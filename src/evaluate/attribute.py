"""
Core attribution module for generating high-resolution EEG feature relevances.
This module implements the Layer-wise Relevance Propagation (LRP) framework 
specifically optimized for foundational EEG Transformers (LaBraM).

It utilizes attention-aware LRP (AttnLRP) to propagate relevance through transformer 
attention layers, as described in 'From Clever Hans to Hypotheses: 
Interpreting EEG Foundational Transformers'.
"""
from models.lighting_labram import NeuralTransformerLightning
from lxt.efficient import monkey_patch
from models.LaBraM_LXT import attnLRP, zennit_composite
from models import modeling_finetune
import torch
import pyarrow as pa
import logging
import os
from tqdm.auto import tqdm
import torch.multiprocessing as mp
import shutil
import os
import numpy as np

LOGGER = logging.getLogger(__name__)

### GPU Attribution Configuration
# Due to some weird issue with GPU attribution leading to filehandle leaks in certain environments, 
# we keep the default to CPU attribution.
NO_GPU_ATTRIBUTION = os.environ.get("no_gpu_attribution", "true").lower() == "true"

if torch.cuda.is_available():
    if torch.cuda.get_device_name(0) == "AMD Instinct MI300A":
        # Fallback to CPU if AMD Instinct MI300A is detected, which we found to have a bug in backward pass for LRP
        LOGGER.warning("Detected AMD Instinct MI300A, falling back to CPU for attribution due to known issues.")
        DEVICE = torch.device("cpu")
    elif NO_GPU_ATTRIBUTION:
        DEVICE = torch.device("cpu")
        LOGGER.info("GPU not used during attribution. Using CPU for attribution. Set 'no_gpu_attribution' to 'false' to enable GPU attribution.")
    else:
        LOGGER.warning("Using GPU for attribution. Experimental. Set 'no_gpu_attribution' to 'true' to disable GPU attribution.")
        DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


# Global variables for multiprocessing workers
worker_model = None
worker_subsample_time_steps = None

def init_worker(model, subsample_time_steps=None):
    """
    Initializer for each worker process in the multiprocessing pool.
    
    Pre-loads the model into worker memory and configures the LRP environment 
    by applying architecture-specific monkey patches (AttnLRP) and 
    registering the Zennit composite for convolutional layers.
    
    Args:
        model (NeuralTransformerLightning): The pre-trained model instance.
        subsample_time_steps (int, optional): Temporal subsampling factor.
    """
    LOGGER.info("Initializing worker process for attribution.")
    global worker_model, worker_subsample_time_steps
    worker_model = model
    worker_subsample_time_steps = subsample_time_steps
    
    if not worker_model.device == DEVICE:
        # Note: Moving model to device inside worker might freeze processes if not handled carefully
        LOGGER.debug(f'Worker model is on {worker_model.device}')
    
    worker_model.eval()

    # Apply LRP patches to the model architecture
    monkey_patch(modeling_finetune, attnLRP)
    zennit_composite.register(worker_model)

def attribute_example_worker(args):
    """
    Worker function wrapper for multiprocessing. Uses the pre-loaded global model.
    """
    index, example = args
    global worker_model, worker_subsample_time_steps
    assert worker_model is not None, "Model not loaded in worker process."    
    result = attribute_example(index, example, worker_model, subsample_time_steps=worker_subsample_time_steps)
    del example
    del index
    return result

def attribute_example(index, example, model, subsample_time_steps=None):
    """
    Attributes the EEG data for a given example using Layer-wise Relevance Propagation (LRP).
    
    This function performs a backward pass to calculate the contribution of each input
    feature (channel x time) to the model's prediction.
    
    Optimized Storage:
        - Attributions are cast to float16 to reduce disk usage.
        - Attributions are flattened into a 1D list to avoid nested list overhead in Parquet.
        - Original shape is stored in 'attribution_shape' for reconstruction.

    Args:
        index (int): Index of the example in the dataset.
        example (dict): Dictionary containing:
            - "signal": (Channels, Time) EEG data tensor.
            - "eeg_channel_indices": Tensor of channel indices.
            - "label": Ground truth label.
        model (torch.nn.Module): The model to use for attribution.
        subsample_time_steps (int, optional): If set, randomly samples this many time steps 
                                              for attribution calculation.

    Returns:
        pa.RecordBatch: A record batch containing:
            - example_index: int64
            - attribution: list<float16> (flattened)
            - attribution_shape: list<int64>
            - prediction: int64
            - label: int64
            - attributed_class_logit: int64 (which logit was attributed)
            - attributed_time_step: int64
            - logit_values: list<float32> (full output vector)
            - sample_key: string, "<subject_id>/<run_file>" if the dataset provides them (else "").
              Lets downstream analyses (e.g. AOPC) verify that example_index still points at the same
              recording when the dataset is rebuilt elsewhere.
    """
    i = index
    sample_key = (f"{example['subject_id']}/{example['run_file']}"
                  if 'subject_id' in example and 'run_file' in example else "")
    for key in ["signal", "eeg_channel_indices", "label"]:
        assert key in example, f"Key '{key}' not found in example at index {i}."
    
    eeg_data = example["signal"].unsqueeze(0).to(DEVICE)
    channels = example["eeg_channel_indices"].unsqueeze(0).to(DEVICE)
    label = example["label"]

    model.zero_grad()
    eeg_data = eeg_data.to(model.device)
    channels = channels.to(model.device)
    eeg_data.requires_grad_(True)
    
    if eeg_data.grad is not None:
        eeg_data.grad.zero_()
    
    model.eval() 
    output = model(eeg_data, channels) # (1, classes) or (1, classes, time)

    # Determine prediction and number of time steps
    if output.dim() == 3:
        predicted_class = torch.argmax(output, dim=1) if output.shape[1] > 1 else (output > 0).squeeze(dim=1)  # single logit: decision at 0
        num_time_steps = output.shape[2]
    else:
        predicted_class = torch.argmax(output, dim=1) if output.shape[1] > 1 else (output > 0).squeeze(dim=1)  # single logit: decision at 0
        num_time_steps = 1
    
    results = {
        'example_index': [],
        'attribution': [],
        'attribution_shape': [],
        'prediction': [],
        'label': [],
        'attributed_class_logit': [],
        'attributed_time_step': [],
        'logit_values': [],
        'sample_key': [],
    }

    # Explicit schema for Float16 enforcement
    schema = pa.schema([
        ('example_index', pa.int64()),
        ('attribution', pa.list_(pa.float16())),
        ('attribution_shape', pa.list_(pa.int64())),
        ('prediction', pa.int64()),
        ('label', pa.int64()),
        ('attributed_class_logit', pa.int64()),
        ('attributed_time_step', pa.int64()),
        ('logit_values', pa.list_(pa.float32())),
        ('sample_key', pa.string()),
    ])
    
    # Handle temporal subsampling if requested
    if subsample_time_steps is not None and subsample_time_steps < num_time_steps:
        all_indices = np.arange(num_time_steps)
        predicted_class_cpu = predicted_class[0].cpu().numpy()
        positive_indices = all_indices[predicted_class_cpu == True]
        negative_indices = all_indices[predicted_class_cpu == False]
        
        n_pos = min(len(positive_indices), subsample_time_steps // 2)
        n_neg = min(len(negative_indices), subsample_time_steps - n_pos)
        # Re-adjust positive if negative was sparse
        if n_pos + n_neg < subsample_time_steps:
             n_pos = min(len(positive_indices), subsample_time_steps - n_neg)

        pos_samples = np.random.choice(positive_indices, size=n_pos, replace=False) if n_pos > 0 else np.array([])
        neg_samples = np.random.choice(negative_indices, size=n_neg, replace=False) if n_neg > 0 else np.array([])
        time_steps_to_attribute = np.sort(np.concatenate([pos_samples, neg_samples])).astype(int)
    else:
        if num_time_steps > 10:
            LOGGER.warning(
                f"Subsampling time steps is not set, but the number of time steps ({num_time_steps}) is pretty big.\
                The output file will be {num_time_steps} the size of your input data. Are you sure you wanted this?.")
        time_steps_to_attribute = range(num_time_steps)

    for t in time_steps_to_attribute:
        if output.dim() == 3:
            predicted_logits = output[0, :, t]
            current_predicted_class = predicted_class[0, t]
            current_label = label[t] if label.dim() > 0 else label
        else:
            predicted_logits = output[0]
            current_predicted_class = predicted_class[0]
            current_label = label

        for logit_idx in range(predicted_logits.shape[0]):
            eeg_data.grad = None
            model.zero_grad()

            to_grad = predicted_logits[logit_idx] 
            to_grad.backward(retain_graph=True)
            attributions = eeg_data.grad

            # CAST TO FLOAT16 for optimized storage
            attributions_cpu = attributions.squeeze(0).detach().cpu().numpy().astype(np.float16)

            results['example_index'].append(i)
            # FLATTEN for optimized storage
            results['attribution'].append(attributions_cpu.flatten().tolist())
            results['attribution_shape'].append(list(attributions_cpu.shape))
            results['prediction'].append(int(current_predicted_class.item()))
            results['label'].append(int(current_label.item()))
            results['attributed_class_logit'].append(logit_idx)
            results['attributed_time_step'].append(t)
            results['logit_values'].append(predicted_logits.detach().cpu().numpy().tolist())
            results['sample_key'].append(sample_key)
    
    return pa.RecordBatch.from_pydict(results, schema=schema)  

def attributioniterator(example_enumeration, model, num_workers=4, subsample_time_steps=None):
    """
    Generator that attributes examples, optionally parallelized.
    """
    if num_workers > 0:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=num_workers, initializer=init_worker, initargs=(model, subsample_time_steps)) as pool:
            pbar = pool.imap_unordered(attribute_example_worker, example_enumeration, chunksize=100)
            for result in pbar:
                yield result
    else:
        for index, example in example_enumeration:
            res = attribute_example(index, example, model, subsample_time_steps=subsample_time_steps)
            del example
            del index
            yield res

def attribute_dataset(dataset, model, output_path=None, num_workers=4, subsample_time_steps=None):
    """
    Attributes an entire dataset and saves results to a Parquet directory/file.
    
    Compression: Uses Zstandard (level 5) for high-ratio compression of Float16 data.
    
    Args:
        dataset: PyTorch Dataset of EEG examples.
        model: Trained NeuralTransformerLightning model.
        output_path: Destination path for Parquet output.
        num_workers: Number of parallel processes.
        subsample_time_steps: Temporal subsampling factor.
    """
    if output_path is None:
        LOGGER.warning("No output path provided. Results will be saved in working directory.")
        output_path = os.path.join(os.getcwd(), "attribution_results.parquet")
    else:
        parent_dir = os.path.dirname(output_path)
        if parent_dir and not os.path.exists(parent_dir):
            os.makedirs(parent_dir, exist_ok=True)
        
        if os.path.isdir(output_path):
            assert len(os.listdir(output_path)) == 0, f"Output directory {output_path} is not empty."
    
    first_example = dataset[0]
    schema = attribute_example(0, first_example, model, subsample_time_steps=subsample_time_steps).schema

    record_batches = attributioniterator(enumerate(dataset), model, num_workers=num_workers, subsample_time_steps=subsample_time_steps)
    if output_path:
        # Use Zstd for optimized compression of numeric data
        format_options = pa.dataset.ParquetFileFormat().make_write_options(compression='zstd', compression_level=5)
        pa.dataset.write_dataset(
            tqdm(record_batches, total=len(dataset), desc=f"Attributing examples with {num_workers} workers ..."),
            output_path,
            format='parquet',
            schema=schema,
            partitioning=None,
            file_options=format_options,
        )

def attribute_dataset_on_labram(dataset, model, output_path=None, num_workers=4, subsample_time_steps=None):
    """
    High-level entry point for attributing LaBraM models.
    Handles model setup, patching, and sequential/parallel execution.
    """
    LOGGER.info(f"Attributing dataset on LaBraM...")
    model.loss_fn = None  # Disable loss for serializability
    assert isinstance(model, NeuralTransformerLightning), "Model must be NeuralTransformerLightning."
    
    # Ensure model is on the correct device (CPU for multiprocessing)
    model.to(DEVICE)
    model.share_memory()

    if num_workers == 0:
        # For sequential mode, apply patches in the main process
        monkey_patch(modeling_finetune, attnLRP)
        zennit_composite.register(model)

    try:
        results = attribute_dataset(dataset, model, output_path, num_workers=num_workers, subsample_time_steps=subsample_time_steps)
    except OSError as e:
        LOGGER.warning(f"OS error during parallel attribution: {e}. Falling back to sequential.")
        shutil.rmtree(output_path, ignore_errors=True)
        results = attribute_dataset(dataset, model, output_path, num_workers=0)
    
    if num_workers == 0:
        zennit_composite.remove()
    
    LOGGER.info(f"Attribution results saved to {output_path}.")
    return results
