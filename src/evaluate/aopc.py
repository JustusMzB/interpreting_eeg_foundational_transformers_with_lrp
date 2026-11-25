import torch
from typing import Union, Tuple, Optional, Callable, Iterable
import inspect
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

def get_best_device(device: Union[str, torch.device] = "auto") -> torch.device:
    """Auto-discovers the best available hardware accelerator."""
    if isinstance(device, torch.device):
        return device
    if device is None or device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    return torch.device(device)

def replace_with_uniform_noise(orig: torch.Tensor) -> torch.Tensor:
    """Replaces values with uniform noise based on the min and max of the tensor."""
    _min = orig.min()
    _max = orig.max()
    return torch.rand_like(orig) * (_max - _min) + _min

def replace_with_global_mean(orig: torch.Tensor) -> torch.Tensor:
    """Replaces values with the mean of the tensor.

    Called per sample on a single channel, i.e. the channel is replaced by its own mean
    over time within that sample.
    """
    return torch.full_like(orig, orig.mean().item())

# Literal to function map
PERTURBATION_STRATEGIES = {
    'uniform_noise': replace_with_uniform_noise,
    'global_mean': replace_with_global_mean,
    'channel_mean': replace_with_global_mean,  # explicit alias: per-sample channel mean
}

def perturb_at_location(samples: torch.Tensor, location: torch.Tensor, strategy: Union[str, Callable] = 'uniform_noise') -> torch.Tensor:
    """
    Applies a perturbation to `samples` at the given `location` (channel indices).
    
    NOTE: This function assumes that all previous perturbations have been done already.
    The modification is applied recursively on the current state of `samples`.
    
    Args:
        samples (torch.Tensor): The input tensor of shape (batch_size, num_channels, ...)
        location (torch.Tensor): The channel index to perturb. Can be 0D/1D (single global channel) 
                                 or 1D of shape (batch_size,) for per-sample channels.
        strategy (str or Callable): Strategy literal ('uniform_noise', 'global_mean') 
                                    or a callable that takes a tensor and returns a perturbed tensor.
        
    Returns:
        torch.Tensor: The recursively perturbed samples.
    """
    perturbed = samples.clone()
    batch_size = samples.shape[0]

    if callable(strategy):
        apply_replacement = strategy
    elif strategy in PERTURBATION_STRATEGIES:
        apply_replacement = PERTURBATION_STRATEGIES[strategy]
    else:
        raise ValueError(f"Unknown perturbation strategy: {strategy}. Available: {list(PERTURBATION_STRATEGIES.keys())}")

    if location.dim() == 0 or (location.dim() == 1 and len(location) == 1 and batch_size > 1):
        # Global order: same channel for every sample, but the replacement value is computed
        # per sample (e.g. that sample's channel mean), so results do not depend on batch composition.
        loc_idx = location.item() if location.dim() == 0 else location[0].item()
        for i in range(batch_size):
            perturbed[i, loc_idx] = apply_replacement(perturbed[i, loc_idx])
    elif location.dim() == 1 and len(location) == batch_size:
        for i in range(batch_size):
            loc_idx = location[i].item()
            perturbed[i, loc_idx] = apply_replacement(perturbed[i, loc_idx])
    else:
        raise ValueError(f"Invalid location shape: {location.shape} for batch size {batch_size}")

    return perturbed

def perturbation_curve_area(
    model: torch.nn.Module, 
    samples: torch.Tensor, 
    order: torch.Tensor, 
    perturbation_strategy: Union[str, Callable] = 'uniform_noise', 
    target_classes: Optional[torch.Tensor] = None,
    return_logits: bool = False,
    apply_sigmoid: bool = False,
    **model_kwargs
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Executes the recursive perturbation over steps, computes logit drops and their cumulative sum.
    
    Args:
        model: The PyTorch model.
        samples: The batch of input samples. Shape (batch_size, channels, ...)
        order: The channel perturbation order. Shape (num_steps,) for global or (batch_size, num_steps) for per-sample.
        perturbation_strategy: Strategy to use ('uniform_noise', 'global_mean' or Callable).
        target_classes: Optional target classes for each sample in the batch to compute logit drops.
                        If None, uses the predicted classes from the unperturbed logits.
        return_logits: If True, returns (cumsum_logit_difference, all_logits).
        apply_sigmoid: If True, applies sigmoid to the logits before computing drops.
        **model_kwargs: Additional arguments passed to the model forward pass.
                        
    Returns:
        If return_logits is False:
            cumsum_logit_difference (torch.Tensor): Shape (order_steps, batch_size)
        If return_logits is True:
            (cumsum_logit_difference, all_logits) 
            where all_logits is (order_steps + 1, batch_size, num_classes) with unperturbed logits at step 0.
    """
    model.eval()
    
    with torch.no_grad():
        original_logits = model(samples, **model_kwargs)
        
    is_binary = original_logits.dim() == 1 or (original_logits.dim() == 2 and original_logits.shape[1] == 1)
    
    if is_binary:
        original_logits = original_logits.view(-1)
        batch_size = original_logits.shape[0]
        num_classes = 1
    else:
        batch_size, num_classes = original_logits.shape
    
    if target_classes is None:
        if is_binary:
            target_classes = (original_logits > 0).long()
        else:
            target_classes = original_logits.argmax(dim=-1)
            
    if order.dim() == 1:
        order_steps = len(order)
        # expand to (batch_size, order_steps) to standardize iteration
        order = order.unsqueeze(0).expand(batch_size, -1)
    else:
        order_steps = order.shape[1]
        
    if is_binary:
        perturbed_logits_accum = torch.zeros((order_steps, batch_size), device=original_logits.device, dtype=original_logits.dtype)
    else:
        perturbed_logits_accum = torch.zeros((order_steps, batch_size, num_classes), device=original_logits.device, dtype=original_logits.dtype)
    
    current_samples = samples.clone()
    
    for i in tqdm(range(order_steps), desc="Perturbing Steps", position=1, leave=False):
        location = order[:, i]
        current_samples = perturb_at_location(current_samples, location, strategy=perturbation_strategy)
        with torch.no_grad():
            out = model(current_samples, **model_kwargs)
            if is_binary:
                perturbed_logits_accum[i] = out.view(-1)
            else:
                perturbed_logits_accum[i] = out

    if apply_sigmoid:
        original_logits = torch.sigmoid(original_logits)
        perturbed_logits_accum = torch.sigmoid(perturbed_logits_accum)
            
    # Extract target class logits
    if is_binary:
        # For binary (1 logit per sample), logit represents log(p(y=1)/p(y=0)).
        # If target is 1, we want logit drop: L - L_pert.
        # If target is 0, we want logit drop: (-L) - (-L_pert) = L_pert - L.
        # We can achieve this by multiplying the logits by (2 * target - 1), which maps 1 -> 1, 0 -> -1.
        sign = (2 * target_classes.float() - 1)
        orig_target_logits = original_logits * sign
        pert_target_logits = perturbed_logits_accum * sign.unsqueeze(0)
    else:
        batch_indices = torch.arange(batch_size, device=original_logits.device)
        orig_target_logits = original_logits[batch_indices, target_classes] # (batch_size,)
        pert_target_logits = perturbed_logits_accum[:, batch_indices, target_classes] # (order_steps, batch_size)
    
    # Drops: f(x) - f(x_perturbed)
    drops = orig_target_logits.unsqueeze(0) - pert_target_logits # (order_steps, batch_size)
    
    # Cumulative sum
    cumsum_logit_difference = drops.cumsum(dim=0) # (order_steps, batch_size)
    
    if return_logits:
        # all_logits: prepend original_logits
        all_logits = torch.cat([original_logits.unsqueeze(0), perturbed_logits_accum], dim=0)
        return cumsum_logit_difference, all_logits
        
    return cumsum_logit_difference

def evaluate_aopc_for_dataset(
    model: torch.nn.Module,
    dataloader: Iterable,
    order: torch.Tensor,
    perturbation_strategy: str = "mean",
    device: Union[str, torch.device] = "auto",
    use_targets_from_dataloader: bool = False,
    return_example_logits: bool = True,
    total_batches: Optional[int] = None,
    apply_sigmoid: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    High-level wrapper to efficiently evaluate the perturbation curve over a dataset in batches.
    
    Args:
        model: PyTorch model.
        dataloader: Iterable PyTorch DataLoader yielding at least `(samples, ...)` and optionally targets.
        order: Tensor of shape (num_steps,) for global or (num_samples, num_steps) for per-sample order.
        perturbation_strategy: Strategy to use.
        device: Target compute device ('auto' detects cuda/mps/cpu).
        return_example_logits: If True, returns example logits from the first batch.
        use_targets_from_dataloader: If True, uses dataset labels as targets. Default False: the target is the
            model's own prediction, i.e. the sign(f(x)) of Eq. 2 in the paper.
        apply_sigmoid: If True, applies sigmoid to the logits before computing drops.
        
    Returns:
        Concatenated results over the entire dataset (always on CPU to save accelerator memory).
        If return_example_logits is False:
            cumsum_logit_difference (torch.Tensor): Shape (order_steps, num_samples)
        If return_example_logits is True:
            (cumsum_logit_difference, all_logits)
    """
    dev = get_best_device(device)
    model = model.to(dev)
    model.eval()
    
    all_cumsums = []
    example_logits = None
    order_idx = 0
    
    # Introspect model's forward signature to only pass accepted kwargs
    sig = inspect.signature(model.forward)
    valid_kwargs = set(sig.parameters.keys())
    
    for batch in tqdm(dataloader, desc=f"Evaluating AOPC", position=0, leave=True, total=total_batches):
        if isinstance(batch, dict):
            b_samples = batch.get('signal', batch.get('samples')).to(dev)
            b_target = batch.get('label', batch.get('target')) if use_targets_from_dataloader else None
            if b_target is not None:
                b_target = b_target.to(dev)
                
            model_kwargs = {}
            for k, v in batch.items():
                if k not in ['signal', 'samples', 'label', 'target']:
                    mapped_k = 'input_chans' if k == 'eeg_channel_indices' else k
                    if mapped_k in valid_kwargs or 'kwargs' in valid_kwargs:
                        model_kwargs[mapped_k] = v.to(dev) if isinstance(v, torch.Tensor) else v
        else:
            b_samples = batch[0].to(dev)
            b_target = None
            if use_targets_from_dataloader and len(batch) > 1:
                b_target = batch[1].to(dev)
            model_kwargs = {}
            
        if order.dim() == 1:
            b_order = order
        else:
            rem_order = order[order_idx:order_idx + b_samples.size(0)]
            if rem_order.size(0) == 0:
                break
            if rem_order.size(0) < b_samples.size(0):
                b_samples = b_samples[:rem_order.size(0)]
                if b_target is not None:
                    b_target = b_target[:rem_order.size(0)]
                for k in model_kwargs:
                    if isinstance(model_kwargs[k], torch.Tensor) and model_kwargs[k].size(0) > rem_order.size(0):
                        model_kwargs[k] = model_kwargs[k][:rem_order.size(0)]
            b_order = rem_order
            order_idx += b_samples.size(0)
        
        need_logits = return_example_logits and example_logits is None
        
        if need_logits:
            res, batch_logits = perturbation_curve_area(
                model=model,
                samples=b_samples,
                order=b_order,
                target_classes=b_target,
                perturbation_strategy=perturbation_strategy,
                return_logits=True,
                apply_sigmoid=apply_sigmoid,
                **model_kwargs
            )
            example_logits = batch_logits.cpu()
        else:
            res = perturbation_curve_area(
                model=model,
                samples=b_samples,
                order=b_order,
                target_classes=b_target,
                perturbation_strategy=perturbation_strategy,
                return_logits=False,
                apply_sigmoid=apply_sigmoid,
                **model_kwargs
            )
            
        all_cumsums.append(res.cpu())
            
    # Concatenate batches along the batch dimension (dim=1)
    final_cumsum = torch.cat(all_cumsums, dim=1)
    
    if return_example_logits:
        return final_cumsum, example_logits
        
    return final_cumsum

def aopcs_at_step(cumsum_logit_difference: torch.Tensor, step: int) -> torch.Tensor:
    """
    Returns the Area Over Perturbation Curve (AOPC) at the specified step.
    
    Args:
        cumsum_logit_difference: The cumulative sum of logit drops. Shape (order_steps, batch_size)
        step: 0-indexed step to query the AOPC.
        
    Returns:
        torch.Tensor: The AOPC values for the batch at the given step. Shape (batch_size,)
    """
    # Eq. 2 after L = step + 1 perturbations: sum of drops / (L + 1)
    return cumsum_logit_difference[step] / (step + 2)


def aopc_per_sample(cumsum_logit_difference: torch.Tensor) -> torch.Tensor:
    """Eq. 2 per sample over all L steps: sum_{s=1..L} (f(x) - f(x_s)) / (L + 1). Shape (batch_size,)."""
    L = cumsum_logit_difference.shape[0]
    return cumsum_logit_difference[-1] / (L + 1)


def aopc_curve(cumsum_logit_difference: torch.Tensor) -> torch.Tensor:
    """Mean Eq. 2 AOPC after each number of steps L = 1..L_max. Shape (order_steps,)."""
    L = torch.arange(1, cumsum_logit_difference.shape[0] + 1, dtype=cumsum_logit_difference.dtype)
    return (cumsum_logit_difference / (L + 1).unsqueeze(1)).mean(dim=1)


def legacy_running_mean_score(cumsum_logit_difference: torch.Tensor) -> float:
    """Score used before the Eq. 2 fix: mean over steps k of cumsum_k / k. Kept for comparison only."""
    k = torch.arange(1, cumsum_logit_difference.shape[0] + 1, dtype=cumsum_logit_difference.dtype)
    return (cumsum_logit_difference / k.unsqueeze(1)).mean().item()

def _aggregate_attributions(attributions: torch.Tensor) -> torch.Tensor:
    """Aggregates attributions across non-channel dimensions (e.g. time)."""
    if attributions.dim() > 2:
        return attributions.sum(dim=list(range(2, attributions.dim())))
    return attributions

def morf(attributions: torch.Tensor) -> torch.Tensor:
    """
    Most Relevant First.
    
    Args:
        attributions: Attribution tensor of shape (batch_size, num_channels, ...)
    Returns:
        order: Channel order sorted descending by attribution. Shape (batch_size, num_channels)
    """
    agg_attrs = _aggregate_attributions(attributions)
    return torch.argsort(agg_attrs, dim=1, descending=True)

def lerf(attributions: torch.Tensor) -> torch.Tensor:
    """
    Least Relevant First.
    
    Args:
        attributions: Attribution tensor of shape (batch_size, num_channels, ...)
    Returns:
        order: Channel order sorted ascending by attribution. Shape (batch_size, num_channels)
    """
    agg_attrs = _aggregate_attributions(attributions)
    return torch.argsort(agg_attrs, dim=1, descending=False)

def random_order(batch_size: int, num_channels: int, device: torch.device, generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """
    Random Order.
    Generates a unique random permutation of channels for each sample.
    
    Args:
        batch_size: Size of the batch.
        num_channels: Total number of channels.
        device: Device to create the tensor on.
        generator: Optional torch.Generator for reproducible orders.
    Returns:
        order: Random permutations. Shape (batch_size, num_channels)
    """
    return torch.stack([torch.randperm(num_channels, generator=generator).to(device) for _ in range(batch_size)])


def check_attribution_alignment(example_index, labels, dataset, sample_keys=None) -> None:
    """Verifies that stored attributions refer to the same samples as `dataset[example_index]`.

    AOPC pairs attributions with data purely by position. If the dataset was rebuilt with a different
    sample order (e.g. on another machine), per-sample perturbation orders would silently be applied to
    the wrong samples. This raises instead.

    Args:
        example_index: Sequence of dataset positions, one per attribution row.
        labels: Stored labels, one per attribution row.
        dataset: Indexable dataset (HF datasets are read column-wise).
        sample_keys: Optional stored "<subject_id>/<run_file>" keys (empty strings are ignored).
    """
    ds = dataset.with_format(None) if hasattr(dataset, "with_format") else dataset
    idx = [int(i) for i in example_index]
    if hasattr(ds, "column_names"):
        ds_labels = [int(l) for l in ds.select(idx)["label"]] if idx else []
        cols = ds.column_names
        ds_keys = ([f"{s}/{r}" for s, r in zip(ds.select(idx)["subject_id"], ds.select(idx)["run_file"])]
                   if {"subject_id", "run_file"} <= set(cols) else None)
    else:
        rows = [ds[i] for i in idx]
        ds_labels = [int(r["label"]) for r in rows]
        ds_keys = ([f"{r['subject_id']}/{r['run_file']}" for r in rows]
                   if rows and "subject_id" in rows[0] and "run_file" in rows[0] else None)

    n_label_mismatch = sum(int(a) != b for a, b in zip(labels, ds_labels))
    if n_label_mismatch:
        raise ValueError(
            f"Attributions do not match the dataset order: {n_label_mismatch}/{len(idx)} labels differ at the "
            "same example_index. The dataset was likely built with a different sample order than when the "
            "attributions were computed. Regenerate the attributions with the current dataset.")
    if sample_keys is not None and ds_keys is not None:
        n_key_mismatch = sum(bool(k) and k != d for k, d in zip(sample_keys, ds_keys))
        if n_key_mismatch:
            raise ValueError(
                f"Attributions do not match the dataset order: {n_key_mismatch}/{len(idx)} sample keys "
                "(subject/run file) differ at the same example_index. Regenerate the attributions.")


PER_SAMPLE_ORDERINGS = ("magnitude", "prediction", "signed")
GLOBAL_ORDERINGS = ("magnitude", "prediction", "signed", "legacy-timewise-magnitude")


def _prediction_sign(logits: torch.Tensor) -> torch.Tensor:
    """+1 where the single binary logit predicts class 1, -1 otherwise. Shape (n, 1)."""
    return torch.where(logits.reshape(-1) > 0, 1.0, -1.0).unsqueeze(1)


def per_sample_order(attributions: torch.Tensor, logits: torch.Tensor, mode: str = "magnitude") -> torch.Tensor:
    """Per-sample most-relevant-first channel order.

    Channel scores are attributions summed over time. Modes:
      - "magnitude":  sort by |score| (independent of the sign convention of the attribution method).
      - "prediction": sort by score * sign(f(x)), i.e. channels supporting the model's own decision first.
      - "signed":     sort by the raw signed score (pre-fix behaviour; only decision-oriented for class 1).
    Returns: (n, num_channels)
    """
    scores = _aggregate_attributions(attributions)
    if mode == "magnitude":
        key = scores.abs()
    elif mode == "prediction":
        key = scores * _prediction_sign(logits).to(scores.dtype)
    elif mode == "signed":
        key = scores
    else:
        raise ValueError(f"Unknown per-sample ordering '{mode}'. Choose from {PER_SAMPLE_ORDERINGS}.")
    return torch.argsort(key, dim=1, descending=True)


def global_order(attributions: torch.Tensor, logits: Optional[torch.Tensor] = None, mode: str = "magnitude") -> torch.Tensor:
    """Most-relevant-first channel order of the dataset-level aggregate.

    Modes:
      - "magnitude":  |sum_t mean_n R(c, t)|, the magnitude of the visualised global heatmap.
      - "prediction": mean_n of sign(f(x)) * score, an aggregate oriented to each sample's decision.
      - "signed":     sum_t mean_n R(c, t), the signed visualised heatmap.
      - "legacy-timewise-magnitude": sum_t |mean_n R(c, t)| (pre-fix behaviour; not the visualised map).
    Returns: (num_channels,)
    """
    scores = _aggregate_attributions(attributions)
    if mode == "magnitude":
        key = scores.mean(dim=0).abs()
    elif mode == "prediction":
        if logits is None:
            raise ValueError("global ordering 'prediction' needs the model logits.")
        key = (scores * _prediction_sign(logits).to(scores.dtype)).mean(dim=0)
    elif mode == "signed":
        key = scores.mean(dim=0)
    elif mode == "legacy-timewise-magnitude":
        key = attributions.mean(dim=0).abs()
        key = key.sum(dim=list(range(1, key.dim()))) if key.dim() > 1 else key
    else:
        raise ValueError(f"Unknown global ordering '{mode}'. Choose from {GLOBAL_ORDERINGS}.")
    return torch.argsort(key, descending=True)
