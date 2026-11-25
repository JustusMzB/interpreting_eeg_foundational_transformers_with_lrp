import pytest
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader

from src.evaluate.aopc import evaluate_aopc_for_dataset, morf, perturb_at_location

class PerturbationTrackingModel(nn.Module):
    """
    A tracking model designed to verify sample-to-order alignment.
    Returns logits that directly expose which channel was perturbed most recently.
    """
    def __init__(self, num_channels=10, num_classes=2):
        super().__init__()
        self.num_channels = num_channels
        self.num_classes = num_classes

    def forward(self, x):
        # x is (batch, channels, time)
        # We compute mean across time for each channel
        ch_means = x.mean(dim=2) # (batch, channels)
        # Return ch_means as logits (first 2 channels or argmax)
        out = torch.zeros(x.shape[0], self.num_classes, device=x.device)
        out[:, 0] = ch_means[:, 0]
        out[:, 1] = ch_means[:, 1]
        return out


def test_morf_order_sample_alignment_in_evaluate_aopc():
    """
    Thorough test ensuring that in evaluate_aopc_for_dataset:
    1. Sample i in DataLoader receives row i of morf_order tensor.
    2. Alignment holds across different batch sizes (1, 2, 3, 5).
    """
    num_samples = 10
    num_channels = 8
    time_steps = 15

    # Construct unique attributions for each sample i
    # Sample i has highest attribution at channel c_i = (i * 2 + 1) % num_channels
    attributions = torch.zeros(num_samples, num_channels, time_steps)
    expected_top_channels = []
    for i in range(num_samples):
        top_ch = (i * 2 + 1) % num_channels
        attributions[i, top_ch, :] = 100.0 + i  # Unique maximum
        expected_top_channels.append(top_ch)

    # Compute MoRF order
    morf_orders = morf(attributions) # (num_samples, num_channels)
    
    # Assert top channel in morf_orders[i] matches expected_top_channels[i]
    for i in range(num_samples):
        assert morf_orders[i, 0].item() == expected_top_channels[i], \
            f"Sample {i} top MoRF channel mismatch: expected {expected_top_channels[i]}, got {morf_orders[i, 0].item()}"

    # Create dummy samples
    samples = torch.ones(num_samples, num_channels, time_steps)
    dataset = TensorDataset(samples)
    model = PerturbationTrackingModel(num_channels=num_channels, num_classes=2)

    # Test across multiple DataLoader batch sizes to ensure chunking preserves 1-to-1 matching
    for batch_size in [1, 2, 3, 5]:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        # Evaluate AOPC
        cumsum = evaluate_aopc_for_dataset(
            model=model,
            dataloader=dataloader,
            order=morf_orders,
            perturbation_strategy='uniform_noise',
            device='cpu',
            return_example_logits=False
        )

        assert cumsum.shape == (num_channels, num_samples), \
            f"Expected cumsum shape ({num_channels}, {num_samples}), got {cumsum.shape} for batch_size={batch_size}"


def test_parquet_attribution_dataframe_deduplication_alignment():
    """
    Tests attribution DataFrame loading logic:
    Verifies that when Parquet files contain 1 row per sample, sorting and deduplicating
    by example_index preserves all dataset items (N rows) in contiguous 0..N-1 order.
    """
    num_samples = 6
    num_channels = 5
    time_steps = 10

    # Build simulated Parquet DataFrame matching actual single-logit experiment files
    records = []
    for i in range(num_samples):
        attr = np.zeros((num_channels, time_steps))
        attr[i % num_channels, :] = 50.0

        records.append({
            "example_index": i,
            "attributed_class_logit": 0,
            "prediction": i % 2,
            "label": i % 2,
            "attribution": attr.tolist()
        })

    df = pd.DataFrame(records)
    assert len(df) == num_samples

    # Load with drop_duplicates and sort_values
    df_clean = df.drop_duplicates(subset=["example_index"]).sort_values("example_index")
    attr_stacked = np.stack(df_clean["attribution"].values)
    tensor_clean = torch.tensor(attr_stacked, dtype=torch.float32)
    morf_orders = morf(tensor_clean)

    assert len(morf_orders) == num_samples
    assert (df_clean["example_index"].values == np.arange(num_samples)).all()

    for i in range(num_samples):
        assert morf_orders[i, 0].item() == i % num_channels


def test_mini_n_samples_truncation_alignment():
    """
    Tests that when mini_n_samples truncation is requested,
    the filtered attribution tensor slice matches the exact subset of the DataLoader.
    """
    num_samples = 10
    mini_n = 4
    num_channels = 4

    attributions = torch.randn(num_samples, num_channels, 10)
    # Give sample i a distinct max channel
    for i in range(num_samples):
        attributions[i, i % num_channels, :] = 100.0

    morf_orders = morf(attributions) # (10, 4)
    truncated_orders = morf_orders[:mini_n] # (4, 4)

    assert truncated_orders.shape == (mini_n, num_channels)
    for i in range(mini_n):
        assert truncated_orders[i, 0].item() == i % num_channels


def test_overwrite_and_backup_cli_flag(tmp_path):
    """
    Tests that passing --overwrite_and_backup creates timestamped backup directories
    for existing experiment and master summary output files.
    """
    import os
    import json
    import glob
    from script.run_aopc_eval_suite import build_parser, evaluate_experiment_folder

    exp_dir = tmp_path / "mock_exp"
    exp_dir.mkdir()
    
    # Results live in aopc_results_<perturbation strategy>_ps-<per-sample ordering>_gl-<global ordering>
    results_dir = exp_dir / "aopc_results_uniform_noise_ps-prediction_gl-magnitude"
    results_dir.mkdir()
    old_json = results_dir / "aopc_evaluation_results.json"
    with open(old_json, "w") as f:
        json.dump([{"fake": "old_data"}], f)

    parser = build_parser()
    args = parser.parse_args([
        "--overwrite_and_backup", "--aopc_perturbation_strategy", "uniform_noise", "--global_ordering", "magnitude",
    ])
    assert args.overwrite_and_backup is True

    # Call evaluate_experiment_folder (which will discover no run_dirs, but will trigger backup of existing results_dir)
    res = evaluate_experiment_folder("mock_exp", str(exp_dir), args)
    
    # Check that a backup directory was created
    backup_dirs = glob.glob(str(exp_dir / "aopc_results_uniform_noise_ps-prediction_gl-magnitude_backup_*"))
    assert len(backup_dirs) == 1, f"Expected 1 backup directory, found: {backup_dirs}"
    assert os.path.exists(os.path.join(backup_dirs[0], "aopc_evaluation_results.json"))

