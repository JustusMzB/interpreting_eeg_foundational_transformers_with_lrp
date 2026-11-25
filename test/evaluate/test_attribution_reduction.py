import os
import torch
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import shutil
import pytest
from torch.utils.data import Dataset
from evaluate.attribute import attribute_dataset, attribute_example
from evaluate.analysis_utils import AttributionAnalyzer, ExperimentResults

class MockEEGDataset(Dataset):
    def __init__(self, num_samples=10, signal_length=100, in_chans=4):
        self.num_samples = num_samples
        self.signal_length = signal_length
        self.in_chans = in_chans

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            # Return a subset of the dataset (though simplified for mock)
            return [self[i] for i in range(*idx.indices(self.num_samples))]
        if idx >= self.num_samples:
            raise IndexError
        return {
            "signal": torch.randn(self.in_chans, self.signal_length),
            "eeg_channel_indices": torch.arange(self.in_chans),
            "label": torch.tensor(idx % 2)
        }

class MockModel(torch.nn.Module):
    def __init__(self, in_chans=4, signal_length=100):
        super().__init__()
        self.device = torch.device("cpu")
        self.linear = torch.nn.Linear(in_chans * signal_length, 2)
    
    def forward(self, x, channels):
        return self.linear(x.flatten(1))

@pytest.fixture
def test_env(tmp_path):
    env_dir = tmp_path / "test_env"
    env_dir.mkdir()
    return env_dir

def create_legacy_run(run_dir, dataset, model):
    run_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for i in range(len(dataset)):
        # We need to create a legacy-style pydict without attribution_shape
        batch = attribute_example(i, dataset[i], model)
        data = batch.to_pydict()
        
        # Manually reconstruct legacy nested list format
        # and remove attribution_shape
        flat_attr = data['attribution'][0]
        shape = data['attribution_shape'][0]
        nested_attr = np.array(flat_attr).reshape(shape).tolist()
        
        legacy_data = {
            'example_index': [data['example_index'][0]],
            'attribution': [nested_attr],
            'prediction': [data['prediction'][0]],
            'label': [data['label'][0]],
            'attributed_class_logit': [data['attributed_class_logit'][0]],
            'attributed_time_step': [data['attributed_time_step'][0]],
            'logit_values': [data['logit_values'][0]],
        }
        results.append(pa.RecordBatch.from_pydict(legacy_data))
    
    table = pa.Table.from_batches(results)
    pq.write_table(table, run_dir / "attribution_results_test.parquet", compression='snappy')

def test_numeric_stability(test_env):
    """Verifies that float16 conversion and storage doesn't introduce unexpected errors."""
    dataset = MockEEGDataset(num_samples=1, signal_length=100, in_chans=4)
    model = MockModel(in_chans=4, signal_length=100)
    
    # Get original float16 attribution from the current implementation
    example = dataset[0]
    batch = attribute_example(0, example, model)
    stored_attr = np.array(batch.to_pydict()['attribution'][0]).astype(np.float16)
    stored_shape = tuple(batch.to_pydict()['attribution_shape'][0])
    
    output_path = test_env / "stability.parquet"
    attribute_dataset(dataset, model, output_path=str(output_path), num_workers=0)
    
    # Load back
    table = pq.read_table(output_path)
    loaded_attr = np.array(table.to_pydict()['attribution'][0]).astype(np.float16)
    loaded_shape = tuple(table.to_pydict()['attribution_shape'][0])
    
    assert stored_shape == loaded_shape
    np.testing.assert_array_almost_equal(stored_attr, loaded_attr)

def test_experiment_discovery_mixed_formats(test_env):
    """Verifies that ExperimentResults can discover both new (float16) and old (float32) runs."""
    dataset = MockEEGDataset(num_samples=5, signal_length=50, in_chans=4)
    model = MockModel(in_chans=4, signal_length=50)
    
    # Run 0: New format
    run_0 = test_env / "run_0" / "task" / "version_0"
    attribute_dataset(dataset, model, output_path=str(run_0 / "attribution_results_test.parquet"), num_workers=0)
    
    # Run 1: Legacy format
    run_1 = test_env / "run_1" / "task" / "version_0"
    create_legacy_run(run_1, dataset, model)
    
    results = ExperimentResults(str(test_env))
    assert len(results.run_dirs) == 2
    
    analyzer = AttributionAnalyzer(results)
    
    # Load Run 0 (float16)
    ddf0 = analyzer.load_run_data(0)
    stats0 = analyzer.calculate_statistics(ddf0)
    assert len(stats0) > 0
    
    # Load Run 1 (float32)
    ddf1 = analyzer.load_run_data(1)
    stats1 = analyzer.calculate_statistics(ddf1)
    assert len(stats1) > 0

def test_classwise_statistics(test_env):
    """Verifies calculate_classwise_statistics handles float16 and produces expected keys."""
    dataset = MockEEGDataset(num_samples=10, signal_length=50, in_chans=4)
    model = MockModel(in_chans=4, signal_length=50)
    
    run_dir = test_env / "run_0" / "task" / "version_0"
    attribute_dataset(dataset, model, output_path=str(run_dir / "attribution_results_test.parquet"), num_workers=0)
    
    results = ExperimentResults(str(test_env))
    analyzer = AttributionAnalyzer(results)
    ddf = analyzer.load_run_data(0)
    
    class_stats = analyzer.calculate_classwise_statistics(ddf, class_names={0: "ClassA", 1: "ClassB"})
    
    for label in [0, 1]:
        assert label in class_stats
        assert f"all_mean" in class_stats[label]
        assert f"all_mean" in class_stats[label] and (f"correct_mean" in class_stats[label] or f"incorrect_mean" in class_stats[label])
        assert class_stats[label]["all_mean"].dtype == np.float64
        assert class_stats[label]["all_mean"].shape == (4, 50)

def test_multi_file_dataset(test_env):
    """Simulates a partitioned dataset (multiple parquet files in a directory)."""
    dataset = MockEEGDataset(num_samples=20, signal_length=50, in_chans=4)
    model = MockModel(in_chans=4, signal_length=50)
    
    run_dir = test_env / "run_dir" / "task" / "version_0"
    run_dir.mkdir(parents=True)
    attr_path = run_dir / "attribution_results_test.parquet"
    
    # Split dataset into two halves and write manually to simulate partitions
    # because pa.dataset.write_dataset with small data might only create one file.
    os.makedirs(attr_path, exist_ok=True)
    
    # File 1
    attribute_dataset(dataset[:10], model, output_path=str(attr_path / "part-0.parquet"), num_workers=0)
    # File 2
    attribute_dataset(dataset[10:], model, output_path=str(attr_path / "part-1.parquet"), num_workers=0)
    
    results = ExperimentResults(str(test_env))
    analyzer = AttributionAnalyzer(results)
    ddf = analyzer.load_run_data(0)
    
    # Dask should see 40 rows (20 samples * 2 logits)
    assert len(ddf) == 40
    stats = analyzer.calculate_statistics(ddf)
    assert sum(s['count'] for s in stats.values()) == 40

@pytest.mark.skip(reason="Manual verification of size reduction. Run with -s to see results.")
def test_size_comparison_report(test_env):
    """Compares file sizes between legacy (Float32, Nested, Snappy) and new (Float16, Flattened, Zstd)."""
    # Use a larger sample to make differences more apparent
    dataset = MockEEGDataset(num_samples=50, signal_length=1000, in_chans=32)
    model = MockModel(in_chans=32, signal_length=1000)
    
    # New format
    new_path = test_env / "size_new.parquet"
    attribute_dataset(dataset, model, output_path=str(new_path), num_workers=0)
    
    # Legacy format
    legacy_dir = test_env / "legacy_size"
    create_legacy_run(legacy_dir, dataset, model)
    legacy_path = legacy_dir / "attribution_results_test.parquet"
    
    def get_dir_size(path):
        if os.path.isfile(path):
            return os.path.getsize(path)
        return sum(os.path.getsize(os.path.join(path, f)) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)))

    new_size = get_dir_size(new_path)
    legacy_size = get_dir_size(legacy_path)
    
    reduction = (1 - new_size / legacy_size) * 100
    print(f"\n[SIZE REPORT]")
    print(f"Legacy Format: {legacy_size / 1024:.2f} KB")
    print(f"New Format:    {new_size / 1024:.2f} KB")
    print(f"Reduction:     {reduction:.2f}%")
    
    assert new_size < legacy_size
