import os
import shutil
import pytest
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from evaluate.analysis_utils import ExperimentResults, AttributionAnalyzer

@pytest.fixture
def mock_experiment_dir(tmp_path):
    """Creates a mock experiment directory with run subdirectories and dummy data."""
    exp_dir = tmp_path / "test_experiment"
    exp_dir.mkdir()
    
    # Create evaluation_summary.csv
    summary_data = pd.DataFrame({
        "run": [0, 1],
        "test_balanced_accuracy": [0.65, 0.70]
    })
    summary_data.to_csv(exp_dir / "evaluation_summary.csv", index=False)
    
    # Create run directories
    for run in [0, 1]:
        run_dir = exp_dir / f"run_{run}" / "task" / "version_0"
        run_dir.mkdir(parents=True)
        
        # Create dummy attribution parquet
        # Attribution shape: 3 channels, 10 timepoints
        data = {
            "example_index": [0, 1, 2, 3],
            "prediction": [0, 1, 0, 1],
            "label": [0, 1, 1, 0],
            "attribution": [
                np.random.rand(3, 10).tolist() for _ in range(4)
            ]
        }
        table = pa.Table.from_pydict(data)
        pq.write_table(table, run_dir / "attribution_results_test.parquet")
        
    return str(exp_dir)

def test_experiment_results_discovery(mock_experiment_dir):
    results = ExperimentResults(mock_experiment_dir)
    assert len(results.run_dirs) == 2
    assert results.performance_df is not None
    assert len(results.performance_df) == 2
    
    path = results.get_attribution_path(0, "test")
    assert path is not None
    assert os.path.exists(path)

def test_attribution_analyzer_stats(mock_experiment_dir):
    results = ExperimentResults(mock_experiment_dir)
    analyzer = AttributionAnalyzer(results)
    
    ddf = analyzer.load_run_data(0, "test")
    stats = analyzer.calculate_statistics(ddf)
    
    # Predictions were [0, 1, 0, 1], so we expect stats for class 0 and 1
    assert 0 in stats
    assert 1 in stats
    assert stats[0]["count"] == 2
    assert stats[1]["count"] == 2
    assert stats[0]["mean"].shape == (3, 10)
    assert "std" in stats[0]
    assert stats[0]["std"].shape == (3, 10)

def test_attribution_analyzer_classwise_stats(mock_experiment_dir):
    results = ExperimentResults(mock_experiment_dir)
    analyzer = AttributionAnalyzer(results)
    
    ddf = analyzer.load_run_data(0, "test")
    # In mock data: labels are [0, 1, 1, 0], predictions are [0, 1, 0, 1]
    # Class 0: 1 correct (0,0), 1 incorrect (0,1)
    # Class 1: 1 correct (1,1), 1 incorrect (1,0)
    
    class_stats = analyzer.calculate_classwise_statistics(ddf, class_names={0: "Low", 1: "High"})
    
    assert 0 in class_stats
    assert 1 in class_stats
    assert class_stats[0]["class_name"] == "Low"
    assert "all_mean" in class_stats[0]
    assert "correct_mean" in class_stats[0]
    assert "incorrect_mean" in class_stats[0]
    assert "all_std" in class_stats[0]
    assert "correct_std" in class_stats[0]
    assert "incorrect_std" in class_stats[0]
    assert class_stats[0]["all_count"] == 2
    assert class_stats[0]["correct_count"] == 1
    assert class_stats[0]["incorrect_count"] == 1

def test_map_subjects(mock_experiment_dir):
    results = ExperimentResults(mock_experiment_dir)
    analyzer = AttributionAnalyzer(results)
    ddf = analyzer.load_run_data(0, "test")
    
    # Mock dataset
    class MockDataset:
        def __getitem__(self, i):
            return {"subject": f"sub-{i}"}
    
    ddf_with_subjects = analyzer.map_subjects(ddf, MockDataset())
    computed = ddf_with_subjects.compute()
    
    assert "subject" in computed.columns
    assert computed.loc[computed.example_index == 0, "subject"].iloc[0] == "sub-0"

def test_spray_clustering(mock_experiment_dir):
    results = ExperimentResults(mock_experiment_dir)
    analyzer = AttributionAnalyzer(results)
    ddf = analyzer.load_run_data(0, "test")
    
    # Run with small n_clusters
    spray_results = analyzer.spray_clustering(ddf, n_clusters=2, downsample_factor=2)
    
    assert "labels" in spray_results
    assert len(spray_results["labels"]) == 4 # 4 examples in mock data
    assert spray_results["n_clusters"] == 2

def test_downsampling():
    results = None # Not needed for pure method
    analyzer = AttributionAnalyzer(results)
    
    attr = np.random.rand(3, 100)
    downsampled = analyzer.downsample_attributions(attr, factor=10)
    assert downsampled.shape == (3, 10)
    
    # Verify value (first window)
    expected_val = attr[0, :10].mean()
    assert np.allclose(downsampled[0, 0], expected_val)

def test_load_run_data_to_memory(mock_experiment_dir):
    results = ExperimentResults(mock_experiment_dir)
    analyzer = AttributionAnalyzer(results)
    
    df = analyzer.load_run_data_to_memory(0, "test")
    
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 4
    assert "attribution" in df.columns
    # Check if attribution is a numpy array (as converted by _convert_to_numpy)
    assert isinstance(df["attribution"].iloc[0], np.ndarray)
    assert df["attribution"].iloc[0].shape == (3, 10)

def test_load_all_runs_data(mock_experiment_dir):
    results = ExperimentResults(mock_experiment_dir)
    analyzer = AttributionAnalyzer(results)
    
    combined_ddf = analyzer.load_all_runs_data(split="test")
    df = combined_ddf.compute()
    
    # 2 runs with 4 samples each = 8 total samples
    assert len(df) == 8
    assert "run_idx" in df.columns
    assert set(df["run_idx"].unique()) == {0, 1}

def test_calculate_aggregate_statistics(mock_experiment_dir):
    results = ExperimentResults(mock_experiment_dir)
    analyzer = AttributionAnalyzer(results)
    
    agg_stats = analyzer.calculate_aggregate_statistics(split="test", class_names={0: "Low", 1: "High"})
    
    assert "pooled_classwise" in agg_stats
    assert "per_run_classwise" in agg_stats
    assert "cross_run_aggregates" in agg_stats
    
    # Check pooled statistics
    pooled = agg_stats["pooled_classwise"]
    assert 0 in pooled
    assert 1 in pooled
    assert pooled[0]["all_count"] == 4 # 2 samples per run * 2 runs
    
    # Check per run statistics
    assert 0 in agg_stats["per_run_classwise"]
    assert 1 in agg_stats["per_run_classwise"]
    
    # Check cross-run aggregates
    cross = agg_stats["cross_run_aggregates"]
    assert 0 in cross
    assert "all_mean_across_runs" in cross[0]
    assert "all_std_across_runs" in cross[0]
    assert cross[0]["all_mean_across_runs"].shape == (3, 10)

