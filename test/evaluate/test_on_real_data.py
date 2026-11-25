import os
import pytest
import pandas as pd
import numpy as np
from evaluate.analysis_utils import ExperimentResults, AttributionAnalyzer

REAL_DATA_PATH = os.environ.get("REAL_DATA_PATH", "")

@pytest.fixture
def real_results():
    if not REAL_DATA_PATH or not os.path.exists(REAL_DATA_PATH):
        pytest.skip("REAL_DATA_PATH is not set or does not exist")
    return ExperimentResults(REAL_DATA_PATH)

@pytest.mark.long
def test_discovery_on_real_data(real_results):
    assert len(real_results.run_dirs) >= 1
    # Check if Run 0 exists and has the expected structure
    attr_path = real_results.get_attribution_path(0, split="test")
    assert attr_path is not None
    assert os.path.exists(attr_path)

@pytest.mark.long
def test_analyzer_on_real_data(real_results):
    analyzer = AttributionAnalyzer(real_results)
    ddf = analyzer.load_run_data(0, split="test")
    
    # Test head() to ensure data is readable
    head_df = ddf.head()
    assert not head_df.empty
    assert "attribution" in head_df.columns
    assert "prediction" in head_df.columns
    
    # Test statistics calculation
    stats = analyzer.calculate_statistics(ddf)
    assert len(stats) > 0
    for label in stats:
        assert "mean" in stats[label]
        assert "count" in stats[label]
        assert isinstance(stats[label]["mean"], np.ndarray)

@pytest.mark.long
def test_classwise_stats_on_real_data(real_results):
    analyzer = AttributionAnalyzer(real_results)
    ddf = analyzer.load_run_data(0, split="test")
    
    class_stats = analyzer.calculate_classwise_statistics(ddf)
    assert len(class_stats) > 0
    for label in class_stats:
        assert "all_mean" in class_stats[label]
        assert "all_count" in class_stats[label]
