
import os
import pytest
import json
import glob
from pathlib import Path

# src is handled by pytest.ini pythonpath
from main import main as main_func

def physionet_found_locally():
    """Helper method to check if the Physionet dataset is found locally."""
    for candidate in ['$HOME/Projects/datasets/physiomi', '/ptmp/$USER/physiomi']:
        if os.path.exists(os.path.expanduser(candidate)):
            return os.path.expanduser(candidate)
    return None

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

@pytest.mark.long
@pytest.mark.skipif(
    not physionet_found_locally(),
    reason="PhysioNet dataset not found locally."
)
def test_full_physiomi_pipeline_direct_call(tmp_path, monkeypatch):
    """
    Test the full physiomi pipeline by calling main() directly.
    This facilitates debugging as it runs in the same process.
    """
    data_dir = physionet_found_locally()
    output_dir = tmp_path / "out"
    
    # Paths must be relative to the PROJECT ROOT because main.py handles them
    # But since we'll run from project root, we should be fine.
    config_path = "src/config/test_training_physiomi.yaml"
    checkpoint_path = "./checkpoint/labram-base.pth"
    
    # Ensure checkpoint exists
    if not os.path.exists(checkpoint_path):
        pytest.skip(f"Checkpoint not found at {checkpoint_path}")

    # Set environment variable for stability (no GPU attribution)
    monkeypatch.setenv("no_gpu_attribution", "true")
    
    # Construct the argument list for main(argv=...)
    argv = [
        "--config", config_path,
        "--output_dir", str(output_dir),
        "--data_dir", data_dir,
        "--debug",
        "--do_testing",
        "--attribute"
    ]
    
    # IMPORTANT: Change working directory to project root for main_func
    # so that relative paths in config and main.py work correctly.
    old_cwd = os.getcwd()
    os.chdir(project_root)
    
    print(f"Calling main with argv: {argv}")
    
    try:
        # Call main directly
        main_func(argv=argv)
    except Exception as e:
        os.chdir(old_cwd)
        pytest.fail(f"Pipeline failed with exception: {e}")
    finally:
        os.chdir(old_cwd)

    # Verify outputs
    dataset_name = "physionetmi_lr"
    log_dir_pattern = os.path.join(output_dir, dataset_name, "version_*")
    log_dirs = glob.glob(log_dir_pattern)
    
    assert len(log_dirs) > 0, f"No log directory found at {log_dir_pattern}"
    log_dir = log_dirs[0]
    
    # 1. Check for test results
    test_results_path = os.path.join(log_dir, "test_results.json")
    assert os.path.exists(test_results_path), f"Test results not found at {test_results_path}"
    with open(test_results_path, 'r') as f:
        test_results = json.load(f)
    assert "test_loss" in test_results

    # 2. Check for attribution results
    val_attribution_path = os.path.join(log_dir, "attribution_results_val.parquet")
    test_attribution_path = os.path.join(log_dir, "attribution_results_test.parquet")
    
    assert os.path.exists(val_attribution_path), f"Validation attribution results not found at {val_attribution_path}"
    assert os.path.exists(test_attribution_path), f"Test attribution results not found at {test_attribution_path}"
    
    # 3. Check Parquet content
    import pyarrow.parquet as pq
    val_table = pq.read_table(val_attribution_path)
    assert val_table.num_rows > 0, "Validation attribution table is empty"
    
    test_table = pq.read_table(test_attribution_path)
    assert test_table.num_rows > 0, "Test attribution table is empty"

    print("Direct pipeline call test passed successfully!")

if __name__ == "__main__":
    # Allow running this script directly
    import tempfile
    
    class MockMonkeyPatch:
        def setenv(self, name, value):
            os.environ[name] = value
            
    with tempfile.TemporaryDirectory() as tmp_dir:
        test_full_physiomi_pipeline_direct_call(Path(tmp_dir), MockMonkeyPatch())
