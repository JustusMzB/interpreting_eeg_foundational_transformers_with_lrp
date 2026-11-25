"""
Layer 4 — End-to-end pipeline tests for the PhysioMI workflow.

Requires RUN_LONG_TESTS=1 and the physiomi data directory to be present.
Uses mini-dataset mode (a few subjects) so the full 109-subject tree is
not required — but if data/physiomi/files/ already exists the download
is skipped entirely.

Run with:
    RUN_LONG_TESTS=1 uv run pytest test/test_physiomi_long.py -v -m long
"""
import json
import os
import subprocess
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN_EXPERIMENT = os.path.join(REPO_ROOT, "src", "run_experiment.py")
FINETUNED_CONFIG = os.path.join(REPO_ROOT, "src", "config", "production", "physionet_finetuned.yaml")
BEST_CKPT = os.path.join(REPO_ROOT, "checkpoint", "physionet_finetuned_best.ckpt")
DATA_DIR = os.path.join(REPO_ROOT, "data", "physiomi")

MINI_TRAIN = 3
MINI_VAL = 2
MINI_TEST = 2


def _data_available():
    return os.path.isdir(os.path.join(DATA_DIR, "files"))


def _run(args, **kwargs):
    return subprocess.run(
        [sys.executable, RUN_EXPERIMENT] + args,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        **kwargs,
    )


@pytest.mark.long
def test_full_pipeline_train_test_attribute(tmp_path):
    """4a — train → test → attribute on a mini PhysioMI subset."""
    if not _data_available():
        pytest.skip("data/physiomi/files not present; set up the dataset first")

    result = _run([
        "--config", FINETUNED_CONFIG,
        "--data_dir", DATA_DIR,
        "--output_dir", str(tmp_path),
        "--mini_n_train", str(MINI_TRAIN),
        "--mini_n_val", str(MINI_VAL),
        "--mini_n_test", str(MINI_TEST),
    ])

    assert result.returncode == 0, (
        f"run_experiment.py exited {result.returncode}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )

    # test_results.json must exist and contain test_loss
    results_files = list(tmp_path.rglob("test_results.json"))
    assert results_files, "No test_results.json found under output dir"
    with open(results_files[0]) as f:
        metrics = json.load(f)
    assert "test_loss" in metrics, f"test_loss missing from {metrics}"

    # Both attribution parquets must exist and be non-empty
    val_parquets = list(tmp_path.rglob("attribution_results_val.parquet"))
    test_parquets = list(tmp_path.rglob("attribution_results_test.parquet"))
    assert val_parquets, "attribution_results_val.parquet not found"
    assert test_parquets, "attribution_results_test.parquet not found"
    assert os.path.getsize(val_parquets[0]) > 0
    assert os.path.getsize(test_parquets[0]) > 0


@pytest.mark.long
@pytest.mark.network
def test_attribution_only_downloads_mini(tmp_path):
    """Downloads a fresh mini PhysioMI subset into a temp dir and runs attribution-only.

    Does not rely on any pre-existing data directory.  Marked network because
    it fetches EDF files from physiomi.org.
    """
    if not os.path.exists(BEST_CKPT):
        pytest.skip("checkpoint/physionet_finetuned_best.ckpt not present")

    data_dir = tmp_path / "physiomi"
    out_dir = tmp_path / "output"

    result = _run([
        "--config", FINETUNED_CONFIG,
        "--data_dir", str(data_dir),
        "--output_dir", str(out_dir),
        "--attribute_only",
        "--checkpoint", BEST_CKPT,
        "--mini_n_train", str(MINI_TRAIN),
        "--mini_n_val", str(MINI_VAL),
        "--mini_n_test", str(MINI_TEST),
    ])

    assert result.returncode == 0, (
        f"run_experiment.py exited {result.returncode}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )

    val_parquets = list(out_dir.rglob("attribution_results_val.parquet"))
    test_parquets = list(out_dir.rglob("attribution_results_test.parquet"))
    assert val_parquets, "attribution_results_val.parquet not found"
    assert test_parquets, "attribution_results_test.parquet not found"
    assert os.path.getsize(val_parquets[0]) > 0
    assert os.path.getsize(test_parquets[0]) > 0


@pytest.mark.long
def test_attribution_only_pipeline(tmp_path):
    """4b — attribution-only using the published checkpoint; no training dir created."""
    if not _data_available():
        pytest.skip("data/physiomi/files not present; set up the dataset first")
    if not os.path.exists(BEST_CKPT):
        pytest.skip("checkpoint/physionet_finetuned_best.ckpt not present")

    result = _run([
        "--config", FINETUNED_CONFIG,
        "--data_dir", DATA_DIR,
        "--output_dir", str(tmp_path),
        "--attribute_only",
        "--checkpoint", BEST_CKPT,
        "--mini_n_train", str(MINI_TRAIN),
        "--mini_n_val", str(MINI_VAL),
        "--mini_n_test", str(MINI_TEST),
    ])

    assert result.returncode == 0, (
        f"run_experiment.py exited {result.returncode}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )

    # No training logs should have been created
    assert not list(tmp_path.rglob("test_results.json")), \
        "test_results.json should not exist in attribution-only mode"

    val_parquets = list(tmp_path.rglob("attribution_results_val.parquet"))
    test_parquets = list(tmp_path.rglob("attribution_results_test.parquet"))
    assert val_parquets, "attribution_results_val.parquet not found"
    assert test_parquets, "attribution_results_test.parquet not found"
    assert os.path.getsize(val_parquets[0]) > 0
    assert os.path.getsize(test_parquets[0]) > 0
