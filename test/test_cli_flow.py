"""
CLI flow tests for src/run_experiment.py.

Layer 1 — Argument parsing (no network, no GPU)
Layer 2 — Config sanity (parametrized over all 11 production configs)
"""
import os
import subprocess
import sys

import pytest
import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRODUCTION_CONFIGS_DIR = os.path.join(REPO_ROOT, "src", "config", "production")
RUN_EXPERIMENT = os.path.join(REPO_ROOT, "src", "run_experiment.py")

REQUIRED_FIELDS = ["dataset", "finetune_ckpt", "dataset_path", "nb_classes", "epochs", "batch_size", "lr"]

_production_configs = sorted(
    os.path.join(PRODUCTION_CONFIGS_DIR, f)
    for f in os.listdir(PRODUCTION_CONFIGS_DIR)
    if f.endswith(".yaml")
)


def _run(args):
    return subprocess.run(
        [sys.executable, RUN_EXPERIMENT] + args,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )


def _parser():
    sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
    from run_experiment import build_parser
    return build_parser()


# ---------------------------------------------------------------------------
# Layer 1 — Argument parsing
# ---------------------------------------------------------------------------

def test_run_experiment_accepts_config_flag(tmp_path):
    config = tmp_path / "test.yaml"
    config.write_text("dataset: physionetmi_lr\n")
    args = _parser().parse_args(["--config", str(config), "--data_dir", "d"])
    assert args.config == str(config)
    assert args.data_dir == "d"


def test_run_experiment_requires_config():
    result = _run([])
    assert result.returncode == 2


def test_run_experiment_config_not_found():
    result = _run(["--config", "definitely_missing_xyz.yaml"])
    assert result.returncode == 1
    combined = result.stdout + result.stderr
    assert "not found" in combined.lower()


def test_attribute_only_requires_checkpoint(tmp_path):
    config = tmp_path / "test.yaml"
    config.write_text("dataset: physionetmi_lr\n")
    result = _run(["--config", str(config), "--attribute_only"])
    assert result.returncode == 1


def test_attribute_only_checkpoint_not_found(tmp_path):
    config = tmp_path / "test.yaml"
    config.write_text("dataset: physionetmi_lr\n")
    result = _run(["--config", str(config), "--attribute_only", "--checkpoint", "missing.ckpt"])
    assert result.returncode == 1


def test_workers_default():
    args = _parser().parse_args(["--config", "dummy.yaml"])
    assert args.workers == 4


# ---------------------------------------------------------------------------
# Layer 2 — Config sanity
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("config_path", _production_configs, ids=[os.path.basename(p) for p in _production_configs])
def test_all_production_configs_are_valid_yaml(config_path):
    with open(config_path) as f:
        data = yaml.safe_load(f)
    assert data is not None


@pytest.mark.parametrize("config_path", _production_configs, ids=[os.path.basename(p) for p in _production_configs])
def test_all_production_configs_have_required_fields(config_path):
    with open(config_path) as f:
        data = yaml.safe_load(f)
    missing = [field for field in REQUIRED_FIELDS if field not in data]
    assert not missing, f"{os.path.basename(config_path)} missing fields: {missing}"


@pytest.mark.parametrize("config_path", _production_configs, ids=[os.path.basename(p) for p in _production_configs])
def test_all_production_configs_have_relative_data_dir(config_path):
    with open(config_path) as f:
        data = yaml.safe_load(f)
    data_dir = data.get("data_dir")
    if data_dir is not None:
        assert not str(data_dir).startswith("/ptmp"), \
            f"{os.path.basename(config_path)}: data_dir must not start with /ptmp (got: {data_dir})"


@pytest.mark.parametrize("config_path", _production_configs, ids=[os.path.basename(p) for p in _production_configs])
def test_all_production_configs_have_relative_output_dir(config_path):
    with open(config_path) as f:
        data = yaml.safe_load(f)
    output_dir = data.get("output_dir")
    if output_dir is not None:
        assert not str(output_dir).startswith("/ptmp"), \
            f"{os.path.basename(config_path)}: output_dir must not start with /ptmp (got: {output_dir})"


def test_finetune_ckpt_path_exists():
    ckpt_path = os.path.join(REPO_ROOT, "checkpoint", "labram-base.pth")
    if not os.path.exists(ckpt_path):
        pytest.skip("checkpoint/labram-base.pth not present")
    assert os.path.exists(ckpt_path)


def test_physionet_finetuned_best_ckpt_exists():
    ckpt_path = os.path.join(REPO_ROOT, "checkpoint", "physionet_finetuned_best.ckpt")
    if not os.path.exists(ckpt_path):
        pytest.skip("checkpoint/physionet_finetuned_best.ckpt not present")
    assert os.path.exists(ckpt_path)
