"""
Unit tests for the AOPC evaluation module (src/evaluate/aopc.py).

Complements test/test_aopc.py (shapes and API surface) with behavioral checks
on a linear model whose channel importances are known.
"""
import os

import pytest
import torch
import torch.nn as nn

from evaluate.aopc import (
    get_best_device,
    lerf,
    morf,
    perturb_at_location,
    perturbation_curve_area,
)


def zero_out(t: torch.Tensor) -> torch.Tensor:
    return torch.zeros_like(t)


class LinearChannelModel(nn.Module):
    """Single-logit model: logit = sum_c w_c * sum_t x[c, t]."""

    def __init__(self, weights):
        super().__init__()
        self.weights = nn.Parameter(torch.tensor(weights, dtype=torch.float32))

    def forward(self, x):
        return (x.sum(dim=-1) * self.weights).sum(dim=-1)


def test_get_best_device():
    assert isinstance(get_best_device("auto"), torch.device)
    assert get_best_device("cpu") == torch.device("cpu")
    assert get_best_device(torch.device("cpu")) == torch.device("cpu")


def test_production_configs_exist():
    """The AOPC suite resolves run directories to these production configs."""
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    prod_config_dir = os.path.join(repo_root, "src", "config", "production")
    for name in [
        "physionet_from_scratch.yaml",
        "physionet_frozen.yaml",
        "physionet_finetuned.yaml",
        "arousal_from_scratch.yaml",
        "arousal_frozen.yaml",
        "arousal_finetuned.yaml",
    ]:
        assert os.path.exists(os.path.join(prod_config_dir, name)), name


def test_perturbation_accumulates_across_steps():
    """Step k must keep the perturbation applied at step k-1 (recursive perturbation)."""
    torch.manual_seed(0)
    x = torch.randn(2, 4, 10)
    order = torch.tensor([[0, 1, 2, 3], [3, 2, 1, 0]])

    x1 = perturb_at_location(x, order[:, 0], strategy="uniform_noise")
    x2 = perturb_at_location(x1, order[:, 1], strategy="uniform_noise")

    # First perturbed channel keeps its step-1 value
    assert torch.equal(x2[0, 0], x1[0, 0])
    assert torch.equal(x2[1, 3], x1[1, 3])
    # Second channel is newly perturbed, untouched channels stay original
    assert not torch.equal(x2[0, 1], x[0, 1])
    assert torch.equal(x2[0, 2], x[0, 2])
    # Input tensor is not modified in place
    assert not torch.equal(x1[0, 0], x[0, 0])


def test_morf_beats_lerf_on_known_importances():
    """Removing the most relevant channels first must give the larger AOPC."""
    model = LinearChannelModel([4.0, 3.0, 2.0, 1.0])
    x = torch.ones(3, 4, 10)
    attributions = torch.tensor([4.0, 3.0, 2.0, 1.0]).view(1, 4, 1).expand(3, 4, 10)

    morf_order = morf(attributions)
    lerf_order = lerf(attributions)
    assert torch.equal(morf_order[0], torch.tensor([0, 1, 2, 3]))

    morf_cumsum = perturbation_curve_area(model, x, morf_order, perturbation_strategy=zero_out)
    lerf_cumsum = perturbation_curve_area(model, x, lerf_order, perturbation_strategy=zero_out)

    # Logit drops after each step (x sums to 10 per channel): MoRF 40, 70, 90, 100
    expected_morf = torch.tensor([40.0, 110.0, 200.0, 300.0]).unsqueeze(1).expand(4, 3)
    assert torch.allclose(morf_cumsum, expected_morf)
    assert (morf_cumsum[-1] > lerf_cumsum[-1]).all()


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_binary_drops_are_oriented_to_predicted_class(sign):
    """Drops towards the opposite class are positive for both classes (paper Eq. 2)."""
    model = LinearChannelModel([sign * 2.0, sign * 1.0])
    x = torch.ones(2, 2, 5)
    order = torch.tensor([0, 1])

    cumsum = perturbation_curve_area(model, x, order, perturbation_strategy=zero_out)

    # |logit| = 15; removing channel 0 drops it by 10, then by 15 in total
    assert torch.allclose(cumsum, torch.tensor([[10.0, 10.0], [25.0, 25.0]]))
