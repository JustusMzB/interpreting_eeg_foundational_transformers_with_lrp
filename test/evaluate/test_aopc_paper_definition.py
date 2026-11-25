"""Tests pinning the AOPC implementation to the definition in the paper (App. E.1, Eq. 2)."""
import torch

from evaluate.aopc import (
    aopc_per_sample,
    aopcs_at_step,
    evaluate_aopc_for_dataset,
    legacy_running_mean_score,
    perturb_at_location,
    perturbation_curve_area,
    random_order,
)


class ChannelSum(torch.nn.Module):
    """Binary model with a single logit: f(x) = sum over time of channel 0 minus channel 1."""

    def forward(self, x):
        return (x[:, 0].sum(-1) - x[:, 1].sum(-1)).unsqueeze(1)


def _samples():
    # sample 0: logit > 0 (predicted class 1); sample 1: logit < 0 (predicted class 0)
    x = torch.zeros(2, 3, 4)
    x[0, 0] = torch.tensor([1.0, 2.0, 3.0, 2.0])   # channel 0 sum 8, mean 2
    x[1, 1] = torch.tensor([2.0, 2.0, 1.0, 3.0])   # channel 1 sum 8, mean 2
    x[0, 0, 0] += 0.0
    return x


def test_sign_comes_from_prediction_not_label():
    """Eq. 2 multiplies by sign(f(x)); dataset labels must not be used by default."""
    model = ChannelSum()
    x = _samples()
    # Remove the channel that drives each prediction: ch 0 for sample 0, ch 1 for sample 1.
    order = torch.tensor([[0], [1]])
    drop = lambda t: torch.zeros_like(t)
    cs = perturbation_curve_area(model, x, order, perturbation_strategy=drop)
    # Both lose their full evidence (8) -> positive drop for both, regardless of class.
    assert torch.allclose(cs[0], torch.tensor([8.0, 8.0]))

    # Wrong labels must not flip the result in the dataset-level evaluation (default).
    batch = {"signal": x, "label": torch.tensor([0, 1])}
    cs_ds, _ = evaluate_aopc_for_dataset(model, [batch], order, perturbation_strategy=drop, device="cpu")
    assert torch.allclose(cs_ds[0], torch.tensor([8.0, 8.0]))


def test_eq2_known_answer():
    """AOPC = sum_{s=1..L} (f(x) - f(x_s)) / (L + 1)."""
    cumsum = torch.tensor([[1.0], [3.0], [6.0]])  # drops 1, 2, 3 over L = 3 steps
    assert torch.allclose(aopc_per_sample(cumsum), torch.tensor([6.0 / 4]))
    assert torch.allclose(aopcs_at_step(cumsum, 0), torch.tensor([1.0 / 2]))
    assert torch.allclose(aopcs_at_step(cumsum, 2), torch.tensor([6.0 / 4]))
    # The pre-fix score was mean_k(cumsum_k / k) = (1 + 1.5 + 2) / 3
    assert abs(legacy_running_mean_score(cumsum) - 1.5) < 1e-6


def test_global_order_replacement_is_per_sample():
    """With a global order, each sample's channel is replaced by its own channel mean."""
    x = _samples()
    out = perturb_at_location(x, torch.tensor(0), strategy="channel_mean")
    assert torch.allclose(out[0, 0], torch.full((4,), 2.0))   # sample 0's own mean
    assert torch.allclose(out[1, 0], torch.zeros(4))          # sample 1's own mean (0), not the batch mean


def test_global_order_independent_of_batch_composition():
    model = ChannelSum()
    torch.manual_seed(0)
    x = torch.randn(6, 3, 4)
    order = torch.tensor([0, 1])
    together = perturbation_curve_area(model, x, order, perturbation_strategy="global_mean")
    alone = torch.cat([perturbation_curve_area(model, x[i:i + 1], order, perturbation_strategy="global_mean")
                       for i in range(len(x))], dim=1)
    assert torch.allclose(together, alone, atol=1e-5)


def test_random_order_is_seedable():
    a = random_order(5, 8, "cpu", generator=torch.Generator().manual_seed(3))
    b = random_order(5, 8, "cpu", generator=torch.Generator().manual_seed(3))
    assert torch.equal(a, b)


def test_alignment_check_rejects_shuffled_attributions():
    import pytest
    from datasets import Dataset
    from evaluate.aopc import check_attribution_alignment

    ds = Dataset.from_dict({"label": [0, 1, 1, 0], "subject_id": ["S1", "S1", "S2", "S2"],
                            "run_file": ["a", "a", "b", "b"]})
    idx = [0, 1, 2, 3]
    check_attribution_alignment(idx, [0, 1, 1, 0], ds, sample_keys=["S1/a", "S1/a", "S2/b", "S2/b"])
    check_attribution_alignment(idx, [0, 1, 1, 0], ds)  # legacy files without keys
    with pytest.raises(ValueError, match="labels differ"):
        check_attribution_alignment(idx, [1, 0, 1, 0], ds)
    with pytest.raises(ValueError, match="sample keys"):
        check_attribution_alignment(idx, [0, 1, 1, 0], ds, sample_keys=["S2/b", "S1/a", "S2/b", "S1/a"])


def test_orderings():
    from evaluate.aopc import per_sample_order, global_order
    # 2 samples, 3 channels, 2 time points; sample 0 predicts class 1, sample 1 class 0
    attr = torch.tensor([[[3.0, 0.0], [-5.0, 0.0], [1.0, 0.0]],
                         [[2.0, 0.0], [-4.0, 0.0], [0.5, 0.0]]])
    logits = torch.tensor([1.0, -1.0])
    assert per_sample_order(attr, logits, "signed")[:, 0].tolist() == [0, 0]
    assert per_sample_order(attr, logits, "magnitude")[:, 0].tolist() == [1, 1]
    # prediction-oriented: sample 0 keeps the most positive (ch 0), sample 1 takes the most negative (ch 1)
    assert per_sample_order(attr, logits, "prediction")[:, 0].tolist() == [0, 1]
    assert global_order(attr, mode="magnitude")[0].item() == 1          # |mean| = [2.5, 4.5, 0.75]
    assert global_order(attr, mode="signed")[0].item() == 0
    assert global_order(attr, logits, mode="prediction")[0].item() == 0  # mean of [3,-5,1] and [-2,4,-0.5]


def test_global_magnitude_is_visualised_map():
    from evaluate.aopc import global_order
    torch.manual_seed(1)
    attr = torch.randn(7, 5, 11)
    visualised = attr.mean(0).sum(-1)  # the map shown in the figures (signed)
    assert torch.equal(global_order(attr, mode="magnitude"), torch.argsort(visualised.abs(), descending=True))
