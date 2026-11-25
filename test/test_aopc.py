import torch
import torch.nn as nn
import pytest

from src.evaluate.aopc import (
    perturb_at_location,
    perturbation_curve_area,
    evaluate_aopc_for_dataset,
    aopcs_at_step,
    morf,
    lerf,
    random_order,
    replace_with_uniform_noise,
    replace_with_global_mean,
)

class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)
        
    def forward(self, x):
        # x is (batch, channels, time)
        # Flatten time and pass to linear layer
        x = x.mean(dim=2) # (batch, channels)
        return self.fc(x)


@pytest.fixture
def dummy_setup():
    torch.manual_seed(42)
    model = DummyModel()
    batch_size = 6
    num_channels = 10
    time = 20
    samples = torch.randn(batch_size, num_channels, time)
    return model, samples, batch_size, num_channels, time


def test_perturb_functions():
    orig = torch.tensor([1.0, 2.0, 3.0])
    
    # Global mean should replace all with 2.0
    res_mean = replace_with_global_mean(orig)
    assert torch.allclose(res_mean, torch.tensor([2.0, 2.0, 2.0]))
    
    # Uniform noise should be bounded by min and max
    res_noise = replace_with_uniform_noise(orig)
    assert (res_noise >= 1.0).all()
    assert (res_noise <= 3.0).all()


def test_perturbation_curve_global_order(dummy_setup):
    """Tests batchwise perturbing (same order for all items in batch)"""
    model, samples, batch_size, num_channels, time = dummy_setup
    
    # 1D global order
    global_order = torch.arange(num_channels)
    
    cumsum, logits = perturbation_curve_area(
        model, samples, global_order, 
        perturbation_strategy='global_mean', return_logits=True
    )
    
    assert cumsum.shape == (10, batch_size)
    assert logits.shape == (11, batch_size, 2)


def test_perturbation_curve_itemwise_order(dummy_setup):
    """Tests itemwise perturbing (different order for each item in batch)"""
    model, samples, batch_size, num_channels, time = dummy_setup
    
    # 2D per-sample order using morf
    attributions = torch.randn(batch_size, num_channels, time)
    itemwise_order = morf(attributions)
    
    assert itemwise_order.shape == (batch_size, 10)
    
    cumsum, logits = perturbation_curve_area(
        model, samples, itemwise_order, 
        perturbation_strategy='uniform_noise', return_logits=True
    )
    
    assert cumsum.shape == (10, batch_size)
    assert logits.shape == (11, batch_size, 2)


def test_evaluate_aopc_for_dataset(dummy_setup):
    """Tests the high-level dataset wrapper"""
    model, samples, batch_size, num_channels, time = dummy_setup
    
    attributions = torch.randn(batch_size, num_channels, time)
    itemwise_order = morf(attributions)
    
    # Run with small batch size to force dataloader chunking
    wrapper_batch_size = 2
    from torch.utils.data import TensorDataset, DataLoader
    dataset = TensorDataset(samples)
    dataloader = DataLoader(dataset, batch_size=wrapper_batch_size, shuffle=False)
    
    cumsum, logits = evaluate_aopc_for_dataset(
        model=model,
        dataloader=dataloader,
        order=itemwise_order,
        perturbation_strategy='uniform_noise',
        device="cpu",
        return_example_logits=True
    )
    
    assert cumsum.shape == (10, batch_size)
    assert logits.shape == (11, wrapper_batch_size, 2)
    
    # Also test global order handling
    global_order = torch.arange(num_channels)
    cumsum_global = evaluate_aopc_for_dataset(
        model=model,
        dataloader=dataloader,
        order=global_order,
        perturbation_strategy='global_mean',
        device="cpu",
        return_example_logits=False
    )
    assert cumsum_global.shape == (10, batch_size)


def test_aopcs_computation(dummy_setup):
    """Tests AOPC computation from cumsum"""
    model, samples, batch_size, num_channels, time = dummy_setup
    
    itemwise_order = random_order(batch_size, num_channels, samples.device)
    
    cumsum = perturbation_curve_area(
        model, samples, itemwise_order, 
        perturbation_strategy='uniform_noise', return_logits=False
    )
    
    step = 4
    aopc_at_4 = aopcs_at_step(cumsum, step)
    
    assert aopc_at_4.shape == (batch_size,)
    expected_aopc = cumsum[step] / (step + 2)  # Eq. 2: L = step + 1 perturbations, divided by L + 1
    assert torch.allclose(aopc_at_4, expected_aopc)


def test_order_functions():
    batch_size = 2
    num_channels = 3
    time = 4
    
    attrs = torch.zeros(batch_size, num_channels, time)
    attrs[:, 0, :] = 1.0
    attrs[:, 1, :] = 5.0
    attrs[:, 2, :] = 10.0
    
    morf_ord = morf(attrs)
    assert torch.all(morf_ord == torch.tensor([[2, 1, 0], [2, 1, 0]]))
    
    lerf_ord = lerf(attrs)
    assert torch.all(lerf_ord == torch.tensor([[0, 1, 2], [0, 1, 2]]))
    
    rand_ord = random_order(batch_size, num_channels, attrs.device)
    assert rand_ord.shape == (batch_size, num_channels)
    assert set(rand_ord[0].tolist()) == {0, 1, 2}
