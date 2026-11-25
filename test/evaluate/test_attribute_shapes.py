
import pytest
import torch
import torch.nn as nn
import numpy as np
import pyarrow as pa
from src.evaluate.attribute import attribute_example

class RealTinyModel3D(nn.Module):
    def __init__(self, num_classes, num_time_steps):
        super().__init__()
        self.num_classes = num_classes
        self.num_time_steps = num_time_steps
        self.linear = nn.Linear(400, num_classes * num_time_steps)
        self.device = torch.device("cpu")
    def forward(self, x, chans):
        # x: (1, 4, 100)
        out = self.linear(x.view(1, -1))
        return out.view(1, self.num_classes, self.num_time_steps)

class RealTinyModel2D(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.linear = nn.Linear(400, num_classes)
        self.device = torch.device("cpu")
    def forward(self, x, chans):
        return self.linear(x.view(1, -1))

@pytest.fixture
def mock_example():
    # Example EEG data: (num_channels, time_points) = (4, 100)
    signal = torch.randn(4, 100)
    eeg_channel_indices = torch.tensor([0, 1, 2, 3])
    label = torch.tensor(1) # Global label
    return {
        "signal": signal,
        "eeg_channel_indices": eeg_channel_indices,
        "label": label
    }

def test_attribute_example_3d_output(mock_example):
    """Test attribute_example with 3D output (batch, classes, time_steps)."""
    num_classes = 2
    num_time_steps = 5
    model = RealTinyModel3D(num_classes, num_time_steps)
    
    # Call attribute_example
    result = attribute_example(0, mock_example, model)
    
    # Verify result
    assert isinstance(result, pa.RecordBatch)
    assert result.num_rows == num_time_steps * num_classes

def test_attribute_example_2d_output(mock_example):
    """Test attribute_example with 2D output (batch, classes) - Global mode."""
    num_classes = 2
    model = RealTinyModel2D(num_classes)
    
    # Call attribute_example
    result = attribute_example(0, mock_example, model)
    
    # Verify result
    assert isinstance(result, pa.RecordBatch)
    assert result.num_rows == num_classes
    assert result.to_pydict()["attributed_time_step"][0] == 0

def test_attribute_example_binary_2d(mock_example):
    """Test attribute_example with 2D output and 1 class (binary case)."""
    class BinaryModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(400, 1)
            self.device = torch.device("cpu")
        def forward(self, x, chans):
            return torch.sigmoid(self.linear(x.view(1, -1)))
            
    model = BinaryModel()
    # Force the weight to ensure prediction > 0.5
    with torch.no_grad():
        model.linear.weight.fill_(0.0)
        model.linear.bias.fill_(10.0)

        
    result = attribute_example(0, mock_example, model)
    
    assert result.num_rows == 1
    # prediction column should contain 1 (as integer or float that is 1.0)
    pred = result.to_pydict()["prediction"][0]
    assert int(pred) == 1

def test_attribute_example_3d_binary_subsampling(mock_example):
    """Test attribute_example with 3D output and subsampling."""
    num_classes = 1
    num_time_steps = 10
    subsample = 4
    
    class Binary3DModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(400, 10)
            self.device = torch.device("cpu")
        def forward(self, x, chans):
            out = torch.sigmoid(self.linear(x.view(1, -1)))
            return out.view(1, 1, 10)

    model = Binary3DModel()
    with torch.no_grad():
        model.linear.weight.fill_(1.0)
        model.linear.bias.fill_(1.0)
    
    result = attribute_example(0, mock_example, model, subsample_time_steps=subsample)
    
    # Should have subsample * num_classes rows
    assert result.num_rows == subsample

def test_attribute_example_3d_time_dependent_label():
    """Test attribute_example with 3D output and time-dependent labels."""
    num_classes = 2
    num_time_steps = 5
    model = RealTinyModel3D(num_classes, num_time_steps)
    
    signal = torch.randn(4, 100)
    eeg_channel_indices = torch.tensor([0, 1, 2, 3])
    label = torch.tensor([0, 1, 0, 1, 0])
    
    example = {
        "signal": signal,
        "eeg_channel_indices": eeg_channel_indices,
        "label": label
    }
    
    result = attribute_example(0, example, model)
    
    labels = result.to_pydict()["label"]
    time_steps = result.to_pydict()["attributed_time_step"]
    
    for l, t in zip(labels, time_steps):
        assert l == label[t].item()
