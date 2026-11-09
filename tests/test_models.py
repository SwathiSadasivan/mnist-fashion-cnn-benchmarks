import torch
import pytest
from src.models.cnn import CustomCNN
from src.models.baselines import BaselineFNN


def test_custom_cnn_output_shape():
    """Verify that CustomCNN outputs expected tensor shape (batch_size, num_classes)."""
    batch_size = 8
    num_classes = 10
    
    # Create dummy input tensor matching 28x28 grayscale image dimensions (batch, channel, height, width)
    dummy_input = torch.randn(batch_size, 1, 28, 28)
    model = CustomCNN(dropout_rate=0.25, num_classes=num_classes)
    
    output = model(dummy_input)
    
    # Check output shape
    assert output.shape == (batch_size, num_classes), f"Expected shape {(batch_size, num_classes)}, got {output.shape}"


def test_baseline_fnn_output_shape():
    """Verify that BaselineFNN outputs expected tensor shape (batch_size, num_classes)."""
    batch_size = 16
    num_classes = 10
    
    dummy_input = torch.randn(batch_size, 1, 28, 28)
    model = BaselineFNN(input_dim=784, hidden_dim=128, num_classes=num_classes)
    
    output = model(dummy_input)
    
    assert output.shape == (batch_size, num_classes), f"Expected shape {(batch_size, num_classes)}, got {output.shape}"


def test_custom_cnn_forward_pass_no_nan():
    """Verify forward pass completes without returning NaN or Inf values."""
    dummy_input = torch.randn(4, 1, 28, 28)
    model = CustomCNN()
    
    output = model(dummy_input)
    
    assert not torch.isnan(output).any(), "Model output contains NaN values"
    assert not torch.isinf(output).any(), "Model output contains Inf values"