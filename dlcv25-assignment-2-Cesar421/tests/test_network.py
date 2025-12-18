import torch
import pytest
from dlcv.models import CustomizableNetwork  # Adjust the import according to your project structure

# Test output shapes for different layer configurations
@pytest.mark.parametrize("conv_layers, filters_conv1, filters_conv2, filters_conv3, dense_units, expected_output_size", [
    (2, 16, 32, 64, 128, (10,)),  # Expected size of the output tensor for 2 conv layers
    (3, 16, 32, 64, 128, (10,))   # Expected size of the output tensor for 3 conv layers
])
def test_output_shape(conv_layers, filters_conv1, filters_conv2, filters_conv3, dense_units, expected_output_size):
    net = CustomizableNetwork(conv_layers, filters_conv1, filters_conv2, filters_conv3, dense_units)
    input_tensor = torch.randn(1, 3, 92, 92)  # Batch size of 1, 3 channels, 92x92 images
    output = net(input_tensor)
    assert output.shape == (1,) + expected_output_size, f"Output shape {output.shape} does not match expected {expected_output_size}"

# Test the layers are correctly created in the model
@pytest.mark.parametrize("conv_layers, filters_conv1, filters_conv2, filters_conv3, dense_units, expected_layers", [
    (2, 16, 32, 64, 128, ['conv1', 'bn1', 'pool1', 'conv2', 'bn2', 'pool2', 'fc1', 'fc2']),
    (3, 16, 32, 64, 128, ['conv1', 'bn1', 'pool1', 'conv2', 'bn2', 'pool2', 'conv3', 'bn3', 'pool3', 'fc1', 'fc2'])
])
def test_layers_existence(conv_layers, filters_conv1, filters_conv2, filters_conv3, dense_units, expected_layers):
    net = CustomizableNetwork(conv_layers, filters_conv1, filters_conv2, filters_conv3, dense_units)
    actual_layers = list(net.layers.keys())
    for layer in expected_layers:
        assert layer in actual_layers, f"Layer {layer} is expected but not found in the model."

# Test output shape for different input sizes
@pytest.mark.parametrize("conv_layers, filters_conv1, filters_conv2, filters_conv3, dense_units, height, width", [
    (2, 16, 32, 64, 128, 92, 92),
    (2, 16, 32, 64, 128, 100, 100)
])
def test_different_input_sizes(conv_layers, filters_conv1, filters_conv2, filters_conv3, dense_units, height, width):
    net = CustomizableNetwork(conv_layers, filters_conv1, filters_conv2, filters_conv3, dense_units)
    input_tensor = torch.randn(1, 3, height, width)  # Test with different image sizes
    output = net(input_tensor)
    assert output.shape == (1, 10), f"Output shape for input size {height}x{width} is incorrect."

