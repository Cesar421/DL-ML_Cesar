import pytest
import torch
from mynet.model import ThreeLayerFullyConnectedNetwork

def test_layer_types_in_order():
    model = ThreeLayerFullyConnectedNetwork()
    input_tensor = torch.randn(16, 1, 28, 28)

    # List to store modules and outputs after each operation
    layer_info = []

    # Register a hook for each module
    for module in model.children():
        module.register_forward_hook(lambda module, input, output, mod=module: layer_info.append((mod, output)))

    # Execute forward pass
    with torch.no_grad():  # We do not need to track gradients here
        model(input_tensor)

    # Assertions based on expected types
    assert isinstance(layer_info[0][0], torch.nn.Flatten), "First layer should be Flatten"
    assert isinstance(layer_info[1][0], torch.nn.Linear), "Second layer should be a Linear layer"
    assert isinstance(layer_info[2][0], torch.nn.ReLU), "Third layer should be a ReLU activation"
    assert isinstance(layer_info[3][0], torch.nn.Linear), "Fourth layer should be a Linear layer"
    assert isinstance(layer_info[4][0], torch.nn.ReLU), "Fifth layer should be a ReLU activation"
    assert isinstance(layer_info[5][0], torch.nn.Linear), "Sixth layer should be a Linear layer"

def test_layer_operations_order():
    model = ThreeLayerFullyConnectedNetwork()
    input_tensor = torch.randn(16, 1, 28, 28)

    # List to store outputs after each operation
    layer_outputs = []

    # Register a hook for each module
    for module in model.children():
        module.register_forward_hook(lambda module, input, output, mod=module: layer_outputs.append((mod, output)))

    # Execute forward pass
    with torch.no_grad():  # We do not need to track gradients here
        model(input_tensor)

    # Assertions based on expected transformations
    assert layer_outputs[0][1].shape == (16, 784), "Output after Flatten should be reshaped to (batch_size, 784)"
    assert layer_outputs[1][1].shape == (16, 32), "Output after first Linear should be (batch_size, 32)"
    assert layer_outputs[3][1].shape == (16, 64), "Output after second Linear should be (batch_size, 64)"
    assert layer_outputs[5][1].shape == (16, 10), "Output after third Linear should be (batch_size, 10)"

def test_forward_output():
    model = ThreeLayerFullyConnectedNetwork()
    input_tensor = torch.randn(32, 1, 28, 28)  # Simulate a batch of 32 images
    output = model(input_tensor)
    assert output.shape == (32, 10), "The output shape should be (batch_size, 10)."
