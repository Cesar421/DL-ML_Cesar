import pytest
import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader

from dlcv.utils import freeze_layers

def test_freeze_layers():
    # Define a simple network with identifiable layers
    class SimpleNetwork(nn.Module):
        def __init__(self):
            super(SimpleNetwork, self).__init__()
            self.layer1 = nn.Linear(2, 2)
            self.layer2 = nn.Linear(2, 2)

        def forward(self, x):
            x = self.layer1(x)
            x = self.layer2(x)
            return x

    # Initialize the model
    model = SimpleNetwork()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    criterion = nn.MSELoss()

    # Freeze the first layer
    freeze_layers(model, ['layer1'])

    # Check initial state of requires_grad
    assert model.layer1.weight.requires_grad == False, "Layer1 should be frozen."
    assert model.layer2.weight.requires_grad == True, "Layer2 should be trainable."

    # Record the initial weights before training
    initial_weights_layer1 = model.layer1.weight.clone().detach()
    initial_weights_layer2 = model.layer2.weight.clone().detach()

    # Create some mock data
    inputs = torch.randn(10, 2)
    targets = torch.randn(10, 2)
    dataset = TensorDataset(inputs, targets)
    loader = DataLoader(dataset, batch_size=5)

    # Train for one epoch
    model.train()
    for data, target in loader:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

    # Check if the weights of layer1 remained the same
    assert torch.all(torch.eq(model.layer1.weight, initial_weights_layer1)), "Weights of layer1 should not have changed."

    # Check if the weights of layer2 have been updated
    assert not torch.all(torch.eq(model.layer2.weight, initial_weights_layer2)), "Weights of layer2 should have changed due to training."
