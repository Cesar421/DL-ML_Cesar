import numpy as np
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pytest
from dlcv.training import evaluate_one_epoch

def test_evaluate_one_epoch_simple():
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    # Set device
    device = "cpu"

    # Create simple test dataset
    num_samples = 10
    images = torch.randn(num_samples, 3, 16, 16)
    labels = torch.randint(0, 5, (num_samples,))
    dataset = TensorDataset(images, labels)
    data_loader = DataLoader(dataset, batch_size=10)

    # Initialize model parameters
    model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 16 * 16, 5)).to(device)
    criterion = nn.CrossEntropyLoss()

    # Evaluate epoch
    epoch_loss, accuracy = evaluate_one_epoch(model, data_loader, criterion, device)

    # Check if the returned loss is a float
    assert isinstance(epoch_loss, float), "The returned epoch loss should be a float."

    # Check if the returned loss is non-negative
    assert epoch_loss >= 0, "The returned epoch loss should be non-negative."

    # Check if the returned accuracy is a float
    assert isinstance(accuracy, float), "The returned accuracy should be a float."

    # Check if the returned accuracy is between 0 and 1
    assert 0 <= accuracy <= 1, "The returned accuracy should be between 0 and 1."

def test_evaluate_one_epoch_accuracy():
    # Set device
    device = "cpu"

    # Create simple test dataset
    num_samples = 10
    images = torch.randn(num_samples, 3, 16, 16)
    labels = torch.zeros(num_samples, dtype=torch.long)  # All labels are 0
    dataset = TensorDataset(images, labels)
    data_loader = DataLoader(dataset, batch_size=10)

    # Initialize model parameters
    model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 16 * 16, 5)).to(device)
    model[1].weight.data.zero_()  # Set weights to zeros
    model[1].bias.data.zero_()  # Set biases to zeros

    # Since all weights and biases are zero, the output should be a tensor of zeros,
    # which means the predicted class will always be 0, and the accuracy should be 1.0.

    criterion = nn.CrossEntropyLoss()

    # Evaluate epoch
    epoch_loss, accuracy = evaluate_one_epoch(model, data_loader, criterion, device)

    # Check if the accuracy is 1.0
    assert accuracy == 1.0, "The accuracy should be 1.0 for this test case."