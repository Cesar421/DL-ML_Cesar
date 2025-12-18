import numpy as np
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pytest
from dlcv.training import train_and_evaluate_model

def test_train_and_evaluate_model_early_stopping():
    # Set device and random seeds for reproducibility
    seed = 41
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    device = "cpu"

    # Create simple test datasets
    num_samples = 10
    dataset_train = TensorDataset(torch.randn(num_samples, 3, 16, 16), torch.randint(0, 5, (num_samples,)))
    train_loader = DataLoader(dataset_train, batch_size=10)
    dataset_test = TensorDataset(torch.randn(num_samples, 3, 16, 16), torch.randint(0, 5, (num_samples,)))
    test_loader = DataLoader(dataset_test, batch_size=10)

    # Define model, criterion, and optimizer
    model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 16 * 16, 5)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    # Train the model with early stopping enabled
    num_epochs = 10
    train_losses, test_losses, test_accuracies = train_and_evaluate_model(
        model, train_loader, test_loader, criterion, optimizer, num_epochs, device, early_stopping=True
    )

    # Check if the training stopped early
    assert len(train_losses) < num_epochs, "Training should have stopped early due to non-decreasing test loss."
