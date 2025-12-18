import numpy as np
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pytest
from dlcv.training import train_and_evaluate_model

def test_train_and_evaluate_model_with_scheduler():
    # Set device and random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
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
    optimizer = optim.SGD(model.parameters(), lr=0.1)  # higher initial lr for more noticeable change

    # Define a mock scheduler that reduces lr based on test loss
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2)

    # Train the model
    num_epochs = 5
    train_losses, test_losses, test_accuracies = train_and_evaluate_model(
        model, train_loader, test_loader, criterion, optimizer, num_epochs, device, scheduler=scheduler
    )

    # Check if the learning rate has been reduced by the scheduler
    for i, param_group in enumerate(optimizer.param_groups):
        print(f"Final learning rate for group {i}: {param_group['lr']}")
        assert param_group['lr'] < 0.1, "Scheduler should have reduced the learning rate."
