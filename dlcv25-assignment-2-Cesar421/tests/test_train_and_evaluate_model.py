import numpy as np
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pytest
from dlcv.training import train_and_evaluate_model

def test_train_and_evaluate_model_simple():
    # set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    # set device
    device = "cpu"

    # create simple test dataset
    num_samples = 10
    images_train = torch.randn(num_samples, 3, 16, 16)
    labels_train = torch.randint(0, 5, (num_samples,))
    dataset_train = TensorDataset(images_train, labels_train)
    train_loader = DataLoader(dataset_train, batch_size=10)

    images_test = torch.randn(num_samples, 3, 16, 16)
    labels_test = torch.randint(0, 5, (num_samples,))
    dataset_test = TensorDataset(images_test, labels_test)
    test_loader = DataLoader(dataset_test, batch_size=10)

    # init model parameters
    model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 16 * 16, 5)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    # Train and evaluate the model
    num_epochs = 5
    train_losses, test_losses, test_accuracies = train_and_evaluate_model(model, train_loader, test_loader, criterion,
                                                                          optimizer, num_epochs, device)


    # Check the length of the returned lists
    assert len(train_losses) == num_epochs, "The length of train_losses should be equal to num_epochs."
    assert len(test_losses) == num_epochs, "The length of test_losses should be equal to num_epochs."
    assert len(test_accuracies) == num_epochs, "The length of test_accuracies should be equal to num_epochs."

    # Check if the returned values are floats
    assert all(isinstance(value, float) for value in train_losses), "All values in train_losses should be floats."
    assert all(isinstance(value, float) for value in test_losses), "All values in test_losses should be floats."
    assert all(isinstance(value, float) for value in test_accuracies), "All values in test_accuracies should be floats."

def test_train_and_evaluate_model_parameters_update():
    # set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    # set device
    device = "cpu"

    # create simple test dataset
    num_samples = 10
    images_train = torch.randn(num_samples, 3, 16, 16)
    labels_train = torch.randint(0, 5, (num_samples,))
    dataset_train = TensorDataset(images_train, labels_train)
    train_loader = DataLoader(dataset_train, batch_size=10)

    images_test = torch.randn(num_samples, 3, 16, 16)
    labels_test = torch.randint(0, 5, (num_samples,))
    dataset_test = TensorDataset(images_test, labels_test)
    test_loader = DataLoader(dataset_test, batch_size=10)

    # init model parameters
    model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 16 * 16, 5)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    # Store initial model parameters for comparison
    initial_parameters = [param.clone().detach() for param in model.parameters()]

    # Train and evaluate the model
    num_epochs = 5
    _, _, _ = train_and_evaluate_model(model, train_loader, test_loader, criterion,
                                                                          optimizer, num_epochs, device)

    # Check if model parameters have been updated
    updated_parameters = [param for param in model.parameters()]
    assert any([not torch.equal(initial_param, updated_param) for initial_param, updated_param in
                zip(initial_parameters, updated_parameters)]), "Model parameters should be updated after training."

def test_train_and_evaluate_model_ci():
    # set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    # set device
    device = "cpu"

    # create simple test dataset
    num_samples = 10
    images_train = torch.randn(num_samples, 3, 16, 16)
    labels_train = torch.randint(0, 5, (num_samples,))
    dataset_train = TensorDataset(images_train, labels_train)
    train_loader = DataLoader(dataset_train, batch_size=10)

    images_test = torch.randn(num_samples, 3, 16, 16)
    labels_test = torch.randint(0, 5, (num_samples,))
    dataset_test = TensorDataset(images_test, labels_test)
    test_loader = DataLoader(dataset_test, batch_size=10)

    # init model parameters
    model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 16 * 16, 5)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    # Train and evaluate the model
    num_epochs = 5
    train_losses, test_losses, test_accuracies = train_and_evaluate_model(model, train_loader, test_loader, criterion,
                                                                          optimizer, num_epochs, device)

    assert np.allclose(np.array(train_losses), np.array([1.8410866260528564, 1.771812081336975, 1.7045332193374634, 1.6392799615859985, 1.5760761499404907]), atol=1e-04), "Either you ran this test locally or the training loss calculation in the train_and_evaluate_model function is wrong."
    assert np.allclose(np.array(test_losses), np.array([1.6680980920791626, 1.6698821783065796, 1.6716432571411133, 1.6733801364898682, 1.675092101097107]),atol=1e-04), "Either you ran this test locally or the test loss calculation in the train_and_evaluate_model function is wrong."
    assert np.all( np.array(test_accuracies) == np.array([0.3, 0.3, 0.3, 0.3, 0.3])), "Either you ran this test locally or the accuracy calculation in the train_and_evaluate_model function is wrong."