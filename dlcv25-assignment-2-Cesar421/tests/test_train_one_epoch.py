import torch
from torch import nn
import numpy as np
import random
from torch.utils.data import DataLoader, Subset, TensorDataset
import torch.optim as optim
from dlcv.training import train_one_epoch
from torchvision.datasets import MNIST
from torchvision import transforms

def test_train_one_epoch():
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    mnist = MNIST(root="data", train=True, download=True, transform=transforms.ToTensor())
    mnist_subset = Subset(mnist, indices=range(10))
    data_loader = DataLoader(mnist_subset, batch_size=2, shuffle=True)

    model = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    train_one_epoch(model, data_loader, criterion, optimizer, "cpu")

    with torch.no_grad():
        input_tensor = torch.randn((1,1,28,28))
        output_tensor = model(input_tensor)

    assert torch.allclose(output_tensor, torch.tensor([[-0.2163,  0.5442, -0.4179,  1.2289, -0.0313,  0.3519, -0.0281, -0.1910,
         -0.1538,  0.0922]]), atol=1e-04), "Either you ran this test locally or the there is a mistake in your train_one_epoch code."

def test_train_one_epoch_false():
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    mnist = MNIST(root="data", train=True, download=True, transform=transforms.ToTensor())
    mnist_subset = Subset(mnist, indices=range(10))
    data_loader = DataLoader(mnist_subset, batch_size=2, shuffle=True)

    model = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    train_one_epoch(model, data_loader, criterion, optimizer, "cpu")

    with torch.no_grad():
        input_tensor = torch.randn((1,1,28,28))
        output_tensor = model(input_tensor)

    assert not torch.allclose(output_tensor, torch.tensor([[ 0.0679,  0.0464, -0.3475,  0.5245, -0.5716,  0.2021,  0.1776,
         0.0243, 0.2569, -0.1695]]), atol=1e-04), "Either you ran this test locally or the there is a mistake in your train_one_epoch code."

def test_train_one_epoch_ci():
    # set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    # set device
    device = "cpu"

    # create simple test dataset
    num_samples = 10
    images = torch.randn(num_samples, 3, 16, 16)
    labels = torch.randint(0, 5, (num_samples,))
    dataset = TensorDataset(images, labels)
    data_loader = DataLoader(dataset, batch_size=10)

    # init model parameters
    model = nn.Sequential(nn.Flatten(), nn.Linear(3*16*16, 5)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    # train epoch
    epoch_loss = train_one_epoch(model, data_loader, criterion, optimizer, device)

    with torch.no_grad():
        input_tensor = torch.randn((1,3,16,16)).to(device)
        output_tensor = model(input_tensor)

    assert torch.allclose(output_tensor, torch.tensor([[ 0.1834,  0.3355, -0.2396, -0.4707,  0.6003]]), atol=1e-04), \
        "Either you ran this test locally or the there is a mistake in your train_one_epoch code."