import pytest
import atexit
import torch
from torchvision import transforms
from torchvision.datasets import MNIST
import os
from mynet.dataset import CustomMNISTDataset

@pytest.fixture()
def custom_mnist_dataset():
    if os.path.exists("data/MNIST"):
        return CustomMNISTDataset(root="data/MNIST", subset="train", transformation= transforms.ToTensor())
    else:
        atexit.register(report)

def test_dataset_length(custom_mnist_dataset):
    mnist_dataset = MNIST(root="data", train=True, download=True)
    assert len(custom_mnist_dataset) == len(mnist_dataset), f"Expected dataset length: {len(mnist_dataset)}, but got: {len(custom_mnist_dataset)}"

def test_dataset_getitem(custom_mnist_dataset):
    image, label = custom_mnist_dataset[0]
    assert isinstance(image, torch.Tensor), "Expected image to be a torch.Tensor"
    assert image.dtype == torch.float32, "Expected image to have dtype torch.float32"
    assert image.size() == (1, 28, 28), "Expected image size to be (1, 28, 28)"
    assert isinstance(label, int), "Expected label to be an int"

def test_dataset_transforms(custom_mnist_dataset):
    image, _ = custom_mnist_dataset[0]
    assert image.max() <= 1.0 and image.min() >= 0.0, "Expected pixel values to be in range [0, 1]"

def report():
    # This test checks if you are using the correct
    print("Your tests are failing because you either did not:")
    print("\t- Run the test from the wrong working directory. Your current working directory is {}, but it should be the repository root.".format(os.getcwd()))
    print("\t- You did not use the utils script to download MNIST to data/MNIST .")

