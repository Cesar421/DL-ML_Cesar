from torch.utils.data import DataLoader, TensorDataset

from dlcv3.dataset import get_cifar10_train_transforms


def test_train_null_model():
    get_cifar10_train_transforms()
