from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from dlcv3.main import summarize_model


def test_train_null_model():
    model = nn.Sequential(nn.Tanh())
    summarize_model(model, batch_size=1)
