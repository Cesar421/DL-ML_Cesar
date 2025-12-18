import torch
from torch.utils.data import DataLoader, TensorDataset

from dlcv3.model import (build_cnn_model, build_cnn_transformer_hybrid_model,
                         build_null_model)


def test_null_model():
    model = build_null_model()

    assert isinstance(model, torch.nn.Module)


def test_cnn_model():
    model = build_cnn_model()

    assert isinstance(model, torch.nn.Module)


def test_hybrid_model():
    model = build_cnn_transformer_hybrid_model()

    assert isinstance(model, torch.nn.Module)
