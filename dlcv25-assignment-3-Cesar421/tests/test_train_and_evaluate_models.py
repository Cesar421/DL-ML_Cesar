from torch.utils.data import DataLoader, TensorDataset

from dlcv3.main import (train_and_evaluate_cnn_model,
                        train_and_evaluate_hybrid_model,
                        train_and_evaluate_null_model)


def test_train_null_model():
    train_and_evaluate_null_model(1, 1.0e-3, debug_run=True)


def test_train_cnn_model():
    train_and_evaluate_cnn_model(1, 1.0e-3, debug_run=True)


def test_train_hybrid_model():
    train_and_evaluate_hybrid_model(1, 1.0e-3, debug_run=True)
