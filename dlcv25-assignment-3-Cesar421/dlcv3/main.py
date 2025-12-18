import argparse
import os
import sys

# Add parent directory to path to allow imports when running from dlcv3/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (EarlyStopping, LearningRateMonitor,
                                         ModelCheckpoint)
from pytorch_lightning.loggers import TensorBoardLogger
from torchinfo import summary

from dlcv3.model import (CIFAR10Classifier, CIFAR10DataModule, build_cnn_model,
                         build_cnn_transformer_hybrid_model, build_null_model)


def summarize_model(model, batch_size):
    input_shape = (batch_size, 3, 32, 32)

    # Summarize model using torchinfo
    summary(
        model,
        input_size=input_shape,
        col_names=["input_size", "output_size", "num_params", "trainable"],
        depth=5,
    )


def _create_cifar10_datamodule(batch_size, val_split=0.2):
    return CIFAR10DataModule(
        data_dir="data_dir",
        batch_size=batch_size,
<<<<<<< HEAD
        num_workers=0,  # Set to 0 on Windows to avoid multiprocessing issues
=======
        num_workers=8,  # Set to 0 on Windows to avoid multiprocessing issues
>>>>>>> 762751f54dc0e46e9b01d64bf3d7bc9614d80e56
        img_size=32,
        val_split=val_split,
    )


def _create_logger(experiment_name):
    return TensorBoardLogger(
        save_dir="tb_logs",
        name=experiment_name,
    )


def _create_callbacks(experiment_name):
    checkpoint_cb = ModelCheckpoint(
        dirpath=os.path.join("checkpoints", experiment_name),
        save_top_k=1,
        monitor="val/acc",
        mode="max",
        filename="{epoch:02d}-{val/acc:.4f}",
        auto_insert_metric_name=False,
    )

    early_stop_cb = EarlyStopping(
        monitor="val/loss",
        mode="min",
        patience=20,
        verbose=True,
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    return [checkpoint_cb, early_stop_cb, lr_monitor]


def _get_trainer_device_config(debug_run=False):
    if debug_run:
        accelerator = "cpu"
        devices = 1
    else:
        accelerator = "gpu" if torch.cuda.is_available() else "cpu"
        devices = torch.cuda.device_count() if accelerator == "gpu" else 1

    return {"accelerator": accelerator, "devices": devices}


def _create_trainer(logger, callbacks, debug_run=False):
    device_cfg = _get_trainer_device_config(debug_run=debug_run)

    return Trainer(
        max_time="00:00:20:00",
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=1,
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        fast_dev_run=debug_run,
        **device_cfg,
    )


def _train_and_evaluate(
        model_builder,
        experiment_name,
        batch_size,
        learning_rate,
        debug_run=False,
        summarize_only=False,
):
    if summarize_only:
        model = model_builder()
        summarize_model(model, batch_size)
        exit(0)

    dm = _create_cifar10_datamodule(batch_size=batch_size)

    model = model_builder()

    summarize_model(model, dm.batch_size)

    lit_model = CIFAR10Classifier(model=model, lr=learning_rate)

    logger = _create_logger(experiment_name)
    callbacks = _create_callbacks(experiment_name)
    trainer = _create_trainer(logger=logger, callbacks=callbacks, debug_run=debug_run)

    trainer.fit(lit_model, datamodule=dm)
    trainer.validate(lit_model, datamodule=dm, verbose=True)
    trainer.test(lit_model, datamodule=dm, verbose=True)


def train_and_evaluate_null_model(batch_size, learning_rate, debug_run=False):
    _train_and_evaluate(
        model_builder=build_null_model,
        experiment_name="experiment-null-model",
        batch_size=batch_size,
        learning_rate=learning_rate,
        debug_run=debug_run,
    )


def train_and_evaluate_cnn_model(batch_size, learning_rate, debug_run=False):
    _train_and_evaluate(
        model_builder=build_cnn_model,
        experiment_name="experiment-cnn-model",
        batch_size=batch_size,
        learning_rate=learning_rate,
        debug_run=debug_run,
    )


def train_and_evaluate_hybrid_model(batch_size, learning_rate, debug_run=False):
    _train_and_evaluate(
        model_builder=build_cnn_transformer_hybrid_model,
        experiment_name="experiment-hybrid-model",
        batch_size=batch_size,
        learning_rate=learning_rate,
        debug_run=debug_run,
    )


MODEL_REGISTRY = {
    "null": (build_null_model, "experiment-null-model"),
    "cnn": (build_cnn_model, "experiment-cnn-model"),
    "hybrid": (build_cnn_transformer_hybrid_model, "experiment-hybrid-model"),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Train CIFAR-10 classifier.")

    parser.add_argument(
        "--model",
        type=str,
        choices=list(MODEL_REGISTRY.keys()),
        default="null",
        help="Model kind to train: 'null', 'cnn', or 'hybrid'.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for training.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Learning rate for optimizer.",
    )
    parser.add_argument(
        "--summarize-only",
        type=bool,
        default=False,
        help="Summarize the chosen model only, do not train.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    model_builder, experiment_name = MODEL_REGISTRY[args.model]

    _train_and_evaluate(
        model_builder=model_builder,
        experiment_name=experiment_name,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        summarize_only=args.summarize_only,
    )


if __name__ == "__main__":
    main()
