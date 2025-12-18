import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import timm
import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_lightning import LightningDataModule, LightningModule
from torch.utils.data import DataLoader, Dataset, random_split
from torchmetrics.classification import (MulticlassAccuracy,
                                         MulticlassConfusionMatrix)
from torchvision.datasets import CIFAR10
from torchvision.utils import make_grid
from vit_pytorch import SimpleViT

from dlcv3.dataset import (CIFAR10Albumentations, get_cifar10_train_transforms,
                           get_cifar10_val_transforms)

CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


class CIFAR10DataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "./data",
        batch_size: int = 128,
        num_workers: int = 8,
        img_size: int = 32,
        val_split: float = 0.1,
        seed: int = 42,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_size = img_size
        self.val_split = val_split
        self.seed = seed

        self.train_set = None
        self.val_set = None
        self.test_set = None

    def prepare_data(self):
        CIFAR10(self.data_dir, train=True, download=True)
        CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            full_train = CIFAR10(self.data_dir, train=True, download=False)
            n_total = len(full_train)
            n_val = int(self.val_split * n_total)
            n_train = n_total - n_val

            generator = torch.Generator().manual_seed(self.seed)
            train_base, val_base = random_split(
                full_train, [n_train, n_val], generator=generator
            )

            train_tf = get_cifar10_train_transforms(self.img_size)
            val_tf = get_cifar10_val_transforms(self.img_size)

            self.train_set = CIFAR10Albumentations(train_base, train_tf)
            self.val_set = CIFAR10Albumentations(val_base, val_tf)

        if stage == "test" or stage is None:
            test_base = CIFAR10(self.data_dir, train=False, download=False)
            test_tf = get_cifar10_val_transforms(self.img_size)
            self.test_set = CIFAR10Albumentations(test_base, test_tf)

    def _build_dataloader(self, dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def train_dataloader(self):
        return self._build_dataloader(self.train_set, shuffle=True)

    def val_dataloader(self):
        return self._build_dataloader(self.val_set, shuffle=False)

    def test_dataloader(self):
        return self._build_dataloader(self.test_set, shuffle=False)


class NullModel(nn.Module):
    def __init__(self):
        super().__init__()

        # Simple CNN to classify into 10 classes
        # Input: 3x32x32
        # Block 1: conv(3->32, 3x3, padding of 1) + ReLU + 2x2 max pool  -> 32x16x16
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Block 2: conv(32->64, 3x3, padding of 1) + ReLU + 2x2 max pool -> 64x8x8
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Head: flatten -> FC(64*8*8 -> 128) + ReLU -> FC(128 -> 10 logits)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x: torch.Tensor):
        x = self.block1(x)
        x = self.block2(x)
        x = self.head(x)
        return x


def build_null_model() -> nn.Module:
    # Simple CNN model
    return NullModel()


def build_cnn_model() -> nn.Module:
    # Return a pre-trained CNN backbone using TIMM library
    model = timm.create_model(
        "efficientnet_b0",
        pretrained=True,
        num_classes=10,
    )
    return model


class HybridModel(nn.Module):
    def __init__(self):
        super().__init__()

        # Implement a pre-trained CNN backbone using TIMM library with features_only=True
        self.backbone = timm.create_model(
            "resnet18",
            pretrained=True,
            features_only=True
        )

        # Use the 2nd feature layer output (index 1)
        # For resnet18 with 32x32 input, feature_info[1] has 64 channels and 8x8 spatial size
        # Implement a transformer using SimpleViT
        self.transformer = SimpleViT(
            channels=self.backbone.feature_info[1]["num_chs"],  # 64 channels
            image_size=8,   # 8x8 feature map from backbone
            patch_size=2,   # 8 / 2 = 4 patches per dimension (16 total patches)
            depth=4,        # 4 transformer blocks
            heads=4,        # 4 attention heads
            dim=256,        # transformer dimension
            mlp_dim=512,    # MLP hidden dimension
            num_classes=10,
        )

    def forward(self, x: torch.Tensor):
        # x: (B, 3, 32, 32)
        feat = self.backbone(x)[1]  # (B, 64, 8, 8)
        logits = self.transformer(feat)  # (B, 10)
        return logits


def build_cnn_transformer_hybrid_model() -> nn.Module:
    # Implement a hybrid model based on a pre-trained CNN backbone (TIMM) + transformer (vit-pytorch)
    return HybridModel()


def build_classification_loss() -> nn.Module:
    # Drop-in an appropriate classification loss
    return nn.CrossEntropyLoss()


class CIFAR10Classifier(LightningModule):
    def __init__(
        self,
        model: nn.Module,
        lr: float,
    ):
        super().__init__()
        self.model = model
        self.lr = lr

        self.criterion = build_classification_loss()

        self.train_acc = MulticlassAccuracy(num_classes=10)
        self.val_acc = MulticlassAccuracy(num_classes=10)
        self.test_acc = MulticlassAccuracy(num_classes=10)

        self.train_confmat = MulticlassConfusionMatrix(num_classes=10)
        self.val_confmat = MulticlassConfusionMatrix(num_classes=10)
        self.test_confmat = MulticlassConfusionMatrix(num_classes=10)

        # buffers for qualitative logging
        self.train_example_images = None
        self.train_example_preds = None
        self.train_example_targets = None

        self.val_example_images = None
        self.val_example_preds = None
        self.val_example_targets = None

        self.test_example_images = None
        self.test_example_preds = None
        self.test_example_targets = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    # ---------- TRAIN/VAL/TEST STEPS ----------

    def _step(self, batch, batch_idx: int, stage: str):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = logits.argmax(dim=1)

        acc_metric: MulticlassAccuracy = getattr(self, f"{stage}_acc")
        confmat: MulticlassConfusionMatrix = getattr(self, f"{stage}_confmat")

        acc_metric(preds, y)
        confmat.update(preds, y)

        if batch_idx == 0:
            self._store_examples(stage, x, preds, y)

        # Log loss and accuracy on epoch
        self.log(f"{stage}/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"{stage}/acc", acc_metric, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, stage="train")

    def validation_step(self, batch, batch_idx):
        self._step(batch, batch_idx, stage="val")

    def test_step(self, batch, batch_idx):
        self._step(batch, batch_idx, stage="test")

    # ---------- EPOCH-END HOOKS ----------

    def _epoch_end(self, stage: str):
        confmat: MulticlassConfusionMatrix = getattr(self, f"{stage}_confmat")
        cm = confmat.compute().detach().cpu().numpy()
        confmat.reset()

        fig = self._plot_confusion_matrix(cm, CIFAR10_CLASSES)
        # Log confusion matrix figure to TensorBoard
        self.logger.experiment.add_figure(
            f"{stage}/confmat",
            fig,
            global_step=self.current_epoch
        )

        plt.close(fig)

        self._log_examples(stage)

    def on_train_epoch_end(self):
        self._epoch_end("train")

    def on_validation_epoch_end(self):
        self._epoch_end("val")

    def on_test_epoch_end(self):
        self._epoch_end("test")

    # ---------- OPTIMIZER ----------

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        return {
            "optimizer": optimizer,
        }

    # ---------- HELPERS ----------

    def _store_examples(
        self,
        stage: str,
        images: torch.Tensor,
        preds: torch.Tensor,
        targets: torch.Tensor,
        k: int = 16,
    ) -> None:
        k = min(k, images.size(0))
        setattr(self, f"{stage}_example_images", images[:k].detach().cpu())
        setattr(self, f"{stage}_example_preds", preds[:k].detach().cpu())
        setattr(self, f"{stage}_example_targets", targets[:k].detach().cpu())

    def _log_examples(self, stage: str) -> None:
        images = getattr(self, f"{stage}_example_images")
        preds = getattr(self, f"{stage}_example_preds")
        targets = getattr(self, f"{stage}_example_targets")

        # --- Qualitative examples ---
        if (
            images is not None
            and preds is not None
            and targets is not None
            and self.logger is not None
        ):
            imgs = self._denormalize(images)
            grid = make_grid(imgs, nrow=4)

            # Log image grid to TensorBoard
            self.logger.experiment.add_image(
                f"{stage}/samples",
                grid,
                global_step=self.current_epoch
            )

            caption_lines = []
            for p, t in zip(preds, targets):
                caption_lines.append(
                    f"pred: {CIFAR10_CLASSES[int(p)]}, "
                    f"true: {CIFAR10_CLASSES[int(t)]}"
                )
            caption_lines = "\n".join(caption_lines)

            # Log caption text to TensorBoard
            self.logger.experiment.add_text(
                f"{stage}/samples_labels",
                caption_lines,
                global_step=self.current_epoch
            )

    @staticmethod
    def _denormalize(x: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
        std = torch.tensor([0.2470, 0.2435, 0.2616]).view(3, 1, 1)
        return x * std + mean

    @staticmethod
    def _plot_confusion_matrix(cm: np.ndarray, class_names):
        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(cm, interpolation="nearest")
        ax.figure.colorbar(im, ax=ax)

        ax.set(
            xticks=np.arange(len(class_names)),
            yticks=np.arange(len(class_names)),
            xticklabels=class_names,
            yticklabels=class_names,
            ylabel="True label",
            xlabel="Predicted label",
        )
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                val = cm[i, j]
                if val > 0:
                    ax.text(
                        j,
                        i,
                        int(val),
                        ha="center",
                        va="center",
                        color="white" if val > thresh else "black",
                        fontsize=7,
                    )

        fig.tight_layout()
        return fig
