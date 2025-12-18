import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset


def get_cifar10_train_transforms(img_size: int = 32) -> A.Compose:
    # Compose an augmentation pipeline with various augmentations
    return A.Compose(
        [
            A.PadIfNeeded(min_height=img_size + 4, min_width=img_size + 4, p=1.0),
            A.RandomCrop(height=img_size, width=img_size, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
            A.CoarseDropout(
                num_holes_range=(1, 1),
                hole_height_range=(4, 8),
                hole_width_range=(4, 8),
                p=0.3,
            ),
            A.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2470, 0.2435, 0.2616),
            ),
            ToTensorV2(),
        ]
    )


def get_cifar10_val_transforms(img_size: int = 32) -> A.Compose:
    return A.Compose(
        [
            A.Resize(height=img_size, width=img_size),
            A.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2470, 0.2435, 0.2616),
            ),
            ToTensorV2(),
        ]
    )


class CIFAR10Albumentations(Dataset):
    """
    Wraps any CIFAR10-like dataset (including Subset) and applies albumentations.
    """

    def __init__(self, base_dataset, transform: A.Compose):
        self.base_dataset = base_dataset
        self.transform = transform

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        img, label = self.base_dataset[idx]
        img = np.array(img)
        if self.transform is not None:
            augmented = self.transform(image=img)
            img = augmented["image"]
        else:
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        return img, label
