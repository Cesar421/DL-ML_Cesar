import pytest
from torchvision import transforms

from dlcv.utils import get_transforms

def test_transforms_for_testing():
    # Create transform for testing (no augmentations)
    transform = get_transforms(train=False)
    expected_transforms = [transforms.ToTensor, transforms.Normalize]
    assert len(transform.transforms) == len(expected_transforms), "Unexpected number of transformations for testing"
    for t, expected in zip(transform.transforms, expected_transforms):
        assert isinstance(t, expected), f"Expected {expected} in testing transforms"

def test_transforms_for_training_no_augmentation():
    # Create transform for training without specific augmentations
    transform = get_transforms(train=True)
    expected_transforms = [transforms.ToTensor, transforms.Normalize]
    assert len(transform.transforms) == len(expected_transforms), "Unexpected number of transformations for training without augmentations"
    for t, expected in zip(transform.transforms, expected_transforms):
        assert isinstance(t, expected), f"Expected {expected} in training transforms with no augmentation"

def test_transforms_for_training_horizontal_flip():
    # Create transform for training with horizontal flip
    transform = get_transforms(train=True, horizontal_flip_prob=0.5)
    expected_transform_types = {transforms.RandomHorizontalFlip, transforms.ToTensor, transforms.Normalize}
    # Extract types of actual transformations
    actual_transform_types = {type(t) for t in transform.transforms}
    assert len(transform.transforms) == 3, "Unexpected number of transformations with horizontal flip"
    assert expected_transform_types == actual_transform_types, "Some expected transforms are missing"

def test_transforms_for_training_rotation():
    # Create transform for training with rotation
    transform = get_transforms(train=True, rotation_degrees=15)
    expected_transform_types = {transforms.RandomRotation, transforms.ToTensor, transforms.Normalize}
    actual_transform_types = {type(t) for t in transform.transforms}
    assert len(transform.transforms) == 3, "Unexpected number of transformations with rotation"
    assert expected_transform_types == actual_transform_types, "Some expected transforms are missing"

def test_both_augmentations():
    # Create transform for training with both augmentations
    transform = get_transforms(train=True, rotation_degrees=15, horizontal_flip_prob=0.5)
    expected_transform_types = {transforms.RandomHorizontalFlip, transforms.RandomRotation, transforms.ToTensor, transforms.Normalize}
    actual_transform_types = {type(t) for t in transform.transforms}
    assert len(transform.transforms) == 4, "Unexpected number of transformations with both augmentations"
    assert expected_transform_types == actual_transform_types, "Some expected transforms are missing"
