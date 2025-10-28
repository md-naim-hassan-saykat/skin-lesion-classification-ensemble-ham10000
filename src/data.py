# src/data.py
from __future__ import annotations

from typing import Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def _make_tfms(image_size: int) -> transforms.Compose:
    """Basic augmentation + normalization for HAM10000-like images."""
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )


def build_loaders(
    data_root: str,
    image_size: int,
    batch_size: int,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> Tuple[DataLoader, DataLoader, transforms.Compose]:
    """Return train/val DataLoaders built from ImageFolder and the transform."""
    tfms = _make_tfms(image_size)
    train_ds = datasets.ImageFolder(f"{data_root}/train", transform=tfms)
    val_ds = datasets.ImageFolder(f"{data_root}/val", transform=tfms)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader, tfms


def compute_class_weights(dataset: datasets.ImageFolder, num_classes: int) -> torch.Tensor:
    """Inverse-frequency class weights for CrossEntropyLoss."""
    targets = torch.tensor(dataset.targets, dtype=torch.long)
    counts = torch.bincount(targets, minlength=num_classes).float()
    weights = 1.0 / (counts + 1e-6)
    weights = weights * (num_classes / weights.sum())  # normalize
    return weights
