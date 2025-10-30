# src/data.py
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def _make_tfms(image_size: int) -> transforms.Compose:
    """Basic augmentation + normalization for HAM10000-like images."""
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def _make_val_tfms(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def build_loaders(
    data_root: str,
    image_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 4,
) -> tuple[DataLoader, DataLoader, list[str]]:
    train_tfms = _make_tfms(image_size)
    val_tfms = _make_val_tfms(image_size)

    train_ds = datasets.ImageFolder(f"{data_root}/train", transform=train_tfms)
    val_ds = datasets.ImageFolder(f"{data_root}/val", transform=val_tfms)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, train_ds.classes


def compute_class_weights(dataset: datasets.ImageFolder, num_classes: int) -> torch.Tensor:
    """Inverse-frequency class weights for CrossEntropyLoss."""
    counts = np.zeros(num_classes, dtype=np.int64)
    for _, label in dataset.samples:
        counts[label] += 1
    weights = 1.0 / np.maximum(counts, 1)
    weights = weights / weights.sum() * num_classes
    return torch.tensor(weights, dtype=torch.float32)
