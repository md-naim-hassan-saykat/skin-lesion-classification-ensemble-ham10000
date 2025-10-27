"""Data loading utilities for HAM10000 or ImageFolder datasets."""

from __future__ import annotations
from pathlib import Path
from typing import Tuple, List

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np


def build_loaders(
    data_root: str | Path,
    image_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader, List[str]]:
    """Build train and validation dataloaders for ImageFolder structure."""
    data_root = Path(data_root)
    train_dir = data_root / "train"
    val_dir = data_root / "val"

    # Basic augmentations for training; resize only for validation
    train_tfms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    val_tfms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    train_ds = datasets.ImageFolder(train_dir, transform=train_tfms)
    val_ds = datasets.ImageFolder(val_dir, transform=val_tfms)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    classes = train_ds.classes
    return train_loader, val_loader, classes


def compute_class_weights(dataset: datasets.ImageFolder, num_classes: int) -> torch.Tensor:
    """Compute class weights inversely proportional to class frequency."""
    counts = np.zeros(num_classes, dtype=np.int64)
    for _, label in dataset.samples:
        counts[label] += 1

    weights = 1.0 / np.maximum(counts, 1)
    weights = weights / weights.sum() * num_classes
    return torch.tensor(weights, dtype=torch.float32)
