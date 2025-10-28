# src/models.py
from __future__ import annotations

from typing import Callable, Dict

import torch.nn as nn
from torchvision import models as tv


def _replace_last_linear(module: nn.Module, out_features: int) -> nn.Module:
    """Replace the last nn.Linear inside a classifier/head with one of size out_features."""
    # Try common Sequential pattern
    if hasattr(module, "classifier") and isinstance(module.classifier, nn.Sequential):
        seq = list(module.classifier.children())
        for i in range(len(seq) - 1, -1, -1):
            if isinstance(seq[i], nn.Linear):
                in_features = seq[i].in_features
                seq[i] = nn.Linear(in_features, out_features)
                module.classifier = nn.Sequential(*seq)
                return module
    # Try attribute named 'fc' (e.g., ResNet)
    if hasattr(module, "fc") and isinstance(getattr(module, "fc"), nn.Linear):
        in_features = module.fc.in_features
        module.fc = nn.Linear(in_features, out_features)
        return module
    return module


def get_model(name: str, num_classes: int = 7) -> nn.Module:
    name = name.lower()
    builders: Dict[str, Callable[[], nn.Module]] = {
        "resnet50": tv.resnet50,
        "densenet121": tv.densenet121,
        "efficientnetb3": tv.efficientnet_b3,
        "convnext_tiny": tv.convnext_tiny,
    }
    if name not in builders:
        raise ValueError(f"Unknown model: {name}")

    model = builders[name](weights=None)  # load weights externally if needed
    model = _replace_last_linear(model, num_classes)
    return model
