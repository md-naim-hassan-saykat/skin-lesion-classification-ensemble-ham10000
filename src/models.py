# src/models.py
from __future__ import annotations

from typing import Callable, Dict

import torch.nn as nn
from torchvision import models as tv


def _replace_last_linear(module: nn.Module, out_features: int) -> nn.Module:
    """Replace the last nn.Linear inside a classifier/head with one of size out_features."""
    if isinstance(module, nn.Sequential) and len(module) > 0:
        for i in range(len(module) - 1, -1, -1):
            if isinstance(module[i], nn.Linear):
                in_f = module[i].in_features
                module[i] = nn.Linear(in_f, out_features)
                return module
    if isinstance(module, nn.Linear):
        return nn.Linear(module.in_features, out_features)
    # direct attribute checks (avoid getattr constant: B009)
    if hasattr(module, "fc") and isinstance(module.fc, nn.Linear):  # type: ignore[attr-defined]
        in_features = module.fc.in_features  # type: ignore[attr-defined]
        module.fc = nn.Linear(in_features, out_features)  # type: ignore[attr-defined]
        return module
    # fallback: search by name
    last_linear_name = None
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            last_linear_name = name
    if last_linear_name is not None:
        old: nn.Linear = getattr(module, last_linear_name)  # type: ignore[assignment]
        setattr(module, last_linear_name, nn.Linear(old.in_features, out_features))
    return module


def get_model(name: str, num_classes: int) -> nn.Module:
    n = (name or "resnet50").lower()
    builders: Dict[str, Callable[[], nn.Module]] = {
        "resnet50": lambda: tv.resnet50(weights=tv.ResNet50_Weights.IMAGENET1K_V2),
        "densenet121": lambda: tv.densenet121(weights=tv.DenseNet121_Weights.IMAGENET1K_V1),
        "efficientnet_b3": lambda: tv.efficientnet_b3(weights=tv.EfficientNet_B3_Weights.IMAGENET1K_V1),
        "convnext_tiny": lambda: tv.convnext_tiny(weights=tv.ConvNeXt_Tiny_Weights.IMAGENET1K_V1),
        "vit_b_16": lambda: tv.vit_b_16(weights=tv.ViT_B_16_Weights.IMAGENET1K_V1),
        "mobilenet_v3_large": lambda: tv.mobilenet_v3_large(weights=tv.MobileNet_V3_Large_Weights.IMAGENET1K_V2),
    }
    if n not in builders:
        raise ValueError(f"Unknown model '{name}'. Choose from: {', '.join(sorted(builders.keys()))}")

    m = builders[n]()

    if n.startswith("resnet"):
        m.fc = nn.Linear(m.fc.in_features, num_classes)  # type: ignore[attr-defined]
    elif n.startswith("densenet"):
        m.classifier = nn.Linear(m.classifier.in_features, num_classes)  # type: ignore[attr-defined]
    elif n.startswith("efficientnet"):
        m.classifier = _replace_last_linear(m.classifier, num_classes)  # type: ignore[attr-defined]
    elif n.startswith("convnext"):
        m.classifier = _replace_last_linear(m.classifier, num_classes)  # type: ignore[attr-defined]
    elif n.startswith("vit"):
        m.heads.head = nn.Linear(m.heads.head.in_features, num_classes)  # type: ignore[attr-defined]
    elif n.startswith("mobilenet"):
        m.classifier = _replace_last_linear(m.classifier, num_classes)  # type: ignore[attr-defined]
    else:
        if hasattr(m, "classifier"):
            m.classifier = _replace_last_linear(m.classifier, num_classes)  # type: ignore[attr-defined]
        elif hasattr(m, "heads"):
            m.heads = _replace_last_linear(m.heads, num_classes)  # type: ignore[attr-defined]
        else:
            raise ValueError(f"Don’t know how to set classifier for model '{name}'")

    return m
