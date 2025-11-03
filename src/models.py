from __future__ import annotations

from collections.abc import Callable

import torch.nn as nn


# Optional torchvision import — only needed when actually building models
try:
    from torchvision import models as tv
except Exception:
    tv = None


def _replace_last_linear(module: nn.Module, out_features: int) -> nn.Module:
    """Replace the last nn.Linear in a classifier/head with one sized to out_features."""
    if isinstance(module, nn.Sequential) and len(module) > 0:
        for i in range(len(module) - 1, -1, -1):
            if isinstance(module[i], nn.Linear):
                in_f = module[i].in_features
                module[i] = nn.Linear(in_f, out_features)
                return module
    return module  # nothing to replace


def get_model(name: str, num_classes: int) -> nn.Module:
    """Return a torchvision model whose classifier matches num_classes."""
    if tv is None:
        raise RuntimeError(
            "torchvision not installed. Install 'torchvision' to build models, "
            "or run only CLI parsing with --help."
        )

    n = (name or "resnet50").lower()
    builders: dict[str, Callable[[], nn.Module]] = {
        "resnet50": lambda: tv.resnet50(weights=tv.ResNet50_Weights.IMAGENET1K_V2),
        "densenet121": lambda: tv.densenet121(weights=tv.DenseNet121_Weights.IMAGENET1K_V1),
        "efficientnet_b3": lambda: tv.efficientnet_b3(
            weights=tv.EfficientNet_B3_Weights.IMAGENET1K_V1
        ),
        "convnext_tiny": lambda: tv.convnext_tiny(weights=tv.ConvNeXt_Tiny_Weights.IMAGENET1K_V1),
        "vit_b_16": lambda: tv.vit_b_16(weights=tv.ViT_B_16_Weights.IMAGENET1K_V1),
        "mobilenet_v3_large": lambda: tv.mobilenet_v3_large(
            weights=tv.MobileNet_V3_Large_Weights.IMAGENET1K_V2
        ),
    }

    if n not in builders:
        raise ValueError(f"Unknown model name: {n}")

    model = builders[n]()

    # Adjust final classifier head
    if hasattr(model, "classifier"):
        model.classifier = _replace_last_linear(model.classifier, num_classes)
    elif hasattr(model, "fc"):  # resnet-style
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        raise ValueError(f"Cannot modify classifier for model: {n}")

    return model


__all__ = ["get_model"]
