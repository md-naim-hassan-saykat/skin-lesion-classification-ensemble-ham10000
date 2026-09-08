from __future__ import annotations

import torch
import torch.nn as nn

try:
    from torchvision import models as tv
except Exception:
    tv = None


class CustomCNN(nn.Module):
    """
    Four-block CNN matching the architecture family described in the repository.

    IMPORTANT:
    Historical checkpoint compatibility depends on the exact archived CNN
    architecture. If the original CNN checkpoint used different channel widths
    or classifier dimensions, this class must be adjusted to match that checkpoint.
    """

    def __init__(self, num_classes: int = 7) -> None:
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def get_model(
    name: str,
    num_classes: int = 7,
    pretrained: bool = False,
) -> nn.Module:
    """
    Build one of the seven study architectures.

    Evaluation of archived checkpoints should normally use pretrained=False,
    because all learned weights should come from the checkpoint itself.
    """

    n = (name or "").lower().replace("-", "_")

    aliases = {
        "cnn": "cnn",
        "custom_cnn": "cnn",
        "resnet_50": "resnet50",
        "resnet50": "resnet50",
        "densenet_121": "densenet121",
        "densenet121": "densenet121",
        "efficientnet_b3": "efficientnet_b3",
        "convnext_tiny": "convnext_tiny",
        "mobilenet_v3_l": "mobilenet_v3_large",
        "mobilenet_v3_large": "mobilenet_v3_large",
        "vit": "vit_b_16",
        "vit_b16": "vit_b_16",
        "vit_b_16": "vit_b_16",
    }

    if n not in aliases:
        raise ValueError(f"Unknown model name: {name}")

    n = aliases[n]

    if n == "cnn":
        return CustomCNN(num_classes=num_classes)

    if tv is None:
        raise RuntimeError("torchvision is required to build this model.")

    if n == "resnet50":
        weights = tv.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        model = tv.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    if n == "densenet121":
        weights = tv.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        model = tv.densenet121(weights=weights)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        return model

    if n == "efficientnet_b3":
        weights = tv.EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None
        model = tv.efficientnet_b3(weights=weights)
        model.classifier[-1] = nn.Linear(
            model.classifier[-1].in_features,
            num_classes,
        )
        return model

    if n == "convnext_tiny":
        weights = tv.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        model = tv.convnext_tiny(weights=weights)
        model.classifier[-1] = nn.Linear(
            model.classifier[-1].in_features,
            num_classes,
        )
        return model

    if n == "mobilenet_v3_large":
        weights = tv.MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None
        model = tv.mobilenet_v3_large(weights=weights)
        model.classifier[-1] = nn.Linear(
            model.classifier[-1].in_features,
            num_classes,
        )
        return model

    if n == "vit_b_16":
        weights = tv.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
        model = tv.vit_b_16(weights=weights)
        model.heads.head = nn.Linear(
            model.heads.head.in_features,
            num_classes,
        )
        return model

    raise RuntimeError(f"Unhandled model: {n}")


__all__ = ["CustomCNN", "get_model"]
