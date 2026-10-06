from __future__ import annotations

import torch
import torch.nn as nn

try:
    from torchvision import models as tv
except Exception:
    tv = None


class CustomCNN(nn.Module):
    """
    Four-block baseline CNN, identical to the archived HAM10000 checkpoint.

    Each block is Conv(3x3) -> BatchNorm -> ReLU -> MaxPool(2). For a
    224 x 224 input the last block yields a 256 x 14 x 14 feature map, which is
    flattened and passed to a two-layer classifier with dropout. Parameter
    names match the archived state dict, so it loads with strict=True.

    The flattened size fixes the input resolution at 224 x 224.
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

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 14 * 14, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class LogitsOnly(nn.Module):
    """Return a plain logits tensor from models that return an output object."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
        return out.logits if hasattr(out, "logits") else out


def build_archived_vit(
    num_classes: int = 7,
    config_dir: str | None = None,
) -> nn.Module:
    """
    Build the Hugging Face ViT-B/16 used for the archived ViT checkpoint.

    The archived checkpoint is a ViTForImageClassification state dict
    (google/vit-base-patch16-224 backbone, seven-class head), not a torchvision
    ViT, so it cannot be loaded into torchvision.models.vit_b_16. If a
    config.json is present in config_dir it is used; otherwise the ViT-B/16
    defaults (224 x 224, patch 16, hidden 768, 12 layers, 12 heads) are used,
    which need no download.

    The returned module yields a logits tensor.
    """

    try:
        from transformers import ViTConfig, ViTForImageClassification
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError(
            "The archived ViT checkpoint requires the 'transformers' package."
        ) from exc

    config = None
    if config_dir:
        from pathlib import Path

        if (Path(config_dir) / "config.json").is_file():
            config = ViTConfig.from_pretrained(config_dir, local_files_only=True)

    if config is None:
        config = ViTConfig(
            image_size=224,
            patch_size=16,
            num_channels=3,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
        )

    config.num_labels = num_classes
    return ViTForImageClassification(config)


def get_model(
    name: str,
    num_classes: int = 7,
    pretrained: bool = False,
) -> nn.Module:
    """
    Build one of the seven study architectures.

    Evaluation of archived checkpoints should normally use pretrained=False,
    because all learned weights should come from the checkpoint itself.

    For "vit_b_16" this returns a torchvision ViT, which is suitable for new
    training runs. The archived ViT checkpoint is a Hugging Face model; build
    it with build_archived_vit() instead.
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


__all__ = ["CustomCNN", "LogitsOnly", "build_archived_vit", "get_model"]
