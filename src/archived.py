"""
Audited evaluation settings for the seven archived HAM10000 checkpoints.

The archived checkpoints were not trained with one common pipeline. Evaluating
them with a single shared preprocessing, or scoring two of them against the
wrong class order, produces wrong results: this is how the near-chance ISIC
2019 values originally reported for ViT and ResNet-50 arose. The values below
were extracted from the archived training code (Supplementary Table S6 of the
manuscript) and are the ones used for every reported result.

This module has no PyTorch dependency so the settings can be inspected and
tested on their own.
"""

from __future__ import annotations

from typing import Any

CANONICAL_CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]

# Order of the class indices used when ResNet-50 and ViT were trained: the
# unsorted order in which diagnoses first appear in HAM10000_metadata.csv.
METADATA_CLASS_ORDER = ["bkl", "nv", "df", "mel", "vasc", "bcc", "akiec"]

# canonical_probs = raw_probs[:, permutation]
IDENTITY = [0, 1, 2, 3, 4, 5, 6]
METADATA_TO_CANONICAL = [METADATA_CLASS_ORDER.index(c) for c in CANONICAL_CLASSES]

NORMALIZATION = {
    "imagenet": ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    # Normalize([0.5]*3, [0.5]*3). This is also what ViTImageProcessor for
    # google/vit-base-patch16-224 applies after resizing to 224 x 224.
    "half": ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
}

ARCHIVED_CHECKPOINTS: dict[str, dict[str, Any]] = {
    "cnn": {
        "file": "cnn_ham10000_best_model.pth",
        "loader": "torchvision",
        "image_size": 224,
        "normalization": "half",
        "output_permutation": IDENTITY,
    },
    "resnet50": {
        "file": "resnet50_ham10000.pth",
        "loader": "torchvision",
        "image_size": 224,
        "normalization": "imagenet",
        "output_permutation": METADATA_TO_CANONICAL,
    },
    "densenet121": {
        "file": "densenet121_ham10000.pth",
        "loader": "torchvision",
        "image_size": 224,
        "normalization": "half",
        "output_permutation": IDENTITY,
    },
    "efficientnet_b3": {
        "file": "efficientnet_b3_best.pth",
        "loader": "torchvision",
        "image_size": 300,
        "normalization": "half",
        "output_permutation": IDENTITY,
    },
    "convnext_tiny": {
        "file": "convnext_tiny_ham10000.pth",
        "loader": "torchvision",
        "image_size": 224,
        "normalization": "imagenet",
        "output_permutation": IDENTITY,
    },
    "mobilenet_v3_large": {
        "file": "mobilenetv3_ham10000.pth",
        "loader": "torchvision",
        "image_size": 224,
        "normalization": "half",
        "output_permutation": IDENTITY,
    },
    "vit_b_16": {
        "file": "vit_ham10000_best_model.pth",
        # Hugging Face ViTForImageClassification (google/vit-base-patch16-224
        # backbone) state dict, not a torchvision ViT.
        "loader": "hf_vit",
        "image_size": 224,
        "normalization": "half",
        "output_permutation": METADATA_TO_CANONICAL,
    },
}

_ALIASES = {
    "custom_cnn": "cnn",
    "resnet_50": "resnet50",
    "densenet_121": "densenet121",
    "mobilenet_v3_l": "mobilenet_v3_large",
    "mobilenet_v3": "mobilenet_v3_large",
    "vit": "vit_b_16",
    "vit_b16": "vit_b_16",
}


def canonical_model_name(name: str) -> str:
    n = (name or "").lower().replace("-", "_")
    return _ALIASES.get(n, n)


def validate_permutation(permutation: list[int]) -> list[int]:
    permutation = [int(x) for x in permutation]
    if sorted(permutation) != IDENTITY:
        raise ValueError("Permutation must contain every index 0..6 exactly once.")
    return permutation


def resolve_eval_settings(
    model_name: str,
    config: dict[str, Any] | None = None,
    image_size: int | None = None,
    normalization: str | None = None,
    output_permutation: list[int] | None = None,
) -> dict[str, Any]:
    """
    Return the evaluation settings for one archived checkpoint.

    Explicit arguments win; otherwise the value comes from the
    `archived_checkpoints` section of the configuration file, and finally from
    the audited defaults in this module.
    """

    name = canonical_model_name(model_name)
    if name not in ARCHIVED_CHECKPOINTS:
        raise ValueError(f"Unknown model name: {model_name}")

    settings = dict(ARCHIVED_CHECKPOINTS[name])
    if config:
        settings.update((config.get("archived_checkpoints") or {}).get(name) or {})

    if image_size is not None:
        settings["image_size"] = int(image_size)
    if normalization is not None:
        settings["normalization"] = normalization
    if output_permutation is not None:
        settings["output_permutation"] = output_permutation

    if settings["normalization"] not in NORMALIZATION:
        raise ValueError(
            f"Unknown normalization '{settings['normalization']}'; "
            f"expected one of {sorted(NORMALIZATION)}."
        )
    if settings.get("output_permutation") is None:
        raise ValueError(f"No audited output permutation is available for {name}.")

    settings["output_permutation"] = validate_permutation(settings["output_permutation"])
    settings["image_size"] = int(settings["image_size"])
    settings["model"] = name
    return settings


__all__ = [
    "ARCHIVED_CHECKPOINTS",
    "CANONICAL_CLASSES",
    "IDENTITY",
    "METADATA_CLASS_ORDER",
    "METADATA_TO_CANONICAL",
    "NORMALIZATION",
    "canonical_model_name",
    "resolve_eval_settings",
    "validate_permutation",
]
