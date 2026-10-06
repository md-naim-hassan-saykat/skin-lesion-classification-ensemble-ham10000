"""
Regression tests for the audited archived-checkpoint evaluation settings.

The original external evaluation scored ResNet-50 and ViT-B/16 against the
wrong class order and applied one preprocessing pipeline to every checkpoint.
These tests pin the corrected settings so neither defect can return silently.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from src.archived import (
    ARCHIVED_CHECKPOINTS,
    CANONICAL_CLASSES,
    IDENTITY,
    METADATA_CLASS_ORDER,
    METADATA_TO_CANONICAL,
    resolve_eval_settings,
)

ROOT = Path(__file__).resolve().parents[1]


def _config() -> dict:
    with open(ROOT / "src" / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_metadata_permutation_value():
    """ResNet-50 and ViT were trained in HAM10000 metadata order."""
    assert METADATA_CLASS_ORDER == ["bkl", "nv", "df", "mel", "vasc", "bcc", "akiec"]
    assert METADATA_TO_CANONICAL == [6, 5, 0, 2, 3, 1, 4]


def test_metadata_permutation_reorders_columns():
    """raw[:, permutation] must place each class in its canonical column."""
    raw = np.eye(7)  # row i: certain prediction for metadata-order class i
    canonical = raw[:, METADATA_TO_CANONICAL]
    for i, name in enumerate(METADATA_CLASS_ORDER):
        assert CANONICAL_CLASSES[int(canonical[i].argmax())] == name


def test_permutations_per_model():
    perms = {k: v["output_permutation"] for k, v in ARCHIVED_CHECKPOINTS.items()}
    assert perms.pop("resnet50") == METADATA_TO_CANONICAL
    assert perms.pop("vit_b_16") == METADATA_TO_CANONICAL
    assert all(p == IDENTITY for p in perms.values())


def test_preprocessing_per_model():
    """Input size and normalization extracted from the archived training code."""
    expected = {
        "cnn": (224, "half"),
        "resnet50": (224, "imagenet"),
        "densenet121": (224, "half"),
        "efficientnet_b3": (300, "half"),
        "convnext_tiny": (224, "imagenet"),
        "mobilenet_v3_large": (224, "half"),
        "vit_b_16": (224, "half"),
    }
    for name, (size, norm) in expected.items():
        settings = resolve_eval_settings(name)
        assert settings["image_size"] == size, name
        assert settings["normalization"] == norm, name


def test_config_matches_module():
    """src/config.yaml and src/archived.py must describe the same settings."""
    cfg = _config()["archived_checkpoints"]
    assert set(cfg) == set(ARCHIVED_CHECKPOINTS)
    for name, values in ARCHIVED_CHECKPOINTS.items():
        for key, value in values.items():
            assert cfg[name][key] == value, (name, key)


def test_config_ensemble_members_have_settings():
    cfg = _config()
    assert set(cfg["ensemble"]["models"]) == set(cfg["archived_checkpoints"])


def test_overrides_and_aliases():
    s = resolve_eval_settings("ViT", image_size=256, normalization="imagenet")
    assert s["model"] == "vit_b_16"
    assert s["image_size"] == 256
    assert s["normalization"] == "imagenet"
    assert s["output_permutation"] == METADATA_TO_CANONICAL


def test_invalid_permutation_rejected():
    with pytest.raises(ValueError):
        resolve_eval_settings("cnn", output_permutation=[0, 0, 1, 2, 3, 4, 5])


def test_unknown_model_rejected():
    with pytest.raises(ValueError):
        resolve_eval_settings("resnet18")


def test_archived_cnn_matches_checkpoint_layout():
    """The CNN must expose the parameter names of the archived state dict."""
    torch = pytest.importorskip("torch")
    from src.models import CustomCNN

    model = CustomCNN(num_classes=7)
    keys = set(model.state_dict())
    assert {"classifier.1.weight", "classifier.4.weight"} <= keys
    assert tuple(model.state_dict()["classifier.1.weight"].shape) == (512, 256 * 14 * 14)

    model.eval()
    with torch.no_grad():
        out = model(torch.zeros(2, 3, 224, 224))
    assert tuple(out.shape) == (2, 7)
