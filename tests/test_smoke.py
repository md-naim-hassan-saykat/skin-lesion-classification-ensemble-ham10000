from __future__ import annotations

import subprocess
import sys

import pytest


SOURCE_MODULES = [
    "src.data",
    "src.ensemble",
    "src.evaluate",
    "src.models",
    "src.train",
    "src.utils",
]


@pytest.mark.parametrize("module_name", SOURCE_MODULES)
def test_imports(module_name):
    """Verify that the main source modules import successfully."""
    __import__(module_name)


@pytest.mark.timeout(10)
@pytest.mark.parametrize(
    "script",
    [
        "src/evaluate.py",
        "src/ensemble.py",
        "src/train.py",
    ],
)
def test_cli_help(script):
    """Verify that command-line entry points expose working --help output."""
    result = subprocess.run(
        [sys.executable, script, "--help"],
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode == 0, result.stderr
    assert "usage" in result.stdout.lower()


def test_canonical_classes():
    """Verify the fixed canonical seven-class output order."""
    from src.utils import CANONICAL_CLASSES

    assert CANONICAL_CLASSES == [
        "akiec",
        "bcc",
        "bkl",
        "df",
        "mel",
        "nv",
        "vasc",
    ]

    assert len(CANONICAL_CLASSES) == 7
    assert len(set(CANONICAL_CLASSES)) == 7


def test_ece_perfect_predictions():
    """Perfectly confident correct predictions should have zero ECE."""
    import numpy as np

    from src.utils import expected_calibration_error

    y_true = np.arange(7)

    y_prob = np.eye(7, dtype=float)

    ece = expected_calibration_error(
        y_true,
        y_prob,
        n_bins=15,
    )

    assert ece == pytest.approx(0.0)


def test_compute_metrics_perfect_predictions():
    """Verify manuscript-aligned HAM10000 metrics on perfect predictions."""
    import numpy as np

    from src.utils import compute_metrics

    y_true = np.arange(7)
    y_prob = np.eye(7, dtype=float)

    metrics = compute_metrics(
        y_true,
        y_prob=y_prob,
        dataset="ham10000",
        ece_bins=15,
    )

    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["weighted_f1"] == pytest.approx(1.0)
    assert metrics["macro_roc_auc_ovr"] == pytest.approx(1.0)
    assert metrics["micro_roc_auc_ovr"] == pytest.approx(1.0)
    assert metrics["ece"] == pytest.approx(0.0)
