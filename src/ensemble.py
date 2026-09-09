from __future__ import annotations

# ruff: noqa: E402
import sys
from pathlib import Path as _P

_PROJECT_ROOT = _P(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from src.utils import CANONICAL_CLASSES, compute_metrics, save_json

EXPECTED_MODELS = [
    "cnn",
    "resnet50",
    "densenet121",
    "efficientnet_b3",
    "convnext_tiny",
    "mobilenet_v3_large",
    "vit_b_16",
]


def read_prediction_csv(
    path: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns sample_index, y_true, probability matrix.

    All files must represent the same samples in exactly the same order.
    """

    indices = []
    labels = []
    probabilities = []

    expected_prob_columns = [f"p_{c}" for c in CANONICAL_CLASSES]

    with open(
        path,
        newline="",
        encoding="utf-8",
    ) as f:
        reader = csv.DictReader(f)

        if reader.fieldnames is None:
            raise ValueError(f"{path}: missing CSV header.")

        required = {
            "sample_index",
            "y_true",
            *expected_prob_columns,
        }

        missing = required.difference(reader.fieldnames)

        if missing:
            raise ValueError(f"{path}: missing columns: {sorted(missing)}")

        for row in reader:
            indices.append(int(row["sample_index"]))
            labels.append(int(row["y_true"]))

            probabilities.append([float(row[c]) for c in expected_prob_columns])

    return (
        np.asarray(indices, dtype=int),
        np.asarray(labels, dtype=int),
        np.asarray(probabilities, dtype=float),
    )


def validate_alignment(
    arrays: list[
        tuple[
            np.ndarray,
            np.ndarray,
            np.ndarray,
        ]
    ],
) -> None:
    ref_idx, ref_y, _ = arrays[0]

    for i, (idx, y, p) in enumerate(
        arrays[1:],
        start=2,
    ):
        if len(idx) != len(ref_idx):
            raise ValueError(
                f"Prediction file {i} has {len(idx)} samples; " f"expected {len(ref_idx)}."
            )

        if not np.array_equal(
            idx,
            ref_idx,
        ):
            raise ValueError(f"Prediction file {i} sample ordering differs.")

        if not np.array_equal(
            y,
            ref_y,
        ):
            raise ValueError(f"Prediction file {i} labels differ.")

        if p.shape != (
            len(ref_idx),
            7,
        ):
            raise ValueError(f"Prediction file {i} has invalid probability shape {p.shape}.")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Equal-weight probability ensemble across exactly seven " "canonical model outputs."
        )
    )

    ap.add_argument(
        "--csvs",
        nargs="+",
        required=True,
    )
    ap.add_argument(
        "--dataset",
        choices=["ham10000", "isic2019"],
        default="ham10000",
    )
    ap.add_argument(
        "--out",
        required=True,
    )

    args = ap.parse_args()

    if len(args.csvs) != 7:
        raise ValueError("The final manuscript ensemble requires exactly seven model CSVs.")

    outputs = [read_prediction_csv(path) for path in args.csvs]

    validate_alignment(outputs)

    sample_index = outputs[0][0]
    y_true = outputs[0][1]

    probs = np.stack(
        [output[2] for output in outputs],
        axis=0,
    )

    ensemble_prob = probs.mean(
        axis=0,
    )

    metrics = compute_metrics(
        y_true,
        y_prob=ensemble_prob,
        dataset=args.dataset,
        ece_bins=15,
    )

    metrics.update(
        {
            "samples": int(len(y_true)),
            "num_models": 7,
            "ensemble_method": "equal_weight_probability_mean",
            "weight_per_model": 1.0 / 7.0,
            "canonical_classes": CANONICAL_CLASSES,
            "input_csvs": args.csvs,
        }
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_json(
        metrics,
        out_path,
    )

    csv_path = out_path.parent / "ensemble_predictions.csv"

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.writer(f)

        writer.writerow(
            [
                "sample_index",
                "y_true",
                "y_pred",
                *[f"p_{c}" for c in CANONICAL_CLASSES],
            ]
        )

        y_pred = ensemble_prob.argmax(
            axis=1,
        )

        for idx, target, pred, p in zip(
            sample_index,
            y_true,
            y_pred,
            ensemble_prob,
            strict=True,
        ):
            writer.writerow(
                [
                    int(idx),
                    int(target),
                    int(pred),
                    *[f"{float(x):.10f}" for x in p],
                ]
            )

    print(
        json.dumps(
            metrics,
            indent=2,
        )
    )

    print(f"[csv] wrote {csv_path}")


if __name__ == "__main__":
    main()
