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
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.archived import NORMALIZATION, resolve_eval_settings
from src.models import LogitsOnly, build_archived_vit, get_model
from src.utils import compute_metrics, load_yaml, save_json

CANONICAL_CLASSES = [
    "akiec",
    "bcc",
    "bkl",
    "df",
    "mel",
    "nv",
    "vasc",
]


def best_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")

    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def build_eval_transform(
    image_size: int,
    normalization: str,
) -> transforms.Compose:
    """
    Evaluation transform for one archived checkpoint.

    Resize to image_size x image_size, convert to a tensor, and normalize with
    the statistics the checkpoint was trained with ("imagenet" or "half"; see
    src/archived.py). For the ViT checkpoint, resize to 224 with "half"
    normalization is what ViTImageProcessor for google/vit-base-patch16-224
    applies.
    """

    mean, std = NORMALIZATION[normalization]

    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def clean_state_dict(state: dict) -> dict:
    cleaned = {}

    for key, value in state.items():
        k = key

        for prefix in (
            "module.",
            "model.",
            "net.",
        ):
            if k.startswith(prefix):
                k = k[len(prefix) :]

        cleaned[k] = value

    return cleaned


def load_checkpoint_strict(
    model: torch.nn.Module,
    checkpoint: str,
    device: torch.device,
) -> None:
    """
    Load a checkpoint without silently discarding unmatched learned parameters.
    """

    raw = torch.load(
        checkpoint,
        map_location=device,
    )

    if isinstance(raw, dict):
        if "model" in raw and isinstance(raw["model"], dict):
            raw = raw["model"]
        elif "state_dict" in raw and isinstance(raw["state_dict"], dict):
            raw = raw["state_dict"]

    if not isinstance(raw, dict):
        raise TypeError(f"Unsupported checkpoint format: {type(raw)}")

    state = clean_state_dict(raw)

    try:
        model.load_state_dict(
            state,
            strict=True,
        )
    except RuntimeError as exc:
        raise RuntimeError(
            "\nCheckpoint does not exactly match the selected architecture.\n"
            "Do not silently drop checkpoint parameters for manuscript evaluation.\n"
            f"Checkpoint: {checkpoint}\n"
            f"Model: {model.__class__.__name__}\n\n"
            f"{exc}"
        ) from exc


def validate_imagefolder_classes(
    ds: datasets.ImageFolder,
) -> None:
    if ds.classes != CANONICAL_CLASSES:
        raise ValueError(
            "ImageFolder class ordering is not canonical.\n"
            f"Expected: {CANONICAL_CLASSES}\n"
            f"Found:    {ds.classes}"
        )


def parse_permutation(value: str | None) -> list[int] | None:
    if value is None:
        return None

    permutation = [int(x.strip()) for x in value.split(",")]

    if sorted(permutation) != list(range(7)):
        raise ValueError("Permutation must contain every index 0..6 exactly once.")

    return permutation


def build_archived_model(
    settings: dict,
    checkpoint: str,
    device: torch.device,
) -> torch.nn.Module:
    """Build the architecture for an archived checkpoint and load it strictly."""

    if settings["loader"] == "hf_vit":
        model = build_archived_vit(
            num_classes=7,
            config_dir=str(Path(checkpoint).parent),
        )
        load_checkpoint_strict(model, checkpoint, device)
        return LogitsOnly(model).to(device)

    model = get_model(
        settings["model"],
        num_classes=7,
        pretrained=False,
    )
    load_checkpoint_strict(model, checkpoint, device)
    return model.to(device)


def run_inference(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    permutation: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    y_true = []
    probabilities = []

    model.eval()

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)

            logits = model(images)

            if hasattr(logits, "logits"):
                logits = logits.logits

            if logits.ndim != 2 or logits.shape[1] != 7:
                raise ValueError(f"Expected model output (N, 7), got {tuple(logits.shape)}")

            p = (
                torch.softmax(
                    logits,
                    dim=1,
                )
                .cpu()
                .numpy()
            )

            p = p[:, permutation]

            probabilities.append(p)
            y_true.extend(labels.numpy().astype(int).tolist())

    return (
        np.asarray(y_true, dtype=int),
        np.concatenate(probabilities, axis=0),
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Evaluate one archived checkpoint with its audited preprocessing and "
            "output permutation, reporting canonical seven-class outputs."
        )
    )

    ap.add_argument(
        "--checkpoint",
        required=True,
    )
    ap.add_argument(
        "--data_dir",
        required=True,
    )
    ap.add_argument(
        "--model",
        required=True,
    )
    ap.add_argument(
        "--dataset",
        choices=["ham10000", "isic2019"],
        default="ham10000",
    )
    ap.add_argument(
        "--num_classes",
        type=int,
        default=7,
    )
    ap.add_argument(
        "--config",
        default=str(_PROJECT_ROOT / "src" / "config.yaml"),
        help="Configuration file holding the audited archived-checkpoint settings.",
    )
    ap.add_argument(
        "--image_size",
        type=int,
        default=None,
        help="Override the audited input resolution for this checkpoint.",
    )
    ap.add_argument(
        "--normalization",
        choices=sorted(NORMALIZATION),
        default=None,
        help="Override the audited normalization for this checkpoint.",
    )
    ap.add_argument(
        "--batch_size",
        type=int,
        default=32,
    )
    ap.add_argument(
        "--num_workers",
        type=int,
        default=2,
    )
    ap.add_argument(
        "--output_permutation",
        default=None,
        help=(
            "Override the audited mapping from raw checkpoint output columns "
            "to canonical order (canonical = raw[:, permutation]). "
            "Example: 6,5,0,2,3,1,4"
        ),
    )
    ap.add_argument(
        "--out",
        required=True,
    )
    ap.add_argument(
        "--save_csv",
        default=None,
    )

    args = ap.parse_args()

    if args.num_classes != 7:
        raise ValueError("This study uses exactly seven canonical classes.")

    config = load_yaml(args.config) if args.config and Path(args.config).is_file() else None

    settings = resolve_eval_settings(
        args.model,
        config=config,
        image_size=args.image_size,
        normalization=args.normalization,
        output_permutation=parse_permutation(args.output_permutation),
    )
    permutation = settings["output_permutation"]

    device = best_device()

    model = build_archived_model(
        settings,
        args.checkpoint,
        device,
    )

    transform = build_eval_transform(
        settings["image_size"],
        settings["normalization"],
    )

    ds = datasets.ImageFolder(
        args.data_dir,
        transform=transform,
    )

    validate_imagefolder_classes(ds)

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    y_true, y_prob = run_inference(
        model,
        loader,
        device,
        permutation,
    )

    metrics = compute_metrics(
        y_true,
        y_prob=y_prob,
        dataset=args.dataset,
        ece_bins=15,
    )

    metrics.update(
        {
            "samples": int(len(y_true)),
            "model": settings["model"],
            "dataset": args.dataset,
            "canonical_classes": CANONICAL_CLASSES,
            "image_size": settings["image_size"],
            "normalization": settings["normalization"],
            "output_permutation": permutation,
            "checkpoint": str(args.checkpoint),
        }
    )

    save_json(
        metrics,
        args.out,
    )

    if args.save_csv:
        p = Path(args.save_csv)
        p.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with p.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as f:
            writer = csv.writer(f)

            writer.writerow(["sample_index", "y_true"] + [f"p_{c}" for c in CANONICAL_CLASSES])

            for idx, (target, probs) in enumerate(
                zip(
                    y_true,
                    y_prob,
                    strict=True,
                )
            ):
                writer.writerow(
                    [
                        idx,
                        int(target),
                        *[f"{float(x):.10f}" for x in probs],
                    ]
                )

        print(f"[csv] wrote {args.save_csv}")

    print(
        json.dumps(
            metrics,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
