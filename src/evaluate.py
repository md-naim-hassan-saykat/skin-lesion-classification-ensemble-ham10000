# src/evaluate.py
from __future__ import annotations

import argparse
from typing import Dict

import numpy as np
import torch
from torchvision import datasets, transforms

from src.models import get_model
from src.utils import compute_metrics, save_json


def _best_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _val_tfms(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ]
    )


def _safe_load(model: torch.nn.Module, checkpoint_path: str, device: torch.device) -> None:
    """Load a checkpoint but drop any params that are missing or shape-mismatched."""
    state = torch.load(checkpoint_path, map_location=device)
    ckpt = state["model"] if isinstance(state, dict) and "model" in state else state
    msd = model.state_dict()
    filtered = {k: v for k, v in ckpt.items() if k in msd and msd[k].shape == v.shape}
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    dropped = sorted(set(ckpt.keys()) - set(filtered.keys()))
    if dropped or missing or unexpected:
        print(f"[warn] non-strict filtered load:"
              f" dropped={len(dropped)} keys, missing={list(missing)}, unexpected={list(unexpected)}")


@torch.no_grad()
def evaluate_once(
    checkpoint: str,
    data_dir: str,
    model_name: str,
    num_classes: int,
    image_size: int,
) -> Dict[str, float | None]:
    device = _best_device()
    model = get_model(model_name, num_classes=num_classes).to(device)
    _safe_load(model, checkpoint, device)
    model.eval()

    ds = datasets.ImageFolder(data_dir, transform=_val_tfms(image_size))
    loader = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=False, num_workers=2)

    y_true, y_pred, probs = [], [], []
    for images, targets in loader:
        logits = model(images.to(device))
        p = torch.softmax(logits, dim=1).cpu().numpy()  # (B, C)
        probs.append(p)
        y_pred.extend(p.argmax(1).tolist())
        y_true.extend(targets.numpy().tolist())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.concatenate(probs, axis=0) if probs else None
    return compute_metrics(y_true, y_pred, y_prob=y_prob)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Evaluate a single checkpoint on an ImageFolder validation set.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--model", default="densenet121")
    ap.add_argument("--num_classes", type=int, default=7)
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--out", required=True)
    ap.add_argument("--save_csv", default=None,
                    help="Optional path to write per-sample probabilities.")
    args = ap.parse_args()

    metrics = evaluate_once(
        checkpoint=args.checkpoint,
        data_dir=args.data_dir,
        model_name=args.model,
        num_classes=args.num_classes,
        image_size=args.image_size,
    )
    save_json(metrics, args.out)

    if args.save_csv:
        import csv

        device = _best_device()
        model = get_model(args.model, num_classes=args.num_classes).to(device)
        _safe_load(model, args.checkpoint, device)
        model.eval()

        ds = datasets.ImageFolder(args.data_dir, transform=_val_tfms(args.image_size))
        loader = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=False, num_workers=2)

        with open(args.save_csv, "w", newline="") as f:
            w = csv.writer(f)
            # exact header: integer y_true then p_0..p_{C-1}
            w.writerow(["y_true"] + [f"p_{i}" for i in range(args.num_classes)])

            with torch.no_grad():
                for images, targets in loader:
                    logits = model(images.to(device))
                    probs = torch.softmax(logits, dim=1).detach().cpu().numpy()  # (B, C)
                    labels = targets.detach().cpu().numpy().astype(int).tolist() # (B,)
                    for t, row in zip(labels, probs.tolist()):
                        w.writerow([int(t)] + [f"{float(x):.8f}" for x in row])

        print(f"[csv] wrote {args.save_csv}")


if __name__ == "__main__":
    main()
