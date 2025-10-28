# src/evaluate.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

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


def _make_val_loader(data_dir: str, image_size: int, batch_size: int, num_workers: int = 0):
    tfms = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )
    ds = datasets.ImageFolder(data_dir, transform=tfms)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return ds, loader


@torch.no_grad()
def _eval(model: torch.nn.Module, loader, device) -> Dict[str, float | None]:
    model.eval()
    y_true, y_pred, probs = [], [], []
    for images, targets in loader:
        images = images.to(device)
        logits = model(images)
        p = torch.softmax(logits, dim=1).cpu().numpy()
        preds = p.argmax(1)
        probs.append(p)
        y_pred.extend(preds.tolist())
        y_true.extend(targets.numpy().tolist())
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.concatenate(probs, axis=0) if probs else None
    return compute_metrics(y_true, y_pred, y_prob=y_prob)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--save_csv", default=None)
    ap.add_argument("--model", default="resnet50")
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--batch_size", type=int, default=64)
    args = ap.parse_args()

    device = _best_device()
    model = get_model(args.model).to(device)
    state = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state["model"] if "model" in state else state)

    ds, loader = _make_val_loader(args.data_dir, args.image_size, args.batch_size)
    metrics = _eval(model, loader, device)

    save_json(metrics, args.out)

    if args.save_csv:
        # write y_true + per-class probs
        import csv  # local to avoid top-level unused import

        model.eval()
        with open(args.save_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            header = ["y_true"] + [f"p_{i}" for i in range(len(ds.classes))]
            writer.writerow(header)
            for images, targets in loader:
                p = torch.softmax(model(images.to(device)), dim=1).cpu().numpy()
                for t, row in zip(targets.numpy().tolist(), p.tolist()):
                    writer.writerow([t] + row)


if __name__ == "__main__":
    main()
