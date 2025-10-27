"""Evaluate a saved checkpoint on an ImageFolder split.

Usage:
  python -m src.evaluate \
      --checkpoint ./outputs/resnet50/resnet50_best.pth \
      --data_dir ./data/HAM10000/val \
      --out ./outputs/resnet50/val_metrics.json \
      --save_csv ./outputs/resnet50/val_preds.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import torch
from torchvision import datasets, transforms

from src.utils import compute_metrics, save_json
from src.models import get_model


def _best_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_state_dict(ckpt_path: str) -> Dict[str, Any]:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict):
        for k in ("model", "state_dict", "model_state_dict"):
            if k in ckpt and isinstance(ckpt[k], dict):
                return ckpt[k]
        if all(isinstance(k, str) and hasattr(v, "size") for k, v in ckpt.items()):
            return ckpt  # looks like a plain state_dict
    raise RuntimeError(f"Unrecognized checkpoint format: {ckpt_path}")


def infer_model_name_from_ckpt(path: str) -> str:
    return Path(path).stem.replace("_best", "")


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate a checkpoint on an ImageFolder split.")
    ap.add_argument("--checkpoint", required=True, type=str)
    ap.add_argument("--data_dir", required=True, type=str, help="Folder with class subdirs (ImageFolder).")
    ap.add_argument("--out", required=True, type=str, help="Output JSON for metrics.")
    ap.add_argument("--save_csv", default=None, type=str, help="Optional CSV of predictions.")
    ap.add_argument("--num_classes", default=7, type=int)
    ap.add_argument("--image_size", default=224, type=int)
    ap.add_argument("--batch_size", default=32, type=int)
    args = ap.parse_args()

    device = _best_device()

    tfms = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    ds = datasets.ImageFolder(args.data_dir, transform=tfms)
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=min(4, (torch.get_num_threads() or 1)),
        pin_memory=(device.type == "cuda"),
        persistent_workers=False,
    )

    model_name = infer_model_name_from_ckpt(args.checkpoint)
    model = get_model(model_name, args.num_classes)

    state_dict = _load_state_dict(args.checkpoint)
    model.load_state_dict(state_dict, strict=False)
    model.to(device).eval()

    y_true, y_pred, probs_batches = [], [], []
    filenames = [p for p, _ in ds.samples]

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = probs.argmax(1)
            y_true.extend(targets.numpy().tolist())
            y_pred.extend(preds.tolist())
            probs_batches.append(probs)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.concatenate(probs_batches, axis=0)

    metrics = compute_metrics(y_true, y_pred, y_prob=y_prob)
    save_json(metrics, args.out)
    print("Metrics saved to:", args.out, metrics)

    if args.save_csv:
        import csv
        idx_to_class = {v: k for k, v in ds.class_to_idx.items()}
        header = ["index", "filename", "true", "true_name", "pred", "pred_name"] + [
            f"p_{i}_{idx_to_class.get(i, str(i))}" for i in range(args.num_classes)
        ]
        with open(args.save_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            for i in range(len(y_true)):
                w.writerow([
                    i,
                    filenames[i],
                    int(y_true[i]),
                    idx_to_class.get(int(y_true[i]), str(int(y_true[i]))),
                    int(y_pred[i]),
                    idx_to_class.get(int(y_pred[i]), str(int(y_pred[i]))),
                    *y_prob[i].tolist(),
                ])
        print("Predictions CSV saved to:", args.save_csv)


if __name__ == "__main__":
    main()
