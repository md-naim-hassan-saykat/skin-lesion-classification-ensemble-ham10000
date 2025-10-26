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
from torchvision import datasets, transforms, models

from src.utils import compute_metrics, save_json

try:
    from tqdm import tqdm
except Exception:
    tqdm = None  # optional


def infer_model_name_from_ckpt(path: str) -> str:
    """Infer model name from checkpoint filename or folder."""
    p = Path(path)
    return p.stem.replace("_best", "")


def get_model(name: str, num_classes: int):
    name = name.lower()
    if name == "resnet50":
        m = models.resnet50()
        m.fc = torch.nn.Linear(m.fc.in_features, num_classes)
    elif name == "densenet121":
        m = models.densenet121()
        m.classifier = torch.nn.Linear(m.classifier.in_features, num_classes)
    elif name == "efficientnet_b3":
        m = models.efficientnet_b3()
        m.classifier[1] = torch.nn.Linear(m.classifier[1].in_features, num_classes)
    elif name == "convnext_tiny":
        m = models.convnext_tiny()
        m.classifier[2] = torch.nn.Linear(m.classifier[2].in_features, num_classes)
    elif name == "mobilenet_v3_small":
        m = models.mobilenet_v3_small()
        m.classifier[3] = torch.nn.Linear(m.classifier[3].in_features, num_classes)
    elif name == "vit_b_16":
        m = models.vit_b_16()
        m.heads.head = torch.nn.Linear(m.heads.head.in_features, num_classes)
    else:
        raise ValueError(f"Unknown model name: {name}")
    return m


def _best_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    # support Apple Silicon
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_state_dict(ckpt_path: str) -> Dict[str, Any]:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict):
        # common keys used in training scripts
        for key in ("model", "state_dict", "model_state_dict"):
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]
        # maybe the dict itself is the state_dict
        # (e.g., torch.save(model.state_dict(), path))
        if all(isinstance(k, str) and hasattr(v, "size") for k, v in ckpt.items()):
            return ckpt  # looks like a state_dict
    raise RuntimeError(f"Unrecognized checkpoint format at {ckpt_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, type=str)
    ap.add_argument("--data_dir", required=True, type=str, help="Folder with class subdirs (ImageFolder)")
    ap.add_argument("--out", required=True, type=str, help="Output JSON for metrics")
    ap.add_argument("--save_csv", default=None, type=str, help="Optional CSV to save predictions (for ensembling)")
    ap.add_argument("--num_classes", default=7, type=int)
    args = ap.parse_args()

    device = _best_device()

    # Note: 224 works for most backbones above; EfficientNet-B3 is typically 300px.
    # We keep 224 for consistency with provided checkpoints and speed.
    tfms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    ds = datasets.ImageFolder(args.data_dir, transform=tfms)

    # safer defaults on macOS; pin memory only helps on CUDA
    num_workers = min(4, (torch.get_num_threads() or 1))
    pin_memory = device.type == "cuda"

    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=32,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=False,
    )

    model_name = infer_model_name_from_ckpt(args.checkpoint)
    model = get_model(model_name, args.num_classes)

    state_dict = _load_state_dict(args.checkpoint)
    model.load_state_dict(state_dict, strict=False)
    model.to(device).eval()

    y_true, y_pred, all_probs = [], [], []
    filenames = [p for p, _ in ds.samples]  # for CSV traceability

    it = loader
    if tqdm is not None:
        it = tqdm(loader, desc=f"Evaluating {model_name}", unit="batch")

    with torch.no_grad():
        for images, targets in it:
            images = images.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
            preds = probs.argmax(1)
            y_true.extend(targets.numpy().tolist())
            y_pred.extend(preds.tolist())
            all_probs.append(probs)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.concatenate(all_probs, axis=0)

    metrics = compute_metrics(y_true, y_pred, y_prob=y_prob)
    save_json(metrics, args.out)
    print("Metrics saved to:", args.out, metrics)

    if args.save_csv:
        import csv
        # map idx->class name for readability
        idx_to_class = {v: k for k, v in ds.class_to_idx.items()}
        header = ["index", "filename", "true", "true_name", "pred", "pred_name"] + [
            f"p_{i}_{idx_to_class.get(i, str(i))}" for i in range(args.num_classes)
        ]
        with open(args.save_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for i in range(len(y_true)):
                row = [
                    i,
                    filenames[i],
                    int(y_true[i]),
                    idx_to_class.get(int(y_true[i]), str(int(y_true[i]))),
                    int(y_pred[i]),
                    idx_to_class.get(int(y_pred[i]), str(int(y_pred[i]))),
                ] + y_prob[i].tolist()
                writer.writerow(row)
        print("Predictions CSV saved to:", args.save_csv)


if __name__ == "__main__":
    main()
