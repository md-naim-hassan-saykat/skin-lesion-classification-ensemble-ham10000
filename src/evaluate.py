"""Evaluate a saved checkpoint on an ImageFolder split.

Usage:
  python src/evaluate.py --checkpoint ./outputs/resnet50/resnet50_best.pth --data_dir ./data/HAM10000/test --out ./outputs/resnet50/test_metrics.json
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from torchvision import datasets, transforms, models

from utils import compute_metrics, save_json


def infer_model_name_from_ckpt(path: str) -> str:
    # Heuristic: extract parent folder name or file prefix
    p = Path(path)
    name = p.stem.replace("_best", "")
    return name


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, type=str)
    ap.add_argument("--data_dir", required=True, type=str, help="Folder with class subdirs (ImageFolder)")
    ap.add_argument("--out", required=True, type=str, help="Output JSON for metrics")
    ap.add_argument("--save_csv", default=None, type=str, help="Optional CSV to save predictions (for ensembling)")
    ap.add_argument("--num_classes", default=7, type=int)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tfms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
    ])
    ds = datasets.ImageFolder(args.data_dir, transform=tfms)
    loader = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    model_name = infer_model_name_from_ckpt(args.checkpoint)
    model = get_model(model_name, args.num_classes)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=False)
    model.to(device).eval()

    y_true, y_pred = [], []
    all_probs = []

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
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
        with open(args.save_csv, "w", newline="") as f:
            writer = csv.writer(f)
            header = ["index", "true", "pred"] + [f"p_{i}" for i in range(args.num_classes)]
            writer.writerow(header)
            for i in range(len(y_true)):
                writer.writerow([i, int(y_true[i]), int(y_pred[i])] + y_prob[i].tolist())
        print("Predictions CSV saved to:", args.save_csv)


if __name__ == "__main__":
    main()
