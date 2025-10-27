"""Train a single model on an ImageFolder dataset.

Usage:
  python src/train.py --config config.yaml --model resnet50 --data_root ./data/HAM10000 --out_dir ./outputs/resnet50

- Expects folders: <data_root>/train and <data_root>/val with class subdirectories.
- Saves: best model state_dict (.pth), metrics.json, and val_predictions.csv (with probabilities).
"""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from src.utils import compute_metrics, load_yaml, save_json, seed_everything
from src.data import build_loaders, compute_class_weights           
from src.models import get_model                                    
from src.evaluate import validate                                   


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a classifier on HAM10000-like ImageFolder data.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    parser.add_argument("--model", type=str, default=None, help="Override model name from config.")
    parser.add_argument("--data_root", type=str, default=None, help="Override data root from config.")
    parser.add_argument("--out_dir", type=str, default=None, help="Override output dir.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = load_yaml(args.config)
    seed_everything(cfg.get("seed", 42))

    model_name = (args.model or cfg.get("model", {}).get("name") or "resnet50").lower()
    data_root = args.data_root or cfg.get("data", {}).get("root", "./data/HAM10000")
    out_dir = Path(args.out_dir or os.path.join(cfg.get("output", {}).get("dir", "./outputs"), model_name))
    out_dir.mkdir(parents=True, exist_ok=True)

    # Device selection (CUDA -> MPS -> CPU)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    num_classes = int(cfg.get("num_classes", 7))
    image_size = int(cfg.get("image_size", 224))
    train_cfg = cfg.get("train", {})
    batch_size = int(train_cfg.get("batch_size", 32))
    epochs = int(train_cfg.get("epochs", 10))
    lr = float(train_cfg.get("lr", 3e-4))
    weight_decay = float(train_cfg.get("weight_decay", 1e-4))
    patience = int(train_cfg.get("early_stop_patience", 5))

    # Dataloaders
    train_loader, val_loader, classes = build_loaders(data_root, image_size, batch_size)

    # Model
    model = get_model(model_name, num_classes).to(device)

    # Weighted CrossEntropy for class imbalance
    class_weights = compute_class_weights(train_loader.dataset, num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))

    best_f1 = -1.0
    patience_left = patience

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        scheduler.step()

        val_metrics, y_true, y_pred, y_prob = validate(model, val_loader, device, num_classes)
        avg_loss = running_loss / max(1, len(train_loader.dataset))

        print(
            f"Epoch {epoch:03d}: "
            f"loss={avg_loss:.4f} | "
            f"val_acc={val_metrics['accuracy']:.4f} | "
            f"val_f1={val_metrics['f1']:.4f} | "
            f"val_auc={val_metrics['auc']}"
        )

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            patience_left = patience

            # Save best checkpoint + metrics + predictions
            ckpt_path = out_dir / f"{model_name}_best.pth"
            torch.save({"model": model.state_dict(), "classes": classes}, ckpt_path)

            save_json({"epoch": epoch, **val_metrics}, out_dir / "metrics.json")

            pred_csv = out_dir / "val_predictions.csv"
            with open(pred_csv, "w", newline="") as f:
                writer = csv.writer(f)
                header = ["index", "true", "pred"] + [f"p_{i}" for i in range(num_classes)]
                writer.writerow(header)
                for i, (t, p) in enumerate(zip(y_true, y_pred)):
                    row = [i, t, p]
                    if y_prob is not None:
                        row += y_prob[i].tolist()
                    writer.writerow(row)
        else:
            patience_left -= 1
            if patience_left <= 0:
                print("Early stopping triggered.")
                break

    print("Training finished. Outputs saved to:", str(out_dir))


if __name__ == "__main__":
    main()
