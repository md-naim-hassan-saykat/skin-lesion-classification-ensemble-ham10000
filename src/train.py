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
from typing import Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

from utils import compute_metrics, seed_everything, load_yaml, save_json


def get_model(name: str, num_classes: int) -> torch.nn.Module:
    name = name.lower()
    if name == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif name == "densenet121":
        m = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        m.classifier = nn.Linear(m.classifier.in_features, num_classes)
    elif name == "efficientnet_b3":
        m = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    elif name == "convnext_tiny":
        m = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        m.classifier[2] = nn.Linear(m.classifier[2].in_features, num_classes)
    elif name == "mobilenet_v3_small":
        m = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, num_classes)
    elif name == "vit_b_16":
        m = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        m.heads.head = nn.Linear(m.heads.head.in_features, num_classes)
    else:
        raise ValueError(f"Unknown model name: {name}")
    return m


def build_loaders(root: str, image_size: int, batch_size: int) -> Tuple[DataLoader, DataLoader, List[str]]:
    train_dir = os.path.join(root, "train")
    val_dir = os.path.join(root, "val")

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_tfms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(0.2,0.2,0.2,0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    val_tfms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_ds = datasets.ImageFolder(train_dir, transform=train_tfms)
    val_ds = datasets.ImageFolder(val_dir, transform=val_tfms)

    classes = train_ds.classes

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, val_loader, classes


def compute_class_weights(dataset: datasets.ImageFolder, num_classes: int) -> torch.Tensor:
    counts = np.zeros(num_classes, dtype=np.int64)
    for _, label in dataset.samples:
        counts[label] += 1
    weights = 1.0 / np.clip(counts, 1, None)
    weights = weights * (num_classes / weights.sum())
    return torch.tensor(weights, dtype=torch.float32)


def validate(model, loader, device, num_classes: int):
    model.eval()
    y_true, y_pred = [], []
    all_probs = []

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(1)

            y_true.extend(targets.cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())
            all_probs.append(probs.cpu().numpy())

    y_prob = np.concatenate(all_probs, axis=0) if all_probs else None
    metrics = compute_metrics(np.array(y_true), np.array(y_pred), y_prob=y_prob)
    return metrics, y_true, y_pred, y_prob


def main():
    parser = argparse.ArgumentParser(description="Train a classifier on HAM10000-like ImageFolder data.")
    parser.add_argument("--config", default="config.yaml", type=str)
    parser.add_argument("--model", default=None, type=str, help="Override model name from config.yaml")
    parser.add_argument("--data_root", default=None, type=str, help="Override data root directory")
    parser.add_argument("--out_dir", default=None, type=str, help="Where to save outputs (folder will be created)")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    seed_everything(cfg.get("seed", 42))

    model_name = args.model or cfg["model"]["name"]
    data_root = args.data_root or cfg["data"]["root"]
    out_dir = Path(args.out_dir or os.path.join(cfg["output"]["dir"], model_name))
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = cfg.get("num_classes", 7)
    image_size = cfg.get("image_size", 224)
    batch_size = cfg["train"]["batch_size"]
    epochs = cfg["train"]["epochs"]
    lr = cfg["train"]["lr"]
    weight_decay = cfg["train"].get("weight_decay", 1e-4)
    patience = cfg["train"].get("early_stop_patience", 5)

    train_loader, val_loader, classes = build_loaders(data_root, image_size, batch_size)

    model = get_model(model_name, num_classes).to(device)

    # Weighted CrossEntropy for class imbalance (weights computed from training set)
    train_ds = train_loader.dataset
    class_weights = compute_class_weights(train_ds, num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))

    best_f1 = -1.0
    patience_left = patience

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for images, targets in train_loader:
            images = images.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        scheduler.step()

        # Validate
        val_metrics, y_true, y_pred, y_prob = validate(model, val_loader, device, num_classes)
        avg_loss = running_loss / len(train_loader.dataset)

        print(
            f"Epoch {epoch:03d}: "
            f"loss={avg_loss:.4f} | "
            f"val_acc={val_metrics['accuracy']:.4f} | "
            f"val_f1={val_metrics['f1']:.4f} | "
            f"val_auc={val_metrics['auc']}"
       )

        # Early stopping on F1
        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            patience_left = patience

            # Save checkpoint
            ckpt_path = out_dir / f"{model_name}_best.pth"
            torch.save({"model": model.state_dict(), "classes": classes}, ckpt_path)

            # Save metrics
            save_json({"epoch": epoch, **val_metrics}, out_dir / "metrics.json")


            # Save predictions CSV for ensembling later
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
