# src/train.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from src.utils import load_yaml, save_json, seed_everything, compute_metrics
from src.data import build_loaders, compute_class_weights
from src.models import get_model


def best_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
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


def train_one_epoch(model, loader, optimizer, criterion, device) -> float:
    model.train()
    running = 0.0
    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        running += loss.item() * images.size(0)
    return running / len(loader.dataset)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="src/config.yaml", type=str)
    ap.add_argument("--outdir", default=None, type=str, help="Override output dir in config.")
    args = ap.parse_args()

    cfg: Dict[str, Any] = load_yaml(args.config)
    seed_everything(cfg.get("seed", 42))

    # Paths & hyperparams
    data_cfg = cfg["data"]
    train_cfg = cfg["train"]
    model_name = cfg["model"]["name"]
    num_classes = int(cfg["num_classes"])
    image_size = int(cfg.get("image_size", 224))
    outdir = Path(args.outdir or cfg["output"]["dir"]) / model_name
    outdir.mkdir(parents=True, exist_ok=True)

    # Device
    device = best_device()
    pin_memory = device.type == "cuda"  # MPS/CPU -> False

    # Data
    train_loader, val_loader, _ = build_loaders(
        data_root=data_cfg["root"],
        image_size=image_size,
        batch_size=int(train_cfg["batch_size"]),
        num_workers=min(4, (torch.get_num_threads() or 1)),
    )

    # Model
    model = get_model(model_name, num_classes=num_classes).to(device)

    # Loss (with class weights)
    class_weights = compute_class_weights(train_loader.dataset, num_classes).to(device)  # type: ignore[arg-type]
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Optim + sched + early stop
    optimizer = optim.AdamW(model.parameters(),
                            lr=float(train_cfg["lr"]),
                            weight_decay=float(train_cfg["weight_decay"]))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=2, factor=0.5)
    max_epochs = int(train_cfg["epochs"])
    es_patience = int(train_cfg.get("early_stop_patience", 5))

    history = []
    best_f1 = -1.0
    es_count = 0

    for epoch in range(1, max_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, device)
        scheduler.step(val_metrics["f1"] if val_metrics.get("f1") is not None else val_metrics["accuracy"])

        row = {"epoch": epoch, "train_loss": train_loss, **val_metrics}
        history.append(row)
        print(f"[{epoch:03d}/{max_epochs}] loss={train_loss:.4f} "
              f"acc={val_metrics['accuracy']:.4f} f1={val_metrics['f1']:.4f} auc={val_metrics.get('auc')}")

        # Early stop on best F1 (fallback to accuracy)
        score = val_metrics["f1"] if val_metrics.get("f1") is not None else val_metrics["accuracy"]
        if score > best_f1:
            best_f1 = score
            es_count = 0
            torch.save({"model": model.state_dict()}, outdir / f"{model_name}_best.pth")
            save_json(val_metrics, outdir / "best_val_metrics.json")
        else:
            es_count += 1
            if es_count >= es_patience:
                print(f"Early stopping (patience={es_patience})")
                break

    # Save training history and last weights
    save_json({"history": history}, outdir / "train_history.json")
    torch.save({"model": model.state_dict()}, outdir / f"{model_name}_last.pth")
    print("Training complete. Best F1:", best_f1, "Artifacts in:", outdir)


if __name__ == "__main__":
    main()
