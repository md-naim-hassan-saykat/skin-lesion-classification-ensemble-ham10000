from __future__ import annotations

# ruff: noqa: E402

import sys
from pathlib import Path as _P


_PROJECT_ROOT = _P(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from src.data import (
    build_loaders,
    compute_class_weights,
)
from src.models import get_model
from src.utils import (
    compute_metrics,
    load_yaml,
    save_json,
    seed_everything,
)


def best_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")

    if (
        getattr(torch.backends, "mps", None)
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")

    return torch.device("cpu")


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float | None]:
    model.eval()

    y_true = []
    probabilities = []

    for images, targets in loader:
        images = images.to(device)

        logits = model(images)

        p = torch.softmax(
            logits,
            dim=1,
        ).cpu().numpy()

        probabilities.append(p)
        y_true.extend(
            targets.numpy().tolist()
        )

    y_true_arr = np.asarray(
        y_true,
        dtype=int,
    )

    y_prob = np.concatenate(
        probabilities,
        axis=0,
    )

    return compute_metrics(
        y_true_arr,
        y_prob=y_prob,
        dataset="ham10000",
        ece_bins=15,
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()

    running_loss = 0.0

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad(
            set_to_none=True
        )

        logits = model(images)
        loss = criterion(
            logits,
            targets,
        )

        loss.backward()
        optimizer.step()

        running_loss += (
            loss.item()
            * images.size(0)
        )

    return (
        running_loss
        / len(loader.dataset)
    )


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--config",
        default="src/config.yaml",
    )

    ap.add_argument(
        "--model",
        required=True,
        help=(
            "cnn, resnet50, densenet121, efficientnet_b3, "
            "convnext_tiny, mobilenet_v3_large, vit_b_16"
        ),
    )

    ap.add_argument(
        "--outdir",
        default=None,
    )

    args = ap.parse_args()

    cfg: dict[str, Any] = load_yaml(
        args.config
    )

    seed_everything(
        int(cfg.get("seed", 42))
    )

    data_cfg = cfg["data"]
    train_cfg = cfg["train"]

    num_classes = int(
        cfg["num_classes"]
    )

    model_cfg = (
        cfg.get("preprocessing", {})
        .get(args.model, {})
    )

    image_size = int(
        model_cfg.get(
            "image_size",
            224,
        )
    )

    outdir = (
        Path(
            args.outdir
            or cfg["output"]["dir"]
        )
        / args.model
    )

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = best_device()

    train_loader, val_loader, classes = build_loaders(
        data_root=data_cfg["root"],
        image_size=image_size,
        batch_size=int(
            train_cfg["batch_size"]
        ),
        num_workers=int(
            data_cfg.get(
                "num_workers",
                2,
            )
        ),
    )

    expected_classes = cfg[
        "canonical_classes"
    ]

    if classes != expected_classes:
        raise ValueError(
            "Training ImageFolder class ordering differs from canonical order.\n"
            f"Expected: {expected_classes}\n"
            f"Found:    {classes}"
        )

    model = get_model(
        args.model,
        num_classes=num_classes,
        pretrained=bool(
            train_cfg.get(
                "pretrained",
                True,
            )
        ),
    ).to(device)

    class_weights = compute_class_weights(
        train_loader.dataset,  # type: ignore[arg-type]
        num_classes,
    ).to(device)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    optimizer = optim.AdamW(
        model.parameters(),
        lr=float(
            train_cfg["lr"]
        ),
        weight_decay=float(
            train_cfg["weight_decay"]
        ),
    )

    max_epochs = int(
        train_cfg["epochs"]
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max_epochs,
    )

    patience = int(
        train_cfg.get(
            "early_stop_patience",
            7,
        )
    )

    best_f1 = -1.0
    no_improvement = 0
    history = []

    for epoch in range(
        1,
        max_epochs + 1,
    ):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
        )

        metrics = evaluate(
            model,
            val_loader,
            device,
        )

        scheduler.step()

        weighted_f1 = float(
            metrics["weighted_f1"]
            or 0.0
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                **metrics,
            }
        )

        print(
            f"[{epoch:03d}/{max_epochs}] "
            f"loss={train_loss:.4f} "
            f"acc={metrics['accuracy']:.4f} "
            f"weighted_f1={weighted_f1:.4f}"
        )

        if weighted_f1 > best_f1:
            best_f1 = weighted_f1
            no_improvement = 0

            torch.save(
                {
                    "model": model.state_dict(),
                    "model_name": args.model,
                    "classes": classes,
                },
                outdir
                / f"{args.model}_best.pth",
            )

            save_json(
                metrics,
                outdir
                / "best_val_metrics.json",
            )

        else:
            no_improvement += 1

            if no_improvement >= patience:
                print(
                    f"Early stopping "
                    f"(patience={patience})"
                )
                break

    save_json(
        {
            "history": history,
        },
        outdir
        / "train_history.json",
    )

    torch.save(
        {
            "model": model.state_dict(),
            "model_name": args.model,
            "classes": classes,
        },
        outdir
        / f"{args.model}_last.pth",
    )

    print(
        "Training complete. "
        f"Best weighted F1: {best_f1:.6f}"
    )


if __name__ == "__main__":
    main()
