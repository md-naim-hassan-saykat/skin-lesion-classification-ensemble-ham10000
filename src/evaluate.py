# src/evaluate.py
from __future__ import annotations
import argparse, numpy as np, torch
from torchvision import datasets, transforms
from src.models import get_model
from src.utils import compute_metrics, save_json

def _device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def _tfms(img_size: int):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

def _safe_load(model: torch.nn.Module, ckpt_path: str, device: torch.device) -> None:
    """Load only weights that exist in model AND match shape (drops 1000-class heads)."""
    state = torch.load(ckpt_path, map_location=device)
    raw = state["model"] if isinstance(state, dict) and "model" in state else state
    msd = model.state_dict()
    filt = {k: v for k, v in raw.items() if k in msd and msd[k].shape == v.shape}
    missing, unexpected = model.load_state_dict(filt, strict=False)
    if missing or unexpected or len(filt) != len(raw):
        print(f"[safe_load] kept={len(filt)} dropped={len(raw)-len(filt)} "
              f"missing={len(missing)} unexpected={len(unexpected)}")

@torch.no_grad()
def evaluate_once(checkpoint: str, data_dir: str, model_name: str,
                  num_classes: int, image_size: int):
    device = _device()
    model = get_model(model_name, num_classes=num_classes).to(device)
    _safe_load(model, checkpoint, device)
    model.eval()

    ds = datasets.ImageFolder(data_dir, transform=_tfms(image_size))
    dl = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=False, num_workers=2)

    y_true, y_pred, probs = [], [], []
    for imgs, labels in dl:
        logits = model(imgs.to(device))
        p = torch.softmax(logits, dim=1).cpu().numpy()
        probs.append(p)
        y_true.extend(labels.numpy())
        y_pred.extend(p.argmax(1))
    y_true = np.array(y_true); y_pred = np.array(y_pred)
    y_prob = np.concatenate(probs, axis=0)
    return compute_metrics(y_true, y_pred, y_prob=y_prob)

def main():
    ap = argparse.ArgumentParser(
        description="Evaluate a single checkpoint on an ImageFolder validation set."
    )
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--num_classes", type=int, default=7)
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--out", required=True)
    ap.add_argument("--save_csv", default=None)
    args = ap.parse_args()

    metrics = evaluate_once(args.checkpoint, args.data_dir, args.model,
                            args.num_classes, args.image_size)
    save_json(metrics, args.out)

    if args.save_csv:
        import csv
        device = _device()
        model = get_model(args.model, num_classes=args.num_classes).to(device)
        _safe_load(model, args.checkpoint, device)
        model.eval()

        ds = datasets.ImageFolder(args.data_dir, transform=_tfms(args.image_size))
        dl = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=False, num_workers=2)

        with open(args.save_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["y_true"] + [f"p_{i}" for i in range(args.num_classes)])
            for imgs, labels in dl:
                logits = model(imgs.to(device))
                p = torch.softmax(logits, dim=1).cpu().numpy().tolist()
                for t, row in zip(labels.numpy().astype(int).tolist(), p):
                    w.writerow([t] + [f"{float(x):.8f}" for x in row])
        print(f"[csv] wrote {args.save_csv}")

if __name__ == "__main__":
    main()
