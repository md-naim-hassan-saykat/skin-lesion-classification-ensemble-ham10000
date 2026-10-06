"""
Computational cost of each model and of the seven-model ensemble (manuscript Table 6).

Run this ON KAGGLE with the T4 accelerator enabled, in a notebook that has the seven
checkpoints attached as datasets. It measures, per model and for the full ensemble:

    trainable parameters, MACs at the model's own evaluation resolution,
    checkpoint size on disk, peak GPU memory during inference,
    and per-image latency at batch size 32.

It prints a LaTeX table body for the computational-cost table of the manuscript
and writes computational_overhead.json.

Nothing here retrains or modifies any checkpoint; it is inference-only.

Notes on robustness:

  * Checkpoint paths are auto-discovered. If a configured path does not exist the
    script searches /kaggle/input recursively for a file of the same name, so the
    exact dataset mount form (/kaggle/input/<slug>/ versus
    /kaggle/input/datasets/<user>/<slug>/) no longer matters.

  * The ViT config no longer depends on a local config.json. It tries the checkpoint's
    own directory, then the Hub id, then falls back to ViTConfig() defaults, which are
    exactly ViT-B/16 at 224 with patch 16. The fallback needs no files and no network.

  * MACs are counted with torch.utils.flop_counter.FlopCounterMode, built into
    PyTorch 2.x, so thop does not need to be installed. thop is used only if
    FlopCounterMode is unavailable.

  * A failure on one model no longer aborts the run; that row is reported as n/a and
    the remaining models still measure.
"""

import json
import os
import time
from glob import glob

import torch
import torch.nn as nn
from torchvision.models import (
    convnext_tiny,
    densenet121,
    efficientnet_b3,
    mobilenet_v3_large,
    resnet50,
)

# --- MACs backend: prefer the built-in counter, fall back to thop -------------
FLOP_BACKEND = None
try:
    from torch.utils.flop_counter import FlopCounterMode

    FLOP_BACKEND = "torch"
except ImportError:
    try:
        from thop import profile as thop_profile

        FLOP_BACKEND = "thop"
    except ImportError:
        print("No MACs backend available; MACs will be reported as n/a.")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH = 32
N_WARMUP = 50
N_MEASURE = 200
NUM_CLASSES = 7
KAGGLE_INPUT = "/kaggle/input"

# Configured paths. If one does not exist, the basename is searched for under
# /kaggle/input, so these act as hints rather than hard requirements.
CKPT = {
    "CNN": f"{KAGGLE_INPUT}/kaggleinputham10000-model-weights/cnn_ham10000_best_model.pth",
    "ResNet-50": f"{KAGGLE_INPUT}/resnet50-ham10000-model-weights/resnet50_ham10000.pth",
    "DenseNet-121": f"{KAGGLE_INPUT}/densenet121-ham10000-model-weights/densenet121_ham10000.pth",
    "EfficientNet-B3": f"{KAGGLE_INPUT}/efficientnetb3-ham10000-model-weights/efficientnet_b3_best.pth",
    "ConvNeXt-Tiny": f"{KAGGLE_INPUT}/ham10000-convnext-tiny-weights/convnext_tiny_ham10000.pth",
    "MobileNetV3-L": f"{KAGGLE_INPUT}/mobilenetv3-ham10000-model-weights/mobilenetv3_ham10000.pth",
    "ViT-B/16": f"{KAGGLE_INPUT}/vit-ham10000-model-weights/vit_ham10000_best_model.pth",
}

# Evaluation resolution per model - must match Supplementary Table S6.
RESOLUTION = {
    "CNN": 224,
    "ResNet-50": 224,
    "DenseNet-121": 224,
    "EfficientNet-B3": 300,
    "ConvNeXt-Tiny": 224,
    "MobileNetV3-L": 224,
    "ViT-B/16": 224,
}

_resolved = {}


def resolve_ckpt(name):
    """Return the real path of a checkpoint, searching /kaggle/input if needed."""
    if name in _resolved:
        return _resolved[name]
    hint = CKPT[name]
    path = hint if os.path.exists(hint) else None
    if path is None:
        hits = glob(f"{KAGGLE_INPUT}/**/{os.path.basename(hint)}", recursive=True)
        if hits:
            path = sorted(hits, key=len)[0]
            print(f"  [{name}] found at {path}")
        else:
            print(f"  [{name}] checkpoint not found; size reported as n/a")
    _resolved[name] = path
    return path


class SimpleCNN(nn.Module):
    """The custom 4-block baseline, as defined in the evaluation notebook."""

    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 14 * 14, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def build_vit():
    """ViT-B/16 with a 7-class head, without depending on a local config.json."""
    from transformers import ViTConfig, ViTForImageClassification

    ckpt = resolve_ckpt("ViT-B/16")
    candidates = []
    if ckpt:
        candidates.append(os.path.dirname(ckpt))
    candidates.append("google/vit-base-patch16-224")

    for src in candidates:
        try:
            if src.startswith("/") and not os.path.exists(os.path.join(src, "config.json")):
                continue  # no local config: do not let transformers treat it as a Hub id
            cfg = ViTConfig.from_pretrained(src)
            cfg.num_labels = NUM_CLASSES
            print(f"  [ViT-B/16] config from {src}")
            return ViTForImageClassification(cfg)
        except Exception as exc:
            print(f"  [ViT-B/16] config source {src} unusable ({type(exc).__name__})")

    # ViTConfig() defaults are exactly ViT-B/16 at 224 with patch size 16:
    # hidden 768, 12 layers, 12 heads, intermediate 3072. No files, no network.
    print("  [ViT-B/16] config from ViTConfig() defaults (ViT-B/16 224, patch 16)")
    cfg = ViTConfig(num_labels=NUM_CLASSES)
    return ViTForImageClassification(cfg)


def build(name):
    """Return an architecture with a 7-class head (weights loaded separately)."""
    if name == "CNN":
        return SimpleCNN()
    if name == "ResNet-50":
        m = resnet50(weights=None)
        m.fc = nn.Linear(m.fc.in_features, NUM_CLASSES)
        return m
    if name == "DenseNet-121":
        m = densenet121(weights=None)
        m.classifier = nn.Linear(m.classifier.in_features, NUM_CLASSES)
        return m
    if name == "EfficientNet-B3":
        m = efficientnet_b3(weights=None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, NUM_CLASSES)
        return m
    if name == "ConvNeXt-Tiny":
        m = convnext_tiny(weights=None)
        m.classifier[2] = nn.Linear(m.classifier[2].in_features, NUM_CLASSES)
        return m
    if name == "MobileNetV3-L":
        m = mobilenet_v3_large(weights=None)
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, NUM_CLASSES)
        return m
    if name == "ViT-B/16":
        return build_vit()
    raise ValueError(name)


class LogitsOnly(nn.Module):
    """Unwrap HuggingFace outputs so the counters and timing loop see a plain tensor."""

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        out = self.m(x)
        return out.logits if hasattr(out, "logits") else out


def count_macs(model, res):
    """MACs in G at the given square resolution, or nan if no backend works."""
    x = torch.randn(1, 3, res, res, device=DEVICE)
    try:
        if FLOP_BACKEND == "torch":
            counter = FlopCounterMode(display=False)
            with counter, torch.no_grad():
                model(x)
            # FlopCounterMode reports 2 flops per multiply-accumulate.
            return counter.get_total_flops() / 2 / 1e9
        if FLOP_BACKEND == "thop":
            with torch.no_grad():
                return thop_profile(model, inputs=(x,), verbose=False)[0] / 1e9
    except Exception as exc:
        print(f"  MACs counting failed: {type(exc).__name__}: {exc}")
    return float("nan")


def measure(name):
    res = RESOLUTION[name]
    nan = float("nan")
    try:
        model = build(name)
    except Exception as exc:
        print(f"  [{name}] BUILD FAILED: {type(exc).__name__}: {exc}")
        return dict(
            model=name,
            params_M=nan,
            input=res,
            macs_G=nan,
            ckpt_MB=nan,
            peak_mem_MB=nan,
            ms_per_image=nan,
        )

    ckpt_mb = nan
    path = resolve_ckpt(name)
    if path:
        ckpt_mb = os.path.getsize(path) / 1024**2
        try:
            sd = torch.load(path, map_location="cpu", weights_only=False)
            if isinstance(sd, dict) and "state_dict" in sd:
                sd = sd["state_dict"]
            model.load_state_dict(sd)
        except Exception as exc:
            print(
                f"  [{name}] weights not loaded ({type(exc).__name__}); cost metrics "
                f"are architecture-determined and remain valid"
            )

    model = LogitsOnly(model).to(DEVICE).eval()
    params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    macs = count_macs(model, res)

    x = torch.randn(BATCH, 3, res, res, device=DEVICE)
    with torch.no_grad():
        for _ in range(N_WARMUP):
            model(x)
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        for _ in range(N_MEASURE):
            model(x)
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

    peak_mb = torch.cuda.max_memory_allocated() / 1024**2 if DEVICE.type == "cuda" else nan
    ms_per_image = elapsed / (N_MEASURE * BATCH) * 1000

    del model, x
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    return dict(
        model=name,
        params_M=params,
        input=res,
        macs_G=macs,
        ckpt_MB=ckpt_mb,
        peak_mem_MB=peak_mb,
        ms_per_image=ms_per_image,
    )


def fmt(v, nd=2):
    return "n/a" if v != v else f"{v:.{nd}f}"


def nansum(vals):
    good = [v for v in vals if v == v]
    return sum(good) if good else float("nan")


def main():
    print(
        f"device: {DEVICE}"
        f"{' - ' + torch.cuda.get_device_name(0) if DEVICE.type == 'cuda' else ''}"
    )
    print(f"MACs backend: {FLOP_BACKEND or 'none'}")
    print(f"batch {BATCH}, {N_WARMUP} warm-up + {N_MEASURE} measured iterations\n")

    rows = []
    for name in CKPT:
        print(f"measuring {name} ...")
        rows.append(measure(name))

    total = dict(
        model="Ensemble (all 7)",
        params_M=nansum([r["params_M"] for r in rows]),
        input="-",
        macs_G=nansum([r["macs_G"] for r in rows]),
        ckpt_MB=nansum([r["ckpt_MB"] for r in rows]),
        peak_mem_MB=max(
            [r["peak_mem_MB"] for r in rows if r["peak_mem_MB"] == r["peak_mem_MB"]]
            or [float("nan")]
        ),
        ms_per_image=nansum([r["ms_per_image"] for r in rows]),
    )

    with open("computational_overhead.json", "w") as fh:
        json.dump(rows + [total], fh, indent=2)

    print("\n" + "=" * 78)
    print("LaTeX table body - paste over the red [measure] cells in the manuscript")
    print("=" * 78)
    for r in rows:
        print(
            f"{r['model']:<16} & {fmt(r['params_M'], 3):>7} & ${r['input']}$ & "
            f"{fmt(r['macs_G']):>6} & {fmt(r['ckpt_MB'], 0):>5} & "
            f"{fmt(r['ms_per_image']):>6} \\\\"
        )
    print("\\midrule")
    print(
        f"{'Ensemble (all 7)':<16} & {fmt(total['params_M'], 3):>7} & - & "
        f"{fmt(total['macs_G']):>6} & {fmt(total['ckpt_MB'], 0):>5} & "
        f"{fmt(total['ms_per_image']):>6} \\\\"
    )
    print("=" * 78)
    print(
        f"\nPeak GPU memory, largest single model: {fmt(total['peak_mem_MB'], 0)} MB "
        f"at batch {BATCH}."
    )
    print("Components run sequentially, so ensemble peak memory equals the largest single")
    print("component; quote that in the caption rather than a sum.")
    print("Saved: computational_overhead.json")


if __name__ == "__main__":
    main()
