"""
Run this ONE CELL FIRST in the Kaggle notebook, before benchmark_overhead.py.

It prints what is actually mounted and whether each checkpoint can be found,
so a wrong dataset path is caught in two seconds instead of mid-benchmark.
Nothing is loaded or modified.
"""

import os
from glob import glob

print(
    "torch:",
    __import__("torch").__version__,
    "| cuda:",
    __import__("torch").cuda.is_available(),
    "|",
    (
        __import__("torch").cuda.get_device_name(0)
        if __import__("torch").cuda.is_available()
        else "NO GPU - enable the T4 accelerator"
    ),
)

try:
    from torch.utils.flop_counter import FlopCounterMode  # noqa: F401

    print("MACs backend: torch.utils.flop_counter (built in, no install needed)")
except ImportError:
    print("MACs backend: FlopCounterMode unavailable, will try thop")

print("\nMounted datasets under /kaggle/input:")
for d in sorted(glob("/kaggle/input/*")):
    print("  ", d)

WANTED = [
    "cnn_ham10000_best_model.pth",
    "resnet50_ham10000.pth",
    "densenet121_ham10000.pth",
    "efficientnet_b3_best.pth",
    "convnext_tiny_ham10000.pth",
    "mobilenetv3_ham10000.pth",
    "vit_ham10000_best_model.pth",
]
print("\nCheckpoint discovery:")
missing = []
for w in WANTED:
    hits = glob(f"/kaggle/input/**/{w}", recursive=True)
    if hits:
        p = sorted(hits, key=len)[0]
        print(f"  OK      {w:<34} {os.path.getsize(p)/1024**2:7.1f} MB  {p}")
    else:
        print(f"  MISSING {w}")
        missing.append(w)

print("\nViT config files next to the ViT checkpoint:")
vit = glob("/kaggle/input/**/vit_ham10000_best_model.pth", recursive=True)
if vit:
    d = os.path.dirname(sorted(vit, key=len)[0])
    found = sorted(os.listdir(d))
    print("  ", d)
    print("   contents:", ", ".join(found) if found else "(empty)")
    print(
        "   config.json present:",
        "config.json" in found,
        "(not required; the benchmark falls back to ViTConfig() defaults)",
    )
else:
    print("   ViT checkpoint not found")

print(
    "\n"
    + (
        "All checkpoints located."
        if not missing
        else f"{len(missing)} checkpoint(s) missing: attach those datasets, "
        "or accept n/a in the Checkpoint (MB) column."
    )
)
