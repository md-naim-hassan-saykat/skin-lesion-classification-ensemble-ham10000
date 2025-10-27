### **2. `training_pipeline.md`**
**Goal:** Match new data loaders, utils, and CLI structure.

**Proposed rewrite:**
```markdown
# Training Pipeline

This document describes the **training process** for the ensemble-based skin lesion classification models.

---

## 1. Overview
The pipeline trains individual CNN architectures (**ResNet50**, **EfficientNet-B3**, **ConvNeXt-Tiny**, **DenseNet121**) on the **HAM10000** dataset using consistent splits and augmentation strategies.

Each model:
- Loads data using `src/data.py`
- Trains via `src/train.py`
- Saves best-performing checkpoints (`.pth`) under `outputs/models/`

---

## 2. Data Loading
Defined in [`src/data.py`](../src/data.py).

Key features:
- Automatic split detection (`train/`, `val/`)
- Configurable image resizing and augmentation
- Support for reproducible `torch.utils.data.DataLoader` pipelines

```python
from src.data import build_loaders
train_loader, val_loader, classes = build_loaders("data/HAM10000_split")
```
