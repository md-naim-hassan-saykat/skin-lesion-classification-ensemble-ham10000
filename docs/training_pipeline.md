# Training Pipeline

This document describes the modular training workflow for the **Generalizable Ensemble Deep Learning for Skin Lesion Classification** project.  
The implementation emphasizes reproducibility, transferability, and cross-architecture comparison.

---

## 1. Overview

The pipeline automates:
1. Data loading and augmentation  
2. Model initialization  
3. Training with validation checkpoints  
4. Metric evaluation  
5. Result storage and logging  

All components are implemented in `src/train.py` and `src/evaluate.py`.

---

## 2. Supported Architectures

- ResNet-50  
- DenseNet-121  
- EfficientNet-B3  
- ConvNeXt-Tiny  
- MobileNetV3-Large  
- Vision Transformer (ViT-B/16)

Each uses identical preprocessing and augmentation for fair comparison.

---

## 3. Workflow

```text
1. Load config.yaml (hyperparameters, paths, seeds)
2. Prepare data loaders (train/val)
3. Initialize model + optimizer + scheduler
4. Train until early stopping / max epochs
5. Save best checkpoints
6. Evaluate on HAM10000 and ISIC 2019
7. Store metrics in results/
```

---

## 4. Execution
Local run
```bash
source .venv/bin/activate
export PYTHONPATH="$PWD"
python src/train.py --config src/config.yaml
python src/evaluate.py --weights checkpoints/model_best.pt
```

Batch evaluation
```bash
bash scripts/eval_all.sh
```

---

## 5. Outputs
Results include:

	•	Model checkpoints in checkpoints/
	•	Metrics JSONs in results/
	•	Figures (ROC, Confusion Matrix, Grad-CAM) in results/figures/
	•	Summary tables (LaTeX-ready) in results/tables/

---

## 6. Reproducibility Notes

	•	Random seeds fixed in config.yaml
	•	Deterministic PyTorch backend
	•	Evaluation identical across models and datasets

