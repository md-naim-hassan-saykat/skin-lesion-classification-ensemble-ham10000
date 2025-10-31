# Training Pipeline

This document describes the modular training workflow for the **Generalizable Ensemble Deep Learning for Skin Lesion Classification: Internal and External Validation on HAM10000 and ISIC 2019** project.  
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

### Local evaluation

```bash
# Activate virtual environment
source .venv/bin/activate
export PYTHONPATH="$PWD"

# Evaluate pretrained models on HAM10000 split
# (Ensure your dataset is in: data/HAM10000/split/{akiec,bcc,bkl,df,mel,nv,vasc})

# ConvNeXt-Tiny
PYTHONPATH="$PWD" python src/evaluate.py \
  --model convnext_tiny \
  --checkpoint checkpoints/convnext_tiny_ham10000.pt \
  --data_dir data/HAM10000/split \
  --out outputs/eval_convnext_tiny.json \
  --save_csv outputs/convnext_val_preds.csv

# DenseNet-121
PYTHONPATH="$PWD" python src/evaluate.py \
  --model densenet121 \
  --checkpoint checkpoints/densenet121_ham10000.pt \
  --data_dir data/HAM10000/split \
  --out outputs/eval_densenet121.json \
  --save_csv outputs/densenet_val_preds.csv

# ResNet-50
PYTHONPATH="$PWD" python src/evaluate.py \
  --model resnet50 \
  --checkpoint checkpoints/resnet50_ham10000.pt \
  --data_dir data/HAM10000/split \
  --out outputs/eval_resnet50.json \
  --save_csv outputs/resnet_val_preds.csv
```
### Ensemble Evaluation
```bash
PYTHONPATH="$PWD" python src/ensemble.py \
  --csvs outputs/convnext_val_preds.csv \
         outputs/densenet_val_preds.csv \
         outputs/resnet_val_preds.csv \
  --out outputs/ensemble_metrics.json
```

### Batch evaluation
```bash
# Run all evaluations and the ensemble in one go
bash scripts/eval_all.sh
```

### (Optional) Retrain locally
```bash
source .venv/bin/activate
export PYTHONPATH="$PWD"

# Train from scratch or fine-tune using default config
python src/train.py --config src/config.yaml
```
### Notes

	•	Dataset structure
    data/HAM10000/split/
    akiec/
    bcc/
    bkl/
    df/
    mel/
    nv/
    vasc/

    	•	Checkpoints must be placed in checkpoints/
    (convnext_tiny_ham10000.pt, densenet121_ham10000.pt, resnet50_ham10000.pt)
	    •	Results and prediction CSVs are saved automatically in outputs/
    (eval_*.json, *_val_preds.csv, ensemble_metrics.json)

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

