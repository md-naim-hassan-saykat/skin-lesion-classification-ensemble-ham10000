# Training Pipeline

This document describes the training workflow used in the **Generalizable Ensemble Deep Learning for Skin Lesion Classification: Internal and External Validation on HAM10000 and ISIC 2019** project.  
The pipeline is implemented using **PyTorch** and is designed for modularity, reproducibility, and scalability across multiple CNN architectures.

---

## 1. Overview

The training pipeline handles all stages of model development — from data loading to checkpointing and evaluation.  
It supports multiple architectures such as:

- **ResNet-50**
- **DenseNet-121**
- **EfficientNet-B3**
- **ConvNeXt-Tiny**
- **MobileNetV3-Large**
- **Vision Transformer (ViT-B/16)**

Each model is trained independently with the same preprocessing and augmentation settings to ensure fair comparison and ensemble compatibility.

---

## 2. Pipeline Structure

The high-level workflow consists of the following stages:

```text
1. Load configuration from config.yaml
2. Prepare training and validation data loaders
3. Initialize model and optimizer
4. Train model with periodic validation
5. Save best-performing checkpoints
6. Evaluate final model performance
```
