# Installation Guide

This document explains how to set up the environment for running experiments on the **Skin Lesion Classification Ensemble** project.

## Requirements
- Python 3.9+
- CUDA 11+ (for GPU training)
- Git and Git LFS installed

## Setup Steps
```bash
# Clone the repository
git clone https://github.com/md-naim-hassan-saykat/skin-lesion-classification-ensemble-ham10000.git
cd skin-lesion-classification-ensemble-ham10000
```

# Create virtual environment
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate    # Windows
```

# Install dependencies
```bash
pip install -r requirements.txt -r requirements-ci.txt
```

# Enable Git LFS
```bash
git lfs install
```
