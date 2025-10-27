# Installation Guide

This document explains how to set up the environment for running experiments on the **Generalizable Ensemble Deep Learning for Skin Lesion Classification: Internal and External Validation on HAM10000 and ISIC 2019** project.

## Requirements
- Python 3.9+
- CUDA 11+ (for GPU training)
- Git and Git LFS installed

## Setup Steps
```bash
# Clone the repository
git clone https://github.com/md-naim-hassan-saykat/skin-lesion-classification-ensemble-ham10000.git
cd skin-lesion-classification-ensemble-ham10000

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate    # Windows

# Install dependencies
pip install -r requirements.txt -r requirements-ci.txt

# Enable Git LFS
git lfs install
```
