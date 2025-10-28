"""Utility helpers for training/evaluation.

- compute_metrics: accuracy / weighted F1 / (optional) ROC-AUC OvR
- seed_everything: reproducibility across NumPy / PyTorch / cuDNN
- load_yaml / save_json: simple IO helpers
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
