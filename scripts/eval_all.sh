#!/usr/bin/env bash
#
# Evaluate all seven archived HAM10000-trained checkpoints using the
# harmonized evaluation protocol reported in the accompanying manuscript.
#
# Models:
#   - CNN
#   - ResNet-50
#   - DenseNet-121
#   - EfficientNet-B3
#   - ConvNeXt-Tiny
#   - MobileNetV3-Large
#   - ViT-B/16
#
# The final ensemble is an equal-weight probability average across all
# seven models (1/7 per model).
#
# IMPORTANT:
# The standardized HAM10000 evaluation cohort contains 2,003 images and
# is a retrospective harmonized evaluation cohort. It must not be
# described as a universally untouched test split shared by all models.
#
# Usage:
#
#   bash scripts/eval_all.sh
#
# Optional environment overrides:
#
#   HAM_EVAL_DIR=/path/to/ham10000/eval \
#   CHECKPOINT_DIR=/path/to/checkpoints \
#   OUT_DIR=/path/to/output \
#   PYTHON=/path/to/python \
#   bash scripts/eval_all.sh
#

set -euo pipefail

# ---------------------------------------------------------------------
# Repository paths
# ---------------------------------------------------------------------

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"

HAM_EVAL_DIR="${HAM_EVAL_DIR:-$ROOT/data/evaluation/HAM10000_standardized}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$ROOT/checkpoints}"
OUT_DIR="${OUT_DIR:-$ROOT/outputs/harmonized_ham10000}"

mkdir -p "$OUT_DIR"

# ---------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------

if [[ ! -x "$PYTHON" ]]; then
  echo "ERROR: Python executable not found:"
  echo "  $PYTHON"
  echo
  echo "Create the repository environment first, for example:"
  echo "  python -m venv .venv"
  echo "  source .venv/bin/activate"
  echo "  pip install -r requirements.txt"
  exit 1
fi

if [[ ! -d "$HAM_EVAL_DIR" ]]; then
  echo "ERROR: Standardized HAM10000 evaluation cohort not found:"
  echo "  $HAM_EVAL_DIR"
  echo
  echo "Set HAM_EVAL_DIR to the released/reconstructed 2,003-image"
  echo "standardized HAM10000 evaluation cohort."
  exit 1
fi

if [[ ! -d "$CHECKPOINT_DIR" ]]; then
  echo "ERROR: Checkpoint directory not found:"
  echo "  $CHECKPOINT_DIR"
  echo
  echo "Set CHECKPOINT_DIR to the directory containing the fixed"
  echo "archived checkpoints used in the final analysis."
  exit 1
fi

# ---------------------------------------------------------------------
# Evaluation configuration
# ---------------------------------------------------------------------
#
# Format:
#
#   key | model_name | checkpoint_pattern
#
# `key` is used for output file names.
# `model_name` must correspond to the model identifier supported by
# the harmonized evaluator.
#

MODEL_CONFIGS=(
  "cnn|cnn|*cnn*.pth"
  "resnet50|resnet50|*resnet*50*.pth"
  "densenet121|densenet121|*densenet*121*.pth"
  "efficientnet_b3|efficientnet_b3|*efficientnet*b3*.pth"
  "convnext_tiny|convnext_tiny|*convnext*tiny*.pth"
  "mobilenet_v3_large|mobilenet_v3_large|*mobilenet*v3*large*.pth"
  "vit_b_16|vit_b_16|*vit*b*16*.pth"
)

PREDICTION_CSVS=()

# ---------------------------------------------------------------------
# Resolve exactly one checkpoint per model
# ---------------------------------------------------------------------

find_checkpoint() {
  local pattern="$1"

  local matches=()

  while IFS= read -r -d '' f; do
    matches+=("$f")
  done < <(
    find "$CHECKPOINT_DIR" \
      -type f \
      -iname "$pattern" \
      -print0
  )

  if (( ${#matches[@]} == 0 )); then
    return 1
  fi

  if (( ${#matches[@]} > 1 )); then
    echo "ERROR: Multiple checkpoints matched pattern:" >&2
    echo "  $pattern" >&2
    echo >&2
    printf '  %s\n' "${matches[@]}" >&2
    echo >&2
    echo "The final analysis requires one fixed checkpoint per model." >&2
    return 2
  fi

  printf '%s\n' "${matches[0]}"
}

# ---------------------------------------------------------------------
# Evaluate individual models
# ---------------------------------------------------------------------

echo
echo "============================================================"
echo "HAM10000 harmonized evaluation"
echo "============================================================"
echo "Evaluation cohort:"
echo "  $HAM_EVAL_DIR"
echo
echo "Checkpoint directory:"
echo "  $CHECKPOINT_DIR"
echo
echo "Output directory:"
echo "  $OUT_DIR"
echo

for config in "${MODEL_CONFIGS[@]}"; do

  IFS='|' read -r key model_name pattern <<< "$config"

  echo "------------------------------------------------------------"
  echo "Model: $key"
  echo "------------------------------------------------------------"

  ckpt="$(find_checkpoint "$pattern")" || {
    echo "ERROR: Could not resolve fixed checkpoint for $key."
    exit 1
  }

  metrics="$OUT_DIR/${key}_metrics.json"
  predictions="$OUT_DIR/${key}_predictions.csv"

  echo "Checkpoint:"
  echo "  $ckpt"
  echo

  "$PYTHON" "$ROOT/src/evaluate_harmonized.py" \
    --checkpoint "$ckpt" \
    --model "$model_name" \
    --dataset ham10000 \
    --data-dir "$HAM_EVAL_DIR" \
    --metrics-out "$metrics" \
    --predictions-out "$predictions"

  if [[ ! -s "$predictions" ]]; then
    echo "ERROR: Prediction file was not produced:"
    echo "  $predictions"
    exit 1
  fi

  PREDICTION_CSVS+=("$predictions")

done

# ---------------------------------------------------------------------
# Verify ensemble inputs
# ---------------------------------------------------------------------

EXPECTED_MODELS=7

if (( ${#PREDICTION_CSVS[@]} != EXPECTED_MODELS )); then
  echo
  echo "ERROR: Expected $EXPECTED_MODELS model prediction files,"
  echo "but found ${#PREDICTION_CSVS[@]}."
  exit 1
fi

# ---------------------------------------------------------------------
# Equal-weight seven-model ensemble
# ---------------------------------------------------------------------

echo
echo "============================================================"
echo "Equal-weight seven-model ensemble"
echo "============================================================"
echo

"$PYTHON" "$ROOT/src/ensemble.py" \
  --csvs "${PREDICTION_CSVS[@]}" \
  --out "$OUT_DIR/ensemble_metrics.json" \
  --num_classes 7

# ---------------------------------------------------------------------
# Full manuscript analyses
# ---------------------------------------------------------------------

if [[ -f "$ROOT/src/analyze_ham10000.py" ]]; then

  echo
  echo "============================================================"
  echo "HAM10000 statistical/calibration analysis"
  echo "============================================================"
  echo

  "$PYTHON" "$ROOT/src/analyze_ham10000.py" \
    --predictions-dir "$OUT_DIR" \
    --bootstrap-resamples 1000 \
    --seed 42 \
    --ece-bins 15 \
    --out-dir "$OUT_DIR/analysis"

else

  echo
  echo "NOTE:"
  echo "  src/analyze_ham10000.py is not present."
  echo "  Individual inference and ensemble averaging are complete,"
  echo "  but bootstrap confidence intervals, McNemar tests, paired"
  echo "  bootstrap comparisons, ECE, ROC, and confusion-matrix"
  echo "  analyses must be generated separately."

fi

# ---------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------

echo
echo "============================================================"
echo "Evaluation complete"
echo "============================================================"
echo

echo "Individual model metrics:"
for config in "${MODEL_CONFIGS[@]}"; do
  IFS='|' read -r key _ _ <<< "$config"

  file="$OUT_DIR/${key}_metrics.json"

  if [[ -f "$file" ]]; then
    echo "  $file"
  fi
done

echo
echo "Ensemble metrics:"
echo "  $OUT_DIR/ensemble_metrics.json"

echo
echo "Prediction outputs:"
for csv in "${PREDICTION_CSVS[@]}"; do
  echo "  $csv"
done

echo
echo "Important:"
echo "  The ensemble must contain exactly seven fixed model outputs"
echo "  with equal probability weighting (1/7 each)."
echo
echo "  Architecture-specific preprocessing and canonical class"
echo "  harmonization must be performed by the harmonized evaluator."
