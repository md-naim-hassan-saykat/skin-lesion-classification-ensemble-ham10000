#!/usr/bin/env bash
#
# Evaluate all seven archived HAM10000-trained checkpoints using the
# repository's canonical seven-class evaluation pipeline.
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
# The ensemble is the equal-weight arithmetic mean of the seven
# model probability distributions (1/7 per model).
#
# IMPORTANT:
# The standardized HAM10000 evaluation cohort contains 2,003 images and
# represents the retrospective harmonized evaluation cohort used for the
# final analysis. It should not be described as a universally untouched
# test split that was identically held out during development of every
# archived model.
#
# The evaluation directory must follow the canonical ImageFolder layout:
#
#   <HAM_EVAL_DIR>/
#     akiec/
#     bcc/
#     bkl/
#     df/
#     mel/
#     nv/
#     vasc/
#
# Usage:
#
#   bash scripts/eval_all.sh
#
# Optional environment overrides:
#
#   HAM_EVAL_DIR=/path/to/ham10000/evaluation \
#   CHECKPOINT_DIR=/path/to/checkpoints \
#   OUT_DIR=/path/to/output \
#   PYTHON=/path/to/python \
#   bash scripts/eval_all.sh
#

set -euo pipefail

# -----------------------------------------------------------------------------
# Repository paths
# -----------------------------------------------------------------------------

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
HAM_EVAL_DIR="${HAM_EVAL_DIR:-$ROOT/data/evaluation/HAM10000_standardized}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$ROOT/checkpoints}"
OUT_DIR="${OUT_DIR:-$ROOT/outputs/harmonized_ham10000}"

EVALUATOR="$ROOT/src/evaluate.py"
ENSEMBLER="$ROOT/src/ensemble.py"

mkdir -p "$OUT_DIR"

# -----------------------------------------------------------------------------
# Sanity checks
# -----------------------------------------------------------------------------

if [[ ! -x "$PYTHON" ]]; then
  echo "ERROR: Python executable not found or not executable:"
  echo "  $PYTHON"
  echo
  echo "Create the repository environment first, for example:"
  echo "  python -m venv .venv"
  echo "  source .venv/bin/activate"
  echo "  python -m pip install -r requirements.txt"
  exit 1
fi

if [[ ! -f "$EVALUATOR" ]]; then
  echo "ERROR: Evaluation script not found:"
  echo "  $EVALUATOR"
  exit 1
fi

if [[ ! -f "$ENSEMBLER" ]]; then
  echo "ERROR: Ensemble script not found:"
  echo "  $ENSEMBLER"
  exit 1
fi

if [[ ! -d "$HAM_EVAL_DIR" ]]; then
  echo "ERROR: Standardized HAM10000 evaluation cohort not found:"
  echo "  $HAM_EVAL_DIR"
  echo
  echo "Set HAM_EVAL_DIR to the reconstructed 2,003-image"
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

# -----------------------------------------------------------------------------
# Canonical HAM10000 class layout
# -----------------------------------------------------------------------------

CANONICAL_CLASSES=(
  "akiec"
  "bcc"
  "bkl"
  "df"
  "mel"
  "nv"
  "vasc"
)

for class_name in "${CANONICAL_CLASSES[@]}"; do
  if [[ ! -d "$HAM_EVAL_DIR/$class_name" ]]; then
    echo "ERROR: Canonical class directory is missing:"
    echo "  $HAM_EVAL_DIR/$class_name"
    echo
    echo "Expected class directories:"
    printf '  %s\n' "${CANONICAL_CLASSES[@]}"
    exit 1
  fi
done

# -----------------------------------------------------------------------------
# Evaluation configuration
# -----------------------------------------------------------------------------
#
# Format:
#
#   key | model_name | checkpoint_pattern
#
# `key` controls output file names.
# `model_name` must be supported by src/models.py.
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

EXPECTED_MODELS=7
PREDICTION_CSVS=()

# -----------------------------------------------------------------------------
# Resolve exactly one checkpoint per model
# -----------------------------------------------------------------------------

find_checkpoint() {
  local pattern="$1"
  local matches=()
  local file

  while IFS= read -r -d '' file; do
    matches+=("$file")
  done < <(
    find "$CHECKPOINT_DIR" \
      -type f \
      -iname "$pattern" \
      -print0
  )

  if (( ${#matches[@]} == 0 )); then
    echo "ERROR: No checkpoint matched pattern:" >&2
    echo "  $pattern" >&2
    return 1
  fi

  if (( ${#matches[@]} > 1 )); then
    echo "ERROR: Multiple checkpoints matched pattern:" >&2
    echo "  $pattern" >&2
    echo >&2
    printf '  %s\n' "${matches[@]}" >&2
    echo >&2
    echo "Exactly one fixed checkpoint is required per model." >&2
    return 2
  fi

  printf '%s\n' "${matches[0]}"
}

# -----------------------------------------------------------------------------
# Evaluate individual models
# -----------------------------------------------------------------------------

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

  ckpt="$(find_checkpoint "$pattern")" || exit $?

  metrics="$OUT_DIR/${key}_metrics.json"
  predictions="$OUT_DIR/${key}_predictions.csv"

  echo "Checkpoint:"
  echo "  $ckpt"
  echo

  "$PYTHON" "$EVALUATOR" \
    --checkpoint "$ckpt" \
    --data_dir "$HAM_EVAL_DIR" \
    --model "$model_name" \
    --dataset ham10000 \
    --out "$metrics" \
    --save_csv "$predictions"

  if [[ ! -s "$metrics" ]]; then
    echo "ERROR: Metrics file was not produced:"
    echo "  $metrics"
    exit 1
  fi

  if [[ ! -s "$predictions" ]]; then
    echo "ERROR: Prediction file was not produced:"
    echo "  $predictions"
    exit 1
  fi

  PREDICTION_CSVS+=("$predictions")
done

# -----------------------------------------------------------------------------
# Verify ensemble inputs
# -----------------------------------------------------------------------------

if (( ${#PREDICTION_CSVS[@]} != EXPECTED_MODELS )); then
  echo
  echo "ERROR: Expected $EXPECTED_MODELS model prediction files,"
  echo "but found ${#PREDICTION_CSVS[@]}."
  exit 1
fi

# -----------------------------------------------------------------------------
# Equal-weight seven-model ensemble
# -----------------------------------------------------------------------------

echo
echo "============================================================"
echo "Equal-weight seven-model ensemble"
echo "============================================================"
echo

"$PYTHON" "$ENSEMBLER" \
  --csvs "${PREDICTION_CSVS[@]}" \
  --dataset ham10000 \
  --out "$OUT_DIR/ensemble_metrics.json"

if [[ ! -s "$OUT_DIR/ensemble_metrics.json" ]]; then
  echo "ERROR: Ensemble metrics file was not produced."
  exit 1
fi

if [[ ! -s "$OUT_DIR/ensemble_predictions.csv" ]]; then
  echo "ERROR: Ensemble prediction file was not produced."
  exit 1
fi

# -----------------------------------------------------------------------------
# Optional manuscript-level statistical analyses
# -----------------------------------------------------------------------------

if [[ -f "$ROOT/src/analyze_ham10000.py" ]]; then
  echo
  echo "============================================================"
  echo "HAM10000 statistical and calibration analysis"
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
  echo "  src/analyze_ham10000.py is not currently present."
  echo "  Individual inference and equal-weight ensemble evaluation"
  echo "  are complete."
  echo
  echo "  Additional manuscript analyses such as bootstrap confidence"
  echo "  intervals, McNemar tests, paired bootstrap comparisons, and"
  echo "  derived calibration/figure generation must therefore be"
  echo "  reproduced using the corresponding released analysis workflow."
fi

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------

echo
echo "============================================================"
echo "Evaluation complete"
echo "============================================================"
echo

echo "Individual model metrics:"
for config in "${MODEL_CONFIGS[@]}"; do
  IFS='|' read -r key _ _ <<< "$config"
  echo "  $OUT_DIR/${key}_metrics.json"
done

echo
echo "Individual model predictions:"
for csv in "${PREDICTION_CSVS[@]}"; do
  echo "  $csv"
done

echo
echo "Ensemble metrics:"
echo "  $OUT_DIR/ensemble_metrics.json"

echo
echo "Ensemble predictions:"
echo "  $OUT_DIR/ensemble_predictions.csv"

echo
echo "Important:"
echo "  The ensemble contains exactly seven fixed model outputs"
echo "  combined using equal probability weighting (1/7 each)."
echo
echo "  Canonical class ordering and any checkpoint-specific output"
echo "  permutation must remain consistent with the archived models."
echo
echo "  If an archived checkpoint requires preprocessing different"
echo "  from the defaults in src/evaluate.py, reproduce that exact"
echo "  preprocessing before treating the resulting metrics as"
echo "  manuscript-equivalent."
