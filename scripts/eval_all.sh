#!/usr/bin/env bash
set -euo pipefail

EXTRACT_DIR="$HOME/Desktop/HAM10000_raw"
SPLIT_ROOT="$HOME/Desktop/HAM10000_split"   # must contain val/
REPO="$HOME/Desktop/skin-lesion-classification-ensemble-ham10000"
OUT_DIR="$REPO/outputs/from_zip_eval"
VENV_PY="$REPO/.venv/bin/python"

# Sanity checks
[[ -x "$VENV_PY" ]] || { echo "Missing: $VENV_PY"; exit 1; }
[[ -d "$SPLIT_ROOT/val" ]] || { echo "Missing val split at $SPLIT_ROOT/val"; exit 1; }
[[ -d "$EXTRACT_DIR/skin_lesion_models_outputs/models" ]] || { echo "Missing models dir under $EXTRACT_DIR"; exit 1; }

mkdir -p "$OUT_DIR"

# Evaluate selected checkpoints (handles MANY files + spaces)
while IFS= read -r -d '' ckpt; do
  case "$ckpt" in
    *densenet121*.pth|*efficientnet*b3*.pth|*convnext*tiny*.pth)
      base="$(basename "$ckpt")"; name="${base%.*}"
      echo "==> Evaluating $name"
      "$VENV_PY" "$REPO/src/evaluate.py" \
        --checkpoint "$ckpt" \
        --data_dir "$SPLIT_ROOT/val" \
        --out "$OUT_DIR/${name}_metrics.json" \
        --save_csv "$OUT_DIR/${name}_val_preds.csv"
      ;;
  esac
done < <(find "$EXTRACT_DIR/skin_lesion_models_outputs/models" -type f -name '*.pth' -print0)

# Collect CSVs without mapfile
csvs=()
while IFS= read -r -d '' f; do csvs+=("$f"); done \
  < <(find "$OUT_DIR" -maxdepth 1 -name '*_val_preds.csv' -print0)

if ((${#csvs[@]}==0)); then
  echo "No prediction CSVs found in $OUT_DIR; skipping ensemble."
  exit 0
fi

"$VENV_PY" "$REPO/src/ensemble.py" \
  --csvs "${csvs[@]}" \
  --out "$OUT_DIR/ensemble_metrics.json"

echo "=== Individual metrics JSONs ==="
ls -1 "$OUT_DIR"/*_metrics.json 2>/dev/null || echo "No metrics yet."
echo "=== Ensemble metrics ==="
[ -f "$OUT_DIR/ensemble_metrics.json" ] && cat "$OUT_DIR/ensemble_metrics.json" || echo "No ensemble yet."
