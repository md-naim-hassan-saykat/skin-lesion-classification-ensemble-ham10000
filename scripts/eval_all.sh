#!/usr/bin/env bash
set -euo pipefail

EXTRACT_DIR="$HOME/Desktop/HAM10000_raw"
SPLIT_ROOT="$HOME/Desktop/HAM10000_split"
REPO="$HOME/Desktop/skin-lesion-classification-ensemble-ham10000"
OUT_DIR="$REPO/outputs/from_zip_eval"
VENV_PY="$REPO/.venv/bin/python"

mkdir -p "$OUT_DIR"

# Only evaluate models that loaded & performed sensibly:
find "$EXTRACT_DIR/skin_lesion_models_outputs/models" -type f -name '*.pth' \
| egrep '/(densenet121|efficientnet[_-]?b3|convnext[_-]?tiny).*\.pth$' \
| while read -r ckpt; do
  base="$(basename "$ckpt")"; name="${base%.*}"
  echo "==> Evaluating $name"
  "$VENV_PY" "$REPO/src/evaluate.py" \
    --checkpoint "$ckpt" \
    --data_dir "$SPLIT_ROOT/val" \
    --out "$OUT_DIR/${name}_metrics.json" \
    --save_csv "$OUT_DIR/${name}_val_preds.csv"
done

"$VENV_PY" "$REPO/src/ensemble.py" \
  --csvs "$OUT_DIR"/*_val_preds.csv \
  --out "$OUT_DIR/ensemble_metrics.json"

echo "=== Individual metrics JSONs ==="
ls -1 "$OUT_DIR"/*_metrics.json 2>/dev/null || echo "No metrics yet."
echo "=== Ensemble metrics ==="
[ -f "$OUT_DIR/ensemble_metrics.json" ] && cat "$OUT_DIR/ensemble_metrics.json" || echo "No ensemble yet."
