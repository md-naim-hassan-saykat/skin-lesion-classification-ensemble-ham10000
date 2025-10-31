#!/usr/bin/env bash
# Download / collect HAM10000 images into data/HAM10000
# - Copies from common local locations (Downloads/Desktop)
# - Unzips parts if needed
# - Verifies image count at the end

set -euo pipefail

# repo root (one level up from this script)
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$ROOT/data/HAM10000"
mkdir -p "$DEST"

# Bash-only nicety; if not available, ignore
if (set -o | grep -q 'noglob'); then :; fi
shopt -s nullglob || true

echo "Target folder: $DEST"

# Candidate folders (already extracted)
PART1_CANDIDATES=(
  "$HOME/Downloads/HAM10000_images_part_1"
  "$HOME/Desktop/HAM10000_images_part_1"
)
PART2_CANDIDATES=(
  "$HOME/Downloads/HAM10000_images_part_2"
  "$HOME/Desktop/HAM10000_images_part_2"
)

# Candidate zip files (to unzip if folders don’t exist)
ZIP1="$HOME/Downloads/HAM10000_images_part_1.zip"
ZIP2="$HOME/Downloads/HAM10000_images_part_2.zip"

ensure_part_folder () {
  local zip="$1"
  shift
  local -a candidates=("$@")

  for d in "${candidates[@]}"; do
    if [[ -d "$d" ]]; then
      echo "Found extracted folder: $d"
      echo "$d"
      return 0
    fi
  done

  if [[ -f "$zip" ]]; then
    echo "Found zip: $zip"
    local outdir="${zip%.zip}"
    echo "Unzipping to: $outdir"
    mkdir -p "$outdir"
    unzip -n "$zip" -d "$outdir" >/dev/null
    echo "$outdir"
    return 0
  fi

  echo ""
  return 1
}

PART1_DIR="$(ensure_part_folder "$ZIP1" "${PART1_CANDIDATES[@]}")" || true
PART2_DIR="$(ensure_part_folder "$ZIP2" "${PART2_CANDIDATES[@]}")" || true

if [[ -z "$PART1_DIR" && -z "$PART2_DIR" ]]; then
  echo "Couldn’t find HAM10000 parts."
  echo "   Put the extracted folders or ZIPs in ~/Downloads or ~/Desktop:"
  echo "     - HAM10000_images_part_1  (or HAM10000_images_part_1.zip)"
  echo "     - HAM10000_images_part_2  (or HAM10000_images_part_2.zip)"
  exit 1
fi

copy_part () {
  local src="$1"
  if [[ -z "$src" || ! -d "$src" ]]; then
    return 0
  fi
  echo "Copying from $src -> $DEST"
  cp -rf "$src"/* "$DEST"/ 2>/dev/null || true
}

copy_part "$PART1_DIR"
copy_part "$PART2_DIR"

# Verify count
COUNT="$(find "$DEST" -type f -iname '*.jpg' | wc -l | awk '{print $1}')"
echo "Image files in $DEST: $COUNT"

EXPECTED=10015
if [[ "$COUNT" -eq "$EXPECTED" ]]; then
  echo "Looks perfect: found exactly $EXPECTED images."
else
  echo "Found $COUNT images (expected $EXPECTED)."
fi

echo "Done."
