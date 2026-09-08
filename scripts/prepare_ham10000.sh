#!/usr/bin/env bash
#
# Prepare the HAM10000 dataset for this repository.
#
# This script does NOT download HAM10000 from the internet.
# It searches for locally downloaded/extracted HAM10000 image archives
# in common locations, extracts them when necessary, and consolidates
# the images into:
#
#   data/HAM10000/images/
#
# The complete HAM10000 dataset contains 10,015 dermoscopic images.
#
# IMPORTANT:
# The full HAM10000 dataset prepared here is not the same as the
# standardized 2,003-image retrospective evaluation cohort used in
# the final manuscript. Evaluation-cohort construction/harmonization
# is handled separately.
#
# Expected source files:
#   HAM10000_images_part_1.zip
#   HAM10000_images_part_2.zip
#
# Optional metadata:
#   HAM10000_metadata.csv
#
# Usage:
#   bash scripts/download_ham10000.sh
#
# Requirements:
#   bash, unzip, find
#

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DATA_ROOT="$ROOT/data/HAM10000"
IMAGE_DEST="$DATA_ROOT/images"

mkdir -p "$IMAGE_DEST"

shopt -s nullglob

echo "Repository root: $ROOT"
echo "HAM10000 destination: $IMAGE_DEST"
echo

# ---------------------------------------------------------------------
# Candidate locations
# ---------------------------------------------------------------------

SEARCH_ROOTS=(
  "$HOME/Downloads"
  "$HOME/Desktop"
)

PART1_NAME="HAM10000_images_part_1"
PART2_NAME="HAM10000_images_part_2"

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

find_or_extract_part() {
  local part_name="$1"

  local root
  local extracted
  local zip_path
  local extract_dir

  for root in "${SEARCH_ROOTS[@]}"; do

    extracted="$root/$part_name"

    if [[ -d "$extracted" ]]; then
      echo "Found extracted folder: $extracted" >&2
      printf '%s\n' "$extracted"
      return 0
    fi

    zip_path="$root/${part_name}.zip"

    if [[ -f "$zip_path" ]]; then
      extract_dir="$root/$part_name"

      echo "Found archive: $zip_path" >&2
      echo "Extracting to: $extract_dir" >&2

      mkdir -p "$extract_dir"
      unzip -n "$zip_path" -d "$extract_dir" >/dev/null

      printf '%s\n' "$extract_dir"
      return 0
    fi

  done

  return 1
}

copy_images() {
  local src="$1"

  if [[ -z "$src" || ! -d "$src" ]]; then
    return 0
  fi

  echo "Collecting images from:"
  echo "  $src"

  local copied=0
  local img

  while IFS= read -r -d '' img; do
    cp -n "$img" "$IMAGE_DEST/"
    copied=$((copied + 1))
  done < <(
    find "$src" -type f \
      \( -iname '*.jpg' -o -iname '*.jpeg' \) \
      -print0
  )

  echo "Processed $copied image files."
  echo
}

copy_metadata_if_available() {
  local root
  local metadata

  for root in "${SEARCH_ROOTS[@]}"; do
    metadata="$root/HAM10000_metadata.csv"

    if [[ -f "$metadata" ]]; then
      cp -n "$metadata" "$DATA_ROOT/HAM10000_metadata.csv"
      echo "Metadata found and copied:"
      echo "  $metadata"
      return 0
    fi
  done

  echo "HAM10000_metadata.csv was not found."
  echo "This is not required for image counting, but may be required"
  echo "for training, label reconstruction, or dataset analysis."
}

# ---------------------------------------------------------------------
# Locate / extract dataset parts
# ---------------------------------------------------------------------

PART1_DIR="$(find_or_extract_part "$PART1_NAME" || true)"
PART2_DIR="$(find_or_extract_part "$PART2_NAME" || true)"

if [[ -z "$PART1_DIR" && -z "$PART2_DIR" ]]; then
  echo
  echo "ERROR: HAM10000 image archives or extracted folders were not found."
  echo
  echo "Place the following in ~/Downloads or ~/Desktop:"
  echo
  echo "  HAM10000_images_part_1.zip"
  echo "  HAM10000_images_part_2.zip"
  echo
  echo "or the corresponding extracted directories."
  echo
  echo "Obtain HAM10000 from its official/public distribution before"
  echo "running this script."
  exit 1
fi

# ---------------------------------------------------------------------
# Consolidate images
# ---------------------------------------------------------------------

copy_images "$PART1_DIR"
copy_images "$PART2_DIR"

copy_metadata_if_available

# ---------------------------------------------------------------------
# Verify dataset
# ---------------------------------------------------------------------

EXPECTED=10015

COUNT="$(
  find "$IMAGE_DEST" \
    -maxdepth 1 \
    -type f \
    \( -iname '*.jpg' -o -iname '*.jpeg' \) \
    | wc -l \
    | awk '{print $1}'
)"

echo
echo "HAM10000 image verification"
echo "---------------------------"
echo "Found:    $COUNT"
echo "Expected: $EXPECTED"

if [[ "$COUNT" -eq "$EXPECTED" ]]; then
  echo
  echo "Dataset preparation completed successfully."
  echo "All $EXPECTED HAM10000 images are present."
else
  echo
  echo "WARNING: Dataset is incomplete or contains an unexpected number of images."
  echo
  echo "Expected exactly $EXPECTED images but found $COUNT."
  echo "Check that both HAM10000 image archives were obtained and extracted correctly."
  exit 2
fi

echo
echo "Prepared dataset:"
echo "  $IMAGE_DEST"

if [[ -f "$DATA_ROOT/HAM10000_metadata.csv" ]]; then
  echo "Metadata:"
  echo "  $DATA_ROOT/HAM10000_metadata.csv"
fi

echo
echo "Note:"
echo "  This directory contains the complete raw HAM10000 dataset."
echo "  It is not, by itself, the standardized 2,003-image evaluation"
echo "  cohort reported in the final manuscript."
