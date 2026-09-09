#!/usr/bin/env bash
#
# Prepare the complete HAM10000 dataset for this repository.
#
# This script does NOT download HAM10000 from the internet.
# It locates locally downloaded or extracted HAM10000 image archives,
# extracts them when necessary, and consolidates the images into:
#
#   data/HAM10000/images/
#
# The complete HAM10000 dataset contains 10,015 dermoscopic images.
#
# IMPORTANT:
# The full HAM10000 dataset prepared by this script is not the same as
# the standardized 2,003-image retrospective evaluation cohort used in
# the final manuscript. Construction or reconstruction of that cohort
# is handled separately.
#
# Expected source archives:
#
#   HAM10000_images_part_1.zip
#   HAM10000_images_part_2.zip
#
# Optional metadata:
#
#   HAM10000_metadata.csv
#
# Usage:
#
#   bash scripts/prepare_ham10000.sh
#
# Optional source-directory override:
#
#   HAM_SOURCE_DIR=/path/to/downloaded/HAM10000 \
#   bash scripts/prepare_ham10000.sh
#
# Requirements:
#
#   bash
#   find
#   unzip
#

set -euo pipefail

# -----------------------------------------------------------------------------
# Repository paths
# -----------------------------------------------------------------------------

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DATA_ROOT="$ROOT/data/HAM10000"
IMAGE_DEST="$DATA_ROOT/images"

mkdir -p "$IMAGE_DEST"

# -----------------------------------------------------------------------------
# Candidate source locations
# -----------------------------------------------------------------------------

SEARCH_ROOTS=()

if [[ -n "${HAM_SOURCE_DIR:-}" ]]; then
  SEARCH_ROOTS+=("$HAM_SOURCE_DIR")
fi

SEARCH_ROOTS+=(
  "$HOME/Downloads"
  "$HOME/Desktop"
)

PART1_NAME="HAM10000_images_part_1"
PART2_NAME="HAM10000_images_part_2"

echo "Repository root:"
echo "  $ROOT"
echo
echo "HAM10000 destination:"
echo "  $IMAGE_DEST"
echo
echo "Candidate source locations:"
printf '  %s\n' "${SEARCH_ROOTS[@]}"
echo

# -----------------------------------------------------------------------------
# Dependency checks
# -----------------------------------------------------------------------------

for command_name in find unzip; do
  if ! command -v "$command_name" >/dev/null 2>&1; then
    echo "ERROR: Required command is not available:"
    echo "  $command_name"
    exit 1
  fi
done

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

find_or_extract_part() {
  local part_name="$1"
  local root
  local extracted
  local zip_path
  local extract_dir

  for root in "${SEARCH_ROOTS[@]}"; do
    [[ -d "$root" ]] || continue

    extracted="$root/$part_name"

    if [[ -d "$extracted" ]]; then
      echo "Found extracted directory:" >&2
      echo "  $extracted" >&2
      printf '%s\n' "$extracted"
      return 0
    fi

    zip_path="$root/${part_name}.zip"

    if [[ -f "$zip_path" ]]; then
      extract_dir="$root/$part_name"

      echo "Found archive:" >&2
      echo "  $zip_path" >&2
      echo "Extracting to:" >&2
      echo "  $extract_dir" >&2

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
  local processed=0
  local img
  local destination

  if [[ -z "$src" || ! -d "$src" ]]; then
    return 0
  fi

  echo "Collecting images from:"
  echo "  $src"

  while IFS= read -r -d '' img; do
    destination="$IMAGE_DEST/$(basename "$img")"

    if [[ ! -e "$destination" ]]; then
      cp "$img" "$destination"
    fi

    processed=$((processed + 1))
  done < <(
    find "$src" \
      -type f \
      \( -iname '*.jpg' -o -iname '*.jpeg' \) \
      -print0
  )

  echo "Processed $processed image files."
  echo
}

copy_metadata_if_available() {
  local root
  local metadata

  for root in "${SEARCH_ROOTS[@]}"; do
    [[ -d "$root" ]] || continue

    metadata="$root/HAM10000_metadata.csv"

    if [[ -f "$metadata" ]]; then
      if [[ ! -f "$DATA_ROOT/HAM10000_metadata.csv" ]]; then
        cp "$metadata" "$DATA_ROOT/HAM10000_metadata.csv"
      fi

      echo "Metadata available:"
      echo "  $metadata"
      return 0
    fi
  done

  echo "HAM10000_metadata.csv was not found."
  echo "Image preparation can still be verified, but metadata is"
  echo "required for workflows that reconstruct labels or metadata."
}

# -----------------------------------------------------------------------------
# Locate or extract dataset parts
# -----------------------------------------------------------------------------

PART1_DIR="$(find_or_extract_part "$PART1_NAME" || true)"
PART2_DIR="$(find_or_extract_part "$PART2_NAME" || true)"

if [[ -z "$PART1_DIR" || -z "$PART2_DIR" ]]; then
  echo
  echo "ERROR: Both HAM10000 image parts are required."
  echo

  if [[ -z "$PART1_DIR" ]]; then
    echo "Missing:"
    echo "  ${PART1_NAME}.zip"
    echo "  or extracted directory: $PART1_NAME"
    echo
  fi

  if [[ -z "$PART2_DIR" ]]; then
    echo "Missing:"
    echo "  ${PART2_NAME}.zip"
    echo "  or extracted directory: $PART2_NAME"
    echo
  fi

  echo "Place the dataset files in one of the searched locations or set:"
  echo
  echo "  HAM_SOURCE_DIR=/path/to/HAM10000"
  echo
  echo "Obtain HAM10000 from its official/public distribution before"
  echo "running this script."
  exit 1
fi

# -----------------------------------------------------------------------------
# Consolidate images and metadata
# -----------------------------------------------------------------------------

copy_images "$PART1_DIR"
copy_images "$PART2_DIR"
copy_metadata_if_available

# -----------------------------------------------------------------------------
# Verify dataset
# -----------------------------------------------------------------------------

EXPECTED_IMAGES=10015

COUNT="$(
  find "$IMAGE_DEST" \
    -maxdepth 1 \
    -type f \
    \( -iname '*.jpg' -o -iname '*.jpeg' \) \
    -print \
    | wc -l \
    | awk '{print $1}'
)"

echo
echo "============================================================"
echo "HAM10000 image verification"
echo "============================================================"
echo "Found:"
echo "  $COUNT"
echo
echo "Expected:"
echo "  $EXPECTED_IMAGES"

if [[ "$COUNT" -ne "$EXPECTED_IMAGES" ]]; then
  echo
  echo "ERROR: HAM10000 preparation is incomplete or unexpected."
  echo "Expected exactly $EXPECTED_IMAGES images but found $COUNT."
  echo
  echo "Check that both HAM10000 image archives were obtained and"
  echo "extracted correctly."
  exit 2
fi

echo
echo "Dataset preparation completed successfully."
echo "All $EXPECTED_IMAGES HAM10000 images are present."

echo
echo "Prepared image directory:"
echo "  $IMAGE_DEST"

if [[ -f "$DATA_ROOT/HAM10000_metadata.csv" ]]; then
  echo
  echo "Metadata:"
  echo "  $DATA_ROOT/HAM10000_metadata.csv"
fi

echo
echo "Important:"
echo "  This directory contains the complete raw HAM10000 image set."
echo "  It is not, by itself, the standardized 2,003-image evaluation"
echo "  cohort used for the final manuscript analysis."
