#!/usr/bin/env bash
# ------------------------------------------------------------------------------
#
# Usage:
#   ./_1_make_symb_links.sh <dataset_id> <training_dataset_id> <ct_cat_id>
#
# Example:
#   chmod +x _1_make_symb_links.sh
#   ./_1_make_symb_links.sh ds_ct1_20x_224_o112 ds_ct1_20x_224_o112_1 ct_1
#
# What it does:
#   Create needed symbolic links inside training_dataset_id folder.
#
# Notes:
#   - Detects Jean Zay vs local.
#   - Missing sources => WARNING (no hard error).
#   - Existing destination link/file/dir => removed then recreated.
#
# ------------------------------------------------------------------------------

set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage:
  _1_make_symb_links.sh <dataset_id> <training_dataset_id> <ct_cat_id>
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -ne 3 ]]; then
  usage
  exit 2
fi

DATASET_ID="$1"
TRAINING_DATASET_ID="$2"
CT_CAT_ID="$3"

# --- Root selection via Python -----------------------

get_root_path() {
  python3 - <<'PY'
import sys
from pathlib import Path

# Add the project root to sys.path so that 'src' can be imported
project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.constants import require_root_path

print(require_root_path())
PY
}

ROOT="$(get_root_path)"

if [[ -z "$ROOT" ]]; then
  echo "[ERROR] Could not resolve ROOT_PATH from Python." >&2
  exit 1
fi

echo "[INFO] ROOT resolved from Python: $ROOT"

# --- Helpers -----------------------------------------------------------------

log()   { printf '[INFO] %s\n' "$*"; }
warn()  { printf '[WARN] %s\n' "$*" >&2; }

ensure_dir() {
  local d="$1"
  mkdir -p "$d"
}

remove_if_exists() {
  local dst="$1"
  # Remove existing symlink, file, or directory at destination.
  if [[ -L "$dst" || -e "$dst" ]]; then
    rm -rf -- "$dst"
  fi
}

link_one() {
  local src="$1"
  local dst="$2"

  if [[ ! -e "$src" && ! -L "$src" ]]; then
    warn "Source missing, skipping: $src"
    return 0
  fi

  remove_if_exists "$dst"
  ln -s -- "$src" "$dst"
  log "Linked: $dst -> $src"
}

# --- Paths -------------------------------------------------------------------

SRC_DATASET_DIR="$ROOT/train/datasets/$DATASET_ID"
DST_TRAINING_DIR="$ROOT/train/training_datasets/$DATASET_ID/$TRAINING_DATASET_ID"

ensure_dir "$DST_TRAINING_DIR"

# --- 1) Dataset artifacts symlinks -------------------------------------------

# Required (link if exists; missing => warn)
ARTIFACTS_REQUIRED=(
  "images.zip"
  "types.csv"
  "masks_cell_ids_nuclei.zip"
  "nucleus_centroids"
  "patch_bboxes.parquet"
)

# Optional
ARTIFACT_OPTIONAL="labels.zip"

log "ROOT selected: $ROOT"
log "Source dataset dir:      $SRC_DATASET_DIR"
log "Destination training dir: $DST_TRAINING_DIR"

for item in "${ARTIFACTS_REQUIRED[@]}"; do
  link_one "$SRC_DATASET_DIR/$item" "$DST_TRAINING_DIR/$item"
done

# Optional labels.zip (only link if exists, otherwise no warning)
if [[ -e "$SRC_DATASET_DIR/$ARTIFACT_OPTIONAL" || -L "$SRC_DATASET_DIR/$ARTIFACT_OPTIONAL" ]]; then
  link_one "$SRC_DATASET_DIR/$ARTIFACT_OPTIONAL" "$DST_TRAINING_DIR/$ARTIFACT_OPTIONAL"
else
  log "Optional not present, skipping: $SRC_DATASET_DIR/$ARTIFACT_OPTIONAL"
fi

# --- 2) cID2cT link (rename ct_cat_id -> cID2cT) ------------------------------

SRC_CID2CT="$ROOT/STHELAR_HistoST/cID2cT/$CT_CAT_ID"
DST_CID2CT="$DST_TRAINING_DIR/cID2cT"
link_one "$SRC_CID2CT" "$DST_CID2CT"

# --- 3) adata_graph link (rename adata_graph -> adata) ------------------------

SRC_ADATA="$ROOT/STHELAR_HistoST/adata_graph"
DST_ADATA="$DST_TRAINING_DIR/adatas"
link_one "$SRC_ADATA" "$DST_ADATA"

log "Done."