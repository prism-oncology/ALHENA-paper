#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------------------------
# Sync a training_dataset_id from the local ROOT_PATH to Jean Zay's ROOT_PATH_JeanZay
# while preserving the exact relative file/folder organization under ROOT_PATH.
#
# Usage:
#   chmod +x _3_send_data_to_JeanZay.sh
#   ./_3_send_data_to_JeanZay.sh <dataset_id> <training_dataset_id> <ct_cat_id>
#
# Example:
#   ./_3_send_data_to_JeanZay.sh ds_ct1_20x_224_o112 ds_ct1_20x_224_o112_1 ct_1
#
# Notes:
# - Skips missing files/folders and prints warnings (no symlinks created)
# ------------------------------------------------------------------------------

usage() {
  cat <<'EOF'
Usage:
  ./_3_send_data_to_JeanZay.sh <dataset_id> <training_dataset_id> <ct_cat_id>

Positional arguments:
  dataset_id            Dataset identifier
  training_dataset_id   Training dataset identifier
  ct_cat_id             Cell type category identifier

Optional environment variables:
  JZ_HOST               SSH host alias for Jean Zay (default: jeanzay)
  REPO_ROOT             Explicit project root containing src/utils/constants.py
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -ne 3 ]]; then
  echo "ERROR: Expected 3 arguments, got $#." >&2
  usage >&2
  exit 1
fi

DATASET_ID="$1"
TRAINING_DATASET_ID="$2"
CT_CAT_ID="$3"
JZ_HOST="${JZ_HOST:-jeanzay}"

# --- Resolve project root (contains src/utils/constants.py) --------------------

find_repo_root() {
  if [[ -n "${REPO_ROOT:-}" ]]; then
    if [[ -f "${REPO_ROOT}/src/utils/constants.py" ]]; then
      printf '%s\n' "$REPO_ROOT"
      return 0
    else
      echo "ERROR: REPO_ROOT is set but src/utils/constants.py not found under it: $REPO_ROOT" >&2
      return 1
    fi
  fi

  local dir="$PWD"
  while [[ "$dir" != "/" ]]; do
    if [[ -f "$dir/src/utils/constants.py" ]]; then
      printf '%s\n' "$dir"
      return 0
    fi
    dir="$(dirname "$dir")"
  done

  echo "ERROR: Could not find project root (src/utils/constants.py) from current directory upward." >&2
  echo "       Run the script from inside the project repo, or set REPO_ROOT=/path/to/repo" >&2
  return 1
}

REPO_ROOT_RESOLVED="$(find_repo_root)"

# --- Read ROOT_PATH and ROOT_PATH_JeanZay from Python constants ----------------

get_root_paths() {
  local repo_root="$1"
  REPO_ROOT_FOR_PY="$repo_root" python3 - <<'PY'
import os, sys
from pathlib import Path

repo_root = Path(os.environ["REPO_ROOT_FOR_PY"]).resolve()
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.utils.constants import get_root_paths_dict

d = get_root_paths_dict()
print(d["local"])
print(d["jeanzay"])
PY
}

ROOT_PATH=""
ROOT_PATH_JZ=""
i=0
while IFS= read -r line; do
  if [[ $i -eq 0 ]]; then
    ROOT_PATH="$line"
  elif [[ $i -eq 1 ]]; then
    ROOT_PATH_JZ="$line"
    break
  fi
  i=$((i+1))
done < <(get_root_paths "$REPO_ROOT_RESOLVED")

if [[ -z "$ROOT_PATH" || -z "$ROOT_PATH_JZ" ]]; then
  echo "ERROR: Failed to read ROOT_PATH / ROOT_PATH_JeanZay from src/utils/constants.py" >&2
  exit 1
fi

# --- Pre-flight checks ---------------------------------------------------------

command -v rsync >/dev/null 2>&1 || { echo "ERROR: rsync is not installed." >&2; exit 1; }
command -v ssh   >/dev/null 2>&1 || { echo "ERROR: ssh is not installed." >&2; exit 1; }
command -v python3 >/dev/null 2>&1 || { echo "ERROR: python3 is not installed." >&2; exit 1; }

if [[ ! -d "$ROOT_PATH" ]]; then
  echo "ERROR: Local ROOT_PATH does not exist: $ROOT_PATH" >&2
  exit 1
fi

# Check remote connectivity. Fatal, so a connection problem is not reported later as a local one.
if ! ssh -o BatchMode=yes -o ConnectTimeout=8 "$JZ_HOST" "echo ok" >/dev/null 2>&1; then
  echo "ERROR: Cannot connect to '$JZ_HOST' via SSH. Check SSH config / VPN / alias." >&2
  exit 1
fi

echo "Local ROOT_PATH       : $ROOT_PATH"
echo "Jean Zay ROOT_PATH    : $ROOT_PATH_JZ"
echo "Jean Zay host         : $JZ_HOST"
echo "dataset_id            : $DATASET_ID"
echo "training_dataset_id   : $TRAINING_DATASET_ID"
echo "ct_cat_id             : $CT_CAT_ID"
echo

# --- Build list of required/optional relative paths ----------------------------

declare -a REQUIRED_PATHS=()
declare -a OPTIONAL_PATHS=()

# STHELAR_HistoST
REQUIRED_PATHS+=("STHELAR_HistoST/adata_graph")
REQUIRED_PATHS+=("STHELAR_HistoST/cID2cT/${CT_CAT_ID}")

# train/datasets/{dataset_id}
REQUIRED_PATHS+=("train/datasets/${DATASET_ID}/images.zip")
REQUIRED_PATHS+=("train/datasets/${DATASET_ID}/masks_cell_ids_nuclei.zip")
REQUIRED_PATHS+=("train/datasets/${DATASET_ID}/types.csv")
REQUIRED_PATHS+=("train/datasets/${DATASET_ID}/nucleus_centroids")
REQUIRED_PATHS+=("train/datasets/${DATASET_ID}/patch_bboxes.parquet")

# train/training_datasets/{dataset_id}/{training_dataset_id}
TD_BASE="train/training_datasets/${DATASET_ID}/${TRAINING_DATASET_ID}"
REQUIRED_PATHS+=("${TD_BASE}/cell_count_train.csv")
REQUIRED_PATHS+=("${TD_BASE}/cell_count_valid.csv")
REQUIRED_PATHS+=("${TD_BASE}/cell_count_test.csv")
REQUIRED_PATHS+=("${TD_BASE}/slide_ids.txt")
REQUIRED_PATHS+=("${TD_BASE}/cell_ids_sets.npz")
REQUIRED_PATHS+=("${TD_BASE}/genes_training_${TRAINING_DATASET_ID}.json")

# train/pretrained_foundation_models
PFM_BASE="train/pretrained_foundation_models"
REQUIRED_PATHS+=("${PFM_BASE}/virchow2")
REQUIRED_PATHS+=("${PFM_BASE}/uni2h")
REQUIRED_PATHS+=("${PFM_BASE}/hoptimus0")
REQUIRED_PATHS+=("${PFM_BASE}/hoptimus1")
REQUIRED_PATHS+=("${PFM_BASE}/scGPT_human")
REQUIRED_PATHS+=("${PFM_BASE}/scGPT_pancancer")
REQUIRED_PATHS+=("${PFM_BASE}/scConcept_Corpus-30M")               # scConcept v1 model
REQUIRED_PATHS+=("${PFM_BASE}/scConcept_corpus40M-model30M")       # scConcept v2 model

# Optional numbered cell_count files
shopt -s nullglob
for f in "$ROOT_PATH/$TD_BASE"/cell_count_train_*.csv; do
  OPTIONAL_PATHS+=("${f#"$ROOT_PATH/"}")
done
for f in "$ROOT_PATH/$TD_BASE"/cell_count_valid_*.csv; do
  OPTIONAL_PATHS+=("${f#"$ROOT_PATH/"}")
done
shopt -u nullglob

# --- rsync helper --------------------------------------------------------------

WARN_COUNT=0
SYNC_COUNT=0
SKIP_COUNT=0

warn_missing() {
  local rel="$1"
  echo "WARNING: Missing, skipping: $rel" >&2
  ((WARN_COUNT+=1))
  ((SKIP_COUNT+=1))
}

sync_one() {
  local rel="$1"
  local src="$ROOT_PATH/$rel"

  if [[ ! -e "$src" ]]; then
    warn_missing "$rel"
    return 0
  fi

  echo ">>> Syncing: $rel"
  # --relative preserves the path hierarchy under the source working directory
  # Change into ROOT_PATH and sync ./<rel> to ROOT_PATH_JeanZay on the remote side
  (
    cd "$ROOT_PATH"
    # Exclude macOS metadata: the exFAT ROOT_PATH emits AppleDouble '._*' sidecars
    # (and .DS_Store). Downstream *.csv globs match them (e.g. the scConcept panel
    # loader) and crash on their binary content, so never ship them to Jean Zay.
    rsync --partial --progress -avh --relative \
      --exclude='._*' --exclude='.DS_Store' \
      "./$rel" "${JZ_HOST}:${ROOT_PATH_JZ}/"
  )
  ((SYNC_COUNT+=1))
}

# --- Run syncs ----------------------------------------------------------------

echo "Starting sync..."
echo

for rel in "${REQUIRED_PATHS[@]}"; do
  sync_one "$rel"
done

for rel in "${OPTIONAL_PATHS[@]}"; do
  sync_one "$rel"
done

echo
echo "Done."
echo "Synced items : $SYNC_COUNT"
echo "Skipped items: $SKIP_COUNT"
echo "Warnings     : $WARN_COUNT"