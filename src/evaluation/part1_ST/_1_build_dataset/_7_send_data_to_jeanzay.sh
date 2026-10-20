#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------------------------
# Sync only the evaluation data needed for one training_dataset_id,
# from the local ROOT_PATH to the Jean Zay ROOT_PATH_JeanZay.
#
# Transferred paths:
#   eval/eval_datasets/st_eval_data/hest/patches
#   eval/eval_datasets/st_eval_data/hest/st
#   eval/eval_datasets/st_eval_data/eval_st_<training_dataset_id>/cell_token_membership_*
#     (excluding cell_token_membership_missing_cells_*)
#   eval/eval_datasets/st_eval_data/eval_st_<training_dataset_id>/metadata_eval_st_<training_dataset_id>.csv
#   eval/eval_datasets/st_eval_data/eval_st_<training_dataset_id>/patches
#   eval/eval_datasets/st_eval_data/eval_st_<training_dataset_id>/st
#
# Symlinks are neither transferred nor followed: only regular files and real directories are sent.
#
# Usage:
#   chmod +x _7_send_data_to_JeanZay.sh
#   ./_7_send_data_to_JeanZay.sh <training_dataset_id>
#
# Example:
#   ./_7_send_data_to_JeanZay.sh ds_ct1_20x_224_o112_1
#
# Optional environment variables:
#   JZ_HOST   SSH host alias for Jean Zay (default: jeanzay)
#   REPO_ROOT Explicit project root containing src/utils/constants.py
#   DRY_RUN   Set to 1 for a dry run
# ------------------------------------------------------------------------------

usage() {
  cat <<'EOF'
Usage:
  ./_7_send_data_to_JeanZay.sh <training_dataset_id>

Positional arguments:
  training_dataset_id   Training dataset identifier

Optional environment variables:
  JZ_HOST               SSH host alias for Jean Zay (default: jeanzay)
  REPO_ROOT             Explicit project root containing src/utils/constants.py
  DRY_RUN               Set to 1 to preview without copying
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -ne 1 ]]; then
  echo "ERROR: Expected 1 argument, got $#." >&2
  usage >&2
  exit 1
fi

TRAINING_DATASET_ID="$1"
EVAL_ST_ID="eval_st_${TRAINING_DATASET_ID}"
JZ_HOST="${JZ_HOST:-jeanzay}"
DRY_RUN="${DRY_RUN:-0}"

# --- Resolve project root -----------------------------------------------------

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

# --- Read ROOT_PATH and ROOT_PATH_JeanZay from Python constants ---------------

get_root_paths() {
  local repo_root="$1"
  REPO_ROOT_FOR_PY="$repo_root" python3 - <<'PY'
import os
import sys
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

# --- Pre-flight checks --------------------------------------------------------

command -v rsync   >/dev/null 2>&1 || { echo "ERROR: rsync is not installed." >&2; exit 1; }
command -v ssh     >/dev/null 2>&1 || { echo "ERROR: ssh is not installed." >&2; exit 1; }
command -v python3 >/dev/null 2>&1 || { echo "ERROR: python3 is not installed." >&2; exit 1; }
command -v find    >/dev/null 2>&1 || { echo "ERROR: find is not installed." >&2; exit 1; }

if [[ ! -d "$ROOT_PATH" ]]; then
  echo "ERROR: Local ROOT_PATH does not exist: $ROOT_PATH" >&2
  exit 1
fi

if ! ssh -o BatchMode=yes -o ConnectTimeout=8 "$JZ_HOST" "command -v rsync >/dev/null 2>&1 && echo ok" >/dev/null 2>&1; then
  echo "ERROR: Cannot connect to '$JZ_HOST' via SSH, or rsync is missing on the remote side." >&2
  exit 1
fi

ROOT_PATH_JZ_Q="$(printf '%q' "$ROOT_PATH_JZ")"
if ! ssh -o BatchMode=yes -o ConnectTimeout=8 "$JZ_HOST" "mkdir -p ${ROOT_PATH_JZ_Q}" >/dev/null 2>&1; then
  echo "ERROR: Could not create/access remote ROOT_PATH on '$JZ_HOST': $ROOT_PATH_JZ" >&2
  exit 1
fi

echo "Local ROOT_PATH       : $ROOT_PATH"
echo "Jean Zay ROOT_PATH    : $ROOT_PATH_JZ"
echo "Jean Zay host         : $JZ_HOST"
echo "training_dataset_id   : $TRAINING_DATASET_ID"
echo "eval_st id            : $EVAL_ST_ID"
echo "Dry run               : $DRY_RUN"
echo

# --- Build requested roots ----------------------------------------------------

ST_EVAL_BASE="eval/eval_datasets/st_eval_data"
HEST_BASE="${ST_EVAL_BASE}/hest"
EVAL_BASE="${ST_EVAL_BASE}/${EVAL_ST_ID}"

declare -a REQUESTED_ROOTS=()

# Exact paths requested
REQUESTED_ROOTS+=("${HEST_BASE}/patches")
REQUESTED_ROOTS+=("${EVAL_BASE}/metadata_${EVAL_ST_ID}.csv")
REQUESTED_ROOTS+=("${EVAL_BASE}/patches")
REQUESTED_ROOTS+=("${EVAL_BASE}/st")

# All cell_token_membership_* directories except missing_cells
shopt -s nullglob
for p in "$ROOT_PATH/$EVAL_BASE"/cell_token_membership_*; do
  base="$(basename "$p")"
  if [[ "$base" == cell_token_membership_* && "$base" != cell_token_membership_missing_cells_* ]]; then
    REQUESTED_ROOTS+=("${p#"$ROOT_PATH/"}")
  fi
done
shopt -u nullglob

echo "Requested roots:"
for rel in "${REQUESTED_ROOTS[@]}"; do
  echo "  - $rel"
done
echo

# --- Build rsync manifest: regular files + real dirs only ---------------------

WARN_COUNT=0
ROOT_ACCEPTED_COUNT=0
MANIFEST_ENTRY_COUNT=0

MANIFEST_FILE="$(mktemp)"
trap 'rm -f "$MANIFEST_FILE"' EXIT

warn() {
  echo "WARNING: $*" >&2
  ((WARN_COUNT+=1))
}

should_exclude_relpath() {
  local rel="$1"
  local base
  base="$(basename "$rel")"

  case "$base" in
    ._*)       return 0 ;;  # macOS AppleDouble metadata files
    .DS_Store) return 0 ;;  # macOS Finder metadata
  esac

  return 1
}

append_manifest_entry() {
  local rel="$1"
  if should_exclude_relpath "$rel"; then
    return 0
  fi
  printf '%s\0' "$rel" >> "$MANIFEST_FILE"
  ((MANIFEST_ENTRY_COUNT+=1))
}

append_tree_without_symlinks() {
  local rel="$1"
  local src="$ROOT_PATH/$rel"

  if [[ ! -e "$src" && ! -L "$src" ]]; then
    warn "Missing, skipping: $rel"
    return 0
  fi

  if [[ -L "$src" ]]; then
    warn "Root path is itself a symlink, skipping entirely: $rel"
    return 0
  fi

  if [[ -f "$src" ]]; then
    append_manifest_entry "$rel"
    ((ROOT_ACCEPTED_COUNT+=1))
    return 0
  fi

  if [[ -d "$src" ]]; then
    ((ROOT_ACCEPTED_COUNT+=1))
    while IFS= read -r -d '' abs_path; do
      # find without -L: regular files and real directories only, symlinks not followed
      append_manifest_entry "${abs_path#"$ROOT_PATH/"}"
    done < <(find "$src" \( -type d -o -type f \) -print0)
    return 0
  fi

  warn "Not a regular file or real directory, skipping: $rel"
}

echo "Building transfer manifest..."
echo "Symlinks will be skipped and never dereferenced."
echo

for rel in "${REQUESTED_ROOTS[@]}"; do
  append_tree_without_symlinks "$rel"
done

if [[ ! -s "$MANIFEST_FILE" ]]; then
  echo "Nothing to sync."
  echo "Warnings: $WARN_COUNT"
  exit 0
fi

echo "Accepted roots   : $ROOT_ACCEPTED_COUNT"
echo "Manifest entries : $MANIFEST_ENTRY_COUNT"
echo "Warnings         : $WARN_COUNT"
echo

# --- Run rsync ----------------------------------------------------------------

RSYNC_OPTS=(
  -avh
  --partial
  --progress
  --from0
  --files-from="$MANIFEST_FILE"
  --exclude='._*'
  --exclude='.DS_Store'
)

if [[ "$DRY_RUN" == "1" ]]; then
  RSYNC_OPTS+=(--dry-run --itemize-changes)
fi

echo "Starting sync..."
rsync "${RSYNC_OPTS[@]}" "$ROOT_PATH/" "${JZ_HOST}:${ROOT_PATH_JZ}/"

echo
echo "Done."
echo "Accepted roots   : $ROOT_ACCEPTED_COUNT"
echo "Manifest entries : $MANIFEST_ENTRY_COUNT"
echo "Warnings         : $WARN_COUNT"