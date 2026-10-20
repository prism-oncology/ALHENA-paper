#!/usr/bin/env bash
set -euo pipefail

# This script creates symbolic links from the shared HEST folder to the evaluation folder of a training dataset ID,
# for the HEST slides of its metadata CSV: patch and ST files, and the per-slide mouse->human gene mapping parquet
# of "Mus musculus" slides. With the optional argument "restrict", only the patch symlinks are created.
#
# The symbolic links are thus inside:
#   <ROOT_HistoST> / eval / eval_datasets / st_eval_data / eval_st_<training_dataset_id> / patches
#   and, unless "restrict" is provided, also / st and / mouse_to_human_gene_mappings

usage() {
  cat <<'EOF'
Usage:
  ./_5_symbolic_links_hest.sh TRAINING_DATASET_ID [restrict]

Example:
  ./_5_symbolic_links_hest.sh ds_ct1_20x_224_o112_1
  ./_5_symbolic_links_hest.sh ds_ct1_20x_224_o112_1 restrict
EOF
}


# Parse args
training_dataset_id=""
restrict_mode=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    restrict)
      restrict_mode=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      if [[ -z "$training_dataset_id" ]]; then
        training_dataset_id="$1"
        shift
      else
        echo "ERROR: Unknown argument: $1" >&2
        usage
        exit 2
      fi
      ;;
  esac
done

[[ -n "$training_dataset_id" ]] || { echo "ERROR: TRAINING_DATASET_ID is required" >&2; usage; exit 2; }

# Locate the project root as in the Python scripts: Path(__file__).resolve().parents[4]
SCRIPT_PATH="$(python3 - "$0" <<'PY'
from pathlib import Path
import sys
print(str(Path(sys.argv[1]).resolve()))
PY
)"

PROJECT_ROOT="$(python3 - "$SCRIPT_PATH" <<'PY'
from pathlib import Path
import sys
p = Path(sys.argv[1]).resolve()
try:
  print(str(p.parents[4]))
except IndexError:
  raise SystemExit(f"ERROR: Cannot compute parents[4] from script path: {p}")
PY
)"

# Helper to call get_full_path from the project (path segments passed as argv), returning an absolute path string
get_full_path_py() {
  python3 - <<'PY' "$PROJECT_ROOT" "$@"
import sys
from pathlib import Path

project_root = Path(sys.argv[1]).resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from src.utils.constants import get_full_path
except Exception as e:
    raise SystemExit(f"ERROR: Failed to import get_full_path from src.utils.constants (project_root={project_root}): {e}")

parts = sys.argv[2:]
try:
    p = get_full_path(*parts)
except Exception as e:
    raise SystemExit(f"ERROR: get_full_path({parts}) failed: {e}")

print(str(Path(p)))
PY
}

# Source directories for HEST data
hest_root="$(get_full_path_py eval eval_datasets st_eval_data hest)"
hest_patches_dir="${hest_root}/patches"

# Basic checks
if [[ ! -d "$hest_patches_dir" ]]; then
  echo "ERROR: HEST patches folder does not exist: $hest_patches_dir" >&2
  exit 1
fi

if [[ $restrict_mode -eq 0 ]]; then
  hest_st_dir="${hest_root}/st"
  if [[ ! -d "$hest_st_dir" ]]; then
    echo "ERROR: HEST ST folder does not exist: $hest_st_dir" >&2
    exit 1
  fi
fi

# eval_st_training_dataset_id path
eval_folder="$(get_full_path_py eval eval_datasets st_eval_data "eval_st_${training_dataset_id}")"
if [[ ! -d "$eval_folder" ]]; then
  echo "ERROR: Evaluation folder does not exist: $eval_folder" >&2
  exit 1
fi

# Metadata csv path: must be exactly one matching file
shopt -s nullglob
metadata_csvs=( "${eval_folder}"/metadata_eval_st_*.csv )
shopt -u nullglob

if [[ ${#metadata_csvs[@]} -eq 0 ]]; then
  echo "ERROR: No metadata CSV files found in $eval_folder matching pattern 'metadata_eval_st_*.csv'." >&2
  exit 1
elif [[ ${#metadata_csvs[@]} -gt 1 ]]; then
  echo "ERROR: Multiple metadata CSV files found in $eval_folder matching pattern 'metadata_eval_st_*.csv'. Ensure there is exactly one." >&2
  printf 'Found:\n' >&2
  for f in "${metadata_csvs[@]}"; do printf '  %s\n' "$f" >&2; done
  exit 1
fi

metadata_path="${metadata_csvs[0]}"
echo "Using metadata CSV: $metadata_path"

# Extract unique HEST ids and their unique species
declare -a ids=()
declare -a species_list=()

while IFS=$'\t' read -r id_ species; do
  ids+=( "$id_" )
  species_list+=( "$species" )
done < <(python3 - <<'PY' "$metadata_path"
import sys
import pandas as pd

metadata_path = sys.argv[1]
try:
    df = pd.read_csv(metadata_path)
except Exception as e:
    raise SystemExit(f"ERROR: Failed to read CSV {metadata_path}: {e}")

for col in ("source", "slide_id", "species"):
    if col not in df.columns:
        raise SystemExit(f"ERROR: Expected column '{col}' not found in metadata CSV {metadata_path}")

sub = df.loc[df["source"] == "HEST", ["slide_id", "species"]].dropna(subset=["slide_id"]).copy()
sub["slide_id"] = sub["slide_id"].astype(str)
sub["species"] = sub["species"].astype(str)

# Keep first-appearance order of IDs, but enforce a unique species per slide ID
for slide_id, g in sub.groupby("slide_id", sort=False):
    species_values = [str(x) for x in g["species"].dropna().unique().tolist()]
    if len(species_values) == 0:
        raise SystemExit(f"ERROR: No species value found for HEST slide id '{slide_id}' in metadata CSV {metadata_path}")
    if len(species_values) > 1:
        raise SystemExit(
            f"ERROR: Multiple species values found for HEST slide id '{slide_id}' in metadata CSV {metadata_path}: "
            f"{species_values}"
        )
    print(f"{slide_id}\t{species_values[0]}")
PY
)

if [[ ${#ids[@]} -eq 0 ]]; then
  echo "No rows found with source == 'HEST'. Nothing to do."
  exit 0
fi
echo "Found ${#ids[@]} unique slide IDs with source 'HEST'."

# Target directories for symlinks
target_patches_dir="${eval_folder}/patches"
mkdir -p "$target_patches_dir"

if [[ $restrict_mode -eq 0 ]]; then
  target_st_dir="${eval_folder}/st"
  target_mappings_dir="${eval_folder}/mouse_to_human_gene_mappings"
  mkdir -p "$target_st_dir" "$target_mappings_dir"
fi

# Remove an existing link/file (error if the path is a directory)
remove_existing_link_or_file() {
  local path="$1"
  local label="$2" # just for messages

  if [[ -e "$path" || -L "$path" ]]; then
    # If it's a symlink or a regular file -> remove
    # If it's a directory or something else -> error
    if [[ -L "$path" || -f "$path" ]]; then
      rm -f -- "$path"
      echo "${label} Removed existing link/file: $path"
    else
      echo "ERROR: Expected $path to be a file or symlink, but it exists and is not. Please resolve this manually." >&2
      exit 1
    fi
  fi
}

# Process each slide ID
for i in "${!ids[@]}"; do
  echo
  id_="${ids[$i]}"
  species="${species_list[$i]}"

  patch_source="${hest_patches_dir}/${id_}.h5"

  st_source=""
  if [[ $restrict_mode -eq 0 ]]; then
    st_source="${hest_st_dir}/${id_}.h5ad"
  fi
  mapping_source="${hest_root}/${id_}_mouse2human_mapping.parquet"

  patch_exists=0
  st_exists=0
  mapping_exists=0
  [[ -f "$patch_source" ]] && patch_exists=1
  [[ -f "$st_source" ]] && st_exists=1
  [[ -f "$mapping_source" ]] && mapping_exists=1

  mapping_required=0
  if [[ "$species" == "Mus musculus" ]]; then
    mapping_required=1
  fi

  if [[ $restrict_mode -eq 1 ]]; then
    if [[ $patch_exists -eq 0 ]]; then
      echo "ERROR: [id=${id_}] Restrict mode requires patch file, but it is missing: ${patch_source}" >&2
      exit 1
    fi

    patch_link="${target_patches_dir}/$(basename "$patch_source")"
    remove_existing_link_or_file "$patch_link" "[id=${id_}]"

    ln -s -- "$patch_source" "$patch_link"
    echo "[id=${id_}] Created patch symlink: $patch_link -> $patch_source"

    if [[ $st_exists -eq 1 ]]; then
      echo "[id=${id_}] Restrict mode enabled: skipping ST symlink"
    fi
    if [[ $mapping_required -eq 1 && $mapping_exists -eq 1 ]]; then
      echo "[id=${id_}] Restrict mode enabled: skipping mapping symlink"
    fi
    continue
  fi

  if [[ $patch_exists -eq 1 && $st_exists -eq 1 ]]; then

    if [[ $mapping_required -eq 1 && $mapping_exists -eq 0 ]]; then
      echo "ERROR: [id=${id_}] species='${species}' requires mapping file but it is missing: ${mapping_source}" >&2
      exit 1
    fi

    patch_link="${target_patches_dir}/$(basename "$patch_source")"
    st_link="${target_st_dir}/$(basename "$st_source")"

    remove_existing_link_or_file "$patch_link" "[id=${id_}]"
    remove_existing_link_or_file "$st_link"   "[id=${id_}]"
  
    if [[ $mapping_required -eq 1 ]]; then
      mapping_link="${target_mappings_dir}/$(basename "$mapping_source")"
      remove_existing_link_or_file "$mapping_link" "[id=${id_}]"
    fi

    ln -s -- "$patch_source" "$patch_link"
    echo "[id=${id_}] Created patch symlink: $patch_link -> $patch_source"
    ln -s -- "$st_source" "$st_link"
    echo "[id=${id_}] Created ST symlink: $st_link -> $st_source"

    if [[ $mapping_required -eq 1 ]]; then
      ln -s -- "$mapping_source" "$mapping_link"
      echo "[id=${id_}] Created mapping symlink: $mapping_link -> $mapping_source"
    fi

  elif [[ $patch_exists -eq 1 && $st_exists -eq 0 ]]; then
    echo "ERROR: [id=${id_}] Found patch file ${patch_source} but missing ST file ${st_source}" >&2
    exit 1

  elif [[ $patch_exists -eq 0 && $st_exists -eq 1 ]]; then
    echo "ERROR: [id=${id_}] Found ST file ${st_source} but missing patch file ${patch_source}" >&2
    exit 1

  else
    echo "ERROR: [id=${id_}] Slide is listed in metadata, but neither patch nor ST file was found: patch=${patch_source}, st=${st_source}" >&2
    exit 1
  fi
done