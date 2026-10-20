#!/usr/bin/env bash

# Usage for a single training:
# ./scripts/train/set_up_train_histost.sh send_data training_46
# ./scripts/train/set_up_train_histost.sh get_data exclude training_46
# ./scripts/train/set_up_train_histost.sh get_data noexclude training_46
# ./scripts/train/set_up_train_histost.sh sync_wandb training_46 group1

# Usage for a list of trainings:
# for tid in training_40 training_41 training_42; do ./scripts/train/set_up_train_histost.sh send_data "$tid"; done
# for tid in training_40 training_41 training_42; do ./scripts/train/set_up_train_histost.sh get_data exclude "$tid"; done
# for tid in training_40 training_41 training_42; do ./scripts/train/set_up_train_histost.sh sync_wandb "$tid" group1; done

set -euo pipefail

# ---- Configuration ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

_PATHS=()
while IFS= read -r line; do
    _PATHS+=("$line")
done < <(
    PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" python3 - <<'PY'
from src.utils.constants import ROOT_PATH, ROOT_PATH_JeanZay
print(ROOT_PATH)
print(ROOT_PATH_JeanZay)
PY
)

if [[ "${#_PATHS[@]}" -ne 2 ]]; then
    echo "Could not load ROOT_PATH and ROOT_PATH_JeanZay from src/utils/constants.py" >&2
    exit 1
fi

LOCAL_BASE="${_PATHS[0]}"
REMOTE_BASE="jeanzay:${_PATHS[1]}"

# ---- Functions ----

remote_mkdir_p() {
    local remote_path="$1"
    local remote_host="${REMOTE_BASE%%:*}"
    ssh "$remote_host" "mkdir -p \"$remote_path\""
}

send_data() {
    local training="$1"
    local remote_dir="${_PATHS[1]}/train/trainings/trainings_HistoST/$training/training"

    # rsync of a single file does not create missing remote parent directories
    remote_mkdir_p "$remote_dir"

    rsync --partial --progress -r \
        "$LOCAL_BASE/train/trainings/trainings_HistoST/$training/training/${training}_histost.yaml" \
        "$REMOTE_BASE/train/trainings/trainings_HistoST/$training/training/"
}

get_data() {
    local exclude_flag="$1"
    local training="$2"
    local local_dir="$LOCAL_BASE/train/trainings/trainings_HistoST/$training"

    mkdir -p "$local_dir"

    # Default for 'noexclude': still skip eval and HEencoder_checkpoints
    local exclude_option="--exclude=eval --exclude=HEencoder_checkpoints"
    if [[ "$exclude_flag" == "exclude" ]]; then
        # For 'exclude': also skip log/checkpoints
        exclude_option="--exclude=log/checkpoints --exclude=eval --exclude=HEencoder_checkpoints"
    fi

    rsync --partial --progress -r $exclude_option \
        "$REMOTE_BASE/train/trainings/trainings_HistoST/$training/" \
        "$LOCAL_BASE/train/trainings/trainings_HistoST/$training/"
}

sync_wandb() {
    local training="$1"
    local name_group="$2"
    local wandb_dir="$LOCAL_BASE/train/trainings/trainings_HistoST/$training/training/wandb"
    local synced_file="$wandb_dir/.synced"

    source "$HOME/miniforge3/etc/profile.d/conda.sh"
    conda activate histost

    mkdir -p "$wandb_dir"
    touch "$synced_file"

    # Get all offline-run folders sorted by date
    find "$wandb_dir" -maxdepth 1 -type d -name 'offline-run-*' | sort | while read -r run_folder; do
        run_id=$(basename "$run_folder" | sed -E 's/.*-([a-z0-9]{8})$/\1/')

        # Check if already synced
        if grep -q "$run_folder" "$synced_file"; then
            echo "Already synced: $run_folder"
            continue
        fi

        # Point wandb sync at the .wandb file inside the run folder, not the folder itself
        wandb_file=$(find "$run_folder" -maxdepth 2 -name 'run-*.wandb' | head -n 1)
        if [[ -z "$wandb_file" ]]; then
            echo "No .wandb file found in $run_folder, skipping."
            continue
        fi

        if ! grep -q 'first_sync_done' "$synced_file"; then
            wandb sync --include-offline --project histost --job_type "$name_group" --id "$run_id" "$wandb_file"

            echo "first_sync_done" >> "$synced_file"
        else
            wandb sync --include-offline --project histost --job_type "$name_group" --id "$run_id" --append "$wandb_file"
        fi

        echo "$run_folder" >> "$synced_file"
    done
}

# ---- Dispatcher ----
case "$1" in
    send_data)
        send_data "$2"
        ;;
    get_data)
        get_data "$2" "$3"
        ;;
    sync_wandb)
        sync_wandb "$2" "$3"
        ;;
    *)
        echo "Usage:"
        echo "  $0 send_data <training_id>"
        echo "  $0 get_data <exclude|noexclude> <training_id>"
        echo "  $0 sync_wandb <training_id> <name_group>"
        exit 1
        ;;
esac