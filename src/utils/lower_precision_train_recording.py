"""
Round the floats of backup_training_metrics.jsonl to 4 decimals and keep a subset of epochs (given, or evenly spaced across the run).
Saved in <ROOT_HistoST> / train / trainings / trainings_HistoST / <training_id_histost> / training / log / backup_training_metrics_low-precision.jsonl
"""

import json
import os
import sys
import argparse

# Add the project root to sys.path so that 'src' can be imported
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.constants import get_full_path


def round_floats(obj, decimals=3):
    if isinstance(obj, float):
        return round(obj, decimals)
    if isinstance(obj, list):
        return [round_floats(x, decimals) for x in obj]
    if isinstance(obj, dict):
        return {k: round_floats(v, decimals) for k, v in obj.items()}
    return obj


def choose_evenly_spaced_epochs(all_epochs, max_epochs):
    """Keep epochs spread across the whole run."""
    all_epochs = sorted(set(all_epochs))
    n = len(all_epochs)

    if max_epochs is None or max_epochs >= n:
        return set(all_epochs)

    if max_epochs <= 1:
        return {all_epochs[-1]}

    idxs = []
    for i in range(max_epochs):
        idx = round(i * (n - 1) / (max_epochs - 1))
        idxs.append(idx)

    return {all_epochs[i] for i in sorted(set(idxs))}


def main(args):

    INPUT_FILE = get_full_path('train', 'trainings', 'trainings_HistoST', args.training_id_histost, 'training', 'log', "backup_training_metrics.jsonl")
    OUTPUT_FILE = get_full_path('train', 'trainings', 'trainings_HistoST', args.training_id_histost, 'training', 'log', "backup_training_metrics_low-precision.jsonl")

    # Read all records
    records = []
    with open(INPUT_FILE, "r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            records.append(data)

    # Check epoch field exists
    records_with_epoch = [r for r in records if "epoch" in r]
    if not records_with_epoch:
        raise ValueError("No 'epoch' field found in the JSONL records.")

    all_epochs = [r["epoch"] for r in records_with_epoch]

    # Select epochs
    if args.keep_epochs is not None:
        selected_epochs = set(args.keep_epochs)
    else:
        selected_epochs = choose_evenly_spaced_epochs(all_epochs, args.max_epochs)

    # Filter + round precision
    filtered_records = []
    for r in records:
        if r.get("epoch") in selected_epochs:
            filtered_records.append(round_floats(r, decimals=4))

    # Save
    with open(OUTPUT_FILE, "w", encoding="utf-8") as fout:
        for r in filtered_records:
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Saved: {OUTPUT_FILE}")
    print(f"Kept epochs: {sorted(selected_epochs)}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Round floating-point numbers in training metrics to lower precision.")

    parser.add_argument("--training_id_histost", type=str, 
                        default="training_46",
                        help="Training ID for HistoST (e.g., training_46)")
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--keep_epochs", type=int, nargs="+",
                        help="List of specific epochs to keep (e.g., --keep_epochs 0 1 2 5 10)")
    group.add_argument("--max_epochs", type=int,
                        help="Maximum number of epochs to keep, automatically selected to be spread across the run (e.g., --max_epochs 10)")

    args = parser.parse_args()
    main(args)


