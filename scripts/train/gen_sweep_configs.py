#!/usr/bin/env python
"""
Generate the training configs of the LoRA-rank sweep (training_40/41/42: rank 8/4/16, alpha = rank)
from configs/template_train_histost_config.yaml.

Saved in <ROOT_HistoST> / train / trainings / trainings_HistoST / <training_id> / training / <training_id>_histost.yaml
"""

import os
import sys
import copy
import argparse
from typing import Any, Mapping

import yaml

# Add the project root to sys.path so that 'src' can be imported
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils.constants import get_full_path


# =====================================================================================
# WHAT TO GENERATE
# =====================================================================================
TEMPLATE_PATH = os.path.join(project_root, "configs", "template_train_histost_config.yaml")

# W&B group of the generated configs
SWEEP_GROUP = "rank_frontier"

# Values applied to every generated config, on top of the template
BASELINE = {
    "data_select": {
        # Number of training patches with n_min_cells=5 (must match the train loader length)
        "num_train_samples": 327955,
    },
    "training": {
        "epochs": 20,
        "save_every": 1,        # save a checkpoint every epoch
        "early_stopping": {
            # Early stopping disabled
            "patience": None,
        },
        "scheduler": {
            "warmup_steps": 500,   # ~1 epoch
        },
        # Per-slide head-cap sampling (the template default is "random")
        "sampling_strategy": "slide_headcap",
        # LoRA rank/alpha (overridden per run in RUNS)
        "lora_rank": 8,
        "lora_alpha": 8,
    },
    "advanced_scheduling": {
        # No CLIP-head freeze (warmup epoch > epochs); advanced_scheduling creates the separate lora param group
        "clip_head_warmup_epoch": 999,
    },
}

# (training_id, override, role) of each generated config; alpha == rank keeps the LoRA scaling at 1.0
RUNS = [
    ("training_40", {"training": {"lora_rank": 8, "lora_alpha": 8}},
     "LoRA rank 8, alpha 8"),
    ("training_41", {"training": {"lora_rank": 4, "lora_alpha": 4}},
     "LoRA rank 4, alpha 4"),
    ("training_42", {"training": {"lora_rank": 16, "lora_alpha": 16}},
     "LoRA rank 16, alpha 16"),
]


# =====================================================================================
# Implementation
# =====================================================================================
def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict:
    """Recursively merge `override` onto a deepcopy of `base` (dict leaves win)."""
    out = copy.deepcopy(dict(base))
    for key, val in override.items():
        if key in out and isinstance(out[key], Mapping) and isinstance(val, Mapping):
            out[key] = deep_merge(out[key], val)
        else:
            out[key] = copy.deepcopy(val)
    return out


def out_yaml_path(new_id: str) -> str:
    return get_full_path(
        "train", "trainings", "trainings_HistoST", new_id, "training",
        f"{new_id}_histost.yaml", as_str=True,
    )


def set_identity(cfg: dict, new_id: str, role: str) -> None:
    """Rewrite every id-bearing logging field so the new run is fully independent."""
    log = cfg.setdefault("logging", {})
    log["training_id_histost"] = new_id
    log["notes"] = f"{new_id}: {role}"
    log["log_comment"] = new_id
    log["tags"] = ["HistoST", new_id, SWEEP_GROUP]
    log["group"] = SWEEP_GROUP


def _get(cfg: Mapping[str, Any], *keys: str) -> Any:
    cur: Any = cfg
    for key in keys:
        if not isinstance(cur, Mapping) or key not in cur:
            return None
        cur = cur[key]
    return cur


def sanity_check(cfg: Mapping[str, Any], tid: str) -> None:
    """Check the config values required for these runs (raises on failure)."""
    errors: list[str] = []

    # num_train_samples must be resolved (the template has a placeholder)
    nts = _get(cfg, "data_select", "num_train_samples")
    if not isinstance(nts, int):
        errors.append(f"data_select.num_train_samples must be an int, got {nts!r}")

    # Head-cap sampling requires sampling_strategy=slide_headcap and batch_slide_mode=mixed
    if _get(cfg, "training", "sampling_strategy") != "slide_headcap":
        errors.append("training.sampling_strategy must be 'slide_headcap'")
    if _get(cfg, "training", "batch_slide_mode") != "mixed":
        errors.append("training.batch_slide_mode must be 'mixed' "
                      "(single_slide/alternate disable the head-cap)")

    # Teacher centering and target temperatures
    if _get(cfg, "model", "clip", "center_teacher_target") is not True:
        errors.append("model.clip.center_teacher_target must be true")
    for key in ("target_temperature", "token_target_temperature"):
        if _get(cfg, "model", "clip", key) != 0.1:
            errors.append(f"model.clip.{key} must be 0.1")

    # LoRA weight decay must be > 0 (code default is 0.0)
    wd_lora = _get(cfg, "advanced_scheduling", "weight_decay_lora")
    if not (isinstance(wd_lora, (int, float)) and wd_lora > 0):
        errors.append(f"advanced_scheduling.weight_decay_lora must be > 0, got {wd_lora!r}")

    # advanced_scheduling creates the lora param group; no head freeze within the run
    if _get(cfg, "advanced_scheduling", "enabled") is not True:
        errors.append("advanced_scheduling.enabled must be true (needed for the lora param group)")
    warm = _get(cfg, "advanced_scheduling", "clip_head_warmup_epoch")
    epochs = _get(cfg, "training", "epochs")
    if not (isinstance(warm, int) and isinstance(epochs, int) and warm > epochs):
        errors.append(f"advanced_scheduling.clip_head_warmup_epoch ({warm}) must be > "
                      f"training.epochs ({epochs}) => no head freeze")

    # Token loss warmup must end before any head freeze
    tok_warm = _get(cfg, "model", "clip", "lambda_token_warmup_epochs")
    if isinstance(tok_warm, int) and isinstance(warm, int) and tok_warm >= warm:
        errors.append(f"model.clip.lambda_token_warmup_epochs ({tok_warm}) must be < "
                      f"clip_head_warmup_epoch ({warm})")

    # LoRA fine-tuning of the H&E encoder
    if _get(cfg, "training", "HE_training_option") != "lora":
        errors.append("training.HE_training_option must be 'lora'")
    if _get(cfg, "model", "he", "use_mlp_head") is not False:
        errors.append("model.he.use_mlp_head must be False (enforced under advanced_scheduling)")

    # alpha == rank (LoRA scaling 1.0)
    rank, alpha = _get(cfg, "training", "lora_rank"), _get(cfg, "training", "lora_alpha")
    if rank != alpha:
        errors.append(f"lora_alpha ({alpha}) must equal lora_rank ({rank}) (LoRA scaling 1.0)")

    if errors:
        raise SystemExit(f"[{tid}] config sanity check FAILED:\n  - " + "\n  - ".join(errors))


def summarize_override(ov: Mapping[str, Any], prefix: str = "") -> list[str]:
    """Flatten an override dict to 'a.b.c = v' strings for a readable per-config diff."""
    items: list[str] = []
    for k, v in ov.items():
        path = f"{prefix}.{k}" if prefix else k
        if isinstance(v, Mapping):
            items.extend(summarize_override(v, path))
        else:
            items.append(f"{path} = {v!r}")
    return items


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the LoRA-rank sweep training configs.")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite an existing training_{id} config if the folder already exists.")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print what would be generated without writing any file.")
    args = parser.parse_args()

    if not os.path.isfile(TEMPLATE_PATH):
        parser.error(f"Template config not found: {TEMPLATE_PATH}")
    with open(TEMPLATE_PATH, "r") as f:
        template_cfg = yaml.safe_load(f)

    print(f"Template : {TEMPLATE_PATH}")
    print(f"Baseline : {summarize_override(BASELINE)}")
    print(f"Runs     : {len(RUNS)} configs, wandb group '{SWEEP_GROUP}'\n")

    written = []
    for new_id, override, role in RUNS:
        cfg = deep_merge(deep_merge(template_cfg, BASELINE), override)
        set_identity(cfg, new_id, role)
        sanity_check(cfg, new_id)

        print(f"{new_id}:  " + "  |  ".join(summarize_override(override)) + f"   # {role}")
        if args.dry_run:
            continue

        dest = out_yaml_path(new_id)
        run_dir = os.path.dirname(os.path.dirname(dest))
        if os.path.isdir(run_dir) and not args.force:
            parser.error(f"Refusing to overwrite existing run folder for {new_id} ({run_dir}). "
                         f"Use --force to override.")
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        header = (f"# {new_id} -- {role}\n"
                  f"# generated by gen_sweep_configs.py from "
                  f"configs/template_train_histost_config.yaml (group {SWEEP_GROUP})\n\n")
        with open(dest, "w") as f:
            f.write(header)
            yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False, allow_unicode=True)
        written.append(new_id)

    ids = [nid for nid, _, _ in RUNS]
    print("\n" + "=" * 78)
    if args.dry_run:
        print("DRY RUN - nothing written.  (sanity checks still ran on every config)")
    else:
        print(f"Wrote {len(written)} config(s) under ROOT_PATH/train/trainings/trainings_HistoST/")
    print("\nBefore launching:")
    print("  1. teacher_center_means.h5 must exist, precomputed at n_min_cells=5:")
    print("       python src/dataset/precompute_teacher_mean.py --novae_training_id training_4 \\")
    print("           --dataset_id ds_ct1_20x_224_o112 --training_dataset_id ds_ct1_20x_224_o112_1 \\")
    print("           --n_min_cells 5")
    print("  2. At launch, confirm the optimizer log prints:  - lora:  lr=0.0001 wd=0.2")
    print("\nPaste into scripts/train/slurm_sweep_histost_jeanzay.sh:")
    print("  TIDS=(" + " ".join(ids) + ")")
    print(f"  #SBATCH --array=0-{len(ids) - 1}%3      # %3 = 3 runs in parallel")
    print("\nThen send the configs to Jean Zay:")
    print("  for tid in " + " ".join(ids) + "; do \\")
    print("    ./scripts/train/set_up_train_histost.sh send_data \"$tid\"; done")
    print("=" * 78)


if __name__ == "__main__":
    main()
