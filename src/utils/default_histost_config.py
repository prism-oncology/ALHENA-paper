from copy import deepcopy
from typing import Any, Mapping


# HistoST runtime defaults used in code (not the recommended values of the YAML template)
DEFAULT_HISTOST_CONFIG = {
    "seed": 42,
    "data": {
        "cached_dataset": "no",
    },
    "data_select": {
        "n_min_cells": 0,
        # K cells sampled per patch for the scConcept token loss (see model.clip.lambda_token).
        "scconcept_max_cells": 32,
        # The ST-graph edge radius is set at precompute time (precompute_novae_teacher.py --edge_distance_threshold)
    },
    "model": {
        "he": {
            "use_mlp_head": False,
            "he_mlp_hidden_dim": 1024,
            "he_mlp_out_dim": 512,
            "he_mlp_dropout": 0.1,
        },
        "st": {
            # scConcept token teacher (ST cell branch; used only when model.clip.lambda_token > 0).
            "scconcept_training_id": None,
            "scconcept_embeddings_dir": None,
            # Precomputed Novae ST teacher (cache built by src/dataset/precompute_novae_teacher.py)
            "novae_training_id": None,          # selects novae_teacher_cache/{novae_training_id}/ (required)
            # --- Resolved at runtime from the cache manifest by Experiment.__init__ (not to be set by hand):
            "novae_teacher_cache_dir": None,    # absolute path to the cache dir
            "novae_teacher_fingerprint": None,  # cache content fingerprint (checked at every per-slide h5 open)
            "st_feature_dim": None,             # st_feat dim = the cache manifest 'dim' (= 128); st_proj input
            "novae_pretrained_ckpt": None,      # provenance only (which teacher checkpoint produced the cache)
            "teacher_center_means_path": None,  # resolved at runtime (teacher_center_means.h5)
        },
        "clip": {
            "clip_embed_dim": 512,
            "loss_type": "soft_targets_st",
            "target_temperature": 0.1,
            "lambda_patch": 1.0,
            "patch_local_loss": True,
            "patch_gather_with_grad": False,
            "patch_reinsert_local": True,
            "token_embed_dim": None,
            "token_loss_type": "soft_targets_st",
            "token_target_temperature": 0.1,
            "lambda_token": 0.5,
            "lambda_token_warmup_epochs": 5,
            "token_local_loss": True,
            "token_gather_with_grad": False,
            "token_reinsert_local": True,
            # Center the teacher vectors by their global mean (requires teacher_center_means.h5, precompute_teacher_mean.py)
            "center_teacher_target": True,
        },
        "pretrained_histost": None,
    },
    "training": {
        "batch_size": 128,
        "num_workers": 23,
        "optimizer_hyperparameters": {
            "lr": 1e-3,
            "betas": [0.9, 0.999],
            "weight_decay": 0.01,
            "eps": 1e-8,
        },
        "early_stopping": {
            "patience": None,
            "monitor": "val_loss",
            "mode": "min",
        },
        "scheduler": {
            "scheduler_type": "constant",
            "warmup_steps": 0,
        },
        "batch_slide_mode": "mixed",
        "alternate_start_with": "mixed",
        "val_step_slide_mode": "inherit",
        "sampling_strategy": "random",
        "sampling_gamma": 0.5,
        # slide_headcap strategy: w(slide) = min(1, C/count), C derived from headcap_max_oversampling unless headcap_C is set
        "headcap_max_oversampling": 1.3,
        "headcap_C": None,
        "ignore_cat": None,
        "mixed_precision": False,
        "amp_dtype": "bf16",
        "eval_every": 1,
        "cka_every": 1,
        "save_every": 1,  # keep 1 (W&B and JSON metrics are saved every epoch)
        "HE_training_option": "lora",
        "lora_rank": 64,
        "lora_alpha": 16,
    },
    "advanced_scheduling": {
        "enabled": False,
        "clip_head_warmup_epoch": 1,
        # AdamW weight decay on the LoRA factors (advanced scheduling only; 0.0 = no decay)
        "weight_decay_lora": 0.0,
    },
    "he_transformations": {},
    "validation": {
        # Min patches per slide for the within-slide retrieval metric (effective floor: max(this, max_topk + 1))
        "within_slide_min_patches": 16,
    },
}

_MISSING = object()


def _deep_merge(defaults: Mapping[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    merged = deepcopy(dict(defaults))
    for key, value in overrides.items():
        if (
            key in merged
            and isinstance(merged[key], Mapping)
            and isinstance(value, Mapping)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _deep_get(mapping: Mapping[str, Any], keys: tuple[str, ...], default: Any = _MISSING) -> Any:
    cur: Any = mapping
    for key in keys:
        if not isinstance(cur, Mapping) or key not in cur:
            return default
        cur = cur[key]
    return cur


def get_default(*keys: str, default: Any = _MISSING) -> Any:
    value = _deep_get(DEFAULT_HISTOST_CONFIG, keys, _MISSING)
    if value is _MISSING:
        if default is not _MISSING:
            return default
        raise KeyError(f"No HistoST runtime default is registered for path: {'.'.join(keys)}")
    return deepcopy(value)


def cfg_get(cfg: Mapping[str, Any], *keys: str, default: Any = _MISSING) -> Any:
    value = _deep_get(cfg, keys, _MISSING)
    if value is not _MISSING:
        return value
    return get_default(*keys, default=default)


def merge_with_defaults(cfg: Mapping[str, Any] | None) -> dict[str, Any]:
    """
    Return a fully materialized config where user-provided values override
    DEFAULT_HISTOST_CONFIG recursively.
    """
    if cfg is None:
        cfg = {}
    if not isinstance(cfg, Mapping):
        raise TypeError(f"Config must be a mapping/dict, got {type(cfg)!r}")
    return _deep_merge(DEFAULT_HISTOST_CONFIG, cfg)