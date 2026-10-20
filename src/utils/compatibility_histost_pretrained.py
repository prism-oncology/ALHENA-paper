import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import h5py
import torch

from src.utils.constants import get_full_path
from src.utils.default_histost_config import cfg_get


def resolve_scconcept_embeddings_dir_from_cfg(cfg: Mapping[str, Any]) -> Optional[str]:
    if float(cfg_get(cfg, "model", "clip", "lambda_token")) <= 0.0:
        return None

    sc_dir = cfg_get(cfg, "model", "st", "scconcept_embeddings_dir")
    if sc_dir:
        return str(sc_dir)

    scconcept_training_id = cfg_get(cfg, "model", "st", "scconcept_training_id")
    if not scconcept_training_id:
        return None

    return get_full_path(
        "train",
        "trainings",
        "trainings_scConcept",
        scconcept_training_id,
        "embeddings",
        as_str=True,
    )


def infer_scconcept_embedding_dim_from_cfg(
    cfg: Mapping[str, Any],
    logger: Optional[logging.Logger] = None,
) -> Optional[int]:
    logger = logger or logging.getLogger(__name__)

    sc_dir = resolve_scconcept_embeddings_dir_from_cfg(cfg)
    if not sc_dir or not os.path.isdir(sc_dir):
        return None

    first_h5 = next(Path(sc_dir).glob("*.h5"), None)
    if first_h5 is None:
        return None

    try:
        with h5py.File(first_h5, "r") as f:
            if "cls_emb" not in f:
                return None
            return int(f.attrs.get("embedding_dim", f["cls_emb"].shape[1]))
    except Exception as exc:
        logger.warning("Could not infer scConcept embedding dim from %s: %s", first_h5, exc)
        return None


def effective_st_spec_from_cfg(
    cfg: Mapping[str, Any],
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """
    ST-side properties compared when reusing pretrained HistoST weights: st_feature_dim (st_proj input dim),
    novae_training_id and scconcept_training_id (teacher identity).
    """
    logger = logger or logging.getLogger(__name__)
    st_dim = cfg_get(cfg, "model", "st", "st_feature_dim", default=None)
    return {
        "st_feature_dim": int(st_dim) if st_dim is not None else None,
        "novae_training_id": cfg_get(cfg, "model", "st", "novae_training_id", default=None),
        "scconcept_training_id": cfg_get(cfg, "model", "st", "scconcept_training_id", default=None),
    }


def check_pretrained_histost_config_compatibility(
    current_config: Mapping[str, Any],
    checkpoint_config: Mapping[str, Any],
    checkpoint_path: str,
    logger: Optional[logging.Logger] = None,
    context_label: str = "model.pretrained_histost",
) -> None:
    logger = logger or logging.getLogger(__name__)

    mismatches: List[str] = []
    warnings_list: List[str] = []

    def add_mismatch(field: str, old: Any, new: Any, reason: str) -> None:
        if old != new:
            mismatches.append(f"  - {field}: checkpoint={old!r} | current={new!r}. {reason}")

    def add_warning(field: str, old: Any, new: Any, reason: str) -> None:
        if old != new:
            warnings_list.append(f"  - {field}: checkpoint={old!r} | current={new!r}. {reason}")

    # 1) H&E side
    old_he = checkpoint_config.get("model", {}).get("he", {})
    new_he = current_config.get("model", {}).get("he", {})

    add_mismatch(
        "model.he.backbone_img_encoder",
        str(old_he.get("backbone_img_encoder", None)).lower(),
        str(new_he.get("backbone_img_encoder", None)).lower(),
        "The H&E backbone must stay identical because it changes the image encoder weights and token interface.",
    )

    old_use_mlp = bool(cfg_get(checkpoint_config, "model", "he", "use_mlp_head"))
    new_use_mlp = bool(cfg_get(current_config, "model", "he", "use_mlp_head"))
    add_mismatch(
        "model.he.use_mlp_head",
        old_use_mlp,
        new_use_mlp,
        "Adding or removing the H&E MLP head changes the instantiated modules and parameter shapes.",
    )
    if old_use_mlp and new_use_mlp:
        add_mismatch(
            "model.he.he_mlp_hidden_dim",
            int(cfg_get(checkpoint_config, "model", "he", "he_mlp_hidden_dim")),
            int(cfg_get(current_config, "model", "he", "he_mlp_hidden_dim")),
            "The H&E MLP hidden size must match exactly.",
        )
        add_mismatch(
            "model.he.he_mlp_out_dim",
            int(cfg_get(checkpoint_config, "model", "he", "he_mlp_out_dim")),
            int(cfg_get(current_config, "model", "he", "he_mlp_out_dim")),
            "The H&E MLP output size must match exactly because it changes the downstream CLIP input dimension.",
        )

    old_he_training_option = str(cfg_get(checkpoint_config, "training", "HE_training_option")).lower()
    new_he_training_option = str(cfg_get(current_config, "training", "HE_training_option")).lower()
    old_use_lora = old_he_training_option == "lora"
    new_use_lora = new_he_training_option == "lora"

    if old_use_lora != new_use_lora:
        add_mismatch(
            "training.HE_training_option (LoRA presence)",
            old_he_training_option,
            new_he_training_option,
            "Switching LoRA on/off is not allowed when reloading a previous HistoST checkpoint. "
            "Changing freeze<->train is fine, but any transition involving 'lora' changes the module structure.",
        )

    if old_use_lora and new_use_lora:
        add_mismatch(
            "training.lora_rank",
            int(cfg_get(checkpoint_config, "training", "lora_rank")),
            int(cfg_get(current_config, "training", "lora_rank")),
            "LoRA rank changes adapter parameter shapes.",
        )
        add_mismatch(
            "training.lora_alpha",
            float(cfg_get(checkpoint_config, "training", "lora_alpha")),
            float(cfg_get(current_config, "training", "lora_alpha")),
            "LoRA alpha should remain identical when reloading trained adapters.",
        )

    # 2) ST side
    old_st = effective_st_spec_from_cfg(checkpoint_config, logger=logger)
    new_st = effective_st_spec_from_cfg(current_config, logger=logger)

    # st_feature_dim sets the st_proj input dim
    add_mismatch(
        "model.st.st_feature_dim",
        old_st["st_feature_dim"],
        new_st["st_feature_dim"],
        "The ST teacher output dimensionality (st_feat dim) must match because it changes the st_proj shape.",
    )

    # 3) CLIP bridge
    old_clip_embed_dim = int(cfg_get(checkpoint_config, "model", "clip", "clip_embed_dim"))
    new_clip_embed_dim = int(cfg_get(current_config, "model", "clip", "clip_embed_dim"))
    add_mismatch(
        "model.clip.clip_embed_dim",
        old_clip_embed_dim,
        new_clip_embed_dim,
        "The patch-level CLIP embedding dimension must match because it changes the projection head shapes.",
    )

    old_token_enabled = float(cfg_get(checkpoint_config, "model", "clip", "lambda_token")) > 0.0
    new_token_enabled = float(cfg_get(current_config, "model", "clip", "lambda_token")) > 0.0
    add_mismatch(
        "model.clip.token_branch_enabled (derived from lambda_token > 0)",
        old_token_enabled,
        new_token_enabled,
        "Adding or removing the token-level CLIP branch changes the instantiated projection heads.",
    )

    if old_token_enabled and new_token_enabled:
        old_token_embed_dim = cfg_get(checkpoint_config, "model", "clip", "token_embed_dim")
        new_token_embed_dim = cfg_get(current_config, "model", "clip", "token_embed_dim")
        old_token_embed_dim = int(old_token_embed_dim if old_token_embed_dim is not None else old_clip_embed_dim)
        new_token_embed_dim = int(new_token_embed_dim if new_token_embed_dim is not None else new_clip_embed_dim)

        add_mismatch(
            "model.clip.token_embed_dim_effective",
            old_token_embed_dim,
            new_token_embed_dim,
            "The token-level CLIP embedding dimension must match exactly.",
        )

        old_sc_dim = infer_scconcept_embedding_dim_from_cfg(checkpoint_config, logger=logger)
        new_sc_dim = infer_scconcept_embedding_dim_from_cfg(current_config, logger=logger)
        if old_sc_dim is not None and new_sc_dim is not None:
            add_mismatch(
                "model.st.scconcept_embedding_dim",
                old_sc_dim,
                new_sc_dim,
                "The scConcept input dimension must match because it changes the token-level ST projection shape.",
            )

        add_mismatch(
            "model.st.scconcept_training_id",
            cfg_get(checkpoint_config, "model", "st", "scconcept_training_id"),
            cfg_get(current_config, "model", "st", "scconcept_training_id"),
            "The scConcept training ID must match."
        )

    add_warning(
        "model.st.novae_training_id",
        old_st["novae_training_id"],
        new_st["novae_training_id"],
        "The precomputed Novae ST teacher differs: the checkpoint loads (same st_feat dim) but was aligned to a "
        "different teacher.",
    )

    if mismatches:
        msg = [
            f"Incompatible config for {context_label}='{checkpoint_path}'.",
            "The following fields must match exactly when loading a previous HistoST checkpoint:",
            *mismatches,
            "",
            "Allowed changes include: logging, seed, device, dataset split/folds, caching, sampling, "
            "optimizer/scheduler, early stopping, freeze schedules, augmentations, and CLIP loss weights/types.",
        ]
        if warnings_list:
            msg.extend([
                "",
                "Potentially risky but still loadable differences that were detected:",
                *warnings_list,
            ])
        raise ValueError("\n".join(msg))

    logger.info("Current config is compatible with the pretrained HistoST checkpoint.")
    if warnings_list:
        logger.warning(
            "Potentially risky but still loadable differences detected with %s:\n%s",
            context_label,
            "\n".join(warnings_list),
        )


def assert_pretrained_state_dict_compatible(
    model: torch.nn.Module,
    checkpoint_state_dict: Mapping[str, torch.Tensor],
    checkpoint_path: str,
) -> None:
    model_sd = model.state_dict()

    model_keys = set(model_sd.keys())
    ckpt_keys = set(checkpoint_state_dict.keys())

    missing = sorted(model_keys - ckpt_keys)
    unexpected = sorted(ckpt_keys - model_keys)
    shape_mismatches: List[str] = []

    for key in sorted(model_keys & ckpt_keys):
        if tuple(model_sd[key].shape) != tuple(checkpoint_state_dict[key].shape):
            shape_mismatches.append(
                f"  - {key}: checkpoint_shape={tuple(checkpoint_state_dict[key].shape)} | "
                f"current_shape={tuple(model_sd[key].shape)}"
            )

    if missing or unexpected or shape_mismatches:
        msg = [
            f"Pretrained HistoST weights are incompatible with the instantiated model: '{checkpoint_path}'.",
            "This usually means that an architecture-defining option changed or the code evolved since the checkpoint was created.",
        ]
        if missing:
            msg.extend([
                "",
                f"Missing keys in checkpoint ({len(missing)}):",
                *[f"  - {k}" for k in missing[:50]],
            ])
            if len(missing) > 50:
                msg.append(f"  ... and {len(missing) - 50} more")
        if unexpected:
            msg.extend([
                "",
                f"Unexpected keys in checkpoint ({len(unexpected)}):",
                *[f"  - {k}" for k in unexpected[:50]],
            ])
            if len(unexpected) > 50:
                msg.append(f"  ... and {len(unexpected) - 50} more")
        if shape_mismatches:
            msg.extend([
                "",
                f"Tensor shape mismatches ({len(shape_mismatches)}):",
                *shape_mismatches[:50],
            ])
            if len(shape_mismatches) > 50:
                msg.append(f"  ... and {len(shape_mismatches) - 50} more")
        raise ValueError("\n".join(msg))


def check_cross_run_resume_compatibility(
    new_config: Mapping[str, Any],
    source_config: Mapping[str, Any],
    checkpoint_path: str,
    source_step_per_epoch: Optional[float] = None,
    new_step_per_epoch: Optional[int] = None,
    allow_schedule_mismatch: bool = False,
    logger: Optional[logging.Logger] = None,
) -> None:
    """
    Check that a new run's config can resume the full training state (weights, optimizer, scheduler, counters) of a
    checkpoint from another run (--resume_state_from_checkpoint). Raises on architecture, optimizer group layout or
    schedule mismatches (steps/epoch: unless allow_schedule_mismatch); warns on loadable differences.
    """
    logger = logger or logging.getLogger(__name__)

    # 1) Architecture check (same as model.pretrained_histost)
    check_pretrained_histost_config_compatibility(
        current_config=new_config,
        checkpoint_config=source_config,
        checkpoint_path=checkpoint_path,
        logger=logger,
        context_label="cross-run resume (--resume_state_from_checkpoint)",
    )

    # 2) Constraints specific to a full-state resume
    hard: List[str] = []

    def _hard(field: str, getter) -> None:
        old = getter(source_config)
        new = getter(new_config)
        if old != new:
            hard.append(f"  - {field}: checkpoint={old!r} | current={new!r}")

    _hard("advanced_scheduling.enabled",
          lambda c: bool(cfg_get(c, "advanced_scheduling", "enabled")))
    _hard("training.scheduler.scheduler_type",
          lambda c: str(cfg_get(c, "training", "scheduler", "scheduler_type")))
    _hard("training.epochs",
          lambda c: int(cfg_get(c, "training", "epochs")))
    _hard("training.batch_size",
          lambda c: int(cfg_get(c, "training", "batch_size")))
    _hard("data_select.num_train_samples",
          lambda c: int(cfg_get(c, "data_select", "num_train_samples")))
    _hard("training.scheduler.warmup_steps",
          lambda c: int(cfg_get(c, "training", "scheduler", "warmup_steps")))

    if hard:
        raise ValueError("\n".join([
            f"Incompatible config for cross-run resume (--resume_state_from_checkpoint)='{checkpoint_path}'.",
            "These fields define the restored optimizer/scheduler state shape and must match the source run:",
            *hard,
            "",
            "To change any of these, start a fresh run instead (optionally loading the weights only via "
            "model.pretrained_histost).",
        ]))

    # 3) Effective steps per epoch (5% tolerance, as in Experiment.run_experiment)
    if source_step_per_epoch is None or new_step_per_epoch is None or source_step_per_epoch <= 0:
        logger.warning(
            "Cross-run resume: could not derive the source steps-per-epoch (source=%r, new=%r); "
            "skipping the effective-schedule check. Ensure the new run uses the same effective batch "
            "(GPU count x batch_size) as the source.",
            source_step_per_epoch, new_step_per_epoch,
        )
    else:
        rel = abs(source_step_per_epoch - new_step_per_epoch) / max(1, new_step_per_epoch)
        if rel > 0.05:
            msg = (
                f"Cross-run resume schedule mismatch: source steps/epoch ~= {source_step_per_epoch:.1f} "
                f"but this run computes {new_step_per_epoch} (relative diff {rel:.0%}): different effective batch "
                "(GPU count x batch_size) or num_train_samples than the source run."
            )
            if allow_schedule_mismatch:
                logger.warning(msg + " Proceeding because --allow_schedule_mismatch was set.")
            else:
                raise ValueError(
                    msg + " Re-launch with the same effective batch, or pass --allow_schedule_mismatch to override."
                )

    # 4) Warnings: loadable differences
    soft: List[str] = []

    def _soft(field: str, getter, note: str) -> None:
        old = getter(source_config)
        new = getter(new_config)
        if old != new:
            soft.append(f"  - {field}: checkpoint={old!r} | current={new!r}. {note}")

    # default=None: keys without a registered default give None instead of raising
    _soft("seed", lambda c: cfg_get(c, "seed", default=None),
          "Different seed: the data order after the resume point differs from the source run.")
    _soft("data.training_dataset_id", lambda c: cfg_get(c, "data", "training_dataset_id", default=None),
          "Metrics before and after the resume point are not comparable (different dataset).")
    _soft("data.dataset_id", lambda c: cfg_get(c, "data", "dataset_id", default=None),
          "Metrics before and after the resume point are not comparable (different dataset).")
    _soft("training.early_stopping.monitor", lambda c: cfg_get(c, "training", "early_stopping", "monitor", default=None),
          "Early-stopping monitor/mode are restored from the checkpoint; the new YAML value is ignored on resume.")
    _soft("training.early_stopping.mode", lambda c: cfg_get(c, "training", "early_stopping", "mode", default=None),
          "Early-stopping monitor/mode are restored from the checkpoint; the new YAML value is ignored on resume.")
    _soft("training.optimizer_hyperparameters.lr",
          lambda c: cfg_get(c, "training", "optimizer_hyperparameters", "lr", default=None),
          "Optimizer LR/wd and scheduler base_lrs are restored from the checkpoint; new-YAML LR changes do not "
          "take effect on resume.")
    _soft("advanced_scheduling.weight_decay_lora",
          lambda c: cfg_get(c, "advanced_scheduling", "weight_decay_lora", default=None),
          "LoRA weight decay lives in the restored optimizer param_groups; the new-YAML value does not take "
          "effect on resume (start a fresh run to change it).")
    # Per-block optimizer keys (not registered defaults, hence default=None)
    _soft("advanced_scheduling.lr_lora", lambda c: cfg_get(c, "advanced_scheduling", "lr_lora", default=None),
          "Per-block LRs live in the restored optimizer param_groups / scheduler base_lrs; the new-YAML value "
          "does not take effect on resume (start a fresh run to change it).")
    _soft("advanced_scheduling.lr_logit", lambda c: cfg_get(c, "advanced_scheduling", "lr_logit", default=None),
          "Per-block LRs live in the restored optimizer param_groups / scheduler base_lrs; the new-YAML value "
          "does not take effect on resume (start a fresh run to change it).")
    _soft("advanced_scheduling.weight_decay_heads",
          lambda c: cfg_get(c, "advanced_scheduling", "weight_decay_heads", default=None),
          "Head weight decay lives in the restored optimizer param_groups; the new-YAML value does not take "
          "effect on resume (start a fresh run to change it).")
    _soft("advanced_scheduling.weight_decay_other",
          lambda c: cfg_get(c, "advanced_scheduling", "weight_decay_other", default=None),
          "The 'other' weight decay lives in the restored optimizer param_groups; the new-YAML value does not "
          "take effect on resume (start a fresh run to change it). This group is expected to be empty.")
    # lr_heads is re-applied from the new config when the head schedule un-freezes the heads
    _soft("advanced_scheduling.lr_heads", lambda c: cfg_get(c, "advanced_scheduling", "lr_heads", default=None),
          "Head LR lives in the restored optimizer param_groups; the new-YAML value does not take effect on "
          "resume, unless the head schedule un-freezes the heads (clip_head_warmup_epoch > the resume epoch).")

    if soft:
        logger.warning(
            "Cross-run resume: loadable differences that change the continuation (not blocking):\n%s\n"
            "Only per-epoch-read keys (e.g. advanced_scheduling.clip_head_warmup_epoch, "
            "model.clip.lambda_token_warmup_epochs) take effect from the resume point; the optimizer values "
            "in force are recorded in checkpoint['effective_optimizer'].",
            "\n".join(soft),
        )
    else:
        logger.info("Cross-run resume: new config is compatible with the source checkpoint (no soft warnings).")