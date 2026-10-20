import logging
import contextlib
# Get the logger for this module
logger = logging.getLogger(__name__)

import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import gc
import numpy as np
import random
import torch.distributed as dist
import json
import math
from typing import Optional

from src.utils.utils import is_main_process, reconcile_state_dict, get_dist_rank, assert_finite
from src.training.metrics import collect_step_counts, global_metrics_computation, within_slide_metrics_computation
from src.models.clip import pool_scconcept_per_patch
from src.training.samplers import _normalize_batch_slide_mode
from src.utils.default_histost_config import cfg_get


def collapse_wandb_key(key: str) -> str:
    """
    Collapse a W&B metric path to one section: prefixes joined with '__', last token kept as metric name.
    Example: "a/b/c" -> "a__b/c"
    """
    parts = [p for p in key.split("/") if p]  # drop empty tokens defensively
    if len(parts) >= 2:
        section = "__".join(parts[:-1])
        return f"{section}/{parts[-1]}"
    return key


# ---------------------------------------------------------------------------------------------
# Optimizer provenance (read-only)
# ---------------------------------------------------------------------------------------------

_EFFECTIVE_OPTIMIZER_SCHEMA_VERSION = 1

# CLIP-head param groups (frozen by maybe_apply_clip_head_schedule after warmup)
HEAD_GROUP_NAMES = ("heads_decay", "heads_no_decay", "heads")

# (config key path, param group, effective field) compared by collect_effective_optimizer.
# Keys absent from the config are reported under "not_compared".
_ADV_SCHED_CHECKS = (
    (("advanced_scheduling", "lr_lora"),            "lora",           "base_lr"),
    (("advanced_scheduling", "lr_heads"),           "heads_decay",    "base_lr"),
    (("advanced_scheduling", "lr_heads"),           "heads_no_decay", "base_lr"),
    (("advanced_scheduling", "lr_logit"),           "logit",          "base_lr"),
    (("advanced_scheduling", "weight_decay_lora"),  "lora",           "weight_decay"),
    (("advanced_scheduling", "weight_decay_heads"), "heads_decay",    "weight_decay"),
    (("advanced_scheduling", "weight_decay_other"), "other_decay",    "weight_decay"),
)


def _same_scalar(a, b) -> bool:
    """True if a configured and an effective scalar agree (rel_tol absorbs float noise only)."""
    try:
        return math.isclose(float(a), float(b), rel_tol=1e-12, abs_tol=0.0)
    except (TypeError, ValueError):
        return a == b


def _as_betas(v) -> tuple:
    """Normalize AdamW betas to a tuple of floats (the config may store them as a list)."""
    try:
        return tuple(float(x) for x in v)
    except (TypeError, ValueError):
        return ()


def _betas_equal(a: tuple, b: tuple) -> bool:
    return len(a) == len(b) and all(_same_scalar(x, y) for x, y in zip(a, b))


def collect_effective_optimizer(optimizer, scheduler, config, heads_frozen=None, restored_from=None) -> dict:
    """
    Describe the optimizer hyperparameters in force and compare them with the config (read-only).
    Returns a dict saved in checkpoint["effective_optimizer"], listing "divergences" (config values not in
    force, e.g. after a resume) and "inert_config_keys" (config values applied to an empty param group).
    """
    adv_enabled = bool(cfg_get(config, "advanced_scheduling", "enabled"))

    groups: dict = {}
    empty_groups: list[str] = []

    for gi, g in enumerate(optimizer.param_groups):
        # Unnamed groups (advanced_scheduling disabled) fall back to their index
        name = g.get("group_name", None) or f"group_{gi}"

        # Base lr (schedule-invariant), from scheduler.base_lrs when available
        base_lr, base_lr_source = None, None
        if scheduler is not None and hasattr(scheduler, "base_lrs"):
            base_lrs = scheduler.base_lrs
            if base_lrs is not None and gi < len(base_lrs):
                base_lr, base_lr_source = float(base_lrs[gi]), "scheduler.base_lrs"
        if base_lr is None:
            base_lr, base_lr_source = float(g.get("lr", 0.0)), "param_group.lr"

        params = g.get("params", []) or []
        if len(params) == 0:
            empty_groups.append(name)

        groups[name] = {
            "index": gi,
            "n_params": len(params),
            "n_trainable": sum(1 for p in params if getattr(p, "requires_grad", False)),
            "n_elements": sum(int(p.numel()) for p in params),
            "lr": float(g.get("lr", 0.0)),   # live / mid-schedule -> informational only
            "base_lr": base_lr,              # the comparable quantity
            "base_lr_source": base_lr_source,
            "weight_decay": float(g.get("weight_decay", 0.0)),
            "betas": list(_as_betas(g.get("betas", ()))),
            "eps": float(g.get("eps", 0.0)),
            "lr_is_zero": (base_lr == 0.0),
        }

    config_intent: dict = {}
    not_compared: list[str] = []
    _not_compared_keys: set = set()
    divergences: list[str] = []
    inert_config_keys: list[str] = []

    if adv_enabled:
        for key_path, group_name, field in _ADV_SCHED_CHECKS:
            key = ".".join(key_path)
            rec = groups.get(group_name)
            if rec is None:
                continue  # group absent

            cfg_val = cfg_get(config, *key_path, default=None)
            if cfg_val is None:
                if key not in _not_compared_keys:
                    _not_compared_keys.add(key)
                    not_compared.append(
                        f"{key}: absent from config (Experiment.get_optimizer applies its own fallback)"
                    )
                continue
            config_intent[key] = cfg_val

            # Empty group: a non-zero configured value has no effect (reported as inert)
            if rec["n_params"] == 0:
                if not _same_scalar(cfg_val, 0.0):
                    inert_config_keys.append(
                        f"{key} (group={group_name}): config={cfg_val!r} but the group holds 0 "
                        f"parameters -- no effect on this run"
                    )
                continue

            # Head groups at base_lr=0 are frozen after warmup (not a divergence)
            if field == "base_lr" and group_name in HEAD_GROUP_NAMES and rec["base_lr"] == 0.0:
                continue

            if not _same_scalar(cfg_val, rec[field]):
                divergences.append(
                    f"{key} (group={group_name}): config={cfg_val!r} effective={rec[field]!r}"
                )
    else:
        # advanced_scheduling disabled: unnamed groups, compared without relying on their order
        cfg_lr = cfg_get(config, "training", "optimizer_hyperparameters", "lr", default=None)
        if cfg_lr is not None:
            config_intent["training.optimizer_hyperparameters.lr"] = cfg_lr
            # lr is set in the constructor, so all groups must match
            for name, rec in groups.items():
                if rec["n_params"] == 0:
                    continue
                if not _same_scalar(cfg_lr, rec["base_lr"]):
                    divergences.append(
                        f"training.optimizer_hyperparameters.lr (group={name}): "
                        f"config={cfg_lr!r} effective={rec['base_lr']!r}"
                    )

        cfg_wd = cfg_get(config, "training", "optimizer_hyperparameters", "weight_decay", default=None)
        if cfg_wd is not None and groups:
            config_intent["training.optimizer_hyperparameters.weight_decay"] = cfg_wd
            # The decayed group carries the configured weight decay; the other one has 0
            eff_wd = max(rec["weight_decay"] for rec in groups.values())
            if not _same_scalar(cfg_wd, eff_wd):
                divergences.append(
                    f"training.optimizer_hyperparameters.weight_decay (max over groups): "
                    f"config={cfg_wd!r} effective={eff_wd!r}"
                )

    # betas / eps are constructor-level in both optimizer paths and land on every group.
    cfg_betas = cfg_get(config, "training", "optimizer_hyperparameters", "betas", default=None)
    if cfg_betas is not None:
        want = _as_betas(cfg_betas)
        config_intent["training.optimizer_hyperparameters.betas"] = list(want)
        bad = [
            f"{name}={rec['betas']!r}"
            for name, rec in groups.items()
            if not _betas_equal(want, _as_betas(rec["betas"]))
        ]
        if bad:
            divergences.append(
                f"training.optimizer_hyperparameters.betas: config={list(want)!r} "
                f"effective={', '.join(bad)}"
            )

    cfg_eps = cfg_get(config, "training", "optimizer_hyperparameters", "eps", default=None)
    if cfg_eps is not None:
        config_intent["training.optimizer_hyperparameters.eps"] = cfg_eps
        bad = [
            f"{name}={rec['eps']!r}" for name, rec in groups.items()
            if not _same_scalar(cfg_eps, rec["eps"])
        ]
        if bad:
            divergences.append(
                f"training.optimizer_hyperparameters.eps: config={cfg_eps!r} "
                f"effective={', '.join(bad)}"
            )

    return {
        "schema_version": _EFFECTIVE_OPTIMIZER_SCHEMA_VERSION,
        "advanced_scheduling_enabled": adv_enabled,
        "optimizer_class": type(optimizer).__name__,
        "scheduler_class": (type(scheduler).__name__ if scheduler is not None else None),
        "optimizer_state_restored_from": (str(restored_from) if restored_from else None),
        "heads_frozen": heads_frozen,
        "empty_groups": empty_groups,
        "groups": groups,
        # Config values that were compared
        "config_intent": config_intent,
        "not_compared": not_compared,
        "divergences": divergences,
        # Config keys applied to a param group with no parameters
        "inert_config_keys": inert_config_keys,
    }


class Trainer:

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        scheduler=None,
        use_amp: bool = False,
        amp_dtype: str = "bf16",
    ):
        """
        Args:
            model: The CLIP model that wraps the image and ST encoders.
            optimizer: Optimizer for training.
            device: Device to run the training (cuda or cpu or mps)
            scheduler: (Optional) Learning rate scheduler.
            use_amp: (Optional) Whether to use mixed precision training (AMP).
            amp_dtype: (Optional) Data type for AMP, default is "bf16".
        """
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.use_amp = use_amp

        # Initialize AMP GradScaler if mixed precision is enabled
        if use_amp:
            if amp_dtype.lower() == "fp16":
                self.amp_dtype = torch.float16
                self.scaler = GradScaler(init_scale=2**14)   # fp16 needs scaling / set to 2**14 to avoid overflow when large batch size
                logger.info("Using mixed precision training with fp16 => GradScaler initialized.")
            elif amp_dtype.lower() == "bf16":
                self.amp_dtype = torch.bfloat16
                self.scaler = None                           # bf16 is scale-free
                logger.info("Using mixed precision training with bf16 => no GradScaler needed.")
            else:
                raise ValueError(f"amp_dtype must be 'fp16' or 'bf16', got '{amp_dtype}'.")
        else:
            self.amp_dtype = torch.float32                   # plain FP32 path
            self.scaler = None
            logger.info("Using plain FP32 training => no GradScaler needed.")

        # Early stopping state
        self.best_early_stopping_value: Optional[float] = None
        self.best_early_stopping_epoch: Optional[int] = None
        self.patience_counter = 0
        self.early_stopping_patience: Optional[int] = None
        self.early_stopping_monitor: Optional[str] = None
        self.early_stopping_mode: Optional[str] = None

        self.global_step = 0  # Keep track of how many iterations so far

        # Lambda weights (patch/token); updated epoch-wise in fit()
        self.current_lambda_patch: float | None = None
        self.current_lambda_token: float | None = None

        # Advanced scheduling state (heads warmup -> freeze)
        self._clip_heads_frozen: Optional[bool] = None

        # Checkpoint the optimizer state was restored from (None on a fresh run)
        self._optimizer_state_restored_from: Optional[str] = None



    def _fmt_metric_value(self, v):
        """
        Robust string formatter for metrics:
          - float/int -> '%.4f'
          - None      -> 'None'
          - anything else -> str(v)
        """
        if v is None:
            return "None"
        if isinstance(v, (int, float)):
            return f"{v:.4f}"
        return str(v)



    def _validate_and_init_early_stopping(self, train_mode_cfg: str, val_mode_cfg: str) -> None:
        """
        Validate early-stopping config and initialize trainer-side state.

        Supported monitors:
          - "val_loss" (special case)
          - any validation metric key produced in val_metrics, e.g.
              "val_patch_loss", "val_acc_i2st", "val_global_acc_i2st", ...
        """
        es_cfg = cfg_get(self.config, "training", "early_stopping")
        if not isinstance(es_cfg, dict):
            raise ValueError("training.early_stopping must be a mapping with keys: patience, monitor, mode.")

        patience = es_cfg.get("patience", None)
        monitor = str(es_cfg.get("monitor", "val_loss")).strip()
        mode = str(es_cfg.get("mode", "min")).strip().lower()

        if patience is not None:
            try:
                patience = int(patience)
            except (TypeError, ValueError):
                raise ValueError("training.early_stopping.patience must be an integer or null.")
            if patience < 0:
                raise ValueError("training.early_stopping.patience must be >= 0 or null.")

        if not monitor:
            raise ValueError("training.early_stopping.monitor must be a non-empty string.")

        if mode not in ("min", "max"):
            raise ValueError("training.early_stopping.mode must be either 'min' or 'max'.")

        if monitor != "val_loss" and not monitor.startswith("val_"):
            raise ValueError("training.early_stopping.monitor must be 'val_loss' or a validation metric key starting with 'val_'.")

        unstable_val_protocol = (
            (val_mode_cfg == "alternate") or
            (val_mode_cfg == "inherit" and train_mode_cfg == "alternate")
        )
        if patience is not None and monitor == "val_loss" and unstable_val_protocol:
            raise ValueError(
                "Early stopping is based on val_loss, so validation step protocol must be fixed "
                "across epochs. Use training.val_step_slide_mode='mixed' or 'single_slide' when "
                "training.batch_slide_mode='alternate'."
            )

        self.early_stopping_patience = patience
        self.early_stopping_monitor = monitor
        self.early_stopping_mode = mode




    def _get_early_stopping_value(self, val_loss, val_metrics: dict | None) -> float:
        """
        Resolve the configured monitored quantity for early stopping.
        """
        if self.early_stopping_monitor is None:
            raise RuntimeError("Early stopping monitor is not initialized.")

        if self.early_stopping_monitor == "val_loss":
            value = val_loss
        else:
            value = None if val_metrics is None else val_metrics.get(self.early_stopping_monitor, None)

        if value is None:
            available = []
            if isinstance(val_metrics, dict):
                available = sorted(k for k, v in val_metrics.items() if v is not None)
            preview = ", ".join(available[:20])
            more = "" if len(available) <= 20 else f" ... (+{len(available) - 20} more)"
            raise ValueError(
                f"Configured early_stopping.monitor='{self.early_stopping_monitor}' is unavailable "
                "for this validation epoch. The monitored quantity must be computed on every validation pass. "
                "For example, some metrics such as CKA may be absent when training.cka_every does not match "
                f"the current validation epoch. Available non-None validation metrics: {preview}{more}"
            )

        value = float(value)
        if not math.isfinite(value):
            raise ValueError(f"Configured early stopping value for monitor='{self.early_stopping_monitor}' is not finite: {value}")
        return value



    def _is_early_stopping_improvement(self, current_value: float) -> bool:
        if self.best_early_stopping_value is None:
            return True
        if self.early_stopping_mode == "max":
            return current_value > self.best_early_stopping_value
        return current_value < self.best_early_stopping_value



    def _adv_enabled(self) -> bool:
        return bool(cfg_get(self.config, "advanced_scheduling", "enabled"))



    def _set_group_lr(self, group_name, new_lr: float) -> None:
        """
        Set optimizer group LR by group_name, and also update scheduler.base_lrs
        so LR schedulers (cosine/warmup LambdaLR) do not resurrect frozen groups.
        """
        if self.optimizer is None:
            return
        if isinstance(group_name, (list, tuple, set)):
            names = set(group_name)
        else:
            names = {group_name}
        for gi, g in enumerate(self.optimizer.param_groups):
            if g.get("group_name", None) not in names:
                continue
            g["lr"] = float(new_lr)
            if self.scheduler is not None and hasattr(self.scheduler, "base_lrs"):
                # base_lrs aligns with optimizer.param_groups order
                try:
                    self.scheduler.base_lrs[gi] = float(new_lr)
                except Exception:
                    pass

    def _get_group_lrs(self, group_names) -> list[float]:
        if self.optimizer is None:
            return []
        if isinstance(group_names, (list, tuple, set)):
            names = set(group_names)
        else:
            names = {group_names}
        out = []
        for g in self.optimizer.param_groups:
            if g.get("group_name", None) in names:
                out.append(float(g.get("lr", 0.0)))
        return out

    def _validate_advanced_scheduling_runtime(self) -> None:
        """
        Validate the advanced_scheduling configuration (same checks as in Experiment).
        """
        if not self._adv_enabled():
            return

        cfg = self.config
        warm = int(cfg_get(cfg, "advanced_scheduling", "clip_head_warmup_epoch"))
        if warm < 0:
            raise ValueError("advanced_scheduling.clip_head_warmup_epoch must be >= 0")

        if bool(cfg_get(cfg, "model", "he", "use_mlp_head")):
            raise ValueError("advanced_scheduling requires model.he.use_mlp_head=False")

        if str(cfg_get(cfg, "training", "HE_training_option")).lower() != "lora":
            raise ValueError("advanced_scheduling requires training.HE_training_option='lora'")

        # The ST-graph teacher is the precomputed st_feat cache (always frozen); require it.
        if not cfg_get(cfg, "model", "st", "novae_teacher_cache_dir"):
            raise ValueError("advanced_scheduling requires the precomputed Novae-teacher cache (model.st.novae_training_id).")
        # model.st.novae_unfreeze_st_epoch is not supported (the ST teacher is precomputed)
        _legacy_unfreeze = (cfg.get("model", {}).get("st", {}) or {}).get("novae_unfreeze_st_epoch")
        if _legacy_unfreeze is not None and int(_legacy_unfreeze) >= 0:
            raise ValueError(
                "ST-teacher unfreeze is unsupported with the precomputed Novae teacher "
                "(remove model.st.novae_unfreeze_st_epoch)."
            )

        # Ensure the optimizer has named param groups
        has_named = any(("group_name" in g) for g in self.optimizer.param_groups)
        if not has_named:
            raise ValueError(
                "advanced_scheduling is enabled but optimizer has no named param groups. "
                "Ensure Experiment.get_optimizer built groups with group_name fields."
            )

    def _log_optimizer_provenance(self) -> None:
        """
        Log once at startup (rank 0) the optimizer config values that are not in force (after a resume)
        or that apply to an empty param group.
        """
        # Log on rank 0 only
        if not is_main_process():
            return

        eff = collect_effective_optimizer(
            self.optimizer, self.scheduler, self.config,
            heads_frozen=self._clip_heads_frozen,
            restored_from=self._optimizer_state_restored_from,
        )
        inert = eff.get("inert_config_keys", [])
        if inert:
            logger.warning(
                "Optimizer: %d config key(s) apply to a param group with no parameters (no effect "
                "on this run):\n%s\nSet them to 0.0 or remove them.",
                len(inert),
                "\n".join(f"  - {k}" for k in inert),
            )

        divergences = eff.get("divergences", [])
        if not divergences:
            return

        logger.warning(
            "Optimizer: %d config key(s) are not in force (optimizer state restored from %s):\n%s\n"
            "Expected when resuming the same run; for a cross-run resume the new config values are not applied.",
            len(divergences),
            eff.get("optimizer_state_restored_from"),
            "\n".join(f"  - {d}" for d in divergences),
        )

    def maybe_apply_clip_head_schedule(self, epoch: int) -> None:
        """
        Two-stage schedule:
          Stage A (epoch < warmup): heads trainable (lr=lr_heads)
          Stage B (epoch >= warmup): freeze heads (requires_grad=False) + lr=0

        Heads include:
          - image_proj, st_proj
          - img_tok_proj, st_node_proj (if present)
        """
        if not self._adv_enabled():
            return
        adv = cfg_get(self.config, "advanced_scheduling")
        warmup_epoch = int(cfg_get(self.config, "advanced_scheduling", "clip_head_warmup_epoch"))
        lr_heads = float(adv.get("lr_heads", float(cfg_get(self.config, "training", "optimizer_hyperparameters", "lr"))))
        head_groups = HEAD_GROUP_NAMES

        freeze = bool(epoch >= warmup_epoch)
        if self._clip_heads_frozen is not None and self._clip_heads_frozen == freeze:
            # already applied; avoid repeated parameter iteration/log spam
            return

        backbone = self.model.module if hasattr(self.model, "module") else self.model

        # Toggle requires_grad on head params
        head_prefixes = ("image_proj.", "st_proj.", "img_tok_proj.", "st_node_proj.")
        affected = 0
        for n, p in backbone.named_parameters():
            if n.startswith(head_prefixes):
                p.requires_grad = (not freeze)
                affected += p.numel()

        # Set head LR to 0 after warmup
        if freeze:
            self._set_group_lr(head_groups, 0.0)
            logger.info(
                f"[epoch {epoch}] Advanced scheduling: freezing CLIP heads after warmup "
                f"(warmup_epoch={warmup_epoch}). Affected params={affected:,}. Heads lr forced to 0."
            )
        else:
            # Don't overwrite LR on resume if scheduler already set a nonzero LR.
            # Only "restore" if heads are currently at lr=0 (i.e., they were frozen).
            cur = self._get_group_lrs(head_groups)
            need_restore = (len(cur) > 0) and (max(cur) == 0.0)
            if need_restore:
                self._set_group_lr(head_groups, lr_heads)
                msg = f"Heads lr restored to {lr_heads} (were 0)."
            else:
                msg = "Heads lr left untouched (resume-safe)."
            logger.info(
                f"[epoch {epoch}] Advanced scheduling: heads trainable during warmup "
                f"(warmup_epoch={warmup_epoch}). Affected params={affected:,}. {msg}"
            )

        self._clip_heads_frozen = freeze




    def train_step(self, batch):
        """
        Performs one training step.
        """
        self.model.train()
        optimizer_step_executed = True  # will be False if AMP over-flow skips the step
        batch = batch.to(self.device, non_blocking=True)

        # Training -----------
        self.optimizer.zero_grad(set_to_none=True)

        if self.use_amp: # - Mixed precision training -
            
            if self.device.type != "cuda":
                raise ValueError("Mixed precision training is only supported on CUDA devices.")
            with autocast("cuda", dtype=self.amp_dtype):
                (loss, logits_i2st, logits_st2i, labels,
                 img_embed, st_embed, logit_scale, logit_scale_token,
                 img_global, st_global,
                 patch_loss, token_loss, token_metrics) = self.model(batch)
                assert_finite(img_embed, st_embed, where="embeddings")

            if self.scaler is not None:          # ─ fp16 ─
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0, norm_type=2.0)
                scale_before = self.scaler.get_scale()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                optimizer_step_executed = (self.scaler.get_scale() >= scale_before) # overflow => scale decreases; valid step => scale same or grows
            else:                                # ─ bf16 ─
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0, norm_type=2.0)
                self.optimizer.step()
                optimizer_step_executed = True

            if self.scheduler is not None and optimizer_step_executed:  # advance LR once per real optimizer step
                self.scheduler.step()       
                
        else:           # - Plain FP32 training -
            (loss, logits_i2st, logits_st2i, labels,
             img_embed, st_embed, logit_scale, logit_scale_token,
             img_global, st_global,
             patch_loss, token_loss, token_metrics) = self.model(batch)
            assert_finite(img_embed, st_embed, where="embeddings")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0, norm_type=2.0)
            self.optimizer.step()   # always executed in FP32
            if self.scheduler is not None:  # FP32 path has no overflow logic
                self.scheduler.step()

        with torch.no_grad():
            backbone = self.model.module if hasattr(self.model, "module") else self.model
            backbone.logit_scale.clamp_(0, math.log(100))
            if hasattr(backbone, "logit_scale_token"):
                backbone.logit_scale_token.clamp_(0, math.log(100))

        # Batch/Step-level metrics -----------
        counts_global = collect_step_counts(
            logits_i2st.detach(), logits_st2i.detach(),
            labels.detach(), logit_scale.detach(),
            img_embed.detach(), st_embed.detach(),
            img_global=img_global, st_global=st_global,
            topk_list_step=(10, 20, 100)
        )

        # Losses record -----------
        bs_local = batch.img.size(0)
        # Token loss is averaged over tok_total (valid cell anchors), not over patches
        # Use tok_total for correct global aggregation (esp. when valid cells vary across ranks/batches)
        tok_total_local = int(token_metrics["tok_total"]) if token_metrics is not None else 0

        backbone = self.model.module if hasattr(self.model, "module") else self.model
        lam_patch = float(backbone.lambda_patch)
        lam_token = float(backbone.lambda_token)

        # patch_loss sum
        patch_loss_sum = torch.tensor(
            float((patch_loss if patch_loss is not None else loss).item()) * bs_local,
            device=self.device
        )
        # token_loss sum (sum over valid cell anchors)
        token_loss_sum = torch.tensor(
            float((token_loss.item() if token_loss is not None else 0.0)) * tok_total_local,
            device=self.device
        )
        # Objective-weighted sums, with correct denominators:
        #   patch part is per-patch, token part is per-cell.
        patch_obj_sum = patch_loss_sum * lam_patch
        token_obj_sum = token_loss_sum * lam_token
        # reduce across all processes
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(patch_loss_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(token_loss_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(patch_obj_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(token_obj_sum, op=dist.ReduceOp.SUM)
        counts_global["patch_loss_sum"] = patch_loss_sum.item()
        counts_global["token_loss_sum"] = token_loss_sum.item()
        counts_global["patch_obj_sum"] = patch_obj_sum.item()
        counts_global["token_obj_sum"] = token_obj_sum.item()

        # Token retrieval metrics (present only if token head ran) -----------
        if token_metrics is not None:
            t1_sum = torch.tensor(token_metrics["tok_top1_correct_sum"], device=self.device, dtype=torch.long)
            tK_sum = torch.tensor(token_metrics["tok_topK_correct_sum"], device=self.device, dtype=torch.long)
            tt = torch.tensor(token_metrics["tok_total"], device=self.device, dtype=torch.long)
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(t1_sum, op=dist.ReduceOp.SUM)
                dist.all_reduce(tK_sum, op=dist.ReduceOp.SUM)
                dist.all_reduce(tt, op=dist.ReduceOp.SUM)
            counts_global["tok_top1_correct_sum"] = int(t1_sum.item())
            counts_global["tok_topK_correct_sum"] = int(tK_sum.item())
            counts_global["tok_total"] = int(tt.item())
            counts_global["tok_K"] = token_metrics["tok_K"]

        self.global_step += 1

        # Dataset-quality counters (dummy ST graphs) -----------
        # DDP safety: always participate in the collective (treat missing attr as 0)
        dummy_initial = torch.tensor(0, device=self.device, dtype=torch.long)
        di = getattr(batch, "dummy_initial", None)
        if di is not None:
            dummy_initial = di.to(dtype=torch.long).sum()
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(dummy_initial, op=dist.ReduceOp.SUM)
        counts_global["dummy_initial_count"] = int(dummy_initial.item())

        return counts_global, logit_scale.detach().item(), logit_scale_token.detach().item()




    def train_epoch(self, dataloader):
        """
        Runs one training epoch.
        Returns the average loss and averaged metrics over the epoch.
        """
        rank = get_dist_rank() if (dist.is_available() and dist.is_initialized()) else 0

        metrics_accum = {}
        steps = 0

        pbar = tqdm(dataloader, desc="Training Epoch", leave=False)

        # Iterate over all training batches ---------------
        # -------------------------------------------------
        for batch in pbar:

            # Ensure warmup applies to the very first optimizer update
            if self.scheduler is not None and self.global_step == 0 and getattr(self.scheduler, "last_epoch", -1) == -1:
                self.scheduler.step()

            # Perform one training step
            counts_global, logit_scale_print, logit_scale_token_print = self.train_step(batch)
            steps += 1

            # Aggregate counts across steps
            for key, value in counts_global.items():
                if key == "tok_K":
                    # tok_K is constant over the run: keep its value (not accumulated)
                    metrics_accum["tok_K"] = value
                else:
                    metrics_accum[key] = metrics_accum.get(key, 0.0) + value

            # Calculate running averages and show in progress bar
            if is_main_process():
                # Total objective uses two denominators: λ_patch * E[patch_loss per patch] + λ_token * E[token_loss per cell]
                patch_term = metrics_accum.get("patch_obj_sum", 0.0) / metrics_accum["total_samples"]
                tok_total_so_far = metrics_accum.get("tok_total", 0.0)
                token_term = (metrics_accum.get("token_obj_sum", 0.0) / tok_total_so_far) if tok_total_so_far > 0 else 0.0
                avg_loss_so_far = patch_term + token_term
                avg_patch_loss_so_far = metrics_accum.get("patch_loss_sum", 0.0) / metrics_accum["total_samples"]
                avg_token_loss_so_far = (metrics_accum.get("token_loss_sum", 0.0) / tok_total_so_far) if tok_total_so_far > 0 else 0.0
                acc_i2st_so_far = metrics_accum["correct_i2st_sum"] / metrics_accum["total_samples"]
                acc_st2i_so_far = metrics_accum["correct_st2i_sum"] / metrics_accum["total_samples"]
                display_metrics = {
                    "loss": f"{avg_loss_so_far:.4f}",
                    "acc_i2st": f"{acc_i2st_so_far:.4f}",
                    "acc_st2i": f"{acc_st2i_so_far:.4f}",
                    "patch_loss": f"{avg_patch_loss_so_far:.4f}",
                    "token_loss": f"{avg_token_loss_so_far:.4f}",
                    "logit_scale": f"{logit_scale_print:.4f}",
                    "logit_scale_token": f"{logit_scale_token_print:.4f}",
                }
                pbar.set_postfix(display_metrics)

        # Calculate STEP (=batch) losses and metrics ---------------
        # ----------------------------------------------------------
        # Average losses
        patch_term = metrics_accum.get("patch_obj_sum", 0.0) / metrics_accum["total_samples"]
        tok_total = metrics_accum.get("tok_total", 0)
        token_term = (metrics_accum.get("token_obj_sum", 0.0) / tok_total) if tok_total > 0 else 0.0
        avg_loss = patch_term + token_term
        train_patch_loss = metrics_accum.get("patch_loss_sum", 0.0) / metrics_accum["total_samples"]
        train_token_loss = (metrics_accum.get("token_loss_sum", 0.0) / tok_total) if tok_total > 0 else 0.0
        # Average token metrics
        tok_top1 = (metrics_accum.get("tok_top1_correct_sum", 0) / tok_total) if tok_total > 0 else None
        tok_topK = (metrics_accum.get("tok_topK_correct_sum", 0) / tok_total) if tok_total > 0 else None
        # Average accuracies
        acc_i2st = metrics_accum["correct_i2st_sum"] / metrics_accum["total_samples"]
        acc_st2i = metrics_accum["correct_st2i_sum"] / metrics_accum["total_samples"]
        # Average MRRs
        mrr_i2st = metrics_accum["rr_i2st_sum"] / metrics_accum["total_samples"]
        mrr_st2i = metrics_accum["rr_st2i_sum"] / metrics_accum["total_samples"]
        # Average similarity metrics
        pos_sim = metrics_accum["pos_sim_sum"] / metrics_accum["total_samples"]
        neg_sim = metrics_accum["neg_sim_sum"] / metrics_accum["total_samples"]
        # Average nDCG and Spearman
        ndcg10_i2st = metrics_accum.get("ndcg_i2st@10_sum", 0.0) / metrics_accum["total_samples"]
        ndcg10_st2i = metrics_accum.get("ndcg_st2i@10_sum", 0.0) / metrics_accum["total_samples"]
        ndcg20_i2st = metrics_accum.get("ndcg_i2st@20_sum", 0.0) / metrics_accum["total_samples"]
        ndcg20_st2i = metrics_accum.get("ndcg_st2i@20_sum", 0.0) / metrics_accum["total_samples"]
        ndcg100_i2st = metrics_accum.get("ndcg_i2st@100_sum", 0.0) / metrics_accum["total_samples"]
        ndcg100_st2i = metrics_accum.get("ndcg_st2i@100_sum", 0.0) / metrics_accum["total_samples"]
        spr_i2st = metrics_accum.get("spearman_i2st_sum", 0.0) / metrics_accum["total_samples"]
        spr_st2i = metrics_accum.get("spearman_st2i_sum", 0.0) / metrics_accum["total_samples"]

        # Store average metrics
        avg_metrics = {
            "train_patch_loss": train_patch_loss,
            "train_token_loss": train_token_loss, 
            "train_tok_top1": tok_top1,
            "train_tok_topK": tok_topK, 
            "train_tok_K": metrics_accum.get("tok_K", None),         
            "train_acc_i2st": acc_i2st,
            "train_acc_st2i": acc_st2i,
            "train_mrr_i2st": mrr_i2st,
            "train_mrr_st2i": mrr_st2i,
            "train_pos_sim": pos_sim,
            "train_neg_sim": neg_sim,
            "train_pos_neg_gap":  pos_sim - neg_sim,
            "train_ndcg_i2st@10": ndcg10_i2st,
            "train_ndcg_st2i@10": ndcg10_st2i,
            "train_ndcg_i2st@20": ndcg20_i2st,
            "train_ndcg_st2i@20": ndcg20_st2i,
            "train_ndcg_i2st@100": ndcg100_i2st,
            "train_ndcg_st2i@100": ndcg100_st2i,
            "train_spearman_i2st": spr_i2st,
            "train_spearman_st2i": spr_st2i,
        }
        
        # Add barrier to ensure synchronization
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

        # Aggregate dummy-graph statistics for this epoch
        dummy_initial_total = float(metrics_accum.get("dummy_initial_count", 0.0))
        total_seen = float(metrics_accum.get("total_samples", 0.0))
        if total_seen > 0.0:
            avg_metrics["train_dummy_over_total"] = dummy_initial_total / total_seen
        else:
            avg_metrics["train_dummy_over_total"] = 0.0
        avg_metrics["train_dummy_count"] = dummy_initial_total

        # Log epoch metrics
        metrics_str = "\n".join(f"    {k}: {self._fmt_metric_value(v)}" for k, v in avg_metrics.items())
        logger.info(f"Training Epoch Completed - Avg Loss: {avg_loss:.4f}\nAvg Metrics:\n{metrics_str}\n")
 
        return avg_loss, avg_metrics, logit_scale_print, logit_scale_token_print




    def validation_step(self, batch):
        """
        Performs one validation step.
        """
        batch = batch.to(self.device, non_blocking=True)
        autocast_ctx = (
            autocast("cuda", dtype=self.amp_dtype)
            if (self.use_amp and self.device.type == "cuda")
            else contextlib.nullcontext()
        )
        with torch.no_grad(), autocast_ctx:

            # Validation -----------
            (loss, logits_i2st, logits_st2i, labels,
             img_embed, st_embed, logit_scale, logit_scale_token,
             img_global, st_global,
             patch_loss, token_loss, token_metrics) = self.model(batch)

            # Batch/Step-level metrics -----------
            counts_global = collect_step_counts(
                logits_i2st.detach(), logits_st2i.detach(),
                labels.detach(), logit_scale.detach(),
                img_embed.detach(), st_embed.detach(),
                img_global=img_global, st_global=st_global,
                topk_list_step=(10, 20, 100)
            )
            
            # Losses record -----------
            bs_local = batch.img.size(0)
            # Token loss is averaged over tok_total (valid cell anchors), not over patches
            # Use tok_total for correct global aggregation
            tok_total_local = int(token_metrics["tok_total"]) if token_metrics is not None else 0

            backbone = self.model.module if hasattr(self.model, "module") else self.model
            lam_patch = float(backbone.lambda_patch)
            lam_token = float(backbone.lambda_token)

            _patch = (patch_loss if patch_loss is not None else loss)
            patch_loss_sum = torch.tensor(float(_patch.item()) * bs_local, device=self.device)
            token_loss_sum = torch.tensor(float(token_loss.item() if token_loss is not None else 0.0) * tok_total_local,device=self.device)
            patch_obj_sum = patch_loss_sum * lam_patch
            token_obj_sum = token_loss_sum * lam_token
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(patch_loss_sum, op=dist.ReduceOp.SUM)
                dist.all_reduce(token_loss_sum, op=dist.ReduceOp.SUM)
                dist.all_reduce(patch_obj_sum, op=dist.ReduceOp.SUM)
                dist.all_reduce(token_obj_sum, op=dist.ReduceOp.SUM)
            counts_global["patch_loss_sum"] = patch_loss_sum.item()
            counts_global["token_loss_sum"] = token_loss_sum.item()
            counts_global["patch_obj_sum"] = patch_obj_sum.item()
            counts_global["token_obj_sum"] = token_obj_sum.item()

            # Token retrieval metrics (present only if token head ran)
            if token_metrics is not None:
                t1_sum = torch.tensor(token_metrics["tok_top1_correct_sum"], device=self.device, dtype=torch.long)
                tK_sum = torch.tensor(token_metrics["tok_topK_correct_sum"], device=self.device, dtype=torch.long)
                tt = torch.tensor(token_metrics["tok_total"], device=self.device, dtype=torch.long)
                if dist.is_available() and dist.is_initialized():
                    dist.all_reduce(t1_sum, op=dist.ReduceOp.SUM)
                    dist.all_reduce(tK_sum, op=dist.ReduceOp.SUM)
                    dist.all_reduce(tt, op=dist.ReduceOp.SUM)
                counts_global["tok_top1_correct_sum"] = int(t1_sum.item())
                counts_global["tok_topK_correct_sum"] = int(tK_sum.item())
                counts_global["tok_total"] = int(tt.item())
                counts_global["tok_K"] = token_metrics["tok_K"]

        return counts_global




    def validation_epoch(self, dataloader_step, dataloader_global=None, compute_cka: bool = False, slide_partitions: dict | None = None):
        """
        Runs one validation epoch.
        Returns the average loss and averaged metrics.
        """

        if dataloader_global is None:
            dataloader_global = dataloader_step

        self.model.eval()

        # ---------------------------
        # PASS A: step metrics / loss
        # ---------------------------
        metrics_accum = {}
        pbar = tqdm(dataloader_step, desc="Validation Epoch (step)", leave=False)
        for batch in pbar:
            counts_global = self.validation_step(batch)

            for key, value in counts_global.items():
                if key == "tok_K":
                    metrics_accum["tok_K"] = value
                else:
                    metrics_accum[key] = metrics_accum.get(key, 0.0) + value

            if is_main_process() and hasattr(pbar, "set_postfix"):
                patch_term = metrics_accum.get("patch_obj_sum", 0.0) / metrics_accum["total_samples"]
                tok_total_so_far = metrics_accum.get("tok_total", 0.0)
                token_term = (metrics_accum.get("token_obj_sum", 0.0) / tok_total_so_far) if tok_total_so_far > 0 else 0.0
                avg_loss_so_far = patch_term + token_term
                acc_i2st_so_far = metrics_accum["correct_i2st_sum"] / metrics_accum["total_samples"]
                acc_st2i_so_far = metrics_accum["correct_st2i_sum"] / metrics_accum["total_samples"]
                pbar.set_postfix({
                    "loss": f"{avg_loss_so_far:.4f}",
                    "acc_i2st": f"{acc_i2st_so_far:.4f}",
                    "acc_st2i": f"{acc_st2i_so_far:.4f}",
                })

        # Calculate STEP (=batch) losses and metrics ---------------
        # Average losses
        patch_term = metrics_accum.get("patch_obj_sum", 0.0) / metrics_accum["total_samples"]
        tok_total = metrics_accum.get("tok_total", 0)
        token_term = (metrics_accum.get("token_obj_sum", 0.0) / tok_total) if tok_total > 0 else 0.0
        avg_loss = patch_term + token_term
        val_patch_loss = metrics_accum.get("patch_loss_sum", 0.0) / metrics_accum["total_samples"]
        val_token_loss = (metrics_accum.get("token_loss_sum", 0.0) / tok_total) if tok_total > 0 else 0.0
        # Average token metrics
        tok_top1 = (metrics_accum.get("tok_top1_correct_sum", 0) / tok_total) if tok_total > 0 else None
        tok_topK = (metrics_accum.get("tok_topK_correct_sum", 0) / tok_total) if tok_total > 0 else None
        # Average accuracies
        acc_i2st = metrics_accum["correct_i2st_sum"] / metrics_accum["total_samples"]
        acc_st2i = metrics_accum["correct_st2i_sum"] / metrics_accum["total_samples"]
        # Average MRRs
        mrr_i2st = metrics_accum["rr_i2st_sum"] / metrics_accum["total_samples"]
        mrr_st2i = metrics_accum["rr_st2i_sum"] / metrics_accum["total_samples"]
        # Average similarity metrics
        pos_sim = metrics_accum["pos_sim_sum"] / metrics_accum["total_samples"]
        neg_sim = metrics_accum["neg_sim_sum"] / metrics_accum["total_samples"]
        # Average nDCG and Spearman
        ndcg10_i2st = metrics_accum.get("ndcg_i2st@10_sum", 0.0) / metrics_accum["total_samples"]
        ndcg10_st2i = metrics_accum.get("ndcg_st2i@10_sum", 0.0) / metrics_accum["total_samples"]
        ndcg20_i2st = metrics_accum.get("ndcg_i2st@20_sum", 0.0) / metrics_accum["total_samples"]
        ndcg20_st2i = metrics_accum.get("ndcg_st2i@20_sum", 0.0) / metrics_accum["total_samples"]
        ndcg100_i2st = metrics_accum.get("ndcg_i2st@100_sum", 0.0) / metrics_accum["total_samples"]
        ndcg100_st2i = metrics_accum.get("ndcg_st2i@100_sum", 0.0) / metrics_accum["total_samples"]
        spr_i2st = metrics_accum.get("spearman_i2st_sum", 0.0) / metrics_accum["total_samples"]
        spr_st2i = metrics_accum.get("spearman_st2i_sum", 0.0) / metrics_accum["total_samples"]

        # Store average metrics
        avg_metrics = {
            "val_patch_loss": val_patch_loss,
            "val_token_loss": val_token_loss,
            "val_tok_top1": tok_top1,
            "val_tok_topK": tok_topK,
            "val_tok_K": metrics_accum.get("tok_K", None),
            "val_acc_i2st": acc_i2st,
            "val_acc_st2i": acc_st2i,
            "val_mrr_i2st": mrr_i2st,
            "val_mrr_st2i": mrr_st2i,
            "val_pos_sim": pos_sim,
            "val_neg_sim": neg_sim,
            "val_pos_neg_gap":  pos_sim - neg_sim,
            "val_ndcg_i2st@10": ndcg10_i2st,
            "val_ndcg_st2i@10": ndcg10_st2i,
            "val_ndcg_i2st@20": ndcg20_i2st,
            "val_ndcg_st2i@20": ndcg20_st2i,
            "val_ndcg_i2st@100": ndcg100_i2st,
            "val_ndcg_st2i@100": ndcg100_st2i,
            "val_spearman_i2st": spr_i2st,
            "val_spearman_st2i": spr_st2i,
        }

        # -------------------------------------------
        # PASS B: global embeddings for retrieval eval
        # -------------------------------------------
        # Local embeddings only (no distributed collectives here); gathered afterwards
        emb_out = self.compute_embeddings(
            dataloader_global,
            return_slide_idx=True,
            compute_cka=compute_cka,
            desc="Validation Epoch (global embeddings)",
        )
        all_img_embeds, all_st_embeds = emb_out["img_embeds"], emb_out["st_embeds"]
        all_slide_idx = emb_out["slide_idx"]
        all_img_feats = emb_out.get("img_feats", None)
        all_st_feats = emb_out.get("st_feats", None)
        all_scconcept_feats = emb_out.get("scconcept_feats", None)


        # Compute global metrics (CKA included iff features are provided)
        # => With the following metrics: 
        #          - Retrieval metrics: Accuracies topK, MRR, Cosine similarities, Graded nDCG@K, 
        #          - Other metrics: Hubness, ECE, kNN-agreement, CKA
        #          - Geometric metrics: Alignment, Spectrum, Uniformity, MMD
        logit_scale = self.model.module.logit_scale if hasattr(self.model, "module") else self.model.logit_scale  # get logit scale
        logit_scale = logit_scale.detach().exp().cpu()
        with torch.no_grad():
            global_metrics = global_metrics_computation(
                all_img_embeds,
                all_st_embeds,
                logit_scale=logit_scale,
                mode="val",
                acc_top_k_list=[3, 5, 10],      # accuracy: top-1 (implicit), top-3, top-5, top-10
                ndcg_top_k_list=[10, 20, 100],   # nDCG: top-10, top-20, top-100
                chunk_size=15000,
                device=self.device,
                slide_idx=all_slide_idx,
                slide_id2idx=self.slide_id2idx_val,
                subset_slide_idx=None,  # ALL (INT+EXT)
                all_img_feats=all_img_feats,
                all_st_feats=all_st_feats,
                all_scconcept_feats=all_scconcept_feats,
            )

            # Same but for partitioned metrics: INT and EXT subsets (if provided)
            if slide_partitions is not None:
                for part_name, idx_list in slide_partitions.items():
                    if idx_list is None or len(idx_list) == 0:
                        continue
                    idx_list = sorted(set(idx_list))  # ensure unique and sorted
                    part_metrics = global_metrics_computation(
                        all_img_embeds,
                        all_st_embeds,
                        logit_scale=logit_scale,
                        mode=f"val_{part_name}",
                        acc_top_k_list=[3, 5, 10],
                        ndcg_top_k_list=[10, 20, 100],
                        chunk_size=15000,
                        device=self.device,
                        slide_idx=all_slide_idx,
                        slide_id2idx=self.slide_id2idx_val,
                        compute_per_slide_ndcg=False,    # skip per-slide for INT/EXT
                        subset_slide_idx=idx_list,  # INT or EXT only
                        all_img_feats=all_img_feats,
                        all_st_feats=all_st_feats,
                        all_scconcept_feats=all_scconcept_feats,
                    )
                    global_metrics.update(part_metrics)

            # Within-slide retrieval (gallery = the query's own slide), macro-averaged over slides
            within_metrics = within_slide_metrics_computation(
                all_img_embeds,
                all_st_embeds,
                logit_scale=logit_scale,
                mode="val",
                acc_top_k_list=[3, 5, 10],   # top-1 added implicitly
                min_patches=cfg_get(self.config, "validation", "within_slide_min_patches"),
                chunk_size=15000,
                device=self.device,
                slide_idx=all_slide_idx,
                slide_id2idx=self.slide_id2idx_val,
            )
            global_metrics.update(within_metrics)

        # Merge the globally computed retrieval metrics
        avg_metrics.update(global_metrics)

        metrics_str = "\n".join(f"    {k}: {self._fmt_metric_value(v)}" for k, v in avg_metrics.items())
        logger.info(f"Validation Epoch Completed - Avg Loss: {avg_loss:.4f}\nAvg Metrics:\n{metrics_str}\n")

        # Free up memory
        del all_img_embeds, all_st_embeds
        try:
            del all_slide_idx, all_img_feats, all_st_feats, all_scconcept_feats
        except UnboundLocalError:
            pass
        gc.collect()

        return avg_loss, avg_metrics
    



    def compute_embeddings(self, dataloader, return_slide_idx: bool = False, compute_cka: bool = False, desc: str = "Computing Embeddings"):
        """
        Compute embeddings locally (no distributed collectives; the caller gathers the results).

        Args:
            dataloader: dataloader over a (possibly rank-partitioned) subset
            return_slide_idx: also return concatenated batch.slide_idx
            compute_cka: also return pre-CLIP (pre-projection) features cached in CLIPModel
            desc: tqdm description

        Returns:
            dict with:
              - img_embeds: (N, D)
              - st_embeds:  (N, D)
              - slide_idx:  (N,) or None
              - img_feats:  (N, d_img) or None   (pre-CLIP feats for CKA)
              - st_feats:   (N, d_st)  or None   (pre-CLIP feats for CKA)
              - scconcept_feats: (N, d_scconcept) or None  (patch-pooled scConcept feats for CKA)
        """
        self.model.eval()
        all_img_embeds = []
        all_st_embeds = []
        all_slide_idx = [] if return_slide_idx else None
        all_img_feats = [] if compute_cka else None
        all_st_feats = [] if compute_cka else None
        # Patch-pooled scConcept features for CKA (only if the token branch is on)
        token_branch_on = float(cfg_get(self.config, "model", "clip", "lambda_token")) > 0.0
        all_scconcept_feats = [] if (compute_cka and token_branch_on) else None

        backbone = getattr(self.model, "module", self.model)  # unwrap if DDP
        backbone.eval()

        autocast_ctx = (
            autocast("cuda", dtype=self.amp_dtype)
            if (self.use_amp and self.device.type == "cuda")
            else contextlib.nullcontext()
        )
        with torch.no_grad(), autocast_ctx:
            for batch in tqdm(dataloader, desc=desc, leave=False):
                
                batch = batch.to(self.device, non_blocking=True)
                img_embed = backbone.encode_image(batch.img)
                st_embed  = backbone.encode_st(batch)

                all_img_embeds.append(img_embed.detach().cpu())
                all_st_embeds.append(st_embed.detach().cpu())

                if return_slide_idx:
                    all_slide_idx.append(batch.slide_idx.detach().cpu())
                if compute_cka:
                    # Cached by encode_image/encode_st (pre-projection)
                    all_img_feats.append(backbone._last_img_feat.detach().cpu())
                    all_st_feats.append(backbone._last_st_feat.detach().cpu())
                    if token_branch_on:
                        # Patch-pooled frozen scConcept teacher (masked mean over valid cells)
                        pooled_scc = pool_scconcept_per_patch(batch.scconcept_emb, batch.scconcept_cell_id)
                        all_scconcept_feats.append(pooled_scc.detach().cpu())

        out = {
            "img_embeds": torch.cat(all_img_embeds, dim=0) if len(all_img_embeds) else torch.empty((0, backbone.embed_dim)),
            "st_embeds":  torch.cat(all_st_embeds,  dim=0) if len(all_st_embeds)  else torch.empty((0, backbone.embed_dim)),
            "slide_idx":  (torch.cat(all_slide_idx, dim=0) if (return_slide_idx and len(all_slide_idx)) else None),
            "img_feats":  (torch.cat(all_img_feats, dim=0) if (compute_cka and len(all_img_feats)) else None),
            "st_feats":   (torch.cat(all_st_feats,  dim=0) if (compute_cka and len(all_st_feats))  else None),
            "scconcept_feats": (torch.cat(all_scconcept_feats, dim=0) if (compute_cka and token_branch_on and len(all_scconcept_feats)) else None),
        }
        return out
    

    

    def save_checkpoint(self, epoch, config, checkpoint_path):
        """
        Save model, optimizer, scheduler, scaler states, plus
        epoch, early-stopping state, global_step, etc.
        """
        logger.info(f"Saving checkpoint to {checkpoint_path}")

        monitor = self.early_stopping_monitor
        mode = self.early_stopping_mode
        best_value = self.best_early_stopping_value

        # Do not save the runtime-only resume pointer
        config_to_save = dict(config)
        config_to_save.pop("checkpoint_start_path", None)

        checkpoint_dict = {
            "epoch": epoch,
            "config": config_to_save,
            # Optimizer hyperparameters in force (may differ from "config" after a resume)
            "effective_optimizer": collect_effective_optimizer(
                self.optimizer, self.scheduler, config,
                heads_frozen=self._clip_heads_frozen,
                restored_from=self._optimizer_state_restored_from,
            ),
            "was_ddp": isinstance(self.model, nn.parallel.DistributedDataParallel),
            # model and optimizer states
            "model_state_dict": (
                self.model.module.state_dict()
                if isinstance(self.model, nn.parallel.DistributedDataParallel)
                else self.model.state_dict()
            ),
            "optimizer_state_dict": self.optimizer.state_dict(),
            # scheduler/scaler (only if present)
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "scaler_state_dict": self.scaler.state_dict() if self.scaler is not None else None,
            "amp_dtype": ("bf16" if self.amp_dtype == torch.bfloat16
                          else "fp16" if self.amp_dtype == torch.float16
                          else "fp32"),
            # early stopping state
            "best_early_stopping_value": best_value,
            "best_early_stopping_epoch": self.best_early_stopping_epoch,
            "early_stopping_monitor": monitor,
            "early_stopping_mode": mode,
            "patience_counter": self.patience_counter,
            # logging info
            "wandb_run_id": wandb.run.id if wandb.run else None,
            # iteration count
            "global_step": self.global_step,
        }

        torch.save(checkpoint_dict, checkpoint_path)

        logger.info(f"Checkpoint saved successfully.")




    def load_checkpoint(self, checkpoint_path):
        """
        Load a checkpoint from disk and restore the model, optimizer, scheduler, scaler states,
        epoch, global_step, early-stopping state.
        """
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        # Map location need to take into account DDP option
        if self.device.type == "cuda":
            map_location = {"cuda:0": f"cuda:{self.device.index}"}
        else:
            map_location = self.device
        checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)

        # Model
        logger.info("** Loading model state dict")
        sd = reconcile_state_dict(checkpoint["model_state_dict"], self.model)
        self.model.load_state_dict(sd, strict=True)

        # Optimizer
        if checkpoint.get("optimizer_state_dict", None) is not None:
            logger.info("** Loading optimizer state dict")
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            # Record where the optimizer state comes from (compared with the config at fit() startup)
            self._optimizer_state_restored_from = str(checkpoint_path)

            # make sure all state tensors sit on the right GPU
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.device, non_blocking=True)

        # Scheduler
        if self.scheduler and checkpoint.get("scheduler_state_dict", None) is not None:
            logger.info("** Loading scheduler state dict")
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        # AMP scaler
        if self.use_amp and self.scaler is not None and checkpoint.get("scaler_state_dict", None) is not None:
            logger.info("** Loading scaler state dict")
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

            # Internal tensors are lazily materialised in recent PT versions.
            # Move them only when they are real tensors.
            if self.scaler._scale is not None:
                self.scaler._scale = self.scaler._scale.to(self.device)
            if self.scaler._growth_tracker is not None:
                self.scaler._growth_tracker = self.scaler._growth_tracker.to(self.device)

        # Log dtype used in the checkpoint
        if "amp_dtype" in checkpoint:
            logger.info(f"Checkpoint AMP dtype: {checkpoint['amp_dtype']}")

        # Early stopping info
        logger.info("** Loading early stopping state")
        self.best_early_stopping_value = checkpoint.get("best_early_stopping_value", None)
        self.best_early_stopping_epoch = checkpoint.get("best_early_stopping_epoch", None)
        self.early_stopping_monitor = checkpoint.get("early_stopping_monitor", None)
        self.early_stopping_mode = checkpoint.get("early_stopping_mode", None)
        self.patience_counter = checkpoint.get("patience_counter", 0)

        # Global step
        logger.info("** Loading global step")
        self.global_step = checkpoint.get("global_step", 0)

        start_epoch = checkpoint.get("epoch", 0) + 1

        logger.info(
            f"Checkpoint loaded. Resuming from epoch {start_epoch-1} | "
            f"best_early_stopping_value={self._fmt_metric_value(self.best_early_stopping_value)} | "
            f"best_early_stopping_epoch={self.best_early_stopping_epoch} | "
            f"early_stopping_monitor={self.early_stopping_monitor} | "
            f"early_stopping_mode={self.early_stopping_mode} | "
            f"global_step={self.global_step} | patience_counter={self.patience_counter}\n"
        )

        return start_epoch




    def wandb_collapse_to_one_section(self, key: str) -> str:
        """Collapse a W&B metric path to one section (see collapse_wandb_key)."""
        return collapse_wandb_key(key)




    def fit(self, 
            train_loader, val_loader_step, slide_id2idx_val, 
            num_epochs, config, checkpoint_path=None, start_epoch=0, slide_partitions_val: dict | None = None, 
            val_loader_global=None, val_loader_step_single=None):
        """
        Runs the full training and validation loop for a given number of epochs.
        
        Args:
            train_loader: DataLoader for training data.
            val_loader_step: DataLoader for validation data (step-level).
            slide_id2idx_val: Dictionary mapping slide IDs to indices for valid set, used for global metrics.
            num_epochs: Total number of epochs.
            config: Configuration dictionary
            checkpoint_path: (Optional) A format string to save model checkpoints.
                             E.g., "checkpoint_epoch_{epoch}.pth". If provided, a checkpoint
                             is saved after every epoch.
            start_epoch: (Optional) The epoch to start training from. Useful for resuming training.
            slide_partitions_val: (Optional) A dictionary defining partitions of the validation slides for partitioned metrics (e.g., {"INT": [...], "EXT": [...]} using slide_idx
            val_loader_global: DataLoader for validation data (global-level).
            val_loader_step_single: DataLoader for validation data (step-level) with single-slide batches.
        
        Returns:
            A history dictionary containing losses and metrics per epoch.
        """
        self.config = config
        self.slide_id2idx_val = slide_id2idx_val
        self.slide_partitions_val = slide_partitions_val

        # Validate the advanced scheduling config
        self._validate_advanced_scheduling_runtime()

        # Log optimizer config values not in force (config and restored optimizer are both set here)
        self._log_optimizer_provenance()

        # Normalize configured modes (train can be "alternate")
        train_mode_cfg = _normalize_batch_slide_mode(cfg_get(config, "training", "batch_slide_mode"))

        vraw = str(cfg_get(config, "training", "val_step_slide_mode")).strip().lower()
        if vraw in ("inherit", "same"):
            val_mode_cfg = "inherit"
        else:
            val_mode_cfg = _normalize_batch_slide_mode(vraw)  # mixed | single_slide | alternate

        # Validate and initialize early stopping
        self._validate_and_init_early_stopping(train_mode_cfg, val_mode_cfg)
        if self.early_stopping_patience is None:
            logger.info(
                f"Early stopping disabled (training.early_stopping.patience=None). "
                f"Configured monitor='{self.early_stopping_monitor}', mode='{self.early_stopping_mode}'.\n\n"
            )
        else:
            logger.info(
                f"Early stopping enabled with patience={self.early_stopping_patience}, "
                f"monitor='{self.early_stopping_monitor}', mode='{self.early_stopping_mode}'.\n\n"
            )

        def _resolve_train_epoch_mode(epoch: int) -> str:
            # Prefer sampler-reported mode when available (covers alternate_start_with != "mixed")
            s = getattr(train_loader, "sampler", None)
            m = getattr(s, "current_mode", None)
            if m in ("mixed", "single_slide"):
                return m
            # Fallback to config
            if train_mode_cfg == "alternate":
                return "mixed" if (int(epoch) % 2 == 0) else "single_slide"
            return train_mode_cfg

        def _resolve_val_epoch_mode(epoch: int, train_epoch_mode: str) -> str:
            if val_mode_cfg == "inherit":
                return train_epoch_mode
            if val_mode_cfg == "alternate":
                return "mixed" if (int(epoch) % 2 == 0) else "single_slide"
            return val_mode_cfg

        def _select_val_step_loader(val_epoch_mode: str):
            if val_epoch_mode == "mixed":
                return val_loader_step  # expected to be the mixed/global loader
            if val_epoch_mode == "single_slide":
                if val_loader_step_single is None:
                    raise ValueError(
                        "Validation step mode requires single-slide loader, but val_loader_step_single is None. "
                        "Build it in Experiment when val_step_slide_mode is 'single_slide' or 'alternate' or when "
                        "train is 'alternate' and val_step_slide_mode='inherit'."
                    )
                return val_loader_step_single
            raise ValueError(f"Unknown val_epoch_mode='{val_epoch_mode}'")

        history = {
            "train_loss": [],
            "val_loss": [],
            "train_metrics": [],
            "val_metrics": []
        }

        # Base lambda weights and token warmup schedule (epoch-wise)
        base_lambda_patch = float(cfg_get(self.config, "model", "clip", "lambda_patch"))
        base_lambda_token = float(cfg_get(self.config, "model", "clip", "lambda_token"))
        lambda_token_warmup_epochs = int(cfg_get(self.config, "model", "clip", "lambda_token_warmup_epochs"))
        logger.info(
            f"Lambda configuration - lambda_patch={base_lambda_patch}, "
            f"lambda_token={base_lambda_token}, "
            f"lambda_token_warmup_epochs={lambda_token_warmup_epochs}"
        )

        eval_every = cfg_get(self.config, "training", "eval_every")
        cka_every = cfg_get(self.config, "training", "cka_every")
        save_every = cfg_get(self.config, "training", "save_every")

        # Training
        for epoch in range(start_epoch, num_epochs):

            # Uncomment to save the pretrained model without training (checkpoint epoch=-1)
            # if is_main_process():
            #     epoch = -1
            #     cp_path = checkpoint_path.format(epoch=epoch)
            #     self.save_checkpoint(epoch, config, cp_path)
            # raise ValueError("Pre-trained model saved using epoch=-1 for name. Stop the training process here.")


            logger.info(f"========== Epoch {epoch} / with num_epochs = {num_epochs} ==========\n")

            # Two-stage schedule: heads trained during warmup, then frozen (applied before this epoch's training)
            self.maybe_apply_clip_head_schedule(epoch)

            # Lambda scheduling (epoch-wise):
            # Simple linear warmup for the token loss:
            #   - if base_lambda_token == 0 or warmup_epochs == 0: no warmup, constant λ
            #   - else: linearly ramp from 0 -> base_lambda_token over [0, lambda_token_warmup_epochs) epochs, then keep constant
            if base_lambda_token > 0.0 and lambda_token_warmup_epochs > 0:
                if epoch < lambda_token_warmup_epochs:
                    frac = float(epoch) / float(lambda_token_warmup_epochs)
                else:
                    frac = 1.0
                self.current_lambda_token = base_lambda_token * frac
            else:
                self.current_lambda_token = base_lambda_token

            # Patch loss: constant weight (no warmup)
            self.current_lambda_patch = base_lambda_patch

            # Push effective lambdas into the underlying CLIPModel (works with or without DDP)
            backbone = self.model.module if hasattr(self.model, "module") else self.model
            backbone.lambda_patch = float(self.current_lambda_patch)
            backbone.lambda_token = float(self.current_lambda_token)
            logger.info(
                f"Epoch {epoch}: effective lambda_patch={self.current_lambda_patch:.6f}, "
                f"lambda_token={self.current_lambda_token:.6f}\n"
            )
            
            # Train: epoch-dependent sampling for any sampler that supports it (DistributedSampler, DistributedWeightedSampler, etc.)
            sampler = getattr(train_loader, "sampler", None)
            if sampler is not None and hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)

            # Determine actual train/val step modes for this epoch (for logging)
            train_epoch_mode = _resolve_train_epoch_mode(epoch)
            val_epoch_mode = _resolve_val_epoch_mode(epoch, train_epoch_mode)
            will_run_val = (epoch % eval_every == 0)
            logger.info(
                f"Epoch {epoch}: train_batch_slide_mode={train_epoch_mode} | "
                f"val_step_slide_mode={val_epoch_mode} | will_run_val={will_run_val}"
            )

            train_loss, train_metrics, logit_scale_print, logit_scale_token_print = self.train_epoch(train_loader)
            history["train_loss"].append(train_loss)
            history["train_metrics"].append(train_metrics)

            # Validate
            if epoch % eval_every == 0:
                do_cka = (cka_every is not None and epoch % cka_every == 0)
                # Select the correct step-loader for this epoch (mixed vs single-slide) if needed
                val_loader_step_epoch = _select_val_step_loader(val_epoch_mode)
                # If the chosen loader has an epoch-aware sampler/batch_sampler, set it (no-op if not present)
                _vs = getattr(val_loader_step_epoch, "sampler", None)
                if _vs is not None and hasattr(_vs, "set_epoch"):
                    _vs.set_epoch(epoch)
                _vbs = getattr(val_loader_step_epoch, "batch_sampler", None)
                if _vbs is not None and hasattr(_vbs, "set_epoch"):
                    _vbs.set_epoch(epoch)
                val_loss, val_metrics = self.validation_epoch(
                    val_loader_step_epoch,
                    dataloader_global=(val_loader_global if val_loader_global is not None else val_loader_step),
                    compute_cka=do_cka,
                    slide_partitions=self.slide_partitions_val,
                )
                history["val_loss"].append(val_loss)
                history["val_metrics"].append(val_metrics)
                logger.info(f"### Epoch {epoch} finished - Train loss: {train_loss:.4f}, "
                            f"Train patch loss: {train_metrics.get('train_patch_loss', None):.4f}, "
                            f"Train token loss: {train_metrics.get('train_token_loss', None):.4f}, "
                            f"Val loss: {val_loss:.4f}, "
                            f"Val patch loss: {val_metrics.get('val_patch_loss', None):.4f}, "
                            f"Val token loss: {val_metrics.get('val_token_loss', None):.4f} ###")
            else:
                val_loss, val_metrics = None, {}
                history["val_loss"].append(None)
                history["val_metrics"].append(None)
                logger.info(f"### Epoch {epoch} finished - Train loss: {train_loss:.4f}, "
                            f"Train patch loss: {train_metrics.get('train_patch_loss', None):.4f}, "
                            f"Train token loss: {train_metrics.get('train_token_loss', None):.4f} ###")

            # Log lr and logit_scale in logger
            if any(("group_name" in g) for g in self.optimizer.param_groups):
                lr_map = {g.get("group_name", str(i)): float(g.get("lr", 0.0)) for i, g in enumerate(self.optimizer.param_groups)}
                logger.info(f"Learning rates (by group): {lr_map}")
            else:
                logger.info(f"Learning rate was: {self.optimizer.param_groups[0]['lr']}")
            logger.info(f"Last logit_scale was: {logit_scale_print}")
            logger.info(f"Last logit_scale_token was: {logit_scale_token_print}")
            should_stop = False
            
            # Log epoch metrics to wandb
            if is_main_process():
                logger.info(f"Logging metrics to wandb...")
                metrics_dict = {
                    "epoch": int(epoch),
                    "lr": float(self.optimizer.param_groups[0]["lr"]),
                    "last_logit_scale": logit_scale_print,
                    "last_logit_scale_token": logit_scale_token_print,

                    "early_stopping/monitor_value": None,
                    "early_stopping/best_value": self.best_early_stopping_value,
                    "early_stopping/patience_counter": int(self.patience_counter),
                    "early_stopping/is_best_epoch": 0,
                    "early_stopping/best_epoch": self.best_early_stopping_epoch,

                    "data/train_epoch_batch_slide_mode": train_epoch_mode,
                    "data/train_epoch_is_single_slide": int(train_epoch_mode == "single_slide"),
                    "data/val_epoch_step_slide_mode": val_epoch_mode,
                    "data/val_epoch_is_single_slide": int(val_epoch_mode == "single_slide"),
                    "data/val_epoch_ran": int(epoch % eval_every == 0),
                    
                    "losses/train_loss": train_loss,
                    "losses/val_loss": val_loss,
                    "losses/train_patch_loss": train_metrics.get("train_patch_loss", None),
                    "losses/train_token_loss": train_metrics.get("train_token_loss", None),
                    "losses/val_patch_loss": val_metrics.get("val_patch_loss", None),
                    "losses/val_token_loss": val_metrics.get("val_token_loss", None),

                    "lambda/lambda_patch": float(self.current_lambda_patch),
                    "lambda/lambda_token": float(self.current_lambda_token),

                    "token/train_tok_top1": train_metrics.get("train_tok_top1", None),
                    "token/val_tok_top1": val_metrics.get("val_tok_top1", None),

                    # 1. Step-level metrics (averaged over the epoch) -------------------------

                    # Accuracy
                    "step/step_acc/train_acc_i2st": train_metrics.get("train_acc_i2st", None),
                    "step/step_acc/train_acc_st2i": train_metrics.get("train_acc_st2i", None),
                    "step/step_acc/val_acc_i2st": val_metrics.get("val_acc_i2st", None),
                    "step/step_acc/val_acc_st2i": val_metrics.get("val_acc_st2i", None),

                    # MRR
                    "step/step_mrr/train_mrr_i2st": train_metrics.get("train_mrr_i2st", None),
                    "step/step_mrr/train_mrr_st2i": train_metrics.get("train_mrr_st2i", None),
                    "step/step_mrr/val_mrr_i2st": val_metrics.get("val_mrr_i2st", None),
                    "step/step_mrr/val_mrr_st2i": val_metrics.get("val_mrr_st2i", None),

                    # Similarity
                    "step/step_sim_i2st/train_pos_sim": train_metrics.get("train_pos_sim", None),
                    "step/step_sim_i2st/train_neg_sim": train_metrics.get("train_neg_sim", None),
                    "step/step_sim_i2st/train_pos_neg_gap": train_metrics.get("train_pos_neg_gap", None),
                    "step/step_sim_i2st/val_pos_sim": val_metrics.get("val_pos_sim",  None),
                    "step/step_sim_i2st/val_neg_sim": val_metrics.get("val_neg_sim", None),
                    "step/step_sim_i2st/val_pos_neg_gap": val_metrics.get("val_pos_neg_gap", None),

                    # nDCG
                    "step/step_ndcg/train_ndcg_i2st@10": train_metrics.get("train_ndcg_i2st@10", None),
                    "step/step_ndcg/train_ndcg_st2i@10": train_metrics.get("train_ndcg_st2i@10", None),
                    "step/step_ndcg/train_ndcg_i2st@20": train_metrics.get("train_ndcg_i2st@20", None),
                    "step/step_ndcg/train_ndcg_st2i@20": train_metrics.get("train_ndcg_st2i@20", None),
                    "step/step_ndcg/train_ndcg_i2st@100": train_metrics.get("train_ndcg_i2st@100", None),
                    "step/step_ndcg/train_ndcg_st2i@100": train_metrics.get("train_ndcg_st2i@100", None),
                    "step/step_ndcg/val_ndcg_i2st@10": val_metrics.get("val_ndcg_i2st@10", None),
                    "step/step_ndcg/val_ndcg_st2i@10": val_metrics.get("val_ndcg_st2i@10", None),
                    "step/step_ndcg/val_ndcg_i2st@20": val_metrics.get("val_ndcg_i2st@20", None),
                    "step/step_ndcg/val_ndcg_st2i@20": val_metrics.get("val_ndcg_st2i@20", None),
                    "step/step_ndcg/val_ndcg_i2st@100": val_metrics.get("val_ndcg_i2st@100", None),
                    "step/step_ndcg/val_ndcg_st2i@100": val_metrics.get("val_ndcg_st2i@100", None),

                    # Spearman
                    "step/step_spearman/train_spearman_i2st": train_metrics.get("train_spearman_i2st", None),
                    "step/step_spearman/train_spearman_st2i": train_metrics.get("train_spearman_st2i", None),
                    "step/step_spearman/val_spearman_i2st": val_metrics.get("val_spearman_i2st", None),
                    "step/step_spearman/val_spearman_st2i": val_metrics.get("val_spearman_st2i", None),

                    # 2. Global metrics (computed on the entire valid set) -------------------------

                    # 2.1. Global retrieval 

                    # Accuracy
                    "global/all/global_acc/val_global_acc_i2st_top1": val_metrics.get("val_global_acc_i2st", None),
                    "global/all/global_acc/val_global_acc_st2i_top1": val_metrics.get("val_global_acc_st2i", None),
                    "global/all/global_acc/val_global_acc_i2st_top3": val_metrics.get("val_global_acc_i2st_top3", None),
                    "global/all/global_acc/val_global_acc_st2i_top3": val_metrics.get("val_global_acc_st2i_top3", None),
                    "global/all/global_acc/val_global_acc_i2st_top5": val_metrics.get("val_global_acc_i2st_top5", None),
                    "global/all/global_acc/val_global_acc_st2i_top5": val_metrics.get("val_global_acc_st2i_top5", None),
                    "global/all/global_acc/val_global_acc_i2st_top10":val_metrics.get("val_global_acc_i2st_top10", None),
                    "global/all/global_acc/val_global_acc_st2i_top10":val_metrics.get("val_global_acc_st2i_top10", None),

                    # MRR
                    "global/all/global_mrr/val_global_mrr_i2st":   val_metrics.get("val_global_mrr_i2st",   None),
                    "global/all/global_mrr/val_global_mrr_st2i":   val_metrics.get("val_global_mrr_st2i",   None),

                    # Similarity
                    "global/all/global_sim_i2st/val_global_pos_sim": val_metrics.get("val_global_pos_sim", None),
                    "global/all/global_sim_i2st/val_global_neg_sim": val_metrics.get("val_global_neg_sim", None),
                    "global/all/global_sim_i2st/val_global_pos_neg_gap": val_metrics.get("val_global_pos_neg_gap",  None),

                    # nDCG
                    "global/all/global_ndcg/val_global_ndcg_i2st_top10": val_metrics.get("val_global_ndcg_i2st_top10", None),
                    "global/all/global_ndcg/val_global_ndcg_st2i_top10": val_metrics.get("val_global_ndcg_st2i_top10", None),
                    "global/all/global_ndcg/val_global_ndcg_i2st_top20": val_metrics.get("val_global_ndcg_i2st_top20", None),
                    "global/all/global_ndcg/val_global_ndcg_st2i_top20": val_metrics.get("val_global_ndcg_st2i_top20", None),
                    "global/all/global_ndcg/val_global_ndcg_i2st_top100": val_metrics.get("val_global_ndcg_i2st_top100", None),
                    "global/all/global_ndcg/val_global_ndcg_st2i_top100": val_metrics.get("val_global_ndcg_st2i_top100", None),

                    # 2.2. Global diverse diagnostics

                    # Hubness
                    "global/all/global_hub/val_gini_i2st@10": val_metrics.get("val_global_hub_gini_i2st@10", None),
                    "global/all/global_hub/val_top1p_i2st@10": val_metrics.get("val_global_hub_top1p_i2st@10", None),
                    "global/all/global_hub/val_gini_st2i@10": val_metrics.get("val_global_hub_gini_st2i@10", None),
                    "global/all/global_hub/val_top1p_st2i@10": val_metrics.get("val_global_hub_top1p_st2i@10", None),

                    # Calibration (ECE)
                    "global/all/global_calib/val_ece_i2st_bins15": val_metrics.get("val_global_ece_i2st_bins15", None),
                    "global/all/global_calib/val_ece_st2i_bins15": val_metrics.get("val_global_ece_st2i_bins15", None),

                    # KNN agreement
                    "global/all/global_knn_agree_imgs/overlap@10_i":  val_metrics.get("val_global_knn_overlap@10_i_anchor", None),
                    "global/all/global_knn_agree_imgs/jaccard@10_i": val_metrics.get("val_global_knn_jaccard@10_i_anchor", None),

                    # CKA
                    "global/all/global_cka/val_global_cka_he":  val_metrics.get("val_global_cka_he",  None),
                    "global/all/global_cka/val_global_cka_st":  val_metrics.get("val_global_cka_st",  None),
                    "global/all/global_cka/val_global_cka_he_st_pre": val_metrics.get("val_global_cka_he_st_pre", None),
                    "global/all/global_cka/val_global_cka_he_scconcept_pre": val_metrics.get("val_global_cka_he_scconcept_pre", None),

                    # 2.3.Geometry diagnostics
                    
                    # Alignment
                    "global/all/global_geom/val_global_align_mse": val_metrics.get("val_global_align_mse", None),

                    # Spectrum
                    "global/all/global_geom/val_global_effrank_img": val_metrics.get("val_global_effrank_img", None),
                    "global/all/global_geom/val_global_effrank_st": val_metrics.get("val_global_effrank_st", None),
                    "global/all/global_geom/val_global_effrank_both": val_metrics.get("val_global_effrank_both", None),
                    "global/all/global_geom/val_global_partratio_img": val_metrics.get("val_global_partratio_img", None),
                    "global/all/global_geom/val_global_partratio_st": val_metrics.get("val_global_partratio_st", None),
                    "global/all/global_geom/val_global_partratio_both": val_metrics.get("val_global_partratio_both", None),

                    # Uniformity
                    "global/all/global_geom/val_global_uni_img": val_metrics.get("val_global_uni_img", None),
                    "global/all/global_geom/val_global_uni_st": val_metrics.get("val_global_uni_st", None),
                    "global/all/global_geom/val_global_uni_both": val_metrics.get("val_global_uni_both", None),

                    # MMD
                    "global/all/global_geom/val_global_mmd_rbf": val_metrics.get("val_global_mmd_rbf", None),

                    # 2.4. Within-slide retrieval (batch-controlled; gallery = query's own slide, macro-averaged)
                    "global/within_slide/global_acc/val_within_slide_acc_i2st_top1":  val_metrics.get("val_within_slide_acc_i2st_top1",  None),
                    "global/within_slide/global_acc/val_within_slide_acc_st2i_top1":  val_metrics.get("val_within_slide_acc_st2i_top1",  None),
                    "global/within_slide/global_acc/val_within_slide_acc_i2st_top3":  val_metrics.get("val_within_slide_acc_i2st_top3",  None),
                    "global/within_slide/global_acc/val_within_slide_acc_st2i_top3":  val_metrics.get("val_within_slide_acc_st2i_top3",  None),
                    "global/within_slide/global_acc/val_within_slide_acc_i2st_top5":  val_metrics.get("val_within_slide_acc_i2st_top5",  None),
                    "global/within_slide/global_acc/val_within_slide_acc_st2i_top5":  val_metrics.get("val_within_slide_acc_st2i_top5",  None),
                    "global/within_slide/global_acc/val_within_slide_acc_i2st_top10": val_metrics.get("val_within_slide_acc_i2st_top10", None),
                    "global/within_slide/global_acc/val_within_slide_acc_st2i_top10": val_metrics.get("val_within_slide_acc_st2i_top10", None),
                    "global/within_slide/global_mrr/val_within_slide_mrr_i2st": val_metrics.get("val_within_slide_mrr_i2st", None),
                    "global/within_slide/global_mrr/val_within_slide_mrr_st2i": val_metrics.get("val_within_slide_mrr_st2i", None),
                    "global/within_slide/global_count/val_within_slide_n_slides_used":    val_metrics.get("val_within_slide_n_slides_used",    None),
                    "global/within_slide/global_count/val_within_slide_n_slides_skipped": val_metrics.get("val_within_slide_n_slides_skipped", None),

                    # 3. Data quality metrics (if present) -------------------------

                    "data_quality/train_dummy_over_total": train_metrics.get("train_dummy_over_total", None),
                    "data_quality/train_dummy_count": train_metrics.get("train_dummy_count", None),                                                
                }

                # Add per-group LR (especially useful in advanced scheduling)
                for i, g in enumerate(self.optimizer.param_groups):
                    gn = g.get("group_name", None)
                    if gn is not None:
                        metrics_dict[f"lr/{gn}"] = float(g.get("lr", 0.0))
                    else:
                        metrics_dict[f"lr/group_{i}"] = float(g.get("lr", 0.0))

                # Early stopping state (only on validation epochs)
                if val_loss is not None:
                    early_stopping_value = self._get_early_stopping_value(val_loss, val_metrics)
                    metrics_dict["early_stopping/monitor_value"] = early_stopping_value

                    if self._is_early_stopping_improvement(early_stopping_value):
                        previous_best = self.best_early_stopping_value
                        self.best_early_stopping_value = early_stopping_value
                        self.best_early_stopping_epoch = epoch
                        self.patience_counter = 0
                        metrics_dict["early_stopping/is_best_epoch"] = 1

                        logger.info(
                            f"Early stopping monitor improved: {self.early_stopping_monitor}="
                            f"{early_stopping_value:.6f} at epoch {epoch} "
                            f"(mode={self.early_stopping_mode}, previous_best={self._fmt_metric_value(previous_best)})."
                        )
                        if wandb.run:
                            wandb.run.summary["best_early_stopping_value"] = early_stopping_value
                            wandb.run.summary["best_early_stopping_metric"] = self.early_stopping_monitor
                            wandb.run.summary["best_early_stopping_mode"] = self.early_stopping_mode
                            wandb.run.summary["best_epoch"] = epoch
                    else:
                        self.patience_counter += 1
                        logger.info(
                            f"No improvement in {self.early_stopping_monitor} for "
                            f"{self.patience_counter} validation check(s). "
                            f"current={early_stopping_value:.6f} | "
                            f"best={self._fmt_metric_value(self.best_early_stopping_value)} | "
                            f"mode={self.early_stopping_mode}"
                        )

                    metrics_dict["early_stopping/best_value"] = self.best_early_stopping_value
                    metrics_dict["early_stopping/patience_counter"] = int(self.patience_counter)
                    metrics_dict["early_stopping/best_epoch"] = self.best_early_stopping_epoch

                    if self.early_stopping_patience is not None and self.patience_counter >= self.early_stopping_patience:
                        logger.info(
                            f"Early stopping triggered at epoch {epoch} "
                            f"(patience={self.early_stopping_patience}, monitor='{self.early_stopping_monitor}', mode='{self.early_stopping_mode}')."
                        )
                        should_stop = True

                # 4. Per-slide & partitioned diagnostics -------------------------
                # Helper to map "val_*" keys to (partition, leaf) and normalize leaf to always start with "val_"
                def _part_and_leaf(key: str):
                    if key.startswith("val_INT_"):
                        return "INT", "val_" + key[len("val_INT_"):]
                    if key.startswith("val_EXT_"):
                        return "EXT", "val_" + key[len("val_EXT_"):]
                    if key.startswith("val_"):
                        return "all", key
                    return None, None

                # 4.1. Per-slide nDCG@100 (from global_metric), routed to global/{part}/global_slide_ndcg/...
                for k, v in val_metrics.items():
                    if v is None:
                        continue
                    part, leaf = _part_and_leaf(k)
                    if part is None:
                        continue
                    # only keep true per-slide nDCG@100 keys "..._ndcg_{i2st|st2i}_top100"
                    if "_ndcg_" not in leaf or not leaf.endswith("_top100"):
                        continue
                    # leaf is "val_{slide_id}_ndcg_{direction}_top100"
                    core = leaf[len("val_"):]  # "{slide_id}_ndcg_i2st_top100"
                    if "_ndcg_" not in core:
                        continue
                    slide_id, rest = core.split("_ndcg_", 1)
                    if (self.slide_id2idx_val is not None) and (slide_id not in self.slide_id2idx_val):
                        continue
                    # Namespace: global/{part}/global_slide_ndcg/<leaf>
                    metrics_dict[f"global/{part}/global_slide_ndcg/{leaf}"] = v

                # 4.2. Partitioned diagnostics (INT/EXT): mirror the "global/all/..." layout
                for k, v in val_metrics.items():
                    if v is None:
                        continue
                    part, leaf = _part_and_leaf(k)
                    if part not in ("INT", "EXT"):
                        continue  # only partitioned ones here

                    # Accuracy
                    if leaf.startswith("val_global_acc_"):
                        # special-case top1 which has no suffix in the leaf name
                        if leaf == "val_global_acc_i2st":
                            metrics_dict[f"global/{part}/global_acc/val_global_acc_i2st_top1"] = v
                            continue
                        if leaf == "val_global_acc_st2i":
                            metrics_dict[f"global/{part}/global_acc/val_global_acc_st2i_top1"] = v
                            continue
                        metrics_dict[f"global/{part}/global_acc/{leaf}"] = v
                        continue
                    
                    # MRR
                    if leaf.startswith("val_global_mrr_"):
                        metrics_dict[f"global/{part}/global_mrr/{leaf}"] = v
                        continue
                    
                    # Similarity (image->st only)
                    if leaf in ("val_global_pos_sim", "val_global_neg_sim", "val_global_pos_neg_gap"):
                        metrics_dict[f"global/{part}/global_sim_i2st/{leaf}"] = v
                        continue
                    
                    # nDCG
                    if leaf.startswith("val_global_ndcg_"):
                        metrics_dict[f"global/{part}/global_ndcg/{leaf}"] = v
                        continue

                    # Hubness (place under global/{part}/global_hub/..., drop "global_hub_" in leaf)
                    if "_global_hub_" in leaf:
                        leaf_out = leaf.replace("val_global_hub_", "val_")
                        metrics_dict[f"global/{part}/global_hub/{leaf_out}"] = v
                        continue

                    # Calibration/ECE (global/{part}/global_calib/..., drop "global_ece_")
                    if "_global_ece_" in leaf:
                        leaf_out = leaf.replace("val_global_ece_", "val_")
                        metrics_dict[f"global/{part}/global_calib/{leaf_out}"] = v
                        continue

                    # kNN agreement (global/{part}/global_knn_agree/overlap@10_i | jaccard@10_i)
                    if leaf.endswith("_global_knn_overlap@10_i_anchor"):
                        metrics_dict[f"global/{part}/global_knn_agree_imgs/overlap@10_i"] = v
                        continue
                    if leaf.endswith("_global_knn_jaccard@10_i_anchor"):
                        metrics_dict[f"global/{part}/global_knn_agree_imgs/jaccard@10_i"] = v
                        continue

                    # CKA (global/{part}/global_cka/...)
                    if "_global_cka_" in leaf:
                        metrics_dict[f"global/{part}/global_cka/{leaf}"] = v
                        continue

                    # Geometry bundle (global/{part}/global_geom/val_global_align_mse | val_global_effrank_* | val_global_partratio_* | val_global_uni_* | val_global_mmd_rbf)
                    if ("_align_mse" in leaf) or ("_uni_" in leaf) or ("_effrank_" in leaf) or ("_partratio_" in leaf) or ("_mmd_rbf" in leaf):
                        metrics_dict[f"global/{part}/global_geom/{leaf}"] = v
                        continue

                # 4.3. Per-slide within-slide retrieval -> global/within_slide/global_slide_{acc,mrr}/...
                if self.slide_id2idx_val is not None:
                    for sid in self.slide_id2idx_val:
                        for leaf in ("acc_i2st_top1", "acc_st2i_top1", "acc_i2st_top5", "acc_st2i_top5"):
                            mk = f"val_within_slide_{sid}_{leaf}"
                            if mk in val_metrics:
                                metrics_dict[f"global/within_slide/global_slide_acc/{mk}"] = val_metrics[mk]
                        for leaf in ("mrr_i2st", "mrr_st2i"):
                            mk = f"val_within_slide_{sid}_{leaf}"
                            if mk in val_metrics:
                                metrics_dict[f"global/within_slide/global_slide_mrr/{mk}"] = val_metrics[mk]

                # Drop non-finite within-slide values (skipped slides)
                for _k in [kk for kk, vv in metrics_dict.items()
                           if "/within_slide/" in kk and (vv is None or not math.isfinite(float(vv)))]:
                    del metrics_dict[_k]

                # 5. Fill dynamic token topK key post-hoc -------------------------
                tok_K = train_metrics.get("train_tok_K", None)
                if tok_K is not None:
                    metrics_dict[f"token/train_tok_top{tok_K}"] = train_metrics.get("train_tok_topK", None)
                    metrics_dict[f"token/val_tok_top{tok_K}"] = val_metrics.get("val_tok_topK", None)

                flat_metrics = {
                    self.wandb_collapse_to_one_section(k): v
                    for k, v in metrics_dict.items()
                    if v is not None
                }
                wandb.log(flat_metrics, step=epoch, commit=True)

                # Also save metrics to a JSONL file for backup
                with open(config["logging"]["log_dir"] + "/backup_training_metrics.jsonl", "a") as f:
                    f.write(json.dumps(metrics_dict) + "\n")

            # Save checkpoint if needed (after early-stopping state update so resume is exact)
            if checkpoint_path is not None and epoch % save_every == 0:
                if is_main_process():
                    cp_path = checkpoint_path.format(epoch=epoch)
                    self.save_checkpoint(epoch, config, cp_path)

            # Add a synchronization barrier at the end of each epoch to ensure all processes are in sync
            if dist.is_available() and dist.is_initialized():
                dist.barrier()

            if should_stop:
                break
            
            logger.info("===" * 20 + "\n")

        return history
