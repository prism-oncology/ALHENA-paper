"""
Measure how far a HistoST LoRA finetune moved the H&E backbone from its frozen UNI2-h initialization,
from the saved LoRA factors of each checkpoint (Q and V of every ViT block):
    - dW = (alpha/rank) * D @ U: Frobenius, spectral and nuclear norms, relative drift, effective rank,
      participation ratio, 99%-energy rank
    - intruder dimensions of W0 + dW vs W0 (Shuttleworth et al. 2024, arXiv:2410.21228)

Saved in <ROOT_HistoST> / train / trainings / trainings_HistoST / analysis / lora_drift / (or --out_dir)
    - lora_drift_<tag>.csv: stats per adapted matrix (<tag> = t<id> of each training_id)
    - lora_drift_intruders_<tag>.csv: max |cos| of each inspected singular vector
    - lora_drift_<tag>.md: summary tables (with --markdown)
    - lora_drift_base_norms.json: norms of the frozen base weights (cache)
"""

import argparse
import json
import logging
import math
import os
import re
import resource
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.constants import get_full_path  # noqa: E402

logger = logging.getLogger(__name__)


# Suffixes of the four LoRA parameters inserted per ViT block by src/models/lora.py
DOWN_SUFFIX = "_downsample"
UP_SUFFIX = "_upsample"
PROJECTIONS = ("Q", "V")

# Row-block of the fused qkv weight ([3*d, d]) that each projection writes into.
# forward() builds cat([x_q, zeros, x_v]) -> Q is block 0, K is block 1 (never adapted), V is block 2.
PROJECTION_ROW_BLOCK = {"Q": 0, "V": 2}

# Per-adapter columns of the intruder pass (written as NaN when the base weight is missing)
INTRUDER_SUMMARY_FIELDS = (
    "n_intruders",
    "intruder_frac",
    "mean_max_cos",
    "min_max_cos",
    "subspace_overlap_topk",
    "mean_best_match_base_index",
    "max_best_match_base_index",
)


# ---------------------------------------------------------------------------------------------
# Checkpoint access
# ---------------------------------------------------------------------------------------------


def load_checkpoint_lazy(path: Path, allow_full_load: bool = True) -> Dict[str, Any]:
    """
    Load a checkpoint memory-mapped, with a full load as fallback (if allow_full_load).

    arguments:
        path [Path]: path to checkpoint_epoch_{e}.pth
        allow_full_load [bool]: permit the non-mmap fallback
    returns:
        checkpoint dict [Dict[str, Any]]
    """
    attempts = [
        {"mmap": True, "weights_only": True},
        {"mmap": True, "weights_only": False},
    ]
    if allow_full_load:
        attempts.append({"mmap": False, "weights_only": False})

    last_err: Optional[Exception] = None
    for kwargs in attempts:
        try:
            ckpt = torch.load(path, map_location="cpu", **kwargs)
            logger.debug("Loaded %s with %s", path.name, kwargs)
            return ckpt
        except Exception as exc:  # noqa: BLE001 - fall through to the next strategy
            last_err = exc
            continue

    raise RuntimeError(
        f"Could not load {path}. Last error: {last_err}. "
        "If this is a login node and only the non-mmap fallback would work, the file is "
        "likely not zipfile-serialized; run on a compute node instead."
    )


def resolve_state_dict(ckpt: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    """Return the model state dict, tolerating a bare state-dict file and a DDP prefix."""
    sd = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    if not isinstance(sd, dict):
        raise RuntimeError("Checkpoint does not contain a model state dict.")
    if any(k.startswith("module.") for k in sd):
        sd = {k[len("module."):] if k.startswith("module.") else k: v for k, v in sd.items()}
    return sd


def resolve_scaling(
    ckpt: Dict[str, Any],
    rank_from_shape: int,
    yaml_path: Optional[Path],
    alpha_override: Optional[float],
) -> Tuple[float, float, str]:
    """
    LoRA scaling alpha/rank (not stored in the state dict), from --alpha, else the checkpoint config, else the run YAML.

    returns:
        (alpha, scaling, source) [Tuple[float, float, str]]
    """
    if alpha_override is not None:
        alpha = float(alpha_override)
        return alpha, alpha / rank_from_shape, "cli --alpha"

    def _from_cfg(cfg: Any, source: str) -> Optional[Tuple[float, float, str]]:
        try:
            tcfg = cfg["training"]
            alpha = float(tcfg["lora_alpha"])
            rank = int(tcfg["lora_rank"])
        except (KeyError, TypeError, ValueError):
            return None
        if rank != rank_from_shape:
            raise RuntimeError(
                f"lora_rank in {source} is {rank} but the saved adapter tensors have rank "
                f"{rank_from_shape}. Refusing to guess."
            )
        return alpha, alpha / rank, source

    if isinstance(ckpt, dict) and isinstance(ckpt.get("config"), dict):
        got = _from_cfg(ckpt["config"], "checkpoint config")
        if got is not None:
            return got

    if yaml_path is not None and yaml_path.is_file():
        with open(yaml_path) as fh:
            got = _from_cfg(yaml.safe_load(fh), f"run YAML {yaml_path.name}")
        if got is not None:
            return got

    raise RuntimeError(
        "Could not determine lora_alpha (it is a plain float in LoRA.__init__, so it is not "
        "in the state dict). Pass --alpha explicitly."
    )


# ---------------------------------------------------------------------------------------------
# Core math -- never materializes the [d, d] delta
# ---------------------------------------------------------------------------------------------


def delta_spectrum(down: torch.Tensor, up: torch.Tensor, scaling: float) -> Dict[str, float]:
    """
    Exact norms and singular-value spectrum of dW = scaling * (down @ up), via r x r problems.

    arguments:
        down [torch.Tensor]: D, shape [d_in, r]  (adapter_*_downsample)
        up   [torch.Tensor]: U, shape [r, d_in]  (adapter_*_upsample)
        scaling [float]: alpha / rank
    returns:
        dict of scalar stats [Dict[str, float]]
    """
    D = down.detach().to(torch.float64)
    U = up.detach().to(torch.float64)
    r = D.shape[1]
    if U.shape[0] != r:
        raise RuntimeError(f"Rank mismatch between factors: down {tuple(D.shape)}, up {tuple(U.shape)}")

    # ||D U||_F^2 = trace( (D^T D) (U U^T) ); both Grams are r x r and symmetric.
    gram_d = D.T @ D
    gram_u = U @ U.T
    fro_sq = float((gram_d * gram_u).sum().clamp(min=0.0))

    # svals(D U) = svals(R_D @ R_U^T)  with D = Q_D R_D and U^T = Q_U R_U.
    _, r_d = torch.linalg.qr(D, mode="reduced")
    _, r_u = torch.linalg.qr(U.T, mode="reduced")
    svals = torch.linalg.svdvals(r_d @ r_u.T).clamp(min=0.0)

    # Self-check: the two independent routes to the Frobenius norm must agree.
    fro_from_svals = float((svals.pow(2).sum()))
    if fro_sq > 0 and abs(fro_from_svals - fro_sq) / max(fro_sq, 1e-30) > 1e-6:
        raise RuntimeError(
            f"Internal inconsistency: ||dW||_F^2 = {fro_sq:.6e} via Grams but "
            f"{fro_from_svals:.6e} via singular values."
        )

    svals = svals * float(scaling)
    lam = svals.pow(2)  # eigenvalues of dW^T dW
    total = float(lam.sum())

    if total <= 0.0:
        return {
            "fro": 0.0, "spectral": 0.0, "nuclear": 0.0,
            "eff_rank": 0.0, "participation_ratio": 0.0, "rank99": 0.0, "rank_max": float(r),
        }

    p = (lam / total).clamp(min=1e-30)
    entropy = float(-(p * p.log()).sum())
    energy = torch.cumsum(lam, dim=0) / total
    rank99 = int(torch.searchsorted(energy, torch.tensor(0.99, dtype=energy.dtype)).item()) + 1

    return {
        "fro": math.sqrt(total),
        "spectral": float(svals.max()),
        "nuclear": float(svals.sum()),
        "eff_rank": math.exp(entropy),
        "participation_ratio": float((total * total) / float(lam.pow(2).sum())),
        "rank99": float(min(rank99, r)),
        "rank_max": float(r),
    }


def base_block_view(qkv_weight: torch.Tensor, projection: str) -> torch.Tensor:
    """
    The [d, d] row-block of the frozen fused qkv weight that this adapter actually modifies.

    The fused qkv weight is [3*d, d] laid out as [Q; K; V]; K is never adapted.
    """
    out_dim = qkv_weight.shape[0]
    if out_dim % 3 != 0:
        raise RuntimeError(f"Expected fused qkv with out_features divisible by 3, got {out_dim}")
    d = out_dim // 3
    idx = PROJECTION_ROW_BLOCK[projection]
    return qkv_weight[idx * d:(idx + 1) * d]


def base_block_fro(qkv_weight: torch.Tensor, projection: str) -> float:
    """Frobenius norm of the frozen base weight sub-block (Q or V) this adapter modifies."""
    block = base_block_view(qkv_weight, projection)
    return float(block.detach().to(torch.float64).pow(2).sum().sqrt())


# ---------------------------------------------------------------------------------------------
# Intruder dimensions (Shuttleworth et al. 2024, arXiv:2410.21228)
# ---------------------------------------------------------------------------------------------


def base_left_singular(
    block: torch.Tensor, base_k: int = 0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Left singular vectors + singular values (float32) of the frozen base sub-block.

    arguments:
        block [torch.Tensor]: [d, d] frozen base sub-block, nn.Linear.weight [out, in] convention
        base_k [int]: keep only the top-k base singular vectors (0 = all, the exact/default case)
    returns:
        (U0 [d, K] float32, S0 [K] float32) [Tuple[torch.Tensor, torch.Tensor]]
    """
    u0, s0, _ = torch.linalg.svd(block.detach().to(torch.float64), full_matrices=False)
    if base_k and base_k > 0:
        u0, s0 = u0[:, :base_k], s0[:base_k]
    return u0.to(torch.float32).contiguous(), s0.to(torch.float32).contiguous()


def intruder_stats(
    base_u: torch.Tensor,
    base_s: torch.Tensor,
    block: torch.Tensor,
    down: torch.Tensor,
    up: torch.Tensor,
    scaling: float,
    top_k: int,
    epsilon: float,
) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    """
    Compare the singular structure of the merged W0 + dW against W0's own.

    A left singular vector y_j of W_tuned is an intruder dimension iff
    max_i |cos(y_j, x_i)| < epsilon over the retained left singular vectors x_i of W0.

    arguments:
        base_u [torch.Tensor]: [d, K] float32 left singular vectors of W0 (from base_left_singular)
        base_s [torch.Tensor]: [K] float32 singular values of W0
        block [torch.Tensor]: [d, d] frozen base sub-block, nn.Linear.weight [out, in] convention
        down [torch.Tensor]: D, shape [d_in, r]
        up [torch.Tensor]: U, shape [r, d_in]
        scaling [float]: alpha / rank
        top_k [int]: how many leading singular vectors of W_tuned to classify
        epsilon [float]: similarity threshold below which a vector counts as an intruder
    returns:
        (per-adapter summary, one row per inspected singular vector) [Tuple[Dict, List[Dict]]]
    """
    # Merged matrix in the nn.Linear.weight [out, in] convention: W0 + scaling * (D @ U)^T
    # (the adapter computes x @ D @ U); (D @ U)^T is computed as U^T @ D^T.
    w_tuned = block.detach().to(torch.float64).clone()
    w_tuned += (up.detach().to(torch.float64).T @ down.detach().to(torch.float64).T) * float(scaling)

    u_t, s_t, _ = torch.linalg.svd(w_tuned, full_matrices=False)
    del w_tuned

    k = int(min(top_k, u_t.shape[1]))
    y = u_t[:, :k].to(torch.float32)  # [d, k]
    del u_t

    cos = (base_u.T @ y).abs()  # [K, k]
    max_cos, best_idx = cos.max(dim=0)
    del cos

    # Fraction of the top-k finetuned subspace energy inside the top-k pretrained subspace (in [0, 1])
    kk = int(min(k, base_u.shape[1]))
    overlap_den = float(kk) if kk > 0 else float("nan")
    overlap = float((base_u[:, :kk].T @ y).pow(2).sum()) / overlap_den

    n_intruders = int((max_cos < epsilon).sum())
    summary = {
        "n_intruders": float(n_intruders),
        "intruder_frac": n_intruders / k if k else float("nan"),
        "mean_max_cos": float(max_cos.mean()),
        "min_max_cos": float(max_cos.min()),
        "subspace_overlap_topk": overlap,
        "mean_best_match_base_index": float(best_idx.to(torch.float64).mean()),
        "max_best_match_base_index": float(best_idx.max()),
    }

    rows: List[Dict[str, Any]] = []
    for j in range(k):
        rows.append({
            "sv_index": j,
            "sigma_tuned": float(s_t[j]),
            "sigma_base": float(base_s[j]) if j < base_s.numel() else float("nan"),
            "max_cos": float(max_cos[j]),
            "best_match_base_index": int(best_idx[j]),
            "is_intruder": int(float(max_cos[j]) < epsilon),
        })
    return summary, rows


# ---------------------------------------------------------------------------------------------
# Per-run driver
# ---------------------------------------------------------------------------------------------


def discover_adapters(state_dict: Dict[str, torch.Tensor]) -> List[Tuple[str, str]]:
    """
    Find every adapted module by parameter-name suffix.

    returns:
        sorted list of (module_prefix, projection) [List[Tuple[str, str]]]
    """
    found = []
    for key in state_dict:
        for proj in PROJECTIONS:
            suffix = f".adapter_{proj}{DOWN_SUFFIX}"
            if key.endswith(suffix):
                found.append((key[: -len(suffix)], proj))

    def _block_index(item: Tuple[str, str]) -> Tuple[int, str]:
        m = re.search(r"blocks\.(\d+)\.", item[0])
        return (int(m.group(1)) if m else 10**6, item[1])

    return sorted(set(found), key=_block_index)


def analyze_checkpoint(
    ckpt_path: Path,
    yaml_path: Optional[Path],
    alpha_override: Optional[float],
    base_cache: Dict[str, float],
    skip_base: bool,
    allow_full_load: bool,
    intruder_cfg: Optional[Dict[str, Any]] = None,
    base_svd_cache: Optional[Dict[str, Tuple[torch.Tensor, torch.Tensor]]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], List[Dict[str, Any]]]:
    """
    Compute per-adapter drift stats for one checkpoint, and optionally its intruder dimensions.

    arguments:
        intruder_cfg [Optional[Dict]]: None disables the intruder pass, else {top_k, epsilon, base_k}
        base_svd_cache [Optional[Dict]]: in-memory cache of W0's left singular vectors, keyed "{prefix}|{projection}"
    returns:
        (per-adapter rows, run-level summary, per-singular-vector intruder rows)
        [Tuple[List[Dict], Dict, List[Dict]]]
    """
    ckpt = load_checkpoint_lazy(ckpt_path, allow_full_load=allow_full_load)
    state_dict = resolve_state_dict(ckpt)
    adapters = discover_adapters(state_dict)
    if not adapters:
        raise RuntimeError(
            f"No LoRA adapters found in {ckpt_path}. Was this run trained with "
            "HE_training_option='lora'?"
        )

    rank_from_shape = int(state_dict[f"{adapters[0][0]}.adapter_{adapters[0][1]}{DOWN_SUFFIX}"].shape[1])
    alpha, scaling, alpha_source = resolve_scaling(ckpt, rank_from_shape, yaml_path, alpha_override)
    # Epoch stored in the checkpoint (the reported epoch is the one of the filename, set in main())
    ckpt_epoch = int(ckpt.get("epoch", -1)) if isinstance(ckpt, dict) else -1

    logger.info(
        "%s: %d adapted matrices, rank=%d, alpha=%g (from %s), scaling=%.4f",
        ckpt_path.name, len(adapters), rank_from_shape, alpha, alpha_source, scaling,
    )

    rows: List[Dict[str, Any]] = []
    intruder_rows: List[Dict[str, Any]] = []
    sum_delta_sq = 0.0
    sum_base_sq = 0.0

    for prefix, proj in adapters:
        down = state_dict[f"{prefix}.adapter_{proj}{DOWN_SUFFIX}"]
        up = state_dict[f"{prefix}.adapter_{proj}{UP_SUFFIX}"]
        stats = delta_spectrum(down, up, scaling)

        cache_key = f"{prefix}|{proj}"
        wkey = f"{prefix}.linear_layer.weight"

        base_fro = float("nan")
        if not skip_base:
            if cache_key not in base_cache:
                if wkey in state_dict:
                    base_cache[cache_key] = base_block_fro(state_dict[wkey], proj)
                else:
                    logger.warning("Frozen base weight %s not in checkpoint; relative drift unavailable.", wkey)
                    base_cache[cache_key] = float("nan")
            base_fro = base_cache[cache_key]

        block_match = re.search(r"blocks\.(\d+)\.", prefix)
        block_idx = int(block_match.group(1)) if block_match else -1
        row = {
            "block": block_idx,
            "projection": proj,
            "rank": rank_from_shape,
            "alpha": alpha,
            "scaling": scaling,
            "delta_fro": stats["fro"],
            "base_fro": base_fro,
            "relative_drift": stats["fro"] / base_fro if base_fro == base_fro and base_fro > 0 else float("nan"),
            "delta_spectral": stats["spectral"],
            "delta_nuclear": stats["nuclear"],
            "eff_rank": stats["eff_rank"],
            "participation_ratio": stats["participation_ratio"],
            "rank99": stats["rank99"],
            "rank_max": stats["rank_max"],
        }

        if intruder_cfg is not None:
            # Intruder fields are always written (NaN if the base weight is missing)
            if wkey not in state_dict:
                logger.warning(
                    "Frozen base weight %s not in checkpoint; intruder dimensions unavailable.", wkey
                )
                row.update({f: float("nan") for f in INTRUDER_SUMMARY_FIELDS})
            else:
                svd_cache = base_svd_cache if base_svd_cache is not None else {}
                blk = base_block_view(state_dict[wkey], proj)
                if cache_key not in svd_cache:
                    svd_cache[cache_key] = base_left_singular(blk, intruder_cfg["base_k"])
                b_u, b_s = svd_cache[cache_key]
                isummary, irows = intruder_stats(
                    base_u=b_u, base_s=b_s, block=blk, down=down, up=up, scaling=scaling,
                    top_k=intruder_cfg["top_k"], epsilon=intruder_cfg["epsilon"],
                )
                row.update(isummary)
                for ir in irows:
                    ir["block"] = block_idx
                    ir["projection"] = proj
                intruder_rows.extend(irows)
                del blk

        rows.append(row)

        sum_delta_sq += stats["fro"] ** 2
        if base_fro == base_fro:
            sum_base_sq += base_fro ** 2

        del down, up  # drop references so mmap pages can be reclaimed

    total_delta = math.sqrt(sum_delta_sq)
    total_base = math.sqrt(sum_base_sq) if sum_base_sq > 0 else float("nan")
    eff_ranks = [r["eff_rank"] for r in rows]
    summary = {
        "ckpt_epoch": ckpt_epoch,
        "n_adapted_matrices": len(rows),
        "rank": rank_from_shape,
        "alpha": alpha,
        "scaling": scaling,
        "alpha_source": alpha_source,
        "total_delta_fro": total_delta,
        "total_base_fro": total_base,
        "total_relative_drift": total_delta / total_base if total_base == total_base and total_base > 0 else float("nan"),
        "mean_eff_rank": sum(eff_ranks) / len(eff_ranks),
        "mean_eff_rank_frac": (sum(eff_ranks) / len(eff_ranks)) / rank_from_shape,
        "mean_rank99": sum(r["rank99"] for r in rows) / len(rows),
        "max_relative_drift": max(
            (r["relative_drift"] for r in rows if r["relative_drift"] == r["relative_drift"]), default=float("nan")
        ),
    }

    if intruder_cfg is not None:
        def _mean(field: str) -> float:
            vals = [r[field] for r in rows if r.get(field) == r.get(field)]  # drop NaN
            return sum(vals) / len(vals) if vals else float("nan")

        summary.update({
            "intruder_top_k": intruder_cfg["top_k"],
            "intruder_epsilon": intruder_cfg["epsilon"],
            "intruder_base_k": intruder_cfg["base_k"],
            "total_intruders": sum(r["is_intruder"] for r in intruder_rows),
            "n_vectors_inspected": len(intruder_rows),
            "mean_n_intruders": _mean("n_intruders"),
            "mean_intruder_frac": _mean("intruder_frac"),
            "mean_max_cos": _mean("mean_max_cos"),
            "mean_subspace_overlap_topk": _mean("subspace_overlap_topk"),
            "mean_best_match_base_index": _mean("mean_best_match_base_index"),
        })

    del state_dict, ckpt
    return rows, summary, intruder_rows


# ---------------------------------------------------------------------------------------------
# Paths & IO
# ---------------------------------------------------------------------------------------------


def run_dir(training_id: str) -> Path:
    return get_full_path("train", "trainings", "trainings_HistoST", training_id, "training", as_str=False)


def default_out_dir() -> Path:
    """Default output folder: trainings_HistoST/analysis/lora_drift/."""
    return get_full_path(
        "train", "trainings", "trainings_HistoST", "analysis", "lora_drift", as_str=False
    )


def checkpoint_path(training_id: str, epoch: int) -> Path:
    return run_dir(training_id) / "log" / "checkpoints" / f"checkpoint_epoch_{epoch}.pth"


def yaml_path_for(training_id: str) -> Path:
    return run_dir(training_id) / f"{training_id}_histost.yaml"


def available_epochs(training_id: str) -> List[int]:
    ckpt_dir = run_dir(training_id) / "log" / "checkpoints"
    if not ckpt_dir.is_dir():
        return []
    epochs = []
    for p in ckpt_dir.glob("checkpoint_epoch_*.pth"):
        if p.name.startswith("."):  # hidden files (e.g. macOS ._*)
            continue
        m = re.match(r"checkpoint_epoch_(-?\d+)\.pth$", p.name)
        if m:
            epochs.append(int(m.group(1)))
    return sorted(epochs)


def write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Wrote per-layer table: %s (%d rows)", path, len(rows))


def write_markdown(summaries: List[Dict[str, Any]], rows: List[Dict[str, Any]], path: Path) -> None:
    """Write the Markdown summary tables."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# LoRA drift from frozen UNI2-h",
        "",
        "`dW = (alpha/rank) * D @ U` per adapted matrix (Q and V of every ViT block).",
        "Relative drift is against the same row-block of the frozen fused qkv weight.",
        "Descriptive diagnostic in the adapted subspace only,",
        "computed from the saved LoRA factors of each checkpoint.",
        "",
        "## Per-run summary",
        "",
        "| run | epoch | rank | alpha | total ||dW||_F | total relative drift | mean eff_rank | eff_rank / rank | mean rank99 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for s in summaries:
        lines.append(
            f"| {s['training_id']} | {s['epoch']} | {s['rank']} | {s['alpha']:g} | "
            f"{s['total_delta_fro']:.4f} | {s['total_relative_drift']:.5f} | "
            f"{s['mean_eff_rank']:.2f} | {s['mean_eff_rank_frac']:.3f} | {s['mean_rank99']:.1f} |"
        )
    blocks = sorted({r["block"] for r in rows})
    lines += [
        "",
        "## Per-block relative drift (mean over Q and V)",
        "",
        "| run | epoch | " + " | ".join(f"b{b}" for b in blocks) + " |",
        "|---" * (2 + len(blocks)) + "|",
    ]
    # One table row per analyzed checkpoint (run, epoch)
    by_cell: Dict[Tuple[str, int, int], List[float]] = defaultdict(list)
    for r in rows:
        if r["relative_drift"] == r["relative_drift"]:  # drop NaN (base norm unavailable)
            by_cell[(r["training_id"], r["epoch"], r["block"])].append(r["relative_drift"])

    for s in summaries:
        cells = []
        for b in blocks:
            vals = by_cell.get((s["training_id"], s["epoch"], b), [])
            cells.append(f"{sum(vals) / len(vals):.4f}" if vals else "-")
        lines.append(f"| {s['training_id']} | {s['epoch']} | " + " | ".join(cells) + " |")

    if summaries and "total_intruders" in summaries[0]:
        top_k = summaries[0]["intruder_top_k"]
        eps = summaries[0]["intruder_epsilon"]
        base_k = summaries[0]["intruder_base_k"]
        lines += [
            "",
            "## Intruder dimensions",
            "",
            f"A left singular vector `y` of the merged `W_tuned = W0 + dW` is an intruder iff "
            f"`max_i |cos(y, x_i)| < {eps:g}` over "
            + ("all" if base_k <= 0 else f"the top-{base_k}")
            + f" left singular vectors `x_i` of `W0` (Shuttleworth et al. 2024, arXiv:2410.21228). "
              f"Top-{top_k:g} vectors of each adapted matrix are inspected.",
            "",
            "`overlap` is the threshold-free companion (top-k subspace energy retained; 1.0 = untouched).",
            "`mean base idx` is the mean rank of the best-matching pretrained direction: a large value",
            "alongside a high `mean max|cos|` means old but low-ranked directions were promoted, which is",
            "not the same as a new direction appearing. Per-vector `max_cos` is in the",
            "companion `lora_drift_intruders_*.csv`, so any epsilon can be recomputed without re-running.",
            "",
            "| run | epoch | rank | intruders | of | per matrix | mean max\\|cos\\| | overlap | mean base idx |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for s in summaries:
            lines.append(
                f"| {s['training_id']} | {s['epoch']} | {s['rank']} | {s['total_intruders']} | "
                f"{s['n_vectors_inspected']} | {s['mean_n_intruders']:.2f} | "
                f"{s['mean_max_cos']:.3f} | {s['mean_subspace_overlap_topk']:.3f} | "
                f"{s['mean_best_match_base_index']:.1f} |"
            )

        # Per-block intruder counts
        lines += [
            "",
            "### Per-block intruder count (summed over Q and V)",
            "",
            "| run | epoch | " + " | ".join(f"b{b}" for b in blocks) + " |",
            "|---" * (2 + len(blocks)) + "|",
        ]
        intr_by_cell: Dict[Tuple[str, int, int], int] = defaultdict(int)
        for r in rows:
            if r.get("n_intruders") == r.get("n_intruders"):  # drop NaN
                intr_by_cell[(r["training_id"], r["epoch"], r["block"])] += int(r["n_intruders"])
        for s in summaries:
            cells = [
                str(intr_by_cell.get((s["training_id"], s["epoch"], b), 0)) for b in blocks
            ]
            lines.append(f"| {s['training_id']} | {s['epoch']} | " + " | ".join(cells) + " |")

    path.write_text("\n".join(lines) + "\n")
    logger.info("Wrote markdown summary: %s", path)


# ---------------------------------------------------------------------------------------------


def main(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir) if args.out_dir else default_out_dir()
    logger.info("Writing outputs to %s", out_dir)

    # --intruder: None (default) = on unless --skip_base; True/False = explicit (--intruder with --skip_base raises)
    if args.intruder is None:
        run_intruder = not args.skip_base
        if args.skip_base:
            logger.info(
                "--skip_base is set, so the intruder pass is auto-disabled (it needs the frozen "
                "base weights). Pass --intruder explicitly to make this a hard error instead."
            )
    else:
        run_intruder = bool(args.intruder)

    intruder_cfg: Optional[Dict[str, Any]] = None
    if run_intruder:
        if args.skip_base:
            raise SystemExit(
                "--intruder needs the frozen base weights (it compares W0 + dW against W0), "
                "so it is incompatible with --skip_base."
            )
        intruder_cfg = {
            "top_k": args.intruder_top_k,
            "epsilon": args.intruder_epsilon,
            "base_k": args.intruder_base_k,
        }
        logger.info(
            "Intruder pass ON: top_k=%d, epsilon=%.2f, base_k=%s.",
            args.intruder_top_k, args.intruder_epsilon,
            "all" if args.intruder_base_k <= 0 else str(args.intruder_base_k),
        )

    all_rows: List[Dict[str, Any]] = []
    all_intruder_rows: List[Dict[str, Any]] = []
    summaries: List[Dict[str, Any]] = []
    base_cache: Dict[str, float] = {}
    base_svd_cache: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}

    cache_path = out_dir / "lora_drift_base_norms.json"
    if not args.skip_base and args.reuse_base_cache and cache_path.is_file():
        base_cache = json.loads(cache_path.read_text())
        logger.info("Reusing %d cached frozen-base norms from %s", len(base_cache), cache_path)

    for training_id in args.training_ids:
        epochs = args.epochs if args.epochs else available_epochs(training_id)
        if not epochs:
            logger.warning("No checkpoints found for %s -- skipping.", training_id)
            continue
        if args.last_only:
            epochs = epochs[-1:]

        for epoch in epochs:
            ckpt = checkpoint_path(training_id, epoch)
            if not ckpt.is_file():
                logger.warning("Missing %s -- skipping.", ckpt)
                continue
            rows, summary, intruder_rows = analyze_checkpoint(
                ckpt_path=ckpt,
                yaml_path=yaml_path_for(training_id),
                alpha_override=args.alpha,
                base_cache=base_cache,
                skip_base=args.skip_base,
                allow_full_load=not args.strict_low_memory,
                intruder_cfg=intruder_cfg,
                base_svd_cache=base_svd_cache,
            )
            for r in rows:
                r["training_id"] = training_id
                r["epoch"] = epoch
            for r in intruder_rows:
                r["training_id"] = training_id
                r["epoch"] = epoch
            all_intruder_rows.extend(intruder_rows)
            summary["training_id"] = training_id
            # Epoch from the filename (same as the rows)
            summary["epoch"] = epoch
            if summary["ckpt_epoch"] not in (epoch, -1):
                logger.warning(
                    "%s: filename says epoch %d but the checkpoint's own counter says %d; "
                    "reporting the filename epoch.", ckpt.name, epoch, summary["ckpt_epoch"],
                )
            all_rows.extend(rows)
            summaries.append(summary)

            logger.info(
                "  %s e%-3d rank=%-3d ||dW||_F=%.4f  rel=%.5f  eff_rank=%.2f/%d (%.0f%%)  rank99=%.1f",
                training_id, epoch, summary["rank"], summary["total_delta_fro"],
                summary["total_relative_drift"], summary["mean_eff_rank"], summary["rank"],
                100 * summary["mean_eff_rank_frac"], summary["mean_rank99"],
            )
            if intruder_cfg is not None:
                logger.info(
                    "       intruders: %d/%d vectors (mean %.2f per matrix)  mean max|cos|=%.3f  "
                    "top-%d subspace overlap=%.3f",
                    summary["total_intruders"], summary["n_vectors_inspected"],
                    summary["mean_n_intruders"], summary["mean_max_cos"],
                    intruder_cfg["top_k"], summary["mean_subspace_overlap_topk"],
                )

    if not all_rows:
        raise SystemExit("No checkpoints analyzed -- nothing to write.")

    tag = "_".join(f"t{re.sub(r'[^0-9-]', '', t)}" for t in args.training_ids)
    write_csv(all_rows, out_dir / f"lora_drift_{tag}.csv")
    if all_intruder_rows:
        # Per-vector max_cos (any epsilon can be applied afterwards)
        write_csv(all_intruder_rows, out_dir / f"lora_drift_intruders_{tag}.csv")
    if not args.skip_base and base_cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(base_cache, indent=2))
    if args.markdown:
        write_markdown(summaries, all_rows, out_dir / f"lora_drift_{tag}.md")

    print("\n" + "=" * 104)
    print("LoRA DRIFT FROM FROZEN UNI2-h  (adapted subspace: Q,V of every ViT block)")
    print("=" * 104)
    print(f"{'run':>12s}{'epoch':>7s}{'rank':>6s}{'alpha':>7s}{'||dW||_F':>11s}"
          f"{'relative':>11s}{'eff_rank':>10s}{'/rank':>8s}{'rank99':>8s}")
    for s in summaries:
        print(f"{s['training_id']:>12s}{s['epoch']:>7d}{s['rank']:>6d}{s['alpha']:>7g}"
              f"{s['total_delta_fro']:>11.4f}{s['total_relative_drift']:>11.5f}"
              f"{s['mean_eff_rank']:>10.2f}{s['mean_eff_rank_frac']:>8.3f}{s['mean_rank99']:>8.1f}")

    if intruder_cfg is not None:
        print("\n" + "=" * 104)
        print(f"INTRUDER DIMENSIONS  (left singular vectors of W0+dW vs W0; "
              f"top-{intruder_cfg['top_k']}, eps={intruder_cfg['epsilon']:g})")
        print("=" * 104)
        print(f"{'run':>12s}{'epoch':>7s}{'rank':>6s}{'intruders':>11s}{'of':>7s}"
              f"{'per-mat':>9s}{'mean|cos|':>11s}{'overlap':>9s}{'mean idx':>10s}")
        for s in summaries:
            print(f"{s['training_id']:>12s}{s['epoch']:>7d}{s['rank']:>6d}"
                  f"{s['total_intruders']:>11d}{s['n_vectors_inspected']:>7d}"
                  f"{s['mean_n_intruders']:>9.2f}{s['mean_max_cos']:>11.3f}"
                  f"{s['mean_subspace_overlap_topk']:>9.3f}{s['mean_best_match_base_index']:>10.1f}")
        print("\n'per-mat' = mean intruders per adapted matrix (of top-"
              f"{intruder_cfg['top_k']}); 'overlap' = top-k subspace energy retained (1.0 = untouched);")
        print("'mean idx' = mean rank of the best-matching pretrained direction -- a large value with a")
        print("high mean|cos| means old-but-low-ranked directions were promoted, not that new ones appeared.")

    peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_mb = peak_kb / (1024 * 1024) if sys.platform == "darwin" else peak_kb / 1024
    print(f"\npeak resident memory: {peak_mb:.0f} MB")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(
        description=(
            "Measure how far a HistoST LoRA finetune moved the H&E backbone from frozen UNI2-h (read-only, CPU)."
        )
    )
    parser.add_argument(
        "--training_ids", type=str, nargs="+", required=True,
        help="HistoST training ids to analyze, e.g. --training_ids training_40 training_42 training_46",
    )
    parser.add_argument(
        "--epochs", type=int, nargs="*", default=None,
        help="Checkpoint epochs to analyze. Default: every checkpoint present for each run.",
    )
    parser.add_argument(
        "--last_only", action="store_true",
        help="Analyze only the highest available epoch per run.",
    )
    parser.add_argument(
        "--alpha", type=float, default=None,
        help="Override lora_alpha (default: read from the checkpoint config or the run YAML).",
    )
    parser.add_argument(
        "--skip_base", action="store_true",
        help="Skip the frozen-base qkv norms (absolute ||dW||_F only; also disables the intruder pass).",
    )
    parser.add_argument(
        "--reuse_base_cache", action=argparse.BooleanOptionalAction, default=True,
        help="Reuse the cached frozen-base norms (lora_drift_base_norms.json); --no-reuse_base_cache recomputes them.",
    )
    parser.add_argument(
        "--strict_low_memory", action="store_true",
        help="Fail instead of falling back to a full (non-mmap) checkpoint load.",
    )
    parser.add_argument(
        "--out_dir", type=str, default=None,
        help="Output directory (default: train/trainings/trainings_HistoST/analysis/lora_drift/ under ROOT_PATH).",
    )
    parser.add_argument(
        "--markdown", action="store_true",
        help="Also write a markdown summary next to the CSV.",
    )
    parser.add_argument(
        "--intruder", action=argparse.BooleanOptionalAction, default=None,
        help="Intruder-dimension pass (Shuttleworth et al. 2024, arXiv:2410.21228): compare the left singular "
             "vectors of W0+dW with those of W0. On by default (off with --skip_base); --no-intruder skips it.",
    )
    parser.add_argument(
        "--intruder_top_k", type=int, default=10,
        help="Number of leading singular vectors of each merged matrix to classify (default: 10).",
    )
    parser.add_argument(
        "--intruder_epsilon", type=float, default=0.6,
        help="Similarity threshold below which a vector counts as an intruder (default: 0.6).",
    )
    parser.add_argument(
        "--intruder_base_k", type=int, default=0,
        help="Compare against only the top-k singular vectors of W0 (default: 0 = all; a positive k can only "
             "increase the intruder count).",
    )

    main(parser.parse_args())
