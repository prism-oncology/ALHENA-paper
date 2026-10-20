#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pretrain the Novae ST-graph teacher (novae v1.1.1) with scConcept cell embeddings as node features
(embedding_name="X_scConcept"), on the train cells of the project Xenium slides and the public Novae cohort slides.
Runs in the `novae` env.

Saved in <ROOT_HistoST> / train / trainings / trainings_Novae / <training_id> /
    - manifest.json: config, scConcept model and cohort
    - checkpoints/: Lightning checkpoints (monitor metrics/val_mean_fide_score) and last.ckpt
    - figures/: connectivities_radius<radius>.png, domains_n<n>.png
    - results_h5ad/<slide_id>.h5ad: own slides with representations and domains
    - validation/: metrics.json (FIDE / JSD / MNE per n_domains) and validation_report.md
    - run.log, wandb/
"""

import os

os.environ.setdefault("WANDB_MODE", "offline")  # W&B offline by default

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import h5py
import numpy as np
import scanpy as sc
import yaml
from anndata import AnnData

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add the project root to sys.path so that 'src' can be imported
project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.constants import get_full_path
from src.utils.cell_ids import int_cell_id_vec

# novae v1.1.1 public API only
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

import novae
from novae.monitor import jensen_shannon_divergence, mean_fide_score, mean_normalized_entropy
from novae.monitor.callback import (
    LogProtoCovCallback,
    LogTissuePrototypeWeights,
    PrototypeUMAPCallback,
    ValidationCallback,
)

logger = logging.getLogger(__name__)

OBSM_KEY = "X_scConcept"          # per-cell scConcept node feature
SLIDE_KEY = "slide_id"            # display slide-id column; passed to novae as `slide_key`
EMBEDDING_DIM = 512               # corpus40M-model30M
DEFAULT_COHORT_ID = "human_cellbased_nobrain_scc40M30M_1"


# --------------------------------------------------------------------------- logging / io


def setup_logging(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(run_dir / "run.log")],
        force=True,
    )


def load_json(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def load_train_cell_ids(npz_path: Path) -> Dict[str, np.ndarray]:
    """{slide_id: int64 train-only cell ids} from `cell_ids_sets.npz` (key `train_only_by_slide`)."""
    if not npz_path.is_file():
        raise FileNotFoundError(f"Missing cell_ids_sets npz at: {npz_path}")
    data = np.load(npz_path, allow_pickle=True)
    if "train_only_by_slide" not in data:
        raise KeyError(f"'train_only_by_slide' not in {npz_path}; keys={list(data.keys())}")
    mapping = data["train_only_by_slide"].item()
    return {str(sid): np.asarray(arr).astype(np.int64) for sid, arr in mapping.items()}


def list_h5ad(dir_path: Path) -> Dict[str, Path]:
    """{stem: path} for *.h5ad files, skipping macOS ._* files."""
    out: Dict[str, Path] = {}
    for p in sorted(dir_path.glob("*.h5ad")):
        if p.name.startswith("._"):
            continue
        out[p.stem] = p
    return out


# --------------------------------------------------------------------------- scConcept model check


def read_own_scconcept_meta(own_training_id: str) -> dict:
    """Top-level provenance from `trainings_scConcept/{id}/manifest.json`."""
    mp = get_full_path("train", "trainings", "trainings_scConcept", own_training_id, "manifest.json")
    if not mp.is_file():
        raise FileNotFoundError(f"scConcept manifest not found: {mp}")
    m = load_json(mp)
    return {
        "scconcept_training_id": own_training_id,
        "model_name": m.get("model_name"),
        "scConcept_version": m.get("scConcept_version"),
        "species": m.get("species"),
        "adapt_steps": m.get("adapt_steps"),
        "manifest_path": str(mp),
    }


def read_public_scconcept_block(cohort_dir: Path) -> dict:
    """scConcept block of the public cohort manifest (written by _1_scConcept_embs_novae_public.py)."""
    mp = cohort_dir / "manifest.json"
    if not mp.is_file():
        raise FileNotFoundError(f"Public cohort manifest not found: {mp}")
    block = load_json(mp).get("scConcept")
    if not block:
        raise RuntimeError(
            f"Public cohort manifest {mp} has no 'scConcept' block. "
            "Run _1_scConcept_embs_novae_public.py first."
        )
    return block


def verify_scconcept_model_match(own: dict, public: dict) -> None:
    """Check that the own and public scConcept embeddings come from the same model (name, version, species, adapt_steps=0, dim)."""
    for k in ("model_name", "scConcept_version", "species"):
        if own.get(k) != public.get(k):
            raise RuntimeError(
                f"scConcept model mismatch on '{k}': own (training={own['scconcept_training_id']}) "
                f"={own.get(k)!r} vs public={public.get(k)!r}. Fix --own_scconcept_training_id."
            )
    if int(own.get("adapt_steps", -1)) != 0 or int(public.get("adapt_steps", -1)) != 0:
        raise RuntimeError(
            f"scConcept must be raw (adapt_steps==0); got own={own.get('adapt_steps')} "
            f"public={public.get('adapt_steps')}."
        )
    pdim = int(public.get("embedding_dim", -1))
    if pdim != EMBEDDING_DIM:
        raise RuntimeError(f"public scConcept embedding_dim={pdim} != {EMBEDDING_DIM}.")
    logger.info(
        "scConcept model-match OK: model=%s version=%s species=%s adapt_steps=0 dim=%d",
        own.get("model_name"), own.get("scConcept_version"), own.get("species"), EMBEDDING_DIM,
    )


# --------------------------------------------------------------------------- cohort assembly


def _attach_scconcept_obsm(adata: AnnData, cell_int: np.ndarray, scc_h5_path: Path) -> None:
    """Attach obsm['X_scConcept'] to adata (aligned on cell_id_int) from a cell-branch <slide>.h5 file."""
    with h5py.File(scc_h5_path, "r") as f:
        if "cell_id_int" not in f or "cls_emb" not in f:
            raise KeyError(f"Invalid scConcept h5 (missing 'cell_id_int'/'cls_emb'): {scc_h5_path}")
        cid = f["cell_id_int"][:].astype(np.int64)
        if cid.size > 1 and np.any(cid[1:] < cid[:-1]):
            raise ValueError(f"'cell_id_int' must be sorted ascending in {scc_h5_path}")
        emb_dim = int(f.attrs["embedding_dim"]) if "embedding_dim" in f.attrs else int(f["cls_emb"].shape[1])
        if emb_dim != EMBEDDING_DIM:
            raise ValueError(f"scConcept embedding_dim={emb_dim} != {EMBEDDING_DIM} in {scc_h5_path}")

        idx = np.searchsorted(cid, cell_int)
        idx_clip = np.clip(idx, 0, cid.shape[0] - 1)
        ok = (idx < cid.shape[0]) & (cid[idx_clip] == cell_int)
        if not np.all(ok):
            missing = cell_int[~ok]
            raise KeyError(
                f"{missing.size} cell_id_int not found in {scc_h5_path.name}; "
                f"e.g. {missing[:5].tolist()}"
            )
        uniq, inv = np.unique(idx.astype(np.int64), return_inverse=True)
        emb = f["cls_emb"][uniq][inv]  # (N, 512), float16 as stored (L2-normalized)
    adata.obsm[OBSM_KEY] = np.asarray(emb, dtype=np.float16)


def load_own_train_adata(sid: str, h5ad_path: Path, train_only_arr: np.ndarray, scc_h5_path: Path) -> AnnData:
    """Own Xenium slide restricted to TRAIN-only cells, with scConcept obsm attached."""
    ad = sc.read_h5ad(h5ad_path)
    if not ad.obs_names.is_unique:
        ad.obs_names_make_unique()
    if "cell_id" not in ad.obs:
        raise KeyError(f"{sid}: adata.obs['cell_id'] missing (needed for the scConcept join).")
    if "spatial" not in ad.obsm:
        raise KeyError(f"{sid}: adata.obsm['spatial'] missing (needed for spatial_neighbors).")

    cell_int = int_cell_id_vec(ad.obs["cell_id"].astype(str).values)
    train_only_arr = np.sort(np.asarray(train_only_arr, dtype=np.int64))
    keep = np.isin(cell_int, train_only_arr)
    n_keep = int(keep.sum())
    if n_keep == 0:
        raise ValueError(f"{sid}: 0 train-only cells after masking.")
    ad = ad[keep].copy()
    cell_int = cell_int[keep]

    _attach_scconcept_obsm(ad, cell_int, scc_h5_path)
    ad.obs[SLIDE_KEY] = str(sid)
    # Drop the stored graphs (rebuilt by spatial_neighbors)
    for k in list(ad.obsp.keys()):
        del ad.obsp[k]
    logger.info("OWN  %-14s train-only cells kept=%d (of %d)", sid, n_keep, keep.size)
    return ad


def load_public_adata(stem: str, h5ad_path: Path) -> AnnData:
    """Public MERSCOPE/CosMx slide; obsm['X_scConcept'] must already be present."""
    ad = sc.read_h5ad(h5ad_path)
    if not ad.obs_names.is_unique:
        ad.obs_names_make_unique()
    if OBSM_KEY not in ad.obsm:
        raise RuntimeError(
            f"PUBLIC {stem}: obsm['{OBSM_KEY}'] missing. "
            "Run _1_scConcept_embs_novae_public.py."
        )
    d = int(ad.obsm[OBSM_KEY].shape[1])
    if d != EMBEDDING_DIM:
        raise ValueError(f"PUBLIC {stem}: obsm['{OBSM_KEY}'] dim={d} != {EMBEDDING_DIM}.")
    if "spatial" not in ad.obsm:
        raise KeyError(f"PUBLIC {stem}: adata.obsm['spatial'] missing.")
    if SLIDE_KEY not in ad.obs:
        ad.obs[SLIDE_KEY] = str(stem)
    for k in list(ad.obsp.keys()):
        del ad.obsp[k]
    logger.info("PUB  %-32s cells=%d", stem, ad.n_obs)
    return ad


# --------------------------------------------------------------------------- config


def load_config(config_path: Path, training_id: str) -> dict:
    """Load the Novae YAML config (project / model_kwargs / fit_kwargs / post_training); the W&B run name is set to training_id."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f) or {}
    cfg.setdefault("project", "novae_pretraining_st")
    cfg.setdefault("model_kwargs", {})
    cfg.setdefault("fit_kwargs", {})
    cfg.setdefault("post_training", {})
    cfg.setdefault("wandb_init_kwargs", {})
    cfg["wandb_init_kwargs"]["name"] = training_id
    return cfg


# --------------------------------------------------------------------------- figures / post-training


def save_connectivities(train_adatas: List[AnnData], out_path: Path, max_slides: int) -> None:
    """Save the graph QC figure (at most max_slides evenly spaced slides)."""
    ad = train_adatas
    if len(ad) > max_slides:
        idx = sorted(set(np.linspace(0, len(ad) - 1, max_slides).round().astype(int).tolist()))
        ad = [train_adatas[i] for i in idx]
    novae.plot.connectivities(ad, show=False)  # novae 1.1.1 has no path_save=
    plt.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close("all")
    logger.info("Saved connectivities figure (%d slides) -> %s", len(ad), out_path)


def post_training(model, own_adatas: List[AnnData], run_dirs: dict, hw: dict, n_domains_list: List[int], post_cfg: dict) -> dict:
    """Post-training on the own slides: representations, domains, FIDE/JSD/MNE and domain maps."""
    logger.info("Post-training: computing representations on %d own slides ...", len(own_adatas))
    model.compute_representations(own_adatas, accelerator=hw["accelerator"], num_workers=hw["num_workers"])
    try:
        model.swav_head.hierarchical_clustering()  # explicit (also lazily triggered by assign_domains)
    except Exception as e:  # noqa: BLE001
        logger.warning("hierarchical_clustering() raised (continuing via lazy path): %s", e)

    metrics: dict = {}
    for n in n_domains_list:
        try:
            obs_key = model.assign_domains(own_adatas, n_domains=n)
        except Exception:  # noqa: BLE001 -- no tree level yields exactly n domains
            logger.warning("assign_domains(n_domains=%s) failed; falling back to level=%s", n, n)
            obs_key = model.assign_domains(own_adatas, level=n)

        jsd = float(jensen_shannon_divergence(own_adatas, obs_key))
        fide = float(mean_fide_score(own_adatas, obs_key, n_classes=n))
        mne = float(mean_normalized_entropy(own_adatas, n_classes=n, obs_key=obs_key))
        metrics[str(n)] = {"obs_key": obs_key, "fide": fide, "jsd": jsd, "mne": mne, "heuristic": fide * mne}
        logger.info("[n_domains=%d] FIDE=%.4f JSD=%.4f MNE=%.4f", n, fide, jsd, mne)

        novae.plot.domains(own_adatas, obs_key, slide_name_key=SLIDE_KEY, show=False)
        fig_path = run_dirs["figures"] / f"domains_n{n}.png"
        plt.savefig(fig_path, bbox_inches="tight", dpi=200)
        plt.close("all")
        logger.info("Saved domain map -> %s", fig_path)

    if post_cfg.get("save_h5ad", True):
        for ad in own_adatas:
            stem = str(ad.obs[SLIDE_KEY].iloc[0])
            if post_cfg.get("delete_X", True):
                try:
                    del ad.X
                except Exception:  # noqa: BLE001
                    ad.X = None
                if "counts" in ad.layers:
                    del ad.layers["counts"]
            ad.write_h5ad(run_dirs["results"] / f"{stem}.h5ad")
        logger.info("Saved %d post-training h5ad(s) -> %s", len(own_adatas), run_dirs["results"])
    return metrics


# --------------------------------------------------------------------------- manifest / report


def write_manifest(run_dirs, args, cfg, own_meta, public_block, own_slide_ids, public_slide_ids, val_slide_id) -> None:
    manifest = {
        "novae_training_id": args.novae_training_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "created_by": "_2_novae_pretraining.py",
        "novae_version": novae.__version__,
        "node_feature_regime": f"embedding_name={OBSM_KEY}",
        "seed": args.seed,
        "cohort_id": args.cohort_id,
        "graph": {
            "method": "novae.spatial_neighbors (Delaunay, generic)",
            "radius_um": float(args.radius),
            "edge_attr": "micron_distance / 20 (Nums.CELLS_CHARACTERISTIC_DISTANCE)",
            "slide_key": SLIDE_KEY,
        },
        "model_kwargs": cfg["model_kwargs"],
        "fit_kwargs": cfg["fit_kwargs"],
        "post_training": cfg["post_training"],
        "scConcept": {
            "own_scconcept_training_id": args.own_scconcept_training_id,
            "model_name": own_meta.get("model_name"),
            "scConcept_version": own_meta.get("scConcept_version"),
            "species": own_meta.get("species"),
            "adapt_steps": own_meta.get("adapt_steps"),
            "embedding_size": EMBEDDING_DIM,
            "model_match_verified": True,
            "own_manifest": own_meta.get("manifest_path"),
            "public_cohort_block": public_block,
        },
        "cohort": {
            "own_train_cell_provenance": get_full_path(
                "train", "training_datasets", args.dataset_id, args.training_dataset_id,
                "cell_ids_sets.npz", as_str=True,
            ) + " :: train_only_by_slide",
            "own_slides": own_slide_ids,
            "n_own_slides": len(own_slide_ids),
            "val_slide_id": val_slide_id,
            "public_slides": public_slide_ids,
            "n_public_slides": len(public_slide_ids),
        },
    }
    out = run_dirs["root"] / "manifest.json"
    with open(out, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Wrote manifest -> %s", out)


def write_validation_report(run_dirs, args, metrics, own_slide_ids, val_slide_id, n_domains_list) -> None:
    vp = run_dirs["validation"]
    with open(vp / "metrics.json", "w") as f:
        json.dump({"n_domains": n_domains_list, "own_slides": own_slide_ids, "metrics": metrics}, f, indent=2)

    lines = [
        f"# Novae teacher validation — {args.novae_training_id}",
        "",
        f"- novae_version: {novae.__version__}",
        f"- cohort_id: {args.cohort_id}",
        f"- radius: {args.radius} µm | validation slide: {val_slide_id}",
        f"- own slides (n={len(own_slide_ids)}): {', '.join(own_slide_ids)}",
        "",
        "## Domain metrics on the project slides (higher FIDE, lower MNE ≈ crisper spatial domains)",
        "",
        "| n_domains | FIDE | JSD | MNE | heuristic (FIDE·MNE) |",
        "|---|---|---|---|---|",
    ]
    for n in n_domains_list:
        m = metrics[str(n)]
        lines.append(f"| {n} | {m['fide']:.4f} | {m['jsd']:.4f} | {m['mne']:.4f} | {m['heuristic']:.4f} |")
    lines += [
        "",
        "## Domain maps",
        "",
    ]
    lines += [f"- n_domains={n}: `figures/domains_n{n}.png`" for n in n_domains_list]
    lines += [
        "",
        "## Checklist",
        "",
        "- [ ] Domain maps on breast/kidney/skin/… show coherent, spatially-contiguous niches",
        "- [ ] FIDE/JSD/MNE in a reasonable range",
        "- [ ] No degenerate single-domain collapse",
        "- Result: PASS / FAIL — __________ (name, date)",
        "",
    ]
    (vp / "validation_report.md").write_text("\n".join(lines))
    logger.info("Wrote validation report -> %s", vp / "validation_report.md")


# --------------------------------------------------------------------------- checkpoint selection


def _select_best_model(model, ckpt_cb, run_dirs):
    """After fit, load the checkpoint selected by `metrics/val_mean_fide_score` for post-training."""
    best = getattr(ckpt_cb, "best_model_path", "") or ""
    path = None
    if best and os.path.exists(best):
        path = best
        logger.info("Loading best checkpoint (val_mean_fide_score) for post-training: %s", best)
    else:
        last = run_dirs["checkpoints"] / "last.ckpt"
        if last.is_file():
            path = str(last)
            logger.warning("No best_model_path recorded; using last.ckpt: %s", last)
    if path is None:
        logger.warning("No checkpoint found; using in-memory (last-epoch) model for post-training.")
        return model
    m = novae.Novae.load_from_checkpoint(path, map_location="cpu", strict=False)
    m.mode.as_pretrained()  # trained=True, pretrained=True (vanilla Mode API)
    return m


def _load_checkpoint_for_post(run_dirs, ckpt_name):
    ckpt = run_dirs["checkpoints"] / (ckpt_name or "last.ckpt")
    if not ckpt.is_file():
        raise FileNotFoundError(f"--post_only: checkpoint not found: {ckpt}")
    logger.info("post_only: loading %s", ckpt)
    m = novae.Novae.load_from_checkpoint(str(ckpt), map_location="cpu", strict=False)
    m.mode.as_pretrained()
    return m


# --------------------------------------------------------------------------- main


def main(args):
    training_id = args.novae_training_id
    run_root = get_full_path("train", "trainings", "trainings_Novae", training_id)
    run_dirs = {
        "root": run_root,
        "checkpoints": run_root / "checkpoints",
        "figures": run_root / "figures",
        "results": run_root / "results_h5ad",
        "validation": run_root / "validation",
        "wandb": run_root / "wandb",
    }

    # ---- per-run config location
    # {training_id}_novae.yaml in the run folder (copied from configs/template_novae_config.yaml), or --config
    if args.config:
        config_path = Path(args.config)
        if not config_path.is_absolute():
            config_path = project_root / config_path
    else:
        config_path = run_root / f"{training_id}_novae.yaml"

    # Refuse to overwrite a previous run (manifest.json or checkpoints present)
    prior_manifest = (run_root / "manifest.json").exists()
    prior_ckpts = run_dirs["checkpoints"].exists() and any(run_dirs["checkpoints"].iterdir())
    if (prior_manifest or prior_ckpts) and not (args.post_only or args.overwrite):
        raise SystemExit(
            f"Run dir already has outputs (manifest.json / checkpoints): {run_root}\n"
            "Use a new --novae_training_id, or pass --overwrite / --post_only."
        )

    if not config_path.is_file():
        template = project_root / "configs" / "template_novae_config.yaml"
        raise SystemExit(
            f"Per-run Novae config not found: {config_path}\n"
            "Copy the template into the run folder (and edit it if needed), e.g.:\n"
            f"  mkdir -p {run_root}\n"
            f"  cp {template} {config_path}\n"
            "(or pass --config <path>)."
        )

    for d in run_dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    setup_logging(run_root)

    # Seed everything (lightning)
    L.seed_everything(args.seed, workers=True)

    cfg = load_config(config_path, training_id)  # per-run config (read only)
    model_kwargs = dict(cfg["model_kwargs"])
    fit_kwargs = dict(cfg["fit_kwargs"])
    post_cfg = dict(cfg["post_training"])
    n_domains_list = list(post_cfg.get("n_domains", [15, 20, 25]))
    hw = {"accelerator": fit_kwargs.get("accelerator", "cpu"), "num_workers": fit_kwargs.get("num_workers", 0)}
    logger.info("Config: %s", config_path)

    if model_kwargs.get("embedding_name") != OBSM_KEY:
        raise ValueError(f"config model_kwargs.embedding_name must be '{OBSM_KEY}' (scConcept regime).")
    if int(model_kwargs.get("embedding_size", -1)) != EMBEDDING_DIM:
        raise ValueError(f"config model_kwargs.embedding_size must be {EMBEDDING_DIM}.")

    # ---- scConcept model check (own vs public)
    own_meta = read_own_scconcept_meta(args.own_scconcept_training_id)
    cohort_dir = get_full_path("train", "pretraining_datasets", "novae_public", args.cohort_id)
    public_block = read_public_scconcept_block(cohort_dir)
    verify_scconcept_model_match(own_meta, public_block)

    # ---- assemble cohort
    logger.info("Assembling cohort (own train-cells + public) ...")
    npz = get_full_path("train", "training_datasets", args.dataset_id, args.training_dataset_id, "cell_ids_sets.npz")
    train_cells = load_train_cell_ids(npz)
    available_own = list_h5ad(get_full_path("STHELAR_HistoST", "adata_graph"))
    scc_emb_dir = get_full_path("train", "trainings", "trainings_scConcept", args.own_scconcept_training_id, "embeddings")

    own_slide_ids = sorted(set(train_cells) & set(available_own))
    if not own_slide_ids:
        raise RuntimeError("No overlap between train_only_by_slide and adata_graph/*.h5ad.")
    # Skip slides with no train-only cells (held out entirely to valid/test)
    empty_own = [sid for sid in own_slide_ids if np.asarray(train_cells[sid]).size == 0]
    if empty_own:
        logger.info(
            "Skipping %d own slide(s) with 0 train-only cells (held out to valid/test): %s",
            len(empty_own), ", ".join(empty_own),
        )
    if str(args.val_slide_id) in empty_own:
        raise ValueError(
            f"Validation slide {args.val_slide_id} has 0 train-only cells (held out entirely to "
            "valid/test). Pass a --val_slide_id that has train-only cells."
        )
    own_slide_ids = [sid for sid in own_slide_ids if sid not in set(empty_own)]
    if not own_slide_ids:
        raise RuntimeError("All overlapping own slides have 0 train-only cells; nothing to load.")
    own_adatas = []
    for sid in own_slide_ids:
        scc_h5 = scc_emb_dir / f"{sid}.h5"
        if not scc_h5.is_file():
            raise FileNotFoundError(f"scConcept embeddings missing for own slide {sid}: {scc_h5}")
        own_adatas.append(load_own_train_adata(sid, available_own[sid], train_cells[sid], scc_h5))

    public_files = list_h5ad(cohort_dir)
    if not public_files:
        raise RuntimeError(f"No public h5ad in {cohort_dir}.")
    public_adatas = [load_public_adata(stem, p) for stem, p in public_files.items()]
    public_slide_ids = sorted(public_files.keys())

    # ---- split validation slide (own, single)
    val_sid = str(args.val_slide_id)
    val_adatas = [a for a in own_adatas if str(a.obs[SLIDE_KEY].iloc[0]) == val_sid]
    if not val_adatas:
        raise ValueError(f"Validation slide {val_sid} not among own slides {own_slide_ids}.")
    train_own = [a for a in own_adatas if str(a.obs[SLIDE_KEY].iloc[0]) != val_sid]
    train_adatas = train_own + public_adatas
    logger.info(
        "Cohort: %d own-train + %d public = %d train slides; val=%s",
        len(train_own), len(public_adatas), len(train_adatas), val_sid,
    )

    for a in train_adatas + val_adatas:  # also checked by novae
        if OBSM_KEY not in a.obsm or a.obsm[OBSM_KEY].shape[1] != EMBEDDING_DIM:
            raise RuntimeError(f"Missing/!={EMBEDDING_DIM}-d obsm['{OBSM_KEY}'] on slide {a.obs[SLIDE_KEY].iloc[0]}")

    # ---- fresh graphs (novae_sid = slide id)
    novae.settings.auto_preprocessing = False
    logger.info("Building fresh radius=%.1f µm Delaunay graphs (slide_key=%s) ...", args.radius, SLIDE_KEY)
    novae.spatial_neighbors(train_adatas, slide_key=SLIDE_KEY, radius=float(args.radius))
    novae.spatial_neighbors(val_adatas, slide_key=SLIDE_KEY, radius=float(args.radius))

    save_connectivities(
        train_adatas, run_dirs["figures"] / f"connectivities_radius{int(args.radius)}.png", args.connectivities_max_slides
    )

    # ---- manifest (written before training)
    write_manifest(run_dirs, args, cfg, own_meta, public_block, own_slide_ids, public_slide_ids, val_sid)

    # ---- logger + callbacks (saved under the run folder)
    wandb_logger = WandbLogger(save_dir=str(run_dirs["wandb"]), project=cfg["project"], name=training_id, log_model=False)
    try:
        import pandas as pd

        flat = pd.json_normalize(
            {"model_kwargs": model_kwargs, "fit_kwargs": fit_kwargs, "cohort_id": args.cohort_id}, sep="."
        ).to_dict("records")[0]
        wandb_logger.experiment.config.update(flat, allow_val_change=True)
    except Exception as e:  # noqa: BLE001
        logger.warning("Could not push flat config to wandb: %s", e)

    ckpt_cb = ModelCheckpoint(
        dirpath=str(run_dirs["checkpoints"]),
        monitor="metrics/val_mean_fide_score",
        mode="max",
        save_last=True,
        save_top_k=5,
    )
    callbacks = [
        ValidationCallback(val_adatas, accelerator=hw["accelerator"], num_workers=hw["num_workers"], slide_name_key=SLIDE_KEY),
        ckpt_cb,
    ]
    if not args.no_extra_callbacks:
        callbacks += [LogProtoCovCallback(), LogTissuePrototypeWeights(), PrototypeUMAPCallback()]

    own_report_adatas = train_own + val_adatas  # own slides for the validation report

    # ---- train (or post-only)
    if args.post_only:
        model = _load_checkpoint_for_post(run_dirs, args.ckpt)
    else:
        logger.info("Instantiating Novae (embedding regime) + init_slide_queue ...")
        model = novae.Novae(**model_kwargs)  # embedding_name -> cell_embedder=None, no preprocessing
        model.init_slide_queue(train_adatas, model.hparams.min_prototypes_ratio)
        logger.info("Fitting Novae: %s", fit_kwargs)
        model.fit(train_adatas, callbacks=callbacks, logger=wandb_logger, **fit_kwargs)
        model = _select_best_model(model, ckpt_cb, run_dirs)

    metrics = post_training(model, own_report_adatas, run_dirs, hw, n_domains_list, post_cfg)
    write_validation_report(
        run_dirs, args, metrics, [str(a.obs[SLIDE_KEY].iloc[0]) for a in own_report_adatas], val_sid, n_domains_list
    )
    logger.info("DONE. Outputs under %s", run_root)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Novae v1.1.1 teacher pretraining with scConcept node features.")

    p.add_argument("--novae_training_id", type=str, default="training_4",
                   help="trainings_Novae id (must be new, unless --overwrite or --post_only).")
    p.add_argument("--cohort_id", type=str, default=DEFAULT_COHORT_ID,
                   help="Public cohort id under train/pretraining_datasets/novae_public/.")
    p.add_argument("--dataset_id", type=str, default="ds_ct1_20x_224_o112",
                   help="dataset_id locating cell_ids_sets.npz.")
    p.add_argument("--training_dataset_id", type=str, default="ds_ct1_20x_224_o112_1",
                   help="training_dataset_id locating cell_ids_sets.npz (train_only_by_slide).")
    p.add_argument("--own_scconcept_training_id", type=str, default="training_7",
                   help="trainings_scConcept id supplying own-slide obsm['X_scConcept'] (must match the public cohort model).")
    p.add_argument("--val_slide_id", type=str, default="skin_s4",
                   help="Own slide reserved for Novae validation (single-slide ValidationCallback).")
    p.add_argument("--radius", type=float, default=100.0,
                   help="Delaunay edge-pruning radius in µm for the teacher pretraining topology.")
    p.add_argument("--config", type=str, default=None,
                   help="Novae YAML config (default: trainings_Novae/{training_id}/{training_id}_novae.yaml, "
                        "copied from configs/template_novae_config.yaml).")
    p.add_argument("--seed", type=int, default=42, help="Global seed.")
    p.add_argument("--connectivities_max_slides", type=int, default=12,
                   help="Max slides drawn in the connectivities QC figure.")
    p.add_argument("--no_extra_callbacks", action="store_true",
                   help="Disable the extra wandb-figure callbacks (proto covariance / tissue weights / prototype UMAP).")
    p.add_argument("--post_only", action="store_true",
                   help="Skip training; load a checkpoint and only run post-training (representations, domains, metrics).")
    p.add_argument("--ckpt", type=str, default=None,
                   help="Checkpoint filename under checkpoints/ for --post_only (default last.ckpt).")
    p.add_argument("--overwrite", action="store_true",
                   help="Allow writing into an existing run dir that has outputs.")

    main(p.parse_args())
