#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute the scConcept embedding of each cell of the public Novae cohort slides, with the raw scConcept model
(adapt_steps=0) of the cell-branch teacher, following https://prism-oncology.github.io/novae/tutorials/novae_scConcept/.
Runs in the `sc_concept_env2` env.

Saved in <ROOT_HistoST> / train / pretraining_datasets / novae_public / <cohort_id> /
    - <slide_id>.h5ad: embeddings in obsm["X_scConcept"], corpus gene IDs in var[<gene_id_column>]
    - manifest.json: scConcept block (model, extraction parameters, vocabulary coverage per slide)
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

# Add the project root to sys.path so that 'src' can be imported
project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.constants import get_full_path  # noqa: E402

logger = logging.getLogger(__name__)

# corpus40M-model30M produces a 512-dim cls embedding (the 170M model would be 1024).
DEFAULT_EXPECTED_EMBEDDING_DIM: int = 512
OBSM_KEY: str = "X_scConcept"  # the obsm key Novae reads (embedding_name="X_scConcept")


# --- scConcept training provenance (read the teacher's manifest, assert adapt_steps == 0) ----


def validate_scconcept_training_manifest(
    m: dict,
    scconcept_training_id: str,
    *,
    expect_adapt_steps_zero: bool = True,
) -> dict:
    """
    Validate a scConcept training manifest.json (written by v2_scConcept_cellEmb.py) and return the fields used here.
    Raises if adapt_steps != 0 (unless expect_adapt_steps_zero is False).
    """
    if "adapt_steps" not in m:
        raise KeyError(
            f"scConcept training manifest for '{scconcept_training_id}' has no 'adapt_steps' key "
            f"(adapt_steps == 0 cannot be checked). Is this a v2 scConcept run?"
        )
    adapt_steps = int(m["adapt_steps"])
    if expect_adapt_steps_zero and adapt_steps != 0:
        raise ValueError(
            f"scConcept training_id='{scconcept_training_id}' has adapt_steps={adapt_steps} (!= 0); the Novae "
            f"public-cohort embeddings require the raw model. Pick an adapt_steps=0 training_id "
            f"(e.g. training_7), or pass --allow_adapted to override."
        )
    if "model_name" not in m:
        raise KeyError(f"scConcept training manifest for '{scconcept_training_id}' has no 'model_name'.")

    return {
        "scconcept_training_id": scconcept_training_id,
        "scConcept_version": m.get("scConcept_version"),
        "model_name": str(m["model_name"]),
        "species": str(m.get("species", "hsapiens")),
        "adapt_steps": adapt_steps,
        # Extraction config of the cell-branch teacher
        "extract_max_tokens": m.get("extract_max_tokens"),
        "extract_gene_sampling_strategy": m.get("extract_gene_sampling_strategy"),
        "own_gene_id_column": m.get("gene_id_column"),
        "pretrained_vocabulary_used": m.get("pretrained_vocabulary_used"),
        "cache_dir": m.get("cache_dir"),
    }


def load_scconcept_training_meta(
    scconcept_training_id: str,
    *,
    expect_adapt_steps_zero: bool = True,
) -> dict:
    """Read `trainings_scConcept/{id}/manifest.json` and validate it (asserts adapt_steps==0)."""
    manifest_path = get_full_path(
        "train", "trainings", "trainings_scConcept", scconcept_training_id, "manifest.json"
    )
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"scConcept training manifest not found: {manifest_path}. Extract the cell-branch teacher "
            f"first (v2_scConcept_cellEmb.py) or pass the correct --scconcept_training_id."
        )
    with open(manifest_path, "r") as f:
        m = json.load(f)
    meta = validate_scconcept_training_manifest(
        m, scconcept_training_id, expect_adapt_steps_zero=expect_adapt_steps_zero
    )
    meta["manifest_path"] = str(manifest_path)
    return meta


# --- Cohort slide discovery -----------------------------------------------------------------


def list_cohort_slides(cohort_dir: Path) -> List[Tuple[str, Path]]:
    """Return the downloaded slides of the cohort as sorted (slide_id, path) pairs (macOS ._* files skipped)."""
    slides: List[Tuple[str, Path]] = []
    for p in sorted(cohort_dir.glob("*.h5ad")):
        if p.name.startswith("._"):
            continue
        slides.append((p.stem, p))
    return slides


def h5ad_has_obsm_key(path: Path, key: str) -> bool:
    """Cheap check (via h5py, no X load) whether an .h5ad already carries `obsm/{key}`."""
    import h5py

    with h5py.File(path, "r") as f:
        return "obsm" in f and key in f["obsm"]


# --- Gene-name -> gene-ID mapping + vocabulary-coverage gate --------------------------------


def _clean_gene_id(x) -> str:
    """Convert one map_gene_names_to_ids output to a string ('' for unmapped/NaN)."""
    if x is None:
        return ""
    if isinstance(x, float) and np.isnan(x):
        return ""
    s = str(x)
    return "" if s.lower() == "nan" else s


def compute_slide_gene_ids(
    var_names: Sequence[str],
    *,
    mapper: Callable[[Sequence[str]], Sequence],
    gene_names_are_ids: bool,
) -> Tuple[np.ndarray, str]:
    """
    Corpus gene IDs of one slide, in var order: var_names mapped with mapper (unmapped -> ''),
    or var_names as is if gene_names_are_ids. Returns (gene_ids, source).
    """
    names = [str(v) for v in var_names]
    if gene_names_are_ids:
        return np.asarray(names, dtype=object), "var_names_as_ids"
    mapped = list(mapper(names))
    cleaned = [_clean_gene_id(x) for x in mapped]
    return np.asarray(cleaned, dtype=object), "mapped_from_var_names"


def coverage_and_gate(
    slide_id: str,
    gene_ids: np.ndarray,
    var_names: Sequence[str],
    *,
    coverage_fn: Callable,
    tokenizer,
    species: str,
    min_coverage_frac: float,
    min_genes_in_vocab: int,
    gene_names_are_ids: bool,
) -> Dict[str, float]:
    """
    Vocabulary coverage of the gene IDs of a slide; raises RuntimeError if it is below the thresholds.
    coverage_fn is compute_vocab_coverage from v2_scConcept_cellEmb.
    """
    cov = coverage_fn(gene_ids, tokenizer, species=species)
    ok = cov["vocab_coverage_frac"] >= min_coverage_frac and cov["n_genes_in_vocab"] >= min_genes_in_vocab
    if ok:
        return cov

    # Hint if the raw var_names cover the vocabulary (var_names may already be gene IDs)
    hint = ""
    if not gene_names_are_ids:
        cov_raw = coverage_fn(np.asarray([str(v) for v in var_names], dtype=object), tokenizer, species=species)
        if cov_raw["vocab_coverage_frac"] >= min_coverage_frac:
            hint = (
                f" adata.var_names used directly (no name->ID mapping) would cover "
                f"{cov_raw['n_genes_in_vocab']}/{cov_raw['n_genes_panel']} "
                f"({100.0 * cov_raw['vocab_coverage_frac']:.1f}%) — this slide's var_names may already be "
                f"corpus gene IDs; re-run with --gene_names_are_ids."
            )
    raise RuntimeError(
        f"[{slide_id}] scConcept vocabulary coverage too low: {cov['n_genes_in_vocab']}/"
        f"{cov['n_genes_panel']} genes in vocab ({100.0 * cov['vocab_coverage_frac']:.1f}%), below "
        f"thresholds (frac >= {100.0 * min_coverage_frac:.1f}% and >= {min_genes_in_vocab} genes). "
        f"Verify the species / gene-symbol convention, "
        f"or lower --min_vocab_coverage_frac / --min_genes_in_vocab to override." + hint
    )


def n_duplicate_ids(gene_ids: np.ndarray) -> int:
    """Count duplicated non-empty gene IDs (aliases can map two symbols to one Ensembl ID)."""
    ids = [g for g in gene_ids.tolist() if g not in ("", None)]
    return int(len(ids) - len(set(ids)))


# --- Manifest ------------------------------------------------------------------------------


def merge_scconcept_block(existing_manifest: Optional[dict], block: dict) -> dict:
    """Attach/replace the `scConcept` block on the cohort manifest, preserving every other key."""
    manifest = dict(existing_manifest) if isinstance(existing_manifest, dict) else {}
    manifest["scConcept"] = block
    return manifest


def write_json_atomic(path: Path, obj: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=False)
    tmp.replace(path)


# --- Main ----------------------------------------------------------------------------------


def _resolve_device(device_arg: Optional[str]):
    import torch

    if device_arg is not None:
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main(args: argparse.Namespace) -> None:
    # --- 1) scConcept model provenance (assert adapt_steps == 0) ---------------------------
    meta = load_scconcept_training_meta(
        args.scconcept_training_id, expect_adapt_steps_zero=not args.allow_adapted
    )
    model_name = args.scconcept_model_name or meta["model_name"]
    species = args.scconcept_species or meta["species"]
    if args.scconcept_model_name and args.scconcept_model_name != meta["model_name"]:
        logger.warning(
            "Requested model_name=%s differs from the training manifest's model_name=%s; using the "
            "requested one.", args.scconcept_model_name, meta["model_name"],
        )
    if model_name != "corpus40M-model30M":
        logger.warning(
            "model_name=%s is not 'corpus40M-model30M' (512-dim). Continuing, but "
            "verify the embedding_size matches Novae's embedding_size.", model_name,
        )

    # Extraction config of the cell-branch teacher, unless overridden
    extract_max_tokens = args.extract_max_tokens if args.extract_max_tokens is not None else meta["extract_max_tokens"]
    extract_gene_sampling_strategy = (
        args.extract_gene_sampling_strategy
        if args.extract_gene_sampling_strategy is not None
        else meta["extract_gene_sampling_strategy"]
    )
    logger.info(
        "scConcept teacher: training_id=%s | version=%s | model_name=%s | species=%s | adapt_steps=%s",
        meta["scconcept_training_id"], meta["scConcept_version"], model_name, species, meta["adapt_steps"],
    )
    logger.info(
        "Extraction config (reused from the teacher unless overridden): max_tokens=%s | "
        "gene_sampling_strategy=%s", extract_max_tokens, extract_gene_sampling_strategy,
    )

    # --- 2) Cohort slides ------------------------------------------------------------------
    cohort_dir = Path(get_full_path("train", "pretraining_datasets", "novae_public", args.cohort_id))
    if not cohort_dir.is_dir():
        raise FileNotFoundError(
            f"Cohort folder not found: {cohort_dir}. Run _0_download_novae_public_cohort.py first."
        )
    all_slides = list_cohort_slides(cohort_dir)
    if not all_slides:
        raise RuntimeError(f"No {{slide_id}}.h5ad found under {cohort_dir}.")
    if args.limit is not None and args.limit >= 0:
        all_slides = all_slides[: args.limit]
        logger.info("--limit %d: restricting to the first %d slide(s).", args.limit, len(all_slides))

    # Split into to-process vs already-done (unless --force).
    to_process: List[Tuple[str, Path]] = []
    already_done: List[str] = []
    for slide_id, path in all_slides:
        if not args.force and h5ad_has_obsm_key(path, OBSM_KEY):
            already_done.append(slide_id)
        else:
            to_process.append((slide_id, path))
    logger.info(
        "Cohort '%s': %d slide(s) total, %d already have obsm['%s'] (skipped), %d to process.",
        args.cohort_id, len(all_slides), len(already_done), OBSM_KEY, len(to_process),
    )

    # --- 3) Load the raw pretrained scConcept model ---------------------------------------
    device = _resolve_device(args.device)
    logger.info("Using device: %s", device)
    # Lazy import: pulls in `concept` (sc_concept_env2 only) + the shared v2 helpers.
    from src.pretraining_ST.scConcept_pretraining_st.v2_scConcept_cellEmb import (  # noqa: E402
        load_pretrained_scconcept,
        extract_cls_emb_matrix,
        compute_vocab_coverage,
    )

    concept, model_info = load_pretrained_scconcept(model_name, species=species, device=device)
    logger.info(
        "Loaded scConcept model from %s (pretrained_vocabulary_used=%s).",
        model_info["model_dir"], model_info["pretrained_vocabulary_used"],
    )

    def mapper(names: Sequence[str]) -> Sequence:
        # concept.map_gene_names_to_ids: names -> corpus gene IDs (case-insensitive; NaN if unmapped).
        return concept.map_gene_names_to_ids(species, list(names))

    # --- 4) Preflight: vocabulary coverage for all to-process slides (fail fast) -----------
    import anndata as ad

    # Allow writing nullable-string columns (the cohort h5ad files were written in the novae env, pandas 3.0)
    ad.settings.allow_write_nullable_strings = True

    if not to_process:
        logger.info("Nothing to process. Exiting.")
        return

    logger.info("[Preflight] Checking gene-panel vs corpus-vocabulary coverage for %d slide(s) ...", len(to_process))
    for slide_id, path in to_process:
        # Read var only (backed mode) to check the coverage before any embedding
        backed = ad.read_h5ad(path, backed="r")
        try:
            var_names = backed.var_names.astype(str).to_numpy()
        finally:
            if getattr(backed, "isbacked", False) and backed.file is not None:
                backed.file.close()
        gene_ids, source = compute_slide_gene_ids(
            var_names, mapper=mapper, gene_names_are_ids=args.gene_names_are_ids
        )
        cov = coverage_and_gate(
            slide_id, gene_ids, var_names,
            coverage_fn=compute_vocab_coverage, tokenizer=concept.tokenizer, species=species,
            min_coverage_frac=float(args.min_vocab_coverage_frac),
            min_genes_in_vocab=int(args.min_genes_in_vocab),
            gene_names_are_ids=args.gene_names_are_ids,
        )
        logger.info(
            "[Preflight] %s: %d/%d genes in vocab (%.1f%%), source=%s",
            slide_id, cov["n_genes_in_vocab"], cov["n_genes_panel"],
            100.0 * cov["vocab_coverage_frac"], source,
        )
    logger.info("[Preflight] Vocabulary coverage OK for all %d slide(s).", len(to_process))

    if args.dry_run:
        logger.info("[DRY-RUN] Preflight passed; no embeddings extracted, nothing written under ROOT_PATH.")
        return

    # --- 5) Extract + write obsm per slide -------------------------------------------------
    slide_records: Dict[str, dict] = {}
    embedding_dim_seen: Optional[int] = None
    for i, (slide_id, path) in enumerate(to_process, start=1):
        logger.info("[%d/%d] %s: loading adata ...", i, len(to_process), slide_id)
        adata = ad.read_h5ad(path)

        # Gene IDs and coverage recomputed from the full read (aligned with adata.var)
        var_names = adata.var_names.astype(str).to_numpy()
        gene_ids, source = compute_slide_gene_ids(
            var_names, mapper=mapper, gene_names_are_ids=args.gene_names_are_ids
        )
        adata.var[args.gene_id_column] = gene_ids
        cov = compute_vocab_coverage(gene_ids, concept.tokenizer, species=species)
        dup = n_duplicate_ids(gene_ids)
        if dup > 0:
            logger.warning("[%s] %d duplicate corpus gene ID(s) after mapping (gene aliases).", slide_id, dup)

        emb, _stats = extract_cls_emb_matrix(
            concept, adata,
            gene_id_column=args.gene_id_column,
            max_tokens=extract_max_tokens,
            gene_sampling_strategy=extract_gene_sampling_strategy,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            species=species,
            out_dtype=args.obsm_dtype,
            progress_desc=f"[{slide_id}] scConcept",
        )
        emb_dim = int(emb.shape[1])
        if emb_dim != int(args.expected_embedding_dim):
            raise RuntimeError(
                f"[{slide_id}] scConcept embedding_dim={emb_dim} != expected {args.expected_embedding_dim}. "
                f"Wrong model? corpus40M-model30M is 512-dim (the 170M model is 1024). Override with "
                f"--expected_embedding_dim if intentional."
            )
        if embedding_dim_seen is None:
            embedding_dim_seen = emb_dim
        elif emb_dim != embedding_dim_seen:
            raise RuntimeError(
                f"[{slide_id}] embedding_dim={emb_dim} differs from earlier slides ({embedding_dim_seen})."
            )

        if emb.shape[0] != adata.n_obs:
            raise RuntimeError(f"[{slide_id}] embedding rows {emb.shape[0]} != n_obs {adata.n_obs}.")
        adata.obsm[OBSM_KEY] = emb

        # Atomic write (temp file, then rename)
        tmp_path = path.with_suffix(".h5ad.tmp")
        adata.write_h5ad(tmp_path)
        tmp_path.replace(path)
        logger.info(
            "[%d/%d] %s: wrote obsm['%s'] (%d cells x %d dim, %s) -> %s",
            i, len(to_process), slide_id, OBSM_KEY, emb.shape[0], emb_dim, args.obsm_dtype, path.name,
        )

        slide_records[slide_id] = {
            "n_cells": int(adata.n_obs),
            "n_genes_panel": int(cov["n_genes_panel"]),
            "n_genes_in_vocab": int(cov["n_genes_in_vocab"]),
            "vocab_coverage_frac": float(cov["vocab_coverage_frac"]),
            "n_duplicate_gene_ids": dup,
            "embedding_dim": emb_dim,
            "gene_id_source": source,
            "status": "embedded",
        }
        del adata, emb

    # --- 6) Extend the cohort manifest with a self-contained scConcept provenance block ----
    manifest_path = cohort_dir / "manifest.json"
    existing = None
    if manifest_path.is_file():
        with open(manifest_path, "r") as f:
            existing = json.load(f)
    else:
        logger.warning("Cohort manifest.json not found at %s; creating a minimal one.", manifest_path)
        existing = {"cohort_id": args.cohort_id}

    # Keep the records of slides embedded in previous runs
    prev_block = existing.get("scConcept", {}) if isinstance(existing, dict) else {}
    prev_slides = prev_block.get("slides", {}) if isinstance(prev_block, dict) else {}
    merged_slides = dict(prev_slides)
    merged_slides.update(slide_records)

    block = {
        "scconcept_training_id": meta["scconcept_training_id"],
        "scConcept_version": meta["scConcept_version"],
        "model_name": model_name,
        "species": species,
        "adapt_steps": meta["adapt_steps"],
        "obsm_key": OBSM_KEY,
        "gene_id_column": args.gene_id_column,
        "gene_names_are_ids": bool(args.gene_names_are_ids),
        "embedding_dim": int(embedding_dim_seen) if embedding_dim_seen is not None else prev_block.get("embedding_dim"),
        "obsm_dtype": args.obsm_dtype,
        "extract_max_tokens": extract_max_tokens,
        "extract_gene_sampling_strategy": extract_gene_sampling_strategy,
        "min_vocab_coverage_frac": float(args.min_vocab_coverage_frac),
        "min_genes_in_vocab": int(args.min_genes_in_vocab),
        "cache_dir": model_info["cache_dir"],
        "pretrained_vocabulary_used": model_info["pretrained_vocabulary_used"],
        "scconcept_training_manifest": meta["manifest_path"],
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "n_slides_embedded_this_run": len(slide_records),
        "slides": merged_slides,
    }
    manifest = merge_scconcept_block(existing, block)
    write_json_atomic(manifest_path, manifest)

    logger.info(
        "Done. Embedded %d slide(s) this run (%d already present). Manifest updated: %s",
        len(slide_records), len(already_done), manifest_path,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="Compute per-cell scConcept embeddings of the public Novae cohort slides and write them to "
                    "adata.obsm['X_scConcept'] (raw scConcept model and extraction config of the cell-branch teacher)."
    )

    # Cohort + scConcept model
    parser.add_argument(
        "--cohort_id", type=str, default="human_cellbased_nobrain_scc40M30M_1",
        help="Public cohort id / folder under train/pretraining_datasets/novae_public/. "
             "Default: human_cellbased_nobrain_scc40M30M_1",
    )
    parser.add_argument(
        "--scconcept_training_id", type=str, default="training_7",
        help="scConcept training_id whose model and extraction config are reused (adapt_steps must be 0). "
             "Default: training_7.",
    )
    parser.add_argument(
        "--scconcept_model_name", type=str, default=None,
        help="Override the model_name (default: read from the scConcept training manifest, "
             "normally 'corpus40M-model30M').",
    )
    parser.add_argument(
        "--scconcept_species", type=str, default=None,
        help="Override the species key (default: read from the scConcept training manifest, "
             "normally 'hsapiens').",
    )
    parser.add_argument(
        "--allow_adapted", action="store_true",
        help="Allow a scConcept training with adapt_steps != 0.",
    )

    # Gene identifiers
    parser.add_argument(
        "--gene_id_column", type=str, default="scconcept_gene_id",
        help="adata.var column to store the resolved corpus gene IDs into (used as the tokenizer's "
             "var_column). Default: scconcept_gene_id.",
    )
    parser.add_argument(
        "--gene_names_are_ids", action="store_true",
        help="Use adata.var_names as corpus gene IDs (no name->ID mapping).",
    )

    # Vocabulary-coverage fail-fast
    parser.add_argument(
        "--min_vocab_coverage_frac", type=float, default=0.3,
        help="Minimum fraction of the panel genes of a slide in the corpus vocabulary (0 disables the check).",
    )
    parser.add_argument(
        "--min_genes_in_vocab", type=int, default=1,
        help="Minimum number of panel genes of a slide in the corpus vocabulary (0 disables the check).",
    )

    # Extraction params (default: reuse the scConcept training manifest's values)
    parser.add_argument(
        "--extract_max_tokens", type=int, default=None,
        help="max_tokens for extraction (default: extract_max_tokens of the scConcept training manifest).",
    )
    parser.add_argument(
        "--extract_gene_sampling_strategy", type=str, default=None,
        help="Override gene_sampling_strategy for extraction (default: reuse the training manifest's value).",
    )
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for scConcept inference.")
    parser.add_argument("--num_workers", type=int, default=8, help="DataLoader workers for tokenization/collate.")
    parser.add_argument(
        "--obsm_dtype", type=str, default="float16", choices=["float16", "float32"],
        help="Storage dtype of obsm['X_scConcept'].",
    )
    parser.add_argument(
        "--expected_embedding_dim", type=int, default=DEFAULT_EXPECTED_EMBEDDING_DIM,
        help=f"Assert the scConcept cls dim (default {DEFAULT_EXPECTED_EMBEDDING_DIM} for corpus40M-model30M; "
             f"the 170M model is 1024).",
    )

    # Technical / control
    parser.add_argument("--device", type=str, default=None, help="Device: cuda / cuda:0 / cpu. Default: auto.")
    parser.add_argument(
        "--force", action="store_true",
        help="Recompute obsm['X_scConcept'] even for slides that already have it (default: skip).",
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Only load the model and check the vocabulary coverage (writes nothing).",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Process only the first N cohort slides.",
    )

    args = parser.parse_args()
    main(args)
