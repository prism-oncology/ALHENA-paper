#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Download the public slides of the Novae pretraining cohort from the Novae dataset (prism-oncology/novae):
human, MERSCOPE + CosMx, non-brain (the Xenium part of the cohort comes from the project data).

Saved in <ROOT_HistoST> / train / pretraining_datasets / novae_public / <cohort_id> /
    - <slide_id>.h5ad: raw counts and spatial coordinates of each slide
    - manifest.json: source, filter, metadata breakdown and slide records
    - download_log.txt: run log
"""

import argparse
import json
import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

# Add the project root to sys.path so that 'src' can be imported
project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.constants import get_full_path  # noqa: E402

logger = logging.getLogger(__name__)


# --- Cohort definition ---------------------------------------------------------------------
# Recorded in manifest.json

HF_REPO_ID: str = "prism-oncology/novae"
HF_REPO_TYPE: str = "dataset"
HF_METADATA_FILE: str = "metadata.csv"

DEFAULT_COHORT_ID: str = "human_cellbased_nobrain_scc40M30M_1"

SPECIES_KEEP: List[str] = ["human"]
TECHNOLOGY_KEEP: List[str] = ["merscope", "cosmx"]  # cell-based
TISSUE_EXCLUDE: List[str] = ["brain"]
# Excluded by the filter above (recorded in the manifest)
EXCLUDED_BY_CONSTRUCTION: List[str] = ["mouse", "toy", "visium", "visium_hd", "xenium"]

FILTER_DESCRIPTION: str = (
    "Public data: species == 'human', technology in {'merscope', 'cosmx'}, "
    "tissue != 'brain'. Xenium is excluded (the public human Xenium slides are "
    "already part of the project data); mouse / toy / visium / visium_hd are excluded by "
    "construction."
)


# --- Metadata / breakdown ------------------------------------------------------------------


def read_full_metadata():
    """Return the full Novae dataset metadata DataFrame (index = slide name)."""
    import novae

    logger.info("Reading full Novae dataset metadata via novae.load_dataset(dry_run=True) ...")
    metadata = novae.load_dataset(dry_run=True)
    for col in ("species", "technology", "tissue"):
        if col not in metadata.columns:
            raise KeyError(
                f"Expected column '{col}' in Novae metadata; found {list(metadata.columns)}."
            )
    logger.info("Full metadata: %d slide(s) across the whole repo.", len(metadata))
    return metadata


def full_breakdown(metadata) -> dict:
    """Compute the full species x technology x tissue breakdown (for logging + manifest)."""
    cross = (
        metadata.groupby(["species", "technology", "tissue"])
        .size()
        .reset_index(name="n_slides")
        .sort_values(["species", "technology", "tissue"])
    )
    return {
        "n_slides_total": int(len(metadata)),
        "by_species": {str(k): int(v) for k, v in metadata["species"].value_counts().items()},
        "by_technology": {str(k): int(v) for k, v in metadata["technology"].value_counts().items()},
        "by_tissue": {str(k): int(v) for k, v in metadata["tissue"].value_counts().items()},
        "by_species_technology_tissue": [
            {
                "species": str(r.species),
                "technology": str(r.technology),
                "tissue": str(r.tissue),
                "n_slides": int(r.n_slides),
            }
            for r in cross.itertuples(index=False)
        ],
    }


def log_breakdown(breakdown: dict) -> None:
    """Log the full breakdown line by line."""
    logger.info("===== Full Novae repo breakdown (species x technology x tissue) =====")
    logger.info("Total slides in repo: %d", breakdown["n_slides_total"])
    logger.info("By species:    %s", breakdown["by_species"])
    logger.info("By technology: %s", breakdown["by_technology"])
    logger.info("By tissue:     %s", breakdown["by_tissue"])
    logger.info("Cross-tab (species | technology | tissue | n_slides):")
    for row in breakdown["by_species_technology_tissue"]:
        logger.info(
            "    %-8s | %-10s | %-16s | %d",
            row["species"], row["technology"], row["tissue"], row["n_slides"],
        )
    logger.info("=====================================================================")


def apply_cohort_filter(metadata):
    """Apply the cohort filter (case-insensitive) and return the kept rows."""
    species = metadata["species"].astype(str).str.lower()
    technology = metadata["technology"].astype(str).str.lower()
    tissue = metadata["tissue"].astype(str).str.lower()

    keep = (
        species.isin([s.lower() for s in SPECIES_KEEP])
        & technology.isin([t.lower() for t in TECHNOLOGY_KEEP])
        & ~tissue.isin([t.lower() for t in TISSUE_EXCLUDE])
    )
    kept = metadata[keep].copy()

    logger.info("----- Cohort filter -----")
    logger.info("%s", FILTER_DESCRIPTION)
    logger.info("Kept %d / %d slide(s) after filter.", len(kept), len(metadata))
    if kept.empty:
        raise RuntimeError(
            "Cohort filter kept 0 slides — the Novae metadata schema/values may have changed. "
            "Inspect the full breakdown above before proceeding."
        )
    logger.info("Kept by technology: %s", {str(k): int(v) for k, v in kept["technology"].value_counts().items()})
    logger.info("Kept by tissue:     %s", {str(k): int(v) for k, v in kept["tissue"].value_counts().items()})
    logger.info("Kept slide names (n=%d):", len(kept))
    for name in kept.index:
        row = kept.loc[name]
        logger.info("    %-40s  %-8s %-10s %s", name, row["species"], row["technology"], row["tissue"])
    logger.info("-------------------------")
    return kept


# --- Provenance ----------------------------------------------------------------------------


def dataset_commit_sha() -> Optional[str]:
    """Best-effort current commit SHA of the HF dataset repo (None if unavailable/offline)."""
    try:
        from huggingface_hub import HfApi

        info = HfApi().dataset_info(HF_REPO_ID)
        return getattr(info, "sha", None)
    except Exception as e:  # noqa: BLE001 - provenance is best-effort, never fatal
        logger.warning("Could not resolve HF dataset commit SHA (offline?): %s", e)
        return None


def novae_version() -> Optional[str]:
    try:
        import novae

        return str(getattr(novae, "__version__", None))
    except Exception:  # noqa: BLE001
        return None


# --- Per-slide download --------------------------------------------------------------------


def sanitize_obs_columns(adata, name: str) -> None:
    """Drop duplicate adata.obs column labels in place, keeping the first occurrence."""
    columns = adata.obs.columns
    if columns.is_unique:
        return
    duplicated_labels = sorted(set(columns[columns.duplicated()]))
    adata.obs = adata.obs.iloc[:, ~columns.duplicated(keep="first")]
    logger.warning(
        "[%s] Collapsed duplicate adata.obs column(s) %s (kept the first occurrence of each).",
        name, duplicated_labels,
    )


def download_one_slide(name: str):
    """Download one slide by its exact metadata name with novae.load_dataset(pattern=...)."""
    import novae

    if "/" in name or "\\" in name:
        raise ValueError(f"Unexpected path separator in slide name {name!r}.")

    pattern = re.escape(name) + r"\Z"
    # Raw counts + spatial coordinates only (no embeddings, no annotations)
    adatas = novae.load_dataset(pattern=pattern)
    if len(adatas) != 1:
        got = [str(a.obs["name"].iloc[0]) if "name" in a.obs else "?" for a in adatas]
        raise RuntimeError(f"Expected exactly 1 slide for name {name!r}, got {len(adatas)}: {got}")

    adata = adatas[0]
    got_name = str(adata.obs["name"].iloc[0]) if "name" in adata.obs else None
    if got_name is not None and got_name != name:
        raise RuntimeError(f"Downloaded slide name {got_name!r} != requested {name!r}.")

    # Non-fatal checks (spatial coordinates, genes)
    if "spatial" not in adata.obsm:
        logger.warning("[%s] No adata.obsm['spatial'] — spatial coords missing (graph build will fail).", name)
    if adata.n_vars == 0:
        logger.warning("[%s] AnnData has 0 genes.", name)

    # Drop duplicate obs columns (write_h5ad rejects repeated column names)
    sanitize_obs_columns(adata, name)

    return adata


def n_cells_backed(path: Path) -> Optional[int]:
    """Read only the cell count of an existing .h5ad (backed mode; no X load)."""
    try:
        import anndata as ad

        return int(ad.read_h5ad(str(path), backed="r").n_obs)
    except Exception as e:  # noqa: BLE001
        logger.warning("Could not read n_obs from %s: %s", path, e)
        return None


# --- Manifest ------------------------------------------------------------------------------


def load_existing_manifest(manifest_path: Path) -> Optional[dict]:
    if manifest_path.is_file():
        try:
            with open(manifest_path, "r") as f:
                return json.load(f)
        except Exception as e:  # noqa: BLE001
            logger.warning("Could not read existing manifest %s: %s", manifest_path, e)
    return None


def write_manifest(manifest_path: Path, manifest: dict) -> None:
    manifest["updated_utc"] = datetime.now(timezone.utc).isoformat()
    tmp = manifest_path.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=False)
    tmp.replace(manifest_path)


def full_cohort_spec(cohort_id: str) -> dict:
    """Description of the full cohort (project Xenium train cells + public slides) for the manifest."""
    return {
        "description": (
            "The full Novae pretraining cohort is the union of (a) the project Xenium train cells "
            "(train-only masks per slide) and (b) the public MERSCOPE + CosMx human non-brain "
            "slides listed in this manifest. Public Xenium slides are not downloaded: the public "
            "human Xenium slides are already part of the project data."
        ),
        "own_xenium_train_cells": {
            "included": True,
            "copied_here": False,
            "source": "train/training_datasets/{dataset_id}/{training_dataset_id}/cell_ids_sets.npz "
                      "(key 'train_only_by_slide')",
            "assembled_at": "Novae pretraining time (_2_novae_pretraining.py); not stored in this cohort folder.",
        },
        "public_cellbased": {
            "included": True,
            "copied_here": True,
            "cohort_id": cohort_id,
            "source": f"HF {HF_REPO_TYPE} {HF_REPO_ID}",
            "slides": "see manifest 'slides' list",
        },
    }


# --- Main ----------------------------------------------------------------------------------


def main(args: argparse.Namespace) -> None:
    cohort_id = args.cohort_id

    # 1) Read the full metadata and compute the breakdown.
    metadata = read_full_metadata()
    breakdown = full_breakdown(metadata)

    # 2) Create the cohort folder and the file log download_log.txt (not with --dry_run)
    cohort_dir: Optional[Path] = None
    if not args.dry_run:
        cohort_dir = Path(get_full_path("train", "pretraining_datasets", "novae_public", cohort_id))
        cohort_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(cohort_dir / "download_log.txt")
        fh.setLevel(logging.INFO)
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        logging.getLogger().addHandler(fh)
        logger.info("Cohort folder: %s", cohort_dir)

    # 3) Log the full breakdown and apply the cohort filter.
    log_breakdown(breakdown)
    kept = apply_cohort_filter(metadata)
    kept_names = list(kept.index)
    if args.limit is not None and args.limit >= 0:
        kept_names = kept_names[: args.limit]
        logger.info("--limit %d: restricting to the first %d kept slide(s).", args.limit, len(kept_names))

    if args.dry_run:
        logger.info("[DRY-RUN] Would download %d slide(s) into "
                    "train/pretraining_datasets/novae_public/%s/. Nothing written.",
                    len(kept_names), cohort_id)
        return

    assert cohort_dir is not None  # created above in the non-dry-run branch
    manifest_path = cohort_dir / "manifest.json"
    existing = load_existing_manifest(manifest_path)
    prev_slides: Dict[str, dict] = (
        {s["slide_id"]: s for s in existing.get("slides", [])} if existing else {}
    )

    # 4) Assemble the manifest skeleton (rewritten after each slide).
    now_iso = datetime.now(timezone.utc).isoformat()
    manifest: dict = {
        "cohort_id": cohort_id,
        "created_utc": existing.get("created_utc", now_iso) if existing else now_iso,
        "updated_utc": now_iso,
        "source": {
            "hf_repo_id": HF_REPO_ID,
            "hf_repo_type": HF_REPO_TYPE,
            "metadata_file": HF_METADATA_FILE,
            "dataset_commit_sha": dataset_commit_sha(),
            "novae_version": novae_version(),
        },
        "filter": {
            "description": FILTER_DESCRIPTION,
            "species_keep": SPECIES_KEEP,
            "technology_keep": TECHNOLOGY_KEEP,
            "tissue_exclude": TISSUE_EXCLUDE,
            "excluded_by_construction": EXCLUDED_BY_CONSTRUCTION,
        },
        "full_cohort_spec": full_cohort_spec(cohort_id),
        "metadata_breakdown_full": breakdown,
        "n_slides_kept": len(kept_names),
        "slides": [],
    }

    # Keep the scConcept block written by _1_scConcept_embs_novae_public.py
    if existing and isinstance(existing.get("scConcept"), dict):
        manifest["scConcept"] = existing["scConcept"]

    # 5) Download (or skip) each kept slide, updating the manifest incrementally.
    slides_records: List[dict] = []
    n_downloaded, n_skipped, n_failed = 0, 0, 0
    for i, name in enumerate(kept_names, start=1):
        row = kept.loc[name]
        out_path = cohort_dir / f"{name}.h5ad"
        rec = {
            "slide_id": name,
            "species": str(row["species"]),
            "technology": str(row["technology"]),
            "tissue": str(row["tissue"]),
            "n_cells": None,
            "relpath": f"{cohort_id}/{name}.h5ad",
            "status": None,
            "downloaded_utc": None,
        }

        if out_path.is_file() and not args.force:
            prev = prev_slides.get(name, {})
            rec["n_cells"] = prev.get("n_cells") or n_cells_backed(out_path)
            rec["downloaded_utc"] = prev.get("downloaded_utc")
            rec["status"] = "already_present"
            logger.info("[%d/%d] %s already present — skipping.", i, len(kept_names), name)
            n_skipped += 1
            slides_records.append(rec)
            manifest["slides"] = slides_records
            write_manifest(manifest_path, manifest)
            continue

        try:
            logger.info("[%d/%d] Downloading %s (%s / %s) ...",
                        i, len(kept_names), name, row["technology"], row["tissue"])
            adata = download_one_slide(name)
            rec["n_cells"] = int(adata.n_obs)
            # Write to a temp file, then rename it
            tmp_path = out_path.with_suffix(".h5ad.tmp")
            adata.write_h5ad(tmp_path)
            tmp_path.replace(out_path)
            rec["status"] = "downloaded"
            rec["downloaded_utc"] = datetime.now(timezone.utc).isoformat()
            logger.info("[%d/%d] Saved %s (%d cells, %d genes) -> %s",
                        i, len(kept_names), name, adata.n_obs, adata.n_vars, out_path)
            n_downloaded += 1
        except Exception as e:  # noqa: BLE001
            rec["status"] = f"failed: {e}"
            logger.error("[%d/%d] FAILED to download %s: %s", i, len(kept_names), name, e)
            n_failed += 1

        slides_records.append(rec)
        manifest["slides"] = slides_records
        write_manifest(manifest_path, manifest)

    # 6) Final summary.
    logger.info("Done. downloaded=%d, already_present=%d, failed=%d, total_kept=%d.",
                n_downloaded, n_skipped, n_failed, len(kept_names))
    logger.info("Manifest written to %s", manifest_path)
    if n_failed:
        logger.error("%d slide(s) failed to download — re-run to retry (idempotent).", n_failed)
        sys.exit(1)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Download the public human MERSCOPE + CosMx non-brain slides of the Novae pretraining "
                    "cohort from the prism-oncology/novae HF dataset."
    )
    parser.add_argument(
        "--cohort_id", type=str, default=DEFAULT_COHORT_ID,
        help=f"Cohort identifier / folder name under train/pretraining_datasets/novae_public/. "
             f"Default: {DEFAULT_COHORT_ID}",
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Only print the metadata breakdown and the kept slides (writes nothing).",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-download slides even if their {slide_id}.h5ad already exists (default: skip).",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Download only the first N kept slides.",
    )

    args = parser.parse_args()
    main(args)
