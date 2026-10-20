#!/usr/bin/env python
"""
Compute the H&E patch and cell embeddings of the PanNuke pseudo-slides prepared by _4_prepare_patches.py, with
extract_embeddings_for_slide (_8_compute_HEembs.py) and process_slide (_9_main_HEemb_cellID.py).
Runs in the `histost` env.

Saved in <ROOT_HistoST> / eval / eval_datasets / pannuke_eval_data / prepared / eval_st_pannuke_<recipe> /
         he_embeddings / HEembs_t<n>e<c> /
    - <pseudo_slide>.h5: patch and cell embeddings
    - main_HEemb_cellID/<pseudo_slide>.parquet: main patch-cell row of each cell
    - extract_manifest.json, extract_log.txt
"""

import argparse
import logging
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

# Project imports
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src_A2.common import configure_logger, ensure_dir, resolve_device, write_json
from pannuke import pannuke_embeddings as pe
from pannuke import pannuke_upstream as upstream
from pannuke.pannuke_common import (
    GEOMETRY_RECIPE_NAMES,
    git_head,
    pannuke_root,
    prepared_root,
)

logger = logging.getLogger(__name__)

STAGES: Tuple[str, ...] = ("extract", "main_rows", "verify")


# ---------------------------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------------------------

def load_encoder(checkpoint_path: Path, device_str: str) -> Tuple[Any, Dict[str, Any], Any]:
    """Load the exported H&E encoder bundle on device_str. Returns (encoder, facts, device)."""
    import torch

    sym = upstream.extraction()
    device = torch.device(device_str)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True  # mirrors `_8.main()`

    encoder = sym["load_he_encoder_bundle"](Path(checkpoint_path), device)

    tok_proj = getattr(encoder, "img_tok_proj", None)
    img_proj = getattr(encoder, "image_proj", None)
    facts: Dict[str, Any] = {
        "checkpoint_path": str(checkpoint_path),
        "class_name": type(encoder).__name__,
        "backbone": str(getattr(encoder, "backbone", "")),
        "resize_size": int(getattr(encoder, "resize_size", 0)),
        "crop_size": int(getattr(encoder, "crop_size", 0)),
        "interpolation_code": int(getattr(encoder, "interpolation_code", -1)),
        "has_image_proj": img_proj is not None,
        "has_token_proj": tok_proj is not None,
        "has_encode_with_tokens": type(encoder).__name__ == "HEClipEncoderWithTokens",
        "patch_he_clip_dim": int(getattr(img_proj, "out_features", 0)) if img_proj else 0,
        "patch_cell_he_clip_dim": int(getattr(tok_proj, "out_features", 0)) if tok_proj else 0,
        "membership_group_derived": sym["membership_group_for_encoder"](encoder),
    }
    stat = Path(checkpoint_path).stat()
    facts["checkpoint_size_bytes"] = int(stat.st_size)
    facts["checkpoint_mtime"] = float(stat.st_mtime)
    return encoder, facts, device


# ---------------------------------------------------------------------------------------------
# Per-slide stages
# ---------------------------------------------------------------------------------------------

def extract_one_slide(
    encoder: Any,
    expectation: pe.SlideExpectation,
    preflight: pe.RecipePreflight,
    emb_dir: Path,
    device: Any,
    args: argparse.Namespace,
    checkpoint_path: Path,
    log: logging.Logger,
) -> Tuple[Path, bool, float]:
    """
    Run extract_embeddings_for_slide (_8) on one pseudo-slide. Returns (out_h5, extracted_in_this_run, seconds).
    Slides already complete (see pannuke_embeddings.is_extract_complete) are skipped unless --overwrite.
    """
    out_h5 = pe.embedding_h5_path(emb_dir, expectation.slide_id)

    if not args.overwrite and pe.is_extract_complete(out_h5, preflight.recipe_name, expectation):
        log.info(
            "  %-22s SKIP (already complete: %s groups)",
            expectation.slide_id,
            f"{expectation.n_expected_groups:,}",
        )
        return out_h5, False, 0.0

    if out_h5.exists():
        # Delete the old file first (it may carry the completion attrs of a previous run)
        out_h5.unlink()

    started = time.time()
    upstream.extraction()["extract_embeddings_for_slide"](
        encoder=encoder,
        slide_h5=expectation.patches_h5,
        out_h5=out_h5,
        eval_root=preflight.prepared_root,
        membership_group=preflight.membership_group,
        device=device,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        out_dtype=str(args.dtype),
        model_checkpoint=str(checkpoint_path),
    )
    seconds = time.time() - started

    if not out_h5.exists():
        raise RuntimeError(
            f"{preflight.recipe_name}/{expectation.slide_id}: extract_embeddings_for_slide "
            f"returned without creating {out_h5} ({expectation.n_patches} patches expected)."
        )
    pe.mark_extract_complete(
        out_h5, preflight.recipe_name, expectation, datetime.now().isoformat(timespec="seconds")
    )
    log.info(
        "  %-22s extracted %s patches -> %s groups in %.1fs",
        expectation.slide_id,
        f"{expectation.n_patches:,}",
        f"{expectation.n_expected_groups:,}",
        seconds,
    )
    return out_h5, True, seconds


def select_main_rows_one_slide(
    expectation: pe.SlideExpectation,
    preflight: pe.RecipePreflight,
    emb_dir: Path,
    h5_path: Path,
    force_overwrite: bool,
    log: logging.Logger,
) -> Tuple[Path, float]:
    """
    Run process_slide (_9) on one pseudo-slide. Returns (out_parquet, seconds).
    force_overwrite rewrites an existing parquet (needed whenever the h5 was rewritten).
    """
    out_parquet = pe.main_rows_parquet_path(emb_dir, expectation.slide_id)
    ensure_dir(out_parquet.parent)
    reused = out_parquet.exists() and not force_overwrite

    started = time.time()
    upstream.selection()["process_slide"](
        slide_id=expectation.slide_id,
        emb_h5_path=h5_path,
        slide_patch_h5=expectation.patches_h5,
        centroid_parquet=expectation.centroids_parquet,
        out_parquet=out_parquet,
        overwrite=bool(force_overwrite),
    )
    seconds = time.time() - started

    if not out_parquet.exists():
        raise RuntimeError(
            f"{preflight.recipe_name}/{expectation.slide_id}: `_9.process_slide` wrote no parquet "
            f"at {out_parquet} (no patch-cell group in the embedding h5)."
        )
    log.info(
        "  %-22s main rows %s (%.1fs)",
        expectation.slide_id,
        "kept (already present, h5 unchanged)" if reused else "written",
        seconds,
    )
    return out_parquet, seconds


# ---------------------------------------------------------------------------------------------
# Per-recipe driver
# ---------------------------------------------------------------------------------------------

def run_recipe(
    args: argparse.Namespace,
    preflight: pe.RecipePreflight,
    emb_dir: Path,
    encoder: Optional[Any],
    device: Any,
    checkpoint_path: Optional[Path],
    stages: Sequence[str],
    log: logging.Logger,
) -> List[Dict[str, Any]]:
    """Extract -> select -> verify, per slide, for one geometry recipe."""
    per_slide: List[Dict[str, Any]] = []
    # The parquet is verified unless the run stops after extraction
    require_parquet = ("main_rows" in stages) or ("extract" not in stages)

    for expectation in preflight.slides:
        record: Dict[str, Any] = {"slide_id": expectation.slide_id}
        h5_path = pe.embedding_h5_path(emb_dir, expectation.slide_id)
        rewrote_h5 = False

        if "extract" in stages:
            h5_path, rewrote_h5, seconds = extract_one_slide(
                encoder, expectation, preflight, emb_dir, device, args, checkpoint_path, log
            )
            record["seconds_extract"] = round(seconds, 2)
            record["skipped_as_complete"] = not rewrote_h5
        elif not h5_path.exists():
            raise FileNotFoundError(
                f"{preflight.recipe_name}/{expectation.slide_id}: no embedding h5 at {h5_path}, "
                "and 'extract' is not among --stages. Run the extraction first."
            )

        if "main_rows" in stages:
            _, seconds = select_main_rows_one_slide(
                expectation, preflight, emb_dir, h5_path, rewrote_h5 or args.overwrite, log
            )
            record["seconds_main_rows"] = round(seconds, 2)

        if "verify" in stages:
            record.update(
                pe.verify_embedding_h5(
                    h5_path, expectation, preflight.recipe_name, full_scan=bool(args.full_scan)
                )
            )
            if require_parquet:
                record.update(
                    pe.verify_main_rows_parquet(
                        pe.main_rows_parquet_path(emb_dir, expectation.slide_id),
                        h5_path,
                        expectation,
                        preflight.recipe_name,
                    )
                )

        record.update(
            {
                "n_ground_truth_cells": expectation.n_ground_truth_cells,
                "n_missing_cells": expectation.n_missing_cells,
                "n_membership_rows": expectation.n_membership_rows,
                "n_expected_groups": expectation.n_expected_groups,
                **pe.slide_file_info(emb_dir, expectation.slide_id, checksums=bool(args.checksums)),
            }
        )
        per_slide.append(record)

    return per_slide


# ---------------------------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------------------------

def main(args: argparse.Namespace) -> int:
    started_at = datetime.now()
    stages = [s for s in STAGES if s in set(args.stages)]
    if not stages:
        raise ValueError(f"--stages selected nothing; choose from {list(STAGES)}.")
    writes = ("extract" in stages) or ("main_rows" in stages)

    recipes = list(dict.fromkeys(args.geometry_recipe))
    data_root = Path(args.data_dir).expanduser().resolve() if args.data_dir else pannuke_root()

    if sys.platform == "darwin" and int(args.num_workers) > 0 and "extract" in stages:
        # As in `_8.main()`: no DataLoader workers on macOS
        warnings.warn(
            "macOS + PyTorch DataLoader workers + Albumentations/OpenCV is fragile; forcing "
            "num_workers=0."
        )
        args.num_workers = 0

    # --- the encoder, once for every recipe ---------------------------------------------------
    encoder: Optional[Any] = None
    encoder_facts: Dict[str, Any] = {}
    device: Any = None
    checkpoint_path: Optional[Path] = None
    membership_group: Optional[str] = None

    if "extract" in stages:
        checkpoint_path = pe.he_encoder_checkpoint_path(args.training_num, args.checkpoint_num)
        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(
                f"H&E encoder bundle not found: {checkpoint_path}."
            )
        device_str = resolve_device(args.device)
        logger.info("Loading %s onto %s", Path(checkpoint_path).name, device_str)
        encoder, encoder_facts, device = load_encoder(Path(checkpoint_path), device_str)
        encoder_facts["training_num"] = str(args.training_num)
        encoder_facts["checkpoint_num"] = str(args.checkpoint_num)
        encoder_facts["checkpoint_tag"] = pe.checkpoint_tag(args.training_num, args.checkpoint_num)
        encoder_facts["device"] = device_str
        membership_group = encoder_facts["membership_group_derived"]
        logger.info(
            "Encoder %s: backbone=%s, %s, token head %s -> membership group '%s'",
            encoder_facts["checkpoint_tag"],
            encoder_facts["backbone"],
            encoder_facts["class_name"],
            "present" if encoder_facts["has_token_proj"] else "ABSENT",
            membership_group,
        )

    # --- preflight every recipe before any GPU work -------------------------------------------
    preflights: Dict[str, pe.RecipePreflight] = {}
    for recipe in recipes:
        root = prepared_root(data_root, recipe)
        if membership_group is None:
            # No encoder in this run: membership group taken from prepare_manifest.json
            group = str((pe.read_prepare_manifest(root).get("membership") or {}).get("group_name"))
            logger.info(
                "No encoder in this run; taking membership group '%s' from prepare_manifest.json",
                group,
            )
        else:
            group = membership_group
        preflights[recipe] = pe.preflight_recipe(
            root,
            recipe,
            group,
            slides=args.slides,
            allow_partial_tree=bool(args.allow_partial_tree),
        )
        pre = preflights[recipe]
        logger.info(
            "Preflight '%s': %d slides, %s patches, %s ground-truth cells, %s missing, "
            "%s expected groups",
            recipe,
            len(pre.slides),
            f"{pre.total('n_patches'):,}",
            f"{pre.total('n_ground_truth_cells'):,}",
            f"{pre.total('n_missing_cells'):,}",
            f"{pre.total('n_expected_groups'):,}",
        )

    if encoder is not None:
        pe.assert_encoder_matches_tree(encoder_facts, list(preflights.values()))
    pe.assert_ground_truth_identical_across_recipes(list(preflights.values()))

    # --- run ------------------------------------------------------------------------------------
    emb_dirs: Dict[str, Path] = {}
    per_slide_by_recipe: Dict[str, List[Dict[str, Any]]] = {}
    for recipe in recipes:
        preflight = preflights[recipe]
        emb_dir = (
            Path(args.out_dir).expanduser().resolve()
            if args.out_dir
            else pe.he_embeddings_dir(preflight.prepared_root, args.training_num, args.checkpoint_num)
        )
        emb_dirs[recipe] = emb_dir
        if writes:
            ensure_dir(emb_dir)
            # Log file attached to this module's logger
            configure_logger(emb_dir / pe.EXTRACT_LOG_NAME, name=logger.name)
        log = logger

        log.info("=== recipe '%s' -> %s (stages: %s) ===", recipe, emb_dir, ", ".join(stages))
        per_slide_by_recipe[recipe] = run_recipe(
            args, preflight, emb_dir, encoder, device, checkpoint_path, stages, log
        )

    # --- the cross-recipe pairing statement ----------------------------------------------------
    cross: Optional[Dict[str, Any]] = None
    partial = any(p.is_partial for p in preflights.values())
    can_pair = (
        len(recipes) == 2
        and not partial
        and ("verify" in stages)
        and all(("main_rows" in stages) or ("extract" not in stages) for _ in recipes)
    )
    if can_pair:
        cross = pe.cross_recipe_pairing(preflights, emb_dirs)
        first, second = cross["recipes"]
        logger.info(
            "Cross-recipe pairing OK: embedded %s (%s) vs %s (%s); symmetric difference %d == "
            "missing symmetric difference %d. Paired cells: %s.",
            f"{cross[f'n_embedded_{first}']:,}", first,
            f"{cross[f'n_embedded_{second}']:,}", second,
            cross["embedded_symmetric_difference"],
            cross["missing_symmetric_difference"],
            f"{cross['n_paired_cells']:,}",
        )
    elif len(recipes) == 2:
        logger.info(
            "Cross-recipe pairing skipped (%s): it needs the full cohort with both recipes written and verified.",
            "partial cohort" if partial else "stages",
        )

    # --- manifests -------------------------------------------------------------------------------
    if writes:
        finished_at = datetime.now()
        environment = pe.environment_record()
        head = git_head()
        for recipe in recipes:
            manifest = pe.build_extract_manifest(
                preflight=preflights[recipe],
                emb_dir=emb_dirs[recipe],
                encoder_facts=encoder_facts,
                runtime={
                    "dtype": str(args.dtype),
                    "batch_size": int(args.batch_size),
                    "num_workers": int(args.num_workers),
                    "device": encoder_facts.get("device"),
                    "stages_run": list(stages),
                    "full_scan": bool(args.full_scan),
                    "overwrite": bool(args.overwrite),
                },
                per_slide=per_slide_by_recipe[recipe],
                verification={
                    "ran": "verify" in stages,
                    "all_passed": "verify" in stages,
                    "full_scan": bool(args.full_scan),
                    "checksums_computed": bool(args.checksums),
                    "emb_sample_rows": pe.EMB_SAMPLE_ROWS,
                    "parquet_verified": ("main_rows" in stages) or ("extract" not in stages),
                },
                environment=environment,
                argv=sys.argv,
                seed=int(args.seed),
                git_head_value=head,
                started_at=started_at.isoformat(timespec="seconds"),
                finished_at=finished_at.isoformat(timespec="seconds"),
                wall_seconds=(finished_at - started_at).total_seconds(),
                checksums=bool(args.checksums),
                cross_recipe=cross,
            )
            path = emb_dirs[recipe] / pe.EXTRACT_MANIFEST_NAME
            write_json(path, manifest)
            logger.info("Wrote %s", path)
    else:
        logger.info("--stages %s: nothing written.", " ".join(stages))

    total_cells = sum(
        int(r.get("n_groups_written") or r.get("n_expected_groups") or 0)
        for rows in per_slide_by_recipe.values()
        for r in rows
    )
    logger.info(
        "DONE: %d recipe(s), %d slide-runs, %s cells, %.1fs",
        len(recipes),
        sum(len(v) for v in per_slide_by_recipe.values()),
        f"{total_cells:,}",
        (datetime.now() - started_at).total_seconds(),
    )
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    parser = argparse.ArgumentParser(
        description=(
            "Compute the H&E patch and cell embeddings of the prepared PanNuke pseudo-slides "
            "(runs in the `histost` env)."
        )
    )

    # --- what to embed --------------------------------------------------------------------------
    parser.add_argument(
        "--geometry_recipe", type=str, nargs="+", default=list(GEOMETRY_RECIPE_NAMES),
        choices=list(GEOMETRY_RECIPE_NAMES),
        help=(
            "Prepared trees to embed (default: both, in one process)."
        ),
    )
    parser.add_argument(
        "--training_num", type=str, required=True,
        help=(
            "Num of the HistoST training_id (e.g. '46' for training_46)."
        ),
    )
    parser.add_argument(
        "--checkpoint_num", type=str, required=True,
        help="Checkpoint number (e.g. '13' for HEencoder_t46e13.pt).",
    )
    parser.add_argument(
        "--slides", type=str, nargs="*", default=None,
        help=(
            "Restrict to these pseudo-slides, e.g. 'pannuke_kidney' (default: all). The run is then "
            "recorded as partial and the cross-recipe pairing is skipped."
        ),
    )

    # --- where ---------------------------------------------------------------------------------
    parser.add_argument(
        "--data_dir", type=str, default=None,
        help=(
            "PanNuke data root holding prepared/ (default: <ROOT_PATH>/eval/eval_datasets/pannuke_eval_data)."
        ),
    )
    parser.add_argument(
        "--out_dir", type=str, default=None,
        help=(
            "Output folder (default: <prepared_root>/he_embeddings/HEembs_t<n>e<c>/, the only path "
            "read by _6_run_retrieval.py)."
        ),
    )

    # --- compute --------------------------------------------------------------------------------
    parser.add_argument(
        "--batch_size", type=int, default=256,
        help=(
            "Encoder inference batch size (does not change the output)."
        ),
    )
    parser.add_argument(
        "--num_workers", type=int, default=22,
        help=(
            "DataLoader workers (keep below --cpus-per-task; forced to 0 on macOS)."
        ),
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="'cuda' / 'cuda:0' / 'cpu' / 'mps'. Default: cuda if available, else mps, else cpu.",
    )
    parser.add_argument(
        "--dtype", type=str, default="fp32", choices=["fp16", "fp32"],
        help=(
            "On-disk precision of the embeddings (always computed in fp32)."
        ),
    )

    # --- staging and safety ----------------------------------------------------------------------
    parser.add_argument(
        "--stages", type=str, nargs="+", default=list(STAGES), choices=list(STAGES),
        help=(
            "Stages to run (the preflight always runs). `--stages verify` only checks an existing "
            "tree (no GPU, no writes)."
        ),
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help=(
            "Re-extract slides whose embedding h5 is already complete (incomplete ones are always "
            "re-extracted)."
        ),
    )
    parser.add_argument(
        "--full_scan", action="store_true",
        help=(
            f"Scan every row of patch_cell_emb_he for non-finite values instead of a "
            f"{pe.EMB_SAMPLE_ROWS:,}-row sample (patch_cell_emb_he_clip and patch_emb_he are always "
            "scanned in full)."
        ),
    )
    parser.add_argument(
        "--checksums", action="store_true",
        help=(
            "Record the sha256 of every output file in the manifest."
        ),
    )
    parser.add_argument(
        "--allow_partial_tree", action="store_true",
        help=(
            "Accept a prepared tree whose prepare_manifest.json has is_full_run=False (built with "
            "--tissues or --max_tiles)."
        ),
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Seed recorded in the manifest (nothing in the script is random).",
    )

    sys.exit(main(parser.parse_args()))
