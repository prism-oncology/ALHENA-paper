#!/usr/bin/env python
"""
Project each tissue's scRNA-seq reference into the checkpoint's CLIP cell space, retrieve the neighbours of each
embedded PanNuke nucleus, vote, and score the predictions against the PanNuke ground truth.
Runs in the `histost_eval_a2` env.

Saved in <ROOT_HistoST> / eval / A-2_eval_st_predictions / pannuke / results /
    - _reference_pool/<ref_stem>.npz + .json, _projected_reference_cache/: reference pools (shared by the recipes)
    - logs/step4_<timestamp>.log
    - eval_st_pannuke_<recipe>/HEencoder_t<n>e<c>_clip/
        - predictions/<pseudo_slide>.parquet, retrieval/<pseudo_slide>_topk30.npz
        - metrics/: per_tissue, per_class, chance_floor, cohort_summary, paired_tests, confidence_sweep (.csv),
          confusion/<pseudo_slide>_<population>.npy
        - diagnostics/: reference_priors, epithelial_by_source_class, border_vs_interior (.csv), coverage_report.json
        - run_manifest.json
    - cross_recipe/HEencoder_t<n>e<c>_clip/: cross_recipe_manifest.json, metrics/cross_recipe_{per_tissue,paired}.csv
"""

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

# Project imports
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src_A2.common import configure_logger, ensure_dir, now_tag, resolve_device, set_global_seed, write_dataframe, write_json
from pannuke import pannuke_embeddings as pe
from pannuke import pannuke_references as pr
from pannuke import pannuke_retrieval as prt
from pannuke.pannuke_common import (
    GEOMETRY_RECIPE_NAMES,
    PANNUKE_TISSUE_TO_HISTOST,
    normalize_tissue_name,
    pseudo_slide_id_for_tissue,
    tissue_for_pseudo_slide_id,
)

logger = logging.getLogger(__name__)

STAGES: Tuple[str, ...] = ("references", "score", "cross_recipe", "verify")


# ---------------------------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------------------------

def build_reference_pools(
    tissues_histost: Sequence[str],
    bundle_facts: Mapping[str, Any],
    device: str,
    allow_adoption: bool,
    log: logging.Logger,
) -> Dict[str, pr.ReferencePool]:
    """Build (or load from cache) one retrieval pool per tissue_HistoST (shared by the recipes)."""
    specs = pr.resolve_reference_specs(sorted(set(tissues_histost)))
    label_map = pr.coarse_label_map()
    pools: Dict[str, pr.ReferencePool] = {}
    for tissue in sorted(specs):
        started = time.time()
        pools[tissue] = pr.build_reference_pool(
            tissue,
            specs[tissue],
            bundle_facts=bundle_facts,
            label_map=label_map,
            device=device,
            allow_adoption=allow_adoption,
            log=log,
        )
        log.info("Reference pool for %s built in %.1f s.", tissue, time.time() - started)
    return pools


def score_recipe(
    scoring_preflight: prt.ScoringPreflight,
    pools: Mapping[str, pr.ReferencePool],
    *,
    training_num: str,
    checkpoint_num: str,
    bundle_facts: Mapping[str, Any],
    device: str,
    save_retrieval: bool,
    argv: Sequence[str],
    log: logging.Logger,
) -> Dict[str, Any]:
    """Score one geometry recipe end to end and write its whole results tree."""
    from src_A2.retrieval import l2_normalize

    started_at = datetime.now().isoformat(timespec="seconds")
    t0 = time.time()
    checkpoint_dir = prt.checkpoint_results_dir(
        scoring_preflight.recipe_name, training_num, checkpoint_num
    )
    ensure_dir(checkpoint_dir)

    tissue_histost_by_pannuke = pr.tissues_histost_for_pannuke()
    # L2-normalized reference matrix per tissue (recomputed for each recipe)
    ref_norm_cache: Dict[str, Any] = {}

    per_tissue_rows: List[Dict[str, Any]] = []
    per_class_frames: List[pd.DataFrame] = []
    floor_rows: List[Dict[str, Any]] = []
    prediction_frames: List[pd.DataFrame] = []
    coverage_reports = []

    for slide_id in scoring_preflight.slide_ids:
        t_slide = time.time()
        queries = prt.load_slide_queries(
            scoring_preflight,
            slide_id,
            checkpoint_dir,
            training_num,
            checkpoint_num,
            tissue_histost_by_pannuke,
        )
        pool = pools.get(queries.tissue_histost)
        if pool is None:
            raise KeyError(
                f"No reference pool for tissue_HistoST '{queries.tissue_histost}' (slide "
                f"'{slide_id}'). Run --stages references over the same --tissues."
            )
        if queries.tissue_histost not in ref_norm_cache:
            ref_norm_cache[queries.tissue_histost] = l2_normalize(pool.z_ref)

        scores = prt.score_slide(
            queries, pool, ref_z_norm=ref_norm_cache[queries.tissue_histost], log=log
        )
        prt.write_slide_outputs(checkpoint_dir, scores, save_retrieval=save_retrieval)

        per_tissue_rows.extend(scores.per_tissue)
        per_class_frames.extend(scores.per_class)
        floor_rows.extend(scores.chance_floor)
        prediction_frames.append(scores.predictions)

        headline = next(
            row for row in scores.per_tissue if row["population"] == prt.HEADLINE_POPULATION
        )
        gt_labels = scores.predictions["gt_label"].astype(str).to_numpy()
        gt_scorable = scores.predictions["gt_scorable"].to_numpy().astype(bool)
        coverage_reports.append(
            pr.coverage_report_for_slide(
                slide_id=slide_id,
                tissue_histost=queries.tissue_histost,
                pool=pool,
                gt_labels=gt_labels,
                gt_scorable_mask=gt_scorable,
            )
        )
        log.info(
            "Scored %s in %.1f s: n=%d, macro-F1=%.4f (floor %.4f), accuracy=%.4f, classes=%d",
            slide_id,
            time.time() - t_slide,
            headline["n_cells"],
            headline["macro_f1"],
            next(
                r["macro_f1"]
                for r in scores.chance_floor
                if r["population"] == prt.HEADLINE_POPULATION
            ),
            headline["accuracy"],
            headline["n_gt_present_classes"],
        )

    # Coverage check over the whole cohort
    pr.assert_coverage(coverage_reports, log=log)
    prt.write_coverage_report(checkpoint_dir, coverage_reports)

    per_tissue = pd.DataFrame(per_tissue_rows).sort_values(["population", "slide_id"])
    per_class = pd.concat(per_class_frames, ignore_index=True)
    chance_floor = pd.DataFrame(floor_rows).sort_values(["population", "slide_id"])
    predictions = pd.concat(prediction_frames, ignore_index=True)

    population_agreement = prt.assert_populations_agree(per_tissue, log=log)
    cohort = prt.aggregate_cohort(per_tissue, chance_floor, predictions)

    write_dataframe(per_tissue, checkpoint_dir / "metrics" / "per_tissue.csv")
    write_dataframe(per_class, checkpoint_dir / "metrics" / "per_class.csv")
    write_dataframe(chance_floor, checkpoint_dir / "metrics" / "chance_floor.csv")
    write_dataframe(cohort, checkpoint_dir / "metrics" / "cohort_summary.csv")
    write_dataframe(
        prt.confidence_sweep(predictions), checkpoint_dir / "metrics" / "confidence_sweep.csv"
    )
    write_dataframe(
        prt.reference_priors(pools), checkpoint_dir / "diagnostics" / "reference_priors.csv"
    )
    write_dataframe(
        prt.epithelial_by_source_class(predictions),
        checkpoint_dir / "diagnostics" / "epithelial_by_source_class.csv",
    )
    write_dataframe(
        prt.border_vs_interior(predictions),
        checkpoint_dir / "diagnostics" / "border_vs_interior.csv",
    )

    finished_at = datetime.now().isoformat(timespec="seconds")
    manifest = prt.build_run_manifest(
        recipe_name=scoring_preflight.recipe_name,
        training_num=training_num,
        checkpoint_num=checkpoint_num,
        checkpoint_dir=checkpoint_dir,
        scoring_preflight=scoring_preflight,
        bundle_facts=bundle_facts,
        pools=pools,
        coverage_reports=coverage_reports,
        population_agreement=population_agreement,
        per_tissue=per_tissue,
        cohort=cohort,
        device=device,
        argv=argv,
        started_at=started_at,
        finished_at=finished_at,
        wall_seconds=time.time() - t0,
    )
    return {
        "checkpoint_dir": checkpoint_dir,
        "per_tissue": per_tissue,
        "chance_floor": chance_floor,
        "predictions": predictions,
        "cohort": cohort,
        "manifest": manifest,
    }


def run_cross_recipe(
    results_by_recipe: Mapping[str, Mapping[str, Any]],
    scoring_preflights: Mapping[str, prt.ScoringPreflight],
    training_num: str,
    checkpoint_num: str,
    allow_cohort_change: bool,
    argv: Sequence[str],
    log: logging.Logger,
) -> Optional[Path]:
    """Compare the two geometry recipes on the intersection of their embedded cells (skipped unless both are scored)."""
    recipes = sorted(results_by_recipe)
    if len(recipes) != 2:
        log.warning(
            "Skipping the cross-recipe stage: it needs exactly two recipes and this run scored "
            "%s. Re-run with --geometry_recipe %s.",
            recipes,
            " ".join(GEOMETRY_RECIPE_NAMES),
        )
        return None

    pairing = pe.cross_recipe_pairing(
        {r: scoring_preflights[r].preflight for r in recipes},
        {r: scoring_preflights[r].emb_dir for r in recipes},
    )
    n_paired = int(pairing["n_paired_cells"])
    if n_paired != prt.EXPECTED_PAIRED_CELLS:
        message = (
            f"The cross-recipe intersection is {n_paired:,} cells, not the expected "
            f"{prt.EXPECTED_PAIRED_CELLS:,}. If the cohort changed, re-run with --allow_cohort_change "
            "and update EXPECTED_PAIRED_CELLS in pannuke_retrieval.py."
        )
        if not allow_cohort_change:
            raise RuntimeError(message)
        log.warning("%s (continuing: --allow_cohort_change was passed).", message)

    per_tissue, paired = prt.cross_recipe_comparison(
        {r: results_by_recipe[r]["predictions"] for r in recipes}, pairing
    )
    out_dir = prt.cross_recipe_results_dir(training_num, checkpoint_num)
    write_dataframe(per_tissue, out_dir / "metrics" / "cross_recipe_per_tissue.csv")
    write_dataframe(paired, out_dir / "metrics" / "cross_recipe_paired.csv")
    write_json(
        out_dir / "cross_recipe_manifest.json",
        {
            "step": 4,
            "score_schema_version": prt.SCORE_SCHEMA_VERSION,
            "recipes": recipes,
            "argv": list(argv),
            "pairing": pairing,
            "expected_paired_cells": prt.EXPECTED_PAIRED_CELLS,
            "allow_cohort_change": bool(allow_cohort_change),
            "population": prt.HEADLINE_POPULATION,
            "primary_metric": prt.PRIMARY_METRIC,
            "bh_family": prt.FAMILY_CROSS_RECIPE,
            "note": (
                "Computed on the intersection of the two embedded cell sets, a different "
                "population from each recipe's own per-tissue table."
            ),
        },
    )
    log.info(
        "Cross-recipe: %d paired cells, agreement %.4f, delta(%s) = %.4f (p = %s).",
        n_paired,
        float(paired["cell_level_agreement"].iloc[0]),
        prt.PRIMARY_METRIC,
        float(paired["delta"].iloc[0]),
        f"{float(paired['p_value'].iloc[0]):.4g}",
    )
    return out_dir


# ---------------------------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------------------------

def resolve_requested_slides(tissues: Optional[Sequence[str]]) -> Optional[List[str]]:
    """Pseudo-slide ids of the --tissues PanNuke tissues (e.g. Prostate -> pannuke_prostate), or None."""
    if not tissues:
        return None
    known = {normalize_tissue_name(t): t for t in PANNUKE_TISSUE_TO_HISTOST}
    out: List[str] = []
    for raw in tissues:
        key = normalize_tissue_name(str(raw))
        if key not in known:
            raise ValueError(
                f"--tissues names {raw!r}, which is not one of the {len(known)} retained PanNuke "
                f"tissues: {sorted(PANNUKE_TISSUE_TO_HISTOST)}."
            )
        out.append(pseudo_slide_id_for_tissue(known[key]))
    return sorted(set(out))


def main(args: argparse.Namespace) -> int:
    stages = [s for s in STAGES if s in set(args.stages)]
    if not stages:
        raise ValueError(f"--stages selected nothing; choose from {list(STAGES)}.")
    recipes = [r for r in GEOMETRY_RECIPE_NAMES if r in set(args.geometry_recipe)]
    if not recipes:
        raise ValueError(
            f"--geometry_recipe selected nothing; choose from {list(GEOMETRY_RECIPE_NAMES)}."
        )

    # Set before any path helper is called (with --data_dir, inputs and outputs are in that folder)
    pr.set_data_root(Path(args.data_dir) if args.data_dir else None)

    log_path = pr.results_root() / "logs" / f"step4_{now_tag()}.log"
    log = configure_logger(log_path, name="A2_pannuke_step4", level=getattr(logging, args.log_level))
    log.info("Retrieval + scoring: recipes=%s stages=%s", recipes, stages)
    # Input and output roots
    log.info("Dataset root (inputs): %s", pr.data_root())
    log.info("Results root (outputs): %s", pr.results_root())
    log.info("Invocation log: %s", log_path)

    set_global_seed(prt.SEED)
    device = resolve_device(args.device)
    log.info("Seed=%d device=%s", prt.SEED, device)

    requested_slides = resolve_requested_slides(args.tissues)
    if requested_slides:
        log.warning(
            "--tissues restricts this run to %s: the cohort summary, paired tests and diagnostics "
            "describe this subset only.",
            requested_slides,
        )

    bundle_facts = pr.read_st_cell_bundle_facts(
        pr.st_cell_bundle_path(args.training_num, args.checkpoint_num)
    )
    log.info(
        "ST cell bundle: scconcept_training_id=%s in_dim=%d token_embed_dim=%d lambda_token=%.4g",
        bundle_facts["scconcept_training_id"],
        bundle_facts["in_dim"],
        bundle_facts["token_embed_dim"],
        bundle_facts["lambda_token"],
    )

    # The preflight always runs
    scoring_preflights: Dict[str, prt.ScoringPreflight] = {}
    for recipe in recipes:
        scoring_preflights[recipe] = prt.preflight_for_scoring(
            recipe_name=recipe,
            training_num=args.training_num,
            checkpoint_num=args.checkpoint_num,
            slides=requested_slides,
            allow_partial_tree=args.allow_partial_tree,
            log=log,
        )
        log.info(
            "Preflight OK for '%s': %d pseudo-slide(s), %d patches, %d token-eligible cells.",
            recipe,
            len(scoring_preflights[recipe].slide_ids),
            scoring_preflights[recipe].preflight.total("n_patches"),
            scoring_preflights[recipe].preflight.total("n_expected_groups"),
        )

    if "verify" in stages and not ({"references", "score"} & set(stages)):
        for recipe in recipes:
            checkpoint_dir = prt.checkpoint_results_dir(
                recipe, args.training_num, args.checkpoint_num
            )
            prt.verify_outputs(checkpoint_dir, scoring_preflights[recipe], log=log)
        log.info("--stages verify: nothing written.")
        return 0

    # Tissues of the reference pools: union over the requested recipes
    tissue_by_pannuke = pr.tissues_histost_for_pannuke()
    tissues_histost = sorted(
        {
            tissue_by_pannuke[tissue_for_pseudo_slide_id(sid)]
            for recipe in recipes
            for sid in scoring_preflights[recipe].slide_ids
        }
    )

    pools: Dict[str, pr.ReferencePool] = {}
    if {"references", "score"} & set(stages):
        pools = build_reference_pools(
            tissues_histost,
            bundle_facts,
            device=device,
            allow_adoption=not args.recompute_reference_filter,
            log=log,
        )
        log.info(
            "Built %d reference pool(s), %d cells total.",
            len(pools),
            sum(p.n_rows for p in pools.values()),
        )
    if "score" not in stages:
        log.info("--stages %s: reference pools are built and cached; stopping.", " ".join(stages))
        return 0

    results_by_recipe: Dict[str, Dict[str, Any]] = {}
    for recipe in recipes:
        checkpoint_dir = prt.checkpoint_results_dir(recipe, args.training_num, args.checkpoint_num)
        if prt.run_manifest_path(checkpoint_dir).exists() and not args.overwrite:
            raise FileExistsError(
                f"{prt.run_manifest_path(checkpoint_dir)} already exists. Re-run with --overwrite "
                "to replace this recipe's results."
            )
        log.info("=== recipe '%s' -> %s ===", recipe, checkpoint_dir)
        result = score_recipe(
            scoring_preflights[recipe],
            pools,
            training_num=args.training_num,
            checkpoint_num=args.checkpoint_num,
            bundle_facts=bundle_facts,
            device=device,
            save_retrieval=not args.no_save_retrieval,
            argv=sys.argv,
            log=log,
        )
        results_by_recipe[recipe] = result

    paired = prt.paired_he_vs_floor(
        {r: results_by_recipe[r]["per_tissue"] for r in recipes},
        {r: results_by_recipe[r]["chance_floor"] for r in recipes},
    )
    for recipe in recipes:
        checkpoint_dir = results_by_recipe[recipe]["checkpoint_dir"]
        write_dataframe(
            paired.loc[paired["recipe"] == recipe], checkpoint_dir / "metrics" / "paired_tests.csv"
        )
        manifest = dict(results_by_recipe[recipe]["manifest"])
        manifest["invocation_log"] = str(log_path)
        manifest["paired_tests"] = paired.loc[paired["recipe"] == recipe].to_dict("records")
        write_json(prt.run_manifest_path(checkpoint_dir), manifest)
        headline = results_by_recipe[recipe]["cohort"]
        headline = headline.loc[
            (headline["population"] == prt.HEADLINE_POPULATION)
            & (headline["summary_scope"] == "unweighted_tissue_mean")
        ]
        for _, row in headline.iterrows():
            log.info(
                "[%s] %s: H&E %.4f vs floor %.4f (delta %+.4f), unweighted mean over %d tissue(s).",
                recipe,
                row["metric"],
                row["he_value"],
                row["floor_value"],
                row["he_minus_floor"],
                int(row["n_tissues"]),
            )

    if "cross_recipe" in stages:
        run_cross_recipe(
            results_by_recipe,
            scoring_preflights,
            args.training_num,
            args.checkpoint_num,
            allow_cohort_change=args.allow_cohort_change,
            argv=sys.argv,
            log=log,
        )

    if "verify" in stages:
        for recipe in recipes:
            prt.verify_outputs(
                results_by_recipe[recipe]["checkpoint_dir"], scoring_preflights[recipe], log=log
            )

    log.info("Done. Invocation log: %s", log_path)
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Project each tissue's scRNA-seq reference into the checkpoint's CLIP cell space, "
            "retrieve the neighbours of every embedded PanNuke nucleus, vote, and score against the "
            "PanNuke ground truth, with the (k, tau) of the STHELAR run."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--geometry_recipe", type=str, nargs="+", required=True,
        choices=list(GEOMETRY_RECIPE_NAMES),
        help=(
            "Prepared tree(s) to score (the cross_recipe stage needs both)."
        ),
    )
    parser.add_argument(
        "--training_num", type=str, required=True,
        help="HistoST training number of the checkpoint to score, e.g. 46 for training_46.",
    )
    parser.add_argument(
        "--checkpoint_num", type=str, required=True,
        help="Checkpoint epoch of that training, e.g. 13 for checkpoint_epoch_13.pth (tag t46e13).",
    )
    parser.add_argument(
        "--stages", type=str, nargs="+", default=list(STAGES), choices=list(STAGES),
        help=(
            "Stages to run (the preflight always runs): 'references' builds and caches the reference "
            "pools; 'score' retrieves, votes, scores and writes the per-recipe tables; 'cross_recipe' "
            "compares the two recipes on their common cells; 'verify' checks the outputs on disk "
            "(no writes)."
        ),
    )
    parser.add_argument(
        "--tissues", type=str, nargs="*", default=None,
        help=(
            "Restrict the run to these PanNuke tissues (e.g. Prostate Kidney)."
        ),
    )
    parser.add_argument(
        "--data_dir", type=str, default=None,
        help=(
            "Folder from which prepared/ is read and to which results/ is written. Default: "
            "prepared/ under eval/eval_datasets/pannuke_eval_data/ and results/ under "
            "eval/A-2_eval_st_predictions/pannuke/."
        ),
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        help=(
            "Device for the reference projection forward pass: auto | cuda | mps | cpu. Retrieval "
            "and scoring are CPU/faiss regardless."
        ),
    )
    parser.add_argument(
        "--recompute_reference_filter", action="store_true",
        help=(
            "Recompute the reference row filter instead of reusing a cached result (reads the full "
            "expression matrix of each reference)."
        ),
    )
    parser.add_argument(
        "--allow_partial_tree", action="store_true",
        help=(
            "Accept a prepared tree built with is_full_run=False (_4_prepare_patches.py with "
            "--tissues or --max_tiles)."
        ),
    )
    parser.add_argument(
        "--allow_cohort_change", action="store_true",
        help=(
            f"Warn instead of failing when the cross-recipe intersection is not "
            f"EXPECTED_PAIRED_CELLS ({prt.EXPECTED_PAIRED_CELLS:,} cells)."
        ),
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help=(
            "Replace the existing results tree of a recipe (otherwise an existing "
            "run_manifest.json raises)."
        ),
    )
    parser.add_argument(
        "--no_save_retrieval", action="store_true",
        help=(
            f"Skip writing retrieval/{{slide}}_topk{prt.SELECTED_K}.npz (top-k neighbours of each cell)."
        ),
    )
    parser.add_argument(
        "--log_level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity for both the console and the invocation log file.",
    )
    return parser


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    sys.exit(main(build_arg_parser().parse_args()))
