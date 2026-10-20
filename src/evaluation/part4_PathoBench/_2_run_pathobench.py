#!/usr/bin/env python
"""
Run the PathoBench slide-level probes (Task D) on the Trident features computed by _1_extract_features.py.

Saved in <ROOT_HistoST> / eval / D_eval_pathobench / <eval_id> /
    - manifest.json: eval_id protocol
    - arms/<tag>_<space>_<head>.json: record of each (encoder, space, head) run
    - HEencoder_t<n>e<c>_<space>/<head>/metrics.json: AUC per task
"""

import argparse
import csv
import hashlib
import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

# Add the project root to sys.path so that 'src' can be imported
project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.evaluation.part4_PathoBench._pathobench_common import (
    ABMIL_DEFAULTS,
    ABMIL_MODEL_KWARGS,
    ABMIL_MODEL_NAME,
    COMBINE_SLIDES_PER_PATIENT,
    DECLARED_EXCLUDED_TASKS,
    DEFAULT_ABMIL_SEEDS,
    DEFAULT_METRIC,
    EXCLUSION_REGISTRY_VERSION,
    HEADS,
    LINPROBE_DEFAULTS,
    LINPROBE_POOL_MODEL,
    SPACES,
    TASKD_REGISTRY,
    TASKD_REGISTRY_VERSION,
    TASKD_SCHEMA_VERSION,
    TaskDTask,
    arm_record_path,
    audit_registry_against_catalog,
    available_splits_path,
    build_taskd_manifest,
    encoder_tag,
    eval_root,
    excluded_slides,
    expects_clip,
    install_appledouble_tolerant_fs_ops,
    list_h5,
    pathobench_assets_root,
    parse_encoder_tag,
    pool_recipe_id,
    pooled_embeddings_root,
    probe_output_root,
    read_available_splits,
    slide_inventory_path,
    split_task_id,
    trident_features_root,
    validate_declared_exclusions,
    validate_exclusion_registry,
)

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- resume (fingerprinted sidecars)

# Resume sidecar format version
RESUME_SCHEMA_VERSION = "2"

TASK_SIDECAR_NAME = "_taskd_task_result.json"
SEED_SIDECAR_NAME = "_taskd_seed_result.json"


@dataclass(frozen=True)
class FeatureSignature:
    """Signature of one (cohort, encoder, space) feature dir: slide count, hash of the slide names, feature dim."""

    n_slides: int
    slides_sha: str
    dim: int


def feature_signature(features_dir: Path, cache: Dict[Path, Optional["FeatureSignature"]]) -> Optional[FeatureSignature]:
    """Signature of a feature dir (memoized in cache), or None if it holds no .h5 file."""
    if features_dir in cache:
        return cache[features_dir]
    import h5py  # lazy import (pathobench/trident envs)

    files = list_h5(features_dir)  # sorted, dotfiles skipped
    sig: Optional[FeatureSignature] = None
    if files:
        with h5py.File(files[0], "r") as f:
            dim = int(f["features"].shape[1])
        payload = "|".join(p.stem for p in files)
        sig = FeatureSignature(len(files), hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16], dim)
    cache[features_dir] = sig
    return sig


def result_fingerprint(
    *, tag: str, space: str, head: str, task_id: str, feature_sig: FeatureSignature, seeds: Sequence[int]
) -> str:
    """Hash of everything that determines a probe result (identity, protocol constants, feature dim, slide set, seeds)."""
    protocol: Dict[str, object] = (
        {"linprobe_defaults": dict(LINPROBE_DEFAULTS), "pool_model": LINPROBE_POOL_MODEL}
        if head == "linprobe"
        else {"abmil_defaults": dict(ABMIL_DEFAULTS), "model_name": ABMIL_MODEL_NAME,
              "model_kwargs": dict(ABMIL_MODEL_KWARGS)}
    )
    payload = json.dumps(
        {
            "resume_schema": RESUME_SCHEMA_VERSION,
            "registry_version": TASKD_REGISTRY_VERSION,
            "exclusion_registry_version": EXCLUSION_REGISTRY_VERSION,
            "tag": str(tag), "space": str(space), "head": str(head), "task_id": str(task_id),
            "metric": DEFAULT_METRIC,
            "combine_slides_per_patient": bool(COMBINE_SLIDES_PER_PATIENT),
            "protocol": protocol,
            "feature_dim": int(feature_sig.dim),
            "n_feature_slides": int(feature_sig.n_slides),
            "feature_slides_sha": str(feature_sig.slides_sha),
            "seeds": [int(s) for s in seeds],
        },
        sort_keys=True, default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def seed_dir_name(head: str, seed: int) -> str:
    """PathoBench saveto subdirectory of one (task, head, seed) run."""
    return f"seed_{int(seed)}" if head == "abmil" else "run"


# --------------------------------------------------------------------------- fold coverage


def read_fold_coverage(saveto: Path) -> Tuple[Optional[int], Optional[int]]:
    """(n_folds_scored, n_folds_total) of a finished PathoBench run, or (None, None) if there are no per-fold results."""
    fold_dirs = sorted(p for p in (Path(saveto) / "test_metrics").glob("fold_*") if p.is_dir())
    if not fold_dirs:
        return None, None
    scored = 0
    for d in fold_dirs:
        try:
            obj = json.loads((d / "metrics.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if (obj.get("overall") or {}).get(DEFAULT_METRIC) is not None:
            scored += 1
    return scored, len(fold_dirs)


# --------------------------------------------------------------------------- pooled-cache sharing


def assert_cohort_pooling_compatible(cohort: str, tasks: Sequence[TaskDTask]) -> str:
    """Check that all tasks of a cohort use the same sample_col (they share one pooled cache). Returns it."""
    from patho_bench.SplitFactory import SplitFactory  # type: ignore  # lazy: pathobench env

    install_appledouble_tolerant_fs_ops()
    by_col: Dict[str, List[str]] = {}
    for task in tasks:
        source, task_name = split_task_id(task.task_id)
        _split, config_path = SplitFactory.from_hf(str(pathobench_assets_root()), source, task_name)
        col = str(SplitFactory.get_task_info(path_to_config=str(config_path))["sample_col"])
        by_col.setdefault(col, []).append(task.task_id)
    if len(by_col) > 1:
        raise SystemExit(
            f"cohort={cohort}: its tasks disagree on `sample_col`, so they cannot share one pooled "
            "embedding cache:\n  "
            + "\n  ".join(f"{col}: {', '.join(sorted(ids))}" for col, ids in sorted(by_col.items()))
            + "\nGive these tasks their own eval_id, or also key `pool_recipe_id` on sample_col."
        )
    return next(iter(by_col))


def _read_sidecar(path: Path, fingerprint: str) -> Optional[Dict[str, object]]:
    """Return the sidecar payload if it matches fingerprint, else None (missing or unreadable files are ignored)."""
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning("Ignoring unreadable resume sidecar %s (%s) — recomputing.", path, exc)
        return None
    if not isinstance(payload, dict) or payload.get("fingerprint") != fingerprint:
        return None
    return payload


def _write_sidecar(path: Path, fingerprint: str, record: Dict[str, object]) -> None:
    """Write a resume sidecar atomically (tmp file + replace)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps({"fingerprint": fingerprint, **record}, indent=2, sort_keys=True) + "\n",
                   encoding="utf-8")
    tmp.replace(path)


# Task ids already validated in this process
_VALIDATED_TASK_IDS: Set[str] = set()


def ensure_task_ids_validated(tasks: Sequence[TaskDTask]) -> None:
    """Run validate_task_ids on the tasks not validated yet."""
    todo = [t for t in tasks if t.task_id not in _VALIDATED_TASK_IDS]
    if not todo:
        return
    validate_task_ids(todo)
    _VALIDATED_TASK_IDS.update(t.task_id for t in todo)


# --------------------------------------------------------------------------- coverage reconciliation


@dataclass(frozen=True)
class CohortCoverage:
    """Split slides vs extracted slides of one cohort, for one (encoder, space)."""

    cohort: str
    split_slides: Tuple[str, ...]    # every slide the cohort's shipped splits reference
    feature_slides: Tuple[str, ...]  # every slide with a `.h5` under this (encoder, space)
    kept: Tuple[str, ...]            # split ∩ features
    dropped: Tuple[str, ...]         # split slides with no .h5 (declared exclusions)


def read_slide_inventory(cohort: str) -> Dict[str, Dict[str, str]]:
    """{slide_id: row} from the slide_inventory.tsv written by _1_extract_features.py."""
    path = slide_inventory_path(cohort)
    if not path.exists():
        raise SystemExit(
            f"cohort={cohort}: no slide inventory at {path} — run _1_extract_features.py for this "
            "cohort first."
        )
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    return {str(r.get("slide_id", "")).strip(): r for r in rows if str(r.get("slide_id", "")).strip()}


def read_split_slide_ids(split_path: Path) -> List[str]:
    """Unique slide_id values of a PathoBench split TSV, in order."""
    ids: List[str] = []
    with Path(split_path).open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        if "slide_id" not in (reader.fieldnames or []):
            raise SystemExit(f"Split {split_path} has no 'slide_id' column (columns={reader.fieldnames}).")
        for row in reader:
            sid = str(row.get("slide_id", "")).strip()
            if sid and sid.lower() != "nan":
                ids.append(sid)
    return list(dict.fromkeys(ids))


def cohort_split_slide_ids(cohort: str) -> List[str]:
    """Union of the slide ids referenced by the registry tasks of a cohort."""
    from patho_bench.SplitFactory import SplitFactory  # type: ignore  # lazy: pathobench env

    install_appledouble_tolerant_fs_ops()
    ids: List[str] = []
    for task in (t for t in TASKD_REGISTRY if t.cohort == cohort):
        source, task_name = split_task_id(task.task_id)
        split_path, _config = SplitFactory.from_hf(str(pathobench_assets_root()), source, task_name)
        ids.extend(read_split_slide_ids(Path(split_path)))
    return list(dict.fromkeys(ids))


def reconcile_cohort_coverage(
    cohort: str,
    features_dir: Path,
    split_slide_ids: Sequence[str],
    *,
    allow_unexpected_missing: bool = False,
) -> CohortCoverage:
    """
    Check the split slides of a cohort against the slide inventory and the extracted features.
    Raises on any missing slide that is not in KNOWN_EXCLUDED_SLIDES (warns if allow_unexpected_missing).
    """
    inventory = read_slide_inventory(cohort)
    frozen = excluded_slides(cohort)
    split_ids = list(dict.fromkeys(str(s) for s in split_slide_ids))
    feature_ids = {p.stem for p in list_h5(features_dir)}

    problems: List[str] = []

    # 1. every split slide is in the inventory
    unseen = sorted(set(split_ids) - set(inventory))
    if unseen:
        problems.append(
            f"{len(unseen)} slide(s) referenced by the shipped splits are absent from "
            f"{slide_inventory_path(cohort)} — `_0`/`_1` never processed them: {unseen[:5]}"
            + (" …" if len(unseen) > 5 else "")
        )

    # 2. every rejected slide is a declared exclusion, with the same reason
    rejected = {sid: r for sid, r in inventory.items()
                if str(r.get("slide_status", "")).strip() == "rejected" and sid in set(split_ids)}
    unexpected_rejections = sorted(set(rejected) - set(frozen))
    if unexpected_rejections:
        problems.append(
            "the slide gate rejected slide(s) that are not in the exclusion registry "
            f"({EXCLUSION_REGISTRY_VERSION}): "
            + ", ".join(f"{sid} ({rejected[sid].get('failure_reason') or 'unknown'})"
                        for sid in unexpected_rejections[:5])
            + (" …" if len(unexpected_rejections) > 5 else "")
            + " — add them to KNOWN_EXCLUDED_SLIDES (and bump EXCLUSION_REGISTRY_VERSION) if they cannot be fixed."
        )
    reason_drift = sorted(
        f"{sid}: registry says {frozen[sid]!r}, inventory says "
        f"{str(rejected[sid].get('failure_reason') or '').strip()!r}"
        for sid in set(rejected) & set(frozen)
        if str(rejected[sid].get("failure_reason") or "").strip() != frozen[sid]
    )
    if reason_drift:
        problems.append("frozen exclusions rejected for a DIFFERENT reason than recorded: " + "; ".join(reason_drift))

    # 3. every kept slide has features for this (encoder, space)
    kept_ids = {sid for sid, r in inventory.items()
                if str(r.get("slide_status", "")).strip() == "kept" and sid in set(split_ids)}
    missing_features = sorted(kept_ids - feature_ids)
    if missing_features:
        problems.append(
            f"{len(missing_features)} slide(s) passed the gate but have no `.h5` under {features_dir} "
            f"— a per-encoder Trident drop: {missing_features[:5]}" + (" …" if len(missing_features) > 5 else "")
        )

    dropped = sorted(set(split_ids) - feature_ids)
    unexpected_dropped = sorted(set(dropped) - set(frozen))
    if unexpected_dropped and not problems:
        # Gap not explained by the checks above
        problems.append(
            f"{len(unexpected_dropped)} split slide(s) have no features and are not explained by the "
            f"inventory or the exclusion registry: {unexpected_dropped[:5]}"
            + (" …" if len(unexpected_dropped) > 5 else "")
        )

    if problems:
        msg = (f"cohort={cohort} features={features_dir}: slide coverage does not reconcile.\n  - "
               + "\n  - ".join(problems)
               + f"\nInventory: {slide_inventory_path(cohort)}. Pass "
                 "--allow_unexpected_missing_slides to proceed anyway.")
        if allow_unexpected_missing:
            logger.warning("%s", msg)
        else:
            raise SystemExit(msg)

    if dropped:
        logger.info("cohort=%s: %d/%d split slide(s) excluded as declared (%s): %s", cohort, len(dropped),
                    len(split_ids), EXCLUSION_REGISTRY_VERSION, ", ".join(dropped))
    return CohortCoverage(
        cohort=str(cohort),
        split_slides=tuple(split_ids),
        feature_slides=tuple(sorted(feature_ids)),
        kept=tuple(sorted(set(split_ids) & feature_ids)),
        dropped=tuple(dropped),
    )


def write_pruned_split(split_path: Path, config_path: Path, keep_slides: Set[str], saveto: Path) -> Dict[str, object]:
    """
    Copy the PathoBench split TSV without the rows of slides that have no extracted features.
    Saved as <saveto>/split_pruned.tsv. Returns the pruning record.
    """
    from patho_bench.SplitFactory import SplitFactory  # type: ignore  # lazy: pathobench env

    sample_col = str(SplitFactory.get_task_info(path_to_config=str(config_path))["sample_col"])
    with Path(split_path).open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    for col in ("slide_id", sample_col):
        if col not in fieldnames:
            raise SystemExit(f"Split {split_path} has no {col!r} column (columns={fieldnames}).")

    def _sid(row: Dict[str, str]) -> str:
        sid = str(row.get("slide_id", "")).strip()
        return "" if sid.lower() == "nan" else sid

    kept_rows = [r for r in rows if _sid(r) in keep_slides]
    dropped = sorted({_sid(r) for r in rows if _sid(r) and _sid(r) not in keep_slides})
    n_rows_without_slide_id = sum(1 for r in rows if not _sid(r))
    cases_before = {str(r.get(sample_col, "")) for r in rows}
    cases_after = {str(r.get(sample_col, "")) for r in kept_rows}
    if not kept_rows:
        raise SystemExit(f"Pruning {split_path} against {len(keep_slides)} extracted slide(s) left no rows.")

    out = Path(saveto) / "split_pruned.tsv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(kept_rows)
    return {
        "split_path": str(out),
        "n_slides_dropped": len(dropped),
        "dropped_slides": dropped,
        "n_rows_without_slide_id": n_rows_without_slide_id,
        "n_cases_used": len(cases_after),
        "n_cases_dropped": len(cases_before - cases_after),
    }


# --------------------------------------------------------------------------- PathoBench (external)


def validate_task_ids(tasks: Sequence[TaskDTask]) -> None:
    """Resolve every task id with the installed PathoBench (raises on unknown ids) and cache its split files."""
    try:
        from patho_bench.SplitFactory import SplitFactory  # type: ignore  # lazy: pathobench env
    except Exception as exc:  # pragma: no cover - env-dependent
        raise SystemExit(
            "PathoBench is not importable. Activate the `pathobench` env and install patho-bench "
            f"(+ trident + ipython; see docs/installation.md). Original error: {exc}"
        )
    # PathoBench cache cleanup can fail on macOS ._* files
    install_appledouble_tolerant_fs_ops()
    missing: List[str] = []
    for t in tasks:
        source, task_name = split_task_id(t.task_id)
        try:
            SplitFactory.from_hf(str(pathobench_assets_root()), source, task_name)
        except Exception as exc:  # noqa: BLE001 - collect all bad ids before failing
            missing.append(f"{t.task_id} ({type(exc).__name__}: {exc})")
    if missing:
        raise SystemExit(
            "These Task D registry ids could not be resolved from the installed PathoBench:\n  "
            + "\n  ".join(missing)
            + "\nCheck them against PathoBench's HuggingFace task list (MahmoodLab/Patho-Bench)."
        )


def run_pathobench_task(
    task: TaskDTask,
    features_dir: Path,
    head: str,
    work_root: Path,
    *,
    input_feature_dim: int,
    seeds: Sequence[int],
    gpu: int,
    coverage: CohortCoverage,
    pooled_dir: Path,
    seed_fingerprint: Callable[[int], str],
    resume: bool = True,
) -> Dict[str, object]:
    """
    Run one PathoBench probe (linprobe, or ABMIL once per seed) on a task and return its metrics record.
    The shipped split is first pruned to the extracted slides; each finished seed is saved as a resume sidecar.
    """
    from patho_bench.SplitFactory import SplitFactory  # type: ignore  # lazy: pathobench env
    from patho_bench.ExperimentFactory import ExperimentFactory  # type: ignore
    from src.utils.utils import set_global_seed

    # PathoBench cache cleanup can fail on macOS ._* files
    install_appledouble_tolerant_fs_ops()
    source, task_name = split_task_id(task.task_id)
    split_path, task_config_path = SplitFactory.from_hf(str(pathobench_assets_root()), source, task_name)
    pruning = write_pruned_split(Path(split_path), Path(task_config_path), set(coverage.kept),
                                 work_root / task.task_id)

    seed_list = list(seeds) if head == "abmil" else [0]
    per_seed: List[Dict[str, float]] = []
    fold_coverage: List[Tuple[Optional[int], Optional[int]]] = []
    n_reused = 0
    for seed in seed_list:
        saveto = work_root / task.task_id / seed_dir_name(head, seed)
        seed_fp = seed_fingerprint(int(seed))
        cached_seed = _read_sidecar(saveto / SEED_SIDECAR_NAME, seed_fp) if resume else None
        if cached_seed is not None:
            per_seed.append({"seed": int(seed), "score": float(cached_seed["score"])})
            fold_coverage.append((cached_seed.get("n_folds_scored"), cached_seed.get("n_folds_total")))
            n_reused += 1
            logger.info("  task=%s head=%s seed=%d: resumed from sidecar (%s=%.4f).",
                        task.task_id, head, seed, DEFAULT_METRIC, float(cached_seed["score"]))
            continue
        saveto.mkdir(parents=True, exist_ok=True)
        if head == "abmil":
            set_global_seed(int(seed))  # seeds the ABMIL head init
            experiment = ExperimentFactory.finetune(
                split=str(pruning["split_path"]), task_config=str(task_config_path),
                patch_embeddings_dirs=[str(features_dir)], saveto=str(saveto),
                combine_slides_per_patient=COMBINE_SLIDES_PER_PATIENT,
                model_name=ABMIL_MODEL_NAME,
                model_kwargs={**ABMIL_MODEL_KWARGS, "input_feature_dim": int(input_feature_dim)},
                gpu=int(gpu), **ABMIL_DEFAULTS,
            )
            experiment.seed = int(seed)  # train() calls self.set_seed(self.seed)
        else:
            pooled_dir.mkdir(parents=True, exist_ok=True)
            experiment = ExperimentFactory.linprobe(
                split=str(pruning["split_path"]), task_config=str(task_config_path),
                pooled_embeddings_dir=str(pooled_dir), saveto=str(saveto),
                combine_slides_per_patient=COMBINE_SLIDES_PER_PATIENT,
                patch_embeddings_dirs=[str(features_dir)], model_name=LINPROBE_POOL_MODEL,
                gpu=int(gpu), **LINPROBE_DEFAULTS,
            )
        experiment.train()
        experiment.test()  # writes {saveto}/test_metrics_summary.json (report_results reads it)
        score = experiment.report_results(metric=DEFAULT_METRIC)
        n_scored, n_total = read_fold_coverage(saveto)
        if n_scored is not None and n_total is not None and n_scored < n_total:
            # Folds whose test split misses a class have no score
            logger.warning("  task=%s head=%s seed=%d: %s computed on %d/%d CV folds (folds whose "
                           "test split misses a class have no score).", task.task_id, head, seed, DEFAULT_METRIC, n_scored, n_total)
        if score is not None:
            per_seed.append({"seed": int(seed), "score": float(score)})
            fold_coverage.append((n_scored, n_total))
            # Save the seed result for resume (None scores are not cached)
            _write_sidecar(saveto / SEED_SIDECAR_NAME, seed_fp, {
                "task_id": task.task_id, "head": head, "seed": int(seed),
                "metric": DEFAULT_METRIC, "score": float(score),
                "n_folds_scored": n_scored, "n_folds_total": n_total,
            })

    coverage_record = {
        "n_cases_used": pruning["n_cases_used"],
        "n_cases_dropped": pruning["n_cases_dropped"],
        "n_slides_dropped": pruning["n_slides_dropped"],
        "dropped_slides": pruning["dropped_slides"],
        "pooled_embeddings_dir": str(pooled_dir) if head == "linprobe" else None,
    }
    if not per_seed:
        return {"status": "no_result", "score_mean": None, **coverage_record}
    scores = np.array([s["score"] for s in per_seed], dtype=float)
    # Fold coverage: scored folds of the worst seed vs total folds
    scored_vals = [s for s, _t in fold_coverage if s is not None]
    total_vals = [t for _s, t in fold_coverage if t is not None]
    return {
        "status": "ok",
        "score_mean": float(np.mean(scores)),
        "score_std": float(np.std(scores)),   # 0 for the deterministic linear probe (single run)
        "n_seeds": len(seed_list),
        "n_folds_scored": min(scored_vals) if scored_vals else None,
        "n_folds_total": max(total_vals) if total_vals else None,
        "per_seed": per_seed,
        **coverage_record,
    }


# --------------------------------------------------------------------------- per (encoder, space, head)


def run_encoder_space_head(
    eval_id: str,
    training_num: int,
    checkpoint_num: int,
    space: str,
    head: str,
    *,
    tasks: Sequence[TaskDTask],
    seeds: Sequence[int],
    gpu: int,
    allow_unexpected_missing: bool = False,
    resume: bool = True,
    feature_sig_cache: Optional[Dict[Path, Optional[FeatureSignature]]] = None,
) -> Optional[Dict[str, int]]:
    """Run all tasks for one (encoder, space, head). Returns outcome counts, or None if the encoder has no such space."""
    tag = encoder_tag(training_num, checkpoint_num)
    if space == "clip" and not expects_clip(training_num, checkpoint_num):
        logger.info("encoder=%s space=clip: skipped (he-only bundle has no clip).", tag)
        return None

    out = probe_output_root(eval_id, training_num, checkpoint_num, space, head)
    work_root = out / "_pathobench_work"  # PathoBench's own artifacts (split.csv, metrics, ckpts)
    sig_cache = feature_sig_cache if feature_sig_cache is not None else {}

    # Coverage is checked once per cohort
    coverage_cache: Dict[str, CohortCoverage] = {}
    task_records: List[Dict[str, object]] = []
    n_resumed = 0
    for task in tasks:
        features_dir = trident_features_root(task.cohort, training_num, checkpoint_num) / space
        base = {"task_id": task.task_id, "family": task.family, "cohort": task.cohort,
                "label": task.label, "metric": task.metric}
        # Feature signature (None if no features)
        sig = feature_signature(features_dir, sig_cache)
        if sig is None:
            task_records.append({**base, "status": "skipped_no_features", "score_mean": None})
            logger.warning("encoder=%s space=%s task=%s: no features at %s (skipped).", tag, space, task.task_id, features_dir)
            continue

        seed_list = list(seeds) if head == "abmil" else [0]
        task_fp = result_fingerprint(tag=tag, space=space, head=head, task_id=task.task_id,
                                     feature_sig=sig, seeds=seed_list)
        task_sidecar = work_root / task.task_id / TASK_SIDECAR_NAME
        cached_task = _read_sidecar(task_sidecar, task_fp) if resume else None
        if cached_task is not None:
            task_records.append({**base, **cached_task["record"]})  # type: ignore[index]
            n_resumed += 1
            logger.info("encoder=%s space=%s head=%s task=%s: resumed (%s=%.4f).", tag, space, head,
                        task.task_id, DEFAULT_METRIC, float(cached_task["record"]["score_mean"]))  # type: ignore[index]
            continue

        try:
            ensure_task_ids_validated(tasks)
            if task.cohort not in coverage_cache:
                coverage_cache[task.cohort] = reconcile_cohort_coverage(
                    task.cohort, features_dir, cohort_split_slide_ids(task.cohort),
                    allow_unexpected_missing=allow_unexpected_missing,
                )
                if head == "linprobe":
                    # Linear probe: all tasks of a cohort share one pooled cache
                    assert_cohort_pooling_compatible(
                        task.cohort, [t for t in TASKD_REGISTRY if t.cohort == task.cohort]
                    )
            coverage = coverage_cache[task.cohort]
            pooled_dir = pooled_embeddings_root(
                task.cohort, training_num, checkpoint_num, space, pool_recipe_id(coverage.kept)
            )
            res = run_pathobench_task(
                task, features_dir, head, work_root,
                input_feature_dim=sig.dim, seeds=seeds, gpu=gpu,
                coverage=coverage, pooled_dir=pooled_dir, resume=resume,
                seed_fingerprint=lambda s, _t=task.task_id, _sig=sig: result_fingerprint(
                    tag=tag, space=space, head=head, task_id=_t, feature_sig=_sig, seeds=[s]
                ),
            )
        except SystemExit:
            raise
        except Exception as exc:  # a failing task does not stop the run
            logger.warning("encoder=%s space=%s head=%s task=%s failed: %s", tag, space, head, task.task_id, exc)
            res = {"status": "error", "score_mean": None, "error": str(exc)}
        # Only completed tasks are cached (failed tasks are retried on the next run)
        if res.get("status") == "ok":
            _write_sidecar(task_sidecar, task_fp, {"record": res, "seeds": seed_list})
        task_records.append({**base, **res})

    payload = {
        "eval_id": eval_id,
        "training_num": training_num,
        "checkpoint_num": checkpoint_num,
        "tag": tag,
        "space": space,
        "head": head,
        "metric": DEFAULT_METRIC,
        "exclusion_registry_version": EXCLUSION_REGISTRY_VERSION,
        "abmil_seeds": list(seeds) if head == "abmil" else [],
        "tasks": task_records,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    n_ok = sum(1 for t in task_records if t.get("status") == "ok")
    n_error = sum(1 for t in task_records if t.get("status") == "error")
    n_attempted = n_error + (n_ok - n_resumed)
    logger.info("encoder=%s space=%s head=%s: %d/%d tasks ok (%d resumed, %d run this pass) -> %s",
                tag, space, head, n_ok, len(task_records), n_resumed, n_ok - n_resumed, out / "metrics.json")

    # Stop if every task attempted in this pass failed
    if n_attempted > 0 and n_error == n_attempted:
        raise SystemExit(
            f"encoder={tag} space={space} head={head}: all {n_error} task(s) attempted this pass failed "
            f"({n_resumed} other(s) resumed from cache). Per-task errors are in {out / 'metrics.json'}; "
            "the first is:\n  "
            + str(next(t.get("error") for t in task_records if t.get("status") == "error"))
        )
    return {
        "n_tasks": len(task_records),
        "n_ok": n_ok,
        "n_error": n_error,
        "n_skipped_no_features": sum(1 for t in task_records if t.get("status") == "skipped_no_features"),
        "n_resumed": n_resumed,
    }


# --------------------------------------------------------------------------- same slides for all encoders


def assert_encoder_slide_sets_consistent(
    cohorts: Sequence[str],
    encoders: Sequence[Tuple[int, int]],
    spaces: Sequence[str],
    *,
    allow_mismatch: bool = False,
) -> None:
    """Check that all encoders were extracted on the same slides, per (cohort, space). Raises (or warns if allow_mismatch)."""
    problems: List[str] = []
    for cohort in cohorts:
        for space in spaces:
            present: List[Tuple[str, set]] = []
            for tn, cn in encoders:
                if space == "clip" and not expects_clip(tn, cn):
                    continue  # he-only encoder (no clip space)
                fdir = trident_features_root(cohort, tn, cn) / space
                present.append((encoder_tag(tn, cn), {p.stem for p in list_h5(fdir)}))
            nonempty = [(tag, s) for tag, s in present if s]
            if len(nonempty) < 2:
                continue  # fewer than 2 encoders with features
            ref_tag, ref = nonempty[0]
            for tag, s in nonempty[1:]:
                if s != ref:
                    problems.append(
                        f"cohort={cohort} space={space}: {ref_tag} has {len(ref)} slides, {tag} has {len(s)} "
                        f"(only in {ref_tag}: {sorted(ref - s)[:5]}; only in {tag}: {sorted(s - ref)[:5]})"
                    )
    if problems:
        msg = ("Encoder feature slide sets differ across encoders:"
               "\n  " + "\n  ".join(problems)
               + "\nRe-extract the missing slides (_1) or pass --allow_encoder_slide_mismatch to proceed anyway.")
        if allow_mismatch:
            logger.warning("%s", msg)
        else:
            raise SystemExit(msg)


# --------------------------------------------------------------------------- eval_id protocol


def write_or_validate_protocol(eval_id: str, seeds: Sequence[int]) -> None:
    """Write the eval_id protocol to <eval_id>/manifest.json, or check that it matches the existing one."""
    manifest = build_taskd_manifest(eval_id, abmil_seeds=seeds, notes="Task D eval_id protocol.")
    path = eval_root(eval_id) / "manifest.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        found_schema = str(existing.get("schema_version", "?"))
        if found_schema != TASKD_SCHEMA_VERSION:
            raise SystemExit(
                f"eval_id {eval_id!r} has a schema v{found_schema} manifest at {path}, but this code "
                f"writes v{TASKD_SCHEMA_VERSION}: use a new eval_id."
            )
        old, new = existing.get("protocol", {}) or {}, manifest["protocol"]
        drifted = sorted(k for k in set(old) | set(new) if old.get(k) != new.get(k))
        if drifted:
            raise SystemExit(
                f"eval_id {eval_id!r} has a different protocol at {path}.\n  Changed: "
                + ", ".join(drifted)
                + "\nThe protocol cannot change under an existing eval_id: use a new eval_id "
                  "(encoders, spaces and heads can be added as arms)."
            )
        logger.info("eval_id %s: protocol matches the frozen manifest (fingerprint %s).",
                    eval_id, manifest["protocol_fingerprint"])
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    logger.info("Froze eval_id protocol (fingerprint %s) -> %s", manifest["protocol_fingerprint"], path)


def record_arm(
    eval_id: str,
    training_num: int,
    checkpoint_num: int,
    space: str,
    head: str,
    *,
    tasks_run: Sequence[str],
    summary: Dict[str, int],
    pathobench_version: Optional[str],
    trident_version: Optional[str],
) -> None:
    """Create or update the record of one (encoder, space, head) run in <eval_id>/arms/<tag>_<space>_<head>.json."""
    path = arm_record_path(eval_id, training_num, checkpoint_num, space, head)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    previous: Dict[str, object] = {}
    if path.exists():
        try:
            previous = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            logger.warning("Arm record %s is unreadable; rewriting it.", path)
    for key, value in (("pathobench_version", pathobench_version), ("trident_version", trident_version)):
        was = previous.get(key)
        if was is not None and value is not None and str(was) != str(value):
            logger.warning("Arm %s: %s changed %s -> %s since its last run — results in this arm now "
                           "span two toolkit versions.", path.stem, key, was, value)
    record = {
        "eval_id": str(eval_id),
        "tag": encoder_tag(training_num, checkpoint_num),
        "training_num": int(training_num),
        "checkpoint_num": int(checkpoint_num),
        "space": str(space),
        "head": str(head),
        "tasks_run": sorted(str(t) for t in tasks_run),
        "n_tasks_run": len(list(tasks_run)),
        "is_full_registry": sorted(str(t) for t in tasks_run) == sorted(t.task_id for t in TASKD_REGISTRY),
        "pathobench_version": pathobench_version,
        "trident_version": trident_version,
        "first_run": previous.get("first_run", now),
        "last_run": now,
        "n_runs": int(previous.get("n_runs", 0)) + 1,
        **{k: int(v) for k, v in summary.items()},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


# --------------------------------------------------------------------------- main


def _device_to_gpu(device: Optional[str], gpu: Optional[int]) -> int:
    """PathoBench GPU id: --gpu if given, else parsed from the device string (cuda:N -> N, otherwise -1 = auto)."""
    if gpu is not None:
        return int(gpu)
    d = (device or "").strip().lower()
    if d.startswith("cuda:"):
        try:
            return int(d.split(":", 1)[1])
        except ValueError:
            return -1
    return -1


def assert_registry_is_declared_subset(*, allow_missing_catalog: bool = True) -> None:
    """Check that every PathoBench task on a Task D cohort is in TASKD_REGISTRY or DECLARED_EXCLUDED_TASKS."""
    catalog = read_available_splits()
    if catalog is None:
        msg = (f"No PathoBench catalog at {available_splits_path()}: registry completeness is not checked "
               "(the catalog is written by `SplitFactory.from_hf` with the splits cache).")
        if allow_missing_catalog:
            logger.warning("%s", msg)
            return
        raise SystemExit(msg)

    audit = audit_registry_against_catalog(catalog)
    logger.info("Catalog check: %d task(s) across %d dataset(s); %d on Task D cohorts -> %d run, "
                "%d declared-excluded (%s).", audit.n_tasks, audit.n_datasets,
                audit.n_tasks_on_registry_cohorts, len(audit.registered), len(audit.declared_excluded),
                ", ".join(audit.declared_excluded) or "none")
    if audit.ok:
        return
    problems: List[str] = []
    if audit.undeclared:
        problems.append(
            f"{len(audit.undeclared)} PathoBench task(s) on cohorts Task D already holds features for "
            f"are neither run nor declared excluded: {', '.join(audit.undeclared)} — add them to "
            "TASKD_REGISTRY or record them in DECLARED_EXCLUDED_TASKS with a reason"
        )
    if audit.not_in_catalog:
        problems.append(
            f"{len(audit.not_in_catalog)} registry id(s) are absent from the catalog (renamed or "
            f"withdrawn upstream): {', '.join(audit.not_in_catalog)}"
        )
    raise SystemExit(
        "Task D registry is not an explicitly declared subset of the PathoBench catalog:\n  - "
        + "\n  - ".join(problems)
        + f"\nCatalog: {available_splits_path()}. Bump TASKD_REGISTRY_VERSION with any change."
    )


def main(args: argparse.Namespace) -> None:
    encoders = [parse_encoder_tag(t) for t in args.encoders]
    spaces = args.spaces or list(SPACES)
    heads = args.heads or list(HEADS)
    tasks = [t for t in TASKD_REGISTRY if (not args.tasks or t.task_id in set(args.tasks))]
    if not tasks:
        raise SystemExit(f"No tasks selected (registry has {[t.task_id for t in TASKD_REGISTRY]}).")
    validate_exclusion_registry()
    validate_declared_exclusions()
    assert_registry_is_declared_subset()
    write_or_validate_protocol(args.eval_id, args.abmil_seeds)
    assert_encoder_slide_sets_consistent(
        sorted({t.cohort for t in tasks}), encoders, spaces,
        allow_mismatch=args.allow_encoder_slide_mismatch,
    )
    gpu = _device_to_gpu(args.device, args.gpu)
    resume = not args.no_resume
    logger.info("Resume: %s.", "ON (fingerprinted sidecars; pass --no_resume to force a clean recompute)"
                if resume else "OFF (--no_resume) — every (task, seed) will be recomputed")

    # Feature signatures, shared across the run
    feature_sig_cache: Dict[Path, Optional[FeatureSignature]] = {}
    for tn, cn in encoders:
        for space in spaces:
            for head in heads:
                summary = run_encoder_space_head(
                    args.eval_id, tn, cn, space, head, tasks=tasks, seeds=args.abmil_seeds,
                    gpu=gpu, allow_unexpected_missing=args.allow_unexpected_missing_slides,
                    resume=resume, feature_sig_cache=feature_sig_cache,
                )
                if summary is not None:
                    # Record each arm as soon as it completes
                    record_arm(args.eval_id, tn, cn, space, head,
                               tasks_run=[t.task_id for t in tasks], summary=summary,
                               pathobench_version=args.pathobench_version,
                               trident_version=args.trident_version)

    logger.info("Done (eval_id %s).",
                args.eval_id)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Task D step 2: PathoBench slide-level probes on the Trident features.")
    parser.add_argument("--eval_id", type=str, required=True, help="Task D eval_id.")
    parser.add_argument("--encoders", type=str, nargs="+", required=True, help="Encoder tags, e.g. t46e13 t9001e0 t-1e-1.")
    parser.add_argument("--spaces", type=str, nargs="*", default=None,
                        help="Spaces to probe: he, clip (default: both).")
    parser.add_argument("--heads", type=str, nargs="*", default=None,
                        help="Probe heads: linprobe, abmil (default: both).")
    parser.add_argument("--tasks", type=str, nargs="*", default=None,
                        help="Subset of task ids (default: all registry tasks).")
    parser.add_argument("--abmil_seeds", type=int, nargs="*", default=list(DEFAULT_ABMIL_SEEDS),
                        help="Seeds of the ABMIL head (part of the eval_id protocol).")
    parser.add_argument("--no_resume", action="store_true",
                        help="Recompute every (task, seed) instead of reusing the resume sidecars.")
    parser.add_argument("--allow_encoder_slide_mismatch", action="store_true",
                        help="Warn instead of failing when encoders were extracted on different slides.")
    parser.add_argument("--allow_unexpected_missing_slides", action="store_true",
                        help="Warn instead of failing on missing split slides that are not in KNOWN_EXCLUDED_SLIDES.")
    parser.add_argument("--gpu", type=int, default=None, help="PathoBench GPU id (-1 = auto-select best CUDA GPU). Overrides --device.")
    parser.add_argument("--device", type=str, default=None, help="Torch device string mapped to --gpu (cuda:N -> N, otherwise -1 = auto).")
    parser.add_argument("--pathobench_version", type=str, default=None, help="Installed PathoBench version, recorded in the eval_id manifest.")
    parser.add_argument("--trident_version", type=str, default=None, help="Trident version used for the features, recorded in the eval_id manifest.")
    main(parser.parse_args())
