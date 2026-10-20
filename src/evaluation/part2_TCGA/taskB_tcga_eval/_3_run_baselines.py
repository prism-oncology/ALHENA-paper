#!/usr/bin/env python3
"""
Run the TCGA Task B baselines (mean_predictor, tile_count, tissue_area, batch_stain_diagnostic, random_features,
target_permutation) with the same fold-safe ridge protocol as _2_fit_ridge_probe.py.

Saved in <ROOT_HistoST> / eval / eval_datasets / TCGA_eval_data / benchmarks / <benchmark_id> / shared / baseline_features / <baseline_id> /
    - case_features__pool_<pooling>.parquet, feature_manifest__pool_<pooling>.json: cached baseline features
Saved in <ROOT_HistoST> / eval / B_eval_case / <eval_id> / baselines /
    - <baseline_id>/: predictions/, metrics/, diagnostics/, run_manifest.json
    - baselines_run_inventory.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

from _eval_common import (
    FeatureBundle,
    assert_eval_config_matches_protocol_manifest,
    benchmark_baseline_features_root,
    benchmark_encoder_features_root,
    collect_output_hashes,
    encoder_features_run_id,
    eval_case_root,
    load_encoder_feature_bundle,
    load_eval_settings,
    make_feature_manifest,
    manifest_header,
    resolve_benchmark_paths,
    sha256_path,
    write_csv_atomic,
    write_json_atomic,
    write_parquet_atomic,
)
from _feature_builders import (
    build_batch_stain_features,
    build_random_feature_bundle,
    build_tile_count_features,
    build_tissue_area_features,
)
from _ridge_protocol import (
    DEFAULT_POOLING,
    RidgeProtocolConfig,
    evaluate_mean_predictor,
    evaluate_ridge_feature_bundle,
)

BASELINE_CHOICES = [
    "mean_predictor",
    "tile_count",
    "tissue_area",
    "batch_stain_diagnostic",
    "random_features",
    "target_permutation",
    "all",
]


def _ridge_config_from_settings(settings) -> RidgeProtocolConfig:
    protocol = settings.eval_config.get("protocol")
    if not isinstance(protocol, dict):
        raise RuntimeError(
            f"eval_config.protocol missing or malformed in {settings.eval_config_path}"
        )
    cfg = RidgeProtocolConfig(
        alpha_grid=tuple(float(a) for a in protocol["alpha_grid"]),
        inner_folds=int(protocol["inner_folds"]),
        inner_seed=int(protocol["inner_seed"]),
        bootstrap_reps=int(protocol["bootstrap_reps"]),
        bootstrap_seed=int(protocol["bootstrap_seed"]),
        pooling=str(protocol.get("pooling", DEFAULT_POOLING)),
    )
    expected_hash = protocol.get("protocol_hash")
    if expected_hash and cfg.sha256 != expected_hash:
        raise RuntimeError(
            "RidgeProtocolConfig rebuilt from eval_config does not match the "
            f"protocol_hash ({cfg.sha256} vs {expected_hash}); re-run _1_setup.py."
        )
    return cfg


def _baseline_cache_filenames(pooling: str) -> Tuple[str, str]:
    """Per-pooling cache file names (parquet, manifest)."""
    return (f"case_features__pool_{pooling}.parquet",
            f"feature_manifest__pool_{pooling}.json")


def _cache_baseline_features(
    *,
    benchmark_id: str,
    baseline_id: str,
    bundle: FeatureBundle,
    project_root: Optional[Path],
    overwrite: bool,
    pooling: str,
    extra_manifest: Optional[Dict[str, object]] = None,
) -> FeatureBundle:
    """Write baseline features to the shared benchmark cache.

    Returns an updated FeatureBundle whose metadata points at the cache path.
    """
    cache_dir = benchmark_baseline_features_root(benchmark_id, baseline_id, project_root=project_root)
    cache_dir.mkdir(parents=True, exist_ok=True)
    parquet_name, manifest_name = _baseline_cache_filenames(pooling)
    parquet_path = cache_dir / parquet_name
    manifest_path = cache_dir / manifest_name
    write_parquet_atomic(parquet_path, bundle.features, overwrite=overwrite)
    feat_hash = sha256_path(parquet_path)
    projects = (
        sorted(bundle.features["project_id"].astype(str).unique())
        if "project_id" in bundle.features.columns
        else []
    )
    bundle = FeatureBundle(
        features=bundle.features,
        feature_columns=bundle.feature_columns,
        numeric_columns=bundle.numeric_columns,
        categorical_columns=bundle.categorical_columns,
        metadata={
            **bundle.metadata,
            "baseline_id": baseline_id,
            "benchmark_id": benchmark_id,
            "pooling": str(pooling),
            "feature_path": str(parquet_path),
            "feature_table_hash": feat_hash,
            "cache_dir": str(cache_dir),
        },
        feature_hash_by_project={proj: feat_hash for proj in projects},
        feature_table_hash=feat_hash,
    )
    manifest = make_feature_manifest(bundle, parquet_path, extra=extra_manifest)
    write_json_atomic(manifest_path, manifest, overwrite=overwrite)
    return bundle


def _load_cached_baseline_features(
    *,
    benchmark_id: str,
    baseline_id: str,
    project_root: Optional[Path],
    pooling: str,
) -> Optional[FeatureBundle]:
    cache_dir = benchmark_baseline_features_root(benchmark_id, baseline_id, project_root=project_root)
    parquet_name, manifest_name = _baseline_cache_filenames(pooling)
    parquet_path = cache_dir / parquet_name
    manifest_path = cache_dir / manifest_name
    if not parquet_path.exists() or not manifest_path.exists():
        return None
    df = pd.read_parquet(parquet_path)
    feat_cols = [c for c in df.columns if c not in {"case_id", "case_uuid", "project_id"}]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    feat_hash = sha256_path(parquet_path)
    projects = sorted(df["project_id"].astype(str).unique()) if "project_id" in df.columns else []
    return FeatureBundle(
        features=df,
        feature_columns=feat_cols,
        numeric_columns=manifest.get("numeric_columns") or feat_cols,
        categorical_columns=manifest.get("categorical_columns") or [],
        metadata={
            **(manifest.get("metadata") or {}),
            "baseline_id": baseline_id,
            "benchmark_id": benchmark_id,
            "pooling": str(pooling),
            "feature_path": str(parquet_path),
            "feature_table_hash": feat_hash,
            "cache_dir": str(cache_dir),
            "loaded_from_cache": True,
        },
        feature_hash_by_project={proj: feat_hash for proj in projects},
        feature_table_hash=feat_hash,
    )


def _get_or_build_baseline_features(
    *,
    benchmark_id: str,
    baseline_id: str,
    builder,
    project_root: Optional[Path],
    rebuild: bool,
    pooling: str,
    extra_manifest: Optional[Dict[str, object]] = None,
):
    """
    Load the cached baseline features, or build and cache them.
    Returns (FeatureBundle, builder_extras): extra outputs of a builder returning a tuple, else None.
    """
    if not rebuild:
        cached = _load_cached_baseline_features(
            benchmark_id=benchmark_id, baseline_id=baseline_id,
            project_root=project_root, pooling=pooling,
        )
        if cached is not None:
            return cached, None
    out = builder()
    if isinstance(out, tuple):
        bundle, *extras = out
        extras = tuple(extras)
    else:
        bundle, extras = out, None
    bundle = _cache_baseline_features(
        benchmark_id=benchmark_id, baseline_id=baseline_id, bundle=bundle,
        project_root=project_root, overwrite=True, pooling=pooling,
        extra_manifest=extra_manifest,
    )
    return bundle, extras


def _run_feature_baseline(
    *,
    baseline_id: str,
    baseline_output_root: Path,
    paths,
    bundle: FeatureBundle,
    cfg: RidgeProtocolConfig,
    eval_id: str,
    settings,
    overwrite: bool,
    allow_unvalidated: bool,
) -> Dict[str, object]:
    if baseline_output_root.exists() and not overwrite and any(p.is_file() and not p.name.startswith(".") for p in baseline_output_root.rglob("*")):
        raise FileExistsError(f"Baseline output exists; use --force: {baseline_output_root}")
    baseline_output_root.mkdir(parents=True, exist_ok=True)
    run_info = {
        "eval_id": eval_id,
        "baseline_id": baseline_id,
        "encoder_id": baseline_id,
        "training_num": "",
        "checkpoint_num": "",
        "space": f"baseline:{baseline_id}",
        "modelgroup_id": paths.modelgroup_id,
        "tile_feature_postproc": "baseline_feature_table",
    }
    result = evaluate_ridge_feature_bundle(
        output_root=baseline_output_root,
        paths=paths,
        feature_bundle=bundle,
        run_info=run_info,
        cfg=cfg,
        overwrite=overwrite,
        allow_unvalidated=allow_unvalidated,
        progress_desc=f"baseline {baseline_id}",
    )
    manifest = manifest_header(Path(__file__).name, {"baseline_id": baseline_id, "eval_id": eval_id})
    manifest.update({
        "frozen_final": True,
        "eval_id": eval_id,
        "baseline_id": baseline_id,
        "benchmark_id": paths.config.get("benchmark_id", ""),
        "modelgroup_id": paths.modelgroup_id,
        "baseline_feature_path": bundle.metadata.get("feature_path"),
        "baseline_feature_table_hash": bundle.feature_table_hash,
        "baseline_feature_cache_dir": bundle.metadata.get("cache_dir"),
        "eval_config_path": str(settings.eval_config_path),
        "eval_config_hash": settings.eval_config_hash,
        "protocol_manifest_path": str(settings.protocol_manifest_path),
        "protocol_manifest_hash": settings.protocol_manifest_hash,
        "ridge_protocol": cfg.as_dict(),
        "ridge_protocol_hash": cfg.sha256,
        "result": {k: v for k, v in result.items() if k != "output_hashes"},
        "output_hashes": collect_output_hashes(baseline_output_root),
    })
    write_json_atomic(baseline_output_root / "run_manifest.json", manifest, overwrite=overwrite)
    return result


def _run_mean_baseline(
    *,
    baseline_output_root: Path,
    paths,
    cfg: RidgeProtocolConfig,
    eval_id: str,
    settings,
    overwrite: bool,
    allow_unvalidated: bool,
) -> Dict[str, object]:
    if baseline_output_root.exists() and not overwrite and any(p.is_file() and not p.name.startswith(".") for p in baseline_output_root.rglob("*")):
        raise FileExistsError(f"Baseline output exists; use --force: {baseline_output_root}")
    baseline_output_root.mkdir(parents=True, exist_ok=True)
    run_info = {
        "eval_id": eval_id,
        "baseline_id": "mean_predictor",
        "encoder_id": "mean_predictor",
        "training_num": "",
        "checkpoint_num": "",
        "space": "baseline:mean_predictor",
        "modelgroup_id": paths.modelgroup_id,
        "tile_feature_postproc": "none",
    }
    result = evaluate_mean_predictor(
        output_root=baseline_output_root,
        paths=paths,
        run_info=run_info,
        cfg=cfg,
        overwrite=overwrite,
        allow_unvalidated=allow_unvalidated,
        progress_desc="baseline mean_predictor",
    )
    manifest = manifest_header(Path(__file__).name, {"baseline_id": "mean_predictor", "eval_id": eval_id})
    manifest.update({
        "frozen_final": True,
        "eval_id": eval_id,
        "baseline_id": "mean_predictor",
        "benchmark_id": paths.config.get("benchmark_id", ""),
        "modelgroup_id": paths.modelgroup_id,
        "protocol": "outer_train_mean_on_frozen_target_scale",
        "correlation_metrics_forced_na": False,
        "eval_config_path": str(settings.eval_config_path),
        "eval_config_hash": settings.eval_config_hash,
        "protocol_manifest_path": str(settings.protocol_manifest_path),
        "protocol_manifest_hash": settings.protocol_manifest_hash,
        "ridge_protocol_reference_hash": cfg.sha256,
        "result": {k: v for k, v in result.items() if k != "output_hashes"},
        "output_hashes": collect_output_hashes(baseline_output_root),
    })
    write_json_atomic(baseline_output_root / "run_manifest.json", manifest, overwrite=overwrite)
    return result


def _reference_encoder_run_id(args: argparse.Namespace) -> Optional[str]:
    if args.reference_run_id:
        return str(args.reference_run_id)
    if args.reference_training_num is not None and args.reference_checkpoint_num is not None:
        # Feature runs hold all spaces (--reference_space selects the datasets at load time)
        return encoder_features_run_id(args.reference_training_num, args.reference_checkpoint_num)
    return None


def _resolve_reference_feature_bundle(args: argparse.Namespace, *, pooling: str) -> Optional[FeatureBundle]:
    rid = _reference_encoder_run_id(args)
    if rid is None:
        return None
    root = benchmark_encoder_features_root(args.benchmark_id, rid, project_root=args.project_root)
    if not root.exists():
        raise FileNotFoundError(
            f"Reference encoder run root not found at the shared encoder features path: {root}"
        )
    # Space read from the reference feature run (default: he)
    return load_encoder_feature_bundle(root, pooling=pooling, space=(args.reference_space or "he"))


def main(args) -> int:

    requested = set(args.baselines)
    if "all" in requested:
        requested = set(BASELINE_CHOICES) - {"all"}
    if args.target_permutation_repeats < 1:
        raise RuntimeError("--target_permutation_repeats must be >= 1")

    settings = load_eval_settings(args.eval_id, project_root=args.project_root)
    if settings.benchmark_id != str(args.benchmark_id):
        raise RuntimeError(
            f"eval_config.identity.benchmark_id={settings.benchmark_id} disagrees with "
            f"--benchmark_id {args.benchmark_id}; the eval_id was set up against a different benchmark."
        )
    assert_eval_config_matches_protocol_manifest(settings)

    paths = resolve_benchmark_paths(
        args.benchmark_id, project_root=args.project_root, modelgroup_id=args.modelgroup_id
    )
    cfg = _ridge_config_from_settings(settings)
    baselines_output_root = eval_case_root(args.eval_id, project_root=args.project_root) / "baselines"
    baselines_output_root.mkdir(parents=True, exist_ok=True)

    reference_bundle: Optional[FeatureBundle] = None
    if {"random_features", "target_permutation"} & requested:
        reference_bundle = _resolve_reference_feature_bundle(args, pooling=cfg.pooling)
        if reference_bundle is None and "random_features" in requested and args.random_feature_dim is None:
            raise RuntimeError(
                "random_features requires --random_feature_dim or a reference encoder run "
                "(--reference_run_id or --reference_training_num/--reference_checkpoint_num/--reference_space)"
            )
        if reference_bundle is None and "target_permutation" in requested:
            raise RuntimeError(
                "target_permutation requires --reference_run_id or "
                "--reference_training_num/--reference_checkpoint_num/--reference_space"
            )

    results: Dict[str, object] = {}

    if "mean_predictor" in requested:
        results["mean_predictor"] = _run_mean_baseline(
            baseline_output_root=baselines_output_root / "mean_predictor",
            paths=paths,
            cfg=cfg,
            eval_id=str(args.eval_id),
            settings=settings,
            overwrite=args.force,
            allow_unvalidated=bool(args.allow_unvalidated_benchmark),
        )

    feature_baselines = [
        ("tile_count", lambda: build_tile_count_features(paths)),
        ("tissue_area", lambda: build_tissue_area_features(paths)),
        ("batch_stain_diagnostic", lambda: build_batch_stain_features(paths)),
    ]
    for baseline_id, builder in feature_baselines:
        if baseline_id not in requested:
            continue
        bundle, _ = _get_or_build_baseline_features(
            benchmark_id=args.benchmark_id,
            baseline_id=baseline_id,
            builder=builder,
            project_root=args.project_root,
            rebuild=bool(args.rebuild_baseline_features),
            pooling=cfg.pooling,
        )
        results[baseline_id] = _run_feature_baseline(
            baseline_id=baseline_id,
            baseline_output_root=baselines_output_root / baseline_id,
            paths=paths,
            bundle=bundle,
            cfg=cfg,
            eval_id=str(args.eval_id),
            settings=settings,
            overwrite=args.force,
            allow_unvalidated=bool(args.allow_unvalidated_benchmark),
        )

    if "random_features" in requested:
        dim = args.random_feature_dim
        if dim is None:
            assert reference_bundle is not None
            dim = int(reference_bundle.metadata.get("feature_dim_case") or len(reference_bundle.feature_columns))
            print(json.dumps({
                "warning": "random_features dim auto-matched to reference encoder",
                "reference_encoder_run_id": _reference_encoder_run_id(args),
                "random_feature_dim": int(dim),
            }, indent=2))
        random_baseline_id = f"random_features_d{int(dim)}_s{int(args.random_seed)}"
        bundle, _ = _get_or_build_baseline_features(
            benchmark_id=args.benchmark_id,
            baseline_id=random_baseline_id,
            builder=lambda: build_random_feature_bundle(paths, feature_dim=int(dim), seed=int(args.random_seed)),
            project_root=args.project_root,
            rebuild=bool(args.rebuild_baseline_features),
            pooling=cfg.pooling,
            extra_manifest={
                "random_seed": int(args.random_seed),
                "random_feature_dimension": int(dim),
                "feature_dimension_matched_to_reference": bool(args.random_feature_dim is None),
                "reference_encoder_run_id": _reference_encoder_run_id(args),
                "pooling": str(cfg.pooling),
            },
        )
        results["random_features"] = _run_feature_baseline(
            baseline_id="random_features",
            baseline_output_root=baselines_output_root / "random_features",
            paths=paths,
            bundle=bundle,
            cfg=cfg,
            eval_id=str(args.eval_id),
            settings=settings,
            overwrite=args.force,
            allow_unvalidated=bool(args.allow_unvalidated_benchmark),
        )

    if "target_permutation" in requested:
        assert reference_bundle is not None
        perm_root = baselines_output_root / "target_permutation"
        if perm_root.exists() and not args.force and any(p.is_file() and not p.name.startswith(".") for p in perm_root.rglob("*")):
            raise FileExistsError(f"Baseline output exists; use --force: {perm_root}")
        perm_root.mkdir(parents=True, exist_ok=True)
        per_repeat_rows: List[Dict[str, object]] = []
        for rep in tqdm(
            range(int(args.target_permutation_repeats)),
            desc="target_permutation", unit="repeat",
        ):
            rep_seed = int(args.target_permutation_seed) + rep
            rep_root = perm_root / f"repeat_{rep:03d}"
            run_info = {
                "eval_id": str(args.eval_id),
                "baseline_id": "target_permutation",
                "encoder_id": "target_permutation",
                "training_num": reference_bundle.metadata.get("training_num", ""),
                "checkpoint_num": reference_bundle.metadata.get("checkpoint_num", ""),
                "space": "baseline:target_permutation",
                "modelgroup_id": paths.modelgroup_id,
                "tile_feature_postproc": reference_bundle.metadata.get("tile_feature_postproc", ""),
            }
            result = evaluate_ridge_feature_bundle(
                output_root=rep_root,
                paths=paths,
                feature_bundle=reference_bundle,
                run_info=run_info,
                cfg=cfg,
                overwrite=args.force,
                allow_unvalidated=bool(args.allow_unvalidated_benchmark),
                target_permutation_seed=rep_seed,
                repeat_id=rep,
                progress_desc=f"target_permutation rep {rep + 1}/{int(args.target_permutation_repeats)}",
            )
            per_repeat_rows.append({
                "repeat_id": rep, "target_permutation_seed": rep_seed,
                **{k: v for k, v in result.items() if k not in {"output_hashes"}},
            })
        write_csv_atomic(perm_root / "per_repeat_summary.csv", pd.DataFrame(per_repeat_rows), overwrite=args.force)
        manifest = manifest_header(Path(__file__).name, vars(args))
        manifest.update({
            "frozen_final": True,
            "eval_id": str(args.eval_id),
            "baseline_id": "target_permutation",
            "benchmark_id": args.benchmark_id,
            "n_repeats": int(args.target_permutation_repeats),
            "base_seed": int(args.target_permutation_seed),
            "reference_feature_metadata": reference_bundle.metadata,
            "reference_feature_hash_by_project": reference_bundle.feature_hash_by_project,
            "eval_config_path": str(settings.eval_config_path),
            "eval_config_hash": settings.eval_config_hash,
            "protocol_manifest_path": str(settings.protocol_manifest_path),
            "protocol_manifest_hash": settings.protocol_manifest_hash,
            "ridge_protocol": cfg.as_dict(),
            "ridge_protocol_hash": cfg.sha256,
            "output_hashes": collect_output_hashes(perm_root),
        })
        write_json_atomic(perm_root / "run_manifest.json", manifest, overwrite=args.force)
        results["target_permutation"] = {
            "status": "ok",
            "output_root": str(perm_root),
            "n_repeats": int(args.target_permutation_repeats),
        }

    inventory = manifest_header(Path(__file__).name, vars(args))
    inventory.update({
        "frozen_final": True,
        "eval_id": args.eval_id,
        "benchmark_id": args.benchmark_id,
        "modelgroup_id": paths.modelgroup_id,
        "requested_baselines": sorted(requested),
        "eval_config_path": str(settings.eval_config_path),
        "eval_config_hash": settings.eval_config_hash,
        "protocol_manifest_path": str(settings.protocol_manifest_path),
        "protocol_manifest_hash": settings.protocol_manifest_hash,
        "ridge_protocol": cfg.as_dict(),
        "ridge_protocol_hash": cfg.sha256,
        "results": results,
        "output_hashes": collect_output_hashes(baselines_output_root),
    })
    write_json_atomic(baselines_output_root / "baselines_run_inventory.json", inventory, overwrite=True)
    print(json.dumps({
        "status": "ok",
        "baselines_root": str(baselines_output_root),
        "baselines": sorted(results.keys()),
        "ridge_protocol_hash": cfg.sha256,
    }, indent=2))
    return 0


if __name__ == "__main__":

    ap = argparse.ArgumentParser(description="Run the TCGA Task B baselines with the fold-safe ridge protocol.")

    ap.add_argument("--eval_id", type=str,
                    required=True,
                    help="Evaluation ID set up by _1_setup.py (protocol and outputs under "
                         "eval/B_eval_case/<eval_id>/).")

    ap.add_argument("--benchmark_id", type=str,
                    required=True,
                    help="Validated benchmark ID (targets, folds); baseline features are cached under "
                         "benchmarks/<benchmark_id>/shared/baseline_features/.")

    ap.add_argument("--modelgroup_id", type=str,
                    default=None,
                    help="Model group ID. Defaults to the model group referenced by the benchmark.")

    ap.add_argument("--project_root", type=Path,
                    default=None,
                    help="Optional override of the TCGA Task B data root.")

    ap.add_argument("--baselines", nargs="+",
                    default=["all"], choices=BASELINE_CHOICES,
                    help="Baselines to run; 'all' runs every implemented baseline.")

    ap.add_argument("--reference_run_id", type=str,
                    default=None,
                    help="Encoder feature run ID, e.g. HEencoder_t46e13 (space chosen with --reference_space). "
                         "Needed for the random_features dimension and target_permutation.")

    ap.add_argument("--reference_training_num", type=int,
                    default=None,
                    help="Training number of the reference encoder run (alternative to --reference_run_id).")

    ap.add_argument("--reference_checkpoint_num", type=int,
                    default=None,
                    help="Checkpoint number of the reference encoder run (alternative to --reference_run_id).")

    ap.add_argument("--reference_space", type=str, choices=["he", "clip"],
                    default=None,
                    help="Embedding space read from the reference encoder feature run "
                         "(random_features / target_permutation). Default: he.")

    ap.add_argument("--random_feature_dim", type=int,
                    default=None,
                    help="Overrides the reference encoder feature_dim_case for the random_features baseline.")

    ap.add_argument("--random_seed", type=int,
                    default=424242,
                    help="Random seed for the random_features baseline.")

    ap.add_argument("--target_permutation_repeats", type=int,
                    default=3,
                    help="Number of label permutations for the target_permutation baseline.")

    ap.add_argument("--target_permutation_seed", type=int,
                    default=515151,
                    help="Random seed for the target_permutation baseline.")

    ap.add_argument("--rebuild_baseline_features", action="store_true",
                    help="Rebuild the cached baseline features under "
                         "benchmarks/<benchmark_id>/shared/baseline_features/.")

    ap.add_argument("--allow_unvalidated_benchmark", action="store_true",
                    help="Allow running against a benchmark that has not passed _14_validate_benchmark.py.")

    ap.add_argument("--force", action="store_true",
                    help="Overwrite existing baseline probe outputs under the eval_id.")

    args = ap.parse_args()
    raise SystemExit(main(args))
