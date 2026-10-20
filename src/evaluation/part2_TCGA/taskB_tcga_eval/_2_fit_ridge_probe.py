#!/usr/bin/env python3
"""
Fit the fold-safe ridge probes of TCGA Task B on the case-level encoder features (one model per eligible
project x signature), with the protocol written by _1_setup.py.

Saved in <ROOT_HistoST> / eval / B_eval_case / <eval_id> / HEencoder_t<training_num>e<checkpoint_num>_<space> /
    - predictions/<project_id>__<signature>.parquet
    - metrics/cell_metrics.csv, project_macro.csv, signature_macro.csv, global_summary.json, alpha_selected.csv
    - diagnostics/model_fit_summary.json
    - ridge_probe_manifest.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from _eval_common import (
    assert_eval_config_matches_protocol_manifest,
    benchmark_encoder_features_root,
    collect_output_hashes,
    encoder_features_run_id,
    encoder_run_id,
    eval_case_root,
    load_encoder_feature_bundle,
    load_eval_settings,
    manifest_header,
    resolve_benchmark_paths,
    sha256_path,
    write_json_atomic,
)
from _ridge_protocol import DEFAULT_POOLING, RidgeProtocolConfig, evaluate_ridge_feature_bundle


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


def main(args) -> int:

    if args.space == "clip" and int(args.checkpoint_num) < 0:
        raise RuntimeError(
            "Invalid run: checkpoint_num < 0 and space == 'clip' because checkpoint -1 exports no CLIP image projection"
        )
    if args.space == "clip" and args.he_l2_normalize_tile_embeddings:
        raise RuntimeError("--he_l2_normalize_tile_embeddings applies only to space=he")

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

    # run_id: output label of this per-space probe; features_run_id: shared feature run (all spaces)
    run_id = encoder_run_id(
        args.training_num, args.checkpoint_num, args.space,
        he_l2_normalize_tile_embeddings=bool(args.he_l2_normalize_tile_embeddings),
    )
    features_run_id = encoder_features_run_id(
        args.training_num, args.checkpoint_num,
        he_l2_normalize_tile_embeddings=bool(args.he_l2_normalize_tile_embeddings),
    )
    features_root = benchmark_encoder_features_root(
        args.benchmark_id, features_run_id, project_root=args.project_root,
    )
    if not features_root.exists():
        raise FileNotFoundError(
            "Missing encoder feature run root. Run _0_extract_encoder_features.py first: "
            f"{features_root}"
        )
    feature_bundle = load_encoder_feature_bundle(features_root, pooling=cfg.pooling, space=args.space)

    # Consistency checks against the feature extraction manifest
    fm = feature_bundle.metadata
    if str(fm.get("benchmark_id")) != str(args.benchmark_id):
        raise RuntimeError(
            f"Feature run benchmark_id={fm.get('benchmark_id')} disagrees with requested {args.benchmark_id}"
        )
    if str(fm.get("modelgroup_id")) != str(paths.modelgroup_id):
        raise RuntimeError(
            f"Feature run modelgroup_id={fm.get('modelgroup_id')} disagrees with benchmark {paths.modelgroup_id}"
        )
    # The requested space was checked by load_encoder_feature_bundle
    if (
        int(fm.get("training_num")) != int(args.training_num)
        or int(fm.get("checkpoint_num")) != int(args.checkpoint_num)
    ):
        raise RuntimeError("Feature run identity disagrees with requested training/checkpoint")

    output_root = eval_case_root(args.eval_id, project_root=args.project_root) / run_id

    run_info = {
        "eval_id": args.eval_id,
        "encoder_id": run_id,
        "training_num": int(args.training_num),
        "checkpoint_num": int(args.checkpoint_num),
        "space": args.space,
        "modelgroup_id": paths.modelgroup_id,
        "tile_feature_postproc": str(fm.get("tile_feature_postproc", "")),
        "pooling": str(cfg.pooling),
        "feature_dim_case": fm.get("feature_dim_case"),
        "feature_dim_tile": fm.get("feature_dim_tile"),
    }
    result = evaluate_ridge_feature_bundle(
        output_root=output_root,
        paths=paths,
        feature_bundle=feature_bundle,
        run_info=run_info,
        cfg=cfg,
        overwrite=bool(args.force),
        allow_unvalidated=bool(args.allow_unvalidated_benchmark),
        progress_desc=f"ridge {run_id}",
    )

    manifest = manifest_header(Path(__file__).name, vars(args))
    manifest.update({
        "frozen_final": True,
        "eval_id": args.eval_id,
        "encoder_id": run_id,
        "benchmark_id": args.benchmark_id,
        "modelgroup_id": paths.modelgroup_id,
        "features_root": str(features_root),
        "output_root": str(output_root),
        "feature_run_manifest_hash": fm.get("run_manifest_hash"),
        "case_features_index_hash": fm.get("case_features_index_hash"),
        "eval_config_path": str(settings.eval_config_path),
        "eval_config_hash": settings.eval_config_hash,
        "protocol_manifest_path": str(settings.protocol_manifest_path),
        "protocol_manifest_hash": settings.protocol_manifest_hash,
        "ridge_protocol": cfg.as_dict(),
        "ridge_protocol_hash": cfg.sha256,
        "result": {k: v for k, v in result.items() if k != "output_hashes"},
        "output_hashes": collect_output_hashes(output_root),
    })
    write_json_atomic(output_root / "ridge_probe_manifest.json", manifest, overwrite=args.force)
    print(json.dumps({
        "status": "ok",
        "output_root": str(output_root),
        "encoder_id": run_id,
        "primary_macro_spearman": result.get("primary_macro_spearman"),
        "ridge_protocol_hash": cfg.sha256,
    }, indent=2))
    return 0


if __name__ == "__main__":

    ap = argparse.ArgumentParser(
        description="Fit fold-safe ridge probes on TCGA Task B encoder case features."
    )

    ap.add_argument("--eval_id", type=str,
                    required=True,
                    help="Evaluation ID set up by _1_setup.py (protocol and outputs under "
                         "eval/B_eval_case/<eval_id>/).")

    ap.add_argument("--benchmark_id", type=str,
                    required=True,
                    help="Validated benchmark ID (targets, folds, encoder features); must match "
                         "eval_config.identity.benchmark_id.")

    ap.add_argument("--modelgroup_id", type=str,
                    default=None,
                    help="Model group ID. Defaults to the model group referenced by the benchmark.")

    ap.add_argument("--training_num", type=int,
                    required=True,
                    help="HistoST training number of the evaluated encoder.")

    ap.add_argument("--checkpoint_num", type=int,
                    required=True,
                    help="Checkpoint number of the evaluated encoder (-1 for a raw pretrained encoder).")

    ap.add_argument("--space", type=str, choices=["he", "clip"],
                    required=True,
                    help="Embedding space: 'he' (raw backbone) or 'clip' (after the CLIP projection head).")

    ap.add_argument("--he_l2_normalize_tile_embeddings", action="store_true",
                    help="Use the '_he_l2tiles' feature run (l2-normalized he tile embeddings) of "
                         "_0_extract_encoder_features.py. Only for space=he.")

    ap.add_argument("--project_root", type=Path,
                    default=None,
                    help="Optional override of the TCGA Task B data root.")

    ap.add_argument("--allow_unvalidated_benchmark", action="store_true",
                    help="Allow running against a benchmark that has not passed _14_validate_benchmark.py.")

    ap.add_argument("--force", action="store_true",
                    help="Overwrite existing ridge probe outputs.")

    args = ap.parse_args()
    raise SystemExit(main(args))
