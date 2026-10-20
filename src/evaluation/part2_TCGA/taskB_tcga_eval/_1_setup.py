#!/usr/bin/env python3
"""
Set up one TCGA Task B eval_id: write its ridge probe protocol.

Saved in <ROOT_HistoST> / eval / B_eval_case / <eval_id> / settings /
    - eval_config.yaml: identity (eval_id, benchmark_id, modelgroup_id, ...) and ridge probe protocol (with its hash)
    - protocol_manifest.json: SHA256 of the benchmark inputs read by the probe, protocol and identity payload
    - environment.json: python / platform / software versions
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add the project root to sys.path so that 'src' can be imported
project_root = Path(__file__).resolve().parents[4]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.constants import get_full_path  # noqa: F401

from _eval_common import (
    eval_case_root,
    eval_settings_dir,
    resolve_benchmark_paths,
    target_hash_map,
    validation_gate,
)
from _ridge_protocol import (
    DEFAULT_ALPHA_GRID,
    DEFAULT_POOLING,
    VALID_POOLINGS,
    RidgeProtocolConfig,
)

from _taskb_common import (
    manifest_header,
    sha256_obj,
    sha256_path,
    software_versions,
    write_json_atomic,
    write_yaml_atomic,
)


def _parse_alpha_grid(text: Optional[str]) -> tuple[float, ...]:
    if not text:
        return tuple(DEFAULT_ALPHA_GRID)
    vals: List[float] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        vals.append(float(part))
    if not vals:
        raise RuntimeError("Alpha grid cannot be empty")
    if any(v <= 0 for v in vals):
        raise RuntimeError("All alpha values must be positive")
    return tuple(vals)


def _benchmark_input_hashes(paths) -> Dict[str, Any]:
    """SHA256 of every benchmark file read by the probe (checked again at run time)."""
    expected = {
        "benchmark_config.yaml": paths.settings_dir / "benchmark_config.yaml",
        "benchmark_asset_links.json": paths.settings_dir / "benchmark_asset_links.json",
        "encoder_registry.json": paths.settings_dir / "encoder_registry.json",
        "signature_set_tcga_case_v1.json": paths.settings_dir / "signature_set_tcga_case_v1.json",
        "final_case_manifest.csv": paths.cohort_dir / "final_case_manifest.csv",
        "case_tile_index.parquet": paths.tiles_dir / "case_tile_index.parquet",
        "case_tile_diagnostics.parquet": paths.tiles_dir / "case_tile_diagnostics.parquet",
        "case_slide_contributions.parquet": paths.tiles_dir / "case_slide_contributions.parquet",
        "fold_assignments.csv": paths.folds_dir / "fold_assignments.csv",
        "signature_eligibility.csv": paths.rna_dir / "signature_eligibility.csv",
        "benchmark_validation_report.json": paths.benchmark_root / "reports" / "benchmark_validation_report.json",
    }
    hashes: Dict[str, str] = {}
    missing: List[str] = []
    for label, p in expected.items():
        if not p.exists():
            missing.append(f"{label} ({p})")
            continue
        hashes[label] = sha256_path(p)
    target_table_hashes = target_hash_map(paths)
    if not target_table_hashes:
        missing.append(f"target_tables/*.parquet ({paths.target_tables_dir})")
    if missing:
        raise FileNotFoundError("Missing benchmark inputs at setup time: " + "; ".join(missing))
    hashes["target_tables"] = target_table_hashes
    return hashes


def _identity_block(args: argparse.Namespace, paths) -> Dict[str, Any]:
    cfg = paths.config
    return {
        "eval_id": str(args.eval_id),
        "benchmark_id": str(args.benchmark_id),
        "modelgroup_id": str(paths.modelgroup_id),
        "tcga_source_id": str(cfg.get("tcga_source_id", "")),
        "rna_asset_id": str(cfg.get("rna_asset_id", "")),
        "dataset_id": str(cfg.get("dataset_id", "")),
        "training_dataset_id": str(cfg.get("training_dataset_id", "")),
        "package": "src.evaluation.part2_TCGA.taskB_tcga_eval",
    }


def main(args: argparse.Namespace) -> int:

    if args.inner_folds < 2:
        raise RuntimeError("--inner_folds must be >= 2")
    if args.bootstrap_reps < 0:
        raise RuntimeError("--bootstrap_reps must be >= 0")

    paths = resolve_benchmark_paths(
        args.benchmark_id, project_root=args.project_root, modelgroup_id=args.modelgroup_id
    )
    validation_status = validation_gate(paths, allow_unvalidated=bool(args.allow_unvalidated_benchmark))

    min_inner = int(paths.config.get("ridge_inner_cv_folds_min", -1))
    if min_inner < 0:
        raise RuntimeError(
            "benchmark_config.yaml is missing 'ridge_inner_cv_folds_min'. "
            "Rebuild the benchmark with _3_init_benchmark.py."
        )
    if int(args.inner_folds) < min_inner:
        raise RuntimeError(
            f"--inner_folds={args.inner_folds} is below the benchmark's minimum "
            f"feasibility constraint ridge_inner_cv_folds_min={min_inner} "
            f"(benchmark_id={args.benchmark_id})."
        )

    cfg = RidgeProtocolConfig(
        alpha_grid=_parse_alpha_grid(args.alpha_grid),
        inner_folds=int(args.inner_folds),
        inner_seed=int(args.inner_seed),
        bootstrap_reps=int(args.bootstrap_reps),
        bootstrap_seed=int(args.bootstrap_seed),
        pooling=str(args.pooling),
    )

    eval_dir = eval_case_root(args.eval_id, project_root=args.project_root)
    settings_dir = eval_settings_dir(args.eval_id, project_root=args.project_root)
    if eval_dir.exists() and not args.overwrite:
        raise FileExistsError(
            f"Eval folder already exists: {eval_dir}. Use --overwrite to refresh settings."
        )
    if eval_dir.exists() and args.overwrite:
        shutil.rmtree(eval_dir)
    settings_dir.mkdir(parents=True, exist_ok=True)

    protocol_block = {
        **cfg.as_dict(),
        "protocol_hash": cfg.sha256,
    }
    identity_block = _identity_block(args, paths)
    eval_config = {
        "identity": identity_block,
        "protocol": protocol_block,
    }
    eval_config_path = settings_dir / "eval_config.yaml"
    write_yaml_atomic(eval_config_path, eval_config, overwrite=True)

    input_hashes = _benchmark_input_hashes(paths)
    eval_config_identity_payload = {
        "identity": identity_block,
        "protocol": protocol_block,
    }
    protocol_manifest = manifest_header(Path(__file__).name, vars(args))
    protocol_manifest.update({
        "frozen_final": True,
        "eval_id": str(args.eval_id),
        "benchmark_id": str(args.benchmark_id),
        "modelgroup_id": str(paths.modelgroup_id),
        "benchmark_root": str(paths.benchmark_root),
        "benchmark_validation": validation_status,
        "protocol": protocol_block,
        "protocol_hash": cfg.sha256,
        "eval_config_identity_payload": eval_config_identity_payload,
        "eval_config_identity_hash": sha256_obj(eval_config_identity_payload),
        "benchmark_input_hashes": input_hashes,
        "eval_config_path": str(eval_config_path),
        "eval_config_hash": sha256_path(eval_config_path),
    })
    protocol_manifest_path = settings_dir / "protocol_manifest.json"
    write_json_atomic(protocol_manifest_path, protocol_manifest, overwrite=True)

    environment = {
        "created_at_utc": protocol_manifest["created_at_utc"],
        "python": sys.version,
        "platform": sys.platform,
        "software_versions": software_versions(),
        "script_name": Path(__file__).name,
    }
    write_json_atomic(settings_dir / "environment.json", environment, overwrite=True)

    print(json.dumps({
        "status": "ok",
        "eval_id": str(args.eval_id),
        "benchmark_id": str(args.benchmark_id),
        "eval_dir": str(eval_dir),
        "settings_dir": str(settings_dir),
        "protocol_hash": cfg.sha256,
    }, indent=2))
    return 0


if __name__ == "__main__":

    ap = argparse.ArgumentParser(
        description="Set up the ridge probe protocol of one Task B eval_id."
    )

    ap.add_argument("--eval_id", type=str,
                    required=True,
                    help="Evaluation ID; names the run directory under eval/B_eval_case/.")

    ap.add_argument("--benchmark_id", type=str,
                    required=True,
                    help="Validated benchmark ID whose data this eval_id points at.")

    ap.add_argument("--modelgroup_id", type=str,
                    default=None,
                    help="Model group ID. Defaults to the model group referenced by the benchmark.")

    ap.add_argument("--project_root", type=Path,
                    default=None,
                    help="Optional override of the TCGA Task B data root.")

    ap.add_argument("--alpha_grid", type=str,
                    default=None,
                    help="Comma-separated ridge alpha grid; defaults to RidgeProtocolConfig defaults.")

    ap.add_argument("--inner_folds", type=int,
                    default=3,
                    help="Number of inner cross-validation folds for alpha selection. "
                         "Must be >= benchmark_config.yaml::ridge_inner_cv_folds_min.")

    ap.add_argument("--inner_seed", type=int,
                    default=271828,
                    help="Random seed for the inner alpha-selection folds.")

    ap.add_argument("--bootstrap_reps", type=int,
                    default=2000,
                    help="Number of bootstrap repetitions for metric confidence intervals.")

    ap.add_argument("--bootstrap_seed", type=int,
                    default=314159,
                    help="Random seed for the bootstrap resampling.")

    ap.add_argument("--pooling", type=str,
                    choices=list(VALID_POOLINGS),
                    default=DEFAULT_POOLING,
                    help="Case-level pooling of tile embeddings: 'mean_std' (default, concat(mean, std)), "
                         "'mean_std_max' (adds the per-axis max) or 'mean'.")

    ap.add_argument("--allow_unvalidated_benchmark", action="store_true",
                    help="Allow setting up against a benchmark that has not passed _14_validate_benchmark.py.")

    ap.add_argument("--overwrite", action="store_true",
                    help="Delete an existing eval_id folder before recreating it.")

    args = ap.parse_args()
    raise SystemExit(main(args))
