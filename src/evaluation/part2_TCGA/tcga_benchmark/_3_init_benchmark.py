#!/usr/bin/env python3
"""
Initialize one TCGA Task B benchmark: asset links, project_domain map, model group (tile geometry and QC),
encoder registry (from src/utils/models_settings.py), gene-signature panel (data/eval_signatures/) and config.

Saved in <ROOT_HistoST> / eval / eval_datasets / TCGA_eval_data / benchmarks / <benchmark_id> /
    - settings/: benchmark_config.yaml, benchmark_asset_links.json, encoder_registry.json, modelgroups/,
      eval_signatures/, project_domain_map.json, included_projects.json, alias map, provenance.json
    - reports/benchmark_init_report.md
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Optional
import pandas as pd

# Add the project root to sys.path so that 'src' can be imported
project_root = Path(__file__).resolve().parents[4]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.constants import get_full_path
from src.utils.models_settings import BACKBONE_TRANSFORM, BACKBONE_NATIVE_RESOLUTION

from _taskb_common import (
    assert_modelgroup_no_upsampling,
    collect_output_hashes,
    copy_or_hardlink,
    load_domain_map_checked,
    manifest_header,
    read_json,
    read_yaml,
    sha256_path,
    write_json_atomic,
    write_yaml_atomic,
    write_text_atomic
)

EVAL_SIGNATURES_REPO_DIR = project_root / "data" / "eval_signatures"
EVAL_SIGNATURES_FILES = ("signatures.gmt", "sources.json")

# Max ratio between an encoder's native mpp and the model group's nominal_mpp_um_per_px
MODELGROUP_MPP_RATIO_TOL = 1.25

# Max difference (um/px) between the training and model-group resolutions without --allow_resolution_redefinition
RESOLUTION_CONSISTENCY_ABS_TOL_UM = 0.02


def parse_dataset_id_geometry(dataset_id: str) -> Dict[str, int]:
    """Parse a ds_ct{c}_{res}x_{size}_o{overlap} dataset_id into {ct, res_x, size_px, overlap_px} (raises otherwise)."""
    m = re.fullmatch(r"ds_ct(\d+)_(\d+)x_(\d+)_o(\d+)", str(dataset_id))
    if not m:
        raise RuntimeError(
            f"dataset_id {dataset_id!r} does not match the expected convention "
            "ds_ct{c}_{res}x_{size}_o{overlap} (e.g. ds_ct1_20x_224_o112)."
        )
    return {
        "ct": int(m.group(1)),
        "res_x": int(m.group(2)),
        "size_px": int(m.group(3)),
        "overlap_px": int(m.group(4)),
    }


DEFAULT_MODEL_GROUP = {
    "modelgroup_id": "224_112um_nooverlap_noups_qcv1",
    "nominal_magnification": 20,
    "nominal_mpp_um_per_px": 0.5,
    "crop_size": 224,
    "resize_size": 224,
    "tile_size_px_encoder_input": 224,
    "tile_fov_um_read": 112.0,
    "tile_fov_um_effective": 112.0,
    "tile_stride_um": 112.0,
    "overlap": 0,
    "rgba_policy": "openslide_rgba_composite_white",
    "rgb_policy": "convert_to_rgb",
    "icc_profile_policy": "ignore_embedded_icc_use_openslide_returned_rgb",
    "qc_version": "tcga_tile_qc_v1",
    "objective_power_fallback_allowed": False,
    "mpp_policy": "openslide_mpp_required",
    # MPP acceptance bounds of a slide (OpenSlide MPP); no upsampling:
    # mpp_max = nominal_mpp_um_per_px * (1 + max_upsample_tol_frac), checked by assert_modelgroup_no_upsampling
    "mpp_min": 0.20,
    "mpp_max": 0.505,
    "max_upsample_tol_frac": 0.01,
    "mpp_anisotropy_max": 1.10,
    "mpp_prefetch_abs_tol": 0.02,
    "tiling_version": "tcga_grid_v1",
    # Tiling toolchain version (OpenSlide decode, OpenCV resize/QC), part of set_recipe_id: bump it when tiles change
    "tiling_toolchain_version": "v1",
}


# Named CLI inputs are resolved under eval/eval_datasets/TCGA_eval_data/refs/
TCGA_REFS = ("eval", "eval_datasets", "TCGA_eval_data", "refs")


def resolve_ref_path(name: Optional[str], *subdirs: str) -> Optional[Path]:
    """Resolve a CLI name (may contain sub-paths) into a path under TCGA_eval_data/refs/, or None."""
    if name is None:
        return None
    return get_full_path(*TCGA_REFS, *subdirs, name)


def build_encoder_registry(modelgroup_cfg: Mapping[str, Any]) -> Dict[str, Any]:
    """Derive the encoder registry from src/utils/models_settings.py.

    Keeps the encoders whose geometry (resize/crop/tile size) matches the model group and whose native mpp is
    within MODELGROUP_MPP_RATIO_TOL of it; those excluded on resolution are listed under excluded_for_magnification.
    """
    # Both tables must define the same backbones
    if set(BACKBONE_NATIVE_RESOLUTION) != set(BACKBONE_TRANSFORM):
        raise RuntimeError(
            "BACKBONE_NATIVE_RESOLUTION and BACKBONE_TRANSFORM must define the same backbones; "
            f"got {sorted(BACKBONE_NATIVE_RESOLUTION)} vs {sorted(BACKBONE_TRANSFORM)}."
        )
    mg_crop = int(modelgroup_cfg["crop_size"])
    mg_resize = int(modelgroup_cfg["resize_size"])
    mg_tile = int(modelgroup_cfg.get("tile_size_px_encoder_input", mg_crop))
    if "nominal_mpp_um_per_px" not in modelgroup_cfg:
        raise RuntimeError("model group is missing nominal_mpp_um_per_px")
    mg_mpp = float(modelgroup_cfg["nominal_mpp_um_per_px"])
    if not (mg_mpp > 0):
        raise RuntimeError(f"model group nominal_mpp_um_per_px must be positive, got {mg_mpp}")

    encoders = []
    excluded_for_magnification = []
    for backbone, tfm in BACKBONE_TRANSFORM.items():
        crop = int(tfm["crop_size"])
        resize = int(tfm["resize_size"])
        if crop != mg_crop or resize != mg_resize or crop != mg_tile:
            continue
        res = BACKBONE_NATIVE_RESOLUTION[backbone]
        enc_mpp = float(res["nominal_mpp_um_per_px"])
        if not (enc_mpp > 0):
            raise RuntimeError(f"{backbone}: nominal_mpp_um_per_px must be positive, got {enc_mpp}")
        ratio = enc_mpp / mg_mpp
        if ratio > MODELGROUP_MPP_RATIO_TOL or ratio < 1.0 / MODELGROUP_MPP_RATIO_TOL:
            excluded_for_magnification.append({
                "backbone": backbone,
                "encoder_nominal_mpp_um_per_px": enc_mpp,
                "modelgroup_nominal_mpp_um_per_px": mg_mpp,
                "mpp_ratio": ratio,
                "ratio_tolerance": MODELGROUP_MPP_RATIO_TOL,
            })
            continue
        encoders.append({
            "backbone": backbone,
            "resize_size": resize,
            "crop_size": crop,
            "interpolation_code": int(tfm["interpolation"]),
            "mean": [float(x) for x in tfm["mean"]],
            "std": [float(x) for x in tfm["std"]],
            "nominal_mpp_um_per_px": enc_mpp,
            "nominal_magnification": res.get("nominal_magnification"),
        })
    if excluded_for_magnification:
        print(
            "WARNING: the following backbones match the model group pixel geometry but were "
            "excluded for native-resolution mismatch (see encoder_registry.json -> "
            f"excluded_for_magnification): {[e['backbone'] for e in excluded_for_magnification]}"
        )
    if not encoders:
        raise RuntimeError(
            "No model in models_settings matches the model group geometry and native resolution "
            f"(crop_size={mg_crop}, resize_size={mg_resize}, tile_size_px_encoder_input={mg_tile}, "
            f"nominal_mpp_um_per_px={mg_mpp})."
        )
    return {
        "registry_schema_version": "v2",
        "generated_by": "_3_init_benchmark.py",
        "generated_from": "src/utils/models_settings.py::BACKBONE_TRANSFORM + BACKBONE_NATIVE_RESOLUTION",
        "modelgroup_id": str(modelgroup_cfg.get("modelgroup_id")),
        "modelgroup_nominal_mpp_um_per_px": mg_mpp,
        "mpp_ratio_tolerance": MODELGROUP_MPP_RATIO_TOL,
        "encoders": sorted(encoders, key=lambda e: e["backbone"]),
        "excluded_for_magnification": sorted(excluded_for_magnification, key=lambda e: e["backbone"]),
    }


def load_or_default(path: Path | None, default: Mapping[str, Any]) -> Dict[str, Any]:
    if path is None:
        return dict(default)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".json":
        return dict(read_json(path))
    return dict(read_yaml(path))


def copy_eval_signatures(dst_dir: Path, *, overwrite: bool) -> Dict[str, Any]:
    """Copy data/eval_signatures/{signatures.gmt, sources.json} into the benchmark; returns the copy record with SHA256s."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    file_records: Dict[str, Dict[str, str]] = {}
    for name in EVAL_SIGNATURES_FILES:
        src = EVAL_SIGNATURES_REPO_DIR / name
        if not src.exists():
            raise FileNotFoundError(src)
        dst = dst_dir / name
        mode = copy_or_hardlink(src, dst, overwrite=overwrite)
        dst_hash = sha256_path(dst)
        src_hash = sha256_path(src)
        if dst_hash != src_hash:
            raise RuntimeError(f"eval_signatures copy hash mismatch for {name}: src={src_hash} dst={dst_hash}")
        file_records[name] = {
            "source": str(src),
            "destination": str(dst),
            "mode": mode,
            "sha256": dst_hash,
        }
    return {
        "status": "copied",
        "source_dir": str(EVAL_SIGNATURES_REPO_DIR),
        "destination_dir": str(dst_dir),
        "files": file_records,
    }


def count_gmt_signatures(gmt_path: Path) -> int:
    n = 0
    with gmt_path.open("r") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def modelgroup_id_from_config(cfg: Mapping[str, Any], fallback: str) -> str:
    return str(cfg.get("modelgroup_id") or fallback)


def validate_rna_asset_manifest(
    rna_asset_root: Path,
    source_alias_hash: str,
    expected_tcga_source_id: str,
    *,
    allow_missing: bool = False,
) -> Dict[str, Any]:
    """Validate an already-built RNA asset, or record a planned asset.
    """
    manifest_path = rna_asset_root / "rna_asset_manifest.json"
    if not manifest_path.exists():
        if allow_missing:
            return {
                "status": "planned_not_built_yet",
                "rna_asset_root": str(rna_asset_root),
                "rna_asset_manifest_path": str(manifest_path),
                "expected_tcga_source_id": expected_tcga_source_id,
                "expected_alias_map_sha256": source_alias_hash,
            }
        raise FileNotFoundError(manifest_path)
    m = read_json(manifest_path)
    rna_alias_hash = m.get("alias_map_sha256")
    if rna_alias_hash != source_alias_hash:
        raise RuntimeError(f"RNA asset alias map hash mismatch: RNA={rna_alias_hash} source={source_alias_hash}")
    observed_source_id = m.get("tcga_source_id")
    if str(observed_source_id) != str(expected_tcga_source_id):
        raise RuntimeError(
            "RNA asset was built from a different TCGA source release: "
            f"RNA={observed_source_id} expected={expected_tcga_source_id}"
        )
    m = dict(m)
    m.setdefault("status", "built")
    return m


def load_project_universe(source_root: Path, rna_asset_root: Optional[Path] = None) -> set[str]:
    projects: set[str] = set()

    project_inventory = source_root / "inventory" / "project_inventory.parquet"
    if project_inventory.exists():
        df = pd.read_parquet(project_inventory, columns=["project_id"])
        projects.update(str(x) for x in df["project_id"].dropna().astype(str))

    for inventory_name in ["candidate_rna_files.parquet", "candidate_slide_files.parquet", "selected_rna_files.parquet"]:
        inv_path = source_root / "inventory" / inventory_name
        if inv_path.exists():
            try:
                df = pd.read_parquet(inv_path, columns=["project_id"])
            except Exception:
                df = pd.read_parquet(inv_path)
            if "project_id" in df.columns:
                projects.update(str(x) for x in df["project_id"].dropna().astype(str))

    if rna_asset_root is not None:
        expr_path = rna_asset_root / "case_expression_log10_fpkmuq.parquet"
        if expr_path.exists():
            df = pd.read_parquet(expr_path, columns=["project_id"])
            projects.update(str(x) for x in df["project_id"].dropna().astype(str))

    if not projects:
        raise RuntimeError("Could not infer any TCGA projects from the source release inventories")
    return projects


def validate_project_domain_map_coverage(domain_map: Mapping[str, Any], project_universe: set[str]) -> None:
    labels = domain_map.get("project_domain_map") or domain_map.get("projects") or {}
    labeled = set(str(x) for x in labels.keys())
    missing = sorted(project_universe - labeled)
    extra = sorted(labeled - project_universe)
    if missing:
        raise RuntimeError(f"project_domain_map is missing labels for TCGA projects: {missing}")
    if extra:
        raise RuntimeError(f"project_domain_map contains projects absent from the frozen source/RNA universe: {extra}")


def included_projects_from_domain_map(domain_map: Mapping[str, Any]) -> Dict[str, Any]:
    labels = domain_map.get("project_domain_map") or domain_map.get("projects") or {}
    included = sorted([p for p, lab in labels.items() if lab in {"in_domain", "out_of_domain"}])
    excluded = sorted([p for p, lab in labels.items() if lab == "excluded"])
    return {"included_projects": included, "excluded_projects": excluded, "source": "manual_reviewed_project_domain_map"}


def main(args) -> int:

    data_root = get_full_path('eval', 'eval_datasets', 'TCGA_eval_data')
    source_root = data_root / "source_releases" / args.tcga_source_id
    rna_asset_root = data_root / "processed_assets" / "rna" / args.rna_asset_id
    benchmark_root = data_root / "benchmarks" / args.benchmark_id
    settings_dir = benchmark_root / "settings"
    shared_dir = benchmark_root / "shared"
    reports_dir = benchmark_root / "reports"
    modelgroups_dir = settings_dir / "modelgroups"

    # Named CLI inputs resolve under eval/eval_datasets/TCGA_eval_data/refs/.
    modelgroup_yaml = resolve_ref_path(args.modelgroup_yaml)
    benchmark_config_path = resolve_ref_path(args.benchmark_config)

    if not source_root.exists():
        raise FileNotFoundError(source_root)
    # rna_asset_root may not exist yet (built later by _12_build_rna_asset.py)
    if benchmark_root.exists() and not args.force:
        raise FileExistsError(f"Benchmark already exists: {benchmark_root}. Use --force to rebuild.")
    if benchmark_root.exists() and args.force and not args.force_cascade:
        downstream_globs = [data_root / "selected_slide_cache" / args.benchmark_id, data_root / "processed_assets" / "tiles"]
        existing = [p for p in downstream_globs if p.exists()]
        if existing:
            raise RuntimeError(f"Refusing to force rebuild benchmark while downstream assets exist: {existing}. Use --force_cascade if intentional.")
    settings_dir.mkdir(parents=True, exist_ok=True)
    shared_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    modelgroups_dir.mkdir(parents=True, exist_ok=True)

    # Copy the source-release alias map
    alias_candidates = sorted((source_root / "settings").glob("gene_alias_map_hgnc_*.tsv"))
    if len(alias_candidates) != 1:
        raise RuntimeError(f"Expected exactly one source alias map; found {len(alias_candidates)}")
    source_alias_map = alias_candidates[0]
    source_alias_hash = sha256_path(source_alias_map)
    alias_sha_sidecar = source_alias_map.with_suffix(".sha256.json")
    sidecars = sorted((source_root / "settings").glob("gene_alias_map_hgnc_*.sha256.json"))
    if len(sidecars) != 1:
        raise RuntimeError(f"Expected exactly one source alias map sha256 sidecar; found {len(sidecars)}")
    alias_sidecar_src = sidecars[0]
    alias_dst = settings_dir / source_alias_map.name
    alias_copy_mode = copy_or_hardlink(source_alias_map, alias_dst, overwrite=args.force)
    sidecar_dst = settings_dir / alias_sidecar_src.name
    sidecar_copy_mode = copy_or_hardlink(alias_sidecar_src, sidecar_dst, overwrite=args.force)
    benchmark_alias_hash = sha256_path(alias_dst)
    if benchmark_alias_hash != source_alias_hash:
        raise RuntimeError("Benchmark alias map copy hash mismatch")

    rna_manifest = validate_rna_asset_manifest(rna_asset_root, source_alias_hash, args.tcga_source_id, allow_missing=True)

    project_domain_map_path = source_root / "settings" / "project_domain_map.json"
    domain_map = load_domain_map_checked(project_domain_map_path)
    project_universe = load_project_universe(source_root, rna_asset_root if rna_asset_root.exists() else None)
    validate_project_domain_map_coverage(domain_map, project_universe)
    included_projects = included_projects_from_domain_map(domain_map)
    included_projects["project_universe"] = sorted(project_universe)
    included_projects["n_project_universe"] = len(project_universe)
    write_json_atomic(settings_dir / "project_domain_map.json", domain_map, overwrite=args.force)
    write_json_atomic(settings_dir / "included_projects.json", included_projects, overwrite=args.force)

    # Gene-signature panel: copy data/eval_signatures/ into settings/eval_signatures/ (used by _4_build_signature_set.py)
    eval_signatures_dst = settings_dir / "eval_signatures"
    eval_signatures_meta = copy_eval_signatures(eval_signatures_dst, overwrite=args.force)

    # Annotation policy is copied from the source release.
    annotation_policy_src = source_root / "settings" / "annotation_policy.yaml"
    if not annotation_policy_src.exists():
        raise FileNotFoundError(annotation_policy_src)
    annotation_copy_mode = copy_or_hardlink(annotation_policy_src, settings_dir / "annotation_policy.yaml", overwrite=args.force)

    # Modelgroup config: only geometry/QC. Encoder preprocessing goes into registry.
    modelgroup_cfg = load_or_default(modelgroup_yaml, DEFAULT_MODEL_GROUP)
    modelgroup_cfg["modelgroup_id"] = modelgroup_id_from_config(modelgroup_cfg, args.modelgroup_id)
    if modelgroup_cfg["modelgroup_id"] != args.modelgroup_id:
        raise RuntimeError(f"modelgroup_id mismatch: CLI={args.modelgroup_id} cfg={modelgroup_cfg['modelgroup_id']}")

    # Training dataset vs model group: the pixel size must match; a different um/px requires
    # --allow_resolution_redefinition
    ds_geom = parse_dataset_id_geometry(args.dataset_id)
    mg_crop = int(modelgroup_cfg["crop_size"])
    mg_tile = int(modelgroup_cfg.get("tile_size_px_encoder_input", mg_crop))
    if ds_geom["size_px"] != mg_crop or ds_geom["size_px"] != mg_tile:
        raise RuntimeError(
            f"dataset_id pixel size ({ds_geom['size_px']}) disagrees with the model group "
            f"(crop_size={mg_crop}, tile_size_px_encoder_input={mg_tile}). The encoder input "
            "pixel size must match the training patch size."
        )
    if "nominal_mpp_um_per_px" not in modelgroup_cfg:
        raise RuntimeError("model group is missing nominal_mpp_um_per_px")
    mg_mpp = float(modelgroup_cfg["nominal_mpp_um_per_px"])
    # No upsampling: mpp_max must not exceed the target resolution (+ tolerance)
    assert_modelgroup_no_upsampling(modelgroup_cfg)
    resolution_check: Dict[str, Any] = {
        "modelgroup_nominal_mpp_um_per_px": mg_mpp,
        "modelgroup_nominal_magnification": modelgroup_cfg.get("nominal_magnification"),
        "dataset_id_geometry": ds_geom,
        "training_pixel_size_um": (float(args.training_pixel_size_um) if args.training_pixel_size_um is not None else None),
        "abs_tol_um": RESOLUTION_CONSISTENCY_ABS_TOL_UM,
        "allow_resolution_redefinition": bool(args.allow_resolution_redefinition),
        "resolution_rationale": str(args.resolution_rationale),
    }
    if args.training_pixel_size_um is not None:
        diff = abs(float(args.training_pixel_size_um) - mg_mpp)
        resolution_check["abs_diff_um"] = diff
        resolution_check["consistent"] = bool(diff <= RESOLUTION_CONSISTENCY_ABS_TOL_UM)
        if diff > RESOLUTION_CONSISTENCY_ABS_TOL_UM and not args.allow_resolution_redefinition:
            raise RuntimeError(
                f"Training pixel size ({args.training_pixel_size_um} um/px) differs from the model "
                f"group nominal_mpp_um_per_px ({mg_mpp} um/px) by {diff:.4f} um/px "
                f"(> {RESOLUTION_CONSISTENCY_ABS_TOL_UM}). Task B evaluates at the model-group "
                "resolution: re-run with --allow_resolution_redefinition and set --resolution_rationale."
            )
        resolution_check["status"] = "consistent" if resolution_check["consistent"] else "redefined_acknowledged"
    else:
        resolution_check["status"] = "training_pixel_size_not_provided_check_skipped"
        print(
            "WARNING: --training_pixel_size_um not provided; the training-vs-eval resolution check is skipped "
            f"(tiles at nominal_mpp_um_per_px={mg_mpp} um/px)."
        )

    write_yaml_atomic(modelgroups_dir / f"{args.modelgroup_id}.yaml", modelgroup_cfg, overwrite=args.force)

    # Encoder registry derived from src/utils/models_settings.py (backbones matching the model group)
    encoder_registry = build_encoder_registry(modelgroup_cfg)
    write_json_atomic(settings_dir / "encoder_registry.json", encoder_registry, overwrite=args.force)

    user_bench_cfg = load_or_default(benchmark_config_path, {}) if benchmark_config_path else {}
    benchmark_config = {
        "benchmark_id": args.benchmark_id,
        "tcga_source_id": args.tcga_source_id,
        "rna_asset_id": args.rna_asset_id,
        "dataset_id": args.dataset_id,
        "training_dataset_id": args.training_dataset_id,
        "modelgroup_id": args.modelgroup_id,
        "data_root": str(data_root),
        "official_primary_biospecimen_match_levels": ["exact_same_portion", "same_sample", "same_case_primary_tumor", "same_case_primary_tumor_unknown_sublevel"],
        "exploratory_only_biospecimen_match_levels": [],
        "min_tiles_per_case": 256,
        "max_tiles_per_case": 8192,
        "fold_count": 5,
        "fold_seed": 20260518,
        "fold_grouping": "tissue_source_site",
        # Max share of a project's cases in one tissue-source site for TSS-grouped folds (else case-level folds)
        "degenerate_tss_fraction": float(args.degenerate_tss_fraction),
        # Minimum inner CV folds of the ridge (the eval_id ridge protocols must use inner_folds >= this)
        "ridge_inner_cv_folds_min": 3,
        "target_construction": "frozen_project_level_gene_zscores_computed_once_after_final_cohort",
        "manual_review_gate_project_domain_map": True,
        "unknown_stain_policy": "allowed_only_for_Diagnostic_Slide_and_flagged",
        "unknown_ffpe_policy": "allowed_only_for_diagnostic_primary_tumor_HE_compatible_or_unknown_stain_and_flagged",
        # Max cases per TCGA project kept by _6_resolve_rna_and_slides.py (<=0: no cap)
        "max_cases_per_project": int(args.max_cases_per_project),
        # Seed for the deterministic per-project case-cap subsampling.
        "case_cap_seed": int(args.case_cap_seed),
        # Case-cap selection method: 'tss_stratified_round_robin' (interleaves tissue-source sites) or 'random'
        "case_cap_method": str(args.case_cap_method),
        # Max slides per case kept by _6_resolve_rna_and_slides.py at the case's best biospecimen tier
        # (ranked H&E-known > FFPE-known > larger file_size > uuid); <=0: no cap
        "max_slides_per_case": int(args.max_slides_per_case),
        # Over-provision for the case cap: _6 selects ceil(max_cases_per_project * (1 + this)) cases per project,
        # and _11_build_case_tile_index.py trims the tile-QC survivors back to max_cases_per_project (0: disabled)
        "case_cap_overprovision_frac": float(args.case_cap_overprovision_frac),
        # Image codec of the shared tile RGB cache (_15), part of codec_recipe_id
        "tile_recipe": {
            "codec": {"format": "jpeg", "quality": 95, "chroma": "444", "png_level": 6},
        },
    }
    immutable_config_keys = {
        "benchmark_id": args.benchmark_id,
        "tcga_source_id": args.tcga_source_id,
        "rna_asset_id": args.rna_asset_id,
        "dataset_id": args.dataset_id,
        "training_dataset_id": args.training_dataset_id,
        "modelgroup_id": args.modelgroup_id,
        "data_root": str(data_root),
    }
    forbidden_overrides = {
        k: {"cli": v, "override": user_bench_cfg[k]}
        for k, v in immutable_config_keys.items()
        if k in user_bench_cfg and str(user_bench_cfg[k]) != str(v)
    }
    if forbidden_overrides:
        raise RuntimeError(
            "benchmark_config override attempts to change immutable benchmark identity fields: "
            + json.dumps(forbidden_overrides, sort_keys=True, default=str)
        )
    benchmark_config.update(user_bench_cfg)
    benchmark_config.update(immutable_config_keys)
    # Set after the config overrides
    benchmark_config["resolution_consistency"] = resolution_check
    write_yaml_atomic(settings_dir / "benchmark_config.yaml", benchmark_config, overwrite=args.force)

    asset_links = {
        "benchmark_id": args.benchmark_id,
        "tcga_source_id": args.tcga_source_id,
        "source_release_root": str(source_root),
        "source_release_provenance_path": str(source_root / "settings" / "provenance.json"),
        "rna_asset_id": args.rna_asset_id,
        "rna_asset_root": str(rna_asset_root),
        "rna_asset_manifest_path": str(rna_asset_root / "rna_asset_manifest.json"),
        "rna_asset_status_at_init": rna_manifest.get("status"),
        "modelgroup_id": args.modelgroup_id,
        "modelgroup_yaml_path": str(modelgroups_dir / f"{args.modelgroup_id}.yaml"),
        "alias_map_path": str(alias_dst),
        "alias_map_sha256": benchmark_alias_hash,
        "eval_signatures_dir": str(eval_signatures_dst),
        "signatures_gmt_path": eval_signatures_meta["files"]["signatures.gmt"]["destination"],
        "signatures_gmt_sha256": eval_signatures_meta["files"]["signatures.gmt"]["sha256"],
        "signatures_sources_path": eval_signatures_meta["files"]["sources.json"]["destination"],
        "signatures_sources_sha256": eval_signatures_meta["files"]["sources.json"]["sha256"],
        "encoder_registry_path": str(settings_dir / "encoder_registry.json"),
        "project_domain_map_path": str(settings_dir / "project_domain_map.json"),
    }
    write_json_atomic(settings_dir / "benchmark_asset_links.json", asset_links, overwrite=args.force)

    provenance = manifest_header(Path(__file__).name, vars(args))
    provenance.update({
        "benchmark_id": args.benchmark_id,
        "source_release_alias_map_path": str(source_alias_map),
        "source_release_alias_map_hash": source_alias_hash,
        "benchmark_alias_map_path": str(alias_dst),
        "benchmark_alias_map_hash": benchmark_alias_hash,
        "alias_copy_mode": alias_copy_mode,
        "alias_sidecar_copy_mode": sidecar_copy_mode,
        "rna_asset_alias_map_hash": rna_manifest.get("alias_map_sha256"),
        "all_available_alias_hashes_match_source": rna_manifest.get("alias_map_sha256") in (None, source_alias_hash),
        "project_domain_map_hash": sha256_path(settings_dir / "project_domain_map.json"),
        "project_domain_manual_review": (domain_map.get("manual_review") or domain_map.get("metadata")),
        "included_projects_hash": sha256_path(settings_dir / "included_projects.json"),
        "eval_signatures_meta": eval_signatures_meta,
        "annotation_policy_hash": sha256_path(settings_dir / "annotation_policy.yaml"),
        "annotation_policy_copy_mode": annotation_copy_mode,
        "modelgroup_hash": sha256_path(modelgroups_dir / f"{args.modelgroup_id}.yaml"),
        "encoder_registry_hash": sha256_path(settings_dir / "encoder_registry.json"),
        "encoder_registry_excluded_for_magnification": encoder_registry.get("excluded_for_magnification"),
        "resolution_consistency": resolution_check,
        "rna_asset_manifest_hash": sha256_path(rna_asset_root / "rna_asset_manifest.json") if (rna_asset_root / "rna_asset_manifest.json").exists() else None,
        "benchmark_asset_links_hash": sha256_path(settings_dir / "benchmark_asset_links.json"),
        "output_hashes": collect_output_hashes(benchmark_root),
    })
    write_json_atomic(settings_dir / "provenance.json", provenance, overwrite=args.force)

    n_eval_signatures = count_gmt_signatures(eval_signatures_dst / "signatures.gmt")
    report = f"""# Task B benchmark initialization report

- benchmark_id: `{args.benchmark_id}`
- tcga_source_id: `{args.tcga_source_id}`
- rna_asset_id: `{args.rna_asset_id}` ({rna_manifest.get('status')})
- modelgroup_id: `{args.modelgroup_id}`
- encoder registry (auto-derived from models_settings.py): {', '.join(e['backbone'] for e in encoder_registry['encoders'])}
- encoders excluded for native-resolution mismatch: {', '.join(e['backbone'] for e in encoder_registry.get('excluded_for_magnification', [])) or 'none'}
- resolution check: `{resolution_check['status']}` (model group {resolution_check['modelgroup_nominal_mpp_um_per_px']} um/px; training {resolution_check['training_pixel_size_um']} um/px)
- project domain map reviewed: `{(domain_map.get('manual_review') or domain_map.get('metadata')).get('reviewed')}`
- included projects: {len(included_projects['included_projects'])}
- excluded projects: {len(included_projects['excluded_projects'])}
- alias map SHA256: `{benchmark_alias_hash}`
- eval signatures dir: `{eval_signatures_dst}`
- signatures.gmt SHA256: `{eval_signatures_meta['files']['signatures.gmt']['sha256']}`
- sources.json SHA256: `{eval_signatures_meta['files']['sources.json']['sha256']}`
- signatures in panel: {n_eval_signatures}

The signature_set_tcga_case_v1.json and signature_support_training.json artifacts
are produced by _4_build_signature_set.py and _5_build_signature_support_training.py
respectively, from the eval_signatures/ panel copied here.
"""
    write_text_atomic(reports_dir / "benchmark_init_report.md", report, overwrite=args.force)

    print(json.dumps({"status": "ok", "benchmark_root": str(benchmark_root), "benchmark_id": args.benchmark_id}, indent=2))
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Initialize the metadata/configuration directory for one TCGA Task B benchmark.")
    
    ap.add_argument("--benchmark_id", type=str,
                    required=True,
                    help="Unique identifier for this benchmark, e.g. 'tcga_benchmark_1'.")

    ap.add_argument("--tcga_source_id", type=str,
                    required=True,
                    help="Source release ID, e.g. gdc_tcga_YYYYMMDD_starcounts_dxslides_v1.")
    
    ap.add_argument("--rna_asset_id", type=str, 
                    required=True, 
                    help="RNA asset ID, e.g. rna_tcga_YYYYMMDD_starcounts_v1.")
    
    ap.add_argument("--dataset_id", type=str, 
                    required=True, 
                    help="Dataset ID from HistoST, e.g. ds_ct1_20x_224_o112")
    
    ap.add_argument("--training_dataset_id", type=str, 
                    required=True, 
                    help="Training dataset ID from HistoST, e.g. ds_ct1_20x_224_o112_1")

    ap.add_argument("--modelgroup_id", type=str,
                    default="224_112um_nooverlap_noups_qcv1",
                    help="Model group ID, e.g. '224_112um_nooverlap_noups_qcv1'.")

    ap.add_argument("--modelgroup_yaml", type=str,
                    default=None,
                    help="Optional YAML file to override the default model group config, "
                         "given as a path relative to eval/eval_datasets/TCGA_eval_data/refs/.")

    ap.add_argument("--benchmark_config", type=str,
                    default=None,
                    help="Optional YAML/JSON overrides for benchmark_config.yaml, "
                         "given as a path relative to eval/eval_datasets/TCGA_eval_data/refs/.")

    ap.add_argument("--training_pixel_size_um", type=float,
                    default=None,
                    help="Pixel size (um/px) of the HistoST training patches, checked against the model group's "
                         "nominal_mpp_um_per_px (the check is skipped if omitted).")

    ap.add_argument("--allow_resolution_redefinition", action="store_true",
                    help="Allow a training pixel size that differs from the model-group resolution "
                         "(see --training_pixel_size_um).")

    ap.add_argument("--resolution_rationale", type=str,
                    default=("Task B evaluates at the model group's nominal resolution (foundation-native "
                             "0.5 um/px / 20x), the standard deployment resolution for these encoders. Any "
                             "difference from the HistoST training pixel size is intrinsic (the finetuned encoder "
                             "and its foundation baseline have different native resolutions) and is left in place "
                             "by design."),
                    help="Free-text rationale recorded in benchmark_config/provenance for the chosen Task B "
                         "evaluation resolution.")

    ap.add_argument("--max_cases_per_project", type=int,
                    default=75,
                    help="Max cases per TCGA project kept by _6_resolve_rna_and_slides.py (<=0: all eligible cases).")

    ap.add_argument("--case_cap_seed", type=int,
                    default=20260518,
                    help="Seed of the per-project case-cap subsampling.")

    ap.add_argument("--case_cap_method", type=str,
                    default="tss_stratified_round_robin",
                    choices=["tss_stratified_round_robin", "random"],
                    help="Case-cap selection method: 'tss_stratified_round_robin' interleaves tissue-source sites, "
                         "'random' is a seeded subsample.")

    ap.add_argument("--max_slides_per_case", type=int,
                    default=1,
                    help="Max slides per case kept by _6_resolve_rna_and_slides.py at the case's best biospecimen "
                         "tier (<=0: all).")

    ap.add_argument("--case_cap_overprovision_frac", type=float,
                    default=0.15,
                    help="Over-provision fraction for the case cap: _6 selects ceil(max_cases_per_project * (1 + this)) "
                         "cases per project, trimmed back by _11_build_case_tile_index.py (0: disabled).")

    ap.add_argument("--degenerate_tss_fraction", type=float,
                    default=0.5,
                    help="Max share of a project's cases in one tissue-source site for TSS-grouped folds, "
                         "above which case-level folds are used.")

    ap.add_argument("--force",
                    action="store_true",
                    help="Allow overwriting existing benchmark directory if it exists. Does not allow overwriting if downstream assets exist, unless --force_cascade is also passed.")
    
    ap.add_argument("--force_cascade", action="store_true",
                    help="When used with --force, allows overwriting existing benchmark directory even if downstream assets exist.")
    
    args = ap.parse_args()
    raise SystemExit(main(args))
