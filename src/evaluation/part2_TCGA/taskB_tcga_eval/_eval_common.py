#!/usr/bin/env python3
"""Shared helpers for TCGA Task B evaluation (uses tcga_benchmark/_taskb_common.py, with a fallback)."""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import math
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def _bootstrap_common_path() -> None:
    here = Path(__file__).resolve()
    candidates: List[Path] = []
    # Normal project layout: .../part2_TCGA/taskB_tcga_eval/_eval_common.py
    try:
        candidates.append(here.parent.parent / "tcga_benchmark")
        candidates.append(here.parents[3])
    except Exception:
        pass
    candidates.extend([
        Path.cwd(),
        Path.cwd() / "src" / "evaluation" / "part2_TCGA" / "tcga_benchmark",
        Path.cwd() / "tcga_benchmark",
    ])
    for p in candidates:
        try:
            ps = str(p.resolve())
        except Exception:
            ps = str(p)
        if p.exists() and ps not in sys.path:
            sys.path.insert(0, ps)


_bootstrap_common_path()

try:  # pragma: no cover - exercised in the real project tree
    from _taskb_common import (  # type: ignore
        collect_output_hashes,
        manifest_header,
        project_path,
        read_json,
        read_yaml,
        sha256_obj,
        sha256_path,
        taskb_data_root,
        utc_now_iso,
        write_csv_atomic,
        write_json_atomic,
        write_parquet_atomic,
        write_text_atomic,
    )
except Exception:  # pragma: no cover - lightweight fallback for syntax/tests only
    UTC = _dt.timezone.utc

    def utc_now_iso() -> str:
        return _dt.datetime.now(tz=UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")

    def _stable(obj: Any) -> str:
        return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)

    def sha256_obj(obj: Any) -> str:
        return hashlib.sha256(_stable(obj).encode("utf-8")).hexdigest()

    def sha256_path(path: Path, chunk_size: int = 1024 * 1024) -> str:
        h = hashlib.sha256()
        with Path(path).open("rb") as f:
            for chunk in iter(lambda: f.read(chunk_size), b""):
                h.update(chunk)
        return h.hexdigest()

    def read_json(path: Path) -> Any:
        with Path(path).open("r", encoding="utf-8") as f:
            return json.load(f)

    def read_yaml(path: Path) -> Any:
        import yaml
        with Path(path).open("r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _atomic_write(path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
        tmp = Path(tmp_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(text)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
        except Exception:
            tmp.unlink(missing_ok=True)
            raise

    def write_json_atomic(path: Path, obj: Any, *, overwrite: bool = False) -> None:
        if Path(path).exists() and not overwrite:
            raise FileExistsError(path)
        _atomic_write(Path(path), json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n")

    def write_text_atomic(path: Path, text: str, *, overwrite: bool = False) -> None:
        if Path(path).exists() and not overwrite:
            raise FileExistsError(path)
        _atomic_write(Path(path), text)

    def write_csv_atomic(path: Path, df: pd.DataFrame, *, overwrite: bool = False, index: bool = False) -> None:
        if Path(path).exists() and not overwrite:
            raise FileExistsError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.tmp")
        df.to_csv(tmp, index=index)
        os.replace(tmp, path)

    def write_parquet_atomic(path: Path, df: pd.DataFrame, *, overwrite: bool = False) -> None:
        if Path(path).exists() and not overwrite:
            raise FileExistsError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.tmp")
        df.to_parquet(tmp, index=False)
        os.replace(tmp, path)

    def collect_output_hashes(root: Path) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        root = Path(root)
        if not root.exists():
            return out
        for p in sorted(x for x in root.rglob("*") if x.is_file() and not x.name.startswith(".")):
            out[str(p.relative_to(root))] = {"sha256": sha256_path(p), "size_bytes": p.stat().st_size}
        return out

    def manifest_header(script_name: str, args: Mapping[str, Any]) -> Dict[str, Any]:
        return {"created_at_utc": utc_now_iso(), "script_name": script_name, "cli_args": dict(args)}

    def project_path(*parts: str, project_root: Optional[Path] = None) -> Path:
        # project_root is joined directly; otherwise get_full_path (cwd only if constants cannot be imported)
        if project_root is not None:
            return Path(project_root).joinpath(*parts)
        get_full_path = None
        try:
            from src.utils.constants import get_full_path  # type: ignore
        except Exception:
            try:
                from utils.constants import get_full_path  # type: ignore
            except Exception:
                get_full_path = None
        if get_full_path is not None:
            return Path(get_full_path(*parts))
        return Path.cwd().joinpath(*parts)

    def taskb_data_root(project_root: Optional[Path] = None) -> Path:
        return project_path("eval", "eval_datasets", "TCGA_eval_data", project_root=project_root)


PREDICTION_REQUIRED_COLUMNS = [
    "case_id", "case_uuid", "project_id", "fold_id", "y_true", "y_pred",
    "n_tiles_eligible_before_cap", "n_tiles_used_after_cap", "n_slides_retained",
    "match_tier_best_case", "biospecimen_match_level", "official_primary_match",
    "encoder_id", "training_num", "checkpoint_num", "space", "modelgroup_id", "tile_feature_postproc",
    "support_category", "domain_slice", "signature_name", "signature_family", "signature_source_name",
    "signature_source_version", "signature_set_hash", "target_table_hash", "fold_assignment_hash", "feature_file_hash",
    "dominant_slide_tile_fraction_after_cap", "slide_tile_entropy_after_cap", "slide_tile_effective_n_after_cap",
    "single_slide_case", "multi_slide_case", "cap_applied",
]


# Display labels of baseline ids (the baseline_id stays the on-disk key; ids absent here use the raw id)
BASELINE_LABELS: Dict[str, str] = {
    "batch_stain_diagnostic": "site + scanner-MPP confound floor",
}

BASELINE_DESCRIPTIONS: Dict[str, str] = {
    "batch_stain_diagnostic": (
        "Confound floor reachable from acquisition metadata alone, with no morphology: "
        "tissue-source-site code, tile-contribution-weighted stain/vendor/prefetch "
        "categories, and mean scanner MPP."
    ),
}


@dataclass(frozen=True)
class BenchmarkPaths:
    data_root: Path
    benchmark_root: Path
    settings_dir: Path
    shared_dir: Path
    cohort_dir: Path
    tiles_dir: Path
    folds_dir: Path
    rna_dir: Path
    target_tables_dir: Path
    modelgroup_id: str
    config_path: Path
    links_path: Path
    config: Dict[str, Any]
    links: Dict[str, Any]


@dataclass(frozen=True)
class FeatureBundle:
    features: pd.DataFrame
    feature_columns: List[str]
    numeric_columns: List[str]
    categorical_columns: List[str]
    metadata: Dict[str, Any]
    feature_hash_by_project: Dict[str, str]
    feature_table_hash: Optional[str] = None


def truthy(x: Any) -> bool:
    if isinstance(x, bool):
        return x
    if x is None:
        return False
    try:
        if pd.isna(x):
            return False
    except Exception:
        pass
    return str(x).strip().lower() in {"true", "t", "1", "yes", "y"}


def nonempty(x: Any) -> bool:
    if x is None:
        return False
    try:
        if pd.isna(x):
            return False
    except Exception:
        pass
    s = str(x).strip()
    return bool(s and s.lower() not in {"nan", "none", "null", "na", "n/a", "<na>"})


def safe_name(x: Any, max_len: int = 160) -> str:
    s = re.sub(r"[^A-Za-z0-9_.=-]+", "_", str(x).strip())
    s = re.sub(r"_+", "_", s).strip("_") or "item"
    return s[:max_len]


def ordered(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    out = df.copy()
    for c in columns:
        if c not in out.columns:
            out[c] = pd.NA
    extras = [c for c in out.columns if c not in columns]
    return out.loc[:, list(columns) + extras]


def require_columns(df: pd.DataFrame, cols: Sequence[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"{name} missing required columns: {missing}")


def merge_no_suffix(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    on,
    how: str = "left",
    validate: Optional[str] = None,
    name: str = "merge",
) -> pd.DataFrame:
    """Merge that raises if non-key columns collide (instead of creating _x/_y suffixed columns)."""
    on_cols = [on] if isinstance(on, str) else list(on)
    overlap = sorted((set(left.columns) & set(right.columns)) - set(on_cols))
    if overlap:
        raise RuntimeError(
            f"{name}: refusing to merge -- non-key columns collide and would be "
            f"suffixed _x/_y: {overlap} (join on {on_cols}). Drop them from one "
            f"side or use coalesce_merge to let the right side win."
        )
    return left.merge(right, on=on, how=how, validate=validate)


def coalesce_merge(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    on,
    how: str = "left",
    validate: Optional[str] = None,
    overwrite: Sequence[str] = (),
) -> pd.DataFrame:
    """Left-merge where, for each column in overwrite, the right frame wins (other collisions raise, see merge_no_suffix)."""
    drop = [c for c in overwrite if c in left.columns and c in right.columns]
    base = left.drop(columns=drop, errors="ignore")
    return merge_no_suffix(base, right, on=on, how=how, validate=validate, name="coalesce_merge")


def numeric_or_nan(x: Any) -> float:
    if not nonempty(x):
        return float("nan")
    try:
        v = float(str(x).strip())
    except Exception:
        return float("nan")
    return v if math.isfinite(v) else float("nan")


def read_mapping(path: Path) -> Dict[str, Any]:
    obj = read_json(path) if Path(path).suffix.lower() == ".json" else read_yaml(path)
    if not isinstance(obj, Mapping):
        raise RuntimeError(f"Expected mapping at {path}")
    return dict(obj)


def resolve_benchmark_paths(benchmark_id: str, project_root: Optional[Path] = None, modelgroup_id: Optional[str] = None) -> BenchmarkPaths:
    data_root = taskb_data_root(project_root)
    benchmark_root = data_root / "benchmarks" / str(benchmark_id)
    settings_dir = benchmark_root / "settings"
    shared_dir = benchmark_root / "shared"
    cohort_dir = shared_dir / "cohort"
    tiles_dir = shared_dir / "tiles"
    folds_dir = shared_dir / "folds"
    rna_dir = shared_dir / "rna"
    target_tables_dir = rna_dir / "target_tables"
    config_path = settings_dir / "benchmark_config.yaml"
    links_path = settings_dir / "benchmark_asset_links.json"
    if not config_path.exists():
        raise FileNotFoundError(config_path)
    if not links_path.exists():
        raise FileNotFoundError(links_path)
    cfg = read_mapping(config_path)
    links = read_mapping(links_path)
    mg = str(modelgroup_id or cfg.get("modelgroup_id") or links.get("modelgroup_id") or "")
    if not mg:
        raise RuntimeError("Could not resolve modelgroup_id from CLI/config/links")
    linked = str(cfg.get("modelgroup_id") or links.get("modelgroup_id") or mg)
    if mg != linked:
        raise RuntimeError(f"Requested modelgroup_id={mg} disagrees with benchmark modelgroup_id={linked}")
    return BenchmarkPaths(
        data_root=data_root,
        benchmark_root=benchmark_root,
        settings_dir=settings_dir,
        shared_dir=shared_dir,
        cohort_dir=cohort_dir,
        tiles_dir=tiles_dir,
        folds_dir=folds_dir,
        rna_dir=rna_dir,
        target_tables_dir=target_tables_dir,
        modelgroup_id=mg,
        config_path=config_path,
        links_path=links_path,
        config=cfg,
        links=links,
    )


def validation_gate(paths: BenchmarkPaths, allow_unvalidated: bool = False) -> Dict[str, Any]:
    report_path = paths.benchmark_root / "reports" / "benchmark_validation_report.json"
    if not report_path.exists():
        if allow_unvalidated:
            return {"status": "missing_but_allowed", "path": str(report_path)}
        raise RuntimeError(f"Missing benchmark validation report. Run _14_validate_benchmark.py first: {report_path}")
    obj = read_json(report_path)
    if obj.get("status") != "passed" and not allow_unvalidated:
        raise RuntimeError(f"Benchmark validation did not pass: {report_path}")
    return {"status": obj.get("status"), "path": str(report_path), "sha256": sha256_path(report_path)}


def eval_case_root(eval_id: str, project_root: Optional[Path] = None) -> Path:
    """Root folder of one eval_id (settings and per-encoder probe outputs; features are not eval_id-scoped)."""
    return project_path("eval", "B_eval_case", str(eval_id), project_root=project_root)


def eval_settings_dir(eval_id: str, project_root: Optional[Path] = None) -> Path:
    """Settings folder for one eval_id (eval_config.yaml, protocol_manifest.json, ...)."""
    return eval_case_root(eval_id, project_root=project_root) / "settings"


def benchmark_encoder_features_root(
    benchmark_id: str, features_run_id: str, *, project_root: Optional[Path] = None
) -> Path:
    """
    Encoder feature folder benchmarks/<benchmark_id>/shared/encoder_features/<features_run_id>/
    (shared by all eval_ids of the benchmark; one run holds every embedding space).
    """
    return (
        taskb_data_root(project_root)
        / "benchmarks"
        / str(benchmark_id)
        / "shared"
        / "encoder_features"
        / str(features_run_id)
    )


def benchmark_baseline_features_root(
    benchmark_id: str, baseline_id: str, *, project_root: Optional[Path] = None
) -> Path:
    """Baseline feature folder benchmarks/<benchmark_id>/shared/baseline_features/<baseline_id>/ (shared by all eval_ids)."""
    return (
        taskb_data_root(project_root)
        / "benchmarks"
        / str(benchmark_id)
        / "shared"
        / "baseline_features"
        / str(baseline_id)
    )


def encoder_features_run_id(training_num: int, checkpoint_num: int, *, he_l2_normalize_tile_embeddings: bool = False) -> str:
    """
    Id of an encoder feature run (all spaces in one run; `_he_l2tiles` suffix for the he-L2 run).
    Same as _0_extract_encoder_features._feature_run_id.
    """
    suffix = "_he_l2tiles" if he_l2_normalize_tile_embeddings else ""
    return f"HEencoder_t{int(training_num)}e{int(checkpoint_num)}{suffix}"


def encoder_run_id(training_num: int, checkpoint_num: int, space: str, *, he_l2_normalize_tile_embeddings: bool = False) -> str:
    """Per-(encoder, space) id of the ridge probe / baseline outputs: '..._he', '..._clip' or '..._he_l2tiles'."""
    if he_l2_normalize_tile_embeddings:
        return f"HEencoder_t{int(training_num)}e{int(checkpoint_num)}_he_l2tiles"
    return f"HEencoder_t{int(training_num)}e{int(checkpoint_num)}_{space}"


@dataclass(frozen=True)
class EvalSettings:
    """Settings of one eval_id: eval_config (identity + protocol), protocol_manifest and the SHA256 of both files."""
    eval_id: str
    benchmark_id: str
    settings_dir: Path
    eval_config_path: Path
    protocol_manifest_path: Path
    eval_config: Dict[str, Any]
    protocol_manifest: Dict[str, Any]
    eval_config_hash: str
    protocol_manifest_hash: str


def load_eval_settings(eval_id: str, *, project_root: Optional[Path] = None) -> EvalSettings:
    settings_dir = eval_settings_dir(eval_id, project_root=project_root)
    eval_config_path = settings_dir / "eval_config.yaml"
    protocol_manifest_path = settings_dir / "protocol_manifest.json"
    if not eval_config_path.exists():
        raise FileNotFoundError(
            f"Missing eval settings. Run _1_setup.py for eval_id={eval_id} first: {eval_config_path}"
        )
    if not protocol_manifest_path.exists():
        raise FileNotFoundError(
            f"Missing protocol_manifest.json. Run _1_setup.py for eval_id={eval_id} first: {protocol_manifest_path}"
        )
    cfg = read_mapping(eval_config_path)
    pm = read_mapping(protocol_manifest_path)
    identity = cfg.get("identity") if isinstance(cfg.get("identity"), Mapping) else {}
    benchmark_id = str(identity.get("benchmark_id") or "")
    if not benchmark_id:
        raise RuntimeError(f"eval_config.identity.benchmark_id missing in {eval_config_path}")
    return EvalSettings(
        eval_id=str(eval_id),
        benchmark_id=benchmark_id,
        settings_dir=settings_dir,
        eval_config_path=eval_config_path,
        protocol_manifest_path=protocol_manifest_path,
        eval_config=dict(cfg),
        protocol_manifest=dict(pm),
        eval_config_hash=sha256_path(eval_config_path),
        protocol_manifest_hash=sha256_path(protocol_manifest_path),
    )


def assert_eval_config_matches_protocol_manifest(settings: EvalSettings) -> None:
    """Check that the identity and protocol blocks of eval_config.yaml match the ones saved in protocol_manifest.json."""
    expected = settings.protocol_manifest.get("eval_config_identity_payload")
    if not isinstance(expected, Mapping):
        raise RuntimeError(
            f"protocol_manifest.eval_config_identity_payload missing in {settings.protocol_manifest_path}"
        )
    actual = {
        "identity": settings.eval_config.get("identity"),
        "protocol": settings.eval_config.get("protocol"),
    }
    if actual != dict(expected):
        raise RuntimeError(
            "eval_config.yaml does not match protocol_manifest.json. "
            "Revert the edits or create a new eval_id (re-run _1_setup.py)."
        )


def load_final_case(paths: BenchmarkPaths) -> pd.DataFrame:
    p = paths.cohort_dir / "final_case_manifest.csv"
    if not p.exists():
        raise FileNotFoundError(p)
    df = pd.read_csv(p, dtype=str, keep_default_na=False)
    require_columns(df, ["case_id", "case_uuid", "project_id", "final_case_eligible"], "final_case_manifest")
    df = df.loc[df["final_case_eligible"].map(truthy)].copy()
    if df.empty:
        raise RuntimeError("No eligible cases in final_case_manifest.csv")
    return df


def load_folds(paths: BenchmarkPaths) -> pd.DataFrame:
    p = paths.folds_dir / "fold_assignments.csv"
    if not p.exists():
        raise FileNotFoundError(p)
    df = pd.read_csv(p, dtype=str, keep_default_na=False)
    require_columns(df, ["case_id", "case_uuid", "project_id", "fold_id", "benchmark_id"], "fold_assignments")
    if not df["benchmark_id"].astype(str).eq(paths.config.get("benchmark_id", df["benchmark_id"].iloc[0])).all():
        # Not enforced here (benchmark_id is checked elsewhere)
        pass
    return df


def load_signature_eligibility(paths: BenchmarkPaths) -> pd.DataFrame:
    p = paths.rna_dir / "signature_eligibility.csv"
    if not p.exists():
        raise FileNotFoundError(p)
    df = pd.read_csv(p, dtype=str, keep_default_na=False)
    require_columns(df, ["project_id", "signature_name", "eligible", "domain_slice", "support_category"], "signature_eligibility")
    df = df.loc[df["eligible"].map(truthy)].copy()
    if df.empty:
        raise RuntimeError("No eligible project × signature cells in signature_eligibility.csv")
    return df.sort_values(["project_id", "signature_name"]).reset_index(drop=True)


def load_case_tile_diagnostics(paths: BenchmarkPaths) -> pd.DataFrame:
    p = paths.tiles_dir / "case_tile_diagnostics.parquet"
    if not p.exists():
        return pd.DataFrame(columns=["case_id", "case_uuid", "project_id"])
    df = pd.read_parquet(p)
    require_columns(df, ["case_id", "case_uuid", "project_id"], "case_tile_diagnostics")
    return df


def load_case_slide_contributions(paths: BenchmarkPaths) -> pd.DataFrame:
    p = paths.tiles_dir / "case_slide_contributions.parquet"
    if not p.exists():
        raise FileNotFoundError(p)
    df = pd.read_parquet(p)
    require_columns(df, ["case_id", "case_uuid", "project_id", "slide_file_uuid", "n_tiles_used_after_cap_slide"], "case_slide_contributions")
    return df


def load_target_table(paths: BenchmarkPaths, project_id: str) -> Tuple[pd.DataFrame, Path, str, Dict[str, Any]]:
    p = paths.target_tables_dir / f"{project_id}.parquet"
    if not p.exists():
        raise FileNotFoundError(p)
    df = pd.read_parquet(p)
    require_columns(df, ["case_id", "case_uuid", "project_id"], f"target_table[{project_id}]")
    manifest_path = p.with_suffix(".manifest.json")
    manifest = read_json(manifest_path) if manifest_path.exists() else {}
    return df, p, sha256_path(p), manifest


def target_hash_map(paths: BenchmarkPaths) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for p in sorted(paths.target_tables_dir.glob("*.parquet")):
        if p.name.startswith("."):
            continue  # skip macOS ._* files
        out[p.stem] = sha256_path(p)
    return out


def merge_case_metadata(final_case: pd.DataFrame, diagnostics: pd.DataFrame) -> pd.DataFrame:
    meta_cols = [
        "case_id", "case_uuid", "project_id",
        "n_tiles_eligible_before_cap", "n_tiles_used_after_cap", "n_slides_retained",
        "match_tier_best_case", "biospecimen_match_level_best_case", "biospecimen_match_level", "official_primary_match",
        "unknown_stain", "unknown_ffpe",
    ]
    meta = final_case.copy()
    if "biospecimen_match_level" not in meta.columns and "biospecimen_match_level_best_case" in meta.columns:
        meta["biospecimen_match_level"] = meta["biospecimen_match_level_best_case"]
    for c in meta_cols:
        if c not in meta.columns:
            meta[c] = pd.NA
    diag_cols = [
        "case_id", "dominant_slide_tile_fraction_after_cap", "slide_tile_entropy_after_cap",
        "slide_tile_effective_n_after_cap", "single_slide_case", "multi_slide_case", "cap_applied",
    ]
    diag = diagnostics.copy()
    for c in diag_cols:
        if c not in diag.columns:
            diag[c] = pd.NA
    out = meta.loc[:, [c for c in meta_cols if c in meta.columns]].merge(
        diag.loc[:, diag_cols].drop_duplicates("case_id"), on="case_id", how="left", validate="one_to_one"
    )
    return out


# Pooling -> ordered tile statistics concatenated from the H5 (datasets {stat}_emb_{space})
POOLING_STAT_ORDER: Dict[str, Tuple[str, ...]] = {
    "mean": ("mean",),
    "mean_std": ("mean", "std"),
    "mean_std_max": ("mean", "std", "max"),
}


def pooling_components(pooling: str, space: str) -> Tuple[str, ...]:
    """Space-suffixed pooled datasets read for a (pooling, space) selection."""
    stats = POOLING_STAT_ORDER.get(pooling)
    if stats is None:
        raise RuntimeError(
            f"Unknown pooling {pooling!r}; expected one of {sorted(POOLING_STAT_ORDER)}"
        )
    return tuple(f"{stat}_emb_{space}" for stat in stats)


def read_h5_project_features(path: Path, *, pooling: str, space: str) -> pd.DataFrame:
    try:
        import h5py  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("h5py is required to read encoder feature H5 files") from e
    components = pooling_components(pooling, space)
    with h5py.File(path, "r") as f:
        available_spaces = str(f.attrs.get("spaces", ""))
        missing = [c for c in components if c not in f]
        if missing:
            raise RuntimeError(
                f"H5 {path} is missing pooling datasets {missing} for "
                f"pooling={pooling!r}, space={space!r} (this run stored spaces="
                f"{available_spaces!r}). A raw pretrained encoder (checkpoint -1) "
                f"stores he only; otherwise re-run _0_extract_encoder_features.py "
                "to regenerate this feature file."
            )
        arrs = [np.asarray(f[c], dtype=np.float32) for c in components]
        # Check the tile feature dim across the pooling components and against feature_dim_tile_{space}
        widths = [int(a.shape[1]) for a in arrs]
        if len(set(widths)) != 1:
            raise RuntimeError(
                f"H5 {path} pooling components disagree on tile feature dim: "
                f"{dict(zip(components, widths))} (pooling={pooling!r}, space={space!r}); "
                "regenerate via _0_extract_encoder_features.py"
            )
        stored_dim = f.attrs.get(f"feature_dim_tile_{space}")
        if stored_dim is not None and int(stored_dim) != widths[0]:
            raise RuntimeError(
                f"H5 {path} declares feature_dim_tile_{space}={int(stored_dim)} but its "
                f"pooling components {list(components)} have tile feature dim {widths[0]} "
                f"(pooling={pooling!r}); regenerate via _0_extract_encoder_features.py"
            )
        case_ids = [v.decode("utf-8") if isinstance(v, bytes) else str(v) for v in np.asarray(f["case_ids"])]
        case_uuids = [v.decode("utf-8") if isinstance(v, bytes) else str(v) for v in np.asarray(f["case_uuids"])]
        project_raw = f["project_id"][()]
        project_id = project_raw.decode("utf-8") if isinstance(project_raw, bytes) else str(project_raw)
    x = np.concatenate(arrs, axis=1)
    feat_cols = [f"feat_{i:06d}" for i in range(int(x.shape[1]))]
    df = pd.DataFrame(x, columns=feat_cols)
    df.insert(0, "project_id", project_id)
    df.insert(0, "case_uuid", case_uuids)
    df.insert(0, "case_id", case_ids)
    return df


def load_encoder_feature_bundle(run_root: Path, *, pooling: str, space: str) -> FeatureBundle:
    run_root = Path(run_root)
    features_dir = run_root / "features"
    index_path = features_dir / "case_features_index.parquet"
    manifest_path = run_root / "run_manifest.json"
    if not index_path.exists():
        raise FileNotFoundError(index_path)
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    manifest = read_json(manifest_path)
    manifest_spaces = manifest.get("spaces")
    if isinstance(manifest_spaces, (list, tuple)) and str(space) not in [str(s) for s in manifest_spaces]:
        raise RuntimeError(
            f"Feature run {run_root} does not contain space={space!r}; it stored "
            f"spaces={list(manifest_spaces)} (a raw pretrained encoder stores 'he' only)."
        )
    # Tile dim declared by the run manifest, checked against each H5 below
    dim_map = manifest.get("feature_dim_tile_by_space")
    manifest_tile_dim = (
        int(dim_map[space]) if isinstance(dim_map, Mapping) and space in dim_map else None
    )
    n_pooling_components = len(pooling_components(pooling, space))
    idx = pd.read_parquet(index_path)
    require_columns(idx, ["project_id", "feature_file", "case_id", "row_index"], "case_features_index")
    frames: List[pd.DataFrame] = []
    hash_by_project: Dict[str, str] = {}
    for project_id, g in idx.groupby("project_id", sort=True):
        files = sorted(set(map(str, g["feature_file"].tolist())))
        if len(files) != 1:
            raise RuntimeError(f"Expected one feature_file for project {project_id}; found {files}")
        p = Path(files[0])
        if not p.is_absolute():
            p = (features_dir / p).resolve()
        if not p.exists():
            # Fallback: <project_id>.h5 in features_dir
            alt = features_dir / f"{project_id}.h5"
            if alt.exists():
                p = alt
            else:
                raise FileNotFoundError(p)
        df = read_h5_project_features(p, pooling=pooling, space=space)
        if manifest_tile_dim is not None:
            n_feat = sum(str(c).startswith("feat_") for c in df.columns)
            expected_feat = manifest_tile_dim * n_pooling_components
            if n_feat != expected_feat:
                raise RuntimeError(
                    f"Manifest<->H5 tile dim drift for project {project_id} in {run_root}: "
                    f"run_manifest feature_dim_tile_by_space[{space!r}]={manifest_tile_dim} implies "
                    f"{expected_feat} feature columns (pooling={pooling!r}, "
                    f"{n_pooling_components} components) but {p.name} yields {n_feat}; "
                    "regenerate via _0_extract_encoder_features.py"
                )

        expected = g.sort_values("row_index").reset_index(drop=True)
        id_cols = ["case_id"]
        if "case_uuid" in expected.columns and "case_uuid" in df.columns:
            id_cols.append("case_uuid")

        if len(df) != len(expected):
            raise RuntimeError(
                f"Feature/index row-count mismatch for project {project_id}: "
                f"H5 has {len(df)} rows, index has {len(expected)} rows"
            )

        observed_keys = df[id_cols].astype(str).to_dict("records")
        expected_keys = expected[id_cols].astype(str).to_dict("records")
        if observed_keys != expected_keys:
            mismatches = [
                {"row": i, "h5": observed_keys[i], "index": expected_keys[i]}
                for i in range(min(len(observed_keys), len(expected_keys)))
                if observed_keys[i] != expected_keys[i]
            ][:10]
            raise RuntimeError(
                f"Feature/index row-order mismatch for project {project_id}; "
                f"first mismatches: {mismatches}"
            )

        frames.append(df)
        hash_by_project[str(project_id)] = sha256_path(p)
    feat = pd.concat(frames, ignore_index=True)
    feature_cols = [c for c in feat.columns if c.startswith("feat_")]
    if not feature_cols:
        raise RuntimeError(f"No feat_* columns were loaded from {features_dir}")
    # Per-space post-processing from the run manifest
    postproc_map = manifest.get("tile_feature_postproc_by_space")
    meta = {
        "source_type": "encoder_case_features_h5",
        "run_root": str(run_root),
        "features_dir": str(features_dir),
        "case_features_index_path": str(index_path),
        "case_features_index_hash": sha256_path(index_path),
        "run_manifest_path": str(manifest_path),
        "run_manifest_hash": sha256_path(manifest_path),
        "pooling": str(pooling),
        "pooling_components": list(pooling_components(pooling, space)),
        "space": str(space),
        "spaces": list(manifest_spaces) if isinstance(manifest_spaces, (list, tuple)) else None,
        "tile_feature_postproc": (postproc_map.get(space) if isinstance(postproc_map, Mapping) else None),
        "feature_dim_tile": manifest_tile_dim,
        "feature_dim_case": int(len(feature_cols)),
        **{k: manifest.get(k) for k in [
            "encoder_id", "benchmark_id", "modelgroup_id", "training_num", "checkpoint_num",
            "foundation_baseline_only", "contains_he_mlp_head", "contains_lora", "bundle_sha256",
        ]},
    }
    return FeatureBundle(
        features=feat,
        feature_columns=feature_cols,
        numeric_columns=feature_cols,
        categorical_columns=[],
        metadata=meta,
        feature_hash_by_project=hash_by_project,
        feature_table_hash=sha256_path(index_path),
    )


def write_feature_bundle_parquet(bundle: FeatureBundle, out_dir: Path, *, overwrite: bool) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "case_features.parquet"
    write_parquet_atomic(p, bundle.features, overwrite=overwrite)
    return p


def make_feature_manifest(bundle: FeatureBundle, feature_path: Path, *, extra: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    obj: Dict[str, Any] = {
        "created_at_utc": utc_now_iso(),
        "feature_path": str(feature_path),
        "feature_table_hash": sha256_path(feature_path),
        "n_cases": int(bundle.features["case_id"].nunique()) if "case_id" in bundle.features.columns else int(len(bundle.features)),
        "n_rows": int(len(bundle.features)),
        "n_feature_columns": int(len(bundle.feature_columns)),
        "numeric_columns": list(bundle.numeric_columns),
        "categorical_columns": list(bundle.categorical_columns),
        "feature_hash_by_project": dict(bundle.feature_hash_by_project),
        "metadata": dict(bundle.metadata),
    }
    if extra:
        obj.update(dict(extra))
    return obj


def json_dump_compact(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
