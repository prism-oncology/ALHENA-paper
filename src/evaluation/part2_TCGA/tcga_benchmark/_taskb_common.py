#!/usr/bin/env python3
"""
Shared utilities for TCGA Task B benchmark construction.
"""

import csv
import dataclasses
import datetime as _dt
import errno
import hashlib
import importlib
import inspect
import io
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, MutableMapping, Optional, Sequence, Tuple
import numbers
import numpy as np
import pandas as pd
import requests
import yaml


DEFAULT_GDC_API_BASE = "https://api.gdc.cancer.gov"
UTC = _dt.timezone.utc


# -----------------------------------------------------------------------------
# Basic serialization / hashing
# -----------------------------------------------------------------------------

def utc_now_iso() -> str:
    return _dt.datetime.now(tz=UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def stable_json_dumps(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def sha256_bytes(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_obj(obj: Any) -> str:
    return sha256_text(stable_json_dumps(obj))


def json_sanitize(obj: Any) -> Any:
    """Convert objects to strict JSON-safe values (NaN/Infinity -> None)."""
    if obj is None or isinstance(obj, (str, bool)):
        return obj
    if isinstance(obj, numbers.Integral) and not isinstance(obj, bool):
        return int(obj)
    if isinstance(obj, numbers.Real) and not isinstance(obj, bool):
        v = float(obj)
        return v if math.isfinite(v) else None
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, Mapping):
        return {str(k): json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [json_sanitize(v) for v in obj]
    try:
        if pd.isna(obj):
            return None
    except Exception:
        pass
    return obj


def sha256_path(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def md5_path(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.md5()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def hashes_path(path: Path, chunk_size: int = 1024 * 1024) -> Tuple[str, str]:
    """Return (md5_hex, sha256_hex) computed in a single pass over the file."""
    h_md5 = hashlib.md5()
    h_sha = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h_md5.update(chunk)
            h_sha.update(chunk)
    return h_md5.hexdigest(), h_sha.hexdigest()


def file_info(path: Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_path(path),
        "mtime_utc": _dt.datetime.fromtimestamp(path.stat().st_mtime, tz=UTC).isoformat().replace("+00:00", "Z"),
    }


def read_json(path: Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def read_yaml(path: Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_bytes_atomic(path: Path, data: bytes, *, overwrite: bool = False, validate: Optional[Callable[[Path], None]] = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing frozen file without --force: {path}")
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        if validate is not None:
            validate(tmp)
        os.replace(tmp, path)
        try:
            dirfd = os.open(str(path.parent), os.O_DIRECTORY)
            try:
                os.fsync(dirfd)
            finally:
                os.close(dirfd)
        except Exception:
            pass
    except Exception:
        try:
            tmp.unlink(missing_ok=True)
        finally:
            raise


def write_text_atomic(path: Path, text: str, *, overwrite: bool = False, validate: Optional[Callable[[Path], None]] = None) -> None:
    write_bytes_atomic(path, text.encode("utf-8"), overwrite=overwrite, validate=validate)


def write_json_atomic(path: Path, obj: Any, *, overwrite: bool = False) -> None:
    text = json.dumps(json_sanitize(obj), indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False, default=str) + "\n"
    write_text_atomic(path, text, overwrite=overwrite, validate=lambda p: read_json(p))


def write_yaml_atomic(path: Path, obj: Any, *, overwrite: bool = False) -> None:
    text = yaml.safe_dump(json_sanitize(obj), sort_keys=False, allow_unicode=True)
    write_text_atomic(path, text, overwrite=overwrite, validate=lambda p: read_yaml(p))


def write_csv_atomic(path: Path, df: pd.DataFrame, *, overwrite: bool = False, index: bool = False) -> None:
    def validate(p: Path) -> None:
        pd.read_csv(p, nrows=5)
    buf = io.StringIO()
    df.to_csv(buf, index=index)
    write_text_atomic(path, buf.getvalue(), overwrite=overwrite, validate=validate)


def write_tsv_atomic(path: Path, df: pd.DataFrame, *, overwrite: bool = False, index: bool = False) -> None:
    def validate(p: Path) -> None:
        pd.read_csv(p, sep="\t", nrows=5)
    buf = io.StringIO()
    df.to_csv(buf, sep="\t", index=index)
    write_text_atomic(path, buf.getvalue(), overwrite=overwrite, validate=validate)


def write_parquet_atomic(path: Path, df: pd.DataFrame, *, overwrite: bool = False) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing frozen file without --force: {path}")
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        df.to_parquet(tmp, index=False)
        # Validate round trip.
        pd.read_parquet(tmp).head(1)
        with tmp.open("rb+") as f:
            os.fsync(f.fileno())
        os.replace(tmp, path)
        try:
            dir_fd = os.open(str(path.parent), os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except Exception:
            pass
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def _truthy_value(x: Any) -> bool:
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


def version_ok_uuids(
    version_df: pd.DataFrame,
    *,
    allow_low_confidence_fallback: bool = False,
    require_latest: bool = False,
) -> set[str]:
    """Return file UUIDs with acceptable frozen GDC file-version provenance."""
    required = ["queried_uuid", "selected_file_is_released_open", "match_kind"]
    missing = [c for c in required if c not in version_df.columns]
    if missing:
        raise RuntimeError(f"file_versions.parquet missing required provenance columns: {missing}")

    allowed_match_kinds = {"id_field_match"}
    if allow_low_confidence_fallback:
        allowed_match_kinds.add("json_substring_unique_fallback")

    mask = version_df["selected_file_is_released_open"].map(_truthy_value)
    mask &= version_df["match_kind"].astype(str).isin(allowed_match_kinds)

    if require_latest:
        if "selected_uuid_is_latest" not in version_df.columns:
            raise RuntimeError("require_latest=True but file_versions.parquet lacks selected_uuid_is_latest")
        mask &= version_df["selected_uuid_is_latest"].map(_truthy_value)

    return set(version_df.loc[mask, "queried_uuid"].dropna().astype(str))


def copy_or_hardlink(src: Path, dst: Path, *, overwrite: bool = False) -> str:
    src = Path(src)
    dst = Path(dst)
    if not src.exists():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and not overwrite:
        if sha256_path(src) == sha256_path(dst):
            return "already_present_same_hash"
        raise FileExistsError(f"Destination exists with different content: {dst}")
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


# -----------------------------------------------------------------------------
# Shared, content-addressed slide store (benchmark-agnostic)
# -----------------------------------------------------------------------------
# WSIs and prefetch results are stored once under selected_slide_cache/_store/, keyed by slide_file_uuid,
# and shared by all benchmark_ids.

SHARED_SLIDE_STORE_DIRNAME = "_store"


def shared_store_root(data_root: Path) -> Path:
    """Root of the benchmark-agnostic, content-addressed slide store."""
    return Path(data_root) / "selected_slide_cache" / SHARED_SLIDE_STORE_DIRNAME


def shared_slide_file_path(data_root: Path, uuid: str, file_name: str) -> Path:
    """Path of a full WSI in the shared store: _store/slide_files/by_uuid/{uuid}/{file_name}."""
    return shared_store_root(data_root) / "slide_files" / "by_uuid" / str(uuid) / file_name


def shared_prefetch_header_path(data_root: Path, uuid: str) -> Path:
    """Path of a slide's cached prefetch header bytes: _store/prefetch/by_uuid/{uuid}/header_bytes.bin."""
    return shared_store_root(data_root) / "prefetch" / "by_uuid" / str(uuid) / "header_bytes.bin"


def shared_prefetch_metadata_path(data_root: Path, uuid: str) -> Path:
    """Path of a slide's cached technical-metadata result: _store/prefetch/by_uuid/{uuid}/technical_metadata.json."""
    return shared_store_root(data_root) / "prefetch" / "by_uuid" / str(uuid) / "technical_metadata.json"


# -----------------------------------------------------------------------------
# Shared, content-addressed HistoQC mask store (benchmark-agnostic)
# -----------------------------------------------------------------------------
# HistoQC outputs are stored once per slide under processed_assets/histoqc/_store/, keyed by slide uuid and run key
# (config hash, version); each benchmark keeps a manifest under processed_assets/histoqc/<histoqc_asset_id>/.

SHARED_HISTOQC_STORE_DIRNAME = "_store"

# Candidate filenames of the HistoQC tissue-use mask, in priority order (they differ across HistoQC versions)
HISTOQC_MASK_USE_CANDIDATES = (
    "mask_use.png",
    "{stem}_mask_use.png",
    "{stem}.svs_mask_use.png",
)


def histoqc_run_key(histoqc_config_hash: str, histoqc_version: str) -> str:
    """Content key of one HistoQC run config: a hash over (config_hash, version)."""
    return sha256_obj({
        "histoqc_config_hash": str(histoqc_config_hash),
        "histoqc_version": str(histoqc_version),
    })[:16]


def shared_histoqc_store_root(data_root: Path) -> Path:
    """Root of the benchmark-agnostic, content-addressed HistoQC mask store."""
    return Path(data_root) / "processed_assets" / "histoqc" / SHARED_HISTOQC_STORE_DIRNAME


def shared_histoqc_slide_dir(data_root: Path, uuid: str, run_key: str) -> Path:
    """Per-slide HistoQC output dir in the shared store: _store/by_uuid/{uuid}/{run_key}."""
    return shared_histoqc_store_root(data_root) / "by_uuid" / str(uuid) / str(run_key)


def resolve_histoqc_mask_use(slide_outdir: Path, slide_stem: str) -> Optional[Path]:
    """Path of HistoQC's tissue-use mask in a per-slide output dir (first known name, else any *mask_use*.png), or None."""
    slide_outdir = Path(slide_outdir)
    if not slide_outdir.exists():
        return None
    for pat in HISTOQC_MASK_USE_CANDIDATES:
        candidate = slide_outdir / pat.format(stem=slide_stem)
        if candidate.exists():
            return candidate
    # Skip leading-dot names (macOS ._* AppleDouble files)
    matches = sorted(
        p for p in slide_outdir.glob("*mask_use*.png") if not p.name.startswith(".")
    )
    if matches:
        return matches[0]
    # Fallback: recursive search (HistoQC writes OUTDIR/<filename>/<filename>_mask_use.png)
    rmatches = sorted(
        p for p in slide_outdir.rglob("*mask_use*.png") if not p.name.startswith(".")
    )
    return rmatches[0] if rmatches else None


# -----------------------------------------------------------------------------
# Shared, content-addressed tile store (benchmark-agnostic), two-tier
# -----------------------------------------------------------------------------
# Per-slide tile sets (manifest.parquet, slide_qc.json) and their pixels (tiles.h5) are stored once under
# processed_assets/tiles/_store/, keyed by slide uuid, set_recipe_id and codec_recipe_id.

SHARED_TILES_STORE_DIRNAME = "_store"


def tile_set_recipe_id(*, modelgroup_yaml_hash: str, qc_config_hash: str, histoqc_run_key: str) -> str:
    """Content key of one tile-set recipe: a hash over (model-group YAML hash, QC config hash, HistoQC run key)."""
    return sha256_obj({
        "modelgroup_yaml_hash": str(modelgroup_yaml_hash),
        "qc_config_hash": str(qc_config_hash),
        "histoqc_run_key": str(histoqc_run_key),
    })[:16]


def tile_codec_recipe_id(*, set_recipe_id: str, codec_recipe: Mapping[str, Any]) -> str:
    """Content key of one tile-pixels recipe: a hash over (set_recipe_id, codec recipe)."""
    return sha256_obj({
        "set_recipe_id": str(set_recipe_id),
        "codec_recipe": {
            "format": str(codec_recipe.get("format")),
            "quality": codec_recipe.get("quality"),
            "chroma": codec_recipe.get("chroma"),
            "png_level": codec_recipe.get("png_level"),
        },
    })[:16]


def shared_tiles_store_root(data_root: Path) -> Path:
    """Root of the benchmark-agnostic, content-addressed tile store."""
    return Path(data_root) / "processed_assets" / "tiles" / SHARED_TILES_STORE_DIRNAME


def shared_tile_set_dir(data_root: Path, uuid: str, set_recipe_id: str) -> Path:
    """Per-slide tile-set dir in the shared store: _store/by_uuid/{uuid}/{set_recipe_id}."""
    return shared_tiles_store_root(data_root) / "by_uuid" / str(uuid) / str(set_recipe_id)


def shared_tile_pixels_dir(data_root: Path, uuid: str, set_recipe_id: str, codec_recipe_id: str) -> Path:
    """Per-slide tile-pixels dir: _store/by_uuid/{uuid}/{set_recipe_id}/pixels/{codec_recipe_id}."""
    return shared_tile_set_dir(data_root, uuid, set_recipe_id) / "pixels" / str(codec_recipe_id)


def shared_tile_h5_path(data_root: Path, uuid: str, set_recipe_id: str, codec_recipe_id: str) -> Path:
    """Path of a slide's tiles.h5 in the shared tile store, rebuilt from its ids under the current data root."""
    return shared_tile_pixels_dir(data_root, uuid, set_recipe_id, codec_recipe_id) / "tiles.h5"


def shared_tile_manifest_path(data_root: Path, uuid: str, set_recipe_id: str) -> Path:
    """Path of a slide's tile-set manifest.parquet in the shared tile store."""
    return shared_tile_set_dir(data_root, uuid, set_recipe_id) / "manifest.parquet"


def shared_wsi_path(data_root: Path, uuid: str, file_name: str) -> Path:
    """Path of a slide's WSI in the shared slide store, rebuilt from uuid and file name under the current data root."""
    return shared_store_root(data_root) / "slide_files" / "by_uuid" / str(uuid) / str(file_name)


def reroot_under_data_root(data_root: Path, stored_path: Optional[str]) -> Optional[Path]:
    """Re-root a stored absolute path under the current data_root (split on the last data_root.name component).
    Returns None if stored_path is empty or does not contain data_root.name."""
    if not stored_path:
        return None
    data_root = Path(data_root)
    anchor = data_root.name
    parts = Path(str(stored_path)).parts
    if anchor not in parts:
        return None
    idx = len(parts) - 1 - parts[::-1].index(anchor)
    return data_root.joinpath(*parts[idx + 1:])


def verify_blob_sha256(path: Path, expected_sha256: Optional[str], *, kind: str = "blob") -> None:
    """Raise RuntimeError if the sha256 of path differs from expected_sha256 (no-op if expected_sha256 is empty)."""
    if not expected_sha256:
        return
    actual = sha256_path(path)
    if actual != str(expected_sha256):
        raise RuntimeError(f"sha256 mismatch for {kind} at {path}: expected {expected_sha256}, got {actual}")


def shared_tile_set_is_complete(set_dir: Path, *, expected_set_recipe_id: str) -> bool:
    """True if the shared tile-set dir is complete: record status ok, matching set_recipe_id and manifest sha256."""
    set_dir = Path(set_dir)
    record = set_dir / "tile_set_record.json"
    manifest = set_dir / "manifest.parquet"
    slide_qc = set_dir / "slide_qc.json"
    if not (record.exists() and manifest.exists() and slide_qc.exists()):
        return False
    try:
        rec = read_json(record)
    except Exception:
        return False
    if not isinstance(rec, Mapping) or str(rec.get("status") or "") != "ok":
        return False
    if str(rec.get("set_recipe_id") or "") != str(expected_set_recipe_id):
        return False
    recorded = str(rec.get("manifest_sha256") or "")
    return bool(recorded) and recorded == sha256_path(manifest)


def shared_tile_pixels_is_complete(
    pixels_dir: Path,
    *,
    expected_set_recipe_id: str,
    expected_codec_recipe_id: str,
    verify_sha: bool = True,
) -> bool:
    """True if the shared tile-pixels dir is complete: record status ok, matching recipe ids and tiles.h5 size.
    With verify_sha, the tiles.h5 sha256 is also checked (always checked if the record has no h5_bytes).
    """
    pixels_dir = Path(pixels_dir)
    record = pixels_dir / "tile_pixels_record.json"
    h5 = pixels_dir / "tiles.h5"
    if not (record.exists() and h5.exists()):
        return False
    try:
        rec = read_json(record)
    except Exception:
        return False
    if not isinstance(rec, Mapping) or str(rec.get("status") or "") != "ok":
        return False
    if str(rec.get("set_recipe_id") or "") != str(expected_set_recipe_id):
        return False
    if str(rec.get("codec_recipe_id") or "") != str(expected_codec_recipe_id):
        return False
    recorded = str(rec.get("h5_sha256") or "")
    if not recorded:
        return False
    recorded_bytes = rec.get("h5_bytes")
    if recorded_bytes is None:
        # No recorded size: check the full sha256
        return recorded == sha256_path(h5)
    try:
        on_disk_bytes = h5.stat().st_size
    except OSError:
        return False
    if int(recorded_bytes) != int(on_disk_bytes):
        return False
    if not verify_sha:
        return True
    return recorded == sha256_path(h5)


def imaging_versions() -> Dict[str, str]:
    """Imaging toolchain versions (openslide, opencv, pillow), recorded in the tile records (not part of any recipe id)."""
    out: Dict[str, str] = {}
    for import_name, label in (("openslide", "openslide-python"), ("cv2", "opencv"), ("PIL", "Pillow")):
        try:
            mod = importlib.import_module(import_name)
            out[label] = str(getattr(mod, "__version__", "unknown"))
        except Exception:
            out[label] = "not_importable"
    return out


# -----------------------------------------------------------------------------
# Frozen output gates / manifests
# -----------------------------------------------------------------------------

def ensure_empty_or_force(path: Path, *, force: bool, force_cascade: bool = False, downstream_paths: Sequence[Path] = ()) -> None:
    path = Path(path)
    if not path.exists():
        return
    if not force:
        raise FileExistsError(f"Output folder already exists; use --force to rebuild: {path}")
    existing_frozen = [p for p in downstream_paths if p.exists()]
    if existing_frozen and not force_cascade:
        msg = "\n".join(str(p) for p in existing_frozen[:20])
        raise RuntimeError(
            "Refusing --force because downstream frozen files exist. "
            "Use --force_cascade only after intentionally invalidating downstream artifacts.\n" + msg
        )


def manifest_header(script_name: str, args: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "created_at_utc": utc_now_iso(),
        "script_name": script_name,
        "cli_args": dict(sorted(args.items())),
        "python": sys.version,
        "platform": sys.platform,
        "software_versions": software_versions(),
    }


def software_versions() -> Dict[str, str]:
    versions = {"python": sys.version.split()[0]}
    for name in ["pandas", "pyarrow", "requests", "yaml", "numpy"]:
        try:
            mod_name = "PyYAML" if name == "yaml" else name
            mod = importlib.import_module(name if name != "yaml" else "yaml")
            versions[mod_name] = getattr(mod, "__version__", "unknown")
        except Exception:
            versions[name] = "not_importable"
    return versions


def collect_output_hashes(root: Path) -> Dict[str, Dict[str, Any]]:
    root = Path(root)
    out: Dict[str, Dict[str, Any]] = {}
    if not root.exists():
        return out
    for p in sorted(root.rglob("*")):
        # skip hidden files (macOS ._* files) and atomic-write temps
        if p.is_file() and ".tmp" not in p.name and not p.name.startswith("."):
            rel = str(p.relative_to(root))
            out[rel] = file_info(p)
    return out


def parse_simple_version(value: Any) -> Tuple[int, ...]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return tuple()
    text = str(value)
    nums = re.findall(r"\d+", text)
    return tuple(int(x) for x in nums)


# -----------------------------------------------------------------------------
# SLURM-array shard-status aggregation
#
# Each shard of _9_run_histoqc.py / _15_build_tile_rgb_cache.py writes <status_root>/shard_status/shard_XXXX_of_YYYY.json,
# read back by read_shard_records in the --finalize_only pass.
# -----------------------------------------------------------------------------

def shard_status_path(status_root: Path, *, shard_index: int, total_shards: int) -> Path:
    """Per-shard status-file path: <status_root>/shard_status/shard_XXXX_of_YYYY.json."""
    return Path(status_root) / "shard_status" / f"shard_{shard_index:04d}_of_{total_shards:04d}.json"


def read_shard_records(status_root: Path, total_shards: int) -> Tuple[List[Dict[str, Any]], List[int]]:
    """Aggregate the shard-status files under status_root/shard_status/ (only those with this total_shards).

    Returns (per_slide_rows, completed_shard_indices).
    """
    rows: List[Dict[str, Any]] = []
    completed: List[int] = []
    sd = Path(status_root) / "shard_status"
    if not sd.exists():
        return rows, completed
    for p in sorted(sd.glob("shard_*.json")):
        try:
            obj = read_json(p)
        except Exception:
            continue
        if not isinstance(obj, Mapping):
            continue
        ts_raw = obj.get("total_shards")
        si_raw = obj.get("shard_index")
        if ts_raw is None or si_raw is None:
            continue
        ts = int(ts_raw)
        si = int(si_raw)
        if ts != int(total_shards) or si < 0:
            continue
        completed.append(si)
        for r in obj.get("per_slide_records") or []:
            if isinstance(r, Mapping):
                rows.append(dict(r))
    return rows, sorted(set(completed))


# -----------------------------------------------------------------------------
# Schema helpers
# -----------------------------------------------------------------------------

def require_columns(df: pd.DataFrame, cols: Sequence[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"{name} is missing required columns: {missing}")


def keep_ordered_columns(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    for c in cols:
        if c not in df.columns:
            df[c] = pd.NA
    return df.loc[:, list(cols)]


def jsonish(value: Any) -> str:
    return stable_json_dumps(value)


def listify(x: Any) -> List[Any]:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return [x]


def first_or_none(xs: Sequence[Any]) -> Any:
    return xs[0] if xs else None


def nested_get(obj: Mapping[str, Any], dotted: str, default: Any = None) -> Any:
    cur: Any = obj
    for part in dotted.split("."):
        if isinstance(cur, Mapping) and part in cur:
            cur = cur[part]
        else:
            return default
    return cur


def flat_value(hit: Mapping[str, Any], key: str) -> Any:
    # Accepts raw keys and dotted keys (with or without the "files." prefix)
    if key in hit:
        return hit[key]
    parts = key.split(".")
    if parts and parts[0] in {"files", "file"}:
        parts = parts[1:]
    cur: Any = hit
    for part in parts:
        if isinstance(cur, Mapping) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


# -----------------------------------------------------------------------------
# TCGA / image-domain helpers (shared across the tile / encoder / validate paths)
# -----------------------------------------------------------------------------

def tissue_source_site(case_id: Any) -> str:
    """TCGA tissue-source-site code from a barcode (TCGA-<TSS>-<participant>-...), or "<UNK>"."""
    parts = str(case_id).split("-")
    return parts[1] if len(parts) > 1 and parts[1].strip() else "<UNK>"


def stable_project_seed(global_seed: int, project_id: str, *, namespace: str = "fold_seed") -> int:
    """Deterministic per-project RNG seed (hash of namespace, global_seed and project_id)."""
    h = sha256_obj({namespace: int(global_seed), "project_id": str(project_id)})[:8]
    return (int(global_seed) + int(h, 16)) % (2**32 - 1)


def case_cap_order(g: pd.DataFrame, *, method: str, rng: "np.random.Generator") -> List[int]:
    """Positional indices of g (a full permutation) in case-cap priority order; the caller keeps the first cap.

    "tss_stratified_round_robin" takes one case per tissue-source site per pass (seeded site and case order);
    "random" is a seeded permutation.
    """
    n = int(len(g))
    if method == "random":
        return [int(i) for i in rng.permutation(n)]
    if method != "tss_stratified_round_robin":
        raise RuntimeError(f"Unknown case_cap_method={method!r}")
    case_ids = g["case_id"].astype(str).tolist()
    groups: Dict[str, List[int]] = {}
    for i, cid in enumerate(case_ids):
        groups.setdefault(tissue_source_site(cid), []).append(i)
    # Shuffle cases within each site, and the site visiting order, deterministically.
    for tss, idxs in groups.items():
        perm = rng.permutation(len(idxs))
        groups[tss] = [idxs[int(j)] for j in perm]
    site_names = list(groups.keys())
    visit = [site_names[int(j)] for j in rng.permutation(len(site_names))]
    order: List[int] = []
    cursors = {tss: 0 for tss in visit}
    while len(order) < n:
        for tss in visit:
            c = cursors[tss]
            if c < len(groups[tss]):
                order.append(groups[tss][c])
                cursors[tss] = c + 1
    return order


# Accepted fold_grouping_used values (written by _13_build_targets_and_folds.py, checked by _14_validate_benchmark.py)
FOLD_GROUPING_USED_VALUES = (
    "tissue_source_site",
    "case_fallback_insufficient_tss",
    "case_fallback_degenerate_dominant_tss",
    "case",
)


def project_tss_distribution(case_ids: Sequence[Any]) -> Dict[str, int]:
    """Map TCGA tissue-source-site -> number of cases, for one project's case ids."""
    counts: Dict[str, int] = {}
    for cid in case_ids:
        tss = tissue_source_site(cid)
        counts[tss] = counts.get(tss, 0) + 1
    return counts


def projected_fold_verdict(
    n_cases: int, tss_counts: Mapping[str, int], *, n_folds: int, degenerate_frac: float
) -> Tuple[str, float]:
    """Fold grouping of a project for TSS-grouped folds, and the largest single-site share of its cases.

    Returns (fold_grouping_used, max_tss_fraction): "case_fallback_insufficient_tss" if fewer than n_folds sites,
    "case_fallback_degenerate_dominant_tss" if one site exceeds degenerate_frac, else "tissue_source_site".
    """
    n_tss = len(tss_counts)
    max_tss_fraction = (max(tss_counts.values()) / n_cases) if (n_cases > 0 and tss_counts) else 0.0
    if n_tss < n_folds:
        grouping = "case_fallback_insufficient_tss"
    elif max_tss_fraction > float(degenerate_frac):
        grouping = "case_fallback_degenerate_dominant_tss"
    else:
        grouping = "tissue_source_site"
    return grouping, float(max_tss_fraction)


def assert_modelgroup_no_upsampling(modelgroup: Mapping[str, Any]) -> float:
    """
    Check that the model group's mpp_max <= nominal_mpp_um_per_px * (1 + max_upsample_tol_frac) (no upsampling).
    Returns max_upsample_tol_frac (default 0.0).
    """
    target = float(modelgroup["nominal_mpp_um_per_px"])
    tol = float(modelgroup.get("max_upsample_tol_frac", 0.0))
    if tol < 0.0:
        raise RuntimeError(f"model group max_upsample_tol_frac must be >= 0, got {tol}")
    mpp_max = float(modelgroup["mpp_max"])
    ceiling = target * (1.0 + tol)
    if mpp_max > ceiling + 1e-9:
        raise RuntimeError(
            f"model group mpp_max ({mpp_max}) exceeds the no-upsampling ceiling "
            f"nominal_mpp_um_per_px*(1+max_upsample_tol_frac)={ceiling:.6f}. Lower mpp_max, "
            "or raise max_upsample_tol_frac."
        )
    return tol


def composite_rgba_on_white(img: Any) -> Any:
    """Composite an RGBA PIL image on a white background and return RGB."""
    from PIL import Image  # lazy import
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    white = Image.new("RGBA", img.size, (255, 255, 255, 255))
    white.alpha_composite(img)
    return white.convert("RGB")


# -----------------------------------------------------------------------------
# GDC client
# -----------------------------------------------------------------------------

@dataclasses.dataclass
class GDCPageSnapshot:
    endpoint: str
    page_from: int
    page_size: int
    sort: str
    response_sha256: str
    pagination: Dict[str, Any]
    warnings: Dict[str, Any]


class GDCClient:
    def __init__(self, base_url: str = DEFAULT_GDC_API_BASE, timeout: int = 120, retries: int = 4, backoff: float = 1.5):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.retries = retries
        self.backoff = backoff
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "HistoST-TCGA-TaskB/1.0"})

    def _url(self, path: str) -> str:
        return f"{self.base_url}/{path.lstrip('/')}"

    def request(self, method: str, path: str, **kwargs: Any) -> requests.Response:
        url = self._url(path)
        last_exc: Optional[BaseException] = None
        for attempt in range(self.retries + 1):
            try:
                resp = self.session.request(method, url, timeout=self.timeout, **kwargs)
                if resp.status_code in {429, 500, 502, 503, 504} and attempt < self.retries:
                    time.sleep(self.backoff ** attempt)
                    continue
                resp.raise_for_status()
                return resp
            except Exception as e:
                last_exc = e
                if attempt >= self.retries:
                    break
                time.sleep(self.backoff ** attempt)
        assert last_exc is not None
        raise last_exc

    def get_json(self, path: str, **kwargs: Any) -> Any:
        resp = self.request("GET", path, **kwargs)
        return resp.json()

    def post_json(self, path: str, payload: Mapping[str, Any]) -> Any:
        resp = self.request("POST", path, json=payload, headers={"Content-Type": "application/json"})
        return resp.json()

    def status(self) -> Any:
        return self.get_json("status")

    def mapping(self, endpoint: str) -> Any:
        return self.get_json(f"{endpoint.strip('/')}/_mapping")

    def search(
        self,
        endpoint: str,
        *,
        filters: Mapping[str, Any],
        fields: Sequence[str],
        sort: str,
        page_size: int = 5000,
        query_snapshot_dir: Optional[Path] = None,
    ) -> Tuple[List[Dict[str, Any]], List[GDCPageSnapshot], Dict[str, Any]]:
        hits: List[Dict[str, Any]] = []
        snapshots: List[GDCPageSnapshot] = []
        start = 0
        endpoint = endpoint.strip("/")
        payload_base = {
            "filters": filters,
            "fields": ",".join(fields),
            "format": "JSON",
            "sort": sort,
        }
        while True:
            payload = dict(payload_base)
            payload.update({"size": page_size, "from": start})
            raw = self.post_json(endpoint, payload)
            raw_hash = sha256_obj(raw)
            data = raw.get("data", {}) if isinstance(raw, Mapping) else {}
            page_hits = listify(data.get("hits", []))
            pagination = data.get("pagination", {}) or {}
            warnings = raw.get("warnings", {}) if isinstance(raw, Mapping) else {}
            snapshots.append(GDCPageSnapshot(endpoint, start, page_size, sort, raw_hash, pagination, warnings))
            if query_snapshot_dir is not None:
                query_snapshot_dir.mkdir(parents=True, exist_ok=True)
                payload_tag = sha256_obj(payload_base)[:12]
                snap_name = f"{endpoint.replace('/', '_')}_{payload_tag}_from{start:012d}_size{page_size}.json"
                write_json_atomic(query_snapshot_dir / snap_name, raw, overwrite=True)
            hits.extend(page_hits)
            total = int(pagination.get("total", len(hits))) if pagination else len(hits)
            count = int(pagination.get("count", len(page_hits))) if pagination else len(page_hits)
            if count <= 0 or len(hits) >= total:
                break
            start += count
        meta = {
            "endpoint": endpoint,
            "payload_base": payload_base,
            "sort": sort,
            "page_size": page_size,
            "n_hits": len(hits),
            "snapshots": [dataclasses.asdict(s) for s in snapshots],
        }
        return hits, snapshots, meta

    def file_versions(self, uuids: Sequence[str], chunk_size: int = 100) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        chunks_meta: List[Dict[str, Any]] = []
        uuids = [u for u in dict.fromkeys(str(u) for u in uuids if pd.notna(u) and str(u))]
        for i in range(0, len(uuids), chunk_size):
            chunk = uuids[i : i + chunk_size]
            endpoint = f"files/versions/{','.join(chunk)}"
            query_time = utc_now_iso()
            raw = self.get_json(endpoint, params={"pretty": "false"})
            raw_hash = sha256_obj(raw)
            chunk_rows = flatten_file_versions_response(raw, chunk, endpoint=endpoint, query_date=query_time, response_hash=raw_hash)
            rows.extend(chunk_rows)

            match_counts: Dict[str, int] = {}
            for r in chunk_rows:
                mk = str(r.get("match_kind") or "missing_match_kind")
                match_counts[mk] = match_counts.get(mk, 0) + 1
            chunks_meta.append({
                "endpoint": endpoint,
                "query_date": query_time,
                "response_sha256": raw_hash,
                "n_uuids": len(chunk),
                "n_rows": len(chunk_rows),
                "match_kind_counts": match_counts,
                "n_json_substring_unique_fallback": int(match_counts.get("json_substring_unique_fallback", 0)),
                "n_no_match_or_ambiguous": int(match_counts.get("no_match_or_ambiguous", 0)),
                "any_low_confidence_file_version_match": bool(
                    match_counts.get("json_substring_unique_fallback", 0) or match_counts.get("no_match_or_ambiguous", 0)
                ),
            })
        return rows, {"chunks": chunks_meta, "n_rows": len(rows), "queried_uuid_count": len(uuids)}

    def file_history(self, uuid: str) -> Tuple[Any, Dict[str, Any]]:
        endpoint = f"history/{uuid}"
        query_time = utc_now_iso()
        raw = self.get_json(endpoint)
        return raw, {"endpoint": endpoint, "query_date": query_time, "response_sha256": sha256_obj(raw)}

    def download_file(self, file_uuid: str, dest: Path, *, expected_md5: Optional[str] = None, overwrite: bool = False) -> Dict[str, Any]:
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        log: Dict[str, Any] = {
            "file_uuid": file_uuid,
            "destination": str(dest),
            "start_time_utc": utc_now_iso(),
            "expected_md5": expected_md5,
            "attempts": [],
        }
        if dest.exists() and expected_md5 and md5_path(dest).lower() == expected_md5.lower():
            log.update({"status": "skipped_existing_md5_ok", "end_time_utc": utc_now_iso(), "md5": expected_md5.lower(), "sha256": sha256_path(dest)})
            return log
        if dest.exists() and not overwrite:
            raise FileExistsError(f"Existing file does not match expected md5 or overwrite disabled: {dest}")
        # Unique temp file (concurrent downloads of the same uuid); MD5 is checked before os.replace
        fd, tmp_name = tempfile.mkstemp(prefix=f".{dest.name}.", suffix=".part", dir=str(dest.parent))
        os.close(fd)
        tmp = Path(tmp_name)
        url_path = f"data/{file_uuid}"
        h_md5 = hashlib.md5()
        h_sha = hashlib.sha256()
        total = 0
        resp = self.request("GET", url_path, stream=True)
        try:
            with tmp.open("wb") as f:
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue
                    total += len(chunk)
                    h_md5.update(chunk)
                    h_sha.update(chunk)
                    f.write(chunk)
                f.flush()
                os.fsync(f.fileno())
            observed_md5 = h_md5.hexdigest()
            if expected_md5 and observed_md5.lower() != expected_md5.lower():
                raise RuntimeError(f"MD5 mismatch for {file_uuid}: expected {expected_md5}, got {observed_md5}")
            os.replace(tmp, dest)
        except Exception:
            tmp.unlink(missing_ok=True)
            raise
        log.update({
            "status": "downloaded_md5_ok" if expected_md5 else "downloaded_no_expected_md5",
            "end_time_utc": utc_now_iso(),
            "bytes": total,
            "md5": observed_md5,
            "sha256": h_sha.hexdigest(),
            "http_status": resp.status_code,
        })
        return log


# -----------------------------------------------------------------------------
# GDC Data Transfer Tool (gdc-client) delegation
# -----------------------------------------------------------------------------
# _8 downloads the slides with gdc-client (parallel, resumable), then re-checks md5 in Python.

GDC_MANIFEST_COLUMNS: Tuple[str, ...] = ("id", "filename", "md5", "size", "state")


def gdc_client_version(bin_name: str = "gdc-client") -> str:
    """Return the installed gdc-client version string, or 'unknown' on failure."""
    for cmd in ([bin_name, "--version"], [bin_name, "version"]):
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            out = (r.stdout or "").strip() or (r.stderr or "").strip()
            if r.returncode == 0 and out:
                return out.splitlines()[0].strip()
        except Exception:
            continue
    return "unknown"


def write_gdc_manifest(path: Path, rows: Sequence[Mapping[str, Any]], *, overwrite: bool = False) -> int:
    """Write a GDC Data Transfer Tool manifest (id, filename, md5, size, state) and return the row count.

    Rows may use the cohort names (slide_file_uuid, file_name, md5sum, file_size); state defaults to 'released'.
    """
    lines = ["\t".join(GDC_MANIFEST_COLUMNS)]
    for r in rows:
        uuid = str(r.get("id") or r.get("slide_file_uuid") or "").strip()
        if not uuid:
            raise ValueError(f"GDC manifest row missing required 'id': {dict(r)}")
        size_raw = r.get("size") if r.get("size") is not None else r.get("file_size")
        try:
            size = str(int(float(size_raw))) if size_raw is not None and str(size_raw).strip() not in {"", "nan", "<NA>"} else ""
        except Exception:
            size = str(size_raw).strip()
        rec = {
            "id": uuid,
            "filename": str(r.get("filename") or r.get("file_name") or "").strip(),
            "md5": str(r.get("md5") or r.get("md5sum") or "").strip().lower(),
            "size": size,
            "state": str(r.get("state") or "released").strip(),
        }
        lines.append("\t".join(rec[c] for c in GDC_MANIFEST_COLUMNS))
    write_text_atomic(path, "\n".join(lines) + "\n", overwrite=overwrite)
    return len(lines) - 1


def run_gdc_client_download(
    *,
    manifest_path: Path,
    dest_dir: Path,
    bin_name: str = "gdc-client",
    n_processes: int = 8,
    retry_amount: int = 5,
    wait_time: int = 5,
    gdc_api_base: Optional[str] = None,
    extra_args: Sequence[str] = (),
    cwd: Optional[Path] = None,
    timeout: Optional[int] = None,
    capture: bool = True,
) -> Dict[str, Any]:
    """Run `gdc-client download -m <manifest> -d <dest_dir>`; files land in <dest_dir>/<uuid>/<filename>.

    Returns a record of the run (no exception on a non-zero exit). gdc-client writes its logs/ under cwd;
    with capture=False its output is streamed to the terminal instead of being recorded.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    cmd: List[str] = [
        bin_name, "download",
        "-m", str(manifest_path),
        "-d", str(dest_dir),
        "-n", str(int(n_processes)),
        "--retry-amount", str(int(retry_amount)),
        "--wait-time", str(int(wait_time)),
    ]
    # --server only for a non-default API base
    if gdc_api_base and str(gdc_api_base).rstrip("/") != DEFAULT_GDC_API_BASE:
        cmd += ["--server", str(gdc_api_base).rstrip("/")]
    cmd += [str(a) for a in extra_args]
    rec: Dict[str, Any] = {"command": cmd, "cwd": str(cwd) if cwd else None, "captured": bool(capture), "started_utc": utc_now_iso()}
    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd, capture_output=capture, text=True, timeout=timeout, check=False,
            cwd=str(cwd) if cwd else None,
        )
        rec.update({
            "returncode": int(proc.returncode),
            "stdout_tail": (proc.stdout or "")[-8000:] if capture else "not_captured (streamed to terminal)",
            "stderr_tail": (proc.stderr or "")[-8000:] if capture else "not_captured (streamed to terminal)",
        })
    except subprocess.TimeoutExpired as e:
        rec.update({
            "returncode": -1,
            "stdout_tail": (e.stdout or "")[-8000:] if (capture and e.stdout) else "",
            "stderr_tail": f"TIMEOUT after {timeout}s",
        })
    except FileNotFoundError as e:
        rec.update({"returncode": -2, "stderr_tail": f"gdc-client binary not found ('{bin_name}'): {e}"})
    except Exception as e:
        rec.update({"returncode": -3, "stderr_tail": f"{type(e).__name__}: {e}"})
    rec["elapsed_seconds"] = float(time.time() - t0)
    rec["finished_utc"] = utc_now_iso()
    return rec


def gdc_eq(field: str, value: Any) -> Dict[str, Any]:
    return {"op": "=", "content": {"field": field, "value": value}}


def gdc_in(field: str, values: Sequence[Any]) -> Dict[str, Any]:
    return {"op": "in", "content": {"field": field, "value": list(values)}}


def gdc_and(*clauses: Mapping[str, Any]) -> Dict[str, Any]:
    return {"op": "and", "content": list(clauses)}


def _fields_from_properties(properties: Mapping[str, Any], prefix: str = "") -> List[str]:
    out: List[str] = []
    for name, spec in properties.items():
        full = f"{prefix}.{name}" if prefix else str(name)
        out.append(full)
        if isinstance(spec, Mapping):
            child = spec.get("properties")
            if isinstance(child, Mapping):
                out.extend(_fields_from_properties(child, full))
    return out


def mapping_fields(mapping_json: Mapping[str, Any]) -> List[str]:
    """Return endpoint field names from common GDC mapping response shapes."""
    out: set[str] = set()

    def visit(obj: Any, prefix: str = "") -> None:
        if not isinstance(obj, Mapping):
            return
        fields = obj.get("fields")
        if isinstance(fields, list):
            out.update(str(x) for x in fields)
        props = obj.get("properties")
        if isinstance(props, Mapping):
            out.update(_fields_from_properties(props, prefix))
        for key in ("data", "_mapping", "mappings"):
            if isinstance(obj.get(key), Mapping):
                visit(obj[key], prefix)
        for key, value in obj.items():
            if key in {"data", "_mapping", "mappings", "fields", "properties"}:
                continue
            if isinstance(value, Mapping) and "properties" in value:
                visit(value, prefix)

    visit(mapping_json)
    return sorted(out)


def field_available(mapping_json: Mapping[str, Any], field: str) -> bool:
    return field in set(mapping_fields(mapping_json))


# -----------------------------------------------------------------------------
# GDC flattening / biospecimen parent resolution
# -----------------------------------------------------------------------------

RNA_INVENTORY_COLUMNS = [
    "rna_file_uuid", "file_name", "file_size", "md5sum", "version", "data_release",
    "data_category", "data_type", "data_format", "experimental_strategy", "access", "state",
    "created_datetime", "updated_datetime", "workflow_type", "workflow_version",
    "case_uuid", "case_id", "project_id", "program_name", "diagnosis_id",
    "diagnosis_is_primary_disease", "primary_diagnosis", "tissue_or_organ_of_origin",
    "associated_entity_id", "associated_entity_submitter_id", "associated_entity_type",
    "sample_id", "sample_submitter_id", "sample_type", "tumor_descriptor", "tissue_type",
    "portion_id", "portion_submitter_id", "is_ffpe",
    "analyte_id", "analyte_submitter_id", "analyte_type",
    "aliquot_id", "aliquot_submitter_id",
    "parent_resolution_status", "parent_resolution_error",
    "associated_entities_json", "raw_hit_sha256",
]

SLIDE_INVENTORY_COLUMNS = [
    "slide_file_uuid", "file_name", "file_size", "md5sum", "version", "data_release",
    "data_category", "data_type", "data_format", "experimental_strategy", "access", "state",
    "created_datetime", "updated_datetime", "stain_type", "magnification",
    "case_uuid", "case_id", "project_id",
    "associated_entity_id", "associated_entity_submitter_id", "associated_entity_type",
    "sample_id", "sample_submitter_id", "sample_type", "tumor_descriptor", "tissue_type",
    "portion_id", "portion_submitter_id", "is_ffpe",
    "slide_id", "slide_submitter_id", "section_location", "slide_state",
    "percent_tumor_cells", "percent_normal_cells", "percent_necrosis", "percent_stromal_cells",
    "percent_lymphocyte_infiltration", "parent_resolution_status", "parent_resolution_error",
    "case_level_metadata_fallback", "associated_entities_json", "raw_hit_sha256",
]

ANNOTATION_COLUMNS = [
    "annotation_id", "case_id", "case_submitter_id", "entity_id", "entity_submitter_id",
    "entity_type", "classification", "category", "notes", "status", "state",
    "created_datetime", "updated_datetime", "raw_annotation_sha256",
]

FILE_VERSION_COLUMNS = [
    "queried_uuid", "query_date", "endpoint_url", "response_sha256", "match_kind", "latest_id", "latest_version",
    "latest_release", "current_file_uuid", "current_version", "current_data_release",
    "selected_uuid_is_latest", "selected_file_is_released_open", "response_warnings",
    "raw_response_json",
]


def _single_case(hit: Mapping[str, Any]) -> Tuple[Optional[Mapping[str, Any]], str]:
    cases = listify(hit.get("cases"))
    if len(cases) != 1:
        return None, f"expected_one_case_found_{len(cases)}"
    return cases[0], ""


def _first_diagnosis(case: Mapping[str, Any]) -> Mapping[str, Any]:
    diags = listify(case.get("diagnoses"))
    if not diags:
        return {}
    primaries = [d for d in diags if d.get("diagnosis_is_primary_disease") is True]
    return primaries[0] if primaries else diags[0]


def _sample_tree(case: Mapping[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for sample in listify(case.get("samples")):
        sample_base = {
            "sample_id": sample.get("sample_id") or sample.get("id"),
            "sample_submitter_id": sample.get("submitter_id"),
            "sample_type": sample.get("sample_type"),
            "tumor_descriptor": sample.get("tumor_descriptor"),
            "tissue_type": sample.get("tissue_type"),
        }
        portions = listify(sample.get("portions")) or [None]
        for portion in portions:
            portion = portion or {}
            portion_base = {
                **sample_base,
                "portion_id": portion.get("portion_id") or portion.get("id"),
                "portion_submitter_id": portion.get("submitter_id"),
                "is_ffpe": portion.get("is_ffpe"),
            }
            analytes = listify(portion.get("analytes")) or [None]
            for analyte in analytes:
                analyte = analyte or {}
                analyte_base = {
                    **portion_base,
                    "analyte_id": analyte.get("analyte_id") or analyte.get("id"),
                    "analyte_submitter_id": analyte.get("submitter_id"),
                    "analyte_type": analyte.get("analyte_type"),
                }
                aliquots = listify(analyte.get("aliquots")) or [None]
                for aliquot in aliquots:
                    aliquot = aliquot or {}
                    rows.append({
                        **analyte_base,
                        "aliquot_id": aliquot.get("aliquot_id") or aliquot.get("id"),
                        "aliquot_submitter_id": aliquot.get("submitter_id"),
                    })
    return rows


def _slide_tree(case: Mapping[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for sample in listify(case.get("samples")):
        sample_base = {
            "sample_id": sample.get("sample_id") or sample.get("id"),
            "sample_submitter_id": sample.get("submitter_id"),
            "sample_type": sample.get("sample_type"),
            "tumor_descriptor": sample.get("tumor_descriptor"),
            "tissue_type": sample.get("tissue_type"),
        }
        for portion in listify(sample.get("portions")):
            portion_base = {
                **sample_base,
                "portion_id": portion.get("portion_id") or portion.get("id"),
                "portion_submitter_id": portion.get("submitter_id"),
                "is_ffpe": portion.get("is_ffpe"),
            }
            slides = listify(portion.get("slides")) or [None]
            for slide in slides:
                slide = slide or {}
                rows.append({
                    **portion_base,
                    "slide_id": slide.get("slide_id") or slide.get("id"),
                    "slide_submitter_id": slide.get("submitter_id"),
                    "section_location": slide.get("section_location"),
                    "slide_state": slide.get("state"),
                    "percent_tumor_cells": slide.get("percent_tumor_cells"),
                    "percent_normal_cells": slide.get("percent_normal_cells"),
                    "percent_necrosis": slide.get("percent_necrosis"),
                    "percent_stromal_cells": slide.get("percent_stromal_cells"),
                    "percent_lymphocyte_infiltration": slide.get("percent_lymphocyte_infiltration"),
                })
    return rows


def _match_rows(rows: List[Dict[str, Any]], entity_type: str, entity_id: str, entity_submitter_id: str = "") -> List[Dict[str, Any]]:
    et = (entity_type or "").lower().strip()
    ids_by_type = {
        "aliquot": ["aliquot_id", "aliquot_submitter_id"],
        "analyte": ["analyte_id", "analyte_submitter_id"],
        "portion": ["portion_id", "portion_submitter_id"],
        "sample": ["sample_id", "sample_submitter_id"],
        "slide": ["slide_id", "slide_submitter_id"],
        "case": [],
    }
    cols = ids_by_type.get(et, [])
    if et == "case":
        return rows
    out = []
    for row in rows:
        for col in cols:
            val = row.get(col)
            if val and (str(val) == str(entity_id) or (entity_submitter_id and str(val) == str(entity_submitter_id))):
                out.append(row)
                break
    # De-duplicate exact row dictionaries.
    seen = set()
    uniq = []
    for r in out:
        key = stable_json_dumps(r)
        if key not in seen:
            seen.add(key)
            uniq.append(r)
    return uniq


def _associated_entity(hit: Mapping[str, Any]) -> Dict[str, Any]:
    ents = listify(hit.get("associated_entities"))
    if not ents:
        return {"entity_id": None, "entity_submitter_id": None, "entity_type": None, "n_associated_entities": 0}
    if len(ents) > 1:
        # Record the first one only (parent resolution marks it as ambiguous)
        first = ents[0]
        return {**first, "n_associated_entities": len(ents)}
    return {**ents[0], "n_associated_entities": 1}


def flatten_rna_hit(hit: Mapping[str, Any]) -> Dict[str, Any]:
    case, case_err = _single_case(hit)
    assoc = _associated_entity(hit)
    diag = _first_diagnosis(case or {})
    base: Dict[str, Any] = {
        "rna_file_uuid": hit.get("file_id") or hit.get("id"),
        "file_name": hit.get("file_name"),
        "file_size": hit.get("file_size"),
        "md5sum": hit.get("md5sum"),
        "version": hit.get("version"),
        "data_release": hit.get("data_release"),
        "data_category": hit.get("data_category"),
        "data_type": hit.get("data_type"),
        "data_format": hit.get("data_format"),
        "experimental_strategy": hit.get("experimental_strategy"),
        "access": hit.get("access"),
        "state": hit.get("state"),
        "created_datetime": hit.get("created_datetime"),
        "updated_datetime": hit.get("updated_datetime"),
        "workflow_type": nested_get(hit, "analysis.workflow_type"),
        "workflow_version": nested_get(hit, "analysis.workflow_version"),
        "case_uuid": (case or {}).get("case_id") or (case or {}).get("id"),
        "case_id": (case or {}).get("submitter_id"),
        "project_id": nested_get(case or {}, "project.project_id"),
        "program_name": nested_get(case or {}, "project.program.name"),
        "diagnosis_id": diag.get("diagnosis_id") or diag.get("id"),
        "diagnosis_is_primary_disease": diag.get("diagnosis_is_primary_disease"),
        "primary_diagnosis": diag.get("primary_diagnosis"),
        "tissue_or_organ_of_origin": diag.get("tissue_or_organ_of_origin"),
        "associated_entity_id": assoc.get("entity_id"),
        "associated_entity_submitter_id": assoc.get("entity_submitter_id"),
        "associated_entity_type": assoc.get("entity_type"),
        "associated_entities_json": jsonish(hit.get("associated_entities")),
        "raw_hit_sha256": sha256_obj(hit),
    }
    if case_err:
        base.update({"parent_resolution_status": "failed", "parent_resolution_error": case_err})
        return keep_row(base, RNA_INVENTORY_COLUMNS)
    rows = _sample_tree(case or {})
    if assoc.get("n_associated_entities", 0) != 1:
        base.update({"parent_resolution_status": "ambiguous", "parent_resolution_error": f"n_associated_entities={assoc.get('n_associated_entities')}"})
        return keep_row(base, RNA_INVENTORY_COLUMNS)
    matches = _match_rows(rows, str(assoc.get("entity_type") or ""), str(assoc.get("entity_id") or ""), str(assoc.get("entity_submitter_id") or ""))
    if len(matches) != 1:
        base.update({"parent_resolution_status": "failed" if len(matches) == 0 else "ambiguous", "parent_resolution_error": f"matched_parent_rows={len(matches)}"})
        return keep_row(base, RNA_INVENTORY_COLUMNS)
    base.update(matches[0])
    base.update({"parent_resolution_status": "resolved", "parent_resolution_error": ""})
    return keep_row(base, RNA_INVENTORY_COLUMNS)


def flatten_slide_hit(hit: Mapping[str, Any]) -> Dict[str, Any]:
    case, case_err = _single_case(hit)
    assoc = _associated_entity(hit)
    base: Dict[str, Any] = {
        "slide_file_uuid": hit.get("file_id") or hit.get("id"),
        "file_name": hit.get("file_name"),
        "file_size": hit.get("file_size"),
        "md5sum": hit.get("md5sum"),
        "version": hit.get("version"),
        "data_release": hit.get("data_release"),
        "data_category": hit.get("data_category"),
        "data_type": hit.get("data_type"),
        "data_format": hit.get("data_format"),
        "experimental_strategy": hit.get("experimental_strategy"),
        "access": hit.get("access"),
        "state": hit.get("state"),
        "created_datetime": hit.get("created_datetime"),
        "updated_datetime": hit.get("updated_datetime"),
        "stain_type": hit.get("stain_type"),
        "magnification": hit.get("magnification"),
        "case_uuid": (case or {}).get("case_id") or (case or {}).get("id"),
        "case_id": (case or {}).get("submitter_id"),
        "project_id": nested_get(case or {}, "project.project_id"),
        "associated_entity_id": assoc.get("entity_id"),
        "associated_entity_submitter_id": assoc.get("entity_submitter_id"),
        "associated_entity_type": assoc.get("entity_type"),
        "case_level_metadata_fallback": False,
        "associated_entities_json": jsonish(hit.get("associated_entities")),
        "raw_hit_sha256": sha256_obj(hit),
    }
    if case_err:
        base.update({"parent_resolution_status": "failed", "parent_resolution_error": case_err})
        return keep_row(base, SLIDE_INVENTORY_COLUMNS)
    rows = _slide_tree(case or {})
    if assoc.get("n_associated_entities", 0) != 1:
        base.update({"parent_resolution_status": "ambiguous", "parent_resolution_error": f"n_associated_entities={assoc.get('n_associated_entities')}"})
        return keep_row(base, SLIDE_INVENTORY_COLUMNS)
    entity_type = str(assoc.get("entity_type") or "").lower().strip()
    if entity_type == "case":
        base.update({"parent_resolution_status": "case_level_fallback", "parent_resolution_error": "", "case_level_metadata_fallback": True})
        return keep_row(base, SLIDE_INVENTORY_COLUMNS)
    matches = _match_rows(rows, entity_type, str(assoc.get("entity_id") or ""), str(assoc.get("entity_submitter_id") or ""))
    if len(matches) != 1:
        base.update({"parent_resolution_status": "failed" if len(matches) == 0 else "ambiguous", "parent_resolution_error": f"matched_parent_rows={len(matches)}"})
        return keep_row(base, SLIDE_INVENTORY_COLUMNS)
    base.update(matches[0])
    base.update({"parent_resolution_status": "resolved", "parent_resolution_error": ""})
    return keep_row(base, SLIDE_INVENTORY_COLUMNS)


def keep_row(row: Mapping[str, Any], cols: Sequence[str]) -> Dict[str, Any]:
    return {c: row.get(c) for c in cols}


def flatten_annotation_hit(hit: Mapping[str, Any]) -> Dict[str, Any]:
    return keep_row({
        "annotation_id": hit.get("annotation_id") or hit.get("id"),
        "case_id": hit.get("case_id"),
        "case_submitter_id": hit.get("case_submitter_id"),
        "entity_id": hit.get("entity_id"),
        "entity_submitter_id": hit.get("entity_submitter_id"),
        "entity_type": hit.get("entity_type"),
        "classification": hit.get("classification"),
        "category": hit.get("category"),
        "notes": hit.get("notes"),
        "status": hit.get("status"),
        "state": hit.get("state"),
        "created_datetime": hit.get("created_datetime"),
        "updated_datetime": hit.get("updated_datetime"),
        "raw_annotation_sha256": sha256_obj(hit),
    }, ANNOTATION_COLUMNS)


def flatten_file_versions_response(raw: Any, queried_uuids: Sequence[str], *, endpoint: str, query_date: str, response_hash: str) -> List[Dict[str, Any]]:
    # Keep the full raw response and fill normalized columns on a best-effort basis (the GDC response shape varies)
    records: List[Any]
    if isinstance(raw, Mapping) and isinstance(raw.get("data"), list):
        records = raw["data"]
    elif isinstance(raw, Mapping) and isinstance(raw.get("data"), Mapping) and isinstance(raw["data"].get("hits"), list):
        records = raw["data"]["hits"]
    elif isinstance(raw, list):
        records = raw
    elif isinstance(raw, Mapping):
        records = [raw]
    else:
        records = []
    raw_text = stable_json_dumps(raw)

    def _rec_uuid_keys(rec: Any) -> set[str]:
        """Collect UUIDs that *identify* this record (not arbitrary references inside it)."""
        keys: set[str] = set()
        if not isinstance(rec, Mapping):
            return keys
        for top in ("id", "file_id", "uuid", "queried_uuid"):
            v = rec.get(top)
            if isinstance(v, str) and v:
                keys.add(v)
        for nested_key in ("current", "current_file", "current_version", "latest", "latest_file", "latest_version"):
            sub = rec.get(nested_key)
            if isinstance(sub, Mapping):
                for k in ("id", "file_id", "uuid"):
                    v = sub.get(k)
                    if isinstance(v, str) and v:
                        keys.add(v)
        for sub in (rec.get("versions") or []):
            if isinstance(sub, Mapping):
                for k in ("id", "file_id", "uuid"):
                    v = sub.get(k)
                    if isinstance(v, str) and v:
                        keys.add(v)
        return keys

    # Lookup queried_uuid -> record by identifying fields (JSON-substring fallback, marked as low confidence)
    index: Dict[str, Tuple[Any, str]] = {}
    for rec in records:
        for k in _rec_uuid_keys(rec):
            index.setdefault(k, (rec, "id_field_match"))
    rows: List[Dict[str, Any]] = []
    for uuid in queried_uuids:
        rec, match_kind = index.get(str(uuid), (None, None))
        if rec is None:
            # Fallback: the only record containing the uuid
            substr_candidates = [r for r in records if str(uuid) in stable_json_dumps(r)]
            if len(substr_candidates) == 1:
                rec, match_kind = substr_candidates[0], "json_substring_unique_fallback"
            else:
                rec, match_kind = {}, "no_match_or_ambiguous"
        latest = rec.get("latest") if isinstance(rec, Mapping) else None
        current = rec.get("current") if isinstance(rec, Mapping) else None
        if latest is None and isinstance(rec, Mapping):
            latest = rec.get("latest_file") or rec.get("latest_version") or {}
        if current is None and isinstance(rec, Mapping):
            current = rec.get("current_file") or rec.get("current_version") or rec
        def g(obj: Any, *keys: str) -> Any:
            if not isinstance(obj, Mapping):
                return None
            for k in keys:
                if k in obj:
                    return obj[k]
            return None
        latest_id = g(latest, "id", "file_id", "uuid") or g(rec, "latest_id", "latest_uuid")
        current_uuid = g(current, "id", "file_id", "uuid") or g(rec, "current_uuid", "uuid", "id", "file_id")
        row = {
            "queried_uuid": uuid,
            "query_date": query_date,
            "endpoint_url": endpoint,
            "response_sha256": response_hash,
            "match_kind": str(match_kind),
            "latest_id": latest_id,
            "latest_version": g(latest, "version") or g(rec, "latest_version"),
            "latest_release": g(latest, "data_release", "release") or g(rec, "latest_release"),
            "current_file_uuid": current_uuid,
            "current_version": g(current, "version") or g(rec, "current_version", "version"),
            "current_data_release": g(current, "data_release", "release") or g(rec, "current_data_release", "data_release"),
            "selected_uuid_is_latest": (str(uuid) == str(latest_id)) if latest_id else pd.NA,
            "selected_file_is_released_open": pd.NA,
            "response_warnings": jsonish(raw.get("warnings", {}) if isinstance(raw, Mapping) else {}),
            "raw_response_json": raw_text,
        }
        rows.append(keep_row(row, FILE_VERSION_COLUMNS))
    return rows


def default_annotation_policy() -> Dict[str, Any]:
    return {
        "annotation_exclusion_policy": {
            "apply_only_to_active_current_annotations": True,
            "exclude_if_classification": ["Redaction"],
            "exclude_if_category_contains": [
                "Item is noncanonical",
                "Prior malignancy",
                "Synchronous malignancy",
            ],
            "warn_only_if_classification": ["Notification", "Observation", "CenterNotification"],
        }
    }


def annotate_exclusions(annotations: pd.DataFrame, policy: Mapping[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if annotations.empty:
        audit = pd.DataFrame(columns=["annotation_id", "entity_id", "classification", "category", "policy_action", "policy_reason"])
        return audit, audit.iloc[0:0].copy()
    p = policy.get("annotation_exclusion_policy", policy)
    excl_classes = set(p.get("exclude_if_classification", []))
    warn_classes = set(p.get("warn_only_if_classification", []))
    cat_contains = [str(x).lower() for x in p.get("exclude_if_category_contains", [])]
    active_only = bool(p.get("apply_only_to_active_current_annotations", True))
    rows = []
    for _, r in annotations.iterrows():
        state = str(r.get("state") or "").lower()
        status = str(r.get("status") or "").lower()
        active = True
        if active_only:
            # Missing status/state counts as active
            inactive_tokens = {"inactive", "withdrawn", "rescinded", "deleted", "archived"}
            active = not (state in inactive_tokens or status in inactive_tokens)
        action = "none"
        reason = ""
        classification = str(r.get("classification") or "")
        category = str(r.get("category") or "")
        if active and classification in excl_classes:
            action = "exclude"
            reason = f"classification={classification}"
        elif active and any(tok in category.lower() for tok in cat_contains):
            action = "exclude"
            reason = f"category_contains={category}"
        elif active and classification in warn_classes:
            action = "warn"
            reason = f"classification={classification}"
        rows.append({
            "annotation_id": r.get("annotation_id"),
            "entity_id": r.get("entity_id"),
            "entity_submitter_id": r.get("entity_submitter_id"),
            "entity_type": r.get("entity_type"),
            "classification": classification,
            "category": category,
            "state": r.get("state"),
            "status": r.get("status"),
            "policy_action": action,
            "policy_reason": reason,
        })
    audit = pd.DataFrame(rows)
    return audit, audit[audit["policy_action"] == "exclude"].copy()


# -----------------------------------------------------------------------------
# Gene alias map and STAR-counts parser helpers
# -----------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class AliasResolver:
    alias_to_symbol: Mapping[str, str]
    source_path: str
    sha256: str
    date: str
    source: str

    def resolve(self, symbol: Any) -> Tuple[Optional[str], str]:
        if symbol is None or pd.isna(symbol):
            return None, "missing"
        s = str(symbol).strip()
        if not s or s in {"?", "NA", "nan", "None"}:
            return None, "missing"
        key = normalize_gene_symbol_key(s)
        if key in self.alias_to_symbol:
            resolved = self.alias_to_symbol[key]
            if resolved == s:
                return resolved, "approved_exact"
            return resolved, "alias_or_case_resolved"
        # Not in the alias map: keep the original symbol
        return s, "unmapped_kept_original"


def normalize_gene_symbol_key(s: str) -> str:
    return re.sub(r"\s+", "", str(s).strip()).upper()


def split_alias_cell(value: Any) -> List[str]:
    if value is None or pd.isna(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    parts = re.split(r"[|,;]", text)
    return [p.strip() for p in parts if p.strip()]


def load_alias_resolver(path: Path, *, date_hint: Optional[str] = None, source: str = "HGNC frozen TSV") -> AliasResolver:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    lower = {c.lower().strip(): c for c in df.columns}
    approved_candidates = [
        "approved symbol", "approved_symbol", "symbol", "gene_symbol", "hgnc_symbol", "approved gene symbol"
    ]
    approved_col = next((lower[c] for c in approved_candidates if c in lower), None)
    if approved_col is None:
        raise RuntimeError(f"Cannot identify approved HGNC symbol column in {path}. Columns={list(df.columns)}")
    alias_cols = [lower[c] for c in ["alias symbols", "alias_symbol", "alias_symbols", "alias", "aliases"] if c in lower]
    prev_cols = [lower[c] for c in ["previous symbols", "prev_symbol", "previous_symbol", "prev_symbols"] if c in lower]
    mapping: Dict[str, str] = {}
    duplicate_conflicts: Dict[str, List[str]] = {}
    for _, r in df.iterrows():
        approved = str(r.get(approved_col, "")).strip()
        if not approved:
            continue
        for candidate in [approved] + [a for col in alias_cols + prev_cols for a in split_alias_cell(r.get(col))]:
            k = normalize_gene_symbol_key(candidate)
            if not k:
                continue
            old = mapping.get(k)
            if old is not None and old != approved:
                duplicate_conflicts.setdefault(k, sorted({old, approved}))
                # Conflict: approved symbols map to themselves; ambiguous aliases are dropped
                if k != normalize_gene_symbol_key(approved):
                    mapping.pop(k, None)
                continue
            mapping[k] = approved
    if duplicate_conflicts:
        sample = dict(list(duplicate_conflicts.items())[:10])
        raise RuntimeError(f"Ambiguous HGNC alias map entries; fix/freeze a clean alias map. Examples: {sample}")
    date = date_hint or infer_date_from_alias_filename(path) or "unknown"
    return AliasResolver(mapping, str(path), sha256_path(path), date, source)


def infer_date_from_alias_filename(path: Path) -> Optional[str]:
    m = re.search(r"(20\d{6}|19\d{6})", Path(path).name)
    return m.group(1) if m else None


def strip_ensembl_version(gene_id: Any) -> str:
    s = str(gene_id).strip()
    return re.sub(r"\.[0-9]+$", "", s)


def detect_star_counts_columns(columns: Sequence[str]) -> Dict[str, Optional[str]]:
    lower = {c.lower().strip(): c for c in columns}
    def first(cands: Sequence[str]) -> Optional[str]:
        return next((lower[c] for c in cands if c in lower), None)
    gene_id = first(["gene_id", "geneid", "ensembl_gene_id", "ensembl_id"])
    gene_name = first(["gene_name", "genename", "gene", "external_gene_name", "gene_symbol"])
    gene_type = first(["gene_type", "genetype", "gene_biotype", "gene_biotype_name"])
    fpkm_uq = first(["fpkm_uq_unstranded"])
    if fpkm_uq is None:
        matches = [c for lc, c in lower.items() if "fpkm" in lc and "uq" in lc and "tpm" not in lc]
        if matches:
            # Prefer unstranded if present, otherwise fail if ambiguous.
            unstranded = [c for c in matches if "unstranded" in c.lower()]
            if len(unstranded) == 1:
                fpkm_uq = unstranded[0]
            elif len(matches) == 1:
                fpkm_uq = matches[0]
            else:
                raise RuntimeError(f"Ambiguous FPKM-UQ columns: {matches}")
    return {"gene_id": gene_id, "gene_name": gene_name, "gene_type": gene_type, "fpkm_uq": fpkm_uq}


def read_star_counts_tsv(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    # comment='#' skips the leading comment lines
    return pd.read_csv(path, sep="\t", comment="#", dtype=str, low_memory=False)


@dataclasses.dataclass
class ParsedRNAFile:
    expression: Dict[str, float]
    gene_manifest_rows: List[Dict[str, Any]]
    parser_report: Dict[str, Any]


def parse_star_counts_expression(path: Path, resolver: AliasResolver) -> ParsedRNAFile:
    df = read_star_counts_tsv(path)
    cols = detect_star_counts_columns(df.columns)
    if cols["gene_id"] is None:
        raise RuntimeError(f"No gene_id column detected in {path}")
    if cols["gene_name"] is None:
        raise RuntimeError(f"No gene_name column detected in {path}; explicit alias resolver requires gene symbols")
    if cols["fpkm_uq"] is None:
        raise RuntimeError(f"No FPKM-UQ column detected in {path}")
    work = pd.DataFrame({
        "gene_id_original": df[cols["gene_id"]].astype(str),
        "gene_id": df[cols["gene_id"]].map(strip_ensembl_version),
        "gene_name_original": df[cols["gene_name"]].astype(str),
        "gene_type": df[cols["gene_type"]].astype(str) if cols["gene_type"] else pd.NA,
        "fpkm_uq": pd.to_numeric(df[cols["fpkm_uq"]], errors="coerce"),
    })
    # Drop STAR summary rows (N_unmapped, N_multimapping, N_noFeature, N_ambiguous; no FPKM-UQ value)
    summary_mask = work["gene_id_original"].str.startswith("N_") | work["gene_name_original"].str.startswith("N_")
    work = work.loc[~summary_mask].copy()
    if work.empty:
        raise RuntimeError(f"No gene rows after removing summary rows in {path}")
    if work["fpkm_uq"].isna().any():
        n = int(work["fpkm_uq"].isna().sum())
        raise RuntimeError(f"FPKM-UQ column has {n} non-numeric values in gene rows of {path}")
    if (work["fpkm_uq"] < 0).any():
        raise RuntimeError(f"FPKM-UQ column has negative values in {path}")
    work["fpkm_uq"] = work["fpkm_uq"].astype(float)
    if float(work["fpkm_uq"].sum()) <= 0:
        raise RuntimeError(f"All-zero expression file: {path}")
    resolved_symbols = []
    status = []
    for s in work["gene_name_original"]:
        r, st = resolver.resolve(s)
        resolved_symbols.append(r)
        status.append(st)
    work["resolved_symbol"] = resolved_symbols
    work["alias_resolution_status"] = status
    work = work.loc[work["resolved_symbol"].notna() & (work["resolved_symbol"].astype(str) != "")].copy()
    # Per-symbol aggregation: sum/size for all symbols; list columns built directly for single-row symbols
    # and with a groupby for symbols with several rows (same values)
    base = work.groupby("resolved_symbol", sort=True, dropna=True).agg(
        symbol_fpkm_uq=("fpkm_uq", "sum"),
        n_contributing_rows=("gene_id", "size"),
    )
    list_specs = [
        ("contributing_ensembl_ids", "gene_id", False),
        ("contributing_gene_names", "gene_name_original", False),
        ("contributing_gene_types", "gene_type", True),
        ("alias_resolution_statuses", "alias_resolution_status", False),
    ]
    dup_symbols = set(base.index[base["n_contributing_rows"].to_numpy() > 1])
    is_dup = work["resolved_symbol"].isin(dup_symbols)
    singles = work.loc[~is_dup]
    # Single-row symbols: [str(v)] (a missing gene_type gives [])
    single_index = singles["resolved_symbol"].to_numpy()
    single_lists: Dict[str, pd.Series] = {}
    for out_col, src_col, drop_na in list_specs:
        vals = singles[src_col].to_numpy()
        if drop_na:
            col = [[] if pd.isna(v) else [str(v)] for v in vals]
        else:
            col = [[str(v)] for v in vals]
        single_lists[out_col] = pd.Series(col, index=single_index, dtype=object)
    lists_df = pd.DataFrame(single_lists)
    if dup_symbols:
        dup_lists = work.loc[is_dup].groupby("resolved_symbol", sort=True, dropna=True).agg(
            contributing_ensembl_ids=("gene_id", lambda x: sorted(set(map(str, x)))),
            contributing_gene_names=("gene_name_original", lambda x: sorted(set(map(str, x)))),
            contributing_gene_types=("gene_type", lambda x: sorted(set(map(str, x.dropna())))),
            alias_resolution_statuses=("alias_resolution_status", lambda x: sorted(set(map(str, x)))),
        )
        lists_df = pd.concat([lists_df, dup_lists])
    grouped = base.join(lists_df).reset_index()
    grouped = grouped[[
        "resolved_symbol", "symbol_fpkm_uq",
        "contributing_ensembl_ids", "contributing_gene_names",
        "contributing_gene_types", "alias_resolution_statuses",
        "n_contributing_rows",
    ]]
    expression = {
        str(r.resolved_symbol): math.log10(1.0 + float(r.symbol_fpkm_uq))
        for r in grouped.itertuples(index=False)
    }
    gene_rows = []
    for r in grouped.itertuples(index=False):
        gene_rows.append({
            "resolved_symbol": r.resolved_symbol,
            "symbol_fpkm_uq_sum_prelog": float(r.symbol_fpkm_uq),
            "contributing_ensembl_ids": jsonish(r.contributing_ensembl_ids),
            "contributing_gene_names": jsonish(r.contributing_gene_names),
            "contributing_gene_types": jsonish(r.contributing_gene_types),
            "alias_resolution_statuses": jsonish(r.alias_resolution_statuses),
            "n_contributing_rows": int(r.n_contributing_rows),
        })
    report = {
        "path": str(path),
        "n_input_rows": int(len(df)),
        "n_summary_rows_removed": int(summary_mask.sum()),
        "n_gene_rows_after_summary_removal": int(len(work)),
        "n_resolved_symbols": int(len(expression)),
        "fpkm_uq_column": cols["fpkm_uq"],
        "gene_id_column": cols["gene_id"],
        "gene_name_column": cols["gene_name"],
        "gene_type_column": cols["gene_type"],
        "duplicate_symbol_rule": "sum_prelog_fpkm_uq_then_log10_1_plus_sum",
        "n_duplicate_resolved_symbols": int((grouped["n_contributing_rows"] > 1).sum()),
        "alias_status_counts": dict(pd.Series(status).value_counts().sort_index()),
    }
    if len(expression) < 10000:
        report["warning_low_gene_count"] = True
    return ParsedRNAFile(expression, gene_rows, report)


# -----------------------------------------------------------------------------
# Config defaults
# -----------------------------------------------------------------------------

def default_source_config(tcga_source_id: str, *, gdc_api_base: str, alias_map_sha256: str, alias_map_date: str) -> Dict[str, Any]:
    return {
        "tcga_source_id": tcga_source_id,
        "gdc_api_base_url": gdc_api_base,
        "created_at_utc": utc_now_iso(),
        "rna_query": {
            "filters": {
                "program_name": "TCGA",
                "data_category": "Transcriptome Profiling",
                "data_type": "Gene Expression Quantification",
                "experimental_strategy": "RNA-Seq",
                "workflow_type": "STAR - Counts",
                "data_format": "TSV",
                "access": "open",
                "state": "released",
            },
            "sort": "file_id:asc",
        },
        "slide_query": {
            "filters": {
                "program_name": "TCGA",
                "data_category": "Biospecimen",
                "data_type": "Slide Image",
                "experimental_strategy": "Diagnostic Slide",
                "access": "open",
                "state": "released",
            },
            "sort": "file_id:asc",
        },
        "annotation_policy": default_annotation_policy()["annotation_exclusion_policy"],
        "hgnc_alias_map": {
            "alias_map_sha256": alias_map_sha256,
            "alias_map_date": alias_map_date,
            "alias_resolution_policy": "frozen_HGNC_alias_previous_symbols_exact_case_insensitive_keep_unmapped_with_audit",
            "duplicate_symbol_policy": "sum_prelog_fpkm_uq_then_log10_1_plus_sum",
        },
    }


PRIMARY_TUMOR_SAMPLE_TYPES = ("Primary Tumor", "Primary Solid Tumor")
RNA_ENTITY_COLUMNS_FOR_ANNOTATIONS = (
    "case_uuid",
    "associated_entity_id",
    "sample_id",
    "portion_id",
    "analyte_id",
    "aliquot_id",
    "rna_file_uuid",
)


def rna_eligibility_config() -> Dict[str, Any]:
    return {
        "workflow_type": "STAR - Counts",
        "sample_type_accept": list(PRIMARY_TUMOR_SAMPLE_TYPES),
        "access": "open",
        "state": "released",
        "diagnosis_is_primary_disease_required_when_available": True,
        "annotation_policy": default_annotation_policy()["annotation_exclusion_policy"],
        "file_level_associated_entity_must_resolve_unambiguously": True,
        "file_version_release_provenance_required": True,
        "source_release_applies_eligibility_before_download": True,
        "default_source_release_rna_download_scope": "metadata_only_deferred_to_benchmark_final_cohort",
        "rna_tsv_download_stage": "build_rna_asset.py_after_slide_tile_filtering",
        "barcode_sample_code_01_used_only_as_tie_breaker_for_canonical_per_case_selection": True,
    }


def is_primary_tumor_sample_type(sample_type: Any) -> bool:
    return str(sample_type).strip() in set(PRIMARY_TUMOR_SAMPLE_TYPES)


def _normalised_nonempty_strings(values: Iterable[Any]) -> set[str]:
    out: set[str] = set()
    for value in values:
        if value is None:
            continue
        try:
            if pd.isna(value):
                continue
        except Exception:
            pass
        text = str(value).strip()
        if text and text.lower() not in {"nan", "none", "<na>"}:
            out.add(text)
    return out


def rna_annotation_entity_ids(rna_df: pd.DataFrame) -> List[str]:
    """Collect only RNA-relevant entity UUIDs for GDC annotation queries."""
    ids: List[str] = []
    for c in RNA_ENTITY_COLUMNS_FOR_ANNOTATIONS:
        if c in rna_df.columns:
            ids.extend(str(x) for x in rna_df[c].dropna().astype(str).tolist() if x and x.lower() not in {"nan", "none", "<na>"})
    return sorted(dict.fromkeys(ids))


def rna_annotation_excluded_mask(rna_df: pd.DataFrame, excluded_entities: Iterable[Any]) -> pd.Series:
    excluded = _normalised_nonempty_strings(excluded_entities)
    mask = pd.Series(False, index=rna_df.index)
    if not excluded:
        return mask
    for c in RNA_ENTITY_COLUMNS_FOR_ANNOTATIONS:
        if c in rna_df.columns:
            mask |= rna_df[c].astype(str).isin(excluded)
    return mask


def pre_download_rna_eligibility_mask(
    rna_df: pd.DataFrame,
    *,
    excluded_entities: Iterable[Any] = (),
    versioned_uuids: Optional[Iterable[Any]] = None,
    require_versioned: bool = False,
) -> pd.Series:
    """RNA eligibility mask computable before downloading the RNA TSVs (all checks except those on the file itself)."""
    required = [
        "rna_file_uuid",
        "case_id",
        "case_uuid",
        "project_id",
        "workflow_type",
        "sample_type",
        "access",
        "state",
        "parent_resolution_status",
    ]
    missing = [c for c in required if c not in rna_df.columns]
    if missing:
        raise RuntimeError(f"RNA inventory missing columns required for source-release eligibility: {missing}")

    mask = pd.Series(True, index=rna_df.index)
    mask &= rna_df["workflow_type"].astype(str).eq("STAR - Counts")
    mask &= rna_df["sample_type"].map(is_primary_tumor_sample_type)
    mask &= rna_df["access"].astype(str).eq("open")
    mask &= rna_df["state"].astype(str).eq("released")
    mask &= rna_df["parent_resolution_status"].astype(str).eq("resolved")
    if "diagnosis_is_primary_disease" in rna_df.columns:
        diag = rna_df["diagnosis_is_primary_disease"]
        mask &= diag.isna() | diag.astype(str).str.lower().isin(["true", "1", "nan", "<na>", "none"])

    mask &= ~rna_annotation_excluded_mask(rna_df, excluded_entities)

    if versioned_uuids is not None or require_versioned:
        versioned = _normalised_nonempty_strings(versioned_uuids or [])
        mask &= rna_df["rna_file_uuid"].astype(str).isin(versioned)
    return mask


def select_pre_download_eligible_rna_files(
    rna_df: pd.DataFrame,
    *,
    excluded_entities: Iterable[Any] = (),
    versioned_uuids: Optional[Iterable[Any]] = None,
    require_versioned: bool = False,
) -> pd.DataFrame:
    mask = pre_download_rna_eligibility_mask(
        rna_df,
        excluded_entities=excluded_entities,
        versioned_uuids=versioned_uuids,
        require_versioned=require_versioned,
    )
    out = rna_df.loc[mask].copy()
    if out.empty:
        return out.reset_index(drop=True)
    out["annotation_excluded"] = rna_annotation_excluded_mask(out, excluded_entities).values
    out["version_acceptable"] = True if versioned_uuids is None and not require_versioned else out["rna_file_uuid"].astype(str).isin(_normalised_nonempty_strings(versioned_uuids or [])).values
    return out.sort_values(["project_id", "case_id", "rna_file_uuid"]).reset_index(drop=True)


def _sample_barcode_rank(value: Any) -> int:
    text = str(value or "").strip().upper()
    parts = text.split("-")
    code = parts[3] if len(parts) > 3 else ""
    if code.startswith("01A"):
        return 0
    if code.startswith("01"):
        return 1
    return 2


def _numeric_release(value: Any) -> float:
    if value is None:
        return -1.0
    nums = re.findall(r"\d+(?:\.\d+)?", str(value))
    return float(nums[0]) if nums else -1.0


def choose_canonical_rna_per_case(rna_df: pd.DataFrame) -> pd.DataFrame:
    """Choose one RNA file per (project_id, case_id), deterministically.

    Ranking: non-FFPE, RNA analyte, sample barcode 01A/01, newer release/version/update, larger file, then UUID.
    """
    if rna_df.empty:
        return rna_df.copy().reset_index(drop=True)
    required = ["project_id", "case_id", "rna_file_uuid"]
    missing = [c for c in required if c not in rna_df.columns]
    if missing:
        raise RuntimeError(f"Cannot choose canonical RNA per case; missing columns: {missing}")

    work = rna_df.copy()
    group_cols = ["project_id", "case_id"]
    counts = work.groupby(group_cols, dropna=False)["rna_file_uuid"].transform("nunique")
    work["n_eligible_rna_files_for_case"] = counts.astype(int)
    work["_is_ffpe_rank"] = work.get("is_ffpe", pd.Series([None] * len(work), index=work.index)).map(lambda x: 0 if str(x).strip().lower() in {"false", "0", "no", "n"} else (2 if str(x).strip().lower() in {"true", "1", "yes", "y"} else 1))
    work["_analyte_rank"] = work.get("analyte_type", pd.Series([None] * len(work), index=work.index)).map(lambda x: 0 if str(x).strip().lower() in {"rna", "r"} else 1)
    work["_barcode_rank"] = work.get("sample_submitter_id", pd.Series([None] * len(work), index=work.index)).map(_sample_barcode_rank)
    work["_data_release_rank"] = work.get("data_release", pd.Series([None] * len(work), index=work.index)).map(_numeric_release)
    work["_version_rank"] = pd.to_numeric(work.get("version", pd.Series([-1] * len(work), index=work.index)), errors="coerce").fillna(-1)
    updated = pd.to_datetime(work.get("updated_datetime", pd.Series([None] * len(work), index=work.index)), errors="coerce", utc=True)
    work["_updated_rank"] = updated.astype("int64")
    work["_file_size_rank"] = pd.to_numeric(work.get("file_size", pd.Series([-1] * len(work), index=work.index)), errors="coerce").fillna(-1)
    work["_rna_uuid_rank"] = work["rna_file_uuid"].astype(str)

    sort_cols = group_cols + [
        "_is_ffpe_rank",
        "_analyte_rank",
        "_barcode_rank",
        "_data_release_rank",
        "_version_rank",
        "_updated_rank",
        "_file_size_rank",
        "_rna_uuid_rank",
    ]
    ascending = [True, True, True, True, True, False, False, False, False, True]
    ranked = work.sort_values(sort_cols, ascending=ascending, kind="mergesort")
    chosen = ranked.groupby(group_cols, dropna=False, sort=False).head(1).copy()
    chosen["canonical_rna_selection_policy"] = (
        "one_per_project_case:prefer_non_ffpe_then_rna_analyte_then_TCGA_01A_or_01_barcode_"
        "then_newer_release_version_update_larger_file_uuid"
    )
    helper_cols = [c for c in chosen.columns if c.startswith("_")]
    return chosen.drop(columns=helper_cols).sort_values(group_cols + ["rna_file_uuid"]).reset_index(drop=True)


def rna_parser_config() -> Dict[str, Any]:
    return {
        "parser": "dynamic_GDC_STAR_counts_TSV_parser",
        "expression_column": "prefer_fpkm_uq_unstranded_else_unique_FPKM_UQ_column",
        "transform": "log10(1 + FPKM-UQ)",
        "duplicate_resolved_symbol_rule": "sum_prelog_fpkm_uq_then_log10_1_plus_sum",
        "strip_ensembl_version_suffixes": True,
        "remove_summary_rows_prefix": "N_",
        "do_not_zero_fill_missing_genes": True,
        "parser_software_version": "taskb_common.parse_star_counts_expression.v1",
    }


def deterministic_rna_asset_id(tcga_source_id: str, components: Mapping[str, Any]) -> str:
    return f"{tcga_source_id}__rna_log10fpkmuq__{sha256_obj(components)[:16]}"


def load_domain_map_checked(path: Path) -> Dict[str, Any]:
    obj = read_json(path) if path.suffix.lower() == ".json" else read_yaml(path)
    if not isinstance(obj, Mapping):
        raise RuntimeError("project_domain_map must be a JSON/YAML object")
    meta = obj.get("manual_review") or obj.get("metadata") or {}
    if not bool(meta.get("reviewed")):
        raise RuntimeError("project_domain_map manual review is required: reviewed must be true")
    for field in ["reviewer", "review_date", "allowed_labels"]:
        if field not in meta:
            raise RuntimeError(f"project_domain_map manual review metadata missing: {field}")
    allowed = set(meta.get("allowed_labels") or [])
    required = {"in_domain", "out_of_domain", "excluded"}
    if allowed != required:
        raise RuntimeError(f"project_domain_map allowed_labels must be exactly {sorted(required)}")
    labels = obj.get("project_domain_map") or obj.get("projects") or {}
    if not isinstance(labels, Mapping) or not labels:
        raise RuntimeError("project_domain_map has no project labels")
    bad = {k: v for k, v in labels.items() if v not in required}
    if bad:
        raise RuntimeError(f"project_domain_map contains invalid labels: {bad}")
    return dict(obj)
