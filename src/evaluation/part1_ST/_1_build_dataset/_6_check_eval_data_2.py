#!/usr/bin/env python
"""
Check the X matrix of the ST eval .h5ad files of a training_dataset_id: non-negative, integer-like (random sample), no NaN,
sparse. The STHELAR "<patch_id>_cells" rows (weighted sums of cell counts) are not checked.

Saved in <ROOT_HistoST> / eval / eval_datasets / st_eval_data / eval_st_<training_dataset_id> / st / check_eval_data_excluding_sthelar_cells.txt
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import h5py
import numpy as np
import pandas as pd

# Add the project root to sys.path so that 'src' can be imported
project_root = Path(__file__).resolve().parents[4]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.constants import get_full_path


# ============================================================
# Config
# ============================================================

RANDOM_SAMPLE_SIZE = 20_000
INTEGER_ATOL = 1e-6
TARGET_CHUNK_MB = 64
REPORT_FILENAME = "check_eval_data_excluding_sthelar_cells.txt"


# ============================================================
# Generic helpers
# ============================================================

def _is_real_h5ad_file(path: Path) -> bool:
    """True for visible .h5ad files (hidden files such as macOS '._*' files are excluded)."""
    return (
        path.is_file()
        and path.suffix == ".h5ad"
        and not path.name.startswith(".")
    )


def _decode_scalar(x):
    if x is None:
        return None
    if isinstance(x, np.ndarray) and x.shape == ():
        x = x.item()
    if isinstance(x, (bytes, np.bytes_)):
        return x.decode("utf-8")
    return x


def _decode_string_array(arr) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.dtype.kind in {"S", "O", "U"}:
        out = []
        for x in arr:
            if isinstance(x, (bytes, np.bytes_)):
                out.append(bytes(x).decode("utf-8"))
            else:
                out.append(str(x))
        return np.asarray(out, dtype=object)
    return arr.astype(object)


def _get_axis_index_dataset(axis_group: h5py.Group) -> Optional[h5py.Dataset]:
    """
    Return the dataset backing the AnnData axis index, without reading its values.
    """
    if "_index" in axis_group:
        idx_key = "_index"
    else:
        idx_key = _decode_scalar(axis_group.attrs.get("_index"))
        if idx_key is None and "index" in axis_group:
            idx_key = "index"

    if idx_key is None or idx_key not in axis_group:
        return None

    return axis_group[idx_key]


def _dtype_is_numeric(dtype) -> bool:
    try:
        return np.issubdtype(np.dtype(dtype), np.number)
    except Exception:
        return False


def _is_integer_like(values: np.ndarray, atol: float = INTEGER_ATOL) -> bool:
    if values.size == 0:
        return True
    values = np.asarray(values, dtype=np.float64)
    return bool(np.all(np.isclose(values, np.rint(values), atol=atol, rtol=0.0, equal_nan=False)))


def _true_intervals(mask: np.ndarray) -> List[Tuple[int, int]]:
    """
    Convert boolean mask into contiguous [start, end) intervals of True rows.
    """
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []

    splits = np.where(np.diff(idx) != 1)[0] + 1
    groups = np.split(idx, splits)
    return [(int(g[0]), int(g[-1]) + 1) for g in groups]


def _read_obs_index(f: h5py.File) -> np.ndarray:
    """
    Read obs index names from .h5ad with h5py.
    """
    if "obs" not in f:
        raise KeyError("Missing /obs group in .h5ad")

    obs_index_ds = _get_axis_index_dataset(f["obs"])
    if obs_index_ds is None:
        raise KeyError("Could not locate obs index dataset in /obs")

    return _decode_string_array(obs_index_ds[:])


def _read_obs_count(f: h5py.File) -> int:
    """
    Read number of obs rows without decoding the obs names.
    """
    if "obs" not in f:
        raise KeyError("Missing /obs group in .h5ad")

    obs_index_ds = _get_axis_index_dataset(f["obs"])
    if obs_index_ds is None:
        raise KeyError("Could not locate obs index dataset in /obs")

    return int(obs_index_ds.shape[0])


def _read_var_count(f: h5py.File) -> Optional[int]:
    """
    Read number of variables if possible.
    """
    if "var" not in f:
        return None

    var_index_ds = _get_axis_index_dataset(f["var"])
    if var_index_ds is None:
        return None

    return int(var_index_ds.shape[0])


def _load_metadata_source_map(eval_root: Path, training_dataset_id: str) -> Dict[str, str]:
    metadata_csv = eval_root / f"metadata_eval_st_{training_dataset_id}.csv"
    if not metadata_csv.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {metadata_csv}")

    df = pd.read_csv(metadata_csv)
    for col in ("slide_id", "source"):
        if col not in df.columns:
            raise KeyError(f"Metadata CSV must contain column '{col}'")

    out: Dict[str, str] = {}
    grouped = df.groupby(df["slide_id"].astype(str))["source"].apply(lambda s: sorted(set(s.astype(str))))
    for slide_id, sources in grouped.items():
        if len(sources) != 1:
            raise ValueError(f"Slide '{slide_id}' has multiple sources in metadata: {sources}")
        out[str(slide_id)] = str(sources[0])

    return out


# ============================================================
# X storage description
# ============================================================

def _describe_x_storage(x_obj, n_obs: Optional[int] = None, n_var: Optional[int] = None) -> Dict[str, Optional[str]]:
    """Storage of X as {"kind": "dense" | "sparse" | "unknown", "storage_format", "dtype", "shape"}."""
    if isinstance(x_obj, h5py.Dataset):
        return {
            "kind": "dense",
            "storage_format": None,
            "dtype": str(x_obj.dtype),
            "shape": str(tuple(x_obj.shape)),
        }

    if not isinstance(x_obj, h5py.Group):
        return {
            "kind": "unknown",
            "storage_format": None,
            "dtype": None,
            "shape": None,
        }

    enc = _decode_scalar(x_obj.attrs.get("encoding-type"))
    enc_norm = str(enc).lower() if enc is not None else None

    has_sparse_keys = {"data", "indices", "indptr"}.issubset(set(x_obj.keys()))
    shape_attr = _decode_scalar(x_obj.attrs.get("shape"))
    shape_str = str(tuple(shape_attr)) if shape_attr is not None else None

    if not has_sparse_keys and enc_norm not in {"csr_matrix", "csc_matrix", "csr", "csc"}:
        return {
            "kind": "unknown",
            "storage_format": None,
            "dtype": None,
            "shape": shape_str,
        }

    fmt = None
    if enc_norm in {"csr_matrix", "csr"}:
        fmt = "csr_matrix"
    elif enc_norm in {"csc_matrix", "csc"}:
        fmt = "csc_matrix"
    else:
        # Try to infer from dimensions if encoding-type is missing.
        indptr_len = int(x_obj["indptr"].shape[0])
        inferred = None
        if n_obs is not None and indptr_len == n_obs + 1 and (n_var is None or indptr_len != n_var + 1):
            inferred = "csr_matrix"
        elif n_var is not None and indptr_len == n_var + 1 and (n_obs is None or indptr_len != n_obs + 1):
            inferred = "csc_matrix"
        fmt = inferred if inferred is not None else "unknown_sparse"

    return {
        "kind": "sparse",
        "storage_format": fmt,
        "dtype": str(x_obj["data"].dtype),
        "shape": shape_str,
    }


# ============================================================
# Dense helpers
# ============================================================

def _choose_sorted_ranks(
    total: int,
    rng: np.random.Generator,
    sample_size: int,
) -> np.ndarray:
    if total <= 0 or sample_size <= 0:
        return np.empty(0, dtype=np.int64)
    s = min(sample_size, total)
    return np.sort(rng.choice(total, size=s, replace=False).astype(np.int64, copy=False))


def _scan_and_sample_dense_selected_rows(
    x_ds: h5py.Dataset,
    keep_rows: np.ndarray,
    rng: np.random.Generator,
    sample_size: int,
) -> Tuple[bool, bool, np.ndarray]:
    """
    Exact scan over selected dense rows, while collecting a uniform random
    sample (uniform over all kept entries, without replacement).
    """
    if len(x_ds.shape) != 2:
        raise ValueError(f"Dense X must be 2D, got shape={x_ds.shape}")

    n_cols = int(x_ds.shape[1])
    n_kept_rows = int(np.count_nonzero(keep_rows))

    total = n_kept_rows * n_cols
    chosen_ranks = _choose_sorted_ranks(total=total, rng=rng, sample_size=sample_size)
    out = np.empty(chosen_ranks.size, dtype=np.float64)

    itemsize = max(1, np.dtype(x_ds.dtype).itemsize)
    bytes_per_row = max(1, n_cols * itemsize)
    row_chunk = max(1, (TARGET_CHUNK_MB * 1024 * 1024) // bytes_per_row)

    has_nan = False
    has_negative = False
    out_pos = 0
    seen = 0

    intervals = _true_intervals(keep_rows)
    for start_row, end_row in intervals:
        r = start_row
        while r < end_row:
            stop = min(r + row_chunk, end_row)
            block = np.asarray(x_ds[r:stop, :])

            if not has_nan and np.isnan(block).any():
                has_nan = True
            if not has_negative and np.any(block < 0):
                has_negative = True

            block_size = int(block.size)
            if block_size > 0:
                left = np.searchsorted(chosen_ranks, seen, side="left")
                right = np.searchsorted(chosen_ranks, seen + block_size, side="left")
                if right > left:
                    local_ranks = chosen_ranks[left:right] - seen
                    block_flat = np.asarray(block, dtype=np.float64).reshape(-1)
                    picked = block_flat[local_ranks]
                    out[out_pos:out_pos + picked.size] = picked
                    out_pos += picked.size
                seen += block_size

            r = stop

    return has_nan, has_negative, out[:out_pos]


# ============================================================
# Sparse CSR helpers
# ============================================================

def _csr_selected_segments(indptr: np.ndarray, keep_rows: np.ndarray) -> List[Tuple[int, int]]:
    """
    For CSR, selected rows correspond to contiguous data segments:
      [indptr[row_start], indptr[row_end])
    """
    segments: List[Tuple[int, int]] = []
    for row_start, row_end in _true_intervals(keep_rows):
        data_start = int(indptr[row_start])
        data_end = int(indptr[row_end])
        if data_end > data_start:
            segments.append((data_start, data_end))
    return segments


def _scan_and_sample_data_segments(
    data_ds: h5py.Dataset,
    segments: List[Tuple[int, int]],
    rng: np.random.Generator,
    sample_size: int,
) -> Tuple[bool, bool, np.ndarray]:
    """
    Exact scan over selected contiguous 1D data segments, while collecting a
    uniform random sample (uniform over the selected entries, without
    replacement).
    """
    lengths = np.asarray([end - start for start, end in segments], dtype=np.int64)
    total = int(lengths.sum())
    chosen_ranks = _choose_sorted_ranks(total=total, rng=rng, sample_size=sample_size)
    out = np.empty(chosen_ranks.size, dtype=np.float64)

    itemsize = max(1, np.dtype(data_ds.dtype).itemsize)
    chunk_elems = max(1, (TARGET_CHUNK_MB * 1024 * 1024) // itemsize)

    has_nan = False
    has_negative = False
    out_pos = 0
    seen = 0

    for seg_start, seg_end in segments:
        pos = seg_start
        while pos < seg_end:
            stop = min(pos + chunk_elems, seg_end)
            arr = np.asarray(data_ds[pos:stop])

            if not has_nan and np.isnan(arr).any():
                has_nan = True
            if not has_negative and np.any(arr < 0):
                has_negative = True

            block_size = int(arr.size)
            if block_size > 0:
                left = np.searchsorted(chosen_ranks, seen, side="left")
                right = np.searchsorted(chosen_ranks, seen + block_size, side="left")
                if right > left:
                    local_ranks = chosen_ranks[left:right] - seen
                    arr64 = np.asarray(arr, dtype=np.float64).reshape(-1)
                    picked = arr64[local_ranks]
                    out[out_pos:out_pos + picked.size] = picked
                    out_pos += picked.size
                seen += block_size

            pos = stop

    return has_nan, has_negative, out[:out_pos]


# ============================================================
# Sparse CSC helpers
# ============================================================

def _scan_csc_selected_for_nan_neg_and_count(
    data_ds: h5py.Dataset,
    indices_ds: h5py.Dataset,
    keep_rows: np.ndarray,
) -> Tuple[bool, bool, int]:
    """
    Exact scan over CSC stored values restricted to selected rows.
    Returns:
      has_nan, has_negative, n_selected_stored_entries
    """
    n_nnz = int(data_ds.shape[0])
    itemsize_data = max(1, np.dtype(data_ds.dtype).itemsize)
    itemsize_idx = max(1, np.dtype(indices_ds.dtype).itemsize)
    itemsize = max(itemsize_data, itemsize_idx)
    chunk_elems = max(1, (TARGET_CHUNK_MB * 1024 * 1024) // itemsize)

    has_nan = False
    has_negative = False
    n_selected = 0

    for start in range(0, n_nnz, chunk_elems):
        stop = min(start + chunk_elems, n_nnz)
        idx_chunk = np.asarray(indices_ds[start:stop], dtype=np.int64)
        mask = keep_rows[idx_chunk]

        if np.any(mask):
            data_chunk = np.asarray(data_ds[start:stop])
            vals = data_chunk[mask]
            n_selected += int(vals.size)

            if not has_nan and np.isnan(vals).any():
                has_nan = True
            if not has_negative and np.any(vals < 0):
                has_negative = True

            if has_nan and has_negative:
                # cannot break early: the exact n_selected is still needed
                pass

    return has_nan, has_negative, n_selected


def _sample_csc_selected_values(
    data_ds: h5py.Dataset,
    indices_ds: h5py.Dataset,
    keep_rows: np.ndarray,
    n_selected_total: int,
    rng: np.random.Generator,
    sample_size: int,
) -> np.ndarray:
    """Uniform random sample among the selected CSC stored values (n_selected_total counted by a first pass)."""
    if n_selected_total == 0 or sample_size <= 0:
        return np.empty(0, dtype=np.float64)

    chosen_ranks = _choose_sorted_ranks(
        total=n_selected_total,
        rng=rng,
        sample_size=sample_size,
    )

    n_nnz = int(data_ds.shape[0])
    itemsize_data = max(1, np.dtype(data_ds.dtype).itemsize)
    itemsize_idx = max(1, np.dtype(indices_ds.dtype).itemsize)
    itemsize = max(itemsize_data, itemsize_idx)
    chunk_elems = max(1, (TARGET_CHUNK_MB * 1024 * 1024) // itemsize)

    out = np.empty(s, dtype=np.float64)
    out_pos = 0
    seen_selected = 0

    for start in range(0, n_nnz, chunk_elems):
        stop = min(start + chunk_elems, n_nnz)
        idx_chunk = np.asarray(indices_ds[start:stop], dtype=np.int64)
        mask = keep_rows[idx_chunk]

        n_kept_here = int(mask.sum())
        if n_kept_here == 0:
            continue

        chunk_start_rank = seen_selected
        chunk_end_rank = seen_selected + n_kept_here

        left = np.searchsorted(chosen_ranks, chunk_start_rank, side="left")
        right = np.searchsorted(chosen_ranks, chunk_end_rank, side="left")

        if right > left:
            data_chunk = np.asarray(data_ds[start:stop])
            kept_vals = np.asarray(data_chunk[mask], dtype=np.float64).reshape(-1)

            local_ranks = chosen_ranks[left:right] - chunk_start_rank
            picked = kept_vals[local_ranks]
            out[out_pos:out_pos + picked.size] = picked
            out_pos += picked.size

        seen_selected = chunk_end_rank

    return out[:out_pos]


# ============================================================
# Main per-file check
# ============================================================

def check_h5ad_file(
    h5ad_path: Path,
    slide_source: Optional[str],
    rng_seed: int = 0,
) -> Dict:
    slide_id = h5ad_path.stem
    rng = np.random.default_rng(rng_seed)

    result = {
        "slide_id": slide_id,
        "path": str(h5ad_path),
        "source": slide_source,
        "issues": [],
        "storage_kind": None,
        "storage_format": None,
        "dtype": None,
        "shape": None,
        "n_rows_total": None,
        "n_rows_checked": None,
        "n_rows_excluded": None,
        "sample_size_used": 0,
        "row_policy": None,
    }

    try:
        with h5py.File(h5ad_path, "r") as f:
            if "X" not in f:
                result["issues"].append("missing X")
                result["storage_kind"] = "unknown"
                return result

            x_obj = f["X"]

            source = str(slide_source) if slide_source is not None else "UNKNOWN"

            if source == "STHELAR":
                obs_names = _read_obs_index(f)
                n_obs = int(obs_names.shape[0])
                exclude_mask = np.fromiter(
                    (str(name).endswith("_cells") for name in obs_names),
                    dtype=bool,
                    count=n_obs,
                )
                keep_rows = ~exclude_mask
                result["row_policy"] = "STHELAR: excluded rows ending with '_cells'; kept '_all' and cell rows"
            else:
                n_obs = _read_obs_count(f)
                keep_rows = np.ones(n_obs, dtype=bool)
                result["row_policy"] = "HEST/other: checked all rows"

            n_rows_checked = int(keep_rows.sum())
            n_rows_excluded = int((~keep_rows).sum())

            result["n_rows_total"] = n_obs
            result["n_rows_checked"] = n_rows_checked
            result["n_rows_excluded"] = n_rows_excluded

            if n_rows_checked == 0:
                result["issues"].append("no rows left to check after row filtering")
                return result

            desc = _describe_x_storage(x_obj, n_obs=n_obs, n_var=None)
            if desc["kind"] == "sparse" and desc.get("storage_format") == "unknown_sparse":
                n_var = _read_var_count(f)
                if n_var is not None:
                    desc = _describe_x_storage(x_obj, n_obs=n_obs, n_var=n_var)

            result["storage_kind"] = desc["kind"]
            # Accepts both "storage_format" and "sparse_format" keys
            storage_format = desc.get("storage_format", desc.get("sparse_format"))
            result["storage_format"] = storage_format
            result["dtype"] = desc["dtype"]
            result["shape"] = desc["shape"]

            if desc["kind"] == "unknown":
                result["issues"].append("unknown X storage type")
                return result

            # ---------------- Dense ----------------
            if desc["kind"] == "dense":
                result["issues"].append("X is dense (expected sparse)")

                if not _dtype_is_numeric(x_obj.dtype):
                    result["issues"].append(f"X has non-numeric dtype ({x_obj.dtype})")
                    return result

                if len(x_obj.shape) != 2:
                    result["issues"].append(f"dense X is not 2D (shape={x_obj.shape})")
                    return result

                if int(x_obj.shape[0]) != n_obs:
                    result["issues"].append(
                        f"X row count ({x_obj.shape[0]}) does not match obs row count ({n_obs})"
                    )
                    return result

                has_nan, has_negative, sample = _scan_and_sample_dense_selected_rows(
                    x_ds=x_obj,
                    keep_rows=keep_rows,
                    rng=rng,
                    sample_size=RANDOM_SAMPLE_SIZE,
                )
                result["sample_size_used"] = int(sample.size)
                if not _is_integer_like(sample):
                    result["issues"].append("X is not integer-like on the random sample")

                if has_nan:
                    result["issues"].append("X contains NaNs")
                if has_negative:
                    result["issues"].append("X contains negative values")

                return result

            # ---------------- Sparse ----------------
            if desc["kind"] == "sparse":
                if storage_format == "unknown_sparse":
                    result["issues"].append("unknown sparse format")
                    return result

                data_ds = x_obj["data"]
                indices_ds = x_obj["indices"]
                indptr_ds = x_obj["indptr"]

                if not _dtype_is_numeric(data_ds.dtype):
                    result["issues"].append(f"X.data has non-numeric dtype ({data_ds.dtype})")
                    return result

                # CSR
                if storage_format == "csr_matrix":
                    indptr = np.asarray(indptr_ds[:], dtype=np.int64)
                    if indptr.shape[0] != n_obs + 1:
                        result["issues"].append(
                            f"CSR indptr length ({indptr.shape[0]}) does not match n_obs+1 ({n_obs + 1})"
                        )
                        return result

                    segments = _csr_selected_segments(indptr=indptr, keep_rows=keep_rows)

                    has_nan, has_negative, sample = _scan_and_sample_data_segments(
                        data_ds=data_ds,
                        segments=segments,
                        rng=rng,
                        sample_size=RANDOM_SAMPLE_SIZE,
                    )
                    result["sample_size_used"] = int(sample.size)

                    if not _is_integer_like(sample):
                        result["issues"].append("X is not integer-like on the random sample")

                    if has_nan:
                        result["issues"].append("X contains NaNs")
                    if has_negative:
                        result["issues"].append("X contains negative values")

                    return result

                # CSC
                if storage_format == "csc_matrix":
                    has_nan, has_negative, n_selected_nnz = _scan_csc_selected_for_nan_neg_and_count(
                        data_ds=data_ds,
                        indices_ds=indices_ds,
                        keep_rows=keep_rows,
                    )

                    sample = _sample_csc_selected_values(
                        data_ds=data_ds,
                        indices_ds=indices_ds,
                        keep_rows=keep_rows,
                        n_selected_total=n_selected_nnz,
                        rng=rng,
                        sample_size=RANDOM_SAMPLE_SIZE,
                    )
                    result["sample_size_used"] = int(sample.size)

                    if not _is_integer_like(sample):
                        result["issues"].append("X is not integer-like on the random sample")
                    if has_nan:
                        result["issues"].append("X contains NaNs")
                    if has_negative:
                        result["issues"].append("X contains negative values")

                    return result

                result["issues"].append(f"unhandled sparse format ({storage_format})")
                return result

            result["issues"].append("unhandled X storage kind")
            return result

    except Exception as e:
        result["issues"].append(f"failed to read/check file: {type(e).__name__}: {e}")
        return result


# ============================================================
# Report
# ============================================================

def write_report(st_dir: Path, training_dataset_id: str, results: List[Dict]) -> Path:
    out_path = st_dir / REPORT_FILENAME

    total = len(results)
    bad_results = [r for r in results if len(r["issues"]) > 0]
    n_bad = len(bad_results)

    lines: List[str] = []
    lines.append("ST EVAL DATA CHECK REPORT")
    lines.append("=" * 100)
    lines.append(f"training_dataset_id: {training_dataset_id}")
    lines.append(f"st_dir: {st_dir}")
    lines.append("")
    lines.append("Row selection policy:")
    lines.append("  - HEST slides: check all rows")
    lines.append("  - STHELAR slides: exclude rows ending with '_cells'")
    lines.append("  - STHELAR rows ending with '_all' are kept and checked")
    lines.append("  - STHELAR cell rows are kept and checked")
    lines.append("")
    lines.append("A slide is considered good only if the checked rows of X are:")
    lines.append("  - nonnegative")
    lines.append("  - integer-like on a random sample")
    lines.append("  - with no NaNs")
    lines.append("  - sparse")
    lines.append("")
    lines.append(f"Bad slides: {n_bad}/{total}")
    lines.append("")

    if n_bad == 0:
        lines.append("No bad slides found.")
    else:
        lines.append("Bad slide details:")
        lines.append("")
        for r in bad_results:
            lines.append(f"- {r['slide_id']}")
            lines.append(f"    source: {r['source']}")
            lines.append(f"    issues: {', '.join(r['issues'])}")
            lines.append(f"    row_policy: {r['row_policy']}")
            lines.append(f"    n_rows_total: {r['n_rows_total']}")
            lines.append(f"    n_rows_checked: {r['n_rows_checked']}")
            lines.append(f"    n_rows_excluded: {r['n_rows_excluded']}")
            lines.append(f"    storage_kind: {r['storage_kind']}")
            lines.append(f"    storage_format: {r['storage_format']}")
            lines.append(f"    dtype: {r['dtype']}")
            lines.append(f"    shape: {r['shape']}")
            lines.append(f"    sample_size_used: {r['sample_size_used']}")
            lines.append("")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


# ============================================================
# Main
# ============================================================

def main(args) -> None:
    training_dataset_id = args.training_dataset_id

    eval_root = Path(get_full_path("eval", "eval_datasets", "st_eval_data", f"eval_st_{training_dataset_id}"))
    st_dir = eval_root / "st"

    if not st_dir.exists():
        raise FileNotFoundError(f"ST directory not found: {st_dir}")

    source_map = _load_metadata_source_map(eval_root=eval_root, training_dataset_id=training_dataset_id)

    h5ad_files = sorted(p for p in st_dir.glob("*.h5ad") if _is_real_h5ad_file(p))
    if len(h5ad_files) == 0:
        raise FileNotFoundError(f"No .h5ad files found in: {st_dir}")

    print(f"Checking {len(h5ad_files)} .h5ad file(s) in: {st_dir}")

    results: List[Dict] = []

    for i, h5ad_path in enumerate(h5ad_files, start=1):
        slide_id = h5ad_path.stem
        slide_source = source_map.get(slide_id, None)

        print(f"[{i}/{len(h5ad_files)}] Checking {h5ad_path.name} ...")
        if slide_source is None:
            print("  -> WARNING: slide not found in metadata; treating as HEST/other and checking all rows")

        result = check_h5ad_file(
            h5ad_path=h5ad_path,
            slide_source=slide_source,
            rng_seed=i,
        )
        results.append(result)

        if result["issues"]:
            print(f"  -> BAD: {', '.join(result['issues'])}")
        else:
            print("  -> OK")

    out_path = write_report(
        st_dir=st_dir,
        training_dataset_id=training_dataset_id,
        results=results,
    )

    n_bad = sum(1 for r in results if r["issues"])
    print("")
    print(f"Done. Bad slides: {n_bad}/{len(results)}")
    print(f"Report written to: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check ST eval .h5ad files.")
    
    parser.add_argument( "--training_dataset_id", type=str,
                        required=True,
                        help="Training dataset id, e.g. ds_ct1_20x_224_o112_1")
    
    args = parser.parse_args()
    main(args)