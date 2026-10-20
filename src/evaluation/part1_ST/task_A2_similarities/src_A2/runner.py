import gc
import hashlib
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
import anndata as ad
import h5py
import numpy as np
import pandas as pd
from scipy import sparse
import torch

from src_A2.common import (
    configure_logger,
    ensure_dir,
    entropy_from_probs,
    file_info,
    read_json,
    resolve_device,
    set_global_seed,
    write_dataframe,
    write_json,
    sha256_file,
    hash_mapping,
)
from src_A2.embeddings import (
    ensure_selected_main_rows,
    load_gt_cell_embeddings,
    load_gt_expression_matrix,
    load_selected_he_embeddings,
    project_reference_embeddings_if_needed,
    resolve_legacy_he_embedding_dir,
    resolve_raw_reference_embedding_h5,
    resolve_legacy_st_cell_embedding_dir,
)
from src_A2.coverage import (
    STAGE_RUN,
    assert_coverage_contract,
    assert_runtime_coverage_matches_frozen,
    build_slide_coverage_report,
)
from src_A2.label_granularity import apply_label_map, normalize_granularity, resolve_label_map
from src_A2.metrics import (
    agreement_rate,
    assert_unscorable_labels_absent,
    classification_metrics_from_ids,
    macro_min_class_support,
    class_scores_from_neighbors,
    entropy_metrics_from_class_scores,
    labels_to_ids,
    majority_class_metrics,
    predicted_labels_from_scores,
    summarize_expression,
    weighted_expression_prediction,
    weighted_expression_prediction_h5,
)
from src_A2.protocol import get_protocol_paths, load_protocol
from src_A2.references import read_obs_columns
from src_A2.retrieval import (
    RetrievalResult,
    jaccard_at_k,
    l2_normalize,
    mapped_class_mass,
    retrieve_topk,
    weights_from_similarities,
)
from src_A2.umap_utils import run_umap_for_slide

from src_A2.common import src_import
get_full_path = src_import("get_full_path")
BACKBONE_TRANSFORM = src_import("BACKBONE_TRANSFORM")
load_exported_he_encoder = src_import("load_exported_he_encoder")


CHECKPOINT_OUTPUT_SUBDIRS = [
    "selected_main_he_rows",
    "projected_embeddings/reference",
    "retrieval",
    "predictions",
    "metrics",
    "diagnostics",
    "plots",
    "logs",
]


def _prepare_new_checkpoint_output_tree(checkpoint_dir: Path) -> None:
    """Create the checkpoint output subfolders; raises if the checkpoint directory already contains files."""
    if checkpoint_dir.exists():
        existing_files = sorted(
            p for p in checkpoint_dir.rglob("*")
            if p.is_file() or p.is_symlink()
        )
        if existing_files:
            shown = existing_files[:30]
            examples = "\n".join(
                f"  - {p.relative_to(checkpoint_dir)}"
                for p in shown
            )
            remaining = len(existing_files) - len(shown)
            more = f"\n  ... and {remaining} more file(s)" if remaining > 0 else ""
            raise FileExistsError(
                f"Refusing to reuse existing checkpoint output directory:\n"
                f"{checkpoint_dir}\n\n"
                f"Existing files were found:\n"
                f"{examples}{more}\n\n"
                "Create a new eval_id, remove this checkpoint directory, or move it "
                "before rerunning this checkpoint evaluation."
            )

    for sub in CHECKPOINT_OUTPUT_SUBDIRS:
        ensure_dir(checkpoint_dir / sub)


@dataclass
class SlideData:
    slide_id: str
    subset: str
    tissue: str
    reference_stem: str
    reference_h5ad: Path
    cells: pd.DataFrame
    z_he: np.ndarray
    z_gt: np.ndarray
    he_valid: np.ndarray
    he_structural_eligible: np.ndarray
    gt_valid: np.ndarray
    gt_expr_present: np.ndarray
    matched_valid: np.ndarray
    covered: np.ndarray
    gt_mapped_label_scorable: np.ndarray
    gt_label_ids: np.ndarray
    gt_labels: np.ndarray
    # None if expression.enabled is False (otherwise of width len(gene_names))
    gt_expr: Optional[np.ndarray]
    gene_names: List[str]
    gt_gene_names: List[str]
    gene_panel_masks: Dict[str, np.ndarray]
    gene_panel_meta: Dict[str, Dict[str, Any]]
    ref_z: np.ndarray
    ref_labels: np.ndarray
    ref_label_ids: np.ndarray
    ref_fine_labels: np.ndarray
    ref_expr: Optional[np.ndarray]
    ref_expr_global_to_local: Optional[np.ndarray]
    reference_h5ad_valid_rows: np.ndarray
    reference_var_indices: np.ndarray
    he_slide_h5: Path
    gt_emb_h5ad: Path
    gt_eval_h5ad: Path
    raw_ref_h5: Path
    projected_ref_cache: Path
    reference_expression_cache_file: Optional[Path]
    gene_universe_file: Path
    gene_qc_file: Path
    reference_row_filter_file: Path
    umap_ref_rows: Optional[np.ndarray] = None
    retrieval_he: Optional[RetrievalResult] = None
    retrieval_gt: Optional[RetrievalResult] = None
    he_valid_rows: Optional[np.ndarray] = None
    gt_valid_rows: Optional[np.ndarray] = None
    meets_minimum_reference_cells: bool = True
    meets_minimum_class_support: bool = True
    meets_minimum_aligned_genes_official: bool = True
    # Reason of an active declared coverage exception (slide scored but not used for (k, tau) tuning), else None
    coverage_declared_exception: Optional[str] = None



def _membership_groups_from_backbone_transform() -> Dict[str, str]:
    by_sig: Dict[Tuple[int, int], List[str]] = {}
    for m, cfg in BACKBONE_TRANSFORM.items():
        rs = int(cfg.get("resize_size", 0))
        cs = int(cfg.get("crop_size", 0))
        by_sig.setdefault((rs, cs), []).append(m)
    out: Dict[str, str] = {}
    for models in by_sig.values():
        models = sorted(models)
        gname = "_".join(models)
        for m in models:
            out[m] = gname
    return out



def _infer_modelgroup_from_he_encoder(he_encoder_path: Path) -> str:
    if get_full_path is None:
        raise RuntimeError("Could not import project utilities needed to infer modelgroup")

    enc = load_exported_he_encoder(str(he_encoder_path), device=torch.device("cpu"))
    backbone = str(getattr(enc, "backbone", "")).lower()
    if backbone:
        mapping = _membership_groups_from_backbone_transform()
        if backbone in mapping:
            return mapping[backbone]
    rs = int(getattr(enc, "resize_size", 0))
    cs = int(getattr(enc, "crop_size", 0))
    for m, cfg in BACKBONE_TRANSFORM.items():
        if int(cfg.get("resize_size", 0)) == rs and int(cfg.get("crop_size", 0)) == cs:
            return _membership_groups_from_backbone_transform()[m]
    return f"custom_r{rs}_c{cs}"



def _normalize_log1p_on_shared_genes(X: np.ndarray, target_sum: float) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    rs = X.sum(axis=1, keepdims=True)
    scale = np.divide(float(target_sum), rs, out=np.zeros_like(rs), where=rs > 0)
    Xn = X * scale
    return np.log1p(Xn).astype(np.float32, copy=False)



def _hash_int_array(x: np.ndarray) -> str:
    arr = np.ascontiguousarray(np.asarray(x, dtype=np.int64))
    return hashlib.sha1(arr.view(np.uint8)).hexdigest()



def _projected_reference_cache_stem(
    raw_ref_h5: Path,
    reference_h5ad: Path,
    st_cell_bundle_path: Path,
) -> str:
    """
    Build a collision-resistant cache filename for checkpoint-specific projected
    reference embeddings.
    """
    payload = {
        "raw_ref_h5": str(raw_ref_h5.resolve()),
        "reference_h5ad": str(reference_h5ad.resolve()),
        "st_cell_bundle": str(st_cell_bundle_path.resolve()),
    }
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    return f"{reference_h5ad.stem}_{digest}"



def _make_reference_expression_cache_path(
    shared_dir: Path,
    reference_h5ad_path: Path,
    selected_rows: np.ndarray,
    reference_var_indices: np.ndarray,
    normalize_target_sum: float,
) -> Path:
    h = hashlib.sha1()
    h.update(str(reference_h5ad_path.resolve()).encode("utf-8"))
    h.update(np.ascontiguousarray(np.asarray(selected_rows, dtype=np.int64)).view(np.uint8))
    h.update(np.ascontiguousarray(np.asarray(reference_var_indices, dtype=np.int64)).view(np.uint8))
    h.update(repr(float(normalize_target_sum)).encode("utf-8"))
    return shared_dir / "reference_expression" / f"{reference_h5ad_path.stem}_{h.hexdigest()[:16]}.h5"



def _decode_h5_string_array(arr: np.ndarray) -> List[str]:
    arr = np.asarray(arr)
    out: List[str] = []
    for x in arr:
        if isinstance(x, (bytes, np.bytes_)):
            out.append(bytes(x).decode("utf-8"))
        else:
            out.append(str(x))
    return out



def _fast_file_stamp(path: Path) -> str:
    st = path.stat()
    return f"{int(st.st_size)}:{int(st.st_mtime_ns)}"



def _runtime_hdf5_compression(cfg: Mapping[str, Any]) -> Tuple[Optional[str], Optional[int]]:
    value = cfg.get("runtime", {}).get("reference_expression_cache_compression", None)
    if value is None:
        return None, None
    value_s = str(value).strip().lower()
    if value_s in {"", "none", "null", "false"}:
        return None, None

    opts = cfg.get("runtime", {}).get("reference_expression_cache_compression_opts", None)
    return value_s, None if opts is None else int(opts)


def _backed_x_format(a: ad.AnnData) -> str:
    """Storage format of a backed AnnData X ('csr'/'csc' for sparse, else the type name)."""
    x = a.X
    fmt = getattr(x, "format", None)
    if fmt is not None:
        return str(fmt).lower()
    return type(x).__name__


def _backed_x_is_sparse_like(a: ad.AnnData) -> bool:
    """True for in-memory scipy sparse matrices and backed sparse (CSR/CSC) AnnData datasets."""
    x = a.X
    if sparse.issparse(x):
        return True
    fmt = str(getattr(x, "format", "")).lower()
    return fmt in {"csr", "csc"}


def _read_reference_expression_block(
    a: ad.AnnData,
    block_start: int,
    block_stop: int,
    rel_rows: np.ndarray,
    var_sorted: np.ndarray,
    var_sorted_list: List[int],
) -> np.ndarray:
    """
    Read selected rows and genes from a backed reference AnnData block
    (backed sparse: row slice of a.X, then column subset in memory).
    """
    rel_rows = np.asarray(rel_rows, dtype=np.int64)
    var_sorted = np.asarray(var_sorted, dtype=np.int64)

    if _backed_x_is_sparse_like(a):
        try:
            x_block = a.X[block_start:block_stop, :]
            x = x_block[rel_rows, :][:, var_sorted]
        except Exception:
            # Fallback if direct a.X slicing is not supported
            x_block = a[block_start:block_stop, var_sorted_list].X
            x = x_block[rel_rows]
    else:
        x_block = a[block_start:block_stop, var_sorted_list].X
        x = x_block[rel_rows]

    if sparse.issparse(x):
        x = x.toarray()
    return np.asarray(x, dtype=np.float32)



def _ensure_reference_expression_cache(
    cache_path: Path,
    reference_h5ad_path: Path,
    selected_rows: np.ndarray,
    reference_var_indices: np.ndarray,
    gene_names: Sequence[str],
    normalize_target_sum: float,
    logger=None,
    log_label: str = "",
    row_block_size: int = 8192,
    target_block_mb: float = 128.0,
    compression: Optional[str] = None,
    compression_opts: Optional[int] = None,
) -> Path:
    selected_rows = np.asarray(selected_rows, dtype=np.int64)
    reference_var_indices = np.asarray(reference_var_indices, dtype=np.int64)

    if selected_rows.ndim != 1:
        raise ValueError("selected_rows must be a 1D integer array")
    if reference_var_indices.ndim != 1:
        raise ValueError("reference_var_indices must be a 1D integer array")
    if selected_rows.size == 0:
        raise ValueError("selected_rows cannot be empty")
    if reference_var_indices.size == 0:
        raise ValueError("reference_var_indices cannot be empty")
    if np.any(selected_rows < 0):
        raise ValueError("selected_rows contains negative row indices")
    if np.any(reference_var_indices < 0):
        raise ValueError("reference_var_indices contains negative gene indices")
    if selected_rows.size > 1 and np.any(selected_rows[1:] <= selected_rows[:-1]):
        raise ValueError(
            "selected_rows must be strictly increasing and unique. "
            "The caller should pass np.unique(...) rows in reference order."
        )

    selected_rows_hash = _hash_int_array(selected_rows)
    reference_var_indices_hash = _hash_int_array(reference_var_indices)
    reference_h5ad_stamp = _fast_file_stamp(reference_h5ad_path)
    if cache_path.exists():
        cache_ok = False
        try:
            with h5py.File(cache_path, "r") as f:
                cached_genes = _decode_h5_string_array(f["gene"][:]) if "gene" in f else []
                cache_ok = (
                    "ref_expr_log1p" in f
                    and list(cached_genes) == list(gene_names)
                    and str(f.attrs.get("reference_h5ad", "")) == str(reference_h5ad_path)
                    and int(f.attrs.get("n_selected_rows", -1)) == int(selected_rows.size)
                    and str(f.attrs.get("selected_rows_hash", "")) == selected_rows_hash
                    and str(f.attrs.get("reference_var_indices_hash", "")) == reference_var_indices_hash
                    and str(f.attrs.get("reference_h5ad_stamp", "")) == reference_h5ad_stamp
                    and int(f.attrs.get("complete", 0)) == 1
                    and float(f.attrs.get("normalize_target_sum", np.nan)) == float(normalize_target_sum)
                )
                if cache_ok:
                    return cache_path
        except Exception:
            cache_ok = False
        if not cache_ok:
            cache_path.unlink()

    a = ad.read_h5ad(reference_h5ad_path, backed="r")
    try:
        ensure_dir(cache_path.parent)
        n_rows = int(selected_rows.size)
        n_genes = int(reference_var_indices.size)
        n_obs = int(a.n_obs)
        n_vars = int(a.n_vars)

        if int(selected_rows[-1]) >= n_obs:
            raise IndexError(
                f"selected_rows contains row {int(selected_rows[-1])}, "
                f"but reference has only {n_obs} rows"
            )
        if int(reference_var_indices.max()) >= n_vars:
            raise IndexError(
                f"reference_var_indices contains var {int(reference_var_indices.max())}, "
                f"but reference has only {n_vars} variables"
            )

        # Read with sorted variable indices, then restore the original gene order
        var_order = np.argsort(reference_var_indices, kind="mergesort")
        var_sorted = reference_var_indices[var_order]
        var_restore = np.empty_like(var_order)
        var_restore[var_order] = np.arange(var_order.size, dtype=np.int64)
        var_sorted_list = var_sorted.astype(np.int64, copy=False).tolist()
        restore_gene_order = not np.array_equal(var_order, np.arange(var_order.size))

        # Bound the dense temporary memory (smaller row blocks for wide gene universes)
        requested_row_block = max(1, int(row_block_size))
        target_bytes = max(16.0, float(target_block_mb)) * (1024 ** 2)
        max_rows_by_memory = max(1, int(target_bytes // max(1, n_genes * np.dtype(np.float32).itemsize)))
        scan_row_block = max(1, min(requested_row_block, max_rows_by_memory))

        row_span = int(selected_rows[-1] - selected_rows[0] + 1)
        selected_density = float(selected_rows.size) / float(max(1, row_span))
        x_format = _backed_x_format(a)

        gene_chunk = min(512, max(1, n_genes))
        tmp_path = cache_path.with_name(f"{cache_path.name}.tmp.{os.getpid()}")
        if tmp_path.exists():
            tmp_path.unlink()

        if logger is not None:
            logger.info(
                "Building reference-expression cache %s rows=%d genes=%d size=%.2f GB "
                "row_block=%d target_block_mb=%.1f selected_row_span=%d selected_density=%.4f "
                "x_format=%s file=%s",
                log_label,
                n_rows,
                n_genes,
                (n_rows * n_genes * np.dtype(np.float32).itemsize) / (1024 ** 3),
                scan_row_block,
                float(target_block_mb),
                row_span,
                selected_density,
                x_format,
                cache_path,
            )

        try:
            with h5py.File(tmp_path, "w") as f:
                dset_kwargs = {
                    "name": "ref_expr_log1p",
                    "shape": (n_rows, n_genes),
                    "dtype": np.float32,
                    "chunks": (min(max(1, n_rows), scan_row_block), gene_chunk),
                }
                if compression is not None:
                    dset_kwargs["compression"] = compression
                    if compression_opts is not None:
                        dset_kwargs["compression_opts"] = compression_opts
                    dset_kwargs["shuffle"] = True
                dset = f.create_dataset(**dset_kwargs)
                f.create_dataset(
                    "gene",
                    data=np.asarray(list(gene_names), dtype=object),
                    dtype=h5py.string_dtype("utf-8"),
                )
                f.attrs["reference_h5ad"] = str(reference_h5ad_path)
                f.attrs["reference_h5ad_stamp"] = reference_h5ad_stamp
                f.attrs["n_selected_rows"] = int(selected_rows.size)
                f.attrs["selected_rows_hash"] = selected_rows_hash
                f.attrs["reference_var_indices_hash"] = reference_var_indices_hash
                f.attrs["normalize_target_sum"] = float(normalize_target_sum)
                f.attrs["complete"] = 0

                ptr = 0
                next_report = 0.10
                while ptr < n_rows:
                    block_start = int(selected_rows[ptr])
                    block_stop = min(n_obs, block_start + scan_row_block)
                    next_ptr = int(np.searchsorted(selected_rows, block_stop, side="left"))
                    if next_ptr <= ptr:
                        next_ptr = ptr + 1
                        block_stop = min(n_obs, int(selected_rows[ptr]) + 1)

                    rel_rows = selected_rows[ptr:next_ptr] - block_start

                    X = _read_reference_expression_block(
                        a=a,
                        block_start=block_start,
                        block_stop=block_stop,
                        rel_rows=rel_rows,
                        var_sorted=var_sorted,
                        var_sorted_list=var_sorted_list,
                    )

                    if restore_gene_order:
                        X = X[:, var_restore]

                    X = _normalize_log1p_on_shared_genes(X, target_sum=normalize_target_sum)
                    dset[ptr:next_ptr] = X

                    ptr = next_ptr
                    frac = float(ptr) / float(n_rows)
                    if logger is not None and (ptr == n_rows or frac + 1e-12 >= next_report):
                        logger.info(
                            "Reference-expression cache progress %s rows=%d/%d %.1f%%",
                            log_label,
                            ptr,
                            n_rows,
                            100.0 * frac,
                        )
                        while next_report <= frac + 1e-12:
                            next_report += 0.10

                f.attrs["complete"] = 1

            os.replace(tmp_path, cache_path)
        except Exception:
            if tmp_path.exists():
                tmp_path.unlink()
            raise
    finally:
        if getattr(a, "file", None) is not None:
            a.file.close()
    return cache_path



def _load_reference_expression_subset_to_memory(
    reference_h5ad_path: Path,
    selected_rows: np.ndarray,
    reference_var_indices: np.ndarray,
    normalize_target_sum: float,
    logger=None,
    log_label: str = "",
    row_block_size: int = 8192,
    target_block_mb: float = 128.0,
) -> np.ndarray:
    """Normalized log1p reference expression of the selected rows (sorted unique) and genes, loaded in RAM (for tuning)."""
    selected_rows = np.asarray(selected_rows, dtype=np.int64)
    reference_var_indices = np.asarray(reference_var_indices, dtype=np.int64)

    if selected_rows.ndim != 1 or reference_var_indices.ndim != 1:
        raise ValueError("selected_rows and reference_var_indices must be 1D arrays")
    if selected_rows.size == 0:
        raise ValueError("selected_rows cannot be empty")
    if reference_var_indices.size == 0:
        raise ValueError("reference_var_indices cannot be empty")
    if selected_rows.size > 1 and np.any(selected_rows[1:] <= selected_rows[:-1]):
        raise ValueError("selected_rows must be sorted unique")

    a = ad.read_h5ad(reference_h5ad_path, backed="r")
    try:
        n_obs = int(a.n_obs)
        n_vars = int(a.n_vars)
        n_rows = int(selected_rows.size)
        n_genes = int(reference_var_indices.size)

        if int(selected_rows[-1]) >= n_obs:
            raise IndexError(f"selected row {int(selected_rows[-1])} exceeds reference n_obs={n_obs}")
        if int(reference_var_indices.max()) >= n_vars:
            raise IndexError(f"reference var index {int(reference_var_indices.max())} exceeds n_vars={n_vars}")

        var_order = np.argsort(reference_var_indices, kind="mergesort")
        var_sorted = reference_var_indices[var_order]
        var_restore = np.empty_like(var_order)
        var_restore[var_order] = np.arange(var_order.size, dtype=np.int64)
        restore_gene_order = not np.array_equal(var_order, np.arange(var_order.size))
        var_sorted_list = var_sorted.astype(np.int64, copy=False).tolist()

        target_bytes = max(16.0, float(target_block_mb)) * (1024 ** 2)
        max_rows_by_memory = max(1, int(target_bytes // max(1, n_genes * np.dtype(np.float32).itemsize)))
        scan_row_block = max(1, min(int(row_block_size), max_rows_by_memory))

        row_span = int(selected_rows[-1] - selected_rows[0] + 1)
        selected_density = float(selected_rows.size) / float(max(1, row_span))
        x_format = _backed_x_format(a)

        out = np.empty((n_rows, n_genes), dtype=np.float32)

        if logger is not None:
            logger.info(
                "Materializing tuning reference expression in RAM %s rows=%d genes=%d size=%.2f GB "
                "row_block=%d selected_row_span=%d selected_density=%.4f x_format=%s",
                log_label,
                n_rows,
                n_genes,
                out.nbytes / (1024 ** 3),
                scan_row_block,
                row_span,
                selected_density,
                x_format,
            )

        ptr = 0
        next_report = 0.10
        while ptr < n_rows:
            block_start = int(selected_rows[ptr])
            block_stop = min(n_obs, block_start + scan_row_block)
            next_ptr = int(np.searchsorted(selected_rows, block_stop, side="left"))

            if next_ptr <= ptr:
                next_ptr = ptr + 1
                block_stop = min(n_obs, int(selected_rows[ptr]) + 1)

            rel_rows = selected_rows[ptr:next_ptr] - block_start

            X = _read_reference_expression_block(
                a=a,
                block_start=block_start,
                block_stop=block_stop,
                rel_rows=rel_rows,
                var_sorted=var_sorted,
                var_sorted_list=var_sorted_list,
            )
            if restore_gene_order:
                X = X[:, var_restore]

            out[ptr:next_ptr] = _normalize_log1p_on_shared_genes(
                X,
                target_sum=normalize_target_sum,
            )

            ptr = next_ptr
            frac = float(ptr) / float(n_rows)
            if logger is not None and (ptr == n_rows or frac + 1e-12 >= next_report):
                logger.info(
                    "Tuning reference-expression RAM materialization progress %s rows=%d/%d %.1f%%",
                    log_label,
                    ptr,
                    n_rows,
                    100.0 * frac,
                )
                while next_report <= frac + 1e-12:
                    next_report += 0.10

        return out

    finally:
        if getattr(a, "file", None) is not None:
            a.file.close()



def _attach_reference_expression_subset(
    slide: SlideData,
    shared_dir: Path,
    k: int,
    cfg: Mapping[str, Any],
    logger,
    query_paths: Sequence[str] = ("he", "gt"),
    query_row_mask: Optional[np.ndarray] = None,
    materialize_in_memory: bool = False,
) -> None:
    if slide.retrieval_he is None or slide.retrieval_gt is None:
        raise RuntimeError(f"Retrieval cache missing before expression attachment for slide {slide.slide_id}")

    query_paths = tuple(str(x) for x in query_paths)
    bad_paths = sorted(set(query_paths) - {"he", "gt"})
    if bad_paths:
        raise ValueError(f"Unsupported query_paths={bad_paths}; expected only 'he' and/or 'gt'")

    if query_row_mask is not None:
        query_row_mask = np.asarray(query_row_mask, dtype=bool)
        if query_row_mask.shape[0] != slide.cells.shape[0]:
            raise ValueError(
                f"query_row_mask has length {query_row_mask.shape[0]}, "
                f"expected {slide.cells.shape[0]} for slide={slide.slide_id}"
            )

    ref_parts: List[np.ndarray] = []
    k = int(k)

    if "he" in query_paths and slide.retrieval_he.indices.size > 0:
        if query_row_mask is None:
            he_keep = np.ones(slide.retrieval_he.indices.shape[0], dtype=bool)
        else:
            he_keep = query_row_mask[slide.he_valid_rows]
        if np.any(he_keep):
            ref_parts.append(slide.retrieval_he.indices[he_keep, :k].reshape(-1))

    if "gt" in query_paths and slide.retrieval_gt.indices.size > 0:
        if query_row_mask is None:
            gt_keep = np.ones(slide.retrieval_gt.indices.shape[0], dtype=bool)
        else:
            gt_keep = query_row_mask[slide.gt_valid_rows]
        if np.any(gt_keep):
            ref_parts.append(slide.retrieval_gt.indices[gt_keep, :k].reshape(-1))

    used_ref_rows = (
        np.unique(np.concatenate(ref_parts, axis=0)).astype(np.int64, copy=False)
        if ref_parts
        else np.empty((0,), dtype=np.int64)
    )

    if used_ref_rows.size == 0:
        slide.ref_expr = None
        slide.ref_expr_global_to_local = np.full(slide.ref_z.shape[0], -1, dtype=np.int64)
        slide.reference_expression_cache_file = None
        logger.warning(
            "No valid query rows for slide=%s at k=%d; skipping reference-expression cache creation. "
            "Expression metrics for empty populations will be reported as undefined.",
            slide.slide_id,
            int(k),
        )
        return

    global_to_local = np.full(slide.ref_z.shape[0], -1, dtype=np.int64)
    global_to_local[used_ref_rows] = np.arange(used_ref_rows.size, dtype=np.int64)

    if materialize_in_memory:
        estimated_bytes = int(used_ref_rows.size) * int(len(slide.gene_names)) * np.dtype(np.float32).itemsize
        max_gb = float(cfg.get("runtime", {}).get("tuning_ref_expr_max_memory_gb", 8.0))
        max_bytes = int(max_gb * (1024 ** 3))

        if estimated_bytes <= max_bytes:
            slide.ref_expr = _load_reference_expression_subset_to_memory(
                reference_h5ad_path=slide.reference_h5ad,
                selected_rows=slide.reference_h5ad_valid_rows[used_ref_rows],
                reference_var_indices=slide.reference_var_indices,
                normalize_target_sum=float(cfg["expression"]["normalize_target_sum"]),
                logger=logger,
                log_label=f"slide={slide.slide_id}",
                row_block_size=int(cfg.get("runtime", {}).get("reference_expression_cache_row_block_size", 8192)),
                target_block_mb=float(cfg.get("runtime", {}).get("reference_expression_cache_target_block_mb", 128.0)),
            )
            slide.ref_expr_global_to_local = global_to_local
            slide.reference_expression_cache_file = None
            logger.info(
                "Materialized tuning reference expression in RAM for slide=%s using %d/%d reference rows at k=%d",
                slide.slide_id,
                int(used_ref_rows.size),
                int(slide.ref_z.shape[0]),
                int(k),
            )
            return

        logger.info(
            "Tuning reference expression for slide=%s is %.2f GB > %.2f GB RAM limit; falling back to HDF5 cache.",
            slide.slide_id,
            estimated_bytes / (1024 ** 3),
            max_gb,
        )

    cache_path = _make_reference_expression_cache_path(
        shared_dir=shared_dir,
        reference_h5ad_path=slide.reference_h5ad,
        selected_rows=slide.reference_h5ad_valid_rows[used_ref_rows],
        reference_var_indices=slide.reference_var_indices,
        normalize_target_sum=float(cfg["expression"]["normalize_target_sum"]),
    )
    logger.info(
        "Ensuring reference-expression cache for slide=%s used_ref_rows=%d ref_cells=%d genes=%d k=%d",
        slide.slide_id,
        int(used_ref_rows.size),
        int(slide.ref_z.shape[0]),
        int(len(slide.gene_names)),
        int(k),
    )
    cache_compression, cache_compression_opts = _runtime_hdf5_compression(cfg)
    _ensure_reference_expression_cache(
        cache_path=cache_path,
        reference_h5ad_path=slide.reference_h5ad,
        selected_rows=slide.reference_h5ad_valid_rows[used_ref_rows],
        reference_var_indices=slide.reference_var_indices,
        gene_names=slide.gene_names,
        normalize_target_sum=float(cfg["expression"]["normalize_target_sum"]),
        logger=logger,
        log_label=f"slide={slide.slide_id}",
        row_block_size=int(cfg.get("runtime", {}).get("reference_expression_cache_row_block_size", 8192)),
        target_block_mb=float(cfg.get("runtime", {}).get("reference_expression_cache_target_block_mb", 128.0)),
        compression=cache_compression,
        compression_opts=cache_compression_opts,
    )
    slide.ref_expr = None
    slide.ref_expr_global_to_local = global_to_local
    slide.reference_expression_cache_file = cache_path
    logger.info(
        "Prepared reference-expression cache for slide=%s using %d/%d retrieved reference rows at k=%d",
        slide.slide_id,
        int(used_ref_rows.size),
        int(slide.ref_z.shape[0]),
        int(k),
    )



def _release_reference_expression_subset(slide: SlideData) -> None:
    slide.ref_expr = None
    slide.ref_expr_global_to_local = None



def _rank_score(s: pd.Series) -> pd.Series:
    """Percentile rank score (larger is better); NaNs get a value below the worst observed one."""
    s2 = pd.to_numeric(s, errors="coerce").astype(float)
    finite = s2[np.isfinite(s2)]
    if not finite.empty:
        min_value = float(finite.min())
        floor = min_value - abs(min_value) - 1.0
    else:
        floor = -1.0e9
    return s2.fillna(floor).rank(method="average", pct=True)


def _paired_rowwise_pearson_fast(
    a: np.ndarray,
    b: np.ndarray,
    variance_eps: float,
) -> Tuple[np.ndarray, np.ndarray]:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    am = a.mean(axis=1, keepdims=True)
    bm = b.mean(axis=1, keepdims=True)
    ac = a - am
    bc = b - bm
    av = np.mean(ac * ac, axis=1)
    bv = np.mean(bc * bc, axis=1)
    valid = (av >= variance_eps) & (bv >= variance_eps)
    corr = np.full(a.shape[0], np.nan, dtype=np.float32)
    denom = np.sqrt(av[valid] * bv[valid])
    denom = np.maximum(denom, np.finfo(np.float32).tiny)
    corr[valid] = np.mean(ac[valid] * bc[valid], axis=1) / denom
    return corr, valid


def _pearson_metrics_for_tuning(
    pred: np.ndarray,
    gt: np.ndarray,
    cfg: Mapping[str, Any],
) -> Tuple[float, float]:
    """(gene_wise_pearson, mean_cellwise_pearson) of a prediction, for tuning."""
    pred = np.asarray(pred, dtype=np.float32)
    gt = np.asarray(gt, dtype=np.float32)
    if pred.shape != gt.shape or pred.shape[0] == 0:
        return np.nan, np.nan

    variance_eps = float(cfg["expression"]["variance_eps"])
    min_cells_cellwise = int(cfg["expression"]["min_valid_cells_for_cellwise_corr"])
    min_cells_genewise = int(cfg["expression"]["min_valid_cells_for_genewise_corr"])
    min_genes_summary = int(cfg["expression"]["min_valid_genes_for_slide_summary"])

    pearson_cell, valid_cell = _paired_rowwise_pearson_fast(pred, gt, variance_eps=variance_eps)
    cell_p = (
        float(np.nanmean(pearson_cell[valid_cell]))
        if int(valid_cell.sum()) >= min_cells_cellwise
        else np.nan
    )

    if pred.shape[0] < min_cells_genewise:
        return np.nan, cell_p

    pearson_gene, valid_gene = _paired_rowwise_pearson_fast(pred.T, gt.T, variance_eps=variance_eps)
    gene_p = (
        float(np.nanmean(pearson_gene[valid_gene]))
        if int(valid_gene.sum()) >= min_genes_summary
        else np.nan
    )
    return gene_p, cell_p



def _tuning_reads_expression(cfg: Mapping[str, Any]) -> bool:
    """True if the (k, tau) selection uses an expression metric (composite objective or expression tie-break)."""
    tuning = cfg.get("tuning", {}) or {}
    if str(tuning.get("objective", "mapped_macro_f1_matched_covered")) == "composite_rank_matched_covered_he":
        return True
    return any(
        str(tb) in {"gene_wise_pearson", "mean_cellwise_pearson"}
        for tb in (tuning.get("tie_breaks", []) or [])
    )


def _evaluate_tuning_grid_point_he(
    slide: SlideData,
    k: int,
    tau: float,
    label_order: Sequence[str],
    cfg: Mapping[str, Any],
    objective: str,
    ref_expr_local: Optional[np.ndarray],
    reads_expression: bool,
) -> Dict[str, Any]:
    """Evaluate one (k, tau) grid point for tuning (objective population and selection metrics only)."""
    if slide.retrieval_he is None or slide.he_valid_rows is None:
        raise RuntimeError(f"Missing H&E retrieval for tuning slide={slide.slide_id}")

    if objective == "mapped_macro_f1_matched_all":
        cls_mask = slide.matched_valid & slide.gt_mapped_label_scorable
        expr_mask = slide.matched_valid & slide.gt_expr_present
        objective_population = "matched_all_he"
    else:
        cls_mask = slide.matched_valid & slide.gt_mapped_label_scorable & slide.covered
        expr_mask = slide.matched_valid & slide.covered & slide.gt_expr_present
        objective_population = "matched_covered_he"

    cls_rows = np.flatnonzero(cls_mask).astype(np.int64, copy=False)
    expr_rows = np.flatnonzero(expr_mask).astype(np.int64, copy=False)

    row_to_valid_pos = np.full(slide.cells.shape[0], -1, dtype=np.int64)
    row_to_valid_pos[slide.he_valid_rows] = np.arange(slide.he_valid_rows.size, dtype=np.int64)

    k = min(int(k), int(slide.retrieval_he.indices.shape[1]))

    macro_f1 = np.nan
    if cls_rows.size > 0:
        cls_pos = row_to_valid_pos[cls_rows]
        cls_keep = cls_pos >= 0
        cls_rows = cls_rows[cls_keep]
        cls_pos = cls_pos[cls_keep]
        if cls_rows.size > 0:
            nn_idx = slide.retrieval_he.indices[cls_pos, :k]
            nn_sim = slide.retrieval_he.similarities[cls_pos, :k]
            weights = weights_from_similarities(nn_sim, tau=float(tau))
            class_scores = class_scores_from_neighbors(
                nn_idx,
                weights,
                slide.ref_label_ids,
                n_classes=len(label_order),
            )
            pred = predicted_labels_from_scores(class_scores)
            macro_f1 = float(
                classification_metrics_from_ids(
                    y_true=slide.gt_label_ids[cls_rows],
                    y_pred=pred,
                    label_order=label_order,
                    # Same per-slide macro floor as the headline (_evaluate_query_path)
                    macro_min_support=macro_min_class_support(cfg),
                ).metrics["macro_f1"]
            )

    gene_p = np.nan
    cell_p = np.nan

    # Keep only valid H&E query rows (retrieval_he is indexed by slide.he_valid_rows)
    if expr_rows.size > 0:
        expr_pos = row_to_valid_pos[expr_rows]
        expr_keep = expr_pos >= 0
        expr_rows = expr_rows[expr_keep]
        expr_pos = expr_pos[expr_keep]
    else:
        expr_pos = np.empty((0,), dtype=np.int64)
    n_valid_expression_cells = int(expr_rows.size)

    tuning_gene_panel = str(cfg.get("gene_panels", {}).get("tuning", {}).get("expression_gene_panel", "G_base_qc"))
    min_tuning_genes = int(
        cfg.get("gene_panels", {})
        .get("tuning", {})
        .get("min_genes_for_G_base_qc_tuning", 30)
    )
    if tuning_gene_panel not in slide.gene_panel_masks:
        raise KeyError(f"Tuning gene panel '{tuning_gene_panel}' not found for slide={slide.slide_id}")
    tuning_gene_mask = slide.gene_panel_masks[tuning_gene_panel]
    n_tuning_genes = int(tuning_gene_mask.sum())
    tuning_panel_status = slide.gene_panel_meta[tuning_gene_panel]["gene_panel_status"]
    tuning_expression_panel_valid = n_tuning_genes >= min_tuning_genes

    # gene_p / cell_p stay NaN when the tuning does not read expression (reads_expression, from the caller)
    if (
        reads_expression
        and expr_rows.size > 0
        and tuning_expression_panel_valid
    ):
        # reads_expression includes expression.enabled: a missing matrix means a config mismatch
        if slide.gt_expr is None:
            raise RuntimeError(
                f"Slide {slide.slide_id}: the tuning pass reads expression, but the GT expression "
                "matrix was not loaded (expression.enabled was False when the slide was prepared)."
            )
        nn_idx = slide.retrieval_he.indices[expr_pos, :k]
        nn_sim = slide.retrieval_he.similarities[expr_pos, :k]
        weights = weights_from_similarities(nn_sim, tau=float(tau))

        if slide.ref_expr_global_to_local is None:
            raise RuntimeError(f"Missing reference-expression row map for tuning slide={slide.slide_id}")
        nn_idx_local = slide.ref_expr_global_to_local[nn_idx]
        if np.any(nn_idx_local < 0):
            raise RuntimeError(f"Reference-expression local remap failed for tuning slide={slide.slide_id}")

        if ref_expr_local is not None:
            pred_expr = weighted_expression_prediction(
                neighbor_indices=nn_idx_local,
                weights=weights,
                ref_expr=ref_expr_local,
                query_batch_size=int(cfg.get("runtime", {}).get("expression_query_batch_size", 512)),
            )
        else:
            if slide.reference_expression_cache_file is None:
                raise RuntimeError(f"Missing reference-expression cache for tuning slide={slide.slide_id}")
            pred_expr = weighted_expression_prediction_h5(
                neighbor_indices=nn_idx_local,
                weights=weights,
                cache_h5_path=str(slide.reference_expression_cache_file),
                query_batch_size=int(cfg.get("runtime", {}).get("expression_query_batch_size", 512)),
            )

        gene_p, cell_p = _pearson_metrics_for_tuning(
            pred=pred_expr[:, tuning_gene_mask],
            gt=slide.gt_expr[expr_rows][:, tuning_gene_mask],
            cfg=cfg,
        )

    return {
        "objective_population": objective_population,
        "macro_f1": macro_f1,
        "gene_wise_pearson": gene_p,
        "mean_cellwise_pearson": cell_p,
        "n_classification_cells": int(cls_rows.size),
        "n_valid_expression_cells": int(n_valid_expression_cells),
        "tuning_gene_panel": tuning_gene_panel,
        "n_tuning_genes": int(n_tuning_genes),
        "tuning_gene_panel_status": tuning_panel_status,
        "min_tuning_genes": int(min_tuning_genes),
    }



def _save_retrieval_cache(
    out_dir: Path,
    slide_id: str,
    path: str,
    valid_rows: np.ndarray,
    retrieval: RetrievalResult,
) -> None:
    """Save the top-Kmax retrieval cache (valid rows, neighbor indices, similarities)."""
    ensure_dir(out_dir)
    np.savez(
        out_dir / f"{slide_id}_{path}_topKmax.npz",
        valid_rows=np.asarray(valid_rows, dtype=np.int64),
        neighbor_indices=np.asarray(retrieval.indices, dtype=np.int64),
        similarities=np.asarray(retrieval.similarities, dtype=np.float32),
    )



def _subset_pooled_classification_metrics(
    preds_df: pd.DataFrame,
    label_order: Sequence[str],
    label_to_idx: Mapping[str, int],
) -> pd.DataFrame:
    if preds_df.empty:
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []
    for subset, sub in preds_df.groupby("subset", dropna=False):
        y_true_full = labels_to_ids(sub["gt_label"].astype(str).tolist(), label_to_idx)
        if "gt_mapped_label_scorable" in sub.columns:
            mapped_scorable = sub["gt_mapped_label_scorable"].astype(bool).to_numpy()
        else:
            mapped_scorable = np.ones(sub.shape[0], dtype=bool)

        for raw_path, report_path, pred_col, valid_col in [
            ("he", "he", "he_pred_label", "he_valid"),
            ("gt", "gt_anchor", "gt_pred_label", "gt_valid"),
        ]:
            valid = sub[valid_col].astype(bool).to_numpy()
            covered = sub["covered"].astype(bool).to_numpy()
            matched = sub["matched_valid"].astype(bool).to_numpy()
            y_pred_full = labels_to_ids(sub[pred_col].astype(str).tolist(), label_to_idx)

            populations = {
                f"{raw_path}_valid_all": valid & mapped_scorable,
                f"{raw_path}_valid_covered": valid & mapped_scorable & covered,
                f"matched_all_{raw_path}": matched & mapped_scorable,
                f"matched_covered_{raw_path}": matched & mapped_scorable & covered,
            }
            for population, mask in populations.items():
                idx = np.flatnonzero(mask)
                if idx.size == 0:
                    continue
                cls = classification_metrics_from_ids(
                    y_true=y_true_full[idx],
                    y_pred=y_pred_full[idx],
                    label_order=label_order,
                )
                for metric_name, value in cls.metrics.items():
                    rows.append(
                        {
                            "subset": subset,
                            "path": report_path,
                            "population": population,
                            "metric_group": "classification",
                            "metric": metric_name,
                            "value": value,
                            "summary_scope": "pooled_over_cells",
                        }
                    )
    return pd.DataFrame(rows)



def _subset_pooled_expression_metrics(expr_cell_tables: Sequence[pd.DataFrame]) -> pd.DataFrame:
    """Cell-wise expression metrics pooled over the cells of each subset (gene-wise metrics stay slide-macro)."""
    if not expr_cell_tables:
        return pd.DataFrame()

    df = pd.concat(expr_cell_tables, axis=0, ignore_index=True)
    rows: List[Dict[str, Any]] = []
    for (subset, path, population, gene_panel), sub in df.groupby(
        ["subset", "path", "population", "gene_panel"],
        dropna=False,
    ):
        panel_meta = {
            c: sub[c].iloc[0]
            for c in GENE_PANEL_META_COLUMNS
            if c in sub.columns
        }

        def _append(metric: str, value: float) -> None:
            row = {
                "subset": subset,
                "path": path,
                "population": population,
                "metric_group": "expression_pooled_over_cells",
                "metric": metric,
                "value": value,
                "summary_scope": "pooled_over_cells",
            }
            row.update(panel_meta)
            rows.append(row)

        valid_p = sub["valid_pearson"].astype(bool).to_numpy()
        valid_s = sub["valid_spearman"].astype(bool).to_numpy()
        _append("n_cells", float(sub.shape[0]))
        _append("mean_mse", float(sub["mse"].mean()) if sub.shape[0] > 0 else np.nan)
        _append("mean_mae", float(sub["mae"].mean()) if sub.shape[0] > 0 else np.nan)
        _append("n_valid_cellwise_pearson", float(valid_p.sum()))
        _append("n_valid_cellwise_spearman", float(valid_s.sum()))
        _append("mean_cellwise_pearson", float(sub.loc[valid_p, "pearson"].mean()) if valid_p.any() else np.nan)
        _append("mean_cellwise_spearman", float(sub.loc[valid_s, "spearman"].mean()) if valid_s.any() else np.nan)
    return pd.DataFrame(rows)


SLIDE_MACRO_GROUP_COLUMNS = [
    "path",
    "population",
    "metric_group",
    "gene_panel",
    "metric",
]


def _nunique_nonnull(values: pd.Series) -> int:
    return int(pd.Series(values).dropna().nunique())


def _slide_macro_metric_summary(
    df: pd.DataFrame,
    group_cols: Sequence[str] = SLIDE_MACRO_GROUP_COLUMNS,
) -> pd.DataFrame:
    """Slide-macro metric summary (mean over slides, NaNs skipped) with the counts of finite/NaN contributions."""
    if df.empty:
        return pd.DataFrame()

    required = set(group_cols) | {"slide_id", "value"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise KeyError(f"Missing required columns for slide-macro summary: {missing}")

    tmp = df.copy()
    tmp["_value_numeric"] = pd.to_numeric(tmp["value"], errors="coerce")
    finite = np.isfinite(tmp["_value_numeric"].to_numpy(dtype=float))
    tmp["_value_is_finite"] = finite
    tmp["_finite_slide_id"] = tmp["slide_id"].where(tmp["_value_is_finite"], pd.NA)
    tmp["_nan_slide_id"] = tmp["slide_id"].where(~tmp["_value_is_finite"], pd.NA)

    out = (
        tmp.groupby(list(group_cols), dropna=False)
        .agg(
            value=("_value_numeric", "mean"),
            n_metric_rows=("value", "size"),
            n_finite_metric_rows=("_value_is_finite", "sum"),
            n_slides=("slide_id", "nunique"),
            n_slides_with_finite_value=("_finite_slide_id", _nunique_nonnull),
            n_slides_with_nan_value=("_nan_slide_id", _nunique_nonnull),
        )
        .reset_index()
    )
    out["n_nan_metric_rows"] = (
        out["n_metric_rows"].astype(int) - out["n_finite_metric_rows"].astype(int)
    )

    ordered_cols = (
        list(group_cols)
        + [
            "value",
            "n_metric_rows",
            "n_finite_metric_rows",
            "n_nan_metric_rows",
            "n_slides",
            "n_slides_with_finite_value",
            "n_slides_with_nan_value",
        ]
    )
    return out[ordered_cols]


# gene_panel value of the non-expression metric rows (the column must exist even without expression rows)
NO_GENE_PANEL = None

GENE_PANEL_META_COLUMNS = [
    "gene_panel",
    "gene_panel_requested_n",
    "gene_panel_effective_n",
    "gene_panel_status",
    "n_genes_raw_shared",
    "n_genes_qc_shared",
    "n_genes_g_base_intersection_raw",
    "n_genes_g_base_qc",
]

HE_SELECTED_REPORT_COLUMNS = [
    "cell_id_int",
    "selected_alpha",
    "selected_patch_cell_row_index",
    "selected_patch_id",
    "selected_patch_index",
    "n_candidate_patches",
]

HE_SELECTED_OPTIONAL_DEFAULTS = {
    "selected_patch_id": pd.NA,
    "selected_patch_index": pd.NA,
    "n_candidate_patches": pd.NA,
}


def _requested_n_from_panel_name(panel_name: str) -> Optional[int]:
    prefix = "G_eval_slide_hvg"
    if str(panel_name).startswith(prefix):
        return int(str(panel_name)[len(prefix):])
    return None


def _fixed_panel_status(n_genes: int, min_genes: int) -> str:
    if int(n_genes) == 0:
        return "empty_after_qc"
    if int(n_genes) < int(min_genes):
        return "insufficient_genes"
    return "ok"


def _hvg_panel_status(n_genes: int, requested_n: int) -> str:
    if int(n_genes) == 0:
        return "empty_after_qc"
    if int(n_genes) < int(requested_n):
        return "fewer_than_requested"
    return "ok"


def _load_gene_panel_masks_for_slide(
    shared_dir: Path,
    slide_id: str,
    gene_names: Sequence[str],
    cfg: Mapping[str, Any],
) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, Any]], Path]:
    gene_qc_file = shared_dir / "gene_qc" / f"{slide_id}.parquet"
    if not gene_qc_file.exists():
        raise FileNotFoundError(
            f"Missing gene QC artifact for slide={slide_id}: {gene_qc_file}. "
            "Run _2_prepare_shared_artifacts.py."
        )

    qdf = pd.read_parquet(gene_qc_file)
    qc_genes = qdf["gene"].astype(str).tolist()
    if qc_genes != list(map(str, gene_names)):
        raise RuntimeError(f"Gene QC artifact is not aligned to G_s_raw for slide={slide_id}")

    n_raw = int(qdf.shape[0])
    n_qc = int(qdf["in_G_s_qc"].astype(bool).sum()) if "in_G_s_qc" in qdf.columns else 0
    n_base_raw = int(qdf["in_G_base"].astype(bool).sum()) if "in_G_base" in qdf.columns else 0
    n_base_qc = int(qdf["in_G_base_qc"].astype(bool).sum()) if "in_G_base_qc" in qdf.columns else 0
    min_official = int(cfg["expression"]["min_aligned_genes_official"])

    enabled = [str(x) for x in cfg.get("gene_panels", {}).get("enabled", ["G_s_qc", "G_base_qc"])]
    masks: Dict[str, np.ndarray] = {}
    meta: Dict[str, Dict[str, Any]] = {}

    for panel in enabled:
        if panel == "G_s_raw":
            mask = np.ones(n_raw, dtype=bool)
            requested_n = None
            status = "ok" if n_raw > 0 else "empty_after_qc"
        elif panel == "G_s_qc":
            mask = qdf["in_G_s_qc"].astype(bool).to_numpy()
            requested_n = None
            status = _fixed_panel_status(int(mask.sum()), min_official)
        elif panel == "G_base_qc":
            mask = qdf["in_G_base_qc"].astype(bool).to_numpy()
            requested_n = None
            status = _fixed_panel_status(int(mask.sum()), min_official)
        elif panel.startswith("G_eval_slide_hvg"):
            col = f"in_{panel}"
            if col not in qdf.columns:
                raise KeyError(f"Missing expected HVG panel column '{col}' in {gene_qc_file}")
            requested_n = _requested_n_from_panel_name(panel)
            if requested_n is None:
                raise ValueError(f"Could not parse requested N from panel name: {panel}")
            mask = qdf[col].astype(bool).to_numpy()
            status = _hvg_panel_status(int(mask.sum()), int(requested_n))
        else:
            raise ValueError(f"Unsupported gene panel: {panel}")

        masks[panel] = mask.astype(bool, copy=False)
        meta[panel] = {
            "gene_panel": panel,
            "gene_panel_requested_n": requested_n,
            "gene_panel_effective_n": int(mask.sum()),
            "gene_panel_status": status,
            "n_genes_raw_shared": n_raw,
            "n_genes_qc_shared": n_qc,
            "n_genes_g_base_intersection_raw": n_base_raw,
            "n_genes_g_base_qc": n_base_qc,
        }

    return masks, meta, gene_qc_file


def _row_with_gene_panel_meta(row: Dict[str, Any], panel_meta: Mapping[str, Any]) -> Dict[str, Any]:
    out = dict(row)
    for c in GENE_PANEL_META_COLUMNS:
        out[c] = panel_meta.get(c, None)
    return out



def _safe_embedding_validity(z: np.ndarray) -> np.ndarray:
    if z.size == 0:
        return np.zeros((0,), dtype=bool)
    finite = np.all(np.isfinite(z), axis=1)
    norms = np.linalg.norm(z.astype(np.float32), axis=1)
    return finite & (norms > 0)



def _resolve_effective_retrieval_backend(backend: str) -> str:
    backend = str(backend)
    if backend != "auto":
        return backend
    try:
        import faiss  # noqa: F401
        return "faiss"
    except Exception:
        return "exact"



def _faiss_search_prebuilt_index(
    index,
    queries: np.ndarray,
    K: int,
    queries_normalized: bool = False,
) -> RetrievalResult:
    """Search a prebuilt FAISS IndexFlatIP (shared by the H&E and GT queries of a slide)."""
    K = int(K)
    if K <= 0:
        raise ValueError("K must be > 0")
    if queries.shape[0] == 0:
        return RetrievalResult(
            indices=np.zeros((0, K), dtype=np.int64),
            similarities=np.zeros((0, K), dtype=np.float32),
        )

    q = queries if queries_normalized else l2_normalize(queries)
    q = np.ascontiguousarray(q, dtype=np.float32)
    sims, idx = index.search(q, K)

    # Sort as in faiss_topk_cosine(): larger similarity first, then smaller reference index
    if idx.shape[1] > 1:
        order = np.lexsort((idx, -sims), axis=1)
        sims = np.take_along_axis(sims, order, axis=1)
        idx = np.take_along_axis(idx, order, axis=1)

    return RetrievalResult(
        indices=idx.astype(np.int64, copy=False),
        similarities=sims.astype(np.float32, copy=False),
    )



def _weighted_label_vote_from_neighbors(
    neighbor_indices: np.ndarray,
    weights: np.ndarray,
    ref_labels: np.ndarray,
) -> np.ndarray:
    """
    Weighted-mass label vote, with tie-break:
      1) largest weighted mass per class,
      2) earliest first-occurrence position among tied classes,
      3) lexicographically smallest label string.
    """
    neighbor_indices = np.asarray(neighbor_indices)
    if neighbor_indices.ndim != 2:
        raise ValueError("neighbor_indices must be 2D")
    n_rows, k = neighbor_indices.shape
    if n_rows == 0:
        return np.empty(0, dtype=object)

    ref_labels_str = np.asarray(ref_labels, dtype=object).astype(str)
    unique_labels, label_ids = np.unique(ref_labels_str, return_inverse=True)
    n_classes = unique_labels.shape[0]

    nn_label_ids = label_ids[neighbor_indices]
    weights_f32 = np.asarray(weights, dtype=np.float32)
    if weights_f32.shape != neighbor_indices.shape:
        raise ValueError(
            f"weights shape {weights_f32.shape} does not match neighbor_indices shape {neighbor_indices.shape}"
        )

    mass = np.zeros((n_rows, n_classes), dtype=np.float32)
    rows_flat = np.repeat(np.arange(n_rows, dtype=np.int64), k)
    np.add.at(mass, (rows_flat, nn_label_ids.ravel()), weights_f32.ravel())

    pred_label_ids = np.argmax(mass, axis=1)
    top_mass = mass[np.arange(n_rows), pred_label_ids]
    n_tied = (mass == top_mass[:, None]).sum(axis=1)
    tied_rows = np.flatnonzero(n_tied > 1)

    for i in tied_rows.tolist():
        row_mass = mass[i]
        tied_classes = set(np.flatnonzero(row_mass == top_mass[i]).tolist())
        first_pos: Dict[int, int] = {}
        for pos in range(k):
            lab_id = int(nn_label_ids[i, pos])
            if lab_id in tied_classes and lab_id not in first_pos:
                first_pos[lab_id] = pos
                if len(first_pos) == len(tied_classes):
                    break
        best = min(
            tied_classes,
            key=lambda c: (first_pos.get(c, k), str(unique_labels[c])),
        )
        pred_label_ids[i] = best

    return unique_labels[pred_label_ids].astype(object)



def _append_undefined_classification_rows(
    metrics_rows: List[Dict[str, Any]],
    population: str,
    n_cells: int,
    reason: str,
) -> None:
    payload = {
        "accuracy": np.nan,
        "macro_f1": np.nan,
        "balanced_accuracy": np.nan,
        "n_cells": float(n_cells),
        "n_gt_present_classes": 0.0,
        # Same metric set as for non-empty populations
        "n_gt_present_classes_unfloored": 0.0,
        "n_classes_below_macro_floor": 0.0,
    }
    for metric_name, value in payload.items():
        metrics_rows.append(
            {
                "population": population,
                "metric_group": "classification",
                "metric": metric_name,
                "value": value,
                "gene_panel": NO_GENE_PANEL,
            }
        )
    metrics_rows.append(
        {
            "population": population,
            "metric_group": "classification",
            "metric": "reporting_status",
            "value": np.nan,
            "status": reason,
            "gene_panel": NO_GENE_PANEL,
        }
    )



def _save_prediction_arrays(
    out_dir: Path,
    slide_id: str,
    path: str,
    valid_rows: np.ndarray,
    pred_label_ids_full: np.ndarray,
    class_scores_full: np.ndarray,
    pred_expr_full: Optional[np.ndarray],
) -> None:
    ensure_dir(out_dir)
    # pred_expr is not written when expression is disabled
    arrays = {
        "valid_rows": np.asarray(valid_rows, dtype=np.int64),
        "pred_label_ids": np.asarray(pred_label_ids_full[valid_rows], dtype=np.int32),
        "class_scores": np.asarray(class_scores_full[valid_rows], dtype=np.float32),
    }
    if pred_expr_full is not None:
        arrays["pred_expr"] = np.asarray(pred_expr_full[valid_rows], dtype=np.float32)
    np.savez(out_dir / f"{slide_id}_{path}_arrays.npz", **arrays)



def _software_versions() -> Dict[str, str]:
    import sys as _sys
    import numpy as _np
    import pandas as _pd
    import scipy as _scipy
    import anndata as _ad
    import h5py as _h5py

    out = {
        "python": _sys.version.split()[0],
        "numpy": _np.__version__,
        "pandas": _pd.__version__,
        "scipy": _scipy.__version__,
        "anndata": _ad.__version__,
        "h5py": _h5py.__version__,
        "torch": torch.__version__,
    }
    return out



def _git_commit_if_available() -> Optional[str]:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode("utf-8").strip()
    except Exception:
        return None



def _prepare_slide_data(
    slide_row: pd.Series,
    cell_backbone_df: pd.DataFrame,
    cell_backbone_index_by_slide: Mapping[str, np.ndarray],
    checkpoint_dir: Path,
    eval_data_root: Path,
    shared_dir: Path,
    training_num: str,
    checkpoint_num: str,
    label_to_idx: Mapping[str, int],
    mapped_ref_label_col: str,
    fine_ref_label_col: str,
    label_map: Optional[Mapping[str, str]],
    cfg: Mapping[str, Any],
    st_cell_bundle_path: Path,
    raw_ref_embedding_dir: Path,
    device: str,
    logger,
    runtime_cache: Optional[Dict[str, Dict[Any, Any]]] = None,
) -> SlideData:
    slide_id = str(slide_row["slide_id"])
    subset = str(slide_row["subset"])
    tissue = str(slide_row["tissue_HistoST"])
    reference_stem = str(slide_row["reference_stem"])
    reference_h5ad = Path(str(slide_row["reference_h5ad"]))

    legacy_he_dir = resolve_legacy_he_embedding_dir(eval_data_root, training_num, checkpoint_num)
    legacy_st_dir = resolve_legacy_st_cell_embedding_dir(eval_data_root, training_num, checkpoint_num)

    he_slide_h5 = legacy_he_dir / f"{slide_id}.h5"
    gt_emb_h5ad = legacy_st_dir / f"{slide_id}.h5ad"
    gt_eval_h5ad = eval_data_root / "st" / f"{slide_id}.h5ad"
    selected_rows = ensure_selected_main_rows(checkpoint_dir, eval_data_root, training_num, checkpoint_num, slide_id)

    slide_positions = cell_backbone_index_by_slide.get(slide_id)
    if slide_positions is None or len(slide_positions) == 0:
        raise RuntimeError(f"No shared cell backbone rows found for slide {slide_id}")
    cell_backbone = cell_backbone_df.iloc[slide_positions].copy()
    cell_backbone = cell_backbone.sort_values(["cell_id_int", "cell_row_key"]).reset_index(drop=True)

    he = load_selected_he_embeddings(he_slide_h5, selected_rows, require_clip=True, load_raw=False)
    gt = load_gt_cell_embeddings(gt_emb_h5ad)

    gu_path = shared_dir / "gene_universes" / f"{slide_id}.parquet"
    gu = pd.read_parquet(gu_path)
    gene_names = gu["gene"].astype(str).tolist()
    gt_gene_names = (
        gu["gt_gene_name"].astype(str).tolist()
        if "gt_gene_name" in gu.columns
        else list(gene_names)
    )
    if len(gt_gene_names) != len(gene_names):
        raise RuntimeError(
            f"Gene-name metadata length mismatch for slide={slide_id}: "
            f"{len(gt_gene_names)} names vs {len(gene_names)} genes"
        )
    gt_var_idx = gu["gt_var_index"].to_numpy(dtype=np.int64, copy=False)
    ref_var_idx = gu["reference_var_index"].to_numpy(dtype=np.int64, copy=False)

    gene_panel_masks, gene_panel_meta, gene_qc_file = _load_gene_panel_masks_for_slide(
        shared_dir=shared_dir,
        slide_id=slide_id,
        gene_names=gene_names,
        cfg=cfg,
    )

    # With expression disabled, only the GT obs is read (X is None)
    expression_enabled = bool(cfg.get("expression", {}).get("enabled", True))
    gt_expr = load_gt_expression_matrix(
        gt_eval_h5ad,
        slide_id=slide_id,
        gene_id_column=cfg["genes"]["gt_gene_id_column"],
        var_indices=gt_var_idx,
        load_x=expression_enabled,
    )

    with np.load(shared_dir / "reference_audits" / "row_filters" / f"{slide_id}.npz") as npz:
        filtered_rows = np.asarray(npz["filtered_rows"], dtype=np.int64)
        fine_rows = np.asarray(npz["fine_rows"], dtype=np.int64)
    if fine_rows.size != filtered_rows.size:
        raise RuntimeError(
            f"Slide {slide_id} has filtered reference rows without fine labels. "
            "This violates the protocol for fine agreement."
        )
    raw_ref_h5 = resolve_raw_reference_embedding_h5(raw_ref_embedding_dir, reference_h5ad)
    umap_ref_rows_rel = np.load(shared_dir / "umap_sampling" / f"{slide_id}.npy").astype(np.int64, copy=False)
    runtime_cache = {} if runtime_cache is None else runtime_cache
    ref_cache = runtime_cache.setdefault("reference", {})
    projected_ref_cache_path: Optional[Path] = None

    ref_key = (
        str(reference_h5ad.resolve()),
        str(raw_ref_h5.resolve()),
        _hash_int_array(filtered_rows),
        mapped_ref_label_col,
        fine_ref_label_col,
    )
    if ref_key not in ref_cache:
        obs_ref = read_obs_columns(reference_h5ad, [mapped_ref_label_col, fine_ref_label_col])
        projected_cache_name = _projected_reference_cache_stem(
            raw_ref_h5=raw_ref_h5,
            reference_h5ad=reference_h5ad,
            st_cell_bundle_path=st_cell_bundle_path,
        )
        projected_ref_cache = project_reference_embeddings_if_needed(
            raw_ref_h5=raw_ref_h5,
            reference_h5ad_path=reference_h5ad,
            projected_cache_path=checkpoint_dir / "projected_embeddings" / "reference" / f"{projected_cache_name}.h5",
            st_cell_bundle_path=st_cell_bundle_path,
            device=device,
        )
        projected_ref_cache_path = projected_ref_cache.path
        ref_z_all = projected_ref_cache.z_ref[filtered_rows]
        # Reference-side label collapse (GT labels were collapsed by _2); ref_fine_all is not collapsed
        ref_labels_all = apply_label_map(
            obs_ref.iloc[filtered_rows][mapped_ref_label_col].astype(str).to_numpy(),
            label_map,
        )
        ref_fine_all = obs_ref.iloc[filtered_rows][fine_ref_label_col].astype(str).to_numpy()
        ref_label_ids_all = labels_to_ids(ref_labels_all, label_to_idx)

        ref_valid = _safe_embedding_validity(ref_z_all)
        if not np.any(ref_valid):
            raise RuntimeError(f"All filtered reference embeddings are invalid for slide {slide_id}")
        if not np.all(ref_valid):
            logger.warning(
                "Dropping %d/%d invalid reference embeddings for slide=%s",
                int((~ref_valid).sum()),
                int(ref_valid.size),
                slide_id,
            )

        ref_z = ref_z_all[ref_valid]
        ref_labels = ref_labels_all[ref_valid]
        ref_fine = ref_fine_all[ref_valid]
        ref_label_ids = ref_label_ids_all[ref_valid]
        if np.any(ref_label_ids < 0):
            missing = sorted(set(ref_labels[ref_label_ids < 0].tolist()))
            raise ValueError(f"Reference labels missing from protocol vocabulary for slide {slide_id}: {missing}")

        ref_cache[ref_key] = {
            "projected_ref_cache": str(projected_ref_cache_path),
            "ref_valid_mask": ref_valid,
            "ref_z": ref_z,
            "ref_labels": ref_labels,
            "ref_fine": ref_fine,
            "ref_label_ids": ref_label_ids,
            "actual_ref_label_counts": pd.Series(ref_labels, dtype="object").value_counts().to_dict(),
        }

    ref_cached = ref_cache[ref_key]
    projected_ref_cache_path = Path(ref_cached["projected_ref_cache"])
    ref_z = ref_cached["ref_z"]
    ref_labels = ref_cached["ref_labels"]
    ref_fine = ref_cached["ref_fine"]
    ref_label_ids = ref_cached["ref_label_ids"]
    # Computed once with the cache entry
    actual_ref_label_counts = ref_cached["actual_ref_label_counts"]

    # Remap the slide's UMAP reference sample after dropping invalid reference embeddings
    ref_valid_mask = np.asarray(ref_cached["ref_valid_mask"], dtype=bool)
    if not np.all(ref_valid_mask):
        old_to_new = np.full(ref_valid_mask.shape[0], -1, dtype=np.int64)
        old_to_new[np.flatnonzero(ref_valid_mask)] = np.arange(int(ref_valid_mask.sum()), dtype=np.int64)
        umap_ref_rows = old_to_new[umap_ref_rows_rel]
        umap_ref_rows = umap_ref_rows[umap_ref_rows >= 0]
    else:
        umap_ref_rows = umap_ref_rows_rel

    if expression_enabled:
        gt_expr_shared = _normalize_log1p_on_shared_genes(
            gt_expr.X,
            target_sum=float(cfg["expression"]["normalize_target_sum"]),
        )
        # The GT expression width must equal len(gene_names) (used instead of gt_expr.shape[1] when expression is disabled)
        if int(gt_expr_shared.shape[1]) != len(gene_names):
            raise RuntimeError(
                f"GT expression width does not match the frozen gene universe for slide "
                f"{slide_id}: matrix has {int(gt_expr_shared.shape[1])} genes but "
                f"gene_universes/{slide_id}.parquet declares {len(gene_names)}."
            )
    else:
        gt_expr_shared = None
        logger.info(
            "expression.enabled=False: GT expression not read for slide=%s (%d shared genes).",
            slide_id,
            len(gene_names),
        )

    if he.z_he.size and he.z_he.shape[1] != ref_z.shape[1]:
        raise RuntimeError(f"H&E projected dim mismatch for slide {slide_id}: {he.z_he.shape[1]} vs reference {ref_z.shape[1]}")
    if gt.z_gt.size and gt.z_gt.shape[1] != ref_z.shape[1]:
        raise RuntimeError(f"GT projected dim mismatch for slide {slide_id}: {gt.z_gt.shape[1]} vs reference {ref_z.shape[1]}")

    base = cell_backbone[
        [
            "cell_row_key",
            "cell_id_int",
            "cell_id",
            "gt_label",
            "gt_mapped_label_scorable",
            "gt_mapped_label_present_in_reference",
            "reference_stem",
            "reference_h5ad",
            "gene_universe_file",
            "gene_universe_n_shared",
        ]
    ].copy()
    if "is_he_structurally_eligible_modelgroup" in cell_backbone.columns:
        base["is_he_structurally_eligible_modelgroup"] = cell_backbone["is_he_structurally_eligible_modelgroup"].astype(bool).to_numpy()
    else:
        base["is_he_structurally_eligible_modelgroup"] = True
    if base["cell_id_int"].duplicated().any():
        dup = base.loc[base["cell_id_int"].duplicated(), "cell_id_int"].astype(str).unique().tolist()[:20]
        raise ValueError(f"Duplicate cell_id_int in shared backbone for slide {slide_id}. Examples: {dup}")

    if he.table.empty:
        he_table = pd.DataFrame(columns=HE_SELECTED_REPORT_COLUMNS)
    else:
        he_table = he.table.copy()
        for col, default in HE_SELECTED_OPTIONAL_DEFAULTS.items():
            if col not in he_table.columns:
                he_table[col] = default
        he_table = he_table[HE_SELECTED_REPORT_COLUMNS].copy()

    he_table["cell_id_int"] = he_table["cell_id_int"].astype(np.int32)
    gt_obs = gt.obs[["cell_id_int"]].copy()
    gt_obs["_gt_emb_row"] = np.arange(gt_obs.shape[0], dtype=np.int64)
    expr_obs = gt_expr.obs[["cell_id_int"]].copy()
    expr_obs["_gt_expr_row"] = np.arange(expr_obs.shape[0], dtype=np.int64)

    base = base.merge(he_table, on="cell_id_int", how="left")
    base = base.merge(gt_obs, on="cell_id_int", how="left")
    base = base.merge(expr_obs, on="cell_id_int", how="left")

    n_cells = base.shape[0]
    he_dim = int(ref_z.shape[1])
    gt_dim = int(ref_z.shape[1])
    z_he = np.zeros((n_cells, he_dim), dtype=np.float32)
    z_gt = np.zeros((n_cells, gt_dim), dtype=np.float32)
    gt_expr_full = (
        np.zeros((n_cells, gt_expr_shared.shape[1]), dtype=np.float32)
        if expression_enabled
        else None
    )

    he_row = base["selected_patch_cell_row_index"].notna().to_numpy()
    he_structural_eligible = base["is_he_structurally_eligible_modelgroup"].astype(bool).to_numpy()
    if he_row.any() and he.table.shape[0] > 0:
        he_pos = pd.Series(
            np.arange(he.table.shape[0], dtype=np.int64),
            index=he.table["cell_id_int"].astype(np.int32),
        )
        he_take = base["cell_id_int"].astype(np.int32).map(he_pos)
        he_take_mask = he_take.notna().to_numpy()
        if he_take_mask.any():
            z_he[he_take_mask] = he.z_he[he_take.loc[he_take_mask].astype(np.int64).to_numpy()]

    gt_row = base["_gt_emb_row"].notna().to_numpy()
    gt_expr_present = base["_gt_expr_row"].notna().to_numpy()
    if gt_row.any():
        idx = np.flatnonzero(gt_row)
        z_gt[idx] = gt.z_gt[base.loc[gt_row, "_gt_emb_row"].astype(np.int64).to_numpy()]
    # Only the expression fill depends on expression.enabled
    expr_row = base["_gt_expr_row"].notna().to_numpy()
    if expression_enabled and expr_row.any():
        idx = np.flatnonzero(expr_row)
        gt_expr_full[idx] = gt_expr_shared[base.loc[expr_row, "_gt_expr_row"].astype(np.int64).to_numpy()]

    he_valid = _safe_embedding_validity(z_he)
    gt_valid = _safe_embedding_validity(z_gt)
    min_alpha = float(cfg["runtime"]["he_support"]["min_selected_alpha"])
    alpha_ok = base["selected_alpha"].fillna(0.0).astype(float).to_numpy() > min_alpha
    he_valid = he_valid & alpha_ok & he_structural_eligible
    matched_valid = he_valid & gt_valid

    gt_labels = base["gt_label"].astype(str).to_numpy()
    gt_mapped_label_scorable = base["gt_mapped_label_scorable"].astype(bool).to_numpy()
    gt_label_ids = labels_to_ids(gt_labels, label_to_idx)
    if np.any(gt_label_ids < 0):
        missing = sorted(set(gt_labels[gt_label_ids < 0].tolist()))
        raise ValueError(f"GT labels missing from protocol label vocabulary for slide {slide_id}: {missing}")
    covered = np.isin(gt_labels, np.unique(ref_labels))

    # Coverage contract at the run stage (reference after projected-embedding QC), on the
    # modelgroup-token-eligible rows (as in _2)
    coverage_rows = he_structural_eligible
    runtime_coverage_report = build_slide_coverage_report(
        slide_id=slide_id,
        tissue=tissue,
        reference_stem=reference_stem,
        label_granularity=normalize_granularity(cfg["identity"].get("label_granularity", "fine")),
        stage=STAGE_RUN,
        gt_labels=gt_labels[coverage_rows],
        gt_scorable_mask=gt_mapped_label_scorable[coverage_rows],
        ref_label_counts=actual_ref_label_counts,
        cfg=cfg,
    )
    # One slide per call: the cross-slide check of declared exceptions is disabled
    assert_coverage_contract(
        [runtime_coverage_report],
        cfg,
        stage=STAGE_RUN,
        log=logger,
        check_declared_bookkeeping=False,
    )
    # The per-cell coverage must equal the _2 coverage
    assert_runtime_coverage_matches_frozen(
        slide_id=slide_id,
        frozen_covered=base["gt_mapped_label_present_in_reference"].astype(bool).to_numpy(),
        runtime_covered=covered,
        cfg=cfg,
        log=logger,
    )

    covered_gt_classes = sorted(set(gt_labels[gt_mapped_label_scorable].tolist()) & set(actual_ref_label_counts.keys()))
    covered_class_supports = [int(actual_ref_label_counts[c]) for c in covered_gt_classes]
    meets_minimum_reference_cells = (int(ref_z.shape[0]) >= int(cfg["references"]["filter"]["minimum_reference_cells"]))
    meets_minimum_class_support = (
        len(covered_class_supports) > 0
        and min(covered_class_supports) >= int(cfg["references"]["filter"]["minimum_class_support"])
    )
 
    if umap_ref_rows.size == 0:
        logger.warning("UMAP reference sample became empty after dropping invalid reference rows for slide=%s", slide_id)

    logger.info(
        "Prepared slide=%s subset=%s n_cells=%d he_valid=%d gt_valid=%d matched=%d covered=%d ref_cells=%d G_s_raw=%d G_s_qc=%d G_base_qc=%d",
        slide_id,
        subset,
        n_cells,
        int(he_valid.sum()),
        int(gt_valid.sum()),
        int(matched_valid.sum()),
        int((matched_valid & covered).sum()),
        ref_z.shape[0],
        len(gene_names),
        int(gene_panel_meta.get("G_s_qc", {}).get("gene_panel_effective_n", 0)),
        int(gene_panel_meta.get("G_base_qc", {}).get("gene_panel_effective_n", 0)),
    )
    return SlideData(
        slide_id=slide_id,
        subset=subset,
        tissue=tissue,
        reference_stem=reference_stem,
        reference_h5ad=reference_h5ad,
        cells=base,
        z_he=z_he,
        z_gt=z_gt,
        he_valid=he_valid,
        he_structural_eligible=he_structural_eligible,
        gt_valid=gt_valid,
        gt_expr_present=gt_expr_present,
        matched_valid=matched_valid,
        covered=covered,
        gt_mapped_label_scorable=gt_mapped_label_scorable,
        gt_label_ids=gt_label_ids,
        gt_labels=gt_labels,
        gt_expr=gt_expr_full,
        gene_names=gene_names,
        gt_gene_names=gt_gene_names,
        gene_panel_masks=gene_panel_masks,
        gene_panel_meta=gene_panel_meta,
        ref_z=ref_z,
        ref_labels=ref_labels,
        ref_label_ids=ref_label_ids,
        ref_fine_labels=ref_fine,
        ref_expr=None,
        ref_expr_global_to_local=None,
        reference_h5ad_valid_rows=filtered_rows[ref_cached["ref_valid_mask"]].astype(np.int64, copy=False),
        reference_var_indices=ref_var_idx.astype(np.int64, copy=False),
        he_slide_h5=he_slide_h5,
        gt_emb_h5ad=gt_emb_h5ad,
        gt_eval_h5ad=gt_eval_h5ad,
        raw_ref_h5=raw_ref_h5,
        projected_ref_cache=projected_ref_cache_path,
        reference_expression_cache_file=None,
        gene_universe_file=gu_path,
        gene_qc_file=gene_qc_file,
        reference_row_filter_file=shared_dir / "reference_audits" / "row_filters" / f"{slide_id}.npz",
        umap_ref_rows=umap_ref_rows,
        meets_minimum_reference_cells=bool(meets_minimum_reference_cells),
        meets_minimum_class_support=bool(meets_minimum_class_support),
        meets_minimum_aligned_genes_official=bool(slide_row.get("meets_minimum_aligned_genes_official", True)),
        coverage_declared_exception=runtime_coverage_report.active_exception_reason,
    )



def _precompute_retrieval(slide: SlideData, Kmax: int, backend: str, cfg: Mapping[str, Any], logger) -> None:
    slide.he_valid_rows = np.flatnonzero(slide.he_valid).astype(np.int64)
    slide.gt_valid_rows = np.flatnonzero(slide.gt_valid).astype(np.int64)

    # Normalize the reference once for the H&E and GT searches
    ref_z_norm = l2_normalize(slide.ref_z)
    effective_backend = _resolve_effective_retrieval_backend(backend)

    q_chunk = int(cfg.get("runtime", {}).get("retrieval_query_chunk_size", 2048))
    r_chunk = int(cfg.get("runtime", {}).get("retrieval_reference_chunk_size", 16384))

    logger.info(
        "Retrieval backend slide=%s requested=%s effective=%s",
        slide.slide_id,
        backend,
        effective_backend,
    )

    faiss_index = None
    faiss_k = int(min(int(Kmax), int(ref_z_norm.shape[0])))
    if effective_backend == "faiss":
        import faiss

        faiss_index = faiss.IndexFlatIP(int(ref_z_norm.shape[1]))
        faiss_index.add(np.ascontiguousarray(ref_z_norm, dtype=np.float32))

    def _progress_callback(path_name: str):
        state = {"next_fraction": 0.10}

        def _cb(done: int, total: int) -> None:
            if total <= 0:
                return
            frac = float(done) / float(total)
            if frac + 1e-12 >= state["next_fraction"] or done == total:
                logger.info(
                    "Retrieval progress slide=%s path=%s queries=%d/%d %.1f%%",
                    slide.slide_id,
                    path_name,
                    int(done),
                    int(total),
                    100.0 * frac,
                )
                while state["next_fraction"] <= frac + 1e-12:
                    state["next_fraction"] += 0.10

        return _cb

    if slide.he_valid_rows.size > 0:
        logger.info(
            "Starting retrieval for slide=%s path=he queries=%d ref=%d Kmax=%d score_pairs=%.3fB",
            slide.slide_id,
            int(slide.he_valid_rows.size),
            int(slide.ref_z.shape[0]),
            int(Kmax),
            float(slide.he_valid_rows.size * slide.ref_z.shape[0]) / 1.0e9,
        )
        if faiss_index is not None:
            slide.retrieval_he = _faiss_search_prebuilt_index(
                faiss_index,
                slide.z_he[slide.he_valid_rows],
                K=faiss_k,
                queries_normalized=False,
            )
        else:
            slide.retrieval_he = retrieve_topk(
                slide.z_he[slide.he_valid_rows],
                ref_z_norm,
                K=Kmax,
                backend=effective_backend,
                query_chunk_size=q_chunk,
                reference_chunk_size=r_chunk,
                references_normalized=True,
                progress_callback=_progress_callback("he"),
            )
    else:
        logger.warning(
            "Slide=%s has zero H&E-valid query cells. H&E-path metrics will be undefined for this slide.",
            slide.slide_id,
        )
        slide.retrieval_he = RetrievalResult(
            indices=np.zeros((0, Kmax), dtype=np.int64),
            similarities=np.zeros((0, Kmax), dtype=np.float32),
        )

    if slide.gt_valid_rows.size > 0:
        logger.info(
            "Starting retrieval for slide=%s path=gt queries=%d ref=%d Kmax=%d score_pairs=%.3fB",
            slide.slide_id,
            int(slide.gt_valid_rows.size),
            int(slide.ref_z.shape[0]),
            int(Kmax),
            float(slide.gt_valid_rows.size * slide.ref_z.shape[0]) / 1.0e9,
        )
        if faiss_index is not None:
            slide.retrieval_gt = _faiss_search_prebuilt_index(
                faiss_index,
                slide.z_gt[slide.gt_valid_rows],
                K=faiss_k,
                queries_normalized=False,
            )
        else:
            slide.retrieval_gt = retrieve_topk(
                slide.z_gt[slide.gt_valid_rows],
                ref_z_norm,
                K=Kmax,
                backend=effective_backend,
                query_chunk_size=q_chunk,
                reference_chunk_size=r_chunk,
                references_normalized=True,
                progress_callback=_progress_callback("gt"),
            )
    else:
        logger.warning(
            "Slide=%s has zero GT-valid query cells. GT-anchor metrics will be undefined for this slide.",
            slide.slide_id,
        )
        slide.retrieval_gt = RetrievalResult(
            indices=np.zeros((0, Kmax), dtype=np.int64),
            similarities=np.zeros((0, Kmax), dtype=np.float32),
        )

    if not np.any(slide.matched_valid):
        logger.warning(
            "Slide=%s has zero matched valid cells. Matched-population metrics will be undefined for this slide.",
            slide.slide_id,
        )
    logger.info(
        "Precomputed retrieval for slide=%s he_queries=%d gt_queries=%d ref=%d Kmax=%d",
        slide.slide_id,
        slide.he_valid_rows.size,
        slide.gt_valid_rows.size,
        slide.ref_z.shape[0],
        Kmax,
    )



def _append_undefined_expression_rows(
    metrics_rows: List[Dict[str, Any]],
    population: str,
    n_cells: int,
    n_genes: int,
    reason: str,
    panel_meta: Mapping[str, Any],
) -> None:
    payload = {
        "n_cells": float(n_cells),
        "n_genes": float(n_genes),
        "mean_mse": np.nan,
        "mean_mae": np.nan,
        "n_valid_cellwise_pearson": 0.0,
        "n_valid_cellwise_spearman": 0.0,
        "n_valid_genewise_pearson": 0.0,
        "n_valid_genewise_spearman": 0.0,
        "mean_cellwise_pearson": np.nan,
        "mean_cellwise_spearman": np.nan,
        "gene_wise_pearson": np.nan,
        "gene_wise_spearman": np.nan,
    }
    for metric_name, value in payload.items():
        metrics_rows.append(
            _row_with_gene_panel_meta(
                {
                    "population": population,
                    "metric_group": "expression",
                    "metric": metric_name,
                    "value": value,
                },
                panel_meta,
            )
        )
    metrics_rows.append(
        _row_with_gene_panel_meta(
            {
                "population": population,
                "metric_group": "expression",
                "metric": "reporting_status",
                "value": np.nan,
                "status": reason,
            },
            panel_meta,
        )
    )



def _evaluate_query_path(
    slide: SlideData,
    path: str,
    k: int,
    tau: float,
    label_order: Sequence[str],
    cfg: Mapping[str, Any],
    return_details: bool,
) -> Dict[str, Any]:
    assert path in {"he", "gt"}
    # Expression prediction and scoring are skipped when expression is disabled
    expression_enabled = bool(cfg.get("expression", {}).get("enabled", True))
    # slide.gt_expr is loaded iff expression is enabled
    if expression_enabled != (slide.gt_expr is not None):
        raise RuntimeError(
            f"Slide {slide.slide_id}: expression.enabled={expression_enabled} but "
            f"slide.gt_expr is {'None' if slide.gt_expr is None else 'loaded'}. "
            "The slide was prepared with a different config."
        )
    retrieval = slide.retrieval_he if path == "he" else slide.retrieval_gt
    valid_rows = slide.he_valid_rows if path == "he" else slide.gt_valid_rows
    if retrieval is None or valid_rows is None:
        raise RuntimeError(f"Retrieval cache missing for {path} on slide {slide.slide_id}")

    k = min(int(k), retrieval.indices.shape[1] if retrieval.indices.ndim == 2 else int(k))
    nn_idx = retrieval.indices[:, :k]
    nn_sim = retrieval.similarities[:, :k]
    weights = weights_from_similarities(nn_sim, tau=float(tau))
    class_scores = class_scores_from_neighbors(nn_idx, weights, slide.ref_label_ids, n_classes=len(label_order))
    pred_label_ids_valid = predicted_labels_from_scores(class_scores)
    pred_label_valid = np.asarray(label_order, dtype=object)[pred_label_ids_valid]
    # pred_expr_valid is only used under the same condition below
    if expression_enabled and valid_rows.size > 0:
        if slide.ref_expr_global_to_local is None:
            raise RuntimeError(f"Reference expression row map was not attached for slide {slide.slide_id}")
        nn_idx_local = slide.ref_expr_global_to_local[nn_idx]
        if np.any(nn_idx_local < 0):
            raise RuntimeError(
                f"Reference-expression local remap failed for slide {slide.slide_id}; "
                "retrieval neighbors were not materialized in the expression subset."
            )
        if slide.ref_expr is not None:
            pred_expr_valid = weighted_expression_prediction(
                neighbor_indices=nn_idx_local,
                weights=weights,
                ref_expr=slide.ref_expr,
                query_batch_size=int(cfg.get("runtime", {}).get("expression_query_batch_size", 512)),
            )
        else:
            if slide.reference_expression_cache_file is None:
                raise RuntimeError(f"Reference expression cache was not attached for slide {slide.slide_id}")
            pred_expr_valid = weighted_expression_prediction_h5(
                neighbor_indices=nn_idx_local,
                weights=weights,
                cache_h5_path=str(slide.reference_expression_cache_file),
                query_batch_size=int(cfg.get("runtime", {}).get("expression_query_batch_size", 512)),
            )

    pred_label_ids_full = np.full(slide.cells.shape[0], -1, dtype=np.int32)
    pred_label_ids_full[valid_rows] = pred_label_ids_valid
    pred_label_full = np.full(slide.cells.shape[0], "", dtype=object)
    pred_label_full[valid_rows] = pred_label_valid

    class_scores_full = np.zeros((slide.cells.shape[0], len(label_order)), dtype=np.float32)
    class_scores_full[valid_rows] = class_scores
    # (n_cells, n_genes) prediction array, allocated only when expression is enabled
    pred_expr_full = (
        np.zeros((slide.cells.shape[0], len(slide.gene_names)), dtype=np.float32)
        if expression_enabled
        else None
    )
    if expression_enabled and valid_rows.size > 0:
        pred_expr_full[valid_rows] = pred_expr_valid

    query_valid_mask = slide.he_valid if path == "he" else slide.gt_valid
    classification_population_masks = {
        f"{path}_valid_all": query_valid_mask & slide.gt_mapped_label_scorable,
        f"{path}_valid_covered": query_valid_mask & slide.gt_mapped_label_scorable & slide.covered,
        f"matched_all_{path}": slide.matched_valid & slide.gt_mapped_label_scorable,
        f"matched_covered_{path}": slide.matched_valid & slide.gt_mapped_label_scorable & slide.covered,
    }
    expression_population_masks = {
        f"{path}_valid_all": query_valid_mask & slide.gt_expr_present,
        f"{path}_valid_covered": query_valid_mask & slide.covered & slide.gt_expr_present,
        f"matched_all_{path}": (slide.matched_valid & slide.gt_expr_present),
        f"matched_covered_{path}": ((slide.matched_valid & slide.covered) & slide.gt_expr_present),
    }

    metrics_rows: List[Dict[str, Any]] = []
    per_class_tables: Dict[str, pd.DataFrame] = {}
    confusion_tables: Dict[str, np.ndarray] = {}
    expr_tables_gene: Dict[Tuple[str, str], pd.DataFrame] = {}
    expr_tables_cell: Dict[Tuple[str, str], pd.DataFrame] = {}

    for pop_name, pop_mask in classification_population_masks.items():
        rows = np.flatnonzero(pop_mask)
        if rows.size == 0:
            _append_undefined_classification_rows(
                metrics_rows=metrics_rows,
                population=pop_name,
                n_cells=0,
                reason="empty_population",
            )
        else:
            cls = classification_metrics_from_ids(
                slide.gt_label_ids[rows],
                pred_label_ids_full[rows],
                label_order=label_order,
                # Same per-slide macro floor as the tuning objective and the chance floor
                macro_min_support=macro_min_class_support(cfg),
            )
            for m, v in cls.metrics.items():
                metrics_rows.append({"population": pop_name, "metric_group": "classification", "metric": m, "value": v, "gene_panel": NO_GENE_PANEL})

            # Per-slide macro denominator: classes present vs classes below the macro floor
            # (derived from cls.per_class)
            _support = cls.per_class["support"].to_numpy(dtype=np.int64)
            _n_present_unfloored = int((_support > 0).sum())
            _n_macro_classes = int(cls.metrics["n_gt_present_classes"])
            for _m, _v in (
                ("n_gt_present_classes_unfloored", float(_n_present_unfloored)),
                ("n_classes_below_macro_floor", float(_n_present_unfloored - _n_macro_classes)),
            ):
                metrics_rows.append({"population": pop_name, "metric_group": "classification", "metric": _m, "value": _v, "gene_panel": NO_GENE_PANEL})

            # Unscorable labels must have zero support and pred_support
            assert_unscorable_labels_absent(
                cls.per_class,
                cfg["data"].get("gt_mapped_unscorable_labels", []),
                context=f"slide={slide.slide_id} path={path} population={pop_name}",
            )
            per_class_tables[pop_name] = cls.per_class
            confusion_tables[pop_name] = cls.confusion_matrix

        if not expression_enabled:
            # No expression rows when expression is disabled
            continue

        expr_rows = np.flatnonzero(expression_population_masks[pop_name])
        gene_array = np.asarray(slide.gene_names, dtype=object)
        gt_gene_name_array = np.asarray(slide.gt_gene_names, dtype=object)

        for panel_name, gene_mask in slide.gene_panel_masks.items():
            panel_meta = slide.gene_panel_meta[panel_name]
            n_panel_genes = int(gene_mask.sum())

            if n_panel_genes == 0:
                _append_undefined_expression_rows(
                    metrics_rows=metrics_rows,
                    population=pop_name,
                    n_cells=int(expr_rows.size),
                    n_genes=0,
                    reason=str(panel_meta["gene_panel_status"]),
                    panel_meta=panel_meta,
                )
                continue

            if expr_rows.size == 0:
                _append_undefined_expression_rows(
                    metrics_rows=metrics_rows,
                    population=pop_name,
                    n_cells=0,
                    n_genes=n_panel_genes,
                    reason="empty_expression_scorable_population",
                    panel_meta=panel_meta,
                )
                continue

            expr = summarize_expression(
                pred=pred_expr_full[expr_rows][:, gene_mask],
                gt=slide.gt_expr[expr_rows][:, gene_mask],
                gene_names=gene_array[gene_mask].astype(str).tolist(),
                gene_report_names=gt_gene_name_array[gene_mask].astype(str).tolist(),
                variance_eps=float(cfg["expression"]["variance_eps"]),
                min_valid_cells_for_cellwise_corr=int(cfg["expression"]["min_valid_cells_for_cellwise_corr"]),
                min_valid_cells_for_genewise_corr=int(cfg["expression"]["min_valid_cells_for_genewise_corr"]),
                min_valid_genes_for_slide_summary=int(cfg["expression"]["min_valid_genes_for_slide_summary"]),
                return_details=return_details,
            )

            for m, v in expr.metrics.items():
                metrics_rows.append(
                    _row_with_gene_panel_meta(
                        {
                            "population": pop_name,
                            "metric_group": "expression",
                            "metric": m,
                            "value": v,
                        },
                        panel_meta,
                    )
                )

            if return_details and expr.per_gene is not None:
                pg = expr.per_gene.copy()
                pc = expr.per_cell.copy()

                for c in GENE_PANEL_META_COLUMNS:
                    pg[c] = panel_meta.get(c, None)
                    pc[c] = panel_meta.get(c, None)

                pc["cell_row_index"] = expr_rows.astype(np.int64)
                pc["cell_id_int"] = slide.cells.loc[expr_rows, "cell_id_int"].to_numpy(dtype=np.int32, copy=False)

                expr_tables_gene[(pop_name, panel_name)] = pg
                expr_tables_cell[(pop_name, panel_name)] = pc

    gt_class_mass_valid, gt_class_in_topk_valid = mapped_class_mass(nn_idx, weights, slide.ref_label_ids, slide.gt_label_ids[valid_rows])
    diag = pd.DataFrame(
        {
            "cell_id_int": slide.cells.loc[valid_rows, "cell_id_int"].to_numpy(dtype=np.int32, copy=False),
            "cell_id": slide.cells.loc[valid_rows, "cell_id"].astype(str).to_numpy(),
            "gt_label": slide.gt_labels[valid_rows],
            "gt_mapped_label_scorable": slide.gt_mapped_label_scorable[valid_rows].astype(np.int8),
            "gt_label_covered": slide.covered[valid_rows].astype(np.int8),
            "pred_label": pred_label_valid,
            "top_class_score": class_scores.max(axis=1).astype(np.float32),
            "class_score_entropy": entropy_metrics_from_class_scores(class_scores).astype(np.float32),
            "neighbor_weight_entropy": entropy_from_probs(weights, axis=1).astype(np.float32),
            "gt_class_mass": gt_class_mass_valid.astype(np.float32),
            "gt_class_in_topk": gt_class_in_topk_valid.astype(np.int8),
        }
    )

    out = {
        "pred_label_ids_full": pred_label_ids_full,
        "pred_label_full": pred_label_full,
        "class_scores_full": class_scores_full,
        "pred_expr_full": pred_expr_full,
        "metrics": pd.DataFrame(metrics_rows),
        "diagnostics": diag,
        "per_class_tables": per_class_tables,
        "confusions": confusion_tables,
        "expr_tables_gene": expr_tables_gene,
        "expr_tables_cell": expr_tables_cell,
        "valid_rows": valid_rows,
        "nn_idx": nn_idx,
        "nn_sim": nn_sim,
        "weights": weights,
    }
    return out



# Tie-break name -> (summary column, ascending)
_TIE_BREAK_COLUMNS: Dict[str, Tuple[str, bool]] = {
    "gene_wise_pearson": ("mean_gene_wise_pearson", False),
    "mapped_macro_f1_matched_covered": ("mean_macro_f1", False),
    "mean_cellwise_pearson": ("mean_cellwise_pearson", False),
    "smaller_k": ("k", True),
    "smaller_tau": ("tau", True),
}


def _apply_tie_breaks(cfg: Mapping[str, Any], sort_by: List[str], ascending: List[bool]) -> None:
    """Append the configured tuning.tie_breaks, then k and tau (ascending) if not already present, to the sort keys."""
    for tb in [str(x) for x in cfg["tuning"].get("tie_breaks", [])]:
        col_asc = _TIE_BREAK_COLUMNS.get(tb)
        if col_asc is None:
            continue
        column, asc = col_asc
        if column not in sort_by:
            sort_by.append(column)
            ascending.append(asc)
    for column, asc in (("k", True), ("tau", True)):
        if column not in sort_by:
            sort_by.append(column)
            ascending.append(asc)



def _tune_k_tau(
    train_slides: Sequence[SlideData],
    k_grid: Sequence[int],
    tau_grid: Sequence[float],
    fallback: Mapping[str, Any],
    label_order: Sequence[str],
    cfg: Mapping[str, Any],
    shared_dir: Path,
    logger,
) -> Tuple[int, float, pd.DataFrame]:
    if len(train_slides) == 0:
        logger.warning(
            "No eligible training slides for tuning. Using fallback k=%s tau=%s",
            fallback["k"],
            fallback["tau"],
        )
        return int(fallback["k"]), float(fallback["tau"]), pd.DataFrame()

    rows: List[Dict[str, Any]] = []
    objective = str(cfg["tuning"].get("objective", "mapped_macro_f1_matched_covered"))
    use_composite = objective == "composite_rank_matched_covered_he"
    # No reference expression is attached during tuning when expression is disabled
    expression_enabled = bool(cfg.get("expression", {}).get("enabled", True))
    # Whether the tuning reads expression (used by both tuning expression sites)
    tuning_expression_used = expression_enabled and _tuning_reads_expression(cfg)
    tuning_expression_skip_reason = (
        ""
        if tuning_expression_used
        else (
            "expression.enabled is False"
            if not expression_enabled
            else (
                f"tuning.objective={objective!r} with tie_breaks="
                f"{[str(t) for t in (cfg['tuning'].get('tie_breaks', []) or [])]} reads no "
                "expression metric"
            )
        )
    )
    if not tuning_expression_used:
        logger.info(
            "Tuning expression pass skipped for all slides: %s. cv_table.csv has NaN "
            "gene_wise_pearson / mean_cellwise_pearson and expression_tuning_computed=0 "
            "(expression.enabled=%s).",
            tuning_expression_skip_reason,
            expression_enabled,
        )

    if objective in {"mapped_macro_f1_matched_covered", "composite_rank_matched_covered_he"}:
        objective_population = "matched_covered_he"
    elif objective == "mapped_macro_f1_matched_all":
        objective_population = "matched_all_he"
    else:
        raise RuntimeError(f"Unsupported tuning objective: {objective}")

    Kmax = int(cfg["retrieval"]["Kmax"])
    tuning_k_max = int(max(int(x) for x in k_grid))
    n_grid = len(k_grid) * len(tau_grid)
    for slide_i, val_slide in enumerate(train_slides, start=1):
        logger.info(
            "Starting tuning slide %d/%d slide=%s grid_points=%d",
            slide_i,
            len(train_slides),
            val_slide.slide_id,
            n_grid,
        )
        ref_expr_local = None
        tuning_gene_panel = str(cfg.get("gene_panels", {}).get("tuning", {}).get("expression_gene_panel", "G_base_qc"))
        min_tuning_genes = int(
            cfg.get("gene_panels", {})
            .get("tuning", {})
            .get("min_genes_for_G_base_qc_tuning", 30)
        )
        n_tuning_genes = int(val_slide.gene_panel_masks[tuning_gene_panel].sum())
        tuning_panel_valid = n_tuning_genes >= min_tuning_genes
        if objective == "mapped_macro_f1_matched_all":
            tuning_expr_mask = (
                val_slide.matched_valid
                & val_slide.gt_expr_present
            )
        else:
            tuning_expr_mask = (
                val_slide.matched_valid
                & val_slide.covered
                & val_slide.gt_expr_present
            )

        if tuning_expression_used and tuning_panel_valid and np.any(tuning_expr_mask):
            _attach_reference_expression_subset(
                slide=val_slide,
                shared_dir=shared_dir,
                k=tuning_k_max,
                cfg=cfg,
                logger=logger,
                query_paths=("he",),
                query_row_mask=tuning_expr_mask,
                materialize_in_memory=True,
            )
            ref_expr_local = val_slide.ref_expr
        elif tuning_expression_used:
            # Warn only when the tuning reads expression
            logger.warning(
                "Skipping expression attachment for tuning slide=%s because %s has %d genes < min_tuning_genes=%d or no expression rows.",
                val_slide.slide_id,
                tuning_gene_panel,
                n_tuning_genes,
                min_tuning_genes,
            )
        try:
            grid_i = 0
            for k in k_grid:
                for tau in tau_grid:
                    grid_i += 1
                    k_i = int(k)
                    tau_f = float(tau)

                    if grid_i == 1 or grid_i == n_grid or grid_i % max(1, n_grid // 6) == 0:
                        logger.info(
                            "Tuning progress slide=%s grid=%d/%d k=%d tau=%.6f",
                            val_slide.slide_id,
                            grid_i,
                            n_grid,
                            k_i,
                            tau_f,
                        )

                    ev = _evaluate_tuning_grid_point_he(
                        slide=val_slide,
                        k=k_i,
                        tau=tau_f,
                        label_order=label_order,
                        cfg=cfg,
                        objective=objective,
                        ref_expr_local=ref_expr_local,
                        reads_expression=tuning_expression_used,
                    )

                    macro_f1 = float(ev["macro_f1"])
                    gene_p = float(ev["gene_wise_pearson"])
                    cell_p = float(ev["mean_cellwise_pearson"])
                    n_cells = int(ev["n_classification_cells"])
                    n_valid_expression_cells = int(ev["n_valid_expression_cells"])

                    fold_valid_classification = np.isfinite(macro_f1) and (n_cells > 0)
                    fold_valid_expression = (
                        np.isfinite(gene_p)
                        and np.isfinite(cell_p)
                        and (n_valid_expression_cells > 0)
                        and (int(ev["n_tuning_genes"]) >= int(ev["min_tuning_genes"]))
                    )
                    rows.append(
                        {
                            "fold_id": f"val_{val_slide.slide_id}",
                            "validation_slide_id": val_slide.slide_id,
                            "subset": val_slide.subset,
                            "k": k_i,
                            "tau": tau_f,
                            "objective": objective,
                            "objective_population": ev["objective_population"],
                            "objective_value": macro_f1,
                            "macro_f1": macro_f1,
                            "gene_wise_pearson": gene_p,
                            "mean_cellwise_pearson": cell_p,
                            "n_covered_matched_cells": n_cells,
                            "n_valid_expression_cells": n_valid_expression_cells,
                            "tuning_gene_panel": ev["tuning_gene_panel"],
                            "n_tuning_genes": int(ev["n_tuning_genes"]),
                            "min_tuning_genes": int(ev["min_tuning_genes"]),
                            "tuning_gene_panel_status": ev["tuning_gene_panel_status"],
                            "fold_valid": int(bool(fold_valid_classification)),
                            "fold_valid_classification": int(bool(fold_valid_classification)),
                            "fold_valid_expression": int(bool(fold_valid_expression)),
                            # 0: expression not computed during tuning
                            "expression_tuning_computed": int(bool(tuning_expression_used)),
                            "expression_tuning_skip_reason": tuning_expression_skip_reason,
                        }
                    )
        finally:
            _release_reference_expression_subset(val_slide)
            if ref_expr_local is not None:
                del ref_expr_local
            gc.collect()

    cv = pd.DataFrame(rows)
    if cv.empty:
        return int(fallback["k"]), float(fallback["tau"]), cv

    n_required_folds = int(cv["validation_slide_id"].nunique())
    if n_required_folds <= 0:
        return int(fallback["k"]), float(fallback["tau"]), cv

    if use_composite:
        summary = (
            cv.groupby(["k", "tau"], dropna=False)
            .agg(
                mean_macro_f1=("macro_f1", "mean"),
                mean_gene_wise_pearson=("gene_wise_pearson", "mean"),
                mean_cellwise_pearson=("mean_cellwise_pearson", "mean"),
                n_valid_classification_folds=("fold_valid_classification", "sum"),
                n_valid_expression_folds=("fold_valid_expression", "sum"),
                mean_cells=("n_covered_matched_cells", "mean"),
                mean_valid_expression_cells=("n_valid_expression_cells", "mean"),
                mean_n_tuning_genes=("n_tuning_genes", "mean"),
                min_n_tuning_genes=("n_tuning_genes", "min"),
            )
            .reset_index()
        )
        if summary.empty:
            raise RuntimeError("Hyperparameter tuning failed: no STHELAR_train grid-search rows were produced.")

        summary["rank_macro_f1"] = _rank_score(summary["mean_macro_f1"])
        summary["rank_gene_wise_pearson"] = _rank_score(summary["mean_gene_wise_pearson"])
        summary["rank_cellwise_pearson"] = _rank_score(summary["mean_cellwise_pearson"])

        weights = cfg["tuning"].get("composite_weights", {})
        w_macro = float(weights.get("mapped_macro_f1_matched_covered", 0.40))
        w_gene = float(weights.get("gene_wise_pearson", 0.40))
        w_cell = float(weights.get("mean_cellwise_pearson", 0.20))

        summary["selection_score"] = (
            w_macro * summary["rank_macro_f1"]
            + w_gene * summary["rank_gene_wise_pearson"]
            + w_cell * summary["rank_cellwise_pearson"]
        )
        # Selectable only if every train slide has finite metrics
        summary["n_required_folds"] = n_required_folds
        summary["is_selectable"] = (
            (summary["n_valid_classification_folds"] == n_required_folds)
            & (summary["n_valid_expression_folds"] == n_required_folds)
        ).astype(np.int8)

        if not bool(summary["is_selectable"].any()):
            raise RuntimeError(
                "Hyperparameter tuning failed: no (k, tau) grid point had finite classification "
                "and finite expression metrics on every eligible STHELAR_train slide. "
                f"Required folds={n_required_folds}. Inspect metrics/cv_table.csv."
            )

        sort_by: List[str] = ["is_selectable", "selection_score"]
        ascending: List[bool] = [False, False]
        _apply_tie_breaks(cfg, sort_by, ascending)
        summary = summary.sort_values(by=sort_by, ascending=ascending).reset_index(drop=True)
        selection_reason = (
            "composite_rank_matched_covered_he"
            f"(w_macro={w_macro:.3f},w_gene={w_gene:.3f},w_cell={w_cell:.3f})"
            + " > "
            + " > ".join(sort_by)
        )

    else:
        grouped = cv.groupby(["k", "tau"], dropna=False)
        summary = grouped.agg(
            mean_objective=("objective_value", "mean"),
            mean_macro_f1=("macro_f1", "mean"),
            mean_gene_wise_pearson=("gene_wise_pearson", "mean"),
            mean_cellwise_pearson=("mean_cellwise_pearson", "mean"),
            n_valid_folds=("fold_valid_classification", "sum"),
            mean_cells=("n_covered_matched_cells", "mean"),
            mean_valid_expression_cells=("n_valid_expression_cells", "mean"),
        ).reset_index()
        if summary.empty:
            raise RuntimeError(
                "Hyperparameter tuning failed: every STHELAR_train fold had zero covered matched cells "
                "or undefined primary metric."
            )

        summary["n_required_folds"] = n_required_folds
        summary["is_selectable"] = (
            summary["n_valid_folds"] == n_required_folds
        ).astype(np.int8)

        if not bool(summary["is_selectable"].any()):
            raise RuntimeError(
                "Hyperparameter tuning failed: no (k, tau) grid point had finite "
                f"classification objective on every eligible STHELAR_train slide. "
                f"Required folds={n_required_folds}. Inspect metrics/cv_table.csv."
            )

        sort_by = ["is_selectable", "mean_objective"]
        ascending = [False, False]
        _apply_tie_breaks(cfg, sort_by, ascending)
        summary = summary.sort_values(by=sort_by, ascending=ascending).reset_index(drop=True)
        selection_reason = " > ".join(sort_by)

    best = summary.iloc[0]
    cv = cv.merge(summary, on=["k", "tau"], how="left")
    cv["selected_k"] = int(best["k"])
    cv["selected_tau"] = float(best["tau"])
    cv["is_selected"] = ((cv["k"] == int(best["k"])) & (cv["tau"] == float(best["tau"]))).astype(np.int8)
    cv["selection_reason"] = selection_reason
    return int(best["k"]), float(best["tau"]), cv



def _save_retrieval_outputs(
    out_dir: Path,
    slide_id: str,
    path: str,
    valid_rows: np.ndarray,
    nn_idx: np.ndarray,
    nn_sim: np.ndarray,
    weights: np.ndarray,
) -> None:
    ensure_dir(out_dir)
    np.savez(
        out_dir / f"{slide_id}_{path}.npz",
        valid_rows=valid_rows.astype(np.int64),
        neighbor_indices=nn_idx.astype(np.int64),
        similarities=nn_sim.astype(np.float32),
        weights=weights.astype(np.float32),
    )



def run_checkpoint_evaluation(eval_root: Path, training_num: str, checkpoint_num: str) -> Path:
    cfg = load_protocol(eval_root)
    paths = get_protocol_paths(eval_root)
    checkpoint_name = f"HEencoder_t{training_num}e{checkpoint_num}_clip"
    checkpoint_dir = eval_root / checkpoint_name
    _prepare_new_checkpoint_output_tree(checkpoint_dir)

    logger = configure_logger(checkpoint_dir / "logs" / "run.log", name=f"A2_eval_{training_num}_{checkpoint_num}")
    set_global_seed(int(cfg["runtime"]["seed"]))
    device = resolve_device(cfg["runtime"].get("device", "auto"))
    logger.info("Running A-2 evaluation for checkpoint training_num=%s checkpoint_num=%s on device=%s", training_num, checkpoint_num, device)

    reporting_cfg = cfg.get("reporting", {})
    save_debug_tables = bool(reporting_cfg.get("save_debug_tables", False))
    save_prediction_arrays = bool(reporting_cfg.get("save_prediction_arrays", False))
    save_selected_k_retrieval = bool(reporting_cfg.get("save_selected_k_retrieval", False))
    save_subset_pooled_expression_metrics = bool(reporting_cfg.get("save_subset_pooled_expression_metrics", True))
    # When expression is disabled, the reference-expression path is skipped
    expression_enabled = bool(cfg.get("expression", {}).get("enabled", True))
    logger.info(
        "Output options: save_debug_tables=%s save_prediction_arrays=%s save_selected_k_retrieval=%s "
        "save_subset_pooled_expression_metrics=%s umap_enabled=%s expression_enabled=%s",
        save_debug_tables,
        save_prediction_arrays,
        save_selected_k_retrieval,
        save_subset_pooled_expression_metrics,
        bool(cfg.get("umap", {}).get("enabled", False)),
        expression_enabled,
    )
    if not expression_enabled:
        logger.info(
            "expression.enabled=False: skipping the reference-expression transfer "
            "(cell-type retrieval and tuning are unaffected)."
        )

    primary_expression_definition = {
        "subset": "STHELAR_test",
        "path": "he",
        "population": "matched_covered_he",
        "metric_group": "expression",
        "gene_panel": "G_base_qc",
        "metric": "gene_wise_pearson",
        "aggregation": "slide_macro_average",
    }

    # Effective retrieval backend (auto resolved to faiss or exact), recorded in the manifest
    effective_retrieval_backend = _resolve_effective_retrieval_backend(
        str(cfg["retrieval"]["backend"])
    )

    if get_full_path is None:
        raise RuntimeError("Could not import src.utils.constants.get_full_path from the project")

    trained_encoder_dir = get_full_path("train", "trainings", "trainings_HistoST", f"training_{training_num}", "trained_encoders")
    he_encoder_path = trained_encoder_dir / f"HEencoder_t{training_num}e{checkpoint_num}.pt"
    st_cell_bundle_path = trained_encoder_dir / f"STencoderCell_t{training_num}e{checkpoint_num}.pt"
    raw_checkpoint_path = get_full_path("train", "trainings", "trainings_HistoST", f"training_{training_num}", "training", "log", "checkpoints", f"checkpoint_epoch_{checkpoint_num}.pth")
    if not he_encoder_path.exists() or not st_cell_bundle_path.exists():
        raise FileNotFoundError(f"Missing exported encoder(s) under {trained_encoder_dir}")

    actual_modelgroup = _infer_modelgroup_from_he_encoder(he_encoder_path)
    expected_modelgroup = str(cfg["identity"]["modelgroup"])
    if actual_modelgroup != expected_modelgroup:
        raise RuntimeError(f"Checkpoint modelgroup mismatch. Protocol expects '{expected_modelgroup}' but encoder resolves to '{actual_modelgroup}'.")

    st_bundle = torch.load(st_cell_bundle_path, map_location="cpu", weights_only=False)
    scconcept_training_id = str(st_bundle.get("scconcept_training_id", "")).strip()
    clip_cfg_bundle = st_bundle.get("config", {}).get("model", {}).get("clip", {})
    if float(clip_cfg_bundle.get("lambda_token", 0.0)) <= 0.0:
        raise RuntimeError(
            "A-2 requires the ST token/cell projection head "
            "(lambda_token > 0 during CLIP training). "
            f"Current checkpoint does not provide it: {st_cell_bundle_path}"
        )

    if not scconcept_training_id:
        raise RuntimeError(f"Missing scconcept_training_id in {st_cell_bundle_path}")

    training_dataset_id = str(cfg['identity']['training_dataset_id'])
    eval_data_root = get_full_path('eval', 'eval_datasets', 'st_eval_data', f'eval_st_{training_dataset_id}')
    raw_ref_subdir = str(cfg["references"].get("raw_reference_embedding_subdir", "scConcept_embs"))
    raw_ref_embedding_dir = get_full_path("eval", "eval_datasets", "scRNAseq_references", raw_ref_subdir, scconcept_training_id)
    if not raw_ref_embedding_dir.exists():
        raise FileNotFoundError(raw_ref_embedding_dir)

    slide_df = pd.read_parquet(paths.shared_dir / "manifests" / "slides.parquet")
    cell_backbone_df = pd.read_parquet(paths.shared_dir / "manifests" / "sthelar_cells_backbone.parquet")
    cell_backbone_slide_key = cell_backbone_df["slide_id"].astype(str)
    cell_backbone_index_by_slide = {
        str(slide_id): np.asarray(pos, dtype=np.int64)
        for slide_id, pos in cell_backbone_df.groupby(cell_backbone_slide_key, sort=False).indices.items()
    }
    vocab_payload = read_json(paths.shared_dir / "manifests" / "label_vocabulary.json")
    label_order = vocab_payload["labels"]
    label_to_idx = {str(k): int(v) for k, v in vocab_payload["label_to_index"].items()}
    mapped_ref_label_col = cfg["references"]["mapped_label_column_template"].format(ct_cat_id=cfg["identity"]["ct_cat_id"])
    fine_ref_label_col = cfg["references"]["fine_label_column"]

    # Same label map as in _2 (resolved once per run)
    label_granularity = normalize_granularity(cfg["identity"].get("label_granularity", "fine"))
    _coarse_version_raw = cfg["identity"].get("coarse_mapping_version", None)
    coarse_mapping_version = str(_coarse_version_raw).strip() if _coarse_version_raw else None
    label_map = resolve_label_map(
        cfg["identity"]["ct_cat_id"], label_granularity, coarse_mapping_version
    )
    vocab_granularity = str(vocab_payload.get("label_granularity", "fine")).strip().lower()
    if vocab_granularity != label_granularity:
        raise RuntimeError(
            "Shared artifacts were built at a different label granularity than this "
            f"eval_config.yaml requests: label_vocabulary.json says '{vocab_granularity}', "
            f"identity.label_granularity says '{label_granularity}'. Re-run "
            "_2_prepare_shared_artifacts.py under a new eval_id."
        )

    # Check the coarse mapping version and the mapping itself against the ones used by _2
    frozen_version_raw = vocab_payload.get("coarse_mapping_version", None)
    frozen_version = str(frozen_version_raw).strip() if frozen_version_raw else None
    if frozen_version != coarse_mapping_version:
        raise RuntimeError(
            "Shared artifacts were built with a different coarse mapping version than this "
            f"eval_config.yaml requests: label_vocabulary.json says {frozen_version!r}, "
            f"identity.coarse_mapping_version says {coarse_mapping_version!r}. Re-run "
            "_2_prepare_shared_artifacts.py under a new eval_id."
        )
    frozen_map_raw = vocab_payload.get("coarse_mapping", None)
    frozen_map = (
        None if frozen_map_raw is None else {str(k): str(v) for k, v in dict(frozen_map_raw).items()}
    )
    runtime_map = None if label_map is None else {str(k): str(v) for k, v in label_map.items()}
    if frozen_map != runtime_map:
        raise RuntimeError(
            "The coarse mapping resolved at run time does not match the one in "
            "label_vocabulary.json (written by _2), although the version strings agree "
            f"({coarse_mapping_version!r}): the mapping file was edited after _2. Restore it, "
            "or create a new mapping version and a new eval_id."
        )

    logger.info(
        "A-2 label_granularity=%s%s: scoring %d classes %s",
        label_granularity,
        "" if coarse_mapping_version is None else f" ({coarse_mapping_version})",
        len(label_order),
        list(label_order),
    )

    slides: List[SlideData] = []
    runtime_cache: Dict[str, Dict[Any, Any]] = {}
    logger.info("Caching slide data for %d slides:", len(slide_df))
    for num_slide, (_, slide_row) in enumerate(slide_df.iterrows()):
        logger.info("Slide %d / %d", num_slide + 1, len(slide_df))
        slide = _prepare_slide_data(
            slide_row=slide_row,
            cell_backbone_df=cell_backbone_df,
            cell_backbone_index_by_slide=cell_backbone_index_by_slide,
            checkpoint_dir=checkpoint_dir,
            eval_data_root=eval_data_root,
            shared_dir=paths.shared_dir,
            training_num=training_num,
            checkpoint_num=checkpoint_num,
            label_to_idx=label_to_idx,
            mapped_ref_label_col=mapped_ref_label_col,
            fine_ref_label_col=fine_ref_label_col,
            label_map=label_map,
            cfg=cfg,
            st_cell_bundle_path=st_cell_bundle_path,
            raw_ref_embedding_dir=raw_ref_embedding_dir,
            device=device,
            logger=logger,
            runtime_cache=runtime_cache,
        )
        _precompute_retrieval(
            slide,
            Kmax=int(cfg["retrieval"]["Kmax"]),
            backend=str(cfg["retrieval"]["backend"]),
            cfg=cfg,
            logger=logger,
        )
        slides.append(slide)

    train_subset = str(cfg["data"]["subset_for_tuning"])
    # Slides with a declared reference-coverage exception are scored but not used for tuning
    coverage_excluded = [
        s for s in slides
        if s.subset == train_subset and s.coverage_declared_exception is not None
    ]
    for s in coverage_excluded:
        logger.warning(
            "Excluding slide '%s' from the (k, tau) tuning pool: declared reference-coverage "
            "exception (%s). It is still scored and still reported.",
            s.slide_id,
            s.coverage_declared_exception,
        )

    train_slides = [
        s for s in slides
        if s.subset == train_subset
        and s.coverage_declared_exception is None
        and s.meets_minimum_reference_cells
        and s.meets_minimum_class_support
        and np.any(s.matched_valid & s.gt_mapped_label_scorable)
    ]
    if len(train_slides) == 0:
        raise RuntimeError(
            "No STHELAR_train slide satisfies the thresholds required for tuning: "
            "(1) not a declared reference-coverage exception "
            f"({len(coverage_excluded)} of this subset's slide(s) excluded on that ground: "
            f"{sorted(s.slide_id for s in coverage_excluded)}), "
            "(2) minimum filtered reference cells, "
            "(3) minimum mapped-label class support, and "
            "(4) at least one matched valid cell with a classification-scorable mapped GT label."
        )

    logger.info(
        "Eligible tuning slides after coverage exceptions + classification/reference thresholds "
        "+ matched validity: %d / %d (%d excluded as declared coverage exceptions).",
        len(train_slides),
        sum(1 for s in slides if s.subset == train_subset),
        len(coverage_excluded),
    )

    k_best, tau_best, cv_table = _tune_k_tau(
        train_slides=train_slides,
        k_grid=[int(x) for x in cfg["retrieval"]["k_grid"]],
        tau_grid=[float(x) for x in cfg["retrieval"]["tau_grid"]],
        fallback=cfg["tuning"]["fallback"],
        label_order=label_order,
        cfg=cfg,
        shared_dir=paths.shared_dir,
        logger=logger,
    )
    if not cv_table.empty:
        write_dataframe(cv_table, checkpoint_dir / "metrics" / "cv_table.csv")
    selection_reason = (
        str(cv_table["selection_reason"].iloc[0])
        if (not cv_table.empty and "selection_reason" in cv_table.columns)
        else ("fallback_no_train_slides" if len(train_slides) == 0 else "primary_objective")
    )
    logger.info("Selected hyperparameters k=%d tau=%.6f reason=%s", k_best, tau_best, selection_reason)

    metrics_rows: List[Dict[str, Any]] = []
    gap_rows: List[Dict[str, Any]] = []
    agreement_rows: List[Dict[str, Any]] = []
    summary_pred_rows: List[pd.DataFrame] = []
    pooled_expr_cell_tables: List[pd.DataFrame] = []

    for slide_i, slide in enumerate(slides, start=1):
        logger.info(
            "Starting final evaluation slide %d/%d slide=%s subset=%s",
            slide_i,
            len(slides),
            slide.slide_id,
            slide.subset,
        )

        if expression_enabled:
            _attach_reference_expression_subset(
                slide=slide,
                shared_dir=paths.shared_dir,
                k=k_best,
                cfg=cfg,
                logger=logger,
                materialize_in_memory=True,
            )
        try:
            _save_retrieval_cache(
                checkpoint_dir / "retrieval",
                slide.slide_id,
                "he",
                slide.he_valid_rows,
                slide.retrieval_he,
            )
            _save_retrieval_cache(
                checkpoint_dir / "retrieval",
                slide.slide_id,
                "gt",
                slide.gt_valid_rows,
                slide.retrieval_gt,
            )

            need_expression_cell_tables = (save_debug_tables or save_subset_pooled_expression_metrics)

            he_eval = _evaluate_query_path(
                slide,
                path="he",
                k=k_best,
                tau=tau_best,
                label_order=label_order,
                cfg=cfg,
                return_details=need_expression_cell_tables,
            )
            gt_eval = _evaluate_query_path(
                slide,
                path="gt",
                k=k_best,
                tau=tau_best,
                label_order=label_order,
                cfg=cfg,
                return_details=need_expression_cell_tables,
            )

            if save_selected_k_retrieval:
                _save_retrieval_outputs(
                    checkpoint_dir / "retrieval",
                    slide.slide_id,
                    "he",
                    he_eval["valid_rows"],
                    he_eval["nn_idx"],
                    he_eval["nn_sim"],
                    he_eval["weights"],
                )
                _save_retrieval_outputs(
                    checkpoint_dir / "retrieval",
                    slide.slide_id,
                    "gt",
                    gt_eval["valid_rows"],
                    gt_eval["nn_idx"],
                    gt_eval["nn_sim"],
                    gt_eval["weights"],
                )

            if save_prediction_arrays:
                _save_prediction_arrays(
                    checkpoint_dir / "predictions",
                    slide.slide_id,
                    "he",
                    he_eval["valid_rows"],
                    he_eval["pred_label_ids_full"],
                    he_eval["class_scores_full"],
                    he_eval["pred_expr_full"],
                )
                _save_prediction_arrays(
                    checkpoint_dir / "predictions",
                    slide.slide_id,
                    "gt",
                    gt_eval["valid_rows"],
                    gt_eval["pred_label_ids_full"],
                    gt_eval["class_scores_full"],
                    gt_eval["pred_expr_full"],
                )

            official_cls = int(slide.meets_minimum_reference_cells and slide.meets_minimum_class_support)
            official_expr = int(
                official_cls
                and int(slide.gene_panel_meta.get("G_base_qc", {}).get("gene_panel_effective_n", 0))
                >= int(cfg["expression"]["min_aligned_genes_official"])
            )

            # Flag of slides with a declared reference-coverage exception (for the fully_covered_slides scope)
            coverage_exception_active = int(slide.coverage_declared_exception is not None)

            he_metrics = he_eval["metrics"].copy()
            he_metrics["slide_id"] = slide.slide_id
            he_metrics["subset"] = slide.subset
            he_metrics["path"] = "he"
            he_metrics["official_classification_slide"] = official_cls
            he_metrics["official_expression_slide"] = official_expr
            he_metrics["official_expression_primary_metric"] = 0
            he_metrics["coverage_declared_exception_active"] = coverage_exception_active

            gt_metrics = gt_eval["metrics"].copy()
            gt_metrics["slide_id"] = slide.slide_id
            gt_metrics["subset"] = slide.subset
            gt_metrics["path"] = "gt_anchor"
            gt_metrics["official_classification_slide"] = official_cls
            gt_metrics["official_expression_slide"] = official_expr
            gt_metrics["official_expression_primary_metric"] = 0
            gt_metrics["coverage_declared_exception_active"] = coverage_exception_active

            primary_mask = (
                (he_metrics["subset"].astype(str) == primary_expression_definition["subset"])
                & (he_metrics["path"].astype(str) == primary_expression_definition["path"])
                & (he_metrics["population"].astype(str) == primary_expression_definition["population"])
                & (he_metrics["metric_group"].astype(str) == primary_expression_definition["metric_group"])
                & (he_metrics["gene_panel"].astype(str) == primary_expression_definition["gene_panel"])
                & (he_metrics["metric"].astype(str) == primary_expression_definition["metric"])
                & (he_metrics["official_expression_slide"].astype(int) == 1)
            )
            he_metrics.loc[primary_mask, "official_expression_primary_metric"] = 1

            metrics_rows.extend(he_metrics.to_dict(orient="records"))
            metrics_rows.extend(gt_metrics.to_dict(orient="records"))

            # Chance floor (path="chance_floor"): majority-class predictor on the populations of the
            # H&E headline, with the same per-slide macro floor
            for floor_pop, floor_mask in [
                ("matched_covered_he", slide.matched_valid & slide.gt_mapped_label_scorable & slide.covered),
                ("matched_all_he", slide.matched_valid & slide.gt_mapped_label_scorable),
            ]:
                floor = majority_class_metrics(
                    slide.gt_label_ids[np.flatnonzero(floor_mask)],
                    label_order=label_order,
                    macro_min_support=macro_min_class_support(cfg),
                )
                # n_cells and n_gt_present_classes are saved for comparison with the headline
                for _fm in ("macro_f1", "accuracy", "balanced_accuracy", "n_cells", "n_gt_present_classes"):
                    metrics_rows.append(
                        {
                            "population": floor_pop,
                            "metric_group": "classification",
                            "metric": _fm,
                            "value": floor[_fm],
                            "slide_id": slide.slide_id,
                            "subset": slide.subset,
                            "path": "chance_floor",
                            "gene_panel": NO_GENE_PANEL,
                            "official_classification_slide": official_cls,
                            "official_expression_slide": official_expr,
                            "official_expression_primary_metric": 0,
                            "coverage_declared_exception_active": coverage_exception_active,
                        }
                    )

            for pop_name, alias_name in [
                ("matched_covered_he", "mapped_macro_f1_matched_covered"),
                ("matched_all_he", "mapped_macro_f1_matched_all"),
            ]:
                alias_src = he_metrics.loc[
                    (he_metrics["population"] == pop_name)
                    & (he_metrics["metric_group"] == "classification")
                    & (he_metrics["metric"] == "macro_f1")
                ]
                if not alias_src.empty:
                    alias_row = alias_src.iloc[[0]].copy()
                    alias_row["metric"] = alias_name
                    metrics_rows.extend(alias_row.to_dict(orient="records"))

            he_diag = he_eval["diagnostics"].copy()
            gt_diag = gt_eval["diagnostics"].copy()
            he_diag["slide_id"] = slide.slide_id
            gt_diag["slide_id"] = slide.slide_id
            write_dataframe(he_diag, checkpoint_dir / "diagnostics" / f"{slide.slide_id}_he.csv")
            write_dataframe(gt_diag, checkpoint_dir / "diagnostics" / f"{slide.slide_id}_gt.csv")

            pred_table = slide.cells[
                [
                    "cell_id_int",
                    "cell_id",
                    "gt_label",
                    "selected_alpha",
                    "selected_patch_cell_row_index",
                    "selected_patch_id",
                    "selected_patch_index",
                    "n_candidate_patches",
                    "reference_stem",
                    "gene_universe_n_shared",
                    "gt_mapped_label_present_in_reference",
                ]
            ].copy()
            pred_table["slide_id"] = slide.slide_id
            pred_table["subset"] = slide.subset
            pred_table["he_pred_label"] = he_eval["pred_label_full"]
            pred_table["gt_pred_label"] = gt_eval["pred_label_full"]
            pred_table["he_valid"] = slide.he_valid.astype(np.int8)
            pred_table["gt_valid"] = slide.gt_valid.astype(np.int8)
            pred_table["matched_valid"] = slide.matched_valid.astype(np.int8)
            pred_table["covered"] = slide.covered.astype(np.int8)
            pred_table["gt_mapped_label_scorable"] = slide.gt_mapped_label_scorable.astype(np.int8)

            # Keep the _2 coverage flag (before projected-embedding QC); the run-time one uses the canonical name
            pred_table["gt_mapped_label_present_in_reference_pre_embedding_qc"] = (
                pred_table["gt_mapped_label_present_in_reference"].astype(np.int8)
            )
            pred_table["gt_mapped_label_present_in_reference"] = slide.covered.astype(np.int8)

            pred_table["is_gt_cell_present"] = np.int8(1)
            pred_table["is_he_structurally_eligible"] = (
                slide.he_structural_eligible
                & slide.cells["selected_patch_cell_row_index"].notna()
                & (slide.cells["selected_alpha"].fillna(0.0).astype(float) > float(cfg["runtime"]["he_support"]["min_selected_alpha"]))
            ).astype(np.int8)
            pred_table["is_gt_structurally_eligible"] = (
                slide.cells["_gt_emb_row"].notna()
                & slide.cells["_gt_expr_row"].notna()
                & (len(slide.gene_names) > 0)
            ).astype(np.int8)
            pred_table["is_he_query_valid"] = slide.he_valid.astype(np.int8)
            pred_table["is_gt_query_valid"] = slide.gt_valid.astype(np.int8)
            pred_table["is_matched_query_valid"] = slide.matched_valid.astype(np.int8)
            pred_table["is_mapped_label_scorable_he"] = (slide.he_valid & slide.gt_mapped_label_scorable).astype(np.int8)
            pred_table["is_mapped_label_scorable_gt"] = (slide.gt_valid & slide.gt_mapped_label_scorable).astype(np.int8)
            pred_table["is_fine_agreement_scorable"] = slide.matched_valid.astype(np.int8)
            expr_ready = np.full(
                slide.cells.shape[0],
                int(slide.gene_panel_meta.get("G_base_qc", {}).get("gene_panel_effective_n", 0))
                >= int(cfg["expression"]["min_aligned_genes_official"]),
                dtype=bool,
            )
            pred_table["is_expression_scorable_he"] = (slide.he_valid & slide.gt_expr_present & expr_ready).astype(np.int8)
            pred_table["is_expression_scorable_gt"] = (slide.gt_valid & slide.gt_expr_present & expr_ready).astype(np.int8)
            pred_table["he_embedding_norm"] = np.linalg.norm(slide.z_he.astype(np.float32), axis=1).astype(np.float32)
            pred_table["gt_embedding_norm"] = np.linalg.norm(slide.z_gt.astype(np.float32), axis=1).astype(np.float32)
            pred_table["gt_expr_present"] = slide.gt_expr_present.astype(np.int8)
            pred_table["reference_cell_count"] = int(slide.ref_z.shape[0])
            pred_table["aligned_gene_count"] = int(len(slide.gene_names))
            pred_table["n_genes_raw_shared"] = int(slide.gene_panel_meta["G_s_qc"]["n_genes_raw_shared"])
            pred_table["n_genes_qc_shared"] = int(slide.gene_panel_meta["G_s_qc"]["n_genes_qc_shared"])
            pred_table["n_genes_g_base_intersection_raw"] = int(slide.gene_panel_meta["G_s_qc"]["n_genes_g_base_intersection_raw"])
            pred_table["n_genes_g_base_qc"] = int(slide.gene_panel_meta["G_s_qc"]["n_genes_g_base_qc"])
            pred_table["gene_qc_file"] = str(slide.gene_qc_file)
            pred_table["meets_minimum_reference_cells"] = int(slide.meets_minimum_reference_cells)
            pred_table["meets_minimum_class_support"] = int(slide.meets_minimum_class_support)
            pred_table["meets_minimum_aligned_genes_official"] = int(slide.meets_minimum_aligned_genes_official)
            pred_table["meets_minimum_G_base_qc_genes_official"] = int(
                int(slide.gene_panel_meta.get("G_base_qc", {}).get("gene_panel_effective_n", 0))
                >= int(cfg["expression"]["min_aligned_genes_official"])
            )

            matched_mask = slide.matched_valid
            matched_idx = np.flatnonzero(matched_mask)
            if matched_idx.size > 0:
                zhe = l2_normalize(slide.z_he[matched_idx])
                zgt = l2_normalize(slide.z_gt[matched_idx])
                cos_he_gt = np.sum(zhe * zgt, axis=1)
                pred_table["cos_he_gt"] = np.nan
                pred_table.loc[matched_idx, "cos_he_gt"] = cos_he_gt
                # Map valid-row order back to matched rows in one vectorized pass.
                pred_table["jaccard_he_gt_neighbors"] = np.nan
                he_valid_rows_arr = np.asarray(he_eval["valid_rows"], dtype=np.int64)
                gt_valid_rows_arr = np.asarray(gt_eval["valid_rows"], dtype=np.int64)
                he_pos_by_row = pd.Series(
                    np.arange(he_valid_rows_arr.size, dtype=np.int64),
                    index=he_valid_rows_arr,
                )
                gt_pos_by_row = pd.Series(
                    np.arange(gt_valid_rows_arr.size, dtype=np.int64),
                    index=gt_valid_rows_arr,
                )
                he_pos = he_pos_by_row.reindex(matched_idx).to_numpy(dtype=float)
                gt_pos = gt_pos_by_row.reindex(matched_idx).to_numpy(dtype=float)
                keep = np.isfinite(he_pos) & np.isfinite(gt_pos)
                if keep.any():
                    he_pos_i = he_pos[keep].astype(np.int64)
                    gt_pos_i = gt_pos[keep].astype(np.int64)
                    jvals = jaccard_at_k(
                        he_eval["nn_idx"][he_pos_i],
                        gt_eval["nn_idx"][gt_pos_i],
                        k=int(k_best),
                    )
                    pred_table.loc[matched_idx[keep], "jaccard_he_gt_neighbors"] = jvals
            else:
                pred_table["cos_he_gt"] = np.nan
                pred_table["jaccard_he_gt_neighbors"] = np.nan

            # Classification anchor gaps.
            for pop in ["matched_all_he", "matched_covered_he"]:
                he_m = he_metrics.loc[
                    (he_metrics["population"] == pop)
                    & (he_metrics["metric_group"] == "classification")
                ]
                gt_pop = pop.replace("_he", "_gt")
                gt_m = gt_metrics.loc[
                    (gt_metrics["population"] == gt_pop)
                    & (gt_metrics["metric_group"] == "classification")
                ]

                for metric_name in ["accuracy", "macro_f1", "balanced_accuracy"]:
                    v_he = he_m.loc[he_m["metric"] == metric_name, "value"]
                    v_gt = gt_m.loc[gt_m["metric"] == metric_name, "value"]

                    gap_rows.append(
                        {
                            "slide_id": slide.slide_id,
                            "subset": slide.subset,
                            "population": pop.replace("_he", ""),
                            "metric_group": "classification",
                            "gene_panel": None,
                            "metric": metric_name,
                            "he_value": float(v_he.iloc[0]) if not v_he.empty else np.nan,
                            "gt_anchor_value": float(v_gt.iloc[0]) if not v_gt.empty else np.nan,
                            "gap_he_minus_gt_anchor": (
                                float(v_he.iloc[0]) - float(v_gt.iloc[0])
                                if (not v_he.empty and not v_gt.empty)
                                else np.nan
                            ),
                        }
                    )

            # Expression anchor gaps, separately for every gene panel
            for pop in ["matched_all_he", "matched_covered_he"]:
                gt_pop = pop.replace("_he", "_gt")

                for panel_name, panel_meta in slide.gene_panel_meta.items():
                    he_m = he_metrics.loc[
                        (he_metrics["population"] == pop)
                        & (he_metrics["metric_group"] == "expression")
                        & (he_metrics["gene_panel"] == panel_name)
                    ]
                    gt_m = gt_metrics.loc[
                        (gt_metrics["population"] == gt_pop)
                        & (gt_metrics["metric_group"] == "expression")
                        & (gt_metrics["gene_panel"] == panel_name)
                    ]

                    for metric_name in [
                        "gene_wise_pearson",
                        "gene_wise_spearman",
                        "mean_cellwise_pearson",
                        "mean_cellwise_spearman",
                        "mean_mae",
                        "mean_mse",
                    ]:
                        v_he = he_m.loc[he_m["metric"] == metric_name, "value"]
                        v_gt = gt_m.loc[gt_m["metric"] == metric_name, "value"]

                        row = {
                            "slide_id": slide.slide_id,
                            "subset": slide.subset,
                            "population": pop.replace("_he", ""),
                            "metric_group": "expression",
                            "metric": metric_name,
                            "he_value": float(v_he.iloc[0]) if not v_he.empty else np.nan,
                            "gt_anchor_value": float(v_gt.iloc[0]) if not v_gt.empty else np.nan,
                            "gap_he_minus_gt_anchor": (
                                float(v_he.iloc[0]) - float(v_gt.iloc[0])
                                if (not v_he.empty and not v_gt.empty)
                                else np.nan
                            ),
                        }
                        row.update(panel_meta)
                        gap_rows.append(row)

            mapped_agreement = agreement_rate(
                he_eval["pred_label_full"][matched_mask].astype(str),
                gt_eval["pred_label_full"][matched_mask].astype(str),
            )
            # fine agreement using the same weighted transfer rule, but on original_cell_type
            he_valid_rows = he_eval["valid_rows"]
            gt_valid_rows = gt_eval["valid_rows"]
            he_top_fine = np.full(slide.cells.shape[0], "", dtype=object)
            gt_top_fine = np.full(slide.cells.shape[0], "", dtype=object)
            if he_valid_rows.size > 0:
                he_top_fine[he_valid_rows] = _weighted_label_vote_from_neighbors(
                    he_eval["nn_idx"], he_eval["weights"], slide.ref_fine_labels
                )
            if gt_valid_rows.size > 0:
                gt_top_fine[gt_valid_rows] = _weighted_label_vote_from_neighbors(
                    gt_eval["nn_idx"], gt_eval["weights"], slide.ref_fine_labels
                )
            fine_agreement = agreement_rate(he_top_fine[matched_mask].astype(str), gt_top_fine[matched_mask].astype(str))
            cond_mask = matched_mask & (he_eval["pred_label_full"].astype(str) == gt_eval["pred_label_full"].astype(str))
            fine_agree_cond = agreement_rate(he_top_fine[cond_mask].astype(str), gt_top_fine[cond_mask].astype(str))

            pred_table["he_pred_fine"] = he_top_fine
            pred_table["gt_pred_fine"] = gt_top_fine
            pred_table["gt_projected_embedding_row_index"] = slide.cells["_gt_emb_row"].fillna(-1).astype(np.int64)
            pred_table["gt_expression_row_index"] = slide.cells["_gt_expr_row"].fillna(-1).astype(np.int64)
            pred_table["gene_universe_file"] = str(slide.gene_universe_file)
            pred_table["reference_row_filter_file"] = str(slide.reference_row_filter_file)
            pred_table["projected_reference_cache"] = str(slide.projected_ref_cache)

            write_dataframe(pred_table, checkpoint_dir / "predictions" / f"{slide.slide_id}.csv")
            summary_pred_rows.append(pred_table.copy())

            agreement_rows.append(
                {
                    "slide_id": slide.slide_id,
                    "subset": slide.subset,
                    "mapped_agreement": mapped_agreement,
                    "fine_agreement": fine_agreement,
                    "fine_agreement_given_mapped_agreement": fine_agree_cond,
                    "matched_cells": int(np.sum(matched_mask)),
                }
            )

            # Pooled-over-cells expression inputs (independent of save_debug_tables)
            if save_subset_pooled_expression_metrics:
                for (pop_name, panel_name), tbl in he_eval["expr_tables_cell"].items():
                    tmp_pool = tbl.copy()
                    tmp_pool["subset"] = slide.subset
                    tmp_pool["path"] = "he"
                    tmp_pool["population"] = pop_name
                    tmp_pool["gene_panel"] = panel_name
                    pooled_expr_cell_tables.append(tmp_pool)

                for (pop_name, panel_name), tbl in gt_eval["expr_tables_cell"].items():
                    tmp_pool = tbl.copy()
                    tmp_pool["subset"] = slide.subset
                    tmp_pool["path"] = "gt_anchor"
                    tmp_pool["population"] = pop_name
                    tmp_pool["gene_panel"] = panel_name
                    pooled_expr_cell_tables.append(tmp_pool)

            if save_debug_tables:
                for pop_name, tbl in he_eval["per_class_tables"].items():
                    tmp = tbl.copy()
                    tmp["slide_id"] = slide.slide_id
                    tmp["subset"] = slide.subset
                    tmp["path"] = "he"
                    tmp["population"] = pop_name
                    write_dataframe(tmp, checkpoint_dir / "metrics" / f"{slide.slide_id}_he_{pop_name}_per_class.csv")
                for pop_name, cm in he_eval["confusions"].items():
                    np.save(checkpoint_dir / "metrics" / f"{slide.slide_id}_he_{pop_name}_confusion.npy", cm)
                for pop_name, tbl in gt_eval["per_class_tables"].items():
                    tmp = tbl.copy()
                    tmp["slide_id"] = slide.slide_id
                    tmp["subset"] = slide.subset
                    tmp["path"] = "gt_anchor"
                    tmp["population"] = pop_name
                    write_dataframe(tmp, checkpoint_dir / "metrics" / f"{slide.slide_id}_gt_{pop_name}_per_class.csv")
                for pop_name, cm in gt_eval["confusions"].items():
                    np.save(checkpoint_dir / "metrics" / f"{slide.slide_id}_gt_{pop_name}_confusion.npy", cm)
                for (pop_name, panel_name), tbl in he_eval["expr_tables_gene"].items():
                    write_dataframe(tbl, checkpoint_dir / "metrics" / f"{slide.slide_id}_he_{pop_name}_{panel_name}_gene_metrics.csv")
                for (pop_name, panel_name), tbl in he_eval["expr_tables_cell"].items():
                    write_dataframe(tbl, checkpoint_dir / "metrics" / f"{slide.slide_id}_he_{pop_name}_{panel_name}_cell_metrics.csv")
                for (pop_name, panel_name), tbl in gt_eval["expr_tables_gene"].items():
                    write_dataframe(tbl, checkpoint_dir / "metrics" / f"{slide.slide_id}_gt_{pop_name}_{panel_name}_gene_metrics.csv")
                for (pop_name, panel_name), tbl in gt_eval["expr_tables_cell"].items():
                    write_dataframe(tbl, checkpoint_dir / "metrics" / f"{slide.slide_id}_gt_{pop_name}_{panel_name}_cell_metrics.csv")

            if bool(cfg["umap"]["enabled"]):
                logger.info("Starting UMAP diagnostics for slide=%s", slide.slide_id)
                run_umap_for_slide(
                    slide=slide,
                    he_eval=he_eval,
                    gt_eval=gt_eval,
                    shared_dir=paths.shared_dir,
                    out_dir=checkpoint_dir / "plots",
                    cfg=cfg,
                    logger=logger,
                )

        finally:
            _release_reference_expression_subset(slide)

    metrics_df = pd.DataFrame(metrics_rows)
    gaps_df = pd.DataFrame(gap_rows)
    agreement_df = pd.DataFrame(agreement_rows)
    preds_df = pd.concat(summary_pred_rows, axis=0, ignore_index=True) if summary_pred_rows else pd.DataFrame()

    if not metrics_df.empty:
        write_dataframe(metrics_df, checkpoint_dir / "metrics" / "all_metrics.csv")

        if "official_expression_primary_metric" in metrics_df.columns:
            primary_expr = metrics_df.loc[
                metrics_df["official_expression_primary_metric"].fillna(0).astype(int) == 1
            ].copy()
            if not primary_expr.empty:
                primary_summary = (
                    primary_expr
                    .groupby(
                        ["subset", "path", "population", "metric_group", "gene_panel", "metric"],
                        dropna=False,
                    )["value"]
                    .mean()
                    .reset_index()
                )
                primary_summary["summary_scope"] = "official_primary_expression_slide_macro"
                write_dataframe(
                    primary_summary,
                    checkpoint_dir / "metrics" / "primary_expression_metric.csv",
                )
    if not gaps_df.empty:
        write_dataframe(gaps_df, checkpoint_dir / "metrics" / "anchor_gaps.csv")
    if not agreement_df.empty:
        write_dataframe(agreement_df, checkpoint_dir / "metrics" / "agreement_metrics.csv")
    if not preds_df.empty:
        write_dataframe(preds_df, checkpoint_dir / "predictions" / "all_predictions.csv")

    pooled_cls_df = _subset_pooled_classification_metrics(
        preds_df=preds_df,
        label_order=label_order,
        label_to_idx=label_to_idx,
    )
    if not pooled_cls_df.empty:
        write_dataframe(pooled_cls_df, checkpoint_dir / "metrics" / "subset_pooled_classification_metrics.csv")

    pooled_expr_df = _subset_pooled_expression_metrics(pooled_expr_cell_tables)
    if not pooled_expr_df.empty:
        write_dataframe(pooled_expr_df, checkpoint_dir / "metrics" / "subset_pooled_expression_metrics.csv")

    if not gaps_df.empty:
        gap_summary = (
            gaps_df.groupby(["subset", "population", "metric_group", "gene_panel", "metric"], dropna=False)[["he_value", "gt_anchor_value", "gap_he_minus_gt_anchor"]]
            .mean()
            .reset_index()
        )
        write_dataframe(gap_summary, checkpoint_dir / "metrics" / "subset_anchor_gap_summary.csv")

    if not agreement_df.empty:
        agreement_summary = (
            agreement_df.groupby(["subset"], dropna=False)[["mapped_agreement", "fine_agreement", "fine_agreement_given_mapped_agreement", "matched_cells"]]
            .mean()
            .reset_index()
        )
        write_dataframe(agreement_summary, checkpoint_dir / "metrics" / "subset_agreement_summary.csv")

    slide_macro = []
    if not metrics_df.empty:
        for subset in metrics_df["subset"].dropna().unique().tolist():
            msub = metrics_df.loc[
                (metrics_df["subset"] == subset)
                & (metrics_df["metric"] != "reporting_status")
            ].copy()
            if msub.empty:
                continue

            grouped_all = _slide_macro_metric_summary(msub)
            grouped_all["subset"] = subset
            grouped_all["summary_scope"] = "all_slides"
            slide_macro.append(grouped_all)

            cls_mask = msub["official_classification_slide"] == 1
            expr_mask = cls_mask & (msub["official_expression_slide"] == 1)

            for metric_group, mask in [
                ("classification", cls_mask),
                ("expression", expr_mask),
            ]:
                g = msub.loc[(msub["metric_group"] == metric_group) & mask]
                if g.empty:
                    continue

                grouped = _slide_macro_metric_summary(g)
                grouped["subset"] = subset
                grouped["summary_scope"] = "official_slides"
                slide_macro.append(grouped)

            # Sensitivity scope 'fully_covered_slides': official slides without a declared coverage
            # exception (emitted only if it differs from official_slides)
            if "coverage_declared_exception_active" in msub.columns:
                covered_only = msub["coverage_declared_exception_active"].fillna(0).astype(int) == 0
                n_declared_slides = int(
                    msub.loc[~covered_only, "slide_id"].nunique()
                )
                if n_declared_slides > 0:
                    for metric_group, mask in [
                        ("classification", cls_mask & covered_only),
                        ("expression", expr_mask & covered_only),
                    ]:
                        g = msub.loc[(msub["metric_group"] == metric_group) & mask]
                        if g.empty:
                            continue
                        grouped = _slide_macro_metric_summary(g)
                        grouped["subset"] = subset
                        grouped["summary_scope"] = "fully_covered_slides"
                        slide_macro.append(grouped)
                    logger.info(
                        "Emitted summary_scope='fully_covered_slides' for subset=%s: "
                        "%d declared-coverage-exception slide(s) dropped from the official "
                        "slide-macro.",
                        subset,
                        n_declared_slides,
                    )
                else:
                    logger.info(
                        "summary_scope='fully_covered_slides' not emitted for subset=%s "
                        "(no declared coverage exception).",
                        subset,
                    )
    if slide_macro:
        slide_macro_df = pd.concat(slide_macro, axis=0, ignore_index=True)
        write_dataframe(slide_macro_df, checkpoint_dir / "metrics" / "subset_slide_macro_metrics.csv")
        # Chance-floor rows of the slide-macro summary
        floor_df = slide_macro_df.loc[slide_macro_df["path"].astype(str) == "chance_floor"].copy()
        if not floor_df.empty:
            write_dataframe(floor_df, checkpoint_dir / "metrics" / "chance_floor.csv")
    else:
        slide_macro_df = pd.DataFrame()

    file_info_cache: Dict[str, Dict[str, Any]] = {}

    def cached_file_info(path: Path, compute_sha256: bool = False) -> Dict[str, Any]:
        p = Path(path)
        key = f"{str(p.resolve())}|sha256={int(bool(compute_sha256))}"
        if key not in file_info_cache:
            file_info_cache[key] = file_info(p, compute_sha256=compute_sha256)
        return file_info_cache[key]

    manifest = {
        "eval_id": eval_root.name,
        "training_num": str(training_num),
        "checkpoint_num": str(checkpoint_num),
        "checkpoint_dir": str(checkpoint_dir),
        "protocol_config_hash_sha256": sha256_file(paths.protocol_yaml),
        "metadata_snapshot_hash_sha256": sha256_file(paths.metadata_snapshot_csv),
        "reference_mapping_snapshot_hash_sha256": sha256_file(paths.ref_mapping_snapshot_json),
        "st_cell_bundle_config_hash": hash_mapping(st_bundle.get("config", {})),
        "raw_training_checkpoint": cached_file_info(raw_checkpoint_path, compute_sha256=False) if raw_checkpoint_path.exists() else None,
        "he_encoder": cached_file_info(he_encoder_path, compute_sha256=False),
        "st_cell_bundle": cached_file_info(st_cell_bundle_path, compute_sha256=False),
        "retrieval_topKmax_files": {
            s.slide_id: {
                "he": str(checkpoint_dir / "retrieval" / f"{s.slide_id}_he_topKmax.npz"),
                "gt": str(checkpoint_dir / "retrieval" / f"{s.slide_id}_gt_topKmax.npz"),
            }
            for s in slides
        },
        "protocol_yaml": cached_file_info(paths.protocol_yaml, compute_sha256=True),
        "reference_mapping_snapshot": cached_file_info(paths.ref_mapping_snapshot_json, compute_sha256=True),
        "metric_semantics_json": cached_file_info(paths.metric_semantics_json, compute_sha256=True),
        "metadata_snapshot": cached_file_info(paths.metadata_snapshot_csv, compute_sha256=True),
        "selected_k": int(k_best),
        "selected_tau": float(tau_best),
        "selection_reason": selection_reason,
        # Slides used for the (k, tau) selection
        "tuning_pool": {
            "subset": str(cfg["data"]["subset_for_tuning"]),
            "n_slides_in_subset": int(
                sum(1 for s in slides if s.subset == str(cfg["data"]["subset_for_tuning"]))
            ),
            "slides_used": sorted(s.slide_id for s in train_slides),
            "slides_excluded_declared_coverage_exception": {
                s.slide_id: s.coverage_declared_exception for s in coverage_excluded
            },
        },
        "expression_enabled": bool(expression_enabled),
        # Whether the (k, tau) selection read an expression metric (see _tuning_reads_expression)
        "tuning_expression": {
            "reads_expression": bool(_tuning_reads_expression(cfg)),
            "computed": bool(expression_enabled and _tuning_reads_expression(cfg)),
            "objective": str(cfg["tuning"].get("objective", "mapped_macro_f1_matched_covered")),
            "tie_breaks": [str(t) for t in (cfg["tuning"].get("tie_breaks", []) or [])],
            "expression_enabled": bool(expression_enabled),
        },
        # Retrieval backend (auto is resolved at run time)
        "retrieval": {
            "backend_requested": str(cfg["retrieval"]["backend"]),
            "backend_effective": effective_retrieval_backend,
            "similarity": str(cfg["retrieval"]["similarity"]),
            "Kmax": int(cfg["retrieval"]["Kmax"]),
            "k_grid": [int(x) for x in cfg["retrieval"]["k_grid"]],
            "tau_grid": [float(x) for x in cfg["retrieval"]["tau_grid"]],
        },
        # Per-slide macro support floor
        "metrics": {
            "macro_min_class_support_per_slide": int(macro_min_class_support(cfg)),
            "applied_at": [
                "per_slide_headline_macro",
                "tuning_objective_macro",
                "per_slide_chance_floor_macro",
            ],
            "never_applied_at": ["subset_pooled_classification_metrics"],
        },
        # Slide admission (decided by _2)
        "slide_admission": {
            "min_scorable_gt_cells_per_slide": int(
                cfg["data"].get("min_scorable_gt_cells_per_slide", 0)
            ),
            "min_classes_per_slide": int(cfg["data"].get("min_classes_per_slide", 0)),
            "min_class_support_for_admission": int(
                cfg["data"].get("min_class_support_for_admission", 0)
            ),
            "admission_population": "is_cell_and_modelgroup_token_eligible",
            "decided_at": "_2_prepare_shared_artifacts",
            "excluded_slides_file": str(paths.shared_dir / "manifests" / "excluded_slides.json"),
        },
        "primary_expression_metric": primary_expression_definition,
        "scconcept_training_id": scconcept_training_id,
        "runtime_seed": int(cfg["runtime"]["seed"]),
        "software_versions": _software_versions(),
        "git_commit": _git_commit_if_available(),
        "label_columns": {
            "gt_label_column": cfg["data"]["gt_label_column"],
            "reference_mapped_label_column": mapped_ref_label_col,
            "reference_fine_label_column": fine_ref_label_col,
            # Label granularity of the mapped-label columns (reference_fine_label_column is never collapsed)
            "label_granularity": label_granularity,
            "coarse_mapping_version": coarse_mapping_version,
            "label_granularity_applies_to": [
                "gt_label_column",
                "reference_mapped_label_column",
            ],
            "n_scored_classes": int(len(label_order)),
        },
        "device": device,
        "actual_modelgroup": actual_modelgroup,
        "expected_modelgroup": expected_modelgroup,
        "gene_id_columns": {
            "reference_gene_id_column": cfg["genes"]["reference_gene_id_column"],
            "gt_gene_id_column": cfg["genes"]["gt_gene_id_column"],
            "artifact_alignment_gene_column": "gene",
            "artifact_gt_gene_name_column": "gt_gene_name",
            "gt_gene_name_source": "gt_h5ad.var_names",
            "g_base_match_field": cfg["expression"].get("g_base_match_field", "gt_gene_name"),
            "g_base_match_note": (
                "Only G_base membership uses this field. Expression alignment and metrics use gene IDs."
            ),
        },
        "embedding_dimensions": {
            s.slide_id: {
                "he_clip_dim": int(s.z_he.shape[1]),
                "gt_clip_dim": int(s.z_gt.shape[1]),
                "reference_clip_dim": int(s.ref_z.shape[1]),
            }
            for s in slides
        },
        "selected_main_row_files": {
            s.slide_id: cached_file_info(checkpoint_dir / "selected_main_he_rows" / f"{s.slide_id}.parquet")
            for s in slides
        },
        "upstream_artifacts": {
            s.slide_id: {
                "he_slide_embeddings": cached_file_info(s.he_slide_h5),
                "gt_cell_embeddings": cached_file_info(s.gt_emb_h5ad),
                "gt_eval_h5ad": cached_file_info(s.gt_eval_h5ad),
                "reference_h5ad": cached_file_info(s.reference_h5ad),
                "raw_reference_embedding": cached_file_info(s.raw_ref_h5),
                "projected_reference_cache": cached_file_info(s.projected_ref_cache),
                "reference_expression_cache": (
                    cached_file_info(s.reference_expression_cache_file)
                    if s.reference_expression_cache_file is not None and s.reference_expression_cache_file.exists()
                    else None
                ),
                "gene_universe_file": cached_file_info(s.gene_universe_file),
                "gene_qc_file": cached_file_info(s.gene_qc_file),
                "reference_row_filter_file": cached_file_info(s.reference_row_filter_file),
            }
            for s in slides
        },
        "slides": [
            {
                "slide_id": s.slide_id,
                "subset": s.subset,
                # Declared reference-coverage exception of the slide, or None
                "coverage_declared_exception": s.coverage_declared_exception,
                "n_cells": int(s.cells.shape[0]),
                "he_valid": int(np.sum(s.he_valid)),
                "gt_valid": int(np.sum(s.gt_valid)),
                "matched_valid": int(np.sum(s.matched_valid)),
                "matched_label_scorable": int(np.sum(s.matched_valid & s.gt_mapped_label_scorable)),
                "covered_matched": int(np.sum(s.matched_valid & s.covered)),
                "covered_matched_label_scorable": int(np.sum(s.matched_valid & s.gt_mapped_label_scorable & s.covered)),
                "ref_cells": int(s.ref_z.shape[0]),
                "aligned_genes": int(len(s.gene_names)),
                "n_genes_raw_shared": int(s.gene_panel_meta["G_s_qc"]["n_genes_raw_shared"]),
                "n_genes_qc_shared": int(s.gene_panel_meta["G_s_qc"]["n_genes_qc_shared"]),
                "n_genes_g_base_intersection_raw": int(s.gene_panel_meta["G_s_qc"]["n_genes_g_base_intersection_raw"]),
                "n_genes_g_base_qc": int(s.gene_panel_meta["G_s_qc"]["n_genes_g_base_qc"]),
                "gene_panels": {
                    name: {
                        "gene_panel_effective_n": int(meta["gene_panel_effective_n"]),
                        "gene_panel_requested_n": meta["gene_panel_requested_n"],
                        "gene_panel_status": meta["gene_panel_status"],
                    }
                    for name, meta in s.gene_panel_meta.items()
                },
                "gene_universe_hash": read_json(paths.shared_dir / "reference_audits" / f"{s.slide_id}.json")["gene_universe_hash"],
            }
            for s in slides
        ],
        "cv_table": str(checkpoint_dir / "metrics" / "cv_table.csv") if not cv_table.empty else None,
    }
    write_json(checkpoint_dir / "run_manifest.json", manifest)
    logger.info("Finished A-2 evaluation. Outputs stored in %s", checkpoint_dir)
    return checkpoint_dir
