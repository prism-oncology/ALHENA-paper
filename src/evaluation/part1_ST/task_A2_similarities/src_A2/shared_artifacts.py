import hashlib
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
import anndata as ad
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import sparse
import scanpy as sc

from src_A2.common import (
    CanonicalGeneRules,
    assert_no_duplicates,
    canonicalize_gene_ids,
    coerce_bool_series,
    ensure_dir,
    file_info,
    hash_mapping,
    read_json,
    summarize_counts,
    write_dataframe,
    write_json,
)
from src_A2.coverage import (
    STAGE_FREEZE,
    SlideCoverageReport,
    assert_coverage_contract,
    build_slide_coverage_report,
)
from src_A2.label_granularity import (
    apply_label_map,
    coarse_mapping_filename,
    collapse_label_counts,
    normalize_granularity,
    resolve_label_map,
)
from src_A2.protocol import get_protocol_paths
from src_A2.references import (
    build_label_counts,
    drop_unscorable_reference_rows,
    FilteredReference,
    filter_reference_rows,
    get_canonical_gene_ids_from_var,
    load_reference_mapping,
    read_obs_columns,
    resolve_reference_for_tissue,
)



def _canonical_rules_from_cfg(cfg: Mapping[str, Any]) -> CanonicalGeneRules:
    g = cfg["genes"]["canonicalization"]
    return CanonicalGeneRules(
        strip_whitespace=bool(g.get("strip_whitespace", True)),
        uppercase=bool(g.get("uppercase", False)),
        strip_ensembl_version=bool(g.get("strip_ensembl_version", False)),
    )


def _g_base_match_rules_from_cfg(cfg: Mapping[str, Any]) -> CanonicalGeneRules:
    """Canonicalization rules used only to match the G_base gene list (separate from genes.canonicalization)."""
    g = cfg.get("expression", {}).get("g_base_match_canonicalization", {})
    return CanonicalGeneRules(
        strip_whitespace=bool(g.get("strip_whitespace", True)),
        uppercase=bool(g.get("uppercase", False)),
        strip_ensembl_version=bool(g.get("strip_ensembl_version", False)),
    )



def _stable_int_from_text(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)



def _sample_reference_rows_stratified(
    row_indices: np.ndarray,
    labels: np.ndarray,
    max_total: int,
    min_per_class: int,
    seed: int,
) -> np.ndarray:
    """
    Deterministic stratified sampler that never exceeds max_total.
    It first allocates rows round-robin toward min_per_class, then
    fills any remaining budget round-robin over the remaining capacity.
    """
    row_indices = np.asarray(row_indices, dtype=np.int64)
    labels = np.asarray(labels, dtype=object)
    if row_indices.size <= max_total:
        return np.sort(row_indices.astype(np.int64, copy=False))

    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"row": row_indices, "label": labels.astype(str)})
    grouped_rows = [
        sub["row"].to_numpy(dtype=np.int64, copy=False)
        for _, sub in df.groupby("label", sort=True)
    ]
    n_classes = len(grouped_rows)
    if n_classes == 0:
        return np.empty((0,), dtype=np.int64)

    # Deterministic randomized class order
    order = rng.permutation(n_classes)
    caps = np.asarray([len(g) for g in grouped_rows], dtype=np.int64)
    quota = np.zeros(n_classes, dtype=np.int64)
    remaining = int(max_total)

    # Pass 1: try to satisfy min_per_class without ever exceeding max_total
    target_min = np.minimum(caps, int(min_per_class))
    while remaining > 0 and np.any(quota < target_min):
        progressed = False
        for i in order:
            if remaining == 0:
                break
            if quota[i] < target_min[i]:
                quota[i] += 1
                remaining -= 1
                progressed = True
        if not progressed:
            break

    # Pass 2: fill remaining budget over remaining capacity
    while remaining > 0 and np.any(quota < caps):
        progressed = False
        for i in order:
            if remaining == 0:
                break
            if quota[i] < caps[i]:
                quota[i] += 1
                remaining -= 1
                progressed = True
        if not progressed:
            break

    sampled = []
    for i, rows in enumerate(grouped_rows):
        q = int(quota[i])
        if q <= 0:
            continue
        if q >= rows.size:
            sampled.append(np.sort(rows))
        else:
            sampled.append(np.sort(rng.choice(rows, size=q, replace=False)))

    out = np.concatenate(sampled, axis=0) if sampled else np.empty((0,), dtype=np.int64)
    out = np.unique(out)
    if out.size > max_total:
        out = np.sort(rng.choice(out, size=max_total, replace=False))
    return np.sort(out.astype(np.int64, copy=False))



def _derive_cell_id_int(obs: pd.DataFrame) -> np.ndarray:
    """`cell_id_int` of a GT obs frame (is_cell rows only): the column if present, else the index suffix."""
    if "cell_id_int" in obs.columns:
        return obs["cell_id_int"].to_numpy(dtype=np.int32, copy=False)
    return np.asarray(
        [int(str(idx).rsplit("_", 1)[-1]) for idx in obs.index.astype(str)], dtype=np.int32
    )


def _load_missing_cells_for_modelgroup(
    eval_data_root: Path,
    modelgroup: str,
    slide_id: str,
    require_exists: bool = True,
) -> set:
    """Cells with no token support for the modelgroup (not H&E-structurally eligible for A-2)."""
    p = eval_data_root / f"cell_token_membership_missing_cells_{modelgroup}" / f"{slide_id}.parquet"
    if not p.exists():
        if require_exists:
            raise FileNotFoundError(
                f"Required structural-eligibility artifact is missing for modelgroup='{modelgroup}', "
                f"slide='{slide_id}': {p}"
            )
        return set()
    df = pq.read_table(str(p), columns=["cell_id_int"]).to_pandas()
    return set(df["cell_id_int"].astype(np.int32).tolist())



def _read_gt_cells_and_gene_ids(
    gt_h5ad_path: Path,
    slide_id: str,
    gt_label_column: str,
    gt_gene_id_column: str,
    rules: CanonicalGeneRules,
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Read the GT cell obs rows, the gene ids (used for alignment) and the gene names
    (report-only, from adata.var_names) of one GT h5ad.
    """
    a = ad.read_h5ad(gt_h5ad_path, backed="r")
    try:
        if "is_cell" not in a.obs.columns:
            raise KeyError(
                f"Missing 'is_cell' column in {gt_h5ad_path}. "
                "This column is required to identify cell rows for evaluation."
            )
        if gt_label_column not in a.obs.columns:
            raise KeyError(f"Missing GT label column '{gt_label_column}' in {gt_h5ad_path}")
        if gt_gene_id_column not in a.var.columns:
            raise KeyError(f"Missing var['{gt_gene_id_column}'] in {gt_h5ad_path}")

        obs_cols = ["is_cell", gt_label_column]
        if "cell_id_int" in a.obs.columns:
            obs_cols.append("cell_id_int")
        if "cell_id" in a.obs.columns:
            obs_cols.append("cell_id")

        obs = a.obs.loc[:, obs_cols].copy()
        mask = coerce_bool_series(obs["is_cell"], context=f"{gt_h5ad_path}:obs['is_cell']").to_numpy()
        h5ad_row_index = np.flatnonzero(mask).astype(np.int64)
        obs = obs.loc[mask].copy()
        obs["_h5ad_row_index"] = h5ad_row_index

        bad_label = obs[gt_label_column].isna() | (obs[gt_label_column].astype(str).str.len() == 0)
        if np.any(bad_label):
            examples = obs.index[bad_label][:10].astype(str).tolist()
            raise ValueError(
                f"GT label column '{gt_label_column}' contains null/empty values "
                f"in {gt_h5ad_path}. Examples: {examples}"
            )

        # Same derivation as in _read_gt_labels_for_coverage
        obs["cell_id_int"] = _derive_cell_id_int(obs)
        if "cell_id" not in obs.columns:
            obs["cell_id"] = obs["cell_id_int"].astype(str)

        obs.index = obs.index.astype(str)
        obs["cell_id_int"] = obs["cell_id_int"].astype(np.int32)
        obs["cell_id"] = obs["cell_id"].astype(str)

        if obs["cell_id_int"].duplicated().any():
            dup = obs.loc[obs["cell_id_int"].duplicated(), "cell_id_int"].astype(str).unique().tolist()[:20]
            raise ValueError(f"Duplicate cell_id_int values found in GT eval cells for {gt_h5ad_path}. Examples: {dup}")

        gene_ids = canonicalize_gene_ids(a.var[gt_gene_id_column].tolist(), rules=rules)
        assert_no_duplicates(gene_ids, context=f"{gt_h5ad_path}:{gt_gene_id_column}")

        gt_gene_names = pd.Index(a.var_names).astype(str).tolist()
        if len(gt_gene_names) != len(gene_ids):
            raise RuntimeError(
                f"GT var_names length mismatch in {gt_h5ad_path}: "
                f"{len(gt_gene_names)} names vs {len(gene_ids)} gene ids"
            )

        return obs, gene_ids, gt_gene_names
    finally:
        if getattr(a, "file", None) is not None:
            a.file.close()



def _build_gene_universe_from_gene_ids(
    gt_h5ad_path: Path,
    reference_h5ad_path: Path,
    gt_ids: Sequence[str],
    ref_ids: Sequence[str],
    gt_gene_names: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Shared gene universe of GT and reference from already-loaded gene ids (gt_gene_names are report-only)."""
    if gt_gene_names is not None and len(gt_gene_names) != len(gt_ids):
        raise ValueError(
            f"gt_gene_names length mismatch for {gt_h5ad_path.name}: "
            f"{len(gt_gene_names)} names vs {len(gt_ids)} gene ids"
        )


    gt_index = {g: i for i, g in enumerate(gt_ids)}
    ref_index = {g: i for i, g in enumerate(ref_ids)}
    shared = [g for g in gt_ids if g in ref_index]
    if len(shared) == 0:
        raise ValueError(f"No aligned genes between GT={gt_h5ad_path.name} and reference={reference_h5ad_path.name}")
    shared_gt_gene_names = (
        [str(gt_gene_names[gt_index[g]]) for g in shared]
        if gt_gene_names is not None
        else list(shared)
    )
    return {
        "genes": shared,
        "gt_gene_names": shared_gt_gene_names,
        "gt_var_indices": np.asarray([gt_index[g] for g in shared], dtype=np.int64),
        "reference_var_indices": np.asarray([ref_index[g] for g in shared], dtype=np.int64),
        "n_shared": int(len(shared)),
    }



def _reference_filter_signature(
    reference_h5ad_path: Path,
    reference_h5ad_info: Mapping[str, Any],
    mapped_label_column: str,
    fine_label_column: str,
    require_nonempty_expression: bool,
) -> Dict[str, Any]:
    return {
        "reference_h5ad": str(reference_h5ad_path),
        "reference_h5ad_sha256": (
            None
            if reference_h5ad_info.get("sha256") is None
            else str(reference_h5ad_info["sha256"])
        ),
        "reference_h5ad_size_bytes": int(reference_h5ad_info["size_bytes"]),
        "reference_h5ad_mtime": float(reference_h5ad_info["mtime"]),
        "reference_h5ad_mtime_ns": int(reference_h5ad_info.get("mtime_ns", 0)),
        "mapped_label_column": str(mapped_label_column),
        "fine_label_column": str(fine_label_column),
        "require_nonempty_expression": bool(require_nonempty_expression),
    }



def _reference_filter_cache_path(cache_dir: Path, signature: Mapping[str, Any], reference_h5ad_path: Path) -> Path:
    digest = hash_mapping(signature)[:16]
    return cache_dir / f"{reference_h5ad_path.stem}_{digest}.npz"



def _shared_reference_filter_cache_dir() -> Path:
    """Task-level (not eval_id-scoped) directory of the reference row-filter cache, next to the reference .h5ad files."""
    # Local import
    from src.utils.constants import get_full_path

    return ensure_dir(
        get_full_path("eval", "eval_datasets", "scRNAseq_references", "_a2_reference_filter_cache")
    )



def _filter_reference_rows_cached(
    reference_h5ad_path: Path,
    mapped_label_column: str,
    fine_label_column: str,
    require_nonempty_expression: bool,
    reference_h5ad_info: Mapping[str, Any],
    cache_dir: Path,
    used_cache_files: Optional[Dict[str, Dict[str, Any]]] = None,
) -> FilteredReference:
    """
    Reference row filter (filter_reference_rows) with a persistent cache keyed by _reference_filter_signature.
    used_cache_files, if given, records the cache file used by this run.
    """
    ensure_dir(cache_dir)
    signature = _reference_filter_signature(
        reference_h5ad_path=reference_h5ad_path,
        reference_h5ad_info=reference_h5ad_info,
        mapped_label_column=mapped_label_column,
        fine_label_column=fine_label_column,
        require_nonempty_expression=require_nonempty_expression,
    )
    cache_path = _reference_filter_cache_path(cache_dir, signature, reference_h5ad_path)
    manifest_path = cache_path.with_suffix(".json")

    def _record(cache_hit: bool) -> None:
        if used_cache_files is None:
            return
        used_cache_files[cache_path.name] = {
            "reference_h5ad": str(reference_h5ad_path),
            "cache_hit": bool(cache_hit),
            # False means identity rests on path + size + mtime alone (see `keyed_by`).
            "sha256_in_signature": signature["reference_h5ad_sha256"] is not None,
        }

    if cache_path.exists() and manifest_path.exists():
        try:
            manifest = read_json(manifest_path)
            if manifest.get("signature") == signature:
                obs = read_obs_columns(reference_h5ad_path, [mapped_label_column, fine_label_column])
                with np.load(cache_path) as npz:
                    filtered_rows = np.asarray(npz["filtered_rows"], dtype=np.int64)
                    fine_rows = np.asarray(npz["fine_rows"], dtype=np.int64)
                if filtered_rows.size == 0 or fine_rows.size == 0:
                    raise ValueError("Cached reference filter is empty")
                _record(cache_hit=True)
                return FilteredReference(
                    obs=obs,
                    filtered_rows=filtered_rows,
                    fine_rows=fine_rows,
                    mapped_label_column=mapped_label_column,
                    fine_label_column=fine_label_column,
                )
        except Exception:
            pass

    ref_filtered = filter_reference_rows(
        reference_h5ad_path=reference_h5ad_path,
        mapped_label_column=mapped_label_column,
        fine_label_column=fine_label_column,
        require_nonempty_expression=require_nonempty_expression,
    )
    # Atomic write (tmp file + replace); np.savez_compressed needs the .npz suffix
    tmp_path = cache_path.with_name(cache_path.name + ".tmp.npz")
    np.savez_compressed(
        tmp_path,
        filtered_rows=ref_filtered.filtered_rows,
        fine_rows=ref_filtered.fine_rows,
    )
    tmp_path.replace(cache_path)
    write_json(manifest_path, {"signature": signature, "cache_file": str(cache_path)})
    _record(cache_hit=False)
    return ref_filtered



# --------------------------------------------------------------------------------------------
# Reference-side gene stats (cached)
# Cache of the reference-side gene stats of the gene-QC table, shared across eval_ids (same scheme as the row-filter cache)
# (same scheme as the row-filter cache)

_REFERENCE_GENE_STATS_SCHEMA_VERSION = 1
_REFERENCE_GENE_STATS_ARRAYS = ("nonzero_cells", "nonzero_rate", "mean_log1p", "var_log1p")


def _int_array_digest(values: np.ndarray) -> str:
    """sha256 of an integer index array (as C-contiguous little-endian int64)."""
    arr = np.ascontiguousarray(np.asarray(values, dtype="<i8"))
    return hashlib.sha256(arr.tobytes()).hexdigest()


def _reference_gene_stats_signature(
    reference_h5ad_path: Path,
    reference_h5ad_info: Mapping[str, Any],
    selected_rows: np.ndarray,
    var_indices: np.ndarray,
    normalize_target_sum: float,
) -> Dict[str, Any]:
    """
    Cache key of one reference-side gene-stats block: reference file identity, selected rows,
    gene columns (in order), normalization target.
    """
    return {
        "schema_version": _REFERENCE_GENE_STATS_SCHEMA_VERSION,
        "reference_h5ad": str(reference_h5ad_path),
        "reference_h5ad_sha256": (
            None
            if reference_h5ad_info.get("sha256") is None
            else str(reference_h5ad_info["sha256"])
        ),
        "reference_h5ad_size_bytes": int(reference_h5ad_info["size_bytes"]),
        "reference_h5ad_mtime": float(reference_h5ad_info["mtime"]),
        "reference_h5ad_mtime_ns": int(reference_h5ad_info.get("mtime_ns", 0)),
        "selected_rows_sha256": _int_array_digest(selected_rows),
        "n_selected_rows": int(np.asarray(selected_rows).size),
        "var_indices_sha256": _int_array_digest(var_indices),
        "n_genes": int(np.asarray(var_indices).size),
        "normalize_target_sum": float(normalize_target_sum),
    }


def _reference_gene_stats_cache_path(
    cache_dir: Path, signature: Mapping[str, Any], reference_h5ad_path: Path
) -> Path:
    digest = hash_mapping(signature)[:16]
    return cache_dir / f"{reference_h5ad_path.stem}_genestats_{digest}.npz"


def _shared_reference_gene_stats_cache_dir() -> Path:
    """Task-level (not eval_id-scoped) directory of the reference gene-stats cache."""
    from src.utils.constants import get_full_path

    return ensure_dir(
        get_full_path("eval", "eval_datasets", "scRNAseq_references", "_a2_reference_gene_stats_cache")
    )


def _reference_gene_stats_cached(
    *,
    reference_h5ad_path: Path,
    reference_h5ad_info: Mapping[str, Any],
    selected_rows: np.ndarray,
    var_indices: np.ndarray,
    normalize_target_sum: float,
    row_block_size: int,
    target_block_mb: float,
    cache_dir: Path,
    used_cache_files: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, np.ndarray]:
    """
    Reference-side gene stats (as returned by _expression_stats_for_rows_on_raw_shared with keep_log_matrix=False),
    with a persistent cache (signature and array lengths are checked on every hit).
    """
    ensure_dir(cache_dir)
    selected_rows = np.asarray(selected_rows, dtype=np.int64)
    var_indices = np.asarray(var_indices, dtype=np.int64)
    signature = _reference_gene_stats_signature(
        reference_h5ad_path=reference_h5ad_path,
        reference_h5ad_info=reference_h5ad_info,
        selected_rows=selected_rows,
        var_indices=var_indices,
        normalize_target_sum=normalize_target_sum,
    )
    cache_path = _reference_gene_stats_cache_path(cache_dir, signature, reference_h5ad_path)
    manifest_path = cache_path.with_suffix(".json")

    def _record(cache_hit: bool) -> None:
        if used_cache_files is None:
            return
        used_cache_files[cache_path.name] = {
            "reference_h5ad": str(reference_h5ad_path),
            "cache_hit": bool(cache_hit),
            "n_selected_rows": int(signature["n_selected_rows"]),
            "n_genes": int(signature["n_genes"]),
            # False means identity rests on path + size + mtime alone (see `keyed_by`).
            "sha256_in_signature": signature["reference_h5ad_sha256"] is not None,
        }

    if cache_path.exists() and manifest_path.exists():
        try:
            manifest = read_json(manifest_path)
            if manifest.get("signature") == signature:
                with np.load(cache_path) as npz:
                    stats = {name: np.asarray(npz[name]) for name in _REFERENCE_GENE_STATS_ARRAYS}
                bad = [n for n, v in stats.items() if v.size != var_indices.size]
                if bad:
                    raise ValueError(f"Cached gene stats have the wrong length for {bad}")
                _record(cache_hit=True)
                return stats
        except Exception:
            pass

    stats, _ = _expression_stats_for_rows_on_raw_shared(
        h5ad_path=reference_h5ad_path,
        selected_rows=selected_rows,
        var_indices=var_indices,
        normalize_target_sum=float(normalize_target_sum),
        row_block_size=int(row_block_size),
        target_block_mb=float(target_block_mb),
        keep_log_matrix=False,
    )
    missing = [n for n in _REFERENCE_GENE_STATS_ARRAYS if n not in stats]
    if missing:
        raise RuntimeError(
            "Reference gene stats are missing expected keys "
            f"{missing}; bump _REFERENCE_GENE_STATS_SCHEMA_VERSION and extend "
            "_REFERENCE_GENE_STATS_ARRAYS rather than caching a partial block."
        )
    # Atomic write (tmp file + replace)
    tmp_path = cache_path.with_name(cache_path.name + ".tmp.npz")
    np.savez_compressed(tmp_path, **{name: stats[name] for name in _REFERENCE_GENE_STATS_ARRAYS})
    tmp_path.replace(cache_path)
    write_json(manifest_path, {"signature": signature, "cache_file": str(cache_path)})
    _record(cache_hit=False)
    return stats



def _reference_cache_key(
    *,
    ref_path_key: str,
    mapped_label_column: str,
    fine_label_column: str,
    require_nonempty_expression: bool,
    label_granularity: str,
    unscorable_labels: Sequence[str],
) -> tuple:
    """In-memory key of one eval-ready reference (row-filter identity + label_granularity + unscorable labels)."""
    return (
        str(ref_path_key),
        str(mapped_label_column),
        str(fine_label_column),
        bool(require_nonempty_expression),
        str(label_granularity),
        tuple(sorted(str(x) for x in unscorable_labels)),
    )



def _filtered_reference_for_eval(
    *,
    reference_h5ad_path: Path,
    reference_h5ad_info: Mapping[str, Any],
    cache_key: tuple,
    mapped_label_column: str,
    fine_label_column: str,
    require_nonempty_expression: bool,
    label_map: Optional[Mapping[str, str]],
    unscorable_labels: Sequence[str],
    cache_dir: Path,
    reference_filter_cache: Dict[tuple, FilteredReference],
    reference_scorable_drop_cache: Dict[tuple, Dict[str, int]],
    logger,
    used_cache_files: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Tuple[FilteredReference, Dict[str, int]]:
    """Eval-ready reference: officially filtered rows minus the unscorable ones (memoized in reference_filter_cache)."""
    if cache_key in reference_filter_cache:
        return reference_filter_cache[cache_key], reference_scorable_drop_cache[cache_key]

    ref_filtered = _filter_reference_rows_cached(
        reference_h5ad_path=reference_h5ad_path,
        mapped_label_column=mapped_label_column,
        fine_label_column=fine_label_column,
        require_nonempty_expression=require_nonempty_expression,
        reference_h5ad_info=reference_h5ad_info,
        cache_dir=cache_dir,
        used_cache_files=used_cache_files,
    )
    n_before = int(ref_filtered.filtered_rows.size)
    ref_filtered, drops = drop_unscorable_reference_rows(
        ref_filtered,
        label_map=label_map,
        unscorable_labels=unscorable_labels,
        reference_h5ad_path=reference_h5ad_path,
    )
    if drops:
        n_dropped = int(sum(drops.values()))
        logger.info(
            "Reference '%s': dropped %d/%d officially filtered cell(s) (%.2f%%) with a "
            "classification-unscorable mapped label %s (removed from the retrieval pool). Per label: %s.",
            reference_h5ad_path.stem,
            n_dropped,
            n_before,
            100.0 * n_dropped / float(max(1, n_before)),
            sorted(str(x) for x in unscorable_labels),
            drops,
        )

    reference_filter_cache[cache_key] = ref_filtered
    reference_scorable_drop_cache[cache_key] = drops
    return ref_filtered, drops



def _normalize_log1p_on_shared_genes(X: np.ndarray, target_sum: float) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    row_sum = X.sum(axis=1, keepdims=True)
    scale = np.divide(float(target_sum), row_sum, out=np.zeros_like(row_sum), where=row_sum > 0)
    return np.log1p(X * scale).astype(np.float32, copy=False)


def _backed_x_is_sparse_like(a: ad.AnnData) -> bool:
    x = a.X
    if sparse.issparse(x):
        return True
    return str(getattr(x, "format", "")).lower() in {"csr", "csc"}


def _read_backed_rows_vars_block(
    a: ad.AnnData,
    block_start: int,
    block_stop: int,
    rel_rows: np.ndarray,
    var_sorted: np.ndarray,
    var_restore: np.ndarray,
) -> np.ndarray:
    rel_rows = np.asarray(rel_rows, dtype=np.int64)
    var_sorted = np.asarray(var_sorted, dtype=np.int64)
    var_sorted_list = var_sorted.tolist()

    if _backed_x_is_sparse_like(a):
        try:
            x_block = a.X[block_start:block_stop, :]
            X = x_block[rel_rows, :][:, var_sorted]
        except Exception:
            X = a[block_start:block_stop, var_sorted_list].X
            X = X[rel_rows]
    else:
        X = a[block_start:block_stop, var_sorted_list].X
        X = X[rel_rows]

    if sparse.issparse(X):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float32)

    if var_restore is not None:
        X = X[:, var_restore]
    return X


def _expression_stats_for_rows_on_raw_shared(
    h5ad_path: Path,
    selected_rows: np.ndarray,
    var_indices: np.ndarray,
    normalize_target_sum: float,
    row_block_size: int = 8192,
    target_block_mb: float = 128.0,
    keep_log_matrix: bool = False,
) -> Tuple[Dict[str, np.ndarray], Optional[np.ndarray]]:
    selected_rows = np.asarray(selected_rows, dtype=np.int64)
    var_indices = np.asarray(var_indices, dtype=np.int64)

    if selected_rows.ndim != 1 or var_indices.ndim != 1:
        raise ValueError("selected_rows and var_indices must be 1D arrays")
    if selected_rows.size == 0:
        raise ValueError(f"No selected rows for expression stats: {h5ad_path}")
    if var_indices.size == 0:
        raise ValueError(f"No selected genes for expression stats: {h5ad_path}")
    if selected_rows.size > 1 and np.any(selected_rows[1:] <= selected_rows[:-1]):
        raise ValueError("selected_rows must be sorted unique")

    var_order = np.argsort(var_indices, kind="mergesort")
    var_sorted = var_indices[var_order]
    if np.array_equal(var_order, np.arange(var_order.size)):
        var_restore = None
    else:
        var_restore = np.empty_like(var_order)
        var_restore[var_order] = np.arange(var_order.size, dtype=np.int64)

    n_rows = int(selected_rows.size)
    n_genes = int(var_indices.size)

    nonzero = np.zeros(n_genes, dtype=np.int64)
    sum_log = np.zeros(n_genes, dtype=np.float64)
    sumsq_log = np.zeros(n_genes, dtype=np.float64)
    log_blocks: List[np.ndarray] = []

    target_bytes = max(16.0, float(target_block_mb)) * (1024 ** 2)
    max_rows_by_memory = max(1, int(target_bytes // max(1, n_genes * np.dtype(np.float32).itemsize)))
    scan_row_block = max(1, min(int(row_block_size), max_rows_by_memory))

    a = ad.read_h5ad(h5ad_path, backed="r")
    try:
        n_obs = int(a.n_obs)
        n_vars = int(a.n_vars)
        if int(selected_rows[-1]) >= n_obs:
            raise IndexError(f"selected row exceeds n_obs for {h5ad_path}")
        if int(var_indices.max()) >= n_vars:
            raise IndexError(f"selected var exceeds n_vars for {h5ad_path}")

        ptr = 0
        while ptr < n_rows:
            block_start = int(selected_rows[ptr])
            block_stop = min(n_obs, block_start + scan_row_block)
            next_ptr = int(np.searchsorted(selected_rows, block_stop, side="left"))
            if next_ptr <= ptr:
                next_ptr = ptr + 1
                block_stop = min(n_obs, int(selected_rows[ptr]) + 1)

            rel_rows = selected_rows[ptr:next_ptr] - block_start
            X_raw = _read_backed_rows_vars_block(
                a=a,
                block_start=block_start,
                block_stop=block_stop,
                rel_rows=rel_rows,
                var_sorted=var_sorted,
                var_restore=var_restore,
            )

            nonzero += np.count_nonzero(X_raw > 0, axis=0).astype(np.int64)
            X_log = _normalize_log1p_on_shared_genes(X_raw, target_sum=float(normalize_target_sum))
            sum_log += X_log.sum(axis=0, dtype=np.float64)
            sumsq_log += np.square(X_log, dtype=np.float64).sum(axis=0, dtype=np.float64)

            if keep_log_matrix:
                log_blocks.append(X_log.astype(np.float32, copy=False))

            ptr = next_ptr
    finally:
        if getattr(a, "file", None) is not None:
            a.file.close()

    mean_log = sum_log / float(n_rows)
    var_log = np.maximum((sumsq_log / float(n_rows)) - np.square(mean_log), 0.0)

    stats = {
        "nonzero_cells": nonzero.astype(np.int64),
        "nonzero_rate": (nonzero.astype(np.float64) / float(n_rows)).astype(np.float64),
        "mean_log1p": mean_log.astype(np.float32),
        "var_log1p": var_log.astype(np.float32),
    }
    log_matrix = np.concatenate(log_blocks, axis=0) if keep_log_matrix else None
    return stats, log_matrix


def _load_g_base_gene_set_for_a2(
    eval_data_root: Path,
    cfg: Mapping[str, Any],
    rules: CanonicalGeneRules,
) -> set:
    g_base_id = cfg.get("expression", {}).get("g_base_id")
    if not g_base_id:
        return set()

    p = eval_data_root / "G_base" / str(g_base_id) / "G_base_genes.json"
    if not p.exists():
        raise RuntimeError(f"G_base gene set not found: {p}")

    payload = read_json(p)
    genes = payload.get("genes")
    if not isinstance(genes, list) or len(genes) == 0:
        raise RuntimeError(f"Invalid or empty G_base gene set in {p}; expected a non-empty 'genes' list.")

    genes_canon = canonicalize_gene_ids(genes, rules=rules)
    assert_no_duplicates(genes_canon, context=f"{p}:genes")
    return set(genes_canon)


def _g_base_membership_values(
    genes: Sequence[str],
    gt_gene_names: Sequence[str],
    cfg: Mapping[str, Any],
) -> Tuple[List[str], str]:
    """Per-gene values used for G_base membership only (gene ids or GT gene names, per expression.g_base_match_field)."""
    field = str(cfg.get("expression", {}).get("g_base_match_field", "gt_gene_name"))
    if field == "gene":
        return list(map(str, genes)), field
    if field == "gt_gene_name":
        return list(map(str, gt_gene_names)), field
    raise ValueError(
        f"Unsupported expression.g_base_match_field={field!r}. "
        "Expected one of {'gene', 'gt_gene_name'}."
    )


def _rank_slide_hvg_genes(
    st_log_qc: np.ndarray,
    st_var_qc: np.ndarray,
    genes_qc: Sequence[str],
    flavor: str,
    fallback_to_variance: bool,
    logger,
    slide_id: str,
) -> np.ndarray:
    G = int(st_log_qc.shape[1])
    if G == 0:
        return np.empty((0,), dtype=np.int64)
    tmp = ad.AnnData(X=np.asarray(st_log_qc, dtype=np.float32))
    tmp.var_names = pd.Index(list(genes_qc)).astype(str)
    sc.pp.highly_variable_genes(
        tmp,
        flavor=str(flavor),
        n_top_genes=None,
        inplace=True,
    )
    if "dispersions_norm" not in tmp.var.columns:
        raise RuntimeError("Scanpy HVG output missing dispersions_norm")
    score = tmp.var["dispersions_norm"].to_numpy(dtype=np.float64)
    if not np.any(np.isfinite(score)):
        raise RuntimeError("Scanpy HVG produced no finite dispersions_norm scores")
    score = np.where(np.isfinite(score), score, -np.inf)
    return np.lexsort((np.arange(G, dtype=np.int64), -score)).astype(np.int64)


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


def _build_gene_qc_artifact_for_slide(
    slide_id: str,
    gt_h5ad: Path,
    ref_h5ad: Path,
    gt_cells: pd.DataFrame,
    ref_filtered: FilteredReference,
    gene_universe: Mapping[str, Any],
    eval_data_root: Path,
    cfg: Mapping[str, Any],
    logger,
    ref_h5ad_info: Mapping[str, Any],
    gene_stats_cache_dir: Path,
    gene_stats_cache_files_used: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    genes = list(map(str, gene_universe["genes"]))
    gt_gene_names = list(map(str, gene_universe.get("gt_gene_names", genes)))
    gt_var_idx = np.asarray(gene_universe["gt_var_indices"], dtype=np.int64)
    ref_var_idx = np.asarray(gene_universe["reference_var_indices"], dtype=np.int64)

    if len(gt_gene_names) != len(genes):
        raise RuntimeError(
            f"Gene-name metadata is not aligned for slide={slide_id}: "
            f"{len(gt_gene_names)} names vs {len(genes)} genes"
        )

    if "_h5ad_row_index" not in gt_cells.columns:
        raise KeyError("gt_cells must contain _h5ad_row_index for gene QC")
    st_rows = gt_cells["_h5ad_row_index"].to_numpy(dtype=np.int64, copy=False)
    ref_rows = np.asarray(ref_filtered.filtered_rows, dtype=np.int64)

    expr_cfg = cfg["expression"]
    gp_cfg = cfg.get("gene_panels", {})
    support_cfg = gp_cfg.get("support_filters", {})
    hvg_cfg = gp_cfg.get("slide_hvg", {})

    target_sum = float(expr_cfg["normalize_target_sum"])
    variance_eps = support_cfg.get("st_variance_eps", None)
    if variance_eps is None:
        variance_eps = float(expr_cfg["variance_eps"])
    else:
        variance_eps = float(variance_eps)

    st_stats, st_log = _expression_stats_for_rows_on_raw_shared(
        h5ad_path=gt_h5ad,
        selected_rows=st_rows,
        var_indices=gt_var_idx,
        normalize_target_sum=target_sum,
        row_block_size=int(cfg.get("runtime", {}).get("reference_expression_cache_row_block_size", 8192)),
        target_block_mb=float(cfg.get("runtime", {}).get("reference_expression_cache_target_block_mb", 128.0)),
        keep_log_matrix=True,
    )
    # Reference side: cached (see _reference_gene_stats_cached); GT side: not cached (log matrix kept for the HVG ranking)
    ref_stats = _reference_gene_stats_cached(
        reference_h5ad_path=ref_h5ad,
        reference_h5ad_info=ref_h5ad_info,
        selected_rows=ref_rows,
        var_indices=ref_var_idx,
        normalize_target_sum=target_sum,
        row_block_size=int(cfg.get("runtime", {}).get("reference_expression_cache_row_block_size", 8192)),
        target_block_mb=float(cfg.get("runtime", {}).get("reference_expression_cache_target_block_mb", 128.0)),
        cache_dir=gene_stats_cache_dir,
        used_cache_files=gene_stats_cache_files_used,
    )
    assert st_log is not None

    n_st = int(st_rows.size)
    n_ref = int(ref_rows.size)
    st_min = max(
        int(support_cfg.get("st_min_nonzero_cells_abs", 20)),
        int(np.ceil(float(support_cfg.get("st_min_nonzero_rate", 0.005)) * float(n_st))),
    )
    ref_min = max(
        int(support_cfg.get("ref_min_nonzero_cells_abs", 50)),
        int(np.ceil(float(support_cfg.get("ref_min_nonzero_rate", 0.001)) * float(n_ref))),
    )

    passes_st_support = st_stats["nonzero_cells"] >= st_min
    passes_ref_support = ref_stats["nonzero_cells"] >= ref_min
    if bool(support_cfg.get("use_st_variance_filter", True)):
        passes_st_variance = st_stats["var_log1p"] > variance_eps
    else:
        passes_st_variance = np.ones(len(genes), dtype=bool)

    passes_qc = passes_st_support & passes_ref_support & passes_st_variance

    g_base_match_rules = _g_base_match_rules_from_cfg(cfg)
    g_base_set = _load_g_base_gene_set_for_a2(
        eval_data_root=eval_data_root,
        cfg=cfg,
        rules=g_base_match_rules,
    )

    g_base_values_raw, g_base_match_field = _g_base_membership_values(
        genes=genes,
        gt_gene_names=gt_gene_names,
        cfg=cfg,
    )
    g_base_match_keys = canonicalize_gene_ids(g_base_values_raw, rules=g_base_match_rules)

    # Warn if duplicated G_base match keys (possible with gene names) select several gene-ID rows
    match_index = pd.Index(g_base_match_keys)
    duplicated_match_keys = sorted(match_index[match_index.duplicated()].unique().tolist())
    duplicated_selected_keys = sorted(set(duplicated_match_keys) & set(g_base_set))
    if duplicated_selected_keys and logger is not None:
        logger.warning(
            "Slide=%s has duplicated G_base match keys that are present in G_base. "
            "match_field=%s duplicated_selected_examples=%s. "
            "Expression alignment still uses gene IDs, but G_base membership may include "
            "multiple gene-ID rows for the same match key.",
            slide_id,
            g_base_match_field,
            duplicated_selected_keys[:20],
        )

    in_g_base = np.asarray([k in g_base_set for k in g_base_match_keys], dtype=bool)
    in_g_s_qc = passes_qc.astype(bool)
    in_g_base_qc = in_g_base & in_g_s_qc

    hvg_rank = np.full(len(genes), np.nan, dtype=np.float64)
    qc_idx = np.flatnonzero(in_g_s_qc)
    if qc_idx.size > 0:
        order_rel = _rank_slide_hvg_genes(
            st_log_qc=st_log[:, qc_idx],
            st_var_qc=st_stats["var_log1p"][qc_idx],
            genes_qc=np.asarray(genes, dtype=object)[qc_idx].tolist(),
            flavor=str(hvg_cfg.get("hvg_flavor", "seurat")),
            fallback_to_variance=bool(hvg_cfg.get("fallback_to_variance", True)),
            logger=logger,
            slide_id=slide_id,
        )
        ranked_abs = qc_idx[order_rel]
        hvg_rank[ranked_abs] = np.arange(1, ranked_abs.size + 1, dtype=np.float64)

    qdf = pd.DataFrame(
        {
            "gene": genes,
            "gt_gene_name": gt_gene_names,
            "gt_var_index": gt_var_idx.astype(np.int64),
            "reference_var_index": ref_var_idx.astype(np.int64),
            "st_nonzero_cells": st_stats["nonzero_cells"].astype(np.int64),
            "st_nonzero_rate": st_stats["nonzero_rate"].astype(np.float64),
            "st_mean_log1p": st_stats["mean_log1p"].astype(np.float32),
            "st_var_log1p": st_stats["var_log1p"].astype(np.float32),
            "ref_nonzero_cells": ref_stats["nonzero_cells"].astype(np.int64),
            "ref_nonzero_rate": ref_stats["nonzero_rate"].astype(np.float64),
            "ref_mean_log1p": ref_stats["mean_log1p"].astype(np.float32),
            "ref_var_log1p": ref_stats["var_log1p"].astype(np.float32),
            "passes_st_support": passes_st_support.astype(np.int8),
            "passes_reference_support": passes_ref_support.astype(np.int8),
            "passes_st_variance": passes_st_variance.astype(np.int8),
            "passes_qc": passes_qc.astype(np.int8),
            "g_base_match_field": np.asarray([g_base_match_field] * len(genes), dtype=object),
            "g_base_match_key": np.asarray(g_base_match_keys, dtype=object),
            "in_G_base": in_g_base.astype(np.int8),
            "in_G_s_qc": in_g_s_qc.astype(np.int8),
            "in_G_base_qc": in_g_base_qc.astype(np.int8),
            "hvg_rank": hvg_rank,
        }
    )

    hvg_summary: Dict[str, Any] = {}
    for N in [int(x) for x in hvg_cfg.get("Ns", [50, 100])]:
        col = f"in_G_eval_slide_hvg{N}"
        mask = np.isfinite(hvg_rank) & (hvg_rank <= float(N))
        qdf[col] = mask.astype(np.int8)
        n_eff = int(mask.sum())
        hvg_summary[f"n_genes_G_eval_slide_hvg{N}"] = n_eff
        hvg_summary[f"status_G_eval_slide_hvg{N}"] = _hvg_panel_status(n_eff, N)

    n_raw = int(len(genes))
    n_qc = int(in_g_s_qc.sum())
    n_base_raw = int(in_g_base.sum())
    n_base_qc = int(in_g_base_qc.sum())
    min_official = int(expr_cfg["min_aligned_genes_official"])

    if len(g_base_set) > 0 and n_base_raw == 0 and logger is not None:
        logger.warning(
            "Slide=%s has zero raw overlap between G_s_raw and G_base before QC. "
            "G_base_match_field=%s. A2 gene-ID examples=%s; A2 GT gene-name examples=%s; "
            "G_base examples=%s. If G_base was built from gene names, use "
            "expression.g_base_match_field: gt_gene_name.",
            slide_id,
            g_base_match_field,
            list(map(str, genes[:10])),
            list(map(str, gt_gene_names[:10])),
            sorted(list(map(str, g_base_set)))[:10],
        )

    summary = {
        "n_genes_G_s_raw": n_raw,
        "n_genes_G_s_qc": n_qc,
        "status_G_s_qc": _fixed_panel_status(n_qc, min_official),
        "n_genes_G_base_raw_intersection": n_base_raw,
        "n_genes_G_base_qc": n_base_qc,
        "status_G_base_qc": _fixed_panel_status(n_base_qc, min_official),
        "g_base_match_field": g_base_match_field,
        "g_base_unique_match_keys_in_G_s_raw": int(pd.Index(g_base_match_keys).nunique()),
        "g_base_duplicated_match_keys_in_G_s_raw": int(len(duplicated_match_keys)),
        "g_base_duplicated_selected_match_keys_in_G_s_raw": int(len(duplicated_selected_keys)),
        "st_support_min_nonzero_cells": int(st_min),
        "reference_support_min_nonzero_cells": int(ref_min),
        "st_variance_eps_for_qc": float(variance_eps),
    }
    summary.update(hvg_summary)
    return qdf, summary



def _read_gt_labels_for_coverage(
    gt_h5ad_path: Path,
    gt_label_column: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Raw mapped GT labels and cell_id_int of the is_cell rows of a slide (obs-only read),
    for slide admission and the coverage preflight.
    """
    a = ad.read_h5ad(gt_h5ad_path, backed="r")
    try:
        for col in ("is_cell", gt_label_column):
            if col not in a.obs.columns:
                raise KeyError(f"Missing obs column '{col}' in {gt_h5ad_path}")
        cols = ["is_cell", gt_label_column]
        if "cell_id_int" in a.obs.columns:
            cols.append("cell_id_int")
        obs = a.obs.loc[:, cols].copy()
    finally:
        if getattr(a, "file", None) is not None:
            a.file.close()

    # Mask the is_cell rows before deriving cell_id_int (as in _read_gt_cells_and_gene_ids)
    mask = coerce_bool_series(obs["is_cell"], context=f"{gt_h5ad_path}:obs['is_cell']").to_numpy()
    obs = obs.loc[mask]
    return (
        obs[gt_label_column].astype(str).to_numpy(dtype=object),
        _derive_cell_id_int(obs),
    )


def _select_official_slides(
    *,
    slide_meta: pd.DataFrame,
    slide_col: str,
    subset_col: str,
    tissue_col: str,
    st_dir: Path,
    gt_label_col: str,
    gt_mapped_unscorable_labels: set,
    label_map: Optional[Mapping[str, str]],
    subset_for_tuning: str,
    min_scorable_gt_cells: int,
    min_classes_per_slide: int,
    min_class_support_for_admission: int,
    eval_data_root: Path,
    modelgroup: str,
    logger,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]], Dict[str, np.ndarray]]:
    """
    Apply the slide-admission floors (data.min_scorable_gt_cells_per_slide, data.min_classes_per_slide)
    on the modelgroup-token-eligible cells, after the label_granularity collapse.
    Returns the admitted slide_meta, one record per excluded slide, and the collapsed GT labels per admitted slide.
    """
    gt_labels_cache: Dict[str, np.ndarray] = {}
    rows: List[Dict[str, Any]] = []

    for _, row in slide_meta.iterrows():
        slide = str(row[slide_col])
        gt_h5ad = st_dir / f"{slide}.h5ad"
        if not gt_h5ad.exists():
            raise FileNotFoundError(gt_h5ad)
        raw_labels, cell_id_int = _read_gt_labels_for_coverage(gt_h5ad, gt_label_col)
        n_is_cell = int(raw_labels.size)

        missing_cells = _load_missing_cells_for_modelgroup(
            eval_data_root=eval_data_root,
            modelgroup=modelgroup,
            slide_id=slide,
            require_exists=True,
        )
        eligible = ~pd.Series(cell_id_int).isin(missing_cells).to_numpy()

        labels = apply_label_map(raw_labels[eligible], label_map)
        gt_labels_cache[slide] = np.asarray(labels, dtype=object).reshape(-1)
        scorable_labels = pd.Series(gt_labels_cache[slide], dtype=object).astype(str)
        scorable_labels = scorable_labels[~scorable_labels.isin(gt_mapped_unscorable_labels)]
        class_counts = scorable_labels.value_counts()
        rows.append(
            {
                "slide_id": slide,
                "subset": str(row[subset_col]),
                "tissue_HistoST": str(row[tissue_col]),
                # is_cell rows before the token-eligibility drop (n_gt_cells is the count used by the floors)
                "n_is_cell_rows": n_is_cell,
                "n_gt_cells": int(gt_labels_cache[slide].size),
                "n_cells_missing_after_modelgroup_tokenization": int(n_is_cell - int(eligible.sum())),
                "n_scorable_gt_cells": int(scorable_labels.size),
                # Scorable classes with at least min_class_support_for_admission cells (compared with min_classes_per_slide)
                "n_scorable_gt_classes": int((class_counts >= max(1, int(min_class_support_for_admission))).sum()),
                # Number of scorable classes without the support floor
                "n_scorable_gt_classes_unfloored": int(class_counts.size),
            }
        )

    kept_ids, excluded = _slide_admission_decision(
        counts=pd.DataFrame(rows),
        floor=int(min_scorable_gt_cells),
        min_classes=int(min_classes_per_slide),
        min_class_support=int(min_class_support_for_admission),
        subset_for_tuning=str(subset_for_tuning),
        logger=logger,
    )

    out = slide_meta.loc[slide_meta[slide_col].astype(str).isin(kept_ids)].reset_index(drop=True)
    for slide in set(gt_labels_cache) - kept_ids:
        gt_labels_cache.pop(slide, None)
    return out, excluded, gt_labels_cache



def _slide_admission_decision(
    *,
    counts: pd.DataFrame,
    floor: int,
    min_classes: int = 0,
    min_class_support: int = 0,
    subset_for_tuning: str,
    logger,
) -> Tuple[set, List[Dict[str, Any]]]:
    """
    Admission decision of _select_official_slides (no I/O): a slide is admitted iff
    n_scorable_gt_cells >= floor and n_scorable_gt_classes >= min_classes (a floor <= 0 disables it).
    Raises if no slide, or no tuning slide, is admitted.
    """
    counts = counts.sort_values(["n_scorable_gt_cells", "slide_id"]).reset_index(drop=True)
    floor = int(floor)
    min_classes = int(min_classes)
    all_ids = set(counts["slide_id"].astype(str))
    if counts.empty:
        raise ValueError("Slide admission received an empty cohort.")
    # No n_scorable_gt_classes column: the class floor passes every slide
    if "n_scorable_gt_classes" not in counts.columns:
        counts = counts.assign(n_scorable_gt_classes=np.iinfo(np.int32).max)

    if floor <= 0 and min_classes <= 0:
        logger.info(
            "Slide admission is disabled (data.min_scorable_gt_cells_per_slide=%d, "
            "data.min_classes_per_slide=%d); all %d slide(s) admitted. Smallest: '%s' with %s "
            "scorable GT cell(s).",
            floor,
            min_classes,
            int(counts.shape[0]),
            str(counts.iloc[0]["slide_id"]),
            f"{int(counts.iloc[0]['n_scorable_gt_cells']):,}",
        )
        return all_ids, []

    fails_cells = (counts["n_scorable_gt_cells"] < floor) if floor > 0 else pd.Series(
        False, index=counts.index
    )
    fails_classes = (counts["n_scorable_gt_classes"] < min_classes) if min_classes > 0 else pd.Series(
        False, index=counts.index
    )
    dropped_mask = fails_cells | fails_classes
    dropped = counts.loc[dropped_mask]
    kept = counts.loc[~dropped_mask]

    excluded: List[Dict[str, Any]] = []
    for _, r in dropped.iterrows():
        reasons: List[str] = []
        if floor > 0 and int(r["n_scorable_gt_cells"]) < floor:
            reasons.append(
                f"{int(r['n_scorable_gt_cells']):,} classification-scorable GT cells < "
                f"min_scorable_gt_cells_per_slide={floor:,}"
            )
        if min_classes > 0 and int(r["n_scorable_gt_classes"]) < min_classes:
            # Also report the class count without the support floor
            unfloored = r.get("n_scorable_gt_classes_unfloored")
            unfloored_note = (
                ""
                if unfloored is None or int(unfloored) == int(r["n_scorable_gt_classes"])
                else f" ({int(unfloored)} class(es) present, but only {int(r['n_scorable_gt_classes'])} "
                     f"carry min_class_support_for_admission cells)"
            )
            reasons.append(
                f"{int(r['n_scorable_gt_classes'])} scorable GT class(es) < "
                f"min_classes_per_slide={min_classes}{unfloored_note}"
            )
        record = {
            **{k: (int(v) if isinstance(v, (int, np.integer)) else str(v)) for k, v in r.items()},
            "min_scorable_gt_cells_per_slide": floor,
            "min_classes_per_slide": min_classes,
            "min_class_support_for_admission": int(min_class_support),
            "reason": "; ".join(reasons) + ".",
        }
        excluded.append(record)
        logger.warning(
            "Slide excluded from the official A-2 cohort: '%s' (subset=%s, tissue=%s) -- %s.",
            record["slide_id"],
            record["subset"],
            record["tissue_HistoST"],
            record["reason"],
        )

    if kept.empty:
        raise ValueError(
            f"Slide admission (data.min_scorable_gt_cells_per_slide={floor}, "
            f"data.min_classes_per_slide={min_classes}) excluded every slide of the official "
            f"cohort (largest slide has {int(counts['n_scorable_gt_cells'].max()):,} scorable GT "
            f"cells and {int(counts['n_scorable_gt_classes'].max())} classes). Lower the floors."
        )

    # Log the margin of the smallest retained slide
    smallest = kept.iloc[0]
    logger.info(
        "Slide admission (data.min_scorable_gt_cells_per_slide=%s, data.min_classes_per_slide=%d "
        "counting classes with >= %d cell(s)) on population '%s': %d/%d slide(s) admitted, %d "
        "excluded. Smallest retained slide: '%s' with %s scorable GT cell(s) (%.2fx the floor) and "
        "%d class(es). Fewest classes among retained: %d. Largest excluded: %s.",
        f"{floor:,}",
        min_classes,
        max(1, int(min_class_support)),
        "is_cell_and_modelgroup_token_eligible",
        int(kept.shape[0]),
        int(counts.shape[0]),
        int(dropped.shape[0]),
        str(smallest["slide_id"]),
        f"{int(smallest['n_scorable_gt_cells']):,}",
        (float(smallest["n_scorable_gt_cells"]) / float(floor)) if floor > 0 else float("inf"),
        int(smallest["n_scorable_gt_classes"]),
        int(kept["n_scorable_gt_classes"].min()),
        "none"
        if dropped.empty
        else f"'{dropped.iloc[-1]['slide_id']}' with {int(dropped.iloc[-1]['n_scorable_gt_cells']):,}",
    )

    # Cells removed by the token-eligibility drop, cohort-wide
    if "n_cells_missing_after_modelgroup_tokenization" in counts.columns:
        n_dropped_cells = int(counts["n_cells_missing_after_modelgroup_tokenization"].sum())
        n_is_cell = int(counts["n_is_cell_rows"].sum()) if "n_is_cell_rows" in counts.columns else 0
        worst = counts.loc[counts["n_cells_missing_after_modelgroup_tokenization"].idxmax()]
        logger.info(
            "Modelgroup token-eligibility drop (applied before the floors): %s of %s `is_cell` "
            "row(s) removed cohort-wide (%.2f%%); worst slide '%s' lost %s.",
            f"{n_dropped_cells:,}",
            f"{n_is_cell:,}",
            (100.0 * n_dropped_cells / n_is_cell) if n_is_cell > 0 else float("nan"),
            str(worst["slide_id"]),
            f"{int(worst['n_cells_missing_after_modelgroup_tokenization']):,}",
        )

    # At least one tuning slide is required to select (k, tau)
    if not (kept["subset"].astype(str) == str(subset_for_tuning)).any():
        raise ValueError(
            f"Slide admission (data.min_scorable_gt_cells_per_slide={floor}, "
            f"data.min_classes_per_slide={min_classes}) left the tuning subset "
            f"'{subset_for_tuning}' with no slides (excluded: "
            f"{[e['slide_id'] for e in excluded]}). (k, tau) tuning would have nothing to run on."
        )

    return set(kept["slide_id"].astype(str)), excluded



def _coverage_preflight(
    *,
    slide_meta: pd.DataFrame,
    slide_col: str,
    tissue_col: str,
    st_dir: Path,
    mapping: Mapping[str, str],
    ref_mapping_path: Path,
    ref_label_col: str,
    ref_fine_col: str,
    gt_label_col: str,
    gt_mapped_unscorable_labels: set,
    gt_labels_cache: Mapping[str, np.ndarray],
    label_map: Optional[Mapping[str, str]],
    label_granularity: str,
    require_nonempty: bool,
    cfg: Mapping[str, Any],
    reference_info_cache: Dict[str, Dict[str, Any]],
    reference_filter_cache: Dict[tuple, FilteredReference],
    reference_scorable_drop_cache: Dict[tuple, Dict[str, int]],
    reference_filter_cache_dir: Path,
    used_cache_files: Dict[str, Dict[str, Any]],
    logger,
) -> Dict[str, SlideCoverageReport]:
    """
    Check the reference-coverage contract for every slide before the main loop.
    Fills the reference caches reused by the main loop, and used_cache_files.
    """
    reports: Dict[str, SlideCoverageReport] = {}
    ref_counts_cache: Dict[tuple, Dict[str, int]] = {}

    for _, row in slide_meta.iterrows():
        slide = str(row[slide_col])
        tissue = str(row[tissue_col])
        gt_h5ad = st_dir / f"{slide}.h5ad"
        if not gt_h5ad.exists():
            raise FileNotFoundError(gt_h5ad)

        ref_spec = resolve_reference_for_tissue(tissue, mapping, ref_mapping_path)
        ref_path_key = str(ref_spec.path.resolve())
        if ref_path_key not in reference_info_cache:
            reference_info_cache[ref_path_key] = file_info(ref_spec.path, compute_sha256=False)
        ref_info = reference_info_cache[ref_path_key]

        ref_cache_key = _reference_cache_key(
            ref_path_key=ref_path_key,
            mapped_label_column=ref_label_col,
            fine_label_column=ref_fine_col,
            require_nonempty_expression=require_nonempty,
            label_granularity=label_granularity,
            unscorable_labels=sorted(gt_mapped_unscorable_labels),
        )
        # Coverage is checked on the retrieval pool (after dropping the unscorable labels)
        ref_filtered, _ = _filtered_reference_for_eval(
            reference_h5ad_path=ref_spec.path,
            reference_h5ad_info=ref_info,
            cache_key=ref_cache_key,
            mapped_label_column=ref_label_col,
            fine_label_column=ref_fine_col,
            require_nonempty_expression=require_nonempty,
            label_map=label_map,
            unscorable_labels=sorted(gt_mapped_unscorable_labels),
            cache_dir=reference_filter_cache_dir,
            reference_filter_cache=reference_filter_cache,
            reference_scorable_drop_cache=reference_scorable_drop_cache,
            logger=logger,
            used_cache_files=used_cache_files,
        )

        if ref_cache_key not in ref_counts_cache:
            ref_counts_cache[ref_cache_key] = collapse_label_counts(
                build_label_counts(ref_filtered.obs, ref_filtered.filtered_rows, ref_label_col),
                label_map,
            )

        # Collapsed once by `_select_official_slides`; only admitted slides are in the cache.
        gt_labels = gt_labels_cache[slide]
        scorable = ~pd.Series(gt_labels, dtype=object).astype(str).isin(
            gt_mapped_unscorable_labels
        ).to_numpy()

        reports[slide] = build_slide_coverage_report(
            slide_id=slide,
            tissue=tissue,
            reference_stem=ref_spec.stem,
            label_granularity=label_granularity,
            stage=STAGE_FREEZE,
            gt_labels=gt_labels,
            gt_scorable_mask=scorable,
            ref_label_counts=ref_counts_cache[ref_cache_key],
            cfg=cfg,
        )

    assert_coverage_contract(
        list(reports.values()), cfg, stage=STAGE_FREEZE, log=logger
    )
    return reports


def prepare_shared_artifacts(eval_root: Path, logger) -> Dict[str, Any]:
    from src_A2.protocol import load_protocol

    cfg = load_protocol(eval_root)
    paths = get_protocol_paths(eval_root)

    metadata = pd.read_csv(paths.metadata_snapshot_csv)
    mapping = load_reference_mapping(paths.ref_mapping_snapshot_json)
    rules = _canonical_rules_from_cfg(cfg)

    source_col = cfg["data"]["source_column"]
    subset_col = cfg["data"]["subset_column"]
    slide_col = cfg["data"]["slide_id_column"]
    tissue_col = cfg["data"]["tissue_column"]
    gt_label_col = cfg["data"]["gt_label_column"]
    gt_mapped_unscorable_labels = {str(x) for x in cfg["data"].get("gt_mapped_unscorable_labels", [])}
    reference_label_col_tmpl = cfg["references"]["mapped_label_column_template"]
    ct_cat_id = cfg["identity"]["ct_cat_id"]
    ref_label_col = reference_label_col_tmpl.format(ct_cat_id=ct_cat_id)
    ref_fine_col = cfg["references"]["fine_label_column"]
    require_nonempty = bool(cfg["references"]["filter"]["require_nonempty_expression"])

    # Label map: None for fine, fine -> coarse map for coarse. Applied to the mapped labels of GT and
    # reference as they are read (not to ref_fine_col).
    label_granularity = normalize_granularity(cfg["identity"].get("label_granularity", "fine"))
    _coarse_version_raw = cfg["identity"].get("coarse_mapping_version", None)
    coarse_mapping_version = str(_coarse_version_raw).strip() if _coarse_version_raw else None
    label_map = resolve_label_map(ct_cat_id, label_granularity, coarse_mapping_version)
    logger.info(
        "A-2 label_granularity=%s (ct_cat_id=%s): mapped labels are %s; "
        "references.fine_label_column='%s' is never collapsed.",
        label_granularity,
        ct_cat_id,
        "used as-is"
        if label_map is None
        else f"collapsed via {coarse_mapping_filename(coarse_mapping_version)}",
        ref_fine_col,
    )

    official_subsets = {str(x) for x in cfg["data"]["official_subsets"]}
    sthelar = metadata.loc[
        (metadata[source_col].astype(str) == cfg["data"]["source"])
        & (metadata[subset_col].astype(str).isin(official_subsets))
    ].copy()
    if sthelar.empty:
        raise ValueError("No STHELAR slides found in metadata snapshot")
    seen_subsets = set(sthelar[subset_col].astype(str).unique().tolist())
    missing_subsets = official_subsets - seen_subsets
    if missing_subsets:
        logger.warning("Configured official subsets missing from metadata snapshot: %s", sorted(missing_subsets))

    manifests_dir = ensure_dir(paths.shared_dir / "manifests")
    audits_dir = ensure_dir(paths.shared_dir / "reference_audits")
    gene_dir = ensure_dir(paths.shared_dir / "gene_universes")
    gene_qc_dir = ensure_dir(paths.shared_dir / "gene_qc")
    sampling_dir = ensure_dir(paths.shared_dir / "umap_sampling")
    row_filter_dir = ensure_dir(audits_dir / "row_filters")

    training_dataset_id = str(cfg['identity']['training_dataset_id'])
    from src.utils.constants import get_full_path

    eval_data_root = get_full_path('eval', 'eval_datasets', 'st_eval_data', f'eval_st_{training_dataset_id}')
    st_dir = eval_data_root / 'st'
    modelgroup = str(cfg["identity"]["modelgroup"])
    missing_dir = eval_data_root / f"cell_token_membership_missing_cells_{modelgroup}"
    if not missing_dir.exists():
        raise FileNotFoundError(
            f"Structural-eligibility directory is missing for modelgroup='{modelgroup}': {missing_dir}"
        )

    # Task-level cache directory (see _shared_reference_filter_cache_dir)
    reference_filter_cache_dir = _shared_reference_filter_cache_dir()
    logger.info(
        "Reference row-filter npz cache (shared across eval_ids, content-addressed): %s",
        reference_filter_cache_dir,
    )
    # Task-level cache of the reference-side gene stats (see _reference_gene_stats_cached)
    reference_gene_stats_cache_dir = _shared_reference_gene_stats_cache_dir()
    logger.info(
        "Reference gene-stats npz cache (shared across eval_ids, content-addressed): %s",
        reference_gene_stats_cache_dir,
    )
    reference_filter_cache: Dict[tuple, FilteredReference] = {}
    reference_scorable_drop_cache: Dict[tuple, Dict[str, int]] = {}
    reference_info_cache: Dict[str, Dict[str, Any]] = {}
    # Cache files used by this run
    reference_filter_cache_files_used: Dict[str, Dict[str, Any]] = {}
    reference_gene_stats_cache_files_used: Dict[str, Dict[str, Any]] = {}
    reference_gene_ids_cache: Dict[tuple, List[str]] = {}
    reference_static_cache: Dict[tuple, Dict[str, Any]] = {}

    slide_rows: List[Dict[str, Any]] = []
    cell_backbone_rows: List[pd.DataFrame] = []
    label_vocab = set()
    reference_audits: List[Dict[str, Any]] = []

    slide_meta = sthelar[[slide_col, subset_col, tissue_col]].drop_duplicates().copy()
    consistency = slide_meta.groupby(slide_col).agg({subset_col: "nunique", tissue_col: "nunique"})
    bad = consistency[(consistency[subset_col] > 1) | (consistency[tissue_col] > 1)]
    if not bad.empty:
        raise ValueError(
            "A slide_id maps to multiple subset/tissue values in the metadata snapshot. "
            f"Examples: {bad.head(10).to_dict(orient='index')}"
        )
    slide_meta = (
        slide_meta
        .drop_duplicates(subset=[slide_col])
        .sort_values([subset_col, slide_col])
        .reset_index(drop=True)
    )

    min_ref_cells = int(cfg["references"]["filter"]["minimum_reference_cells"])
    min_class_support = int(cfg["references"]["filter"]["minimum_class_support"])
    min_aligned_genes = int(cfg["expression"]["min_aligned_genes_official"])

    # Slide admission (before the coverage contract and any gene work); also collapses the GT labels
    slide_meta, excluded_slides, gt_labels_cache = _select_official_slides(
        slide_meta=slide_meta,
        slide_col=slide_col,
        subset_col=subset_col,
        tissue_col=tissue_col,
        st_dir=st_dir,
        gt_label_col=gt_label_col,
        gt_mapped_unscorable_labels=gt_mapped_unscorable_labels,
        label_map=label_map,
        subset_for_tuning=str(cfg["data"]["subset_for_tuning"]),
        min_scorable_gt_cells=int(cfg["data"].get("min_scorable_gt_cells_per_slide", 0)),
        min_classes_per_slide=int(cfg["data"].get("min_classes_per_slide", 0)),
        min_class_support_for_admission=int(cfg["data"].get("min_class_support_for_admission", 0)),
        eval_data_root=eval_data_root,
        modelgroup=modelgroup,
        logger=logger,
    )

    # Coverage contract: every scorable GT class of every admitted slide must be covered by its reference
    coverage_reports = _coverage_preflight(
        slide_meta=slide_meta,
        slide_col=slide_col,
        tissue_col=tissue_col,
        st_dir=st_dir,
        mapping=mapping,
        ref_mapping_path=paths.ref_mapping_snapshot_json,
        ref_label_col=ref_label_col,
        ref_fine_col=ref_fine_col,
        gt_label_col=gt_label_col,
        gt_mapped_unscorable_labels=gt_mapped_unscorable_labels,
        gt_labels_cache=gt_labels_cache,
        label_map=label_map,
        label_granularity=label_granularity,
        require_nonempty=require_nonempty,
        cfg=cfg,
        reference_info_cache=reference_info_cache,
        reference_filter_cache=reference_filter_cache,
        reference_scorable_drop_cache=reference_scorable_drop_cache,
        reference_filter_cache_dir=reference_filter_cache_dir,
        used_cache_files=reference_filter_cache_files_used,
        logger=logger,
    )

    for _, row in slide_meta.iterrows():
        slide = str(row[slide_col])
        subset = str(row[subset_col])
        tissue = str(row[tissue_col])
        gt_h5ad = st_dir / f"{slide}.h5ad"
        if not gt_h5ad.exists():
            raise FileNotFoundError(gt_h5ad)

        ref_spec = resolve_reference_for_tissue(tissue, mapping, paths.ref_mapping_snapshot_json)
        ref_path_key = str(ref_spec.path.resolve())
        if ref_path_key not in reference_info_cache:
            reference_info_cache[ref_path_key] = file_info(ref_spec.path, compute_sha256=False)
        ref_info = reference_info_cache[ref_path_key]

        ref_cache_key = _reference_cache_key(
            ref_path_key=ref_path_key,
            mapped_label_column=ref_label_col,
            fine_label_column=ref_fine_col,
            require_nonempty_expression=require_nonempty,
            label_granularity=label_granularity,
            unscorable_labels=sorted(gt_mapped_unscorable_labels),
        )
        # Cached by the coverage preflight
        ref_filtered, ref_scorable_drops = _filtered_reference_for_eval(
            reference_h5ad_path=ref_spec.path,
            reference_h5ad_info=ref_info,
            cache_key=ref_cache_key,
            mapped_label_column=ref_label_col,
            fine_label_column=ref_fine_col,
            require_nonempty_expression=require_nonempty,
            label_map=label_map,
            unscorable_labels=sorted(gt_mapped_unscorable_labels),
            cache_dir=reference_filter_cache_dir,
            reference_filter_cache=reference_filter_cache,
            reference_scorable_drop_cache=reference_scorable_drop_cache,
            logger=logger,
            used_cache_files=reference_filter_cache_files_used,
        )

        ref_gene_cache_key = (
            ref_path_key,
            str(cfg["genes"]["reference_gene_id_column"]),
            rules.strip_whitespace,
            rules.uppercase,
            rules.strip_ensembl_version,
        )
        if ref_gene_cache_key not in reference_gene_ids_cache:
            reference_gene_ids_cache[ref_gene_cache_key] = get_canonical_gene_ids_from_var(
                ref_spec.path,
                cfg["genes"]["reference_gene_id_column"],
                rules,
            )

        # Checked on the narrowed pool (after dropping the unscorable labels)
        if ref_filtered.fine_rows.size != ref_filtered.filtered_rows.size:
            raise ValueError(
                f"Reference {ref_spec.path} contains officially filtered rows without "
                f"'{ref_fine_col}'. Fine-agreement requires original_cell_type on every "
                "officially filtered reference cell."
            )
        # Retrieval pool of the slide (read back by _3)
        np.savez_compressed(
            row_filter_dir / f"{slide}.npz",
            filtered_rows=ref_filtered.filtered_rows,
            fine_rows=ref_filtered.fine_rows,
        )

        gt_cells, gt_gene_ids, gt_gene_names = _read_gt_cells_and_gene_ids(
            gt_h5ad_path=gt_h5ad,
            slide_id=slide,
            gt_label_column=gt_label_col,
            gt_gene_id_column=cfg["genes"]["gt_gene_id_column"],
            rules=rules,
        )
        if gt_cells.empty:
            raise ValueError(f"GT eval cells are empty for slide {slide}")

        gene_universe = _build_gene_universe_from_gene_ids(
            gt_h5ad_path=gt_h5ad,
            reference_h5ad_path=ref_spec.path,
            gt_ids=gt_gene_ids,
            ref_ids=reference_gene_ids_cache[ref_gene_cache_key],
            gt_gene_names=gt_gene_names,
        )
        gu_df = pd.DataFrame(
            {
                "gene": gene_universe["genes"],
                "gt_gene_name": gene_universe["gt_gene_names"],
                "gt_var_index": gene_universe["gt_var_indices"],
                "reference_var_index": gene_universe["reference_var_indices"],
            }
        )
        gu_path = gene_dir / f"{slide}.parquet"
        write_dataframe(gu_df, gu_path)

        gene_qc_df, gene_panel_summary = _build_gene_qc_artifact_for_slide(
            slide_id=slide,
            gt_h5ad=gt_h5ad,
            ref_h5ad=ref_spec.path,
            gt_cells=gt_cells,
            ref_filtered=ref_filtered,
            gene_universe=gene_universe,
            eval_data_root=eval_data_root,
            cfg=cfg,
            logger=logger,
            ref_h5ad_info=ref_info,
            gene_stats_cache_dir=reference_gene_stats_cache_dir,
            gene_stats_cache_files_used=reference_gene_stats_cache_files_used,
        )
        gene_qc_path = gene_qc_dir / f"{slide}.parquet"
        write_dataframe(gene_qc_df, gene_qc_path)

        if ref_cache_key not in reference_static_cache:
            # Reference-side label collapse (derived views only: not the cached obs, not ref_fine_counts)
            reference_static_cache[ref_cache_key] = {
                "ref_label_counts": collapse_label_counts(
                    build_label_counts(ref_filtered.obs, ref_filtered.filtered_rows, ref_label_col),
                    label_map,
                ),
                "ref_fine_counts": build_label_counts(ref_filtered.obs, ref_filtered.fine_rows, ref_fine_col),
                "filtered_labels": apply_label_map(
                    ref_filtered.obs.iloc[ref_filtered.filtered_rows][ref_label_col].astype(str).to_numpy(),
                    label_map,
                ),
            }

        ref_label_counts = reference_static_cache[ref_cache_key]["ref_label_counts"]
        ref_fine_counts = reference_static_cache[ref_cache_key]["ref_fine_counts"]
        filtered_labels = reference_static_cache[ref_cache_key]["filtered_labels"]

        # GT-side label collapse (also the gt_label column of sthelar_cells_backbone.parquet read by _3)
        gt_labels_series = apply_label_map(gt_cells[gt_label_col].astype(str), label_map)
        gt_label_counts = summarize_counts(gt_labels_series)
        gt_mapped_label_scorable = ~gt_labels_series.isin(gt_mapped_unscorable_labels)

        gt_labels_present = gt_labels_series.tolist()
        gt_labels_present_scorable = gt_labels_series.loc[gt_mapped_label_scorable].tolist()
        covered_gt_classes = sorted(set(gt_labels_present_scorable) & set(ref_label_counts.keys()))
        covered_class_supports = [int(ref_label_counts[c]) for c in covered_gt_classes]

        missing_cells_modelgroup = _load_missing_cells_for_modelgroup(
            eval_data_root=eval_data_root,
            modelgroup=modelgroup,
            slide_id=slide,
            require_exists=True,
        )
        he_structural_ok = ~gt_cells["cell_id_int"].astype(np.int32).isin(missing_cells_modelgroup)

        meets_min_ref_cells = int(ref_filtered.filtered_rows.size) >= min_ref_cells
        meets_min_class_support = (
            len(covered_class_supports) > 0
            and min(covered_class_supports) >= min_class_support
        )
        meets_min_aligned_genes = int(gene_panel_summary["n_genes_G_base_qc"]) >= min_aligned_genes
        if not meets_min_ref_cells:
            logger.warning(
                "Slide=%s has only %d filtered reference cells (< %d)",
                slide, int(ref_filtered.filtered_rows.size), min_ref_cells
            )
        if not meets_min_class_support:
            logger.warning(
                "Slide=%s has mapped reference class support below minimum threshold %d",
                slide, min_class_support
            )
        if not meets_min_aligned_genes:
            logger.warning(
                "Slide=%s has only %d G_base_qc genes (< %d); official expression metrics will be undefined",
                slide, int(gene_panel_summary["n_genes_G_base_qc"]), min_aligned_genes
            )
        label_vocab.update(gt_label_counts.keys())
        label_vocab.update(ref_label_counts.keys())

        gt_label_present = gt_labels_series.isin(set(ref_label_counts.keys()))
        cell_manifest = pd.DataFrame(
            {
                "cell_row_key": gt_cells.index.astype(str),
                "cell_id_int": gt_cells["cell_id_int"].to_numpy(dtype=np.int32, copy=False),
                "cell_id": gt_cells["cell_id"].astype(str).to_numpy(),
                "slide_id": slide,
                "subset": subset,
                "source": cfg["data"]["source"],
                "tissue_HistoST": tissue,
                "gt_label": gt_labels_series.to_numpy(),
                "reference_stem": ref_spec.stem,
                "reference_h5ad": str(ref_spec.path),
                "gene_universe_file": str(gu_path),
                "gene_qc_file": str(gene_qc_path),
                "gene_universe_n_shared": int(gene_universe["n_shared"]),
                "n_genes_G_s_qc": int(gene_panel_summary["n_genes_G_s_qc"]),
                "n_genes_G_base_qc": int(gene_panel_summary["n_genes_G_base_qc"]),
                "is_he_structurally_eligible_modelgroup": he_structural_ok.astype(np.int8).to_numpy(),
                "gt_mapped_label_present_in_reference": gt_label_present.astype(np.int8).to_numpy(),
                "gt_mapped_label_scorable": gt_mapped_label_scorable.astype(np.int8).to_numpy(),
            }
        )
        cell_backbone_rows.append(cell_manifest)

        seed = int(cfg["umap"]["random_state"])
        sampled_rows = _sample_reference_rows_stratified(
            row_indices=np.arange(ref_filtered.filtered_rows.size, dtype=np.int64),
            labels=filtered_labels,
            max_total=int(cfg["umap"]["reference_max_cells_per_slide"]),
            min_per_class=int(cfg["umap"]["reference_min_cells_per_class"]),
            seed=seed + (_stable_int_from_text(slide) % 10_000),
        )
        np.save(sampling_dir / f"{slide}.npy", sampled_rows)

        audit = {
            "slide_id": slide,
            "subset": subset,
            "tissue_HistoST": tissue,
            "reference_stem": ref_spec.stem,
            "reference_h5ad": str(ref_spec.path),
            "reference_h5ad_info": ref_info,
            # after dropping the unscorable labels (retrieval pool size)
            "filtered_reference_cells": int(ref_filtered.filtered_rows.size),
            "fine_reference_cells": int(ref_filtered.fine_rows.size),
            "aligned_genes": int(gene_universe["n_shared"]),
            "gt_gene_name_source": "gt_h5ad.var_names",
            "g_base_match_field": gene_panel_summary.get("g_base_match_field"),
            "g_base_note": (
                "G_base membership may be matched using gt_gene_name; expression alignment and "
                "expression metrics use gene IDs."
            ),
            "gene_panel_summary": gene_panel_summary,
            "gene_qc_file": str(gene_qc_path),
            "gt_cells": int(gt_cells.shape[0]),
            # Counts at this eval_id's label_granularity (except reference_fine_label_counts)
            "label_granularity": label_granularity,
            "gt_label_counts": gt_label_counts,
            "reference_label_counts": ref_label_counts,
            "reference_fine_label_counts": ref_fine_counts,
            "gt_cells_with_uncovered_mapped_label": int((~gt_label_present).sum()),
            "gt_mapped_unscorable_labels": sorted(gt_mapped_unscorable_labels),
            "gt_cells_with_unscorable_mapped_label": int((~gt_mapped_label_scorable).sum()),
            "gt_cells_with_uncovered_scorable_mapped_label": int(((~gt_label_present) & gt_mapped_label_scorable).sum()),
            "mapped_scorable_fraction": float(gt_mapped_label_scorable.mean()),
            "mapped_coverage_fraction": float(gt_label_present.mean()),
            # Per-class GT-vs-reference coverage and contract verdict (src_A2/coverage.py)
            "coverage_contract": coverage_reports[slide].as_dict(),
            "all_filtered_rows_have_fine_label": True,
            "reference_filter_signature": {
                "require_mapped_label": True,
                "require_nonempty_expression": bool(require_nonempty),
                "mapped_label_column": ref_label_col,
                "fine_label_column": ref_fine_col,
            },
            # Reference rows removed for data.gt_mapped_unscorable_labels (after the label_granularity collapse)
            "reference_scorable_filter": {
                "gt_mapped_unscorable_labels": sorted(gt_mapped_unscorable_labels),
                "applied_to": "reference_rows_after_label_granularity_collapse",
                "label_granularity": label_granularity,
                "rows_dropped_by_label": dict(ref_scorable_drops),
                "n_rows_dropped": int(sum(ref_scorable_drops.values())),
            },
            "meets_minimum_reference_cells": bool(meets_min_ref_cells),
            "meets_minimum_class_support": bool(meets_min_class_support),
            "meets_minimum_aligned_genes_official": bool(meets_min_aligned_genes),
            "meets_minimum_G_base_qc_genes_official": bool(meets_min_aligned_genes),
            "minimum_class_support_all_reference_classes": min(ref_label_counts.values()) if ref_label_counts else 0,
            "median_class_support_all_reference_classes": float(np.median(list(ref_label_counts.values()))) if ref_label_counts else 0.0,
            "maximum_class_support_all_reference_classes": max(ref_label_counts.values()) if ref_label_counts else 0,
            "covered_gt_classes": covered_gt_classes,
            "minimum_class_support_gt_covered_classes": min(covered_class_supports) if covered_class_supports else 0,
            "median_class_support_gt_covered_classes": float(np.median(covered_class_supports)) if covered_class_supports else 0.0,
            "maximum_class_support_gt_covered_classes": max(covered_class_supports) if covered_class_supports else 0,
            "n_cells_missing_after_modelgroup_tokenization": int(len(missing_cells_modelgroup)),
            "missing_cells_modelgroup_dir": str(missing_dir),
            "gene_universe_hash": hash_mapping({"genes": gene_universe["genes"]}),
            "reference_row_filter_file": str(row_filter_dir / f"{slide}.npz"),
            "umap_sample_file": str(sampling_dir / f"{slide}.npy"),
        }
        write_json(audits_dir / f"{slide}.json", audit)
        reference_audits.append(audit)

        slide_rows.append(
            {
                "slide_id": slide,
                "subset": subset,
                "source": cfg["data"]["source"],
                "tissue_HistoST": tissue,
                "reference_stem": ref_spec.stem,
                "reference_h5ad": str(ref_spec.path),
                "filtered_reference_cells": int(ref_filtered.filtered_rows.size),
                "fine_reference_cells": int(ref_filtered.fine_rows.size),
                "reference_rows_dropped_unscorable": int(sum(ref_scorable_drops.values())),
                "aligned_genes": int(gene_universe["n_shared"]),
                "n_genes_G_s_raw": int(gene_panel_summary["n_genes_G_s_raw"]),
                "n_genes_G_s_qc": int(gene_panel_summary["n_genes_G_s_qc"]),
                "n_genes_G_base_raw_intersection": int(gene_panel_summary["n_genes_G_base_raw_intersection"]),
                "n_genes_G_base_qc": int(gene_panel_summary["n_genes_G_base_qc"]),
                "g_base_match_field": gene_panel_summary.get("g_base_match_field"),
                "status_G_s_qc": gene_panel_summary["status_G_s_qc"],
                "status_G_base_qc": gene_panel_summary["status_G_base_qc"],
                "gt_cells": int(gt_cells.shape[0]),
                "mapped_coverage_fraction": float(gt_label_present.mean()),
                "n_gt_classes": int(coverage_reports[slide].n_gt_classes),
                "n_covered_gt_classes": int(coverage_reports[slide].n_covered_classes),
                "coverage_contract_satisfied": bool(coverage_reports[slide].ok),
                "coverage_declared_exception": coverage_reports[slide].declared_reason,
                "meets_minimum_reference_cells": bool(meets_min_ref_cells),
                "meets_minimum_class_support": bool(meets_min_class_support),
                "meets_minimum_aligned_genes_official": bool(meets_min_aligned_genes),
                "meets_minimum_G_base_qc_genes_official": bool(meets_min_aligned_genes),
            }
        )
        logger.info(
            "Prepared shared artifacts for slide=%s subset=%s tissue=%s ref=%s gt_cells=%d ref_cells=%d G_s_raw=%d G_s_qc=%d G_base_qc=%d",
            slide,
            subset,
            tissue,
            ref_spec.stem,
            gt_cells.shape[0],
            ref_filtered.filtered_rows.size,
            gene_panel_summary["n_genes_G_s_raw"],
            gene_panel_summary["n_genes_G_s_qc"],
            gene_panel_summary["n_genes_G_base_qc"],
        )

    slide_df = pd.DataFrame(slide_rows).sort_values(["subset", "slide_id"]).reset_index(drop=True)
    cell_backbone_df = pd.concat(cell_backbone_rows, axis=0, ignore_index=True)
    label_vocab_list = sorted(label_vocab)
    label_to_index = {lab: i for i, lab in enumerate(label_vocab_list)}

    write_dataframe(slide_df, manifests_dir / "slides.parquet")
    write_dataframe(cell_backbone_df, manifests_dir / "sthelar_cells_backbone.parquet")
    # Written even when empty (excluded slides are not in slides.parquet)
    write_json(
        manifests_dir / "excluded_slides.json",
        {
            "min_scorable_gt_cells_per_slide": int(
                cfg["data"].get("min_scorable_gt_cells_per_slide", 0)
            ),
            "min_classes_per_slide": int(cfg["data"].get("min_classes_per_slide", 0)),
            # Support a class needs to count towards min_classes_per_slide
            "min_class_support_for_admission": int(
                cfg["data"].get("min_class_support_for_admission", 0)
            ),
            "label_granularity": label_granularity,
            "coarse_mapping_version": coarse_mapping_version,
            "gt_mapped_unscorable_labels": sorted(gt_mapped_unscorable_labels),
            # Cells on which the floors are counted
            "admission_population": "is_cell_and_modelgroup_token_eligible",
            "admission_modelgroup": modelgroup,
            "n_excluded": len(excluded_slides),
            "excluded": excluded_slides,
        },
    )
    write_dataframe(pd.DataFrame(reference_audits), audits_dir / "reference_audit_summary.csv")
    _n_hits = sum(1 for v in reference_filter_cache_files_used.values() if v.get("cache_hit"))
    logger.info(
        "Reference row-filter cache: %d blob(s) used, %d hit / %d rebuilt.",
        len(reference_filter_cache_files_used),
        _n_hits,
        len(reference_filter_cache_files_used) - _n_hits,
    )
    # Location of the shared row-filter cache used by this eval_id
    write_json(
        audits_dir / "reference_filter_cache_location.json",
        {
            "shared_cache_dir": str(reference_filter_cache_dir),
            "scope": "task_level_shared_across_eval_ids",
            # sha256 is not computed (compute_sha256=False): identity is path + size + mtime (see sha256_in_signature)
            "keyed_by": "reference_filter_signature: h5ad path + size + mtime (+ mtime_ns) + both "
                        "label columns + require_nonempty_expression; granularity-invariant. The "
                        "signature's reference_h5ad_sha256 field is null unless a caller opts into "
                        "hashing -- see sha256_in_signature per file below.",
            "verified_on_every_hit": True,
            # Cache files used by this run (cache_hit false: rebuilt)
            "cache_files_used": {
                name: reference_filter_cache_files_used[name]
                for name in sorted(reference_filter_cache_files_used)
            },
            "n_cache_files_used": len(reference_filter_cache_files_used),
            "n_cache_hits": sum(
                1 for v in reference_filter_cache_files_used.values() if v.get("cache_hit")
            ),
        },
    )
    _n_gs_hits = sum(1 for v in reference_gene_stats_cache_files_used.values() if v.get("cache_hit"))
    logger.info(
        "Reference gene-stats cache: %d blob(s) used, %d hit / %d rebuilt.",
        len(reference_gene_stats_cache_files_used),
        _n_gs_hits,
        len(reference_gene_stats_cache_files_used) - _n_gs_hits,
    )
    # Location of the shared gene-stats cache used by this eval_id
    write_json(
        audits_dir / "reference_gene_stats_cache_location.json",
        {
            "shared_cache_dir": str(reference_gene_stats_cache_dir),
            "scope": "task_level_shared_across_eval_ids",
            "keyed_by": "reference_gene_stats_signature: h5ad path + size + mtime (+ mtime_ns) + "
                        "sha256 of the selected reference rows + sha256 of the selected gene "
                        "columns (in order) + normalize_target_sum + schema_version. The chunking "
                        "settings (row_block_size / target_block_mb) are not part of the key.",
            "verified_on_every_hit": True,
            "schema_version": _REFERENCE_GENE_STATS_SCHEMA_VERSION,
            "cached_arrays": list(_REFERENCE_GENE_STATS_ARRAYS),
            "cache_files_used": {
                name: reference_gene_stats_cache_files_used[name]
                for name in sorted(reference_gene_stats_cache_files_used)
            },
            "n_cache_files_used": len(reference_gene_stats_cache_files_used),
            "n_cache_hits": _n_gs_hits,
        },
    )
    write_json(
        manifests_dir / "label_vocabulary.json",
        {
            "ct_cat_id": ct_cat_id,
            # Class axis of all mapped-label metrics (coarse_mapping: applied fine -> coarse map, None under fine)
            "label_granularity": label_granularity,
            "coarse_mapping_version": coarse_mapping_version,
            "labels": label_vocab_list,
            "label_to_index": label_to_index,
            "coarse_mapping": (None if label_map is None else {str(k): str(v) for k, v in label_map.items()}),
        },
    )

    summary = {
        "n_slides": int(slide_df.shape[0]),
        "n_cells": int(cell_backbone_df.shape[0]),
        "label_granularity": label_granularity,
        "coarse_mapping_version": coarse_mapping_version,
        # Excluded from both the scored GT population and the retrieval pool
        "gt_mapped_unscorable_labels": sorted(gt_mapped_unscorable_labels),
        # Slide admission (n_slides counts admitted slides only)
        "min_scorable_gt_cells_per_slide": int(
            cfg["data"].get("min_scorable_gt_cells_per_slide", 0)
        ),
        "min_classes_per_slide": int(cfg["data"].get("min_classes_per_slide", 0)),
        "min_class_support_for_admission": int(
            cfg["data"].get("min_class_support_for_admission", 0)
        ),
        "admission_population": "is_cell_and_modelgroup_token_eligible",
        "n_slides_excluded_by_admission": len(excluded_slides),
        "slides_excluded_by_admission": [str(e["slide_id"]) for e in excluded_slides],
        "n_labels": int(len(label_vocab_list)),
        "labels": label_vocab_list,
        "slides_by_subset": slide_df.groupby("subset")["slide_id"].nunique().to_dict(),
    }
    write_json(paths.shared_dir / "shared_summary.json", summary)
    return summary
