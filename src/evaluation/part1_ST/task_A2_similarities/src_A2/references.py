import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from src_A2.common import CanonicalGeneRules, assert_no_duplicates, canonicalize_gene_ids
from src_A2.label_granularity import apply_label_map


@dataclass
class ReferenceSpec:
    tissue: str
    path: Path
    stem: str


@dataclass
class FilteredReference:
    obs: pd.DataFrame
    filtered_rows: np.ndarray
    fine_rows: np.ndarray
    mapped_label_column: str
    fine_label_column: str



def load_reference_mapping(path: Path) -> Dict[str, str]:
    with path.open("r") as f:
        mapping = json.load(f)
    if not isinstance(mapping, dict) or len(mapping) == 0:
        raise ValueError(f"Invalid reference mapping at {path}")
    return {str(k): str(v) for k, v in mapping.items()}



def resolve_reference_for_tissue(tissue: str, mapping: Mapping[str, str], mapping_path: Path) -> ReferenceSpec:
    if tissue not in mapping:
        raise KeyError(f"Tissue '{tissue}' was not found in reference mapping {mapping_path}")
    p = Path(mapping[tissue])
    if not p.is_absolute():
        p = (mapping_path.parent / p).resolve()
    if not p.exists():
        raise FileNotFoundError(p)
    return ReferenceSpec(tissue=tissue, path=p, stem=p.stem)



def read_obs_columns(h5ad_path: Path, columns: Sequence[str]) -> pd.DataFrame:
    a = ad.read_h5ad(h5ad_path, backed="r")
    try:
        missing = [c for c in columns if c not in a.obs.columns]
        if missing:
            raise KeyError(f"Missing obs columns in {h5ad_path}: {missing}")
        obs = a.obs.loc[:, list(columns)].copy()
        obs.index = obs.index.astype(str)
        return obs
    finally:
        if getattr(a, "file", None) is not None:
            a.file.close()



def iter_backed_row_sums(h5ad_path: Path, chunk_size: int = 4096) -> np.ndarray:
    a = ad.read_h5ad(h5ad_path, backed="r")
    try:
        n = int(a.n_obs)
        out = np.zeros(n, dtype=np.float64)
        for start in range(0, n, chunk_size):
            stop = min(start + chunk_size, n)
            X = a[start:stop].X
            if sparse.issparse(X):
                out[start:stop] = np.asarray(X.sum(axis=1)).ravel()
            else:
                out[start:stop] = np.asarray(X).sum(axis=1)
        return out
    finally:
        if getattr(a, "file", None) is not None:
            a.file.close()



def filter_reference_rows(
    reference_h5ad_path: Path,
    mapped_label_column: str,
    fine_label_column: str,
    require_nonempty_expression: bool,
    chunk_size: int = 4096,
) -> FilteredReference:
    obs = read_obs_columns(reference_h5ad_path, [mapped_label_column, fine_label_column])

    mapped_str = obs[mapped_label_column].astype(str).str.strip()
    fine_str = obs[fine_label_column].astype(str).str.strip()

    mapped_ok = obs[mapped_label_column].notna() & (mapped_str.str.len() > 0)
    fine_ok = obs[fine_label_column].notna() & (fine_str.str.len() > 0)

    if require_nonempty_expression:
        row_sums = iter_backed_row_sums(reference_h5ad_path, chunk_size=chunk_size)
        expr_ok = row_sums > 0
    else:
        expr_ok = np.ones(obs.shape[0], dtype=bool)

    official_mask = mapped_ok.to_numpy() & expr_ok
    fine_mask = official_mask & fine_ok.to_numpy()

    filtered_rows = np.flatnonzero(official_mask).astype(np.int64)
    fine_rows = np.flatnonzero(fine_mask).astype(np.int64)

    if filtered_rows.size == 0:
        raise ValueError(f"Reference {reference_h5ad_path} has 0 cells after official filtering")
    if fine_rows.size == 0:
        raise ValueError(f"Reference {reference_h5ad_path} has 0 cells after fine-label filtering")

    return FilteredReference(
        obs=obs,
        filtered_rows=filtered_rows,
        fine_rows=fine_rows,
        mapped_label_column=mapped_label_column,
        fine_label_column=fine_label_column,
    )



def drop_unscorable_reference_rows(
    ref_filtered: FilteredReference,
    *,
    label_map: Optional[Mapping[str, str]],
    unscorable_labels: Iterable[str],
    reference_h5ad_path: Path,
) -> Tuple[FilteredReference, Dict[str, int]]:
    """
    Remove from the filtered reference the rows whose mapped label (after the label_granularity collapse) is unscorable.
    Returns the narrowed reference and {label: n_rows_dropped}.
    """
    unscorable = {str(x) for x in unscorable_labels}
    if not unscorable:
        return ref_filtered, {}

    obs = ref_filtered.obs
    column = ref_filtered.mapped_label_column
    filtered_rows = np.asarray(ref_filtered.filtered_rows, dtype=np.int64)

    # Collapse only the rows kept by the official filter (null/empty labels already removed)
    mapped = apply_label_map(obs.iloc[filtered_rows][column].astype(str), label_map)
    mapped_values = np.asarray(mapped, dtype=object).reshape(-1)
    drop = pd.Series(mapped_values, dtype=object).isin(unscorable).to_numpy()

    if not bool(drop.any()):
        return ref_filtered, {}

    dropped_counts = pd.Series(mapped_values[drop], dtype=object).value_counts().sort_index()
    audit = {str(k): int(v) for k, v in dropped_counts.items()}

    keep_rows = filtered_rows[~drop]
    if keep_rows.size == 0:
        raise ValueError(
            f"Reference {reference_h5ad_path} has 0 cells left after dropping the "
            f"classification-unscorable mapped labels {sorted(unscorable)} "
            f"(dropped {audit}). Fix data.gt_mapped_unscorable_labels or supply a different reference "
            "for this tissue."
        )

    # fine_rows is a subset of filtered_rows: narrow it with the same kept rows
    keep_mask = np.zeros(int(obs.shape[0]), dtype=bool)
    keep_mask[keep_rows] = True
    fine_rows = np.asarray(ref_filtered.fine_rows, dtype=np.int64)

    return (
        FilteredReference(
            obs=obs,
            filtered_rows=keep_rows,
            fine_rows=fine_rows[keep_mask[fine_rows]],
            mapped_label_column=column,
            fine_label_column=ref_filtered.fine_label_column,
        ),
        audit,
    )



def build_label_counts(obs: pd.DataFrame, rows: np.ndarray, column: str) -> Dict[str, int]:
    """Label -> count over rows (labels stripped, as in apply_label_map)."""
    s = obs.iloc[rows][column].astype(str).str.strip()
    vc = s.value_counts().sort_index()
    return {str(k): int(v) for k, v in vc.items()}



def get_canonical_gene_ids_from_var(
    h5ad_path: Path,
    gene_id_column: str,
    rules: CanonicalGeneRules,
) -> List[str]:
    a = ad.read_h5ad(h5ad_path, backed="r")
    try:
        if gene_id_column not in a.var.columns:
            raise KeyError(f"Missing var['{gene_id_column}'] in {h5ad_path}")
        gene_ids = canonicalize_gene_ids(a.var[gene_id_column].tolist(), rules=rules)
        assert_no_duplicates(gene_ids, context=f"{h5ad_path}:{gene_id_column}")
        return gene_ids
    finally:
        if getattr(a, "file", None) is not None:
            a.file.close()
