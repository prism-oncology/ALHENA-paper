"""
Choose the final gene selection for the given training_dataset_id 

Rule 1:
- If a slide has <= MIN_GENES_FOR_HVG genes, keep all genes.
- If a slide has >  MIN_GENES_FOR_HVG genes, keep:
    - Highly variable genes with given min_disp threshold
    - UNION genes present in >= N small slides (<= MIN_GENES_FOR_HVG genes)
    - UNION top-K highly expressed genes on this slide (after normalize/log1p)

Rule 2:
- If a slide has cells in train set and another set (valid/test), do gene selection only based on information from train cells.
- If a slide has only cells in valid set or only in test set, do gene selection based on information from all cells.

Saves a JSON file mapping slide_id -> list of selected genes in the output_folder.
Saved in <ROOT_HistoST>/train/training_datasets/<dataset_id>/<training_dataset_id>/genes_training_<training_dataset_id>.json

With the new scConcept-based Novae teacher, no gene selection is needed: all genes are sent to scConcept.
=> Not needed by the teacher any more; its JSON is still read by tcga_benchmark/_5_build_signature_support_training.py for Task B.
"""

import sys
from pathlib import Path
import argparse
import os
import scanpy as sc
from anndata import AnnData
import json
from typing import Dict, List, Set, Tuple, Optional
import numpy as np
from scipy import sparse
import pandas as pd
import shutil

# Add the project root to sys.path so that 'src' can be imported
project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.constants import get_full_path



MIN_GENES_FOR_HVG = 700
KEY_HIGHLY_VARIABLE = "highly_variable"
DEFAULT_CELL_ID_COL = "cell_id"



def _xenium_str_ids_to_int64(cell_id_strs: pd.Series) -> Optional[np.ndarray]:
    """
    Vectorized conversion of Xenium Explorer string cell IDs (e.g. 'aaaachba-1')
    into int64 IDs (e.g. 10001). Returns None if format doesn't match.
    """
    if cell_id_strs.empty:
        return np.empty((0,), dtype=np.int64)

    # Ensure string dtype, then convert to fixed-width bytes (dtype 'S<maxlen>')
    s = cell_id_strs.astype(str).to_numpy()
    b = s.astype("S")  # bytes; width = max string length in s

    # Lightweight format sanity-check on a small sample (avoid O(n) string ops here)
    n_sample = min(b.shape[0], 32)
    sample = b[:n_sample]
    if not np.all(np.char.endswith(sample, b"-1")):
        return None

    # Bytes layout: (n, maxlen). Take first 8 letters, map a-p -> 0-15, base-16 decode.
    # NOTE: reshape uses maxlen derived from dtype 'S<maxlen>'.
    maxlen = b.dtype.itemsize
    raw = b.tobytes()
    chars = np.frombuffer(raw, dtype=np.uint8).reshape(b.shape[0], maxlen)[:, :8]
    digits = chars.astype(np.int16) - ord("a")  # int16 to safely check negatives

    # Validate all digits are in [0, 15] (a..p); if not, it's not Xenium format.
    if np.any(digits < 0) or np.any(digits > 15):
        return None

    powers = np.array([16**i for i in range(7, -1, -1)], dtype=np.int64)
    out = (digits.astype(np.int64) * powers).sum(axis=1) + 1  # +1 shift, as in int_cell_id
    return out



def _load_cell_ids_sets_npz(cell_ids_sets_path: str) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Load the output of cell_ids_sets.py (npz) and return:
      train_only_by_slide, valid_only_by_slide, test_only_by_slide
    Each is a dict: slide_id -> np.ndarray of cell IDs
    """
    if not os.path.isfile(cell_ids_sets_path):
        raise FileNotFoundError(f"[ERROR] cell_ids_sets file not found: '{cell_ids_sets_path}'.")
    npz = np.load(cell_ids_sets_path, allow_pickle=True)
    # Saved as dict objects in np.savez_compressed => need .item()
    def _get_dict(name: str) -> Dict[str, np.ndarray]:
        if name not in npz.files:
            return {}
        obj = npz[name]
        # obj is a 0-d object array containing the dict
        try:
            return obj.item()
        except Exception:
            # be defensive: some numpy versions behave differently
            return dict(obj)

    train_only = _get_dict("train_only_by_slide")
    valid_only = _get_dict("valid_only_by_slide")
    test_only  = _get_dict("test_only_by_slide")
    return train_only, valid_only, test_only



def _standardize_cell_ids_for_isin(
    obs_cell_ids: pd.Series,
    target_cell_ids: np.ndarray,
) -> Tuple[pd.Series, np.ndarray]:
    """
    Make obs_cell_ids and target_cell_ids comparable for membership testing
    - Prefer int64 comparison when possible.
    - Otherwise fall back to string comparison.
    """
    tgt_arr = np.asarray(target_cell_ids)

    # Try pure-numeric comparison first (fastest when both are numeric-like)
    obs_num = pd.to_numeric(obs_cell_ids, errors="coerce")
    if not obs_num.isna().any():
        try:
            tgt_int = tgt_arr.astype(np.int64, copy=False)
        except Exception:
            tgt_int = tgt_arr.astype(np.int64)
        return obs_num.astype(np.int64), tgt_int

    # Xenium special-case:
    # - npz provides int cell_ids
    # - adata.obs provides Xenium string cell_ids like 'aaaachba-1'
    # Prefer converting strings -> ints in a vectorized way and compare int64.
    if tgt_arr.dtype.kind in ("i", "u"):
        obs_int = _xenium_str_ids_to_int64(obs_cell_ids)
        if obs_int is not None:
            return pd.Series(obs_int, index=obs_cell_ids.index, dtype=np.int64), tgt_arr.astype(np.int64, copy=False)

    # Also handle the reverse (obs ints, target Xenium strings)
    if obs_num.isna().any() and tgt_arr.dtype.kind in ("O", "U", "S"):
        tgt_series = pd.Series(tgt_arr.astype(str))
        tgt_int = _xenium_str_ids_to_int64(tgt_series)
        if tgt_int is not None:
            # obs is non-numeric strings here, so this branch is mostly for completeness;
            # but if obs_cell_ids are already integer dtype stored as object, try coercion again.
            obs_num2 = pd.to_numeric(obs_cell_ids, errors="coerce")
            if not obs_num2.isna().any():
                return obs_num2.astype(np.int64), tgt_int

    # Fallback to string comparison (works for arbitrary IDs)
    return obs_cell_ids.astype(str), tgt_arr.astype(str)



def _subset_adata_to_cell_ids(
    adata: AnnData,
    cell_ids: np.ndarray,
    cell_id_col: str = DEFAULT_CELL_ID_COL,
) -> AnnData:
    """
    Return a copy of adata restricted to obs[cell_id_col] in cell_ids.
    """
    if cell_id_col not in adata.obs.columns:
        raise KeyError(f"[ERROR] '{cell_id_col}' not found in adata.obs columns: {list(adata.obs.columns)}")
    if cell_ids is None or len(cell_ids) == 0:
        # caller decides whether empty is allowed; returning empty adata can break downstream scanpy steps
        return adata[:0].copy()
    obs_std, tgt_std = _standardize_cell_ids_for_isin(adata.obs[cell_id_col], cell_ids)
    mask = obs_std.isin(tgt_std).to_numpy()
    return adata[mask].copy()



def _resolve_cells_for_gene_selection(
    slide_id: str,
    train_only_by_slide: Dict[str, np.ndarray],
    valid_only_by_slide: Dict[str, np.ndarray],
    test_only_by_slide: Dict[str, np.ndarray],
) -> Tuple[Optional[np.ndarray], str]:
    """
    - If a slide has train cells and also valid/test cells => use TRAIN cells only.
    - If a slide has no train cells (only valid/test)      => use all cells (return None).
    Returns (cell_ids_to_use or None, reason_string).
    """
    trn = train_only_by_slide.get(slide_id, np.array([], dtype=np.int64))
    vld = valid_only_by_slide.get(slide_id, np.array([], dtype=np.int64))
    tst = test_only_by_slide.get(slide_id,  np.array([], dtype=np.int64))

    has_train = (len(trn) > 0)
    has_other = (len(vld) > 0) or (len(tst) > 0)

    if has_train and has_other:
        return trn, "shared slide (train + valid/test) => using TRAIN-only cells"
    if has_train and not has_other:
        # Still follow "use train cells only" (here train-only == all train cells of that slide under this split).
        return trn, "train-only slide => using TRAIN-only cells"
    # No train cells: valid/test-only slide => use all cells
    return None, "valid/test-only slide => using ALL cells"



def _maybe_to_memory_matrix(X):
    """
    anndata backed sparse datasets may return a proxy object.
    Convert to an in-memory scipy sparse matrix or ndarray when needed.
    """
    if sparse.issparse(X):
        return X
    if hasattr(X, "to_memory"):
        return X.to_memory()
    return X



def _highly_variable_genes(adata: AnnData, min_disp: float):
    """
    Compute highly variable genes with min_disp threshold.
    """

    key_highly_variable = KEY_HIGHLY_VARIABLE
    min_genes_for_hvg = MIN_GENES_FOR_HVG
    min_genes = 20
    n_hvg_threshold = 500

    if adata.n_vars <= min_genes_for_hvg:  # if too few genes, keep all genes
        print(f"[WARNING] Only {adata.n_vars} genes found. Keeping all genes - no HVG selection.")
        adata.var[key_highly_variable] = True
        return

    # Perform sc.pp.normalize_total(adata, target_sum=1e4) and sc.pp.log1p(adata) before HVG selection,
    # after checking that adata.X holds raw counts
    if adata.X.min() < 0:
        raise ValueError(f"Negative values found in adata.X. Are you sure this is raw counts in adata.X?")
    max_value = adata.X.max()
    if max_value < 10:
        raise ValueError(f"Max value in adata.X is {max_value}, which is less than 10. Are you sure this is raw counts in adata.X?")
    if int(max_value) != max_value:
        raise ValueError(f"Max value in adata.X is {max_value}, which is not an integer. Are you sure this is raw counts in adata.X?")
    print(f"[INFO] Normalizing and logarithmizing the data")
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata) 

    print("[INFO] Computing highly variable genes")
    sc.pp.highly_variable_genes(adata, min_disp=min_disp, flavor="seurat")  # Annotate highly variable genes
                                                                            # Expects logarithmized data, except when flavor='seurat_v3'/'seurat_v3_paper', in which count data is expected. / Default is 'seurat'
                                                                            # For cutoffs, default values are min_mean=0.0125, max_mean=3, min_disp=0.5, max_disp=inf; AND n_top_genes=None

    n_hvg = int(adata.var[key_highly_variable].sum())

    if n_hvg < min_genes:
        print(f"[WARNING] Only {n_hvg} highly variable genes were found.")

    if n_hvg < n_hvg_threshold and n_hvg < adata.n_vars // 2:
        print(f"[WARNING] Only {n_hvg} highly variable genes were found. Keeping at least half of the genes.")
        sc.pp.highly_variable_genes(adata, n_top_genes=adata.n_vars // 2)  # keep at least half of the genes



def _small_slide_gene_support(
    adata_folder: str,
    cache_path: str,
    train_csv_path: str,
    cell_ids_sets_path: str,
    cell_id_col: str = DEFAULT_CELL_ID_COL,
    min_genes_for_hvg: int = MIN_GENES_FOR_HVG,
) -> Dict[str, int]:
    """
    Build or load a dictionary: gene -> number of 'small' TRAIN slides (<= min_genes_for_hvg genes)
    in which the gene is present, using only TRAIN cells on those slides (no leakage).

    TRAIN slides are discovered from `cell_count_train.csv` (slide_id derived from patch IDs).
    For each TRAIN slide the selection is restricted to its TRAIN-only cell IDs from the
    `cell_ids_sets.py` output.
    Gene "presence" is defined as: expressed (non-zero) in at least one selected (train) cell.
    """
    if os.path.isfile(cache_path):
        with open(cache_path, "r") as f:
            payload = json.load(f)
        print(f"[INFO] Loaded small-slide gene support from cache: {cache_path} "
              f"({payload.get('n_small_slides', 0)} small slides).")
        return payload["gene_support"]

    if not os.path.isfile(train_csv_path):
        raise FileNotFoundError(
            f"[ERROR] Could not find train CSV at '{train_csv_path}'. "
            f"Expected a file like 'cell_count_train.csv' in the dataset folder."
        )

    # Load cell ID sets (no-leak)
    train_only_by_slide, valid_only_by_slide, test_only_by_slide = _load_cell_ids_sets_npz(cell_ids_sets_path)

    # Read train patches and derive the list of TRAIN slide_ids
    df_train = pd.read_csv(train_csv_path, usecols=["patch_id"])
    train_slide_ids = sorted(set(df_train["patch_id"].str.rsplit("_", n=1).str[0]))
    paths = [os.path.join(adata_folder, f"{sid}.h5ad") for sid in train_slide_ids]
    n_missing = sum(1 for p in paths if not os.path.isfile(p))
    if n_missing:
        raise FileNotFoundError(f"[WARNING] {n_missing} training slides listed in CSV have no matching .h5ad in '{adata_folder}'.")

    gene_counts: Dict[str, int] = {}
    n_small = 0
    print(f"[INFO] Scanning {len(paths)} TRAIN slides to build small-slide gene support...")
    for p in paths:
        try:
            ad = sc.read_h5ad(p, backed="r")
        except Exception as e:
            print(f"[WARNING] Skipping {p}: {e}")
            continue
        slide_id = os.path.basename(p).replace(".h5ad", "")
        if ad.n_vars <= min_genes_for_hvg:
            # Restrict to TRAIN-only cells for this training slide (conservative, no leakage)
            train_cell_ids, reason = _resolve_cells_for_gene_selection(
                slide_id=slide_id,
                train_only_by_slide=train_only_by_slide,
                valid_only_by_slide=valid_only_by_slide,
                test_only_by_slide=test_only_by_slide,
            )
            if train_cell_ids is None:
                # This would mean "no train cells" which is unexpected for a TRAIN slide; skip to avoid leakage.
                print(f"[WARNING] TRAIN slide '{slide_id}' has no TRAIN-only cells per cell_ids_sets ({reason}). Skipping for support to avoid leakage.")
                ad.file.close()
                continue
            if cell_id_col not in ad.obs.columns:
                print(f"[WARNING] '{cell_id_col}' missing in obs for '{slide_id}'. Skipping for support.")
                ad.file.close()
                continue

            obs_std, tgt_std = _standardize_cell_ids_for_isin(ad.obs[cell_id_col], train_cell_ids)
            idx = np.flatnonzero(obs_std.isin(tgt_std).to_numpy())
            if idx.size == 0:
                print(f"[WARNING] TRAIN slide '{slide_id}' has 0 matching TRAIN-only cells after filtering. Skipping for support.")
                ad.file.close()
                continue

            X_sel = _maybe_to_memory_matrix(ad.X[idx, :])
            # Determine genes with any non-zero in selected cells
            if sparse.issparse(X_sel):
                present = (X_sel.getnnz(axis=0) > 0)
                present = np.asarray(present).ravel()
            else:
                X_arr = np.asarray(X_sel)
                present = np.any(X_arr != 0, axis=0)

            present_genes = ad.var_names[present]
            n_small += 1
            for g in present_genes:
                gene_counts[g] = gene_counts.get(g, 0) + 1

        ad.file.close()  # close backed handle

    payload = {"n_small_slides": n_small, "gene_support": gene_counts, "cell_ids_sets_path": os.path.abspath(cell_ids_sets_path)}
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[INFO] Built small-slide gene support: {n_small} small slides, "
          f"{len(gene_counts)} unique genes. Cached at {cache_path}.")
    return gene_counts




def _top_highly_expressed_genes(adata: AnnData, top_k: int) -> Set[str]:
    """
    Return the set of top_k most highly expressed genes on this slide.
    Assumes adata.X has already been normalized/log1p in the current pipeline,
    which is fine for relative ranking.
    """
    if top_k <= 0:
        return set()
    # Use scanpy to compute per-gene means (works for dense and sparse)
    sc.pp.calculate_qc_metrics(adata, percent_top=None, log1p=False, inplace=True)
    # After the normalize+log1p above, these are means on transformed data, which still
    # gives a robust ranking for "high expression". For very large matrices, avoid materializing.
    if "mean_counts" in adata.var:
        means = adata.var["mean_counts"].to_numpy()
    else:
        # Fallback: compute means directly
        X = adata.X
        if sparse.issparse(X):
            means = (X.mean(axis=0)).A1
        else:
            means = np.asarray(X).mean(axis=0)
    top_k = min(top_k, adata.n_vars)
    idx = np.argpartition(means, -top_k)[-top_k:]
    return set(adata.var_names[idx])




def main(args):

    adata_folder = get_full_path('STHELAR_HistoST', 'adata_graph', as_str=True)
    output_folder = get_full_path('train', 'training_datasets', args.dataset_id, args.training_dataset_id, as_str=True)
    cell_ids_sets_path = os.path.join(output_folder, "cell_ids_sets.npz")
    training_dataset_id = args.training_dataset_id
    support_cache = os.path.join(output_folder, f"small_slide_gene_support_{training_dataset_id}.json")

    # Get list of slide_ids using slide_ids.txt in the output folder
    slide_ids_path = os.path.join(output_folder, "slide_ids.txt")
    if not os.path.isfile(slide_ids_path):
        raise FileNotFoundError(f"[ERROR] slide_ids.txt not found in output folder '{output_folder}'. Expected a file listing the slide IDs to process, one per line.")
    with open(slide_ids_path, "r") as f:
        slide_ids = [line.strip() for line in f if line.strip()]
    print(f"[INFO] Found {len(slide_ids)} slide IDs to process from '{slide_ids_path}': {slide_ids}")

    # Always start from a clean support cache to avoid reusing stale results
    if os.path.exists(support_cache):
        os.remove(support_cache)
        print(f"[INFO] Deleted existing support cache: {support_cache}")

    # Build a fresh mapping
    json_path = os.path.join(output_folder, f"genes_training_{training_dataset_id}.json")
    if os.path.isfile(json_path):
        # Remove any stale file so old slide_ids are never reused accidentally
        os.remove(json_path)
        print(f"[INFO] Deleted existing genes JSON to rebuild cleanly: {json_path}")

    keeping_dict: Dict[str, List[str]] = {}

    # Load cell-id sets once
    train_only_by_slide, valid_only_by_slide, test_only_by_slide = _load_cell_ids_sets_npz(cell_ids_sets_path)

    for slide_id in slide_ids:

        print(f"\n==== Proccessing {slide_id} ====")

        # Resolve which cells to use (no-leakage selection)
        cell_ids_to_use, reason = _resolve_cells_for_gene_selection(
            slide_id=slide_id,
            train_only_by_slide=train_only_by_slide,
            valid_only_by_slide=valid_only_by_slide,
            test_only_by_slide=test_only_by_slide,
        )
        print(f"[INFO] Cell filtering rule for '{slide_id}': {reason}")

        # Load the AnnData object
        print("\n** Loading the AnnData object...")
        adata = sc.read_h5ad(os.path.join(adata_folder, f"{slide_id}.h5ad"))
        print(adata)

        # Apply cell filtering if needed (keeping all genes for now)
        if cell_ids_to_use is not None:
            before_n = adata.n_obs
            adata = _subset_adata_to_cell_ids(adata, cell_ids_to_use, cell_id_col=DEFAULT_CELL_ID_COL)
            after_n = adata.n_obs
            print(f"[INFO] Filtered cells for gene selection: {before_n} -> {after_n}")
            if after_n == 0:
                raise ValueError(
                    f"[ERROR] After filtering to TRAIN-only cells, slide '{slide_id}' has 0 cells. "
                    f"Check consistency between masks cell IDs and adata.obs['{DEFAULT_CELL_ID_COL}']."
                )

        if args.no_selection:
            print("\n** No filtering will be performed, keeping all genes...")
            keeping_genes = adata.var.index.tolist()

        else:
            # Strategy:
            # - If <= MIN_GENES_FOR_HVG: keep all genes.
            # - If >  MIN_GENES_FOR_HVG: HVG(min_disp) UNION genes present in >= args.min_smallslide_support small slides UNION top-K highly expressed genes on this slide.
            print(f"\n** Computing selection (min_disp={args.min_disp}, min_smallslide_support={args.min_smallslide_support}, top_high_expr={args.top_high_expr})...")

            # 1) HVG (or keep-all if small)
            print(f"Step1: HVG selection:")
            _highly_variable_genes(adata, min_disp=args.min_disp)
            base_mask = adata.var[KEY_HIGHLY_VARIABLE].to_numpy()
            selected: Set[str] = set(adata.var_names[base_mask])

            # 2) If this is a 'big' slide, union with cross-slide support + high-expression
            print(f"Step2: Cross-slide support + high-expression:")
            if adata.n_vars > MIN_GENES_FOR_HVG:
                train_csv_path = os.path.join(output_folder, "cell_count_train.csv")
                support_dict = _small_slide_gene_support(
                    adata_folder=adata_folder,
                    cache_path=support_cache,
                    train_csv_path=train_csv_path,
                    cell_ids_sets_path=cell_ids_sets_path,
                    cell_id_col=DEFAULT_CELL_ID_COL,
                )
                # Genes present in >= n small slides
                keep_by_support = {
                    g for g in adata.var_names
                    if support_dict.get(g, 0) >= args.min_smallslide_support
                }
                # Top highly expressed genes on this slide (after normalize/log1p above)
                keep_high_expr = _top_highly_expressed_genes(adata, args.top_high_expr)
                # Union
                before = len(selected)
                selected |= keep_by_support
                added_support = len(selected) - before
                before2 = len(selected)
                selected |= keep_high_expr
                added_high_expr = len(selected) - before2
                print(f"[INFO] HVG base: {before} | +small-slide support: +{added_support} "
                    f"| +high-expression: +{added_high_expr} | total now: {len(selected)}")
            else:
                print(f"[INFO] Again this is a small slide with only {adata.n_vars} genes, and thus no further filtering is performed.")

            keeping_genes = sorted(list(selected))

        
        print(f"\n===> Number of selected genes: {len(keeping_genes)} (over a total of {adata.n_vars} genes)")

        # Store result in-memory
        keeping_dict[slide_id] = keeping_genes

        print("\nDone.")

    # Write final JSON once
    tmp_path = json_path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(keeping_dict, f, indent=2)
    os.replace(tmp_path, json_path)
    print(f"[INFO] Saved selected genes for {len(keeping_dict)} slides to {json_path}.")

    # Clean up cache at end
    if os.path.exists(support_cache):
        os.remove(support_cache)
        print(f"[INFO] Deleted support cache: {support_cache}")



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Get final gene selection for each slide for the training")

    parser.add_argument("--dataset_id", type=str,
                        default="ds_ct1_20x_224_o112",
                        help="Dataset_id, e.g. ds_ct1_20x_224_o112")
    
    parser.add_argument("--training_dataset_id", type=str,
                        default="ds_ct1_20x_224_o112_1",
                        help="Training_dataset_id, e.g. ds_ct1_20x_224_o112_1")
    
    parser.add_argument('--no_selection', 
                        action='store_true', 
                        help="If set, no filtering will be performed, and all genes will be kept")
    
    parser.add_argument('--min_disp', type=float, 
                        default=-1.5, 
                        help="Minimum dispersion_norm for highly variable genes (hvg) selection => to choose, look at src/preprocessing/st/overview_disp_norm_genes.ipynb")
    
    parser.add_argument('--min_smallslide_support', type=int, 
                        default=5, 
                        help="Keep any gene present in at least N slides among the slides with <=700 genes")
    
    parser.add_argument('--top_high_expr', type=int, 
                        default=0,
                        help="Additionally keep the top-K highly expressed genes on the current slide (computed after normalize/log1p). Set 0 to disable. Recommended: 0 because we are using z-score during training.")
    
    args = parser.parse_args()
    main(args)