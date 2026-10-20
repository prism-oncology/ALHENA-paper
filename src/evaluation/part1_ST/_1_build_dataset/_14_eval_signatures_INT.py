"""
Build the internal gene-signature panel of the Task A.1 signature-target mode: co-expression modules among the G_base
genes (consensus of per-slide correlations on TRAIN cells, average-linkage clustering, eigengene weights).

Saved in <ROOT_HistoST> / eval / eval_datasets / eval_signatures_INT / <eval_signatures_INT_id> /
    - eval_signatures_INT_table.parquet: one row per (signature_id, gene, weight)
    - config.json: parameters, input G_base provenance and per-module summary
"""

import sys
from pathlib import Path
import argparse
import logging
import os
import json
import hashlib
import datetime as _dt
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from scipy import sparse

logger = logging.getLogger(__name__)

# Add project root
project_root = Path(__file__).resolve().parents[4]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.constants import get_full_path
from src.utils.utils import set_global_seed


DEFAULT_CELL_ID_COL = "cell_id"
INTERNAL_CATEGORY = "internal_coexpression"
# Signatures are named INT_COEXP_{rank:02d}, rank 1 = largest surviving module.
SIGNATURE_ID_FMT = "INT_COEXP_{rank:02d}"


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

def _save_json_atomic(path: str, payload: dict) -> None:
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp_path, path)


def _save_parquet_atomic(path: str, df: pd.DataFrame) -> None:
    tmp_path = path + ".tmp"
    df.to_parquet(tmp_path, index=False)
    os.replace(tmp_path, path)


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Cell-id handling + dataset loaders
# ---------------------------------------------------------------------------

def _xenium_str_ids_to_int64(cell_id_strs: pd.Series) -> Optional[np.ndarray]:
    """Vectorized conversion of Xenium Explorer string cell IDs ('aaaachba-1')
    into int64 IDs. Returns None if the format does not match."""
    if cell_id_strs.empty:
        return np.empty((0,), dtype=np.int64)

    s = cell_id_strs.astype(str).to_numpy()
    b = s.astype("S")

    if not np.all(np.char.endswith(b, b"-1")):
        return None

    maxlen = b.dtype.itemsize
    raw = b.tobytes()
    chars = np.frombuffer(raw, dtype=np.uint8).reshape(b.shape[0], maxlen)[:, :8]
    digits = chars.astype(np.int16) - ord("a")

    if np.any(digits < 0) or np.any(digits > 15):
        return None

    powers = np.array([16**i for i in range(7, -1, -1)], dtype=np.int64)
    out = (digits.astype(np.int64) * powers).sum(axis=1) + 1
    return out


def _load_cell_ids_sets_npz(cell_ids_sets_path: str) -> Dict[str, np.ndarray]:
    """Load cell_ids_sets.npz and return the TRAIN-only per-slide dict."""
    if not os.path.isfile(cell_ids_sets_path):
        raise FileNotFoundError(
            f"[ERROR] cell_ids_sets file not found: '{cell_ids_sets_path}'."
        )
    npz = np.load(cell_ids_sets_path, allow_pickle=True)
    name = "train_only_by_slide"
    if name not in npz.files:
        raise KeyError(
            f"[ERROR] '{name}' key not found in cell_ids_sets npz: "
            f"'{cell_ids_sets_path}'. Available keys: {list(npz.files)}."
        )
    obj = npz[name]
    try:
        return obj.item()
    except Exception:
        return dict(obj)


def _standardize_cell_ids_for_isin(
    obs_cell_ids: pd.Series,
    target_cell_ids: np.ndarray,
) -> Tuple[pd.Series, np.ndarray]:
    """Make obs_cell_ids and target_cell_ids comparable for membership testing."""
    tgt_arr = np.asarray(target_cell_ids)

    obs_num = pd.to_numeric(obs_cell_ids, errors="coerce")
    if not obs_num.isna().any():
        return obs_num.astype(np.int64), tgt_arr.astype(np.int64, copy=False)

    if tgt_arr.dtype.kind in ("i", "u"):
        obs_int = _xenium_str_ids_to_int64(obs_cell_ids)
        if obs_int is not None:
            return (
                pd.Series(obs_int, index=obs_cell_ids.index, dtype=np.int64),
                tgt_arr.astype(np.int64, copy=False),
            )

    return obs_cell_ids.astype(str), tgt_arr.astype(str)


def _subset_adata_to_cell_ids(
    adata: AnnData,
    cell_ids: np.ndarray,
    cell_id_col: str = DEFAULT_CELL_ID_COL,
) -> AnnData:
    """Return adata restricted to obs[cell_id_col] in cell_ids."""
    if cell_id_col not in adata.obs.columns:
        raise KeyError(
            f"[ERROR] '{cell_id_col}' not found in adata.obs columns: "
            f"{list(adata.obs.columns)}"
        )
    if cell_ids is None or len(cell_ids) == 0:
        return adata[:0].copy()

    obs_std, tgt_std = _standardize_cell_ids_for_isin(adata.obs[cell_id_col], cell_ids)
    mask = obs_std.isin(tgt_std).to_numpy()
    return adata[mask].copy()


def _load_slide_ids(training_folder: str) -> List[str]:
    slide_ids_path = os.path.join(training_folder, "slide_ids.txt")
    if not os.path.isfile(slide_ids_path):
        raise FileNotFoundError(f"[ERROR] slide_ids.txt not found: '{slide_ids_path}'")
    with open(slide_ids_path, "r") as f:
        slide_ids = [line.strip() for line in f if line.strip()]
    if len(slide_ids) == 0:
        raise ValueError(f"[ERROR] No slide IDs found in '{slide_ids_path}'.")
    return slide_ids


def _get_gene_names(adata: AnnData, gene_col: Optional[str]) -> np.ndarray:
    """Return adata.var_names (or adata.var[gene_col]); raises on duplicates."""
    if gene_col is None:
        gene_names = adata.var_names.astype(str).to_numpy()
    else:
        if gene_col not in adata.var.columns:
            raise KeyError(
                f"[ERROR] gene_col='{gene_col}' not found in adata.var.columns: "
                f"{list(adata.var.columns)}"
            )
        gene_names = adata.var[gene_col].astype(str).to_numpy()

    idx = pd.Index(gene_names)
    if idx.has_duplicates:
        duplicated = idx[idx.duplicated()].unique().tolist()
        raise ValueError(
            f"[ERROR] Gene names contain duplicates (showing up to 10): {duplicated[:10]}."
        )
    return gene_names


def _assert_nonnegative_counts(adata: AnnData, slide_id: str) -> None:
    """Sanity-check that adata.X holds raw non-negative integer counts."""
    x_min = adata.X.min()
    if sparse.issparse(adata.X):
        x_min = float(x_min)
    if x_min < 0:
        raise ValueError(
            f"[ERROR] Negative values found in adata.X for slide '{slide_id}'. "
            "Expected raw non-negative counts."
        )
    if sparse.issparse(adata.X):
        data = np.asarray(adata.X.data)
    else:
        data = np.asarray(adata.X).ravel()
    if data.size > 0 and not np.allclose(data, np.round(data), atol=1e-6):
        raise ValueError(
            f"[ERROR] adata.X for slide '{slide_id}' contains non-integer values; "
            "this script expects raw counts (it then applies normalize_total + log1p)."
        )


def _load_g_base(
    training_dataset_id: str,
    G_base_id: str,
) -> Tuple[List[str], str, str, str]:
    """Read the input G_base; return (sorted_genes, scgpt_model_name, g_base_dir,
    g_base_genes_sha256)."""
    g_base_dir = get_full_path(
        "eval", "eval_datasets", "st_eval_data",
        f"eval_st_{training_dataset_id}", "G_base", G_base_id, as_str=True,
    )
    genes_path = os.path.join(g_base_dir, "G_base_genes.json")
    if not os.path.isfile(genes_path):
        raise FileNotFoundError(
            f"[ERROR] G_base_genes.json not found at '{genes_path}'. "
            f"Did you run _11_Gbase.py for --G_base_id '{G_base_id}'?"
        )
    with open(genes_path, "r") as f:
        genes_payload = json.load(f)
    if "genes" not in genes_payload or not isinstance(genes_payload["genes"], list):
        raise ValueError(
            f"[ERROR] '{genes_path}' is missing a list under the 'genes' key."
        )
    raw_genes = [str(g) for g in genes_payload["genes"]]
    if len(raw_genes) == 0:
        raise ValueError(f"[ERROR] G_base '{G_base_id}' contains zero genes.")
    sorted_genes = sorted(set(raw_genes))

    config_path = os.path.join(g_base_dir, "config.json")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(
            f"[ERROR] G_base config.json not found at '{config_path}'."
        )
    with open(config_path, "r") as f:
        g_base_config = json.load(f)
    scgpt_model_name = g_base_config.get("scGPT_model_name")
    if not isinstance(scgpt_model_name, str) or not scgpt_model_name:
        raise ValueError(
            f"[ERROR] G_base config.json '{config_path}' is missing a non-empty "
            f"'scGPT_model_name' field."
        )
    return sorted_genes, scgpt_model_name, g_base_dir, _sha256_file(genes_path)


# ---------------------------------------------------------------------------
# Co-expression consensus
# ---------------------------------------------------------------------------

def _per_slide_gene_correlation(
    X_gb: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Pearson gene x gene correlation over the cells of one slide (X_gb: n_cells x n_genes, log1p, G_base genes).

    Returns (R, valid_mask): R is NaN on the rows/cols of zero-variance genes; valid_mask marks the other genes.
    """
    stds = X_gb.std(axis=0)
    valid = np.isfinite(stds) & (stds > 0.0)
    n_genes = X_gb.shape[1]
    R = np.full((n_genes, n_genes), np.nan, dtype=np.float64)
    if int(valid.sum()) < 2:
        return R, valid
    Xv = X_gb[:, valid].astype(np.float64)
    Rv = np.corrcoef(Xv, rowvar=False)
    idx = np.where(valid)[0]
    R[np.ix_(idx, idx)] = Rv
    return R, valid


def _consensus_correlation(
    per_slide: List[Tuple[np.ndarray, np.ndarray, int]],
    n_genes: int,
    *,
    min_gene_slides: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cell-weighted Fisher-z average of per-slide correlation matrices.

    per_slide: list of (gene_global_idx, R_slide, n_cells), R_slide restricted to the slide's G_base genes.
    Returns:
      consensus_r      : (n_genes, n_genes) consensus correlation (diagonal 1; pairs never co-measured -> 0).
      gene_slide_count : per-gene number of slides with finite variance.
      kept_mask        : genes with finite variance on >= min_gene_slides slides.
    """
    Zsum = np.zeros((n_genes, n_genes), dtype=np.float64)
    Wsum = np.zeros((n_genes, n_genes), dtype=np.float64)
    gene_slide_count = np.zeros(n_genes, dtype=np.int64)

    for gene_idx, R_slide, n_cells in per_slide:
        w = float(max(n_cells - 3, 1))
        finite = np.isfinite(R_slide)
        # Exclude the diagonal (arctanh(1) = inf); set to 1 below
        np.fill_diagonal(finite, False)
        if not finite.any():
            # Slide still marks its valid genes as "present" for the count.
            valid_genes = np.unique(gene_idx[np.isfinite(np.diag(R_slide))])
            gene_slide_count[valid_genes] += 1
            continue
        gi = gene_idx
        block_finite = finite
        # Fisher z only on finite off-diagonal entries (NaN pairs stay 0).
        z = np.zeros_like(R_slide, dtype=np.float64)
        np.arctanh(
            np.clip(R_slide, -0.999999, 0.999999), out=z, where=block_finite,
        )
        # Scatter-add the finite z*w and w into the global accumulators.
        zw = np.where(block_finite, z * w, 0.0)
        ww = np.where(block_finite, w, 0.0)
        Zsum[np.ix_(gi, gi)] += zw
        Wsum[np.ix_(gi, gi)] += ww
        # A gene is "present" on this slide if it had finite variance (finite diag).
        valid_genes = gi[np.isfinite(np.diag(R_slide))]
        gene_slide_count[valid_genes] += 1

    with np.errstate(invalid="ignore", divide="ignore"):
        consensus_z = np.where(Wsum > 0.0, Zsum / Wsum, 0.0)
    consensus_r = np.tanh(consensus_z)
    np.fill_diagonal(consensus_r, 1.0)

    kept_mask = gene_slide_count >= int(min_gene_slides)
    return consensus_r, gene_slide_count, kept_mask


# ---------------------------------------------------------------------------
# Module discovery + eigengene weights
# ---------------------------------------------------------------------------

def _agglomerative_labels(distance: np.ndarray, n_clusters: int) -> np.ndarray:
    """Average-linkage agglomerative clustering on a precomputed distance matrix (sklearn metric=, or affinity= before 1.2)."""
    from sklearn.cluster import AgglomerativeClustering

    try:
        model = AgglomerativeClustering(
            n_clusters=n_clusters, metric="precomputed", linkage="average",
        )
    except TypeError:
        model = AgglomerativeClustering(
            n_clusters=n_clusters, affinity="precomputed", linkage="average",
        )
    return model.fit_predict(distance)


def _eigengene_weights(consensus_sub: np.ndarray) -> np.ndarray:
    """Module-eigengene weights (WGCNA kME-style): leading eigenvector of the module's consensus-correlation
    submatrix, oriented so that its entries sum positive, L2-normalized."""
    # Symmetrize for numerical safety before eigh.
    C = 0.5 * (consensus_sub + consensus_sub.T)
    eigvals, eigvecs = np.linalg.eigh(C)
    v = eigvecs[:, -1]  # eigenvector of the largest eigenvalue
    if float(np.sum(v)) < 0.0:
        v = -v
    norm = float(np.linalg.norm(v))
    if norm <= 0.0:
        # Degenerate: fall back to flat unit weights.
        return np.ones(consensus_sub.shape[0], dtype=np.float64)
    return (v / norm).astype(np.float64)


def _discover_modules(
    consensus_r: np.ndarray,
    genes: List[str],
    kept_mask: np.ndarray,
    *,
    n_modules: int,
    min_module_size: int,
    membership_floor: float,
    weight_scheme: str,
) -> Tuple[List[Dict[str, object]], List[str]]:
    """Cluster the kept genes into modules and assign their weights.

    Returns (modules, grey_genes): modules as {genes, weights} dicts, largest first; grey_genes not in any module.
    """
    genes_arr = np.asarray(genes, dtype=object)
    kept_idx = np.where(kept_mask)[0]
    grey: List[str] = [str(g) for g in genes_arr[~kept_mask]]

    if kept_idx.size < max(2, int(min_module_size)):
        return [], sorted(grey + [str(g) for g in genes_arr[kept_idx]])

    sub_r = consensus_r[np.ix_(kept_idx, kept_idx)]
    distance = 1.0 - sub_r  # signed: anti-correlated genes are far apart
    np.fill_diagonal(distance, 0.0)
    distance = np.clip(distance, 0.0, 2.0)
    distance = 0.5 * (distance + distance.T)  # enforce symmetry

    k = int(min(n_modules, kept_idx.size))
    labels = _agglomerative_labels(distance, k)

    raw_modules: List[List[int]] = []
    for lab in np.unique(labels):
        members_local = np.where(labels == lab)[0]
        raw_modules.append(members_local.tolist())

    surviving: List[Dict[str, object]] = []
    for members_local in raw_modules:
        if len(members_local) < int(min_module_size):
            grey.extend(str(genes_arr[kept_idx[m]]) for m in members_local)
            continue
        local = np.asarray(members_local, dtype=int)
        if weight_scheme == "unit":
            weights = np.ones(local.size, dtype=np.float64)
        else:
            weights = _eigengene_weights(sub_r[np.ix_(local, local)])
        # Membership floor: drop genes whose signed loading is below the floor.
        keep = weights >= float(membership_floor)
        if weight_scheme == "unit":
            keep = np.ones(local.size, dtype=bool)
        if int(keep.sum()) < int(min_module_size):
            grey.extend(str(genes_arr[kept_idx[m]]) for m in local)
            continue
        local = local[keep]
        weights = weights[keep]
        module_genes = [str(genes_arr[kept_idx[m]]) for m in local]
        surviving.append({"genes": module_genes, "weights": weights.tolist()})

    # Deterministic order: largest module first, ties broken by first gene alpha.
    surviving.sort(key=lambda m: (-len(m["genes"]), sorted(m["genes"])[0]))
    return surviving, sorted(set(grey))


def _build_int_table(
    modules: List[Dict[str, object]],
) -> Tuple[pd.DataFrame, Dict[str, dict]]:
    """Build the long (signature_id, gene, weight) table + per-module summary.

    Within a module, genes are ordered by descending |weight| (hub genes first),
    ties broken alphabetically."""
    rows: List[Dict[str, object]] = []
    summary: Dict[str, dict] = {}
    for rank, mod in enumerate(modules, start=1):
        sig_id = SIGNATURE_ID_FMT.format(rank=rank)
        genes = list(mod["genes"])
        weights = [float(w) for w in mod["weights"]]
        order = sorted(
            range(len(genes)), key=lambda i: (-abs(weights[i]), genes[i])
        )
        for i in order:
            rows.append(
                {"signature_id": sig_id, "gene": genes[i], "weight": weights[i]}
            )
        summary[sig_id] = {
            "category": INTERNAL_CATEGORY,
            "n_genes_total": len(genes),
            "module_rank": rank,
        }
    table = pd.DataFrame(rows, columns=["signature_id", "gene", "weight"])
    return table, summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    if args.n_modules < 2:
        raise ValueError("[ERROR] --n_modules must be >= 2.")
    if args.min_module_size < 3:
        raise ValueError(
            "[ERROR] --min_module_size must be >= 3."
        )
    if args.min_gene_slides < 1:
        raise ValueError("[ERROR] --min_gene_slides must be >= 1.")
    if args.weight_scheme not in ("eigengene_kme", "unit"):
        raise ValueError(
            f"[ERROR] --weight_scheme must be 'eigengene_kme' or 'unit', got "
            f"'{args.weight_scheme}'."
        )

    set_global_seed(int(args.seed))

    adata_folder = get_full_path("STHELAR_HistoST", "adata_graph", as_str=True)
    training_folder = get_full_path(
        "train", "training_datasets", args.dataset_id, args.training_dataset_id,
        as_str=True,
    )
    output_folder = get_full_path(
        "eval", "eval_datasets", "eval_signatures_INT",
        args.eval_signatures_INT_id, as_str=True,
    )

    main_output = os.path.join(output_folder, "eval_signatures_INT_table.parquet")
    if os.path.exists(main_output) and not args.overwrite:
        mtime = _dt.datetime.fromtimestamp(
            os.path.getmtime(main_output)
        ).isoformat(timespec="seconds")
        raise FileExistsError(
            f"[ERROR] eval_signatures_INT_table.parquet already exists at "
            f"'{main_output}' (mtime={mtime}). Pass --overwrite to replace it, "
            f"or choose a new --eval_signatures_INT_id."
        )

    os.makedirs(output_folder, exist_ok=True)

    g_base_genes, g_base_scgpt_model_name, g_base_dir, g_base_sha256 = _load_g_base(
        args.training_dataset_id, args.G_base_id,
    )
    g_base_index = {g: i for i, g in enumerate(g_base_genes)}
    n_genes = len(g_base_genes)

    cell_ids_sets_path = os.path.join(training_folder, "cell_ids_sets.npz")
    slide_ids = _load_slide_ids(training_folder)
    train_only_by_slide = _load_cell_ids_sets_npz(cell_ids_sets_path)

    logger.info("dataset_id             = %s", args.dataset_id)
    logger.info("training_dataset_id    = %s", args.training_dataset_id)
    logger.info("G_base_id              = %s", args.G_base_id)
    logger.info("eval_signatures_INT_id = %s", args.eval_signatures_INT_id)
    logger.info("#G_base genes          = %d", n_genes)
    logger.info("n_modules              = %d", args.n_modules)
    logger.info("min_module_size        = %d", args.min_module_size)
    logger.info("weight_scheme          = %s", args.weight_scheme)
    logger.info("output_folder          = %s", output_folder)

    per_slide: List[Tuple[np.ndarray, np.ndarray, int]] = []
    used_slide_ids: List[str] = []
    per_slide_n_cells: Dict[str, int] = {}

    for slide_idx, slide_id in enumerate(slide_ids):
        train_cell_ids = np.asarray(
            train_only_by_slide.get(slide_id, np.array([], dtype=np.int64))
        )
        if train_cell_ids.size == 0:
            logger.info("Slide '%s' has no TRAIN cells. Skipping.", slide_id)
            continue

        adata_path = os.path.join(adata_folder, f"{slide_id}.h5ad")
        if not os.path.isfile(adata_path):
            raise FileNotFoundError(f"[ERROR] Missing AnnData file: '{adata_path}'")

        adata = sc.read_h5ad(adata_path)
        adata = _subset_adata_to_cell_ids(
            adata, train_cell_ids, cell_id_col=args.cell_id_col,
        )
        n_cells_train = int(adata.n_obs)
        if n_cells_train == 0:
            raise ValueError(
                f"[ERROR] Slide '{slide_id}' has 0 cells after filtering to TRAIN "
                f"cell IDs. Check adata.obs['{args.cell_id_col}'] vs cell_ids_sets.npz."
            )
        _assert_nonnegative_counts(adata, slide_id=slide_id)

        # G_base ∩ slide panel.
        gene_names = _get_gene_names(adata, gene_col=args.gene_col)
        g_base_mask = np.fromiter(
            (g in g_base_index for g in gene_names),
            dtype=bool, count=len(gene_names),
        )
        panel_size = int(g_base_mask.sum())
        if panel_size < 2:
            logger.warning(
                "Slide '%s' has %d G_base genes in its panel (<2). Skipping.",
                slide_id, panel_size,
            )
            continue

        # Seeded per-slide cell subsample (generator seeded with (seed, slide index))
        if args.cell_cap > 0 and n_cells_train > args.cell_cap:
            rng = np.random.default_rng([int(args.seed), slide_idx])
            sel = np.sort(
                rng.choice(n_cells_train, size=int(args.cell_cap), replace=False)
            )
            adata = adata[sel].copy()

        sc.pp.normalize_total(adata, target_sum=args.target_sum)
        sc.pp.log1p(adata)
        X = adata.X
        if sparse.issparse(X):
            X_gb = np.asarray(X.tocsr()[:, g_base_mask].todense(), dtype=np.float64)
        else:
            X_gb = np.asarray(X, dtype=np.float64)[:, g_base_mask]

        gene_names_gb = gene_names[g_base_mask]
        gene_global_idx = np.fromiter(
            (g_base_index[g] for g in gene_names_gb),
            dtype=np.int64, count=len(gene_names_gb),
        )

        R_slide, valid = _per_slide_gene_correlation(X_gb)
        if int(valid.sum()) < 2:
            logger.warning(
                "Slide '%s' has <2 finite-variance G_base genes. Skipping.", slide_id,
            )
            continue

        per_slide.append((gene_global_idx, R_slide, int(adata.n_obs)))
        used_slide_ids.append(slide_id)
        per_slide_n_cells[slide_id] = int(adata.n_obs)
        logger.info(
            "Slide '%s': train_cells_used=%d | G_base on slide=%d | valid_genes=%d",
            slide_id, int(adata.n_obs), panel_size, int(valid.sum()),
        )

    if len(used_slide_ids) < args.min_gene_slides:
        raise RuntimeError(
            f"[ERROR] Only {len(used_slide_ids)} TRAIN slides contributed, below "
            f"--min_gene_slides={args.min_gene_slides}. Cannot build a consensus "
            f"co-expression matrix."
        )

    consensus_r, gene_slide_count, kept_mask = _consensus_correlation(
        per_slide, n_genes, min_gene_slides=args.min_gene_slides,
    )
    n_kept = int(kept_mask.sum())
    logger.info(
        "Consensus co-expression over %d/%d G_base genes present on >= %d slides.",
        n_kept, n_genes, args.min_gene_slides,
    )

    modules, grey_genes = _discover_modules(
        consensus_r, g_base_genes, kept_mask,
        n_modules=args.n_modules,
        min_module_size=args.min_module_size,
        membership_floor=args.membership_floor,
        weight_scheme=args.weight_scheme,
    )
    if len(modules) == 0:
        raise RuntimeError(
            "[ERROR] No co-expression module survived the min_module_size filter. "
            "Lower --min_module_size or --n_modules, or check the input data."
        )

    table, summary = _build_int_table(modules)

    under_covered = [
        g for g in g_base_genes
        if int(gene_slide_count[g_base_index[g]]) < int(args.min_gene_slides)
    ]

    config_payload = {
        "eval_signatures_INT_id": args.eval_signatures_INT_id,
        "dataset_id": args.dataset_id,
        "training_dataset_id": args.training_dataset_id,
        "G_base_id": args.G_base_id,
        "G_base_dir": g_base_dir,
        "G_base_genes_sha256": g_base_sha256,
        "scGPT_model_name": g_base_scgpt_model_name,
        "adata_folder": adata_folder,
        "training_folder": training_folder,
        "cell_ids_sets_path": cell_ids_sets_path,
        "cell_id_col": args.cell_id_col,
        "gene_col": args.gene_col,
        "n_g_base_genes": n_genes,
        "n_genes_used": n_kept,
        "n_slide_ids_listed": len(slide_ids),
        "n_training_slides_used": len(used_slide_ids),
        "training_slide_ids_used": used_slide_ids,
        "n_cells_used_per_slide": per_slide_n_cells,
        "n_modules_surviving": len(modules),
        "n_genes_in_modules": int(table["gene"].nunique()),
        "n_genes_grey_unassigned": len(grey_genes),
        "grey_unassigned_genes": grey_genes,
        "under_covered_genes": under_covered,
        "n_signatures": len(summary),
        "signatures": summary,
        "parameters": {
            "n_modules": args.n_modules,
            "min_module_size": args.min_module_size,
            "membership_floor": args.membership_floor,
            "weight_scheme": args.weight_scheme,
            "cell_cap": args.cell_cap,
            "target_sum": args.target_sum,
            "min_gene_slides": args.min_gene_slides,
            "seed": args.seed,
        },
        "recipe": {
            "scope": (
                "Data-driven internal panel scoped to (G_base_id, "
                "training_dataset_id): co-expression modules discovered from the "
                "G_base genes on CLIP-TRAIN cells only (the TRAIN cells used by "
                "G_base, disjoint from the held-out STHELAR_TEST / HEST slides)."
            ),
            "coexpression": (
                "Per-slide gene x gene Pearson correlation on TRAIN cells "
                "(normalize_total(target_sum)+log1p), combined into a consensus "
                "via a cell-weighted (n_cells-3) Fisher-z average."
            ),
            "clustering": (
                "Average-linkage agglomerative clustering on the signed distance "
                "1 - r into n_modules clusters. Clusters smaller than "
                "min_module_size, genes present on < min_gene_slides slides, and "
                "(for eigengene_kme) genes with loading < membership_floor fall "
                "into an unassigned grey bucket."
            ),
            "weights": (
                "eigengene_kme: sign-pinned leading eigenvector of the module's "
                "consensus-correlation submatrix (WGCNA module-eigengene loading, "
                "kME-proportional). unit: flat weights (mean-of-z)."
            ),
            "downstream": (
                "Consumed by the Task A.1 signature-target mode as one of the "
                "signature_sets (signature_set='internal'). The aggregate score is "
                "the weighted mean of z-scored member genes (sum w*z / sum|w|), "
                "identical in form to the external panel and to Task B."
            ),
        },
    }

    _save_parquet_atomic(main_output, table)
    _save_json_atomic(os.path.join(output_folder, "config.json"), config_payload)

    logger.info("==================== DONE ====================")
    logger.info("internal-signatures table: %s", main_output)
    logger.info(
        "#modules=%d  #genes_in_modules=%d  #grey=%d  slides_used=%d",
        len(modules), int(table["gene"].nunique()), len(grey_genes),
        len(used_slide_ids),
    )
    for sig_id, meta in summary.items():
        logger.info("  %s : %d genes", sig_id, meta["n_genes_total"])


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Build the internal gene-signature panel (co-expression modules among the G_base genes, "
        "TRAIN cells only) for the Task A.1 signature-target mode."
    )

    parser.add_argument("--dataset_id", type=str,
                        default="ds_ct1_20x_224_o112",
                        help="Dataset ID, e.g. ds_ct1_20x_224_o112.")
    parser.add_argument("--training_dataset_id", type=str,
                        default="ds_ct1_20x_224_o112_1",
                        help="Training dataset ID, e.g. ds_ct1_20x_224_o112_1.")
    parser.add_argument("--G_base_id", type=str, default="G_base_1",
                        help="Identifier of the input G_base panel (built by _11_Gbase.py) whose genes are clustered.")
    parser.add_argument("--eval_signatures_INT_id", type=str,
                        default="eval_signatures_INT_1",
                        help="Identifier of this internal-signatures build.")

    parser.add_argument("--n_modules", type=int, default=20,
                        help="Number of agglomerative clusters (clusters smaller than --min_module_size are discarded).")
    parser.add_argument("--min_module_size", type=int, default=5,
                        help="Minimum number of genes of a signature (smaller clusters are unassigned; minimum 3).")
    parser.add_argument("--membership_floor", type=float, default=0.0,
                        help="Minimum signed eigengene loading of a module gene (weight_scheme=eigengene_kme only).")
    parser.add_argument("--weight_scheme", type=str, default="eigengene_kme",
                        choices=["eigengene_kme", "unit"],
                        help="eigengene_kme: WGCNA module-eigengene loadings "
                        "(sign-pinned leading eigenvector). unit: flat mean-of-z.")
    parser.add_argument("--cell_cap", type=int, default=200000,
                        help="Maximum TRAIN cells per slide for the correlation (seeded subsample; <=0: all cells).")
    parser.add_argument("--min_gene_slides", type=int, default=3,
                        help="Minimum TRAIN slides on which a gene must have finite "
                        "variance to enter the consensus co-expression matrix.")
    parser.add_argument("--target_sum", type=float, default=1e4,
                        help="Target sum for scanpy normalize_total before log1p.")
    parser.add_argument("--cell_id_col", type=str, default=DEFAULT_CELL_ID_COL,
                        help="Column in adata.obs containing cell IDs.")
    parser.add_argument("--gene_col", type=str, default=None,
                        help="Optional column in adata.var with gene names. If "
                        "omitted, adata.var_names are used.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Global seed (set_global_seed) + per-slide subsample "
                        "seed.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite an existing eval_signatures_INT_id folder.")

    args = parser.parse_args()
    main(args)
