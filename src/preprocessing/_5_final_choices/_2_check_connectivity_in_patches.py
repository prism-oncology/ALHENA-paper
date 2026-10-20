"""
QC on the per-patch spatial-transcriptomics graphs that the frozen Novae teacher consumes, for a
given training_dataset_id. Computes per-patch and per-slide connectivity stats and saves.

The LMDB cache is geometry-only (it holds no ST graph), so this QC does not read an adjacency from
it. Instead it reconstructs each patch's graph exactly as
src/dataset/precompute_novae_teacher.py::precompute_slide does — from the slide AnnData Delaunay
graph (`adatas/{slide}.h5ad` obsp["spatial_distances"], symlinked to STHELAR_HistoST/adata_graph),
sliced to the patch's inside-patch cells and pruned to the teacher radius — reusing the shared
helpers in src/dataset/novae_teacher.py so this QC can never drift from the teacher. It therefore
measures the actual graph the Novae teacher sees per patch.

Requirements:
- Runs in the `histost` conda env (importing src.dataset.novae_teacher pulls in torch /
  torch_geometric).
- The optional "per_patch_qc.csv.gz" output is written only when the lines marked "CHOOSE" are
  uncommented; they are commented out by default.
- Run after preprocessing has produced adatas/ (adata_graph), masks_cell_ids_nuclei.zip,
  nucleus_centroids/ and patch_bboxes.parquet for the training_dataset_id (the same inputs
  precompute_novae_teacher.py reads); no LMDB cache is required.

Output files are stored inside the folder <ROOT_HistoST> / train / training_datasets / <dataset_id> / <training_dataset_id> / checking / connectivity_r<edge_distance_threshold> (or connectivity_r<edge_distance_threshold>_<n_min_cells> if --n_min_cells > 0).
Description of the output files:

1. graph_qc_summary.json ------------------------------------------------------------

Description:
Global statistics computed across all graph patches analyzed.

Variables:
- total_patches: Total number of patches analyzed.
- cells_per_patch: Descriptive stats (mean, std, median, percentiles, min, max, n) for number of cells in a patch.
- edges_per_patch: Same as above, but for number of edges.
- avg_degree_per_patch: Average number of edges per cell (connectivity).
- isolated_cell_ratio_per_patch: Fraction of cells in a patch that have no edges.
- edge_density_per_patch: How dense the graph is (edges / possible edges).
- patches_with_no_edges: Number of patches with zero connections.
- patches_flagged: Number of patches marked as bad (based on thresholds).
- iso_threshold: Threshold for the maximum allowed isolated cell ratio to identify bad patches.
- avg_degree_threshold: Threshold for the minimum required average degree to identify bad patches.
- edge_distance_threshold: Prune radius (µm) applied to the Delaunay graph before QC (teacher radius).

2. graph_qc_slides.json ------------------------------------------------------------

Description:
Per-slide breakdown of QC metrics. Each slide groups multiple patches.

For each slide ID:
- n_patches: Number of patches in the slide.
- cells_per_patch: Stats for number of cells per patch.
- edges_per_patch: Stats for number of edges per patch.
- avg_degree_per_patch: Stats for average degree per patch.
- isolated_cell_ratio_per_patch: Stats for isolated cells.
- edge_density_per_patch: Stats for edge density.
- patches_with_no_edges: Number of patches in the slide with no connections.

3. bad_patches.txt ------------------------------------------------------------

Description:
Plain text file listing image names (patches) that were flagged as low quality.

Flagging criteria:
- Zero edges
- Isolated cell ratio > iso_threshold
- Average degree < avg_degree_threshold

4. *.png (Global histograms & scatter plots) ------------------------------------------------------------

Description:
Visualizations showing distribution of QC metrics across all patches.

Files:
- cells_per_patch_global.png
- edges_per_patch_global.png
- avg_degree_per_patch_global.png
- isolated_cell_ratio_global.png
- edge_density_global.png
- cells_vs_edges_global.png (scatter plot of cells vs. edges)

Each plot shows frequency/distribution of a given metric across the full dataset.

5. slide_<slide_id>/*.png (Per-slide plots) ------------------------------------------------------------

Description:
Same as above, but broken down per slide. A subdirectory is created for each slide with:

- cells_<slide_id>.png
- edges_<slide_id>.png
- avg_deg_<slide_id>.png
- iso_ratio_<slide_id>.png
- density_<slide_id>.png
- cells_vs_edges_<slide_id>.png

Only generated if --plots_per_slide is set.

6. per_patch_qc.csv.gz (optional, commented out by default) ------------------------------------------------------------

Description:
Compressed CSV with raw QC values per patch.

Columns:
- patch_id: Patch name
- slide_id: ID of the slide this patch belongs to
- n_cells: Number of cells in the graph
- n_edges: Number of edges (undirected)
- avg_degree: Average number of edges per cell
- isolated_ratio: Fraction of cells with no connections
- edge_density: Edge density (edges / all possible edges)

To enable this file, uncomment the relevant lines in the code marked with "CHOOSE".
"""

import sys
import io
from pathlib import Path
import argparse, gzip, csv, json, os, warnings
from collections import defaultdict
import numpy as np
import pandas as pd
import anndata as ad
import zipfile
import scipy.sparse as sp
from scipy.sparse import load_npz
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add the project root to sys.path so that 'src' can be imported
project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.constants import get_full_path
# Shared per-patch graph-assembly helpers — the same ones the frozen Novae teacher precompute uses
# (src/dataset/precompute_novae_teacher.py), so this QC measures exactly the graph the teacher sees.
from src.dataset.novae_teacher import (
    list_fold_patch_ids,
    read_patch_cell_ids,
    align_mask_cells_to_obs,
    patch_level_coords,
    inside_patch_keep_idx,
)



def _describe(arr: np.ndarray) -> dict:
    if arr.size == 0:          # safety – shouldn't happen
        return {}
    return {
        "mean":   float(arr.mean()),
        "std":    float(arr.std()),
        "median": float(np.median(arr)),
        "p05":    float(np.percentile(arr, 5)),
        "p95":    float(np.percentile(arr, 95)),
        "min":    float(arr.min()),
        "max":    float(arr.max()),
        "n":      int(arr.size),
    }



def _hist(arr, title, out_dir):
    plt.figure(figsize=(6, 4))
    plt.hist(arr, bins="auto")
    plt.title(title.replace("_", " "))
    plt.xlabel(title.replace("_", " "))
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{title}.png"), dpi=200)
    plt.close()



def _scatter(x, y, title, xlabel, ylabel, out_dir):
    plt.figure(figsize=(6, 6))
    plt.scatter(x, y, s=10, alpha=0.5)
    plt.title(title.replace("_", " "))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{title}.png"), dpi=200)
    plt.close()



def _slide_of(patch_id: str) -> str:
    """slide_id for a patch_id '{slide_id}_{suffix}' (slide_id may contain underscores)."""
    return str(patch_id).rsplit("_", 1)[0]



def _enumerate_patch_ids_by_slide(dataset_dir, folds, n_min_cells=0) -> "dict[str, list[str]]":
    """
    Union of the fold patch lists, de-duplicated, grouped by slide (the same per-fold patch universe
    the teacher precompute covers). With ``n_min_cells > 0`` each fold is read from its filtered
    ``cell_count_{fold}_{n_min_cells}.csv`` (via ``list_fold_patch_ids``), so all requested folds are
    filtered consistently. A missing per-fold cell-count CSV is skipped with a warning so the QC
    still runs on datasets lacking that fold/threshold.
    """
    seen = set()
    by_slide = defaultdict(list)
    for fold in folds:
        try:
            fold_pids = list_fold_patch_ids(dataset_dir, fold, n_min_cells=n_min_cells)
        except FileNotFoundError:
            _csv = f"cell_count_{fold}_{n_min_cells}.csv" if n_min_cells and n_min_cells > 0 else f"cell_count_{fold}.csv"
            print(f"[WARN] {_csv} not found under {dataset_dir}; skipping fold '{fold}'.")
            continue
        for pid in fold_pids:
            if pid not in seen:
                seen.add(pid)
                by_slide[_slide_of(pid)].append(pid)
    return by_slide



def _reconstruct_patch_adj(patch_id, obs_cell_id, obsp, centroids, bboxes,
                           mask_zip, mask_names, edge_distance_threshold):
    """
    Rebuild the per-patch ST subgraph exactly as the frozen Novae teacher precompute does
    (precompute_novae_teacher.precompute_slide): mask cells -> AnnData obs order -> inside-patch
    filter -> obsp submatrix -> prune to the teacher radius. Returns a scipy sparse µm-distance
    matrix (symmetric) or None for a degenerate patch (missing mask/bbox, no cells, no obs match,
    or all cells outside the patch) — None is treated downstream exactly like an empty graph.
    """
    mask_fname = f"{patch_id}.npz"
    if mask_fname not in mask_names or patch_id not in bboxes.index:
        return None
    mask_bytes = mask_zip.read(mask_fname)
    cell_ids_mask = read_patch_cell_ids(mask_bytes)
    if cell_ids_mask.size == 0:
        return None
    H, W = load_npz(io.BytesIO(mask_bytes)).shape  # patch pixel size (raw, pre-resize)

    ordered_ids_int, patch_cell_indices, ordered_str_ids = align_mask_cells_to_obs(
        cell_ids_mask, obs_cell_id
    )
    if patch_cell_indices.size == 0:
        return None

    # Nucleus centroids (slide-level px) for the adata-matched cells, in their order. Mirror the
    # frozen teacher precompute (precompute_novae_teacher.precompute_slide) and the training dataset
    # (histost_dataset._get_patch_cell_data): a tolerant reindex that drops any matched cell lacking
    # a nucleus centroid (NaN row) instead of raising, keeping patch_cell_indices in lockstep. If
    # every matched cell is dropped, coord_slide is empty and the patch falls through to the
    # keep.size == 0 branch below (treated as an empty graph -> None), exactly as the teacher caches
    # zeros for such a patch. (centroids is indexed by str cell_id -- see the read in analyse().)
    coord_df = centroids.reindex(list(ordered_str_ids))[["x", "y"]]
    have = coord_df.notna().all(axis=1).to_numpy()
    if not have.all():
        keep_c = np.flatnonzero(have)
        coord_df = coord_df.iloc[keep_c]
        patch_cell_indices = patch_cell_indices[keep_c]
    coord_slide = coord_df.to_numpy()

    coord_patch = patch_level_coords(coord_slide, bboxes.loc[patch_id])
    keep = inside_patch_keep_idx(coord_patch, (H, W))
    if keep.size == 0:
        return None

    node_idx = patch_cell_indices[keep]
    sub_adj = obsp[node_idx][:, node_idx].tocsr().astype(np.float32, copy=False)
    # Prune to the teacher radius (novae's inclusive '<= r' rule). <=0 disables pruning.
    if edge_distance_threshold is not None and edge_distance_threshold > 0 and sub_adj.nnz:
        sub_adj.data[sub_adj.data > float(edge_distance_threshold)] = 0.0
        sub_adj.eliminate_zeros()
    return sub_adj



def analyse(dataset_dir,
            out_dir: str,
            iso_th: float,
            deg_th: float,
            plots_per_slide: int,
            max_patches: int,
            sample_frac: float,
            seed: int,
            edge_distance_threshold: float,
            folds,
            n_min_cells: int = 0):

    rng = np.random.default_rng(seed=seed)

    processed = 0       # patches kept for QC stats (after filter / sampling)
    scanned = 0         # patches actually scanned (for pbar / scan cap)

    os.makedirs(out_dir, exist_ok=True)
    dataset_dir = Path(dataset_dir)

    # Patch universe (union of folds, filtered by n_min_cells when > 0), grouped by slide.
    by_slide = _enumerate_patch_ids_by_slide(dataset_dir, folds, n_min_cells=n_min_cells)
    n_total = sum(len(v) for v in by_slide.values())

    # Progress-bar target must reflect real scans of patches (not kept patches).
    _scan_total = n_total if max_patches is None else min(n_total, max_patches)

    print(
        f"Found {n_total} patches across {len(by_slide)} slide(s) in folds {list(folds)} "
        f"(n_min_cells={n_min_cells}); scan_cap={max_patches}, sample_frac={sample_frac}, "
        f"edge_distance_threshold={edge_distance_threshold}"
    )

    # Slide-invariant inputs (one bounding-box table + one cell-id mask zip for the whole dataset).
    bboxes = pd.read_parquet(
        dataset_dir / "patch_bboxes.parquet",
        columns=["patch_id", "x_min", "y_min", "x_max", "y_max"],
    ).set_index("patch_id")
    mask_zip = zipfile.ZipFile(dataset_dir / "masks_cell_ids_nuclei.zip", "r")
    mask_names = set(mask_zip.namelist())

    # Global accumulators
    cells, edges, avg_deg, iso_ratio, density = [], [], [], [], []
    bad_patches = set()
    # Per-slide accumulators
    slides = defaultdict(lambda: defaultdict(list))
    # # CSV header - CHOOSE
    # csv_path = os.path.join(out_dir, "per_patch_qc.csv.gz")
    # csv_fp = gzip.open(csv_path, "wt", newline="")
    # csv_writer = csv.writer(csv_fp)
    # csv_writer.writerow(["patch_id", "slide_id", "n_cells", "n_edges", "avg_degree", "isolated_ratio", "edge_density"])

    print("Scanning patches (per slide)...")
    stop = False
    try:
        with tqdm(total=_scan_total, unit="patch") as pbar:
            for slide_id in sorted(by_slide.keys()):
                if stop:
                    break

                # Per-slide inputs: the Delaunay graph + nucleus centroids (loaded once per slide).
                adata_path = dataset_dir / "adatas" / f"{slide_id}.h5ad"
                if not adata_path.is_file():
                    print(f"[WARN] AnnData not found for slide '{slide_id}': {adata_path}; skipping slide.")
                    continue
                adata = ad.read_h5ad(adata_path)
                if "cell_id" not in adata.obs:
                    raise KeyError(f"{slide_id}: adata.obs['cell_id'] missing (needed for the mask<->graph join).")
                if "spatial_distances" not in adata.obsp:
                    raise KeyError(f"{slide_id}: adata.obsp['spatial_distances'] missing (the µm Delaunay graph).")
                obs_cell_id = adata.obs["cell_id"].astype(str)
                obsp = adata.obsp["spatial_distances"].tocsr()
                # The parquet is written with index=False (a plain `cell_id` column + x / y), so
                # promote `cell_id` to the index for label lookups -- exactly as the frozen teacher
                # precompute (precompute_novae_teacher.precompute_slide) and the training dataset
                # (histost_dataset._get_slide_centroids) do (`.set_index("cell_id")`).
                centroids = pd.read_parquet(
                    dataset_dir / "nucleus_centroids" / f"nucleus_centroids_{slide_id}.parquet"
                ).set_index("cell_id")

                for patch_id in by_slide[slide_id]:
                    # Hard cap on scanned patches (real traversal), not on kept patches
                    if max_patches is not None and scanned >= max_patches:
                        stop = True
                        break

                    scanned += 1
                    pbar.update(1)  # count every scanned patch, even if later skipped

                    # Optional random subsampling among the (already n_min_cells-filtered) patches.
                    if sample_frac is not None and rng.random() >= sample_frac:
                        continue

                    processed += 1

                    # Reconstruct the exact graph the Novae teacher consumes for this patch.
                    adj = _reconstruct_patch_adj(
                        patch_id, obs_cell_id, obsp, centroids, bboxes,
                        mask_zip, mask_names, edge_distance_threshold,
                    )

                    # Normalize adjacency to CSR float32 and remove self-loops
                    if adj is None:
                        n = m = 0
                        deg_arr = np.empty(0, dtype=np.int32)
                    else:
                        # accept both float64 and float32 caches
                        if not sp.issparse(adj):
                            adj = sp.csr_matrix(adj)
                        else:
                            adj = adj.tocsr()
                        if adj.dtype != np.float32:
                            adj = adj.astype(np.float32, copy=False)
                        # ensure no self loops before counting edges/degrees
                        adj.setdiag(0)
                        adj.eliminate_zeros()

                        n = int(adj.shape[0])
                        # number of undirected edges (matrix is symmetric with both (i,j) and (j,i))
                        m = int(adj.nnz // 2)
                        # degree = number of neighbors (non-zero entries), not sum of weights
                        deg_arr = np.diff(adj.indptr).astype(np.int32, copy=False)

                    mean_deg = float(deg_arr.mean()) if n else 0.0
                    iso = float((deg_arr == 0).sum() / n) if n else 0.0
                    dens = 2.0 * m / (n * (n - 1)) if n > 1 else 0.0   # edge density of a graph = what fraction of all possible cell pairs actually have an edge

                    # Append globals
                    cells.append(n)
                    edges.append(m)
                    avg_deg.append(mean_deg)
                    iso_ratio.append(iso)
                    density.append(dens)

                    # Append per-slide
                    slides[slide_id]["cells"].append(n)
                    slides[slide_id]["edges"].append(m)
                    slides[slide_id]["avg_deg"].append(mean_deg)
                    slides[slide_id]["iso_ratio"].append(iso)
                    slides[slide_id]["density"].append(dens)

                    # QC rules
                    if m == 0 or iso > iso_th or mean_deg < deg_th:
                        bad_patches.add(patch_id)

            #         # CSV row - CHOOSE
            #         csv_writer.writerow((patch_id, slide_id, n, m, mean_deg, iso, dens))

                del adata
    finally:
        mask_zip.close()

    # csv_fp.close()

    # Global summary
    cells_arr = np.asarray(cells)
    edges_arr = np.asarray(edges)
    adeg_arr = np.asarray(avg_deg)
    iso_arr = np.asarray(iso_ratio)
    dens_arr = np.asarray(density)
    summary = {
        "total_patches": int(cells_arr.size),
        "cells_per_patch": _describe(cells_arr),
        "edges_per_patch": _describe(edges_arr),
        "avg_degree_per_patch": _describe(adeg_arr),
        "isolated_cell_ratio_per_patch": _describe(iso_arr),
        "edge_density_per_patch": _describe(dens_arr),
        "patches_with_no_edges": int(np.sum(edges_arr == 0)) if edges_arr.size else 0,
        "patches_flagged": int(len(bad_patches)),
        "iso_threshold": iso_th,
        "avg_degree_threshold": deg_th,
        "edge_distance_threshold": edge_distance_threshold,
    }
    with open(os.path.join(out_dir, "graph_qc_summary.json"), "w") as fp:
        json.dump(summary, fp, indent=2)

    # Per-slide summary
    slide_json = {}
    for sid, dic in slides.items():
        slide_json[sid] = {
            "n_patches": len(dic["cells"]),
            "cells_per_patch": _describe(np.asarray(dic["cells"])),
            "edges_per_patch": _describe(np.asarray(dic["edges"])),
            "avg_degree_per_patch": _describe(np.asarray(dic["avg_deg"])),
            "isolated_cell_ratio_per_patch": _describe(np.asarray(dic["iso_ratio"])),
            "edge_density_per_patch": _describe(np.asarray(dic["density"])),
            "patches_with_no_edges": int(sum(e == 0 for e in dic["edges"])),
        }
    with open(os.path.join(out_dir, "graph_qc_slides.json"), "w") as fp:
        json.dump(slide_json, fp, indent=2)

    # Global plots
    if cells_arr.size:
        _hist(cells_arr, "cells_per_patch_global", out_dir)
        _hist(edges_arr, "edges_per_patch_global", out_dir)
        _hist(adeg_arr, "avg_degree_per_patch_global", out_dir)
        _hist(iso_arr, "isolated_cell_ratio_global", out_dir)
        _hist(dens_arr, "edge_density_global", out_dir)

        MAX_SCATTER_POINTS = 50_000
        if cells_arr.size > MAX_SCATTER_POINTS:
            idx = rng.choice(cells_arr.size, MAX_SCATTER_POINTS, replace=False)
            _scatter(cells_arr[idx], edges_arr[idx], "cells_vs_edges_global", "cells", "edges", out_dir)
        else:
            _scatter(cells_arr, edges_arr,  "cells_vs_edges_global", "cells", "edges", out_dir)

    # Per-slide plots
    if plots_per_slide != 0:
        max_plots = plots_per_slide if plots_per_slide > 0 else float("inf")
        for k, (sid, dic) in enumerate(slides.items()):
            if k >= max_plots:
                break
            arr_c, arr_e = np.asarray(dic["cells"]), np.asarray(dic["edges"])
            arr_d, arr_i = np.asarray(dic["avg_deg"]), np.asarray(dic["iso_ratio"])
            arr_de = np.asarray(dic["density"])
            subdir = os.path.join(out_dir, f"slide_{sid}")
            os.makedirs(subdir, exist_ok=True)
            _hist(arr_c, f"cells_{sid}", subdir)
            _hist(arr_e, f"edges_{sid}", subdir)
            _hist(arr_d, f"avg_deg_{sid}", subdir)
            _hist(arr_i, f"iso_ratio_{sid}", subdir)
            _hist(arr_de, f"density_{sid}", subdir)

            MAX_SCATTER_POINTS = 50_000
            if arr_c.size > MAX_SCATTER_POINTS:
                idx = rng.choice(arr_c.size, MAX_SCATTER_POINTS, replace=False)
                _scatter(arr_c[idx], arr_e[idx], f"cells_vs_edges_{sid}", "cells", "edges", subdir)
            else:
                _scatter(arr_c, arr_e, f"cells_vs_edges_{sid}", "cells", "edges", subdir)

    # Bad patches list
    with open(os.path.join(out_dir, "bad_patches.txt"), "w") as fp:
        for name in sorted(bad_patches):
            fp.write(f"{name}\n")

    print(f"QC finished – scanned {scanned} patches, kept {processed} for QC. Results in {out_dir}")



def main(args):

    # Check args format
    if args.sample_frac is not None and not (0.0 <= args.sample_frac <= 1.0):
        raise ValueError("--sample-frac must be between 0 and 1")
    if args.max_patches is not None and args.max_patches < 0:
        raise ValueError("--max-patches must be >= 0")
    if args.n_min_cells is not None and args.n_min_cells < 0:
        raise ValueError("--n_min_cells must be >= 0")

    folds = [f.strip() for f in args.folds.split(",") if f.strip()]

    # Training-dataset dir holding adatas/ (adata_graph symlink), masks, centroids and bboxes.
    dataset_dir = get_full_path('train', 'training_datasets', args.dataset_id, args.training_dataset_id)

    # Output folder encodes the prune radius and (when > 0) the n_min_cells filter.
    nmin_suffix = f"_{args.n_min_cells}" if args.n_min_cells and args.n_min_cells > 0 else ""
    out_dir = get_full_path('train', 'training_datasets', args.dataset_id, args.training_dataset_id, 'checking', f'connectivity_r{int(args.edge_distance_threshold)}{nmin_suffix}', as_str=True)

    # Analyze
    analyse(dataset_dir=dataset_dir,
            out_dir=out_dir,
            iso_th=args.iso_th,
            deg_th=args.deg_th,
            plots_per_slide=args.plots_per_slide,
            max_patches=args.max_patches,
            sample_frac=args.sample_frac,
            seed=args.seed,
            edge_distance_threshold=args.edge_distance_threshold,
            folds=folds,
            n_min_cells=args.n_min_cells)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute per-patch ST-graph connectivity QC (reconstructed exactly as the frozen Novae teacher precompute does).")

    # Data

    parser.add_argument("--dataset_id", type=str,
                        default="ds_ct1_20x_224_o112",
                        help="Dataset_id, e.g. ds_ct1_20x_224_o112")

    parser.add_argument("--training_dataset_id", type=str,
                        default="ds_ct1_20x_224_o112_1",
                        help="Training_dataset_id, e.g. ds_ct1_20x_224_o112_1")

    parser.add_argument("--folds", type=str,
                        default="train,valid,test",
                        help="Comma-separated folds whose patches to QC (union). A missing cell_count_{fold}.csv is skipped with a warning.")

    # Parameters

    parser.add_argument("--edge_distance_threshold", type=float,
                        default=100.0,
                        help="Prune graph edges to <= this many µm before QC, matching the frozen Novae teacher's radius. <=0 disables pruning (full patch Delaunay).")

    parser.add_argument("--iso_th", type=float,
                        default=0.4,
                        help="Isolation ratio threshold for flagging patches")

    parser.add_argument("--deg_th", type=float,
                        default=2.0,
                        help="Average degree threshold for flagging patches")

    parser.add_argument("--plots_per_slide", type=int,
                        default=-1,
                        help="Number of slides for which to draw individual histograms (0 = none, -1 = all)")

    parser.add_argument("--max-patches", type=int,
                        default=None,
                        help="Hard cap on number of patches to SCAN (default: no cap). Caps in slide order.")

    parser.add_argument("--sample-frac", type=float,
                        default=None,
                        help="Randomly sample this fraction of patches (0–1, default: all)")

    parser.add_argument("--n_min_cells", type=int,
                        default=0,
                        help=("Minimum cells/patch filter applied to EVERY --folds fold (reads each fold's "
                              "cell_count_{fold}_{n_min_cells}.csv, built by _1_filter_patches_by_min_cells.py). "
                              "0 = no filter (full per-fold lists). A fold lacking its filtered CSV is skipped with a warning."))

    parser.add_argument("--seed", type=int,
                        default=0,
                        help="PRNG seed used when --sample-frac is set (default: 0)")

    args = parser.parse_args()
    main(args)
