#!/usr/bin/env python
"""
Build the ST evaluation dataset (HEST-format H&E .h5 and ST .h5ad) of a training_dataset_id: HEST slides are downloaded
from HuggingFace, STHELAR slides are built from the test patches of the training dataset.

Saved in <ROOT_HistoST> / eval / eval_datasets / st_eval_data /
    - hest / patches / <slide_id>.h5, hest / st / <slide_id>.h5ad: HEST slides (shared by all eval datasets)
    - eval_st_<training_dataset_id> / patches / <slide_id>.h5, eval_st_<training_dataset_id> / st / <slide_id>.h5ad: STHELAR slides
      (patch rows <patch_id>_cells and <patch_id>_all, cell rows <slide_id>_cell_<cell_id_int>)
    - eval_st_<training_dataset_id> / cell_token_membership_<modelgroup> / <slide_id>.parquet: cell-to-token membership
    - eval_st_<training_dataset_id> / cell_token_membership_missing_cells_<modelgroup> / <slide_id>.parquet: cells with no token
"""

import sys
import argparse
import io
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import warnings
import h5py
import cv2
import numpy as np
import pandas as pd
from anndata import AnnData, read_h5ad
from huggingface_hub import snapshot_download
from scipy.sparse import issparse, load_npz, csr_matrix, coo_matrix, vstack as sp_vstack
from tqdm import tqdm
import zipfile
from functools import lru_cache
from PIL import Image
import geopandas as gpd
from shapely.geometry import box
import spatialdata as sd
from sopa.utils import get_spatial_image, to_intrinsic
import pyarrow as pa
import pyarrow.parquet as pq
import re

# Add the project root to sys.path so that 'src' can be imported
project_root = Path(__file__).resolve().parents[4]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.constants import get_full_path
from src.utils.models_settings import BACKBONE_TRANSFORM
from src.utils.cell_ids import str_cell_id
from src.utils.sdata_utils import promote_20x_to_default




# -----------------------------
# Configs
# -----------------------------

# Max number of ST transcripts per spatial-join chunk (patch-level all-RNA expression)
SPATIAL_JOIN_CHUNK_SIZE = 200_000

# Membership is needed for these backbones, grouped by geometric preprocess.
MEMBERSHIP_BACKBONES = ("uni2h", "virchow2", "hoptimus0", "hoptimus1")
VIT_TOKEN_PX = 14  # patch14


def _membership_groups_from_backbone_transform():
    """Group the backbones by (resize_size, crop_size); group name = sorted model names joined by '_'."""
    by_sig = {}
    for m in MEMBERSHIP_BACKBONES:
        if m not in BACKBONE_TRANSFORM:
            raise KeyError(f"BACKBONE_TRANSFORM missing backbone '{m}' needed for membership grouping.")
        cfg = BACKBONE_TRANSFORM[m]
        sig = (int(cfg.get("resize_size", 0)), int(cfg.get("crop_size", 0)))
        by_sig.setdefault(sig, []).append(m)

    groups = []
    for (rs, cs), models in by_sig.items():
        models_sorted = sorted(models)
        group_name = "_".join(models_sorted)
        groups.append(
            {
                "name": group_name,
                "models": models_sorted,
                "resize_size": int(rs),
                "crop_size": int(cs),
            }
        )
    # Deterministic order
    groups = sorted(groups, key=lambda d: d["name"])
    return groups


_EMPTY_MEMBERSHIP_SCHEMA = pa.schema(
    [
        pa.field("cell_id_int", pa.int32()),
        pa.field("cell_id", pa.string()),
        pa.field("patch_id", pa.string()),
        pa.field("token_index", pa.int32()),
        pa.field("nuc_px", pa.int32()),
    ]
)

_EMPTY_MISSING_MEMBERSHIP_SCHEMA = pa.schema(
    [
        pa.field("cell_id_int", pa.int32()),
        pa.field("cell_id", pa.string()),
    ]
)



def _resize_mask_opencv_nearest(mask: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Resize an integer cell-ID mask with OpenCV nearest-neighbor (as A.Resize(..., mask_interpolation=cv2.INTER_NEAREST) in training)."""
    in_h, in_w = int(mask.shape[0]), int(mask.shape[1])
    out_h, out_w = int(out_h), int(out_w)
    if in_h == out_h and in_w == out_w:
        return mask

    out = cv2.resize(mask, dsize=(out_w, out_h), interpolation=cv2.INTER_NEAREST)
    if out.ndim != 2:
        raise ValueError(f"Expected 2D resized mask, got shape={out.shape}")
    return out.astype(mask.dtype, copy=False)



def _center_crop_mask(mask: np.ndarray, crop_size: int) -> np.ndarray:
    crop_size = int(crop_size)
    h, w = int(mask.shape[0]), int(mask.shape[1])
    if crop_size <= 0 or (h == crop_size and w == crop_size):
        return mask
    if crop_size > h or crop_size > w:
        # Crop larger than the mask: mask returned unchanged
        return mask
    top = max((h - crop_size) // 2, 0)
    left = max((w - crop_size) // 2, 0)
    return mask[top : top + crop_size, left : left + crop_size]



def apply_membership_preprocess_to_mask(mask: np.ndarray, resize_size: int, crop_size: int) -> np.ndarray:
    """
    Apply the same geometric preprocessing as the H&E training pipeline:
      A.Resize(..., mask_interpolation=cv2.INTER_NEAREST) -> A.CenterCrop(...)
    on integer cell-ID masks.
    """
    m = mask
    rs = int(resize_size)
    cs = int(crop_size)
    if rs > 0:
        if int(m.shape[0]) != rs or int(m.shape[1]) != rs:
            m = _resize_mask_opencv_nearest(m, rs, rs)
    if cs > 0:
        if int(m.shape[0]) != cs or int(m.shape[1]) != cs:
            m = _center_crop_mask(m, cs)
    return m



# -----------------------------
# Helpers
# -----------------------------



def locate_cid2ct_parquet(cid2ct_dir: Path, slide_id: str) -> Path:
    """
    Locate the per-slide cID2cT parquet named cID2cT_{slide_id}.parquet
    """
    cid2ct_dir = Path(cid2ct_dir)
    p = cid2ct_dir / f"cID2cT_{slide_id}.parquet"
    if p.exists():
        return p
    raise FileNotFoundError(f"Could not find cID2cT parquet for slide '{slide_id}' under: {cid2ct_dir}.")



@lru_cache(maxsize=32)
def load_cid2ct_map(parquet_path_str: str) -> Dict[str, str]:
    """
    Load a mapping {cell_id_str -> cell_type} from a cID2cT parquet.
    """
    parquet_path = Path(parquet_path_str)
    table = pq.read_table(str(parquet_path), columns=["cell_id", "cell_type"])
    df = table.to_pandas()
    df["cell_id"] = df["cell_id"].astype(str)
    df["cell_type"] = df["cell_type"].astype(str)
    if df["cell_id"].duplicated().any():
        raise ValueError(f"Duplicate cell_id values found in cID2cT parquet '{parquet_path}'.")
    return dict(zip(df["cell_id"].values, df["cell_type"].values))



def download_hest_for_slides(slide_ids: Sequence[str], local_dir: Path) -> None:
    """Download the HEST patches/<slide_id>.h5 and st/<slide_id>.h5ad files of the given slides to local_dir (snapshot_download)."""
    if not slide_ids:
        print("No missing HEST slides to download.")
        return

    print(f"Downloading HEST patches + ST (.h5ad) for {len(slide_ids)} slide(s) into {local_dir} ...")

    # Restrict download to only `patches/` and `st/` subdirectories for the given slide IDs
    patterns: List[str] = []
    for sid in slide_ids:
        # ST expression (AnnData)
        patterns.append(f"st/{sid}.h5ad")
        # Image patches
        patterns.append(f"patches/{sid}.h5")

    repo_id = "MahmoodLab/hest"
    local_dir = Path(local_dir)

    snapshot_download(
        repo_id=repo_id,
        allow_patterns=patterns,
        repo_type="dataset",
        local_dir=str(local_dir),
    )

    missing_after_download = [
        sid
        for sid in slide_ids
        if not (local_dir / "patches" / f"{sid}.h5").exists()
        or not (local_dir / "st" / f"{sid}.h5ad").exists()
    ]
    if missing_after_download:
        raise FileNotFoundError(
            "HuggingFace download completed, but some required HEST files are still missing: "
            f"{missing_after_download}"
        )



def load_patch_bboxes(parquet_path: Path) -> pd.DataFrame:
    """
    Load patch_bboxes.parquet and index by raw patch_id.
    """
    bboxes = pd.read_parquet(
        parquet_path,
        columns=["slide_id", "patch_id", "x_min", "y_min", "x_max", "y_max"],
    )
    bboxes.set_index("patch_id", inplace=True)
    return bboxes



def load_cell_count(cell_count_csv: Path) -> pd.DataFrame:
    """
    Load cell_count_test.csv and add a 'slide_id' column derived from 'patch_id'.
    """
    df = pd.read_csv(cell_count_csv)
    if "patch_id" not in df.columns:
        raise ValueError("cell_count_test.csv must contain a 'patch_id' column.")

    # Derive slide_id from patch_id: {slide_id}_{number}
    slide_ids = df["patch_id"].str.rsplit("_", n=1).str[0]
    df["slide_id"] = slide_ids
    return df




def load_patch_image_from_zip(img_zip: zipfile.ZipFile, img_name: str) -> np.ndarray:
    """
    Load an image patch from images.zip into a uint8 numpy array (H, W, 3).
    """
    img_data = img_zip.read(img_name)
    img = Image.open(io.BytesIO(img_data)).convert("RGB")
    return np.array(img, dtype=np.uint8)



def load_mask_from_zip(mask_zip: zipfile.ZipFile, mask_name: str) -> np.ndarray:
    """
    Load the nuclei cell-id mask (.npz) from masks_cell_ids_nuclei.zip.
    """
    mask_data = mask_zip.read(mask_name)
    mask_cell_ids = load_npz(io.BytesIO(mask_data))
    mask_cell_ids = mask_cell_ids.toarray().astype(np.int32)
    return mask_cell_ids




def _iter_nonzero_runs_1d(row: np.ndarray):
    """Yield the contiguous non-zero runs of a 1D integer mask row as (cell_id_int, x_start, x_end_exclusive)."""
    row = np.asarray(row)
    if row.ndim != 1:
        raise ValueError(f"Expected a 1D row, got shape={row.shape}")
    if row.size == 0:
        return

    change = np.flatnonzero(row[1:] != row[:-1]) + 1
    starts = np.concatenate(([0], change))
    ends = np.concatenate((change, [row.shape[0]]))

    for s, e in zip(starts, ends):
        cid = int(row[s])
        if cid > 0:
            yield cid, int(s), int(e)



def compute_slide_cell_total_nucleus_px(
    slide_id: str,
    bboxes: pd.DataFrame,
    mask_zip: zipfile.ZipFile,
) -> Dict[int, int]:
    """
    Total unique nucleus area (pixels) of each cell on a slide, from all the slide patch masks (not only test patches),
    with overlapping patches deduplicated in slide coordinates. Returns {cell_id_int: n_pixels}.
    """
    slide_boxes = bboxes[bboxes["slide_id"] == slide_id][["x_min", "y_min"]].copy()
    if slide_boxes.empty:
        raise ValueError(f"No patch bounding boxes found for slide '{slide_id}'.")

    slide_boxes = slide_boxes.sort_values(["y_min", "x_min"])
    slide_patch_ids = slide_boxes.index.tolist()

    # Keyed by (cell_id_int, global_y) -> list[(global_x_start, global_x_end)]
    intervals_by_row: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}

    for patch_id in tqdm(slide_patch_ids, desc="  0. Compute exact slide-level nucleus areas..."):
        mask_name = f"{patch_id}.npz"
        try:
            mask = load_mask_from_zip(mask_zip, mask_name)
        except KeyError as e:
            raise KeyError(f"Mask '{mask_name}' not found while computing slide-level nucleus areas for slide '{slide_id}'.") from e

        bbox = bboxes.loc[patch_id]
        x0 = int(bbox["x_min"])
        y0 = int(bbox["y_min"])

        for local_y in range(mask.shape[0]):
            row = mask[local_y]
            if not np.any(row):
                continue
            global_y = y0 + local_y
            for cid, xs, xe in _iter_nonzero_runs_1d(row):
                key = (cid, global_y)
                intervals_by_row.setdefault(key, []).append((x0 + xs, x0 + xe))

    total_nuc_px: Dict[int, int] = {}
    for (cid, _global_y), intervals in intervals_by_row.items():
        intervals.sort()
        cur_s, cur_e = intervals[0]
        merged_len = 0
        for s, e in intervals[1:]:
            if s <= cur_e:
                if e > cur_e:
                    cur_e = e
            else:
                merged_len += cur_e - cur_s
                cur_s, cur_e = s, e
        merged_len += cur_e - cur_s
        total_nuc_px[cid] = total_nuc_px.get(cid, 0) + int(merged_len)

    return total_nuc_px



def aggregate_patch_expression_weighted_by_nucleus_fraction(
    slide_adata: AnnData,
    cell_id_to_idx: Dict[str, int],
    cell_ids_int: np.ndarray,
    slide_cell_total_nucleus_px: Dict[int, int],
) -> csr_matrix:
    """
    Patch expression (1 x n_genes CSR row): sum of the whole-cell RNA counts of the cells of the patch nucleus mask,
    weighted by nucleus_pixels_of_cell_inside_patch / total_unique_nucleus_pixels_of_cell_on_slide.
    """
    n_genes = slide_adata.shape[1]

    unique_ids, patch_nuc_px = np.unique(cell_ids_int, return_counts=True)
    keep = unique_ids > 0
    unique_ids = unique_ids[keep].astype(np.int64, copy=False)
    patch_nuc_px = patch_nuc_px[keep].astype(np.float32, copy=False)

    if unique_ids.size == 0:
        return csr_matrix((1, n_genes), dtype=np.float32)

    total_nuc_px = np.fromiter(
        (slide_cell_total_nucleus_px.get(int(cid), 0) for cid in unique_ids),
        dtype=np.float32,
        count=unique_ids.size,
    )
    bad = total_nuc_px <= 0
    if np.any(bad):
        bad_ids = unique_ids[bad][:10].tolist()
        raise ValueError(
            "Some cells present in a patch have no valid total unique nucleus area on the slide "
            f"(examples: {bad_ids})."
        )

    weights = patch_nuc_px / total_nuc_px

    str_ids = [str_cell_id(int(cid)) for cid in unique_ids.tolist()]
    missing = [sid for sid in str_ids if sid not in cell_id_to_idx]
    if missing:
        raise KeyError(
            "Some cell ids from the nuclei mask are not present in slide_adata.obs['cell_id'] "
            f"(examples: {missing[:10]})."
        )
    indices = [cell_id_to_idx[sid] for sid in str_ids]

    X = slide_adata.X[indices]
    if issparse(X):
        X = X.tocsr()
        vec = X.multiply(weights[:, None]).sum(axis=0)
        if not issparse(vec):
            vec = csr_matrix(np.asarray(vec, dtype=np.float32))
        else:
            vec = vec.tocsr()
        vec = vec.astype(np.float32)
        vec.eliminate_zeros()
        return vec
    else:
        X = np.asarray(X, dtype=np.float32)
        vec = (X * weights[:, None]).sum(axis=0, dtype=np.float32)
        return csr_matrix(vec.reshape(1, -1))



@lru_cache(maxsize=32)
def _token_ids_flat_lut(H: int, W: int, tokens_per_dim: int, token_px: int) -> np.ndarray:
    """
    Precompute token id per pixel (row-major) for a given patch shape and token grid.
    Returned array is flat (H*W,) int32.
    """
    if H != tokens_per_dim * token_px or W != tokens_per_dim * token_px:
        raise ValueError(
            f"Token grid mismatch: got HxW={H}x{W} but tokens_per_dim*token_px="
            f"{tokens_per_dim}*{token_px}={tokens_per_dim*token_px}."
        )
    ty = (np.arange(H, dtype=np.int32) // token_px)[:, None]               # (H,1)
    tx = (np.arange(W, dtype=np.int32) // token_px)[None, :]               # (1,W)
    token_ids = (ty * tokens_per_dim + tx).astype(np.int32)                # (H,W)
    return token_ids.ravel()



def compute_patch_expression_all_rna(
    slide_id: str,
    patch_ids: Sequence[str],
    bboxes: pd.DataFrame,
    slide_adata: AnnData,
    spatialdata_dir: Path,
    he_resolution: int,
) -> csr_matrix:
    """
    Patch-level expression summing all RNAs (cells and background) in each patch, from the SpatialData ST points.
    Returns:
        expr_all: (n_patches, n_genes) CSR sparse float32, aligned with patch_ids order and slide_adata.var
    """
    if spatialdata_dir is None:
        raise ValueError("spatialdata_dir must be provided to compute all-RNA patch expression.")

    sdata_path = spatialdata_dir / f"sdata_{slide_id}.zarr"
    if not sdata_path.exists():
        raise FileNotFoundError(f"SpatialData file not found: {sdata_path}")

    # Load SpatialData with images + points
    sdata = sd.read_zarr(sdata_path, selection=("images", "points"))

    # Set correct H&E resolution as default
    he = get_spatial_image(sdata, "he")
    if he_resolution == 20:
        promote_20x_to_default(sdata, key="he")
        he = get_spatial_image(sdata, "he")
    elif he_resolution != 40:
        raise ValueError("Unsupported H&E resolution. Choose 20 or 40.")

    if "st" not in sdata.points:
        raise KeyError(f"'st' points not found in SpatialData for slide '{slide_id}'.")

    # Project ST points into intrinsic pixel coordinates of H&E
    st_df = to_intrinsic(sdata, sdata.points["st"], he)

    # to_intrinsic may return a Dask DataFrame: convert to pandas
    if hasattr(st_df, "compute"):
        # Dask DataFrame -> pandas
        st_df = st_df.compute()
    else:
        # Make sure this is a plain pandas DataFrame
        st_df = pd.DataFrame(st_df)

    # Clean index so boolean masks and numpy indexing are safe
    st_df = st_df.reset_index(drop=True)

    n_patches = len(patch_ids)
    n_genes = slide_adata.shape[1]
    # Build sparse triplets incrementally (chunked) to avoid dense (n_patches x n_genes)
    rows_chunks: List[np.ndarray] = []
    cols_chunks: List[np.ndarray] = []
    data_chunks: List[np.ndarray] = []

    # Nothing to do if there are no transcripts at all
    if len(st_df) == 0:
        return csr_matrix((n_patches, n_genes), dtype=np.float32)

    # Gene names per transcript (work on plain DataFrame first – no geometry yet)
    gene_col = "feature_name"
    if gene_col not in st_df.columns:
        raise KeyError(f"Column '{gene_col}' not found in ST points for slide '{slide_id}'.")

    # Keep only the required columns, to save memory
    st_df = st_df[["x", "y", gene_col]].copy()

    # Map gene names to var indices once and drop unknown genes
    gene_cat = pd.Categorical(st_df[gene_col].astype(str), categories=slide_adata.var_names)
    gene_idx_all = gene_cat.codes.astype(np.int32)
    valid_gene = gene_idx_all >= 0
    if not np.any(valid_gene):
        # All transcripts are for genes not present in slide_adata.var_names
        return csr_matrix((n_patches, n_genes), dtype=np.float32)

    st_df = st_df.loc[valid_gene].copy()
    st_df["gene_idx"] = gene_idx_all[valid_gene]

    # Get bounding boxes for exactly the requested patches, in patch_ids order
    try:
        bboxes_patches = bboxes.loc[list(patch_ids)]
    except KeyError as e:
        missing = sorted(set(patch_ids) - set(bboxes.index))
        raise KeyError(f"The following patch names were not found in bboxes for slide '{slide_id}': {missing}") from e

    x_min = bboxes_patches["x_min"].to_numpy(dtype=np.float32)
    y_min = bboxes_patches["y_min"].to_numpy(dtype=np.float32)
    x_max = bboxes_patches["x_max"].to_numpy(dtype=np.float32)
    y_max = bboxes_patches["y_max"].to_numpy(dtype=np.float32)

    # Keep only transcripts in the global patch bounding box (before building geometries)
    global_xmin = float(x_min.min())
    global_xmax = float(x_max.max())
    global_ymin = float(y_min.min())
    global_ymax = float(y_max.max())

    xs_all = st_df["x"].to_numpy(dtype=np.float32)
    ys_all = st_df["y"].to_numpy(dtype=np.float32)
    in_global_bbox = (
        (xs_all >= global_xmin)
        & (xs_all < global_xmax)
        & (ys_all >= global_ymin)
        & (ys_all < global_ymax)
    )

    if not np.any(in_global_bbox):
        # No transcript falls inside any patch bounding box
        return csr_matrix((n_patches, n_genes), dtype=np.float32)

    st_df = st_df.loc[in_global_bbox].copy()

    # Build patch polygons as a GeoDataFrame, with a fixed integer index per patch
    patch_polys = [
        box(float(x0), float(y0), float(x1), float(y1))
        for x0, y0, x1, y1 in zip(x_min, y_min, x_max, y_max)
    ]

    patches_gdf = gpd.GeoDataFrame(
        {
            "patch_index": np.arange(n_patches, dtype=np.int32),
            "patch_id": list(patch_ids),
            "x_min": x_min,
            "y_min": y_min,
            "x_max": x_max,
            "y_max": y_max,
        },
        geometry=patch_polys,
    )

    # Prepare arrays for chunked processing
    xs = st_df["x"].to_numpy(dtype=np.float32)
    ys = st_df["y"].to_numpy(dtype=np.float32)
    gene_idx = st_df["gene_idx"].to_numpy(dtype=np.int32)

    weight_col = "__weight"
    n_points = gene_idx.shape[0]

    # Process transcripts in chunks to avoid huge GeoDataFrames and sjoins
    for start in tqdm(range(0, n_points, SPATIAL_JOIN_CHUNK_SIZE), desc="  2. Compute patch expression using whole RNA..."):
        end = min(start + SPATIAL_JOIN_CHUNK_SIZE, n_points)
        n_chunk = end - start
        if n_chunk <= 0:
            continue

        chunk_gene_idx = gene_idx[start:end]
        chunk_x = xs[start:end]
        chunk_y = ys[start:end]

        # Chunk GeoDataFrame: only minimal columns + geometry
        points_chunk = gpd.GeoDataFrame(
            {
                "gene_idx": chunk_gene_idx,
                weight_col: np.ones(n_chunk, dtype=np.float32),
                "x": chunk_x,
                "y": chunk_y,
            },
            geometry=gpd.points_from_xy(chunk_x, chunk_y),
        )

        # Spatial join: assign each transcript in the chunk to a patch
        joined = gpd.sjoin(
            points_chunk[["gene_idx", weight_col, "x", "y", "geometry"]],
            patches_gdf[["patch_index", "x_min", "y_min", "x_max", "y_max", "geometry"]],
            how="inner",
            predicate="intersects",
        )

        if joined.empty:
            continue

        # Enforce the intended half-open patch bbox convention:
        #   x_min <= x < x_max, y_min <= y < y_max
        inside = (
            (joined["x"].to_numpy(dtype=np.float32) >= joined["x_min"].to_numpy(dtype=np.float32))
            & (joined["x"].to_numpy(dtype=np.float32) < joined["x_max"].to_numpy(dtype=np.float32))
            & (joined["y"].to_numpy(dtype=np.float32) >= joined["y_min"].to_numpy(dtype=np.float32))
            & (joined["y"].to_numpy(dtype=np.float32) < joined["y_max"].to_numpy(dtype=np.float32))
        )
        if not np.any(inside):
            continue
        joined = joined.loc[inside, ["patch_index", "gene_idx", weight_col]]

        patch_idx_join = joined["patch_index"].to_numpy(dtype=np.int32)
        gene_idx_join = joined["gene_idx"].to_numpy(dtype=np.int32)
        weights_join = joined[weight_col].to_numpy(dtype=np.float32)

        # Aggregate duplicates inside the chunk without building dense matrices
        # Combine (patch, gene) into a single sortable key
        key = patch_idx_join.astype(np.int64) * np.int64(n_genes) + gene_idx_join.astype(np.int64)
        order = np.argsort(key)
        key = key[order]
        w = weights_join[order]

        uniq_key, first_idx = np.unique(key, return_index=True)
        summed = np.add.reduceat(w, first_idx).astype(np.float32)

        rows = (uniq_key // n_genes).astype(np.int32)
        cols = (uniq_key % n_genes).astype(np.int32)

        rows_chunks.append(rows)
        cols_chunks.append(cols)
        data_chunks.append(summed)

    if not rows_chunks:
        return csr_matrix((n_patches, n_genes), dtype=np.float32)

    rows_all = np.concatenate(rows_chunks, axis=0)
    cols_all = np.concatenate(cols_chunks, axis=0)
    data_all = np.concatenate(data_chunks, axis=0)

    expr_all = coo_matrix((data_all, (rows_all, cols_all)), shape=(n_patches, n_genes), dtype=np.float32).tocsr()
    expr_all.sum_duplicates()
    return expr_all




def build_sthelar_slide_data(
    slide_id: str,
    patch_ids: Sequence[str],
    bboxes: pd.DataFrame,
    cell_count_df: pd.DataFrame,
    img_zip: zipfile.ZipFile,
    mask_zip: zipfile.ZipFile,
    adata_dir: Path,
    he_resolution: int,
    spatialdata_dir: Optional[Path] = None,
    cid2ct_dir: Optional[Path] = None,
    cell_token_membership_out_map: Optional[Dict[str, Path]] = None,
    cell_token_membership_missing_out_map: Optional[Dict[str, Path]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, AnnData]:
    """
    HEST-format arrays and ST AnnData of a STHELAR slide (rows <patch_id>_cells, <patch_id>_all if spatialdata_dir
    is given, <slide_id>_cell_<cell_id_int>); also writes the cell-to-token membership files if their paths are given.

    Returns:
        barcodes_arr: (n_patches, 1) patch IDs
        coords_arr:   (n_patches, 2) int coordinates (x_min, y_min)
        imgs_arr:     (n_patches, H, W, 3) uint8 RGB images
        adata_patches: AnnData of the patch and cell rows (var of the slide-level AnnData)
    """
    patch_ids = list(patch_ids)  # ensure indexable & stable order
    if not patch_ids:
        raise ValueError(f"No patch IDs provided for slide {slide_id}.")

    print(f"  Loading slide-level AnnData for '{slide_id}' ...")
    slide_adata_path = adata_dir / f"{slide_id}.h5ad"
    if not slide_adata_path.exists():
        raise FileNotFoundError(f"Could not find slide AnnData: {slide_adata_path}")
    slide_adata = read_h5ad(slide_adata_path)

    # Build cell_id -> row index mapping
    if "cell_id" not in slide_adata.obs.columns:
        raise KeyError(f"AnnData for slide {slide_id} must have 'cell_id' in .obs.")
    cell_id_series = slide_adata.obs["cell_id"].astype(str)
    if cell_id_series.duplicated().any():
        dup = cell_id_series[cell_id_series.duplicated()].unique()[:5].tolist()
        raise ValueError(f"AnnData for slide '{slide_id}' has duplicated cell_id values (examples: {dup}).")
    cell_id_to_idx = {cid: i for i, cid in enumerate(cell_id_series.values)}

    # Compute exact slide-level total nucleus area per cell using all patch masks
    # for this slide (not only eval/test patches), deduplicated in slide coordinates
    slide_cell_total_nucleus_px = compute_slide_cell_total_nucleus_px(
        slide_id=slide_id,
        bboxes=bboxes,
        mask_zip=mask_zip,
    )
    if len(slide_cell_total_nucleus_px) == 0:
        warnings.warn(f"No nucleus pixels were found across all patch masks for slide '{slide_id}'. Patch-level *_cells rows will be empty.")

    # Prepare cell_count for this slide
    cc_slide = cell_count_df[cell_count_df["slide_id"] == slide_id].copy()
    if cc_slide.empty:
        raise ValueError(f"No rows in cell_count_test.csv for slide '{slide_id}'.")

    # Index by patch_id to align rows with patch_ids
    cc_slide.set_index("patch_id", inplace=True)
    if not cc_slide.index.is_unique:
        dup = cc_slide.index[cc_slide.index.duplicated()].unique().tolist()
        raise ValueError(f"cell_count_test.csv has duplicate patch_id entries for slide '{slide_id}'. Examples: {dup[:5]}.")

    # Determine cell-type columns (all except 'slide_set_scope' if present)
    cc_cols = [c for c in cc_slide.columns if c not in ("slide_id", "slide_set_scope")]
    print(f"    Detected cell type columns in cell_count_test.csv: {cc_cols}")

    # Prepare arrays
    n_patches = len(patch_ids)
    # H and W are determined from the first image
    first_img = load_patch_image_from_zip(img_zip, f'{patch_ids[0]}.png')
    H, W, C = first_img.shape
    print(f"    Detected patch image size: {H}x{W} pixels.")
    if C != 3:
        raise ValueError(f"Expected RGB images with 3 channels, got {C}.")

    # Membership writers per preprocess group (optional)
    # Membership is computed in model input space: resize + center-crop, then patch14 tokenization
    membership_writers: Dict[str, pq.ParquetWriter] = {}
    membership_written_any: Dict[str, bool] = {}
    membership_token_lut: Dict[str, Tuple[np.ndarray, int]] = {}  # group -> (token_ids_flat, tokens_per_patch)
    membership_cells_present: Dict[str, set[int]] = {}            # group -> cells with >=1 surviving token anywhere on slide
    membership_enabled = (cell_token_membership_out_map is not None) and (len(cell_token_membership_out_map) > 0)

    if membership_enabled:
        # No-op loop (the token LUTs are built by the loop below)
        for gname, out_path in cell_token_membership_out_map.items():
            pass

        # Token LUT per group at the final (post-preprocess) resolution; resize/crop read from the group's first model
        for gname, out_path in cell_token_membership_out_map.items():
            models = gname.split("_")
            if len(models) == 0:
                raise ValueError(f"Invalid membership group name '{gname}'.")
            m0 = models[0]
            if m0 not in BACKBONE_TRANSFORM:
                raise KeyError(f"Cannot infer transform for membership group '{gname}': unknown backbone '{m0}'.")
            rs = int(BACKBONE_TRANSFORM[m0].get("resize_size", 0))
            cs = int(BACKBONE_TRANSFORM[m0].get("crop_size", 0))
            # Final resolution after preprocess (assume square)
            final_hw = cs if cs > 0 else (rs if rs > 0 else H)
            final_hw = int(final_hw)

            if (final_hw % VIT_TOKEN_PX != 0) or (final_hw <= 0):
                warnings.warn(f"Membership disabled for group '{gname}': final_hw={final_hw} not compatible with patch{VIT_TOKEN_PX}.")
                continue
            tokens_per_dim = final_hw // VIT_TOKEN_PX
            tokens_per_patch = tokens_per_dim * tokens_per_dim
            token_ids_flat = _token_ids_flat_lut(final_hw, final_hw, tokens_per_dim, VIT_TOKEN_PX)
            membership_token_lut[gname] = (token_ids_flat, tokens_per_patch)
            membership_written_any[gname] = False
            membership_cells_present[gname] = set()

        if len(membership_token_lut) == 0:
            warnings.warn("No valid membership groups had a compatible final token grid; membership will not be written.")
            membership_enabled = False

    imgs_arr = np.empty((n_patches, H, W, 3), dtype=np.uint8)
    coords_arr = np.empty((n_patches, 2), dtype=np.int32)

    # Load cell_id str -> cell_type mapping (for cell rows one-hot)
    if cid2ct_dir is None:
        raise ValueError("cid2ct_dir must be provided for STHELAR to build cell-level rows.")
    cid2ct_path = locate_cid2ct_parquet(Path(cid2ct_dir), slide_id=slide_id)
    cid2ct_map = load_cid2ct_map(str(cid2ct_path))

    n_genes = slide_adata.shape[1]
    # Store expression rows sparsely to avoid huge dense allocations
    expr_patch_cells_rows: List[csr_matrix] = []
    barcodes_list: List[str] = []  # for .h5 (patch-level only, no suffix)
    # base patch obs (per patch, shared between *_cells, *_all rows and tokens)
    base_obs_rows: List[pd.Series] = []
    # cell-level containers
    cell_ids_seen: set[int] = set()
    cell_patch_count: Dict[int, int] = {}
    # membership handled via membership_writers dict per preprocess group

    print(f"  Building data for {n_patches} patch(es) of slide '{slide_id}':")
    
    for i, patch_id in enumerate(tqdm(patch_ids, desc=f"  1. Compute patch expression using cells + Compute cell ID membership...")):

        # Stored barcode = patch_id
        barcodes_list.append(patch_id)

        # ----------------- Patch-level metadata -----------------

        # Coords from bboxes
        if patch_id not in bboxes.index:
            raise KeyError(f"Patch '{patch_id}' not found in patch_bboxes.parquet for slide '{slide_id}'.")
        row = bboxes.loc[patch_id]
        coords_arr[i, 0] = int(row["x_min"])
        coords_arr[i, 1] = int(row["y_min"])

        # Image
        img_name = f"{patch_id}.png"
        img = load_patch_image_from_zip(img_zip, img_name)
        if img.shape != (H, W, 3):
            raise ValueError(f"Image '{img_name}' has shape {img.shape}, expected {(H, W, 3)}.")
        imgs_arr[i] = img

        # Cell counts for obs (patch-level)
        if patch_id not in cc_slide.index:
            raise KeyError(f"Patch '{patch_id}' not found in cell_count csv for slide '{slide_id}'.")
        else:
            base_obs = cc_slide.loc[patch_id, cc_cols]
            base_obs.name = patch_id

        base_obs = base_obs.copy()
        base_obs["is_cell"] = False
        base_obs["patch_id"] = patch_id
        base_obs_rows.append(base_obs)

        # ----------------- Patch-level expression using cells -----------------

        # Expression from masks + slide_adata
        mask_name = f"{patch_id}.npz"
        mask = None
        try:
            mask = load_mask_from_zip(mask_zip, mask_name)
        except KeyError:
            raise KeyError(f"Mask '{mask_name}' not found in masks_cell_ids_nuclei.zip for slide '{slide_id}'.")

        expr_vec = aggregate_patch_expression_weighted_by_nucleus_fraction(
            slide_adata=slide_adata,
            cell_id_to_idx=cell_id_to_idx,
            cell_ids_int=mask,
            slide_cell_total_nucleus_px=slide_cell_total_nucleus_px,
        )
        expr_patch_cells_rows.append(expr_vec)

        # ----------------- Collect cell IDs and write membership (cell <-> patch <-> token) -----------------
        # Unique cell ids in this patch (ignore background 0)
        u = np.unique(mask)
        u = u[u > 0]
        if u.size > 0:
            # update union + per-cell patch count (for quick summary in cell obs)
            for cid in u.astype(np.int64, copy=False):
                ic = int(cid)
                cell_ids_seen.add(ic)
                cell_patch_count[ic] = cell_patch_count.get(ic, 0) + 1

        # Stream cell-to-token membership per preprocess group (in model input space).
        # The exact same resize/crop geometry is applied to the nuclei mask (nearest), then token_index is computed at the final resolution.
        if membership_enabled and (u.size > 0):
            for gname, (token_ids_flat, tokens_per_patch) in membership_token_lut.items():
                # Infer resize/crop for this group from its first listed model
                m0 = gname.split("_")[0]
                rs = int(BACKBONE_TRANSFORM[m0].get("resize_size", 0))
                cs = int(BACKBONE_TRANSFORM[m0].get("crop_size", 0))

                mask_t = apply_membership_preprocess_to_mask(mask, resize_size=rs, crop_size=cs)
                if mask_t.size == 0:
                    continue

                # The mask must match the token LUT size (final_hw x final_hw)
                fh, fw = int(mask_t.shape[0]), int(mask_t.shape[1])
                expected_hw = int(np.sqrt(token_ids_flat.shape[0]))
                if fh != expected_hw or fw != expected_hw:
                    raise RuntimeError(
                        f"Membership mask shape mismatch for group '{gname}' on patch '{patch_id}': "
                        f"got {fh}x{fw}, expected {expected_hw}x{expected_hw}."
                    )

                mask_flat = mask_t.ravel()
                valid = mask_flat > 0
                if not np.any(valid):
                    continue

                cell_flat = mask_flat[valid].astype(np.int64, copy=False)
                tok_flat = token_ids_flat[valid].astype(np.int64, copy=False)

                key = cell_flat * np.int64(tokens_per_patch) + tok_flat
                uniq_key, counts = np.unique(key, return_counts=True)

                cell_ids = (uniq_key // tokens_per_patch).astype(np.int32, copy=False)
                tok_ids = (uniq_key % tokens_per_patch).astype(np.int32, copy=False)
                nuc_px = counts.astype(np.int32, copy=False)

                # Track which ground-truth cells still have at least one surviving token
                # anywhere on this slide after this modelgroup's preprocess
                membership_cells_present[gname].update(cell_ids.tolist())

                cell_id_str = np.fromiter(
                    (str_cell_id(int(c)) for c in cell_ids),
                    dtype=object,
                    count=cell_ids.shape[0],
                )
                chunk = pd.DataFrame(
                    {
                        "cell_id_int": cell_ids,
                        "cell_id": cell_id_str,
                        "patch_id": np.repeat(patch_id, cell_ids.shape[0]).astype(object),
                        "token_index": tok_ids,
                        "nuc_px": nuc_px,
                    }
                )

                table = pa.Table.from_pandas(chunk, preserve_index=False, schema=_EMPTY_MEMBERSHIP_SCHEMA)

                if gname not in membership_writers:
                    out_path = cell_token_membership_out_map[gname]
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    membership_writers[gname] = pq.ParquetWriter(
                        str(out_path),
                        _EMPTY_MEMBERSHIP_SCHEMA,
                        compression="zstd",
                        use_dictionary=True,
                    )
                membership_writers[gname].write_table(table)
                membership_written_any[gname] = True

    # ----------------- Compute patch-level all-RNA expression (if SpatialData is available) -----------------
    expr_all: Optional[csr_matrix] = None
    if spatialdata_dir is not None:
        expr_all = compute_patch_expression_all_rna(
            slide_id=slide_id,
            patch_ids=patch_ids,
            bboxes=bboxes,
            slide_adata=slide_adata,
            spatialdata_dir=spatialdata_dir,
            he_resolution=he_resolution,
        )
    else:
        print("    spatialdata_dir not provided, skipping all-RNA patch expression.")

    # Close membership writers and ensure empty files exist
    if membership_enabled:
        for gname, out_path in (cell_token_membership_out_map or {}).items():
            w = membership_writers.get(gname, None)
            if w is not None:
                w.close()
            else:
                # Write an empty parquet for consistency (no membership rows after preprocess)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                empty = pa.Table.from_arrays(
                    [
                        pa.array([], type=pa.int32()),
                        pa.array([], type=pa.string()),
                        pa.array([], type=pa.string()),
                        pa.array([], type=pa.int32()),
                        pa.array([], type=pa.int32()),
                    ],
                    schema=_EMPTY_MEMBERSHIP_SCHEMA,
                )
                pq.write_table(empty, str(out_path), compression="zstd")

    # Write, for each modelgroup, the ground-truth eval cells that never got any
    # surviving token support anywhere on the slide after preprocess
    if cell_token_membership_missing_out_map is not None:
        if cell_ids_seen:
            gt_cell_ids = np.array(sorted(cell_ids_seen), dtype=np.int32)
        else:
            gt_cell_ids = np.empty((0,), dtype=np.int32)

        for gname, out_path in cell_token_membership_missing_out_map.items():
            out_path.parent.mkdir(parents=True, exist_ok=True)

            present_set = membership_cells_present.get(gname, None)
            if present_set is None:
                # Group was not active/valid for membership computation; keep an empty file for consistency
                missing_ids = np.empty((0,), dtype=np.int32)
            elif gt_cell_ids.size == 0:
                missing_ids = np.empty((0,), dtype=np.int32)
            else:
                present_ids = np.fromiter(present_set, dtype=np.int32, count=len(present_set))
                if present_ids.size > 1:
                    present_ids.sort()
                missing_ids = np.setdiff1d(gt_cell_ids, present_ids, assume_unique=True)

            if missing_ids.size == 0:
                empty = pa.Table.from_arrays(
                    [
                        pa.array([], type=pa.int32()),
                        pa.array([], type=pa.string()),
                    ],
                    schema=_EMPTY_MISSING_MEMBERSHIP_SCHEMA,
                )
                pq.write_table(empty, str(out_path), compression="zstd")
            else:
                missing_cell_id_str = np.fromiter(
                    (str_cell_id(int(c)) for c in missing_ids.tolist()),
                    dtype=object,
                    count=missing_ids.shape[0],
                )
                missing_df = pd.DataFrame(
                    {
                        "cell_id_int": missing_ids.astype(np.int32, copy=False),
                        "cell_id": missing_cell_id_str,
                    }
                )
                missing_table = pa.Table.from_pandas(
                    missing_df,
                    preserve_index=False,
                    schema=_EMPTY_MISSING_MEMBERSHIP_SCHEMA,
                )
                pq.write_table(missing_table, str(out_path), compression="zstd")

                warnings.warn(
                    f"Slide '{slide_id}', membership group '{gname}': "
                    f"{missing_ids.size} ground-truth cell ID(s) have no surviving token support "
                    f"across all patches after resize/crop/tokenization. "
                    "Be careful when comparing encoders from different modelgroups. "
                    f"Examples: {missing_df['cell_id'].head(10).tolist()}"
                )
    
    # ----------------- Build final outputs -----------------

    print(f"  3. Combining into final outputs + Add unique cell expression...")

    # Prepare barcode array for .h5 (n_patches, 1), stays patch-level only
    barcodes_arr = np.array(barcodes_list, dtype=object).reshape(-1, 1)  # patch_id

    # Build sparse X and aligned obs
    X_parts: List[csr_matrix] = []
    obs_parts: List[pd.DataFrame] = []

    # Patch-level (cells-only)
    X_cells = sp_vstack(expr_patch_cells_rows, format="csr").astype(np.float32)
    obs_patch = pd.DataFrame(base_obs_rows)
    obs_patch_cells = obs_patch.copy()
    obs_patch_cells.index = [f"{pid}_cells" for pid in patch_ids]
    obs_patch_cells.index.name = None
    X_parts.append(X_cells)
    obs_parts.append(obs_patch_cells)

    # Patch-level (all-RNA)
    if expr_all is not None:
        if expr_all.shape != (n_patches, n_genes):
            raise ValueError(f"expr_all has shape {expr_all.shape}, expected {(n_patches, n_genes)}")
        obs_patch_all = obs_patch.copy()
        obs_patch_all.index = [f"{pid}_all" for pid in patch_ids]
        obs_patch_all.index.name = None
        X_parts.append(expr_all.astype(np.float32))
        obs_parts.append(obs_patch_all)

    # Cell-level rows (one per unique cell_id_int in the slide that appears in any patch mask)
    if cell_ids_seen:
        cell_ids_sorted = np.array(sorted(cell_ids_seen), dtype=np.int32)
        cell_ids_str = [str_cell_id(int(c)) for c in cell_ids_sorted.tolist()]

        # Map to slide_adata row indices
        missing = [sid for sid in cell_ids_str if sid not in cell_id_to_idx]
        if missing:
            raise KeyError(f"Some cell ids from masks are not present in slide AnnData .obs['cell_id'] (examples: {missing[:10]}).")
        cell_rows = np.fromiter((cell_id_to_idx[sid] for sid in cell_ids_str), dtype=np.int32, count=len(cell_ids_str))

        X_cell = slide_adata.X[cell_rows]
        if issparse(X_cell):
            X_cell = X_cell.tocsr().astype(np.float32, copy=False)
            X_cell.sum_duplicates()
            X_cell.eliminate_zeros()
        else:
            X_cell = csr_matrix(np.asarray(X_cell, dtype=np.float32))

        # Build cell obs: same cell-type columns as patch obs (cc_cols) but one-hot
        cell_obs_index = [f"{slide_id}_cell_{int(c)}" for c in cell_ids_sorted.tolist()]
        obs_cell = pd.DataFrame(0, index=cell_obs_index, columns=cc_cols, dtype=np.int32)
        obs_cell.index.name = None
        obs_cell["is_cell"] = True
        obs_cell["patch_id"] = None
        obs_cell["cell_id_int"] = cell_ids_sorted.astype(np.int32, copy=False)
        obs_cell["cell_id"] = pd.Series(cell_ids_str, index=cell_obs_index, dtype="object")
        obs_cell["n_patches"] = np.fromiter(
            (cell_patch_count.get(int(c), 0) for c in cell_ids_sorted.tolist()),
            dtype=np.int32,
            count=cell_ids_sorted.shape[0],
        )

        # Assign cell type (from cID2cT) and one-hot encode into the same cc_cols
        ct = [cid2ct_map.get(s) for s in cell_ids_str]
        ct_arr = np.asarray(ct, dtype=object)
        obs_cell["cell_type"] = ct_arr

        unknown_mask = pd.isna(ct_arr)
        if np.any(unknown_mask):
            missing_ids = np.asarray(cell_ids_str, dtype=object)[unknown_mask][:10].tolist()
            raise ValueError(f"Some cell ids have no cell type in cID2cT mapping (examples: {missing_ids}).")

        # One-hot in cc_cols
        for t in np.unique(ct_arr):
            if t not in cc_cols:
                raise ValueError(f"Cell type '{t}' from cID2cT mapping is not present in cell_count columns (cc_cols).")
            obs_cell.loc[ct_arr == t, t] = 1

        X_parts.append(X_cell)
        obs_parts.append(obs_cell)

    X = sp_vstack(X_parts, format="csr").astype(np.float32)
    obs_df = pd.concat(obs_parts, axis=0)

    # Use the same var as slide-level AnnData
    var_df = slide_adata.var.copy()

    adata_patches = AnnData(X=X, obs=obs_df, var=var_df)

    return barcodes_arr, coords_arr, imgs_arr, adata_patches




def write_hest_h5(
    dest_h5: Path,
    barcodes_arr: np.ndarray,
    coords_arr: np.ndarray,
    imgs_arr: np.ndarray,
) -> None:
    """
    Write HEST-format .h5 file with 'barcode', 'coords', 'img' datasets.
    """
    dest_h5.parent.mkdir(parents=True, exist_ok=True)
    n_patches = barcodes_arr.shape[0]

    if barcodes_arr.shape != (n_patches, 1):
        raise ValueError(f"barcodes_arr must have shape ({n_patches}, 1), got {barcodes_arr.shape}.")
    if coords_arr.shape[0] != n_patches:
        raise ValueError(f"coords_arr must have {n_patches} rows, got {coords_arr.shape[0]}.")
    if imgs_arr.shape[0] != n_patches:
        raise ValueError(f"imgs_arr must have {n_patches} rows, got {imgs_arr.shape[0]}.")

    print(f"  Writing HEST-style .h5 to {dest_h5}")
    with h5py.File(dest_h5, "w") as f:
        str_dt = h5py.string_dtype(encoding="utf-8")
        # Resizable datasets (patches can be appended)
        f.create_dataset(
            "barcode",
            data=barcodes_arr,
            dtype=str_dt,
            maxshape=(None, 1),
            chunks=(max(1, min(1024, barcodes_arr.shape[0])), 1),
        )
        f.create_dataset(
            "coords",
            data=coords_arr.astype(np.int32),
            dtype="int32",
            maxshape=(None, 2),
            chunks=(max(1, min(1024, coords_arr.shape[0])), 2),
        )
        # Compressed images
        H, W, C = imgs_arr.shape[1], imgs_arr.shape[2], imgs_arr.shape[3]
        f.create_dataset(
            "img",
            data=imgs_arr.astype(np.uint8),
            dtype="uint8",
            compression="gzip",
            compression_opts=4,
            shuffle=True,               # improves gzip ratio for uint8 images
            chunks=(1, H, W, C),         # patch-wise access without reading the whole tensor
            maxshape=(None, H, W, C),
        )




def write_hest_h5ad(dest_h5ad: Path, adata: AnnData) -> None:
    """
    Write the ST AnnData to .h5ad (patch rows and, for STHELAR, optional cell rows).
    """
    dest_h5ad.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Writing HEST-style .h5ad to {dest_h5ad}")
    adata.write_h5ad(dest_h5ad)




# -----------------------------
# Main processing logic
# -----------------------------


def process_hest_slides(
    metadata: pd.DataFrame,
    patches_dir: Path,
    st_dir: Path,
) -> None:
    """
    Handle HEST slides: download any missing HEST data from HuggingFace.
    """
    if "source" not in metadata.columns or "slide_id" not in metadata.columns:
        raise KeyError("metadata CSV must contain at least 'slide_id' and 'source' columns.")

    hest_meta = metadata[metadata["source"] == "HEST"]
    if hest_meta.empty:
        print("No HEST slides found in metadata.")
        return

    slide_ids = sorted(hest_meta["slide_id"].astype(str).unique())
    missing_slides: List[str] = []
    for sid in slide_ids:
        h5_path = patches_dir / f"{sid}.h5"
        h5ad_path = st_dir / f"{sid}.h5ad"
        if h5_path.exists() and h5ad_path.exists():
            print(f"HEST slide '{sid}': already downloaded (both .h5 and .h5ad present).")
        else:
            print(f"HEST slide '{sid}': missing data, will download.")
            missing_slides.append(sid)

    if missing_slides:
        download_hest_for_slides(missing_slides, local_dir=patches_dir.parent)


def compute_patch_list_for_slide(
    slide_id: str,
    cell_count_df: pd.DataFrame,
) -> List[str]:
    """
    Compute list of patch IDs to use for a STHELAR slide using cell_count_test.csv for that slide.
    """
    cc_slide = cell_count_df[cell_count_df["slide_id"] == slide_id]
    if cc_slide.empty:
        print(f"WARNING: No entries in cell_count_test.csv for slide '{slide_id}'.")
        return []

    available = sorted(cc_slide["patch_id"].astype(str).unique())
    return available



def get_patch_ids_from_h5(dest_h5: Path) -> List[str]:
    """
    Read only the barcode dataset from an existing HEST-format .h5 (avoid loading images).
    Returns a list of stored patch IDs as strings (for STHELAR, without the ".png" suffix).
    """
    with h5py.File(dest_h5, "r") as f:
        bc = f["barcode"][:].reshape(-1)
    out: List[str] = []
    for x in bc:
        if isinstance(x, (bytes, np.bytes_)):
            out.append(bytes(x).decode("utf-8"))
        else:
            out.append(str(x))
    return out



def process_sthelar_slides(
    metadata: pd.DataFrame,
    patches_dir: Path,
    st_dir: Path,
    training_dataset_dir: Path,
    he_resolution: int,
    spatialdata_dir: Optional[Path] = None,
) -> None:
    """
    Handle STHELAR slides: build HEST-format .h5 (H&E) and .h5ad (ST) from the training_dataset_id folder.
    Only patches coming from the test set (i.e. cell_count_test.csv) are used.
    """
    if "source" not in metadata.columns or "slide_id" not in metadata.columns:
        raise KeyError("metadata CSV must contain at least 'slide_id' and 'source' columns.")

    sthelar_meta = metadata[metadata["source"] == "STHELAR"]
    if sthelar_meta.empty:
        print("No STHELAR slides found in metadata.")
        return

    # Paths inside training_dataset_folder
    images_zip_path = training_dataset_dir / "images.zip"
    masks_zip_path = training_dataset_dir / "masks_cell_ids_nuclei.zip"
    patch_bboxes_path = training_dataset_dir / "patch_bboxes.parquet"
    cell_count_csv_path = training_dataset_dir / "cell_count_test.csv"
    adata_dir = training_dataset_dir / "adatas"
    cid2ct_dir = training_dataset_dir / "cID2cT"

    for p in [
        images_zip_path,
        masks_zip_path,
        patch_bboxes_path,
        cell_count_csv_path,
        adata_dir,
        cid2ct_dir,
    ]:
        if not p.exists():
            raise FileNotFoundError(f"Required training dataset file/folder not found: {p}")

    print("Loading patch_bboxes.parquet and cell_count_test.csv ...")
    bboxes = load_patch_bboxes(patch_bboxes_path)
    cell_count_df = load_cell_count(cell_count_csv_path)

    # Open zip files once
    print("Opening images.zip and masks_cell_ids_nuclei.zip ...")
    with zipfile.ZipFile(images_zip_path, "r") as img_zip, \
         zipfile.ZipFile(masks_zip_path, "r") as mask_zip:

        slide_ids = sorted(sthelar_meta["slide_id"].astype(str).unique())
        for i_slide, slide_id in enumerate(slide_ids):
            
            print(f"\n** Processing STHELAR slide '{slide_id}' ({i_slide + 1}/{len(slide_ids)}) ...")
            dest_h5 = patches_dir / f"{slide_id}.h5"
            dest_h5ad = st_dir / f"{slide_id}.h5ad"
            eval_root = patches_dir.parent
            groups = _membership_groups_from_backbone_transform()
            membership_out_map = {}
            membership_missing_out_map = {}
            for g in groups:
                gdir = eval_root / f"cell_token_membership_{g['name']}"
                membership_out_map[g["name"]] = gdir / f"{slide_id}.parquet"
                gdir_missing = eval_root / f"cell_token_membership_missing_cells_{g['name']}"
                membership_missing_out_map[g["name"]] = gdir_missing / f"{slide_id}.parquet"

            patch_ids = compute_patch_list_for_slide(
                slide_id=slide_id,
                cell_count_df=cell_count_df,
            )

            if not patch_ids:
                print(f"  WARNING: No patch IDs found to build slide '{slide_id}'. Skipping.")
                continue
            print(f"  Building data for {len(patch_ids)} patch(es) ...")

            barcodes_arr, coords_arr, imgs_arr, adata_patches = build_sthelar_slide_data(
                slide_id=slide_id,
                patch_ids=patch_ids,
                bboxes=bboxes,
                cell_count_df=cell_count_df,
                img_zip=img_zip,
                mask_zip=mask_zip,
                adata_dir=adata_dir,
                he_resolution=he_resolution,
                spatialdata_dir=spatialdata_dir,
                cid2ct_dir=cid2ct_dir,
                cell_token_membership_out_map=membership_out_map,
                cell_token_membership_missing_out_map=membership_missing_out_map,
            )
            print(f"  Writing .h5 and .h5ad for slide '{slide_id}' ...")
            write_hest_h5(dest_h5, barcodes_arr, coords_arr, imgs_arr)
            write_hest_h5ad(dest_h5ad, adata_patches)




# -----------------------------
# Process
# -----------------------------

def main(args) -> None:

    # Input paths
    dataset_id = args.dataset_id
    training_dataset_id = args.training_dataset_id
    metadata_csv = get_full_path('eval', 'eval_datasets', 'st_eval_data', f'eval_st_{training_dataset_id}', f'metadata_eval_st_{training_dataset_id}.csv')
    training_dataset_dir = get_full_path('train', 'training_datasets', dataset_id, training_dataset_id)
    spatialdata_dir = get_full_path('STHELAR_HistoST', 'sdata_processed')  # Set to None here to skip ST points and all-RNA patch expression
    he_resolution = args.he_resolution

    # Check that he_resolution str is present in training_dataset_id
    if not re.search(rf"(?:^|_){he_resolution}x(?:_|$)", training_dataset_id):
        raise ValueError(
            f"he_resolution {he_resolution}x was not found as a standalone token in "
            f"training_dataset_id '{training_dataset_id}'. "
            "Expected a pattern like '_20x_' or '_40x_' "
            "(e.g. ds_ct1_20x_224_o112_1 for 20x)."
        )

    # Read metadata
    if not metadata_csv.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {metadata_csv}")
    print(f"\nReading metadata from {metadata_csv} ...")
    metadata = pd.read_csv(metadata_csv)

    # Dir for HEST
    hest_dir = get_full_path('eval', 'eval_datasets', 'st_eval_data', 'hest')
    patches_dir_hest = hest_dir / "patches"
    st_dir_hest = hest_dir / "st"
    patches_dir_hest.mkdir(parents=True, exist_ok=True)
    st_dir_hest.mkdir(parents=True, exist_ok=True)
    # Dir for STHELAR
    sthelar_dir = get_full_path('eval', 'eval_datasets', 'st_eval_data', f'eval_st_{training_dataset_id}')
    patches_dir_sthelar = sthelar_dir / "patches"
    st_dir_sthelar = sthelar_dir / "st"
    # refresh completely for STHELAR outputs (patches, st, and membership folders)
    for d in [patches_dir_sthelar, st_dir_sthelar]:
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    # Remove any previous membership group folders
    if sthelar_dir.exists():
        for p in sthelar_dir.glob("cell_token_membership_*"):
            if p.is_dir():
                shutil.rmtree(p)

    if spatialdata_dir is None:
        warnings.warn("spatialdata_dir is None. STHELAR .h5ad files will not contain '*_all' rows.")
    else:
        if not spatialdata_dir.exists():
            raise FileNotFoundError(f"SpatialData folder not found: {spatialdata_dir}")

    # Process HEST slides
    print("\n====== 1. Processing HEST slides ======")
    process_hest_slides(metadata=metadata, 
                        patches_dir=patches_dir_hest, 
                        st_dir=st_dir_hest)

    # Process STHELAR slides
    print("\n\n\n====== 2. Processing STHELAR slides ======")
    process_sthelar_slides(
        metadata=metadata,
        patches_dir=patches_dir_sthelar,
        st_dir=st_dir_sthelar,
        training_dataset_dir=training_dataset_dir,
        he_resolution=he_resolution,
        spatialdata_dir=spatialdata_dir,
    )

    print("\nDone.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Build evaluation dataset (HEST-style .h5 and .h5ad) for HEST and STHELAR slides for given training_dataset_id.")
    
    parser.add_argument("--dataset_id", type=str,
                        default="ds_ct1_20x_224_o112",
                        help="Dataset_id, e.g. ds_ct1_20x_224_o112")
    
    parser.add_argument("--training_dataset_id", type=str,
                        default="ds_ct1_20x_224_o112_1",
                        help="Training_dataset_id, e.g. ds_ct1_20x_224_o112_1")
    
    parser.add_argument("--he_resolution", type=int, default=20, choices=[20, 40], 
                        help="H&E resolution (20 or 40) used when aligning ST points for all-RNA expression.")

    args = parser.parse_args()
    main(args)