"""
Generating:
    - images.zip, 
    - labels.zip (optional, not needed for HistoST project), 
    - masks_cell_ids_nuclei.zip, 
    - cell_count.csv
    - types.csv
    - patch_metrics.csv (potentially updated with CellViT inference results in a later step by _5_patch_eval_results.py)
    - slide_ids.txt
directly from one or many SpatialData slides.

Saved in <ROOT_HistoST> / "train" / "datasets" / <dataset_id> /
"""

import argparse, json, io, os, gc, zipfile
from pathlib import Path
import sys
from typing import List, Dict, Tuple, Optional
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import random
import spatialdata as sd
from shapely.geometry import box, Polygon, MultiPolygon
from shapely.validation import explain_validity
from skimage.draw import polygon
from PIL import Image
from scipy.sparse import csr_matrix, save_npz
from joblib import Parallel, delayed
from math import ceil

# These come from the sopa version pinned in the process_slide conda environment.
from sopa.utils import get_spatial_image, to_intrinsic
from sopa import make_image_patches

# Add the project root to sys.path so that 'src' can be imported
project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.constants import get_full_path
from src.utils.cell_ids import int_cell_id
from src.utils.sdata_utils import promote_20x_to_default



def update_slide_ids_txt(output_dir: Path, slide_ids: List[str], filename: str = "slide_ids.txt") -> None:
    """
    Maintain <output_dir>/slide_ids.txt as an append-only (deduped) list of slide IDs
    that are present in this dataset_id folder across runs.
    """
    path = output_dir / filename

    existing: List[str] = []
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            existing = [line.strip() for line in f if line.strip()]

    seen = set(existing)
    for sid in slide_ids:
        if sid not in seen:
            existing.append(sid)
            seen.add(sid)

    # Atomic write (avoid partial file if interrupted)
    tmp_path = output_dir / (filename + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        for sid in existing:
            f.write(f"{sid}\n")
    os.replace(tmp_path, path)



def correct_geometry(geometry):
    """
    Corrects the input geometry if invalid due to self-intersection
    """
    if not geometry.is_valid:
        explanation = explain_validity(geometry)
        if "Self-intersection" in explanation:
            # Attempt to fix self-intersection by buffering with zero distance
            corrected_geometry = geometry.buffer(0)
            if corrected_geometry.is_valid:
                return corrected_geometry
    return geometry



def patchs_cells_intersections(patch_df, geo_df):

    # Returns the indices of the patches and corresponding cells that intersect
    patch_idx, cell_idx = geo_df.geometry.sindex.query(patch_df.geometry, predicate=None)

    # Group the cell data by patch index
    patch_idx_2_cells_idx = pd.Series(geo_df.index[cell_idx]).groupby(patch_idx).apply(list)
    patch_cell_pairs_idx = [(patch_idx, patch_idx_2_cells_idx[patch_idx]) for patch_idx in patch_idx_2_cells_idx.index]

    # Return the patch index and their corresponding cell indices in the same order
    return patch_cell_pairs_idx




def rasterize(cell_patch_intersection, xy_min, out_shape) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Rasterize a Polygon or MultiPolygon (already intersected with the patch box) into pixel indices
    Returns (rr, cc) or None if nothing falls inside the patch

    Properly handles MultiPolygon by rasterizing each component separately
    """
    if cell_patch_intersection.is_empty:
        return None

    rr_all = []
    cc_all = []

    def _poly_to_rrcc(poly: Polygon):
        coords = np.asarray(poly.exterior.coords, dtype=np.float32)
        coords[:, 0] -= xy_min[0]
        coords[:, 1] -= xy_min[1]
        rr, cc = polygon(coords[:, 1], coords[:, 0], out_shape)
        return rr, cc

    if isinstance(cell_patch_intersection, Polygon):
        rr, cc = _poly_to_rrcc(cell_patch_intersection)
        if rr.size == 0:
            return None
        return rr, cc

    if isinstance(cell_patch_intersection, MultiPolygon):
        for poly in cell_patch_intersection.geoms:
            if poly.is_empty:
                continue
            rr, cc = _poly_to_rrcc(poly)
            if rr.size:
                rr_all.append(rr)
                cc_all.append(cc)
        if not rr_all:
            return None
        return np.concatenate(rr_all), np.concatenate(cc_all)

    raise ValueError(f"Unsupported geometry type: {type(cell_patch_intersection)}")




def remap_label(pred):
    """
    Rename all instance id so that the id is contiguous i.e [0, 1, 2, 3] not [0, 2, 4, 6]. The ordering of instances (which one comes first)
    """
    pred_id = list(np.unique(pred))
    
    if 0 in pred_id:
        pred_id.remove(0)
    if len(pred_id) == 0:
        return pred  # no label

    new_pred = np.zeros(pred.shape, np.int32)
    for idx, inst_id in enumerate(pred_id):
        new_pred[pred == inst_id] = idx + 1
    return new_pred




def checking_results(img, mask, patch_id, cat2color, idx2color, check_dir, use_cell_ids: bool = False):

    # Create legend handles and labels (only for type_map visualization)
    legend_handles = []
    legend_labels = []
    if (not use_cell_ids) and (cat2color is not None):
        for cat, color in cat2color.items():
            legend_handles.append(Line2D([0], [0], marker='o', color='w', label=cat, markerfacecolor=[c/255 for c in color], markersize=10))
            legend_labels.append(cat)

    plt.figure()
    
    # Plot H&E patch
    plt.subplot(1,2,1)
    plt.imshow(img)
    
    # Plot H&E patch with mask
    plt.subplot(1,2,2)

    plt.imshow(img)
    # Build an RGBA overlay where background stays transparent
    overlay = np.zeros((*mask.shape, 4), dtype=np.uint8)  # H×W×4
    if use_cell_ids:
        # When type_map is disabled the segmentation is visualized from cell_id_map instead.
        # Deterministic pseudo-colors directly from the integer cell_id
        ids = mask.astype(np.uint32, copy=False)
        nz = ids != 0
        x = ids[nz]
        # Mix bits (LCG + xor) to spread ids into color space (deterministic)
        x = (x * np.uint32(2654435761))  # Knuth multiplicative hash
        x ^= (x >> np.uint32(16))
        r = (x & np.uint32(255)).astype(np.uint8)
        g = ((x >> np.uint32(8)) & np.uint32(255)).astype(np.uint8)
        b = ((x >> np.uint32(16)) & np.uint32(255)).astype(np.uint8)
        a = np.full_like(r, 128, dtype=np.uint8)
        overlay[nz] = np.stack([r, g, b, a], axis=-1)
    else:
        for idx, rgba in idx2color.items():
            if idx == 0:                       # keep background fully transparent
                continue
            overlay[mask == idx] = rgba        # Fill the overlay with colors from idx2color

    plt.imshow(overlay, interpolation="none")
    if legend_handles:
        plt.legend(handles=legend_handles, labels=legend_labels, loc='upper left', bbox_to_anchor=(1, 1))
    
    plt.savefig(os.path.join(check_dir, f"patch_{patch_id}.png"), bbox_inches='tight')
    plt.close()



def process_patch(pair: Tuple[int, List[int]], patch_df, geo_df, he_data,
                  cell2cat: Dict[int, int], nb_cat: int, 
                  slide_id: str, tissue: str, cat_names: List[str], he_patch_width: int,
                  save_inst_type_maps: bool = True):

    p_idx, cell_idxs = pair
    p_id = patch_df['patch_id'].iloc[p_idx]
    patch_bounds = patch_df.bounds.iloc[p_idx]
    ymin, ymax, xmin, xmax = int(patch_bounds['miny']), int(patch_bounds['maxy']), int(patch_bounds['minx']), int(patch_bounds['maxx'])
    bnd = [xmin, ymin, xmax, ymax]  # patch bounding box
    patch_w = ymax - ymin

    if patch_w != he_patch_width:
        raise ValueError(f"Patch width {patch_w} does not match expected {he_patch_width}")

    patch_box = box(xmin, ymin, xmax, ymax)
    # Overlap handling: "nearest center wins" (geometry-based, class-agnostic)
    # winner_score stores the best (highest) score at each pixel
    winner_score = np.full((patch_w, patch_w), -np.inf, dtype=np.float32)
    cell_id_map = np.zeros((patch_w, patch_w), dtype=np.int32)
    if save_inst_type_maps:
        inst_map = np.zeros((patch_w, patch_w), dtype=np.int32)
        type_map = np.zeros((patch_w, patch_w), dtype=np.int32)
        inst = 1  # temporary per-cell id, remapped to contiguous at the end
    else:
        inst_map = None
        type_map = None

    # exact_counts must be computed after overlap resolution, otherwise a cell can be
    # counted even when it ends up with 0 pixels.
    exact_counts = {name: 0 for name in cat_names}
    # Map integer cell_id -> cat_idx for the cells considered here
    cell_int2cat: Dict[int, int] = {}

    # Deterministic processing order (ties in distance keep earlier winner)
    cell_idxs = sorted(cell_idxs, key=lambda cid: str(cid))

    for c_idx in cell_idxs:
        cell = geo_df.loc[c_idx].geometry
        if not isinstance(cell, (Polygon, MultiPolygon)):
            print(f"Unexpected type for cell: {type(cell)}, value: {cell}")
            continue
        cell = correct_geometry(cell)
        if not cell.is_valid:
            print(f"Invalid geometry for {c_idx}")
            continue
        if cell.is_empty or not patch_box.intersects(cell):
            continue
        cat_idx = cell2cat.get(c_idx)
        if cat_idx is None:
            continue
        cat_idx = int(cat_idx)
        if not (0 <= cat_idx < nb_cat):
            continue
        # Cache mapping for final counting (after overlap resolution)
        cid_int = int_cell_id(c_idx)
        cell_int2cat[cid_int] = cat_idx
        inter = cell.intersection(patch_box)
        try:
            rc = rasterize(inter, (xmin, ymin), (patch_w, patch_w))
            if rc is None:
                continue
            rr, cc = rc

            # Geometry-based z-buffer:
            # assign overlapped pixels to the cell whose (clipped) center is closest
            # Use a point guaranteed inside the clipped geometry (more robust than centroid for concave shapes)
            p = inter.representative_point()
            cx = np.float32(p.x - xmin)
            cy = np.float32(p.y - ymin)

            # score = -squared_distance_to_center (higher is better)
            dx = cc.astype(np.float32) - cx
            dy = rr.astype(np.float32) - cy
            score = -(dx * dx + dy * dy)

            prev_score = winner_score[rr, cc]
            win = score > prev_score   # strict '>' => deterministic tie handling (keeps earlier winner)
            if np.any(win):
                rrw = rr[win]
                ccw = cc[win]
                winner_score[rrw, ccw] = score[win]
                cell_id_map[rrw, ccw] = cid_int
                if save_inst_type_maps:
                    inst_map[rrw, ccw] = inst
                    type_map[rrw, ccw] = cat_idx + 1
                if save_inst_type_maps:
                    inst += 1
        except ValueError as e:
            print("Error rasterizing cell:", e)
            continue

    # If nothing was rasterized into the patch, skip it
    if not cell_id_map.any():
        return None

    # Final exact per-category counts from the resolved mask (post-overlap)
    present = np.unique(cell_id_map)
    present = present[present != 0]
    for cid in present:
        cat = cell_int2cat.get(int(cid))
        if cat is not None:
            exact_counts[cat_names[cat]] += 1

    # Build the image patch
    img = np.transpose(he_data[:, ymin:ymax, xmin:xmax], (1, 2, 0)).astype(np.uint8)
    if img.shape != (patch_w, patch_w, 3):  # corrupt geometry
        print(f"Patch {p_id} has corrupt geometry, skipping.")
        return None
    img = Image.fromarray(img.astype(np.uint8))

    if save_inst_type_maps:
        # Remap instance ids to contiguous [1..N]
        inst_map = remap_label(inst_map)

        # Debug sanity check: each instance should map to exactly one cell_id
        # (skip background inst_id=0)
        for inst_id in np.unique(inst_map):
            if inst_id == 0:
                continue
            vals = np.unique(cell_id_map[inst_map == inst_id])
            vals = vals[vals != 0]
            if len(vals) > 1:
                print(f"[WARN] Patch {p_id}: inst_id {inst_id} has multiple cell_ids: {vals[:10]}")

    return p_id, img, inst_map, type_map, cell_id_map, tissue, exact_counts, bnd




def process_slide(sdata_dir, output_dir, slide_id, tissue, cfg, label2cat, cat2idx, cat2color, idx2color, cat_names, nb_cat):
    
    # Loading sdata, keeping only the required annotation table
    sdata = sd.read_zarr(sdata_dir / f"sdata_{slide_id}.zarr")
    for t in list(sdata.tables):
        if t != cfg.annots_table:
            del sdata.tables[t]
    gc.collect()
    logging.info(sdata)
    print(sdata)

    # Promote 20x to default if needed
    logging.info(f"\nUsing H&E resolution: {cfg.he_resolution}x")
    print(f"\nUsing H&E resolution: {cfg.he_resolution}x")
    resolution_40 = get_spatial_image(sdata, "he").shape
    logging.info(f"40x resolution: {resolution_40}")
    print(f"40x resolution: {resolution_40}")
    if cfg.he_resolution == 20:
        promote_20x_to_default(sdata, key="he")
        resolution_20 = get_spatial_image(sdata, "he").shape
        logging.info(f"20x resolution: {resolution_20}")
        print(f"20x resolution: {resolution_20}")
        logging.info(f"Check dividing 40x by 2 / Diff are : {resolution_40[1] // 2 - resolution_20[1]} and {resolution_40[2] // 2 - resolution_20[2]}")
        print(f"Check dividing 40x by 2 / Diff are : {resolution_40[1] // 2 - resolution_20[1]} and {resolution_40[2] // 2 - resolution_20[2]}")
    elif cfg.he_resolution != 40:
        raise ValueError("Unsupported H&E resolution. Choose 20 or 40.")
    logging.info(f"\nNew sdata: {sdata}")
    print(f"\nNew sdata: {sdata}")

    # Get HE and segmentation data
    he = get_spatial_image(sdata, "he")  # Gets a DataArray from a SpatialData object (if the image has multiple scale, the `scale0` is returned)
    geo_df = to_intrinsic(sdata, sdata.shapes['nucleus_boundaries'], he)
    geo_df['geometry'] = geo_df['geometry'].apply(correct_geometry)

    # Check names of the columns in geo_df
    logging.info(f"Columns in geo_df: {geo_df.columns}")
    print(f"Columns in geo_df: {geo_df.columns}")

    # Build patches
    name_key_patches = f'sopa_patches_he_{cfg.he_patch_width}_o{cfg.he_patch_overlap}'
    make_image_patches(sdata, patch_width=cfg.he_patch_width, patch_overlap=cfg.he_patch_overlap, image_key='he', key_added=name_key_patches)
    patch_df = sdata.shapes[name_key_patches].copy()
    sdata.delete_element_from_disk(name_key_patches)
    patch_df['patch_id'] = slide_id + '_' + patch_df.index.astype(str)

    # Get cell2cat idx dictionary
    ad_obs = sdata.tables[cfg.annots_table].obs.copy()
    ad_obs['cat_idx'] = ad_obs[cfg.annots_column].map(label2cat).map(cat2idx)
    cell2cat = dict(zip(ad_obs.cell_id, ad_obs.cat_idx))  

    # Already-present files (avoid duplicates on re-run)
    existing_imgs   = set(cfg.images_zip.namelist())
    existing_masks  = set(cfg.cell_ids_zip.namelist()) 
    if cfg.save_inst_type_maps:
        existing_labels = set(cfg.labels_zip.namelist())
    else:
        existing_labels = set()

    # Build spatial index mapping (patch -> cells)
    pairs = patchs_cells_intersections(patch_df, geo_df)
    logging.info(f"Processing {len(pairs)} patches over a total of {len(patch_df)} patches.")
    print(f"Processing {len(pairs)} patches over a total of {len(patch_df)} patches.")

    # Select a random subset of pairs for checking if enabled
    if cfg.checking == 'yes':
        check_dir = output_dir / 'checking_patches' / slide_id
        check_dir.mkdir(parents=True, exist_ok=True)
        patch_to_check = random.sample(range(len(pairs)), min(cfg.N_checks, len(pairs)))
        patch_ids_to_check = [pairs[i][0] for i in patch_to_check]
        patch_ids_to_check = [f"{slide_id}_{p_id}" for p_id in patch_ids_to_check]

    # Preload H&E data into memory
    logging.info("Preloading H&E data into memory")
    print("Preloading H&E data into memory")
    he_data = he.values
    
    # Generate patches (batched if too many)
    logging.info("\nGenerating patches...")
    print("\nGenerating patches...")

    worker = delayed(process_patch)
    rows_cc, rows_tp, patch_ids = [], [], []
    n_pairs = len(pairs)
    
    if n_pairs > cfg.max_patches_in_memory:
        print(f"Too many patches ({n_pairs}) for in-memory processing. Processing in batches of {cfg.max_patches_in_memory}.")
        logging.info(f"Processing in batches of {cfg.max_patches_in_memory}.")
        n_batches = ceil(n_pairs / cfg.max_patches_in_memory)
        pbar = tqdm(total=n_pairs)
        
        for bi in range(n_batches):
            bs = bi * cfg.max_patches_in_memory
            be = min((bi + 1) * cfg.max_patches_in_memory, n_pairs)
            batch_pairs = pairs[bs:be]
            batch_results = Parallel(cfg.num_workers, prefer='threads')(
                worker(pair, patch_df, geo_df, he_data, cell2cat, nb_cat, slide_id, tissue, cat_names, cfg.he_patch_width, cfg.save_inst_type_maps)
                for pair in batch_pairs
            )
            for res in filter(None, batch_results):
                p_id, img, inst_map, type_map, cell_id_mask, tissue_type, counts, bnd = res
                
                # Checking
                if cfg.checking == 'yes' and p_id in patch_ids_to_check:
                    if cfg.save_inst_type_maps:
                        checking_results(img, type_map, p_id, cat2color, idx2color, check_dir, use_cell_ids=False)
                    else:
                        checking_results(img, cell_id_mask, p_id, cat2color, idx2color, check_dir, use_cell_ids=True)
                
                # Names
                fname = f"{p_id}.png"
                npz_name = fname.replace('.png', '.npz')
                
                # images.zip
                buf = io.BytesIO()
                img.save(buf, format='PNG')
                if fname not in existing_imgs:
                    cfg.images_zip.writestr(fname, buf.getvalue())
                    existing_imgs.add(fname)
                
                # labels.zip
                if cfg.save_inst_type_maps:
                    inst_sp = csr_matrix(inst_map)
                    type_sp = csr_matrix(type_map)
                    buf = io.BytesIO()
                    np.savez(buf,
                             inst_map_data=inst_sp.data, inst_map_indices=inst_sp.indices,
                             inst_map_indptr=inst_sp.indptr, inst_map_shape=inst_sp.shape,
                             type_map_data=type_sp.data, type_map_indices=type_sp.indices,
                             type_map_indptr=type_sp.indptr, type_map_shape=type_sp.shape)
                    if npz_name not in existing_labels:
                        cfg.labels_zip.writestr(npz_name, buf.getvalue())
                        existing_labels.add(npz_name)
                
                # masks_cell_ids_nuclei.zip
                cell_id_sp = csr_matrix(cell_id_mask)
                buf = io.BytesIO()
                save_npz(buf, cell_id_sp)
                if npz_name not in existing_masks:
                    cfg.cell_ids_zip.writestr(npz_name, buf.getvalue())
                    existing_masks.add(npz_name)
                
                # rows
                rows_cc.append({'patch_id': p_id, **counts})
                rows_tp.append({'patch_id': p_id, 'type': tissue_type})
                patch_ids.append({'slide_id': slide_id, 'patch_id': p_id, 'x_min': bnd[0], 'y_min': bnd[1], 'x_max': bnd[2], 'y_max': bnd[3]})
            
            del batch_results
            gc.collect()
            pbar.update(be - bs)
        pbar.close()
    
    else:
        
        results = Parallel(cfg.num_workers, prefer='threads')(
            worker(pair, patch_df, geo_df, he_data, cell2cat, nb_cat, slide_id, tissue, cat_names, cfg.he_patch_width, cfg.save_inst_type_maps)
            for pair in tqdm(pairs))
        
        for res in filter(None, results):
            p_id, img, inst_map, type_map, cell_id_mask, tissue_type, counts, bnd = res
            
            # Checking
            if cfg.checking == 'yes' and p_id in patch_ids_to_check:
                if cfg.save_inst_type_maps:
                   checking_results(img, type_map, p_id, cat2color, idx2color, check_dir, use_cell_ids=False)
                else:
                    checking_results(img, cell_id_mask, p_id, cat2color, idx2color, check_dir, use_cell_ids=True)

            # Names
            fname = f"{p_id}.png"
            npz_name = fname.replace('.png', '.npz')
            
            # images.zip
            buf = io.BytesIO()
            img.save(buf, format='PNG')
            if fname not in existing_imgs:
                cfg.images_zip.writestr(fname, buf.getvalue())
                existing_imgs.add(fname)

            # labels.zip
            if cfg.save_inst_type_maps:
                inst_sp = csr_matrix(inst_map)
                type_sp = csr_matrix(type_map)
                buf = io.BytesIO()
                np.savez(buf,
                         inst_map_data=inst_sp.data, inst_map_indices=inst_sp.indices,
                         inst_map_indptr=inst_sp.indptr, inst_map_shape=inst_sp.shape,
                         type_map_data=type_sp.data, type_map_indices=type_sp.indices,
                         type_map_indptr=type_sp.indptr, type_map_shape=type_sp.shape)
                if npz_name not in existing_labels:
                    cfg.labels_zip.writestr(npz_name, buf.getvalue())
                    existing_labels.add(npz_name)
            
            # masks_cell_ids_nuclei.zip
            cell_id_sp = csr_matrix(cell_id_mask)
            buf = io.BytesIO()
            save_npz(buf, cell_id_sp)
            if npz_name not in existing_masks:
                cfg.cell_ids_zip.writestr(npz_name, buf.getvalue())
                existing_masks.add(npz_name)
            
            rows_cc.append({'patch_id': p_id, **counts})
            rows_tp.append({'patch_id': p_id, 'type': tissue_type})
            patch_ids.append({'slide_id': slide_id, 'patch_id': p_id, 'x_min': bnd[0], 'y_min': bnd[1], 'x_max': bnd[2], 'y_max': bnd[3]})

    return pd.DataFrame(rows_cc), pd.DataFrame(rows_tp), pd.DataFrame(patch_ids)





def main(cfg):
    
    if len(cfg.slides) != len(cfg.slide_tissues):
        raise ValueError("Number of slides must match number of slide tissues.")
    print(f"Processing dataset_id: {cfg.dataset_id} with {len(cfg.slides)} slides.")

    # Check that args.annots_table=='table_cells' and that args.annots_column=='final_label'
    if cfg.annots_table != 'table_cells':
        raise ValueError(f"Expected annots_table='table_cells', got '{cfg.annots_table}'")
    if cfg.annots_column != 'final_label':
        raise ValueError(f"Expected annots_column='final_label', got '{cfg.annots_column}'")

    # Path
    sdata_dir = get_full_path("STHELAR_HistoST", "sdata_final")
    output_dir = get_full_path("train", "datasets", cfg.dataset_id)
    
    output_dir.mkdir(parents=True, exist_ok=True)

    # Track all slide IDs seen for this dataset_id across runs
    update_slide_ids_txt(output_dir, cfg.slides, filename="slide_ids.txt")

    # Init logger
    log_path = output_dir / 'get_patches.log'
    print(f"Logging to {log_path}")
    logging.basicConfig(filename=log_path, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Loading annots dictionaries
    with open(cfg.dict_dir / 'label2cat.json') as f: label2cat = json.load(f)
    with open(cfg.dict_dir / 'cat2idx.json') as f: cat2idx = json.load(f)
    with open(cfg.dict_dir / 'cat2color.json') as f: cat2color = json.load(f)
    cat_names = [k for k, _ in sorted(cat2idx.items(), key=lambda kv: kv[1])]
    nb_cat = len(cat2idx)
    idx2color = {idx+1: cat2color[cat]+[0.5*255] for cat, idx in cat2idx.items()}
    idx2color[0] = [0, 0, 0, 0]  # Background color
    logging.info(f"\nNumber of categories: {nb_cat}")
    print(f"\nNumber of categories: {nb_cat}")
    logging.info(f"Categories: {cat_names}")
    print(f"Categories: {cat_names}")

    # Zip files
    zip_mode = 'a' if (output_dir / 'images.zip').exists() else 'w'   # append if files already exist
    cfg.images_zip = zipfile.ZipFile(output_dir / 'images.zip', zip_mode, zipfile.ZIP_DEFLATED)
    if cfg.save_inst_type_maps:
        zip_mode = 'a' if (output_dir / 'labels.zip').exists() else 'w'
        cfg.labels_zip = zipfile.ZipFile(output_dir / 'labels.zip', zip_mode, zipfile.ZIP_DEFLATED)
    else:
        cfg.labels_zip = None
    zip_mode = 'a' if (output_dir / 'masks_cell_ids_nuclei.zip').exists() else 'w'
    cfg.cell_ids_zip = zipfile.ZipFile(output_dir / 'masks_cell_ids_nuclei.zip', zip_mode, zipfile.ZIP_DEFLATED)

    # Process each slide
    all_cc, all_tp, all_pi = [], [], []
    logging.info("Processing slides...")
    print("Processing slides...")
    for i, (slide_id, tissue) in enumerate(zip(cfg.slides, cfg.slide_tissues)):
        logging.info(f"\n\n\n****** Processing slide {i+1}/{len(cfg.slides)}: {slide_id} ({tissue}) ******")
        print(f"\n\n\n****** Processing slide {i+1}/{len(cfg.slides)}: {slide_id} ({tissue}) ******")
        cc, tp, pi = process_slide(sdata_dir, output_dir, slide_id, tissue, cfg, label2cat, cat2idx, cat2color, idx2color, cat_names, nb_cat)
        all_cc.append(cc)
        all_tp.append(tp)
        all_pi.append(pi)
        gc.collect()

    cfg.images_zip.close()
    if cfg.labels_zip is not None:
        cfg.labels_zip.close()
    cfg.cell_ids_zip.close()

    # Save cell counts, types, and patch_ids to CSV
    logging.info("\nSaving cell counts, types, and patch IDs to CSV files...")
    print("\nSaving cell counts, types, and patch IDs to CSV files...")

    cc_path = output_dir / 'cell_count.csv'
    tp_path = output_dir / 'types.csv'
    pi_path = output_dir / 'patch_metrics.csv'

    new_cc = pd.concat(all_cc)
    new_tp = pd.concat(all_tp)
    new_pi = pd.concat(all_pi)

    if cc_path.exists():
        new_cc = pd.concat([pd.read_csv(cc_path), new_cc]).drop_duplicates(subset='patch_id')
    if tp_path.exists():
        new_tp = pd.concat([pd.read_csv(tp_path), new_tp]).drop_duplicates(subset='patch_id')
    if pi_path.exists():
        new_pi = pd.concat([pd.read_csv(pi_path), new_pi]).drop_duplicates(subset='patch_id')

    new_cc.to_csv(cc_path, index=False)
    new_tp.to_csv(tp_path, index=False)
    new_pi.to_csv(pi_path, index=False)

    logging.info("Done.")
    print("Done.")




if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Generate data for the given dataset_id")

    # <dataset_id> (e.g. "ds_ct1_20x_224_o112")
    parser.add_argument('--dataset_id', type=str, default='ds_ct1_20x_224_o112', help='Dataset ID, e.g. ds_ct1_20x_224_o112')
    
    # Slide IDs to use
    parser.add_argument('--slides', nargs='+', required=True, help='slide IDs')
    parser.add_argument('--slide_tissues', nargs='+', required=True, help='tissue type per slide (SAME ORDER AS SLIDES and WITH FIRST LETTER CAPITALIZED, e.g. "Kidney", "Liver")')
    
    # Dataset-specific parameters
    parser.add_argument('--dict_dir', type=Path, default=Path('/path/to/ALHENA-paper/data/annots/annot_dicts_ct_1'), help='Directory with annot dictionaries for given ct_cat_id')
    parser.add_argument('--annots_table', type=str, default='table_cells', help='Name of the annotations table in sdata / Choose table_cells everytime')
    parser.add_argument('--annots_column', type=str, default='final_label', help='Name of the column in the annotations table with cell types / Choose final_label everytime')
    parser.add_argument('--he_resolution', type=int, default=20, help='H&E image resolution between 20x and 40x / Choose 20 or 40')
    parser.add_argument("--he_patch_width", type=int, default=224, help="Patch width for H&E images in pixels")
    parser.add_argument("--he_patch_overlap", type=int, default=112, help="Patch overlap for H&E images in pixels") 
    
    # Processing parameters
    parser.add_argument('--num_workers', type=int, default=10, help='Number of parallel workers for patch processing')
    parser.add_argument("--checking", type=str, default='yes', help="Plot few images per slide for checking, choose 'yes' or 'no'")
    parser.add_argument("--N_checks", type=int, default=50, help="Number of patches to check per slide, if checking is enabled")
    parser.add_argument("--max_patches_in_memory", type=int, default=80000, help="If a slide has more than this number of patches to process, process them in batches of this size to limit peak memory.")
    parser.add_argument("--no_inst_type_maps", action="store_true", help="Do not generate/write instance_map and type_map (labels.zip). CSV outputs are unchanged. Checking overlay uses the cell_id mask.")

    cfg = parser.parse_args()
    cfg.save_inst_type_maps = not cfg.no_inst_type_maps

    main(cfg)