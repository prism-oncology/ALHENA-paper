"""
Generate dictionaries of cell IDs (int) that appear exclusively in each split (train/valid/test) per slide_id, for a given training_dataset_id.
Saved in <ROOT_HistoST> / train / training_datasets / <dataset_id> / <training_dataset_id> / cell_ids_sets.npz
"""

import sys
from pathlib import Path
import os
import argparse
import zipfile
import io
import numpy as np
import multiprocessing as mp
from scipy.sparse import load_npz
import pandas as pd
from tqdm import tqdm
from functools import partial
from collections import defaultdict

# Add the project root to sys.path so that 'src' can be imported
project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.constants import get_full_path



def read_csv_to_set(csv_path: str):
    """
    Reads the CSV file containing a column 'patch_id' (e.g., 'heart_s0_366'),
    and returns a set of patch base names (e.g., 'heart_s0_366').
    """
    df = pd.read_csv(csv_path, usecols=['patch_id'])
    patch_set = set(df['patch_id'])
    return patch_set



g_zf = None   # Global ZipFile handle per worker
g_names = None  # Global set of names per worker
def worker_batch_init(zip_path: str):
    """
    Initialize one ZipFile handle and precompute the member set once per worker.
    """
    global g_zf, g_names
    g_zf = zipfile.ZipFile(zip_path, mode='r')
    # Build a set once for O(1) membership checks inside the worker
    g_names = set(g_zf.namelist())



def process_patch_batch(patch_list, train_patches, valid_patches, test_patches):
    """
    Processes all 'patch_base' in patch_list within a single worker,
    accumulating local dictionaries for train/valid/test cell IDs by slide.

    Returns: (local_train_dict, local_valid_dict, local_test_dict)
       each is a dict[slide_id] -> set_of_cell_ids
    """
    global g_zf, g_names

    local_train = defaultdict(set)
    local_valid = defaultdict(set)
    local_test  = defaultdict(set)

    for patch_base in patch_list:
        npz_name = patch_base + '.npz'
        if npz_name not in g_names:
            print(f"[WARNING] Missing {npz_name} in ZIP, skipping...")
            continue  # skip if missing in the ZIP

        # Derive slide_id
        slide_id, _ = patch_base.rsplit('_', 1)

        # Load the sparse .npz from the zip and extract unique non-zero labels
        # Avoid .toarray(): work directly on the sparse .data (already excludes zeros)
        with g_zf.open(npz_name, 'r') as f:
            with io.BytesIO(f.read()) as bio:
                sp = load_npz(bio)
        data = getattr(sp, "data", None)
        if data is None or data.size == 0:
            continue
        # Unique labels among non-zero entries
        unique_ids = np.unique(data.astype(np.int32, copy=False))
        # Be defensive in case zeros were stored (shouldn't happen with standard sparse)
        if unique_ids.size == 0:
            continue
        if unique_ids[0] == 0 or (unique_ids.size and unique_ids[-1] == 0):
            unique_ids = unique_ids[unique_ids != 0]
            if unique_ids.size == 0:
                continue

        # Skip if no cell IDs
        if unique_ids.size == 0:
            continue

        # Check which set(s) this patch belongs to
        in_train = (patch_base in train_patches)
        in_valid = (patch_base in valid_patches)
        in_test  = (patch_base in test_patches)

        if in_train:
            local_train[slide_id].update(unique_ids)
        if in_valid:
            local_valid[slide_id].update(unique_ids)
        if in_test:
            local_test[slide_id].update(unique_ids)

    return (local_train, local_valid, local_test)



def merge_batch_result(batch_result, train_dict, valid_dict, test_dict):
    """
    Merges the (local_train, local_valid, local_test) dicts from one batch
    into the global dictionaries in the main process.
    """
    local_train, local_valid, local_test = batch_result

    for slide_id, cid_set in local_train.items():
        train_dict.setdefault(slide_id, set()).update(cid_set)
    for slide_id, cid_set in local_valid.items():
        valid_dict.setdefault(slide_id, set()).update(cid_set)
    for slide_id, cid_set in local_test.items():
        test_dict.setdefault(slide_id, set()).update(cid_set)



def chunker(seq, chunk_size):
    """
    Yields consecutive chunk_size-sized slices from seq.
    e.g., chunker([1,2,3,4,5], 2) -> [1,2], [3,4], [5]
    """
    for pos in range(0, len(seq), chunk_size):
        yield seq[pos:pos + chunk_size]



def main(args):

    # 1) Resolve input paths
    training_dataset_id_dir = get_full_path('train', 'training_datasets', args.dataset_id, args.training_dataset_id, as_str=True)
    train_csv_path = os.path.join(training_dataset_id_dir, "cell_count_train.csv")
    valid_csv_path = os.path.join(training_dataset_id_dir, "cell_count_valid.csv")
    test_csv_path = os.path.join(training_dataset_id_dir, "cell_count_test.csv")
    zip_path = os.path.join(training_dataset_id_dir, "masks_cell_ids_nuclei.zip")
    output_path = os.path.join(training_dataset_id_dir, "cell_ids_sets.npz")

    # 2) Read CSV => sets of patch_base
    print("[INFO] Reading split CSVs...")
    train_patches = read_csv_to_set(train_csv_path)
    valid_patches = read_csv_to_set(valid_csv_path)
    test_patches = read_csv_to_set(test_csv_path)

    print(f"  Train patches: {len(train_patches)}")
    print(f"  Valid patches: {len(valid_patches)}")
    print(f"  Test  patches: {len(test_patches)}")

    # Combine all distinct patches
    all_patches = train_patches.union(valid_patches).union(test_patches)
    all_patches_list = sorted(all_patches)  # Sort to keep processing order deterministic across runs
    print(f"[INFO] Total distinct patches to process: {len(all_patches_list)}")

    # 3) Prepare global dictionaries to hold final results
    train_cell_ids = {}
    valid_cell_ids = {}
    test_cell_ids  = {}

    # 4) Create chunked input for each worker
    chunk_size = args.chunk_size
    chunked_patches = list(chunker(all_patches_list, chunk_size))
    print(f"[INFO] Created {len(chunked_patches)} chunks, each up to size {chunk_size}.")

    # Bind the split patch sets into process_patch_batch for the worker pool
    batch_fn = partial(
        process_patch_batch,
        train_patches=train_patches,
        valid_patches=valid_patches,
        test_patches=test_patches
    )

    # 5) Setup multiprocessing (cap workers to number of chunks to avoid oversubscription)
    requested_workers = args.num_workers or mp.cpu_count()
    num_workers = max(1, min(requested_workers, len(chunked_patches)))
    print(f"[INFO] Using {num_workers} worker processes...")

    # 6) Map the chunks -> gather partial results -> merge once per chunk
    # Use a larger dispatcher chunksize and unordered completion for better throughput
    dispatch_chunksize = max(1, len(chunked_patches) // (num_workers * 4))
    with mp.Pool(processes=num_workers,
                 initializer=worker_batch_init,
                 initargs=(zip_path,)) as pool:
        results_iter = pool.imap_unordered(batch_fn, chunked_patches, chunksize=dispatch_chunksize)
        for batch_result in tqdm(results_iter,
                                 total=len(chunked_patches),
                                 desc="Processing patches in batches"):
            merge_batch_result(batch_result, train_cell_ids, valid_cell_ids, test_cell_ids)

    # 7) Compute exclusive cell IDs per slide (train/valid/test) and overlap metrics
    print("[INFO] Computing exclusive cell IDs per slide (train/valid/test)...")

    all_slide_ids = set(train_cell_ids.keys()) \
                    .union(valid_cell_ids.keys()) \
                    .union(test_cell_ids.keys())

    train_only_by_slide = {}
    valid_only_by_slide = {}
    test_only_by_slide  = {}
    overlap_count_by_slide = {}
    overlap_breakdown_by_slide = {}
    total_count_by_slide   = {}

    for slide_id in all_slide_ids:
 
        trn = train_cell_ids.get(slide_id, set())
        vld = valid_cell_ids.get(slide_id, set())
        tst = test_cell_ids.get(slide_id, set())

        # Exclusive (present only in one split)
        train_only = trn - vld - tst
        valid_only = vld - trn - tst
        test_only  = tst - trn - vld

        # Non-exclusive split into disjoint buckets (no double counting)
        train_valid_only = (trn & vld) - tst
        train_test_only  = (trn & tst) - vld
        valid_test_only  = (vld & tst) - trn
        all_three        = trn & vld & tst

        # Store sorted arrays for compact, deterministic output
        train_only_by_slide[slide_id] = np.array(sorted(train_only), dtype=np.int64)
        valid_only_by_slide[slide_id] = np.array(sorted(valid_only), dtype=np.int64)
        test_only_by_slide[slide_id]  = np.array(sorted(test_only),  dtype=np.int64)

        # Overlap & totals (IDs in >=2 splits are non-exclusive)
        total_ids = trn | vld | tst
        overlap_breakdown_by_slide[slide_id] = {
            "train_valid_only": len(train_valid_only),
            "train_test_only": len(train_test_only),
            "valid_test_only": len(valid_test_only),
            "all_three": len(all_three),
        }
        non_exclusive = sum(overlap_breakdown_by_slide[slide_id].values())
        overlap_count_by_slide[slide_id] = non_exclusive
        total_count_by_slide[slide_id] = len(total_ids)

    # 8) Save final dictionaries (single .npz for all splits)
    print(f"[INFO] Saving results to {output_path}...")
    np.savez_compressed(
        output_path,
        train_only_by_slide=train_only_by_slide,
        valid_only_by_slide=valid_only_by_slide,
        test_only_by_slide=test_only_by_slide,
    )

    # 9) Summary (including per-slide non-exclusive counts)
    print("[INFO] Done!")
    print(f"   * Number of slides in final result: {len(all_slide_ids)}")
    trn_sizes = [len(a) for a in train_only_by_slide.values()]
    vld_sizes = [len(a) for a in valid_only_by_slide.values()]
    tst_sizes = [len(a) for a in test_only_by_slide.values()]
    if trn_sizes:
        print(f"   * Train-only cells - min:{min(trn_sizes)}, max:{max(trn_sizes)}, avg:{sum(trn_sizes)/len(trn_sizes):.1f}")
    if vld_sizes:
        print(f"   * Valid-only cells - min:{min(vld_sizes)}, max:{max(vld_sizes)}, avg:{sum(vld_sizes)/len(vld_sizes):.1f}")
    if tst_sizes:
        print(f"   * Test-only cells  - min:{min(tst_sizes)}, max:{max(tst_sizes)}, avg:{sum(tst_sizes)/len(tst_sizes):.1f}")

    print("\n[INFO] Per-slide overlap (non-exclusive) counts:")
    for slide_id in sorted(all_slide_ids):
        overlap = overlap_count_by_slide.get(slide_id, 0)
        total = total_count_by_slide.get(slide_id, 0)
        pct = (100.0 * overlap / total) if total else 0.0
        print(f"   * {slide_id}: non-exclusive {overlap} / total {total} ({pct:.1f}%)")

        if overlap:
            d = overlap_breakdown_by_slide[slide_id]
            print(
                "      - "
                f"train&valid only: {d['train_valid_only']}, "
                f"train&test only: {d['train_test_only']}, "
                f"valid&test only: {d['valid_test_only']}, "
                f"all three: {d['all_three']}"
            )
        


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Identify cell IDs that appear exclusively in each split (train/valid/test), by slide, for a given training_dataset_id.")

    parser.add_argument("--dataset_id", type=str,
                        default="ds_ct1_20x_224_o112",
                        help="Dataset_id, e.g. ds_ct1_20x_224_o112")
    
    parser.add_argument("--training_dataset_id", type=str,
                        default="ds_ct1_20x_224_o112_1",
                        help="Training_dataset_id, e.g. ds_ct1_20x_224_o112_1")
    
    parser.add_argument("--num_workers", type=int,
                        default=8,
                        help="Number of parallel workers")
    
    parser.add_argument("--chunk_size", type=int,
                        default=100,
                        help="Number of patches to process per worker batch")
    
    args = parser.parse_args()
    main(args)