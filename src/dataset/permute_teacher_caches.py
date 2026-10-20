#!/usr/bin/env python
"""
Build the shuffled-pairing control caches: the TRAIN rows of the Novae st_feat cache and of the scConcept embeddings are
permuted across slides (other rows copied). --mode verify_means checks that two teacher_center_means.h5 files are identical.
Runs in the `histost` env.

Saved in <ROOT_HistoST> / train / training_datasets / <dataset_id> / <training_dataset_id> / novae_teacher_cache / <novae_training_id>_SHUFFLED_seed<seed> /
    - <slide_id>.h5: permuted st_feat
    - manifest.json
Saved in <ROOT_HistoST> / train / trainings / trainings_scConcept / <scconcept_training_id>_SHUFFLED_seed<seed> /
    - embeddings/<slide_id>.h5: permuted cls_emb
    - manifest.json
"""

import os
import sys
import json
import shutil
import hashlib
import logging
import argparse
from datetime import datetime, timezone

import numpy as np
import h5py
from tqdm import tqdm

# Add the project root to sys.path so that 'src' can be imported.
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Torch-free imports only (helpers of src.dataset.novae_teacher are copied below)
from src.utils.constants import get_full_path  # noqa: E402

logger = logging.getLogger(__name__)

OUTPUT_SIZE = 128      # Novae st_feat dim
EMBEDDING_SIZE = 512   # scConcept cls_emb dim

# Permutation format version (recorded in the manifests and in every per-slide h5)
SHUFFLE_SCHEMA_VERSION: int = 1

# Marker contained in every sentinel id / fingerprint
SHUFFLE_MARKER: str = "SHUFFLED"

_SCC_READ_BLOCK_ROWS = 65_536   # target rows per contiguous cls_emb read (see _scc_block_rows)
_COLSUM_RTOL = 1e-9             # relative tolerance of the column-sum check


# --------------------------------------------------------------------------- sentinel identity
def shuffled_sentinel(name: str, seed: int) -> str:
    """``{name}_SHUFFLED_seed{seed}``: sentinel for the output novae_training_id, scconcept_training_id and fingerprint."""
    return f"{name}_{SHUFFLE_MARKER}_seed{int(seed)}"


def _assert_marked(value: str, what: str) -> None:
    """Raise if a sentinel does not contain SHUFFLE_MARKER."""
    if SHUFFLE_MARKER not in str(value):
        raise ValueError(
            f"{what}='{value}' does not contain '{SHUFFLE_MARKER}'. Use the default "
            "(source id + --seed) or a name containing it."
        )


# --------------------------------------------------------------------------- helpers
def _split_patch_id(patch_id):
    """``{slide_id}_{n}`` -> ``(slide_id, int(n))`` (slide_id may contain underscores)."""
    slide_id, suffix = str(patch_id).rsplit("_", 1)
    if not suffix.isdigit():
        raise ValueError(f"Unexpected patch_id (numeric suffix expected): '{patch_id}'")
    return slide_id, int(suffix)


def _list_train_patch_ids(dataset_dir, n_min_cells):
    """TRAIN patch ids from ``cell_count_train.csv`` (or ``cell_count_train_{n_min_cells}.csv``), and the CSV path."""
    import pandas as pd  # local import: keeps the module import pandas-optional
    if n_min_cells and int(n_min_cells) > 0:
        csv = os.path.join(str(dataset_dir), f"cell_count_train_{int(n_min_cells)}.csv")
    else:
        csv = os.path.join(str(dataset_dir), "cell_count_train.csv")
    if not os.path.isfile(csv):
        raise FileNotFoundError(f"TRAIN cell-count CSV not found: {csv}")
    return pd.read_csv(csv)["patch_id"].astype(str).tolist(), csv


def _scc_block_rows(dset):
    """Rows per contiguous ``cls_emb`` read, rounded down to a whole number of HDF5 chunks."""
    chunk_rows = int(dset.chunks[0]) if dset.chunks else _SCC_READ_BLOCK_ROWS
    return max(chunk_rows, (_SCC_READ_BLOCK_ROWS // chunk_rows) * chunk_rows)


def _iter_masked_blocks(dset, mask, pbar=None):
    """Yield ``(abs_row_indices, data)`` for the rows selected by ``mask``, block by block, in ascending order."""
    block_rows = _scc_block_rows(dset)
    for start in range(0, dset.shape[0], block_rows):
        stop = min(start + block_rows, dset.shape[0])
        blk_mask = mask[start:stop]
        n_sel = int(blk_mask.sum())
        if n_sel:
            yield start + np.flatnonzero(blk_mask), dset[start:stop][blk_mask]
        if pbar is not None:
            pbar.update(n_sel)


def _load_json(path):
    with open(path) as f:
        return json.load(f)


def _git_commit():
    try:
        import subprocess
        return subprocess.check_output(
            ["git", "-C", project_root, "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return None


def _human(n_bytes):
    return f"{float(n_bytes) / 1e9:.2f} GB"


# --------------------------------------------------------------------------- the permutation
def build_permutation(slide_index, rng):
    """Uniform random permutation over the canonical train-row list, plus its diagnostics.

    Args:
        slide_index: ``[N]`` int array -- the slide ordinal of each canonical row.
        rng: a seeded ``numpy.random.Generator``.

    Returns:
        ``(perm, stats)``: destination slot ``g`` receives source row ``perm[g]``; stats = n_rows, n_fixed_points,
        frac_cross_slide (fraction of rows that changed slide).
    """
    n = int(np.asarray(slide_index).shape[0])
    if n == 0:
        raise RuntimeError("No TRAIN rows to permute (check the split files / n_min_cells).")
    perm = rng.permutation(n).astype(np.int64, copy=False)
    slide_index = np.asarray(slide_index)
    n_fixed = int(np.count_nonzero(perm == np.arange(n, dtype=np.int64)))
    n_cross = int(np.count_nonzero(slide_index[perm] != slide_index))
    return perm, {
        "n_rows": n,
        "n_fixed_points": n_fixed,
        "frac_cross_slide": float(n_cross) / float(n),
    }


def index_map_sha256(slide_ids, ids_by_slide, perm) -> str:
    """Content hash of the whole index map: the canonical row list and the permutation."""
    h = hashlib.sha256()
    h.update(f"permute_teacher_caches/v{SHUFFLE_SCHEMA_VERSION}\n".encode("utf-8"))
    for sid in slide_ids:
        h.update(str(sid).encode("utf-8"))
        h.update(b"\n")
        h.update(np.ascontiguousarray(np.asarray(ids_by_slide[sid], dtype="<i8")).tobytes())
    h.update(np.ascontiguousarray(np.asarray(perm, dtype="<i8")).tobytes())
    return h.hexdigest()


def _canonical_layout(slide_ids, ids_by_slide):
    """Canonical destination layout: slides in the given (sorted) order, ids ascending within.

    Returns ``(starts, counts, slide_index)``; slide ``i`` owns the contiguous slots from ``starts[i]``.
    """
    counts = np.asarray([int(np.asarray(ids_by_slide[s]).size) for s in slide_ids], dtype=np.int64)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(np.int64) if counts.size else counts
    slide_index = np.repeat(np.arange(counts.size, dtype=np.int32), counts)
    return starts, counts, slide_index


def _shuffle_block(*, seed, source_novae_training_id, source_novae_fingerprint,
                   source_scconcept_training_id, novae_sentinel, scconcept_sentinel,
                   fingerprint_sentinel, novae_stats, scconcept_stats):
    """The ``shuffled_pairing`` provenance block shared by both output manifests."""
    return {
        "schema_version": int(SHUFFLE_SCHEMA_VERSION),
        "created_by": "permute_teacher_caches.py",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "arm": "control - shuffled pairing",
        "permutation_unit": "train_rows_across_slides",
        "seed": int(seed),
        "source_novae_training_id": source_novae_training_id,
        "source_novae_fingerprint": source_novae_fingerprint,
        "source_scconcept_training_id": source_scconcept_training_id,
        "novae_training_id": novae_sentinel,
        "scconcept_training_id": scconcept_sentinel,
        "fingerprint": fingerprint_sentinel,
        "novae": novae_stats,
        "scconcept": scconcept_stats,
    }


# --------------------------------------------------------------------------- Novae st_feat cache
def plan_novae_permutation(*, src_dir, dataset_dir, n_min_cells, seed):
    """Read the Novae train rows, check them against the cache and build the permutation (read-only)."""
    manifest_path = os.path.join(str(src_dir), "manifest.json")
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(
            f"Novae cache manifest not found: {manifest_path}. Run precompute_novae_teacher.py."
        )
    src_manifest = _load_json(manifest_path)
    if int(src_manifest.get("dim", -1)) != OUTPUT_SIZE:
        raise ValueError(f"Novae cache manifest dim={src_manifest.get('dim')} != {OUTPUT_SIZE}.")
    if not src_manifest.get("fingerprint"):
        raise KeyError(f"Novae cache manifest has no 'fingerprint': {manifest_path}")

    # Slides of the cache, from the manifest (not from a directory glob)
    all_slides = sorted((src_manifest.get("n_patches_per_slide") or {}).keys())
    if not all_slides:
        raise KeyError(f"Novae cache manifest has no 'n_patches_per_slide': {manifest_path}")

    train_patch_ids, train_csv = _list_train_patch_ids(dataset_dir, n_min_cells)
    idx_by_slide = {}
    for pid in train_patch_ids:
        sid, pidx = _split_patch_id(pid)
        idx_by_slide.setdefault(sid, []).append(pidx)
    idx_by_slide = {s: np.asarray(sorted(v), dtype=np.int64) for s, v in idx_by_slide.items()}

    unknown = sorted(set(idx_by_slide) - set(all_slides))
    if unknown:
        raise ValueError(
            f"{len(unknown)} slide(s) in {train_csv} have no st_feat cache "
            f"(e.g. {unknown[:5]}). Re-run precompute_novae_teacher.py for this dataset."
        )

    # Row of every train patch in the cache (raises on any missing patch)
    train_slides = [s for s in all_slides if s in idx_by_slide and idx_by_slide[s].size]
    row_pos = {}
    for sid in train_slides:
        h5_path = os.path.join(str(src_dir), f"{sid}.h5")
        if not os.path.isfile(h5_path):
            raise FileNotFoundError(f"Novae st_feat cache missing for slide '{sid}': {h5_path}")
        with h5py.File(h5_path, "r") as f:
            if "patch_idx" not in f or "st_feat" not in f:
                raise KeyError(f"Invalid st_feat cache (missing 'patch_idx'/'st_feat'): {h5_path}")
            patch_idx = f["patch_idx"][:].astype(np.int64, copy=False)
            if int(f["st_feat"].shape[1]) != OUTPUT_SIZE:
                raise ValueError(f"{sid}: st_feat dim {f['st_feat'].shape[1]} != {OUTPUT_SIZE}.")
        want = idx_by_slide[sid]
        pos = np.searchsorted(patch_idx, want)
        pos_clipped = np.clip(pos, 0, patch_idx.shape[0] - 1)
        ok = (pos < patch_idx.shape[0]) & (patch_idx[pos_clipped] == want)
        if not np.all(ok):
            missing = want[~ok]
            raise KeyError(
                f"{sid}: {missing.size} TRAIN patch_idx not present in the st_feat cache "
                f"(e.g. {missing[:5].tolist()}). Re-run precompute_novae_teacher.py --folds "
                "train,valid,test for this dataset."
            )
        row_pos[sid] = pos.astype(np.int64, copy=False)

    starts, counts, slide_index = _canonical_layout(train_slides, idx_by_slide)
    perm, stats = build_permutation(slide_index, np.random.default_rng([int(seed), 0]))
    stats.update({
        "train_row_source": os.path.basename(train_csv),
        "n_min_cells": int(n_min_cells),
        "n_train_slides": len(train_slides),
        "index_map_sha256": index_map_sha256(train_slides, idx_by_slide, perm),
    })
    return {
        "src_manifest": src_manifest,
        "all_slides": all_slides,
        "train_slides": train_slides,
        "idx_by_slide": idx_by_slide,
        "row_pos": row_pos,
        "starts": starts,
        "counts": counts,
        "perm": perm,
        "stats": stats,
    }


def permute_novae_cache(*, plan, src_dir, dst_dir, novae_sentinel, scconcept_sentinel,
                        fingerprint_sentinel, shuffle_block, limit_slides=None):
    """Write the permuted Novae st_feat cache (payload only; ``patch_idx`` copied verbatim)."""
    os.makedirs(str(dst_dir), exist_ok=True)
    train_slides, row_pos, perm = plan["train_slides"], plan["row_pos"], plan["perm"]
    starts, counts = plan["starts"], plan["counts"]

    # --- gather the train payload in canonical order ---------------------------------------
    n_rows = int(counts.sum())
    src_payload = np.empty((n_rows, OUTPUT_SIZE), dtype=np.float32)
    for i, sid in enumerate(tqdm(train_slides, desc="[novae] read", unit="slide")):
        with h5py.File(os.path.join(str(src_dir), f"{sid}.h5"), "r") as f:
            src_payload[starts[i]:starts[i] + counts[i]] = f["st_feat"][:][row_pos[sid]]
    dst_payload = src_payload[perm]

    # Check that the column sums are unchanged by the permutation
    _assert_colsum_match(src_payload.sum(axis=0, dtype=np.float64),
                         dst_payload.sum(axis=0, dtype=np.float64), "novae st_feat")
    del src_payload

    # --- write every slide (train rows overwritten, everything else verbatim) ----------------
    slides = plan["all_slides"] if limit_slides is None else plan["all_slides"][:int(limit_slides)]
    dst_index = {sid: i for i, sid in enumerate(train_slides)}
    n_written = 0
    for sid in tqdm(slides, desc="[novae] write", unit="slide"):
        src_path = os.path.join(str(src_dir), f"{sid}.h5")
        dst_path = os.path.join(str(dst_dir), f"{sid}.h5")
        tmp_path = dst_path + ".tmp"
        with h5py.File(src_path, "r") as fs:
            patch_idx = fs["patch_idx"][:]
            st_feat = fs["st_feat"][:]
            src_attrs = dict(fs.attrs)
        if sid in dst_index:
            i = dst_index[sid]
            st_feat[row_pos[sid]] = dst_payload[starts[i]:starts[i] + counts[i]]
            n_written += int(counts[i])
        with h5py.File(tmp_path, "w") as fd:
            fd.create_dataset("patch_idx", data=patch_idx, dtype="int64")
            fd.create_dataset("st_feat", data=st_feat, dtype="float32")
            for k, v in src_attrs.items():
                fd.attrs[k] = v
            fd.attrs["fingerprint"] = fingerprint_sentinel
            fd.attrs["novae_training_id"] = novae_sentinel
            fd.attrs["scconcept_training_id"] = scconcept_sentinel
            _stamp_shuffle_attrs(fd, shuffle_block, plan["stats"], teacher="novae")
        os.replace(tmp_path, dst_path)

    if limit_slides is None and n_written != n_rows:
        raise RuntimeError(f"Novae: wrote {n_written} train rows, expected {n_rows}.")
    logger.info("Novae: %d slide(s) written, %d TRAIN rows permuted (%.1f%% changed slide, "
                "%d fixed point(s)).", len(slides), n_written,
                100.0 * plan["stats"]["frac_cross_slide"], plan["stats"]["n_fixed_points"])


# --------------------------------------------------------------------------- scConcept store
def plan_scconcept_permutation(*, src_dir, cell_ids_sets_path, seed):
    """Read the scConcept train rows, check them against the store and build the permutation (read-only).

    TRAIN cells: ``cell_ids_sets.npz::train_only_by_slide`` (cells in no valid patch; independent of n_min_cells).
    """
    if not os.path.isfile(cell_ids_sets_path):
        raise FileNotFoundError(f"cell_ids_sets.npz not found: {cell_ids_sets_path}")
    z = np.load(str(cell_ids_sets_path), allow_pickle=True)
    if "train_only_by_slide" not in z:
        raise KeyError(f"'train_only_by_slide' missing in {cell_ids_sets_path}")
    train_by_slide_raw = z["train_only_by_slide"].item()

    # Slide list from the npz keys (not from a directory glob)
    all_slides = sorted(train_by_slide_raw.keys())
    ids_by_slide = {s: np.unique(np.asarray(train_by_slide_raw[s], dtype=np.int64))
                    for s in all_slides}
    train_slides = [s for s in all_slides if ids_by_slide[s].size]

    row_pos, n_rows_by_slide, emb_dims = {}, {}, set()
    for sid in all_slides:
        h5_path = os.path.join(str(src_dir), f"{sid}.h5")
        if not os.path.isfile(h5_path):
            raise FileNotFoundError(f"scConcept embeddings not found for slide '{sid}': {h5_path}")
        with h5py.File(h5_path, "r") as f:
            if "cell_id_int" not in f or "cls_emb" not in f:
                raise KeyError(f"Invalid scConcept embeddings (missing 'cell_id_int'/'cls_emb'): {h5_path}")
            cell_id_int = f["cell_id_int"][:].astype(np.int64, copy=False)
            emb_dims.add(int(f.attrs.get("embedding_dim", f["cls_emb"].shape[1])))
            n_rows_by_slide[sid] = int(f["cls_emb"].shape[0])
        # searchsorted lookups (here and in the training dataset) require ascending ids.
        if cell_id_int.size > 1 and np.any(cell_id_int[1:] < cell_id_int[:-1]):
            raise ValueError(
                f"{sid}: scConcept 'cell_id_int' is not sorted ascending in {h5_path}, but lookups "
                "use np.searchsorted(). Re-export the embeddings sorted by cell_id_int."
            )
        want = ids_by_slide[sid]
        if want.size == 0:
            row_pos[sid] = np.empty(0, dtype=np.int64)
            continue
        pos = np.searchsorted(cell_id_int, want)
        pos_clipped = np.clip(pos, 0, cell_id_int.shape[0] - 1)
        ok = (pos < cell_id_int.shape[0]) & (cell_id_int[pos_clipped] == want)
        if not np.all(ok):
            missing = want[~ok]
            raise KeyError(
                f"{sid}: {missing.size} TRAIN cell_id_int not present in the scConcept store "
                f"(e.g. {missing[:5].tolist()})."
            )
        row_pos[sid] = pos.astype(np.int64, copy=False)

    if len(emb_dims) != 1:
        raise ValueError(f"Inconsistent scConcept embedding dims across slides: {sorted(emb_dims)}")
    emb_dim = emb_dims.pop()
    if emb_dim != EMBEDDING_SIZE:
        raise ValueError(f"scConcept embedding_dim={emb_dim} != {EMBEDDING_SIZE}.")

    starts, counts, slide_index = _canonical_layout(train_slides, ids_by_slide)
    perm, stats = build_permutation(slide_index, np.random.default_rng([int(seed), 1]))
    stats.update({
        "train_row_source": "cell_ids_sets.npz::train_only_by_slide",
        "n_min_cells": None,   # the scConcept train set is cell-set-driven, not patch-filtered
        "n_train_slides": len(train_slides),
        "index_map_sha256": index_map_sha256(train_slides, ids_by_slide, perm),
    })
    return {
        "all_slides": all_slides,
        "train_slides": train_slides,
        "ids_by_slide": ids_by_slide,
        "row_pos": row_pos,
        "n_rows_by_slide": n_rows_by_slide,
        "emb_dim": emb_dim,
        "starts": starts,
        "counts": counts,
        "perm": perm,
        "stats": stats,
    }


def permute_scconcept_store(*, plan, src_dir, dst_dir, scconcept_sentinel, shuffle_block,
                            dest_slide_chunks=2, dest_compression="none", resume=False,
                            limit_slides=None):
    """Write the permuted scConcept store (``cls_emb`` payload only; ``cell_id_int`` verbatim).

    Destination slides are built in ``dest_slide_chunks`` groups; for each group the source slides are streamed once.
    """
    os.makedirs(str(dst_dir), exist_ok=True)
    train_slides, row_pos, perm = plan["train_slides"], plan["row_pos"], plan["perm"]
    starts, counts, emb_dim = plan["starts"], plan["counts"], plan["emb_dim"]
    n_rows_total = int(counts.sum())
    dst_index = {sid: i for i, sid in enumerate(train_slides)}

    slides = plan["all_slides"] if limit_slides is None else plan["all_slides"][:int(limit_slides)]
    todo = [s for s in slides if not (resume and _already_shuffled(
        os.path.join(str(dst_dir), f"{s}.h5"), shuffle_block, plan["stats"]))]
    if len(todo) != len(slides):
        logger.info("scConcept: --resume skips %d slide(s) already built with this index map.",
                    len(slides) - len(todo))

    # Slides with no TRAIN cells: copied (HistoStDataset needs one file per slide)
    plain = [s for s in todo if s not in dst_index]
    for sid in tqdm(plain, desc="[scConcept] copy", unit="slide") if plain else []:
        _copy_and_stamp(os.path.join(str(src_dir), f"{sid}.h5"),
                        os.path.join(str(dst_dir), f"{sid}.h5"),
                        shuffle_block, plan["stats"], teacher="scconcept",
                        extra_attrs={"scconcept_training_id": scconcept_sentinel})

    dest_todo = [s for s in todo if s in dst_index]
    if not dest_todo:
        logger.info("scConcept: nothing to permute.")
        return

    n_groups = max(1, min(int(dest_slide_chunks), len(dest_todo)))
    groups = [list(g) for g in np.array_split(np.asarray(dest_todo, dtype=object), n_groups)]
    sum_read = np.zeros(emb_dim, dtype=np.float64)
    sum_written = np.zeros(emb_dim, dtype=np.float64)
    n_scattered = 0

    for gi, group in enumerate(groups, 1):
        gslots = np.concatenate([
            np.arange(starts[dst_index[s]], starts[dst_index[s]] + counts[dst_index[s]],
                      dtype=np.int64) for s in group])
        gsrc = perm[gslots]                       # global source row of each destination slot
        bufs = {s: np.empty((int(counts[dst_index[s]]), emb_dim), dtype=np.float16) for s in group}
        covered = {s: np.zeros(int(counts[dst_index[s]]), dtype=bool) for s in group}
        logger.info("scConcept group %d/%d: %d slide(s), %d rows (~%s of buffers).",
                    gi, n_groups, len(group), int(gslots.size),
                    _human(int(gslots.size) * emb_dim * 2))

        with tqdm(total=int(gslots.size), desc=f"[scConcept] gather {gi}/{n_groups}",
                  unit="cell", unit_scale=True) as pbar:
            for si, ssid in enumerate(train_slides):
                lo, hi = int(starts[si]), int(starts[si] + counts[si])
                sel = (gsrc >= lo) & (gsrc < hi)
                if not sel.any():
                    continue
                # Source rows to read from this slide, and where each lands.
                src_rows = row_pos[ssid][gsrc[sel] - lo]
                dest_slot_for_row = np.full(plan["n_rows_by_slide"][ssid], -1, dtype=np.int64)
                dest_slot_for_row[src_rows] = gslots[sel]
                mask = np.zeros(plan["n_rows_by_slide"][ssid], dtype=bool)
                mask[src_rows] = True
                pbar.set_postfix_str(ssid, refresh=False)
                with h5py.File(os.path.join(str(src_dir), f"{ssid}.h5"), "r") as fs:
                    for abs_idx, block in _iter_masked_blocks(fs["cls_emb"], mask, pbar):
                        sum_read += block.astype(np.float64).sum(axis=0)
                        dslots = dest_slot_for_row[abs_idx]
                        # Destination slot -> (slide, local row): slides own contiguous blocks.
                        which = np.searchsorted(starts, dslots, side="right") - 1
                        local = dslots - starts[which]
                        for u in np.unique(which):
                            m = which == u
                            dsid = train_slides[int(u)]
                            bufs[dsid][local[m]] = block[m]
                            covered[dsid][local[m]] = True
                        n_scattered += int(abs_idx.size)

        for sid in group:
            if not covered[sid].all():
                raise RuntimeError(
                    f"scConcept: {int((~covered[sid]).sum())} destination row(s) of slide '{sid}' "
                    "were never written -- the scatter is incomplete, refusing to write."
                )
            sum_written += bufs[sid].astype(np.float64).sum(axis=0)
            _write_scconcept_slide(
                src_path=os.path.join(str(src_dir), f"{sid}.h5"),
                dst_path=os.path.join(str(dst_dir), f"{sid}.h5"),
                buf=bufs[sid], buf_rows=row_pos[sid], dest_compression=dest_compression,
                shuffle_block=shuffle_block, stats=plan["stats"],
                extra_attrs={"scconcept_training_id": scconcept_sentinel},
            )
        del bufs, covered

    if limit_slides is None:
        if n_scattered != n_rows_total:
            raise RuntimeError(
                f"scConcept: scattered {n_scattered} rows, expected {n_rows_total}.")
        _assert_colsum_match(sum_read, sum_written, "scConcept cls_emb")
    logger.info("scConcept: %d slide(s) written, %d TRAIN rows permuted (%.1f%% changed slide, "
                "%d fixed point(s)).", len(todo), n_scattered,
                100.0 * plan["stats"]["frac_cross_slide"], plan["stats"]["n_fixed_points"])


def _write_scconcept_slide(*, src_path, dst_path, buf, buf_rows, dest_compression,
                           shuffle_block, stats, extra_attrs):
    """Write one destination ``{slide}.h5`` (ids verbatim, ``cls_emb`` with train rows replaced), slab by slab and atomically."""
    tmp_path = str(dst_path) + ".tmp"
    with h5py.File(src_path, "r") as fs:
        d_src = fs["cls_emb"]
        n_rows = int(d_src.shape[0])
        # Row -> position in the permuted buffer (-1 = not a train row, copied verbatim).
        buf_pos = np.full(n_rows, -1, dtype=np.int64)
        buf_pos[buf_rows] = np.arange(buf.shape[0], dtype=np.int64)

        kw = {"chunks": d_src.chunks}
        if str(dest_compression) == "same":
            kw.update({"compression": d_src.compression,
                       "compression_opts": d_src.compression_opts,
                       "shuffle": d_src.shuffle})
        with h5py.File(tmp_path, "w") as fd:
            ids = fs["cell_id_int"]
            fd.create_dataset("cell_id_int", data=ids[:], dtype=ids.dtype,
                              chunks=ids.chunks, compression=ids.compression,
                              compression_opts=ids.compression_opts, shuffle=ids.shuffle)
            d_dst = fd.create_dataset("cls_emb", shape=d_src.shape, dtype=d_src.dtype, **kw)
            block_rows = _scc_block_rows(d_src)
            for start in range(0, n_rows, block_rows):
                stop = min(start + block_rows, n_rows)
                slab = d_src[start:stop]
                bp = buf_pos[start:stop]
                sel = bp >= 0
                if sel.any():
                    slab[sel] = buf[bp[sel]]
                d_dst[start:stop] = slab
            for k, v in fs.attrs.items():
                fd.attrs[k] = v
            for k, v in (extra_attrs or {}).items():
                fd.attrs[k] = v
            _stamp_shuffle_attrs(fd, shuffle_block, stats, teacher="scconcept")
    os.replace(tmp_path, dst_path)


def _copy_and_stamp(src_path, dst_path, shuffle_block, stats, teacher, extra_attrs=None):
    """Byte-copy a slide with no TRAIN rows, then stamp the provenance attrs."""
    tmp_path = str(dst_path) + ".tmp"
    shutil.copyfile(src_path, tmp_path)
    with h5py.File(tmp_path, "a") as fd:
        for k, v in (extra_attrs or {}).items():
            fd.attrs[k] = v
        _stamp_shuffle_attrs(fd, shuffle_block, stats, teacher=teacher)
    os.replace(tmp_path, dst_path)


def _stamp_shuffle_attrs(fd, shuffle_block, stats, teacher):
    """Write the permutation provenance attrs (seed, index-map hash, sources) to one slide file."""
    fd.attrs["shuffled_pairing"] = "true"
    fd.attrs["shuffle_schema_version"] = int(SHUFFLE_SCHEMA_VERSION)
    fd.attrs["shuffle_seed"] = int(shuffle_block["seed"])
    fd.attrs["shuffle_teacher"] = str(teacher)
    fd.attrs["shuffle_permutation_unit"] = shuffle_block["permutation_unit"]
    fd.attrs["shuffle_index_map_sha256"] = stats["index_map_sha256"]
    fd.attrs["shuffle_source_novae_training_id"] = shuffle_block["source_novae_training_id"]
    fd.attrs["shuffle_source_scconcept_training_id"] = shuffle_block["source_scconcept_training_id"]


def _already_shuffled(dst_path, shuffle_block, stats) -> bool:
    """True if ``dst_path`` was already built from this exact seed + index map (``--resume``)."""
    if not os.path.isfile(dst_path):
        return False
    try:
        with h5py.File(dst_path, "r") as f:
            return (str(f.attrs.get("shuffle_index_map_sha256", "")) == stats["index_map_sha256"]
                    and int(f.attrs.get("shuffle_seed", -1)) == int(shuffle_block["seed"]))
    except Exception:
        return False


def _assert_colsum_match(a, b, what):
    """Raise if the fp64 column sums before/after the permutation differ by more than _COLSUM_RTOL."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    scale = max(float(np.abs(a).max()), 1e-30)
    dev = float(np.abs(a - b).max()) / scale
    if not np.isfinite(dev) or dev > _COLSUM_RTOL:
        raise RuntimeError(
            f"{what}: column sums changed under the permutation (max relative deviation {dev:.3e} "
            f"> {_COLSUM_RTOL:.0e}): the train multiset was not preserved."
        )
    logger.info("%s: train-row column sums preserved (max relative deviation %.2e).", what, dev)


# --------------------------------------------------------------------------- verify_means
def verify_means(path_a, path_b):
    """True iff the mean arrays of two ``teacher_center_means.h5`` files are bit-identical (attrs are not compared)."""
    ok = True
    with h5py.File(str(path_a), "r") as fa, h5py.File(str(path_b), "r") as fb:
        for key in ("novae_mean", "scconcept_mean"):
            in_a, in_b = key in fa, key in fb
            if not in_a and not in_b:
                logger.warning("%s: absent from both files (not checked).", key)
                continue
            if in_a != in_b:
                logger.error("%s: present in %s but not the other file.", key, "A" if in_a else "B")
                ok = False
                continue
            A, B = fa[key][:], fb[key][:]
            if A.shape != B.shape or A.dtype != B.dtype:
                logger.error("%s: shape/dtype differ (%s %s vs %s %s).",
                             key, A.shape, A.dtype, B.shape, B.dtype)
                ok = False
                continue
            if np.array_equal(A, B):
                logger.info("%s: bit-identical (%d values, %s). PASS", key, A.size, A.dtype)
                continue
            ok = False
            # Diagnostics of the difference (the check itself is exact equality)
            n_diff = int(np.count_nonzero(A != B))
            max_abs = float(np.abs(A.astype(np.float64) - B.astype(np.float64)).max())
            ulp = ""
            if A.dtype == np.float32:
                d = np.abs(A.view(np.int32).astype(np.int64) - B.view(np.int32).astype(np.int64))
                ulp = f", max {int(d.max())} ulp"
            logger.error("%s: differs in %d/%d value(s) (max |delta| %.3e%s). FAIL",
                         key, n_diff, A.size, max_abs, ulp)
    return ok


# --------------------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser(
        description="Build the permuted (shuffled-pairing) ST-teacher caches, or compare their "
                    "teacher-centering means with the source ones (--mode verify_means)."
    )
    parser.add_argument("--mode", type=str, choices=["build", "verify_means"], default="build",
                        help="'build' (default): write the permuted caches; 'verify_means': check that the "
                             "permuted and source teacher-centering means are bit-identical.")
    parser.add_argument("--dataset_id", type=str, required=True,
                        help="Dataset id, e.g. ds_ct1_20x_224_o112.")
    parser.add_argument("--training_dataset_id", type=str, required=True,
                        help="Training dataset id, e.g. ds_ct1_20x_224_o112_1 (defines the TRAIN split).")
    parser.add_argument("--novae_training_id", type=str, required=True,
                        help="Source Novae training id whose st_feat cache is permuted (e.g. training_4).")
    parser.add_argument("--scconcept_training_id", type=str, default=None,
                        help="Source scConcept training id (default: the one of the Novae cache manifest, "
                             "which it must match).")
    parser.add_argument("--n_min_cells", type=int, required=True,
                        help="Min-cell filter defining the Novae train rows (must match data_select.n_min_cells "
                             "of the training run).")
    parser.add_argument("--seed", type=int, default=1234,
                        help="Seed of the permutation (default 1234).")
    parser.add_argument("--out_novae_training_id", type=str, default=None,
                        help="Sentinel Novae id = the output cache folder name. "
                             "Default '{novae_training_id}_SHUFFLED_seed{seed}'. Must contain 'SHUFFLED'.")
    parser.add_argument("--out_scconcept_training_id", type=str, default=None,
                        help="Sentinel scConcept id = the output embeddings folder name. "
                             "Default '{scconcept_training_id}_SHUFFLED_seed{seed}'. Must contain 'SHUFFLED'.")
    parser.add_argument("--dest_slide_chunks", type=int, default=2,
                        help="Number of groups of destination scConcept slides (peak RAM ~ payload / N, source "
                             "read N times; default 2).")
    parser.add_argument("--dest_compression", type=str, choices=["none", "same"], default="none",
                        help="scConcept payload compression: 'none' (default, source chunking without gzip) "
                             "or 'same' (source layout).")
    parser.add_argument("--overwrite", action="store_true",
                        help="Allow writing into an existing output folder (default: refuse).")
    parser.add_argument("--resume", action="store_true",
                        help="Skip destination slides already built from this exact seed + index map.")
    parser.add_argument("--dry_run", action="store_true",
                        help="Plan only: report the permutation diagnostics, output size and free space, "
                             "without writing.")
    parser.add_argument("--limit_slides", type=int, default=None,
                        help="Write only the first N slides (no manifests).")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )

    # --- paths ------------------------------------------------------------------------------
    dataset_dir = get_full_path("train", "training_datasets", args.dataset_id,
                                args.training_dataset_id, as_str=True, scratch=True)
    src_cache_dir = get_full_path("train", "training_datasets", args.dataset_id,
                                  args.training_dataset_id, "novae_teacher_cache",
                                  args.novae_training_id, as_str=True, scratch=True)
    if not os.path.isdir(src_cache_dir):
        raise FileNotFoundError(f"Novae teacher cache dir not found: {src_cache_dir}. "
                                "Run precompute_novae_teacher.py.")

    src_manifest = _load_json(os.path.join(src_cache_dir, "manifest.json"))
    src_scc_id = args.scconcept_training_id or src_manifest.get("scconcept_training_id")
    if not src_scc_id:
        raise KeyError("--scconcept_training_id not given and absent from the Novae cache manifest.")
    if args.scconcept_training_id and args.scconcept_training_id != src_manifest.get("scconcept_training_id"):
        raise ValueError(
            f"--scconcept_training_id={args.scconcept_training_id} != the Novae cache manifest's "
            f"{src_manifest.get('scconcept_training_id')!r}. The cell teacher must match the "
            "scConcept model the Novae node features came from."
        )

    novae_sentinel = args.out_novae_training_id or shuffled_sentinel(args.novae_training_id, args.seed)
    scc_sentinel = args.out_scconcept_training_id or shuffled_sentinel(src_scc_id, args.seed)
    _assert_marked(novae_sentinel, "--out_novae_training_id")
    _assert_marked(scc_sentinel, "--out_scconcept_training_id")
    fingerprint_sentinel = shuffled_sentinel(str(src_manifest.get("fingerprint")), args.seed)

    dst_cache_dir = get_full_path("train", "training_datasets", args.dataset_id,
                                  args.training_dataset_id, "novae_teacher_cache",
                                  novae_sentinel, as_str=True, scratch=True)

    # --- verify_means -----------------------------------------------------------------------
    if args.mode == "verify_means":
        a = os.path.join(src_cache_dir, "teacher_center_means.h5")
        b = os.path.join(dst_cache_dir, "teacher_center_means.h5")
        for p, label in ((a, "source"), (b, "permuted")):
            if not os.path.isfile(p):
                raise FileNotFoundError(
                    f"{label} teacher_center_means.h5 not found: {p}. Run "
                    f"precompute_teacher_mean.py --novae_training_id "
                    f"{args.novae_training_id if label == 'source' else novae_sentinel} "
                    f"--n_min_cells {args.n_min_cells}."
                )
        logger.info("Checking that the teacher-centering means are bit-identical.")
        logger.info("  A (source)   : %s", a)
        logger.info("  B (permuted) : %s", b)
        if verify_means(a, b):
            logger.info("PASSED: the permuted and source teacher-centering means are "
                        "bit-identical.")
            return
        raise SystemExit(
            "FAILED: the permuted teacher-centering means are not bit-identical to the source ones "
            "(TRAIN multiset not preserved, or means recomputed with a different --n_min_cells / --which)."
        )

    src_scc_dir = get_full_path("train", "trainings", "trainings_scConcept", src_scc_id,
                                "embeddings", as_str=True)
    dst_scc_root = get_full_path("train", "trainings", "trainings_scConcept", scc_sentinel, as_str=True)
    dst_scc_dir = os.path.join(dst_scc_root, "embeddings")
    if not os.path.isdir(src_scc_dir):
        raise FileNotFoundError(f"scConcept embeddings dir not found: {src_scc_dir}")

    logger.info("Shuffled-pairing caches, seed %d", args.seed)
    logger.info("  novae      : %s  ->  %s", args.novae_training_id, novae_sentinel)
    logger.info("  scConcept  : %s  ->  %s", src_scc_id, scc_sentinel)
    logger.info("  fingerprint: %s  ->  %s", src_manifest.get("fingerprint"), fingerprint_sentinel)

    # --- plan both teachers (read-only) -----------------------------------------------------
    novae_plan = plan_novae_permutation(src_dir=src_cache_dir, dataset_dir=dataset_dir,
                                        n_min_cells=args.n_min_cells, seed=args.seed)
    scc_plan = plan_scconcept_permutation(
        src_dir=src_scc_dir,
        cell_ids_sets_path=os.path.join(dataset_dir, "cell_ids_sets.npz"),
        seed=args.seed,
    )
    for label, st in (("novae", novae_plan["stats"]), ("scConcept", scc_plan["stats"])):
        logger.info("  %-9s: %d TRAIN rows over %d slide(s) from %s | %.2f%% changed slide, "
                    "%d fixed point(s) | index_map=%s",
                    label, st["n_rows"], st["n_train_slides"], st["train_row_source"],
                    100.0 * st["frac_cross_slide"], st["n_fixed_points"],
                    st["index_map_sha256"][:16])

    # --- size + free-space check ------------------------------------------------------------
    novae_bytes = sum(os.path.getsize(os.path.join(src_cache_dir, f"{s}.h5"))
                      for s in novae_plan["all_slides"])
    if args.dest_compression == "same":
        scc_bytes = sum(os.path.getsize(os.path.join(src_scc_dir, f"{s}.h5"))
                        for s in scc_plan["all_slides"])
    else:
        scc_bytes = sum(int(plan_rows) * (scc_plan["emb_dim"] * 2 + 8)
                        for plan_rows in scc_plan["n_rows_by_slide"].values())
    need = int((novae_bytes + scc_bytes) * 1.10)
    free = shutil.disk_usage(os.path.dirname(os.path.dirname(str(dst_cache_dir)))).free
    logger.info("  output size: ~%s novae + ~%s scConcept (+10%% margin => %s); free: %s",
                _human(novae_bytes), _human(scc_bytes), _human(need), _human(free))
    peak = int(scc_plan["stats"]["n_rows"]) * scc_plan["emb_dim"] * 2 / max(1, int(args.dest_slide_chunks))
    logger.info("  peak scConcept buffer RAM ~%s at --dest_slide_chunks %d",
                _human(peak), int(args.dest_slide_chunks))
    if free < need:
        raise RuntimeError(f"Not enough free space: need ~{_human(need)}, have {_human(free)}.")

    if args.dry_run:
        logger.info("Dry run: nothing written (drop --dry_run to build).")
        return

    for d in (dst_cache_dir, dst_scc_dir):
        if os.path.isdir(d) and os.listdir(d) and not (args.overwrite or args.resume):
            raise FileExistsError(f"{d} exists and is not empty; pass --overwrite (or --resume).")

    shuffle_block = _shuffle_block(
        seed=args.seed,
        source_novae_training_id=args.novae_training_id,
        source_novae_fingerprint=src_manifest.get("fingerprint"),
        source_scconcept_training_id=src_scc_id,
        novae_sentinel=novae_sentinel, scconcept_sentinel=scc_sentinel,
        fingerprint_sentinel=fingerprint_sentinel,
        novae_stats=novae_plan["stats"], scconcept_stats=scc_plan["stats"],
    )

    # --- build ------------------------------------------------------------------------------
    permute_novae_cache(plan=novae_plan, src_dir=src_cache_dir, dst_dir=dst_cache_dir,
                        novae_sentinel=novae_sentinel, scconcept_sentinel=scc_sentinel,
                        fingerprint_sentinel=fingerprint_sentinel, shuffle_block=shuffle_block,
                        limit_slides=args.limit_slides)
    permute_scconcept_store(plan=scc_plan, src_dir=src_scc_dir, dst_dir=dst_scc_dir,
                            scconcept_sentinel=scc_sentinel, shuffle_block=shuffle_block,
                            dest_slide_chunks=args.dest_slide_chunks,
                            dest_compression=args.dest_compression, resume=args.resume,
                            limit_slides=args.limit_slides)

    if args.limit_slides is not None:
        logger.info("--limit_slides: manifests not written.")
        return

    write_manifests(dst_cache_dir=dst_cache_dir, dst_scc_root=dst_scc_root,
                    src_manifest=src_manifest, src_scc_root=os.path.dirname(src_scc_dir),
                    novae_sentinel=novae_sentinel, scconcept_sentinel=scc_sentinel,
                    fingerprint_sentinel=fingerprint_sentinel, shuffle_block=shuffle_block,
                    novae_plan=novae_plan, scc_plan=scc_plan)

    logger.info("Done.")
    logger.info("  permuted novae cache : %s", dst_cache_dir)
    logger.info("  permuted scConcept   : %s", dst_scc_dir)
    logger.info("Next steps:")
    logger.info("  1. python3 src/dataset/precompute_teacher_mean.py --dataset_id %s "
                "--training_dataset_id %s --novae_training_id %s --n_min_cells %d",
                args.dataset_id, args.training_dataset_id, novae_sentinel, args.n_min_cells)
    logger.info("  2. python3 src/dataset/permute_teacher_caches.py --mode verify_means "
                "--dataset_id %s --training_dataset_id %s --novae_training_id %s "
                "--n_min_cells %d --seed %d",
                args.dataset_id, args.training_dataset_id, args.novae_training_id,
                args.n_min_cells, args.seed)


def write_manifests(*, dst_cache_dir, dst_scc_root, src_manifest, src_scc_root,
                    novae_sentinel, scconcept_sentinel, fingerprint_sentinel,
                    shuffle_block, novae_plan, scc_plan):
    """Write both sentinel manifests (the Novae one is the source manifest with the identity fields replaced)."""
    manifest = dict(src_manifest)
    manifest["novae_training_id"] = novae_sentinel
    manifest["scconcept_training_id"] = scconcept_sentinel
    manifest["fingerprint"] = fingerprint_sentinel
    manifest["created_by"] = "permute_teacher_caches.py"
    manifest["created_utc"] = shuffle_block["created_utc"]
    manifest["source_created_by"] = src_manifest.get("created_by")
    manifest["n_patches_per_slide"] = {s: int(n) for s, n in
                                       (src_manifest.get("n_patches_per_slide") or {}).items()}
    manifest["shuffled_pairing"] = shuffle_block
    with open(os.path.join(str(dst_cache_dir), "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Wrote %s", os.path.join(str(dst_cache_dir), "manifest.json"))

    scc_manifest = {}
    src_scc_manifest_path = os.path.join(str(src_scc_root), "manifest.json")
    if os.path.isfile(src_scc_manifest_path):
        scc_manifest = dict(_load_json(src_scc_manifest_path))
        # Drop the per-slide block (absolute paths into the source store)
        scc_manifest.pop("slides", None)
        scc_manifest["source_slides_block_dropped"] = (
            "the source manifest's 'slides' map held absolute paths into the source store")
    scc_manifest["scconcept_training_id"] = scconcept_sentinel
    scc_manifest["source_scconcept_training_id"] = shuffle_block["source_scconcept_training_id"]
    scc_manifest["created_by"] = "permute_teacher_caches.py"
    scc_manifest["created_utc"] = shuffle_block["created_utc"]
    scc_manifest["n_slides"] = len(scc_plan["all_slides"])
    scc_manifest["embedding_size"] = int(scc_plan["emb_dim"])
    scc_manifest["shuffled_pairing"] = shuffle_block
    os.makedirs(str(dst_scc_root), exist_ok=True)
    with open(os.path.join(str(dst_scc_root), "manifest.json"), "w") as f:
        json.dump(scc_manifest, f, indent=2)
    logger.info("Wrote %s", os.path.join(str(dst_scc_root), "manifest.json"))


if __name__ == "__main__":
    main()
