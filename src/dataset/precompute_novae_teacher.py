#!/usr/bin/env python
"""
Precompute the st_feat of the frozen Novae ST teacher (encoder output on the patch cell graph, with scConcept
node features) for every patch of a training dataset. Runs in the `novae` env.

Saved in <ROOT_HistoST> / train / training_datasets / <dataset_id> / <training_dataset_id> / novae_teacher_cache / <novae_training_id> /
    - <slide_id>.h5: patch_idx [n], st_feat [n, 128]
    - manifest.json
"""

import os
import sys
import json
import logging
import argparse
import zipfile
import io
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.sparse import load_npz
import h5py
import torch
import anndata as ad
from tqdm import tqdm

# Add the project root to sys.path so that 'src' can be imported.
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import novae  # noqa: E402  (novae env only)

from src.utils.constants import get_full_path  # noqa: E402
from src.dataset.novae_teacher import (  # noqa: E402
    build_novae_teacher_graph,
    novae_teacher_fingerprint,
    fetch_scconcept_for_cells,
    list_fold_patch_ids,
    read_patch_cell_ids,
    align_mask_cells_to_obs,
    patch_level_coords,
    inside_patch_keep_idx,
    NOVAE_CELLS_CHARACTERISTIC_DISTANCE,
    NOVAE_SCALE_TO_MICRONS,
    NOVAE_TEACHER_SCHEMA_VERSION,
)

logger = logging.getLogger(__name__)

OUTPUT_SIZE = 128       # Novae GAT output_size (= st_feat dim) for corpus40M-model30M teachers
EMBEDDING_SIZE = 512    # Novae node-feature (scConcept) dim


# --------------------------------------------------------------------------- small helpers
def _split_patch_id(patch_id: str):
    """``{slide_id}_{n}`` -> ``(slide_id, int(n))`` (slide_id may contain underscores)."""
    slide_id, suffix = str(patch_id).rsplit("_", 1)
    if not suffix.isdigit():
        raise ValueError(f"Unexpected patch_id (numeric suffix expected): '{patch_id}'")
    return slide_id, int(suffix)


def _load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def _git_commit():
    try:
        import subprocess
        return subprocess.check_output(
            ["git", "-C", project_root, "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return None


# --------------------------------------------------------------------------- provenance
def read_teacher_provenance(novae_training_id: str, scconcept_training_id_arg: str | None) -> dict:
    """
    Read the scConcept provenance of the Novae teacher from its ``manifest.json`` (written by ``_2_novae_pretraining.py``).
    ``scconcept_training_id`` defaults to, and must match, the scConcept model the teacher was trained with.
    """
    manifest_path = get_full_path("train", "trainings", "trainings_Novae", novae_training_id, "manifest.json")
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(
            f"Novae manifest not found: {manifest_path} (written by _2_novae_pretraining.py)."
        )
    manifest = _load_json(manifest_path)
    scc = manifest.get("scConcept") or {}
    own_scc_id = scc.get("own_scconcept_training_id")
    if own_scc_id is None:
        raise KeyError(f"Novae manifest {manifest_path} has no scConcept.own_scconcept_training_id.")

    scconcept_training_id = scconcept_training_id_arg or own_scc_id
    if scconcept_training_id != own_scc_id:
        raise ValueError(
            f"--scconcept_training_id={scconcept_training_id} does not match the Novae teacher's "
            f"own_scconcept_training_id={own_scc_id} (the node features must come from the scConcept "
            "model the teacher was trained with)."
        )

    if int(scc.get("adapt_steps", -1)) != 0:
        raise ValueError(f"scConcept teacher must be raw (adapt_steps==0); got {scc.get('adapt_steps')}.")
    if int(scc.get("embedding_size", -1)) != EMBEDDING_SIZE:
        raise ValueError(f"scConcept embedding_size={scc.get('embedding_size')} != {EMBEDDING_SIZE}.")

    model_kwargs = manifest.get("model_kwargs") or {}
    graph = manifest.get("graph") or {}
    return {
        "scconcept_training_id": scconcept_training_id,
        "scconcept_model_name": scc.get("model_name"),
        "scconcept_model_version": scc.get("scConcept_version"),
        "scconcept_species": scc.get("species"),
        "manifest_output_size": model_kwargs.get("output_size"),
        "manifest_embedding_size": model_kwargs.get("embedding_size"),
        "manifest_radius_um": graph.get("radius_um"),
        "novae_manifest_path": manifest_path,
    }


def resolve_ckpt_path(novae_training_id: str, ckpt_name: str | None) -> str:
    """Resolve the checkpoint path (default ``last.ckpt``; ``--ckpt_name`` overrides)."""
    ckpt_dir = get_full_path("train", "trainings", "trainings_Novae", novae_training_id, "checkpoints")
    name = ckpt_name or "last.ckpt"
    ckpt = os.path.join(str(ckpt_dir), name)
    if not os.path.isfile(ckpt):
        raise FileNotFoundError(
            f"Novae checkpoint not found: {ckpt}. Pass --ckpt_name to select a specific checkpoint "
            f"(available under {ckpt_dir})."
        )
    return ckpt


def load_frozen_novae(ckpt_path: str, device: torch.device):
    """Load the frozen Novae teacher and validate it is the expected unimodal 128/512 model."""
    # strict=False: the swav_head.queue buffer (shape depends on the number of slides) is not registered without adata
    model = novae.Novae.load_from_checkpoint(ckpt_path, map_location="cpu", strict=False)
    model.eval()
    model.requires_grad_(False)

    hp = model.hparams
    output_size = int(hp.get("output_size"))
    embedding_size = int(hp.get("embedding_size"))
    if output_size != OUTPUT_SIZE:
        raise ValueError(f"Novae teacher output_size={output_size} != {OUTPUT_SIZE} (st_feat dim).")
    if embedding_size != EMBEDDING_SIZE:
        raise ValueError(f"Novae teacher embedding_size={embedding_size} != {EMBEDDING_SIZE} (scConcept dim).")
    mode_kwargs = hp.get("mode_kwargs") or {}
    if mode_kwargs.get("multimodal", False):
        raise RuntimeError(
            "Novae teacher was trained multimodal (mode_kwargs.multimodal=True), but the graph fed here "
            "is unimodal (no histo_embeddings). Use a unimodal Novae teacher."
        )
    model.to(device)
    return model, output_size, embedding_size


# --------------------------------------------------------------------------- per-slide compute
def precompute_slide(
    model,
    device,
    slide_id: str,
    slide_patch_items,          # list of (patch_idx:int, patch_id:str)
    dataset_dir,
    scconcept_dir,
    edge_distance_threshold: float,
    fingerprint: str,
    fingerprint_inputs: dict,
    self_check: bool,
):
    """Compute + return (patch_idx array sorted, st_feat [n,128] float32) for one slide."""
    # --- open the slide's inputs -------------------------------------------------------------
    adata_path = os.path.join(str(dataset_dir), "adatas", f"{slide_id}.h5ad")
    if not os.path.isfile(adata_path):
        raise FileNotFoundError(f"AnnData not found for slide '{slide_id}': {adata_path}")
    adata = ad.read_h5ad(adata_path)
    if "cell_id" not in adata.obs:
        raise KeyError(f"{slide_id}: adata.obs['cell_id'] missing (needed for the mask<->graph join).")
    if "spatial_distances" not in adata.obsp:
        raise KeyError(f"{slide_id}: adata.obsp['spatial_distances'] missing (the µm Delaunay graph).")
    obs_cell_id = adata.obs["cell_id"].astype(str)
    obsp = adata.obsp["spatial_distances"].tocsr()

    centroids_path = os.path.join(str(dataset_dir), "nucleus_centroids", f"nucleus_centroids_{slide_id}.parquet")
    if not os.path.isfile(centroids_path):
        raise FileNotFoundError(f"Nucleus centroids not found for slide '{slide_id}': {centroids_path}")
    centroids = pd.read_parquet(centroids_path).set_index("cell_id")  # str cell id -> x / y

    bboxes = pd.read_parquet(
        os.path.join(str(dataset_dir), "patch_bboxes.parquet"),
        columns=["patch_id", "x_min", "y_min", "x_max", "y_max"],
    ).set_index("patch_id")

    scc_h5_path = os.path.join(str(scconcept_dir), f"{slide_id}.h5")
    if not os.path.isfile(scc_h5_path):
        raise FileNotFoundError(f"scConcept embeddings not found for slide '{slide_id}': {scc_h5_path}")
    scc_f = h5py.File(scc_h5_path, "r")
    scc_cell_id_int = scc_f["cell_id_int"][:].astype(np.int32, copy=False)
    # fetch_scconcept_for_cells (np.searchsorted) requires cell_id_int sorted ascending
    if scc_cell_id_int.size > 1 and np.any(scc_cell_id_int[1:] < scc_cell_id_int[:-1]):
        scc_f.close()
        raise ValueError(
            f"{slide_id}: scConcept 'cell_id_int' is not sorted ascending in {scc_h5_path}, but "
            "lookups use np.searchsorted(). Re-export the scConcept embeddings sorted by cell_id_int "
            "(or sort the file contents) before running the precompute."
        )
    scc_emb_dim = int(scc_f.attrs.get("embedding_dim", scc_f["cls_emb"].shape[1]))
    if scc_emb_dim != EMBEDDING_SIZE:
        scc_f.close()
        raise ValueError(f"{slide_id}: scConcept embedding_dim={scc_emb_dim} != {EMBEDDING_SIZE}.")

    mask_zip = zipfile.ZipFile(os.path.join(str(dataset_dir), "masks_cell_ids_nuclei.zip"), "r")
    mask_names = set(mask_zip.namelist())

    n_zero_cells = n_no_match = n_all_outside = n_nonfinite = n_missing_centroid = 0
    checked = False

    idx_list, feat_list = [], []
    try:
        for patch_idx, patch_id in tqdm(
            slide_patch_items, desc=f"[st_feat] {slide_id}", unit="patch", leave=False
        ):
            mask_fname = f"{patch_id}.npz"
            if mask_fname not in mask_names:
                raise FileNotFoundError(f"Mask missing in zip for patch '{patch_id}'.")
            mask_bytes = mask_zip.read(mask_fname)
            cell_ids_mask = read_patch_cell_ids(mask_bytes)
            H, W = load_npz(io.BytesIO(mask_bytes)).shape  # patch pixel size (raw, pre-resize)

            st_feat = None
            if cell_ids_mask.size == 0:
                n_zero_cells += 1
            else:
                ordered_ids_int, patch_cell_indices, ordered_str_ids = align_mask_cells_to_obs(
                    cell_ids_mask, obs_cell_id
                )
                if patch_cell_indices.size == 0:
                    n_no_match += 1
                else:
                    # Nucleus centroids (slide-level px) of the matched cells; cells without a centroid are dropped
                    coord_df = centroids.reindex(list(ordered_str_ids))[["x", "y"]]
                    have = coord_df.notna().all(axis=1).to_numpy()
                    if not have.all():
                        keep_c = np.flatnonzero(have)
                        n_missing_centroid += int(have.size - keep_c.size)
                        logger.warning(
                            "Patch '%s': %d matched cell(s) missing a nucleus centroid (dropped). Example(s): %s",
                            patch_id,
                            int(have.size - keep_c.size),
                            [s for s, ok in zip(list(ordered_str_ids), have.tolist()) if not ok][:5],
                        )
                        coord_df = coord_df.iloc[keep_c]
                        patch_cell_indices = patch_cell_indices[keep_c]
                        ordered_ids_int = [ordered_ids_int[i] for i in keep_c.tolist()]
                    coord_slide = coord_df.to_numpy()
                    bbox = bboxes.loc[patch_id]
                    coord_patch = patch_level_coords(coord_slide, bbox)
                    keep = inside_patch_keep_idx(coord_patch, (H, W))
                    if keep.size == 0:
                        n_all_outside += 1
                    else:
                        node_adata_idx = patch_cell_indices[keep]
                        node_ids_int = [ordered_ids_int[i] for i in keep.tolist()]
                        sub_adj = obsp[node_adata_idx][:, node_adata_idx]
                        node_feats = fetch_scconcept_for_cells(
                            scc_f, scc_cell_id_int, node_ids_int, emb_dim=scc_emb_dim
                        )
                        data = build_novae_teacher_graph(
                            sub_adj, node_feats, edge_distance_threshold=edge_distance_threshold
                        ).to(device)
                        with torch.no_grad():
                            out = model.encoder(data)  # [1, output_size]
                        vec = out.detach().float().cpu().numpy().reshape(-1)
                        if vec.shape[0] != OUTPUT_SIZE or not np.isfinite(vec).all():
                            n_nonfinite += 1
                            logger.warning(
                                "Non-finite/degenerate st_feat for patch '%s' (n_cells=%d); caching zeros.",
                                patch_id, keep.size,
                            )
                            st_feat = None
                        else:
                            st_feat = vec
                            # Determinism self-check on the first real patch of the run.
                            if self_check and not checked:
                                with torch.no_grad():
                                    out2 = model.encoder(
                                        build_novae_teacher_graph(
                                            sub_adj, node_feats, edge_distance_threshold=edge_distance_threshold
                                        ).to(device)
                                    )
                                if not np.allclose(vec, out2.detach().float().cpu().numpy().reshape(-1), atol=1e-5):
                                    raise RuntimeError(
                                        f"Determinism self-check failed on patch '{patch_id}': a second "
                                        "model.encoder call gave a different st_feat."
                                    )
                                logger.info("Determinism self-check OK on patch '%s'.", patch_id)
                                checked = True

            if st_feat is None:
                st_feat = np.zeros((OUTPUT_SIZE,), dtype=np.float32)
            idx_list.append(int(patch_idx))
            feat_list.append(st_feat.astype(np.float32, copy=False))
    finally:
        scc_f.close()
        mask_zip.close()
        del adata

    order = np.argsort(np.asarray(idx_list, dtype=np.int64), kind="stable")
    patch_idx_arr = np.asarray(idx_list, dtype=np.int64)[order]
    st_feat_arr = np.stack(feat_list, axis=0)[order].astype(np.float32, copy=False)

    if n_zero_cells or n_no_match or n_all_outside or n_nonfinite or n_missing_centroid:
        logger.info(
            "slide %s: zeros for %d empty-mask + %d no-adata-match + %d all-cells-outside + %d non-finite patches; "
            "%d matched cell(s) dropped for a missing nucleus centroid.",
            slide_id, n_zero_cells, n_no_match, n_all_outside, n_nonfinite, n_missing_centroid,
        )
    return patch_idx_arr, st_feat_arr


def write_slide_cache(out_path, patch_idx_arr, st_feat_arr, fingerprint, fingerprint_inputs):
    tmp = str(out_path) + ".tmp"
    with h5py.File(tmp, "w") as f:
        f.create_dataset("patch_idx", data=patch_idx_arr, dtype="int64")
        f.create_dataset("st_feat", data=st_feat_arr, dtype="float32")
        f.attrs["fingerprint"] = fingerprint
        f.attrs["dim"] = int(OUTPUT_SIZE)
        f.attrs["schema_version"] = int(NOVAE_TEACHER_SCHEMA_VERSION)
        for k, v in fingerprint_inputs.items():
            f.attrs[k] = "" if v is None else v
    os.replace(tmp, out_path)


# --------------------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser(
        description="Precompute the per-patch st_feat of the frozen Novae ST teacher."
    )
    parser.add_argument("--novae_training_id", type=str, required=True,
                        help="Novae training id whose frozen teacher produces the cache (e.g. training_4).")
    parser.add_argument("--dataset_id", type=str, required=True, help="Dataset id, e.g. ds_ct1_20x_224_o112.")
    parser.add_argument("--training_dataset_id", type=str, required=True,
                        help="Training dataset id, e.g. ds_ct1_20x_224_o112_1.")
    parser.add_argument("--scconcept_training_id", type=str, default=None,
                        help="scConcept training id of the node features (default: the Novae teacher's "
                             "own_scconcept_training_id, which it must match).")
    parser.add_argument("--folds", type=str, default="train,valid,test",
                        help="Comma-separated folds to cover (default 'train,valid,test').")
    parser.add_argument("--ckpt_name", type=str, default=None,
                        help="Checkpoint filename under checkpoints/ (default 'last.ckpt').")
    parser.add_argument("--edge_distance_threshold", type=float, default=100.0,
                        help="Prune patch-subgraph edges to <= this many µm (novae radius rule; default 100).")
    parser.add_argument("--device", type=str, default="cpu", help="torch device: cpu | cuda | mps (default cpu).")
    parser.add_argument("--overwrite", action="store_true",
                        help="Recompute + overwrite existing per-slide caches (default: skip a slide whose "
                             "cache already matches the fingerprint).")
    parser.add_argument("--no_self_check", action="store_true", help="Skip the determinism self-check.")
    parser.add_argument("--limit_slides", type=int, default=None, help="Process only the first N slides (manifest not written).")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )

    device = torch.device(args.device)
    folds = [x.strip() for x in args.folds.split(",") if x.strip()]

    # --- resolve provenance + fingerprint ----------------------------------------------------
    prov = read_teacher_provenance(args.novae_training_id, args.scconcept_training_id)
    scconcept_training_id = prov["scconcept_training_id"]
    logger.info("Novae teacher %s <- scConcept %s (%s / %s).", args.novae_training_id, scconcept_training_id,
                prov["scconcept_model_name"], prov["scconcept_model_version"])

    ckpt_path = resolve_ckpt_path(args.novae_training_id, args.ckpt_name)
    logger.info("Loading frozen Novae teacher: %s", ckpt_path)
    model, output_size, embedding_size = load_frozen_novae(ckpt_path, device)
    if prov["manifest_output_size"] not in (None, output_size):
        logger.warning("manifest output_size=%s != checkpoint output_size=%d.", prov["manifest_output_size"], output_size)

    fingerprint = novae_teacher_fingerprint(
        novae_training_id=args.novae_training_id,
        scconcept_training_id=scconcept_training_id,
        scconcept_model_name=prov["scconcept_model_name"] or "",
        scconcept_model_version=prov["scconcept_model_version"] or "",
        edge_distance_threshold=args.edge_distance_threshold,
        edge_char=NOVAE_CELLS_CHARACTERISTIC_DISTANCE,
        scale_to_microns=NOVAE_SCALE_TO_MICRONS,
        prune_op="le",
        output_size=output_size,
        embedding_size=embedding_size,
        dataset_id=args.dataset_id,
        training_dataset_id=args.training_dataset_id,
    )
    fingerprint_inputs = {
        "novae_training_id": args.novae_training_id,
        "scconcept_training_id": scconcept_training_id,
        "scconcept_model_name": prov["scconcept_model_name"] or "",
        "scconcept_model_version": prov["scconcept_model_version"] or "",
        "edge_distance_threshold": float(args.edge_distance_threshold),
        "edge_char": float(NOVAE_CELLS_CHARACTERISTIC_DISTANCE),
        "scale_to_microns": float(NOVAE_SCALE_TO_MICRONS),
        "prune_op": "le",
        "output_size": int(output_size),
        "embedding_size": int(embedding_size),
        "dataset_id": args.dataset_id,
        "training_dataset_id": args.training_dataset_id,
    }
    logger.info("Teacher cache fingerprint = %s", fingerprint)

    # --- paths -------------------------------------------------------------------------------
    dataset_dir = get_full_path("train", "training_datasets", args.dataset_id, args.training_dataset_id, scratch=True)
    scconcept_dir = get_full_path("train", "trainings", "trainings_scConcept", scconcept_training_id, "embeddings")
    cache_dir = get_full_path("train", "training_datasets", args.dataset_id, args.training_dataset_id,
                              "novae_teacher_cache", args.novae_training_id, scratch=True)
    os.makedirs(cache_dir, exist_ok=True)

    # --- enumerate patches across folds, group by slide --------------------------------------
    by_slide = defaultdict(list)
    for fold in folds:
        for patch_id in list_fold_patch_ids(dataset_dir, fold, n_min_cells=0):
            sid, pidx = _split_patch_id(patch_id)
            by_slide[sid].append((pidx, patch_id))
    slide_ids = sorted(by_slide.keys())
    if args.limit_slides is not None:
        slide_ids = slide_ids[: args.limit_slides]
    logger.info("Covering %d slide(s), %d patch(es) across folds %s.",
                len(slide_ids), sum(len(by_slide[s]) for s in slide_ids), folds)

    # --- per-slide compute + write -----------------------------------------------------------
    per_slide_counts = {}
    for i, slide_id in enumerate(slide_ids, 1):
        out_path = os.path.join(str(cache_dir), f"{slide_id}.h5")
        if os.path.isfile(out_path) and not args.overwrite:
            try:
                with h5py.File(out_path, "r") as f:
                    if f.attrs.get("fingerprint") == fingerprint:
                        per_slide_counts[slide_id] = int(f["patch_idx"].shape[0])
                        logger.info("[%d/%d] slide %s: up-to-date cache, skipping.", i, len(slide_ids), slide_id)
                        continue
            except Exception:
                pass  # fall through to recompute
        logger.info("[%d/%d] slide %s: computing st_feat for %d patches...",
                    i, len(slide_ids), slide_id, len(by_slide[slide_id]))
        patch_idx_arr, st_feat_arr = precompute_slide(
            model, device, slide_id, by_slide[slide_id], dataset_dir, scconcept_dir,
            edge_distance_threshold=args.edge_distance_threshold,
            fingerprint=fingerprint, fingerprint_inputs=fingerprint_inputs,
            self_check=(not args.no_self_check),
        )
        write_slide_cache(out_path, patch_idx_arr, st_feat_arr, fingerprint, fingerprint_inputs)
        per_slide_counts[slide_id] = int(patch_idx_arr.shape[0])

    # --- manifest ----------------------------------------------------------------------------
    manifest = {
        "schema_version": int(NOVAE_TEACHER_SCHEMA_VERSION),
        "dim": int(output_size),
        "fingerprint": fingerprint,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "created_by": "precompute_novae_teacher.py",
        "git_commit": _git_commit(),
        "novae_training_id": args.novae_training_id,
        "novae_ckpt_path": ckpt_path,
        "novae_version": getattr(novae, "__version__", None),
        "scconcept_training_id": scconcept_training_id,
        "scconcept_model_name": prov["scconcept_model_name"],
        "scconcept_model_version": prov["scconcept_model_version"],
        "scconcept_species": prov["scconcept_species"],
        "dataset_id": args.dataset_id,
        "training_dataset_id": args.training_dataset_id,
        "folds": folds,
        "edge_distance_threshold": float(args.edge_distance_threshold),
        "edge_char": float(NOVAE_CELLS_CHARACTERISTIC_DISTANCE),
        "scale_to_microns": float(NOVAE_SCALE_TO_MICRONS),
        "prune_op": "le",
        "embedding_size": int(embedding_size),
        "n_slides": len(per_slide_counts),
        "n_patches_per_slide": per_slide_counts,
    }
    if args.limit_slides is None:  # manifest not written with --limit_slides
        with open(os.path.join(str(cache_dir), "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)
        logger.info("Wrote cache manifest -> %s", os.path.join(str(cache_dir), "manifest.json"))
    else:
        logger.info("--limit_slides: manifest not written.")
    logger.info("Done. Cache at %s", cache_dir)


if __name__ == "__main__":
    main()
