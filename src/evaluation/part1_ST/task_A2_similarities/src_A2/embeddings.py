from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import anndata as ad
import h5py
import numpy as np
import pandas as pd
from scipy import sparse

from src_A2.common import coerce_bool_series, ensure_dir, file_info, read_json, src_import, write_json


def _manifest_file_entry_matches_path(entry: Dict[str, Any], path: Path) -> bool:
    """
    Fast cache-validity check.
    """
    if not isinstance(entry, dict):
        return False
    try:
        st = path.stat()
        return (
            int(entry.get("size_bytes", -1)) == int(st.st_size)
            and abs(float(entry.get("mtime", -1.0)) - float(st.st_mtime)) < 1e-6
        )
    except Exception:
        return False

@dataclass
class ProjectedReferenceCache:
    path: Path
    obs_name: np.ndarray
    z_ref: np.ndarray
    row_idx: np.ndarray
    embedding_dim: int


@dataclass
class HESelectedEmbeddings:
    table: pd.DataFrame
    z_he: np.ndarray
    z_he_raw: Optional[np.ndarray]


@dataclass
class GTCellEmbeddings:
    obs: pd.DataFrame
    z_gt: np.ndarray


@dataclass
class GTExpressionMatrix:
    obs: pd.DataFrame
    # None if loaded with load_x=False (obs and gene_ids are still built and validated)
    X: Optional[np.ndarray]
    gene_ids: List[str]



def _decode_h5_string_array(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr[:, 0]
    out = []
    for x in arr:
        if isinstance(x, (bytes, np.bytes_)):
            out.append(bytes(x).decode("utf-8"))
        else:
            out.append(str(x))
    return np.asarray(out, dtype=object)



def resolve_legacy_he_embedding_dir(eval_data_root: Path, training_num: str, checkpoint_num: str) -> Path:
    path = eval_data_root / "he_embeddings" / f"HEembs_t{training_num}e{checkpoint_num}"
    if not path.exists():
        raise FileNotFoundError(path)
    return path



def resolve_legacy_st_cell_embedding_dir(eval_data_root: Path, training_num: str, checkpoint_num: str) -> Path:
    path = eval_data_root / "st_sthelar_embeddings_clip" / f"CELLembs_t{training_num}e{checkpoint_num}"
    if not path.exists():
        raise FileNotFoundError(path)
    return path



def resolve_raw_reference_embedding_h5(raw_ref_embedding_dir: Path, reference_h5ad_path: Path) -> Path:
    manifest_path = raw_ref_embedding_dir / "manifest.json"
    ref_resolved = str(reference_h5ad_path.resolve())

    if manifest_path.exists():
        manifest = read_json(manifest_path)
        files = manifest.get("files", {})
        matches = []
        for out_stem, info in files.items():
            source = info.get("input_h5ad") or info.get("source_h5ad")
            if source is None:
                continue
            try:
                if str(Path(source).resolve()) == ref_resolved:
                    cand = raw_ref_embedding_dir / f"{out_stem}.h5"
                    if cand.exists():
                        matches.append(cand)
            except Exception:
                continue
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise RuntimeError(f"More than one raw reference embedding file matched {reference_h5ad_path}: {matches}")

    fallback = raw_ref_embedding_dir / f"{reference_h5ad_path.stem}.h5"
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"Could not resolve raw reference embedding file for {reference_h5ad_path} under {raw_ref_embedding_dir}")



def ensure_selected_main_rows(
    checkpoint_dir: Path,
    eval_data_root: Path,
    training_num: str,
    checkpoint_num: str,
    slide_id: str,
) -> Path:
    dst = checkpoint_dir / "selected_main_he_rows" / f"{slide_id}.parquet"
    if dst.exists():
        return dst
    legacy = resolve_legacy_he_embedding_dir(eval_data_root, training_num, checkpoint_num) / "main_HEemb_cellID" / f"{slide_id}.parquet"
    if not legacy.exists():
        raise FileNotFoundError(f"Selected main H&E row parquet not found for slide '{slide_id}'. Expected either {dst} or {legacy}.")
    ensure_dir(dst.parent)
    df = pd.read_parquet(legacy)
    df.to_parquet(dst, index=False)
    return dst



def load_selected_he_embeddings(
    he_slide_h5: Path,
    selected_rows_parquet: Path,
    require_clip: bool = True,
    load_raw: bool = False,
) -> HESelectedEmbeddings:
    table = pd.read_parquet(selected_rows_parquet).copy()
    required_cols = [
        "cell_id_int",
        "selected_patch_cell_row_index",
        "selected_alpha",
    ]
    missing = [c for c in required_cols if c not in table.columns]
    if missing:
        raise KeyError(f"Missing required columns in {selected_rows_parquet}: {missing}")
    if table.empty:
        return HESelectedEmbeddings(table=table, z_he=np.zeros((0, 0), dtype=np.float32), z_he_raw=None)
    if table["cell_id_int"].duplicated().any():
        dup = table.loc[table["cell_id_int"].duplicated(), "cell_id_int"].astype(str).unique().tolist()[:20]
        raise ValueError(f"Duplicate cell_id_int in selected-main-row parquet {selected_rows_parquet}. Examples: {dup}")
    if table["selected_patch_cell_row_index"].isna().any():
        raise ValueError(f"NaN selected_patch_cell_row_index found in {selected_rows_parquet}")

    row_idx = table["selected_patch_cell_row_index"].to_numpy(dtype=np.int64, copy=False)
    if pd.Index(row_idx).has_duplicates:
        dup = pd.Index(row_idx)[pd.Index(row_idx).duplicated()].unique().tolist()[:20]
        raise ValueError(f"Duplicate selected_patch_cell_row_index in {selected_rows_parquet}. Examples: {dup}")
    order = np.argsort(row_idx, kind="mergesort")
    inv_order = np.empty_like(order)
    inv_order[order] = np.arange(order.size, dtype=np.int64)
    row_idx_sorted = row_idx[order]

    with h5py.File(he_slide_h5, "r") as f:
        if require_clip and "patch_cell_emb_he_clip" not in f:
            raise KeyError(f"Missing patch_cell_emb_he_clip in {he_slide_h5}")
        if require_clip and int(f["patch_cell_emb_he_clip"].shape[1]) <= 0:
            raise RuntimeError(
                f"{he_slide_h5} contains patch_cell_emb_he_clip with dimension 0. "
                "A-2 requires a trained H&E token/cell projection head for the cell-level "
                "CLIP-space evaluation."
            )
        n_rows = int(f["patch_cell_emb_he_clip"].shape[0])
        if np.any(row_idx < 0) or np.any(row_idx >= n_rows):
            raise IndexError(f"selected_patch_cell_row_index out of bounds for {he_slide_h5}")
        z_clip_sorted = np.asarray(f["patch_cell_emb_he_clip"][row_idx_sorted], dtype=np.float32)
        z_clip = z_clip_sorted[inv_order]
        if load_raw:
            z_raw_sorted = np.asarray(f["patch_cell_emb_he"][row_idx_sorted], dtype=np.float32)
            z_raw = z_raw_sorted[inv_order]
        else:
            z_raw = None
    return HESelectedEmbeddings(table=table, z_he=z_clip, z_he_raw=z_raw)



def load_gt_cell_embeddings(st_cell_h5ad: Path) -> GTCellEmbeddings:
    a = ad.read_h5ad(st_cell_h5ad)
    try:
        obs = a.obs.copy()
        obs.index = obs.index.astype(str)
        if obs.index.duplicated().any():
            dup = obs.index[obs.index.duplicated()].unique().tolist()[:20]
            raise ValueError(f"Duplicate obs index in GT cell embedding file {st_cell_h5ad}. Examples: {dup}")
        if "cell_id_int" not in obs.columns:
            raise KeyError(f"Missing cell_id_int in {st_cell_h5ad}")
        obs["cell_id_int"] = obs["cell_id_int"].astype(np.int32)
        if obs["cell_id_int"].duplicated().any():
            dup = obs.loc[obs["cell_id_int"].duplicated(), "cell_id_int"].astype(str).unique().tolist()[:20]
            raise ValueError(f"Duplicate cell_id_int in GT cell embedding file {st_cell_h5ad}. Examples: {dup}")
        z_gt = np.asarray(a.X, dtype=np.float32)
        if z_gt.shape[0] != obs.shape[0]:
            raise RuntimeError(f"Row-count mismatch in {st_cell_h5ad}: obs has {obs.shape[0]} rows but X has {z_gt.shape[0]} rows.")
        return GTCellEmbeddings(obs=obs, z_gt=z_gt)
    finally:
        if getattr(a, "file", None) is not None:
            try:
                a.file.close()
            except Exception:
                pass



def load_gt_expression_matrix(
    gt_eval_h5ad: Path,
    slide_id: str,
    gene_id_column: str,
    var_indices: Optional[np.ndarray] = None,
    *,
    load_x: bool = True,
) -> GTExpressionMatrix:
    """
    Load the GT expression of the cell rows (only the genes in var_indices if given).
    With load_x=False, X is None (expression not read); obs and gene_ids are still built and validated.
    """
    a = ad.read_h5ad(gt_eval_h5ad, backed="r")
    try:
        obs = a.obs.copy()
        if "is_cell" in obs.columns:
            mask = coerce_bool_series(obs["is_cell"], context=f"{gt_eval_h5ad}:obs['is_cell']").to_numpy()
        else:
            raise KeyError(f"Missing is_cell column in {gt_eval_h5ad}: unable to identify cell rows for slide '{slide_id}'")
        obs = obs.loc[mask].copy()

        if gene_id_column not in a.var.columns:
            raise KeyError(f"Missing var['{gene_id_column}'] in {gt_eval_h5ad}")

        all_gene_ids = [str(x) for x in a.var[gene_id_column].tolist()]

        if var_indices is None:
            # `load_x=False` skips only this read; `gene_ids` below is unaffected.
            X = a[mask, :].X if load_x else None
            gene_ids = all_gene_ids
        else:
            var_indices = np.asarray(var_indices, dtype=np.int64)
            if var_indices.ndim != 1:
                raise ValueError("var_indices must be a 1D integer array")
            if var_indices.size == 0:
                raise ValueError("var_indices cannot be empty")
            if np.any(var_indices < 0) or int(var_indices.max()) >= int(a.n_vars):
                raise IndexError(
                    f"var_indices out of bounds for {gt_eval_h5ad}: "
                    f"min={int(var_indices.min())}, max={int(var_indices.max())}, n_vars={int(a.n_vars)}"
                )

            # Read with sorted column indices, then restore the requested gene order
            var_order = np.argsort(var_indices, kind="mergesort")
            var_sorted = var_indices[var_order]
            var_restore = np.empty_like(var_order)
            var_restore[var_order] = np.arange(var_order.size, dtype=np.int64)

            # load_x=False skips only this read
            X = a[mask, var_sorted.astype(np.int64, copy=False).tolist()].X if load_x else None
            gene_ids = [all_gene_ids[int(i)] for i in var_indices.tolist()]

        if X is not None:
            if sparse.issparse(X):
                X = X.toarray()
            X = np.asarray(X, dtype=np.float32)

            if var_indices is not None and not np.array_equal(var_order, np.arange(var_order.size)):
                X = X[:, var_restore]

        obs.index = obs.index.astype(str)
        if "cell_id_int" not in obs.columns:
            obs["cell_id_int"] = [int(str(idx).rsplit("_", 1)[-1]) for idx in obs.index.astype(str)]
        if "cell_id" not in obs.columns:
            obs["cell_id"] = obs["cell_id_int"].astype(str)
        obs["cell_id_int"] = obs["cell_id_int"].astype(np.int32)
        obs["cell_id"] = obs["cell_id"].astype(str)
        if obs.index.duplicated().any():
            dup = obs.index[obs.index.duplicated()].unique().tolist()[:20]
            raise ValueError(f"Duplicate cell-row index in GT eval h5ad {gt_eval_h5ad}. Examples: {dup}")
        if obs["cell_id_int"].duplicated().any():
            dup = obs.loc[obs["cell_id_int"].duplicated(), "cell_id_int"].astype(str).unique().tolist()[:20]
            raise ValueError(f"Duplicate cell_id_int in GT eval h5ad {gt_eval_h5ad}. Examples: {dup}")
        # Row-count check (only when X is loaded)
        if X is not None and X.shape[0] != obs.shape[0]:
            raise RuntimeError(f"Row-count mismatch in {gt_eval_h5ad}: obs has {obs.shape[0]} rows but X has {X.shape[0]} rows after cell filtering.")
        return GTExpressionMatrix(obs=obs, X=X, gene_ids=gene_ids)
    finally:
        if getattr(a, "file", None) is not None:
            try:
                a.file.close()
            except Exception:
                pass



def validate_reference_alignment(reference_h5ad_path: Path, raw_ref_h5: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Align the raw reference embedding rows with the reference h5ad rows (using row_idx, raw_h5_row -> h5ad_row, if present).

    Returns:
        obs_names: h5ad obs names in h5ad row order.
        h5ad_to_raw: integer array where h5ad_to_raw[h5ad_row] = raw_h5_row.
    """
    a = ad.read_h5ad(reference_h5ad_path, backed="r")
    try:
        obs_names = a.obs_names.astype(str).to_numpy()
    finally:
        if getattr(a, "file", None) is not None:
            a.file.close()

    with h5py.File(raw_ref_h5, "r") as f:
        if "obs_name" not in f or "cls_emb" not in f:
            raise KeyError(f"Raw reference embedding file is missing obs_name or cls_emb: {raw_ref_h5}")
        raw_obs_names = _decode_h5_string_array(f["obs_name"][:])
        has_row_idx = "row_idx" in f
        stored_row_idx = (
            np.asarray(f["row_idx"][:], dtype=np.int64)
            if has_row_idx
            else np.arange(raw_obs_names.shape[0], dtype=np.int64)
        )
    if stored_row_idx.shape[0] != raw_obs_names.shape[0]:
        raise RuntimeError(f"Invalid row_idx length in {raw_ref_h5}")
    if pd.Index(stored_row_idx).has_duplicates:
        raise RuntimeError(f"Duplicate row_idx entries in raw reference embedding file: {raw_ref_h5}")
    if np.any(np.sort(stored_row_idx) != np.arange(stored_row_idx.shape[0], dtype=np.int64)):
        raise RuntimeError(f"row_idx in raw reference embedding file is not a permutation of [0..n-1]: {raw_ref_h5}")
    if obs_names.shape[0] != raw_obs_names.shape[0]:
        raise RuntimeError(f"Reference/raw-embedding size mismatch for {reference_h5ad_path.name}: {obs_names.shape[0]} vs {raw_obs_names.shape[0]}")
    if pd.Index(obs_names).has_duplicates:
        raise RuntimeError(f"Duplicate obs_names in reference h5ad: {reference_h5ad_path}")
    if pd.Index(raw_obs_names).has_duplicates:
        raise RuntimeError(f"Duplicate obs_name entries in raw reference embedding file: {raw_ref_h5}")

    if has_row_idx:
        # row_idx should describe which h5ad row each raw embedding row came from.
        expected_raw_obs_names = obs_names[stored_row_idx]
        if not np.array_equal(expected_raw_obs_names.astype(str), raw_obs_names.astype(str)):
            bad = np.flatnonzero(expected_raw_obs_names.astype(str) != raw_obs_names.astype(str))[:10]
            examples = [
                {
                    "raw_row": int(i),
                    "row_idx_h5ad_row": int(stored_row_idx[i]),
                    "obs_name_from_row_idx": str(expected_raw_obs_names[i]),
                    "raw_obs_name": str(raw_obs_names[i]),
                }
                for i in bad.tolist()
            ]
            raise RuntimeError(
                f"row_idx/obs_name inconsistency in raw reference embedding file {raw_ref_h5}. "
                f"Examples: {examples}"
            )

        h5ad_to_raw = np.empty_like(stored_row_idx)
        h5ad_to_raw[stored_row_idx] = np.arange(stored_row_idx.shape[0], dtype=np.int64)
    elif np.array_equal(obs_names, raw_obs_names):
        h5ad_to_raw = np.arange(obs_names.shape[0], dtype=np.int64)
    else:
        raw_index = pd.Index(raw_obs_names)
        h5ad_to_raw = raw_index.get_indexer(obs_names).astype(np.int64, copy=False)
        if np.any(h5ad_to_raw < 0):
            missing = obs_names[h5ad_to_raw < 0][:10].tolist()
            raise RuntimeError(
                f"Reference/raw-embedding obs_name mismatch for {reference_h5ad_path.name}; "
                f"missing names include {missing}"
            )
    return obs_names, h5ad_to_raw



def resolve_st_cell_bundle_dims(
    config: Dict[str, Any],
    state_dict: Dict[str, Any],
) -> Tuple[int, int]:
    """
    Resolve (in_dim, token_embed_dim) of an exported ST cell bundle.
    token_embed_dim falls back to clip_embed_dim when None; in_dim is read by infer_scconcept_embedding_dim.
    """
    clip_cfg = config["model"]["clip"]
    embed_dim = int(clip_cfg["clip_embed_dim"])

    token_embed_dim = clip_cfg.get("token_embed_dim", None)
    token_embed_dim = int(token_embed_dim) if token_embed_dim is not None else embed_dim

    infer_scconcept_embedding_dim = src_import("infer_scconcept_embedding_dim")
    in_dim, _in_dim_source = infer_scconcept_embedding_dim(config, state_dict)
    return int(in_dim), int(token_embed_dim)


def project_reference_embeddings_if_needed(
    raw_ref_h5: Path,
    reference_h5ad_path: Path,
    projected_cache_path: Path,
    st_cell_bundle_path: Path,
    device: str,
    batch_size: int = 4096,
) -> ProjectedReferenceCache:
    cache_ok = False
    manifest_path = projected_cache_path.with_suffix(".manifest.json")
    if projected_cache_path.exists():
        try:
            manifest_ok = False
            if manifest_path.exists():
                manifest = read_json(manifest_path)
                manifest_ok = (
                    _manifest_file_entry_matches_path(manifest.get("raw_reference_embedding_file", {}), raw_ref_h5)
                    and _manifest_file_entry_matches_path(manifest.get("reference_h5ad", {}), reference_h5ad_path)
                    and _manifest_file_entry_matches_path(manifest.get("st_cell_bundle", {}), st_cell_bundle_path)
                )
            with h5py.File(projected_cache_path, "r") as f:
                cache_ok = (
                    "obs_name" in f
                    and "z_ref_clip" in f
                    and "row_idx" in f
                    and manifest_ok
                    and str(f.attrs.get("raw_reference_embedding_file", "")) == str(raw_ref_h5)
                    and str(f.attrs.get("reference_h5ad", "")) == str(reference_h5ad_path)
                    and str(f.attrs.get("st_cell_bundle", "")) == str(st_cell_bundle_path)
                    and f["obs_name"].shape[0] == f["z_ref_clip"].shape[0] == f["row_idx"].shape[0]
                )
        except Exception:
            cache_ok = False

    if cache_ok:
        with h5py.File(projected_cache_path, "r") as f:
            return ProjectedReferenceCache(
                path=projected_cache_path,
                obs_name=_decode_h5_string_array(f["obs_name"][:]),
                z_ref=np.asarray(f["z_ref_clip"][:], dtype=np.float32),
                row_idx=np.asarray(f["row_idx"][:], dtype=np.int64),
                embedding_dim=int(f.attrs["embedding_dim"]),
            )

    if projected_cache_path.exists():
        projected_cache_path.unlink()
        if manifest_path.exists():
            manifest_path.unlink()

    obs_names, h5ad_to_raw = validate_reference_alignment(reference_h5ad_path, raw_ref_h5)

    import torch

    # ST cell encoder shared with extract_encoders.py: st_node_proj(raw scConcept), bias-free, no centering
    STCellClipEncoder = src_import("STCellClipEncoder")

    bundle = torch.load(st_cell_bundle_path, map_location="cpu", weights_only=False)
    config = bundle["config"]
    state_dict = bundle["state_dict"]
    clip_cfg = config["model"]["clip"]
    lambda_token = float(clip_cfg["lambda_token"])
    has_proj = lambda_token > 0.0
    if not has_proj:
        # The trained ST cell projection head is required to project the reference into the CLIP cell space
        raise ValueError(
            "Projection is required for A-2 reference embedding projection, but the "
            "checkpoint's clip config has lambda_token <= 0.0 (no st_node_proj head)."
        )

    # Resolved after the has_proj check (st_node_proj.weight is then present)
    in_dim, token_embed_dim = resolve_st_cell_bundle_dims(config, state_dict)

    model = STCellClipEncoder(in_dim=in_dim, out_dim=token_embed_dim, has_proj=has_proj)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    torch_device = torch.device(device)
    model.to(torch_device)

    # h5ad_to_raw: h5ad_row -> raw_h5_row; its inverse writes the projected batches directly in h5ad order
    n = int(h5ad_to_raw.shape[0])
    if np.any(np.sort(h5ad_to_raw) != np.arange(n, dtype=np.int64)):
        raise RuntimeError(
            f"Reference/raw embedding alignment for {reference_h5ad_path.name} is not "
            "a permutation of [0..n-1]."
        )
    h5ad_pos_for_raw = np.empty_like(h5ad_to_raw)
    h5ad_pos_for_raw[h5ad_to_raw] = np.arange(n, dtype=np.int64)

    with h5py.File(raw_ref_h5, "r") as f:
        cls_emb = f["cls_emb"]
        if int(cls_emb.shape[1]) != int(in_dim):
            raise RuntimeError(
                f"Raw scConcept reference embedding dim mismatch for {raw_ref_h5}: "
                f"{int(cls_emb.shape[1])} vs expected {int(in_dim)} from ST cell bundle."
            )
        n = int(cls_emb.shape[0])
        out_dim = int(token_embed_dim if has_proj else in_dim)
        z_out = np.empty((n, out_dim), dtype=np.float32)
        with torch.inference_mode():
            for start in range(0, n, batch_size):
                stop = min(start + batch_size, n)
                x = torch.from_numpy(np.asarray(cls_emb[start:stop], dtype=np.float32)).to(torch_device)
                y = model(x).detach().cpu().float().numpy()
                dest_rows = h5ad_pos_for_raw[start:stop]
                z_out[dest_rows] = y
    aligned_row_idx = np.arange(obs_names.shape[0], dtype=np.int64)

    ensure_dir(projected_cache_path.parent)
    with h5py.File(projected_cache_path, "w") as f:
        f.create_dataset("row_idx", data=aligned_row_idx)
        f.create_dataset("h5ad_to_raw_row", data=h5ad_to_raw)
        f.create_dataset("obs_name", data=np.asarray(obs_names, dtype=object), dtype=h5py.string_dtype("utf-8"))
        f.create_dataset("z_ref_clip", data=z_out)
        f.attrs["embedding_dim"] = int(z_out.shape[1])
        f.attrs["raw_reference_embedding_file"] = str(raw_ref_h5)
        f.attrs["reference_h5ad"] = str(reference_h5ad_path)
        f.attrs["st_cell_bundle"] = str(st_cell_bundle_path)

    manifest = {
        "projected_cache": file_info(projected_cache_path, compute_sha256=False),
        "raw_reference_embedding_file": file_info(raw_ref_h5, compute_sha256=False),
        "reference_h5ad": file_info(reference_h5ad_path, compute_sha256=False),
        "st_cell_bundle": file_info(st_cell_bundle_path, compute_sha256=False),
    }
    write_json(projected_cache_path.with_suffix(".manifest.json"), manifest)
    return ProjectedReferenceCache(
        path=projected_cache_path,
        obs_name=obs_names,
        z_ref=z_out,
        row_idx=aligned_row_idx,
        embedding_dim=int(z_out.shape[1]),
    )
