"""
Helpers shared by the Novae-teacher precompute and the training dataset (ST-graph assembly, cache fingerprint, cell lookups).
Imported in both the `histost` and `novae` envs (only numpy / scipy / torch / torch_geometric at top level).
"""

import hashlib
import io
import json
from pathlib import Path

import numpy as np
from scipy.sparse import load_npz
import torch
from torch_geometric.data import Data
from torch_geometric.utils import from_scipy_sparse_matrix

from src.utils.cell_ids import str_cell_id


# Novae edge-distance normalization in µm (Nums.CELLS_CHARACTERISTIC_DISTANCE in novae/_constants.py)
NOVAE_CELLS_CHARACTERISTIC_DISTANCE: float = 20.0

# Novae scale_to_microns setting (novae/_settings.py); 1.0 = coordinates already in µm
NOVAE_SCALE_TO_MICRONS: float = 1.0

# st_feat cache format version (part of the fingerprint)
NOVAE_TEACHER_SCHEMA_VERSION: int = 1


# ---------------------------------------------------------------------------------------
# Per-patch ST-graph assembly (fed to novae.Novae.encoder)
# ---------------------------------------------------------------------------------------
def build_novae_teacher_graph(
    sub_adj,
    node_feats,
    edge_distance_threshold: float | None = 100.0,
    edge_char: float = NOVAE_CELLS_CHARACTERISTIC_DISTANCE,
    scale_to_microns: float = NOVAE_SCALE_TO_MICRONS,
) -> Data:
    """
    Assemble the single-patch PyG graph fed to ``novae.Novae.encoder(...)``, as novae's AnnData -> PyG conversion
    (``novae/data/dataset.py``) does: edge_attr = distance * scale_to_microns / edge_char, no self-loops.

    Args:
        sub_adj: scipy sparse ``[n, n]`` symmetric matrix of Delaunay edge distances (µm) between the patch cells.
        node_feats: array-like ``[n, D]`` per-cell scConcept embeddings.
        edge_distance_threshold: keep edges with distance ``<=`` this value in µm (``None`` disables pruning).
        edge_char: characteristic distance in µm (20).

    Returns:
        ``torch_geometric.data.Data`` with ``x`` ``[n, D]`` float32, ``edge_index`` ``[2, E]``,
        ``edge_attr`` ``[E, 1]`` float32, ``batch`` ``[n]`` long (all zeros = one graph).
    """
    x = torch.as_tensor(np.asarray(node_feats), dtype=torch.float32)
    n_nodes = int(x.shape[0])

    sub_adj = sub_adj.astype(np.float32, copy=False)
    edge_index, edge_weight = from_scipy_sparse_matrix(sub_adj)  # both directions (symmetric)
    edge_weight = edge_weight.to(torch.float32)

    if edge_distance_threshold is not None and edge_weight.numel() > 0:
        keep = edge_weight <= float(edge_distance_threshold)  # '<=' matches novae radius=[0, r]
        edge_index = edge_index[:, keep]
        edge_weight = edge_weight[keep]

    edge_attr = (edge_weight[:, None] * float(scale_to_microns) / float(edge_char)).to(torch.float32)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=n_nodes)
    data.batch = torch.zeros(n_nodes, dtype=torch.long)
    return data


# ---------------------------------------------------------------------------------------
# Cache fingerprint of the st_feat cache
# ---------------------------------------------------------------------------------------
def novae_teacher_fingerprint(
    *,
    novae_training_id: str,
    scconcept_training_id: str,
    scconcept_model_name: str,
    scconcept_model_version: str,
    edge_distance_threshold: float | None,
    edge_char: float = NOVAE_CELLS_CHARACTERISTIC_DISTANCE,
    scale_to_microns: float = NOVAE_SCALE_TO_MICRONS,
    prune_op: str = "le",
    output_size: int = 128,
    embedding_size: int = 512,
    dataset_id: str,
    training_dataset_id: str,
    schema_version: int = NOVAE_TEACHER_SCHEMA_VERSION,
) -> str:
    """
    Content hash of everything that determines a cached ``st_feat`` value.
    """
    payload = {
        "novae_training_id": str(novae_training_id),
        "scconcept_training_id": str(scconcept_training_id),
        "scconcept_model_name": str(scconcept_model_name),
        "scconcept_model_version": str(scconcept_model_version),
        "edge_distance_threshold": (None if edge_distance_threshold is None
                                    else float(edge_distance_threshold)),
        "edge_char": float(edge_char),
        "scale_to_microns": float(scale_to_microns),
        "prune_op": str(prune_op),
        "output_size": int(output_size),
        "embedding_size": int(embedding_size),
        "dataset_id": str(dataset_id),
        "training_dataset_id": str(training_dataset_id),
        "schema_version": int(schema_version),
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------------------
# On-disk helpers shared by the precompute and the training dataset
# ---------------------------------------------------------------------------------------
def list_fold_patch_ids(dataset_dir, fold: str, n_min_cells: int = 0) -> list[str]:
    """
    Natsorted ``patch_id`` list of a fold (the full list when ``n_min_cells == 0``), as in ``HistoStDataset.__init__``.
    """
    import pandas as pd
    from natsort import natsorted

    dataset_dir = Path(dataset_dir)
    if n_min_cells and int(n_min_cells) > 0:
        csv = dataset_dir / f"cell_count_{fold}_{int(n_min_cells)}.csv"
    else:
        csv = dataset_dir / f"cell_count_{fold}.csv"
    if not csv.is_file():
        raise FileNotFoundError(f"cell-count file not found for fold '{fold}': {csv}")
    patch_ids = pd.read_csv(csv)["patch_id"].tolist()
    return natsorted(patch_ids)


def read_patch_cell_ids(mask_npz_bytes) -> np.ndarray:
    """
    Unique non-zero integer cell ids of a patch's cell-id mask (0 = background).
    """
    mask_sparse = load_npz(io.BytesIO(mask_npz_bytes))
    if mask_sparse.nnz == 0:
        return np.array([], dtype=np.int32)
    return np.unique(mask_sparse.data.astype(np.int32, copy=False))


def align_mask_cells_to_obs(cell_ids_patch, obs_cell_id):
    """
    Match a patch's integer mask cell ids to the order of the slide AnnData ``obs['cell_id']``.

    Args:
        cell_ids_patch: array-like of integer (background-shifted) cell ids from the mask.
        obs_cell_id: pandas Series of the slide AnnData's string ``cell_id`` column.

    Returns:
        ``(ordered_cell_ids_int, patch_cell_indices, ordered_cell_str_ids)`` of the patch cells, in AnnData
        row order: int ids (list), AnnData row indices (``np.ndarray``) and string ids (pandas Series).
    """
    cell_ids_patch = np.asarray(cell_ids_patch, dtype=np.int32)
    cell_ids_patch_str = [str_cell_id(int(cid)) for cid in cell_ids_patch.tolist()]
    str_to_int = {s: int(cid) for s, cid in zip(cell_ids_patch_str, cell_ids_patch.tolist())}

    requested_cell_ids = list(str_to_int.keys())
    patch_cell_mask = obs_cell_id.isin(requested_cell_ids).values
    patch_cell_indices = np.flatnonzero(patch_cell_mask)
    ordered_cell_str_ids = obs_cell_id.iloc[patch_cell_indices]
    ordered_cell_ids_int = [str_to_int[s] for s in ordered_cell_str_ids.tolist()]
    return ordered_cell_ids_int, patch_cell_indices, ordered_cell_str_ids


def patch_level_coords(coord_slide, bbox) -> np.ndarray:
    """
    Convert slide-level nucleus centroids to patch-local pixel coords (subtract the bbox ``x_min`` / ``y_min``).
    """
    coord = np.asarray(coord_slide, dtype=np.float32).copy()
    coord[:, 0] = coord[:, 0] - float(bbox["x_min"])
    coord[:, 1] = coord[:, 1] - float(bbox["y_min"])
    return coord


def inside_patch_keep_idx(coord_patch, patch_hw) -> np.ndarray:
    """
    Indices of the cells whose patch-local centroid lies inside ``[0, W) x [0, H)``, with ``patch_hw = (H, W)``.
    """
    H, W = float(patch_hw[0]), float(patch_hw[1])
    coord_patch = np.asarray(coord_patch, dtype=np.float32)
    inside = (
        (coord_patch[:, 0] >= 0.0) & (coord_patch[:, 0] < W)
        & (coord_patch[:, 1] >= 0.0) & (coord_patch[:, 1] < H)
    )
    return np.flatnonzero(inside)


def fetch_scconcept_for_cells(h5_file, cell_id_int_sorted, cell_ids, emb_dim=None) -> np.ndarray:
    """
    Per-cell scConcept embeddings ``[N, D]`` in ``cell_ids`` order, by binary search on the sorted ``cell_id_int``
    of an open scConcept ``{slide}.h5`` (stored dtype). Raises ``KeyError`` if an id is missing.
    """
    ids = np.asarray(cell_ids, dtype=np.int32)
    if ids.size == 0:
        D = int(emb_dim) if emb_dim is not None else int(h5_file["cls_emb"].shape[1])
        return np.zeros((0, D), dtype=h5_file["cls_emb"].dtype)

    idx = np.searchsorted(cell_id_int_sorted, ids)
    idx_clipped = np.clip(idx, 0, cell_id_int_sorted.shape[0] - 1)
    ok = (idx < cell_id_int_sorted.shape[0]) & (cell_id_int_sorted[idx_clipped] == ids)
    if not np.all(ok):
        missing = ids[~ok]
        raise KeyError(
            f"{missing.size} cell_id_int values were not found in scConcept embeddings. "
            f"Example missing id(s): {missing[:5].tolist()}"
        )

    # h5py fancy indexing requires increasing indices: unique(sorted) read + inverse map.
    uniq_idx, inv = np.unique(idx.astype(np.int64, copy=False), return_inverse=True)
    emb_uniq = h5_file["cls_emb"][uniq_idx]  # [U, D], typically float16
    return emb_uniq[inv]                      # [N, D] aligned to input order
