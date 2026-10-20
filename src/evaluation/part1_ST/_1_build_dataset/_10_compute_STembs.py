#!/usr/bin/env python
"""
Compute the ST CLIP embeddings of the STHELAR eval slides with the exported projection heads of a HistoST training:
per patch (st_proj on the precomputed Novae teacher st_feat) and per cell (st_node_proj on the precomputed scConcept cls_emb).

Saved in <ROOT_HistoST> / eval / eval_datasets / st_eval_data / eval_st_<training_dataset_id> / st_sthelar_embeddings_clip /
    - GRAPHembs_t<training_num>e<checkpoint_num> / <slide_id>.h5ad: patch graph embeddings
    - CELLembs_t<training_num>e<checkpoint_num> / <slide_id>.h5ad: cell embeddings
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

# Add project root
project_root = Path(__file__).resolve().parents[4]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.constants import get_full_path
from src.utils.cell_ids import str_cell_id
# ST cell encoder and untrained-projection check, shared with extract_encoders.py
from src.evaluation.extract_encoders import STCellClipEncoder, patch_projections_are_untrained


def read_patch_ids_from_h5(h5_path: Path) -> List[str]:
    with h5py.File(h5_path, "r") as f:
        bc = f["barcode"][:].reshape(-1)
    out = []
    for x in bc:
        if isinstance(x, (bytes, np.bytes_)):
            out.append(bytes(x).decode("utf-8"))
        else:
            out.append(str(x))
    return out


def parse_training_num(training_id: str) -> str:
    return str(training_id).split("_")[-1]


def make_embedding_var_names(dim: int) -> pd.Index:
    width = max(4, len(str(int(dim))))
    return pd.Index([f"emb_{i:0{width}d}" for i in range(int(dim))])


def read_eval_cell_obs(eval_h5ad_path: Path, slide_id: str) -> pd.DataFrame:
    """
    Read only obs from eval ST .h5ad and keep cell rows.
    """
    a = ad.read_h5ad(eval_h5ad_path, backed="r")
    try:
        obs = a.obs.copy()
    finally:
        if getattr(a, "file", None) is not None:
            a.file.close()

    if "is_cell" in obs.columns:
        mask = obs["is_cell"].astype(bool).values
    else:
        mask = np.array([str(idx).startswith(f"{slide_id}_cell_") for idx in obs.index], dtype=bool)

    obs = obs.loc[mask].copy()

    if "cell_id_int" not in obs.columns:
        # derive from index {slide_id}_cell_{cell_id_int}
        pat = re.compile(rf"^{re.escape(slide_id)}_cell_(\d+)$")
        vals = []
        for idx in obs.index.astype(str):
            m = pat.match(idx)
            if m is None:
                raise ValueError(f"Could not parse cell_id_int from obs index '{idx}' for slide '{slide_id}'.")
            vals.append(int(m.group(1)))
        obs["cell_id_int"] = np.asarray(vals, dtype=np.int32)

    if "cell_id" not in obs.columns:
        obs["cell_id"] = [str_cell_id(int(x)) for x in obs["cell_id_int"].tolist()]

    obs["cell_id"] = obs["cell_id"].astype(str)
    obs["cell_id_int"] = obs["cell_id_int"].astype(np.int32)
    return obs


# ---------------------------------------------------------------------
# Exported ST encoder wrappers (projection heads only; teachers are precomputed)
# ---------------------------------------------------------------------


class STGraphProjEncoder(nn.Module):
    """ST graph projection head st_proj (st_feature_dim -> embed_dim, no bias), applied to precomputed per-patch st_feat."""
    def __init__(self, st_feature_dim: int, embed_dim: int, has_proj: bool = True) -> None:
        super().__init__()
        self.st_feature_dim = int(st_feature_dim)
        self.embed_dim = int(embed_dim)
        self.has_proj = bool(has_proj)
        if self.has_proj:
            self.st_proj = nn.Linear(self.st_feature_dim, self.embed_dim, bias=False)
            nn.init.normal_(self.st_proj.weight, std=self.st_feature_dim ** -0.5)
        else:
            self.st_proj = None

    def forward(self, st_feat: torch.Tensor, project: bool = True) -> torch.Tensor:
        if project:
            if self.st_proj is None:
                raise RuntimeError(
                    "This exported ST graph encoder does not contain `st_proj` (backbone-only export with "
                    "checkpoint_epoch < 0, or lambda_patch=0 arm with an untrained patch projection)."
                )
            return self.st_proj(st_feat)
        return st_feat

    def encode_graph(self, st_feat: torch.Tensor, project: bool = True) -> torch.Tensor:
        return self.forward(st_feat, project=project)


def load_st_graph_encoder(bundle_path: Path, device: torch.device) -> Tuple[STGraphProjEncoder, Dict, Optional[str]]:
    """Load the exported ST graph projection head (st_proj). Returns (model, config, novae_training_id of the teacher cache)."""
    bundle = torch.load(bundle_path, map_location="cpu", weights_only=False)
    config = bundle["config"]
    state_dict = bundle["state_dict"]

    embed_dim = int(config["model"]["clip"].get("clip_embed_dim", 256))

    # st_feature_dim (Novae output size): from the bundle, else the st_proj weight, else the config (default 128)
    st_feature_dim = bundle.get("st_feature_dim", None)
    if st_feature_dim is None and "st_proj.weight" in state_dict:
        st_feature_dim = int(state_dict["st_proj.weight"].shape[1])
    if st_feature_dim is None:
        st_feature_dim = int(config.get("model", {}).get("st", {}).get("st_feature_dim", 128) or 128)

    has_proj = "st_proj.weight" in state_dict

    # lambda_patch=0: st_proj was never trained, so it is treated as absent (and removed from the state dict)
    if has_proj and patch_projections_are_untrained(config["model"]["clip"]):
        print(
            "lambda_patch=0.0: the `st_proj` in this bundle was never trained (random init). "
            "The ST graph head is treated as absent."
        )
        has_proj = False
        state_dict = {k: v for k, v in state_dict.items() if k != "st_proj.weight"}

    model = STGraphProjEncoder(st_feature_dim=int(st_feature_dim), embed_dim=embed_dim, has_proj=has_proj)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    novae_training_id = bundle.get("novae_training_id", None)
    if novae_training_id is not None:
        novae_training_id = str(novae_training_id).strip() or None
    return model, config, novae_training_id


def load_st_cell_encoder(bundle_path: Path, device: torch.device) -> Tuple[STCellClipEncoder, Dict, str]:
    bundle = torch.load(bundle_path, map_location="cpu", weights_only=False)
    config = bundle["config"]
    state_dict = bundle["state_dict"]

    clip_cfg = config["model"]["clip"]
    embed_dim = int(clip_cfg.get("clip_embed_dim", 256))
    lambda_token = float(clip_cfg.get("lambda_token", 0.0))
    token_embed_dim = clip_cfg.get("token_embed_dim", None)
    if token_embed_dim is None:
        token_embed_dim = embed_dim
    token_embed_dim = int(token_embed_dim)
    in_dim = int(clip_cfg.get("scconcept_embedding_dim", 512))
    has_proj = (lambda_token > 0.0)

    scconcept_training_id = bundle.get("scconcept_training_id", None)
    if scconcept_training_id is None:
        raise KeyError(
            f"Exported ST cell bundle is missing top-level 'scconcept_training_id': {bundle_path}. "
            "Re-run extract_encoders.py."
        )
    scconcept_training_id = str(scconcept_training_id).strip()
    if scconcept_training_id == "" or scconcept_training_id.lower() in ("none", "null"):
        raise ValueError(f"Invalid scconcept_training_id in ST cell bundle: {bundle_path}")

    # Bundle metadata and embedded config must agree
    cfg_scconcept_training_id = str(
        config.get("model", {}).get("st", {}).get("scconcept_training_id", "")
    ).strip()
    if cfg_scconcept_training_id != scconcept_training_id:
        raise ValueError(
            "Mismatch between bundle['scconcept_training_id'] and bundle['config']['model']['st']['scconcept_training_id'] "
            f"for {bundle_path}: '{scconcept_training_id}' vs '{cfg_scconcept_training_id}'."
        )

    model = STCellClipEncoder(in_dim=in_dim, out_dim=token_embed_dim, has_proj=has_proj)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model, config, scconcept_training_id


def resolve_scconcept_embeddings_dir(scconcept_training_id: str) -> Path:
    scconcept_training_id = str(scconcept_training_id).strip()
    if scconcept_training_id == "" or scconcept_training_id.lower() in ("none", "null"):
        raise ValueError("Invalid scconcept_training_id; cannot resolve scConcept embeddings directory.")

    scconcept_dir = get_full_path("train", "trainings", "trainings_scConcept", scconcept_training_id, "embeddings")
    if not scconcept_dir.exists():
        raise FileNotFoundError(
            f"Resolved scConcept embeddings directory does not exist: {scconcept_dir}"
        )
    return scconcept_dir


# ---------------------------------------------------------------------
# scConcept loading (cell branch)
# ---------------------------------------------------------------------


def fetch_scconcept_embeddings(
    h5_path: Path,
    cell_ids_int: np.ndarray,
) -> np.ndarray:
    """
    Fetch embeddings aligned with cell_ids_int order.
    """
    if cell_ids_int.ndim != 1:
        raise ValueError(f"cell_ids_int must be 1D, got shape={cell_ids_int.shape}")

    with h5py.File(h5_path, "r") as f:
        if "cell_id_int" not in f or "cls_emb" not in f:
            raise KeyError(f"Invalid scConcept file '{h5_path}': missing 'cell_id_int' or 'cls_emb'")

        all_ids = f["cell_id_int"][:].astype(np.int32, copy=False)
        if all_ids.ndim != 1:
            raise ValueError(f"'cell_id_int' in '{h5_path}' must be 1D, got shape={all_ids.shape}")

        if all_ids.size == 0:
            if cell_ids_int.size == 0:
                emb_dim = int(f["cls_emb"].shape[1])
                return np.empty((0, emb_dim), dtype=np.float32)
            raise KeyError(f"scConcept file '{h5_path}' contains no cell_id_int entries.")

        # Fast path if file is sorted; safe fallback otherwise
        if np.all(all_ids[1:] >= all_ids[:-1]):
            idx = np.searchsorted(all_ids, cell_ids_int)
            idx_clip = np.clip(idx, 0, max(0, all_ids.shape[0] - 1))
            ok = (idx < all_ids.shape[0]) & (all_ids[idx_clip] == cell_ids_int)
            if not np.all(ok):
                missing = cell_ids_int[~ok][:10].tolist()
                raise KeyError(
                    f"{np.sum(~ok)} cell_id_int values were not found in scConcept file '{h5_path}'. "
                    f"Examples: {missing}"
                )
        else:
            id_to_pos = {int(cid): i for i, cid in enumerate(all_ids.tolist())}
            idx = np.asarray([id_to_pos.get(int(cid), -1) for cid in cell_ids_int.tolist()], dtype=np.int32)
            ok = idx >= 0
            if not np.all(ok):
                missing = cell_ids_int[~ok][:10].tolist()
                raise KeyError(
                    f"{np.sum(~ok)} cell_id_int values were not found in scConcept file '{h5_path}'. "
                    f"Examples: {missing}"
                )

        uniq_idx, inv = np.unique(idx.astype(np.int32, copy=False), return_inverse=True)
        emb_uniq = f["cls_emb"][uniq_idx]
        emb = emb_uniq[inv]

    return np.asarray(emb, dtype=np.float32)


# ---------------------------------------------------------------------
# Frozen Novae ST-graph teacher cache (precomputed per-patch st_feat)
# ---------------------------------------------------------------------


def resolve_novae_teacher_cache_dir(
    novae_training_id: str, dataset_id: str, training_dataset_id: str
) -> Tuple[Path, Dict]:
    """Precomputed Novae teacher cache directory (layout of precompute_novae_teacher.py) and its manifest."""
    cache_dir = get_full_path(
        "train", "training_datasets", dataset_id, training_dataset_id,
        "novae_teacher_cache", novae_training_id, scratch=True,
    )
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"novae_teacher_cache manifest not found: {manifest_path}. Build it with "
            f"src/dataset/precompute_novae_teacher.py --novae_training_id {novae_training_id} "
            "--folds train,valid,test (the 'test' fold covers the STHELAR eval patches)."
        )
    with manifest_path.open("r") as f:
        manifest = json.load(f)
    return cache_dir, manifest


def load_slide_st_feat(
    cache_dir: Path, slide_id: str, expected_fingerprint: Optional[str]
) -> Tuple[np.ndarray, np.ndarray]:
    """Load one slide of the teacher cache: (sorted patch_idx [n], st_feat [n, 128]); checks its fingerprint against the manifest."""
    h5_path = cache_dir / f"{slide_id}.h5"
    if not h5_path.exists():
        raise FileNotFoundError(
            f"novae_teacher_cache slide file missing: {h5_path}. The precompute must cover the "
            "STHELAR eval (test-fold) patches -> run precompute_novae_teacher.py with "
            "--folds train,valid,test."
        )
    with h5py.File(h5_path, "r") as f:
        fp = f.attrs.get("fingerprint", None)
        if expected_fingerprint is not None and fp is not None and str(fp) != str(expected_fingerprint):
            raise ValueError(
                f"Teacher-cache fingerprint mismatch for {h5_path}: cache '{fp}' != manifest "
                f"'{expected_fingerprint}'. The cache is stale; rebuild it with precompute_novae_teacher.py."
            )
        patch_idx = f["patch_idx"][:].astype(np.int64, copy=False)
        st_feat = f["st_feat"][:].astype(np.float32, copy=False)
    order = np.argsort(patch_idx, kind="stable")
    return patch_idx[order], st_feat[order]


def st_feat_for_patch_ids(
    patch_idx_sorted: np.ndarray,
    st_feat_sorted: np.ndarray,
    patch_ids: Sequence[str],
    slide_id: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    st_feat rows of the eval patch_ids (in order), matched on the numeric patch suffix (cache patch_idx).
    Returns (st_feat_rows [P, 128], is_dummy [P]); is_dummy marks all-zero rows (degenerate/empty patches).
    """
    suffixes = []
    for pid in patch_ids:
        s = str(pid).rsplit("_", 1)[-1]
        if not s.isdigit():
            raise ValueError(f"Unexpected patch_id (numeric suffix expected): '{pid}'")
        suffixes.append(int(s))
    suffixes = np.asarray(suffixes, dtype=np.int64)

    idx = np.searchsorted(patch_idx_sorted, suffixes)
    idx_clip = np.clip(idx, 0, max(0, patch_idx_sorted.shape[0] - 1))
    ok = (idx < patch_idx_sorted.shape[0]) & (patch_idx_sorted[idx_clip] == suffixes)
    if not np.all(ok):
        missing = [str(patch_ids[i]) for i in np.flatnonzero(~ok)[:10].tolist()]
        raise KeyError(
            f"{int(np.sum(~ok))} eval patch(es) of slide '{slide_id}' are absent from the "
            f"novae_teacher_cache. Examples: {missing}. Re-run precompute_novae_teacher.py with "
            "the test fold (--folds train,valid,test) so the STHELAR eval patches are covered."
        )
    rows = st_feat_sorted[idx_clip]
    is_dummy = ~np.any(rows != 0.0, axis=1)  # precompute caches zeros for degenerate/empty patches
    return rows, is_dummy


# ---------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------


def save_graph_embeddings_h5ad(
    out_path: Path,
    emb: np.ndarray,
    patch_ids: Sequence[str],
    slide_id: str,
    is_dummy_graph: Sequence[int],
    uns_meta: Dict,
) -> None:
    obs = pd.DataFrame(
        {
            "slide_id": [slide_id] * len(patch_ids),
            "patch_id": list(patch_ids),
            # n_cells_graph = -1 (not stored in the teacher cache); is_dummy_graph=1 for degenerate/empty patches
            "n_cells_graph": np.full(len(patch_ids), -1, dtype=np.int32),
            "is_dummy_graph": np.asarray(is_dummy_graph, dtype=np.int8),
            "st_feat_source": ["novae_teacher_cache"] * len(patch_ids),
        },
        index=pd.Index(list(patch_ids), name=None),
    )
    var = pd.DataFrame(index=make_embedding_var_names(emb.shape[1]))
    adata = ad.AnnData(X=np.asarray(emb, dtype=np.float32), obs=obs, var=var)
    adata.uns.update(uns_meta)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(out_path)


def save_cell_embeddings_h5ad(
    out_path: Path,
    emb: np.ndarray,
    obs_cell: pd.DataFrame,
    uns_meta: Dict,
) -> None:
    obs = obs_cell.copy()
    obs.index = pd.Index(obs.index.astype(str), name=None)
    var = pd.DataFrame(index=make_embedding_var_names(emb.shape[1]))
    adata = ad.AnnData(X=np.asarray(emb, dtype=np.float32), obs=obs, var=var)
    adata.uns.update(uns_meta)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(out_path)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main(args) -> None:

    # Device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Paths
    dataset_id = args.dataset_id
    training_dataset_id = args.training_dataset_id
    training_id = args.training_id
    checkpoint_num = int(args.checkpoint_num)
    training_num = parse_training_num(training_id)

    eval_root = get_full_path("eval", "eval_datasets", "st_eval_data", f"eval_st_{training_dataset_id}", scratch=True)
    metadata_csv = get_full_path("eval", "eval_datasets", "st_eval_data", f"eval_st_{training_dataset_id}", f"metadata_eval_st_{training_dataset_id}.csv", scratch=True)

    patches_eval_dir = eval_root / "patches"
    st_eval_dir = eval_root / "st"

    graph_out_dir = get_full_path(
        "eval", "eval_datasets", "st_eval_data", f"eval_st_{training_dataset_id}",
        "st_sthelar_embeddings_clip", f"GRAPHembs_t{training_num}e{checkpoint_num}",
        scratch=True,
    )
    cell_out_dir = get_full_path(
        "eval", "eval_datasets", "st_eval_data", f"eval_st_{training_dataset_id}",
        "st_sthelar_embeddings_clip", f"CELLembs_t{training_num}e{checkpoint_num}",
        scratch=True,
    )

    # Exported encoders
    trained_encoders_dir = get_full_path("train", "trainings", "trainings_HistoST", training_id, "trained_encoders")
    graph_bundle_path = trained_encoders_dir / f"STencoderGraph_t{training_num}e{checkpoint_num}.pt"
    cell_bundle_path = trained_encoders_dir / f"STencoderCell_t{training_num}e{checkpoint_num}.pt"

    if not metadata_csv.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {metadata_csv}")

    graph_model = None
    cell_model = None
    scconcept_training_id = None
    scconcept_dir = None
    novae_training_id = None
    teacher_cache_dir = None
    teacher_fingerprint = None

    # --- GRAPH branch setup: st_proj head + precomputed Novae teacher cache -------------------
    if args.do_graph:
        if not graph_bundle_path.exists():
            raise FileNotFoundError(f"Graph encoder bundle not found: {graph_bundle_path}")
        print("Loading exported ST graph projection head ...")
        graph_model, _graph_config, bundle_novae_training_id = load_st_graph_encoder(graph_bundle_path, device=device)
        if graph_model.has_proj is False:
            raise ValueError(
                "Graph encoder bundle has no usable st_proj head (checkpoint_num=-1 baseline, or lambda_patch=0 "
                "arm with an untrained st_proj); cannot compute CLIP-space graph embeddings. Run --do_cell only."
            )

        novae_training_id = (args.novae_training_id or bundle_novae_training_id)
        if not novae_training_id:
            raise ValueError(
                "Could not resolve the Novae teacher training id: the STencoderGraph bundle carries no "
                "'novae_training_id' and --novae_training_id was not passed. Re-export with "
                "extract_encoders.py, or pass --novae_training_id."
            )
        teacher_cache_dir, teacher_manifest = resolve_novae_teacher_cache_dir(
            novae_training_id, dataset_id, training_dataset_id
        )
        teacher_fingerprint = teacher_manifest.get("fingerprint", None)
        teacher_dim = int(teacher_manifest.get("dim", graph_model.st_feature_dim))
        if teacher_dim != graph_model.st_feature_dim:
            raise ValueError(
                f"Teacher-cache st_feat dim ({teacher_dim}) != st_proj input dim "
                f"({graph_model.st_feature_dim}). The STencoderGraph bundle and the novae_teacher_cache "
                "do not match; re-export/rebuild consistently."
            )
        print(f"Novae teacher '{novae_training_id}' cache: {teacher_cache_dir} (st_feat dim={teacher_dim}).")

    # --- CELL branch setup: st_node_proj head + precomputed scConcept -------------------------
    if args.do_cell:
        if not cell_bundle_path.exists():
            raise FileNotFoundError(f"Cell encoder bundle not found: {cell_bundle_path}")
        print("Loading exported ST cell encoder ...")
        cell_model, _cell_config, scconcept_training_id = load_st_cell_encoder(cell_bundle_path, device=device)
        if cell_model.has_proj is False:
            raise ValueError("Cell encoder was loaded but it does not contain a projection head. Cannot compute cell embeddings.")
        scconcept_dir = resolve_scconcept_embeddings_dir(scconcept_training_id)
        print(f"Resolved scConcept embeddings dir: {scconcept_dir}")

    # STHELAR slides only in evaluation dataset
    metadata = pd.read_csv(metadata_csv)
    if "source" not in metadata.columns or "slide_id" not in metadata.columns:
        raise KeyError("metadata csv must contain 'source' and 'slide_id'")
    slide_ids = sorted(metadata.loc[metadata["source"] == "STHELAR", "slide_id"].astype(str).unique().tolist())
    if len(slide_ids) == 0:
        raise ValueError("No STHELAR slides found in metadata.")

    if args.do_graph:
        graph_out_dir.mkdir(parents=True, exist_ok=True)
    if args.do_cell:
        cell_out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Will process {len(slide_ids)} STHELAR eval slide(s).")
    if args.do_graph:
        print(f"Graph output dir: {graph_out_dir}")
    if args.do_cell:
        print(f"Cell output dir:  {cell_out_dir}")

    # Process each slide
    for i_slide, slide_id in enumerate(slide_ids, start=1):
        print(f"\n[{i_slide}/{len(slide_ids)}] Slide '{slide_id}'")

        graph_out_path = (graph_out_dir / f"{slide_id}.h5ad") if args.do_graph else None
        cell_out_path = (cell_out_dir / f"{slide_id}.h5ad") if args.do_cell else None

        if not args.overwrite:
            graph_done = (not args.do_graph) or graph_out_path.exists()
            cell_done = (not args.do_cell) or cell_out_path.exists()
            if graph_done and cell_done:
                print("  Outputs already exist, skipping (use --overwrite to recompute).")
                continue

        eval_h5_path = patches_eval_dir / f"{slide_id}.h5"
        eval_h5ad_path = st_eval_dir / f"{slide_id}.h5ad"
        if not eval_h5ad_path.exists():
            raise FileNotFoundError(f"Required file missing for slide '{slide_id}': {eval_h5ad_path}")

        if args.do_graph and not eval_h5_path.exists():
            raise FileNotFoundError(f"Required file missing for slide '{slide_id}': {eval_h5_path}")

        if args.do_cell:
            scconcept_h5_path = scconcept_dir / f"{slide_id}.h5"
            if not scconcept_h5_path.exists():
                raise FileNotFoundError(
                    f"scConcept embeddings file missing for slide '{slide_id}' in resolved run "
                    f"'{scconcept_training_id}': {scconcept_h5_path}"
                )

        # 1) GRAPH embeddings per patch (project the precomputed teacher st_feat) --------------
        if args.do_graph:
            patch_ids = read_patch_ids_from_h5(eval_h5_path)
            if len(patch_ids) == 0:
                raise ValueError(f"No patch IDs found in eval H5: {eval_h5_path}")

            patch_idx_sorted, st_feat_sorted = load_slide_st_feat(teacher_cache_dir, slide_id, teacher_fingerprint)
            if st_feat_sorted.shape[1] != graph_model.st_feature_dim:
                raise ValueError(
                    f"Cached st_feat dim ({st_feat_sorted.shape[1]}) != st_proj input dim "
                    f"({graph_model.st_feature_dim}) for slide '{slide_id}'."
                )

            st_feat_rows, is_dummy = st_feat_for_patch_ids(patch_idx_sorted, st_feat_sorted, patch_ids, slide_id)

            print(f"  Projecting {len(patch_ids)} patch st_feat -> CLIP space ...")
            graph_emb_chunks: List[np.ndarray] = []
            bs = int(args.graph_batch_size)
            for start in tqdm(range(0, st_feat_rows.shape[0], bs), desc=f"  Graph {slide_id}"):
                end = min(start + bs, st_feat_rows.shape[0])
                x = torch.from_numpy(np.ascontiguousarray(st_feat_rows[start:end])).to(device=device, dtype=torch.float32)
                with torch.no_grad():
                    emb = graph_model.encode_graph(x, project=True)
                graph_emb_chunks.append(emb.detach().cpu().float().numpy())

            graph_emb = (
                np.concatenate(graph_emb_chunks, axis=0)
                if graph_emb_chunks else np.empty((0, graph_model.embed_dim), dtype=np.float32)
            )
            if graph_emb.shape[0] != len(patch_ids):
                raise RuntimeError(
                    f"Graph embedding count mismatch for slide '{slide_id}': "
                    f"{graph_emb.shape[0]} embeddings vs {len(patch_ids)} patch ids"
                )

            save_graph_embeddings_h5ad(
                out_path=graph_out_path,
                emb=graph_emb,
                patch_ids=patch_ids,
                slide_id=slide_id,
                is_dummy_graph=is_dummy.astype(np.int8),
                uns_meta={
                    "embedding_type": "ST_graph_clip_projected",
                    "training_id": training_id,
                    "checkpoint_num": checkpoint_num,
                    "training_dataset_id": training_dataset_id,
                    "dataset_id": dataset_id,
                    "st_feat_source": "novae_teacher_cache",
                    "novae_training_id": novae_training_id,
                    "novae_teacher_fingerprint": teacher_fingerprint if teacher_fingerprint is not None else "",
                    "st_feature_dim": int(graph_model.st_feature_dim),
                    "n_dummy_patches": int(np.sum(is_dummy)),
                },
            )
            print(f"  Saved graph embeddings: {graph_out_path}  (dummy patches: {int(np.sum(is_dummy))})")

        # 2) CELL embeddings per evaluation cell -----------------------------------------------
        if args.do_cell:
            print("  Loading evaluation cells + scConcept embeddings ...")
            obs_cell = read_eval_cell_obs(eval_h5ad_path, slide_id=slide_id)
            cell_ids_int = obs_cell["cell_id_int"].to_numpy(dtype=np.int32, copy=False)
            sc_emb = fetch_scconcept_embeddings(scconcept_h5_path, cell_ids_int=cell_ids_int)

            if sc_emb.ndim != 2:
                raise ValueError(f"scConcept embeddings must be 2D, got shape={sc_emb.shape}")

            out_chunks = []
            bs = int(args.cell_batch_size)
            print(f"  Encoding {sc_emb.shape[0]} evaluation cell(s) ...")
            out_dim = int(cell_model.out_dim if cell_model.has_proj else cell_model.in_dim)
            if sc_emb.shape[0] == 0:
                cell_emb = np.empty((0, out_dim), dtype=np.float32)
            else:
                for start in tqdm(range(0, sc_emb.shape[0], bs), desc=f"  Cell  {slide_id}"):
                    end = min(start + bs, sc_emb.shape[0])
                    x = torch.from_numpy(sc_emb[start:end]).to(device=device, dtype=torch.float32)
                    with torch.no_grad():
                        y = cell_model.encode_cell(x, project=True)
                    out_chunks.append(y.detach().cpu().float().numpy())

                cell_emb = np.concatenate(out_chunks, axis=0)
            if cell_emb.shape[0] != obs_cell.shape[0]:
                raise RuntimeError(
                    f"Cell embedding count mismatch for slide '{slide_id}': "
                    f"{cell_emb.shape[0]} embeddings vs {obs_cell.shape[0]} eval cells"
                )

            save_cell_embeddings_h5ad(
                out_path=cell_out_path,
                emb=cell_emb,
                obs_cell=obs_cell,
                uns_meta={
                    "embedding_type": "ST_cell_clip_projected",
                    "training_id": training_id,
                    "checkpoint_num": checkpoint_num,
                    "training_dataset_id": training_dataset_id,
                    "dataset_id": dataset_id,
                    "source_embeddings": "scConcept",
                    "scconcept_training_id": scconcept_training_id,
                    "has_learned_projection": bool(cell_model.has_proj),
                },
            )
            print(f"  Saved cell embeddings:  {cell_out_path}")

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract ST graph and cell CLIP embeddings on STHELAR evaluation data.")

    # Data ---

    parser.add_argument("--dataset_id", type=str,
                        default="ds_ct1_20x_224_o112",
                        help="Dataset ID used (e.g. 'ds_ct1_20x_224_o112')")

    parser.add_argument("--training_dataset_id", type=str,
                        default="ds_ct1_20x_224_o112_1",
                        help="Training dataset ID used (e.g. 'ds_ct1_20x_224_o112_1')")

    parser.add_argument("--training_id", type=str,
                        default="training_46",
                        help="HistoST training ID used (e.g. 'training_46')")

    parser.add_argument("--checkpoint_num", type=int,
                        default=13,
                        help="Checkpoint number to load (e.g. '13' for 'checkpoint_epoch_13.pth')")

    parser.add_argument("--novae_training_id", type=str,
                        default=None,
                        help="Novae teacher training id of the novae_teacher_cache (default: the id stored in the STencoderGraph bundle).")

    # Technical parameters ---

    parser.add_argument("--device", type=str,
                        default=None,
                        help="e.g. cuda, cuda:0, mps, cpu. Default: cuda if available elif mps if available else cpu")

    parser.add_argument("--graph_batch_size", type=int,
                        default=256,
                        help="Batch size for projecting patch st_feat")

    parser.add_argument("--cell_batch_size", type=int,
                        default=8192,
                        help="Batch size for encoding cells")

    # What to compute ---

    parser.add_argument("--do_graph", action="store_true", help="Compute graph embeddings")
    parser.add_argument("--do_cell", action="store_true", help="Compute cell embeddings")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")

    args = parser.parse_args()
    if (not args.do_graph) and (not args.do_cell):  # default = do both if neither flag is explicitly given
        args.do_graph = True
        args.do_cell = True
    main(args)
