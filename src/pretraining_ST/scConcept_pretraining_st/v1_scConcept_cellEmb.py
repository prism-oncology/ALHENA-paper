#!/usr/bin/env python3
"""
- Version 1, for model Corpus-30M -

Light-adapt the pretrained scConcept on the Xenium training cells of a training_dataset_id,
then extract the embeddings of every cell of every slide.

Saved in <ROOT_HistoST> / train / trainings / trainings_scConcept / <training_id> /
    - embeddings/<slide_id>.h5: cell ids and cell embeddings (cls_emb, optional mean_emb)
    - manifest.json: parameters and vocabulary coverage per slide
    - scConcept_<training_id>.ckpt: adapted checkpoint (if --save_adapted_ckpt)
"""

import sys
from pathlib import Path
import argparse
import json
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence
import numpy as np
import torch
import anndata as ad
import h5py
from tqdm import tqdm

# Add the project root to sys.path so that 'src' can be imported
project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.constants import get_full_path
from src.utils.cell_ids import int_cell_id_vec


# On non-CUDA platforms (e.g. Mac), flash_attn is unavailable ----------
# CPU stub for flash_attn (local debugging only)
import sys as _sys

try:
    import flash_attn  # already available (CUDA machine)
    print(f"flash_attn version {flash_attn.__version__} imported successfully.")

except ModuleNotFoundError:
    print("flash_attn not found => using CPU stub (for debugging on non-CUDA machines only).")
    # flash_attn is required on Jean Zay
    from src.utils.constants import on_jeanzay
    if on_jeanzay():
        raise ModuleNotFoundError("flash_attn not found. This should not happen on Jean Zay.")
    
    import types, torch, torch.nn as _nn
    _fa = types.ModuleType("flash_attn")
    _mha = types.ModuleType("flash_attn.modules.mha")

    class _MHA(_nn.Module):
        """
        CPU stub for flash_attn.modules.mha.MHA.
        Stores weights as a single fused Wqkv (matching checkpoint keys),
        then splits them at forward time for F.scaled_dot_product_attention.
        """
        def __init__(self, embed_dim, num_heads, bias=True, **kwargs):
            super().__init__()
            self.embed_dim = embed_dim
            self.num_heads = num_heads
            self.head_dim  = embed_dim // num_heads
            # fused QKV weight — matches checkpoint key "self_attn.Wqkv.weight"
            self.Wqkv = _nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
            self.out_proj = _nn.Linear(embed_dim, embed_dim, bias=bias)

        def forward(self, x, *args, **kwargs):
            B, T, C = x.shape
            qkv = self.Wqkv(x)                        # (B, T, 3*C)
            q, k, v = qkv.chunk(3, dim=-1)
            # reshape to (B, heads, T, head_dim)
            def split_heads(t):
                return t.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
            q, k, v = split_heads(q), split_heads(k), split_heads(v)
            attn_out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
            # merge heads back
            attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, C)
            return self.out_proj(attn_out)

    _mha.MHA = _MHA
    _fa.modules = types.ModuleType("flash_attn.modules")
    _fa.modules.mha = _mha
    _sys.modules["flash_attn"] = _fa
    _sys.modules["flash_attn.modules"] = _fa.modules
    _sys.modules["flash_attn.modules.mha"] = _mha
# -----------------------------------------------------------------------
    
    
# scConcept + lamin_dataloader (dependencies of scConcept)
from concept import scConcept
from lamin_dataloader import TokenizedDataset, BaseCollate, InMemoryCollection
from torch.utils.data import DataLoader


# -----------------------------
# Modify scConcept config for adaptation
# -----------------------------

def _set_cfg_attr(cfg, dotted_key: str, value):
    """
    Set an OmegaConf key that is expected to exist.
    Raises KeyError if the key path does not exist.
    """
    parts = dotted_key.split(".")
    cur = cfg
    for p in parts[:-1]:
        if not hasattr(cur, p):
            raise KeyError(f"Config has no key path '{dotted_key}' (missing '{p}')")
        cur = getattr(cur, p)
    last = parts[-1]
    if not hasattr(cur, last):
        raise KeyError(f"Config has no key path '{dotted_key}' (missing '{last}')")
    setattr(cur, last, value)



def patch_scconcept_cfg_for_adaptation(
    concept: scConcept,
    *,
    token_budget: int,
    gene_sampling_strategy: Optional[str],
    loss_switch_step: int,
    panel_selection: str,
    panel_size_min: int,
    panel_size_max: int,
):
    """
    Set the scConcept config for adaptation: max_tokens, gene sampling strategy, loss_switch_step,
    panel selection and panel_size_min/max.
    """
    cfg = concept.cfg

    # 1) Token budget during adaptation
    _set_cfg_attr(cfg, "datamodule.dataset.train.max_tokens", int(token_budget))

    # 2) Keep sampling strategy consistent between adaptation + extraction
    if gene_sampling_strategy is not None:
        _set_cfg_attr(cfg, "datamodule.gene_sampling_strategy", str(gene_sampling_strategy))

    # 3) Step at which the panel-invariance loss starts
    _set_cfg_attr(cfg, "model.loss_switch_step", int(loss_switch_step))

    # 4) Random panels with bounded panel size
    _set_cfg_attr(cfg, "datamodule.dataset.train.panel_selection", str(panel_selection))
    _set_cfg_attr(cfg, "datamodule.dataset.train.panel_size_min", int(panel_size_min))
    _set_cfg_attr(cfg, "datamodule.dataset.train.panel_size_max", int(panel_size_max))



def get_cell_id_strings(adata: ad.AnnData) -> np.ndarray:
    """
    Get adata.obs['cell_id'] if present; else raise error
    """
    if "cell_id" in adata.obs.columns:
        return adata.obs["cell_id"].astype(str).to_numpy()
    else:
        raise KeyError("adata.obs['cell_id'] not found => cannot extract cell IDs.")


def mask_from_sorted_ids(cell_ids_int: np.ndarray, keep_sorted_ids: np.ndarray) -> np.ndarray:
    """
    Fast membership test when keep_sorted_ids is sorted ascending as in the npz.
    """
    if keep_sorted_ids.size == 0:
        return np.zeros(cell_ids_int.shape[0], dtype=bool)
    idx = np.searchsorted(keep_sorted_ids, cell_ids_int)
    idx = np.clip(idx, 0, keep_sorted_ids.size - 1)
    return keep_sorted_ids[idx] == cell_ids_int


# -----------------------------
# AnnData preparation for scConcept
# -----------------------------
def ensure_gene_id_index(adata: ad.AnnData, gene_id_column: Optional[str]) -> None:
    """
    Set adata.var_names to adata.var[gene_id_column] (scConcept's AnnDataModule reads the gene IDs from var_names).
    """
    if gene_id_column is None:
        return
    if gene_id_column not in adata.var.columns:
        raise KeyError(f"gene_id_column='{gene_id_column}' not found in adata.var. Columns={list(adata.var.columns)}")

    gene_ids = adata.var[gene_id_column].astype(str).to_numpy()
    if len(np.unique(gene_ids)) != len(gene_ids):
        # duplicates would break a clean var_names assignment
        raise ValueError(f"adata.var['{gene_id_column}'] contains duplicate gene IDs => cannot safely set var_names.")
    # keep the gene symbols in var['gene_symbol']
    if "gene_symbol" not in adata.var.columns:
        adata.var["gene_symbol"] = adata.var_names.astype(str)
    adata.var_names = gene_ids



# -----------------------------
# Per-slide gene-panel vs corpus-vocabulary coverage
# -----------------------------
def panel_gene_ids_for_tokenizer(adata: ad.AnnData, gene_id_column: Optional[str]) -> np.ndarray:
    """Gene IDs tokenized by scConcept: adata.var[gene_id_column] if gene_id_column is set, else adata.var_names."""
    if gene_id_column is None:
        return adata.var_names.astype(str).to_numpy()
    if gene_id_column not in adata.var.columns:
        raise KeyError(f"gene_id_column='{gene_id_column}' not found in adata.var. Columns={list(adata.var.columns)}")
    return adata.var[gene_id_column].astype(str).to_numpy()


def compute_vocab_coverage(gene_ids: np.ndarray, tokenizer, species: Optional[str] = None) -> Dict[str, float]:
    """
    Count the panel genes of a slide that are in the scConcept corpus vocabulary.
    species: None for the v1 single-species tokenizer, set for a multi-species tokenizer.
    """
    tokens = tokenizer.encode(gene_ids) if species is None else tokenizer.encode(gene_ids, species)
    tokens = np.asarray(tokens)
    n_panel = int(len(gene_ids))
    n_in_vocab = int(np.count_nonzero(tokens != tokenizer.NOT_FOUND))
    frac = (n_in_vocab / n_panel) if n_panel > 0 else 0.0
    return {"n_genes_panel": n_panel, "n_genes_in_vocab": n_in_vocab, "vocab_coverage_frac": frac}


def preflight_vocab_coverage(
    slide_ids: Sequence[str],
    adatas_dir: Path,
    tokenizer,
    gene_id_column: Optional[str],
    min_coverage_frac: float,
    min_genes_in_vocab: int,
    species: Optional[str] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Vocabulary coverage of every slide (var read in backed mode); raises if a slide is below the thresholds.
    Returns {slide_id: {n_genes_panel, n_genes_in_vocab, vocab_coverage_frac}}.
    """
    coverage_by_slide: Dict[str, Dict[str, float]] = {}
    offenders: List[str] = []
    print("[Preflight] Checking per-slide gene-panel vs corpus-vocabulary coverage ...")
    for sid in slide_ids:
        p = adatas_dir / f"{sid}.h5ad"
        if not p.exists():
            raise FileNotFoundError(f"Missing h5ad for slide_id='{sid}': {p}")
        adata = ad.read_h5ad(p, backed="r")
        try:
            gene_ids = panel_gene_ids_for_tokenizer(adata, gene_id_column)
        finally:
            if getattr(adata, "isbacked", False) and adata.file is not None:
                adata.file.close()

        cov = compute_vocab_coverage(gene_ids, tokenizer, species=species)
        coverage_by_slide[sid] = cov
        ok = cov["vocab_coverage_frac"] >= min_coverage_frac and cov["n_genes_in_vocab"] >= min_genes_in_vocab
        print(
            f"[Preflight] {sid}: {cov['n_genes_in_vocab']}/{cov['n_genes_panel']} genes in vocab "
            f"({100.0 * cov['vocab_coverage_frac']:.1f}%)" + ("" if ok else "  <-- BELOW THRESHOLD")
        )
        if not ok:
            offenders.append(sid)

    if offenders:
        lines = [
            f"  - {sid}: {coverage_by_slide[sid]['n_genes_in_vocab']}/{coverage_by_slide[sid]['n_genes_panel']} "
            f"genes in vocab ({100.0 * coverage_by_slide[sid]['vocab_coverage_frac']:.1f}%)"
            for sid in offenders
        ]
        raise RuntimeError(
            f"scConcept vocabulary coverage too low for {len(offenders)}/{len(slide_ids)} slide(s) "
            f"(require coverage >= {100.0 * min_coverage_frac:.1f}% and >= {min_genes_in_vocab} genes in vocab):\n"
            + "\n".join(lines)
            + "\nVerify gene_id_column maps into the corpus ID space (e.g. unversioned Ensembl IDs), "
            "or lower --min_vocab_coverage_frac / --min_genes_in_vocab to override."
        )

    fracs = [c["vocab_coverage_frac"] for c in coverage_by_slide.values()]
    print(
        f"[Preflight] Vocabulary coverage OK for all {len(slide_ids)} slides "
        f"(min {100.0 * min(fracs):.1f}%, mean {100.0 * (sum(fracs) / len(fracs)):.1f}%)."
    )
    return coverage_by_slide


def subset_minimal_adata_for_scconcept(
    adata: ad.AnnData,
    obs_indexer: np.ndarray | slice,
) -> ad.AnnData:
    """
    Return a compact AnnData containing only X/obs/var for scConcept.
    """
    # SciPy sparse advanced indexing is safest with np.intp for integer indexers
    if isinstance(obs_indexer, np.ndarray) and obs_indexer.dtype != bool:
        obs_indexer = np.asarray(obs_indexer, dtype=np.intp)

    view = adata[obs_indexer, :]

    # Copy only what scConcept needs for train([...])
    X = view.X.copy() if hasattr(view.X, "copy") else np.asarray(view.X)
    obs = view.obs.copy()
    var = view.var.copy()

    return ad.AnnData(X=X, obs=obs, var=var)



# -----------------------------
# Data loading for adaptation
# -----------------------------
def load_train_adatas(
    slide_ids: Sequence[str],
    adatas_dir: Path,
    train_only_by_slide: Dict[str, np.ndarray],
    gene_id_column: Optional[str],
    max_train_cells_per_slide: Optional[int],
    seed: int = 0,
) -> List[ad.AnnData]:
    """
    Load only the slides that have training cells (train_only_by_slide[slide_id] non-empty),
    subset to those cells, and return a list of AnnData objects for scConcept.train([...]).
    """
    rng = np.random.default_rng(seed)
    train_adatas: List[ad.AnnData] = []

    for sid in slide_ids:
        keep_ids = np.asarray(train_only_by_slide.get(sid, np.array([], dtype=np.int64)), dtype=np.int64)
        if keep_ids.size == 0:
            continue  # slide not used for adaptation

        p = adatas_dir / f"{sid}.h5ad"
        if not p.exists():
            raise FileNotFoundError(f"Missing h5ad for slide_id='{sid}': {p}")

        adata = ad.read_h5ad(p)
        ensure_gene_id_index(adata, gene_id_column)

        cell_str = get_cell_id_strings(adata)
        cell_int = int_cell_id_vec(cell_str)
        mask = mask_from_sorted_ids(cell_int, keep_ids)
        n_keep = int(mask.sum())
        if n_keep == 0:
            continue

        # Optional downsampling to cap memory / runtime
        if max_train_cells_per_slide is not None and n_keep > max_train_cells_per_slide:
            keep_idx = np.flatnonzero(mask)
            choose = rng.choice(keep_idx, size=max_train_cells_per_slide, replace=False)
            choose = np.asarray(choose, dtype=np.intp)
            choose.sort()
            adata_sub = subset_minimal_adata_for_scconcept(adata, choose)
        else:
            adata_sub = subset_minimal_adata_for_scconcept(adata, mask)

        # Helpful metadata
        adata_sub.uns["slide_id"] = sid
        train_adatas.append(adata_sub)

        # free original
        del adata

    return train_adatas


# -----------------------------
# Streaming embedding extraction to HDF5 (no giant RAM tensors)
# -----------------------------
@dataclass
class ExtractParams:
    gene_id_column: Optional[str]
    batch_size: int
    max_tokens: Optional[int]
    gene_sampling_strategy: Optional[str]
    num_workers: int
    store_mean_emb: bool = True
    out_dtype: str = "float16"  # float16 recommended for storage


def stream_extract_to_h5(
    concept: scConcept,
    slide_id: str,
    adata_path: Path,
    out_h5_path: Path,
    params: ExtractParams,
    compression_level: int = 4,
    chunk_rows: int = 4096,
    write_cell_id_str: bool = True,
    store_mean_emb: bool = True,
) -> Dict[str, int]:
    """
    Extract embeddings slide-by-slide, writing batches directly into HDF5 datasets.
    Returns small stats dict.
    """
    adata = ad.read_h5ad(adata_path)
    ensure_gene_id_index(adata, params.gene_id_column)

    cell_str = get_cell_id_strings(adata)
    cell_int = int_cell_id_vec(cell_str)
    n_cells = int(cell_int.shape[0])

    out_h5_path.parent.mkdir(parents=True, exist_ok=True)
    if out_h5_path.exists():
        out_h5_path.unlink()  # overwrite to avoid mixing partial outputs

    # Resolve defaults from config if not provided
    max_tokens = params.max_tokens
    if max_tokens is None:
        max_tokens = int(concept.cfg.datamodule.dataset.train.max_tokens)
    gene_sampling_strategy = params.gene_sampling_strategy
    if gene_sampling_strategy is None:
        gene_sampling_strategy = str(concept.cfg.datamodule.gene_sampling_strategy)

    # Build tokenized dataset + collate (mirrors scConcept.api.extract_embeddings but stream-writes)
    collection = InMemoryCollection([adata], var_column=params.gene_id_column)
    dataset = TokenizedDataset(
        collection,
        concept.tokenizer,
        normalization=concept.cfg.datamodule.normalization,
    )
    collate_fn = BaseCollate(
        concept.tokenizer.PAD_TOKEN,
        max_tokens=max_tokens,
        gene_sampling_strategy=gene_sampling_strategy,
    )

    device = next(concept.model.parameters()).device
    pin_memory = device.type == "cuda"

    loader = DataLoader(
        dataset,
        batch_size=params.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=params.num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        persistent_workers=(params.num_workers > 0),
    )

    # Write outputs
    out_dtype = np.float16 if params.out_dtype == "float16" else np.float32

    with h5py.File(out_h5_path, "w") as f:
        f.attrs["slide_id"] = slide_id
        f.attrs["model_name"] = getattr(concept, "model_name", "") or ""
        f.attrs["gene_id_column"] = params.gene_id_column or ""
        f.attrs["max_tokens"] = int(max_tokens)
        f.attrs["gene_sampling_strategy"] = str(gene_sampling_strategy)
        f.attrs["embedding_dtype"] = "float16" if out_dtype == np.float16 else "float32"

        f.create_dataset(
            "cell_id_int",
            data=cell_int.astype(np.int64, copy=False),
            compression="gzip",
            compression_opts=1,
            shuffle=True,
        )
        if write_cell_id_str:
            # store as fixed-width bytes (fast + compact)
            f.create_dataset(
                "cell_id",
                data=np.asarray(cell_str, dtype="S10"),
                compression="gzip",
                compression_opts=1,
                shuffle=True,
            )

        cls_ds = None
        mean_ds = None  # only created if store_mean_emb
        offset = 0

        concept.model.eval()
        with torch.no_grad():
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                for batch_idx, batch in enumerate(tqdm(loader, desc=f"[{slide_id}] embeddings", unit="batch")):
                    batch = {
                        k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
                        for k, v in batch.items()
                    }
                    out = concept.model.predict_step(batch, batch_idx)
                    cls_ = out["cls_cell_emb"].detach().cpu().numpy().astype(out_dtype, copy=False)
                    if store_mean_emb:
                        mean = out["mean_cell_emb"].detach().cpu().numpy().astype(out_dtype, copy=False)
                    else:
                        mean = None

                    if cls_ds is None:
                        d = int(cls_.shape[1])
                        chunks = (min(chunk_rows, n_cells), d)
                        cls_ds = f.create_dataset(
                            "cls_emb",
                            shape=(n_cells, d),
                            dtype=out_dtype,
                            chunks=chunks,
                            compression="gzip",
                            compression_opts=compression_level,
                            shuffle=True,
                        )
                        if store_mean_emb:
                            mean_ds = f.create_dataset(
                                "mean_emb",
                                shape=(n_cells, d),
                                dtype=out_dtype,
                                chunks=chunks,
                                compression="gzip",
                                compression_opts=compression_level,
                                shuffle=True,
                            )
                        f.attrs["embedding_dim"] = d

                    bs = cls_.shape[0]
                    cls_ds[offset : offset + bs] = cls_
                    if store_mean_emb:
                        mean_ds[offset : offset + bs] = mean
                    offset += bs

        if offset != n_cells:
            raise RuntimeError(f"[{slide_id}] wrote {offset} rows but expected {n_cells}")

    # free memory
    del adata
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {"n_cells": n_cells, "max_tokens": int(max_tokens)}


# -----------------------------
# Main
# -----------------------------

def main(args):

    # Paths
    adatas_dir = get_full_path('train', 'training_datasets', args.dataset_id, args.training_dataset_id, 'adatas')
    cell_ids_sets_path = adatas_dir.parent / "cell_ids_sets.npz"
    slide_ids_txt = adatas_dir.parent / "slide_ids.txt"
    out_dir = get_full_path('train', 'trainings', 'trainings_scConcept', args.training_id_scConcept)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Model
    model_name = "Corpus-30M"
    cache_dir = get_full_path('train', 'pretrained_foundation_models', 'scConcept_Corpus-30M', as_str=True)

    _is_cuda = torch.cuda.is_available()
    if not _is_cuda:
        print("[WARNING] CUDA not available — running on CPU (for debugging only).")
    
    # Get list of slide IDs from slide_ids.txt (one per line)
    if not slide_ids_txt.exists():
        raise FileNotFoundError(f"Missing slide_ids.txt at: {slide_ids_txt}")
    with slide_ids_txt.open() as f:
        slide_ids = [line.strip() for line in f if line.strip()]
    if len(slide_ids) == 0:
        raise ValueError(f"No slide IDs found in {slide_ids_txt}")

    # Load split dicts
    npz = np.load(cell_ids_sets_path, allow_pickle=True)
    train_only_by_slide = npz["train_only_by_slide"].item()

    # Initialize and load pretrained scConcept
    print(f"Loading scConcept model '{model_name}' with cache_dir='{cache_dir}' ...")
    concept = scConcept(cache_dir=cache_dir)
    model_dir = Path(cache_dir) / model_name
    concept.load_config_and_model(
        config=str(model_dir / "config.yaml"),
        model_path=str(model_dir / "model.ckpt"),
        gene_mapping_path=str(model_dir / "pc_gene_token_mapping.pkl"),
        panels_dir=str(model_dir / "panels"),
    )

    # Make sure training path uses dataset grouping for AnnData lists (required by scConcept's AnnDataModule)
    assert concept.cfg.datamodule.dataloader.train.within_group_sampling == "dataset" # correspond to no mixing slides in the same batch

    # Preflight: vocabulary coverage of each slide (v1 single-species tokenizer: species=None)
    coverage_by_slide = preflight_vocab_coverage(
        slide_ids=slide_ids,
        adatas_dir=adatas_dir,
        tokenizer=concept.tokenizer,
        gene_id_column=args.gene_id_column,
        min_coverage_frac=float(args.min_vocab_coverage_frac),
        min_genes_in_vocab=int(args.min_genes_in_vocab),
        species=None,
    )

    # Print all information that is inside the config
    print("\n[Adaptation] INITIAL scConcept config:")
    from omegaconf import OmegaConf
    print(OmegaConf.to_yaml(concept.cfg))

    # -----------------
    # 1) Light adaptation (using slide IDs and cell IDs in train set only)
    # -----------------
    t0 = time.time()
    train_adatas = load_train_adatas(
        slide_ids=slide_ids,
        adatas_dir=adatas_dir,
        train_only_by_slide=train_only_by_slide,
        gene_id_column=args.gene_id_column,
        max_train_cells_per_slide=args.max_train_cells_per_slide,
        seed=args.seed,
    )
    if len(train_adatas) == 0:
        raise RuntimeError("No training cells found (train_only_by_slide arrays were empty for all provided slides).")

    # Panel sizes for adaptation: panel_size_min bounded by the smallest training panel
    min_panel_genes = int(min(a.shape[1] for a in train_adatas))
    # Bound min/max so that large panels are downsampled, but small panels still work
    adapt_token_budget = int(args.adapt_max_tokens)
    panel_size_max = adapt_token_budget
    panel_size_min = int(min(args.adapt_panel_size_min, panel_size_max, min_panel_genes))
    if panel_size_min < 1:
        panel_size_min = 1

    patch_scconcept_cfg_for_adaptation(
        concept,
        token_budget=adapt_token_budget,
        gene_sampling_strategy=args.adapt_gene_sampling_strategy,
        loss_switch_step=int(args.adapt_loss_switch_step),
        panel_selection=str(args.adapt_panel_selection),
        panel_size_min=panel_size_min,
        panel_size_max=panel_size_max,
    )

    print("\n[Adaptation] Patched scConcept config:")
    print(OmegaConf.to_yaml(concept.cfg))


    print(f"[Adaptation] Loaded {len(train_adatas)} train AnnData objects.")
    print(f"[Adaptation] Starting scConcept.train(max_steps={args.adapt_steps}, batch_size={args.adapt_batch_size}) ...")

    concept.train(train_adatas, max_steps=args.adapt_steps, batch_size=args.adapt_batch_size)

    # Optionally save adapted checkpoint (trainer is attached to LightningModule after fit)
    if args.save_adapted_ckpt:
        adapted_ckpt_name = f"scConcept_{args.training_id_scConcept}.ckpt"
        ckpt_path = out_dir / adapted_ckpt_name
        trainer = getattr(concept.model, "trainer", None)
        if trainer is None:
            print("[Adaptation] WARNING: trainer not found on concept.model => cannot save checkpoint.")
        else:
            trainer.save_checkpoint(str(ckpt_path))
            print(f"[Adaptation] Saved adapted checkpoint: {ckpt_path}")

    extract_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    concept.model = concept.model.to(extract_device)
    concept.model.eval()
    concept.device = next(concept.model.parameters()).device

    # Release adaptation adatas (important for memory before extraction)
    del train_adatas
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"[Adaptation] Done in {(time.time() - t0)/60:.2f} min")

    # -----------------
    # 2) Extract embeddings for every cell of every slide
    # -----------------
    emb_dir = out_dir / "embeddings"
    emb_dir.mkdir(parents=True, exist_ok=True)

    # Extraction uses the adaptation values unless overridden in args
    if args.extract_max_tokens is None:
        args.extract_max_tokens = args.adapt_max_tokens
    if args.extract_gene_sampling_strategy is None:
        args.extract_gene_sampling_strategy = args.adapt_gene_sampling_strategy

    extract_params = ExtractParams(
        gene_id_column=args.gene_id_column,
        batch_size=args.extract_batch_size,
        max_tokens=args.extract_max_tokens,
        gene_sampling_strategy=args.extract_gene_sampling_strategy,
        num_workers=args.extract_num_workers,
        out_dtype="float16",
        store_mean_emb=args.store_mean_emb,
    )

    manifest = {
        "model_name": model_name,
        "cache_dir": str(Path(cache_dir).resolve()),
        "gene_id_column": args.gene_id_column,
        "adapt_steps": args.adapt_steps,
        "adapt_batch_size": args.adapt_batch_size,
        "adapt_max_tokens": int(args.adapt_max_tokens),
        "adapt_gene_sampling_strategy": args.adapt_gene_sampling_strategy,
        "adapt_panel_selection": args.adapt_panel_selection,
        "adapt_panel_size_min": int(panel_size_min),
        "adapt_panel_size_max": int(panel_size_max),
        "adapt_loss_switch_step": int(args.adapt_loss_switch_step),
        "extract_max_tokens": int(args.extract_max_tokens),
        "extract_gene_sampling_strategy": args.extract_gene_sampling_strategy,
        "min_vocab_coverage_frac": float(args.min_vocab_coverage_frac),
        "min_genes_in_vocab": int(args.min_genes_in_vocab),
        "slides": {},
    }

    for sid in slide_ids:
        adata_path = adatas_dir / f"{sid}.h5ad"
        if not adata_path.exists():
            raise FileNotFoundError(f"Missing h5ad for slide_id='{sid}': {adata_path}")

        out_h5 = emb_dir / f"{sid}.h5"
        stats = stream_extract_to_h5(
            concept=concept,
            slide_id=sid,
            adata_path=adata_path,
            out_h5_path=out_h5,
            params=extract_params,
            compression_level=4,
            chunk_rows=4096,
            write_cell_id_str=(not args.no_cell_id_str),
            store_mean_emb=extract_params.store_mean_emb,
        )

        manifest["slides"][sid] = {
            "h5_path": str(out_h5.resolve()),
            **stats,
            **coverage_by_slide[sid],
        }

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\nAll done. Manifest written to: {manifest_path}")
    print(f"Embeddings directory: {emb_dir}")




if __name__ == "__main__":

    ap = argparse.ArgumentParser(description="Adapt scConcept on Xenium training cells, then extract cell embeddings per slide.")

    # Dataset_id and training_dataset_id, and training_id for scConcept

    ap.add_argument("--dataset_id", type=str,
                    default="ds_ct1_20x_224_o112",
                    help="Dataset_id, e.g. ds_ct1_20x_224_o112")
    
    ap.add_argument("--training_dataset_id", type=str,
                    default="ds_ct1_20x_224_o112_1",
                    help="Training_dataset_id, e.g. ds_ct1_20x_224_o112_1")
    
    ap.add_argument("--training_id_scConcept", type=str,
                    default="training_1",
                    help="Training_id for scConcept, e.g. training_1")

    # Model
    ap.add_argument("--gene_id_column", type=str, default="gene_ids", help="Column in adata.var with Ensembl IDs (usually 'gene_ids').")

    # Vocabulary-coverage check
    ap.add_argument("--min_vocab_coverage_frac", type=float, default=0.3,
                    help="Minimum fraction of the gene panel of a slide in the scConcept corpus vocabulary, checked "
                         "before adaptation (0 disables the check).")
    ap.add_argument("--min_genes_in_vocab", type=int, default=1,
                    help="Minimum number of panel genes of a slide in the corpus vocabulary (0 disables the check).")

    # Adaptation
    ap.add_argument("--adapt_steps", type=int, default=50_000, help="Max steps for light adaptation, recommended is 50,000.")  # scConcept authors' guidance: 50,000 steps for datasets of fewer than 100K cells
    ap.add_argument("--adapt_batch_size", type=int, default=128, help="Batch size for adaptation, recommended is 128.")
    ap.add_argument("--max_train_cells_per_slide", type=int, default=100_000, help="Optional cap to reduce memory/time during adaptation, recommended is 100,000.") # also balances slides
    ap.add_argument("--save_adapted_ckpt", action="store_true", help="Save the adapted Lightning checkpoint to out_dir.")
    ap.add_argument("--adapt_max_tokens", type=int, default=512, help="Token budget during adaptation (and default for extraction).")
    ap.add_argument("--adapt_panel_selection", type=str, default="random", help="Panel sampling mode during adaptation. Recommended: 'random' for mixed Xenium panels.")
    ap.add_argument("--adapt_panel_size_min", type=int, default=256, help="Min random panel size during adaptation (bounded by the smallest slide panel).")
    ap.add_argument("--adapt_gene_sampling_strategy", type=str, default="top-nonzero", help="Sampling strategy for selecting genes within max_tokens. Keep consistent for adapt+extract.")
    ap.add_argument("--adapt_loss_switch_step", type=int, default=0, help="Step at which the stronger loss (same-view negatives) starts during adaptation.")

    # Extraction
    ap.add_argument("--extract_batch_size", type=int, default=256, help="Batch size for embedding extraction.")
    ap.add_argument("--extract_num_workers", type=int, default=10, help="DataLoader workers for extraction.")
    ap.add_argument("--extract_max_tokens", type=int, default=None, help="Override max_tokens (else uses adaptation value).")
    ap.add_argument("--extract_gene_sampling_strategy", type=str, default=None, help="Override gene_sampling_strategy (else uses adaptation value).")

    ap.add_argument("--no_cell_id_str", action="store_true", help="Do not store cell_id strings (i.e. store only int).")
    ap.add_argument("--store_mean_emb", action="store_true", help="Store mean_emb in output HDF5 (default: off).")
    ap.add_argument("--seed", type=int, default=0)

    args = ap.parse_args()
    main(args)