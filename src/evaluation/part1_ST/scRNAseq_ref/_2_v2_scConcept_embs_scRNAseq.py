#!/usr/bin/env python3
"""
Compute the raw scConcept embeddings (no CLIP projection) of each scRNA-seq reference h5ad with the adapted scConcept
model of a sc_concept_training_id (<ROOT_HistoST> / train / trainings / trainings_scConcept / <sc_concept_training_id> /).
Version for the corpus40M-model30M model (scConcept v0.2.x API).

Saved in <ROOT_HistoST> / eval / eval_datasets / scRNAseq_references / scConcept_embs / <sc_concept_training_id> /
    - <ref_stem>.h5: cls_emb of each cell (512-d, L2-normalized), with row_idx, obs_name and the run parameters as attributes
    - manifest.json: run parameters and statistics of each reference
"""

import sys
import os
import json
import time
import argparse
import warnings
from pathlib import Path
from dataclasses import dataclass
from contextlib import nullcontext
from typing import Dict, List, Optional, Sequence, Tuple, Any
import numpy as np
import pandas as pd
import torch
import anndata as ad
import h5py
from tqdm import tqdm
from torch.utils.data import DataLoader
import hashlib
from scipy import sparse

# Add project root to sys.path for imports
project_root = Path(__file__).resolve().parents[4]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.constants import get_full_path


# -----------------------------------------------------------------------------
# flash_attn CPU stub for non-CUDA debug environments
# -----------------------------------------------------------------------------
try:
    import flash_attn

except ModuleNotFoundError:

    # flash_attn is expected to be installed on the GPU cluster; a missing module there is an error.
    from src.utils.constants import on_jeanzay
    if on_jeanzay():
        raise ModuleNotFoundError("flash_attn not found. This should not happen on Jean Zay.")

    import types
    import torch.nn as nn

    _fa = types.ModuleType("flash_attn")
    _mha = types.ModuleType("flash_attn.modules.mha")

    class _MHA(nn.Module):
        def __init__(self, embed_dim, num_heads, bias=True, **kwargs):
            super().__init__()
            self.embed_dim = embed_dim
            self.num_heads = num_heads
            self.head_dim = embed_dim // num_heads
            self.Wqkv = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
            self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        def forward(self, x, *args, **kwargs):
            cu_seqlens = kwargs.get("cu_seqlens", None)

            # flash_attn MHA receives packed (jagged) input when cu_seqlens is given:
            #   x shape: (total_tokens, C)   — 2-D, all sequences concatenated
            # Without cu_seqlens it behaves like a standard MHA:
            #   x shape: (B, T, C)           — 3-D
            if cu_seqlens is not None:
                # Treat the entire packed sequence as one "batch row" to reuse the
                # standard multi-head attention path below.
                total_tokens, C = x.shape
                x = x.unsqueeze(0)      # (1, total_tokens, C)
                B, T = 1, total_tokens
                packed = True
            else:
                B, T, C = x.shape
                packed = False

            qkv = self.Wqkv(x)
            q, k, v = qkv.chunk(3, dim=-1)

            def split_heads(t):
                return t.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

            q, k, v = split_heads(q), split_heads(k), split_heads(v)
            attn_out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
            attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, C)

            if packed:
                attn_out = attn_out.squeeze(0)   # back to (total_tokens, C)

            return self.out_proj(attn_out)

    _mha.MHA = _MHA
    _fa.modules = types.ModuleType("flash_attn.modules")
    _fa.modules.mha = _mha
    sys.modules["flash_attn"] = _fa
    sys.modules["flash_attn.modules"] = _fa.modules
    sys.modules["flash_attn.modules.mha"] = _mha


# -----------------------------------------------------------------------------
# Project imports
# -----------------------------------------------------------------------------

from concept import scConcept
from concept.dataset import MultiSpeciesTokenizedDataset
from lamin_dataloader import BaseCollate, InMemoryCollection



# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r") as f:
        return json.load(f)



def _autocast_context(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()



def _tensor_to_numpy_for_storage(t: torch.Tensor, np_dtype: np.dtype) -> np.ndarray:
    target_torch_dtype = torch.float16 if np_dtype == np.float16 else torch.float32
    return t.to(device="cpu", dtype=target_torch_dtype, non_blocking=True).numpy()



def _load_existing_manifest(manifest_path: Path) -> Dict[str, Any]:
    """
    Load an existing manifest if present and valid.
    Returns an empty dict if missing/invalid.
    """
    if not manifest_path.exists():
        return {}

    try:
        with manifest_path.open("r") as f:
            manifest = json.load(f)
    except Exception as e:
        print(f"[WARNING] Could not read existing manifest at {manifest_path}: {e}")
        return {}

    if not isinstance(manifest, dict):
        print(f"[WARNING] Existing manifest is not a dict: {manifest_path}")
        return {}

    files = manifest.get("files", {})
    if not isinstance(files, dict):
        print(f"[WARNING] Existing manifest['files'] is not a dict: {manifest_path}")
        manifest["files"] = {}

    return manifest



def _load_scconcept_from_training(
    sc_concept_training_id: str,
    device: torch.device,
):
    # Training directory
    training_dir = get_full_path("train", "trainings", "trainings_scConcept", sc_concept_training_id)
    if not training_dir.exists():
        raise FileNotFoundError(f"Training directory not found: {training_dir}")
    
    # Manifest
    manifest_path = training_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Training manifest.json not found in {training_dir}: {manifest_path}")
    training_manifest = _read_json(manifest_path)

    # Model and cache directories
    cache_dir = Path(training_manifest["cache_dir"])
    if not cache_dir.exists():
        # Warn and use fallbach with get_full_path for pretrained_foundation_models
        print(f"[WARNING] Cache directory from manifest not found: {cache_dir}")
        cache_dir_stem = cache_dir.name
        cache_dir = get_full_path("train", "pretrained_foundation_models", cache_dir_stem)

    # After the fallback, the cache directory must exist.
    if not cache_dir.exists():
        raise FileNotFoundError(f"scConcept cache_dir not found: {cache_dir}")

    ckpt_path = training_dir / f"scConcept_{sc_concept_training_id}.ckpt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Adapted scConcept checkpoint not found: {ckpt_path}.")

    model_name = training_manifest["model_name"]
    species = training_manifest.get("species", "hsapiens")
    model_dir = cache_dir / model_name
    config_path = model_dir / "config.yaml"
    gene_mappings_dir = model_dir / "gene_mappings"   # directory of {species}.csv
    panels_dir = model_dir / "panels"

    for p in [config_path, gene_mappings_dir, panels_dir]:
        if not p.exists():
            raise FileNotFoundError(f"Required pretrained scConcept asset not found: {p}")

    # Load pretrained_vocabulary if it was used during adaptation (recorded in the training manifest)
    pretrained_vocab_dir = model_dir / "pretrained_vocabulary"
    pretrained_vocab_present = pretrained_vocab_dir.is_dir() and any(pretrained_vocab_dir.glob("*.csv"))
    pretrained_vocab_expected = bool(training_manifest.get("pretrained_vocabulary_used", False))
    if pretrained_vocab_expected and not pretrained_vocab_present:
        raise FileNotFoundError(
            f"Training manifest says pretrained_vocabulary_used=True but no pretrained_vocabulary/*.csv "
            f"found under {model_dir}. Cannot reproduce the adapted model architecture."
        )

    # Load scConcept from adapted checkpoint (v0.2.x on-disk layout, local files only)
    print(f"Loading scConcept from adapted checkpoint: {ckpt_path}")
    print(f"Using pretrained assets from: {model_dir} (species='{species}')")

    concept = scConcept(cache_dir=str(cache_dir))
    concept.load_config_and_model(
        config=str(config_path),
        model_path=str(ckpt_path),
        gene_mappings_path=str(gene_mappings_dir),
        panels_dir=str(panels_dir),
        pretrained_vocabulary_path=str(pretrained_vocab_dir) if pretrained_vocab_expected else None,
    )
    if species not in list(concept.tokenizer.species):
        raise ValueError(f"species='{species}' not in model species {list(concept.tokenizer.species)}")
    concept.model = concept.model.to(device)
    concept.model.eval()
    for p in concept.model.parameters():
        p.requires_grad_(False)

    return concept, ckpt_path, model_name, training_manifest, species



def _resolve_references(
    ref_json: Optional[Path],
    h5ad_files: Optional[List[str]],
    tissues: Optional[List[str]],
) -> List[Dict[str, Any]]:
    """
    Returns a list of unique reference files:
      [
        {
          "reference_name": "tonsil",
          "path": Path("/.../tonsil.h5ad"),
          "tissue_aliases": ["LymphNode", "Tonsil"],
        },
        ...
      ]
    """
    def _attach_unique_output_stems(refs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Ensure output filenames / manifest keys are unique even when different
        input files share the same stem.
        """
        stem_counts: Dict[str, int] = {}
        for ref in refs:
            stem = str(ref["reference_name"])
            stem_counts[stem] = stem_counts.get(stem, 0) + 1

        for ref in refs:
            stem = str(ref["reference_name"])
            if stem_counts[stem] == 1:
                ref["output_stem"] = stem
            else:
                resolved = str(Path(ref["path"]).resolve())
                digest = hashlib.sha1(resolved.encode("utf-8")).hexdigest()[:10]
                ref["output_stem"] = f"{stem}_{digest}"
        return refs

    if h5ad_files:
        # Deduplicate explicit files by resolved path
        by_path: Dict[Path, Dict[str, Any]] = {}
        for x in h5ad_files:
            p = Path(x).resolve()
            if not p.exists():
                raise FileNotFoundError(f"Reference h5ad not found: {p}")
            by_path[p] = {
                "reference_name": p.stem,
                "path": p,
                "tissue_aliases": [p.stem],
            }

        refs = [by_path[p] for p in sorted(by_path, key=lambda p: p.name.lower())]
        return _attach_unique_output_stems(refs)

    if ref_json is None:
        raise ValueError("Provide either --ref_json or --h5ad_files.")

    if not ref_json.exists():
        raise FileNotFoundError(f"ref_json not found: {ref_json}")

    mapping = _read_json(ref_json)
    if not isinstance(mapping, dict) or len(mapping) == 0:
        raise ValueError(f"ref_json must contain a non-empty dict, got: {type(mapping)}")

    tissues_set = set(tissues) if tissues else None
    by_path: Dict[Path, List[str]] = {}

    for tissue_name, fname in mapping.items():
        if tissues_set is not None and tissue_name not in tissues_set:
            continue
        p = Path(fname)
        scRNAseq_ref_dir = get_full_path("eval", "eval_datasets", "scRNAseq_references", "adatas")
        if not p.is_absolute():
            p = (scRNAseq_ref_dir / p).resolve()
        if not p.exists():
            raise FileNotFoundError(f"Reference file from JSON does not exist: {p}")
        by_path.setdefault(p, []).append(str(tissue_name))

    refs = []
    for p, aliases in sorted(by_path.items(), key=lambda x: x[0].name.lower()):
        refs.append({
            "reference_name": p.stem,
            "path": p,
            "tissue_aliases": sorted(aliases),
        })
    return _attach_unique_output_stems(refs)



def _decode_h5_attr(x: Any) -> str:
    """
    h5py attributes can be stored as bytes depending on writer/version.
    Normalize them to plain strings.
    """
    if x is None:
        return ""
    if isinstance(x, bytes):
        return x.decode("utf-8")
    return str(x)


def _h5_node_shape(node: h5py.Group | h5py.Dataset) -> Tuple[int, int]:
    """
    Return the 2D shape of an H5AD X node.
    Dense X is an h5py.Dataset.
    Sparse X is an h5py.Group with shape stored as an attribute in modern H5AD.
    """
    if isinstance(node, h5py.Dataset):
        return tuple(int(x) for x in node.shape)

    if "shape" in node.attrs:
        return tuple(int(x) for x in node.attrs["shape"])

    # Older/non-standard files may store shape as a dataset
    if "shape" in node:
        return tuple(int(x) for x in node["shape"][()])

    raise ValueError("Could not determine shape of H5AD X node.")


def _read_h5ad_X_rows(
    h5ad_path: Path,
    start: int,
    stop: int,
) -> Any:
    """Read X[start:stop, :] directly from an .h5ad file (dense, CSR or CSC X), without backed AnnData views."""
    with h5py.File(h5ad_path, "r") as f:
        if "X" not in f:
            raise KeyError(f"{h5ad_path}: no root-level 'X' found.")

        x_node = f["X"]
        n_obs, n_vars = _h5_node_shape(x_node)

        if start < 0 or stop > n_obs or start > stop:
            raise IndexError(
                f"Invalid row slice [{start}:{stop}) for X with n_obs={n_obs}."
            )

        # Dense X
        if isinstance(x_node, h5py.Dataset):
            return np.asarray(x_node[start:stop, :])

        # Sparse X
        if not isinstance(x_node, h5py.Group):
            raise TypeError(f"Unsupported H5AD X node type: {type(x_node)}")

        required = {"data", "indices", "indptr"}
        missing = required.difference(x_node.keys())
        if missing:
            raise KeyError(f"Unsupported sparse X group in {h5ad_path}; missing {sorted(missing)}.")

        encoding_type = _decode_h5_attr(x_node.attrs.get("encoding-type"))

        # If encoding metadata is absent, infer from indptr length
        if not encoding_type:
            indptr_len = int(x_node["indptr"].shape[0])
            if indptr_len == n_obs + 1:
                encoding_type = "csr_matrix"
            elif indptr_len == n_vars + 1:
                encoding_type = "csc_matrix"

        if encoding_type in {"csr_matrix", "csr_array"}:
            # Fast row slicing for CSR:
            # each row's non-zeros are contiguous in data/indices
            indptr_abs = np.asarray(x_node["indptr"][start:stop + 1])
            data_start = int(indptr_abs[0])
            data_stop = int(indptr_abs[-1])

            indptr = indptr_abs - data_start
            indices = np.asarray(x_node["indices"][data_start:data_stop])
            data = np.asarray(x_node["data"][data_start:data_stop])

            return sparse.csr_matrix(
                (data, indices, indptr),
                shape=(stop - start, n_vars),
            )

        if encoding_type in {"csc_matrix", "csc_array"}:
            # Correct but slower: row slicing is not contiguous for CSC
            old_indptr = np.asarray(x_node["indptr"][:])
            new_indptr = np.zeros(n_vars + 1, dtype=old_indptr.dtype)
            data_parts = []
            row_parts = []
            nnz = 0

            for j in range(n_vars):
                col_start = int(old_indptr[j])
                col_stop = int(old_indptr[j + 1])

                if col_start == col_stop:
                    new_indptr[j + 1] = nnz
                    continue

                rows = np.asarray(x_node["indices"][col_start:col_stop])
                keep = (rows >= start) & (rows < stop)

                if np.any(keep):
                    row_parts.append((rows[keep] - start).astype(rows.dtype, copy=False))
                    data_parts.append(np.asarray(x_node["data"][col_start:col_stop])[keep])
                    nnz += int(np.sum(keep))

                new_indptr[j + 1] = nnz

            if nnz == 0:
                rows = np.empty(0, dtype=x_node["indices"].dtype)
                data = np.empty(0, dtype=x_node["data"].dtype)
            else:
                rows = np.concatenate(row_parts)
                data = np.concatenate(data_parts)

            return sparse.csc_matrix(
                (data, rows, new_indptr),
                shape=(stop - start, n_vars),
            ).tocsr()

        raise ValueError(
            f"Unsupported H5AD X encoding_type={encoding_type!r} in {h5ad_path}. "
            "Expected dense, csr_matrix/csr_array, or csc_matrix/csc_array."
        )



def _make_minimal_chunk_adata(
    backed_adata: ad.AnnData,
    adata_path: Path,
    start: int,
    stop: int,
) -> ad.AnnData:
    """Materialize only X / obs / var of one chunk of a backed AnnData (X read with _read_h5ad_X_rows)."""
    X = _read_h5ad_X_rows(
        h5ad_path=adata_path,
        start=start,
        stop=stop,
    )
    obs = backed_adata.obs.iloc[start:stop].copy()
    var = backed_adata.var.copy()

    adata_chunk = ad.AnnData(X=X, obs=obs, var=var)
    adata_chunk.obs_names = backed_adata.obs_names[start:stop].copy()
    adata_chunk.var_names = backed_adata.var_names.copy()
    return adata_chunk



@dataclass
class ExtractConfig:
    gene_id_column: str
    batch_size: int
    num_workers: int
    cells_per_chunk: int
    max_tokens: Optional[int]
    gene_sampling_strategy: Optional[str]
    out_dtype: str
    species: str = "hsapiens"
    compression_level: int = 4
    chunk_rows: int = 4096



def _build_loader_for_chunk(
    concept: scConcept,
    adata_chunk: ad.AnnData,
    cfg: ExtractConfig,
    device: torch.device,
) -> DataLoader:
    max_tokens = cfg.max_tokens
    if max_tokens is None:
        max_tokens = int(concept.cfg.datamodule.dataset.train.max_tokens)

    gene_sampling_strategy = cfg.gene_sampling_strategy
    if gene_sampling_strategy is None:
        gene_sampling_strategy = str(concept.cfg.datamodule.gene_sampling_strategy)

    collection = InMemoryCollection([adata_chunk], var_column=cfg.gene_id_column)
    dataset = MultiSpeciesTokenizedDataset(
        metadata={"species": [cfg.species]},
        collection=collection,
        tokenizer=concept.tokenizer,
        normalization=concept.cfg.datamodule.normalization,
    )
    collate_fn = BaseCollate(
        concept.tokenizer.PAD_TOKEN,
        max_tokens=int(max_tokens),
        gene_sampling_strategy=str(gene_sampling_strategy),
    )

    pin_memory = device.type == "cuda"

    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        persistent_workers=(cfg.num_workers > 0),
    )
    return loader



# -----------------------------------------------------------------------------
# Per-reference gene-panel vs corpus-vocabulary coverage
# -----------------------------------------------------------------------------
def compute_vocab_coverage(gene_ids: np.ndarray, tokenizer, species: Optional[str] = None) -> Dict[str, float]:
    """
    Count how many of a reference's panel genes exist in the scConcept corpus vocabulary (the others are masked out).
    species: for the v2 MultiSpeciesTokenizer; None for a single-species tokenizer.
    """
    tokens = tokenizer.encode(gene_ids) if species is None else tokenizer.encode(gene_ids, species)
    tokens = np.asarray(tokens)
    n_panel = int(len(gene_ids))
    n_in_vocab = int(np.count_nonzero(tokens != tokenizer.NOT_FOUND))
    frac = (n_in_vocab / n_panel) if n_panel > 0 else 0.0
    return {"n_genes_panel": n_panel, "n_genes_in_vocab": n_in_vocab, "vocab_coverage_frac": frac}


def preflight_reference_vocab_coverage(
    references: List[Dict[str, Any]],
    tokenizer,
    gene_id_column: str,
    min_coverage_frac: float,
    min_genes_in_vocab: int,
    species: Optional[str] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Compute the vocabulary coverage of every reference to (re)process and raise if a panel barely overlaps the
    corpus vocabulary. Returns {output_stem: {n_genes_panel, n_genes_in_vocab, vocab_coverage_frac}}.
    """
    if not references:
        print("[Preflight] No references to (re)process — skipping vocabulary-coverage check.")
        return {}

    coverage_by_ref: Dict[str, Dict[str, float]] = {}
    offenders: List[Dict[str, Any]] = []
    print("[Preflight] Checking per-reference gene-panel vs corpus-vocabulary coverage ...")
    for ref in references:
        p = Path(ref["path"])
        backed = ad.read_h5ad(p, backed="r")
        try:
            if gene_id_column not in backed.var.columns:
                raise KeyError(
                    f"gene_id_column='{gene_id_column}' not found in adata.var for reference "
                    f"'{ref['reference_name']}' ({p}). Available columns: {list(backed.var.columns)}"
                )
            gene_ids = backed.var[gene_id_column].astype(str).to_numpy()
        finally:
            if hasattr(backed, "file") and backed.file is not None:
                backed.file.close()

        cov = compute_vocab_coverage(gene_ids, tokenizer, species=species)
        coverage_by_ref[ref["output_stem"]] = cov
        ok = cov["vocab_coverage_frac"] >= min_coverage_frac and cov["n_genes_in_vocab"] >= min_genes_in_vocab
        print(
            f"[Preflight] {ref['reference_name']}: {cov['n_genes_in_vocab']}/{cov['n_genes_panel']} genes in vocab "
            f"({100.0 * cov['vocab_coverage_frac']:.1f}%)" + ("" if ok else "  <-- below threshold")
        )
        if not ok:
            offenders.append(ref)

    if offenders:
        lines = [
            f"  - {r['reference_name']}: {coverage_by_ref[r['output_stem']]['n_genes_in_vocab']}/"
            f"{coverage_by_ref[r['output_stem']]['n_genes_panel']} genes in vocab "
            f"({100.0 * coverage_by_ref[r['output_stem']]['vocab_coverage_frac']:.1f}%)"
            for r in offenders
        ]
        raise RuntimeError(
            f"scConcept vocabulary coverage too low for {len(offenders)}/{len(references)} reference(s) "
            f"(require coverage >= {100.0 * min_coverage_frac:.1f}% and >= {min_genes_in_vocab} genes in vocab):\n"
            + "\n".join(lines)
            + "\nVerify that gene_id_column maps into the corpus ID space (e.g. unversioned Ensembl IDs), "
            "or lower --min_vocab_coverage_frac / --min_genes_in_vocab."
        )

    fracs = [c["vocab_coverage_frac"] for c in coverage_by_ref.values()]
    print(
        f"[Preflight] Vocabulary coverage OK for all {len(coverage_by_ref)} reference(s) "
        f"(min {100.0 * min(fracs):.1f}%, mean {100.0 * (sum(fracs) / len(fracs)):.1f}%)."
    )
    return coverage_by_ref



def stream_reference_to_h5(
    concept: scConcept,
    reference_name: str,
    adata_path: Path,
    tissue_aliases: List[str],
    out_h5_path: Path,
    cfg: ExtractConfig,
    device: torch.device,
    sc_concept_training_id: str,
    model_name: str,
    checkpoint_path: Path,
) -> Dict[str, Any]:
    """
    Stream a large reference h5ad in chunks from backed mode -> HDF5 embeddings.
    """
    print(f"\n[Reference] {reference_name}")
    print(f"  source      : {adata_path}")
    print(f"  tissue_alias: {tissue_aliases}")

    backed = ad.read_h5ad(adata_path, backed="r")

    if cfg.gene_id_column not in backed.var.columns:
        raise KeyError(
            f"gene_id_column='{cfg.gene_id_column}' not found in adata.var. "
            f"Available columns: {list(backed.var.columns)}"
        )

    gene_ids = backed.var[cfg.gene_id_column].astype(str).to_numpy()

    if pd.isna(backed.var[cfg.gene_id_column]).any():
        raise ValueError(f"{adata_path.name}: found missing values in adata.var['{cfg.gene_id_column}'].")

    if len(np.unique(gene_ids)) != len(gene_ids):
        raise ValueError(
            f"{adata_path.name}: adata.var['{cfg.gene_id_column}'] contains duplicate gene IDs."
        )

    # Vocabulary coverage of this reference (recorded in the H5 attrs and manifest)
    cov = compute_vocab_coverage(gene_ids, concept.tokenizer, species=cfg.species)
    print(
        f"  vocab cov   : {cov['n_genes_in_vocab']}/{cov['n_genes_panel']} genes in vocab "
        f"({100.0 * cov['vocab_coverage_frac']:.1f}%)"
    )

    try:
        n_obs = int(backed.n_obs)
        n_vars = int(backed.n_vars)

        if n_obs == 0:
            raise ValueError(f"{adata_path.name}: reference contains 0 cells; refusing to write an empty embedding file.")

        print(f"  n_obs       : {n_obs}")
        print(f"  n_vars      : {n_vars}")
        print(f"  gene_id_col : {cfg.gene_id_column}")

        out_h5_path.parent.mkdir(parents=True, exist_ok=True)
        if out_h5_path.exists():
            out_h5_path.unlink()

        np_dtype = np.float16 if cfg.out_dtype == "fp16" else np.float32

        with h5py.File(out_h5_path, "w") as f_out:
            # common attrs
            f_out.attrs["reference_name"] = str(reference_name)
            f_out.attrs["source_h5ad"] = str(adata_path.resolve())
            f_out.attrs["tissue_aliases_json"] = json.dumps(tissue_aliases)
            f_out.attrs["sc_concept_training_id"] = str(sc_concept_training_id)
            f_out.attrs["model_name"] = str(model_name)
            f_out.attrs["checkpoint_path"] = str(checkpoint_path.resolve())
            f_out.attrs["gene_id_column"] = str(cfg.gene_id_column)
            f_out.attrs["max_tokens"] = -1 if cfg.max_tokens is None else int(cfg.max_tokens)
            f_out.attrs["gene_sampling_strategy"] = "" if cfg.gene_sampling_strategy is None else str(cfg.gene_sampling_strategy)
            f_out.attrs["embedding_dtype"] = "float16" if np_dtype == np.float16 else "float32"
            f_out.attrs["n_obs"] = n_obs
            f_out.attrs["n_vars"] = n_vars
            f_out.attrs["n_genes_panel"] = int(cov["n_genes_panel"])
            f_out.attrs["n_genes_in_vocab"] = int(cov["n_genes_in_vocab"])
            f_out.attrs["vocab_coverage_frac"] = float(cov["vocab_coverage_frac"])
            f_out.attrs["scConcept_version"] = "v2"
            f_out.attrs["species"] = str(cfg.species)

            # fixed-size metadata datasets
            f_out.create_dataset(
                "row_idx",
                data=np.arange(n_obs, dtype=np.int64),
                compression="gzip",
                compression_opts=1,
                shuffle=True,
            )
            f_out.create_dataset(
                "obs_name",
                shape=(n_obs,),
                dtype=h5py.string_dtype(encoding="utf-8"),
            )

            emb_initialized = False
            write_offset = 0
            embedding_dim = None

            n_chunks = (n_obs + cfg.cells_per_chunk - 1) // cfg.cells_per_chunk
            for chunk_idx, start in enumerate(range(0, n_obs, cfg.cells_per_chunk), start=1):
                stop = min(start + cfg.cells_per_chunk, n_obs)
                print(f"  chunk {chunk_idx:>3}/{n_chunks}: cells [{start}:{stop})")

                adata_chunk = _make_minimal_chunk_adata(
                    backed_adata=backed,
                    adata_path=adata_path,
                    start=start,
                    stop=stop,
                )

                f_out["obs_name"][start:stop] = np.asarray(adata_chunk.obs_names.astype(str), dtype=object)

                # loader for this chunk
                chunk_cfg = ExtractConfig(
                    gene_id_column=cfg.gene_id_column,
                    batch_size=cfg.batch_size,
                    num_workers=cfg.num_workers,
                    cells_per_chunk=cfg.cells_per_chunk,
                    max_tokens=cfg.max_tokens,
                    gene_sampling_strategy=cfg.gene_sampling_strategy,
                    out_dtype=cfg.out_dtype,
                    species=cfg.species,
                    compression_level=cfg.compression_level,
                    chunk_rows=cfg.chunk_rows,
                )
                loader = _build_loader_for_chunk(
                    concept=concept,
                    adata_chunk=adata_chunk,
                    cfg=chunk_cfg,
                    device=device,
                )

                local_offset = 0
                with torch.inference_mode():
                    with _autocast_context(device):
                        for batch_idx, batch in enumerate(tqdm(loader, desc=f"    embed {reference_name}", unit="batch")):
                            batch = {
                                k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
                                for k, v in batch.items()
                            }
                            out = concept.model.predict_step(batch, batch_idx)

                            # predict_step returns cls_cell_emb (512-dim, L2-normalized)
                            cls_np = _tensor_to_numpy_for_storage(out["cls_cell_emb"], np_dtype)

                            if not emb_initialized:
                                embedding_dim = int(cls_np.shape[1])
                                cls_chunks = (min(cfg.chunk_rows, n_obs), embedding_dim)
                                f_out.create_dataset(
                                    "cls_emb",
                                    shape=(n_obs, embedding_dim),
                                    dtype=np_dtype,
                                    chunks=cls_chunks,
                                    compression="gzip",
                                    compression_opts=cfg.compression_level,
                                    shuffle=True,
                                )

                                f_out.attrs["embedding_dim"] = embedding_dim
                                emb_initialized = True

                            bs = int(cls_np.shape[0])
                            sl = slice(write_offset + local_offset, write_offset + local_offset + bs)
                            f_out["cls_emb"][sl, :] = cls_np
                            local_offset += bs

                expected = stop - start
                if local_offset != expected:
                    raise RuntimeError(
                        f"[{reference_name}] wrote {local_offset} embeddings for chunk [{start}:{stop}) "
                        f"but expected {expected}"
                    )

                write_offset += local_offset

                del loader
                del adata_chunk
                if device.type == "cuda":
                    torch.cuda.empty_cache()

            if write_offset != n_obs:
                raise RuntimeError(f"[{reference_name}] wrote {write_offset} rows but expected {n_obs}")

        return {
            "reference_name": reference_name,
            "input_h5ad": str(adata_path.resolve()),
            "output_h5": str(out_h5_path.resolve()),
            "tissue_aliases": tissue_aliases,
            "n_obs": n_obs,
            "n_vars": n_vars,
            "gene_id_column": cfg.gene_id_column,
            "embedding_dim": embedding_dim,
            **cov,
        }

    finally:
        if hasattr(backed, "file") and backed.file is not None:
            backed.file.close()



# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main(args):

    # Device
    if args.device is not None:
        device = torch.device(args.device)
    else:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    print(f"Using device: {device}")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    # Resolve references
    if args.ref_json is None and args.h5ad_files is None:
        raise ValueError("Provide at least one of --ref_json or --h5ad_files.")
    
    ref_json = Path(args.ref_json) if args.ref_json is not None else None

    references = _resolve_references(
        ref_json=ref_json if not args.h5ad_files else None,
        h5ad_files=args.h5ad_files,
        tissues=args.tissues,
    )
    if len(references) == 0:
        raise RuntimeError("No reference h5ad files selected.")

    print(f"Found {len(references)} unique reference file(s) to process.")

    # Load adapted scConcept
    concept, ckpt_path, model_name, training_manifest, species = _load_scconcept_from_training(
        sc_concept_training_id=args.sc_concept_training_id,
        device=device,
    )

    # Extraction parameters
    max_tokens = training_manifest['extract_max_tokens']
    gene_sampling_strategy = training_manifest['extract_gene_sampling_strategy']

    # Output directory
    out_dir = get_full_path("eval", "eval_datasets", "scRNAseq_references", "scConcept_embs", args.sc_concept_training_id, scratch=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Config for extraction
    cfg = ExtractConfig(
        gene_id_column=args.gene_id_column,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        cells_per_chunk=args.cells_per_chunk,
        max_tokens=max_tokens,
        gene_sampling_strategy=gene_sampling_strategy,
        out_dtype=args.dtype,
        species=species,
        compression_level=args.compression_level,
        chunk_rows=args.chunk_rows,
    )

    manifest_path = out_dir / "manifest.json"
    existing_manifest = _load_existing_manifest(manifest_path)
    existing_files = dict(existing_manifest.get("files", {}))

    manifest = {
        "sc_concept_training_id": args.sc_concept_training_id,
        "scConcept_version": "v2",
        "model_name": model_name,
        "species": species,
        "checkpoint_path": str(ckpt_path.resolve()),
        "device": str(device),
        "output_dir": str(out_dir.resolve()),
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "cells_per_chunk": args.cells_per_chunk,
        "max_tokens": max_tokens,
        "gene_sampling_strategy": gene_sampling_strategy,
        "gene_id_column_arg": args.gene_id_column,
        "dtype": args.dtype,
        "compression_level": args.compression_level,
        "chunk_rows": args.chunk_rows,
        "min_vocab_coverage_frac": float(args.min_vocab_coverage_frac),
        "min_genes_in_vocab": int(args.min_genes_in_vocab),
        "files": existing_files,   # preserve previous per-file entries
    }

    # Check the vocabulary coverage of the references to (re)process before embedding
    refs_to_process = [r for r in references if args.overwrite or not (out_dir / f"{r['output_stem']}.h5").exists()]
    preflight_reference_vocab_coverage(
        references=refs_to_process,
        tokenizer=concept.tokenizer,
        gene_id_column=cfg.gene_id_column,
        min_coverage_frac=float(args.min_vocab_coverage_frac),
        min_genes_in_vocab=int(args.min_genes_in_vocab),
        species=species,
    )

    # Process each reference and stream embeddings to HDF5
    t0 = time.time()
    for i, ref in enumerate(references, start=1):
        ref_name = ref["reference_name"]
        adata_path = ref["path"]
        tissue_aliases = ref["tissue_aliases"]
        out_key = ref["output_stem"]
        out_h5 = out_dir / f"{out_key}.h5"

        if out_h5.exists() and not args.overwrite:
            print(f"\n[{i}/{len(references)}] Skipping existing file: {out_h5.name}")
            continue

        print(f"\n[{i}/{len(references)}] Processing: {adata_path.name}")
        stats = stream_reference_to_h5(
            concept=concept,
            reference_name=ref_name,
            adata_path=adata_path,
            tissue_aliases=tissue_aliases,
            out_h5_path=out_h5,
            cfg=cfg,
            device=device,
            sc_concept_training_id=args.sc_concept_training_id,
            model_name=model_name,
            checkpoint_path=ckpt_path,
        )
        stats["output_stem"] = out_key
        manifest["files"][out_key] = stats

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\nDone in {(time.time() - t0)/60:.2f} min")
    print(f"Manifest written to: {manifest_path}")
    print(f"Embeddings directory  : {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Compute raw scConcept embeddings for scRNA-seq reference h5ad files.")

    # scConcept model
    ap.add_argument("--sc_concept_training_id", type=str,
                    required=True,
                    help="scConcept training ID, e.g. 'training_1'.")

    # Reference selection
    ap.add_argument("--ref_json", type=str,
                    default="/path/to/ALHENA-paper/data/eval_scRNAseq_ref/ref_tissue_files.json",
                    help="Path to ref_tissue_files.json. Default uses the standard eval path.")
    
    ap.add_argument("--h5ad_files", type=str,
                    nargs="*",
                    default=None,
                    help="Optional explicit list of h5ad files to process instead of ref_json.")
    
    ap.add_argument("--tissues", type=str,
                    nargs="*",
                    default=None,
                    help="Optional subset of tissue keys from ref_tissue_files.json.")

    # Gene identifiers-
    ap.add_argument("--gene_id_column", type=str,
                    default="gene_ids",
                    help="Column in adata.var to use for genes.")

    # Vocabulary-coverage check
    ap.add_argument("--min_vocab_coverage_frac", type=float, default=0.3,
                    help="Minimum fraction of a reference's gene panel present in the scConcept corpus vocabulary "
                         "(n_genes_in_vocab / n_genes_panel), checked before embedding. 0 disables the check.")
    ap.add_argument("--min_genes_in_vocab", type=int, default=1,
                    help="Minimum number of a reference's panel genes present in the corpus vocabulary. "
                         "0 disables the check.")

    # Extraction parameters
    ap.add_argument("--batch_size", type=int, default=256, help="Batch size for scConcept inference.")
    ap.add_argument("--num_workers", type=int, default=4, help="DataLoader workers for tokenization/collate.")
    ap.add_argument("--cells_per_chunk", type=int, default=100000, help="How many cells to materialize per backed AnnData chunk.")
    ap.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "fp32"], help="Storage dtype for embeddings.")
    ap.add_argument("--compression_level", type=int, default=4, help="gzip compression level for embedding datasets.")
    ap.add_argument("--chunk_rows", type=int, default=4096, help="HDF5 chunk rows for embedding datasets.")

    # Technical
    ap.add_argument("--device", type=str, default=None, help="Device: cuda / cuda:0 / mps / cpu. Default auto.")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing output .h5 files.")

    args = ap.parse_args()
    main(args)