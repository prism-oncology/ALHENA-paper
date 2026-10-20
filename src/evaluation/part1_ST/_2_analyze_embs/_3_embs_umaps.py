#!/usr/bin/env python
"""
UMAPs of the H&E / ST evaluation embeddings of eval_st_<training_dataset_id> (H&E patch and cell embeddings with and
without CLIP projection, ST graph and cell embeddings, joint H&E + ST embeddings), per group of slides.

Saved in <ROOT_HistoST> / eval / eval_datasets / st_eval_data / eval_st_<training_dataset_id> / embeddings_analysis / umap_embeddings / t<training_num>e<checkpoint_num> /
    - a0_version.json: analysis version and cell-type parameters
    - <config>/ (all_sources, source_<source>, source_<source>_subset_<subset>):
        - umap_*.png (and .pdf): UMAP figures
        - umap_metrics.csv: kNN label purity and geometry metrics of each UMAP family
        - report.json: generated and skipped families, sampling and UMAP parameters
        - sample_*.parquet, umap_*_coords.parquet: sampled rows and UMAP coordinates (unless --no_keep_intermediate_parquet)
    - by_slides_STHELAR/: the same files for each STHELAR slide, prefixed by <slide_id>_
"""

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import anndata as ad
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.sparse import issparse
from sklearn.decomposition import PCA
import umap


# Add the project root to sys.path so that 'src' can be imported
project_root = Path(__file__).resolve().parents[4]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.constants import get_full_path

# Sibling modules in this folder.
from celltype_axis import (  # noqa: E402
    MISSING_ANNOTATION_LABELS,
    apply_label_map,
    build_axes,
)
from umap_structure import (  # noqa: E402
    JOINT_LABEL_COLUMNS,
    POPULATION_HE_ONLY,
    POPULATION_JOINT,
    POPULATION_ST_ONLY,
    MetricsUnresolvableError,
    assert_any_metric_resolvable,
    compute_umap_family_metrics,
    structure_metrics_to_rows,
)

# Output format version of this analysis (written into a0_version.json, checked before writing into an existing folder)
A0_ANALYSIS_VERSION = 4


# ---------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------


def sanitize_name(x: str) -> str:
    x = str(x)
    x = re.sub(r"\s+", "_", x.strip())
    x = re.sub(r"[^A-Za-z0-9_.-]+", "_", x)
    return x if x else "NA"



def decode_h5_string_array(arr) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr[:, 0]
    out = []
    for x in arr:
        if isinstance(x, (bytes, bytearray, np.bytes_)):
            out.append(x.decode("utf-8"))
        else:
            out.append(str(x))
    return np.asarray(out, dtype=object)



def l2_normalize_rows(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return x / norms



def ensure_dense_2d(x) -> np.ndarray:
    if issparse(x):
        x = x.toarray()
    x = np.asarray(x)
    if x.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape={x.shape}")
    return x



def pick_point_size(n: int) -> float:
    if n <= 5_000:
        return 8.0
    if n <= 20_000:
        return 5.0
    if n <= 50_000:
        return 3.0
    return 1.8



def pick_alpha(n: int) -> float:
    if n <= 10_000:
        return 0.85
    if n <= 50_000:
        return 0.65
    return 0.45



def sample_indices(n_total: int, n_keep: int, rng: np.random.Generator) -> np.ndarray:
    if n_total <= 0:
        return np.zeros((0,), dtype=np.int64)
    if n_keep is None or n_keep <= 0 or n_total <= n_keep:
        return np.arange(n_total, dtype=np.int64)
    idx = rng.choice(n_total, size=n_keep, replace=False)
    idx.sort()
    return idx.astype(np.int64, copy=False)



def allocate_total_budget_per_slide(
    available_per_slide: Dict[str, int],
    total_budget: Optional[int],
    rng: np.random.Generator,
) -> Dict[str, int]:
    """
    Allocate a global sampling budget across slides as evenly as possible, within each slide's available count
    (max-min fair integer allocation, leftover budget redistributed). None or <= 0 keeps everything.
    """
    available = {
        str(slide_id): int(n)
        for slide_id, n in available_per_slide.items()
        if int(n) > 0
    }
    if not available:
        return {}

    if total_budget is None or int(total_budget) <= 0:
        return available.copy()

    total_budget = min(int(total_budget), int(sum(available.values())))
    if total_budget <= 0:
        return {}

    slides = list(available.keys())
    alloc = {slide_id: 0 for slide_id in slides}
    remaining_cap = available.copy()
    active = set(slides)
    remaining_budget = int(total_budget)

    while active and remaining_budget > 0:
        active_list = [slide_id for slide_id in slides if slide_id in active]
        floor_share = remaining_budget // len(active_list)

        # If budget is smaller than number of active slides, distribute single
        # extra units to a random subset of active slides
        if floor_share == 0:
            order = np.asarray(active_list, dtype=object)
            if order.size > 1:
                order = order[rng.permutation(order.size)]
            for slide_id in order[:remaining_budget].tolist():
                alloc[slide_id] += 1
            remaining_budget = 0
            break

        # Slides whose remaining capacity is <= the equal-share floor are saturated first
        saturated = [slide_id for slide_id in active_list if remaining_cap[slide_id] <= floor_share]
        if saturated:
            for slide_id in saturated:
                take = remaining_cap[slide_id]
                alloc[slide_id] += take
                remaining_budget -= take
                remaining_cap[slide_id] = 0
                active.remove(slide_id)
            continue

        # Otherwise all active slides can absorb the floor share
        for slide_id in active_list:
            alloc[slide_id] += floor_share
            remaining_cap[slide_id] -= floor_share
        remaining_budget -= floor_share * len(active_list)

        # Distribute the final remainder (+1) fairly across active slides
        if remaining_budget > 0:
            order = np.asarray(active_list, dtype=object)
            if order.size > 1:
                order = order[rng.permutation(order.size)]
            for slide_id in order[:remaining_budget].tolist():
                alloc[slide_id] += 1
                remaining_cap[slide_id] -= 1
            remaining_budget = 0

    return {slide_id: n for slide_id, n in alloc.items() if n > 0}



def read_h5_rows_2d(h5_path: Path, dataset_name: str, row_indices: np.ndarray) -> np.ndarray:
    row_indices = np.asarray(row_indices, dtype=np.int64)
    if row_indices.ndim != 1:
        raise ValueError("row_indices must be 1D")
    if row_indices.size == 0:
        with h5py.File(h5_path, "r") as f:
            d = f[dataset_name]
            return np.zeros((0, d.shape[1]), dtype=np.float32)

    order = np.argsort(row_indices)
    inv = np.empty_like(order)
    inv[order] = np.arange(order.size)
    idx_sorted = row_indices[order]

    with h5py.File(h5_path, "r") as f:
        arr = f[dataset_name][idx_sorted, :]
    arr = np.asarray(arr)
    arr = arr[inv]
    return arr.astype(np.float32, copy=False)



def read_h5_1d_strings(h5_path: Path, dataset_name: str) -> np.ndarray:
    with h5py.File(h5_path, "r") as f:
        arr = f[dataset_name][:]
    return decode_h5_string_array(arr)



def read_h5ad_rows(h5ad_path: Path, row_indices: np.ndarray) -> np.ndarray:
    row_indices = np.asarray(row_indices, dtype=np.int64)
    if row_indices.ndim != 1:
        raise ValueError("row_indices must be 1D")

    a = ad.read_h5ad(h5ad_path, backed="r")
    try:
        if row_indices.size == 0:
            n_vars = a.shape[1]
            return np.zeros((0, n_vars), dtype=np.float32)
        x = a.X[row_indices]
        x = ensure_dense_2d(x)
        return x.astype(np.float32, copy=False)
    finally:
        if getattr(a, "file", None) is not None:
            a.file.close()



def save_dataframe_parquet(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)


def save_intermediate_dataframe_parquet(
    df: pd.DataFrame,
    out_path: Path,
    keep_on_disk: bool,
) -> None:
    """Save a sampled intermediate parquet file (sample_*.parquet) only if keep_on_disk."""
    if keep_on_disk:
        save_dataframe_parquet(df, out_path)


def delete_file_if_exists(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def filter_available_color_cols(meta_df: pd.DataFrame, candidate_cols: Sequence[str]) -> List[str]:
    """Keep only columns that exist and are not entirely missing."""
    out: List[str] = []
    for c in candidate_cols:
        if c not in meta_df.columns:
            continue
        s = meta_df[c]
        if s.isna().all():
            continue
        out.append(c)
    return out



# ---------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------


@dataclass
class ConfigSpec:
    name: str
    title: str
    slides_df: pd.DataFrame
    allow_cells: bool
    allow_st: bool



def build_config_specs(meta: pd.DataFrame) -> List[ConfigSpec]:
    specs: List[ConfigSpec] = []
    all_sources_have_cells = (~meta["source"].astype(str).eq("HEST")).all()
    all_sources_have_st = (~meta["source"].astype(str).eq("HEST")).all()

    specs.append(
        ConfigSpec(
            name="all_sources",
            title="All evaluation slides together",
            slides_df=meta.copy(),
            allow_cells=all_sources_have_cells,  # HEST does not have cell information
            allow_st=all_sources_have_st,        # HEST does not have ST embeddings
        )
    )

    for source in sorted(meta["source"].astype(str).unique().tolist()):
        df_source = meta.loc[meta["source"].astype(str) == source].copy()
        source_has_cells = (str(source) != "HEST")
        source_has_st = (str(source) != "HEST")
        specs.append(
            ConfigSpec(
                name=f"source_{sanitize_name(source)}",
                title=f"Source = {source}",
                slides_df=df_source,
                allow_cells=source_has_cells,
                allow_st=source_has_st,
            )
        )

        for subset in sorted(df_source["subset"].astype(str).unique().tolist()):
            df_subset = df_source.loc[df_source["subset"].astype(str) == subset].copy()
            specs.append(
                ConfigSpec(
                    name=f"source_{sanitize_name(source)}_subset_{sanitize_name(subset)}",
                    title=f"Source = {source} | Subset = {subset}",
                    slides_df=df_subset,
                    allow_cells=source_has_cells,
                    allow_st=source_has_st,
                )
            )

    return specs


def build_sthelar_slide_specs(meta: pd.DataFrame) -> List[ConfigSpec]:
    """Build one config per STHELAR slide (outputs in by_slides_STHELAR/)."""
    specs: List[ConfigSpec] = []
    df_sthelar = meta.loc[meta["source"].astype(str) == "STHELAR"].copy()

    for slide_id in sorted(df_sthelar["slide_id"].astype(str).unique().tolist()):
        df_slide = df_sthelar.loc[df_sthelar["slide_id"].astype(str) == slide_id].copy()
        row = df_slide.iloc[0]
        specs.append(
            ConfigSpec(
                name=f"slide_{sanitize_name(slide_id)}",
                title=(
                    f"Slide = {slide_id} | Tissue = {row['tissue_HistoST']} | "
                    f"Subset = {row['subset']} | Source = STHELAR"
                ),
                slides_df=df_slide,
                allow_cells=True,
                allow_st=True,
            )
        )

    return specs


# ---------------------------------------------------------------------
# Repository for all embeddings / metadata
# ---------------------------------------------------------------------


class EvalEmbeddingRepository:
    
    def __init__(
        self,
        eval_root: Path,
        metadata: pd.DataFrame,
        training_dataset_id: str,
        training_num: str,
        checkpoint_num: str,
        training_id_novae: Optional[str] = None,
    ) -> None:
        
        self.eval_root = Path(eval_root)
        self.metadata = metadata.copy()
        self.training_id_novae = None if training_id_novae is None else str(training_id_novae)

        self.he_root = self.eval_root / "he_embeddings" / f"HEembs_t{training_num}e{checkpoint_num}"
        self.he_main_cell_root = self.he_root / "main_HEemb_cellID"
        self.st_graph_root = (self.eval_root / "st_sthelar_embeddings_clip" / f"GRAPHembs_t{training_num}e{checkpoint_num}")
        self.st_cell_root = (self.eval_root / "st_sthelar_embeddings_clip" / f"CELLembs_t{training_num}e{checkpoint_num}")
        self.eval_st_root = self.eval_root / "st"

        if not self.he_root.exists():
            raise FileNotFoundError(f"H&E embeddings root not found: {self.he_root}")

        self.slide_meta = self.metadata.set_index("slide_id", drop=False)

        # Optional Novae patch-domain metadata
        self.novae_patch_domain_root: Optional[Path] = None
        self.patch_domain_cols: List[str] = []
        self.patch_domain_frac_cols: List[str] = []

        if self.training_id_novae is not None:
            self.novae_patch_domain_root = self.eval_root.parent / f"eval_st_{training_dataset_id}" / "novae_domains_HEpatches" / self.training_id_novae
            if not self.novae_patch_domain_root.exists():
                raise FileNotFoundError(f"Novae patch-domain root not found: {self.novae_patch_domain_root}")

            manifest_path = self.novae_patch_domain_root / "manifest.json"
            if not manifest_path.exists():
                raise FileNotFoundError(f"Novae patch-domain manifest not found: {manifest_path}")

            with manifest_path.open("r") as f:
                manifest = json.load(f)

            requested_n_domains = manifest.get("requested_n_domains", None)
            if requested_n_domains is None:
                raise KeyError(f"'requested_n_domains' missing from manifest: {manifest_path}")

            self.patch_domain_cols = [f"novae_patch_domain_{int(k)}" for k in requested_n_domains]
            self.patch_domain_frac_cols = [f"novae_patch_domain_frac_{int(k)}" for k in requested_n_domains]

    def he_h5_path(self, slide_id: str) -> Path:
        return self.he_root / f"{slide_id}.h5"

    def he_main_cell_path(self, slide_id: str) -> Path:
        return self.he_main_cell_root / f"{slide_id}.parquet"

    def st_graph_h5ad_path(self, slide_id: str) -> Path:
        return self.st_graph_root / f"{slide_id}.h5ad"

    def st_cell_h5ad_path(self, slide_id: str) -> Path:
        return self.st_cell_root / f"{slide_id}.h5ad"

    def eval_st_h5ad_path(self, slide_id: str) -> Path:
        return self.eval_st_root / f"{slide_id}.h5ad"

    def has_he_patch(self, slide_id: str) -> bool:
        return self.he_h5_path(slide_id).exists()

    def has_he_cell(self, slide_id: str) -> bool:
        return self.he_h5_path(slide_id).exists() and self.he_main_cell_path(slide_id).exists()

    def has_st_graph(self, slide_id: str) -> bool:
        return self.st_graph_h5ad_path(slide_id).exists()

    def has_st_cell(self, slide_id: str) -> bool:
        return self.st_cell_h5ad_path(slide_id).exists()

    @lru_cache(maxsize=None)
    def get_he_patch_index(self, slide_id: str) -> pd.DataFrame:
        h5_path = self.he_h5_path(slide_id)
        if not h5_path.exists():
            raise FileNotFoundError(h5_path)
        patch_ids = read_h5_1d_strings(h5_path, "patch_id")
        df = pd.DataFrame(
            {
                "patch_id": patch_ids.astype(object, copy=False),
                "row_index": np.arange(len(patch_ids), dtype=np.int32),
            }
        )
        if df["patch_id"].duplicated().any():
            dup = df.loc[df["patch_id"].duplicated(), "patch_id"].head(10).tolist()
            raise ValueError(f"Duplicated patch_id in H&E patch embeddings for slide '{slide_id}'. Examples: {dup}")
        return df.set_index("patch_id", drop=False)

    @lru_cache(maxsize=None)
    def get_he_main_cell_df(self, slide_id: str) -> pd.DataFrame:
        p = self.he_main_cell_path(slide_id)
        if not p.exists():
            raise FileNotFoundError(p)
        df = pd.read_parquet(p).copy()
        required = [
            "cell_id_int",
            "cell_id",
            "selected_patch_cell_row_index",
        ]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise KeyError(f"{p} missing required columns: {missing}")
        df["cell_id_int"] = df["cell_id_int"].astype(np.int32)
        df["cell_id"] = df["cell_id"].astype(str)
        df["selected_patch_cell_row_index"] = df["selected_patch_cell_row_index"].astype(np.int64)

        if df["cell_id_int"].duplicated().any():
            dup = df.loc[df["cell_id_int"].duplicated(), "cell_id_int"].head(10).tolist()
            raise ValueError(f"Duplicate cell_id_int in {p}. Examples: {dup}")

        return df.set_index("cell_id_int", drop=False)

    @lru_cache(maxsize=None)
    def get_eval_cell_metadata(self, slide_id: str) -> pd.DataFrame:
        p = self.eval_st_h5ad_path(slide_id)
        if not p.exists():
            raise FileNotFoundError(p)

        a = ad.read_h5ad(p, backed="r")
        try:
            obs = a.obs.copy()
        finally:
            if getattr(a, "file", None) is not None:
                a.file.close()

        if "is_cell" in obs.columns:
            mask = obs["is_cell"].astype(bool).to_numpy()
        else:
            idx = obs.index.astype(str)
            mask = np.asarray([x.startswith(f"{slide_id}_cell_") for x in idx], dtype=bool)

        obs = obs.loc[mask].copy()
        if obs.empty:
            return pd.DataFrame(
                columns=["cell_id_int", "cell_id", "cell_type", "row_index"]
            ).set_index("cell_id_int", drop=False)

        if "cell_id_int" not in obs.columns:
            vals = []
            pat = re.compile(rf"^{re.escape(slide_id)}_cell_(\d+)$")
            for x in obs.index.astype(str):
                m = pat.match(x)
                if m is None:
                    raise ValueError(f"Cannot parse cell_id_int from obs index '{x}' for slide '{slide_id}'.")
                vals.append(int(m.group(1)))
            obs["cell_id_int"] = np.asarray(vals, dtype=np.int32)

        if "cell_id" not in obs.columns:
            obs["cell_id"] = obs["cell_id_int"].astype(str)

        if "cell_type" not in obs.columns:
            # Missing annotation: `UNKNOWN` label (distinct from the `Unknown` cell type)
            obs["cell_type"] = MISSING_ANNOTATION_LABELS[0]

        obs["cell_id_int"] = obs["cell_id_int"].astype(np.int32)
        obs["cell_id"] = obs["cell_id"].astype(str)
        obs["cell_type"] = obs["cell_type"].astype(str)
        obs["row_index"] = np.arange(obs.shape[0], dtype=np.int32)

        if obs["cell_id_int"].duplicated().any():
            dup = obs.loc[obs["cell_id_int"].duplicated(), "cell_id_int"].head(10).tolist()
            raise ValueError(f"Duplicate cell_id_int in eval cell metadata for slide '{slide_id}'. Examples: {dup}")

        return obs[["cell_id_int", "cell_id", "cell_type", "row_index"]].set_index("cell_id_int", drop=False)

    @lru_cache(maxsize=None)
    def get_st_graph_index(self, slide_id: str) -> pd.DataFrame:
        p = self.st_graph_h5ad_path(slide_id)
        if not p.exists():
            raise FileNotFoundError(p)

        a = ad.read_h5ad(p, backed="r")
        try:
            obs = a.obs.copy()
        finally:
            if getattr(a, "file", None) is not None:
                a.file.close()

        patch_id = obs["patch_id"].astype(str) if "patch_id" in obs.columns else obs.index.astype(str)
        df = pd.DataFrame(
            {
                "patch_id": patch_id.astype(str),
                "row_index": np.arange(obs.shape[0], dtype=np.int32),
            }
        )
        if df["patch_id"].duplicated().any():
            dup = df.loc[df["patch_id"].duplicated(), "patch_id"].head(10).tolist()
            raise ValueError(f"Duplicated patch_id in ST graph embeddings for slide '{slide_id}'. Examples: {dup}")
        return df.set_index("patch_id", drop=False)

    @lru_cache(maxsize=None)
    def get_st_cell_index(self, slide_id: str) -> pd.DataFrame:
        p = self.st_cell_h5ad_path(slide_id)
        if not p.exists():
            raise FileNotFoundError(p)

        a = ad.read_h5ad(p, backed="r")
        try:
            obs = a.obs.copy()
        finally:
            if getattr(a, "file", None) is not None:
                a.file.close()

        if "cell_id_int" not in obs.columns:
            raise KeyError(f"{p} is missing obs['cell_id_int'].")

        if "cell_id" not in obs.columns:
            obs["cell_id"] = obs.index.astype(str)

        if "cell_type" not in obs.columns:
            # Missing annotation: `UNKNOWN` label (distinct from the `Unknown` cell type)
            obs["cell_type"] = MISSING_ANNOTATION_LABELS[0]

        obs["cell_id_int"] = obs["cell_id_int"].astype(np.int32)
        obs["cell_id"] = obs["cell_id"].astype(str)
        obs["cell_type"] = obs["cell_type"].astype(str)
        obs["row_index"] = np.arange(obs.shape[0], dtype=np.int32)

        if obs["cell_id_int"].duplicated().any():
            dup = obs.loc[obs["cell_id_int"].duplicated(), "cell_id_int"].head(10).tolist()
            raise ValueError(f"Duplicate cell_id_int in ST cell embeddings for slide '{slide_id}'. Examples: {dup}")

        return obs[["cell_id_int", "cell_id", "cell_type", "row_index"]].set_index("cell_id_int", drop=False)

    def get_slide_meta_row(self, slide_id: str) -> pd.Series:
        if slide_id not in self.slide_meta.index:
            raise KeyError(f"slide_id '{slide_id}' missing from metadata.")
        return self.slide_meta.loc[slide_id]

    @lru_cache(maxsize=None)
    def get_patch_domain_df(self, slide_id: str) -> pd.DataFrame:
        """
        Load per-patch Novae domains for one slide.
        Returns a DataFrame indexed by patch_id with:
          - patch_id
          - novae_patch_domain_{K}
          - novae_patch_domain_frac_{K}
        If Novae domains are not enabled, returns an empty schema-only DataFrame.
        For HEST slides, returns an empty schema-only DataFrame.
        """
        base_cols = ["patch_id"] + self.patch_domain_cols + self.patch_domain_frac_cols

        if self.novae_patch_domain_root is None or len(self.patch_domain_cols) == 0:
            return pd.DataFrame(columns=base_cols).set_index("patch_id", drop=False)

        source = str(self.get_slide_meta_row(slide_id)["source"])
        p = self.novae_patch_domain_root / f"{slide_id}.parquet"

        if not p.exists():
            if source == "HEST":
                return pd.DataFrame(columns=base_cols).set_index("patch_id", drop=False)
            raise FileNotFoundError(f"Missing Novae patch-domain parquet for STHELAR slide '{slide_id}': {p}")

        needed = ["patch_id"] + self.patch_domain_cols + self.patch_domain_frac_cols
        df = pd.read_parquet(p, columns=needed).copy()
        df["patch_id"] = df["patch_id"].astype(str)

        if df["patch_id"].duplicated().any():
            dup = df.loc[df["patch_id"].duplicated(), "patch_id"].head(10).tolist()
            raise ValueError(f"Duplicate patch_id in Novae patch-domain parquet for slide '{slide_id}'. Examples: {dup}")

        for c in self.patch_domain_cols:
            if c not in df.columns:
                df[c] = pd.Series([pd.NA] * df.shape[0], dtype="string")
            else:
                df[c] = df[c].astype("string")

        for c in self.patch_domain_frac_cols:
            if c not in df.columns:
                df[c] = np.full(df.shape[0], np.nan, dtype=np.float32)
            else:
                df[c] = df[c].astype("float32")

        return df.set_index("patch_id", drop=False)

    def attach_patch_domains(self, slide_id: str, df: pd.DataFrame) -> pd.DataFrame:
        """Left-join patch-domain metadata on patch_id (unchanged if domains are not enabled)."""
        if len(self.patch_domain_cols) == 0:
            return df

        dom = self.get_patch_domain_df(slide_id)
        if dom.empty:
            out = df.copy()
            for c in self.patch_domain_cols:
                out[c] = pd.Series([pd.NA] * out.shape[0], dtype="string")
            for c in self.patch_domain_frac_cols:
                out[c] = np.full(out.shape[0], np.nan, dtype=np.float32)
            return out

        out = df.merge(
            dom.reset_index(drop=True),
            on="patch_id",
            how="left",
            validate="one_to_one",
        )

        for c in self.patch_domain_cols:
            if c not in out.columns:
                out[c] = pd.Series([pd.NA] * out.shape[0], dtype="string")
            else:
                out[c] = out[c].astype("string")

        for c in self.patch_domain_frac_cols:
            if c not in out.columns:
                out[c] = np.full(out.shape[0], np.nan, dtype=np.float32)
            else:
                out[c] = out[c].astype("float32")

        return out

    # ------------------ sampling ------------------

    def sample_he_patches(self, slides_df: pd.DataFrame, np_total: int, rng: np.random.Generator) -> pd.DataFrame:
        slide_ids = slides_df["slide_id"].astype(str).tolist()
        per_slide_idx: Dict[str, pd.DataFrame] = {}
        available_per_slide: Dict[str, int] = {}

        for slide_id in slide_ids:
            if not self.has_he_patch(slide_id):
                continue
            idx_df = self.get_he_patch_index(slide_id).reset_index(drop=True)
            if idx_df.empty:
                continue
            per_slide_idx[slide_id] = idx_df
            available_per_slide[slide_id] = int(idx_df.shape[0])

        keep_per_slide = allocate_total_budget_per_slide(available_per_slide, np_total, rng)

        rows = []
        for slide_id in slide_ids:
            n_keep = keep_per_slide.get(slide_id, 0)
            if n_keep <= 0:
                continue
            idx_df = per_slide_idx[slide_id]
            sel = sample_indices(idx_df.shape[0], n_keep, rng)
            if sel.size == 0:
                continue
            take = idx_df.iloc[sel].copy()
            meta = self.get_slide_meta_row(slide_id)
            take["slide_id"] = slide_id
            take["source"] = str(meta["source"])
            take["subset"] = str(meta["subset"])
            take["tissue_HistoST"] = str(meta["tissue_HistoST"])
            take = self.attach_patch_domains(slide_id, take)
            rows.append(
                take[
                    ["slide_id", "patch_id", "row_index", "source", "subset", "tissue_HistoST"]
                    + self.patch_domain_cols
                    + self.patch_domain_frac_cols
                ]
            )

        return pd.concat(rows, axis=0, ignore_index=True) if rows else pd.DataFrame(
            columns=["slide_id", "patch_id", "row_index", "source", "subset", "tissue_HistoST"]
            + self.patch_domain_cols
            + self.patch_domain_frac_cols
        )

    def sample_he_cells(self, slides_df: pd.DataFrame, nc_total: int, rng: np.random.Generator) -> pd.DataFrame:
        slide_ids = slides_df["slide_id"].astype(str).tolist()
        per_slide_merged: Dict[str, pd.DataFrame] = {}
        available_per_slide: Dict[str, int] = {}

        for slide_id in slide_ids:
            if not self.has_he_cell(slide_id):
                continue
            main_df = self.get_he_main_cell_df(slide_id).reset_index(drop=True)
            cell_meta = self.get_eval_cell_metadata(slide_id).reset_index(drop=True)
            merged = main_df.merge(
                cell_meta[["cell_id_int", "cell_id", "cell_type"]],
                on=["cell_id_int", "cell_id"],
                how="inner",
            )
            if merged.empty:
                continue
            per_slide_merged[slide_id] = merged
            available_per_slide[slide_id] = int(merged.shape[0])

        keep_per_slide = allocate_total_budget_per_slide(available_per_slide, nc_total, rng)

        rows = []
        for slide_id in slide_ids:
            n_keep = keep_per_slide.get(slide_id, 0)
            if n_keep <= 0:
                continue
            merged = per_slide_merged[slide_id]
            sel = sample_indices(merged.shape[0], n_keep, rng)
            if sel.size == 0:
                continue
            take = merged.iloc[sel].copy()
            meta = self.get_slide_meta_row(slide_id)
            take["slide_id"] = slide_id
            take["source"] = str(meta["source"])
            take["subset"] = str(meta["subset"])
            take["tissue_HistoST"] = str(meta["tissue_HistoST"])
            rows.append(
                take[
                    [
                        "slide_id",
                        "cell_id_int",
                        "cell_id",
                        "cell_type",
                        "selected_patch_cell_row_index",
                        "source",
                        "subset",
                        "tissue_HistoST",
                    ]
                ]
            )

        return pd.concat(rows, axis=0, ignore_index=True) if rows else pd.DataFrame(
            columns=[
                "slide_id", "cell_id_int", "cell_id", "cell_type",
                "selected_patch_cell_row_index", "source", "subset", "tissue_HistoST",
            ]
        )

    def sample_st_graphs(self, slides_df: pd.DataFrame, np_total: int, rng: np.random.Generator) -> pd.DataFrame:
        slide_ids = slides_df["slide_id"].astype(str).tolist()
        per_slide_idx: Dict[str, pd.DataFrame] = {}
        available_per_slide: Dict[str, int] = {}

        for slide_id in slide_ids:
            if not self.has_st_graph(slide_id):
                continue
            idx_df = self.get_st_graph_index(slide_id).reset_index(drop=True)
            if idx_df.empty:
                continue
            per_slide_idx[slide_id] = idx_df
            available_per_slide[slide_id] = int(idx_df.shape[0])

        keep_per_slide = allocate_total_budget_per_slide(available_per_slide, np_total, rng)

        rows = []
        for slide_id in slide_ids:
            n_keep = keep_per_slide.get(slide_id, 0)
            if n_keep <= 0:
                continue
            idx_df = per_slide_idx[slide_id]
            sel = sample_indices(idx_df.shape[0], n_keep, rng)
            if sel.size == 0:
                continue
            take = idx_df.iloc[sel].copy()
            meta = self.get_slide_meta_row(slide_id)
            take["slide_id"] = slide_id
            take["source"] = str(meta["source"])
            take["subset"] = str(meta["subset"])
            take["tissue_HistoST"] = str(meta["tissue_HistoST"])
            take = self.attach_patch_domains(slide_id, take)
            rows.append(
                take[
                    ["slide_id", "patch_id", "row_index", "source", "subset", "tissue_HistoST"]
                    + self.patch_domain_cols
                    + self.patch_domain_frac_cols
                ]
            )

        return pd.concat(rows, axis=0, ignore_index=True) if rows else pd.DataFrame(
            columns=["slide_id", "patch_id", "row_index", "source", "subset", "tissue_HistoST"]
            + self.patch_domain_cols
            + self.patch_domain_frac_cols
        )

    def sample_st_cells(self, slides_df: pd.DataFrame, nc_total: int, rng: np.random.Generator) -> pd.DataFrame:
        slide_ids = slides_df["slide_id"].astype(str).tolist()
        per_slide_idx: Dict[str, pd.DataFrame] = {}
        available_per_slide: Dict[str, int] = {}

        for slide_id in slide_ids:
            if not self.has_st_cell(slide_id):
                continue
            idx_df = self.get_st_cell_index(slide_id).reset_index(drop=True)
            if idx_df.empty:
                continue
            per_slide_idx[slide_id] = idx_df
            available_per_slide[slide_id] = int(idx_df.shape[0])

        keep_per_slide = allocate_total_budget_per_slide(available_per_slide, nc_total, rng)

        rows = []
        for slide_id in slide_ids:
            n_keep = keep_per_slide.get(slide_id, 0)
            if n_keep <= 0:
                continue
            idx_df = per_slide_idx[slide_id]
            sel = sample_indices(idx_df.shape[0], n_keep, rng)
            if sel.size == 0:
                continue
            take = idx_df.iloc[sel].copy()
            meta = self.get_slide_meta_row(slide_id)
            take["slide_id"] = slide_id
            take["source"] = str(meta["source"])
            take["subset"] = str(meta["subset"])
            take["tissue_HistoST"] = str(meta["tissue_HistoST"])
            rows.append(
                take[["slide_id", "cell_id_int", "cell_id", "cell_type", "row_index", "source", "subset", "tissue_HistoST"]]
            )

        return pd.concat(rows, axis=0, ignore_index=True) if rows else pd.DataFrame(
            columns=["slide_id", "cell_id_int", "cell_id", "cell_type", "row_index", "source", "subset", "tissue_HistoST"]
        )

    def sample_joint_patches(self, slides_df: pd.DataFrame, np_total: int, rng: np.random.Generator) -> pd.DataFrame:
        slide_ids = slides_df["slide_id"].astype(str).tolist()
        per_slide_common: Dict[str, pd.DataFrame] = {}
        available_per_slide: Dict[str, int] = {}

        for slide_id in slide_ids:
            if (not self.has_he_patch(slide_id)) or (not self.has_st_graph(slide_id)):
                continue
            he_idx = self.get_he_patch_index(slide_id)[["row_index"]].rename(columns={"row_index": "he_row_index"})
            st_idx = self.get_st_graph_index(slide_id)[["row_index"]].rename(columns={"row_index": "st_row_index"})
            common = he_idx.join(st_idx, how="inner")
            if common.empty:
                continue
            common = common.reset_index(drop=False)
            per_slide_common[slide_id] = common
            available_per_slide[slide_id] = int(common.shape[0])

        keep_per_slide = allocate_total_budget_per_slide(available_per_slide, np_total, rng)

        rows = []
        for slide_id in slide_ids:
            n_keep = keep_per_slide.get(slide_id, 0)
            if n_keep <= 0:
                continue
            common = per_slide_common[slide_id]
            sel = sample_indices(common.shape[0], n_keep, rng)
            if sel.size == 0:
                continue
            take = common.iloc[sel].copy()
            meta = self.get_slide_meta_row(slide_id)
            take["slide_id"] = slide_id
            take["source"] = str(meta["source"])
            take["subset"] = str(meta["subset"])
            take["tissue_HistoST"] = str(meta["tissue_HistoST"])
            take = self.attach_patch_domains(slide_id, take)
            rows.append(
                take[
                    ["slide_id", "patch_id", "he_row_index", "st_row_index", "source", "subset", "tissue_HistoST"]
                    + self.patch_domain_cols
                    + self.patch_domain_frac_cols
                ]
            )

        return pd.concat(rows, axis=0, ignore_index=True) if rows else pd.DataFrame(
            columns=["slide_id", "patch_id", "he_row_index", "st_row_index", "source", "subset", "tissue_HistoST"]
            + self.patch_domain_cols
            + self.patch_domain_frac_cols
        )

    def sample_joint_cells(self, slides_df: pd.DataFrame, nc_total: int, rng: np.random.Generator) -> pd.DataFrame:
        slide_ids = slides_df["slide_id"].astype(str).tolist()
        per_slide_common: Dict[str, pd.DataFrame] = {}
        available_per_slide: Dict[str, int] = {}

        for slide_id in slide_ids:
            if (not self.has_he_cell(slide_id)) or (not self.has_st_cell(slide_id)):
                continue
            he_df = (
                self.get_he_main_cell_df(slide_id)[
                    ["cell_id_int", "cell_id", "selected_patch_cell_row_index"]
                ]
                .reset_index(drop=True)
                .rename(columns={"selected_patch_cell_row_index": "he_row_index"})
            )
            st_df = (
                self.get_st_cell_index(slide_id)[
                    ["cell_id_int", "cell_id", "cell_type", "row_index"]
                ]
                .reset_index(drop=True)
                .rename(columns={"row_index": "st_row_index"})
            )
            common = he_df.merge(
                st_df,
                on=["cell_id_int", "cell_id"],
                how="inner",
                validate="one_to_one",
            )
            if common.empty:
                continue
            per_slide_common[slide_id] = common
            available_per_slide[slide_id] = int(common.shape[0])

        keep_per_slide = allocate_total_budget_per_slide(available_per_slide, nc_total, rng)

        rows = []
        for slide_id in slide_ids:
            n_keep = keep_per_slide.get(slide_id, 0)
            if n_keep <= 0:
                continue
            common = per_slide_common[slide_id]
            sel = sample_indices(common.shape[0], n_keep, rng)
            if sel.size == 0:
                continue
            take = common.iloc[sel].copy()
            meta = self.get_slide_meta_row(slide_id)
            take["slide_id"] = slide_id
            take["source"] = str(meta["source"])
            take["subset"] = str(meta["subset"])
            take["tissue_HistoST"] = str(meta["tissue_HistoST"])
            rows.append(
                take[
                    [
                        "slide_id", "cell_id_int", "cell_id", "cell_type",
                        "he_row_index", "st_row_index", "source", "subset", "tissue_HistoST",
                    ]
                ]
            )

        return pd.concat(rows, axis=0, ignore_index=True) if rows else pd.DataFrame(
            columns=[
                "slide_id", "cell_id_int", "cell_id", "cell_type",
                "he_row_index", "st_row_index", "source", "subset", "tissue_HistoST",
            ]
        )

    # ------------------ loaders ------------------

    def load_he_patch_embeddings(self, sample_df: pd.DataFrame, dataset_name: str) -> np.ndarray:
        chunks = []
        for slide_id, sub in sample_df.groupby("slide_id", sort=False):
            arr = read_h5_rows_2d(self.he_h5_path(slide_id), dataset_name, sub["row_index"].to_numpy(dtype=np.int64))
            chunks.append(arr)
        return np.concatenate(chunks, axis=0) if chunks else np.zeros((0, 0), dtype=np.float32)

    def load_he_cell_embeddings(self, sample_df: pd.DataFrame, dataset_name: str) -> np.ndarray:
        chunks = []
        for slide_id, sub in sample_df.groupby("slide_id", sort=False):
            arr = read_h5_rows_2d(
                self.he_h5_path(slide_id),
                dataset_name,
                sub["selected_patch_cell_row_index"].to_numpy(dtype=np.int64),
            )
            chunks.append(arr)
        return np.concatenate(chunks, axis=0) if chunks else np.zeros((0, 0), dtype=np.float32)

    def load_st_graph_embeddings(self, sample_df: pd.DataFrame) -> np.ndarray:
        chunks = []
        for slide_id, sub in sample_df.groupby("slide_id", sort=False):
            arr = read_h5ad_rows(self.st_graph_h5ad_path(slide_id), sub["row_index"].to_numpy(dtype=np.int64))
            chunks.append(arr)
        return np.concatenate(chunks, axis=0) if chunks else np.zeros((0, 0), dtype=np.float32)

    def load_st_cell_embeddings(self, sample_df: pd.DataFrame) -> np.ndarray:
        chunks = []
        for slide_id, sub in sample_df.groupby("slide_id", sort=False):
            arr = read_h5ad_rows(self.st_cell_h5ad_path(slide_id), sub["row_index"].to_numpy(dtype=np.int64))
            chunks.append(arr)
        return np.concatenate(chunks, axis=0) if chunks else np.zeros((0, 0), dtype=np.float32)

    def load_joint_patch_embeddings(self, sample_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        he_chunks = []
        st_chunks = []
        for slide_id, sub in sample_df.groupby("slide_id", sort=False):
            he_arr = read_h5_rows_2d(
                self.he_h5_path(slide_id),
                "patch_emb_he_clip",
                sub["he_row_index"].to_numpy(dtype=np.int64),
            )
            st_arr = read_h5ad_rows(
                self.st_graph_h5ad_path(slide_id),
                sub["st_row_index"].to_numpy(dtype=np.int64),
            )
            he_chunks.append(he_arr)
            st_chunks.append(st_arr)
        he = np.concatenate(he_chunks, axis=0) if he_chunks else np.zeros((0, 0), dtype=np.float32)
        st = np.concatenate(st_chunks, axis=0) if st_chunks else np.zeros((0, 0), dtype=np.float32)
        return he, st

    def load_joint_cell_embeddings(self, sample_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        he_chunks = []
        st_chunks = []
        for slide_id, sub in sample_df.groupby("slide_id", sort=False):
            he_arr = read_h5_rows_2d(
                self.he_h5_path(slide_id),
                "patch_cell_emb_he_clip",
                sub["he_row_index"].to_numpy(dtype=np.int64),
            )
            st_arr = read_h5ad_rows(
                self.st_cell_h5ad_path(slide_id),
                sub["st_row_index"].to_numpy(dtype=np.int64),
            )
            he_chunks.append(he_arr)
            st_chunks.append(st_arr)
        he = np.concatenate(he_chunks, axis=0) if he_chunks else np.zeros((0, 0), dtype=np.float32)
        st = np.concatenate(st_chunks, axis=0) if st_chunks else np.zeros((0, 0), dtype=np.float32)
        return he, st


# ---------------------------------------------------------------------
# UMAP / plotting
# ---------------------------------------------------------------------


def compute_umap_coords(
    emb: np.ndarray,
    seed: int,
    n_neighbors: int,
    min_dist: float,
    spread: float,
    metric: str,
    pca_dim: int,
    l2_norm: bool,
) -> Tuple[np.ndarray, Dict]:
    emb = np.asarray(emb, dtype=np.float32)
    n, d = emb.shape

    if n < 3:
        raise ValueError(f"Need at least 3 points for UMAP, got {n}.")

    x = emb
    if l2_norm:
        x = l2_normalize_rows(x)

    pca_used = 0
    if pca_dim > 0 and d > pca_dim and n > 3:
        pca_used = min(int(pca_dim), int(d), int(n - 1))
        x = PCA(
            n_components=pca_used,
            svd_solver="randomized",
            random_state=seed,
        ).fit_transform(x)

    nn = min(int(n_neighbors), max(2, n - 1))
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=nn,
        min_dist=float(min_dist),
        spread=float(spread),
        metric=str(metric),
        random_state=int(seed),
        low_memory=True,
        transform_seed=int(seed),
        init="spectral",
        verbose=False,
    )
    coords = reducer.fit_transform(x).astype(np.float32, copy=False)

    info = {
        "n_points": int(n),
        "input_dim": int(d),
        "pca_dim_used": int(pca_used),
        "umap_n_neighbors_used": int(nn),
        "metric": str(metric),
        "min_dist": float(min_dist),
        "spread": float(spread),
        "l2_norm": bool(l2_norm),
        "seed": int(seed),
    }
    return coords, info



def categorical_order(series: pd.Series, name: str) -> List[str]:
    vals = series.fillna("NA").astype(str)
    uniq = sorted(vals.unique().tolist())
    if name == "modality":
        pref = ["HE", "ST"]
        extra = [x for x in uniq if x not in pref]
        return [x for x in pref if x in uniq] + extra
    if name == "source":
        pref = ["HEST", "STHELAR"]
        extra = [x for x in uniq if x not in pref]
        return [x for x in pref if x in uniq] + extra
    return uniq



def build_palette(categories: Sequence[str]) -> Dict[str, Tuple[float, float, float, float]]:
    n = len(categories)
    if n <= 10:
        cmap = plt.get_cmap("tab10")
    elif n <= 20:
        cmap = plt.get_cmap("tab20")
    else:
        cmap = plt.get_cmap("gist_ncar")
    return {cat: cmap(i / max(1, n - 1)) for i, cat in enumerate(categories)}



def plot_one_categorical_umap(
    ax,
    coords: np.ndarray,
    meta_df: pd.DataFrame,
    color_col: str,
    title: str,
    show_legend: bool = True,
    plot_seed: int = 0,
) -> None:
    vals = meta_df[color_col].fillna("NA").astype(str)
    vals_np = vals.to_numpy()
    cats = categorical_order(vals, color_col)
    palette = build_palette(cats)

    n_points = coords.shape[0]
    if len(vals_np) != n_points:
        raise ValueError(
            f"coords/meta length mismatch for color_col='{color_col}': "
            f"coords={n_points}, meta={len(vals_np)}"
        )

    s = pick_point_size(coords.shape[0])
    base_alpha = pick_alpha(coords.shape[0])

    # One RGBA color per point (all points drawn in one scatter call, in shuffled order)
    rgba = np.zeros((n_points, 4), dtype=np.float32)

    for cat in cats:
        mask = vals_np == cat
        if not np.any(mask):
            continue

        rgba_cat = np.asarray(palette[cat], dtype=np.float32)
        rgba[mask, :3] = rgba_cat[:3]
        rgba[mask, 3] = base_alpha

    # Joint H&E/ST plots: ST points more transparent in every panel
    joint_st_alpha = 0.47
    if "modality" in meta_df.columns:
        modality = meta_df["modality"].fillna("NA").astype(str).to_numpy()
        is_joint_he_st = np.any(modality == "HE") and np.any(modality == "ST")
        if is_joint_he_st:
            joint_st_alpha = float(np.clip(joint_st_alpha, 0.0, 1.0))
            rgba[modality == "ST", 3] = joint_st_alpha

    # Random draw order (fixed seed)
    if n_points > 1:
        rng = np.random.default_rng(int(plot_seed))
        draw_order = rng.permutation(n_points)
    else:
        draw_order = np.arange(n_points, dtype=np.int64)

    ax.scatter(
        coords[draw_order, 0],
        coords[draw_order, 1],
        s=s,
        c=rgba[draw_order],
        edgecolors="none",
        rasterized=True,
    )

    ax.set_title(title, fontsize=11)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", adjustable="datalim")

    if show_legend:
        legend_cats = cats
        if color_col == "slide_id" and len(cats) > 30 and "source" in meta_df.columns:
            sthelar_mask = meta_df["source"].fillna("NA").astype(str).to_numpy() == "STHELAR"
            legend_cats = [
                cat for cat in cats
                if np.any((vals_np == cat) & sthelar_mask)
            ]

        legend_handles = []
        for cat in legend_cats:
            if cat not in palette:
                continue

            legend_alpha = base_alpha
            if color_col == "modality" and cat == "ST" and joint_st_alpha is not None:
                legend_alpha = joint_st_alpha

            color = tuple(palette[cat][:3]) + (legend_alpha,)
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    linestyle="none",
                    markerfacecolor=color,
                    markeredgecolor="none",
                    markersize=6,
                    label=cat,
                )
            )

        if legend_handles:
            ax.legend(
                legend_handles,
                legend_cats,
                loc="upper left",
                bbox_to_anchor=(1.02, 1.0),
                borderaxespad=0.0,
                frameon=False,
                fontsize=8,
                markerscale=2.0,
                ncol=(1 if len(legend_cats) <= 18 else 2),
            )



def plot_umap_panel(
    coords: np.ndarray,
    meta_df: pd.DataFrame,
    color_cols: Sequence[str],
    title_prefix: str,
    out_png: Path,
    out_pdf: Optional[Path] = None,
    plot_seed: int = 0,
) -> None:
    n_panels = len(color_cols)
    ncols = 2 if n_panels <= 4 else 3
    nrows = int(math.ceil(n_panels / ncols))

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(7.0 * ncols, 6.0 * nrows),
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes).ravel()

    for i, color_col in enumerate(color_cols):
        plot_one_categorical_umap(
            ax=axes[i],
            coords=coords,
            meta_df=meta_df,
            color_col=color_col,
            title=f"{title_prefix}\ncolored by {color_col}",
            show_legend=True,
            plot_seed=int(plot_seed) + 1009 * i,
        )

    for j in range(n_panels, len(axes)):
        axes[j].axis("off")

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=250, bbox_inches="tight")
    if out_pdf is not None:
        fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)



def run_single_umap_figure(
    emb: np.ndarray,
    meta_df: pd.DataFrame,
    color_cols: Sequence[str],
    title_prefix: str,
    stem: str,
    out_dir: Path,
    args,
    keep_coords_parquet: bool,
    population: str = POPULATION_HE_ONLY,
) -> Dict:
    if emb.shape[0] == 0:
        raise ValueError("No points.")
    if emb.shape[0] < 3:
        raise ValueError(f"Need at least 3 points, got {emb.shape[0]}.")

    # Secondary cell-type axes as extra colouring panels (columns added by attach_celltype_axis_columns)
    meta_df = meta_df.copy()
    color_cols = list(color_cols)
    celltype_axes = getattr(args, "_resolved_celltype_axes", {}) or {}
    if "cell_type" in meta_df.columns:
        for axis, mapping in celltype_axes.items():
            if mapping is None:
                continue
            col = f"cell_type_{axis}"
            if col in meta_df.columns and col not in color_cols:
                color_cols.insert(color_cols.index("cell_type") + 1, col)

    # Structure metrics on the raw embedding matrix (not the UMAP coordinates); only `modality` for joint families
    candidate_label_cols = (
        JOINT_LABEL_COLUMNS if population == POPULATION_JOINT else
        ("slide_id", "tissue_HistoST", "modality", "source", "subset", "cell_type",
         *[f"cell_type_{a}" for a, m in celltype_axes.items() if m is not None])
    )
    metric_label_cols = [c for c in candidate_label_cols if c in meta_df.columns]
    structure_metrics = compute_umap_family_metrics(
        emb=emb,
        meta_df=meta_df,
        label_cols=metric_label_cols,
        group_col="slide_id",
        k=int(getattr(args, "metrics_knn_k", 15)),
        min_points_per_label=int(getattr(args, "min_points_per_label", 30)),
        seed=int(args.seed),
        population=population,
    )

    coords, info = compute_umap_coords(
        emb=emb,
        seed=args.seed,
        n_neighbors=args.umap_n_neighbors,
        min_dist=args.umap_min_dist,
        spread=args.umap_spread,
        metric=args.umap_metric,
        pca_dim=args.pca_dim,
        l2_norm=(not args.no_l2_norm),
    )

    out_png = out_dir / f"{stem}.png"
    out_pdf = out_dir / f"{stem}.pdf" if args.save_pdf else None
    plot_umap_panel(
        coords=coords,
        meta_df=meta_df,
        color_cols=color_cols,
        title_prefix=title_prefix,
        out_png=out_png,
        out_pdf=out_pdf,
        plot_seed=args.seed,
    )

    coords_df = meta_df.copy()
    coords_df["umap_1"] = coords[:, 0]
    coords_df["umap_2"] = coords[:, 1]

    coords_parquet_path = out_dir / f"{stem}_coords.parquet"
    if keep_coords_parquet:
        save_dataframe_parquet(coords_df, coords_parquet_path)

    return {
        "figure_png": str(out_png),
        "figure_pdf": str(out_pdf) if out_pdf is not None else None,
        "n_points": int(emb.shape[0]),
        "embedding_dim": int(emb.shape[1]),
        "umap": info,
        "color_columns": list(color_cols),
        "structure_metrics": structure_metrics,
        **(
            {"coords_parquet": str(coords_parquet_path)}
            if keep_coords_parquet else {}
        ),
    }


# ---------------------------------------------------------------------
# Main per-config generation
# ---------------------------------------------------------------------


def normalize_metadata_df(df: pd.DataFrame) -> pd.DataFrame:

    out = df.copy()
    out["slide_id"] = out["slide_id"].astype(str)
    out["source"] = out["source"].astype(str)
    out["subset"] = out["subset"].astype(str)
    out["tissue_HistoST"] = out["tissue_HistoST"].astype(str)
    out = out[["slide_id", "source", "subset", "tissue_HistoST"]].drop_duplicates().reset_index(drop=True)

    if out["slide_id"].duplicated().any():
        dup = out.loc[out["slide_id"].duplicated(), "slide_id"].head(10).tolist()
        raise ValueError(f"Duplicate slide_id rows in metadata. Examples: {dup}")

    return out



def write_report(report: Dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(report, f, indent=2)


def default_umap_stem_by_key() -> Dict[str, str]:
    """
    Canonical stems used by the standard per-config outputs.
    """
    return {
        "umap_he_patch_clip": "umap_he_patch_clip",
        "umap_he_patch_raw": "umap_he_patch_raw",
        "umap_st_graph_clip": "umap_st_graph_clip",
        "umap_joint_patch_graph_clip": "umap_joint_patch_graph_clip",
        "umap_he_cell_clip": "umap_he_cell_clip",
        "umap_he_cell_raw": "umap_he_cell_raw",
        "umap_st_cell_clip": "umap_st_cell_clip",
        "umap_joint_cell_clip": "umap_joint_cell_clip",
    }


# RNG stream ordinal of each UMAP family (one independent stream per family)
FAMILY_RNG_ORDINAL: Dict[str, int] = {
    "he_patch": 0,
    "st_graph": 1,
    "joint_patch": 2,
    "he_cell": 3,
    "st_cell": 4,
    "joint_cell": 5,
}


def attach_celltype_axis_columns(
    df: pd.DataFrame, celltype_axes: Dict[str, Optional[Dict[str, str]]]
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Add one `cell_type_{axis}` column per non-`fine` axis (raises on an unmapped label). Returns `(df, added_column_names)`.
    A frame without a `cell_type` column is returned unchanged.
    """
    if df.empty or "cell_type" not in df.columns:
        return df, []
    out = df.copy()
    added: List[str] = []
    for axis, mapping in celltype_axes.items():
        if mapping is None:            # `fine`: already in `cell_type`
            continue
        col = f"cell_type_{axis}"
        out[col] = apply_label_map(out["cell_type"], mapping)
        added.append(col)
    return out, added


def generate_group_outputs(
    repo: EvalEmbeddingRepository,
    slides_df: pd.DataFrame,
    group_name: str,
    group_title: str,
    allow_cells: bool,
    allow_st: bool,
    out_dir: Path,
    args,
    rng_master: np.random.Generator,
    stem_by_key: Optional[Dict[str, str]] = None,
    sample_prefix: str = "",
    report_filename: str = "report.json",
) -> None:
    """Generate the UMAP outputs for one group of slides (a config folder, or one slide in by_slides_STHELAR)."""
    canonical_stems = default_umap_stem_by_key()
    stem_by_key = canonical_stems if stem_by_key is None else dict(stem_by_key)
    # Cell-type axes resolved in main
    celltype_axes = getattr(args, "_resolved_celltype_axes", {}) or {}

    missing_stem_keys = sorted(set(canonical_stems).difference(stem_by_key))
    if missing_stem_keys:
        raise KeyError(f"Missing UMAP stem mappings for keys: {missing_stem_keys}")

    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.keep_intermediate_parquet:
        stale_parquet_paths = [
            out_dir / f"{sample_prefix}sample_he_patch.parquet",
            out_dir / f"{sample_prefix}sample_st_graph.parquet",
            out_dir / f"{sample_prefix}sample_joint_patch_graph.parquet",
            out_dir / f"{sample_prefix}sample_he_cell.parquet",
            out_dir / f"{sample_prefix}sample_st_cell.parquet",
            out_dir / f"{sample_prefix}sample_joint_cell.parquet",
        ]
        stale_parquet_paths.extend(
            out_dir / f"{stem_by_key[k]}_coords.parquet"
            for k in sorted(canonical_stems.keys())
        )
        for p in stale_parquet_paths:
            delete_file_if_exists(p)

    print(f"  {group_title}")

    report = {
        "config_name": group_name,
        "config_title": group_title,
        "slides_requested": slides_df["slide_id"].astype(str).tolist(),
        "n_slides_requested": int(slides_df.shape[0]),
        "novae_training_id": repo.training_id_novae,
        "keep_intermediate_parquet": bool(args.keep_intermediate_parquet),
        "patch_domain_columns": list(repo.patch_domain_cols),
        "generated": {},
        "skipped": {},
        "sampling": {
            "budget_scope": "overall_per_umap_family",
            "allocation_strategy": "max_min_fair_split_across_eligible_slides_with_capacity_redistribution",
            "seed": int(args.seed),
        },
    }

    # fresh child RNG per group for reproducibility + independence
    cfg_seed = int(rng_master.integers(0, 2**31 - 1))
    report["sampling"]["config_seed"] = cfg_seed

    # One independent RNG stream per family
    def family_rng(family: str) -> np.random.Generator:
        return np.random.default_rng([cfg_seed, FAMILY_RNG_ORDINAL[family]])

    # ---------------- patch-level H&E ----------------

    he_patch_df = repo.sample_he_patches(slides_df, args.Np, family_rng("he_patch"))
    save_intermediate_dataframe_parquet(
        he_patch_df,
        out_dir / f"{sample_prefix}sample_he_patch.parquet",
        keep_on_disk=args.keep_intermediate_parquet,
    )

    if he_patch_df.shape[0] >= 3:
        try:
            emb = repo.load_he_patch_embeddings(he_patch_df, "patch_emb_he_clip")
            meta_cols = filter_available_color_cols(
                he_patch_df,
                ["slide_id", "tissue_HistoST", "source", "subset"] + repo.patch_domain_cols,
            )
            report["generated"]["umap_he_patch_clip"] = run_single_umap_figure(
                emb=emb,
                meta_df=he_patch_df[meta_cols].copy(),
                color_cols=meta_cols,
                title_prefix=f"{group_title}\nH&E patches WITH CLIP projection (n={emb.shape[0]})",
                stem=stem_by_key["umap_he_patch_clip"],
                out_dir=out_dir,
                keep_coords_parquet=args.keep_intermediate_parquet,
                args=args,
                population=POPULATION_HE_ONLY,
            )
        except Exception as e:
            report["skipped"]["umap_he_patch_clip"] = str(e)

        try:
            emb = repo.load_he_patch_embeddings(he_patch_df, "patch_emb_he")
            meta_cols = filter_available_color_cols(
                he_patch_df,
                ["slide_id", "tissue_HistoST", "source", "subset"] + repo.patch_domain_cols,
            )
            report["generated"]["umap_he_patch_raw"] = run_single_umap_figure(
                emb=emb,
                meta_df=he_patch_df[meta_cols].copy(),
                color_cols=meta_cols,
                title_prefix=f"{group_title}\nH&E patches WITHOUT CLIP projection (n={emb.shape[0]})",
                stem=stem_by_key["umap_he_patch_raw"],
                out_dir=out_dir,
                keep_coords_parquet=args.keep_intermediate_parquet,
                args=args,
                population=POPULATION_HE_ONLY,
            )
        except Exception as e:
            report["skipped"]["umap_he_patch_raw"] = str(e)
    else:
        report["skipped"]["umap_he_patch_clip"] = "Fewer than 3 sampled H&E patches."
        report["skipped"]["umap_he_patch_raw"] = "Fewer than 3 sampled H&E patches."

    # ---------------- patch-level ST graph ----------------

    if not allow_st:
        report["skipped"]["umap_st_graph_clip"] = (
            "At least one source of this configuration has no ST embeddings."
        )
    else:
        st_graph_df = repo.sample_st_graphs(slides_df, args.Np, family_rng("st_graph"))
        save_intermediate_dataframe_parquet(
            st_graph_df,
            out_dir / f"{sample_prefix}sample_st_graph.parquet",
            keep_on_disk=args.keep_intermediate_parquet,
        )

        if st_graph_df.shape[0] >= 3:
            try:
                emb = repo.load_st_graph_embeddings(st_graph_df)
                meta_cols = filter_available_color_cols(
                    st_graph_df,
                    ["slide_id", "tissue_HistoST", "source", "subset"] + repo.patch_domain_cols,
                )
                report["generated"]["umap_st_graph_clip"] = run_single_umap_figure(
                    emb=emb,
                    meta_df=st_graph_df[meta_cols].copy(),
                    color_cols=meta_cols,
                    title_prefix=f"{group_title}\nST graphs WITH CLIP projection (n={emb.shape[0]})",
                    stem=stem_by_key["umap_st_graph_clip"],
                    out_dir=out_dir,
                    keep_coords_parquet=args.keep_intermediate_parquet,
                    args=args,
                    population=POPULATION_ST_ONLY,
                )
            except Exception as e:
                report["skipped"]["umap_st_graph_clip"] = str(e)
        else:
            report["skipped"]["umap_st_graph_clip"] = (
                "Fewer than 3 sampled ST graph embeddings. "
                "This can happen if graph embeddings were not extracted for this configuration."
            )

    # ---------------- joint patch / graph ----------------

    if not allow_st:
        report["skipped"]["umap_joint_patch_graph_clip"] = (
            "At least one source of this configuration has no ST embeddings."
        )
    else:
        joint_patch_df = repo.sample_joint_patches(slides_df, args.Np, family_rng("joint_patch"))
        save_intermediate_dataframe_parquet(
            joint_patch_df,
            out_dir / f"{sample_prefix}sample_joint_patch_graph.parquet",
            keep_on_disk=args.keep_intermediate_parquet,
        )

        if joint_patch_df.shape[0] >= 3:
            try:
                he_clip, st_clip = repo.load_joint_patch_embeddings(joint_patch_df)
                if he_clip.shape[1] != st_clip.shape[1]:
                    raise ValueError(
                        f"Joint patch/graph dims mismatch: H&E={he_clip.shape[1]} vs ST={st_clip.shape[1]}"
                    )

                base_patch_cols = ["slide_id", "tissue_HistoST", "source", "subset"] + repo.patch_domain_cols
                base_patch_cols = filter_available_color_cols(joint_patch_df, base_patch_cols)

                joint_meta_he = joint_patch_df[base_patch_cols].copy()
                joint_meta_he["modality"] = "HE"
                joint_meta_st = joint_patch_df[base_patch_cols].copy()
                joint_meta_st["modality"] = "ST"

                joint_meta = pd.concat([joint_meta_he, joint_meta_st], axis=0, ignore_index=True)
                emb_joint = np.concatenate([he_clip, st_clip], axis=0)

                joint_color_cols = filter_available_color_cols(
                    joint_meta,
                    ["modality", "slide_id", "tissue_HistoST", "source", "subset"] + repo.patch_domain_cols,
                )

                report["generated"]["umap_joint_patch_graph_clip"] = run_single_umap_figure(
                    emb=emb_joint,
                    meta_df=joint_meta[joint_color_cols].copy(),
                    color_cols=joint_color_cols,
                    title_prefix=(
                        f"{group_title}\n"
                        f"Joint H&E patches + ST graphs WITH CLIP projection (matched IDs, n={emb_joint.shape[0]})"
                    ),
                    stem=stem_by_key["umap_joint_patch_graph_clip"],
                    out_dir=out_dir,
                    keep_coords_parquet=args.keep_intermediate_parquet,
                    args=args,
                    population=POPULATION_JOINT,
                )
            except Exception as e:
                report["skipped"]["umap_joint_patch_graph_clip"] = str(e)
        else:
            report["skipped"]["umap_joint_patch_graph_clip"] = (
                "Fewer than 3 matched patch IDs with both H&E patch and ST graph embeddings."
            )

    # ---------------- cell-level plots ----------------

    if not allow_cells:
        report["skipped"]["umap_he_cell_clip"] = "At least one source of this configuration has no cell information."
        report["skipped"]["umap_he_cell_raw"] = "At least one source of this configuration has no cell information."
        report["skipped"]["umap_st_cell_clip"] = "At least one source of this configuration has no cell information."
        report["skipped"]["umap_joint_cell_clip"] = "At least one source of this configuration has no cell information."
    else:
        # H&E cells
        he_cell_df = repo.sample_he_cells(slides_df, args.Nc, family_rng("he_cell"))
        # Secondary cell-type axis columns (added outside the try blocks below, so an unmapped label raises)
        he_cell_df, he_cell_axis_cols = attach_celltype_axis_columns(he_cell_df, celltype_axes)
        he_cell_meta_cols = ["slide_id", "tissue_HistoST", "cell_type", *he_cell_axis_cols,
                             "source", "subset"]
        save_intermediate_dataframe_parquet(
            he_cell_df,
            out_dir / f"{sample_prefix}sample_he_cell.parquet",
            keep_on_disk=args.keep_intermediate_parquet,
        )

        if he_cell_df.shape[0] >= 3:
            try:
                emb = repo.load_he_cell_embeddings(he_cell_df, "patch_cell_emb_he_clip")
                report["generated"]["umap_he_cell_clip"] = run_single_umap_figure(
                    emb=emb,
                    meta_df=he_cell_df[he_cell_meta_cols].copy(),
                    color_cols=list(he_cell_meta_cols),
                    title_prefix=f"{group_title}\nH&E cells WITH CLIP projection (n={emb.shape[0]})",
                    stem=stem_by_key["umap_he_cell_clip"],
                    out_dir=out_dir,
                    keep_coords_parquet=args.keep_intermediate_parquet,
                    args=args,
                    population=POPULATION_HE_ONLY,
                )
            except Exception as e:
                report["skipped"]["umap_he_cell_clip"] = str(e)

            try:
                emb = repo.load_he_cell_embeddings(he_cell_df, "patch_cell_emb_he")
                report["generated"]["umap_he_cell_raw"] = run_single_umap_figure(
                    emb=emb,
                    meta_df=he_cell_df[he_cell_meta_cols].copy(),
                    color_cols=list(he_cell_meta_cols),
                    title_prefix=f"{group_title}\nH&E cells WITHOUT CLIP projection (n={emb.shape[0]})",
                    stem=stem_by_key["umap_he_cell_raw"],
                    out_dir=out_dir,
                    keep_coords_parquet=args.keep_intermediate_parquet,
                    args=args,
                    population=POPULATION_HE_ONLY,
                )
            except Exception as e:
                report["skipped"]["umap_he_cell_raw"] = str(e)
        else:
            report["skipped"]["umap_he_cell_clip"] = "Fewer than 3 sampled H&E cells."
            report["skipped"]["umap_he_cell_raw"] = "Fewer than 3 sampled H&E cells."

        # ST cells
        if not allow_st:
            report["skipped"]["umap_st_cell_clip"] = (
                "At least one source of this configuration has no ST embeddings."
            )
            report["skipped"]["umap_joint_cell_clip"] = (
                "At least one source of this configuration has no ST embeddings."
            )
        else:
            st_cell_df = repo.sample_st_cells(slides_df, args.Nc, family_rng("st_cell"))
            st_cell_df, st_cell_axis_cols = attach_celltype_axis_columns(st_cell_df, celltype_axes)
            st_cell_meta_cols = ["slide_id", "tissue_HistoST", "cell_type", *st_cell_axis_cols,
                                 "source", "subset"]
            save_intermediate_dataframe_parquet(
                st_cell_df,
                out_dir / f"{sample_prefix}sample_st_cell.parquet",
                keep_on_disk=args.keep_intermediate_parquet,
            )

            if st_cell_df.shape[0] >= 3:
                try:
                    emb = repo.load_st_cell_embeddings(st_cell_df)
                    report["generated"]["umap_st_cell_clip"] = run_single_umap_figure(
                        emb=emb,
                        meta_df=st_cell_df[st_cell_meta_cols].copy(),
                        color_cols=list(st_cell_meta_cols),
                        title_prefix=f"{group_title}\nST cells WITH CLIP projection (n={emb.shape[0]})",
                        stem=stem_by_key["umap_st_cell_clip"],
                        out_dir=out_dir,
                        keep_coords_parquet=args.keep_intermediate_parquet,
                        args=args,
                        population=POPULATION_ST_ONLY,
                    )
                except Exception as e:
                    report["skipped"]["umap_st_cell_clip"] = str(e)
            else:
                report["skipped"]["umap_st_cell_clip"] = "Fewer than 3 sampled ST cells."

            # joint cells
            joint_cell_df = repo.sample_joint_cells(slides_df, args.Nc, family_rng("joint_cell"))
            joint_cell_df, joint_cell_axis_cols = attach_celltype_axis_columns(
                joint_cell_df, celltype_axes
            )
            joint_cell_base_cols = ["slide_id", "tissue_HistoST", "cell_type",
                                    *joint_cell_axis_cols, "source", "subset"]
            save_intermediate_dataframe_parquet(
                joint_cell_df,
                out_dir / f"{sample_prefix}sample_joint_cell.parquet",
                keep_on_disk=args.keep_intermediate_parquet,
            )

            if joint_cell_df.shape[0] >= 3:
                try:
                    he_clip, st_clip = repo.load_joint_cell_embeddings(joint_cell_df)
                    if he_clip.shape[1] != st_clip.shape[1]:
                        raise ValueError(
                            f"Joint cell dims mismatch: H&E={he_clip.shape[1]} vs ST={st_clip.shape[1]}"
                        )

                    joint_meta_he = joint_cell_df[joint_cell_base_cols].copy()
                    joint_meta_he["modality"] = "HE"
                    joint_meta_st = joint_cell_df[joint_cell_base_cols].copy()
                    joint_meta_st["modality"] = "ST"

                    joint_meta = pd.concat([joint_meta_he, joint_meta_st], axis=0, ignore_index=True)
                    emb_joint = np.concatenate([he_clip, st_clip], axis=0)

                    joint_cell_meta_cols = ["modality", *joint_cell_base_cols]
                    report["generated"]["umap_joint_cell_clip"] = run_single_umap_figure(
                        emb=emb_joint,
                        meta_df=joint_meta[joint_cell_meta_cols].copy(),
                        color_cols=list(joint_cell_meta_cols),
                        title_prefix=(
                            f"{group_title}\n"
                            f"Joint H&E cells + ST cells WITH CLIP projection (matched IDs, n={emb_joint.shape[0]})"
                        ),
                        stem=stem_by_key["umap_joint_cell_clip"],
                        out_dir=out_dir,
                        keep_coords_parquet=args.keep_intermediate_parquet,
                        args=args,
                        population=POPULATION_JOINT,
                    )
                except Exception as e:
                    report["skipped"]["umap_joint_cell_clip"] = str(e)
            else:
                report["skipped"]["umap_joint_cell_clip"] = (
                    "Fewer than 3 matched cell IDs with both H&E cell and ST cell embeddings."
                )

    # Long-format structure metrics (umap_metrics.csv)
    metric_rows = []
    for family, entry in report.get("generated", {}).items():
        sm = entry.get("structure_metrics") if isinstance(entry, dict) else None
        if sm:
            metric_rows.extend(
                structure_metrics_to_rows(sm, config_name=report.get("config_name", ""), family=family)
            )
    if metric_rows:
        # sample_prefix: one table per slide in by_slides_STHELAR/
        metrics_csv_path = out_dir / f"{sample_prefix}umap_metrics.csv"
        pd.DataFrame(metric_rows).to_csv(metrics_csv_path, index=False)
        print(f"  structure metrics -> {metrics_csv_path}")
    else:
        # No metrics: remove any existing umap_metrics.csv
        delete_file_if_exists(out_dir / f"{sample_prefix}umap_metrics.csv")

    write_report(report, out_dir / report_filename)
    print(f"  done -> {out_dir / report_filename}\n")

    # Raise (after writing the report) if no label metric is resolvable (single-slide configs exempt)
    if metric_rows:
        assert_any_metric_resolvable(
            metric_rows,
            config_name=str(report.get("config_name", "")),
            n_slides=int(slides_df["slide_id"].astype(str).nunique()),
        )


def main(args) -> None:

    # Eval root
    eval_root = get_full_path("eval", "eval_datasets", "st_eval_data", f"eval_st_{args.training_dataset_id}")
    
    # Metadata csv
    metadata_csv = eval_root / f"metadata_eval_st_{args.training_dataset_id}.csv"
    if not metadata_csv.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {metadata_csv}")

    metadata_raw = pd.read_csv(metadata_csv)
    metadata = normalize_metadata_df(metadata_raw)

    # Out root
    out_root = eval_root / "embeddings_analysis" / "umap_embeddings" / f"t{args.training_num}e{args.checkpoint_num}"
    out_root.mkdir(parents=True, exist_ok=True)

    # Resolve the cell-type axes once (stored on args)
    args._resolved_celltype_axes = build_axes(
        args.ct_cat_id, args.celltype_axes, args.coarse_mapping_version
    )

    # Refuse to write into a tree produced by a different analysis version.
    version_path = out_root / "a0_version.json"
    if version_path.exists():
        try:
            prev_version = int(json.loads(version_path.read_text()).get("a0_analysis_version", 1))
        except (json.JSONDecodeError, TypeError, ValueError):
            prev_version = 1
        if prev_version != A0_ANALYSIS_VERSION and not args.overwrite:
            raise RuntimeError(
                f"{out_root} was written by A0_ANALYSIS_VERSION={prev_version}, but this is "
                f"version {A0_ANALYSIS_VERSION}. Re-run into a clean directory, or pass "
                "--overwrite to replace it."
            )
    version_path.write_text(json.dumps({
        "a0_analysis_version": int(A0_ANALYSIS_VERSION),
        "ct_cat_id": args.ct_cat_id,
        "celltype_axes": list(args.celltype_axes),
        "coarse_mapping_version": args.coarse_mapping_version,
        "metrics_knn_k": int(args.metrics_knn_k),
        "min_points_per_label": int(args.min_points_per_label),
    }, indent=2, sort_keys=True))

    # Evaluation embedding repository
    repo = EvalEmbeddingRepository(
        eval_root=eval_root,
        metadata=metadata,
        training_dataset_id=str(args.training_dataset_id),
        training_num=str(args.training_num),
        checkpoint_num=str(args.checkpoint_num),
        training_id_novae=args.training_id_novae,
    )

    # Configs per group (i.e. all slides, only one source, only one subset in a given source)
    configs = build_config_specs(metadata)
    rng_master = np.random.default_rng(args.seed)

    print(f"Metadata CSV : {metadata_csv}")
    print(f"Eval root     : {eval_root}")
    print(f"Output root   : {out_root}")
    print(f"Configs       : {len(configs)}")
    print(f"Np total/fam. : {args.Np}")
    print(f"Nc total/fam. : {args.Nc}")
    print(f"UMAP metric   : {args.umap_metric}")
    print(f"L2 norm       : {not args.no_l2_norm}")
    print(f"Keep interm. pq: {args.keep_intermediate_parquet}")
    print(f"Novae domains : {repo.training_id_novae if repo.training_id_novae is not None else 'disabled'}")
    if len(repo.patch_domain_cols) > 0:
        print(f"Domain cols   : {repo.patch_domain_cols}")
    print()

    # Configs with no resolvable metric are collected and raised together at the end
    gate_failures: List[Tuple[str, str]] = []

    # Process each standard config
    for i_cfg, cfg in enumerate(configs, start=1):

        try:
            generate_group_outputs(
                repo=repo,
                slides_df=cfg.slides_df,
                group_name=cfg.name,
                group_title=cfg.title,
                allow_cells=cfg.allow_cells,
                allow_st=cfg.allow_st,
                out_dir=out_root / cfg.name,
                args=args,
                rng_master=rng_master,
            )
        except MetricsUnresolvableError as exc:
            print(f"  [gate] {cfg.name}: {exc}")
            gate_failures.append((cfg.name, str(exc)))

    # Additional flat outputs: one pass per STHELAR slide
    sthelar_slide_specs = build_sthelar_slide_specs(metadata)
    by_slides_out = out_root / "by_slides_STHELAR"
    by_slides_out.mkdir(parents=True, exist_ok=True)

    print(f"STHELAR single-slide outputs: {len(sthelar_slide_specs)}")
    print(f"By-slides root               : {by_slides_out}")
    print()

    for i_slide, cfg in enumerate(sthelar_slide_specs, start=1):
        slide_id = str(cfg.slides_df["slide_id"].iloc[0])
        print(f"[{i_slide}/{len(sthelar_slide_specs)}] by_slides_STHELAR/{slide_id}")

        slide_stem_by_key = {
            "umap_he_patch_clip": f"{slide_id}_umap_patch_clip",
            "umap_he_patch_raw": f"{slide_id}_umap_patch_raw",
            "umap_st_graph_clip": f"{slide_id}_umap_st_graph_clip",
            "umap_joint_patch_graph_clip": f"{slide_id}_umap_joint_patch_graph_clip",
            "umap_he_cell_clip": f"{slide_id}_umap_he_cell_clip",
            "umap_he_cell_raw": f"{slide_id}_umap_he_cell_raw",
            "umap_st_cell_clip": f"{slide_id}_umap_st_cell_clip",
            "umap_joint_cell_clip": f"{slide_id}_umap_joint_cell_clip",
        }

        try:
            generate_group_outputs(
                repo=repo,
                slides_df=cfg.slides_df,
                group_name=cfg.name,
                group_title=cfg.title,
                allow_cells=cfg.allow_cells,
                allow_st=cfg.allow_st,
                out_dir=by_slides_out,
                args=args,
                rng_master=rng_master,
                stem_by_key=slide_stem_by_key,
                sample_prefix=f"{slide_id}_",
                report_filename=f"{slide_id}_report.json",
            )
        except MetricsUnresolvableError as exc:
            print(f"  [gate] {cfg.name}: {exc}")
            gate_failures.append((cfg.name, str(exc)))

    if gate_failures:
        detail = "\n".join(f"  - {name}: {msg}" for name, msg in gate_failures)
        raise MetricsUnresolvableError(
            f"{len(gate_failures)} config(s) produced no resolvable UMAP structure metric. Every "
            f"other config completed and its figures / report.json / umap_metrics.csv are on "
            f"disk.\n{detail}"
        )

    print("Done.")




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UMAP analysis for H&E / ST evaluation embeddings.")

    parser.add_argument("--training_dataset_id", type=str,
                        required=True,
                        help="Training dataset id, e.g. ds_ct1_20x_224_o112_1")
    
    parser.add_argument("--training_num", type=str,
                        required=True,
                        help="Training number, e.g. 46 for training_46")
    
    parser.add_argument("--checkpoint_num", type=str,
                        required=True,
                        help="Checkpoint number, e.g. 13 for 'checkpoint_epoch_13.pth'")

    parser.add_argument("--training_id_novae", type=str,
                        default=None,
                        help="Optional Novae training id used to load per-patch domain labels from "
                             "eval_datasets/st_eval_data/eval_st_{training_dataset_id}/novae_domains_HEpatches/{training_id_novae}/ . "
                             "If omitted, domain coloring is disabled.")

    parser.add_argument("--Np", type=int,
                        default=1500,
                        help="Maximum total number of sampled patches/graphs per UMAP family across eligible slides.")
    
    parser.add_argument("--Nc", type=int,
                        default=3000,
                        help="Maximum total number of sampled cells per UMAP family across eligible slides.")
    
    parser.add_argument("--ct_cat_id", type=str, default="ct_1",
                        help=("Cell-type category id of data/annots/annot_dicts_{ct_cat_id}/ (e.g. ct_1)."))
    parser.add_argument("--celltype_axes", type=str, nargs="+",
                        default=["fine", "coarse"], choices=["fine", "coarse"],
                        help=("Cell-type axes to colour by (the first one is the primary axis, used for the cell_type panel)."))
    parser.add_argument("--coarse_mapping_version", type=str, default="v1",
                        help="Version of the coarse cell-type mapping (required with 'coarse').")
    parser.add_argument("--metrics_knn_k", type=int, default=15,
                        help=("k of the kNN label-purity metrics (computed on the raw embeddings, not the UMAP coordinates)."))
    parser.add_argument("--min_points_per_label", type=int, default=30,
                        help=("Minimum number of points of a label for the kNN purity (fewer than 2 such labels: "
                              "status='insufficient_resolution')."))
    parser.add_argument("--overwrite", action="store_true",
                        help=("Allow writing into a tree produced by a different "
                              "A0_ANALYSIS_VERSION."))
    parser.add_argument("--seed", type=int,
                        default=42,
                        help="Random seed for per-slide sampling and UMAP.")
    
    parser.add_argument("--pca_dim", type=int,
                        default=50,
                        help="PCA dimension before UMAP. Set 0 to disable PCA.")
    
    parser.add_argument("--umap_metric", type=str,
                        default="cosine",
                        choices=["cosine", "euclidean", "manhattan"],
                        help="UMAP metric. 'cosine' is recommended.")
    
    parser.add_argument("--umap_n_neighbors", type=int,
                        default=30,
                        help="UMAP n_neighbors.")
    
    parser.add_argument("--umap_min_dist", type=float,
                        default=0.15,
                        help="UMAP min_dist.")
    
    parser.add_argument("--umap_spread", type=float,
                        default=1.0,
                        help="UMAP spread.")
    
    parser.add_argument("--no_l2_norm", 
                        action="store_true",
                        help="Disable row-wise L2 normalization before PCA/UMAP.")
    
    parser.add_argument("--save_pdf",
                        action="store_true",
                        help="Also save each figure as PDF.")

    parser.set_defaults(keep_intermediate_parquet=True)
    parser.add_argument("--no_keep_intermediate_parquet",
                        dest="keep_intermediate_parquet",
                        action="store_false",
                        help=("Do not save sampled intermediate parquet files on disk"))

    args = parser.parse_args()
    main(args)