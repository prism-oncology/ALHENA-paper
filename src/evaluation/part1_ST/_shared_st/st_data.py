"""Per-slide ST row manifests and H&E embedding loaders (patch and cell level), shared by Task A.1 and Task A.3."""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple
import warnings

import anndata as ad
import h5py
import numpy as np
import pandas as pd


def _task_a1_dir() -> Optional[Path]:
    here = Path(__file__).resolve()
    for parent in here.parents:
        cand = parent / "task_A1_gene_predict"
        if (cand / "src_A1" / "common.py").exists():
            return cand
    return None


_A1_DIR = _task_a1_dir()
if _A1_DIR is not None and str(_A1_DIR) not in sys.path:
    sys.path.insert(0, str(_A1_DIR))

from src_A1.common import (  # noqa: E402
    SlideRecord,
    get_full_path,
    int_cell_id,
    read_metadata,
    resolve_eval_dataset_root,
    resolve_training_dataset_dir,
)


@dataclass
class SlideRowBundle:
    slide_id: str
    row_keys: np.ndarray
    coords_xy: np.ndarray
    patch_ids: Optional[np.ndarray]
    cell_id_int: Optional[np.ndarray]
    cell_types: Optional[np.ndarray]
    obs_df: pd.DataFrame
    row_filter_report: Optional[Dict[str, Any]] = None


def _decode_h5_strings(arr: np.ndarray) -> np.ndarray:
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


def _infer_patch_hw(shape: Tuple[int, ...]) -> Tuple[int, int]:
    if len(shape) != 4:
        raise ValueError(f"Expected patch img shape to be 4D, got {shape}.")
    if shape[-1] == 3:
        return int(shape[1]), int(shape[2])
    if shape[1] == 3:
        return int(shape[2]), int(shape[3])
    raise ValueError(f"Unrecognized patch image tensor shape: {shape}.")


def _load_patch_centers_from_eval_h5(slide_patch_h5: Path, expected_patch_ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    with h5py.File(slide_patch_h5, "r") as f:
        if "coords" not in f or "img" not in f or "barcode" not in f:
            raise KeyError(f"{slide_patch_h5} must contain 'coords', 'img', and 'barcode'.")
        patch_ids_h5 = _decode_h5_strings(f["barcode"][:])
        coords = f["coords"][:].astype(np.float64)
        ph, pw = _infer_patch_hw(f["img"].shape)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"coords in '{slide_patch_h5}' must have shape (N, 2), got {coords.shape}.")
    if len(patch_ids_h5) != len(expected_patch_ids) or not np.array_equal(patch_ids_h5, expected_patch_ids):
        raise ValueError(f"Patch order mismatch between eval patch file '{slide_patch_h5}' and embedding file.")
    center_x = coords[:, 0] + float(pw) / 2.0
    center_y = coords[:, 1] + float(ph) / 2.0
    return center_x, center_y


def _normalize_patch_id(x: object) -> str:
    """Patch id without the _cells / _all / .png suffixes (used only for cross-file matching)."""
    s = str(x)
    if s.endswith("_cells"):
        s = s[:-6]
    elif s.endswith("_all"):
        s = s[:-4]
    if s.endswith(".png"):
        s = s[:-4]
    return s


def load_patch_geometry(eval_dataset_root: Path, slide_id: str) -> pd.DataFrame:
    patch_h5 = eval_dataset_root / "patches" / f"{slide_id}.h5"
    with h5py.File(patch_h5, "r") as f:
        patch_ids = _decode_h5_strings(f["barcode"][:])
        coords = f["coords"][:].astype(np.float64)
        ph, pw = _infer_patch_hw(f["img"].shape)
    if coords.shape[1] != 2:
        raise ValueError(f"coords in '{patch_h5}' must have shape (N,2).")
    x_min = coords[:, 0]
    y_min = coords[:, 1]
    x_max = x_min + pw
    y_max = y_min + ph
    x_center = x_min + pw / 2.0
    y_center = y_min + ph / 2.0
    patch_ids_norm = np.asarray([_normalize_patch_id(pid) for pid in patch_ids], dtype=object)
    return pd.DataFrame(
        {
            "patch_id": patch_ids,
            "patch_id_norm": patch_ids_norm,
            "x_min": x_min,
            "y_min": y_min,
            "x_max": x_max,
            "y_max": y_max,
            "x_center": x_center,
            "y_center": y_center,
            "patch_h": float(ph),
            "patch_w": float(pw),
        }
    )


def _match_patch_obs_rows_to_geometry(
    *,
    obs: pd.DataFrame,
    candidate_mask: np.ndarray,
    geom: pd.DataFrame,
    slide_id: str,
    level: str,
    allow_drop_unmatched: bool,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, Optional[Dict[str, Any]]]:
    """
    Match AnnData obs rows to H&E patch geometry. Unmatched rows (HEST spots without an H&E patch)
    are dropped if allow_drop_unmatched, otherwise raise.

    Returns:
        candidate_pos: positions in obs of the matched rows.
        row_keys: matched obs index values, in protocol order.
        patch_df: geometry rows aligned to row_keys.
        report: count report when rows were dropped, otherwise None.
    """
    candidate_mask = np.asarray(candidate_mask, dtype=bool)
    candidate_pos = np.flatnonzero(candidate_mask)
    row_keys_all = obs.index.to_numpy(dtype=object, copy=False)[candidate_pos]

    patch_df_all = (
        pd.DataFrame(
            {
                "row_key": row_keys_all,
                "patch_id_norm": [_normalize_patch_id(rk) for rk in row_keys_all],
            }
        )
        .merge(
            geom[["patch_id", "patch_id_norm", "x_center", "y_center"]],
            on="patch_id_norm",
            how="left",
            validate="many_to_one",
        )
    )

    unmatched = patch_df_all["patch_id"].isna().to_numpy()
    report: Optional[Dict[str, Any]] = None

    if np.any(unmatched):
        missing_examples = row_keys_all[unmatched][:10].tolist()
        n_unmatched = int(np.sum(unmatched))
        n_candidate = int(row_keys_all.shape[0])

        if not allow_drop_unmatched:
            raise KeyError(
                f"Could not match some {level} rows to patch geometry on slide '{slide_id}'. "
                f"Examples: {missing_examples}"
            )

        warnings.warn(
            f"Slide '{slide_id}' level='{level}': dropping {n_unmatched}/{n_candidate} "
            ".h5ad row(s) because they have no matching H&E patch geometry. "
            f"Examples: {missing_examples}"
        )

        keep = ~unmatched
        candidate_pos = candidate_pos[keep]
        row_keys_all = row_keys_all[keep]
        patch_df_all = patch_df_all.loc[keep].reset_index(drop=True)

        report = {
            "slide_id": str(slide_id),
            "level": str(level),
            "n_candidate_rows_before_geometry_filter": n_candidate,
            "n_rows_dropped_no_patch_geometry": n_unmatched,
            "n_rows_retained_after_geometry_filter": int(row_keys_all.shape[0]),
            "reason": "h5ad_obs_row_has_no_matching_he_patch_geometry",
        }

    if row_keys_all.shape[0] == 0:
        raise RuntimeError(
            f"Slide '{slide_id}' produced zero valid {level} rows after patch-geometry matching."
        )

    return candidate_pos, row_keys_all, patch_df_all, report


def _load_centroids(training_dataset_dir: Path, slide_id: str) -> pd.DataFrame:
    p = training_dataset_dir / "nucleus_centroids" / f"nucleus_centroids_{slide_id}.parquet"
    df = pd.read_parquet(p).copy()
    if not {"x", "y"}.issubset(df.columns):
        raise KeyError(f"Centroid file '{p}' must contain columns 'x' and 'y'.")

    if "cell_id_int" in df.columns:
        out = df[["cell_id_int", "x", "y"]].copy()
        out["cell_id_int"] = out["cell_id_int"].astype(np.int32)
        return out

    if "cell_id" not in df.columns:
        raise KeyError(f"Centroid file '{p}' must contain either 'cell_id_int' or 'cell_id'.")

    df["cell_id"] = df["cell_id"].astype(str)
    if df["cell_id"].str.endswith("-1").all():
        df["cell_id_int"] = df["cell_id"].map(int_cell_id).astype(np.int32)
    else:
        raise ValueError(
            f"Centroid file '{p}' is missing cell_id_int and cell_id does not look like Xenium strings."
        )
    return df[["cell_id_int", "x", "y"]].copy()


def load_missing_cells_for_modelgroup(
    eval_dataset_root: Path,
    modelgroup: str,
    slide_id: str,
    require_exists: bool = True,
) -> np.ndarray:
    """cell_id_int of the cells with no token under the modelgroup geometry (raises if the file is missing and require_exists)."""
    p = eval_dataset_root / f"cell_token_membership_missing_cells_{modelgroup}" / f"{slide_id}.parquet"
    if not p.exists():
        if require_exists:
            raise FileNotFoundError(
                f"Required structural-eligibility artifact is missing for modelgroup='{modelgroup}', "
                f"slide='{slide_id}': {p}"
            )
        return np.zeros((0,), dtype=np.int32)
    df = pd.read_parquet(p)
    if df.empty:
        return np.zeros((0,), dtype=np.int32)
    return df["cell_id_int"].to_numpy(dtype=np.int32, copy=False)


_NULL_DOMAIN_STRINGS = {"", "na", "nan", "none", "null", "<na>"}

_TRUE_BOOL_STRINGS = {"true", "t", "1", "yes", "y"}
_FALSE_BOOL_STRINGS = {"false", "f", "0", "no", "n"}


def _obs_bool_array(s: pd.Series, *, name: str) -> np.ndarray:
    """Parse a boolean-like obs column (bool, 0/1 or strings such as "true"/"false") to a bool array."""
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False).to_numpy(dtype=bool)

    if pd.api.types.is_numeric_dtype(s):
        x = pd.to_numeric(s, errors="coerce")
        bad = x.notna() & ~x.isin([0, 1])
        if bool(bad.any()):
            examples = s.loc[bad].head(10).tolist()
            raise ValueError(
                f"obs[{name!r}] must be boolean-like 0/1, got examples: {examples}"
            )
        return x.fillna(0).to_numpy(dtype=np.float64) != 0.0

    ss = s.astype("string").str.strip().str.lower()
    is_true = ss.isin(_TRUE_BOOL_STRINGS).fillna(False).to_numpy(dtype=bool)
    is_false = (
        ss.isin(_FALSE_BOOL_STRINGS | _NULL_DOMAIN_STRINGS)
        | ss.isna()
    ).fillna(False).to_numpy(dtype=bool)

    bad = ~(is_true | is_false)
    if bool(np.any(bad)):
        examples = s.iloc[np.flatnonzero(bad)[:10]].tolist()
        raise ValueError(
            f"obs[{name!r}] must be boolean-like, got examples: {examples}"
        )

    return is_true


def _normalize_nullable_domain_series(s: pd.Series) -> pd.Series:
    out = s.astype("string")
    bad = out.isna() | out.str.strip().str.lower().isin(_NULL_DOMAIN_STRINGS)
    out = out.mask(bad, pd.NA)
    return out


def load_novae_patch_domains(
    eval_dataset_root: Path,
    novae_training_id: Optional[str],
    slide_id: str,
    novae_domain_n: Optional[int] = None,
) -> Optional[pd.DataFrame]:
    """
    Load the patch-level Novae domain labels of a slide, with columns patch_id, novae_domain and
    (optional) novae_domain_frac. Returns None if novae_training_id is not provided.
    """
    if novae_training_id is None or str(novae_training_id).strip() == "":
        return None

    root = Path(eval_dataset_root) / "novae_domains_HEpatches" / str(novae_training_id)
    p = root / f"{slide_id}.parquet"
    if not p.exists():
        raise FileNotFoundError(f"Novae patch-domain parquet not found for slide '{slide_id}': {p}")

    df = pd.read_parquet(p)
    if "patch_id" not in df.columns:
        raise KeyError(f"Novae parquet '{p}' is missing required column 'patch_id'.")

    # Canonical columns, if already present
    if "novae_domain" in df.columns:
        keep = ["patch_id", "novae_domain"]
        if "novae_domain_frac" in df.columns:
            keep.append("novae_domain_frac")
        out = df[keep].copy()
        out["patch_id"] = out["patch_id"].astype(str)
        out["novae_domain"] = _normalize_nullable_domain_series(out["novae_domain"])
        if "novae_domain_frac" in out.columns:
            out["novae_domain_frac"] = out["novae_domain_frac"].astype(np.float32, copy=False)
        return out

    domain_cols = sorted(
        c
        for c in df.columns
        if c.startswith("novae_patch_domain_") and not c.startswith("novae_patch_domain_frac_")
    )
    if not domain_cols:
        raise KeyError(f"Novae parquet '{p}' has no 'novae_patch_domain_*' column.")

    if novae_domain_n is not None:
        dom_col = f"novae_patch_domain_{int(novae_domain_n)}"
        if dom_col not in df.columns:
            raise KeyError(
                f"Requested Novae domain column '{dom_col}' was not found in '{p}'. "
                f"Available: {domain_cols}"
            )
    else:
        if len(domain_cols) != 1:
            raise ValueError(
                f"Novae parquet '{p}' contains multiple domain columns {domain_cols}. "
                "Set protocol.structure.novae_domain_n explicitly in eval_config.yaml."
            )
        dom_col = domain_cols[0]

    suffix = dom_col.split("novae_patch_domain_", 1)[1]
    frac_col = f"novae_patch_domain_frac_{suffix}"

    keep = ["patch_id", dom_col] + ([frac_col] if frac_col in df.columns else [])
    out = df[keep].copy().rename(
        columns={
            dom_col: "novae_domain",
            frac_col: "novae_domain_frac",
        }
    )
    out["patch_id"] = out["patch_id"].astype(str)
    out["novae_domain"] = _normalize_nullable_domain_series(out["novae_domain"])
    if "novae_domain_frac" in out.columns:
        out["novae_domain_frac"] = out["novae_domain_frac"].astype(np.float32, copy=False)
    return out


def build_row_manifest_for_slide(
    *,
    dataset_id: str,
    training_dataset_id: str,
    level: str,
    modelgroup: str,
    slide_record: SlideRecord,
    novae_training_id: Optional[str] = None,
    novae_domain_n: Optional[int] = None,
) -> SlideRowBundle:
    eval_root = resolve_eval_dataset_root(training_dataset_id)
    st_path = eval_root / "st" / f"{slide_record.slide_id}.h5ad"
    if not st_path.exists():
        raise FileNotFoundError(f"Missing slide h5ad: {st_path}")
    adata = ad.read_h5ad(st_path, backed="r")
    try:
        obs = adata.obs.copy()
        
        if level == "patch_all":
            if slide_record.source == "HEST":
                if "is_cell" in obs.columns:
                    mask = ~_obs_bool_array(obs["is_cell"], name="is_cell")
                else:
                    mask = np.ones(obs.shape[0], dtype=bool)
            else:
                mask = obs.index.to_series().astype(str).str.endswith("_all").to_numpy()
            geom = load_patch_geometry(eval_root, slide_record.slide_id)

            allow_drop_unmatched = slide_record.source == "HEST"
            obs_pos, row_keys, patch_df, row_filter_report = _match_patch_obs_rows_to_geometry(
                obs=obs,
                candidate_mask=mask,
                geom=geom,
                slide_id=slide_record.slide_id,
                level=level,
                allow_drop_unmatched=allow_drop_unmatched,
            )

            coords = patch_df[["x_center", "y_center"]].to_numpy(dtype=np.float64)
            out_obs = obs.iloc[obs_pos].copy()
            out_obs["x"] = coords[:, 0]
            out_obs["y"] = coords[:, 1]
            out_obs["patch_id"] = patch_df["patch_id"].to_numpy(dtype=object)
            novae_df = None
            if slide_record.source == "STHELAR":
                novae_df = load_novae_patch_domains(
                    eval_dataset_root=eval_root,
                    novae_training_id=novae_training_id,
                    slide_id=slide_record.slide_id,
                    novae_domain_n=novae_domain_n,
                )
            if novae_df is not None:
                patch_meta = pd.DataFrame({"patch_id": patch_df["patch_id"].to_numpy(dtype=object)})
                patch_meta = patch_meta.merge(novae_df, on="patch_id", how="left", validate="many_to_one")
                # Patches with no Novae domain keep a null label; raise only if all labels are null
                if bool(patch_meta["novae_domain"].isna().all()):
                    raise ValueError(
                        f"All Novae domain labels are null for patch_all rows on slide "
                        f"'{slide_record.slide_id}' (patch_id mismatch / labels not attached?)."
                    )
                out_obs["novae_domain"] = patch_meta["novae_domain"].to_numpy(dtype=object)
                if "novae_domain_frac" in patch_meta.columns:
                    out_obs["novae_domain_frac"] = patch_meta["novae_domain_frac"].to_numpy(dtype=np.float32, copy=False)
            return SlideRowBundle(
                slide_id=slide_record.slide_id,
                row_keys=row_keys,
                coords_xy=coords,
                patch_ids=patch_df["patch_id"].to_numpy(dtype=object),
                cell_id_int=None,
                cell_types=None,
                obs_df=out_obs,
                row_filter_report=row_filter_report,
            )

        if level == "cell":
            if "is_cell" not in obs.columns:
                raise KeyError(
                    f"Slide '{slide_record.slide_id}' is missing obs['is_cell'] "
                    "required for level='cell'."
                )
            mask = _obs_bool_array(obs["is_cell"], name="is_cell")
            out_obs = obs.iloc[np.flatnonzero(mask)].copy()
            if "cell_id_int" not in out_obs.columns:
                raise KeyError(f"Slide '{slide_record.slide_id}' cell rows are missing obs['cell_id_int'].")
            n_is_cell = int(out_obs.shape[0])
            missing_cells = set(
                load_missing_cells_for_modelgroup(
                    eval_root, modelgroup, slide_record.slide_id, require_exists=True
                ).tolist()
            )
            if missing_cells:
                keep = ~out_obs["cell_id_int"].astype(np.int32).isin(missing_cells).to_numpy()
                out_obs = out_obs.iloc[np.flatnonzero(keep)].copy()
            # Record of the cells dropped by the token-eligibility filter
            row_filter_report = {
                "modelgroup": str(modelgroup),
                "n_is_cell_rows": n_is_cell,
                "n_cells_missing_after_modelgroup_tokenization": n_is_cell - int(out_obs.shape[0]),
                "n_token_eligible_cells": int(out_obs.shape[0]),
            }
            row_keys = out_obs.index.to_numpy(dtype=object, copy=False)
            cell_ids = out_obs["cell_id_int"].to_numpy(dtype=np.int32, copy=False)
            centroids = _load_centroids(resolve_training_dataset_dir(dataset_id, training_dataset_id), slide_record.slide_id)
            centroids = centroids.set_index("cell_id_int")
            ctr = centroids.loc[cell_ids].reset_index()
            coords = ctr[["x", "y"]].to_numpy(dtype=np.float64)
            out_obs["x"] = coords[:, 0]
            out_obs["y"] = coords[:, 1]
            cell_types = out_obs["cell_type"].astype(str).to_numpy(dtype=object) if "cell_type" in out_obs.columns else None
            return SlideRowBundle(
                slide_id=slide_record.slide_id,
                row_keys=row_keys,
                coords_xy=coords,
                patch_ids=None,
                cell_id_int=cell_ids,
                cell_types=cell_types,
                obs_df=out_obs,
                row_filter_report=row_filter_report,
            )

        raise ValueError(f"Unsupported level='{level}'.")
    finally:
        if getattr(adata, "file", None) is not None:
            adata.file.close()


def load_slide_records(training_dataset_id: str) -> Dict[str, SlideRecord]:
    return {r.slide_id: r for r in read_metadata(training_dataset_id)}


def _take_rows_from_h5_dataset(dset: h5py.Dataset, row_idx: Sequence[int]) -> np.ndarray:
    """Read rows of an h5py dataset in any order (duplicates allowed): sorted unique read, then the requested order is restored."""
    idx = np.asarray(row_idx, dtype=np.int64).reshape(-1)
    tail_shape = tuple(dset.shape[1:])
    if idx.size == 0:
        return np.empty((0, *tail_shape), dtype=dset.dtype)

    if int(idx.min()) < 0 or int(idx.max()) >= int(dset.shape[0]):
        raise IndexError(
            f"Requested HDF5 row indices are out of bounds for dataset "
            f"shape={tuple(dset.shape)}. min={int(idx.min())}, max={int(idx.max())}."
        )

    order = np.argsort(idx, kind="stable")
    idx_sorted = idx[order]
    unique_idx, inverse = np.unique(idx_sorted, return_inverse=True)
    taken_unique = dset[unique_idx]
    taken_sorted = taken_unique[inverse]

    out = np.empty((idx.size, *tail_shape), dtype=taken_sorted.dtype)
    out[order] = taken_sorted
    return np.asarray(out)


def load_patch_embeddings(
    emb_h5_path: Path,
    patch_ids_ordered: Sequence[str],
    space: str,
) -> np.ndarray:
    dset_name = "patch_emb_he" if space in {"he", "he_l2"} else "patch_emb_he_clip"
    with h5py.File(emb_h5_path, "r") as f:
        if dset_name not in f:
            raise KeyError(f"Embedding file '{emb_h5_path}' is missing dataset '{dset_name}'.")
        patch_ids = _decode_h5_strings(f["patch_id"][:])
        lut = {str(pid): i for i, pid in enumerate(patch_ids.tolist())}
        missing = [str(pid) for pid in patch_ids_ordered if str(pid) not in lut]
        if missing:
            raise KeyError(
                f"Embedding file '{emb_h5_path}' is missing {len(missing)} requested patch ids. "
                f"Examples: {missing[:10]}"
            )
        idx = np.array([lut[str(pid)] for pid in patch_ids_ordered], dtype=np.int64)
        X = _take_rows_from_h5_dataset(f[dset_name], idx).astype(np.float32, copy=False)
    return X


def load_cell_embeddings_via_main_assignment(
    emb_h5_path: Path,
    main_assign_path: Path,
    ordered_cell_ids: Sequence[int],
    space: str,
) -> np.ndarray:
    assign_df = pd.read_parquet(main_assign_path)
    if assign_df["cell_id_int"].duplicated().any():
        dup = (
            assign_df.loc[assign_df["cell_id_int"].duplicated(), "cell_id_int"]
            .head(10)
            .tolist()
        )
        raise ValueError(
            f"Main cell assignment file '{main_assign_path}' contains duplicated "
            f"cell_id_int values. Examples: {dup}"
        )
    assign_df = assign_df.set_index("cell_id_int")
    requested = pd.Index(np.asarray(ordered_cell_ids, dtype=np.int64))
    available = pd.Index(assign_df.index.astype(np.int64))
    missing = requested.difference(available)
    if len(missing) > 0:
        raise KeyError(
            f"Main cell assignment file '{main_assign_path}' is missing "
            f"{len(missing)} requested cell_id_int values. Examples: {missing[:10].tolist()}"
        )
    row_idx = assign_df.loc[list(ordered_cell_ids), "selected_patch_cell_row_index"].to_numpy(dtype=np.int64)
    dset_name = "patch_cell_emb_he" if space in {"he", "he_l2"} else "patch_cell_emb_he_clip"
    with h5py.File(emb_h5_path, "r") as f:
        if dset_name not in f:
            raise KeyError(f"Embedding file '{emb_h5_path}' is missing dataset '{dset_name}'.")
        X = _take_rows_from_h5_dataset(f[dset_name], row_idx).astype(np.float32, copy=False)
    return X


def resolve_scconcept_cell_embeddings_dir(scconcept_training_id: str) -> Path:
    """Directory of the per-slide scConcept cell-embedding files (<slide_id>.h5: cls_emb, cell_id_int) of one scConcept training."""
    return get_full_path(
        "train", "trainings", "trainings_scConcept", str(scconcept_training_id), "embeddings",
        as_str=False,
    )


def load_scconcept_cell_embeddings(h5_path: Path, ordered_cell_ids: Sequence[int]) -> np.ndarray:
    """Per-cell scConcept cls_emb aligned to ordered_cell_ids (float32 [N, 512]); raises KeyError on missing cells."""
    with h5py.File(h5_path, "r") as f:
        if "cls_emb" not in f or "cell_id_int" not in f:
            raise KeyError(
                f"scConcept embedding file '{h5_path}' must contain 'cls_emb' and 'cell_id_int'."
            )
        cell_ids = f["cell_id_int"][:].astype(np.int64)
        order = np.argsort(cell_ids, kind="stable")
        cell_ids_sorted = cell_ids[order]
        req = np.asarray(ordered_cell_ids, dtype=np.int64)
        pos = np.clip(np.searchsorted(cell_ids_sorted, req), 0, max(len(cell_ids_sorted) - 1, 0))
        matched = (cell_ids_sorted[pos] == req) if cell_ids_sorted.size else np.zeros_like(req, dtype=bool)
        if not bool(matched.all()):
            missing = req[~matched][:10].tolist()
            raise KeyError(
                f"scConcept embeddings in '{h5_path}' are missing "
                f"{int((~matched).sum())} requested cell_id_int value(s). Examples: {missing}"
            )
        row_idx = order[pos]
        X = _take_rows_from_h5_dataset(f["cls_emb"], row_idx).astype(np.float32, copy=False)
    return X


def load_membership_table(eval_dataset_root: Path, modelgroup: str, slide_id: str) -> pd.DataFrame:
    p = eval_dataset_root / f"cell_token_membership_{modelgroup}" / f"{slide_id}.parquet"
    if not p.exists():
        raise FileNotFoundError(f"Membership parquet not found: {p}")
    df = pd.read_parquet(p)
    return df[["cell_id_int", "patch_id", "token_index", "nuc_px"]].copy()


def build_split_specific_cell_embedding_index(
    *,
    emb_h5_path: Path,
    allowed_patch_ids: Sequence[str],
    slide_patch_h5: Path,
    centroid_parquet: Path,
) -> pd.DataFrame:
    with h5py.File(emb_h5_path, "r") as f:
        patch_ids = _decode_h5_strings(f["patch_id"][:])
        lut = {pid: i for i, pid in enumerate(patch_ids.tolist())}
        requested_patch_ids = [str(pid) for pid in allowed_patch_ids]
        missing_patch_ids = sorted(set(requested_patch_ids) - set(lut))
        if missing_patch_ids:
            raise RuntimeError(
                f"The protocol requested patch ids that are missing from '{emb_h5_path.name}'. "
                f"Examples: {missing_patch_ids[:10]}"
            )
        allowed_patch_indices = {lut[pid] for pid in requested_patch_ids}
        patch_idx = f["patch_cell_patch_index"][:].astype(np.int32)
        cell_id = f["patch_cell_cell_id_int"][:].astype(np.int32)
        alpha = f["patch_cell_alpha"][:].astype(np.float32)

    patch_center_x, patch_center_y = _load_patch_centers_from_eval_h5(slide_patch_h5, patch_ids)

    centroids = pd.read_parquet(centroid_parquet).copy()
    if "cell_id_int" not in centroids.columns:
        if "cell_id" not in centroids.columns:
            raise KeyError(f"Centroid parquet '{centroid_parquet}' must contain 'cell_id_int' or 'cell_id'.")
        centroids["cell_id_int"] = centroids["cell_id"].astype(str).map(int_cell_id).astype(np.int32)
    if not {"x", "y"}.issubset(set(centroids.columns)):
        raise KeyError(f"Centroid parquet '{centroid_parquet}' must contain columns 'x' and 'y'.")
    centroids = centroids[["cell_id_int", "x", "y"]].drop_duplicates(subset=["cell_id_int"], keep="first").set_index("cell_id_int")

    keep = np.array([pi in allowed_patch_indices for pi in patch_idx], dtype=bool)
    df = pd.DataFrame({
        "patch_cell_row": np.arange(len(patch_idx), dtype=np.int64),
        "patch_index": patch_idx,
        "cell_id_int": cell_id,
        "alpha": alpha,
    })
    df = df.loc[keep].copy()
    if df.empty:
        return df

    missing_cells = sorted(set(df["cell_id_int"].astype(int).tolist()) - set(centroids.index.astype(int).tolist()))
    if missing_cells:
        raise KeyError(
            f"Some cell ids present in patch-cell embeddings are missing from '{centroid_parquet.name}'. "
            f"Examples: {missing_cells[:10]}"
        )

    df["patch_center_x"] = patch_center_x[df["patch_index"].to_numpy(dtype=np.int64)]
    df["patch_center_y"] = patch_center_y[df["patch_index"].to_numpy(dtype=np.int64)]
    cent = centroids.loc[df["cell_id_int"].to_numpy(dtype=np.int32)]
    dx = cent["x"].to_numpy(dtype=np.float64, copy=False) - df["patch_center_x"].to_numpy(dtype=np.float64, copy=False)
    dy = cent["y"].to_numpy(dtype=np.float64, copy=False) - df["patch_center_y"].to_numpy(dtype=np.float64, copy=False)
    df["center_dist2"] = dx * dx + dy * dy

    df = df.sort_values(
        ["cell_id_int", "alpha", "center_dist2", "patch_index", "patch_cell_row"],
        ascending=[True, False, True, True, True],
    )
    df = df.drop_duplicates(subset=["cell_id_int"], keep="first")
    return df


def load_split_specific_cell_embeddings(
    *,
    emb_h5_path: Path,
    split_index_df: pd.DataFrame,
    ordered_cell_ids: Sequence[int],
    space: str,
) -> np.ndarray:
    idx_df = split_index_df.set_index("cell_id_int")
    requested = pd.Index(np.asarray(ordered_cell_ids, dtype=np.int64))
    available = pd.Index(idx_df.index.astype(np.int64))
    missing = requested.difference(available)
    if len(missing) > 0:
        raise KeyError(
            f"Split-specific cell embedding index is missing {len(missing)} requested "
            f"cell_id_int values. This usually means the area split selected cells "
            f"without a valid patch-cell embedding inside the allowed patch stripe. "
            f"Examples: {missing[:10].tolist()}"
        )
    row_idx = idx_df.loc[list(ordered_cell_ids), "patch_cell_row"].to_numpy(dtype=np.int64)
    dset_name = "patch_cell_emb_he" if space in {"he", "he_l2"} else "patch_cell_emb_he_clip"
    with h5py.File(emb_h5_path, "r") as f:
        if dset_name not in f:
            raise KeyError(f"Embedding file '{emb_h5_path}' is missing dataset '{dset_name}'.")
        X = _take_rows_from_h5_dataset(f[dset_name], row_idx).astype(np.float32, copy=False)
    return X


