import json
import re
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import PatchCollection
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.patches import Rectangle
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
)

try:
    import umap
except Exception:  # no cover
    umap = None

from .common import mkdir, resolve_eval_dataset_root, save_json, save_text
from .metrics import _location_profile_scores, _safe_pearson, _safe_spearman, _mae, _rmse


# -----------------------------------------------------------------------------
# General reporting helpers
# -----------------------------------------------------------------------------


_H5_NULL_STRINGS = {"", "na", "nan", "none", "null", "<na>"}


def _as_h5_utf8_strings(
    values,
    *,
    name: str,
    n_expected: Optional[int] = None,
    allow_missing: bool = False,
    missing_value: str = "unknown",
) -> np.ndarray:
    """Object array of str for an h5py UTF-8 dataset (missing values raise, or become missing_value if allow_missing)."""
    s = pd.Series(list(values), dtype="string")

    missing = s.isna() | s.str.strip().str.casefold().isin(_H5_NULL_STRINGS)
    if bool(missing.any()):
        if not allow_missing:
            examples = pd.Series(list(values), dtype=object).loc[missing].head(10).tolist()
            raise ValueError(
                f"Cannot write HDF5 string dataset '{name}' because it contains "
                f"{int(missing.sum())} missing/null values. Examples: {examples}"
            )
        s = s.mask(missing, str(missing_value))

    s = s.astype("string").str.strip().fillna(str(missing_value))
    out = s.astype(object).to_numpy()

    if n_expected is not None and out.shape[0] != int(n_expected):
        raise ValueError(
            f"HDF5 string dataset '{name}' has length {out.shape[0]}, "
            f"expected {int(n_expected)}."
        )
    return out


def save_prediction_bundle(preds: Mapping[str, Mapping], out_dir: Path) -> None:
    mkdir(out_dir)
    row_meta_dir = mkdir(out_dir / "row_metadata")
    meta_rows = []
    for slide_id, bundle in preds.items():
        out_path = out_dir / f"{slide_id}.h5"
        obs_path = ""
        with h5py.File(out_path, "w") as f:
            y_true = np.asarray(bundle["y_true"], dtype=np.float32)
            y_pred = np.asarray(bundle["y_pred"], dtype=np.float32)
            n_rows = int(y_true.shape[0])
            n_genes = int(y_true.shape[1])

            str_dt = h5py.string_dtype(encoding="utf-8")

            f.create_dataset(
                "row_keys",
                data=_as_h5_utf8_strings(bundle["row_keys"], name="row_keys", n_expected=n_rows),
                dtype=str_dt,
            )
            f.create_dataset(
                "gene_names",
                data=_as_h5_utf8_strings(bundle["gene_names"], name="gene_names", n_expected=n_genes),
                dtype=str_dt,
            )
            f.create_dataset("coords_xy", data=np.asarray(bundle["coords_xy"], dtype=np.float32))
            f.create_dataset("y_true", data=y_true, compression="gzip", compression_opts=4)
            f.create_dataset("y_pred", data=y_pred, compression="gzip", compression_opts=4)
            if bundle.get("cell_types") is not None:
                f.create_dataset(
                    "cell_types",
                    data=_as_h5_utf8_strings(
                        bundle["cell_types"],
                        name="cell_types",
                        n_expected=n_rows,
                        allow_missing=True,
                        missing_value="unknown",
                    ),
                    dtype=str_dt,
                )
            f.attrs["slide_id"] = str(bundle["slide_id"])
            f.attrs["source"] = str(bundle["source"])
            f.attrs["subset"] = str(bundle["subset"])
        if bundle.get("obs_df") is not None:
            obs_path = str(row_meta_dir / f"{slide_id}.parquet")
            bundle["obs_df"].reset_index(drop=True).to_parquet(obs_path, index=False)
        meta_rows.append(
            {
                "slide_id": slide_id,
                "source": str(bundle["source"]),
                "subset": str(bundle["subset"]),
                "tissue": str(bundle["tissue"]),
                "n_rows": n_rows,
                "n_genes": n_genes,
                "file": str(out_path),
                "row_metadata_file": obs_path,
            }
        )
    pd.DataFrame(meta_rows).to_csv(out_dir / "prediction_files.csv", index=False)


def save_benchmark_outputs(benchmark: Mapping[str, object], out_dir: Path) -> None:
    mkdir(out_dir)
    save_json(out_dir / "benchmark_summary.json", benchmark["benchmark_summary"])
    benchmark["per_slide_gene"].to_csv(out_dir / "per_slide_gene_metrics.csv", index=False)
    benchmark["per_location"].to_csv(out_dir / "per_location_metrics.csv", index=False)
    benchmark["per_slide_onoff"].to_csv(out_dir / "per_slide_gene_onoff_metrics.csv", index=False)
    if "per_subset_benchmark" in benchmark and isinstance(benchmark["per_subset_benchmark"], pd.DataFrame):
        benchmark["per_subset_benchmark"].to_csv(out_dir / "per_subset_benchmark.csv", index=False)
    if "per_tissue_benchmark" in benchmark and isinstance(benchmark["per_tissue_benchmark"], pd.DataFrame):
        benchmark["per_tissue_benchmark"].to_csv(out_dir / "per_tissue_benchmark.csv", index=False)
    for name, df in benchmark.get("macro_tables", {}).items():
        if isinstance(df, pd.DataFrame):
            df.to_csv(out_dir / f"{name}.csv", index=False)
    coverage = benchmark.get("coverage", {})
    for name, value in coverage.items():
        if isinstance(value, pd.DataFrame):
            value.to_csv(out_dir / f"coverage_{name}.csv", index=False)


def save_diagnostics_outputs(diagnostics: Mapping[str, object], out_dir: Path) -> None:
    mkdir(out_dir)
    save_json(out_dir / "diagnostics_summary.json", diagnostics.get("diagnostics_summary", {}))
    diagnostics["per_slide"].to_csv(out_dir / "per_slide_summary.csv", index=False)
    diagnostics["by_tissue"].to_csv(out_dir / "by_tissue_summary.csv", index=False)
    diagnostics["per_gene"].to_csv(out_dir / "per_gene_summary.csv", index=False)
    diagnostics["top_pred_genes"].to_csv(out_dir / "top_pred_genes.csv", index=False)
    diagnostics["bottom_pred_genes"].to_csv(out_dir / "bottom_pred_genes.csv", index=False)
    if "by_cell_type" in diagnostics:
        diagnostics["by_cell_type"].to_csv(out_dir / "by_cell_type_summary.csv", index=False)
    if "calibration_by_slide" in diagnostics:
        diagnostics["calibration_by_slide"].to_csv(out_dir / "calibration_by_slide.csv", index=False)
    if "depth_robustness" in diagnostics:
        diagnostics["depth_robustness"].to_csv(out_dir / "depth_robustness.csv", index=False)
    for name, df in diagnostics.get("gene_property_strata", {}).items():
        df.to_csv(out_dir / f"gene_property_strata__{name}.csv", index=False)
    if isinstance(diagnostics.get("gene_strata_continuous"), pd.DataFrame):
        diagnostics["gene_strata_continuous"].to_csv(out_dir / "gene_strata_continuous.csv", index=False)
    if isinstance(diagnostics.get("gene_signature_strata"), pd.DataFrame):
        diagnostics["gene_signature_strata"].to_csv(out_dir / "gene_signature_strata.csv", index=False)


def save_structure_outputs(structure: Mapping[str, object], out_dir: Path) -> None:
    mkdir(out_dir)
    save_json(out_dir / "structure_summary.json", structure.get("structure_summary", {}))
    structure["coexpression"].to_csv(out_dir / "coexpression.csv", index=False)
    structure["spatial_autocorrelation"].to_csv(out_dir / "spatial_autocorrelation.csv", index=False)
    structure["clustering"].to_csv(out_dir / "clustering.csv", index=False)
    if "within_slide_rank" in structure:
        structure["within_slide_rank"].to_csv(out_dir / "within_slide_rank.csv", index=False)
    # Metric 1 (neighborhood preservation) and Metric 2 (teacher-niche recovery), per slide
    if isinstance(structure.get("neighborhood_preservation"), pd.DataFrame):
        structure["neighborhood_preservation"].to_csv(out_dir / "neighborhood_preservation.csv", index=False)
    if isinstance(structure.get("teacher_niche_recovery"), pd.DataFrame):
        structure["teacher_niche_recovery"].to_csv(out_dir / "teacher_niche_recovery.csv", index=False)
    # Per-slide reasons of the skipped column-gated analyses (written even when empty)
    if isinstance(structure.get("structure_skipped"), pd.DataFrame):
        structure["structure_skipped"].to_csv(out_dir / "structure_skipped.csv", index=False)


def save_run_summary(run_dir: Path, benchmark: Mapping[str, object], diagnostics: Mapping[str, object], structure: Mapping[str, object]) -> None:
    summary_txt = [
        "Run-level summary",
        json.dumps(benchmark["benchmark_summary"], indent=2),
        f"n_per_slide_rows: {len(diagnostics['per_slide'])}",
        f"n_coexpression_rows: {len(structure['coexpression'])}",
    ]
    save_text(run_dir / "run_summary.txt", "\n".join(summary_txt) + "\n")


# -----------------------------------------------------------------------------
# Specific qualitative report helpers
# -----------------------------------------------------------------------------


_NULL_STRINGS = {"", "na", "nan", "none", "null", "<na>"}


def _report_cfg(cfg: Mapping) -> Mapping:
    return cfg.get("protocol", {}).get("reports", {})


def _level(cfg: Mapping) -> str:
    return str(cfg.get("protocol", {}).get("level", "patch_all"))


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
        raise ValueError(f"Expected patch image dataset to be 4D, got shape={shape}.")
    if shape[-1] == 3:
        return int(shape[1]), int(shape[2])
    if shape[1] == 3:
        return int(shape[2]), int(shape[3])
    raise ValueError(f"Unrecognized patch image tensor shape: {shape}.")


def _to_hwc_uint8(img: np.ndarray) -> np.ndarray:
    img = np.asarray(img)
    if img.ndim != 3:
        raise ValueError(f"Expected one patch image to be 3D, got shape={img.shape}.")
    if img.shape[0] == 3 and img.shape[-1] != 3:
        img = np.moveaxis(img, 0, -1)
    if img.shape[-1] != 3:
        raise ValueError(f"Expected RGB image with 3 channels, got shape={img.shape}.")

    if np.issubdtype(img.dtype, np.floating):
        mx = float(np.nanmax(img)) if img.size else 1.0
        if mx <= 1.5:
            img = img * 255.0
        img = np.clip(img, 0, 255).astype(np.uint8)
    elif img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def _normalize_patch_id(x: object) -> str:
    s = str(x)
    if s.endswith("_cells"):
        s = s[:-6]
    elif s.endswith("_all"):
        s = s[:-4]
    if s.endswith(".png"):
        s = s[:-4]
    return s


def _safe_gene_filename(gene: str) -> str:
    gene = str(gene)
    gene = re.sub(r"[^A-Za-z0-9_.-]+", "_", gene)
    return gene[:120] if len(gene) > 120 else gene


def _load_patch_geometry_from_h5(eval_dataset_root: Path, slide_id: str) -> Optional[pd.DataFrame]:
    patch_h5 = eval_dataset_root / "patches" / f"{slide_id}.h5"
    if not patch_h5.exists():
        return None

    with h5py.File(patch_h5, "r") as f:
        if "barcode" not in f or "coords" not in f or "img" not in f:
            return None
        patch_ids = _decode_h5_strings(f["barcode"][:])
        coords = np.asarray(f["coords"][:], dtype=np.float64)
        ph, pw = _infer_patch_hw(tuple(f["img"].shape))

    if coords.ndim != 2 or coords.shape[1] != 2:
        return None

    x_min = coords[:, 0]
    y_min = coords[:, 1]
    return pd.DataFrame(
        {
            "patch_id": patch_ids.astype(object),
            "patch_id_norm": np.asarray([_normalize_patch_id(x) for x in patch_ids], dtype=object),
            "x_min": x_min,
            "y_min": y_min,
            "w": float(pw),
            "h": float(ph),
            "x_max": x_min + float(pw),
            "y_max": y_min + float(ph),
        }
    )


def _patch_rects_for_bundle(bundle: Mapping, eval_dataset_root: Path) -> Optional[np.ndarray]:
    obs_df = bundle.get("obs_df", None)
    if obs_df is None or not isinstance(obs_df, pd.DataFrame) or obs_df.empty:
        return None
    if "patch_id" not in obs_df.columns:
        return None

    geom = _load_patch_geometry_from_h5(eval_dataset_root, str(bundle["slide_id"]))
    if geom is None or geom.empty:
        return None

    geom = geom.drop_duplicates("patch_id_norm", keep="first").set_index("patch_id_norm")
    patch_ids = obs_df["patch_id"].astype(str).map(_normalize_patch_id).to_numpy(dtype=object)
    missing = [pid for pid in patch_ids[:1000] if pid not in geom.index]
    if missing:
        return None

    sub = geom.loc[patch_ids]
    rects = sub[["x_min", "y_min", "w", "h"]].to_numpy(dtype=np.float64)
    if rects.shape[0] != np.asarray(bundle["coords_xy"]).shape[0]:
        return None
    return rects


def _build_he_mosaic(
    eval_dataset_root: Path,
    slide_id: str,
    cfg: Mapping,
) -> Optional[Dict[str, object]]:
    """Low-resolution H&E mosaic of eval_dataset_root/patches/{slide_id}.h5 (overlapping patches averaged)."""
    patch_h5 = eval_dataset_root / "patches" / f"{slide_id}.h5"
    if not patch_h5.exists():
        return None

    rcfg = _report_cfg(cfg)
    requested_ds = max(1, int(rcfg.get("he_mosaic_downsample", 8)))
    max_mpx = float(rcfg.get("he_mosaic_max_megapixels", 24))

    with h5py.File(patch_h5, "r") as f:
        if "img" not in f or "coords" not in f:
            return None

        img_dset = f["img"]
        coords = np.asarray(f["coords"][:], dtype=np.float64)
        if coords.ndim != 2 or coords.shape[1] != 2 or coords.shape[0] == 0:
            return None

        ph, pw = _infer_patch_hw(tuple(img_dset.shape))
        x0 = float(np.min(coords[:, 0]))
        y0 = float(np.min(coords[:, 1]))
        x1 = float(np.max(coords[:, 0] + pw))
        y1 = float(np.max(coords[:, 1] + ph))
        span_x = max(x1 - x0, 1.0)
        span_y = max(y1 - y0, 1.0)

        ds = requested_ds
        while (np.ceil(span_x / ds) * np.ceil(span_y / ds)) > max_mpx * 1_000_000:
            ds *= 2

        W = int(np.ceil(span_x / ds))
        H = int(np.ceil(span_y / ds))
        acc = np.zeros((H, W, 3), dtype=np.float32)
        weight = np.zeros((H, W), dtype=np.float32)

        for i in range(coords.shape[0]):
            patch = _to_hwc_uint8(img_dset[i])
            small = patch[::ds, ::ds, :].astype(np.float32, copy=False)

            xx = int(round((float(coords[i, 0]) - x0) / ds))
            yy = int(round((float(coords[i, 1]) - y0) / ds))
            hh = min(small.shape[0], H - yy)
            ww = min(small.shape[1], W - xx)
            if hh <= 0 or ww <= 0:
                continue

            acc[yy : yy + hh, xx : xx + ww, :] += small[:hh, :ww, :]
            weight[yy : yy + hh, xx : xx + ww] += 1.0

    out = np.full((H, W, 3), 255, dtype=np.float32)
    mask = weight > 0
    out[mask] = acc[mask] / weight[mask, None]
    out = np.clip(out, 0, 255).astype(np.uint8)

    return {
        "image": out,
        "extent": (x0, x0 + W * ds, y0 + H * ds, y0),  # left, right, bottom, top
        "downsample": int(ds),
        "n_patches": int(coords.shape[0]),
    }


def _style_spatial_axis(
    ax,
    coords: np.ndarray,
    patch_rects: Optional[np.ndarray],
    mosaic: Optional[Mapping],
) -> None:
    if mosaic is not None:
        left, right, bottom, top = mosaic["extent"]
        ax.set_xlim(left, right)
        ax.set_ylim(bottom, top)
    elif patch_rects is not None and patch_rects.size:
        x0 = float(np.nanmin(patch_rects[:, 0]))
        y0 = float(np.nanmin(patch_rects[:, 1]))
        x1 = float(np.nanmax(patch_rects[:, 0] + patch_rects[:, 2]))
        y1 = float(np.nanmax(patch_rects[:, 1] + patch_rects[:, 3]))
        ax.set_xlim(x0, x1)
        ax.set_ylim(y1, y0)
    elif coords.size:
        ax.set_xlim(float(np.nanmin(coords[:, 0])), float(np.nanmax(coords[:, 0])))
        ax.set_ylim(float(np.nanmax(coords[:, 1])), float(np.nanmin(coords[:, 1])))

    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])


def _draw_he_background(ax, mosaic: Optional[Mapping], alpha: float = 1.0) -> None:
    if mosaic is None:
        return
    ax.imshow(
        mosaic["image"],
        extent=mosaic["extent"],
        interpolation="nearest",
        alpha=float(alpha),
        zorder=0,
    )


def _draw_he_only(
    ax,
    *,
    coords: np.ndarray,
    patch_rects: Optional[np.ndarray],
    mosaic: Optional[Mapping],
    title: str,
) -> None:
    _draw_he_background(ax, mosaic, alpha=1.0)
    if mosaic is None:
        if patch_rects is not None:
            rects = [Rectangle((x, y), w, h) for x, y, w, h in patch_rects]
            pc = PatchCollection(rects, facecolor="none", edgecolor="0.7", linewidth=0.2)
            ax.add_collection(pc)
        else:
            ax.scatter(coords[:, 0], coords[:, 1], s=0.5, alpha=0.35)
    _style_spatial_axis(ax, coords, patch_rects, mosaic)
    ax.set_title(title)


def _finite_values(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    return values[np.isfinite(values)]


def _robust_limits(values: np.ndarray, percentile: float = 99.0) -> Tuple[float, float]:
    vals = _finite_values(values)
    if vals.size == 0:
        return 0.0, 1.0
    p = float(percentile)
    lo = np.nanpercentile(vals, 100.0 - p)
    hi = np.nanpercentile(vals, p)
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo = float(np.nanmin(vals))
        hi = float(np.nanmax(vals))
    if lo == hi:
        lo -= 0.5
        hi += 0.5
    return float(lo), float(hi)


def _symmetric_abs_limit(values: np.ndarray, percentile: float = 98.0) -> float:
    vals = np.abs(_finite_values(values))
    if vals.size == 0:
        return 1.0
    lim = float(np.nanpercentile(vals, float(percentile)))
    if not np.isfinite(lim) or lim <= 0:
        lim = float(np.nanmax(vals)) if vals.size else 1.0
    return max(lim, 1e-6)


def _plot_indices(n: int, level: str, cfg: Mapping) -> np.ndarray:
    n = int(n)
    if n <= 0:
        return np.zeros((0,), dtype=np.int64)

    # Patch levels: all patches are drawn (as rectangles)
    if str(level).startswith("patch"):
        return np.arange(n, dtype=np.int64)

    max_points = int(_report_cfg(cfg).get("max_plot_points", 60000))
    if n <= max_points:
        return np.arange(n, dtype=np.int64)

    rng = np.random.default_rng(0)
    return np.sort(rng.choice(n, size=max_points, replace=False)).astype(np.int64)


def _draw_spatial_values(
    ax,
    *,
    coords: np.ndarray,
    values: np.ndarray,
    level: str,
    cfg: Mapping,
    patch_rects: Optional[np.ndarray],
    mosaic: Optional[Mapping],
    title: str,
    cmap: str = "viridis",
    norm: Optional[Normalize] = None,
    alpha: Optional[float] = None,
):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    coords = np.asarray(coords, dtype=np.float64)
    valid = np.isfinite(values) & np.all(np.isfinite(coords), axis=1)

    _draw_he_background(ax, mosaic, alpha=1.0)

    if alpha is None:
        alpha = float(_report_cfg(cfg).get("patch_alpha", 0.55))  # A.1 is patch_all only

    mappable = None
    if level.startswith("patch") and patch_rects is not None:
        valid &= np.all(np.isfinite(patch_rects), axis=1)
        idx = np.flatnonzero(valid)
        rects = [
            Rectangle((patch_rects[i, 0], patch_rects[i, 1]), patch_rects[i, 2], patch_rects[i, 3])
            for i in idx.tolist()
        ]
        pc = PatchCollection(
            rects,
            cmap=cmap,
            norm=norm,
            edgecolor="none",
            linewidth=0.0,
            alpha=alpha,
            zorder=2,
        )
        pc.set_array(values[idx].astype(np.float64))
        ax.add_collection(pc)
        mappable = pc
    else:
        idx_all = np.flatnonzero(valid)
        draw_idx = idx_all[_plot_indices(len(idx_all), level, cfg)]
        sc = ax.scatter(
            coords[draw_idx, 0],
            coords[draw_idx, 1],
            c=values[draw_idx],
            s=2.0 if level == "cell" else 7.0,
            cmap=cmap,
            norm=norm,
            alpha=alpha,
            linewidths=0,
            zorder=2,
        )
        mappable = sc

    _style_spatial_axis(ax, coords, patch_rects, mosaic)
    ax.set_title(title)
    return mappable


def _draw_spatial_categories(
    ax,
    *,
    coords: np.ndarray,
    labels: np.ndarray,
    level: str,
    cfg: Mapping,
    patch_rects: Optional[np.ndarray],
    mosaic: Optional[Mapping],
    title: str,
    label_order: Optional[Sequence[object]] = None,
    show_colorbar: bool = True,
):
    coords = np.asarray(coords, dtype=np.float64)
    labels = np.asarray(labels, dtype=object).reshape(-1)

    if label_order is None:
        cats = sorted(pd.Series(labels.astype(str)).dropna().unique().tolist())
    else:
        cats = [str(x) for x in label_order]

    cat = pd.Categorical(labels.astype(str), categories=cats)
    codes = cat.codes.astype(np.int32)
    valid = (codes >= 0) & np.all(np.isfinite(coords), axis=1)

    cmap = plt.get_cmap("tab20", max(len(cats), 1))
    norm = Normalize(vmin=-0.5, vmax=max(len(cats) - 0.5, 0.5))
    alpha = float(_report_cfg(cfg).get("patch_alpha", 0.55))  # A.1 is patch_all only

    _draw_he_background(ax, mosaic, alpha=1.0)

    mappable = None
    if level.startswith("patch") and patch_rects is not None:
        valid &= np.all(np.isfinite(patch_rects), axis=1)
        idx = np.flatnonzero(valid)
        rects = [
            Rectangle((patch_rects[i, 0], patch_rects[i, 1]), patch_rects[i, 2], patch_rects[i, 3])
            for i in idx.tolist()
        ]
        pc = PatchCollection(
            rects,
            cmap=cmap,
            norm=norm,
            edgecolor="none",
            linewidth=0.0,
            alpha=alpha,
            zorder=2,
        )
        pc.set_array(codes[idx].astype(np.float64))
        ax.add_collection(pc)
        mappable = pc
    else:
        idx_all = np.flatnonzero(valid)
        draw_idx = idx_all[_plot_indices(len(idx_all), level, cfg)]
        sc = ax.scatter(
            coords[draw_idx, 0],
            coords[draw_idx, 1],
            c=codes[draw_idx],
            s=2.0 if level == "cell" else 7.0,
            cmap=cmap,
            norm=norm,
            alpha=alpha,
            linewidths=0,
            zorder=2,
        )
        mappable = sc

    _style_spatial_axis(ax, coords, patch_rects, mosaic)
    ax.set_title(title)

    if show_colorbar and mappable is not None and len(cats) <= 12:
        cbar = ax.figure.colorbar(mappable, ax=ax, fraction=0.046, pad=0.02)
        cbar.set_ticks(np.arange(len(cats)))
        cbar.set_ticklabels(cats)

    return mappable


def _add_colorbar(fig, ax, mappable, label: str = "") -> None:
    if mappable is None:
        return
    cbar = fig.colorbar(mappable, ax=ax, fraction=0.046, pad=0.02)
    if label:
        cbar.set_label(label)


def _nanmean_axis1(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    valid = np.isfinite(x)
    denom = np.sum(valid, axis=1)
    num = np.nansum(np.where(valid, x, 0.0), axis=1)
    out = np.full((x.shape[0],), np.nan, dtype=np.float64)
    keep = denom > 0
    out[keep] = num[keep] / denom[keep]
    return out


def _nanrmse_axis1(x: np.ndarray) -> np.ndarray:
    return np.sqrt(_nanmean_axis1(np.square(x, dtype=np.float64)))


def _zscore_by_true(
    X_true: np.ndarray,
    X_pred: np.ndarray,
    cfg: Mapping,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Z-score the common measured genes with the mean/std of X_true (clipped), for the structure plots."""
    X_true = np.asarray(X_true, dtype=np.float32)
    X_pred = np.asarray(X_pred, dtype=np.float32)

    common_gene = np.all(np.isfinite(X_true), axis=0) & np.all(np.isfinite(X_pred), axis=0)
    gene_idx = np.flatnonzero(common_gene)
    if gene_idx.size < 2:
        return None

    Xt = X_true[:, gene_idx].astype(np.float32, copy=False)
    Xp = X_pred[:, gene_idx].astype(np.float32, copy=False)

    mu = Xt.mean(axis=0, keepdims=True)
    sd = Xt.std(axis=0, keepdims=True)
    keep = sd.reshape(-1) > 1e-6
    if int(np.sum(keep)) < 2:
        return None

    Xt = (Xt[:, keep] - mu[:, keep]) / sd[:, keep]
    Xp = (Xp[:, keep] - mu[:, keep]) / sd[:, keep]

    clip = float(_report_cfg(cfg).get("zscore_clip", 5.0))
    Xt = np.clip(Xt, -clip, clip).astype(np.float32, copy=False)
    Xp = np.clip(Xp, -clip, clip).astype(np.float32, copy=False)
    return Xt, Xp, gene_idx[keep]


def _joint_embed_2d(x_true: np.ndarray, x_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    X = np.vstack([x_true, x_pred]).astype(np.float32, copy=False)
    d = min(20, X.shape[1], max(2, X.shape[0] - 1))
    if X.shape[1] > d:
        X = PCA(n_components=d, random_state=0).fit_transform(X)
    if umap is not None and X.shape[0] >= 10:
        Z = umap.UMAP(
            n_components=2,
            n_neighbors=min(30, max(2, X.shape[0] - 1)),
            min_dist=0.3,
            metric="euclidean",
            random_state=0,
        ).fit_transform(X)
    else:
        Z = PCA(n_components=2, random_state=0).fit_transform(X)
    n = x_true.shape[0]
    return Z[:n], Z[n:]


def _cluster_labels_z(X: np.ndarray, k: int = 8, pca_dim: int = 20) -> Optional[np.ndarray]:
    X = np.asarray(X, dtype=np.float32)
    if X.shape[0] < 10 or X.shape[1] < 2:
        return None

    d = min(int(pca_dim), X.shape[1], max(2, X.shape[0] - 1))
    Z = PCA(n_components=d, random_state=0).fit_transform(X) if X.shape[1] > d else X
    k_eff = min(int(k), max(2, int(np.sqrt(X.shape[0]))))
    if k_eff < 2 or X.shape[0] < k_eff:
        return None
    return KMeans(n_clusters=k_eff, n_init=10, random_state=0).fit_predict(Z)


def _hungarian_relabel(reference: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Relabel predicted clusters to the matched true-cluster IDs (Hungarian matching), for visualization only."""
    reference = np.asarray(reference)
    labels = np.asarray(labels)
    ref_classes = np.unique(reference)
    lab_classes = np.unique(labels)

    mat = np.zeros((len(ref_classes), len(lab_classes)), dtype=np.int64)
    ref_lut = {c: i for i, c in enumerate(ref_classes.tolist())}
    lab_lut = {c: i for i, c in enumerate(lab_classes.tolist())}

    for r, l in zip(reference.tolist(), labels.tolist()):
        mat[ref_lut[r], lab_lut[l]] += 1

    row_ind, col_ind = linear_sum_assignment(-mat)
    mapping = {lab_classes[c]: ref_classes[r] for r, c in zip(row_ind, col_ind)}

    next_label = int(np.max(ref_classes)) + 1 if ref_classes.size else 0
    out = np.empty_like(labels)
    for i, l in enumerate(labels.tolist()):
        if l in mapping:
            out[i] = mapping[l]
        else:
            out[i] = next_label
            next_label += 1
    return out


def _contingency_df(a: np.ndarray, b: np.ndarray, name_a: str, name_b: str) -> pd.DataFrame:
    return pd.crosstab(
        pd.Series(a.astype(str), name=name_a),
        pd.Series(b.astype(str), name=name_b),
        dropna=False,
    )


def _plot_contingency_heatmap(ax, df: pd.DataFrame, title: str) -> None:
    if df.empty:
        ax.axis("off")
        ax.set_title(title + " unavailable")
        return
    mat = df.to_numpy(dtype=np.float64)
    im = ax.imshow(mat, aspect="auto", interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel(df.columns.name or "pred")
    ax.set_ylabel(df.index.name or "true")
    if df.shape[0] <= 20:
        ax.set_yticks(np.arange(df.shape[0]))
        ax.set_yticklabels(df.index.astype(str).tolist(), fontsize=7)
    if df.shape[1] <= 20:
        ax.set_xticks(np.arange(df.shape[1]))
        ax.set_xticklabels(df.columns.astype(str).tolist(), rotation=90, fontsize=7)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.02)


def _true_domain_recovery(
    X_true_z: np.ndarray,
    X_pred_z: np.ndarray,
    *,
    k: int,
    pca_dim: int,
) -> Optional[Dict[str, object]]:
    if X_true_z.shape[0] < 10 or X_true_z.shape[1] < 2:
        return None

    d = min(int(pca_dim), X_true_z.shape[1], max(2, X_true_z.shape[0] - 1))
    pca = PCA(n_components=d, random_state=0) if X_true_z.shape[1] > d else None
    Zt = pca.fit_transform(X_true_z) if pca is not None else X_true_z
    Zp = pca.transform(X_pred_z) if pca is not None else X_pred_z

    k_eff = min(int(k), max(2, int(np.sqrt(X_true_z.shape[0]))))
    if k_eff < 2 or X_true_z.shape[0] < k_eff:
        return None

    km = KMeans(n_clusters=k_eff, n_init=10, random_state=0).fit(Zt)
    c_true = km.labels_
    centroids = km.cluster_centers_

    # Assign each predicted profile to the nearest true-expression centroid.
    d2 = np.sum((Zp[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
    c_recovered = np.argmin(d2, axis=1).astype(np.int32)

    return {
        "true_domains": c_true,
        "pred_assigned_to_true_domains": c_recovered,
        "accuracy": float(np.mean(c_recovered == c_true)),
        "ami": float(adjusted_mutual_info_score(c_true, c_recovered)),
        "ari": float(adjusted_rand_score(c_true, c_recovered)),
        "nmi": float(normalized_mutual_info_score(c_true, c_recovered)),
    }


def _clean_reference_labels(labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(labels, dtype=object).reshape(-1)
    s = pd.Series(labels, dtype="string")
    valid = s.notna() & ~s.str.strip().str.lower().isin(_NULL_STRINGS)
    cleaned = s.astype(str).to_numpy(dtype=object)
    return cleaned, valid.to_numpy(dtype=bool)


def _resolve_reference_labels(bundle: Mapping, level: str) -> Tuple[Optional[str], Optional[np.ndarray], Optional[np.ndarray]]:
    """(name, labels, valid mask) of the reference labels: cell_type at cell level, else the Novae domain if present."""
    obs_df = bundle.get("obs_df", None)

    if level == "cell":
        if obs_df is not None and isinstance(obs_df, pd.DataFrame) and "cell_type" in obs_df.columns:
            labels, valid = _clean_reference_labels(obs_df["cell_type"].to_numpy(dtype=object))
            return "cell_type", labels, valid
        if bundle.get("cell_types", None) is not None:
            labels, valid = _clean_reference_labels(np.asarray(bundle["cell_types"], dtype=object))
            return "cell_type", labels, valid

    if obs_df is not None and isinstance(obs_df, pd.DataFrame):
        if "novae_domain" in obs_df.columns:
            labels, valid = _clean_reference_labels(obs_df["novae_domain"].to_numpy(dtype=object))
            return "novae_domain", labels, valid

        domain_cols = sorted(
            c for c in obs_df.columns
            if c.startswith("novae_patch_domain_") and not c.startswith("novae_patch_domain_frac_")
        )
        if len(domain_cols) == 1:
            labels, valid = _clean_reference_labels(obs_df[domain_cols[0]].to_numpy(dtype=object))
            return domain_cols[0], labels, valid

    return None, None, None


def _slide_title(bundle: Mapping, cfg: Mapping, *, extra: str = "") -> str:
    yt = np.asarray(bundle["y_true"])
    base = (
        f"{bundle['slide_id']} | source={bundle['source']} | subset={bundle['subset']} | "
        f"tissue={bundle['tissue']} | level={_level(cfg)} | rows={yt.shape[0]:,} | genes={yt.shape[1]:,}"
    )
    return f"{base} | {extra}" if extra else base


def _set_embedding_limits(ax, emb_true: np.ndarray, emb_pred: np.ndarray) -> None:
    Z = np.vstack([emb_true, emb_pred])
    x0, x1 = float(np.nanmin(Z[:, 0])), float(np.nanmax(Z[:, 0]))
    y0, y1 = float(np.nanmin(Z[:, 1])), float(np.nanmax(Z[:, 1]))
    padx = 0.05 * max(x1 - x0, 1e-6)
    pady = 0.05 * max(y1 - y0, 1e-6)
    ax.set_xlim(x0 - padx, x1 + padx)
    ax.set_ylim(y0 - pady, y1 + pady)
    ax.set_xticks([])
    ax.set_yticks([])


def _draw_embedding_categories(
    ax,
    emb: np.ndarray,
    labels: np.ndarray,
    *,
    title: str,
    label_order: Optional[Sequence[object]] = None,
    valid_mask: Optional[np.ndarray] = None,
) -> None:
    emb = np.asarray(emb, dtype=np.float64)
    labels = np.asarray(labels, dtype=object).reshape(-1)
    if valid_mask is None:
        valid_mask = np.ones((emb.shape[0],), dtype=bool)

    if label_order is None:
        cats = sorted(pd.Series(labels[valid_mask].astype(str)).dropna().unique().tolist())
    else:
        cats = [str(x) for x in label_order]

    cat = pd.Categorical(labels.astype(str), categories=cats)
    codes = cat.codes.astype(np.int32)
    keep = valid_mask & (codes >= 0) & np.all(np.isfinite(emb), axis=1)

    cmap = plt.get_cmap("tab20", max(len(cats), 1))
    norm = Normalize(vmin=-0.5, vmax=max(len(cats) - 0.5, 0.5))

    sc = ax.scatter(
        emb[keep, 0],
        emb[keep, 1],
        c=codes[keep],
        s=4.0,
        cmap=cmap,
        norm=norm,
        alpha=0.7,
        linewidths=0,
    )
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])

    if len(cats) <= 12:
        cbar = ax.figure.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
        cbar.set_ticks(np.arange(len(cats)))
        cbar.set_ticklabels(cats)


def _select_genes_for_slide(
    *,
    slide_id: str,
    gene_names: np.ndarray,
    benchmark: Mapping[str, object],
    diagnostics: Mapping[str, object],
    n_genes: int,
) -> Dict[str, list[str]]:
    """Global and slide-specific best/worst genes, as gene -> selection reasons (shown in the gene panel title)."""
    available = set(map(str, gene_names.tolist()))
    selected: Dict[str, list[str]] = {}

    def add(gene: object, reason: str) -> None:
        g = str(gene)
        if g not in available:
            return
        selected.setdefault(g, [])
        if reason not in selected[g]:
            selected[g].append(reason)

    top = diagnostics.get("top_pred_genes", pd.DataFrame())
    bottom = diagnostics.get("bottom_pred_genes", pd.DataFrame())
    n_top = max(1, int(np.ceil(n_genes / 4)))
    n_bottom = max(1, int(np.ceil(n_genes / 4)))

    if isinstance(top, pd.DataFrame) and "gene" in top.columns:
        for rank, row in enumerate(top.head(n_top).itertuples(index=False), start=1):
            metric = getattr(row, "gene_macro_spearman", np.nan)
            add(getattr(row, "gene"), f"global_best_rank={rank},rho={metric:.3g}")

    if isinstance(bottom, pd.DataFrame) and "gene" in bottom.columns:
        for rank, row in enumerate(bottom.head(n_bottom).itertuples(index=False), start=1):
            metric = getattr(row, "gene_macro_spearman", np.nan)
            add(getattr(row, "gene"), f"global_worst_rank={rank},rho={metric:.3g}")

    per_slide_gene = benchmark.get("per_slide_gene", pd.DataFrame())
    if isinstance(per_slide_gene, pd.DataFrame) and not per_slide_gene.empty:
        sdf = per_slide_gene.loc[
            (per_slide_gene["slide_id"].astype(str) == str(slide_id))
            & per_slide_gene["corr_valid"].fillna(False).astype(bool)
            & np.isfinite(per_slide_gene["spearman"])
        ].copy()

        if not sdf.empty:
            best = sdf.sort_values(["spearman", "n_samples", "gene"], ascending=[False, False, True]).head(max(1, n_genes // 4))
            worst = sdf.sort_values(["spearman", "n_samples", "gene"], ascending=[True, False, True]).head(max(1, n_genes // 4))

            for rank, row in enumerate(best.itertuples(index=False), start=1):
                add(row.gene, f"slide_best_rank={rank},rho={float(row.spearman):.3g}")
            for rank, row in enumerate(worst.itertuples(index=False), start=1):
                add(row.gene, f"slide_worst_rank={rank},rho={float(row.spearman):.3g}")

    # Fill remaining slots with globally ranked genes if slide-specific/global overlap was high.
    if isinstance(top, pd.DataFrame) and "gene" in top.columns:
        for row in top.itertuples(index=False):
            if len(selected) >= n_genes:
                break
            add(getattr(row, "gene"), "global_fill_best")

    if isinstance(bottom, pd.DataFrame) and "gene" in bottom.columns:
        for row in bottom.itertuples(index=False):
            if len(selected) >= n_genes:
                break
            add(getattr(row, "gene"), "global_fill_worst")

    return dict(list(selected.items())[: int(n_genes)])


def _save_slide_overview_panel(
    *,
    bundle: Mapping,
    out_path: Path,
    cfg: Mapping,
    mosaic: Optional[Mapping],
    patch_rects: Optional[np.ndarray],
) -> None:
    level = _level(cfg)
    coords = np.asarray(bundle["coords_xy"], dtype=np.float32)
    yt = np.asarray(bundle["y_true"], dtype=np.float32)
    yp = np.asarray(bundle["y_pred"], dtype=np.float32)
    detected = np.asarray(bundle["detected"], dtype=bool)

    valid = np.isfinite(yt) & np.isfinite(yp)
    residual = np.where(valid, yp - yt, np.nan)

    loc_spearman = _location_profile_scores(yt, yp, detected, cfg)
    local_mae = _nanmean_axis1(np.abs(residual))
    local_rmse = _nanrmse_axis1(residual)
    local_bias = _nanmean_axis1(residual)

    true_total = np.nansum(np.where(np.isfinite(yt), yt, np.nan), axis=1)
    pred_total = np.nansum(np.where(valid, yp, np.nan), axis=1)
    total_diff = pred_total - true_total

    total_vmin, total_vmax = _robust_limits(
        np.concatenate([true_total[np.isfinite(true_total)], pred_total[np.isfinite(pred_total)]]),
        percentile=float(_report_cfg(cfg).get("gene_value_percentile", 99.0)),
    )
    bias_lim = _symmetric_abs_limit(local_bias)
    total_diff_lim = _symmetric_abs_limit(total_diff)

    fig, axes = plt.subplots(3, 3, figsize=(18, 15), constrained_layout=True)
    axes = axes.reshape(-1)

    _draw_he_only(
        axes[0],
        coords=coords,
        patch_rects=patch_rects,
        mosaic=mosaic,
        title="Aligned H&E mosaic",
    )

    panels = [
        ("Location profile Spearman", loc_spearman, "viridis", None, "Spearman"),
        ("Local MAE across genes", local_mae, "magma", None, "MAE"),
        ("Local RMSE across genes", local_rmse, "magma", None, "RMSE"),
        ("Local mean residual pred - true", local_bias, "coolwarm", TwoSlopeNorm(vcenter=0.0, vmin=-bias_lim, vmax=bias_lim), "mean residual"),
        ("True total expression", true_total, "viridis", Normalize(vmin=total_vmin, vmax=total_vmax), "sum log1p"),
        ("Predicted total expression", pred_total, "viridis", Normalize(vmin=total_vmin, vmax=total_vmax), "sum log1p"),
        ("Total expression residual pred - true", total_diff, "coolwarm", TwoSlopeNorm(vcenter=0.0, vmin=-total_diff_lim, vmax=total_diff_lim), "sum residual"),
        ("Measured gene count per location", np.sum(np.isfinite(yt), axis=1), "viridis", None, "n genes"),
    ]

    for ax, (title, vals, cmap, norm, cbar_label) in zip(axes[1:], panels):
        m = _draw_spatial_values(
            ax,
            coords=coords,
            values=vals,
            level=level,
            cfg=cfg,
            patch_rects=patch_rects,
            mosaic=mosaic,
            title=title,
            cmap=cmap,
            norm=norm,
        )
        _add_colorbar(fig, ax, m, cbar_label)

    fig.suptitle(_slide_title(bundle, cfg, extra="overview diagnostics"), fontsize=13)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _save_embedding_cluster_panel(
    *,
    bundle: Mapping,
    out_dir: Path,
    cfg: Mapping,
    mosaic: Optional[Mapping],
    patch_rects: Optional[np.ndarray],
) -> None:
    level = _level(cfg)
    coords = np.asarray(bundle["coords_xy"], dtype=np.float32)
    yt = np.asarray(bundle["y_true"], dtype=np.float32)
    yp = np.asarray(bundle["y_pred"], dtype=np.float32)

    normed = _zscore_by_true(yt, yp, cfg)
    if normed is None:
        save_text(out_dir / "embedding_cluster_panel_skipped.txt", "Skipped: fewer than 2 common finite measured genes after z-scoring.\n")
        return

    X_true_z, X_pred_z, common_gene_idx = normed
    if X_true_z.shape[0] < 10:
        save_text(out_dir / "embedding_cluster_panel_skipped.txt", "Skipped: fewer than 10 locations.\n")
        return

    structure_cfg = cfg.get("protocol", {}).get("structure", {})
    k = int(structure_cfg.get("cluster_k", 8))
    pca_dim = int(structure_cfg.get("cluster_pca_dim", 20))

    emb_true, emb_pred = _joint_embed_2d(X_true_z, X_pred_z)
    c_true = _cluster_labels_z(X_true_z, k=k, pca_dim=pca_dim)
    c_pred_raw = _cluster_labels_z(X_pred_z, k=k, pca_dim=pca_dim)

    if c_true is None or c_pred_raw is None:
        save_text(out_dir / "embedding_cluster_panel_skipped.txt", "Skipped: clustering failed because of insufficient rows/features.\n")
        return

    c_pred_matched = _hungarian_relabel(c_true, c_pred_raw)

    ami = float(adjusted_mutual_info_score(c_true, c_pred_raw))
    ari = float(adjusted_rand_score(c_true, c_pred_raw))
    nmi = float(normalized_mutual_info_score(c_true, c_pred_raw))

    cont = _contingency_df(c_true, c_pred_matched, "true_cluster", "pred_cluster_matched")
    cont.to_csv(out_dir / "cluster_contingency_true_vs_pred.csv")

    recovery = _true_domain_recovery(X_true_z, X_pred_z, k=k, pca_dim=pca_dim)
    if recovery is not None:
        c_recovered = np.asarray(recovery["pred_assigned_to_true_domains"], dtype=np.int32)
        rec_cont = _contingency_df(
            np.asarray(recovery["true_domains"], dtype=np.int32),
            c_recovered,
            "true_expression_domain",
            "pred_nearest_true_centroid",
        )
        rec_cont.to_csv(out_dir / "domain_recovery_contingency.csv")
    else:
        c_recovered = None
        rec_cont = pd.DataFrame()

    label_kind, ref_labels, ref_valid = _resolve_reference_labels(bundle, level)

    fig, axes = plt.subplots(3, 4, figsize=(22, 16), constrained_layout=True)
    ax = axes

    cluster_order = sorted(np.unique(c_true).tolist())

    _draw_embedding_categories(
        ax[0, 0],
        emb_true,
        c_true,
        title="Joint embedding: true profiles / true clusters",
        label_order=cluster_order,
    )
    _set_embedding_limits(ax[0, 0], emb_true, emb_pred)

    _draw_embedding_categories(
        ax[0, 1],
        emb_pred,
        c_pred_matched,
        title=f"Joint embedding: predicted profiles / matched clusters\nAMI={ami:.3f}, ARI={ari:.3f}, NMI={nmi:.3f}",
        label_order=cluster_order,
    )
    _set_embedding_limits(ax[0, 1], emb_true, emb_pred)

    ax[0, 2].scatter(emb_true[:, 0], emb_true[:, 1], s=4, alpha=0.45, linewidths=0, label="true")
    ax[0, 2].scatter(emb_pred[:, 0], emb_pred[:, 1], s=4, alpha=0.45, linewidths=0, label="pred")
    ax[0, 2].legend(frameon=False, fontsize=8)
    ax[0, 2].set_title("Joint embedding overlay\nsame axes for true and predicted profiles")
    _set_embedding_limits(ax[0, 2], emb_true, emb_pred)

    _plot_contingency_heatmap(
        ax[0, 3],
        cont,
        "True vs predicted independent clusters\npred labels Hungarian-matched for display",
    )

    _draw_he_only(
        ax[1, 0],
        coords=coords,
        patch_rects=patch_rects,
        mosaic=mosaic,
        title="Aligned H&E mosaic",
    )

    _draw_spatial_categories(
        ax[1, 1],
        coords=coords,
        labels=c_true,
        level=level,
        cfg=cfg,
        patch_rects=patch_rects,
        mosaic=mosaic,
        title="Slide: true-expression clusters",
        label_order=cluster_order,
    )

    _draw_spatial_categories(
        ax[1, 2],
        coords=coords,
        labels=c_pred_matched,
        level=level,
        cfg=cfg,
        patch_rects=patch_rects,
        mosaic=mosaic,
        title="Slide: predicted clusters\nmatched to true colors for visualization only",
        label_order=cluster_order,
    )

    if c_recovered is not None:
        _draw_spatial_categories(
            ax[1, 3],
            coords=coords,
            labels=c_recovered,
            level=level,
            cfg=cfg,
            patch_rects=patch_rects,
            mosaic=mosaic,
            title=(
                "Domain recovery: pred assigned to nearest true centroid\n"
                f"acc={recovery['accuracy']:.3f}, AMI={recovery['ami']:.3f}, "
                f"ARI={recovery['ari']:.3f}, NMI={recovery['nmi']:.3f}"
            ),
            label_order=cluster_order,
        )
    else:
        ax[1, 3].axis("off")
        ax[1, 3].set_title("Domain recovery unavailable")

    if label_kind is not None and ref_labels is not None and ref_valid is not None and int(ref_valid.sum()) >= 2:
        _draw_embedding_categories(
            ax[2, 0],
            emb_true,
            ref_labels,
            title=f"True embedding colored by reference labels\nreference={label_kind}",
            valid_mask=ref_valid,
        )
        _set_embedding_limits(ax[2, 0], emb_true, emb_pred)

        _draw_embedding_categories(
            ax[2, 1],
            emb_pred,
            ref_labels,
            title=f"Predicted embedding colored by same reference labels\nreference={label_kind}",
            valid_mask=ref_valid,
        )
        _set_embedding_limits(ax[2, 1], emb_true, emb_pred)

        _draw_spatial_categories(
            ax[2, 2],
            coords=coords,
            labels=ref_labels,
            level=level,
            cfg=cfg,
            patch_rects=patch_rects,
            mosaic=mosaic,
            title=f"Slide reference labels\nreference={label_kind}",
            show_colorbar=False,
        )
    else:
        for j in range(3):
            ax[2, j].axis("off")
        ax[2, 0].set_title("No valid reference labels available")

    _plot_contingency_heatmap(
        ax[2, 3],
        rec_cont,
        "True domains vs domain-recovery assignment",
    )

    extra = (
        f"cluster metrics on independent KMeans labels: AMI={ami:.3f}, ARI={ari:.3f}, NMI={nmi:.3f}; "
        f"structure z-score genes from X_true, common_genes={len(common_gene_idx)}"
    )
    fig.suptitle(_slide_title(bundle, cfg, extra=extra), fontsize=13)
    fig.savefig(out_dir / "embedding_cluster_panel.png", dpi=180)
    plt.close(fig)


def _save_gene_panel(
    *,
    bundle: Mapping,
    gene: str,
    reasons: Sequence[str],
    out_path: Path,
    cfg: Mapping,
    mosaic: Optional[Mapping],
    patch_rects: Optional[np.ndarray],
) -> None:
    level = _level(cfg)
    coords = np.asarray(bundle["coords_xy"], dtype=np.float32)
    gene_names = np.asarray(bundle["gene_names"], dtype=object)
    yt = np.asarray(bundle["y_true"], dtype=np.float32)
    yp = np.asarray(bundle["y_pred"], dtype=np.float32)

    if gene not in gene_names.tolist():
        return

    j = int(np.flatnonzero(gene_names == gene)[0])
    t = yt[:, j].astype(np.float64)
    p = yp[:, j].astype(np.float64)
    valid = np.isfinite(t) & np.isfinite(p)
    if int(np.sum(valid)) == 0:
        return

    residual = p - t
    abs_err = np.abs(residual)

    pct = float(_report_cfg(cfg).get("gene_value_percentile", 99.0))
    vmin, vmax = _robust_limits(np.concatenate([t[valid], p[valid]]), percentile=pct)

    rpct = float(_report_cfg(cfg).get("residual_abs_percentile", 98.0))
    rlim = _symmetric_abs_limit(residual[valid], percentile=rpct)

    spearman = _safe_spearman(t[valid], p[valid])
    pearson = _safe_pearson(t[valid], p[valid])
    mae = _mae(t[valid], p[valid])
    rmse = _rmse(t[valid], p[valid])
    n_pos = int(np.sum(t[valid] > 0))

    fig, axes = plt.subplots(2, 3, figsize=(18, 11), constrained_layout=True)
    axes = axes.reshape(-1)

    _draw_he_only(
        axes[0],
        coords=coords,
        patch_rects=patch_rects,
        mosaic=mosaic,
        title="Aligned H&E mosaic",
    )

    shared_norm = Normalize(vmin=vmin, vmax=vmax)
    m = _draw_spatial_values(
        axes[1],
        coords=coords,
        values=np.where(valid, t, np.nan),
        level=level,
        cfg=cfg,
        patch_rects=patch_rects,
        mosaic=mosaic,
        title="Truth",
        cmap="viridis",
        norm=shared_norm,
    )
    _add_colorbar(fig, axes[1], m, "log1p count")

    m = _draw_spatial_values(
        axes[2],
        coords=coords,
        values=np.where(valid, p, np.nan),
        level=level,
        cfg=cfg,
        patch_rects=patch_rects,
        mosaic=mosaic,
        title="Prediction\nsame color scale as truth",
        cmap="viridis",
        norm=shared_norm,
    )
    _add_colorbar(fig, axes[2], m, "log1p count")

    res_norm = TwoSlopeNorm(vcenter=0.0, vmin=-rlim, vmax=rlim)
    m = _draw_spatial_values(
        axes[3],
        coords=coords,
        values=np.where(valid, residual, np.nan),
        level=level,
        cfg=cfg,
        patch_rects=patch_rects,
        mosaic=mosaic,
        title=f"Signed residual pred - true\ncentered at 0, ±p{rpct:g}",
        cmap="coolwarm",
        norm=res_norm,
    )
    _add_colorbar(fig, axes[3], m, "residual")

    m = _draw_spatial_values(
        axes[4],
        coords=coords,
        values=np.where(valid, abs_err, np.nan),
        level=level,
        cfg=cfg,
        patch_rects=patch_rects,
        mosaic=mosaic,
        title="Absolute error",
        cmap="magma",
        norm=None,
    )
    _add_colorbar(fig, axes[4], m, "|error|")

    ax = axes[5]
    tv = t[valid]
    pv = p[valid]
    if tv.size >= 3000:
        hb = ax.hexbin(tv, pv, gridsize=60, mincnt=1)
        fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.02, label="count")
    else:
        ax.scatter(tv, pv, s=6, alpha=0.45, linewidths=0)

    lo = float(np.nanmin(np.concatenate([tv, pv])))
    hi = float(np.nanmax(np.concatenate([tv, pv])))
    if lo == hi:
        lo -= 0.5
        hi += 0.5
    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("truth")
    ax.set_ylabel("prediction")
    ax.set_title(
        f"Truth vs prediction\nSpearman={spearman:.3f}, Pearson={pearson:.3f}, "
        f"MAE={mae:.3g}, RMSE={rmse:.3g}, n_pos={n_pos:,}"
    )

    reason_txt = "; ".join(map(str, reasons))
    fig.suptitle(
        _slide_title(
            bundle,
            cfg,
            extra=f"gene={gene} | selected because: {reason_txt} | valid_locations={int(valid.sum()):,}",
        ),
        fontsize=13,
    )
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _append_unique(dst: list[str], values: Sequence[str], limit: int) -> None:
    seen = set(dst)
    for value in values:
        value = str(value)
        if value in seen:
            continue
        dst.append(value)
        seen.add(value)
        if len(dst) >= limit:
            return


def save_canonical_reports(
    preds: Mapping[str, Mapping],
    benchmark: Mapping[str, object],
    diagnostics: Mapping[str, object],
    out_dir: Path,
    cfg: Mapping,
    n_slides: int = 3,
    n_genes: int = 4,
) -> None:
    mkdir(out_dir)
    per_slide = diagnostics["per_slide"].copy()
    if per_slide.empty:
        save_text(out_dir / "README.txt", "No canonical reports were generated because no valid per-slide metrics were available.\n")
        return

    if "slide_macro_spearman" in per_slide.columns:
        sort_col = "slide_macro_spearman"
        ascending = True
    elif "slide_macro_pearson" in per_slide.columns:
        sort_col = "slide_macro_pearson"
        ascending = True
    elif "location_macro_profile_spearman" in per_slide.columns:
        sort_col = "location_macro_profile_spearman"
        ascending = True
    else:
        save_text(out_dir / "README.txt", "No canonical reports were generated because no sortable per-slide metric column was available.\n")
        return

    per_slide = per_slide.sort_values(sort_col, ascending=ascending)

    n_slides = max(1, int(n_slides))
    n_genes = max(1, int(n_genes))

    selected_slides: list[str] = []
    _append_unique(selected_slides, per_slide.head(max(1, n_slides // 3))["slide_id"].astype(str).tolist(), n_slides)
    _append_unique(
        selected_slides,
        per_slide.iloc[len(per_slide) // 2 : len(per_slide) // 2 + max(1, n_slides // 3)]["slide_id"].astype(str).tolist(),
        n_slides,
    )
    _append_unique(selected_slides, per_slide.tail(max(1, n_slides - len(selected_slides)))["slide_id"].astype(str).tolist(), n_slides)
    _append_unique(selected_slides, per_slide["slide_id"].astype(str).tolist(), n_slides)

    eval_dataset_root = resolve_eval_dataset_root(str(cfg["identity"]["training_dataset_id"]))

    selection_payload: Dict[str, object] = {
        "selected_slides": selected_slides,
        "slide_selection_sort_column": sort_col,
        "level": _level(cfg),
        "per_slide_selected_genes": {},
        "notes": [
            "Spatial plots are overlaid on a reconstructed low-resolution H&E mosaic from eval_dataset_root/patches/{slide_id}.h5 when available.",
            "Patch-level reports render patch rows as semi-transparent rectangles using patch geometry. Cell-level reports render cells as semi-transparent points.",
            "Cluster colors for predicted independent clusters are Hungarian-matched to true cluster IDs for visualization only. AMI/ARI/NMI are computed on raw independent cluster labels.",
            "Domain recovery assigns predicted profiles to nearest true-expression cluster centroids and answers a different question from independent-cluster preservation.",
            "Qualitative UMAP/PCA/KMeans structure plots z-score genes using X_true statistics, apply the same transform to X_pred, clip extreme z-scores, and use the common finite measured gene set per slide.",
        ],
    }

    for slide_id in selected_slides:
        if slide_id not in preds:
            continue

        bundle = preds[slide_id]
        slide_dir = mkdir(out_dir / slide_id)

        coords = np.asarray(bundle["coords_xy"], dtype=np.float32)
        gene_names = np.asarray(bundle["gene_names"], dtype=object)

        mosaic = _build_he_mosaic(eval_dataset_root, slide_id, cfg)
        patch_rects = _patch_rects_for_bundle(bundle, eval_dataset_root) if _level(cfg).startswith("patch") else None

        selected_gene_reasons = _select_genes_for_slide(
            slide_id=slide_id,
            gene_names=gene_names,
            benchmark=benchmark,
            diagnostics=diagnostics,
            n_genes=n_genes,
        )
        selection_payload["per_slide_selected_genes"][slide_id] = selected_gene_reasons

        _save_slide_overview_panel(
            bundle=bundle,
            out_path=slide_dir / "slide_overview_panel.png",
            cfg=cfg,
            mosaic=mosaic,
            patch_rects=patch_rects,
        )

        _save_embedding_cluster_panel(
            bundle=bundle,
            out_dir=slide_dir,
            cfg=cfg,
            mosaic=mosaic,
            patch_rects=patch_rects,
        )

        for gene, reasons in selected_gene_reasons.items():
            _save_gene_panel(
                bundle=bundle,
                gene=gene,
                reasons=reasons,
                out_path=slide_dir / f"gene_{_safe_gene_filename(gene)}_panel.png",
                cfg=cfg,
                mosaic=mosaic,
                patch_rects=patch_rects,
            )

        save_json(
            slide_dir / "report_metadata.json",
            {
                "slide_id": str(slide_id),
                "source": str(bundle["source"]),
                "subset": str(bundle["subset"]),
                "tissue": str(bundle["tissue"]),
                "level": _level(cfg),
                "n_rows": int(coords.shape[0]),
                "n_genes": int(len(gene_names)),
                "selected_genes": selected_gene_reasons,
                "he_mosaic": {
                    "available": mosaic is not None,
                    "downsample": int(mosaic["downsample"]) if mosaic is not None else None,
                    "n_patches": int(mosaic["n_patches"]) if mosaic is not None else None,
                },
                "patch_rectangles_available": bool(patch_rects is not None),
            },
        )

    save_json(out_dir / "report_selection.json", selection_payload)
    save_text(
        out_dir / "README.txt",
        "Canonical qualitative reports\n"
        "=============================\n\n"
        "Each selected slide folder contains:\n"
        "- slide_overview_panel.png: H&E, location-profile Spearman, local MAE/RMSE, local bias, and total-expression diagnostics.\n"
        "- embedding_cluster_panel.png: shared true/predicted embedding, independent cluster preservation, Hungarian-matched cluster colors, contingency heatmap, reference-label views, and true-domain recovery.\n"
        "- gene_<gene>_panel.png: H&E, truth, prediction, signed residual, absolute error, and truth-vs-prediction scatter/hexbin.\n\n"
        "Notes:\n"
        "- The reports do not affect the benchmark metrics.\n"
        "- Predicted cluster labels are relabeled only for visualization; AMI/ARI/NMI are permutation-invariant and use the independent cluster assignments.\n"
        "- Patch-level plots use rectangles; cell-level plots use points.\n"
        "- Truth and prediction maps for a gene share the same color scale.\n"
        "- Residual maps are centered at zero with symmetric robust limits.\n"
        "- Reference labels are explicitly named in panel titles, usually cell_type for cell-level reports and novae_domain for patch-level STHELAR reports when available.\n",
    )