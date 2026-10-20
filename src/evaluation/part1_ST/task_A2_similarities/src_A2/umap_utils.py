import hashlib
from pathlib import Path
from typing import Any, Dict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from src_A2.common import ensure_dir, write_dataframe
from src_A2.retrieval import l2_normalize



def _import_umap():
    try:
        import umap  # type: ignore
        return umap
    except Exception as e:  # pragma: no cover
        raise RuntimeError(f"UMAP is not available: {e}")



def _stable_seed(text: str, base: int) -> int:
    return base + int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16) % 10_000



def _subsample_rows(rows: np.ndarray, max_n: int, seed: int) -> np.ndarray:
    rows = np.asarray(rows, dtype=np.int64)
    if rows.size <= max_n:
        return rows
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(rows, size=max_n, replace=False))



def _maybe_fit_pca(X: np.ndarray, cfg: Dict[str, Any]):
    if not bool(cfg["umap"].get("use_pca", True)):
        return None, X
    from sklearn.decomposition import PCA

    n_comp = int(min(cfg["umap"].get("pca_dim", 50), X.shape[0], X.shape[1]))
    if n_comp < 2:
        return None, X
    pca = PCA(n_components=n_comp, random_state=int(cfg["umap"]["random_state"]))
    Xp = pca.fit_transform(X)
    return pca, Xp



def _as_str_array(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=object).astype(str)



def _label_color_map(*label_arrays: np.ndarray) -> Dict[str, Any]:
    """Build a stable label -> color map for one panel (colors are assigned per point)."""
    labels = []
    for values in label_arrays:
        arr = _as_str_array(values)
        if arr.size:
            labels.extend(arr.tolist())

    uniq = sorted(pd.Index(labels).unique().tolist(), key=str)
    if not uniq:
        return {}

    prop_cycle = plt.rcParams.get("axes.prop_cycle")
    colors = prop_cycle.by_key().get("color", []) if prop_cycle is not None else []

    # Use tab20 when the default color cycle is too short
    if len(colors) < len(uniq):
        cmap = plt.get_cmap("tab20", max(1, len(uniq)))
        colors = [cmap(i) for i in range(len(uniq))]

    return {lab: colors[i % len(colors)] for i, lab in enumerate(uniq)}



def _colors_for(labels: np.ndarray, label_to_color: Dict[str, Any]) -> list:
    return [label_to_color[str(lab)] for lab in labels.tolist()]



def _scatter_reference_then_query_panel(
    ax,
    ref_xy: np.ndarray,
    ref_labels: np.ndarray,
    query_xy: np.ndarray,
    query_labels: np.ndarray,
    title: str,
    query_seed: int,
    reference_alpha: float,
    query_alpha: float,
    reference_size: float,
    query_size: float,
    randomize_query_draw_order: bool,
) -> None:
    """
    Draw one UMAP panel: reference cells first (low alpha), then query cells
    (draw order optionally randomized, for the PNG only).
    """
    ref_xy = np.asarray(ref_xy, dtype=np.float32)
    query_xy = np.asarray(query_xy, dtype=np.float32)
    ref_labels = _as_str_array(ref_labels)
    query_labels = _as_str_array(query_labels)

    if ref_xy.ndim != 2 or ref_xy.shape[1] != 2:
        raise ValueError(f"ref_xy must have shape [N, 2], got {ref_xy.shape}")
    if query_xy.ndim != 2 or query_xy.shape[1] != 2:
        raise ValueError(f"query_xy must have shape [N, 2], got {query_xy.shape}")
    if ref_xy.shape[0] != ref_labels.shape[0]:
        raise ValueError(f"ref_xy/ref_labels length mismatch: {ref_xy.shape[0]} vs {ref_labels.shape[0]}")
    if query_xy.shape[0] != query_labels.shape[0]:
        raise ValueError(f"query_xy/query_labels length mismatch: {query_xy.shape[0]} vs {query_labels.shape[0]}")

    label_to_color = _label_color_map(ref_labels, query_labels)

    if ref_xy.shape[0] > 0:
        ax.scatter(
            ref_xy[:, 0],
            ref_xy[:, 1],
            s=float(reference_size),
            alpha=float(reference_alpha),
            c=_colors_for(ref_labels, label_to_color),
            linewidths=0,
            rasterized=True,
        )

    if query_xy.shape[0] > 0:
        order = np.arange(query_xy.shape[0], dtype=np.int64)
        if bool(randomize_query_draw_order) and order.size > 1:
            rng = np.random.default_rng(int(query_seed))
            rng.shuffle(order)

        query_xy_plot = query_xy[order]
        query_labels_plot = query_labels[order]
        ax.scatter(
            query_xy_plot[:, 0],
            query_xy_plot[:, 1],
            s=float(query_size),
            alpha=float(query_alpha),
            c=_colors_for(query_labels_plot, label_to_color),
            linewidths=0,
            rasterized=True,
        )

    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])

    if label_to_color:
        handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                color="none",
                markerfacecolor=color,
                markeredgecolor="none",
                label=lab,
                markersize=4,
            )
            for lab, color in label_to_color.items()
        ]
        ax.legend(handles=handles, fontsize=6, markerscale=2, frameon=False)



def run_umap_for_slide(slide, he_eval, gt_eval, shared_dir: Path, out_dir: Path, cfg: Dict[str, Any], logger) -> None:
    try:
        umap = _import_umap()
    except Exception as e:
        logger.warning("Skipping UMAP for slide=%s because %s", slide.slide_id, e)
        return

    ref_rows = np.asarray(slide.umap_ref_rows, dtype=np.int64) if slide.umap_ref_rows is not None else np.empty((0,), dtype=np.int64)
    if ref_rows.size == 0:
        logger.warning("Empty UMAP sample file for slide=%s", slide.slide_id)
        return

    ref_sample = l2_normalize(slide.ref_z[ref_rows])
    pca, ref_fit = _maybe_fit_pca(ref_sample, cfg)
    reducer = umap.UMAP(
        n_neighbors=int(cfg["umap"]["n_neighbors"]),
        min_dist=float(cfg["umap"]["min_dist"]),
        metric=str(cfg["umap"]["metric"]),
        random_state=int(cfg["umap"]["random_state"]),
        transform_seed=int(cfg["umap"]["random_state"]),
    )
    ref_xy = reducer.fit_transform(ref_fit)

    he_rows = _subsample_rows(
        np.flatnonzero(slide.he_valid),
        int(cfg["umap"]["query_max_cells_per_slide"]),
        _stable_seed(f"{slide.slide_id}_he", int(cfg["umap"]["random_state"])),
    )
    gt_rows = _subsample_rows(
        np.flatnonzero(slide.gt_valid),
        int(cfg["umap"]["query_max_cells_per_slide"]),
        _stable_seed(f"{slide.slide_id}_gt", int(cfg["umap"]["random_state"])),
    )
    he_q = l2_normalize(slide.z_he[he_rows]) if he_rows.size > 0 else np.zeros((0, slide.ref_z.shape[1]), dtype=np.float32)
    gt_q = l2_normalize(slide.z_gt[gt_rows]) if gt_rows.size > 0 else np.zeros((0, slide.ref_z.shape[1]), dtype=np.float32)
    if pca is not None:
        he_q_fit = pca.transform(he_q) if he_q.shape[0] > 0 else he_q
        gt_q_fit = pca.transform(gt_q) if gt_q.shape[0] > 0 else gt_q
    else:
        he_q_fit = he_q
        gt_q_fit = gt_q
    he_xy = reducer.transform(he_q_fit) if he_q_fit.shape[0] > 0 else np.zeros((0, 2), dtype=np.float32)
    gt_xy = reducer.transform(gt_q_fit) if gt_q_fit.shape[0] > 0 else np.zeros((0, 2), dtype=np.float32)

    ensure_dir(out_dir)
    ref_df = pd.DataFrame({
        "kind": "reference",
        "x": ref_xy[:, 0],
        "y": ref_xy[:, 1],
        "label": slide.ref_labels[ref_rows].astype(str),
        "modality": "reference",
    })
    he_df = pd.DataFrame({
        "kind": "he_query",
        "x": he_xy[:, 0] if he_xy.size else [],
        "y": he_xy[:, 1] if he_xy.size else [],
        "label": slide.gt_labels[he_rows].astype(str) if he_rows.size else [],
        "modality": "he",
        "cell_id_int": slide.cells.loc[he_rows, "cell_id_int"].to_numpy(dtype=np.int32, copy=False) if he_rows.size else [],
    })
    gt_df = pd.DataFrame({
        "kind": "gt_query",
        "x": gt_xy[:, 0] if gt_xy.size else [],
        "y": gt_xy[:, 1] if gt_xy.size else [],
        "label": slide.gt_labels[gt_rows].astype(str) if gt_rows.size else [],
        "modality": "gt",
        "cell_id_int": slide.cells.loc[gt_rows, "cell_id_int"].to_numpy(dtype=np.int32, copy=False) if gt_rows.size else [],
    })
    coords = pd.concat([ref_df, he_df, gt_df], axis=0, ignore_index=True, sort=False)
    write_dataframe(coords, out_dir / f"{slide.slide_id}_umap_coords.csv")

    query_sample = pd.concat(
        [
            pd.DataFrame({"path": "he", "cell_id_int": slide.cells.loc[he_rows, "cell_id_int"].to_numpy(dtype=np.int32, copy=False)}) if he_rows.size else pd.DataFrame(columns=["path", "cell_id_int"]),
            pd.DataFrame({"path": "gt", "cell_id_int": slide.cells.loc[gt_rows, "cell_id_int"].to_numpy(dtype=np.int32, copy=False)}) if gt_rows.size else pd.DataFrame(columns=["path", "cell_id_int"]),
        ],
        axis=0,
        ignore_index=True,
    )
    write_dataframe(query_sample, out_dir / f"{slide.slide_id}_umap_query_sample.csv")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    umap_cfg = cfg["umap"]
    reference_alpha = float(umap_cfg.get("plot_reference_alpha", 0.25))
    query_alpha = float(umap_cfg.get("plot_query_alpha", 0.75))
    reference_size = float(umap_cfg.get("plot_reference_point_size", 3.0))
    query_size = float(umap_cfg.get("plot_query_point_size", 4.0))
    randomize_query_draw_order = bool(umap_cfg.get("randomize_query_draw_order", True))

    base_seed = int(umap_cfg["random_state"])
    he_draw_seed = _stable_seed(f"{slide.slide_id}_he_query_draw_order", base_seed)
    gt_draw_seed = _stable_seed(f"{slide.slide_id}_gt_query_draw_order", base_seed)

    ref_modality_labels = np.full(ref_xy.shape[0], "reference", dtype=object)
    he_modality_labels = np.full(he_xy.shape[0], "he", dtype=object)
    gt_modality_labels = np.full(gt_xy.shape[0], "gt", dtype=object)

    ref_mapped_labels = slide.ref_labels[ref_rows].astype(str)
    he_mapped_labels = slide.gt_labels[he_rows].astype(str) if he_rows.size else np.empty((0,), dtype=object)
    gt_mapped_labels = slide.gt_labels[gt_rows].astype(str) if gt_rows.size else np.empty((0,), dtype=object)

    # A: H&E + ref by modality
    _scatter_reference_then_query_panel(
        axes[0, 0],
        ref_xy=ref_xy,
        ref_labels=ref_modality_labels,
        query_xy=he_xy,
        query_labels=he_modality_labels,
        title=f"{slide.slide_id} A: H&E + reference by modality",
        query_seed=he_draw_seed,
        reference_alpha=reference_alpha,
        query_alpha=query_alpha,
        reference_size=reference_size,
        query_size=query_size,
        randomize_query_draw_order=randomize_query_draw_order,
    )

    # B: H&E + ref by mapped label
    _scatter_reference_then_query_panel(
        axes[0, 1],
        ref_xy=ref_xy,
        ref_labels=ref_mapped_labels,
        query_xy=he_xy,
        query_labels=he_mapped_labels,
        title=f"{slide.slide_id} B: H&E + reference by mapped label",
        query_seed=he_draw_seed,
        reference_alpha=reference_alpha,
        query_alpha=query_alpha,
        reference_size=reference_size,
        query_size=query_size,
        randomize_query_draw_order=randomize_query_draw_order,
    )

    # C: GT + ref by modality
    _scatter_reference_then_query_panel(
        axes[1, 0],
        ref_xy=ref_xy,
        ref_labels=ref_modality_labels,
        query_xy=gt_xy,
        query_labels=gt_modality_labels,
        title=f"{slide.slide_id} C: GT + reference by modality",
        query_seed=gt_draw_seed,
        reference_alpha=reference_alpha,
        query_alpha=query_alpha,
        reference_size=reference_size,
        query_size=query_size,
        randomize_query_draw_order=randomize_query_draw_order,
    )

    # D: GT + ref by mapped label
    _scatter_reference_then_query_panel(
        axes[1, 1],
        ref_xy=ref_xy,
        ref_labels=ref_mapped_labels,
        query_xy=gt_xy,
        query_labels=gt_mapped_labels,
        title=f"{slide.slide_id} D: GT + reference by mapped label",
        query_seed=gt_draw_seed,
        reference_alpha=reference_alpha,
        query_alpha=query_alpha,
        reference_size=reference_size,
        query_size=query_size,
        randomize_query_draw_order=randomize_query_draw_order,
    )

    fig.tight_layout()
    fig.savefig(out_dir / f"{slide.slide_id}_umap_panels.png", dpi=180)
    plt.close(fig)
    logger.info("Saved UMAP panels for slide=%s", slide.slide_id)
