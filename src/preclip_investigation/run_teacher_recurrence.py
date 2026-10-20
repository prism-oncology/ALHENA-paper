#!/usr/bin/env python
"""
Geometry of the precomputed ST teacher embeddings (Novae per patch, scConcept per cell) on the train split: cross-slide
KMeans cluster recurrence, cosine strata (raw and centered), cell-type kNN, temperature tables and sampler verdict.

Saved in <ROOT_HistoST> / train / preclip_investigation / <training_dataset_id> / <investigation_id> /
    - report.md, summary.json, config.json: results of both teachers
    - *.csv: tables of the Novae teacher
    - scconcept/*.csv: tables of the scConcept teacher
"""

import os
import sys
import json
import math
import time
import logging
import argparse
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import h5py
from tqdm import tqdm

# Add the project root to sys.path so that 'src' can be imported.
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.constants import get_full_path  # noqa: E402
from src.preclip_investigation import geometry as G  # noqa: E402

logger = logging.getLogger(__name__)

PRECLIP_SCHEMA_VERSION = 3   # output format version
_NORM_EPS = 1e-8


# --------------------------------------------------------------------------- small helpers
def _git_commit():
    try:
        import subprocess
        return subprocess.check_output(
            ["git", "-C", project_root, "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return None


def _split_patch_id(patch_id):
    """{slide_id}_{n} -> (slide_id, int(n)) (underscores allowed in slide_id)."""
    slide_id, suffix = str(patch_id).rsplit("_", 1)
    if not suffix.isdigit():
        raise ValueError(f"Unexpected patch_id (numeric suffix expected): '{patch_id}'")
    return slide_id, int(suffix)


def _list_fold_patch_ids(training_folder, fold, n_min_cells=0):
    """patch_id list of a fold, from cell_count_<fold>[_<n_min_cells>].csv."""
    training_folder = str(training_folder)
    if n_min_cells and int(n_min_cells) > 0:
        csv = os.path.join(training_folder, f"cell_count_{fold}_{int(n_min_cells)}.csv")
    else:
        csv = os.path.join(training_folder, f"cell_count_{fold}.csv")
    if not os.path.isfile(csv):
        raise FileNotFoundError(f"cell-count file not found for fold '{fold}': {csv}")
    return pd.read_csv(csv)["patch_id"].tolist()


def _tissue_of(slide_id):
    """slide_id ``{tissue}_s{n}`` -> tissue (handles multi-word tissues, e.g. ``lymph_node_s0``)."""
    return str(slide_id).rsplit("_", 1)[0]


def _save_json_atomic(path, payload):
    tmp = str(path) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2, default=_json_default)
    os.replace(tmp, path)


def _save_csv_atomic(path, df):
    tmp = str(path) + ".tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def _save_text_atomic(path, text):
    tmp = str(path) + ".tmp"
    with open(tmp, "w") as f:
        f.write(text)
    os.replace(tmp, path)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    return str(o)


def _read_cache_manifest(cache_dir):
    path = os.path.join(str(cache_dir), "manifest.json")
    if os.path.isfile(path):
        with open(path) as f:
            return json.load(f)
    return {}


def _enumerate_slides(training_folder, cache_dir, manifest):
    """Slides of slide_ids.txt (else of the manifest n_patches_per_slide) that have a <slide>.h5 in the cache (._* files skipped)."""
    slide_ids = []
    slides_txt = os.path.join(str(training_folder), "slide_ids.txt")
    if os.path.isfile(slides_txt):
        with open(slides_txt) as f:
            slide_ids = [ln.strip() for ln in f if ln.strip()]
    elif manifest and manifest.get("n_patches_per_slide"):
        slide_ids = list(manifest["n_patches_per_slide"].keys())
    else:
        raise FileNotFoundError(
            f"No slide list: neither {slides_txt} nor manifest['n_patches_per_slide'] available."
        )
    out = []
    for s in slide_ids:
        if s.startswith("._"):
            continue
        if os.path.isfile(os.path.join(str(cache_dir), f"{s}.h5")):
            out.append(s)
        else:
            logger.warning("No st_feat cache for slide '%s' (skipping).", s)
    return out


# --------------------------------------------------------------------------- teacher loaders
def _load_train_st_feat(cache_dir, training_folder, split, n_min_cells, slides, max_patches, seed):
    """
    Novae patch teacher: st_feat of the patches of the split (the cache holds all folds).
    Returns (X [N, 128] raw float32, slide_ids, tissue_ids, inventory_df, raw_norm_stats, counts, None).
    """
    fold_patch_ids = _list_fold_patch_ids(training_folder, split, n_min_cells)
    train_idx_by_slide = {}
    for pid in fold_patch_ids:
        s, idx = _split_patch_id(pid)
        train_idx_by_slide.setdefault(s, []).append(idx)
    for s in train_idx_by_slide:
        train_idx_by_slide[s] = np.asarray(sorted(train_idx_by_slide[s]), dtype=np.int64)

    X_parts, slide_parts, tissue_parts, norm_parts, inv_rows = [], [], [], [], []
    n_requested_total = int(sum(len(v) for v in train_idx_by_slide.values()))
    for s in tqdm(slides, desc="Loading novae st_feat", unit="slide"):
        train_idx = train_idx_by_slide.get(s)
        if train_idx is None or train_idx.size == 0:
            continue
        with h5py.File(os.path.join(str(cache_dir), f"{s}.h5"), "r") as f:
            patch_idx = f["patch_idx"][:].astype(np.int64, copy=False)
            st = f["st_feat"][:].astype(np.float32, copy=False)
        mask = np.isin(patch_idx, train_idx)
        Xs = st[mask]
        norms = np.linalg.norm(Xs, axis=1)
        X_parts.append(Xs)
        slide_parts.append(np.full(Xs.shape[0], s, dtype=object))
        tissue_parts.append(np.full(Xs.shape[0], _tissue_of(s), dtype=object))
        norm_parts.append(norms.astype(np.float32))
        inv_rows.append({"slide": s, "tissue": _tissue_of(s), "n_cached": int(patch_idx.shape[0]),
                         "n_train_requested": int(train_idx.size), "n_train_loaded": int(Xs.shape[0]),
                         "n_zero_norm": int((norms <= _NORM_EPS).sum())})
    if not X_parts:
        raise RuntimeError("No TRAIN patches loaded — check split, n_min_cells and the cache.")
    # No per-cell ids for the Novae patch teacher
    return _finalize_load(X_parts, slide_parts, tissue_parts, norm_parts, inv_rows,
                          n_requested_total, max_patches, seed, unit="patch")


def _load_train_scconcept_cells(scc_dir, cell_ids_sets_path, slides, cells_per_slide, max_cells, seed):
    """
    scConcept cell teacher: cls_emb of the train cells (cell_ids_sets.npz train_only_by_slide), at most
    cells_per_slide per slide. Returns the same tuple as _load_train_st_feat (X [N, 512], cell ids last).
    """
    z = np.load(str(cell_ids_sets_path), allow_pickle=True)
    train_by_slide = z["train_only_by_slide"].item()
    rng = np.random.default_rng(seed)

    X_parts, slide_parts, tissue_parts, norm_parts, inv_rows, cid_parts = [], [], [], [], [], []
    n_requested_total = 0
    for s in tqdm(slides, desc="Loading scConcept cls_emb", unit="slide"):
        train_ids = train_by_slide.get(s)
        h5_path = os.path.join(str(scc_dir), f"{s}.h5")
        if train_ids is None or len(train_ids) == 0 or not os.path.isfile(h5_path):
            continue
        train_ids = np.asarray(train_ids, dtype=np.int64)
        n_requested_total += int(train_ids.size)
        with h5py.File(h5_path, "r") as f:
            cell_id_int = f["cell_id_int"][:].astype(np.int64, copy=False)
            train_pos = np.nonzero(np.isin(cell_id_int, train_ids, assume_unique=True))[0]
            if train_pos.size == 0:
                continue
            if cells_per_slide and train_pos.size > int(cells_per_slide):
                train_pos = np.sort(rng.choice(train_pos, int(cells_per_slide), replace=False))
            Xs = f["cls_emb"][train_pos].astype(np.float32, copy=False)  # sorted fancy read
            cids = cell_id_int[train_pos]                                # keep cell ids for label join
        norms = np.linalg.norm(Xs, axis=1)
        X_parts.append(Xs)
        slide_parts.append(np.full(Xs.shape[0], s, dtype=object))
        tissue_parts.append(np.full(Xs.shape[0], _tissue_of(s), dtype=object))
        norm_parts.append(norms.astype(np.float32))
        cid_parts.append(cids.astype(np.int64, copy=False))
        inv_rows.append({"slide": s, "tissue": _tissue_of(s), "n_cached": int(cell_id_int.shape[0]),
                         "n_train_requested": int(train_ids.size), "n_train_loaded": int(Xs.shape[0]),
                         "n_zero_norm": int((norms <= _NORM_EPS).sum())})
    if not X_parts:
        raise RuntimeError("No TRAIN scConcept cells loaded — check the embeddings dir and npz.")
    return _finalize_load(X_parts, slide_parts, tissue_parts, norm_parts, inv_rows,
                          n_requested_total, max_cells, seed, unit="cell", extra_parts=cid_parts)


def _xenium_str_ids_to_int64(cell_id_strs):
    """Xenium string cell ids ('aaaachba-1') -> int64 cell_id_int of scConcept; None if the format does not match."""
    s = pd.Series(cell_id_strs).astype(str)
    if s.empty:
        return np.empty((0,), dtype=np.int64)
    b = s.to_numpy().astype("S")
    if not np.all(np.char.endswith(b, b"-1")):
        return None
    maxlen = b.dtype.itemsize
    chars = np.frombuffer(b.tobytes(), dtype=np.uint8).reshape(b.shape[0], maxlen)[:, :8]
    digits = chars.astype(np.int16) - ord("a")
    if np.any(digits < 0) or np.any(digits > 15):
        return None
    powers = np.array([16 ** i for i in range(7, -1, -1)], dtype=np.int64)
    return (digits.astype(np.int64) * powers).sum(axis=1) + 1


def _load_celltype_labels(cell_ids, slide_ids, ct_cat_id, training_folder):
    """
    Cell-type labels of the scConcept cells from cID2cT, joined on (slide, cell_id) after conversion of the Xenium
    string ids to int64. Returns (labels aligned to cell_ids, 'UNKNOWN' if unmatched; stats), or (None, stats) if
    no cID2cT dir is found.
    """
    cands = [os.path.join(str(training_folder), "cID2cT", ct_cat_id),
             os.path.join(str(training_folder), "cID2cT")]
    try:
        cands.append(get_full_path("STHELAR_HistoST", "cID2cT", ct_cat_id, as_str=True))
    except Exception:
        pass
    ct_dir = None
    for d in cands:
        if d and os.path.isdir(d) and any(
                fn.startswith("cID2cT_") and fn.endswith(".parquet")
                for fn in os.listdir(d) if not fn.startswith("._")):
            ct_dir = d
            break
    if ct_dir is None:
        logger.warning("No cID2cT dir with cID2cT_*.parquet for ct_cat_id=%s (tried %s) — "
                       "skipping supervised cell-type recovery.", ct_cat_id, cands)
        return None, {"ct_dir": None, "ct_cat_id": ct_cat_id}

    cell_ids = np.asarray(cell_ids, dtype=np.int64)
    slide_ids = np.asarray(slide_ids)
    labels = np.full(cell_ids.shape[0], "UNKNOWN", dtype=object)
    n_slides_ok = 0
    for s in sorted(set(slide_ids.tolist())):
        p = os.path.join(ct_dir, f"cID2cT_{s}.parquet")
        if not os.path.isfile(p):
            logger.warning("cID2cT parquet missing for slide %s (%s).", s, p)
            continue
        df = pd.read_parquet(p, columns=["cell_id", "cell_type"])
        cid_int = _xenium_str_ids_to_int64(df["cell_id"])
        if cid_int is None:                              # already-numeric cell ids
            num = pd.to_numeric(df["cell_id"], errors="coerce").to_numpy()
            ok = np.isfinite(num)
            cid_int = num[ok].astype(np.int64)
            ct_vals = df["cell_type"].astype(str).to_numpy()[ok]
        else:
            ct_vals = df["cell_type"].astype(str).to_numpy()
        m = {int(a): b for a, b in zip(cid_int, ct_vals)}
        sel = np.nonzero(slide_ids == s)[0]
        labels[sel] = [m.get(int(c), "UNKNOWN") for c in cell_ids[sel]]
        n_slides_ok += 1
    n_known = int((labels != "UNKNOWN").sum())
    stats = {"ct_dir": ct_dir, "ct_cat_id": ct_cat_id, "n_slides_with_labels": n_slides_ok,
             "n_cells_labeled": n_known, "n_cells_total": int(labels.shape[0]),
             "label_match_fraction": round(n_known / max(1, labels.shape[0]), 4)}
    logger.info("cID2cT[%s]: matched %d/%d cells (%.1f%%) over %d slides.", ct_cat_id, n_known,
                labels.shape[0], 100 * stats["label_match_fraction"], n_slides_ok)
    return labels, stats


def _finalize_load(X_parts, slide_parts, tissue_parts, norm_parts, inv_rows,
                   n_requested_total, max_rows, seed, unit, extra_parts=None):
    """Concatenate the per-slide loads, apply the optional global cap max_rows, and return a 7-tuple (7th: extra or None)."""
    X = np.concatenate(X_parts, axis=0)
    slide_ids = np.concatenate(slide_parts, axis=0)
    tissue_ids = np.concatenate(tissue_parts, axis=0)
    raw_norms = np.concatenate(norm_parts, axis=0)
    extra = np.concatenate(extra_parts, axis=0) if extra_parts is not None else None
    if max_rows and X.shape[0] > int(max_rows):
        rng = np.random.default_rng(seed)
        sub = rng.choice(X.shape[0], int(max_rows), replace=False)
        X, slide_ids, tissue_ids, raw_norms = X[sub], slide_ids[sub], tissue_ids[sub], raw_norms[sub]
        if extra is not None:
            extra = extra[sub]
    inventory_df = pd.DataFrame(inv_rows)
    raw_norm_stats = {"min": float(np.min(raw_norms)), "p05": float(np.quantile(raw_norms, 0.05)),
                      "median": float(np.median(raw_norms)), "mean": float(np.mean(raw_norms)),
                      "p95": float(np.quantile(raw_norms, 0.95)), "max": float(np.max(raw_norms))}
    counts = {"n_slides": int(inventory_df.shape[0]), "n_train_requested": int(n_requested_total),
              "n_train_loaded": int(X.shape[0]), "unit": unit,
              "n_zero_norm_total": int(inventory_df["n_zero_norm"].sum()),
              "fold_survivor_fraction": round(float(X.shape[0]) / n_requested_total, 4) if n_requested_total else 0.0}
    return X, slide_ids, tissue_ids, inventory_df, raw_norm_stats, counts, extra


_CT_RELABEL = {"same_cluster_same_slide": "same_celltype_same_slide",
               "same_cluster_diff_slide": "same_celltype_diff_slide"}


# --------------------------------------------------------------------------- core (per teacher)
def _analyze_one_embedding(label, unit, X, slide_ids, tissue_ids, inventory_df, raw_norm_stats,
                           counts, out_dir, *, k_list, K_list, P, batch_size, T_grid, fit_subsample,
                           n_init, pair_samples, seed, slide_probe, minibatch, pca_dim,
                           celltype_ids=None, coherence_min=0.3, fit_balance_slide=True,
                           knn_k=15, knn_ref_per_slide=800, knn_query_per_celltype=400):
    """Run the geometry analysis on one teacher embedding, write its CSVs into out_dir and return its results (cell-type analysis if celltype_ids is given)."""
    os.makedirs(out_dir, exist_ok=True)
    logger.info("[%s] L2-normalizing %d rows (%s-level).", label, X.shape[0], unit)
    Xn, keep_mask = G.l2_normalize(X, eps=_NORM_EPS)
    slide_ids = slide_ids[keep_mask]
    tissue_ids = tissue_ids[keep_mask]
    if celltype_ids is not None:
        celltype_ids = np.asarray(celltype_ids)[keep_mask]
    n_total = int(Xn.shape[0])
    counts = dict(counts, n_rows_analyzed=n_total)
    _save_csv_atomic(os.path.join(out_dir, "inventory.csv"), inventory_df)

    # Centered geometry (global mean of the normalized teacher, as in the training soft-target)
    global_mean = Xn.mean(axis=0)
    Xn_c = G.center_by(Xn, global_mean)

    Xcluster = Xn
    if pca_dim and int(pca_dim) > 0:
        from sklearn.decomposition import PCA
        rng = np.random.default_rng(seed)
        fit = Xn[rng.choice(n_total, min(fit_subsample, n_total), replace=False)]
        pca = PCA(n_components=int(pca_dim), whiten=False, random_state=seed).fit(fit)
        Xp = pca.transform(Xn).astype(np.float32)
        Xp /= np.clip(np.linalg.norm(Xp, axis=1, keepdims=True), _NORM_EPS, None)
        Xcluster = Xp

    # Slide-balanced KMeans fit indices (same for every k)
    fit_idx = (G.balanced_subsample(slide_ids, fit_subsample, seed=seed)
               if fit_balance_slide else None)

    ceiling = G.tissue_ceiling(slide_ids, tissue_ids)   # within-tissue ceiling

    # recurrence over k -------------------------------------------------------------------
    ref_k = k_list[len(k_list) // 2]
    gate_K = min(K_list, key=lambda K: abs(K - 32))   # verdict gate: nearest available K to 32
    per_k, labels_ref, cluster_df_ref = [], None, None
    pbar = tqdm(k_list, desc=f"[{label}] KMeans over k", unit="k")
    for k in pbar:
        pbar.set_postfix_str(f"k={k} fitting (slowest step)")
        t0 = time.time()
        labels = G.fit_kmeans(Xcluster, k, fit_subsample=fit_subsample, n_init=n_init,
                              seed=seed, minibatch=minibatch, fit_idx=fit_idx)
        cluster_df = G.cluster_recurrence(labels, slide_ids, tissue_ids, Xn_centered=Xn_c)
        summ = G.recurrence_summary(cluster_df, n_total, ceiling, K_list=K_list,
                                    coherence_min=coherence_min)
        _save_csv_atomic(os.path.join(out_dir, f"clusters_{k}.csv"), cluster_df)
        row = {"k": k, "R_all": round(summ["R_all"], 4),
               "R_within_tissue": round(summ["R"], 4),
               "R_cond_within_tissue": round(summ["R_cond"], 4) if np.isfinite(summ["R_cond"]) else float("nan"),
               "ceiling_within_tissue": round(summ["ceiling"], 4)}
        for K in K_list:
            row[f"N_usable_all_{K}"] = summ["N_usable_all"][K]
            row[f"N_usable_coherent_{K}"] = summ["N_usable_coherent"][K]
            row[f"N_usable_within_tissue_{K}"] = summ["N_usable"][K]
        per_k.append(row)
        tqdm.write(f"  [{label}] k={k}: R_all={summ['R_all']:.3f} "
                   f"N_coherent({gate_K})={summ['N_usable_coherent'].get(gate_K, -1)} "
                   f"(within-tissue R_cond={summ['R_cond']:.3f})  ({time.time() - t0:.0f}s)")
        if k == ref_k:
            labels_ref, cluster_df_ref = labels, cluster_df
    per_k_df = pd.DataFrame(per_k)
    _save_csv_atomic(os.path.join(out_dir, "recurrence_by_k.csv"), per_k_df)

    med_R_all = float(np.median(per_k_df["R_all"]))
    med_R = float(np.median(per_k_df["R_within_tissue"]))
    med_R_cond = float(np.nanmedian(per_k_df["R_cond_within_tissue"]))
    med_N_all_gate = float(np.median(per_k_df[f"N_usable_all_{gate_K}"]))
    med_N_coh_gate = float(np.median(per_k_df[f"N_usable_coherent_{gate_K}"]))
    iqr_R_all = float(np.nanquantile(per_k_df["R_all"], 0.75) - np.nanquantile(per_k_df["R_all"], 0.25))

    # per-tissue recurrent mass at reference k ---------------------------------------------
    rbt_df = G.recurrent_mass_by_tissue(cluster_df_ref, labels_ref, tissue_ids,
                                        coherence_min=coherence_min)
    _save_csv_atomic(os.path.join(out_dir, "recurrent_by_tissue.csv"), rbt_df)
    n_tissues_total = int(rbt_df.shape[0])
    n_tissues_rec = int(rbt_df["recurrent_here"].sum())

    # cosine strata (KMeans clusters): raw and centered -----------------------------------
    logger.info("[%s] Cosine strata from %d random pairs (reference k=%d), raw + centered.",
                label, pair_samples, ref_k)
    strata = G.cosine_pair_stats(Xn, labels_ref, slide_ids, tissue_ids, n_pairs=pair_samples, seed=seed)
    strata_c = G.cosine_pair_stats(Xn_c, labels_ref, slide_ids, tissue_ids, n_pairs=pair_samples, seed=seed)
    _save_csv_atomic(os.path.join(out_dir, "cosine_strata.csv"),
                     pd.DataFrame([{"stratum": s, **v} for s, v in strata.items()]))
    _save_csv_atomic(os.path.join(out_dir, "cosine_strata_centered.csv"),
                     pd.DataFrame([{"stratum": s, **v} for s, v in strata_c.items()]))

    pos, neg = strata["same_cluster_diff_slide"]["median"], strata["random"]["median"]
    same_slide = strata["same_slide"]["median"]
    gap = pos - neg if (np.isfinite(pos) and np.isfinite(neg)) else float("nan")
    batch_dominated = bool(np.isfinite(gap) and gap > 0 and (same_slide - pos) > 0.5 * gap)
    pos_c, neg_c = strata_c["same_cluster_diff_slide"]["median"], strata_c["random"]["median"]
    gap_c = pos_c - neg_c if (np.isfinite(pos_c) and np.isfinite(neg_c)) else float("nan")
    centering = {
        "pos_raw": _r(pos), "neg_raw": _r(neg), "gap_raw": _r(gap),
        "pos_centered": _r(pos_c), "neg_centered": _r(neg_c), "gap_centered": _r(gap_c),
        "abs_neg_centered": _r(abs(neg_c) if np.isfinite(neg_c) else float("nan")),
        "sharpens": bool(np.isfinite(gap) and np.isfinite(gap_c) and gap_c > gap + 0.02),
    }

    # supervised cross-slide cell-type recovery (real labels; scConcept only) -------------
    supervised = None
    if celltype_ids is not None:
        known = celltype_ids != "UNKNOWN"
        n_known = int(known.sum())
        if n_known >= max(200, 10 * knn_k):
            logger.info("[%s] Supervised cross-slide cell-type recovery on %d labeled cells.",
                        label, n_known)
            Xk, slk, tik, ctk = Xn[known], slide_ids[known], tissue_ids[known], celltype_ids[known]
            knn_df, knn_macro = G.cross_slide_celltype_knn(
                Xk, slk, ctk, k=knn_k, n_ref_per_slide=knn_ref_per_slide,
                n_query_per_celltype=knn_query_per_celltype, seed=seed)
            _save_csv_atomic(os.path.join(out_dir, "celltype_cross_slide_knn.csv"), knn_df)
            ct_str = G.cosine_pair_stats(Xk, ctk, slk, tik, n_pairs=pair_samples, seed=seed)
            ct_str_c = G.cosine_pair_stats(G.center_by(Xk, global_mean), ctk, slk, tik,
                                           n_pairs=pair_samples, seed=seed)
            ct_str = {_CT_RELABEL.get(kk, kk): v for kk, v in ct_str.items()}
            ct_str_c = {_CT_RELABEL.get(kk, kk): v for kk, v in ct_str_c.items()}
            _save_csv_atomic(os.path.join(out_dir, "cosine_strata_celltype.csv"),
                             pd.DataFrame([{"stratum": s, **v} for s, v in ct_str.items()]))
            _save_csv_atomic(os.path.join(out_dir, "cosine_strata_celltype_centered.csv"),
                             pd.DataFrame([{"stratum": s, **v} for s, v in ct_str_c.items()]))
            ct_pos = ct_str["same_celltype_diff_slide"]["median"]
            ct_pos_c = ct_str_c["same_celltype_diff_slide"]["median"]
            ct_neg, ct_neg_c = ct_str["random"]["median"], ct_str_c["random"]["median"]
            supervised = {
                "n_cells_known": n_known,
                "celltype_knn": knn_macro,
                "celltype_knn_records": knn_df.to_dict("records"),
                "celltype_cosine_strata": {s: {"median": v["median"], "n": v["n"]} for s, v in ct_str.items()},
                "celltype_cosine_strata_centered": {s: {"median": v["median"], "n": v["n"]} for s, v in ct_str_c.items()},
                "celltype_centering": {
                    "pos_raw": _r(ct_pos), "neg_raw": _r(ct_neg),
                    "pos_centered": _r(ct_pos_c), "neg_centered": _r(ct_neg_c),
                    "gap_raw": _r(ct_pos - ct_neg if np.isfinite(ct_pos) and np.isfinite(ct_neg) else float("nan")),
                    "gap_centered": _r(ct_pos_c - ct_neg_c if np.isfinite(ct_pos_c) and np.isfinite(ct_neg_c) else float("nan"))},
            }
        else:
            logger.warning("[%s] Only %d labeled cells — skipping supervised recovery.", label, n_known)

    # per-slide heterogeneity (reference k) ----------------------------------------------
    het_rows = []
    for s in inventory_df["slide"]:
        m = slide_ids == s
        if not m.any():
            continue
        het_rows.append({"slide": s, "tissue": _tissue_of(s), "n_rows": int(m.sum()),
                         "occupied_cluster_hill": round(G.occupied_cluster_hill(labels_ref[m]), 3),
                         "effective_rank": round(G.effective_rank(Xn[m], seed=seed), 3)})
    het_df = pd.DataFrame(het_rows)
    _save_csv_atomic(os.path.join(out_dir, "per_slide_heterogeneity.csv"), het_df)

    # temperature diagnostic on the raw geometry (training T: select_training_params.py) -------
    K_synth = max(2, int(batch_size) // int(P))
    idx, synth_labels, P_used = G.build_synthetic_batch(labels_ref, slide_ids, cluster_df_ref,
                                                        int(P), K_synth, seed=seed)
    temp_df = G.temperature_table(Xn[idx], synth_labels, T_grid) if idx.size else pd.DataFrame(
        columns=["T", "perplexity", "in_niche_mass"])
    _save_csv_atomic(os.path.join(out_dir, "temperature_table.csv"), temp_df)
    T_star = G.suggest_temperature(pos, neg, int(P))
    suggested_T = float("nan")
    if not temp_df.empty:
        suggested_T = float(temp_df.iloc[(temp_df["perplexity"] - K_synth).abs().argmin()]["T"])

    probe_records = None
    if slide_probe:
        logger.info("[%s] Within-tissue slide probe (batch-leakage check)...", label)
        probe_df = G.within_tissue_slide_probe(Xn, slide_ids, tissue_ids, seed=seed)
        if probe_df is not None and not probe_df.empty:
            _save_csv_atomic(os.path.join(out_dir, "slide_probe.csv"), probe_df)
            probe_records = probe_df.to_dict("records")

    verdict, reasons = G.decide_verdict(med_R_all, med_N_coh_gate, n_tissues_rec, n_tissues_total,
                                        iqr_R_all, gate_K=gate_K)
    logger.info("[%s] Sampler verdict: %s (%s)", label, verdict, "; ".join(reasons))

    return {
        "label": label, "granularity": unit, "verdict": verdict, "verdict_reasons": reasons,
        "verdict_kind": "sampler_feasibility", "reference_k": ref_k,
        "headline": {
            # all-slide recurrence
            "median_R_all": _r(med_R_all),
            "median_N_usable_all_gate": int(med_N_all_gate) if np.isfinite(med_N_all_gate) else None,
            "median_N_usable_coherent_gate": int(med_N_coh_gate) if np.isfinite(med_N_coh_gate) else None,
            "gate_K": int(gate_K), "iqr_R_all": _r(iqr_R_all),
            "n_tissues_recurrent": int(n_tissues_rec), "n_tissues_total": int(n_tissues_total),
            # within-tissue recurrence (dominant tissue)
            "median_R_within_tissue": _r(med_R), "median_R_cond_within_tissue": _r(med_R_cond),
            "ceiling_within_tissue": _r(ceiling),
            # cosine geometry (KMeans clusters)
            "cosine_pos_same_cluster_diff_slide": _r(pos), "cosine_neg_random": _r(neg),
            "cosine_same_slide": _r(same_slide), "cosine_gap_pos_minus_neg": _r(gap),
            "batch_dominated": batch_dominated,
            # temperature (raw geometry)
            "T_star_analytic": _r(T_star), "suggested_target_temperature_RAW_geometry": _r(suggested_T),
            "synthetic_batch_P_used": int(P_used), "synthetic_batch_K": int(K_synth),
        },
        "centering": centering,
        "supervised": supervised,
        "recurrent_by_tissue": rbt_df.to_dict("records"),
        "per_k": per_k,
        "cosine_strata": {s: {"median": v["median"], "n": v["n"]} for s, v in strata.items()},
        "cosine_strata_centered": {s: {"median": v["median"], "n": v["n"]} for s, v in strata_c.items()},
        "per_slide_heterogeneity": het_df.to_dict("records"),
        "temperature_table": temp_df.to_dict("records"),
        "slide_probe": probe_records,
        "counts": counts, "raw_norm_stats": raw_norm_stats,
    }


def _r(x, nd=4):
    return round(float(x), nd) if (x is not None and np.isfinite(x)) else None


# --------------------------------------------------------------------------- orchestration
def analyze(*, cache_dir, training_folder, output_dir, dataset_id, training_dataset_id,
            novae_training_id, investigation_id, split="train", n_min_cells=0,
            k_list=(15, 20, 25, 50, 100, 200, 400, 800), K_list=(16, 32, 64), P=32, batch_size=128,
            T_grid=(0.02, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.5), fit_subsample=200_000, n_init=3,
            pair_samples=2_000_000, seed=0, slide_probe=False, minibatch=False, pca_dim=0,
            max_patches=0, analyze_scconcept=True, scconcept_training_id=None,
            scconcept_dir=None, cell_ids_sets_path=None, scconcept_cells_per_slide=15000,
            scconcept_max_cells=0, ct_cat_id="ct_1", coherence_min=0.3, fit_balance_slide=True,
            knn_k=15, knn_ref_per_slide=800, knn_query_per_celltype=400, overwrite=False):
    """Run the analysis of both teachers on explicit paths and write all outputs."""
    cache_dir, training_folder, output_dir = str(cache_dir), str(training_folder), str(output_dir)
    k_list = sorted(int(k) for k in k_list)
    K_list = [int(k) for k in K_list]

    sentinel = os.path.join(output_dir, "report.md")
    if os.path.exists(sentinel) and not overwrite:
        mtime = datetime.fromtimestamp(os.path.getmtime(sentinel)).isoformat(timespec="seconds")
        raise FileExistsError(
            f"[ERROR] report.md already exists at '{sentinel}' (mtime={mtime}). "
            f"Pass --overwrite, or choose a new --investigation_id.")
    os.makedirs(output_dir, exist_ok=True)

    manifest = _read_cache_manifest(cache_dir)
    common = dict(k_list=k_list, K_list=K_list, P=P, batch_size=batch_size, T_grid=T_grid,
                  fit_subsample=fit_subsample, n_init=n_init, pair_samples=pair_samples, seed=seed,
                  slide_probe=slide_probe, minibatch=minibatch, pca_dim=pca_dim,
                  coherence_min=coherence_min, fit_balance_slide=fit_balance_slide,
                  knn_k=knn_k, knn_ref_per_slide=knn_ref_per_slide,
                  knn_query_per_celltype=knn_query_per_celltype)

    # ---- Novae patch teacher (top-level outputs) ------------------------------------------
    slides = _enumerate_slides(training_folder, cache_dir, manifest)
    Xn, sl, ti, inv, rns, cnt, _ = _load_train_st_feat(
        cache_dir, training_folder, split, n_min_cells, slides, max_patches, seed)
    novae_block = _analyze_one_embedding("novae_patch", "patch", Xn, sl, ti, inv, rns, cnt,
                                         output_dir, **common)   # celltype_ids=None (per-patch)
    del Xn, sl, ti

    # ---- scConcept cell teacher (scconcept/ subfolder) ------------------------------------
    scc_block, scc_used = None, {}
    if analyze_scconcept:
        scc_training_id = scconcept_training_id or manifest.get("scconcept_training_id")
        if scconcept_dir is None and scc_training_id:
            scconcept_dir = get_full_path("train", "trainings", "trainings_scConcept",
                                          scc_training_id, "embeddings", as_str=True)
        if cell_ids_sets_path is None:
            cell_ids_sets_path = os.path.join(training_folder, "cell_ids_sets.npz")
        scc_used = {"scconcept_training_id": scc_training_id, "scconcept_dir": scconcept_dir,
                    "cell_ids_sets_path": cell_ids_sets_path,
                    "cells_per_slide": scconcept_cells_per_slide, "ct_cat_id": ct_cat_id}
        if not (scconcept_dir and os.path.isdir(str(scconcept_dir)) and os.path.isfile(str(cell_ids_sets_path))):
            logger.warning("scConcept inputs not found (dir=%s, npz=%s) — skipping scConcept.",
                           scconcept_dir, cell_ids_sets_path)
        else:
            scc_slides = [s for s in os.listdir(str(scconcept_dir))
                          if s.endswith(".h5") and not s.startswith("._")]
            scc_slides = sorted(s[:-3] for s in scc_slides)
            Xc, sc, tc, invc, rnsc, cntc, cid_c = _load_train_scconcept_cells(
                scconcept_dir, cell_ids_sets_path, scc_slides, scconcept_cells_per_slide,
                scconcept_max_cells, seed)
            # Real per-cell cell-type labels (cID2cT) for the supervised cross-slide recovery.
            ct_labels, ct_label_stats = _load_celltype_labels(cid_c, sc, ct_cat_id, training_folder)
            scc_used["celltype_labels"] = ct_label_stats
            scc_block = _analyze_one_embedding(
                "scconcept_cell", "cell", Xc, sc, tc, invc, rnsc, cntc,
                os.path.join(output_dir, "scconcept"), celltype_ids=ct_labels, **common)
            del Xc, sc, tc

    # ---- combined summary.json (Novae top-level + scConcept block) ------------------------
    params = dict(split=split, n_min_cells=int(n_min_cells), k_list=k_list, K_list=K_list, P=int(P),
                  batch_size=int(batch_size), T_grid=list(T_grid), fit_subsample=int(fit_subsample),
                  n_init=int(n_init), pair_samples=int(pair_samples), seed=int(seed),
                  slide_probe=bool(slide_probe), minibatch=bool(minibatch), pca_dim=int(pca_dim),
                  max_patches=int(max_patches), analyze_scconcept=bool(analyze_scconcept),
                  ct_cat_id=ct_cat_id, coherence_min=float(coherence_min),
                  fit_balance_slide=bool(fit_balance_slide), knn_k=int(knn_k),
                  knn_ref_per_slide=int(knn_ref_per_slide), knn_query_per_celltype=int(knn_query_per_celltype),
                  reference_k=novae_block["reference_k"], **{f"scconcept_{k}": v for k, v in scc_used.items()})
    summary = {
        "verdict_kind": "sampler_feasibility",
        "verdict": novae_block["verdict"], "verdict_reasons": novae_block["verdict_reasons"],
        "headline": novae_block["headline"], "per_k": novae_block["per_k"],
        "recurrent_by_tissue": novae_block["recurrent_by_tissue"],
        "cosine_strata": novae_block["cosine_strata"], "counts": novae_block["counts"],
        "raw_norm_stats": novae_block["raw_norm_stats"],
        "novae_centering": novae_block["centering"],
        "scconcept": (None if scc_block is None else {
            "verdict": scc_block["verdict"], "verdict_reasons": scc_block["verdict_reasons"],
            "granularity": scc_block["granularity"], "reference_k": scc_block["reference_k"],
            "headline": scc_block["headline"], "per_k": scc_block["per_k"],
            "recurrent_by_tissue": scc_block["recurrent_by_tissue"],
            "cosine_strata": scc_block["cosine_strata"],
            "cosine_strata_centered": scc_block["cosine_strata_centered"],
            "centering": scc_block["centering"], "supervised": scc_block["supervised"],
            "counts": scc_block["counts"], "raw_norm_stats": scc_block["raw_norm_stats"]}),
        "params": params,
    }
    _save_json_atomic(os.path.join(output_dir, "summary.json"), summary)

    config = {
        "schema_version": PRECLIP_SCHEMA_VERSION, "created_utc": datetime.now(timezone.utc).isoformat(),
        "created_by": "run_teacher_recurrence.py", "git_commit": _git_commit(),
        "dataset_id": dataset_id, "training_dataset_id": training_dataset_id,
        "novae_training_id": novae_training_id, "investigation_id": investigation_id,
        "cache_dir": cache_dir, "cache_fingerprint": manifest.get("fingerprint"),
        "cache_dim": manifest.get("dim"), "scconcept_training_id": manifest.get("scconcept_training_id"),
        "scconcept_model_name": manifest.get("scconcept_model_name"),
        "scconcept_model_version": manifest.get("scconcept_model_version"),
        "parameters": params, "novae_counts": novae_block["counts"],
        "scconcept_counts": (scc_block["counts"] if scc_block else None),
    }
    _save_json_atomic(os.path.join(output_dir, "config.json"), config)
    _save_text_atomic(sentinel, _build_report_md(summary, config, novae_block, scc_block))
    logger.info("Wrote outputs to %s", output_dir)
    return summary


# --------------------------------------------------------------------------- report
def _fmt(x, nd=3):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "n/a"
    return f"{x:.{nd}f}" if isinstance(x, float) else str(x)


def _recurrence_section(block, K_list):
    h = block["headline"]
    gk = h["gate_K"]
    L = [f"- **R_all = {_fmt(h['median_R_all'])}** — fraction of rows in all-slide-recurrent "
         f"clusters (S_eff over slides of any tissue ≥ 2), used by the sampler verdict.\n",
         f"- **cross-slide recurrence in {h['n_tissues_recurrent']}/{h['n_tissues_total']} tissues** "
         f"(spread criterion; see the per-tissue table below)\n",
         f"- N_coherent({gk}) = {_fmt(h['median_N_usable_coherent_gate'])} coherent recurrent niches "
         f"≥{gk} rows (of N_all({gk}) = {_fmt(h['median_N_usable_all_gate'])}) · "
         f"IQR(R_all) = {_fmt(h['iqr_R_all'])} · reference k = {block['reference_k']}\n",
         f"- *secondary (within-tissue niche, dominant-tissue-restricted): R_cond = "
         f"{_fmt(h['median_R_cond_within_tissue'])} — not a cross-slide recurrence "
         f"(single-slide tissues are excluded).*\n\n",
         "| k | R_all | " + " | ".join(f"N_all_{K}" for K in K_list) + " | "
         + " | ".join(f"N_coh_{K}" for K in K_list) + " | R_cond(within-tissue) |\n",
         "|---|---|" + "---|" * (2 * len(K_list)) + "---|\n"]
    for r in block["per_k"]:
        L.append(f"| {r['k']} | {_fmt(r['R_all'])} | "
                 + " | ".join(str(r[f'N_usable_all_{K}']) for K in K_list) + " | "
                 + " | ".join(str(r[f'N_usable_coherent_{K}']) for K in K_list) + " | "
                 + f"{_fmt(r['R_cond_within_tissue'])} |\n")
    return L


def _tissue_spread_section(block):
    L = ["\n### Cross-slide recurrence by tissue\n\n",
         "*Fraction of each tissue's rows in an all-slide-recurrent (and coherent) cluster.*\n\n",
         "| tissue | n_rows | frac_recurrent_all | frac_recurrent_coherent | sampler-usable |\n",
         "|---|---|---|---|---|\n"]
    for r in block["recurrent_by_tissue"]:
        L.append(f"| {r['tissue']} | {int(r['n_rows'])} | {_fmt(r['frac_in_recurrent_all'])} | "
                 f"{_fmt(r['frac_in_recurrent_coherent'])} | {'yes' if r['recurrent_here'] else 'no'} |\n")
    return L


def _supervised_section(block):
    sup = block.get("supervised")
    if not sup:
        return []
    m = sup["celltype_knn"]
    cc = sup["celltype_centering"]
    L = ["\n### Supervised cross-slide cell-type recovery (real cID2cT labels)\n\n",
         f"- **macro cross-slide kNN purity = {_fmt(m.get('macro_purity'))}** vs chance "
         f"{_fmt(m.get('macro_chance'))} → **excess {_fmt(m.get('macro_excess'))}** "
         f"(k={m.get('k')}, {m.get('n_celltypes')} cell types, {sup['n_cells_known']} labeled cells). "
         f"High excess ⇒ the teacher places the same cell type from different slides together.\n\n",
         "| cell type | n | cross-slide kNN purity | chance | excess |\n|---|---|---|---|---|\n"]
    for r in sup["celltype_knn_records"]:
        L.append(f"| {r['celltype']} | {int(r['n_query'])} | {_fmt(r['cross_slide_knn_purity'])} | "
                 f"{_fmt(r['chance_prevalence'])} | {_fmt(r['excess_over_chance'])} |\n")
    L.append("\n**Real-label centering test** (same-celltype/diff-slide vs random):\n\n")
    L.append("| | positives | negatives | gap |\n|---|---|---|---|\n")
    L.append(f"| raw | {_fmt(cc['pos_raw'])} | {_fmt(cc['neg_raw'])} | {_fmt(cc['gap_raw'])} |\n")
    L.append(f"| centered | {_fmt(cc['pos_centered'])} | {_fmt(cc['neg_centered'])} | {_fmt(cc['gap_centered'])} |\n")
    return L


def _cosine_section(block):
    h, c = block["headline"], block["centering"]
    L = [f"- positives (same-cluster/diff-slide): **{_fmt(h['cosine_pos_same_cluster_diff_slide'])}** · "
         f"negatives (random): {_fmt(h['cosine_neg_random'])} · gap = {_fmt(h['cosine_gap_pos_minus_neg'])}\n",
         f"- same-slide (batch confound): {_fmt(h['cosine_same_slide'])} · "
         f"batch-dominated = {h['batch_dominated']}\n\n",
         "**Centering test** (does removing the shared mean sharpen the geometry?)\n\n",
         "| | positives | negatives | gap |\n|---|---|---|---|\n",
         f"| raw | {_fmt(c['pos_raw'])} | {_fmt(c['neg_raw'])} | {_fmt(c['gap_raw'])} |\n",
         f"| centered | {_fmt(c['pos_centered'])} | {_fmt(c['neg_centered'])} | {_fmt(c['gap_centered'])} |\n",
         f"\n→ **centering {'SHARPENS' if c['sharpens'] else 'does NOT sharpen'}** the contrast "
         f"(gap {_fmt(c['gap_raw'])} → {_fmt(c['gap_centered'])}; |neg| after centering "
         f"{_fmt(c['abs_neg_centered'])}).\n"]
    return L


def _temperature_section(block):
    h = block["headline"]
    L = [f"- analytic anchor T* = {_fmt(h['T_star_analytic'])} · "
         f"suggested T (raw geometry) = {_fmt(h['suggested_target_temperature_RAW_geometry'])} "
         f"(synthetic batch P={h['synthetic_batch_P_used']} × K={h['synthetic_batch_K']})\n",
         "- raw-geometry T; the training T (centered target) comes from "
         "`select_training_params.py` (centered precision curve).\n"]
    if block["temperature_table"]:
        L.append("\n| T | perplexity | in-niche mass |\n|---|---|---|\n")
        for r in block["temperature_table"]:
            L.append(f"| {_fmt(r['T'])} | {_fmt(r['perplexity'])} | {_fmt(r['in_niche_mass'])} |\n")
    return L


def _heterogeneity_line(block):
    hh = [r["occupied_cluster_hill"] for r in block["per_slide_heterogeneity"]]
    if not hh:
        return "- (no per-slide heterogeneity)\n"
    return (f"- occupied-niche Hill number across slides: min {_fmt(min(hh))}, "
            f"median {_fmt(float(np.median(hh)))}, max {_fmt(max(hh))} "
            f"({max(hh) / max(min(hh), 1e-9):.1f}× spread)\n")


def _teacher_report(block, K_list, title):
    c = block["centering"]
    L = [f"\n## {title}\n",
         f"- **Sampler feasibility: {block['verdict']}** ({'; '.join(block['verdict_reasons'])})\n",
         f"- **Centering: {'SHARPENS' if c['sharpens'] else 'does NOT sharpen'}** the target "
         f"(random-negative cosine {_fmt(c['neg_raw'])} → {_fmt(c['neg_centered'])})\n",
         f"*granularity: {block['granularity']} · "
         f"{block['counts']['n_rows_analyzed']} rows · {block['counts']['n_slides']} slides · "
         f"{block['counts']['n_zero_norm_total']} zero-norm dropped*\n",
         "\n### Recurrence (all-slide)\n"]
    L += _recurrence_section(block, K_list)
    L += _tissue_spread_section(block)
    L += _supervised_section(block)
    L.append("\n### Cosine geometry (biology vs batch) + centering\n")
    L += _cosine_section(block)
    L.append("\n### Temperature\n")
    L += _temperature_section(block)
    L.append("\n### Per-slide heterogeneity\n")
    L.append(_heterogeneity_line(block))
    rns = block["raw_norm_stats"]
    L.append(f"- raw norms: median {_fmt(rns['median'])} (p05 {_fmt(rns['p05'])}, p95 {_fmt(rns['p95'])})\n")
    if block["slide_probe"]:
        L.append("\n### Within-tissue slide probe (batch leakage)\n\n")
        L.append("| tissue | n_slides | chance | balanced_acc | excess |\n|---|---|---|---|---|\n")
        for r in block["slide_probe"]:
            L.append(f"| {r['tissue']} | {int(r['n_slides'])} | {_fmt(r['chance'])} | "
                     f"{_fmt(r['balanced_accuracy'])} | {_fmt(r['excess_over_chance'])} |\n")
    return L


def _build_report_md(summary, config, novae_block, scc_block):
    K_list = config["parameters"]["K_list"]
    L = [f"# Pre-CLIP teacher-geometry diagnostic — `{config['investigation_id']}`\n",
         f"- teacher: novae `{config['novae_training_id']}` · scConcept "
         f"`{config.get('scconcept_model_name')}` `{config.get('scconcept_model_version')}` "
         f"· cache fingerprint `{config.get('cache_fingerprint')}`\n",
         f"- dataset: `{config['training_dataset_id']}` · split `{config['parameters']['split']}`\n"]
    L.append("\nTwo analyses per teacher: **(1) sampler feasibility** (coherent cross-slide recurrence "
             "spread across tissues) and **(2) centering** (effect of removing the global mean on the "
             "soft-target geometry).\n")
    L += _teacher_report(novae_block, K_list, "Novae teacher (patch branch)")
    if scc_block is not None:
        L += _teacher_report(scc_block, K_list, "scConcept teacher (cell branch)")
    else:
        L.append("\n## scConcept teacher (cell branch)\n- not analyzed (inputs unavailable or disabled).\n")
    L.append("\n*Heuristic verdicts; thresholds and full numbers in summary.json.*\n")
    return "".join(L)


# --------------------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser(
        description="Pre-CLIP geometry analysis of the ST teachers (Novae patch, scConcept cell).")
    parser.add_argument("--dataset_id", type=str, default="ds_ct1_20x_224_o112")
    parser.add_argument("--training_dataset_id", type=str, default="ds_ct1_20x_224_o112_1")
    parser.add_argument("--novae_training_id", type=str, default="training_4",
                        help="Novae training id whose st_feat cache is analyzed (e.g. training_4).")
    parser.add_argument("--investigation_id", type=str, default="preclip_investigation_1",
                        help="Output id (one per parameter set).")
    parser.add_argument("--split", type=str, default="train", help="Fold to analyze (default train).")
    parser.add_argument("--n_min_cells", type=int, default=0, help="Match training's min-cell filter.")
    parser.add_argument("--k_list", type=int, nargs="+", default=[15, 20, 25, 50, 100, 200, 400, 800],
                        help="KMeans cluster counts (the verdict uses the median over k).")
    parser.add_argument("--K_list", type=int, nargs="+", default=[16, 32, 64],
                        help="Per-niche thresholds for N_usable(K) (verdict uses N_usable(32)).")
    parser.add_argument("--P", type=int, default=32, help="Niches per batch, for the temperature diagnostic.")
    parser.add_argument("--batch_size", type=int, default=128, help="K_synth = batch_size // P.")
    parser.add_argument("--T_grid", type=float, nargs="+",
                        default=[0.02, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.5])
    parser.add_argument("--fit_subsample", type=int, default=200_000, help="Rows to fit KMeans on.")
    parser.add_argument("--n_init", type=int, default=3, help="KMeans n_init.")
    parser.add_argument("--pair_samples", type=int, default=2_000_000, help="Random pairs for cosine strata.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--slide_probe", action="store_true", help="Also run the within-tissue slide probe.")
    parser.add_argument("--minibatch", action="store_true", help="Use MiniBatchKMeans.")
    parser.add_argument("--pca_dim", type=int, default=0, help="If >0, non-whitening PCA before clustering.")
    parser.add_argument("--max_patches", type=int, default=0, help="If >0, cap Novae patches (large caches).")
    parser.add_argument("--skip_scconcept", action="store_true",
                        help="Skip the scConcept cell-branch analysis (Novae patch branch only).")
    parser.add_argument("--scconcept_training_id", type=str, default=None,
                        help="scConcept training id (default: read from the Novae cache manifest).")
    parser.add_argument("--scconcept_cells_per_slide", type=int, default=15000,
                        help="Max train cells per slide for scConcept.")
    parser.add_argument("--scconcept_max_cells", type=int, default=0,
                        help="If >0, global cap on total scConcept cells analyzed.")
    parser.add_argument("--ct_cat_id", type=str, default="ct_1",
                        help="Cell-type category id for the supervised cross-slide recovery "
                             "(cID2cT/{ct_cat_id}/cID2cT_{slide}.parquet).")
    parser.add_argument("--coherence_min", type=float, default=0.3,
                        help="Min centered coherence of a recurrent cluster to count as usable.")
    parser.add_argument("--no_fit_balance_slide", action="store_true",
                        help="Disable slide-balanced KMeans fitting (default: balanced).")
    parser.add_argument("--knn_k", type=int, default=15,
                        help="k for the supervised cross-slide cell-type kNN recovery.")
    parser.add_argument("--knn_ref_per_slide", type=int, default=800,
                        help="Reference cells per slide for the kNN bank.")
    parser.add_argument("--knn_query_per_celltype", type=int, default=400,
                        help="Query cells per cell type for the kNN recovery.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite an existing investigation_id.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                        handlers=[logging.StreamHandler()])

    training_folder = get_full_path("train", "training_datasets", args.dataset_id,
                                    args.training_dataset_id, as_str=True)
    cache_dir = get_full_path("train", "training_datasets", args.dataset_id, args.training_dataset_id,
                              "novae_teacher_cache", args.novae_training_id, as_str=True, scratch=True)
    output_dir = get_full_path("train", "preclip_investigation", args.training_dataset_id,
                               args.investigation_id, as_str=True)

    analyze(cache_dir=cache_dir, training_folder=training_folder, output_dir=output_dir,
            dataset_id=args.dataset_id, training_dataset_id=args.training_dataset_id,
            novae_training_id=args.novae_training_id, investigation_id=args.investigation_id,
            split=args.split, n_min_cells=args.n_min_cells, k_list=args.k_list, K_list=args.K_list,
            P=args.P, batch_size=args.batch_size, T_grid=args.T_grid, fit_subsample=args.fit_subsample,
            n_init=args.n_init, pair_samples=args.pair_samples, seed=args.seed,
            slide_probe=args.slide_probe, minibatch=args.minibatch, pca_dim=args.pca_dim,
            max_patches=args.max_patches, analyze_scconcept=not args.skip_scconcept,
            scconcept_training_id=args.scconcept_training_id,
            scconcept_cells_per_slide=args.scconcept_cells_per_slide,
            scconcept_max_cells=args.scconcept_max_cells, ct_cat_id=args.ct_cat_id,
            coherence_min=args.coherence_min, fit_balance_slide=not args.no_fit_balance_slide,
            knn_k=args.knn_k, knn_ref_per_slide=args.knn_ref_per_slide,
            knn_query_per_celltype=args.knn_query_per_celltype, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
