#!/usr/bin/env python
"""
Selection of the head-cap sampling cap C (per slide and per tissue) and of the soft-target temperatures (Novae patch,
scConcept token) from the centered ST teacher geometry and simulated batches, before training.

Saved in <ROOT_HistoST> / train / preclip_investigation / <training_dataset_id> / <investigation_id> / param_selection /
    - report.md, summary.json, config.json: recommended values
    - headcap_slide.csv, headcap_tissue.csv: cap C per max-oversampling tolerance
    - temperature_novae.csv, [temperature_scconcept.csv]: soft-target masses and precision per T
"""

import os
import io
import sys
import math
import zipfile
import hashlib
import logging
import argparse
from datetime import datetime, timezone

import numpy as np
import h5py
from scipy.sparse import load_npz

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.constants import get_full_path  # noqa: E402
from src.preclip_investigation import geometry as G  # noqa: E402
from src.preclip_investigation.run_teacher_recurrence import (  # noqa: E402  (internal reuse)
    _load_train_st_feat, _enumerate_slides, _read_cache_manifest, _tissue_of, _git_commit,
    _split_patch_id, _list_fold_patch_ids, _save_json_atomic, _save_csv_atomic, _save_text_atomic,
    _NORM_EPS,
)

logger = logging.getLogger(__name__)
PARAM_SELECT_SCHEMA_VERSION = 3   # output format version


# --------------------------------------------------------------------------- small helpers
def _center(X):
    Xn, keep = G.l2_normalize(X, eps=_NORM_EPS)
    return G.center_then_normalize(Xn), keep


def _counts(group_of_row):
    u, c = np.unique(np.asarray(group_of_row), return_counts=True)
    return {str(t): int(n) for t, n in zip(u, c)}


def _stable_seed(patch_id, seed):
    h = hashlib.md5(f"{patch_id}:{seed}".encode()).digest()
    return int.from_bytes(h[:8], "little")


def _patch_cell_ids(mask_bytes):
    """Unique non-zero int cell ids in a patch's nucleus mask npz."""
    m = load_npz(io.BytesIO(mask_bytes))
    if m.nnz == 0:
        return np.array([], dtype=np.int32)
    return np.unique(m.data.astype(np.int32, copy=False))


def _fetch_cls_emb(h5, cid_int_sorted, cell_ids):
    """cls_emb [n, D] of the cell ids present in the scConcept h5 (searchsorted); ids without embedding are dropped."""
    D = int(h5["cls_emb"].shape[1])
    ids = np.asarray(cell_ids, dtype=np.int32)
    if ids.size == 0:
        return np.zeros((0, D), np.float32)
    idx = np.searchsorted(cid_int_sorted, ids)
    idxc = np.clip(idx, 0, cid_int_sorted.shape[0] - 1)
    ok = (idx < cid_int_sorted.shape[0]) & (cid_int_sorted[idxc] == ids)
    idx = idx[ok]
    if idx.size == 0:
        return np.zeros((0, D), np.float32)
    u, inv = np.unique(idx.astype(np.int64, copy=False), return_inverse=True)
    return h5["cls_emb"][u].astype(np.float32)[inv]


def _headcap_search(group_of_row, tol_grid, rec_tol):
    """Head-cap sweep for one grouping (slide or tissue). Returns (counts, rows-for-CSV, rec_C)."""
    counts = _counts(group_of_row)
    rows = []
    for tol in tol_grid:
        C = G.recommend_headcap(counts, tol)
        mtr = G.headcap_metrics(counts, C)
        rows.append({"tol_max_oversampling": round(float(tol), 3), **{k: v for k, v in mtr.items()
                     if k != "capped_groups"}, "capped_groups": ";".join(mtr["capped_groups"])})
    return counts, rows, G.recommend_headcap(counts, rec_tol)


def _induced_tissue_shares(slide_counts, slide_to_tissue, C):
    """Per-tissue sampling share induced by a per-slide head-cap at C (P ∝ min(count_slide, C))."""
    share = {}
    tot = 0.0
    for s, n in slide_counts.items():
        m = min(float(n), float(C))
        share[slide_to_tissue[s]] = share.get(slide_to_tissue[s], 0.0) + m
        tot += m
    return {t: v / tot for t, v in share.items()} if tot > 0 else {}


def _auto_sim_threshold(strata_centered_csv, override, default):
    """
    Centered-cosine threshold of similar pairs, from the cosine_strata_centered.csv of run_teacher_recurrence.py:
    midpoint of the random-negative q75 and the same-cluster/diff-slide q25 (midpoint of the medians if they
    overlap), clamped inside (neg median, pos median). override: manual value; default if the CSV is unusable.
    Returns (threshold, source).
    """
    import pandas as pd
    if override is not None:
        return float(override), "manual_override"
    try:
        df = pd.read_csv(strata_centered_csv).set_index("stratum")
        neg_m, neg_hi = float(df.loc["random", "median"]), float(df.loc["random", "q75"])
        pos_m, pos_lo = (float(df.loc["same_cluster_diff_slide", "median"]),
                         float(df.loc["same_cluster_diff_slide", "q25"]))
    except Exception as e:
        logger.warning("Could not auto-derive sim_threshold from %s (%s) — using default %.3f.",
                       strata_centered_csv, e, default)
        return float(default), "default_fallback"
    if not (np.isfinite(neg_m) and np.isfinite(pos_m) and pos_m > neg_m):
        logger.warning("Degenerate centered strata (neg=%.3f pos=%.3f) — using default %.3f.",
                       neg_m, pos_m, default)
        return float(default), "default_fallback"
    thr = 0.5 * (neg_hi + pos_lo) if pos_lo > neg_hi else 0.5 * (neg_m + pos_m)
    thr = float(np.clip(thr, neg_m + 0.02, pos_m - 0.02))
    logger.info("Auto sim_threshold = %.3f  (neg median %.3f q75 %.3f | pos median %.3f q25 %.3f)  "
                "from %s", thr, neg_m, neg_hi, pos_m, pos_lo,
                os.path.join(os.path.basename(os.path.dirname(strata_centered_csv)),
                             os.path.basename(strata_centered_csv)))
    return thr, "auto_from_centered_strata"


def _curve_and_rec(label, sims_by_cond, T_grid, sim_threshold, precision_floor):
    import pandas as pd
    curves, recs, cooc = [], {}, {}
    for cond, sims in sims_by_cond.items():
        if not sims:
            recs[cond] = {"T": None, "precise": None}
            continue
        cooc[cond] = round(G.batch_cooccurrence_rate(sims, sim_threshold), 5)
        curve = G.soft_target_precision_curve(sims, T_grid, sim_threshold=sim_threshold)
        curve.insert(0, "condition", cond)
        T_rec, ok = G.recommend_temperature(curve, precision_floor=precision_floor)
        recs[cond] = {"T": T_rec, "precise": bool(ok)}
        curves.append(curve)
        logger.info("  [%s] %s -> recommended T=%.3f (precise=%s)", label, cond, T_rec, ok)
    return (pd.concat(curves, ignore_index=True) if curves else None), recs, cooc


def _scconcept_token_sims(scconcept_dir, mask_zip_path, patch_ids, weights_by_cond, K,
                          token_patch_batch, n_sim_batches, seed):
    """
    scConcept token batches rebuilt from the nucleus masks (cells grouped by patch), per sampling condition.
    Returns lists of centered token-batch cosine matrices.
    """
    patch_ids = np.asarray(patch_ids)
    pbatches_by_c, needed = {}, set()
    for cond, w in weights_by_cond.items():
        pb = G.simulate_batches(w, token_patch_batch, n_sim_batches, seed=seed)
        pbatches_by_c[cond] = pb
        for b in pb:
            needed.update(int(i) for i in b)
    needed = sorted(needed)
    logger.info("[scconcept_cell] reconstructing token cells for %d unique patches from masks...", len(needed))

    scc = {}

    def _h(slide):
        if slide not in scc:
            scc[slide] = None
            p = os.path.join(str(scconcept_dir), f"{slide}.h5")
            if os.path.isfile(p):
                h5 = h5py.File(p, "r")
                scc[slide] = (h5, h5["cell_id_int"][:].astype(np.int32, copy=False))
        return scc[slide]

    raw = {}
    with zipfile.ZipFile(str(mask_zip_path), "r") as mz:
        members = set(mz.namelist())
        for i in needed:
            pid = str(patch_ids[i])
            if f"{pid}.npz" not in members:
                continue
            slide, _ = _split_patch_id(pid)
            handle = _h(slide)
            if handle is None:
                continue
            cids = _patch_cell_ids(mz.read(f"{pid}.npz"))
            if cids.size == 0:
                continue
            h5, cid_int = handle
            emb = _fetch_cls_emb(h5, cid_int, cids)
            if emb.shape[0] == 0:
                continue
            if emb.shape[0] > K:
                r = np.random.default_rng(_stable_seed(pid, seed))
                emb = emb[r.choice(emb.shape[0], K, replace=False)]
            nrm = np.clip(np.linalg.norm(emb, axis=1, keepdims=True), _NORM_EPS, None)
            raw[i] = (emb / nrm).astype(np.float32)
    for v in scc.values():
        if v is not None:
            v[0].close()

    if not raw:
        return None
    mean = np.concatenate(list(raw.values()), axis=0).mean(axis=0)
    cblk = {i: G.center_by(b, mean) for i, b in raw.items()}
    sims_by_c = {}
    for cond, pb in pbatches_by_c.items():
        sims = []
        for b in pb:
            mats = [cblk[int(i)] for i in b if int(i) in cblk]
            if len(mats) < 2:
                continue
            X = np.concatenate(mats, axis=0)
            if X.shape[0] >= 2:
                sims.append(X @ X.T)
        sims_by_c[cond] = sims
    return sims_by_c


# --------------------------------------------------------------------------- orchestration
def select_params(*, cache_dir, training_folder, output_dir, dataset_id, training_dataset_id,
                  novae_training_id, investigation_id, split="train", n_min_cells=0,
                  headcap_tol_grid=(1.1, 1.2, 1.3, 1.5, 2.0), max_oversampling=1.3,
                  T_grid=(0.02, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.5), batch_size=128,
                  cell_batch_size=1024, n_sim_batches=200, sim_threshold_novae=None,
                  sim_threshold_scconcept=None, precision_floor=0.9, seed=0, max_patches=0,
                  analyze_scconcept=True, scconcept_training_id=None, scconcept_dir=None,
                  scconcept_max_cells_per_patch=32, overwrite=False):
    """
    Run the head-cap and temperature selection on explicit paths and write the outputs.
    sim_threshold_{novae,scconcept}=None: derived from cosine_strata_centered.csv (_auto_sim_threshold).
    """
    import pandas as pd
    cache_dir, training_folder, output_dir = str(cache_dir), str(training_folder), str(output_dir)
    headcap_tol_grid = sorted(float(t) for t in headcap_tol_grid)
    inv_dir = os.path.dirname(output_dir.rstrip("/"))   # investigation dir of run_teacher_recurrence.py (centered strata)
    scc_thr_src = None

    sentinel = os.path.join(output_dir, "report.md")
    if os.path.exists(sentinel) and not overwrite:
        mtime = datetime.fromtimestamp(os.path.getmtime(sentinel)).isoformat(timespec="seconds")
        raise FileExistsError(
            f"[ERROR] report.md already exists at '{sentinel}' (mtime={mtime}). "
            f"Pass --overwrite, or choose a new --investigation_id.")
    os.makedirs(output_dir, exist_ok=True)
    manifest = _read_cache_manifest(cache_dir)

    # ---- load + center Novae patches (with slide and tissue labels) ------------------------
    slides = _enumerate_slides(training_folder, cache_dir, manifest)
    Xp, sl_p, ti_p, inv_p, rns_p, cnt_p, _ = _load_train_st_feat(
        cache_dir, training_folder, split, n_min_cells, slides, max_patches, seed)
    Xp_c, keep = _center(Xp)
    sl_p, ti_p = sl_p[keep], ti_p[keep]
    del Xp
    logger.info("Novae: %d centered patches over %d slides / %d tissues.",
                Xp_c.shape[0], len(_counts(sl_p)), len(_counts(ti_p)))

    # ---- head-cap search: per slide (recommended) + per tissue (comparison) ---------------
    slide_counts, slide_rows, rec_C_slide = _headcap_search(sl_p, headcap_tol_grid, max_oversampling)
    tissue_counts, tissue_rows, rec_C_tissue = _headcap_search(ti_p, headcap_tol_grid, max_oversampling)
    _save_csv_atomic(os.path.join(output_dir, "headcap_slide.csv"), pd.DataFrame(slide_rows))
    _save_csv_atomic(os.path.join(output_dir, "headcap_tissue.csv"), pd.DataFrame(tissue_rows))
    slide_to_tissue = {s: _tissue_of(s) for s in slide_counts}
    slide_metrics = G.headcap_metrics(slide_counts, rec_C_slide)
    induced_tissue = _induced_tissue_shares(slide_counts, slide_to_tissue, rec_C_slide)
    logger.info("Recommended per-slide head-cap C = %d (caps %d slides; max slide share %.1f%%, "
                "induced breast %.1f%%).", int(rec_C_slide), slide_metrics["n_capped"],
                100 * slide_metrics["max_share"], 100 * induced_tissue.get("breast", 0.0))

    # ---- temperature under natural vs the recommended per-slide head-cap -------------------
    sim_threshold_novae, novae_thr_src = _auto_sim_threshold(
        os.path.join(inv_dir, "cosine_strata_centered.csv"), sim_threshold_novae, default=0.5)
    conds = {"natural": np.ones(sl_p.shape[0], dtype=np.float64),
             f"headcap_slide_C{int(rec_C_slide)}": G.headcap_weights(sl_p, rec_C_slide)}
    novae_sims = {c: G.batch_sims(Xp_c, G.simulate_batches(w, batch_size, n_sim_batches, seed=seed))
                  for c, w in conds.items()}
    tn_df, tn_recs, tn_cooc = _curve_and_rec("novae_patch", novae_sims, T_grid,
                                             sim_threshold_novae, precision_floor)
    _save_csv_atomic(os.path.join(output_dir, "temperature_novae.csv"), tn_df)
    del Xp_c

    # ---- temperature: scConcept (token batches from masks) --------------------------------
    ts_df, ts_recs, ts_cooc, scc_used = None, None, {}, {}
    if analyze_scconcept:
        scc_training_id = scconcept_training_id or manifest.get("scconcept_training_id")
        if scconcept_dir is None and scc_training_id:
            scconcept_dir = get_full_path("train", "trainings", "trainings_scConcept",
                                          scc_training_id, "embeddings", as_str=True)
        mask_zip = os.path.join(training_folder, "masks_cell_ids_nuclei.zip")
        K = int(scconcept_max_cells_per_patch)
        token_pb = max(1, int(cell_batch_size) // K)
        scc_used = {"scconcept_training_id": scc_training_id, "scconcept_dir": scconcept_dir,
                    "mask_zip": mask_zip, "token_patch_batch": token_pb,
                    "token_batches": "reconstructed_from_masks"}
        if not (scconcept_dir and os.path.isdir(str(scconcept_dir)) and os.path.isfile(mask_zip)):
            logger.warning("scConcept dir or mask zip not found — skipping scConcept temperature.")
        else:
            fold_pids = _list_fold_patch_ids(training_folder, split, n_min_cells)
            p_ids, p_slides = [], []
            for pid in fold_pids:
                s, _ = _split_patch_id(pid)
                if os.path.isfile(os.path.join(str(scconcept_dir), f"{s}.h5")):
                    p_ids.append(pid)
                    p_slides.append(s)
            p_slides = np.asarray(p_slides)
            wcond = {"natural": np.ones(p_slides.shape[0], dtype=np.float64),
                     f"headcap_slide_C{int(rec_C_slide)}": G.headcap_weights(p_slides, rec_C_slide)}
            sim_threshold_scconcept, scc_thr_src = _auto_sim_threshold(
                os.path.join(inv_dir, "scconcept", "cosine_strata_centered.csv"),
                sim_threshold_scconcept, default=0.3)
            sims_by_c = _scconcept_token_sims(scconcept_dir, mask_zip, p_ids, wcond, K,
                                              token_pb, n_sim_batches, seed)
            if sims_by_c:
                ts_df, ts_recs, ts_cooc = _curve_and_rec("scconcept_cell", sims_by_c, T_grid,
                                                         sim_threshold_scconcept, precision_floor)
                if ts_df is not None:
                    _save_csv_atomic(os.path.join(output_dir, "temperature_scconcept.csv"), ts_df)

    # ---- summary + config + report --------------------------------------------------------
    hc_cond = f"headcap_slide_C{int(rec_C_slide)}"
    params = dict(split=split, n_min_cells=int(n_min_cells), headcap_tol_grid=headcap_tol_grid,
                  max_oversampling=float(max_oversampling), T_grid=list(T_grid), batch_size=int(batch_size),
                  cell_batch_size=int(cell_batch_size), n_sim_batches=int(n_sim_batches),
                  sim_threshold_novae=float(sim_threshold_novae), sim_threshold_novae_source=novae_thr_src,
                  sim_threshold_scconcept=(float(sim_threshold_scconcept) if sim_threshold_scconcept is not None else None),
                  sim_threshold_scconcept_source=scc_thr_src,
                  precision_floor=float(precision_floor), seed=int(seed), centered=True,
                  analyze_scconcept=bool(analyze_scconcept),
                  scconcept_max_cells_per_patch=int(scconcept_max_cells_per_patch),
                  **{f"scconcept_{k}": v for k, v in scc_used.items()})
    rec = {
        "headcap_grouping_recommended": "slide",
        "recommended_slide_headcap_C": int(rec_C_slide),
        "slide_headcap_effect": {"max_slide_share": slide_metrics["max_share"],
                                 "effective_slides": slide_metrics["effective_groups"],
                                 "n_slides_capped": slide_metrics["n_capped"],
                                 "capped_slides": slide_metrics["capped_groups"],
                                 "max_oversampling": slide_metrics["max_oversampling"],
                                 "induced_breast_tissue_share": round(induced_tissue.get("breast", 0.0), 4),
                                 "induced_max_tissue_share": round(max(induced_tissue.values()) if induced_tissue else 0.0, 4)},
        "recommended_tissue_headcap_C": int(rec_C_tissue),
        "recommended_target_temperature_novae": tn_recs.get(hc_cond, {}).get("T"),
        "novae_temperature_precise": tn_recs.get(hc_cond, {}).get("precise"),
        "recommended_token_target_temperature_scconcept": (ts_recs.get(hc_cond, {}).get("T") if ts_recs else None),
        "scconcept_temperature_precise": (ts_recs.get(hc_cond, {}).get("precise") if ts_recs else None),
        "temperature_by_condition": {"novae": tn_recs, "scconcept": ts_recs},
        "cooccurrence_by_condition": {"novae": tn_cooc, "scconcept": ts_cooc},
        "slide_counts": slide_counts, "tissue_counts": tissue_counts,
    }
    summary = {"recommendations": rec, "params": params}
    _save_json_atomic(os.path.join(output_dir, "summary.json"), summary)

    config = {"schema_version": PARAM_SELECT_SCHEMA_VERSION,
              "created_utc": datetime.now(timezone.utc).isoformat(),
              "created_by": "select_training_params.py", "git_commit": _git_commit(),
              "dataset_id": dataset_id, "training_dataset_id": training_dataset_id,
              "novae_training_id": novae_training_id, "investigation_id": investigation_id,
              "cache_fingerprint": manifest.get("fingerprint"),
              "scconcept_model_name": manifest.get("scconcept_model_name"),
              "scconcept_model_version": manifest.get("scconcept_model_version"), "parameters": params}
    _save_json_atomic(os.path.join(output_dir, "config.json"), config)
    _save_text_atomic(sentinel, _build_report_md(summary, config, slide_rows, tissue_rows, tn_df, ts_df))
    logger.info("Wrote param-selection outputs to %s", output_dir)
    return summary


# --------------------------------------------------------------------------- report
def _fmt(x, nd=3):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "n/a"
    return f"{x:.{nd}f}" if isinstance(x, float) else str(x)


def _headcap_table(rows):
    L = ["\n| max_oversampling | cap C | max_share | effective_groups | n_capped | capped_groups |\n",
         "|---|---|---|---|---|---|\n"]
    for r in rows:
        cg = r["capped_groups"] if r["capped_groups"] else "—"
        L.append(f"| {_fmt(r['tol_max_oversampling'], 2)} | {r['cap']} | {_fmt(r['max_share'])} | "
                 f"{_fmt(r['effective_groups'])} | {r['n_capped']} | {cg} |\n")
    return L


def _temp_tables(df):
    L = []
    for cond in sorted(df["condition"].unique()):
        sub = df[df["condition"] == cond]
        L.append(f"\n*{cond}*\n\n| T | self_mass | offdiag_mass | precision_similar |\n|---|---|---|---|\n")
        for _, r in sub.iterrows():
            L.append(f"| {_fmt(r['T'])} | {_fmt(r['self_mass'])} | {_fmt(r['offdiag_mass'])} | "
                     f"{_fmt(r['precision_similar'])} |\n")
    return L


def _build_report_md(summary, config, slide_rows, tissue_rows, tn_df, ts_df):
    rec = summary["recommendations"]
    p = config["parameters"]
    eff = rec["slide_headcap_effect"]
    L = [f"# Pre-CLIP training-parameter selection — `{config['investigation_id']}`\n",
         f"- teacher: novae `{config['novae_training_id']}` · scConcept "
         f"`{config.get('scconcept_model_name')}` `{config.get('scconcept_model_version')}` · "
         f"**centered target** · dataset `{config['training_dataset_id']}`\n\n",
         "## Recommendations\n",
         f"- **per-slide head-cap C = {rec['recommended_slide_headcap_C']}** — cap each slide's "
         f"effective size at ≤C patches (`min(count, C)`), max repetition ≤ {_fmt(p['max_oversampling'], 2)}×.\n",
         f"  → largest slide {_fmt(eff['max_slide_share'])} share · induced breast tissue "
         f"{_fmt(eff['induced_breast_tissue_share'])} · effective slides {_fmt(eff['effective_slides'])} · "
         f"caps {eff['n_slides_capped']} slides: {', '.join(eff['capped_slides']) or '—'}\n",
         f"- Novae **`target_temperature` = {_fmt(rec['recommended_target_temperature_novae'])}** "
         f"(precise={rec['novae_temperature_precise']}; under the head-cap; vs default 0.1)\n"]
    if rec["recommended_token_target_temperature_scconcept"] is not None:
        L.append(f"- scConcept **`token_target_temperature` = "
                 f"{_fmt(rec['recommended_token_target_temperature_scconcept'])}** "
                 f"(precise={rec['scconcept_temperature_precise']}; token batches from masks)\n")
    else:
        L.append("- scConcept token temperature: not analyzed (inputs unavailable / disabled)\n")

    L.append("\n## Head-cap search — per slide (recommended axis)\n")
    L.append("*`P ∝ min(count(slide), C)`. Recommended C = smallest whose max_oversampling "
             f"≤ `--max_oversampling` ({_fmt(p['max_oversampling'], 2)}).*\n")
    L += _headcap_table(slide_rows)
    L.append("\n## Head-cap search — per tissue (for comparison)\n")
    L.append("*`P ∝ min(count(tissue), C)`. Does not limit the dominance of one slide within a tissue.*\n")
    L += _headcap_table(tissue_rows)

    L.append("\n## Temperature — Novae (patch, centered soft-target)\n")
    L.append(f"*Similar-pair threshold = **{_fmt(p['sim_threshold_novae'])}** "
             f"({p.get('sim_threshold_novae_source')}). Softest T (largest off-diagonal mass) whose "
             f"precision_similar stays ≥ {_fmt(p['precision_floor'], 2)}, under natural vs the "
             f"recommended head-cap sampling.*\n")
    if tn_df is not None:
        L += _temp_tables(tn_df)
    if ts_df is not None:
        L.append("\n## Temperature — scConcept (cell, centered token soft-target)\n")
        L.append(f"*Similar-pair threshold = **{_fmt(p['sim_threshold_scconcept'])}** "
                 f"({p.get('sim_threshold_scconcept_source')}). Token batches reconstructed from the "
                 f"nucleus masks (cells grouped by their patch).*\n")
        L += _temp_tables(ts_df)

    L.append("\n*Offline selection from the frozen teacher geometry and simulated batches. The similar-pair "
             "thresholds are derived from `cosine_strata_centered.csv` of run_teacher_recurrence.py (between "
             "the centered random negatives and the same-cluster/diff-slide positives); pass "
             "`--sim_threshold_*` to override.*\n")
    return "".join(L)


# --------------------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser(
        description="Selection of the head-cap C (slide / tissue) and soft-target temperatures (both teachers).")
    parser.add_argument("--dataset_id", type=str, default="ds_ct1_20x_224_o112")
    parser.add_argument("--training_dataset_id", type=str, default="ds_ct1_20x_224_o112_1")
    parser.add_argument("--novae_training_id", type=str, default="training_4")
    parser.add_argument("--investigation_id", type=str, default="preclip_investigation_1",
                        help="Outputs go under this id's param_selection/ subfolder.")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--n_min_cells", type=int, default=0)
    parser.add_argument("--headcap_tol_grid", type=float, nargs="+", default=[1.1, 1.2, 1.3, 1.5, 2.0],
                        help="max-oversampling tolerances to tabulate the head-cap C for.")
    parser.add_argument("--max_oversampling", type=float, default=1.3,
                        help="Recommended head-cap = smallest C whose worst-group repetition stays ≤ this value.")
    parser.add_argument("--T_grid", type=float, nargs="+",
                        default=[0.02, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.5])
    parser.add_argument("--batch_size", type=int, default=128, help="Simulated patch-batch size (Novae).")
    parser.add_argument("--cell_batch_size", type=int, default=1024,
                        help="Target scConcept token-batch cell count (→ token_patch_batch = //max_cells_per_patch).")
    parser.add_argument("--scconcept_max_cells_per_patch", type=int, default=32,
                        help="Cells sampled per patch for a token batch (match training scconcept_max_cells).")
    parser.add_argument("--n_sim_batches", type=int, default=200, help="Batches to simulate per condition.")
    parser.add_argument("--sim_threshold_novae", type=float, default=None,
                        help="Centered-cosine threshold of similar pairs (Novae). Default: derived from "
                             "cosine_strata_centered.csv of run_teacher_recurrence.py.")
    parser.add_argument("--sim_threshold_scconcept", type=float, default=None,
                        help="Centered-cosine threshold of similar pairs (scConcept). Default: derived from "
                             "scconcept/cosine_strata_centered.csv of run_teacher_recurrence.py.")
    parser.add_argument("--precision_floor", type=float, default=0.9,
                        help="Min fraction of off-diagonal soft-target mass on similar pairs.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_patches", type=int, default=0)
    parser.add_argument("--skip_scconcept", action="store_true")
    parser.add_argument("--scconcept_training_id", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                        handlers=[logging.StreamHandler()])

    training_folder = get_full_path("train", "training_datasets", args.dataset_id,
                                    args.training_dataset_id, as_str=True)
    cache_dir = get_full_path("train", "training_datasets", args.dataset_id, args.training_dataset_id,
                              "novae_teacher_cache", args.novae_training_id, as_str=True, scratch=True)
    output_dir = get_full_path("train", "preclip_investigation", args.training_dataset_id,
                               args.investigation_id, "param_selection", as_str=True)

    select_params(cache_dir=cache_dir, training_folder=training_folder, output_dir=output_dir,
                  dataset_id=args.dataset_id, training_dataset_id=args.training_dataset_id,
                  novae_training_id=args.novae_training_id, investigation_id=args.investigation_id,
                  split=args.split, n_min_cells=args.n_min_cells, headcap_tol_grid=args.headcap_tol_grid,
                  max_oversampling=args.max_oversampling, T_grid=args.T_grid, batch_size=args.batch_size,
                  cell_batch_size=args.cell_batch_size, n_sim_batches=args.n_sim_batches,
                  sim_threshold_novae=args.sim_threshold_novae, sim_threshold_scconcept=args.sim_threshold_scconcept,
                  precision_floor=args.precision_floor, seed=args.seed, max_patches=args.max_patches,
                  analyze_scconcept=not args.skip_scconcept, scconcept_training_id=args.scconcept_training_id,
                  scconcept_max_cells_per_patch=args.scconcept_max_cells_per_patch, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
