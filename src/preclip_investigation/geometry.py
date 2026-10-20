"""Geometry functions (numpy, no repo imports) for the pre-CLIP analysis of the ST teacher embeddings (rows assumed L2-normalized)."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- normalization
def l2_normalize(X: np.ndarray, eps: float = 1e-8):
    """Row-L2-normalize and drop zero-norm rows. Returns (Xn [M, dim] float32, keep_mask [N] over the input rows)."""
    X = np.asarray(X, dtype=np.float32)
    norms = np.linalg.norm(X, axis=1)
    keep_mask = norms > eps
    Xn = X[keep_mask] / norms[keep_mask, None]
    return Xn.astype(np.float32, copy=False), keep_mask


# --------------------------------------------------------------------------- centering
def center_by(X: np.ndarray, mean: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Subtract a given mean vector, then re-L2-normalize (tiny norms floored, rows kept)."""
    Xc = np.asarray(X, dtype=np.float32) - np.asarray(mean, dtype=np.float32)[None, :]
    norms = np.clip(np.linalg.norm(Xc, axis=1), eps, None)
    return (Xc / norms[:, None]).astype(np.float32)


def center_then_normalize(Xn: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Subtract the mean of the rows, then re-L2-normalize (tiny norms floored, rows kept)."""
    X = np.asarray(Xn, dtype=np.float32)
    return center_by(X, X.mean(axis=0), eps=eps)


# --------------------------------------------------------------------------- balanced subsampling
def balanced_subsample(group_ids: np.ndarray, n: int, seed: int = 0) -> np.ndarray:
    """Up to n row indices spread evenly across groups (round-robin over shuffled per-group indices)."""
    group_ids = np.asarray(group_ids)
    rng = np.random.default_rng(seed)
    by: dict = {}
    for i, g in enumerate(group_ids.tolist()):
        by.setdefault(g, []).append(i)
    for g in by:
        rng.shuffle(by[g])
    keys = list(by.keys())
    rng.shuffle(keys)
    n = int(min(n, group_ids.shape[0]))
    out: list = []
    while len(out) < n:
        progressed = False
        for g in keys:
            if by[g]:
                out.append(by[g].pop())
                progressed = True
                if len(out) >= n:
                    break
        if not progressed:
            break
    return np.asarray(out, dtype=np.int64)


# --------------------------------------------------------------------------- clustering
def fit_kmeans(
    Xn: np.ndarray,
    k: int,
    fit_subsample: int = 200_000,
    n_init: int = 3,
    seed: int = 0,
    minibatch: bool = False,
    chunk: int = 100_000,
    fit_idx: np.ndarray | None = None,
) -> np.ndarray:
    """
    KMeans (MiniBatchKMeans if minibatch) fitted on the fit_idx rows or a random subsample, then all rows
    assigned in chunks. Returns int32 labels [N].
    """
    from sklearn.cluster import KMeans, MiniBatchKMeans

    Xn = np.asarray(Xn, dtype=np.float32)
    N = Xn.shape[0]
    rng = np.random.default_rng(seed)
    if fit_idx is not None:
        Xfit = Xn[np.asarray(fit_idx, dtype=np.int64)]
    elif N > fit_subsample:
        Xfit = Xn[rng.choice(N, size=fit_subsample, replace=False)]
    else:
        Xfit = Xn
    k_eff = int(min(k, max(2, Xfit.shape[0] - 1)))
    if minibatch:
        km = MiniBatchKMeans(n_clusters=k_eff, n_init=n_init, random_state=seed, batch_size=10_000)
    else:
        km = KMeans(n_clusters=k_eff, n_init=n_init, random_state=seed)
    km.fit(Xfit)
    labels = np.empty(N, dtype=np.int32)
    for start in range(0, N, chunk):
        stop = min(start + chunk, N)
        labels[start:stop] = km.predict(Xn[start:stop])
    return labels


# --------------------------------------------------------------------------- diversity helpers
def _inverse_simpson(counts: Sequence[float]) -> float:
    """Effective number = 1 / Σ pᵢ² (Hill order 2)."""
    counts = np.asarray(counts, dtype=np.float64)
    tot = counts.sum()
    if tot <= 0:
        return 0.0
    p = counts / tot
    denom = float(np.square(p).sum())
    return (1.0 / denom) if denom > 0 else 0.0


def _shannon_hill(counts: Sequence[float]) -> float:
    """Effective number = exp(H) (Hill order 1)."""
    counts = np.asarray(counts, dtype=np.float64)
    tot = counts.sum()
    if tot <= 0:
        return 0.0
    p = counts[counts > 0] / tot
    H = -float((p * np.log(p)).sum())
    return float(np.exp(H))


# --------------------------------------------------------------------------- recurrence
def cluster_coherence(Xn: np.ndarray, labels: np.ndarray) -> dict:
    """Per-cluster mean cosine of the members to their unit centroid, {cluster: coherence} (pass centered unit vectors)."""
    Xn = np.asarray(Xn, dtype=np.float32)
    labels = np.asarray(labels)
    out = {}
    for cl in np.unique(labels):
        m = labels == cl
        C = Xn[m].mean(axis=0)
        nrm = float(np.linalg.norm(C))
        out[int(cl)] = 0.0 if nrm < 1e-12 else float((Xn[m] @ (C / nrm)).mean())
    return out


def cluster_recurrence(
    labels: np.ndarray,
    slide_ids: np.ndarray,
    tissue_ids: np.ndarray,
    Xn_centered: np.ndarray | None = None,
    s_eff_min: float = 2.0,
) -> pd.DataFrame:
    """
    Per-cluster cross-slide recurrence: S_eff_all = 1 / Σ pᵢ² over the slides of the cluster (recurrent_all if
    >= s_eff_min), the same over the slides of the dominant tissue, and coherence if Xn_centered is given.
    """
    df = pd.DataFrame(
        {
            "cluster": np.asarray(labels),
            "slide": np.asarray(slide_ids),
            "tissue": np.asarray(tissue_ids),
        }
    )
    coh = cluster_coherence(Xn_centered, labels) if Xn_centered is not None else None
    rows = []
    for cl, g in df.groupby("cluster", sort=True):
        n_c = len(g)
        all_slide_counts = g["slide"].value_counts().to_numpy()
        s_eff_all = _inverse_simpson(all_slide_counts)
        tissue_counts = g["tissue"].value_counts()
        dom_tissue = tissue_counts.index[0]
        dom_count = int(tissue_counts.iloc[0])
        dom = g[g["tissue"] == dom_tissue]
        dom_slide_counts = dom["slide"].value_counts().to_numpy()
        s_eff_dom = _inverse_simpson(dom_slide_counts)
        rows.append(
            {
                "cluster": int(cl),
                "n_rows": int(n_c),
                "dominant_tissue": dom_tissue,
                "purity": round(dom_count / n_c, 4),
                "n_slides_total": int(g["slide"].nunique()),
                # all-slide recurrence
                "S_eff_all": round(float(s_eff_all), 4),
                "recurrent_all": bool(s_eff_all >= s_eff_min),
                "coherence": round(float(coh[int(cl)]), 4) if coh is not None else float("nan"),
                # dominant-tissue recurrence
                "n_slides_dom_tissue": int(len(dom_slide_counts)),
                "dom_tissue_row_count": dom_count,
                "S_eff_dom_tissue": round(float(s_eff_dom), 4),
                "recurrent_dom_tissue": bool(s_eff_dom >= s_eff_min),
            }
        )
    return pd.DataFrame(rows)


def tissue_ceiling(slide_ids: np.ndarray, tissue_ids: np.ndarray) -> float:
    """Fraction of patches on tissues with ≥2 distinct slides — the max attainable recurrence R."""
    df = pd.DataFrame({"slide": np.asarray(slide_ids), "tissue": np.asarray(tissue_ids)})
    if len(df) == 0:
        return 0.0
    slides_per_tissue = df.groupby("tissue")["slide"].nunique()
    multi = set(slides_per_tissue[slides_per_tissue >= 2].index)
    return float(df["tissue"].isin(multi).mean())


def recurrence_summary(
    cluster_df: pd.DataFrame,
    n_total: int,
    ceiling: float,
    K_list: Sequence[int] = (16, 32, 64),
    coherence_min: float = 0.3,
) -> dict:
    """
    Recurrence metrics at one k: R_all (fraction of rows in recurrent_all clusters), N_usable_all[K] and
    N_usable_coherent[K] (recurrent clusters with >= K rows, and with coherence >= coherence_min), and the
    dominant-tissue versions R, R_cond = R / ceiling and N_usable[K].
    """
    z = {int(K): 0 for K in K_list}
    if cluster_df.empty or not n_total:
        return {"R_all": 0.0, "N_usable_all": dict(z), "N_usable_coherent": dict(z),
                "R": 0.0, "R_cond": float("nan"), "ceiling": float(ceiling), "N_usable": dict(z)}
    rec_all = cluster_df[cluster_df["recurrent_all"]]
    R_all = float(rec_all["n_rows"].sum()) / float(n_total)
    coh_ok = rec_all[rec_all["coherence"] >= coherence_min] if "coherence" in rec_all.columns else rec_all
    N_usable_all = {int(K): int((rec_all["n_rows"] >= K).sum()) for K in K_list}
    N_usable_coh = {int(K): int((coh_ok["n_rows"] >= K).sum()) for K in K_list}
    rec_dom = cluster_df[cluster_df["recurrent_dom_tissue"]]
    R = float(rec_dom["n_rows"].sum()) / float(n_total)
    R_cond = float(R / ceiling) if ceiling > 0 else float("nan")
    N_usable = {int(K): int((rec_dom["dom_tissue_row_count"] >= K).sum()) for K in K_list}
    return {"R_all": R_all, "N_usable_all": N_usable_all, "N_usable_coherent": N_usable_coh,
            "R": R, "R_cond": R_cond, "ceiling": float(ceiling), "N_usable": N_usable}


def recurrent_mass_by_tissue(
    cluster_df: pd.DataFrame,
    labels: np.ndarray,
    tissue_ids: np.ndarray,
    coherence_min: float = 0.3,
) -> pd.DataFrame:
    """Per tissue: fraction of its rows in recurrent clusters and in coherent recurrent clusters (recurrent_here if >= 0.3)."""
    labels = np.asarray(labels)
    tissue_ids = np.asarray(tissue_ids)
    rec_all = set(cluster_df.loc[cluster_df["recurrent_all"], "cluster"].astype(int).tolist())
    if "coherence" in cluster_df.columns:
        coh_ok = set(cluster_df.loc[(cluster_df["recurrent_all"]) &
                                    (cluster_df["coherence"] >= coherence_min),
                                    "cluster"].astype(int).tolist())
    else:
        coh_ok = rec_all
    lab_int = labels.astype(np.int64)
    row_rec = np.isin(lab_int, list(rec_all)) if rec_all else np.zeros(lab_int.shape, bool)
    row_coh = np.isin(lab_int, list(coh_ok)) if coh_ok else np.zeros(lab_int.shape, bool)
    rows = []
    for t in sorted(set(tissue_ids.tolist())):
        m = tissue_ids == t
        frac_coh = float(row_coh[m].mean()) if m.any() else 0.0
        rows.append({"tissue": t, "n_rows": int(m.sum()),
                     "frac_in_recurrent_all": round(float(row_rec[m].mean()) if m.any() else 0.0, 4),
                     "frac_in_recurrent_coherent": round(frac_coh, 4),
                     "recurrent_here": bool(frac_coh >= 0.3)})
    return pd.DataFrame(rows).sort_values("frac_in_recurrent_coherent", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------- supervised cross-slide recovery
def cross_slide_celltype_knn(
    Xn: np.ndarray,
    slide_ids: np.ndarray,
    celltype_ids: np.ndarray,
    k: int = 15,
    n_ref_per_slide: int = 800,
    n_query_per_celltype: int = 400,
    seed: int = 0,
    chunk: int = 256,
) -> tuple:
    """
    Cross-slide cell-type kNN purity: fraction of the k nearest references from other slides with the query's cell
    type, vs the cell-type prevalence in the slide-balanced references. Returns (per_celltype_df, macro_dict).
    """
    Xn = np.asarray(Xn, dtype=np.float32)
    slides = np.asarray(slide_ids)
    cts = np.asarray(celltype_ids)
    rng = np.random.default_rng(seed)
    n_slides = len(set(slides.tolist()))

    ref_idx = balanced_subsample(slides, n_ref_per_slide * max(1, n_slides), seed=seed)
    Xr, sr, cr = Xn[ref_idx], slides[ref_idx], cts[ref_idx]
    prevalence = {c: float(np.mean(cr == c)) for c in set(cr.tolist())}

    q_parts = []
    for c in sorted(set(cts.tolist())):
        idc = np.where(cts == c)[0]
        if idc.size > n_query_per_celltype:
            idc = rng.choice(idc, n_query_per_celltype, replace=False)
        q_parts.append(idc)
    q_idx = np.concatenate(q_parts) if q_parts else np.empty(0, dtype=np.int64)
    if q_idx.size == 0 or Xr.shape[0] <= k:
        return pd.DataFrame(columns=["celltype", "n_query", "cross_slide_knn_purity",
                                     "chance_prevalence", "excess_over_chance"]), {}
    Xq, sq, cq = Xn[q_idx], slides[q_idx], cts[q_idx]

    purity = np.zeros(Xq.shape[0], dtype=np.float64)
    for s0 in range(0, Xq.shape[0], chunk):
        s1 = min(s0 + chunk, Xq.shape[0])
        S = Xq[s0:s1] @ Xr.T                                  # [c, R] cosine
        same = sq[s0:s1][:, None] == sr[None, :]             # mask same-slide references
        S = np.where(same, -np.inf, S)
        topk = np.argpartition(-S, k, axis=1)[:, :k]          # top-k by similarity
        nbr_ct = cr[topk]                                     # [c, k] neighbour cell types
        purity[s0:s1] = (nbr_ct == cq[s0:s1][:, None]).mean(axis=1)

    df = pd.DataFrame({"celltype": cq, "purity": purity})
    rows = []
    for c, g in df.groupby("celltype"):
        p = float(g["purity"].mean())
        ch = prevalence.get(c, float("nan"))
        rows.append({"celltype": c, "n_query": int(len(g)),
                     "cross_slide_knn_purity": round(p, 4),
                     "chance_prevalence": round(ch, 4),
                     "excess_over_chance": round(p - ch, 4)})
    out = pd.DataFrame(rows).sort_values("excess_over_chance", ascending=False).reset_index(drop=True)
    macro = {"k": int(k), "n_celltypes": int(len(out)), "n_ref": int(Xr.shape[0]),
             "n_query": int(Xq.shape[0]),
             "macro_purity": round(float(out["cross_slide_knn_purity"].mean()), 4),
             "macro_chance": round(float(out["chance_prevalence"].mean()), 4),
             "macro_excess": round(float(out["excess_over_chance"].mean()), 4)}
    return out, macro


# --------------------------------------------------------------------------- cosine strata
def cosine_pair_stats(
    Xn: np.ndarray,
    labels: np.ndarray,
    slide_ids: np.ndarray,
    tissue_ids: np.ndarray,
    n_pairs: int = 2_000_000,
    seed: int = 0,
    chunk: int = 500_000,
) -> dict:
    """
    Cosine of random pairs per stratum: random, cross_tissue, same_slide, same_cluster_same_slide,
    same_cluster_diff_slide (no N x N matrix).
    """
    Xn = np.asarray(Xn, dtype=np.float32)
    labels = np.asarray(labels)
    slides = np.asarray(slide_ids)
    tissues = np.asarray(tissue_ids)
    N = Xn.shape[0]
    rng = np.random.default_rng(seed)

    ii = rng.integers(0, N, size=n_pairs)
    jj = rng.integers(0, N, size=n_pairs)
    keep = ii != jj
    ii, jj = ii[keep], jj[keep]

    cos_parts, ss_parts, sc_parts, stt_parts = [], [], [], []
    for start in range(0, ii.shape[0], chunk):
        a = ii[start:start + chunk]
        b = jj[start:start + chunk]
        cos_parts.append(np.einsum("ij,ij->i", Xn[a], Xn[b]).astype(np.float32))
        ss_parts.append(slides[a] == slides[b])
        sc_parts.append(labels[a] == labels[b])
        stt_parts.append(tissues[a] == tissues[b])
    cos = np.concatenate(cos_parts)
    ss = np.concatenate(ss_parts)
    sc = np.concatenate(sc_parts)
    stt = np.concatenate(stt_parts)

    def stat(mask: np.ndarray) -> dict:
        v = cos[mask]
        if v.size == 0:
            return {"n": 0, "median": float("nan"), "mean": float("nan"),
                    "q25": float("nan"), "q75": float("nan")}
        return {"n": int(v.size), "median": float(np.median(v)), "mean": float(np.mean(v)),
                "q25": float(np.quantile(v, 0.25)), "q75": float(np.quantile(v, 0.75))}

    return {
        "random": stat(np.ones(cos.shape[0], dtype=bool)),
        "cross_tissue": stat(~stt),
        "same_slide": stat(ss),
        "same_cluster_same_slide": stat(sc & ss),
        "same_cluster_diff_slide": stat(sc & ~ss),
    }


# --------------------------------------------------------------------------- per-slide heterogeneity
def occupied_cluster_hill(labels_of_slide: np.ndarray) -> float:
    """Effective number of niches one slide occupies (Hill order 1). Higher ⇒ more heterogeneous."""
    labels_of_slide = np.asarray(labels_of_slide)
    if labels_of_slide.size == 0:
        return 0.0
    _, counts = np.unique(labels_of_slide, return_counts=True)
    return _shannon_hill(counts)


def effective_rank(Xs: np.ndarray, max_points: int = 20_000, seed: int = 0) -> float:
    """Effective rank = exp(entropy of the squared singular values), uncentered; rows subsampled above max_points."""
    Xs = np.asarray(Xs, dtype=np.float64)
    if Xs.shape[0] < 2:
        return float("nan")
    if Xs.shape[0] > max_points:
        rng = np.random.default_rng(seed)
        Xs = Xs[rng.choice(Xs.shape[0], max_points, replace=False)]
    s = np.linalg.svd(Xs, compute_uv=False)
    p = np.square(s)
    tot = p.sum()
    if tot <= 0:
        return float("nan")
    p = p[p > 0] / tot
    H = -float((p * np.log(p)).sum())
    return float(np.exp(H))


# --------------------------------------------------------------------------- temperature diagnostic
def build_synthetic_batch(
    labels: np.ndarray,
    slide_ids: np.ndarray,
    cluster_df: pd.DataFrame,
    P: int,
    K: int,
    seed: int = 0,
):
    """
    Row indices of a synthetic P x K batch: up to P recurrent clusters with >= K rows (largest first), K rows
    each sampled round-robin across slides. Returns (idx, synth_labels, P_used).
    """
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    slides = np.asarray(slide_ids)
    if cluster_df.empty:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64), 0
    usable = cluster_df[(cluster_df["recurrent_all"]) & (cluster_df["n_rows"] >= K)]
    usable = usable.sort_values("n_rows", ascending=False)
    chosen = usable["cluster"].tolist()[:P]

    idx_out, lab_out = [], []
    for new_lab, cl in enumerate(chosen):
        members = np.where(labels == cl)[0]
        by_slide: dict = {}
        for m in members:
            by_slide.setdefault(slides[m], []).append(int(m))
        for lst in by_slide.values():
            rng.shuffle(lst)
        slide_keys = list(by_slide.keys())
        rng.shuffle(slide_keys)
        picked: list = []
        while len(picked) < K:
            progressed = False
            for s in slide_keys:
                if by_slide[s]:
                    picked.append(by_slide[s].pop())
                    progressed = True
                    if len(picked) >= K:
                        break
            if not progressed:
                break
        idx_out.extend(picked)
        lab_out.extend([new_lab] * len(picked))
    return np.asarray(idx_out, dtype=np.int64), np.asarray(lab_out, dtype=np.int64), len(chosen)


def suggest_temperature(pos: float, neg: float, P: int) -> float:
    """T* ≈ (pos - neg) / ln(P - 1), pos = median same-cluster-diff-slide cosine, neg = median random cosine (NaN if P < 3 or pos <= neg)."""
    if P is None or P < 3 or not np.isfinite(pos) or not np.isfinite(neg) or pos <= neg:
        return float("nan")
    return float((pos - neg) / np.log(P - 1))


def _softmax_rows(G: np.ndarray, T: float) -> np.ndarray:
    Z = G / T
    Z = Z - Z.max(axis=1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(axis=1, keepdims=True)


def temperature_table(X_synth: np.ndarray, synth_labels: np.ndarray, T_grid: Sequence[float]) -> pd.DataFrame:
    """
    Per T: mean perplexity exp(H(softmax(row / T))) and mean in-niche mass on a synthetic batch
    (softmax over G = X_synth @ X_synthᵀ, self included).
    """
    X_synth = np.asarray(X_synth, dtype=np.float64)
    synth_labels = np.asarray(synth_labels)
    if X_synth.shape[0] < 2:
        return pd.DataFrame(columns=["T", "perplexity", "in_niche_mass"])
    G = X_synth @ X_synth.T
    same = synth_labels[:, None] == synth_labels[None, :]
    rows = []
    for T in T_grid:
        Pm = _softmax_rows(G, float(T))
        H = -(np.where(Pm > 0, Pm * np.log(Pm), 0.0)).sum(axis=1)
        rows.append(
            {
                "T": float(T),
                "perplexity": round(float(np.exp(H).mean()), 4),
                "in_niche_mass": round(float((Pm * same).sum(axis=1).mean()), 4),
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- batch-leakage probe
def within_tissue_slide_probe(
    Xn: np.ndarray,
    slide_ids: np.ndarray,
    tissue_ids: np.ndarray,
    seed: int = 0,
    max_samples: int = 100_000,
    min_per_slide: int = 50,
) -> pd.DataFrame:
    """Per tissue with >= 2 slides: balanced accuracy of a logistic probe predicting the slide, vs chance 1 / n_slides."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import balanced_accuracy_score

    Xn = np.asarray(Xn, dtype=np.float32)
    slides = np.asarray(slide_ids)
    tissues = np.asarray(tissue_ids)
    rng = np.random.default_rng(seed)

    rows = []
    for t in sorted(set(tissues.tolist())):
        mask = tissues == t
        if np.unique(slides[mask]).size < 2:
            continue
        Xt, yt = Xn[mask], slides[mask]
        if Xt.shape[0] > max_samples:
            sub = rng.choice(Xt.shape[0], max_samples, replace=False)
            Xt, yt = Xt[sub], yt[sub]
        uy, cy = np.unique(yt, return_counts=True)
        keep_sl = set(uy[cy >= min_per_slide].tolist())
        sel = np.isin(yt, list(keep_sl))
        Xt, yt = Xt[sel], yt[sel]
        if np.unique(yt).size < 2:
            continue
        Xtr, Xte, ytr, yte = train_test_split(Xt, yt, test_size=0.3, random_state=seed, stratify=yt)
        clf = make_pipeline(
            StandardScaler(with_mean=True),
            LogisticRegression(max_iter=200, class_weight="balanced"),
        )
        clf.fit(Xtr, ytr)
        n_sl = int(np.unique(yt).size)
        bacc = float(balanced_accuracy_score(yte, clf.predict(Xte)))
        rows.append(
            {
                "tissue": t,
                "n_slides": n_sl,
                "chance": round(1.0 / n_sl, 4),
                "balanced_accuracy": round(bacc, 4),
                "excess_over_chance": round(bacc - 1.0 / n_sl, 4),
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- training-param selection
def headcap_weights(group_of_row: np.ndarray, C: float) -> np.ndarray:
    """Per-row head-cap sampling weight w = min(1, C / count(group)) (float64, aligned to the rows)."""
    g = np.asarray(group_of_row)
    uniq, counts = np.unique(g, return_counts=True)
    w_by = {t: min(1.0, float(C) / float(n)) for t, n in zip(uniq, counts)}
    return np.array([w_by[t] for t in g], dtype=np.float64)


def headcap_metrics(counts: dict, C: float) -> dict:
    """Balance metrics of P ∝ min(count(g), C) from a {group: count} dict (max_oversampling = N / Σ min(count, C))."""
    groups = list(counts.keys())
    c = np.array([counts[t] for t in groups], dtype=np.float64)
    N = c.sum()
    m = np.minimum(c, float(C))
    p = m / m.sum()
    eff = float(np.exp(-(p[p > 0] * np.log(p[p > 0])).sum()))
    capped = [groups[i] for i in np.argsort(-c) if c[i] > float(C)]
    return {"cap": int(round(float(C))), "max_share": round(float(p.max()), 4),
            "effective_groups": round(eff, 3), "n_groups": len(groups),
            "max_oversampling": round(float(N / m.sum()), 3), "n_capped": len(capped),
            "capped_groups": capped}


def recommend_headcap(counts: dict, max_oversampling: float = 1.3) -> float:
    """Smallest cap C with N / Σ min(count, C) <= max_oversampling (binary search)."""
    c = np.array(list(counts.values()), dtype=np.float64)
    N = c.sum()
    lo, hi = 1.0, float(c.max())
    for _ in range(64):
        C = 0.5 * (lo + hi)
        if N / np.minimum(c, C).sum() > max_oversampling:
            lo = C
        else:
            hi = C
    return float(hi)


def simulate_batches(weights: np.ndarray, batch_size: int, n_batches: int, seed: int = 0):
    """List of batch index arrays: weighted sampling without replacement within each batch (Efraimidis-Spirakis keys)."""
    weights = np.asarray(weights, dtype=np.float64)
    n = weights.shape[0]
    bs = int(min(batch_size, n))
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(int(n_batches)):
        keys = rng.exponential(size=n) / np.clip(weights, 1e-12, None)
        out.append(np.argpartition(keys, bs - 1)[:bs])
    return out


def batch_sims(Xn: np.ndarray, batches):
    """Per-batch cosine similarity matrices, from index arrays into Xn."""
    Xn = np.asarray(Xn, dtype=np.float32)
    return [Xn[b] @ Xn[b].T for b in batches]


def batch_cooccurrence_rate(sims, sim_threshold: float = 0.5) -> float:
    """Mean fraction of within-batch off-diagonal pairs with cosine >= sim_threshold, over per-batch similarity matrices."""
    fr = []
    for S in sims:
        B = S.shape[0]
        m = (S >= sim_threshold)
        np.fill_diagonal(m, False)
        fr.append(m.sum() / max(B * (B - 1), 1))
    return float(np.mean(fr)) if fr else float("nan")


def soft_target_precision_curve(sims, T_grid, sim_threshold: float = 0.5) -> pd.DataFrame:
    """
    Per T, over per-batch cosine similarity matrices: self_mass (mean diagonal of softmax(sim / T)), offdiag_mass
    (1 - self_mass) and precision_similar (fraction of the off-diagonal mass on pairs with cosine >= sim_threshold).
    """
    rows = []
    for T in T_grid:
        self_m, off_m = [], []
        prec_num = prec_den = 0.0
        for S in sims:
            Z = S / float(T)
            Z = Z - Z.max(axis=1, keepdims=True)
            P = np.exp(Z)
            P /= P.sum(axis=1, keepdims=True)
            diag = np.diagonal(P)
            offdiag = 1.0 - diag
            simmask = (S >= sim_threshold).copy()
            np.fill_diagonal(simmask, False)
            sim_mass = (P * simmask).sum(axis=1)
            self_m.append(float(diag.mean()))
            off_m.append(float(offdiag.mean()))
            prec_num += float(sim_mass.sum())
            prec_den += float(offdiag.sum())
        rows.append({"T": float(T), "self_mass": round(float(np.mean(self_m)), 4),
                     "offdiag_mass": round(float(np.mean(off_m)), 4),
                     "precision_similar": round(prec_num / prec_den, 4) if prec_den > 1e-9 else float("nan")})
    return pd.DataFrame(rows)


def recommend_temperature(curve_df: pd.DataFrame, precision_floor: float = 0.9,
                          min_offdiag: float = 0.02):
    """
    Highest T with offdiag_mass >= min_offdiag and precision_similar >= precision_floor. Returns (T, ok);
    if no T qualifies, the most precise T with ok=False.
    """
    df = curve_df.dropna(subset=["precision_similar"])
    ok = df[(df["precision_similar"] >= precision_floor) & (df["offdiag_mass"] >= min_offdiag)]
    if len(ok):
        return float(ok.sort_values("T").iloc[-1]["T"]), True
    if not len(df):
        return float("nan"), False
    return float(df.sort_values("precision_similar").iloc[-1]["T"]), False


# --------------------------------------------------------------------------- verdict
def decide_verdict(
    med_R_all: float,
    med_N_coherent: float,
    n_tissues_recurrent: int,
    n_tissues_total: int,
    iqr_R_all: float,
    gate_K: int = 32,
):
    """Sampler verdict (GO / CONDITIONAL / NO-GO) from median-over-k metrics and fixed thresholds. Returns (verdict, reasons)."""
    reasons = []
    ng = f"N_coherent({gate_K})"
    spread = int(n_tissues_recurrent)
    n_lo = bool(np.isfinite(med_N_coherent) and med_N_coherent < 15)

    if (spread <= 2) or (med_R_all < 0.30) or n_lo:
        if spread <= 2:
            reasons.append(f"cross-slide recurrence in only {spread}/{n_tissues_total} tissues")
        if med_R_all < 0.30:
            reasons.append(f"R_all={med_R_all:.2f} < 0.30")
        if n_lo:
            reasons.append(f"{ng}={int(med_N_coherent)} < 15")
        return "NO-GO", reasons or ["below thresholds"]

    if (spread >= max(6, n_tissues_total // 2) and med_R_all >= 0.50
            and np.isfinite(med_N_coherent) and med_N_coherent >= 30 and iqr_R_all <= 0.15):
        return "GO", [f"recurrence in {spread}/{n_tissues_total} tissues; R_all={med_R_all:.2f}; "
                      f"{ng}={int(med_N_coherent)}"]

    reasons.append(f"recurrence in {spread}/{n_tissues_total} tissues")
    reasons.append(f"R_all={med_R_all:.2f}")
    if np.isfinite(med_N_coherent):
        reasons.append(f"{ng}={int(med_N_coherent)}")
    if iqr_R_all > 0.15:
        reasons.append(f"IQR(R_all)={iqr_R_all:.2f} > 0.15")
    return "CONDITIONAL", reasons
