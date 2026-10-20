#!/usr/bin/env python3
"""
Geometry of the Thunder tile embeddings, per encoder and dataset: mean pairwise cosine (raw and centred),
mean cosine within and between classes and their gap (raw and centred), and effective dimensionality.

Saved in <ROOT_HistoST> / eval / C_eval_thunder /
    - thunder_geometry_n<n>_<splits>_seed<seed>[_ds-...][_noeffdim].csv: one row per (encoder, dataset)
"""
import argparse, csv, glob, os, re, sys
from pathlib import Path

import numpy as np

try:
    import h5py
except ImportError:
    sys.exit("h5py not found -- run inside the thunder env")

# Add the project root to sys.path so that 'src' can be imported
project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.constants import get_full_path

# Task C root (as in _1_run_thunder_benchmark.build_paths())
TASK_C_ROOT_PARTS = ("eval", "C_eval_thunder")

ARMS = {"t-1e-1": "UNI2-h", "t46e13": "ALHENA", "t-2e-1": "Virchow2",
        "t-3e-1": "H-optimus-1", "t9001e0": "SEAL"}


def arm_name(d):
    b = os.path.basename(d.rstrip("/"))
    tag = b[len("HEencoder_"):-len("_he")] if b.startswith("HEencoder_") else b
    return ARMS.get(tag, tag), tag


def output_csv_name(args, splits):
    """Output CSV name, including the sampling parameters."""
    parts = ["thunder_geometry", "n%d" % args.n, "-".join(splits), "seed%d" % args.seed]
    wanted = [x for x in args.datasets.split(",") if x]
    if wanted:
        parts.append("ds-" + "-".join(wanted))
    if args.no_eff_dim:
        parts.append("noeffdim")
    return re.sub(r"[^A-Za-z0-9._-]+", "_", "_".join(parts)) + ".csv"


def file_keys(fe):
    """(n_datasets, key list) of one embeddings.h5; the key list is None when the keys are "0".."N-1"."""
    n_total = len(fe)
    if n_total > 0 and "0" in fe and str(n_total - 1) in fe:
        return n_total, None
    return n_total, list(fe.keys())


def load(emb_paths, n, rng):
    """Sample n (embedding, label) pairs. Returns row-normalised X and labels."""
    handles, sizes, keylists = [], [], []
    for p in emb_paths:
        lp = os.path.join(os.path.dirname(p), "labels.h5")
        try:
            fe = h5py.File(p, "r")
            fl = h5py.File(lp, "r") if os.path.exists(lp) else None
        except OSError:
            continue
        handles.append((fe, fl))
        n_total, explicit = file_keys(fe)
        sizes.append(n_total)
        keylists.append(explicit)

    def _close():
        for fe, fl in handles:
            fe.close()
            if fl is not None:
                fl.close()

    total = int(sum(sizes))
    if total == 0:
        _close()
        return None, None

    take = min(n, total)
    sel = rng.choice(total, take, replace=False) if total > take else np.arange(total)
    # Sorted indices: sequential reads (the statistics below do not depend on the order)
    sel = np.sort(np.asarray(sel))
    offs = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)

    vecs, labs, dim = [], [], None
    for fi, (fe, fl) in enumerate(handles):
        local = sel[(sel >= offs[fi]) & (sel < offs[fi + 1])] - offs[fi]
        explicit = keylists[fi]
        for j in local:
            k = str(int(j)) if explicit is None else explicit[int(j)]
            v = np.asarray(fe[k], dtype=np.float32).ravel()
            if dim is None:
                dim = v.size
            elif v.size != dim:
                continue
            nrm = float(np.linalg.norm(v))
            if not np.isfinite(nrm) or nrm == 0:
                continue
            vecs.append(v / nrm)
            labs.append(int(np.asarray(fl[k]).ravel()[0]) if fl is not None else -1)
    _close()
    if len(vecs) < 10:
        return None, None
    return np.asarray(vecs, dtype=np.float64), np.asarray(labs)


def mean_pairwise(U):
    """mean_{i != j} u_i . u_j, from ||sum u||^2 -- no N x N matrix."""
    n = len(U)
    s = U.sum(axis=0)
    return (float(s @ s) - n) / (n * (n - 1))


def within_between(U, y):
    """Mean cosine within a class and between classes, same identity per class."""
    n = len(U)
    s = U.sum(axis=0)
    tot_sum, tot_cnt = float(s @ s) - n, n * (n - 1)
    w_sum = w_cnt = 0.0
    for c in np.unique(y):
        if c < 0:
            return None, None
        m = y == c
        nc = int(m.sum())
        if nc < 2:
            continue
        sc = U[m].sum(axis=0)
        w_sum += float(sc @ sc) - nc
        w_cnt += nc * (nc - 1)
    b_sum, b_cnt = tot_sum - w_sum, tot_cnt - w_cnt
    if w_cnt == 0 or b_cnt == 0:
        return None, None
    return w_sum / w_cnt, b_sum / b_cnt


def eff_dim(U):
    """Participation ratio of the centred covariance, (sum L)^2 / sum L^2, computed from the Gram matrix traces."""
    V = U - U.mean(axis=0)
    # V.T @ V (d x d) and V @ V.T (n x n) share their non-zero spectrum; take the smaller.
    G = V.T @ V if V.shape[0] >= V.shape[1] else V @ V.T
    tot = float(np.trace(G))
    sq = float((G * G).sum())
    return float(tot * tot / sq) if tot > 0 and sq > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=2000)
    ap.add_argument("--splits", default="train,val")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--datasets", default="")
    ap.add_argument("--no-eff-dim", action="store_true", help="skip the effective dimensionality")
    args = ap.parse_args()

    splits = [s for s in args.splits.split(",") if s]
    wanted = [x for x in args.datasets.split(",") if x]

    # scratch=True is a no-op off Jean Zay; there it selects the $SCRATCH root.
    root = Path(get_full_path(*TASK_C_ROOT_PARTS, as_str=True, scratch=True))
    print("Task C root: %s" % root)
    arm_dirs = sorted(glob.glob(os.path.join(str(root), "HEencoder_*")))
    if not arm_dirs:
        msg = "no HEencoder_* directories under " + str(root)
        alt = Path(get_full_path(*TASK_C_ROOT_PARTS, as_str=True))
        if alt != root:
            msg += ("\n(the non-scratch Task C root is %s -- that is where "
                    "_1_run_thunder_benchmark.py writes)" % alt)
        sys.exit(msg)

    rows = []
    hdr = ("%-12s %-20s %6s %8s %9s %9s %9s %8s %9s %8s"
           % ("arm", "dataset", "n", "cos_raw", "cos_cent", "within", "between",
              "gap", "gap_cent", "eff_dim"))
    print(hdr)
    print("-" * len(hdr))

    for d in arm_dirs:
        name, tag = arm_name(d)
        base = os.path.join(d, "_thunder_base", "embeddings")
        if not os.path.isdir(base):
            continue
        rng = np.random.default_rng(args.seed)
        for ds in sorted(x for x in os.listdir(base)
                         if os.path.isdir(os.path.join(base, x)) and (not wanted or x in wanted)):
            paths = []
            for sp in splits:
                paths += glob.glob(os.path.join(base, ds, "*", sp, "embeddings.h5"))
            U, y = load(paths, args.n, rng)
            if U is None:
                continue
            craw = mean_pairwise(U)
            Vc = U - U.mean(axis=0)
            Vc /= np.linalg.norm(Vc, axis=1, keepdims=True)
            ccen = mean_pairwise(Vc)
            w, b = within_between(U, y)
            wc, bc = within_between(Vc, y)
            ed = float("nan") if args.no_eff_dim else eff_dim(U)
            gap = (w - b) if (w is not None) else float("nan")
            gapc = (wc - bc) if (wc is not None) else float("nan")
            f = lambda v: ("%.2f" % (100 * v)) if v is not None else "--"
            print("%-12s %-20s %6d %8.2f %9.2f %9s %9s %8s %9s %8.1f"
                  % (name, ds, len(U), 100 * craw, 100 * ccen, f(w), f(b), f(gap), f(gapc), ed))
            rows.append(dict(arm=name, tag=tag, dataset=ds, n=len(U), cos_raw=craw, cos_centred=ccen,
                             within=w, between=b, gap=gap if w is not None else None,
                             gap_centred=gapc if wc is not None else None, eff_dim=ed))

    if not rows:
        sys.exit("no embeddings found under " + str(root))

    out_csv = root / output_csv_name(args, splits)
    with open(out_csv, "w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wtr.writeheader()
        wtr.writerows(rows)
    print("\nwrote %s" % out_csv)


if __name__ == "__main__":
    main()
