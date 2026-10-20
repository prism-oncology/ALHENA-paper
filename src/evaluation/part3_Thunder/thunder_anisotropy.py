#!/usr/bin/env python3
"""
Mean pairwise cosine between random tiles, per encoder and dataset, from the Thunder embeddings.

Saved in <ROOT_HistoST> / eval / C_eval_thunder /
    - thunder_anisotropy_n<n>_<splits>_seed<seed>[_ds-...].csv: one row per (encoder, dataset), then one overall row per encoder
"""
import argparse, csv, glob, os, re, sys
from pathlib import Path

import numpy as np

try:
    import h5py
except ImportError:
    sys.exit("h5py not found -- run inside the thunder env (module load / conda activate)")

# Add the project root to sys.path so that 'src' can be imported
project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.constants import get_full_path

# Task C root (as in _1_run_thunder_benchmark.build_paths())
TASK_C_ROOT_PARTS = ("eval", "C_eval_thunder")

# Encoder tag -> name
ARMS = {"t-1e-1": "UNI2-h", "t46e13": "ALHENA", "t-2e-1": "Virchow2",
        "t-3e-1": "H-optimus-1", "t9001e0": "SEAL"}

# Reference encoder of the differences (frozen backbone)
REFERENCE_ARM = "UNI2-h"

CSV_HEADER = ["scope", "arm", "tag", "dataset", "n_sampled",
              "mean_pairwise_cosine", "mean_pairwise_cosine_x100",
              "reference_arm", "diff_from_reference_x100", "n_datasets"]


def arm_name(d):
    b = os.path.basename(d.rstrip("/"))
    tag = b[len("HEencoder_"):-len("_he")] if b.startswith("HEencoder_") else b
    return ARMS.get(tag, tag), tag


def output_csv_name(args, splits):
    """Output CSV name, including the sampling parameters."""
    parts = ["thunder_anisotropy", "n%d" % args.n, "-".join(splits), "seed%d" % args.seed]
    wanted = [x for x in args.datasets.split(",") if x]
    if wanted:
        parts.append("ds-" + "-".join(wanted))
    return re.sub(r"[^A-Za-z0-9._-]+", "_", "_".join(parts)) + ".csv"


def mean_pairwise_cosine(paths, n, rng):
    """(mean pairwise cosine, count) of up to n L2-normalized embeddings sampled from the embeddings.h5 files."""
    handles, keys = [], []
    for p in paths:
        try:
            f = h5py.File(p, "r")
        except OSError:
            continue
        handles.append(f)
        keys += [(len(handles) - 1, k) for k in f.keys()]
    if not keys:
        return None, 0
    if len(keys) > n:
        keys = [keys[i] for i in rng.choice(len(keys), n, replace=False)]

    acc, count, dim = None, 0, None
    for hi, k in keys:
        v = np.asarray(handles[hi][k], dtype=np.float64).ravel()
        nrm = np.linalg.norm(v)
        if not np.isfinite(nrm) or nrm == 0:
            continue
        v /= nrm
        if acc is None:
            acc, dim = np.zeros_like(v), v.size
        elif v.size != dim:
            continue
        acc += v
        count += 1
    for f in handles:
        f.close()
    if count < 2:
        return None, count
    total = float(acc @ acc)                      # = sum_{i,j} u_i . u_j
    return (total - count) / (count * (count - 1)), count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=2000, help="tiles sampled per dataset (default 2000)")
    ap.add_argument("--splits", default="train,val", help="Thunder splits to sample from (default train,val)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--datasets", default="", help="optional comma-separated subset")
    args = ap.parse_args()

    splits = [s for s in args.splits.split(",") if s]
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

    per_arm, tags, order = {}, {}, []
    for d in arm_dirs:
        name, tag = arm_name(d)
        base = os.path.join(d, "_thunder_base", "embeddings")
        if not os.path.isdir(base):
            print("  (skipping %s -- no _thunder_base/embeddings)" % name, file=sys.stderr)
            continue
        wanted = [x for x in args.datasets.split(",") if x]
        dsets = sorted(x for x in os.listdir(base) if os.path.isdir(os.path.join(base, x))
                       and (not wanted or x in wanted))
        rng = np.random.default_rng(args.seed)
        res = {}
        for ds in dsets:
            paths = []
            for sp in splits:
                paths += glob.glob(os.path.join(base, ds, "*", sp, "embeddings.h5"))
            c, cnt = mean_pairwise_cosine(paths, args.n, rng)
            if c is not None:
                res[ds] = (c, cnt)
                print("  %-10s %-20s %7.2f  (n=%d)" % (name, ds, 100 * c, cnt), flush=True)
        if res:
            per_arm[name] = res
            tags[name] = tag
            order.append(name)

    if not per_arm:
        sys.exit("no embeddings found")

    common = sorted(set.intersection(*(set(v) for v in per_arm.values())))
    ref = per_arm.get(REFERENCE_ARM)
    ref_mean = None if ref is None else 100 * float(np.mean([ref[d][0] for d in common]))

    rows = []
    for name in order:
        for ds in sorted(per_arm[name]):
            c, cnt = per_arm[name][ds]
            diff = ""
            if ref is not None and name != REFERENCE_ARM and ds in ref:
                diff = round(100 * (c - ref[ds][0]), 4)
            rows.append(["per_dataset", name, tags[name], ds, cnt,
                         c, round(100 * c, 4), REFERENCE_ARM, diff, ""])

    print("\n=== mean pairwise cosine between random tiles, x100, over %d datasets ===" % len(common))
    for name in order:
        m = 100 * float(np.mean([per_arm[name][d][0] for d in common]))
        line = "%-12s %7.2f" % (name, m)
        diff = ""
        if ref_mean is not None and name != REFERENCE_ARM:
            diff = round(m - ref_mean, 4)
            line += "   difference from %s %+6.2f" % (REFERENCE_ARM, diff)
        print(line)
        rows.append(["overall", name, tags[name], "", "",
                     m / 100.0, round(m, 4), REFERENCE_ARM, diff, len(common)])

    out_csv = root / output_csv_name(args, splits)
    with open(out_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(CSV_HEADER)
        w.writerows(rows)
    print("\nwrote %s" % out_csv)

    print("\nCompare ALHENA's difference with its transformation-invariance gain.")
    print("Near 0: the invariance gain is not explained by anisotropy.")


if __name__ == "__main__":
    main()
