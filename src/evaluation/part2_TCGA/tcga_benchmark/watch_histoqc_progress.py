#!/usr/bin/env python3
"""
Read-only progress monitor for _9_run_histoqc.py: counts the masks published in the shared HistoQC store for the
current run_key (--histoqc_config must match the one of _9) and prints the progress and ETA.
"""
from __future__ import annotations

import argparse
import glob
import importlib.metadata
import sys
import time
from pathlib import Path

# Add the project root to sys.path so that 'src' can be imported, and this
# script's own dir so that the sibling '_taskb_common' module is importable.
project_root = Path(__file__).resolve().parents[4]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

from src.utils.constants import get_full_path
from _taskb_common import histoqc_run_key, sha256_path, shared_histoqc_slide_dir


DEFAULT_HISTOQC_CONFIG = (
    Path(__file__).resolve().parent / "histoqc_configs" / "tcga_taskb_v1.ini"
)
OK_DOWNLOAD_STATUSES = {
    "downloaded_md5_ok",
    "skipped_existing_md5_ok",
    "skipped_existing_size_ok",
}


def _detect_histoqc_version() -> str:
    try:
        return importlib.metadata.version("histoqc")
    except Exception:
        return "unknown"


def main(args: argparse.Namespace) -> int:
    data_root = get_full_path("eval", "eval_datasets", "TCGA_eval_data")

    config = Path(args.histoqc_config) if args.histoqc_config else DEFAULT_HISTOQC_CONFIG
    if not config.exists():
        raise FileNotFoundError(config)
    version = _detect_histoqc_version()
    run_key = histoqc_run_key(sha256_path(config), version)

    # by_uuid root of the shared content-addressed mask store.
    by_uuid = shared_histoqc_slide_dir(data_root, "X", run_key).parent.parent
    pattern = str(by_uuid / "*" / run_key / "mask_record.json")

    manifest = (
        data_root / "selected_slide_cache" / args.benchmark_id
        / "manifests" / "selected_slide_download_manifest.tsv"
    )
    if not manifest.exists():
        raise FileNotFoundError(f"Run _8_download_selected_slides.py first; missing {manifest}")
    df = pd.read_csv(manifest, sep="\t", dtype=str, keep_default_na=False)
    total = int(df["download_status"].isin(OK_DOWNLOAD_STATUSES).sum())

    print(f"benchmark : {args.benchmark_id}")
    print(f"run_key   : {run_key}  (histoqc {version})")
    print(f"store     : {by_uuid}")
    print(f"total     : {total} slides   (polling every {args.interval}s; Ctrl-C to stop)\n")

    start_n = start_t = None
    while True:
        n = len(glob.glob(pattern))
        now = time.time()
        if start_n is None:
            start_n, start_t = n, now
        did = n - start_n
        elapsed = now - start_t
        rate = did / elapsed if (elapsed > 0 and did > 0) else 0.0  # slides/sec since start
        eta = (max(0, total - n) / rate) if rate > 0 else None
        pct = (100.0 * n / total) if total else 0.0
        eta_s = f"{eta / 60:.0f} min" if eta is not None else "warming up..."
        print(
            f"{time.strftime('%H:%M:%S', time.localtime())}  {n:4d}/{total}  "
            f"({pct:.1f}%)  +{did} new  rate={rate * 60:.1f}/min  ETA={eta_s}"
        )
        if n >= total:
            print(f"\nDONE -- all {total} masks present in the store.")
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Read-only live progress monitor for _9_run_histoqc.py.")
    
    ap.add_argument("--benchmark_id", type=str, 
                    required=True,
                    help="Benchmark ID whose HistoQC progress to watch.")
    
    ap.add_argument("--histoqc_config", type=Path, 
                    default=None,
                    help=("HistoQC .ini config; must match the one passed to _9. Defaults to the "
                          f"bundled {DEFAULT_HISTOQC_CONFIG.name}."))
    
    ap.add_argument("--interval", type=int, 
                    default=20,
                    help="Polling interval in seconds (default 20).")
    
    args = ap.parse_args()
    raise SystemExit(main(args))
