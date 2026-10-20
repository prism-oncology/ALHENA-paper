#!/usr/bin/env python3
"""
Read-only progress monitor for _15_build_tile_rgb_cache.py: counts the tiles.h5 files published in the shared tile
store for the final-cohort slides of the benchmark and prints the progress and ETA.
"""
from __future__ import annotations

import argparse
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
from _taskb_common import (
    _truthy_value,
    read_json,
    read_yaml,
    shared_tile_pixels_dir,
    shared_tiles_store_root,
    tile_codec_recipe_id,
)


# Codec defaults (must match _15_build_tile_rgb_cache.py)
DEFAULT_CACHE_FORMAT = "jpeg"
DEFAULT_QUALITY = 95
DEFAULT_CHROMA = "444"
DEFAULT_PNG_LEVEL = 6


def _resolve_codec_recipe(settings_dir: Path, args: argparse.Namespace) -> dict:
    """Tile codec from benchmark_config.yaml (tile_recipe.codec) with the CLI overrides, as in _15._resolve_codec."""
    cfg_path = settings_dir / "benchmark_config.yaml"
    cfg = read_yaml(cfg_path) if cfg_path.exists() else {}
    codec = dict(((cfg or {}).get("tile_recipe") or {}).get("codec") or {})
    codec.setdefault("format", DEFAULT_CACHE_FORMAT)
    codec.setdefault("quality", DEFAULT_QUALITY)
    codec.setdefault("chroma", DEFAULT_CHROMA)
    codec.setdefault("png_level", DEFAULT_PNG_LEVEL)
    if args.cache_format is not None:
        codec["format"] = str(args.cache_format)
    if args.quality is not None:
        codec["quality"] = int(args.quality)
    if args.png_compression is not None:
        codec["png_level"] = int(args.png_compression)
    if args.chroma is not None:
        codec["chroma"] = str(args.chroma)
    codec["format"] = str(codec["format"]).lower()
    codec["quality"] = int(codec["quality"])
    codec["png_level"] = int(codec["png_level"])
    codec["chroma"] = str(codec["chroma"])
    return codec


def main(args: argparse.Namespace) -> int:
    data_root = get_full_path("eval", "eval_datasets", "TCGA_eval_data")
    benchmark_root = data_root / "benchmarks" / args.benchmark_id
    settings_dir = benchmark_root / "settings"

    # Set recipe (benchmark_asset_links.json) and codec
    links_path = settings_dir / "benchmark_asset_links.json"
    if not links_path.exists():
        raise FileNotFoundError(f"Run _10_build_tile_asset.py first; missing {links_path}")
    links = read_json(links_path)
    if not isinstance(links, dict):
        raise RuntimeError("benchmark_asset_links.json must be a mapping")
    set_recipe_id = str(links.get("tile_set_recipe_id") or "")
    if not set_recipe_id:
        raise RuntimeError(
            "tile_set_recipe_id missing in benchmark_asset_links.json. Re-run _10_build_tile_asset.py."
        )
    codec = _resolve_codec_recipe(settings_dir, args)
    codec_recipe_id = tile_codec_recipe_id(set_recipe_id=set_recipe_id, codec_recipe=codec)

    # Total: slides of the final_case_eligible cases (as cached by _15)
    idx_path = benchmark_root / "shared" / "tiles" / "case_tile_index.parquet"
    if not idx_path.exists():
        raise FileNotFoundError(f"Run _11_build_case_tile_index.py first; missing {idx_path}")
    final_case_path = benchmark_root / "shared" / "cohort" / "final_case_manifest.csv"
    if not final_case_path.exists():
        raise FileNotFoundError(f"Run _11_build_case_tile_index.py first; missing {final_case_path}")
    used = pd.read_parquet(idx_path, columns=["slide_file_uuid", "case_id"])
    final_case = pd.read_csv(final_case_path, dtype=str, keep_default_na=False)
    if not {"final_case_eligible", "case_id"}.issubset(final_case.columns):
        raise RuntimeError("final_case_manifest.csv missing final_case_eligible/case_id columns")
    eligible_cases = set(
        final_case.loc[final_case["final_case_eligible"].map(_truthy_value), "case_id"].astype(str)
    )
    used = used.loc[used["case_id"].astype(str).isin(eligible_cases)]
    used_uuids = sorted(used["slide_file_uuid"].astype(str).unique().tolist())
    total = len(used_uuids)
    if total == 0:
        raise RuntimeError(f"No final-eligible slides in {idx_path} (final_case_manifest.csv)")

    by_uuid = shared_tiles_store_root(data_root) / "by_uuid"
    print(f"benchmark : {args.benchmark_id}")
    print(
        f"recipe    : set={set_recipe_id}  codec={codec_recipe_id}  "
        f"({codec['format']} q{codec['quality']} {codec['chroma']})"
    )
    print(f"store     : {by_uuid}")
    print(f"total     : {total} slides   (polling every {args.interval}s; Ctrl-C to stop)\n")

    # Count the slides whose tiles.h5 exists (published atomically, so complete)
    start_n = start_t = None
    while True:
        n = sum(
            1 for u in used_uuids
            if (shared_tile_pixels_dir(data_root, u, set_recipe_id, codec_recipe_id) / "tiles.h5").exists()
        )
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
            print(f"\nDONE -- all {total} tile blobs present in the store.")
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Read-only live progress monitor for _15_build_tile_rgb_cache.py.")

    ap.add_argument("--benchmark_id", type=str,
                    required=True,
                    help="Benchmark ID whose tile RGB cache progress to watch.")

    ap.add_argument("--interval", type=int,
                    default=20,
                    help="Polling interval in seconds (default 20).")

    # Debug codec overrides (only if _15 was run with the same overrides)
    ap.add_argument("--cache_format", type=str,
                    choices=["jpeg", "webp", "png"],
                    default=None,
                    help="Debug override; must match the --cache_format passed to _15.")

    ap.add_argument("--quality", type=int,
                    default=None,
                    help="Debug override of the lossy quality; must match _15.")

    ap.add_argument("--chroma", type=str,
                    default=None,
                    help="Debug override of the jpeg chroma tag (e.g. 444); must match _15.")

    ap.add_argument("--png_compression", type=int,
                    default=None,
                    help="Debug override of the png compression level; must match _15.")

    args = ap.parse_args()
    raise SystemExit(main(args))
