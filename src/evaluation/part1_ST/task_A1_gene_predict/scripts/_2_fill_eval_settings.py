#!/usr/bin/env python
"""
Fill the settings folder of a Task A.1 evaluation (protocol manifests shared by all H&E encoders).

Saved in <ROOT_HistoST> / eval / A-1_eval_st_predictions / <eval_id> / settings /
    - rows/, splits/, area_splits/, genes/, target_norm/, noise_ceiling/: protocol files
    - protocol_manifest.json: protocol manifest
"""

import argparse
import logging
from pathlib import Path
import sys

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src_A1.common import resolve_eval_task_root, rmtree_appledouble_safe
from src_A1.protocols import fill_settings_folder


logger = logging.getLogger(__name__)


# Settings subfolders written by this script (removed by --overwrite)
PROTOCOL_OWNED_SETTINGS_DIRS = (
    "rows",
    "splits",
    "area_splits",
    "genes",
    "target_norm",
    "noise_ceiling",
)

# informations/ files written by _1_setup.py (kept by --overwrite)
INFO_DIR_NAME = "informations"
SETUP_OWNED_INFO_FILES = frozenset({
    "README.md",
    "dataset_slide_summary.csv",
    "dataset_summary_by_subset.csv",
})


def _clear_protocol_settings(settings_dir: Path) -> None:
    """Remove the settings files written by this script (the files written by _1_setup.py are kept)."""
    for name in PROTOCOL_OWNED_SETTINGS_DIRS:
        target = settings_dir / name
        if target.is_dir():
            rmtree_appledouble_safe(target)

    info_dir = settings_dir / INFO_DIR_NAME
    if info_dir.is_dir():
        # Keep the files written by _1_setup.py (and their macOS ._ sidecars)
        for p in sorted(info_dir.iterdir()):
            if p.name.removeprefix("._") in SETUP_OWNED_INFO_FILES:
                continue
            if p.is_dir():
                rmtree_appledouble_safe(p)
            else:
                p.unlink(missing_ok=True)

    # Top-level files written by this script (.tmp: leftover of save_json_atomic)
    for stale in ("protocol_manifest.json", "protocol_manifest.json.tmp", "scgpt_manifest.json"):
        p = settings_dir / stale
        if p.is_file():
            p.unlink()


def main(args: argparse.Namespace) -> None:

    eval_dir = resolve_eval_task_root() / args.eval_id
    if not eval_dir.exists():
        raise FileNotFoundError(f"Eval directory not found: {eval_dir}")

    settings_dir = eval_dir / "settings"
    manifest_path = settings_dir / "protocol_manifest.json"
    # protocol_manifest.json is written last: its presence marks a completed fill
    if manifest_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Settings are already filled for eval_id='{args.eval_id}' "
            f"({manifest_path}). Pass --overwrite to rebuild them."
        )
    if args.overwrite and settings_dir.is_dir():
        logger.info("Clearing protocol-owned settings under %s", settings_dir)
        _clear_protocol_settings(settings_dir)

    logger.info("Filling settings for eval_id='%s'...", args.eval_id)

    fill_settings_folder(eval_dir)

    logger.info("Done.")
    logger.info("Protocol manifest: %s", manifest_path)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Persist all protocol-fixed manifests for a Task A.1 evaluation.")

    parser.add_argument("--eval_id", type=str,
                        required=True,
                        help="Evaluation ID, e.g. 'eval_1'")

    parser.add_argument("--overwrite", action="store_true",
                        help="Rebuild the settings from scratch (eval_config.yaml and the files written "
                             "by _1_setup.py are kept).")

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    main(args)
