#!/usr/bin/env python
"""
Fill the settings folder of a Task A.3 eval_id from its eval_config.yaml (frozen cell rows, class vocabulary,
slide and area splits, per-fold class coverage).

Saved in <ROOT_HistoST> / eval / A-3_eval_celltype_predictions / <eval_id> / settings /
    - protocol_manifest.json: frozen protocol and paths of the files below
    - rows/: per-cell rows
    - labels/: class vocabulary, class support, per-fold class coverage
    - splits/, area_splits/: slide-level and within-slide splits
    - manifests/excluded_slides.json: excluded slides
    - informations/freeze_log.txt: log of this step
"""

import argparse
import logging
from pathlib import Path
import sys

# Add the task_A3 dir to sys.path so that 'src_A3' can be imported
# (ensure_imports() then adds the repo root and the task_A1 dir)
task_A3_dir = Path(__file__).resolve().parents[1]
if str(task_A3_dir) not in sys.path:
    sys.path.insert(0, str(task_A3_dir))

from src_A3.common import ensure_imports

ensure_imports()

from src_A3.common import mkdir, resolve_a3_eval_task_root
from src_A3.protocols import fill_a3_settings_folder

logger = logging.getLogger(__name__)

FREEZE_LOG_NAME = "freeze_log.txt"




def _configure_freeze_logging(eval_dir: Path) -> Path:
    """Configure the root logger (INFO) to write to stdout and to settings/informations/freeze_log.txt (overwritten)."""
    info_dir = mkdir(eval_dir / "settings" / "informations")
    log_path = info_dir / FREEZE_LOG_NAME
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, mode="w", encoding="utf-8"),
        ],
        force=True,
    )
    return log_path


def main(args: argparse.Namespace) -> None:
    """Freeze all protocol-fixed manifests for one Task A.3 eval_id."""

    eval_dir = resolve_a3_eval_task_root() / args.eval_id
    if not eval_dir.exists():
        raise FileNotFoundError(f"Eval directory not found: {eval_dir}")

    log_path = _configure_freeze_logging(eval_dir)

    logger.info("Filling settings for eval_id='%s'...", args.eval_id)
    logger.info("Freeze log: %s", log_path)

    protocol_manifest = fill_a3_settings_folder(eval_dir)

    logger.info("Done.")
    logger.info("Protocol manifest: %s", eval_dir / "settings" / "protocol_manifest.json")




if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Persist all protocol-fixed manifests for a Task A.3 (cell-type prediction) evaluation."
    )

    parser.add_argument("--eval_id", type=str,
                        required=True,
                        help="Evaluation ID, e.g. 'eval_1'")

    args = parser.parse_args()
    main(args)
