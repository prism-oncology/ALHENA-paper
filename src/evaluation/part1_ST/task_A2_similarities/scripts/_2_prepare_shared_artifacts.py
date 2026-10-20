#!/usr/bin/env python
"""
Prepare the shared artifacts of a Task A.2 eval_id (cell-level evaluation).

Saved in <ROOT_HistoST> / eval / A-2_eval_st_predictions / <eval_id> / shared /
"""

import argparse
from pathlib import Path
import sys

# Project imports
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src_A2.common import configure_logger
from src_A2.shared_artifacts import prepare_shared_artifacts

from src_A2.common import src_import
get_full_path = src_import("get_full_path")



def main(args: argparse.Namespace) -> None:
    eval_root = get_full_path("eval", "A-2_eval_st_predictions", args.eval_id)
    logger = configure_logger(eval_root / "shared" / "logs" / "prepare_shared_artifacts.log", name=f"A2_prepare_{args.eval_id}")
    summary = prepare_shared_artifacts(eval_root=eval_root, logger=logger)
    print(f"Prepared shared artifacts under: {eval_root / 'shared'}")
    print(summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build shared artifacts for A-2 cell-level evaluation.")
    
    parser.add_argument("--eval_id", type=str, 
                        required=True, 
                        help="Existing eval_id created by _1_setup.py")
    
    args = parser.parse_args()
    main(args)
