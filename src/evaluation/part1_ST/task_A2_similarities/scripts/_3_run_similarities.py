#!/usr/bin/env python
"""
Run the Task A.2 cell-level similarity evaluation of one HistoST checkpoint for an eval_id.

Saved in <ROOT_HistoST> / eval / A-2_eval_st_predictions / <eval_id> / HEencoder_t<training_num>e<checkpoint_num>_clip /
"""

import argparse
from pathlib import Path
import sys

# Project imports
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src_A2.runner import run_checkpoint_evaluation

from src_A2.common import src_import
get_full_path = src_import("get_full_path")




def main(args: argparse.Namespace) -> None:
    eval_root = get_full_path("eval", "A-2_eval_st_predictions", args.eval_id)
    out = run_checkpoint_evaluation(eval_root=eval_root, training_num=str(args.training_num), checkpoint_num=str(args.checkpoint_num))
    print(f"Finished checkpoint evaluation. Output root: {out}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run one-checkpoint A-2 cell-level similarity / transfer evaluation.")
    
    parser.add_argument("--eval_id", type=str, 
                        required=True, 
                        help="Existing eval_id")
    
    parser.add_argument("--training_num", type=str, 
                        required=True, 
                        help="HistoST training number, e.g. '46' for 'training_46'")
    
    parser.add_argument("--checkpoint_num", type=str, 
                        required=True, 
                        help="Checkpoint number, e.g. '13' for 'checkpoint_epoch_13.pth'")
    
    args = parser.parse_args()
    main(args)
