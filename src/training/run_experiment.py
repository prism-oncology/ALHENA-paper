#!/usr/bin/env python
"""
This script is the entry point for running the CLIP experiment for H&E and Spatial Transcriptomics.
"""

import os
os.environ["WANDB_DISABLE_CODE"] = "True"  
os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "1"  # keep ranks alive during long I/O
os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "INFO"  # set to INFO for more verbose logging
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"  # disable albumentations update messages
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # enable CPU fallback for MPS

import yaml
import logging
import argparse
import sys
import torch
import torch.distributed as dist
import datetime

# Add the project root to sys.path so that 'src' can be imported.
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.constants import get_full_path
from src.training.experiment import Experiment
from src.utils.utils import set_global_seed, const_distributed
from src.utils.default_histost_config import cfg_get, merge_with_defaults



def load_config(config_path: str) -> dict:
    """
    Loads the experiment configuration from a YAML file.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return merge_with_defaults(config)



def init_config(config_path: str, checkpoint_start_path: str = None, cross_run: bool = False) -> tuple:
    """
    Loads the configuration and the W&B run id to resume.

    - Normal run: config from the YAML.
    - Same-id resume (--checkpoint_name): config and W&B run id from the checkpoint.
    - Cross-run resume (--resume_state_from_checkpoint): config from the YAML, new W&B run.
    """
    # Load the configuration
    if cross_run:
        config = load_config(config_path)
        wandb_run_id = None
    elif not checkpoint_start_path:
        config = load_config(config_path)
        wandb_run_id = None
    else:
        init_checkpoint = torch.load(checkpoint_start_path, map_location="cpu", weights_only=False)
        config = merge_with_defaults(init_checkpoint["config"])
        wandb_run_id = init_checkpoint.get("wandb_run_id", None)
    return config, wandb_run_id



def init_logger(config: dict, rank: int) -> logging.Logger:

    # Get the log file path
    log_dir = get_full_path("train", "trainings", "trainings_HistoST", config["logging"]["training_id_histost"], "training", "log", as_str=True)
    config["logging"]["log_dir"] = log_dir
    os.makedirs(log_dir, exist_ok=True)
    log_path = log_dir + "/logfile.log"

    # Get wandb_dir parent
    wandb_dir = get_full_path("train", "trainings", "trainings_HistoST", config["logging"]["training_id_histost"], "training", as_str=True)
    config["logging"]["wandb_dir"] = wandb_dir
    
    # Configure the logging system
    level_str = config["logging"].get("level", "info")
    log_level = getattr(logging, level_str.upper(), logging.INFO)
    if (not config["distributed"]) or (rank == 0):
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(log_path),
                logging.StreamHandler()
            ]
        )
    else:
        # Other ranks either disable logging or set a high log level.
        logging.getLogger().setLevel(logging.WARNING)
    
    logger = logging.getLogger(__name__)
    logger.info("Starting experiment\n\n")
    logger.info(f"Logging to {log_path}")
    logger.info(f"Wandb dir inside: {wandb_dir}")

    return logger


def resolve_device(local_rank: int, cfg: dict) -> torch.device:
    if cfg["device_type"].startswith("cuda") and torch.cuda.is_available():
        return torch.device(f"cuda:{local_rank}")
    if cfg["device_type"] == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Run CLIP Experiment for H&E and Spatial Transcriptomics")

    parser.add_argument("--training_id", type=str, required=True, help="Training ID, used to find the config path")
    parser.add_argument("--checkpoint_name", type=str, help="Name of the checkpoint to restart from if training was interrupted (same-id resume: config reloaded from the checkpoint)")
    parser.add_argument("--resume_state_from_checkpoint", type=str, default=None,
                        help="Absolute path to a checkpoint of another run: restore its full training state, with the "
                             "config of this --training_id and a new W&B run (the source epochs are copied into the "
                             "JSONL and W&B). Mutually exclusive with --checkpoint_name.")
    parser.add_argument("--allow_schedule_mismatch", action="store_true",
                        help="For --resume_state_from_checkpoint: proceed even if the steps/epoch (GPU count x batch_size) "
                             "differ from the source run (the restored LR schedule is then invalid).")

    args = parser.parse_args()

    if args.checkpoint_name and args.resume_state_from_checkpoint:
        parser.error("--checkpoint_name and --resume_state_from_checkpoint are mutually exclusive "
                     "(same-id resume vs cross-run full-state resume).")

    # Initialize config
    config_path = get_full_path("train", "trainings", "trainings_HistoST", args.training_id, "training", f"{args.training_id}_histost.yaml", as_str=True)
    
    cross_run = bool(args.resume_state_from_checkpoint)

    checkpoint_start_path = None
    if args.checkpoint_name:
        checkpoint_start_path = get_full_path("train", "trainings", "trainings_HistoST", args.training_id, "training", "log", "checkpoints", args.checkpoint_name, as_str=True)
    elif cross_run:
        # Absolute path (get_full_path rejects absolute parts), checked before any heavy work
        checkpoint_start_path = args.resume_state_from_checkpoint
        if not os.path.isfile(checkpoint_start_path):
            parser.error(f"--resume_state_from_checkpoint not found: {checkpoint_start_path}")

    config, wandb_run_id = init_config(config_path, checkpoint_start_path, cross_run=cross_run)

    # Check if the cached dataset option is valid
    valid_cache = ["no", "lmdb_init"]
    cached_option = cfg_get(config, "data", "cached_dataset")
    if cached_option not in valid_cache:
        raise ValueError(f"cached_dataset must be one of {valid_cache}")

    # Look for the distributed training
    local_rank, world_size, rank = const_distributed()
    config["distributed"] = world_size > 1
    n_gpus = world_size

    # Initialize distributed process group if needed.
    if config["distributed"]:
        if not torch.cuda.is_available():
            raise RuntimeError("DDP with NCCL needs GPUs.")
        dist.init_process_group(backend="nccl", init_method="env://", timeout=datetime.timedelta(hours=12), world_size=world_size, rank=rank)
        torch.cuda.set_device(local_rank)

    # Set the device
    device = resolve_device(local_rank, config)
    config["device"] = device
    if device.type == "cuda":
        torch.cuda.set_device(device)

    # The logger can only be built once the config and rank are known
    logger = init_logger(config, rank)

    # Print the number of GPUs available and if distributed training is enabled
    logger.info(f"Number of GPUs available: {n_gpus}")
    logger.info(f"Distributed training enabled: {config['distributed']}")
    if config["distributed"]:
        logger.info(f"Global batch size will be {cfg_get(config, 'training', 'batch_size') * n_gpus}")

    # Print config
    if cross_run:
        logger.info(f"Cross-run full-state resume from: {checkpoint_start_path} (config from this training_id's YAML). config is: {config}\n\n")
    elif not checkpoint_start_path:
        logger.info(f"Using configuration: {config}\n\n")
    else:
        logger.info(f"Using configuration from checkpoint: {checkpoint_start_path}, config is: {config}\n\n")

    # Set the global seed for reproducibility
    base_seed = cfg_get(config, "seed")
    set_global_seed(base_seed)

    # If a checkpoint path is provided, add it to the configuration
    if checkpoint_start_path:
        config["checkpoint_start_path"] = checkpoint_start_path

    # Log the cached dataset option and its implications
    if config["distributed"] and cached_option == "lmdb_init":
        logger.info("Using 'lmdb_init' in distributed mode: the cache is built once on disk and then reused read-only by all ranks.")

    # Initialize the Experiment
    experiment = Experiment(config=config, wandb_run_id=wandb_run_id, local_rank=local_rank, world_size=world_size, rank=rank, distributed=world_size > 1, cross_run_resume=cross_run, allow_schedule_mismatch=args.allow_schedule_mismatch)
    experiment.run_experiment()

    # If using DDP, cleanup
    if config["distributed"]:
        logger.info("Cleaning up distributed process group.")
        dist.destroy_process_group()
