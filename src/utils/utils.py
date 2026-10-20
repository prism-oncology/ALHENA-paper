import logging
# Get the logger for this module
logger = logging.getLogger(__name__)

import numpy as np
import hashlib
import os
import random
import torch
import torch.nn.functional as F
from scipy.sparse import csr_matrix
import torch.distributed as dist
import torch.nn as nn
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present


try:
    import idr_torch                        # provided by the cluster
    _USING_JEANZAY = True
    logger.info("Running on Jeanzay (idr_torch is available)")
except ImportError:                         # running elsewhere
    _USING_JEANZAY = False
    logger.info("Not running on Jeanzay (idr_torch is not available)")


def is_using_jeanzay():
    """Returns True if running on Jeanzay (idr_torch is available)"""
    return _USING_JEANZAY


def const_distributed():
    """Get constants for distributed training, with support for Jeanzay"""
    if not dist.is_available() or not torch.cuda.is_available():
        return 0, 1, 0
    
    if _USING_JEANZAY:
        local_rank = idr_torch.local_rank
        world_size = idr_torch.size
        rank = idr_torch.rank
    else:
        local_rank = int(os.environ["SLURM_LOCALID"])
        world_size = int(os.environ["SLURM_NTASKS"])
        rank = int(os.environ["SLURM_PROCID"])
    
    return local_rank, world_size, rank


def get_dist_rank():
    """Gets the process rank in a distributed setup, with support for Jeanzay"""
    if not dist.is_available() or not dist.is_initialized():
        return 0
        
    if _USING_JEANZAY:
        return idr_torch.rank
    else:
        return dist.get_rank()


def get_dist_local_rank():
    """Gets the local rank for the current node, with support for Jeanzay"""
    if _USING_JEANZAY:
        return idr_torch.local_rank
    else:
        return int(os.environ.get("LOCAL_RANK", 0))


def get_dist_world_size():
    """Gets the world size in a distributed setup, with support for Jeanzay"""
    if not dist.is_available() or not dist.is_initialized():
        return 1
        
    if _USING_JEANZAY:
        return idr_torch.size
    else:
        return dist.get_world_size()



def set_global_seed(seed: int):
    """
    Sets the seed for Python random, NumPy, and PyTorch to ensure reproducibility.
    
    Args:
        seed (int): The seed value to be used.
    """
    # Python's built-in random module
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch for CPU and GPU
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    
    # Ensure reproducibility for hash-based operations
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Configure cuBLAS to prevent nondeterministic behavior
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    
    # Configure cuDNN for deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    logger.info(f"Global seed set to: {seed}\n")


def stable_seed(*parts: object) -> int:
    """Deterministic 32-bit seed derived from parts via SHA-256 (stable across processes)."""
    payload = "\x00".join(str(p) for p in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def sparse_std(a: csr_matrix, axis=None) -> np.matrix:
    a_squared = a.multiply(a)
    return np.sqrt(a_squared.mean(axis) - np.square(a.mean(axis)))


def is_main_process():
    """
    Returns True if this is the main (rank 0) process,
    or if no distributed environment is initialized at all.
    """
    if not dist.is_available():
        return True
    if not dist.is_initialized():
        return True
    return get_dist_rank() == 0



def reconcile_state_dict(state_dict: dict, model: nn.Module) -> dict:
    """
    Make `state_dict` compatible with `model` regardless of DDP wrappers.
    Returns a new dict (keeps the original untouched).
    """
    is_ckpt_ddp  = any(k.startswith("module.") for k in state_dict)
    is_model_ddp = isinstance(model, nn.parallel.DistributedDataParallel)

    # --- strip ---
    if is_ckpt_ddp and not is_model_ddp:
        state_dict = state_dict.copy()
        consume_prefix_in_state_dict_if_present(state_dict, "module.")

    # --- add ---
    elif not is_ckpt_ddp and is_model_ddp:
        state_dict = {f"module.{k}": v for k, v in state_dict.items()}

    return state_dict


def loading_checkpoint_cpu(checkpoint_path: str, weights_only: bool):
    """
    Loads a checkpoint from the specified path.
    If using distributed training, it ensures that the checkpoint is loaded
    correctly across all processes.
    """
    distributed = dist.is_available() and dist.is_initialized()
    rank = get_dist_rank() if distributed else 0

    if rank == 0:
        checkpoint_obj = torch.load(checkpoint_path,
                                    map_location="cpu",
                                    weights_only=weights_only)
    else:
        checkpoint_obj = None

    if distributed:
        obj_list = [checkpoint_obj]
        dist.broadcast_object_list(obj_list, src=0)
        checkpoint_obj = obj_list[0]

    return checkpoint_obj



def safe_normalize(x: torch.Tensor, dim: int, eps: float = 1e-6) -> torch.Tensor:
    """
    Safe L2 normalisation of a tensor.
    1. Replace any NaN / ±Inf by 0  
    2. Do a standard L2 normalisation
    """
    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return F.normalize(x, dim=dim, eps=eps)


def assert_finite(*tensors, where=""):
    for t in tensors:
        if not torch.isfinite(t).all():
            raise RuntimeError(f"NaN/Inf detected in {where}")