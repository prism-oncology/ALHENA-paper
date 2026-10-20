"""
Constants for the project.
"""

import os
from pathlib import Path
from typing import Union


# Root paths to store data and results
ROOT_PATH: Path = Path("/path/to/ALHENA_data")
ROOT_PATH_JeanZay: Path = Path("/lustre/fswork/projects/rech/<group>/<login>/ST_HE/HistoST")
ROOT_PATH_JeanZay_scratch: Path = Path("/lustre/fsn1/projects/rech/<group>/<login>/ST_HE/HistoST")


# Format version of teacher_center_means.h5 (written by src/dataset/precompute_teacher_mean.py)
TEACHER_CENTER_MEANS_SCHEMA_VERSION: int = 1






# HELPERS --------------------------------------------------------------------------------------------


def on_jeanzay() -> bool:
    
    if os.getenv("SLURM_CLUSTER_NAME", "").lower() == "jean-zay":  # SLURM job on Jean Zay
        return True
    
    if os.getenv("SLURM_JOB_ID"):  # often set on Jean Zay compute nodes
        return Path("/lustre").exists()  # extra safety: Jean Zay filesystem is visible

    return Path("/lustre/fswork/projects/rech").exists()  # fallback: filesystem-only check (works in login nodes too)



def require_root_path(scratch: bool = False) -> Path:
    """Return root path depending on environment and scratch preference."""
    if on_jeanzay():
        return ROOT_PATH_JeanZay_scratch if scratch else ROOT_PATH_JeanZay
    return ROOT_PATH



def get_root_paths_dict() -> dict[str, Path]:
    """Return both root paths with explicit keys."""
    return {
        "local": ROOT_PATH,
        "jeanzay": ROOT_PATH_JeanZay,
        "jeanzay_scratch": ROOT_PATH_JeanZay_scratch,
    }



def get_full_path(
    *parts: Union[str, Path],
    as_str: bool = False,
    scratch: bool = False,
) -> Union[Path, str]:
    """
    Build a path under ROOT_PATH from any number of parts.

    Examples:
      get_full_path("data_1", "adata.h5ad")                 -> Path
      get_full_path("data_1", "adata.h5ad", as_str=True)    -> str
      get_full_path("data_1", "adata.h5ad", scratch=True)   -> Path (Jean Zay scratch only)
    """
    if not parts:
        raise ValueError("get_full_path() needs at least one path part")

    root = require_root_path(scratch=scratch)

    rel = Path()
    for part in parts:
        p = Path(part).expanduser()
        if p.is_absolute():
            raise ValueError(f"Expected relative parts under ROOT_PATH, got absolute: {p}")
        rel /= p

    full = root / rel
    return str(full) if as_str else full
