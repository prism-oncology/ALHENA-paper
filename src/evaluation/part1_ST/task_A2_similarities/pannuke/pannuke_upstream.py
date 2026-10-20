#!/usr/bin/env python
"""Import the membership, decoding, H&E extraction and main-row selection functions of the numbered scripts."""

import importlib.util
import logging
import sys
import types
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# Conda env of _4_prepare_patches.py (named in the error message)
REQUIRED_CONDA_ENV = "process_slide"

# Modules needed by `_1_download_eval_data.py` at import time
_HEAVY_DEPENDENCIES = ("geopandas", "shapely", "spatialdata", "sopa")

# Conda envs of _5_compute_HEembs.py (local and cluster names)
EMBEDDING_CONDA_ENV_LOCAL = "histost"
EMBEDDING_CONDA_ENV_CLUSTER = "histost_env"

# Modules needed by `_8_compute_HEembs.py` at import time
_EMBEDDING_DEPENDENCIES = ("albumentations", "cv2", "torch", "anndata")

# Expected ViT geometry of the membership group
EXPECTED_FINAL_HW = 224
EXPECTED_TOKENS_PER_DIM = 16
EXPECTED_TOKENS_PER_PATCH = EXPECTED_TOKENS_PER_DIM * EXPECTED_TOKENS_PER_DIM

# Symbols taken from `_1_download_eval_data.py`
CONTRACT_SYMBOLS: Tuple[str, ...] = (
    "VIT_TOKEN_PX",
    "_membership_groups_from_backbone_transform",
    "_EMPTY_MEMBERSHIP_SCHEMA",
    "_EMPTY_MISSING_MEMBERSHIP_SCHEMA",
    "apply_membership_preprocess_to_mask",
    "_token_ids_flat_lut",
    "write_hest_h5",
)

# Decoding helpers taken from `_2_characterize_pannuke.py`
DECODER_SYMBOLS: Tuple[str, ...] = (
    "read_hf_feature_names",
    "iter_tiles",
    "read_specific_tiles",
    "_image_bytes",
    "_open_image",
    "select_overlay_tiles",
)

# Symbols taken from `_8_compute_HEembs.py` (used by _5_compute_HEembs.py)
EXTRACTION_SYMBOLS: Tuple[str, ...] = (
    "extract_embeddings_for_slide",
    "membership_parquet_path",
    "membership_group_for_encoder",
    "load_he_encoder_bundle",
)

# Symbols taken from `_9_main_HEemb_cellID.py` (used by _5_compute_HEembs.py)
SELECTION_SYMBOLS: Tuple[str, ...] = (
    "process_slide",
    "select_main_rows",
    "load_centroids",
    "load_patch_centers_from_eval_h5",
)


def eval_data_builder_path() -> Path:
    """Path to `_1_download_eval_data.py`, the owner of the membership contract."""
    return Path(__file__).resolve().parents[2] / "_1_build_dataset" / "_1_download_eval_data.py"


def characterizer_path() -> Path:
    """Path to `_2_characterize_pannuke.py`, the owner of the PanNuke decoders."""
    return Path(__file__).resolve().parent / "_2_characterize_pannuke.py"


def he_embeddings_path() -> Path:
    """Path to `_8_compute_HEembs.py`, the owner of the H&E patch/cell extraction."""
    return Path(__file__).resolve().parents[2] / "_1_build_dataset" / "_8_compute_HEembs.py"


def main_row_selector_path() -> Path:
    """Path to `_9_main_HEemb_cellID.py`, the owner of the one-row-per-cell selection."""
    return Path(__file__).resolve().parents[2] / "_1_build_dataset" / "_9_main_HEemb_cellID.py"


def _step2_env_message(missing: str, path: Path) -> str:
    """Error message for a wrong conda env when loading the modules of _4_prepare_patches.py."""
    return (
        f"_4_prepare_patches.py must run in the `{REQUIRED_CONDA_ENV}` conda env: importing "
        f"{path.name} needs '{missing}', and {REQUIRED_CONDA_ENV} is the env with "
        f"{list(_HEAVY_DEPENDENCIES)}.\n"
        f"    conda activate {REQUIRED_CONDA_ENV}"
    )


def _step3_env_message(missing: str, path: Path) -> str:
    """Error message for a wrong conda env when loading the modules of _5_compute_HEembs.py."""
    return (
        f"_5_compute_HEembs.py must run in the `{EMBEDDING_CONDA_ENV_LOCAL}` conda env "
        f"(`{EMBEDDING_CONDA_ENV_CLUSTER}` on the cluster): importing {path.name} needs '{missing}', "
        f"and the `{REQUIRED_CONDA_ENV}` env of _4_prepare_patches.py does not have "
        f"{list(_EMBEDDING_DEPENDENCIES)}.\n"
        f"    conda activate {EMBEDDING_CONDA_ENV_LOCAL}\n"
        "No single env has both the dependencies of _4_prepare_patches.py "
        f"({list(_HEAVY_DEPENDENCIES)}) and those of _5_compute_HEembs.py."
    )


def _load_module_by_path(
    path: Path,
    module_name: str,
    *,
    heavy_dependencies: Tuple[str, ...] = _HEAVY_DEPENDENCIES,
    env_message: Optional[Callable[[str, Path], str]] = None,
) -> types.ModuleType:
    """Execute a numbered script as a module (registered in sys.modules); a missing dependency raises an env error."""
    if module_name in sys.modules:
        return sys.modules[module_name]
    if not path.exists():
        raise FileNotFoundError(
            f"Cannot load '{module_name}': {path} does not exist."
        )
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not build an import spec for {path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except ModuleNotFoundError as exc:
        del sys.modules[module_name]
        missing = str(getattr(exc, "name", "") or "")
        if missing in heavy_dependencies:
            builder = env_message or _step2_env_message
            raise ImportError(builder(missing, path)) from exc
        del_msg = f"Failed to import {path.name}: missing module '{missing}'."
        raise ImportError(del_msg) from exc
    except Exception:
        del sys.modules[module_name]
        raise
    return module


@lru_cache(maxsize=1)
def eval_data_builder() -> types.ModuleType:
    """`_1_download_eval_data.py` (STHELAR/HEST eval-data builder), loaded for its membership contract."""
    return _load_module_by_path(eval_data_builder_path(), "_1_download_eval_data")


@lru_cache(maxsize=1)
def characterizer() -> types.ModuleType:
    """`_2_characterize_pannuke.py`, loaded for its HF Arrow decoders."""
    return _load_module_by_path(characterizer_path(), "_2_characterize_pannuke")


@lru_cache(maxsize=1)
def he_embedder() -> types.ModuleType:
    """`_8_compute_HEembs.py`, loaded for extract_embeddings_for_slide (importing it sets BLAS/OMP threads to 1)."""
    return _load_module_by_path(
        he_embeddings_path(),
        "_8_compute_HEembs",
        heavy_dependencies=_EMBEDDING_DEPENDENCIES,
        env_message=_step3_env_message,
    )


@lru_cache(maxsize=1)
def main_row_selector() -> types.ModuleType:
    """`_9_main_HEemb_cellID.py`, loaded for process_slide."""
    return _load_module_by_path(
        main_row_selector_path(),
        "_9_main_HEemb_cellID",
        heavy_dependencies=_EMBEDDING_DEPENDENCIES,
        env_message=_step3_env_message,
    )


def _require(module: types.ModuleType, names: Tuple[str, ...], origin: str) -> Dict[str, Any]:
    """Named symbols of a loaded module (raises if one is missing)."""
    out: Dict[str, Any] = {}
    missing: List[str] = []
    for name in names:
        if not hasattr(module, name):
            missing.append(name)
        else:
            out[name] = getattr(module, name)
    if missing:
        raise AttributeError(
            f"{origin} does not define {missing}."
        )
    return out


@dataclass(frozen=True)
class MembershipContract:
    """The cell-token-membership functions and schemas of `_1_download_eval_data.py`."""

    source_path: Path
    vit_token_px: int
    membership_schema: Any                      # pa.Schema for cell_token_membership_*
    missing_membership_schema: Any              # pa.Schema for cell_token_membership_missing_cells_*
    apply_membership_preprocess_to_mask: Callable[..., Any]
    token_ids_flat_lut: Callable[..., Any]
    write_hest_h5: Callable[..., Any]
    membership_groups: Tuple[Dict[str, Any], ...]


@lru_cache(maxsize=1)
def contract() -> MembershipContract:
    """Load and return the membership contract (cached)."""
    module = eval_data_builder()
    sym = _require(module, CONTRACT_SYMBOLS, origin=str(eval_data_builder_path()))
    groups = tuple(dict(g) for g in sym["_membership_groups_from_backbone_transform"]())
    return MembershipContract(
        source_path=eval_data_builder_path(),
        vit_token_px=int(sym["VIT_TOKEN_PX"]),
        membership_schema=sym["_EMPTY_MEMBERSHIP_SCHEMA"],
        missing_membership_schema=sym["_EMPTY_MISSING_MEMBERSHIP_SCHEMA"],
        apply_membership_preprocess_to_mask=sym["apply_membership_preprocess_to_mask"],
        token_ids_flat_lut=sym["_token_ids_flat_lut"],
        write_hest_h5=sym["write_hest_h5"],
        membership_groups=groups,
    )


@lru_cache(maxsize=1)
def decoders() -> Dict[str, Callable[..., Any]]:
    """The HF Arrow decoders of `_2_characterize_pannuke.py`."""
    module = characterizer()
    return _require(module, DECODER_SYMBOLS, origin=str(characterizer_path()))


@lru_cache(maxsize=1)
def extraction() -> Dict[str, Any]:
    """The H&E extraction functions of `_8_compute_HEembs.py`."""
    return _require(he_embedder(), EXTRACTION_SYMBOLS, origin=str(he_embeddings_path()))


@lru_cache(maxsize=1)
def selection() -> Dict[str, Any]:
    """The main-row selection functions of `_9_main_HEemb_cellID.py`."""
    return _require(main_row_selector(), SELECTION_SYMBOLS, origin=str(main_row_selector_path()))


def resolve_membership_group() -> Dict[str, Any]:
    """
    The single membership group derived from BACKBONE_TRANSFORM, with its token grid.
    Raises unless all backbones share one group with a 224 px input and a 16x16 token grid.
    """
    groups = contract().membership_groups
    if len(groups) != 1:
        names = [g.get("name") for g in groups]
        raise RuntimeError(
            f"Expected exactly one membership group across the backbones, got {len(groups)}: "
            f"{names}."
        )
    group = dict(groups[0])
    rs, cs = int(group["resize_size"]), int(group["crop_size"])
    if (rs, cs) != (EXPECTED_FINAL_HW, EXPECTED_FINAL_HW):
        raise RuntimeError(
            f"Membership group '{group['name']}' resolves to (resize, crop) = ({rs}, {cs}), not "
            f"({EXPECTED_FINAL_HW}, {EXPECTED_FINAL_HW})."
        )
    token_px = contract().vit_token_px
    if EXPECTED_FINAL_HW % token_px != 0:
        raise RuntimeError(
            f"Final size {EXPECTED_FINAL_HW} is not divisible by the ViT token size {token_px}."
        )
    tokens_per_dim = EXPECTED_FINAL_HW // token_px
    if tokens_per_dim != EXPECTED_TOKENS_PER_DIM:
        raise RuntimeError(
            f"Token grid resolves to {tokens_per_dim}x{tokens_per_dim}, expected "
            f"{EXPECTED_TOKENS_PER_DIM}x{EXPECTED_TOKENS_PER_DIM}."
        )
    group["tokens_per_dim"] = tokens_per_dim
    group["tokens_per_patch"] = tokens_per_dim * tokens_per_dim
    group["vit_token_px"] = token_px
    return group


def membership_dirname(group_name: str) -> str:
    """Folder holding the per-pseudo-slide membership parquet, as `_8_compute_HEembs.py` reads it."""
    return f"cell_token_membership_{group_name}"


def missing_membership_dirname(group_name: str) -> str:
    """Folder holding the cells with no surviving token support after the geometry transform."""
    return f"cell_token_membership_missing_cells_{group_name}"
