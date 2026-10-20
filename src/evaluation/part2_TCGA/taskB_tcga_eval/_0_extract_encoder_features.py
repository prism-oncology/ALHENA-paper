#!/usr/bin/env python3
"""
Extract the H&E encoder tile features and the case-level pooled features (mean/std/max) of the TCGA Task B benchmark
cohort, for every available embedding space (he, and clip if the encoder exports image_proj) in one backbone pass.

Saved in <ROOT_HistoST> / eval / eval_datasets / TCGA_eval_data / benchmarks / <benchmark_id> / shared / encoder_features / HEencoder_t<training_num>e<checkpoint_num>[_he_l2tiles] /
    - features/<project_id>.h5: mean/std/max_emb_<space> datasets
    - features/case_features_index.parquet, features/run_manifest.json
    - diagnostics/preprocessing_audit.json, feature_extraction_summary.json, failed_tiles.csv
    - run_manifest.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple
import shutil
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import atexit

# Add the project root and tcga_benchmark/ (for _taskb_common) to sys.path
project_root = Path(__file__).resolve().parents[4]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

_taskb_common_dir = Path(__file__).resolve().parent.parent / "tcga_benchmark"
if str(_taskb_common_dir) not in sys.path:
    sys.path.insert(0, str(_taskb_common_dir))

from src.utils.constants import get_full_path

from _taskb_common import (
    collect_output_hashes,
    manifest_header,
    read_json,
    read_yaml,
    sha256_obj,
    sha256_path,
    shared_tile_h5_path,
    shared_wsi_path,
    tissue_source_site,
    verify_blob_sha256,
    write_csv_atomic,
    write_json_atomic,
    write_parquet_atomic,
    write_text_atomic,
)
from _eval_common import (
    benchmark_encoder_features_root,
    encoder_features_run_id as _encoder_features_run_id_helper,
)

try:  # required at runtime
    import torch
    from torch.utils.data import DataLoader, Dataset
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    DataLoader = object  # type: ignore
    Dataset = object  # type: ignore

try:
    import h5py
except Exception:  # pragma: no cover
    h5py = None  # type: ignore

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore

try:
    import openslide  # type: ignore
except Exception:  # pragma: no cover
    openslide = None  # type: ignore

FAILED_TILE_COLUMNS = [
    "tile_id", "case_id", "case_uuid", "project_id", "slide_file_uuid", "top_left_x_l0", "top_left_y_l0",
    "read_window_px_l0_x", "read_window_px_l0_y", "failure_stage", "failure_reason", "benchmark_id", "modelgroup_id",
]

# Case-features index: one row per (project, case); row_index indexes the pooled datasets of every space
FEATURE_INDEX_COLUMNS = [
    "case_id", "case_uuid", "project_id", "feature_file", "row_index", "n_tiles_used_after_cap", "n_tiles_encoded",
    "n_slides_retained", "benchmark_id", "modelgroup_id", "encoder_id", "spaces",
]

# Pooling statistics saved in each H5 (combined downstream according to the eval_config pooling)
POOLING_VARIANTS: Tuple[str, ...] = ("mean", "std", "max")

# Embedding spaces: he (raw backbone output) and clip (CLIP image projection, L2-normalized per tile;
# only if the bundle exports image_proj)
SPACE_ORDER: Tuple[str, ...] = ("he", "clip")
TILE_POSTPROC_NONE = "none"
TILE_POSTPROC_L2 = "l2_normalize_tile_embeddings"

_SLIDE_CACHE: "OrderedDict[str, Any]" = None  # type: ignore[assignment]
_SLIDE_CACHE_MAX = 16

# Per-worker tile-RGB-cache HDF5 handles: (h5py.File, tile_id -> row dict, tile_asset_id)
_TILE_CACHE_HANDLES: "OrderedDict[str, Any]" = None  # type: ignore[assignment]
_TILE_CACHE_HANDLES_MAX = 8

# Supported `format` attr labels of the tile RGB cache (keep in sync with _15_build_tile_rgb_cache.py)
CACHE_FORMAT_LABELS_SUPPORTED = frozenset({
    "png_rgb_vlen_uint8_v1",       # lossless PNG
    "jpeg444_rgb_vlen_uint8_v1",   # lossy JPEG, 4:4:4 chroma
    "webp_rgb_vlen_uint8_v1",      # lossy WebP
})


def _load_mapping(path: Path) -> Dict[str, Any]:
    obj = read_json(path) if path.suffix.lower() == ".json" else read_yaml(path)
    if not isinstance(obj, Mapping):
        raise RuntimeError(f"Expected mapping at {path}")
    return dict(obj)


def _truthy(x: Any) -> bool:
    if isinstance(x, bool):
        return x
    if x is None:
        return False
    try:
        if pd.isna(x):
            return False
    except Exception:
        pass
    return str(x).strip().lower() in {"true", "t", "1", "yes", "y"}


def _nonempty(x: Any) -> bool:
    if x is None:
        return False
    try:
        if pd.isna(x):
            return False
    except Exception:
        pass
    s = str(x).strip()
    return bool(s and s.lower() not in {"nan", "none", "null", "na", "n/a", "<na>"})


def _as_int(x: Any, name: str) -> int:
    try:
        return int(float(str(x)))
    except Exception as e:
        raise RuntimeError(f"Invalid integer for {name}: {x!r}") from e


def _resolve_bundle_path(training_num: int, checkpoint_num: int) -> Path:
    # Exported bundle of (training_num, checkpoint_num), also for the foundation baseline (checkpoint -1)
    training_id = f"training_{int(training_num)}"
    return get_full_path(
        "train", "trainings", "trainings_HistoST", training_id, "trained_encoders",
        f"HEencoder_t{int(training_num)}e{int(checkpoint_num)}.pt",
    )


def _torch_load_bundle(path: Path, device: str = "cpu") -> Dict[str, Any]:
    if torch is None:
        raise RuntimeError("PyTorch is required for feature extraction")
    if not path.exists():
        raise FileNotFoundError(path)
    bundle = torch.load(str(path), map_location=torch.device(device), weights_only=False)
    if not isinstance(bundle, dict):
        raise RuntimeError(f"Encoder bundle is not a dict: {type(bundle).__name__}")
    if bundle.get("export_type") != "state_dict_bundle":
        raise RuntimeError(f"Encoder bundle export_type must be state_dict_bundle, got {bundle.get('export_type')!r}")
    for key in ["config", "state_dict"]:
        if key not in bundle:
            raise RuntimeError(f"Encoder bundle missing key: {key}")
    return bundle


def _nested_get(obj: Any, paths: Sequence[Sequence[str]]) -> Any:
    for path in paths:
        cur = obj
        ok = True
        for key in path:
            if isinstance(cur, Mapping) and key in cur:
                cur = cur[key]
            else:
                ok = False
                break
        if ok:
            return cur
    return None


def _bool_or_unknown(x: Any) -> Optional[bool]:
    if isinstance(x, bool):
        return x
    if x is None:
        return None
    if isinstance(x, (int, float)) and not isinstance(x, bool):
        if math.isfinite(float(x)):
            return bool(x)
    if isinstance(x, str):
        s = x.strip().lower()
        if s in {"true", "t", "1", "yes", "y", "lora"}:
            return True
        if s in {"false", "f", "0", "no", "n", "none", "frozen", "linear_probe"}:
            return False
    return None


def _state_keys(bundle: Mapping[str, Any]) -> List[str]:
    sd = bundle.get("state_dict")
    if not isinstance(sd, Mapping):
        raise RuntimeError("bundle.state_dict is not a mapping")
    return [str(k) for k in sd.keys()]


def _has_state_tensor(keys: Sequence[str], suffix: str) -> bool:
    return any(k == suffix or k.endswith("." + suffix) for k in keys)


def _has_state_module(keys: Sequence[str], module_names: Sequence[str]) -> bool:
    names = tuple(str(n) for n in module_names)
    for k in map(str, keys):
        for name in names:
            if k == name or k.startswith(name + ".") or f".{name}." in k:
                return True
    return False


def _has_module_attr(model: Any, module_names: Sequence[str]) -> bool:
    return any(getattr(model, str(name), None) is not None for name in module_names)


def _specific_he_mlp_from_state(keys: Sequence[str]) -> Optional[bool]:
    # Avoid treating ViT transformer block MLPs as the H&E projection MLP head.
    patterns = [
        r"(^|\.)image_encoder\.mlp_head\.",
        r"(^|\.)image_encoder\.he_mlp\.",
        r"(^|\.)image_encoder\.head_mlp\.",
        r"(^|\.)image_encoder\.projection_mlp\.",
    ]
    for k in keys:
        if any(re.search(p, k) for p in patterns):
            return True
    return None


def _state_has_lora(keys: Sequence[str]) -> bool:
    # LoRA weights are named adapter_{Q,V}_{down,up}sample (src/models/lora.py); "lora" substring kept as fallback
    adapter_re = r"(^|\.)adapter_[QV]_(down|up)sample(\.|$)"
    return any(("lora" in k.lower()) or re.search(adapter_re, k) for k in keys)


def _module_has_lora(model: Any) -> bool:
    for name, mod in model.named_modules():
        text = f"{name} {mod.__class__.__name__}".lower()
        if "lora" in text:
            return True
    return False


def _module_has_he_mlp(model: Any) -> Optional[bool]:
    img = getattr(model, "image_encoder", None)
    if img is None:
        return None
    if "HEEncoderWithMLP".lower() in img.__class__.__name__.lower():
        return True
    for attr in ["mlp_head", "he_mlp", "head_mlp", "projection_mlp"]:
        if hasattr(img, attr):
            return True
    # MLP head as a submodule of the image_encoder
    for name, mod in img.named_modules():
        if name in {"mlp_head", "he_mlp", "head_mlp", "projection_mlp"}:
            return True
    return False


def _consensus_flag(name: str, evidence: Mapping[str, Optional[bool]]) -> bool:
    known = {k: v for k, v in evidence.items() if v is not None}
    if not known:
        raise RuntimeError(f"Cannot infer {name}; evidence={evidence}")
    vals = set(known.values())
    if len(vals) != 1:
        raise RuntimeError(f"Conflicting {name} evidence: {known}")
    return bool(next(iter(vals)))


def _extract_backbone(config: Any, model: Any) -> str:
    for attr in ["backbone", "model_name"]:
        v = getattr(model, attr, None)
        if _nonempty(v):
            return str(v)
    val = _nested_get(config, [
        ["model", "he", "backbone_img_encoder"], ["model", "backbone_img_encoder"], ["he", "backbone_img_encoder"],
        ["model", "he", "backbone"], ["model", "backbone"], ["backbone"],
    ])
    return str(val) if _nonempty(val) else "unknown"


def _tensor_or_attr_array(model: Any, name: str) -> List[float]:
    v = getattr(model, name, None)
    if v is None:
        raise RuntimeError(f"Loaded encoder missing preprocessing attribute/buffer {name}")
    if torch is not None and isinstance(v, torch.Tensor):
        arr = v.detach().cpu().numpy().astype(float).reshape(-1)
    else:
        arr = np.asarray(v, dtype=float).reshape(-1)
    return [float(x) for x in arr.tolist()]


def _scalar_attr(model: Any, name: str, default: Any = None) -> Any:
    v = getattr(model, name, default)
    if torch is not None and isinstance(v, torch.Tensor):
        if v.numel() == 1:
            return v.detach().cpu().item()
        return v.detach().cpu().numpy().tolist()
    return v


def _preprocessing_metadata(model: Any, config: Any) -> Dict[str, Any]:
    backbone = _extract_backbone(config, model)
    resize = int(_scalar_attr(model, "resize_size", 0) or 0)
    crop = int(_scalar_attr(model, "crop_size", 0) or 0)
    interpolation = int(_scalar_attr(model, "interpolation_code", 1) or 1)
    mean = _tensor_or_attr_array(model, "mean")
    std = _tensor_or_attr_array(model, "std")
    if len(mean) != 3 or len(std) != 3:
        raise RuntimeError(f"Expected 3-channel mean/std, got mean={mean}, std={std}")
    if any((not math.isfinite(x)) for x in mean + std) or any(x <= 0 for x in std):
        raise RuntimeError(f"Invalid preprocessing mean/std: mean={mean}, std={std}")
    return {
        "backbone": backbone,
        "resize_size": resize,
        "crop_size": crop,
        "mean": mean,
        "std": std,
        "interpolation_code": interpolation,
        "expects_preprocessed": True,
        "preprocessing_hash": sha256_obj({
            "backbone": backbone, "resize_size": resize, "crop_size": crop, "mean": mean, "std": std,
            "interpolation_code": interpolation, "expects_preprocessed": True,
        }),
    }


def _config_flag_evidence(config: Any) -> Dict[str, Optional[bool]]:
    he_mlp = _bool_or_unknown(_nested_get(config, [["model", "he", "use_mlp_head"], ["model", "use_mlp_head"], ["he", "use_mlp_head"], ["use_mlp_head"]]))
    train_option = _nested_get(config, [["training", "HE_training_option"], ["training", "he_training_option"], ["HE_training_option"], ["he_training_option"]])
    lora: Optional[bool]
    if train_option is None:
        lora = _bool_or_unknown(_nested_get(config, [["training", "use_lora"], ["model", "he", "use_lora"], ["use_lora"]]))
    else:
        lora = str(train_option).strip().lower() == "lora"
    return {"contains_he_mlp_head": he_mlp, "contains_lora": lora}


def _infer_encoder_flags(bundle: Mapping[str, Any], model: Any, checkpoint_num: int) -> Dict[str, Any]:
    keys = _state_keys(bundle)
    projection_flags = bundle.get("projection_flags") if isinstance(bundle.get("projection_flags"), Mapping) else {}
    cfg_flags = _config_flag_evidence(bundle.get("config"))
    bundle_foundation = _bool_or_unknown(bundle.get("foundation_baseline_only"))
    ckpt_epoch = bundle.get("checkpoint_epoch")
    ckpt_foundation: Optional[bool] = None
    if ckpt_epoch is not None:
        try:
            ckpt_foundation = int(ckpt_epoch) < 0
        except Exception:
            ckpt_foundation = None
    filename_foundation = bool(checkpoint_num < 0)

    evidence: Dict[str, Dict[str, Optional[bool]]] = {
        "bundle_has_image_proj": {
            "bundle_projection_flags": _bool_or_unknown(projection_flags.get("has_image_proj")),
            "state_dict": _has_state_module(keys, ["image_proj", "img_proj", "image_projection"]),
            "module": _has_module_attr(model, ["image_proj", "img_proj", "image_projection"]),
        },
        "bundle_has_img_tok_proj": {
            "bundle_projection_flags": _bool_or_unknown(projection_flags.get("has_img_tok_proj")),
            "state_dict": _has_state_module(keys, ["img_tok_proj", "image_token_proj", "img_token_proj"]),
            "module": _has_module_attr(model, ["img_tok_proj", "image_token_proj", "img_token_proj"]),
        },
        "contains_lora": {
            "config": cfg_flags["contains_lora"],
            "state_dict": _state_has_lora(keys),
            "module": _module_has_lora(model),
        },
        "contains_he_mlp_head": {
            "config": cfg_flags["contains_he_mlp_head"],
            "state_dict": _specific_he_mlp_from_state(keys),
            "module": _module_has_he_mlp(model),
        },
        "foundation_baseline_only": {
            "bundle": bundle_foundation,
            "checkpoint_epoch": ckpt_foundation,
            "requested_checkpoint_num": filename_foundation,
        },
    }
    flags = {name: _consensus_flag(name, ev) for name, ev in evidence.items()}
    flags["evidence"] = evidence

    if checkpoint_num < 0:
        required = {
            "foundation_baseline_only": True,
            "contains_he_mlp_head": False,
            "contains_lora": False,
            "bundle_has_image_proj": False,
            "bundle_has_img_tok_proj": False,
        }
        bad = {k: {"observed": flags[k], "expected": v} for k, v in required.items() if flags[k] != v}
        if bad:
            raise RuntimeError(f"Checkpoint -1 must be a strict clean foundation baseline; violations={bad}")
    return flags


def _actual_output_dims(model: Any, crop_size: int, device: Any, *, has_image_proj: bool) -> Dict[str, Optional[int]]:
    size = int(crop_size or _scalar_attr(model, "crop_size", 224) or 224)
    rng_devices = [int(device.index or 0)] if getattr(device, "type", None) == "cuda" else []
    with torch.random.fork_rng(devices=rng_devices, enabled=True):
        torch.manual_seed(12345)
        x = torch.randn((2, 3, size, size), dtype=torch.float32, device=device)
    out: Dict[str, Optional[int]] = {}
    with torch.inference_mode():
        y = model.encode_image(x, project=False)
        if y.ndim != 2:
            raise RuntimeError(f"encode_image(project=False) returned shape {tuple(y.shape)}")
        out["actual_output_dim_project_false"] = int(y.shape[1])
        if has_image_proj:
            yp = model.encode_image(x, project=True)
            if yp.ndim != 2:
                raise RuntimeError(f"encode_image(project=True) returned shape {tuple(yp.shape)}")
            out["actual_output_dim_project_true"] = int(yp.shape[1])
        else:
            out["actual_output_dim_project_true"] = None
    return out


def _flatten_registry_entries(obj: Any) -> List[Mapping[str, Any]]:
    entries: List[Mapping[str, Any]] = []
    if isinstance(obj, Mapping):
        if any(k in obj for k in ["backbone", "backbone_img_encoder", "model_name", "resize_size", "crop_size", "mean", "std"]):
            entries.append(obj)
        for v in obj.values():
            entries.extend(_flatten_registry_entries(v))
    elif isinstance(obj, list):
        for v in obj:
            entries.extend(_flatten_registry_entries(v))
    return entries


def _entry_backbone(entry: Mapping[str, Any]) -> Optional[str]:
    for k in ["backbone", "backbone_img_encoder", "model_name", "encoder_backbone", "foundation_backbone"]:
        if _nonempty(entry.get(k)):
            return str(entry.get(k))
    return None


def _as_float_list(x: Any) -> Optional[List[float]]:
    if x is None:
        return None
    try:
        arr = np.asarray(x, dtype=float).reshape(-1)
        return [float(v) for v in arr.tolist()]
    except Exception:
        return None


def _validate_registry(registry_path: Path, preprocess: Mapping[str, Any], modelgroup: Mapping[str, Any]) -> Dict[str, Any]:
    if not registry_path.exists():
        raise FileNotFoundError(registry_path)
    registry = read_json(registry_path)
    entries = _flatten_registry_entries(registry)
    backbone = str(preprocess.get("backbone"))
    matches = [e for e in entries if _entry_backbone(e) == backbone]
    if not matches:
        raise RuntimeError(f"encoder_registry.json has no entry matching backbone={backbone!r}")
    disagreements: List[str] = []
    for e in matches:
        for key in ["resize_size", "crop_size", "interpolation_code"]:
            if key in e and _nonempty(e.get(key)):
                try:
                    if int(e[key]) != int(preprocess[key]):
                        disagreements.append(f"{key}: registry={e[key]} bundle={preprocess[key]}")
                except Exception:
                    disagreements.append(f"{key}: registry_unparseable={e[key]}")
        for key in ["mean", "std"]:
            if key in e:
                val = _as_float_list(e.get(key))
                # mean/std compared at float32 precision (the encoder buffers are float32)
                if val is not None and not np.allclose(
                    np.asarray(val, dtype=np.float32),
                    np.asarray(preprocess[key], dtype=np.float32),
                    rtol=0, atol=1e-6,
                ):
                    disagreements.append(f"{key}: registry={val} bundle={preprocess[key]}")
    if disagreements:
        raise RuntimeError("Encoder preprocessing metadata disagrees with encoder_registry.json: " + "; ".join(disagreements))

    geom_disagreements: List[str] = []
    if "crop_size" in modelgroup and int(modelgroup["crop_size"]) != int(preprocess["crop_size"]):
        geom_disagreements.append(f"crop_size modelgroup={modelgroup['crop_size']} encoder={preprocess['crop_size']}")
    if "resize_size" in modelgroup and int(modelgroup["resize_size"]) != int(preprocess["resize_size"]):
        geom_disagreements.append(f"resize_size modelgroup={modelgroup['resize_size']} encoder={preprocess['resize_size']}")
    if "tile_size_px_encoder_input" in modelgroup and int(modelgroup["tile_size_px_encoder_input"]) != int(preprocess["crop_size"]):
        geom_disagreements.append(f"tile_size_px_encoder_input modelgroup={modelgroup['tile_size_px_encoder_input']} encoder_crop={preprocess['crop_size']}")
    # Native resolution: registry entries vs model group
    if "nominal_mpp_um_per_px" in modelgroup:
        mg_mpp = float(modelgroup["nominal_mpp_um_per_px"])
        for e in matches:
            if "nominal_mpp_um_per_px" in e and _nonempty(e.get("nominal_mpp_um_per_px")):
                if not np.isclose(float(e["nominal_mpp_um_per_px"]), mg_mpp, rtol=0, atol=1e-8):
                    geom_disagreements.append(
                        f"nominal_mpp_um_per_px registry={e['nominal_mpp_um_per_px']} modelgroup={mg_mpp}"
                    )
    if geom_disagreements:
        raise RuntimeError("Encoder geometry disagrees with modelgroup YAML: " + "; ".join(geom_disagreements))
    return {"matched_entries": len(matches), "registry_hash": sha256_path(registry_path), "matched_backbone": backbone}


def _composite_rgba_on_white(img: Image.Image) -> Image.Image:
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    white = Image.new("RGBA", img.size, (255, 255, 255, 255))
    white.alpha_composite(img)
    return white.convert("RGB")


def _resize_rgb(rgb: np.ndarray, size: int, interpolation_code: int) -> np.ndarray:
    # OpenCV required (interpolation_code is a cv2 constant)
    if size <= 0:
        return rgb
    if rgb.shape[0] == size and rgb.shape[1] == size:
        return rgb
    return cv2.resize(rgb, (size, size), interpolation=int(interpolation_code))


def _center_crop(rgb: np.ndarray, crop_size: int) -> np.ndarray:
    if crop_size <= 0:
        return rgb
    h, w = rgb.shape[:2]
    if h < crop_size or w < crop_size:
        raise RuntimeError(f"Cannot center-crop {crop_size} from image with shape {rgb.shape}")
    y0 = (h - crop_size) // 2
    x0 = (w - crop_size) // 2
    return rgb[y0:y0 + crop_size, x0:x0 + crop_size, :]


def _preprocess_rgb_to_tensor(rgb: np.ndarray, preprocess: Mapping[str, Any]) -> Any:
    rgb = _resize_rgb(rgb, int(preprocess["resize_size"]), int(preprocess["interpolation_code"]))
    rgb = _center_crop(rgb, int(preprocess["crop_size"]))
    arr = rgb.astype(np.float32) / 255.0
    mean = np.asarray(preprocess["mean"], dtype=np.float32).reshape(1, 1, 3)
    std = np.asarray(preprocess["std"], dtype=np.float32).reshape(1, 1, 3)
    arr = (arr - mean) / std
    if not np.isfinite(arr).all():
        raise RuntimeError("Non-finite value after image preprocessing")
    arr = np.transpose(arr, (2, 0, 1)).copy()
    return torch.from_numpy(arr)


def _slide_for_path(path: str) -> Any:
    from collections import OrderedDict
    if openslide is None:
        raise RuntimeError("openslide-python is required for WSI feature extraction")
    global _SLIDE_CACHE
    if _SLIDE_CACHE is None:
        _SLIDE_CACHE = OrderedDict()
    if path in _SLIDE_CACHE:
        _SLIDE_CACHE.move_to_end(path)
        return _SLIDE_CACHE[path]
    slide = openslide.OpenSlide(path)
    _SLIDE_CACHE[path] = slide
    while len(_SLIDE_CACHE) > _SLIDE_CACHE_MAX:
        _, evicted = _SLIDE_CACHE.popitem(last=False)
        try:
            evicted.close()
        except Exception:
            pass
    return slide


def _close_slide_cache() -> None:
    global _SLIDE_CACHE
    if _SLIDE_CACHE is None:
        return
    for s in list(_SLIDE_CACHE.values()):
        try:
            s.close()
        except Exception:
            pass
    _SLIDE_CACHE.clear()


def _tile_cache_handle(path: str, *, expected_set_recipe_id: str, expected_codec_recipe_id: str) -> Tuple[Any, Dict[str, int]]:
    """
    Per-worker LRU of tile-RGB-cache HDF5 handles (opened with swmr=True). Returns (h5_file, tile_id_to_row_idx).
    Checks the set_recipe_id / codec_recipe_id attributes of the file.
    """
    from collections import OrderedDict
    if h5py is None:
        raise RuntimeError("h5py is required for the tile RGB cache reader")
    expected = (str(expected_set_recipe_id), str(expected_codec_recipe_id))
    global _TILE_CACHE_HANDLES
    if _TILE_CACHE_HANDLES is None:
        _TILE_CACHE_HANDLES = OrderedDict()
    if path in _TILE_CACHE_HANDLES:
        _TILE_CACHE_HANDLES.move_to_end(path)
        h5f, idx_map, declared = _TILE_CACHE_HANDLES[path]
        if declared != expected:
            raise RuntimeError(
                f"Tile RGB cache recipe mismatch on {path}: declared={declared!r}, expected={expected!r}"
            )
        return h5f, idx_map
    # Read-only with SWMR (plain mode if not supported)
    try:
        h5f = h5py.File(path, "r", swmr=True)
    except Exception:
        h5f = h5py.File(path, "r")
    declared = (str(h5f.attrs.get("set_recipe_id", "")), str(h5f.attrs.get("codec_recipe_id", "")))
    if declared != expected:
        try:
            h5f.close()
        except Exception:
            pass
        raise RuntimeError(
            f"Tile RGB cache recipe mismatch on {path}: declared={declared!r}, expected={expected!r}"
        )
    fmt = str(h5f.attrs.get("format", ""))
    if fmt and fmt not in CACHE_FORMAT_LABELS_SUPPORTED:
        try:
            h5f.close()
        except Exception:
            pass
        raise RuntimeError(
            f"Tile RGB cache format unsupported on {path}: got {fmt!r}, "
            f"expected one of {sorted(CACHE_FORMAT_LABELS_SUPPORTED)}"
        )
    tile_ids = h5f["tile_ids"][:]
    idx_map = {(t.decode("utf-8") if isinstance(t, (bytes, bytearray)) else str(t)): i for i, t in enumerate(tile_ids)}
    _TILE_CACHE_HANDLES[path] = (h5f, idx_map, expected)
    while len(_TILE_CACHE_HANDLES) > _TILE_CACHE_HANDLES_MAX:
        _, evicted = _TILE_CACHE_HANDLES.popitem(last=False)
        try:
            evicted[0].close()
        except Exception:
            pass
    return h5f, idx_map


def _close_tile_cache() -> None:
    global _TILE_CACHE_HANDLES
    if _TILE_CACHE_HANDLES is None:
        return
    for h5f, _idx, _decl in list(_TILE_CACHE_HANDLES.values()):
        try:
            h5f.close()
        except Exception:
            pass
    _TILE_CACHE_HANDLES.clear()


atexit.register(_close_slide_cache)
atexit.register(_close_tile_cache)


# ---------------------------------------------------------------------------
# Resume support: per-case checkpoint "shards"
# Each fully encoded case is saved as an npz shard (per-space mean / M2 / max accumulators) under
# run_root/.partial/case_shards/, with the run fingerprint; shards of another configuration are ignored.
PARTIAL_DIRNAME = ".partial"
SHARDS_DIRNAME = "case_shards"
RESUME_META_NAME = "resume_meta.json"
RESUME_SCHEMA = "taskb_feature_resume_v1"


def _resume_run_fingerprint(parts: Mapping[str, Any]) -> str:
    """Content hash of everything that determines a case's pooled feature values."""
    return sha256_obj(dict(parts))


def _write_case_shard(shards_dir: Path, st: Mapping[str, Any], *, run_fingerprint: str, spaces: Sequence[str]) -> None:
    """Save the accumulators of one fully encoded case as an npz shard (atomic write)."""
    cid = str(st["case_id"])
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", cid) or "case"
    path = shards_dir / f"{safe}__{sha256_obj(cid)[:12]}.npz"
    arrays: Dict[str, np.ndarray] = {
        "run_fingerprint": np.array(str(run_fingerprint)),
        "case_id": np.array(cid),
        "case_uuid": np.array(str(st.get("case_uuid", ""))),
        "project_id": np.array(str(st.get("project_id", ""))),
        "n": np.array(int(st["n"]), dtype=np.int64),
        "n_tiles_used_after_cap": np.array(int(st.get("n_tiles_used_after_cap", 0)), dtype=np.int64),
        "n_slides_retained": np.array(int(st.get("n_slides_retained", 0)), dtype=np.int64),
        "spaces": np.array([str(s) for s in spaces]),
    }
    for s in spaces:
        a = st["acc"][s]
        arrays[f"mean__{s}"] = np.asarray(a["mean"], dtype=np.float64)
        arrays[f"M2__{s}"] = np.asarray(a["M2"], dtype=np.float64)
        arrays[f"max__{s}"] = np.asarray(a["max_emb"], dtype=np.float64)
    shards_dir.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.unlink(missing_ok=True)
    with open(tmp, "wb") as f:
        np.savez(f, **arrays)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _load_case_shards(
    shards_dir: Path, *, run_fingerprint: str, spaces: Sequence[str], dim_by_space: Mapping[str, int]
) -> Dict[str, Dict[str, Any]]:
    """Load the valid case shards as case_state records (unreadable or mismatching shards are ignored and re-encoded)."""
    out: Dict[str, Dict[str, Any]] = {}
    if not shards_dir.exists():
        return out
    spaces = list(spaces)
    for path in sorted(shards_dir.glob("*.npz")):
        if path.name.startswith("."):  # temp file / macOS ._ file
            continue
        try:
            with np.load(path, allow_pickle=False) as z:
                if str(z["run_fingerprint"].item()) != str(run_fingerprint):
                    continue
                if [str(x) for x in z["spaces"].tolist()] != spaces:
                    continue
                acc: Dict[str, Any] = {}
                ok = True
                for s in spaces:
                    mean = np.asarray(z[f"mean__{s}"], dtype=np.float64)
                    if int(mean.shape[0]) != int(dim_by_space[s]):
                        ok = False
                        break
                    acc[s] = {
                        "mean": mean,
                        "M2": np.asarray(z[f"M2__{s}"], dtype=np.float64),
                        "max_emb": np.asarray(z[f"max__{s}"], dtype=np.float64),
                    }
                if not ok:
                    continue
                cid = str(z["case_id"].item())
                out[cid] = {
                    "case_id": cid,
                    "case_uuid": str(z["case_uuid"].item()),
                    "project_id": str(z["project_id"].item()),
                    "acc": acc,
                    "n": int(z["n"].item()),
                    "n_tiles_used_after_cap": int(z["n_tiles_used_after_cap"].item()),
                    "n_slides_retained": int(z["n_slides_retained"].item()),
                }
        except Exception as e:  # pragma: no cover - a bad shard is recomputed
            print(f"WARN: ignoring unreadable resume shard {path.name}: {e}", file=sys.stderr)
            continue
    return out


class TileDataset(Dataset):  # type: ignore[misc]
    def __init__(
        self,
        rows: pd.DataFrame,
        preprocess: Mapping[str, Any],
        benchmark_id: str,
        modelgroup_id: str,
        *,
        expected_set_recipe_id: str = "",
        expected_codec_recipe_id: str = "",
        cache_paths_by_uuid: Optional[Mapping[str, str]] = None,
        cache_mode: str = "never",
    ) -> None:
        self.rows = rows.reset_index(drop=True)
        self.preprocess = dict(preprocess)
        self.benchmark_id = benchmark_id
        self.modelgroup_id = modelgroup_id
        self.expected_set_recipe_id = str(expected_set_recipe_id)
        self.expected_codec_recipe_id = str(expected_codec_recipe_id)
        self.cache_paths_by_uuid: Dict[str, str] = (
            {str(k): str(v) for k, v in (cache_paths_by_uuid or {}).items()}
        )
        # cache_mode: "auto" (use cache if path present), "always" (must hit),
        # "never" (ignore cache entirely). Validated in `_main_impl`.
        if cache_mode not in {"auto", "always", "never"}:
            raise RuntimeError(f"invalid cache_mode {cache_mode!r}")
        self.cache_mode = cache_mode

    def __len__(self) -> int:
        return int(len(self.rows))

    def _try_cache(self, tile_id: str, slide_file_uuid: str) -> Optional[np.ndarray]:
        """
        Load a tile from the RGB cache. Returns None only if the slide has no cache file;
        raises if the slide has a cache file without this tile (no WSI fallback within a cached slide).
        """
        if self.cache_mode == "never":
            return None
        cache_path = self.cache_paths_by_uuid.get(slide_file_uuid)
        if not cache_path:
            return None
        if cv2 is None:
            raise RuntimeError("opencv-python/cv2 is required to decode cached tiles")
        h5f, idx_map = _tile_cache_handle(
            cache_path,
            expected_set_recipe_id=self.expected_set_recipe_id,
            expected_codec_recipe_id=self.expected_codec_recipe_id,
        )
        row_idx = idx_map.get(tile_id)
        if row_idx is None:
            raise RuntimeError(
                f"Tile {tile_id!r} for slide {slide_file_uuid!r} is absent from its tile RGB cache "
                f"({cache_path}); the cache is incomplete. Rebuild it with _15_build_tile_rgb_cache.py "
                f"or rerun with --use_tile_rgb_cache never."
            )
        buf = np.asarray(h5f["tiles_png"][int(row_idx)], dtype=np.uint8)
        bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"cv2.imdecode failed for cached tile {tile_id!r} ({cache_path})")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        r = self.rows.iloc[int(idx)].to_dict()
        slide_file_uuid = str(r.get("slide_file_uuid", ""))
        tile_id = str(r.get("tile_id", ""))
        base_meta = {
            "tile_id": tile_id,
            "case_id": str(r.get("case_id", "")),
            "case_uuid": str(r.get("case_uuid", "")),
            "project_id": str(r.get("project_id", "")),
            "slide_file_uuid": slide_file_uuid,
            "top_left_x_l0": r.get("top_left_x_l0", ""),
            "top_left_y_l0": r.get("top_left_y_l0", ""),
            "read_window_px_l0_x": r.get("read_window_px_l0_x", ""),
            "read_window_px_l0_y": r.get("read_window_px_l0_y", ""),
            "benchmark_id": self.benchmark_id,
            "modelgroup_id": self.modelgroup_id,
        }
        cache_hit = False
        try:
            rgb = self._try_cache(tile_id, slide_file_uuid)
            if rgb is not None:
                cache_hit = True
            else:
                # _try_cache returns None only when the slide has no cache file at all.
                if self.cache_mode == "always":
                    raise RuntimeError(
                        f"--use_tile_rgb_cache=always but slide {slide_file_uuid!r} has no per-slide "
                        f"cache file (cache_path={self.cache_paths_by_uuid.get(slide_file_uuid)!r}); "
                        f"this should have been caught at preflight."
                    )
                local_path = str(r["local_path"])
                slide = _slide_for_path(local_path)
                x = _as_int(r["top_left_x_l0"], "top_left_x_l0")
                y = _as_int(r["top_left_y_l0"], "top_left_y_l0")
                wx = _as_int(r["read_window_px_l0_x"], "read_window_px_l0_x")
                wy = _as_int(r["read_window_px_l0_y"], "read_window_px_l0_y")
                region = slide.read_region((x, y), 0, (wx, wy))
                rgb_img = _composite_rgba_on_white(region)
                rgb = np.asarray(rgb_img, dtype=np.uint8)
            tensor = _preprocess_rgb_to_tensor(rgb, self.preprocess)
            return {"ok": True, "image": tensor, "meta": base_meta, "error": "", "cache_hit": cache_hit}
        except Exception as e:
            return {"ok": False, "image": None, "meta": base_meta, "error": f"{type(e).__name__}: {e}", "cache_hit": cache_hit}


def _collate_tiles(batch: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    ok_items = [b for b in batch if b.get("ok")]
    fail_items = [b for b in batch if not b.get("ok")]
    images = torch.stack([b["image"] for b in ok_items], dim=0) if ok_items else None
    n_cache_hits = int(sum(1 for b in batch if bool(b.get("cache_hit"))))
    n_cache_misses = int(len(batch) - n_cache_hits)
    return {
        "images": images,
        "metas": [b["meta"] for b in ok_items],
        "failures": [{**b["meta"], "failure_stage": "tile_read_preprocess", "failure_reason": b.get("error", "unknown_error")} for b in fail_items],
        "n_cache_hits": n_cache_hits,
        "n_cache_misses": n_cache_misses,
    }


def _l2_normalize_or_abort(emb: Any, *, context: str) -> Any:
    norm = torch.linalg.vector_norm(emb, dim=1, keepdim=True)
    if torch.any(norm <= 0) or torch.any(~torch.isfinite(norm)):
        raise RuntimeError(f"Invalid {context} embedding norm before L2 normalization")
    return emb / torch.clamp(norm, min=1e-12)


def _encode_batch(model: Any, images: Any, *, spaces: Sequence[str], he_l2_normalize: bool, device: Any) -> Dict[str, np.ndarray]:
    """
    Encode one tile batch into every requested space with a single backbone pass. Returns {space: [B, D] float64 array}.
    he: raw backbone output (per-tile L2 if he_l2_normalize); clip: image_proj of it, per-tile L2.
    """
    x = images.to(device=device, dtype=torch.float32, non_blocking=True)
    if torch.any(~torch.isfinite(x)):
        raise RuntimeError("NaN/Inf in preprocessed input batch")
    emb_by_space: Dict[str, Any] = {}
    with torch.inference_mode():
        he = model.encode_image(x, project=False)  # backbone output, shared by both spaces
        if "he" in spaces:
            emb_by_space["he"] = _l2_normalize_or_abort(he, context="HE-space") if he_l2_normalize else he
        if "clip" in spaces:
            emb_by_space["clip"] = _l2_normalize_or_abort(model.image_proj(he), context="CLIP")
    result: Dict[str, np.ndarray] = {}
    for space, emb in emb_by_space.items():
        if emb.ndim != 2:
            raise RuntimeError(f"Encoder output for space={space!r} must be [B, D], got {tuple(emb.shape)}")
        if torch.any(~torch.isfinite(emb)):
            raise RuntimeError(f"NaN/Inf in encoder output batch for space={space!r}")
        result[space] = emb.detach().cpu().numpy().astype(np.float64, copy=False)
    return result


def _compute_tss_clustering(
    case_state: Mapping[str, Mapping[str, Any]],
    project_id_by_case: Mapping[str, str],
    *,
    space: str,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    TSS (tissue source site) clustering diagnostic of the case-level mean_std embeddings of one space: cosine silhouette,
    global (label project__tss) and per project (label tss). Returns (summary_dict, per_case_dataframe).
    """
    try:
        from sklearn.metrics import silhouette_score  # type: ignore
    except Exception as e:
        return (
            {
                "space": str(space),
                "error": f"sklearn not available: {e}",
                "pooling_used_for_diagnostic": "mean_std (Task B default; matches default pooling)",
            },
            pd.DataFrame(columns=["space", "case_id", "case_uuid", "project_id", "tss", "embedding_norm"]),
        )

    rows: List[Dict[str, Any]] = []
    embeddings_list: List[np.ndarray] = []
    for cid, st in case_state.items():
        n = int(st.get("n") or 0)
        if n <= 0:
            continue
        acc = st["acc"][space]
        mean = np.asarray(acc["mean"], dtype=np.float64)
        var = np.maximum(np.asarray(acc["M2"], dtype=np.float64) / float(n), 0.0)
        std = np.sqrt(var)
        emb = np.concatenate([mean, std])
        rows.append({
            "space": str(space),
            "case_id": str(cid),
            "case_uuid": str(st.get("case_uuid") or ""),
            "project_id": str(project_id_by_case.get(cid, "<UNK>")),
            "tss": tissue_source_site(cid),
            "embedding_norm": float(np.linalg.norm(emb)),
        })
        embeddings_list.append(emb)

    per_case_df = pd.DataFrame(rows)

    base_summary: Dict[str, Any] = {
        "space": str(space),
        "pooling_used_for_diagnostic": "mean_std (Task B default; matches default pooling)",
        "embedding_dim_used": int(embeddings_list[0].shape[0]) if embeddings_list else 0,
        "n_cases": int(len(rows)),
        "silhouette_metric": "cosine",
    }
    if not rows:
        base_summary["error"] = "no cases in case_state"
        return base_summary, per_case_df

    embeddings = np.stack(embeddings_list, axis=0)
    project_ids = [r["project_id"] for r in rows]
    tsss = [r["tss"] for r in rows]

    # Global silhouette: label = "{project}__{tss}"
    global_labels = np.asarray([f"{p}__{t}" for p, t in zip(project_ids, tsss)])
    n_unk_global = int(sum(1 for t in tsss if t == "<UNK>"))
    global_summary: Dict[str, Any] = {
        "n_unique_labels": int(len(set(global_labels.tolist()))),
        "n_unk_tss": n_unk_global,
    }
    if n_unk_global > 0.5 * len(rows):
        global_summary["silhouette_score_cosine"] = None
        global_summary["skipped_reason"] = "n_unk_tss > 0.5 * n_cases (possible non-TCGA contamination)"
    elif global_summary["n_unique_labels"] < 2:
        global_summary["silhouette_score_cosine"] = None
        global_summary["skipped_reason"] = "less_than_two_unique_labels"
    else:
        try:
            global_summary["silhouette_score_cosine"] = float(silhouette_score(embeddings, global_labels, metric="cosine"))
        except Exception as e:
            global_summary["silhouette_score_cosine"] = None
            global_summary["skipped_reason"] = f"silhouette_failed:{type(e).__name__}:{e}"

    # Per-project silhouette: label = tss
    per_project: Dict[str, Any] = {}
    for project_id in sorted(set(project_ids)):
        mask = np.fromiter((p == project_id for p in project_ids), dtype=bool, count=len(project_ids))
        proj_emb = embeddings[mask]
        proj_tss = np.asarray([t for i, t in enumerate(tsss) if mask[i]])
        n_unique = int(len(set(proj_tss.tolist())))
        tss_counts: Dict[str, int] = {}
        for t in proj_tss.tolist():
            tss_counts[t] = tss_counts.get(t, 0) + 1
        rec: Dict[str, Any] = {
            "n_cases": int(mask.sum()),
            "n_tss": n_unique,
            "n_unk_tss": int(sum(1 for t in proj_tss.tolist() if t == "<UNK>")),
            "tss_counts": dict(sorted(tss_counts.items(), key=lambda x: (-x[1], x[0]))),
        }
        if n_unique < 2:
            rec["silhouette_score_cosine"] = None
            rec["skipped_reason"] = "single_tss"
        else:
            try:
                rec["silhouette_score_cosine"] = float(silhouette_score(proj_emb, proj_tss, metric="cosine"))
            except Exception as e:
                rec["silhouette_score_cosine"] = None
                rec["skipped_reason"] = f"silhouette_failed:{type(e).__name__}:{e}"
        per_project[project_id] = rec

    base_summary["global"] = global_summary
    base_summary["per_project"] = per_project
    return base_summary, per_case_df


def _init_case_state(meta: Mapping[str, Any], dim_by_space: Mapping[str, int], frozen: Mapping[str, Any]) -> Dict[str, Any]:
    # Tile count n shared by all spaces; Welford mean/M2/max accumulators per space
    return {
        "case_id": str(meta["case_id"]),
        "case_uuid": str(meta.get("case_uuid", frozen.get("case_uuid", ""))),
        "project_id": str(meta["project_id"]),
        "acc": {
            str(space): {
                "mean": np.zeros(int(dim), dtype=np.float64),
                "M2": np.zeros(int(dim), dtype=np.float64),
                "max_emb": np.full(int(dim), -np.inf, dtype=np.float64),
            }
            for space, dim in dim_by_space.items()
        },
        "n": 0,
        "n_tiles_used_after_cap": int(float(frozen.get("n_tiles_used_after_cap", 0))),
        "n_slides_retained": int(float(frozen.get("n_slides_retained", 0))),
    }


def _write_project_h5(
    path: Path,
    cases: Sequence[Mapping[str, Any]],
    *,
    emb_by_space: Mapping[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    dim_by_space: Mapping[str, int],
    postproc_by_space: Mapping[str, str],
    spaces: Sequence[str],
    project_id: str,
) -> None:
    if h5py is None:
        raise RuntimeError("h5py is required to write feature H5 files")
    if not spaces:
        raise RuntimeError("No embedding spaces to write")
    n_cases = len(cases)
    for space in spaces:
        mean_emb, std_emb, max_emb = emb_by_space[space]
        if not (mean_emb.shape == std_emb.shape == max_emb.shape):
            raise RuntimeError(
                f"[{space}] per-stat arrays must share shape, got mean={mean_emb.shape}, "
                f"std={std_emb.shape}, max={max_emb.shape}"
            )
        if mean_emb.shape[0] != n_cases:
            raise RuntimeError(f"[{space}] expected {n_cases} case rows, got {mean_emb.shape[0]}")
        if mean_emb.shape[1] != int(dim_by_space[space]):
            raise RuntimeError(
                f"[{space}] per-stat arrays must have feature_dim_tile={dim_by_space[space]} columns, "
                f"got {mean_emb.shape[1]}"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.unlink(missing_ok=True)
    str_dt = h5py.string_dtype(encoding="utf-8")
    with h5py.File(tmp, "w") as f:
        # Per-space pooled datasets ({stat}_emb_{space}); no clip without image_proj
        for space in spaces:
            mean_emb, std_emb, max_emb = emb_by_space[space]
            f.create_dataset(f"mean_emb_{space}", data=mean_emb.astype(np.float32), dtype="float32")
            f.create_dataset(f"std_emb_{space}", data=std_emb.astype(np.float32), dtype="float32")
            f.create_dataset(f"max_emb_{space}", data=max_emb.astype(np.float32), dtype="float32")
            f.attrs[f"feature_dim_tile_{space}"] = int(dim_by_space[space])
            f.attrs[f"tile_feature_postproc_{space}"] = str(postproc_by_space[space])
        f.create_dataset("case_ids", data=np.asarray([str(c["case_id"]) for c in cases], dtype=object), dtype=str_dt)
        f.create_dataset("case_uuids", data=np.asarray([str(c["case_uuid"]) for c in cases], dtype=object), dtype=str_dt)
        f.create_dataset("project_id", data=str(project_id), dtype=str_dt)
        f.create_dataset("n_tiles_used_after_cap", data=np.asarray([int(c["n_tiles_used_after_cap"]) for c in cases], dtype=np.int64))
        f.create_dataset("n_slides_retained", data=np.asarray([int(c["n_slides_retained"]) for c in cases], dtype=np.int64))
        f.attrs["project_id"] = str(project_id)
        f.attrs["spaces"] = ",".join(spaces)
        f.attrs["pooling_variants"] = ",".join(POOLING_VARIANTS)
        f.attrs["std_ddof"] = 0
    os.replace(tmp, path)


def _write_failed_tiles(path: Path, rows: Sequence[Mapping[str, Any]], *, overwrite: bool) -> None:
    df = pd.DataFrame(list(rows)) if rows else pd.DataFrame(columns=FAILED_TILE_COLUMNS)
    for c in FAILED_TILE_COLUMNS:
        if c not in df.columns:
            df[c] = pd.NA
    write_csv_atomic(path, df.loc[:, FAILED_TILE_COLUMNS], overwrite=overwrite)


def _feature_run_id(training_num: int, checkpoint_num: int, *, he_l2_normalize_tile_embeddings: bool = False) -> str:
    # Same as _eval_common.encoder_features_run_id
    return _encoder_features_run_id_helper(
        training_num, checkpoint_num,
        he_l2_normalize_tile_embeddings=he_l2_normalize_tile_embeddings,
    )


def _validate_requested_bundle(bundle: Mapping[str, Any], bundle_path: Path, training_num: int, checkpoint_num: int) -> None:
    ckpt_epoch = bundle.get("checkpoint_epoch")
    if ckpt_epoch is not None and int(ckpt_epoch) != int(checkpoint_num):
        raise RuntimeError(f"Bundle checkpoint_epoch={ckpt_epoch} does not match requested checkpoint_num={checkpoint_num}")
    m = re.search(r"HEencoder_t(-?\d+)e(-?\d+)\.pt$", bundle_path.name)
    if m:
        t, e = int(m.group(1)), int(m.group(2))
        if t != int(training_num) or e != int(checkpoint_num):
            raise RuntimeError(f"Bundle filename {bundle_path.name} does not match requested t={training_num}, e={checkpoint_num}")


def _validation_gate(benchmark_root: Path, *, allow_unvalidated: bool) -> Dict[str, Any]:
    path = benchmark_root / "reports" / "benchmark_validation_report.json"
    if not path.exists():
        if allow_unvalidated:
            return {"status": "not_found_but_allowed", "path": str(path)}
        raise RuntimeError(f"Benchmark validation report is missing. Run _14_validate_benchmark.py first, or pass --allow_unvalidated_benchmark: {path}")
    obj = read_json(path)
    if obj.get("status") != "passed" and not allow_unvalidated:
        raise RuntimeError(f"Benchmark validation did not pass: {path}")
    return {"status": obj.get("status"), "path": str(path), "sha256": sha256_path(path)}


def _worker_init(_):
    global _SLIDE_CACHE, _TILE_CACHE_HANDLES
    _SLIDE_CACHE = None
    _TILE_CACHE_HANDLES = None
    # One OpenCV thread per data-loading worker (avoids CPU oversubscription; pixels unchanged),
    # so --num_workers can be set close to --cpus-per-task
    if cv2 is not None:
        cv2.setNumThreads(0)


def _resolve_device(requested_device: str):
    """Resolve the torch device; raises if a CUDA device is requested but not available (no silent CPU fallback)."""
    if torch is None:
        raise RuntimeError("PyTorch is required for feature extraction but could not be imported.")
    req = str(requested_device).strip()
    wants_cuda = req == "cuda" or req.startswith("cuda:")
    if wants_cuda and not torch.cuda.is_available():
        raise RuntimeError(
            f"--device {req!r} was requested but torch.cuda.is_available() is False: no usable "
            "CUDA device in this environment. Check the GPU allocation, the CUDA module and the "
            "torch build, or pass --device cpu."
        )
    device = torch.device(req)
    if device.type == "cuda":
        idx = device.index if device.index is not None else torch.cuda.current_device()
        print(
            f"Using CUDA device {idx}: {torch.cuda.get_device_name(idx)} "
            f"(torch {torch.__version__} sees {torch.cuda.device_count()} CUDA device(s)).",
            file=sys.stderr,
        )
    else:
        print(f"Using device '{device.type}' for encoder inference (CUDA not requested).", file=sys.stderr)
    return device


def _main_impl(args) -> int:

    if args.batch_size <= 0:
        raise RuntimeError("batch_size must be positive")
    if args.num_workers < 0:
        raise RuntimeError("num_workers must be >= 0")
    if bool(getattr(args, "resume", False)) and bool(args.allow_tile_read_failures):
        raise RuntimeError(
            "--resume is incompatible with --allow_tile_read_failures (only fully encoded cases "
            "are saved as resume shards). Re-run without --allow_tile_read_failures, or use --force "
            "for a fresh run."
        )

    if torch is None:
        raise RuntimeError("PyTorch is required")
    if h5py is None:
        raise RuntimeError("h5py is required")
    if cv2 is None:
        raise RuntimeError("OpenCV (cv2) is required for tile resizing; install opencv-python-headless")

    from src.evaluation.extract_encoders import load_exported_he_encoder

    data_root = get_full_path('eval', 'eval_datasets', 'TCGA_eval_data')
    benchmark_root = data_root / "benchmarks" / args.benchmark_id
    settings_dir = benchmark_root / "settings"
    cohort_dir = benchmark_root / "shared" / "cohort"
    tiles_dir = benchmark_root / "shared" / "tiles"
    if not settings_dir.exists():
        raise FileNotFoundError(settings_dir)
    validation_status = _validation_gate(benchmark_root, allow_unvalidated=bool(args.allow_unvalidated_benchmark))

    cfg = _load_mapping(settings_dir / "benchmark_config.yaml")
    links = _load_mapping(settings_dir / "benchmark_asset_links.json")
    modelgroup_id = str(args.modelgroup_id or cfg.get("modelgroup_id") or links.get("modelgroup_id"))
    if modelgroup_id != str(cfg.get("modelgroup_id") or links.get("modelgroup_id")):
        raise RuntimeError(f"Requested modelgroup_id={modelgroup_id} disagrees with benchmark config/links")
    # Settings files resolved under the current root (settings_dir), not from the *_path values of
    # benchmark_asset_links.json
    modelgroup_path = settings_dir / "modelgroups" / f"{modelgroup_id}.yaml"
    modelgroup = _load_mapping(modelgroup_path)

    case_tile_path = tiles_dir / "case_tile_index.parquet"
    final_case_path = cohort_dir / "final_case_manifest.csv"
    download_manifest_path = data_root / "selected_slide_cache" / args.benchmark_id / "manifests" / "selected_slide_download_manifest.tsv"
    encoder_registry_path = settings_dir / "encoder_registry.json"  # current root (see above)
    for p in [case_tile_path, final_case_path, download_manifest_path, encoder_registry_path, modelgroup_path]:
        if not p.exists():
            raise FileNotFoundError(p)

    he_l2 = bool(args.he_l2_normalize_tile_embeddings)
    # Default run: he (raw) + clip (per-tile L2, if exported); he-L2 run: he only, non-primary
    official_primary_feature_run = (
        args.limit_cases is None
        and not args.allow_tile_read_failures
        and not he_l2
    )
    run_id = _feature_run_id(
        args.training_num,
        args.checkpoint_num,
        he_l2_normalize_tile_embeddings=he_l2,
    )
    run_root = benchmark_encoder_features_root(args.benchmark_id, run_id)
    features_dir = run_root / "features"
    diagnostics_dir = run_root / "diagnostics"
    resume_enabled = bool(args.resume)
    # With --force or --resume, all outputs are rewritten (overwrite allowed)
    finalize_overwrite = bool(args.force or resume_enabled)
    if run_root.exists() and args.force:
        # Remove the outputs of the previous run
        shutil.rmtree(run_root)
    elif (run_root / "run_manifest.json").exists():
        # A completed run (the root manifest is written last). Resume has nothing to do.
        if resume_enabled:
            print(json.dumps({"status": "already_complete", "run_root": str(run_root), "encoder_id": run_id}, indent=2))
            return 0
        raise FileExistsError(f"Run output already exists; use --force to rebuild: {run_root}")
    elif run_root.exists() and not resume_enabled:
        # A plain re-run is refused if outputs or .partial shards exist (use --resume or --force)
        non_dot_entries = [
            p for p in run_root.rglob("*")
            if not any(seg.startswith(".") for seg in p.relative_to(run_root).parts)
        ]
        if non_dot_entries or (run_root / PARTIAL_DIRNAME).exists():
            raise FileExistsError(
                f"Run output already exists (incomplete); use --resume to continue or --force to rebuild: {run_root}"
            )
    features_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    bundle_path = _resolve_bundle_path(args.training_num, args.checkpoint_num)
    bundle = _torch_load_bundle(bundle_path, device="cpu")
    _validate_requested_bundle(bundle, bundle_path, args.training_num, args.checkpoint_num)
    bundle_sha = sha256_path(bundle_path)

    device = _resolve_device(args.device)
    model = load_exported_he_encoder(str(bundle_path), device=device).to(device).eval()
    preprocess = _preprocessing_metadata(model, bundle.get("config"))
    flags = _infer_encoder_flags(bundle, model, int(args.checkpoint_num))
    registry_audit = _validate_registry(encoder_registry_path, preprocess, modelgroup)
    has_image_proj = bool(flags["bundle_has_image_proj"])
    dims = _actual_output_dims(model, int(preprocess["crop_size"]), device, has_image_proj=has_image_proj)

    # Spaces of this run: clip needs an exported image_proj; the he-L2 run is he only
    if he_l2:
        spaces = ["he"]
        postproc_by_space = {"he": TILE_POSTPROC_L2}
    else:
        spaces = ["he"] + (["clip"] if has_image_proj else [])
        postproc_by_space = {"he": TILE_POSTPROC_NONE}
        if has_image_proj:
            postproc_by_space["clip"] = TILE_POSTPROC_L2
    spaces = [s for s in SPACE_ORDER if s in spaces]  # canonical (he, clip) order
    dim_by_space: Dict[str, int] = {"he": int(dims["actual_output_dim_project_false"])}
    if "clip" in spaces:
        dim_by_space["clip"] = int(dims["actual_output_dim_project_true"])
    for space in spaces:
        if int(dim_by_space[space]) <= 0:
            raise RuntimeError(f"Invalid actual output dimension for space={space!r}: {dim_by_space[space]}")

    final_case = pd.read_csv(final_case_path, dtype=str, keep_default_na=False)
    final_case = final_case.loc[final_case["final_case_eligible"].map(_truthy)].copy()
    if args.limit_cases is not None:
        final_case = final_case.sort_values(["project_id", "case_id"]).groupby("project_id", sort=True).head(int(args.limit_cases)).copy()
    if final_case.empty:
        raise RuntimeError("No final eligible cases to extract")

    case_tile = pd.read_parquet(case_tile_path)
    if str(case_tile["benchmark_id"].iloc[0]) != args.benchmark_id or not case_tile["modelgroup_id"].astype(str).eq(modelgroup_id).all():
        raise RuntimeError("case_tile_index benchmark_id/modelgroup_id mismatch")
    final_cases = set(final_case["case_id"].astype(str))
    case_tile = case_tile.loc[case_tile["case_id"].astype(str).isin(final_cases)].copy()
    if case_tile.empty:
        raise RuntimeError("case_tile_index has no rows for final cases")

    # The download manifest is only validated: WSI paths are rebuilt from slide_file_uuid + slide_file_name
    # under the current root (shared slide store)
    dl = pd.read_csv(download_manifest_path, sep="\t", dtype=str, keep_default_na=False)
    if "local_path" not in dl.columns or "slide_file_uuid" not in dl.columns:
        raise RuntimeError("selected_slide_download_manifest.tsv missing local_path/slide_file_uuid")
    if "slide_file_name" not in case_tile.columns:
        raise RuntimeError("case_tile_index.parquet missing slide_file_name (needed to locate WSIs)")
    # Slides without a file name map to "" (checked only if read from the WSI).
    # --wsi_on_scratch: WSI store under the scratch root of get_full_path
    wsi_data_root = (
        get_full_path("eval", "eval_datasets", "TCGA_eval_data", scratch=True)
        if bool(getattr(args, "wsi_on_scratch", False))
        else data_root
    )
    case_tile["local_path"] = [
        str(shared_wsi_path(wsi_data_root, str(u), str(fn)))
        if (fn is not None and not pd.isna(fn) and str(fn) != "")
        else ""
        for u, fn in zip(case_tile["slide_file_uuid"], case_tile["slide_file_name"])
    ]
    # WSI existence is checked below, only for slides read from the WSI

    # Deterministic extraction order favors slide locality while preserving frozen tile IDs for pooling.
    case_tile = case_tile.sort_values(["slide_file_uuid", "top_left_y_l0", "top_left_x_l0", "case_id", "tile_id"]).reset_index(drop=True)
    frozen_case_meta = {str(r["case_id"]): r for r in final_case.to_dict("records")}

    # Resolve tile_asset_id once for the case-index consistency check.
    tile_asset_id_resolved = str(case_tile["tile_asset_id"].iloc[0]) if "tile_asset_id" in case_tile.columns else ""
    if tile_asset_id_resolved and not case_tile["tile_asset_id"].astype(str).eq(tile_asset_id_resolved).all():
        raise RuntimeError("case_tile_index contains rows from multiple tile_asset_ids — refusing to extract.")

    # Expected tile counts over the full case set (before --resume filtering)
    expected_counts_full = case_tile.groupby("case_id")["tile_id"].size().astype(int).to_dict()
    n_selected_tiles_total = int(len(case_tile))
    # case_state is created here so that resume shards can be loaded before encoding
    case_state: MutableMapping[str, Dict[str, Any]] = {}

    # Tile RGB cache paths from tile_rgb_cache_manifest.json (a single set recipe and codec recipe required)
    benchmark_root = data_root / "benchmarks" / args.benchmark_id
    links_path = benchmark_root / "settings" / "benchmark_asset_links.json"
    links = read_json(links_path) if links_path.exists() else {}
    links = links if isinstance(links, Mapping) else {}
    set_recipe_id_resolved = str(links.get("tile_set_recipe_id") or "")
    codec_recipe_id_resolved = str(links.get("tile_codec_recipe_id") or "")
    tile_rgb_cache_dir = (
        data_root / "processed_assets" / "tiles" / tile_asset_id_resolved / "tile_rgb_cache"
        if tile_asset_id_resolved
        else None
    )
    cache_paths_by_uuid: Dict[str, str] = {}
    if tile_rgb_cache_dir is not None:
        # Cache manifest resolved under the current root (not the absolute path stored in the links)
        cache_mf_path = tile_rgb_cache_dir / "tile_rgb_cache_manifest.json"
        if cache_mf_path.exists():
            cache_mf = read_json(cache_mf_path)
            ref_tiles = cache_mf.get("referenced_shared_tiles") if isinstance(cache_mf, Mapping) else None
            if isinstance(ref_tiles, Mapping):
                set_ids: set = set()
                codec_ids: set = set()
                for uuid, info in ref_tiles.items():
                    if not isinstance(info, Mapping):
                        continue
                    set_id = str(info.get("set_recipe_id") or "")
                    codec_id = str(info.get("codec_recipe_id") or "")
                    # tiles.h5 path rebuilt from its content-addressed ids under the current root
                    if set_id and codec_id:
                        h5 = shared_tile_h5_path(data_root, str(uuid), set_id, codec_id)
                        if h5.exists():
                            if args.verify_cache_sha256:
                                verify_blob_sha256(h5, info.get("h5_sha256"), kind=f"tile cache {uuid}")
                            cache_paths_by_uuid[str(uuid)] = str(h5)
                    if set_id:
                        set_ids.add(set_id)
                    if codec_id:
                        codec_ids.add(codec_id)
                if len(set_ids) > 1 or len(codec_ids) > 1:
                    raise RuntimeError(
                        f"Tile RGB cache references multiple recipes (set={sorted(set_ids)}, "
                        f"codec={sorted(codec_ids)}) — refusing to extract from a heterogeneous cache."
                    )
                if set_ids and set_recipe_id_resolved and set_recipe_id_resolved not in set_ids:
                    raise RuntimeError(f"links tile_set_recipe_id={set_recipe_id_resolved!r} not in cache set recipes {sorted(set_ids)}")
                if codec_ids and codec_recipe_id_resolved and codec_recipe_id_resolved not in codec_ids:
                    raise RuntimeError(f"links tile_codec_recipe_id={codec_recipe_id_resolved!r} not in cache codec recipes {sorted(codec_ids)}")
                if not set_recipe_id_resolved and len(set_ids) == 1:
                    set_recipe_id_resolved = next(iter(set_ids))
                if not codec_recipe_id_resolved and len(codec_ids) == 1:
                    codec_recipe_id_resolved = next(iter(codec_ids))

    cache_mode = str(args.use_tile_rgb_cache or "auto").lower()
    if cache_mode not in {"auto", "always", "never"}:
        raise RuntimeError(f"--use_tile_rgb_cache must be one of auto/always/never; got {cache_mode!r}")

    # ------------------------------------------------------------------ resume
    # Resume fingerprint: everything that determines a case's pooled features (encoder, preprocessing,
    # tile set, pixel source)
    run_fingerprint = _resume_run_fingerprint({
        "schema": RESUME_SCHEMA,
        "benchmark_id": str(args.benchmark_id),
        "modelgroup_id": str(modelgroup_id),
        "encoder_id": str(run_id),
        "training_num": int(args.training_num),
        "checkpoint_num": int(args.checkpoint_num),
        "bundle_sha256": str(bundle_sha),
        "preprocessing_hash": str(preprocess.get("preprocessing_hash", "")),
        "spaces": list(spaces),
        "tile_feature_postproc_by_space": dict(postproc_by_space),
        "feature_dim_tile_by_space": {s: int(dim_by_space[s]) for s in spaces},
        "he_l2_normalize_tile_embeddings": bool(he_l2),
        "tile_cache_mode": str(cache_mode),
        "tile_set_recipe_id": str(set_recipe_id_resolved),
        "tile_codec_recipe_id": str(codec_recipe_id_resolved),
        "case_tile_index_sha256": sha256_path(case_tile_path),
        "final_case_manifest_sha256": sha256_path(final_case_path),
    })
    shards_dir = run_root / PARTIAL_DIRNAME / SHARDS_DIRNAME
    done_case_ids: set = set()
    if resume_enabled:
        partial_dir = run_root / PARTIAL_DIRNAME
        partial_dir.mkdir(parents=True, exist_ok=True)
        meta_path = partial_dir / RESUME_META_NAME
        if meta_path.exists():
            prev_meta = read_json(meta_path)
            prev_fp = str(prev_meta.get("run_fingerprint")) if isinstance(prev_meta, Mapping) else ""
            if prev_fp != run_fingerprint:
                raise RuntimeError(
                    "Cannot --resume: the partial run has a different configuration "
                    f"(fingerprint {prev_fp!r} != current {run_fingerprint!r}: encoder weights, "
                    "preprocessing, tile set or pixel source changed). Use --force to rebuild from scratch."
                )
        else:
            write_json_atomic(meta_path, {
                "schema": RESUME_SCHEMA,
                "run_fingerprint": run_fingerprint,
                "benchmark_id": str(args.benchmark_id),
                "encoder_id": str(run_id),
                "training_num": int(args.training_num),
                "checkpoint_num": int(args.checkpoint_num),
                "bundle_sha256": str(bundle_sha),
                "spaces": list(spaces),
                "tile_cache_mode": str(cache_mode),
                "tile_set_recipe_id": str(set_recipe_id_resolved),
                "tile_codec_recipe_id": str(codec_recipe_id_resolved),
            }, overwrite=True)
        loaded = _load_case_shards(shards_dir, run_fingerprint=run_fingerprint, spaces=spaces, dim_by_space=dim_by_space)
        for cid, st in loaded.items():
            # Skip shards of cases outside the current cohort
            if cid not in final_cases:
                continue
            case_state[cid] = st
            done_case_ids.add(cid)
        print(
            f"RESUME: loaded {len(done_case_ids)} previously-encoded case shard(s) from {shards_dir}.",
            file=sys.stderr,
        )

    # Skip the tiles of already encoded cases
    if done_case_ids:
        before = int(len(case_tile))
        case_tile = case_tile.loc[~case_tile["case_id"].astype(str).isin(done_case_ids)].reset_index(drop=True)
        print(
            f"RESUME: skipping {before - int(len(case_tile))} tile(s) of {len(done_case_ids)} "
            f"already-encoded case(s); {len(case_tile)} tile(s) remain to encode.",
            file=sys.stderr,
        )

    unique_uuids = case_tile["slide_file_uuid"].astype(str).unique().tolist()
    if cache_mode == "always":
        missing_cache = [u for u in unique_uuids if u not in cache_paths_by_uuid]
        if missing_cache:
            raise RuntimeError(
                f"--use_tile_rgb_cache=always but per-slide cache HDF5 is missing for "
                f"{len(missing_cache)} slides (first 20: {missing_cache[:20]}). "
                f"Run _15_build_tile_rgb_cache.py first or use --use_tile_rgb_cache auto."
            )
    if cache_mode == "auto" and not args.allow_partial_cache:
        n_cached = sum(1 for u in unique_uuids if u in cache_paths_by_uuid)
        if 0 < n_cached < len(unique_uuids):
            uncached = [u for u in unique_uuids if u not in cache_paths_by_uuid]
            raise RuntimeError(
                f"--use_tile_rgb_cache=auto found a partial tile RGB cache: {n_cached} of "
                f"{len(unique_uuids)} final slides are cached (e.g. uncached: {uncached[:10]}). "
                f"Complete the cache with _15_build_tile_rgb_cache.py and use --use_tile_rgb_cache always, "
                f"read all slides from the WSI with --use_tile_rgb_cache never (default), or pass "
                f"--allow_partial_cache."
            )
    if cache_mode == "auto" and not cache_paths_by_uuid:
        print(
            "INFO: --use_tile_rgb_cache=auto but no per-slide cache HDF5 found under "
            f"{tile_rgb_cache_dir} — falling back to WSI read_region for every tile.",
            file=sys.stderr,
        )

    # Which slides will be served from the cache (and thus need no WSI on disk)?
    #   never  -> none (all WSI)
    #   always -> all (per-tile coverage is enforced lazily in _try_cache)
    #   auto   -> those with a present cache file; the rest are read from the WSI
    if cache_mode == "never":
        served_from_cache: set = set()
    elif cache_mode == "always":
        served_from_cache = set(unique_uuids)
    else:  # auto
        served_from_cache = set(cache_paths_by_uuid.keys())

    wsi_needed = [u for u in unique_uuids if u not in served_from_cache]
    if wsi_needed:
        slide_paths = case_tile[["slide_file_uuid", "local_path"]].drop_duplicates("slide_file_uuid")
        slide_paths = slide_paths.loc[slide_paths["slide_file_uuid"].astype(str).isin(set(wsi_needed))]
        missing_path = slide_paths.loc[
            slide_paths["local_path"].isna()
            | (slide_paths["local_path"].astype(str) == "")
            | (~slide_paths["local_path"].map(lambda p: Path(str(p)).exists()))
        ]
        if not missing_path.empty:
            bad = missing_path[["slide_file_uuid", "local_path"]].head(20).to_dict("records")
            raise RuntimeError(
                f"Missing WSI local_path for {len(missing_path)} slide(s) not served from the tile RGB "
                f"cache (cache_mode={cache_mode!r}): {bad}. Build/complete the cache with "
                f"_15_build_tile_rgb_cache.py and use --use_tile_rgb_cache always, restore the WSIs, "
                f"or pass --wsi_on_scratch if they are under the scratch root."
            )

    # WSI readability preflight: open each WSI to be read and decode one region (disable with --skip_wsi_preflight)
    if wsi_needed and not args.skip_wsi_preflight:
        reps = case_tile.drop_duplicates("slide_file_uuid")[
            ["slide_file_uuid", "local_path", "top_left_x_l0", "top_left_y_l0",
             "read_window_px_l0_x", "read_window_px_l0_y"]
        ].copy()
        reps["slide_file_uuid"] = reps["slide_file_uuid"].astype(str)
        reps = reps[reps["slide_file_uuid"].isin(set(wsi_needed))].set_index("slide_file_uuid")
        preflight_failures: List[Dict[str, str]] = []
        n_wsi = len(wsi_needed)
        for i, u in enumerate(wsi_needed, 1):
            row = reps.loc[u]
            lp = str(row["local_path"])
            try:
                slide = _slide_for_path(lp)
                x = int(float(row["top_left_x_l0"])); y = int(float(row["top_left_y_l0"]))
                wx = int(float(row["read_window_px_l0_x"])); wy = int(float(row["read_window_px_l0_y"]))
                slide.read_region((x, y), 0, (wx, wy))
            except Exception as e:
                preflight_failures.append({
                    "slide_file_uuid": str(u), "local_path": lp,
                    "error": f"{type(e).__name__}: {e}",
                })
            if i % 100 == 0 or i == n_wsi:
                print(f"WSI preflight: {i}/{n_wsi} checked, {len(preflight_failures)} unreadable",
                      file=sys.stderr)
        if preflight_failures:
            raise RuntimeError(
                f"WSI-readability preflight failed for {len(preflight_failures)} of {n_wsi} slide(s) "
                f"before extraction (corrupt/truncated WSIs). Re-download them with "
                f"_8_download_selected_slides.py, or pass --skip_wsi_preflight. "
                f"First failures: {preflight_failures[:10]}"
            )

    # Tile RGB cache provenance (codec/quality) for the run manifest
    tile_rgb_cache_provenance: Optional[Dict[str, Any]] = None
    if cache_mode != "never" and cache_paths_by_uuid and tile_rgb_cache_dir is not None:
        _cache_mf_path = tile_rgb_cache_dir / "tile_rgb_cache_manifest.json"
        _cache_mf = read_json(_cache_mf_path) if _cache_mf_path.exists() else None
        tile_rgb_cache_provenance = {
            "cache_dir": str(tile_rgb_cache_dir),
            "manifest_path": str(_cache_mf_path) if _cache_mf_path.exists() else None,
            "manifest_sha256": sha256_path(_cache_mf_path) if _cache_mf_path.exists() else None,
            "set_recipe_id": set_recipe_id_resolved,
            "codec_recipe_id": codec_recipe_id_resolved,
            "format": (_cache_mf or {}).get("format"),
            "cache_format": (_cache_mf or {}).get("cache_format"),
            "quality": (_cache_mf or {}).get("quality"),
            "chroma": (_cache_mf or {}).get("chroma"),
            "frozen_final": (_cache_mf or {}).get("frozen_final"),
            "n_slides_served_from_cache": int(len(served_from_cache)),
        }

    dataset = TileDataset(
        case_tile, preprocess,
        benchmark_id=args.benchmark_id, modelgroup_id=modelgroup_id,
        expected_set_recipe_id=set_recipe_id_resolved,
        expected_codec_recipe_id=codec_recipe_id_resolved,
        cache_paths_by_uuid=cache_paths_by_uuid if cache_mode != "never" else {},
        cache_mode=cache_mode,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=(device.type == "cuda"),
        collate_fn=_collate_tiles,
        persistent_workers=bool(args.num_workers > 0),
        worker_init_fn=_worker_init
    )

    # case_state already holds the resumed cases; sharded_this_run: cases checkpointed in this run
    failures: List[Dict[str, Any]] = []
    n_batches = 0
    n_tiles_encoded = 0
    n_cache_hits_total = 0
    n_cache_misses_total = 0
    sharded_this_run: set = set()

    n_cases_to_encode = int(case_tile["case_id"].astype(str).nunique())
    print(
        f"Encoding {len(case_tile)} tile(s) from {n_cases_to_encode} case(s) in "
        f"{len(loader)} batch(es) of up to {int(args.batch_size)} on {device.type}...",
        file=sys.stderr,
    )
    pbar = tqdm(loader, total=len(loader), desc="Encoding tiles", unit="batch")
    for batch in pbar:
        n_batches += 1
        failures.extend(batch["failures"])
        n_cache_hits_total += int(batch.get("n_cache_hits") or 0)
        n_cache_misses_total += int(batch.get("n_cache_misses") or 0)
        images = batch["images"]
        metas = batch["metas"]
        if images is None or not metas:
            continue
        emb_by_space = _encode_batch(model, images, spaces=spaces, he_l2_normalize=he_l2, device=device)
        for space in spaces:
            if emb_by_space[space].shape[1] != dim_by_space[space]:
                raise RuntimeError(
                    f"Encoder dimension changed during extraction for space={space!r}: "
                    f"first={dim_by_space[space]}, batch={emb_by_space[space].shape[1]}"
                )
        for i, meta in enumerate(metas):
            cid = str(meta["case_id"])
            if cid not in frozen_case_meta:
                raise RuntimeError(f"Encoded tile for non-final case: {cid}")
            if cid in done_case_ids or cid in sharded_this_run:
                raise RuntimeError(
                    f"Encoded a tile for case {cid} that was already fully encoded/checkpointed — "
                    f"the frozen tile set is inconsistent (over-count)."
                )
            st = case_state.get(cid)
            if st is None:
                st = _init_case_state(meta, dim_by_space, frozen_case_meta[cid])
                case_state[cid] = st
            # One Welford step per tile (shared count n, per-space accumulators)
            st["n"] += 1
            n = float(st["n"])
            for space in spaces:
                vec = emb_by_space[space][i]
                a = st["acc"][space]
                delta = vec - a["mean"]
                a["mean"] += delta / n
                a["M2"] += delta * (vec - a["mean"])
                np.maximum(a["max_emb"], vec, out=a["max_emb"])
            n_tiles_encoded += 1
        # Checkpoint the cases completed in this batch (--resume only)
        if resume_enabled:
            for cid in {str(m["case_id"]) for m in metas}:
                if cid in sharded_this_run:
                    continue
                st = case_state.get(cid)
                if st is not None and int(st["n"]) == int(expected_counts_full.get(cid, -1)):
                    _write_case_shard(shards_dir, st, run_fingerprint=run_fingerprint, spaces=spaces)
                    sharded_this_run.add(cid)
        pbar.set_postfix(tiles=n_tiles_encoded, refresh=False)
    pbar.close()

    # Checkpoint any remaining completed case
    if resume_enabled:
        for cid, st in case_state.items():
            if cid in done_case_ids or cid in sharded_this_run:
                continue
            if int(st["n"]) == int(expected_counts_full.get(cid, -1)):
                _write_case_shard(shards_dir, st, run_fingerprint=run_fingerprint, spaces=spaces)
                sharded_this_run.add(cid)

    failed_path = diagnostics_dir / "failed_tiles.csv"
    _write_failed_tiles(failed_path, failures, overwrite=True if finalize_overwrite else not failed_path.exists())
    if failures and not args.allow_tile_read_failures:
        raise RuntimeError(f"Tile read/preprocess failures occurred ({len(failures)}). See {failed_path}")

    missing_cases = sorted(final_cases - set(case_state.keys()))
    if missing_cases:
        raise RuntimeError(f"Final cases with no encoded tiles (and no resumed shard): {missing_cases[:20]}")

    # Counts over the full cohort (resumed cases included)
    expected_counts = {cid: expected_counts_full[cid] for cid in expected_counts_full if cid in final_cases}
    bad_counts = []
    for cid, expected in expected_counts.items():
        observed = int(case_state[str(cid)]["n"])
        if observed != int(expected):
            bad_counts.append({"case_id": str(cid), "expected_tiles": int(expected), "encoded_tiles": observed})
    if bad_counts and not args.allow_tile_read_failures:
        raise RuntimeError(f"Encoded tile counts do not match frozen case_tile_index: {bad_counts[:20]}")

    # TSS clustering diagnostic for every produced space
    project_id_by_case = {str(r["case_id"]): str(r["project_id"]) for r in final_case.to_dict("records")}
    tss_summary_by_space: Dict[str, Any] = {}
    tss_per_case_frames: List[pd.DataFrame] = []
    for space in spaces:
        s_summary, s_per_case = _compute_tss_clustering(case_state, project_id_by_case, space=space)
        tss_summary_by_space[space] = s_summary
        tss_per_case_frames.append(s_per_case)
    tss_per_case_df = (
        pd.concat(tss_per_case_frames, ignore_index=True) if tss_per_case_frames else pd.DataFrame()
    )
    tss_clustering_path = diagnostics_dir / "tss_clustering.json"
    tss_per_case_path = diagnostics_dir / "tss_per_case.csv"
    write_json_atomic(
        tss_clustering_path,
        {
            "benchmark_id": args.benchmark_id,
            "modelgroup_id": modelgroup_id,
            "encoder_id": run_id,
            "spaces": list(spaces),
            "by_space": tss_summary_by_space,
        },
        overwrite=finalize_overwrite,
    )
    write_csv_atomic(tss_per_case_path, tss_per_case_df, overwrite=finalize_overwrite)

    feature_hashes: Dict[str, str] = {}
    feature_index_rows: List[Dict[str, Any]] = []
    for project_id, g in final_case.sort_values(["project_id", "case_id"]).groupby("project_id", sort=True):
        project_cases: List[Dict[str, Any]] = []
        rows_by_space: Dict[str, Dict[str, List[np.ndarray]]] = {
            space: {"mean": [], "std": [], "max": []} for space in spaces
        }
        for row_idx, r in enumerate(g.sort_values(["case_id", "case_uuid"]).to_dict("records")):
            cid = str(r["case_id"])
            st = case_state[cid]
            n = int(st["n"])
            if n <= 0:
                raise RuntimeError(f"No encoded tiles for case {cid}")
            for space in spaces:
                a = st["acc"][space]
                mean = a["mean"]
                # Population std (ddof=0), 0 for single-tile cases (keep min_tiles_per_case > 1)
                var = np.maximum(a["M2"] / float(n), 0.0)
                std = np.sqrt(var)
                max_emb = a["max_emb"]
                for name, arr in (("mean", mean), ("std", std), ("max", max_emb)):
                    if not np.isfinite(arr).all():
                        raise RuntimeError(f"Non-finite {name} pooled feature for case {cid} (space={space})")
                rows_by_space[space]["mean"].append(mean.astype(np.float32))
                rows_by_space[space]["std"].append(std.astype(np.float32))
                rows_by_space[space]["max"].append(max_emb.astype(np.float32))
            case_rec = {
                "case_id": cid,
                "case_uuid": str(r["case_uuid"]),
                "project_id": str(project_id),
                "n_tiles_used_after_cap": int(float(r.get("n_tiles_used_after_cap", n))),
                "n_tiles_encoded": n,
                "n_slides_retained": int(float(r.get("n_slides_retained", 0))),
            }
            project_cases.append(case_rec)
            feature_index_rows.append({
                "case_id": cid,
                "case_uuid": str(r["case_uuid"]),
                "project_id": str(project_id),
                "feature_file": str(features_dir / f"{project_id}.h5"),
                "row_index": int(row_idx),
                "n_tiles_used_after_cap": int(float(r.get("n_tiles_used_after_cap", n))),
                "n_tiles_encoded": n,
                "n_slides_retained": int(float(r.get("n_slides_retained", 0))),
                "benchmark_id": args.benchmark_id,
                "modelgroup_id": modelgroup_id,
                "encoder_id": run_id,
                "spaces": ",".join(spaces),
            })
        emb_by_space = {
            space: (
                np.vstack(rows_by_space[space]["mean"]).astype(np.float32),
                np.vstack(rows_by_space[space]["std"]).astype(np.float32),
                np.vstack(rows_by_space[space]["max"]).astype(np.float32),
            )
            for space in spaces
        }
        fpath = features_dir / f"{project_id}.h5"
        if fpath.exists() and not finalize_overwrite:
            raise FileExistsError(fpath)
        _write_project_h5(
            fpath,
            project_cases,
            emb_by_space=emb_by_space,
            dim_by_space=dim_by_space,
            postproc_by_space=postproc_by_space,
            spaces=spaces,
            project_id=str(project_id),
        )
        feature_hashes[str(project_id)] = sha256_path(fpath)

    feature_index = pd.DataFrame(feature_index_rows)
    for c in FEATURE_INDEX_COLUMNS:
        if c not in feature_index.columns:
            feature_index[c] = pd.NA
    index_path = features_dir / "case_features_index.parquet"
    if index_path.exists() and not finalize_overwrite:
        raise FileExistsError(index_path)
    write_parquet_atomic(index_path, feature_index.loc[:, FEATURE_INDEX_COLUMNS], overwrite=finalize_overwrite)
    feature_hashes["case_features_index.parquet"] = sha256_path(index_path)

    used_image_proj = bool("clip" in spaces)
    foundation_baseline_only = bool(flags["foundation_baseline_only"])
    manifest_required_fields = {
        "spaces": list(spaces),
        "tile_feature_postproc_by_space": dict(postproc_by_space),
        "feature_dim_tile_by_space": {s: int(dim_by_space[s]) for s in spaces},
        "checkpoint_num": int(args.checkpoint_num),
        "foundation_baseline_only": foundation_baseline_only,
        "contains_he_mlp_head": bool(flags["contains_he_mlp_head"]),
        "contains_lora": bool(flags["contains_lora"]),
        "bundle_has_image_proj": bool(flags["bundle_has_image_proj"]),
        "used_image_proj": used_image_proj,
        "bundle_has_img_tok_proj": bool(flags["bundle_has_img_tok_proj"]),
    }
    # Check the manifest fields expected for each run type
    if int(args.checkpoint_num) < 0:
        # Pretrained baseline: he only, no heads (he per-tile L2 with --he_l2_normalize_tile_embeddings)
        expected = {
            "spaces": ["he"], "foundation_baseline_only": True, "contains_he_mlp_head": False,
            "contains_lora": False, "bundle_has_image_proj": False, "used_image_proj": False,
            "bundle_has_img_tok_proj": False,
            "tile_feature_postproc_by_space": {"he": (TILE_POSTPROC_L2 if he_l2 else TILE_POSTPROC_NONE)},
        }
        bad = {k: {"observed": manifest_required_fields[k], "expected": v} for k, v in expected.items() if manifest_required_fields[k] != v}
        if bad:
            raise RuntimeError(f"Strict pretrained baseline manifest fields invalid: {bad}")
    elif he_l2:
        # Finetuned he-L2 run: he only, per-tile L2 (structural flags recorded, not asserted)
        expected = {
            "foundation_baseline_only": False,
            "spaces": ["he"],
            "used_image_proj": False,
            "tile_feature_postproc_by_space": {"he": TILE_POSTPROC_L2},
        }
        bad = {k: {"observed": manifest_required_fields[k], "expected": v} for k, v in expected.items() if manifest_required_fields[k] != v}
        if bad:
            raise RuntimeError(f"Finetuned he-L2 sensitivity manifest fields invalid: {bad}")
    elif not foundation_baseline_only:
        # Default finetuned run: he (raw), plus clip (per-tile L2) iff the encoder exports image_proj
        has_clip = bool(manifest_required_fields["bundle_has_image_proj"])
        expected = {
            "foundation_baseline_only": False,
            "spaces": (["he", "clip"] if has_clip else ["he"]),
            "used_image_proj": has_clip,
            "tile_feature_postproc_by_space": (
                {"he": TILE_POSTPROC_NONE, "clip": TILE_POSTPROC_L2} if has_clip
                else {"he": TILE_POSTPROC_NONE}
            ),
        }
        bad = {k: {"observed": manifest_required_fields[k], "expected": v} for k, v in expected.items() if manifest_required_fields[k] != v}
        if bad:
            raise RuntimeError(f"Finetuned default-run manifest fields invalid: {bad}")

    preprocessing_audit = {
        "benchmark_id": args.benchmark_id,
        "modelgroup_id": modelgroup_id,
        "encoder_id": run_id,
        "bundle_path": str(bundle_path),
        "bundle_sha256": bundle_sha,
        "preprocessing_source_of_truth": "loaded_encoder_bundle_attributes_and_buffers",
        "preprocessing": preprocess,
        "registry_audit": registry_audit,
        "modelgroup_path": str(modelgroup_path),
        "modelgroup_hash": sha256_path(modelgroup_path),
        "encoder_registry_hash": sha256_path(encoder_registry_path),
    }
    write_json_atomic(diagnostics_dir / "preprocessing_audit.json", preprocessing_audit, overwrite=finalize_overwrite)

    # Tiles in the features (resumed + encoded); n_tiles_encoded: tiles encoded in this run
    n_tiles_encoded_total = int(sum(int(st["n"]) for cid, st in case_state.items() if cid in final_cases))
    n_cases_resumed = int(len(done_case_ids))
    # Cases newly encoded this invocation (== all cohort cases on a non-resume run).
    n_cases_encoded_this_run = int(len([c for c in case_state if c in final_cases and c not in done_case_ids]))
    extraction_summary = {
        "benchmark_id": args.benchmark_id,
        "modelgroup_id": modelgroup_id,
        "encoder_id": run_id,
        "n_cases": int(len(final_case)),
        "n_projects": int(final_case["project_id"].nunique()),
        "n_selected_tiles": int(n_selected_tiles_total),
        "n_tiles_encoded": int(n_tiles_encoded_total),
        "n_tiles_encoded_this_run": int(n_tiles_encoded),
        "resume_enabled": bool(resume_enabled),
        "n_cases_resumed_from_shards": n_cases_resumed,
        "n_cases_encoded_this_run": n_cases_encoded_this_run,
        "n_batches": int(n_batches),
        "n_failed_tiles": int(len(failures)),
        "allow_tile_read_failures": bool(args.allow_tile_read_failures),
        "spaces": list(spaces),
        "feature_dim_tile_by_space": {s: int(dim_by_space[s]) for s in spaces},
        "accumulator_dtype": "float64",
        "stored_feature_dtype": "float32",
        "pooling_variants": list(POOLING_VARIANTS),
        "std_ddof": 0,
        "debug_limit_cases": args.limit_cases,
        "tss_clustering_path": str(tss_clustering_path),
        "tss_clustering_sha256": sha256_path(tss_clustering_path),
        "tss_per_case_path": str(tss_per_case_path),
        "tss_per_case_sha256": sha256_path(tss_per_case_path),
        "tile_cache_mode": cache_mode,
        "tile_cache_n_slides_available": int(len(cache_paths_by_uuid)),
        "tile_cache_n_slides_total": int(len(unique_uuids)),
        "tile_cache_hits": int(n_cache_hits_total),
        "tile_cache_misses": int(n_cache_misses_total),
        "tile_cache_path_resolution": "rebuilt_from_ids",
        "tile_cache_sha256_verified": bool(args.verify_cache_sha256),
        "wsi_path_resolution": "rebuilt_from_ids",
        "tile_rgb_cache": tile_rgb_cache_provenance,
    }
    write_json_atomic(diagnostics_dir / "feature_extraction_summary.json", extraction_summary, overwrite=finalize_overwrite)

    run_manifest = manifest_header(Path(__file__).name, vars(args))
    run_manifest.update({
        "frozen_final": bool(official_primary_feature_run),
        "official_primary_feature_run": bool(official_primary_feature_run),
        "encoder_id": run_id,
        "benchmark_id": args.benchmark_id,
        "modelgroup_id": modelgroup_id,
        "training_num": int(args.training_num),
        "checkpoint_num": int(args.checkpoint_num),
        "encoder_bundle": str(bundle_path),
        "bundle_path": str(bundle_path),
        "bundle_sha256": bundle_sha,
        "checkpoint_epoch": bundle.get("checkpoint_epoch"),
        "projection_flags": bundle.get("projection_flags"),
        **manifest_required_fields,
        "preprocessing": preprocess,
        "image_feature_dim": int(getattr(model, "image_feature_dim", dims["actual_output_dim_project_false"])),
        "embed_dim": int(getattr(model, "embed_dim", dims["actual_output_dim_project_true"] or dims["actual_output_dim_project_false"])),
        "actual_output_dim_project_false": dims["actual_output_dim_project_false"],
        "actual_output_dim_project_true": dims["actual_output_dim_project_true"],
        "pooling_variants": list(POOLING_VARIANTS),
        "std_ddof": 0,
        "feature_file_hashes": feature_hashes,
        "case_tile_index_hash": sha256_path(case_tile_path),
        "final_case_manifest_hash": sha256_path(final_case_path),
        "download_manifest_hash": sha256_path(download_manifest_path),
        "modelgroup_hash": sha256_path(modelgroup_path),
        "encoder_registry_hash": sha256_path(encoder_registry_path),
        "tile_rgb_cache": tile_rgb_cache_provenance,
        "benchmark_validation": validation_status,
        "structural_flag_evidence": flags["evidence"],
        "case_features_index_hash": sha256_path(index_path),
        "diagnostic_hashes": collect_output_hashes(diagnostics_dir),
        "features_output_hashes": collect_output_hashes(features_dir),
        "run_root": str(run_root),
        # Resume gives the same features as a single pass
        "resume_enabled": bool(resume_enabled),
        "resume_run_fingerprint": run_fingerprint,
        "n_cases_resumed_from_shards": n_cases_resumed,
        "n_cases_encoded_this_run": n_cases_encoded_this_run,
    })
    write_json_atomic(features_dir / "run_manifest.json", run_manifest, overwrite=finalize_overwrite)
    # Include the just-written feature manifest hash in the root manifest.
    run_manifest["features_run_manifest_hash"] = sha256_path(features_dir / "run_manifest.json")
    write_json_atomic(run_root / "run_manifest.json", run_manifest, overwrite=finalize_overwrite)

    # Run complete: remove the per-case resume shards
    if resume_enabled:
        shutil.rmtree(run_root / PARTIAL_DIRNAME, ignore_errors=True)

    print(json.dumps({"status": "ok", "run_root": str(run_root), "encoder_id": run_id, **extraction_summary}, indent=2))
    return 0


def main(args) -> int:
    try:
        return _main_impl(args)
    finally:
        _close_slide_cache()


if __name__ == "__main__":

    ap = argparse.ArgumentParser(description="Extract H&E encoder features for TCGA Task B.")

    ap.add_argument("--benchmark_id", type=str,
                    required=True,
                    help=("Validated benchmark ID whose tiles are encoded. "
                          "Features are written under "
                          "benchmarks/{benchmark_id}/shared/encoder_features/ and reused by every eval_id."))

    ap.add_argument("--modelgroup_id", type=str,
                    default=None,
                    help="Model group ID. Defaults to the model group referenced by the benchmark.")

    ap.add_argument("--training_num", type=int,
                    required=True,
                    help=("HistoST training number of the encoder (bundle: trainings_HistoST/training_<training_num>/"
                          "trained_encoders/HEencoder_t<training_num>e<checkpoint_num>.pt)."))

    ap.add_argument("--checkpoint_num", type=int,
                    required=True,
                    help="Checkpoint number of the encoder (-1 for a raw pretrained encoder).")

    ap.add_argument("--batch_size", type=int,
                    default=64,
                    help="Tile batch size for encoder inference.")

    ap.add_argument("--device", type=str,
                    default="cuda",
                    help="Torch device used for encoder inference.")

    ap.add_argument("--num_workers", type=int,
                    default=4,
                    help="Number of data-loading workers.")

    ap.add_argument("--he_l2_normalize_tile_embeddings", action="store_true",
                    help=("Sensitivity run: separate he-only run (folder suffix '_he_l2tiles') with "
                          "L2-normalized he tiles before pooling."))

    ap.add_argument("--allow_tile_read_failures", action="store_true",
                    help="Continue after tile read failures if every case remains valid (marks the run non-primary).")

    ap.add_argument("--allow_unvalidated_benchmark", action="store_true",
                    help="Allow running against a benchmark that has not passed _14_validate_benchmark.py.")

    ap.add_argument("--limit_cases", type=int,
                    default=None,
                    help="Debug only: process only the first N cases; marks the run non-primary in the manifest.")

    ap.add_argument("--use_tile_rgb_cache", type=str,
                    choices=["auto", "always", "never"],
                    default="never",
                    help=(
                        "Tile source: 'never' (default, read from the WSI), 'always' (per-slide tile RGB "
                        "cache of _15_build_tile_rgb_cache.py, no WSI needed) or 'auto' (cache when present, "
                        "else WSI; a partial cache is refused, see --allow_partial_cache)."
                    ))

    ap.add_argument("--allow_partial_cache", action="store_true",
                    help=(
                        "With --use_tile_rgb_cache auto: allow a partial cache (some slides from the "
                        "cache, the rest from the WSI)."
                    ))

    ap.add_argument("--skip_wsi_preflight", action="store_true",
                    help=(
                        "Skip the WSI readability preflight (one region decoded per WSI before extraction)."
                    ))

    ap.add_argument("--wsi_on_scratch", action="store_true",
                    help=(
                        "Read the WSI shared store under the scratch root (get_full_path(..., scratch=True)); "
                        "other data stays under the default root."
                    ))

    ap.add_argument("--verify_cache_sha256", action="store_true",
                    help=(
                        "Check each per-slide tiles.h5 against the h5_sha256 of the tile RGB cache "
                        "manifest before reading it."
                    ))

    ap.add_argument("--force", action="store_true",
                    help="Overwrite an existing encoder feature run.")

    ap.add_argument("--resume", action="store_true",
                    help=(
                        "Resumable run: each fully encoded case is saved as a shard under "
                        "run_root/.partial/case_shards/ and skipped on a re-run with the same configuration "
                        "(shards removed at the end). Incompatible with --allow_tile_read_failures."
                    ))

    args = ap.parse_args()
    raise SystemExit(main(args))
