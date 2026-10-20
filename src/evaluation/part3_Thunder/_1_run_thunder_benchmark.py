#!/usr/bin/env python
"""
Run the Thunder benchmark for an exported HistoST H&E encoder.
Saved in <ROOT_HistoST> / eval / C_eval_thunder / HEencoder_t<n>e<c>_<space> /
"""

import argparse
import csv
import json
import hashlib
import logging
import os
import shutil
import sys
from dataclasses import dataclass
from importlib import resources as importlib_resources
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import cv2
import numpy as np
import torch
import yaml
from PIL import Image


# Add the project root to sys.path so that 'src' can be imported
project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.constants import get_full_path
from src.evaluation.extract_encoders import load_exported_he_encoder

# Thunder imports
from thunder import benchmark as thunder_benchmark
from thunder.models import PretrainedModel


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

# Known Thunder datasets (availability is read from the installed Thunder Hydra configs)
KNOWN_CLASSIFICATION_DATASETS = {
    "bach",
    "bracs",
    "break_his",
    "ccrcc",
    "crc",
    "esca",
    "mhist",
    "patch_camelyon",
    "tcga_crc_msi",
    "tcga_tils",
    "tcga_uniform",
    "wilds",
    "spider_breast",
    "spider_colorectal",
    "spider_skin",
    "spider_thorax",
}

KNOWN_SEGMENTATION_DATASETS = {
    "ocelot",
    "pannuke",
    "segpath_epithelial",
    "segpath_lymphocytes",
}

KNOWN_DATASETS = KNOWN_CLASSIFICATION_DATASETS | KNOWN_SEGMENTATION_DATASETS

RESUME_SCHEMA_VERSION = 1

CLASSIFICATION_TASKS = {
    "adversarial_attack",
    # "alignment_scoring",  # Disabled: requires local baseline embeddings from other models
    "image_retrieval",
    "knn",
    "linear_probing",
    "pre_computing_embeddings",
    "simple_shot",
    "transformation_invariance",
}

SEGMENTATION_TASKS = {
    "pre_computing_embeddings",
    "segmentation",
}

UNSUPPORTED_FOR_HE_ONLY = {
    "zero_shot_vlm",
}

TASK_ALIASES = {
    "classification_suite": [
        "pre_computing_embeddings",
        "knn",
        "linear_probing",
        "simple_shot",
        "image_retrieval",
        # "alignment_scoring",  # Disabled: requires local baseline embeddings from other models
        "transformation_invariance",
        "adversarial_attack",
    ],
    "segmentation_suite": [
        "pre_computing_embeddings",
        "segmentation",
    ],
    "all_non_vlm": [
        "pre_computing_embeddings",
        "knn",
        "linear_probing",
        "simple_shot",
        "image_retrieval",
        # "alignment_scoring",  # Disabled: requires local baseline embeddings from other models
        "transformation_invariance",
        "adversarial_attack",
        "segmentation",
    ],
}


_THUNDER_DATASET_CONFIG_CACHE: Dict[str, Dict[str, Any]] = {}


def discover_installed_thunder_dataset_names() -> set[str]:
    """Names of the datasets with a Hydra config in the installed Thunder package (thunder/config/dataset/*.yaml)."""
    try:
        dataset_dir = importlib_resources.files("thunder").joinpath("config", "dataset")
        names = {
            p.name.rsplit(".", 1)[0]
            for p in dataset_dir.iterdir()
            if p.is_file()
            and p.name.endswith((".yaml", ".yml"))
            and not p.name.startswith("_")
        }
    except Exception as exc:
        raise RuntimeError(
            "Could not discover installed Thunder dataset configs under "
            "thunder/config/dataset. Is the thunder package installed correctly?"
        ) from exc

    if not names:
        raise RuntimeError(
            "No Thunder dataset configs were found under thunder/config/dataset. "
            "The thunder installation looks incomplete."
        )

    return names


def load_thunder_dataset_config(dataset: str) -> Dict[str, Any]:
    """Load the installed Thunder dataset YAML (cached)."""
    if dataset in _THUNDER_DATASET_CONFIG_CACHE:
        return _THUNDER_DATASET_CONFIG_CACHE[dataset]

    try:
        cfg_path = importlib_resources.files("thunder").joinpath(
            "config",
            "dataset",
            f"{dataset}.yaml",
        )
        with cfg_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except FileNotFoundError as exc:
        installed = sorted(discover_installed_thunder_dataset_names())
        raise FileNotFoundError(
            f"Thunder dataset config not found for dataset='{dataset}'. "
            f"Installed Thunder dataset configs are: {installed}"
        ) from exc
    except Exception as exc:
        raise RuntimeError(
            f"Could not load Thunder dataset config for dataset='{dataset}'."
        ) from exc

    if not isinstance(cfg, dict):
        raise TypeError(
            f"Thunder dataset config for dataset='{dataset}' must be a dict, "
            f"got {type(cfg).__name__}."
        )

    _THUNDER_DATASET_CONFIG_CACHE[dataset] = cfg
    return cfg


def infer_installed_thunder_dataset_groups(
    installed_datasets: set[str],
) -> Tuple[set[str], set[str]]:
    """
    Split datasets into classification and segmentation, from Thunder's compatible_tasks field
    ('segmentation' in compatible_tasks -> segmentation, else classification).
    """
    classification: set[str] = set()
    segmentation: set[str] = set()

    for dataset in installed_datasets:
        cfg = load_thunder_dataset_config(dataset)
        compatible_tasks = cfg.get("compatible_tasks")
        compatible = (
            {str(t) for t in compatible_tasks}
            if isinstance(compatible_tasks, list)
            else set()
        )

        if "segmentation" in compatible:
            segmentation.add(dataset)
        else:
            classification.add(dataset)

    return classification, segmentation


INSTALLED_THUNDER_DATASETS = discover_installed_thunder_dataset_names()
CLASSIFICATION_DATASETS, SEGMENTATION_DATASETS = infer_installed_thunder_dataset_groups(INSTALLED_THUNDER_DATASETS)
SUPPORTED_DATASETS = CLASSIFICATION_DATASETS | SEGMENTATION_DATASETS
KNOWN_BUT_NOT_INSTALLED_DATASETS = KNOWN_DATASETS - INSTALLED_THUNDER_DATASETS

DATASET_ALIASES = {
    "classification": sorted(CLASSIFICATION_DATASETS),
    "segmentation": sorted(SEGMENTATION_DATASETS),
    "all": sorted(SUPPORTED_DATASETS),
}


# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------


def setup_logger(log_path: Path) -> logging.Logger:
    """Configure the root logger (stdout + log file, also used by Thunder) and return the wrapper logger."""

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)
        handler.close()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(stream_handler)
    root_logger.addHandler(file_handler)

    logger = logging.getLogger("histost_thunder")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    logger.propagate = True

    return logger


def safe_json_dump(obj: Any, path: Path) -> None:
    """Atomically write JSON (tmp file + replace)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp.{os.getpid()}")

    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())

    os.replace(tmp_path, path)

    try:
        dir_fd = os.open(str(path.parent), os.O_DIRECTORY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except Exception:
        pass


def flatten_dict(
    obj: Any,
    prefix: str = "",
    out: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if out is None:
        out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            new_prefix = f"{prefix}.{k}" if prefix else str(k)
            flatten_dict(v, new_prefix, out)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            new_prefix = f"{prefix}[{i}]"
            flatten_dict(v, new_prefix, out)
    else:
        out[prefix] = obj
    return out


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_symlink(src: Path, dst: Path) -> None:
    src = src.resolve()
    if dst.is_symlink():
        if dst.resolve() == src:
            return
        dst.unlink()
    elif dst.exists():
        raise FileExistsError(f"Cannot create symlink {dst} -> {src} because {dst} already exists and is not a symlink.")
    os.symlink(src, dst, target_is_directory=src.is_dir())


def resolve_aliases(values: Sequence[str], alias_map: Dict[str, List[str]]) -> List[str]:
    out: List[str] = []
    for v in values:
        if v in alias_map:
            out.extend(alias_map[v])
        else:
            out.append(v)
    dedup: List[str] = []
    seen = set()
    for v in out:
        if v not in seen:
            dedup.append(v)
            seen.add(v)
    return dedup


def get_training_id(training_num: int) -> str:
    return f"training_{training_num}"


def get_encoder_filename(training_num: int, checkpoint_num: int) -> str:
    return f"HEencoder_t{training_num}e{checkpoint_num}.pt"


def load_json_if_exists(path: Path) -> Optional[Dict[str, Any]]:
    if path is None or not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def compute_file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """SHA-256 of the exported encoder file."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def stable_json_dumps(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def sha256_json(obj: Any) -> str:
    return hashlib.sha256(stable_json_dumps(obj).encode("utf-8")).hexdigest()


def get_thunder_model_name(training_num: int, checkpoint_num: int, space: str) -> str:
    return f"histost_t{training_num}e{checkpoint_num}_{space}"


def run_done_marker_path(summary_dir: Path, dataset: str, task: str) -> Path:
    return summary_dir / "_done" / f"{dataset}__{task}.json"


def normalize_completed_payload(payload: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Record of a done marker ({"schema_version": 1, "record": {...}}), or None."""
    if not isinstance(payload, dict):
        return None
    record = payload.get("record")
    if isinstance(record, dict):
        return record
    return None


def completed_artifacts_present(record: Dict[str, Any]) -> bool:
    """True if the result folder (and outputs.json) of a completed record still exist."""
    copied_result_dir = record.get("copied_result_dir")
    runtime_result_dir = record.get("runtime_result_dir")

    has_result_dir = False
    for p in [copied_result_dir, runtime_result_dir]:
        if p and Path(p).exists():
            has_result_dir = True
            break

    if not has_result_dir:
        return False

    outputs_json = record.get("outputs_json")
    if outputs_json and not Path(outputs_json).is_file():
        return False

    return True


def build_run_config_payload(
    *,
    args: argparse.Namespace,
    encoder_sha256: str,
    model_name: str,
    dataset: str,
    task: str,
    loading_mode: str,
) -> Dict[str, Any]:
    """Identity of a benchmark result (excludes num_workers, batch sizes and machine-dependent settings)."""
    return {
        "resume_schema_version": RESUME_SCHEMA_VERSION,
        "training_num": int(args.training_num),
        "checkpoint_num": int(args.checkpoint_num),
        "space": str(args.space),
        "encoder_sha256": encoder_sha256,
        "model_name": model_name,
        "dataset": dataset,
        "task": task,
        "loading_mode": loading_mode,
    }


def load_completed_record_if_safe(
    *,
    summary_dir: Path,
    dataset: str,
    task: str,
) -> Optional[Dict[str, Any]]:
    """
    Previous successful record of (dataset, task) in this encoder eval folder, if its outputs still exist, else None.
    Checks the _done marker first, then summary.json.
    """

    def _is_skippable(record: Optional[Dict[str, Any]]) -> bool:
        return (
            isinstance(record, dict)
            and record.get("dataset") == dataset
            and record.get("task") == task
            and record.get("success") is True
            and completed_artifacts_present(record)
        )

    # Per-(dataset, task) done marker
    marker = normalize_completed_payload(
        load_json_if_exists(run_done_marker_path(summary_dir, dataset, task))
    )
    if _is_skippable(marker):
        return marker

    # Fallback: completed record in summary.json
    summary = load_json_if_exists(summary_dir / "summary.json")
    if isinstance(summary, dict):
        for r in reversed(summary.get("runs", [])):
            if _is_skippable(r):
                return r

    return None



# -----------------------------------------------------------------------------
# OpenCV preprocessing transform
# -----------------------------------------------------------------------------


class OpenCVPreprocessTransform:
    """
    Deterministic preprocessing matching the metadata stored in the exported
    H&E encoder bundle: resize -> center crop -> float[0,1] -> normalize.
    """

    def __init__(
        self,
        resize_size: int,
        crop_size: int,
        mean: torch.Tensor,
        std: torch.Tensor,
        interpolation_code: int,
    ) -> None:
        self.resize_size = int(resize_size)
        self.crop_size = int(crop_size)
        self.mean = mean.detach().cpu().view(3, 1, 1).float()
        self.std = std.detach().cpu().view(3, 1, 1).float()
        self.interpolation_code = int(interpolation_code)

    @classmethod
    def from_encoder(cls, encoder: torch.nn.Module) -> "OpenCVPreprocessTransform":
        resize_size = int(getattr(encoder, "resize_size", 0))
        crop_size = int(getattr(encoder, "crop_size", 0))
        mean = getattr(encoder, "mean")
        std = getattr(encoder, "std")
        interpolation_code = int(getattr(encoder, "interpolation_code", cv2.INTER_LINEAR))
        return cls(
            resize_size=resize_size,
            crop_size=crop_size,
            mean=mean,
            std=std,
            interpolation_code=interpolation_code,
        )

    def _to_rgb_uint8_numpy(self, image: Any) -> np.ndarray:
        if isinstance(image, Image.Image):
            arr = np.asarray(image.convert("RGB"))
        elif torch.is_tensor(image):
            x = image.detach().cpu()
            if x.ndim == 3 and x.shape[0] in {1, 3}:
                x = x.permute(1, 2, 0)
            arr = x.numpy()
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
        else:
            arr = np.asarray(image)
            if arr.ndim == 2:
                arr = np.stack([arr, arr, arr], axis=-1)
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).astype(np.uint8)

        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(f"Expected RGB image with shape [H,W,3], got {arr.shape}.")
        return np.ascontiguousarray(arr)

    def _resize_keep_aspect(self, arr: np.ndarray) -> np.ndarray:
        if self.resize_size <= 0:
            return arr
        h, w = arr.shape[:2]
        if min(h, w) == self.resize_size:
            return arr
        scale = float(self.resize_size) / float(min(h, w))
        new_h = max(1, int(round(h * scale)))
        new_w = max(1, int(round(w * scale)))
        return cv2.resize(arr, (new_w, new_h), interpolation=self.interpolation_code)

    def _center_crop(self, arr: np.ndarray) -> np.ndarray:
        if self.crop_size <= 0:
            return arr
        h, w = arr.shape[:2]
        crop_h = min(self.crop_size, h)
        crop_w = min(self.crop_size, w)
        top = max(0, (h - crop_h) // 2)
        left = max(0, (w - crop_w) // 2)
        arr = arr[top : top + crop_h, left : left + crop_w]

        if crop_h == self.crop_size and crop_w == self.crop_size:
            return arr

        pad_h = self.crop_size - crop_h
        pad_w = self.crop_size - crop_w
        return cv2.copyMakeBorder(
            arr,
            0,
            max(0, pad_h),
            0,
            max(0, pad_w),
            borderType=cv2.BORDER_REFLECT_101,
        )

    def __call__(self, image: Any) -> torch.Tensor:
        arr = self._to_rgb_uint8_numpy(image)
        arr = self._resize_keep_aspect(arr)
        arr = self._center_crop(arr)
        x = torch.from_numpy(arr).permute(2, 0, 1).contiguous().float().div_(255.0)
        x = (x - self.mean) / self.std
        return x


# -----------------------------------------------------------------------------
# Thunder custom model adapter for the exported HistoST encoder
# -----------------------------------------------------------------------------


class HistoSTThunderModel(PretrainedModel):
    def __init__(
        self,
        encoder_path: Path,
        model_name: str,
        space: str,
    ) -> None:
        super().__init__()
        if space not in {"he", "clip"}:
            raise ValueError(f"space must be one of ['he', 'clip'], got: {space}")

        self.encoder_path = str(encoder_path)
        self.name = model_name
        self.space = space
        self.vlm = False

        self.encoder = load_exported_he_encoder(self.encoder_path, device=torch.device("cpu")).eval()
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        self._state_keys = set(self.encoder.state_dict().keys())
        self._transform = OpenCVPreprocessTransform.from_encoder(self.encoder)

        self.project_patch = self.space == "clip"
        self.project_tokens = self.space == "clip"
        self.has_token_proj = any(k.startswith("img_tok_proj.") for k in self._state_keys)

        patch_dim, token_dim, supports_tokens = self._infer_dims()
        self.emb_dim = int(patch_dim)
        self.supports_tokens = bool(supports_tokens)
        if token_dim is not None:
            self.emb_dim_seg = int(token_dim)

        self.backbone = str(getattr(self.encoder, "backbone", "unknown"))

    def _dummy_input_size(self) -> int:
        crop_size = int(getattr(self.encoder, "crop_size", 0))
        resize_size = int(getattr(self.encoder, "resize_size", 0))
        return crop_size or resize_size or 224

    @torch.inference_mode()
    def _infer_dims(self) -> Tuple[int, Optional[int], bool]:
        size = self._dummy_input_size()
        x = torch.zeros(1, 3, size, size, dtype=torch.float32)

        patch = self.encoder.encode_image(x, project=self.project_patch)
        patch_dim = int(patch.shape[-1])

        token_dim: Optional[int] = None
        supports_tokens = False
        try:
            _, tok, _ = self.encoder.encode_with_tokens(
                x,
                project_img=self.project_patch,
                project_tokens=(self.project_tokens and self.has_token_proj),
            )
            if tok is not None:
                supports_tokens = True
                token_dim = int(tok.shape[-1])
        except Exception:
            supports_tokens = False
            token_dim = None

        return patch_dim, token_dim, supports_tokens

    def get_transform(self):
        return self._transform

    @staticmethod
    def _needs_input_grad(x: torch.Tensor) -> bool:
        """True if gradients w.r.t. the input are needed (PGD step of Thunder's adversarial_attack task)."""
        return torch.is_grad_enabled() and torch.is_tensor(x) and x.requires_grad

    def get_linear_probing_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        if self._needs_input_grad(x):
            # Adversarial attack (PGD): keep the graph from the input image
            with torch.inference_mode(False), torch.enable_grad():
                return self.encoder.encode_image(
                    x,
                    project=self.project_patch,
                ).float()

        # Other tasks: no gradients (no_grad rather than inference_mode, to return normal tensors)
        with torch.no_grad():
            return self.encoder.encode_image(
                x,
                project=self.project_patch,
            ).float()

    def get_segmentation_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        if not self.supports_tokens:
            raise RuntimeError(
                f"Encoder {self.name} does not expose token embeddings; "
                "Thunder segmentation cannot run."
            )
        if self.space == "clip" and not self.has_token_proj:
            raise RuntimeError(
                f"Encoder {self.name} was requested in clip space, but the exported "
                "encoder has no token projection head."
            )

        # No gradients (no_grad rather than inference_mode, to return normal tensors)
        with torch.no_grad():
            _, tok, _ = self.encoder.encode_with_tokens(
                x,
                project_img=self.project_patch,
                project_tokens=(self.project_tokens and self.has_token_proj),
            )

        if tok is None:
            raise RuntimeError(f"Token embeddings are unexpectedly None for {self.name}.")
        return tok.float()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.get_linear_probing_embeddings(x)


# -----------------------------------------------------------------------------
# Benchmark planning / validation
# -----------------------------------------------------------------------------


def dataset_allowed_tasks(dataset: str) -> set[str]:
    if dataset not in INSTALLED_THUNDER_DATASETS:
        installed = sorted(INSTALLED_THUNDER_DATASETS)
        extra_msg = ""
        if dataset in KNOWN_BUT_NOT_INSTALLED_DATASETS:
            extra_msg = (
                " This dataset is listed in this wrapper's known dataset list, "
                "but it is not available in the installed Thunder package."
            )
        raise ValueError(
            f"Unsupported Thunder dataset: {dataset}.{extra_msg} "
            f"Installed Thunder datasets are: {installed}"
        )

    thunder_cfg = load_thunder_dataset_config(dataset)
    compatible_tasks = thunder_cfg.get("compatible_tasks")
    if isinstance(compatible_tasks, list) and compatible_tasks:
        return {str(t) for t in compatible_tasks}
    if dataset in CLASSIFICATION_DATASETS:
        return set(CLASSIFICATION_TASKS)
    if dataset in SEGMENTATION_DATASETS:
        return set(SEGMENTATION_TASKS)
    raise ValueError(f"Unsupported Thunder dataset: {dataset}")


def default_loading_mode(task: str) -> str:
    if task in {"linear_probing", "segmentation"}:
        return "embedding_pre_loading"
    return "online_loading"


def thunder_recommended_overrides(
    dataset: str,
    task: str,
    num_workers: int,
) -> Dict[str, Any]:
    """
    Thunder overrides: adaptation.num_workers, and adaptation.epochs for the SegPath segmentation datasets
    (as in the Thunder documentation). All other hyperparameters keep the Thunder defaults.
    """
    overrides: Dict[str, Any] = {}

    if task in {"linear_probing", "segmentation", "transformation_invariance"}:
        overrides["adaptation.num_workers"] = num_workers

    if task == "segmentation":
        if dataset == "segpath_epithelial":
            overrides["adaptation.epochs"] = 9
        elif dataset == "segpath_lymphocytes":
            overrides["adaptation.epochs"] = 21

    return overrides


def compute_requested_runs(
    datasets: Sequence[str],
    tasks: Sequence[str],
) -> Tuple[List[Tuple[str, str]], List[Dict[str, Any]]]:
    runnable: List[Tuple[str, str]] = []
    skipped: List[Dict[str, Any]] = []

    for dataset in datasets:
        if dataset not in INSTALLED_THUNDER_DATASETS:
            installed = sorted(INSTALLED_THUNDER_DATASETS)
            extra_msg = ""
            if dataset in KNOWN_BUT_NOT_INSTALLED_DATASETS:
                extra_msg = (
                    " It appears in this wrapper's known dataset list, "
                    "but the installed Thunder package has no Hydra config for it."
                )
            raise ValueError(
                f"Unsupported or unavailable Thunder dataset: {dataset}.{extra_msg} "
                f"Installed Thunder datasets are: {installed}"
            )

        allowed = dataset_allowed_tasks(dataset)
        for task in tasks:
            if task in UNSUPPORTED_FOR_HE_ONLY:
                skipped.append(
                    {
                        "dataset": dataset,
                        "task": task,
                        "reason": "zero_shot_vlm requires a vision-language model; the HistoST encoder is H&E-only.",
                    }
                )
                continue
            if task not in allowed:
                skipped.append(
                    {
                        "dataset": dataset,
                        "task": task,
                        "reason": f"Task '{task}' is not compatible with dataset '{dataset}'.",
                    }
                )
                continue
            runnable.append((dataset, task))
    return runnable, skipped


# -----------------------------------------------------------------------------
# Filesystem layout
# -----------------------------------------------------------------------------


def build_paths(training_num: int, checkpoint_num: int, space: str) -> Dict[str, Path]:
    training_id = get_training_id(training_num)
    encoder_name = get_encoder_filename(training_num, checkpoint_num)

    encoder_path = Path(
        get_full_path(
            "train",
            "trainings",
            "trainings_HistoST",
            training_id,
            "trained_encoders",
            encoder_name,
            as_str=True,
        )
    )

    eval_encoder_dir = Path(
        get_full_path(
            "eval",
            "C_eval_thunder",
            f"HEencoder_t{training_num}e{checkpoint_num}_{space}",
            as_str=True,
        )
    )

    thunder_shared_datasets_dir = Path(
        get_full_path(
            "eval",
            "eval_datasets",
            "thunder_eval_data",
            "datasets",
            as_str=True,
        )
    )

    runtime_base = eval_encoder_dir / "_thunder_base"
    copied_results_dir = eval_encoder_dir / "results"
    summary_dir = eval_encoder_dir / "summary"

    return {
        "encoder_path": encoder_path,
        "eval_encoder_dir": eval_encoder_dir,
        "thunder_shared_datasets_dir": thunder_shared_datasets_dir,
        "runtime_base": runtime_base,
        "copied_results_dir": copied_results_dir,
        "summary_dir": summary_dir,
    }


def prepare_runtime_base(runtime_base: Path, thunder_shared_datasets_dir: Path) -> None:
    ensure_dir(runtime_base)
    if not thunder_shared_datasets_dir.is_dir():
        raise FileNotFoundError(
            f"Thunder datasets directory not found: {thunder_shared_datasets_dir}\n"
            "Expected the downloaded Thunder data under "
            "get_full_path('eval', 'eval_datasets', 'thunder_eval_data', 'datasets')."
        )
    ensure_symlink(thunder_shared_datasets_dir, runtime_base / "datasets")
    ensure_dir(runtime_base / "outputs")
    ensure_dir(runtime_base / "embeddings")
    ensure_dir(runtime_base / "pretrained_ckpts")


def copy_result_artifacts(
    runtime_base: Path,
    copied_results_dir: Path,
    dataset: str,
    model_name: str,
    task: str,
    adaptation_type: str = "frozen",
) -> Tuple[Path, Optional[Path]]:
    src_res_dir = (
        runtime_base
        / "outputs"
        / "res"
        / dataset
        / model_name
        / task
        / adaptation_type
    )
    dst_res_dir = copied_results_dir / dataset / task / adaptation_type

    if dst_res_dir.exists():
        shutil.rmtree(dst_res_dir)
    if src_res_dir.exists():
        shutil.copytree(src_res_dir, dst_res_dir)
    else:
        ensure_dir(dst_res_dir)

    outputs_json = dst_res_dir / "outputs.json"
    return dst_res_dir, (outputs_json if outputs_json.is_file() else None)


# -----------------------------------------------------------------------------
# Benchmark execution
# -----------------------------------------------------------------------------


@dataclass
class RunRecord:
    dataset: str
    task: str
    loading_mode: str
    model_name: str
    success: bool
    message: str
    runtime_result_dir: Optional[str] = None
    copied_result_dir: Optional[str] = None
    outputs_json: Optional[str] = None
    metrics_raw: Optional[Dict[str, Any]] = None
    metrics_flat: Optional[Dict[str, Any]] = None
    overrides: Optional[Dict[str, Any]] = None
    run_config_hash: Optional[str] = None
    resumed_from_previous_success: bool = False
    encoder_sha256: Optional[str] = None


def run_record_to_dict(r: RunRecord, *, include_metrics: bool = False) -> Dict[str, Any]:
    """Serialize a RunRecord (metrics_raw / metrics_flat only if include_metrics)."""
    d: Dict[str, Any] = {
        "dataset": r.dataset,
        "task": r.task,
        "loading_mode": r.loading_mode,
        "model_name": r.model_name,
        "success": r.success,
        "message": r.message,
        "runtime_result_dir": r.runtime_result_dir,
        "copied_result_dir": r.copied_result_dir,
        "outputs_json": r.outputs_json,
        "overrides": r.overrides,
        "run_config_hash": r.run_config_hash,
        "resumed_from_previous_success": r.resumed_from_previous_success,
        "encoder_sha256": r.encoder_sha256,
    }
    if include_metrics:
        d["metrics_raw"] = r.metrics_raw
        d["metrics_flat"] = r.metrics_flat
    return d


def save_done_marker(summary_dir: Path, record: RunRecord) -> None:
    """Write the done marker of a successful run (record without metrics), used for resume."""
    if not record.success:
        return

    safe_json_dump(
        {
            "schema_version": RESUME_SCHEMA_VERSION,
            "record": run_record_to_dict(record),
        },
        run_done_marker_path(summary_dir, record.dataset, record.task),
    )


def run_one_benchmark(
    *,
    model: HistoSTThunderModel,
    runtime_base: Path,
    copied_results_dir: Path,
    space: str,
    dataset: str,
    task: str,
    loading_mode: str,
    overrides: Dict[str, Any],
    run_config_hash: str,
    encoder_sha256: str,
    logger: logging.Logger,
) -> RunRecord:
    model.eval()
    model_name = model.name

    runtime_res_dir = (
        runtime_base
        / "outputs"
        / "res"
        / dataset
        / model_name
        / task
        / "frozen"
    )

    if dataset in SEGMENTATION_DATASETS:
        if not model.supports_tokens:
            raise RuntimeError("Segmentation dataset requested, but the encoder does not expose token embeddings.")
        if space == "clip" and not model.has_token_proj:
            raise RuntimeError("Segmentation dataset requested in clip space, but the encoder has no token projection head.")

    logger.info(
        "Running Thunder | dataset=%s | task=%s | model=%s | loading_mode=%s",
        dataset,
        task,
        model_name,
        loading_mode,
    )
    if overrides:
        logger.info("Thunder overrides: %s", json.dumps(overrides, sort_keys=True))

    # -- Resume: embeddings cache without a done marker ----------
    # Left by an interrupted run (possibly truncated HDF5): deleted so that Thunder recomputes it
    if task == "pre_computing_embeddings":
        stale_emb_dir = runtime_base / "embeddings" / dataset / model_name
        if stale_emb_dir.exists():
            logger.warning(
                "Purging un-certified embeddings cache before recompute "
                "(no success marker -> a prior attempt was incomplete): %s",
                stale_emb_dir,
            )
            shutil.rmtree(stale_emb_dir, ignore_errors=True)

    old_env = os.environ.get("THUNDER_BASE_DATA_FOLDER")
    try:
        os.environ["THUNDER_BASE_DATA_FOLDER"] = str(runtime_base)
        os.environ.setdefault("WANDB_MODE", "offline")
        os.environ.setdefault("WANDB_SILENT", "true")

        thunder_benchmark(
            model=model,
            dataset=dataset,
            task=task,
            loading_mode=loading_mode,
            lora=False,
            ckpt_save_all=False,
            online_wandb=False,
            recomp_embs=False,
            retrain_model=False,
            **overrides,
        )
    finally:
        if old_env is None:
            os.environ.pop("THUNDER_BASE_DATA_FOLDER", None)
        else:
            os.environ["THUNDER_BASE_DATA_FOLDER"] = old_env

    copied_res_dir, outputs_json_path = copy_result_artifacts(
        runtime_base=runtime_base,
        copied_results_dir=copied_results_dir,
        dataset=dataset,
        model_name=model_name,
        task=task,
        adaptation_type="frozen",
    )

    metrics_raw = load_json_if_exists(outputs_json_path)
    metrics_flat = flatten_dict(metrics_raw) if metrics_raw is not None else None

    logger.info("DONE   | dataset=%s | task=%s", dataset, task)

    return RunRecord(
        dataset=dataset,
        task=task,
        loading_mode=loading_mode,
        model_name=model_name,
        success=True,
        message="ok",
        runtime_result_dir=str(runtime_res_dir),
        copied_result_dir=str(copied_res_dir),
        outputs_json=str(outputs_json_path) if outputs_json_path else None,
        metrics_raw=metrics_raw,
        metrics_flat=metrics_flat,
        overrides=overrides,
        run_config_hash=run_config_hash,
        encoder_sha256=encoder_sha256,
    )


def resolve_outputs_json(record: RunRecord, summary_dir: Path) -> Optional[Path]:
    """
    Path of a run's Thunder outputs.json: the record's outputs_json, else <copied_result_dir>/outputs.json,
    else <eval_encoder_dir>/results/<ds>/<task>/frozen/outputs.json. None if none exists.
    """
    candidates: List[Path] = []
    if record.outputs_json:
        candidates.append(Path(record.outputs_json))
    if record.copied_result_dir:
        candidates.append(Path(record.copied_result_dir) / "outputs.json")
    candidates.append(
        summary_dir.parent / "results" / record.dataset / record.task / "frozen" / "outputs.json"
    )
    for c in candidates:
        if c.is_file():
            return c
    return None


def save_status_summary(
    summary_dir: Path,
    records: Sequence[RunRecord],
    skipped: Sequence[Dict[str, Any]],
) -> None:
    """Write summary.json (status of each run, without metrics) and skipped.json."""
    ensure_dir(summary_dir)

    summary_json = {
        "runs": [run_record_to_dict(r) for r in records],
        "skipped": list(skipped),
    }
    safe_json_dump(summary_json, summary_dir / "summary.json")
    safe_json_dump({"skipped": list(skipped)}, summary_dir / "skipped.json")


def save_flat_csv(summary_dir: Path, records: Sequence[RunRecord]) -> None:
    """
    Write summary_flat.csv (one row per run, with its metrics) once at the end of the run.
    Metrics come from memory, else from the run's outputs.json.
    """
    ensure_dir(summary_dir)

    base_cols = [
        "dataset",
        "task",
        "loading_mode",
        "model_name",
        "success",
        "message",
        "runtime_result_dir",
        "copied_result_dir",
        "outputs_json",
        "run_config_hash",
        "resumed_from_previous_success",
    ]

    # In-memory metrics, else the pair's outputs.json
    flats: List[Dict[str, Any]] = []
    for r in records:
        if r.metrics_flat:
            flats.append(dict(r.metrics_flat))
        else:
            raw = load_json_if_exists(resolve_outputs_json(r, summary_dir))
            flats.append(flatten_dict(raw) if raw is not None else {})

    metric_keys: set[str] = set()
    for flat in flats:
        metric_keys.update(flat.keys())

    fieldnames = base_cols + sorted(metric_keys)

    with (summary_dir / "summary_flat.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r, flat in zip(records, flats):
            row = {
                "dataset": r.dataset,
                "task": r.task,
                "loading_mode": r.loading_mode,
                "model_name": r.model_name,
                "success": r.success,
                "message": r.message,
                "runtime_result_dir": r.runtime_result_dir,
                "copied_result_dir": r.copied_result_dir,
                "outputs_json": r.outputs_json,
                "run_config_hash": r.run_config_hash,
                "resumed_from_previous_success": r.resumed_from_previous_success,
            }
            if flat:
                row.update(flat)
            writer.writerow(row)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def main(args) -> None:
    
    torch.set_float32_matmul_precision("high")

    datasets = resolve_aliases(args.datasets, DATASET_ALIASES)
    tasks = resolve_aliases(args.tasks, TASK_ALIASES)

    paths = build_paths(
        training_num=args.training_num,
        checkpoint_num=args.checkpoint_num,
        space=args.space,
    )
    ensure_dir(paths["eval_encoder_dir"])
    logger = setup_logger(paths["eval_encoder_dir"] / "run.log")

    logger.info(
        "Installed Thunder datasets discovered from package configs: %s",
        ", ".join(sorted(INSTALLED_THUNDER_DATASETS)),
    )
    if KNOWN_BUT_NOT_INSTALLED_DATASETS:
        logger.warning(
            "Known datasets excluded because this Thunder installation has no config for them: %s",
            ", ".join(sorted(KNOWN_BUT_NOT_INSTALLED_DATASETS)),
        )

    if not paths["encoder_path"].is_file():
        raise FileNotFoundError(f"Exported H&E encoder not found: {paths['encoder_path']}")

    encoder_sha256 = compute_file_sha256(paths["encoder_path"])

    prepare_runtime_base(
        runtime_base=paths["runtime_base"],
        thunder_shared_datasets_dir=paths["thunder_shared_datasets_dir"],
    )

    requested_runs, skipped = compute_requested_runs(datasets, tasks)

    total_runs = len(requested_runs)
    logger.info(
        "Thunder run plan | runnable_pairs=%d | skipped_incompatible=%d",
        total_runs,
        len(skipped),
    )

    manifest = {
        "resume_schema_version": RESUME_SCHEMA_VERSION,
        "repo_root": str(project_root),
        "training_num": args.training_num,
        "checkpoint_num": args.checkpoint_num,
        "space": args.space,
        "encoder_path": str(paths["encoder_path"]),
        "encoder_sha256": encoder_sha256,
        "eval_encoder_dir": str(paths["eval_encoder_dir"]),
        "thunder_shared_datasets_dir": str(paths["thunder_shared_datasets_dir"]),
        "runtime_base": str(paths["runtime_base"]),
        "datasets_requested": datasets,
        "tasks_requested": tasks,
        "installed_thunder_datasets": sorted(INSTALLED_THUNDER_DATASETS),
        "classification_datasets_from_installed_thunder": sorted(CLASSIFICATION_DATASETS),
        "segmentation_datasets_from_installed_thunder": sorted(SEGMENTATION_DATASETS),
        "known_but_not_installed_datasets": sorted(KNOWN_BUT_NOT_INSTALLED_DATASETS),
        "requested_runs": [{"dataset": d, "task": t} for d, t in requested_runs],
        "skipped_before_run": skipped,
        "num_workers": int(args.num_workers),
        "resume_policy": (
            "A dataset/task pair is skipped if this eval folder holds a successful run "
            "of it with its outputs present (per-pair marker "
            "summary/_done/<dataset>__<task>.json, else summary.json)."
        ),
        "thunder_hyperparameter_policy": (
            "Thunder defaults are kept. Only adaptation.num_workers is overridden "
            "as a runtime setting, plus adaptation.epochs=9 for segpath_epithelial "
            "segmentation and adaptation.epochs=21 for segpath_lymphocytes segmentation."
        ),
        "thunder_required_overrides": {
            f"{d}__{t}": thunder_recommended_overrides(
                dataset=d,
                task=t,
                num_workers=args.num_workers,
            )
            for d, t in requested_runs
        },
    }
    safe_json_dump(manifest, paths["eval_encoder_dir"] / "manifest.json")

    if not requested_runs:
        logger.warning("No runnable (dataset, task) pairs remain after compatibility checks.")
        save_status_summary(paths["summary_dir"], [], skipped)
        save_flat_csv(paths["summary_dir"], [])
        return

    records: List[RunRecord] = []
    model: Optional[HistoSTThunderModel] = None

    for run_idx, (dataset, task) in enumerate(requested_runs, start=1):

        logger.info("\n\n\n")
        logger.info("%s", "=" * 60)
        logger.info(
            "START  [%d/%d] | dataset=%s | task=%s",
            run_idx,
            total_runs,
            dataset,
            task,
        )

        loading_mode = default_loading_mode(task)
        overrides = thunder_recommended_overrides(
            dataset=dataset,
            task=task,
            num_workers=args.num_workers,
        )

        model_name = get_thunder_model_name(
            training_num=args.training_num,
            checkpoint_num=args.checkpoint_num,
            space=args.space,
        )

        run_config_payload = build_run_config_payload(
            args=args,
            encoder_sha256=encoder_sha256,
            model_name=model_name,
            dataset=dataset,
            task=task,
            loading_mode=loading_mode,
        )
        run_config_hash = sha256_json(run_config_payload)

        previous_success = load_completed_record_if_safe(
            summary_dir=paths["summary_dir"],
            dataset=dataset,
            task=task,
        )

        if previous_success is not None:
            logger.info(
                "SKIP   [%d/%d] | dataset=%s | task=%s | previous successful dataset/task found",
                run_idx,
                total_runs,
                dataset,
                task,
            )
            # Resumed record (no metrics; the done marker is not rewritten)
            record = RunRecord(
                dataset=dataset,
                task=task,
                loading_mode=previous_success.get("loading_mode", loading_mode),
                model_name=previous_success.get("model_name", model_name),
                success=True,
                message="skipped: previous successful dataset/task found",
                runtime_result_dir=previous_success.get("runtime_result_dir"),
                copied_result_dir=previous_success.get("copied_result_dir"),
                outputs_json=previous_success.get("outputs_json"),
                overrides=previous_success.get("overrides", overrides),
                run_config_hash=run_config_hash,
                resumed_from_previous_success=True,
                encoder_sha256=previous_success.get("encoder_sha256"),
            )
            records.append(record)
            logger.info(
                "PROGRESS | done=%d/%d | remaining=%d",
                len(records),
                total_runs,
                total_runs - len(records),
            )
            continue

        if model is None:
            logger.info("Loading HistoST Thunder model once | model=%s", model_name)
            model = HistoSTThunderModel(
                encoder_path=paths["encoder_path"],
                model_name=model_name,
                space=args.space,
            )

        try:
            record = run_one_benchmark(
                model=model,
                runtime_base=paths["runtime_base"],
                copied_results_dir=paths["copied_results_dir"],
                space=args.space,
                dataset=dataset,
                task=task,
                loading_mode=loading_mode,
                overrides=overrides,
                run_config_hash=run_config_hash,
                encoder_sha256=encoder_sha256,
                logger=logger,
            )
        except Exception as exc:
            runtime_res_dir = (
                paths["runtime_base"]
                / "outputs"
                / "res"
                / dataset
                / model_name
                / task
                / "frozen"
            )
            logger.error(
                "FAILED [%d/%d] | dataset=%s | task=%s | aborting immediately | %s: %s",
                run_idx,
                total_runs,
                dataset,
                task,
                type(exc).__name__,
                exc,
            )
            records.append(
                RunRecord(
                    dataset=dataset,
                    task=task,
                    loading_mode=loading_mode,
                    model_name=model_name,
                    success=False,
                    message=f"{type(exc).__name__}: {exc}",
                    runtime_result_dir=str(runtime_res_dir),
                    overrides=overrides,
                    run_config_hash=run_config_hash,
                    encoder_sha256=encoder_sha256,
                )
            )
            # Save the status before raising (the metrics CSV is built at the end)
            save_status_summary(paths["summary_dir"], records, skipped)
            raise

        if record.success:
            save_done_marker(paths["summary_dir"], record)

        records.append(record)
        # Status snapshot after each pair (the metrics CSV is built at the end)
        save_status_summary(paths["summary_dir"], records, skipped)
        logger.info(
            "PROGRESS | done=%d/%d | remaining=%d",
            len(records),
            total_runs,
            total_runs - len(records),
        )

    save_status_summary(paths["summary_dir"], records, skipped)
    save_flat_csv(paths["summary_dir"], records)

    n_ok = sum(r.success for r in records)
    logger.info(
        "Thunder evaluation finished | success=%d | failed=%d | skipped=%d",
        n_ok,
        0,
        len(skipped),
    )




if __name__ == "__main__":
    
    p = argparse.ArgumentParser(description=("Run Thunder benchmark for an exported HistoST H&E encoder"))
    
    p.add_argument("--training_num", type=int, 
                   required=True,
                   help="Num from HistoST training_id (e.g. '46' for training_id=='training_46')")
    
    p.add_argument("--checkpoint_num", type=int, 
                   required=True,
                   help="Checkpoint number to load (e.g. '13' for 'checkpoint_epoch_13.pth')")

    p.add_argument("--space", type=str,
                   required=True,
                   choices=["he", "clip"],
                   help="Embedding space to evaluate: 'he' for pre-projection, 'clip' for projected CLIP space.")
    
    p.add_argument("--datasets",
                   nargs="+",
                   required=True,
                   help="Thunder datasets, or aliases: classification / segmentation / all.")
    
    p.add_argument("--tasks",
                   nargs="+",
                   required=True,
                   help="Thunder tasks, or aliases: classification_suite / segmentation_suite / all_non_vlm. "
                        "Tasks are: pre_computing_embeddings, knn, linear_probing, simple_shot, image_retrieval, alignment_scoring, transformation_invariance, adversarial_attack, segmentation, zero_shot_vlm.")
    
    p.add_argument("--num_workers", type=int,
                   default=7,
                   help=("Number of dataloader workers to pass to Thunder where applicable."))
    
    # To recompute a (dataset, task) pair, delete its summary/_done/<dataset>__<task>.json marker (and its result folder)

    args = p.parse_args()
    main(args)