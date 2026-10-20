"""Experiment class: sets up and runs the CLIP training aligning H&E images and spatial transcriptomics."""

import logging
# Get the logger for this module
logger = logging.getLogger(__name__)

from typing import Optional, Callable, Tuple, Any, Dict
import wandb
import logging
import torch
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import Sampler
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
import json
import numpy as np
import cv2
import math
import random
import pandas as pd
from pathlib import Path
import h5py


from src.utils.constants import get_full_path, TEACHER_CENTER_MEANS_SCHEMA_VERSION
from src.dataset.histost_dataset import HistoStDataset
from src.models.timmEncoders_histo import get_timm_he_encoder, ImageEncoderTimm
from src.models.clip import CLIPModel
from src.training.trainer import Trainer, collapse_wandb_key
from src.models.lora import insert_lora
from src.utils.models_settings import BACKBONE_TRANSFORM
from src.utils.utils import reconcile_state_dict, loading_checkpoint_cpu
from src.models.mlp_head_histo import HEEncoderWithMLP
from src.training.samplers import _normalize_batch_slide_mode, DistributedSingleSlideUniformGlobalBatchSampler, DistributedSingleSlideBatchSampler, DistributedWeightedGlobalBatchUniqueSampler, DistributedAlternatingSampler, DistributedBatchAlignedSampler, DistributedStridedEvalSampler
from src.utils.compatibility_histost_pretrained import (assert_pretrained_state_dict_compatible, check_pretrained_histost_config_compatibility, check_cross_run_resume_compatibility)
from src.utils.default_histost_config import cfg_get





class Experiment:


    def __init__(self, config: dict, wandb_run_id: str, local_rank: int, world_size: int, rank: int, distributed: bool, cross_run_resume: bool = False, allow_schedule_mismatch: bool = False):
        """
        Initializes the Experiment from the run config (resolves the dataset, teacher-cache and scConcept paths).
        cross_run_resume: full-state resume from a checkpoint of another run (--resume_state_from_checkpoint).
        """
        
        # Update config with full paths
        dataset_path = get_full_path('train', 'training_datasets', config["data"]["dataset_id"], config["data"]["training_dataset_id"], as_str=True, scratch=True)
        foundation_models_folder = get_full_path('train', 'pretrained_foundation_models', as_str=True)
        
        config["data"]["dataset_path"] = dataset_path
        config["model"]["foundation_models_folder"] = foundation_models_folder
        logger.info(f"Using dataset path: {dataset_path}")
        logger.info(f"Using pretrained backbone folder: {foundation_models_folder}")

        # --- Precomputed Novae ST-graph teacher: resolve the st_feat cache + its manifest ---
        # st_feat cache written by src/dataset/precompute_novae_teacher.py
        novae_training_id = cfg_get(config, "model", "st", "novae_training_id")
        if not novae_training_id:
            raise ValueError("model.st.novae_training_id is required (selects the precomputed Novae-teacher st_feat cache).")
        novae_teacher_cache_dir = get_full_path(
            "train", "training_datasets", config["data"]["dataset_id"], config["data"]["training_dataset_id"],
            "novae_teacher_cache", novae_training_id, as_str=True, scratch=True,
        )
        teacher_manifest_path = os.path.join(novae_teacher_cache_dir, "manifest.json")
        if not os.path.isfile(teacher_manifest_path):
            raise FileNotFoundError(
                f"Novae-teacher cache manifest not found: {teacher_manifest_path}. "
                "Run src/dataset/precompute_novae_teacher.py first (novae env)."
            )
        with open(teacher_manifest_path) as _mf:
            teacher_manifest = json.load(_mf)
        if teacher_manifest.get("novae_training_id") != novae_training_id:
            raise ValueError(
                f"Novae-teacher cache manifest novae_training_id={teacher_manifest.get('novae_training_id')!r} "
                f"!= config model.st.novae_training_id={novae_training_id!r}."
            )
        config["model"]["st"]["novae_teacher_cache_dir"] = novae_teacher_cache_dir
        config["model"]["st"]["novae_teacher_fingerprint"] = teacher_manifest.get("fingerprint")
        config["model"]["st"]["st_feature_dim"] = int(teacher_manifest["dim"])
        # Teacher checkpoint that produced the cache (recorded only)
        config["model"]["st"]["novae_pretrained_ckpt"] = teacher_manifest.get("novae_ckpt_path")
        # Default scconcept_training_id: the teacher's
        if not cfg_get(config, "model", "st", "scconcept_training_id"):
            config["model"]["st"]["scconcept_training_id"] = teacher_manifest.get("scconcept_training_id")
        # scconcept_training_id must match the teacher's
        _cfg_scc = cfg_get(config, "model", "st", "scconcept_training_id")
        if _cfg_scc and teacher_manifest.get("scconcept_training_id") and _cfg_scc != teacher_manifest.get("scconcept_training_id"):
            raise ValueError(
                f"model.st.scconcept_training_id={_cfg_scc!r} != the Novae teacher's scConcept "
                f"({teacher_manifest.get('scconcept_training_id')!r}). The token teacher must match the "
                "scConcept model the Novae node features came from."
            )
        logger.info(
            f"Novae ST-graph teacher cache: {novae_teacher_cache_dir} (dim={teacher_manifest['dim']}, "
            f"fingerprint={teacher_manifest.get('fingerprint')}, scConcept={_cfg_scc})"
        )

        # Teacher-cache manifest (checked against the teacher-centering means in get_model)
        self._novae_teacher_manifest = teacher_manifest

        # --- Teacher-centering means ---
        # Required if center_teacher_target and a soft-target loss are on (written by src/dataset/precompute_teacher_mean.py)
        _center = bool(cfg_get(config, "model", "clip", "center_teacher_target"))
        _patch_soft = (str(cfg_get(config, "model", "clip", "loss_type")).lower() in ("soft_targets", "soft_targets_st")
                       and float(cfg_get(config, "model", "clip", "lambda_patch")) > 0.0)
        _token_soft = (str(cfg_get(config, "model", "clip", "token_loss_type")).lower() in ("soft_targets", "soft_targets_st")
                       and float(cfg_get(config, "model", "clip", "lambda_token")) > 0.0)
        config["model"]["st"]["teacher_center_means_path"] = None
        if _center and (_patch_soft or _token_soft):
            _means_path = os.path.join(novae_teacher_cache_dir, "teacher_center_means.h5")
            if not os.path.isfile(_means_path):
                raise FileNotFoundError(
                    f"model.clip.center_teacher_target=True but the teacher-centering means artifact was not "
                    f"found: {_means_path}. Run src/dataset/precompute_teacher_mean.py first."
                )
            config["model"]["st"]["teacher_center_means_path"] = _means_path
            logger.info(f"Teacher-centering means artifact: {_means_path}")

        # model.st.novae_unfreeze_st_epoch is not supported (the ST teacher is precomputed)
        _legacy_unfreeze = (config.get("model", {}).get("st", {}) or {}).get("novae_unfreeze_st_epoch")
        if _legacy_unfreeze is not None and int(_legacy_unfreeze) >= 0:
            raise ValueError(
                "ST-teacher unfreeze is unsupported with the precomputed Novae teacher "
                f"(model.st.novae_unfreeze_st_epoch={_legacy_unfreeze}). Remove that key."
            )

        # --- scConcept token teacher (optional; enabled when lambda_token > 0) ---
        use_scconcept_embeddings = float(cfg_get(config, "model", "clip", "lambda_token")) > 0.0
        if use_scconcept_embeddings:
            scconcept_embeddings_dir = cfg_get(config, "model", "st", "scconcept_embeddings_dir")
            if scconcept_embeddings_dir:
                scconcept_embeddings_dir = str(scconcept_embeddings_dir)
            else:
                scconcept_training_id = cfg_get(config, "model", "st", "scconcept_training_id")
                if not scconcept_training_id:
                    raise ValueError(
                        "Token loss is enabled (model.clip.lambda_token > 0), but neither "
                        "model.st.scconcept_embeddings_dir nor model.st.scconcept_training_id is set."
                    )
                scconcept_embeddings_dir = get_full_path("train", "trainings", "trainings_scConcept", scconcept_training_id, "embeddings", as_str=True)
            if not os.path.isdir(scconcept_embeddings_dir):
                raise FileNotFoundError(f"Resolved scConcept embeddings directory does not exist: {scconcept_embeddings_dir}")
            config["model"]["st"]["scconcept_embeddings_dir"] = scconcept_embeddings_dir
            logger.info(f"Using scConcept embeddings from: {scconcept_embeddings_dir}")
        else:
            config["model"]["st"]["scconcept_embeddings_dir"] = None
            logger.info("Token loss disabled: no scConcept embeddings directory will be used.")

        self.config = config
        self.wandb_run_id = wandb_run_id
        self.local_rank = local_rank
        self.world_size = world_size
        self.rank = rank
        self.distributed = distributed
        self.cross_run_resume = cross_run_resume
        self.allow_schedule_mismatch = allow_schedule_mismatch
        self._source_training_id = None  # set for a cross-run resume (history seeding)



    def _seed_cross_run_history(self, start_epoch: int) -> None:
        """
        Cross-run resume: copy the source run's epochs [0, start_epoch) into this run's
        backup_training_metrics.jsonl and W&B run (skipped if the JSONL already exists; failures only warn).
        """
        try:
            if not self._source_training_id:
                logger.warning("Cross-run resume: no source training_id resolved; skipping history seeding.")
                return

            target_jsonl = os.path.join(self.config["logging"]["log_dir"], "backup_training_metrics.jsonl")
            if os.path.exists(target_jsonl):
                logger.info(
                    f"Cross-run resume: target metrics JSONL already exists ({target_jsonl}); "
                    "skipping history seeding (idempotent re-launch)."
                )
                return

            source_jsonl = get_full_path(
                "train", "trainings", "trainings_HistoST", self._source_training_id,
                "training", "log", "backup_training_metrics.jsonl", as_str=True,
            )
            if not os.path.isfile(source_jsonl):
                logger.warning(
                    f"Cross-run resume: source metrics JSONL not found ({source_jsonl}); this run's "
                    f"report/W&B will start at epoch {start_epoch}. Training continues."
                )
                return

            # Read the source rows for epochs [0, start_epoch) (one JSON dict per line).
            by_epoch = {}
            with open(source_jsonl, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    ep = row.get("epoch", None)
                    if ep is None:
                        continue
                    ep = int(ep)
                    if ep < int(start_epoch):
                        by_epoch[ep] = row  # last write per epoch wins

            if not by_epoch:
                logger.warning(
                    f"Cross-run resume: no source rows with epoch < {start_epoch} in {source_jsonl}; "
                    "nothing to seed."
                )
                return

            ordered = [by_epoch[e] for e in sorted(by_epoch)]

            # 1) JSONL seed (creates the file; the trainer appends start_epoch.. afterwards).
            os.makedirs(os.path.dirname(target_jsonl), exist_ok=True)
            with open(target_jsonl, "w", encoding="utf-8") as f:
                for row in ordered:
                    f.write(json.dumps(row) + "\n")
            logger.info(
                f"Cross-run resume: seeded {len(ordered)} epoch row(s) [0, {start_epoch}) into "
                f"{target_jsonl} from source '{self._source_training_id}'."
            )

            # 2) W&B replay: same collapse + non-None filter as the live log, keyed by step=epoch.
            if wandb.run is not None:
                for row in ordered:
                    flat = {collapse_wandb_key(k): v for k, v in row.items() if v is not None}
                    wandb.log(flat, step=int(row["epoch"]))
                logger.info(
                    f"Cross-run resume: replayed {len(ordered)} epoch row(s) into the fresh W&B run "
                    f"(steps 0..{start_epoch - 1})."
                )
        except Exception as exc:
            logger.warning(f"Cross-run resume: history seeding failed ({exc}); training continues.")



    def get_transforms_he(
        self, transform_settings: dict, input_shape: int = 256
    ) -> Tuple[Callable, Callable]:
        """Get Transformations (Albumentation Transformations). Return both training and validation transformations.

        Args:
            transform_settings (dict): dictionay with the transformation settings.
            input_shape (int, optional): Input shape of the images to used. Defaults to 256.

        Returns:
            Tuple[Callable, Callable]: Train Transformations, Validation Transformations

        """

        transform_list = []
        transform_settings = {k.lower(): v for k, v in transform_settings.items()}
        backbone = self.config["model"]["he"]["backbone_img_encoder"].lower()
        backbone_transform = BACKBONE_TRANSFORM[backbone]

        if "RandomRotate90".lower() in transform_settings:
            p = transform_settings["randomrotate90"]["p"]
            if p > 0 and p <= 1:
                transform_list.append(A.RandomRotate90(p=p))
        if "HorizontalFlip".lower() in transform_settings.keys():
            p = transform_settings["horizontalflip"]["p"]
            if p > 0 and p <= 1:
                transform_list.append(A.HorizontalFlip(p=p))
        if "VerticalFlip".lower() in transform_settings:
            p = transform_settings["verticalflip"]["p"]
            if p > 0 and p <= 1:
                transform_list.append(A.VerticalFlip(p=p))
        if "Downscale".lower() in transform_settings:
            p = transform_settings["downscale"]["p"]
            scale_range = transform_settings["downscale"]["scale_range"]
            if p > 0 and p <= 1:
                transform_list.append(
                    A.Downscale(p=p, scale_range=scale_range)
                )
        if "Blur".lower() in transform_settings:
            p = transform_settings["blur"]["p"]
            blur_limit = transform_settings["blur"]["blur_limit"]
            if p > 0 and p <= 1:
                transform_list.append(A.Blur(p=p, blur_limit=blur_limit))
        if "GaussNoise".lower() in transform_settings:
            p = transform_settings["gaussnoise"]["p"]
            if p > 0 and p <= 1:
                transform_list.append(A.GaussNoise(p=p))
        if "RandomBrightnessContrast".lower() in transform_settings:
            p = transform_settings["randombrightnesscontrast"]["p"]
            brightness_limit = transform_settings["randombrightnesscontrast"]["brightness_limit"]
            contrast_limit = transform_settings["randombrightnesscontrast"]["contrast_limit"]
            if p > 0 and p <= 1:
                transform_list.append(
                    A.RandomBrightnessContrast(
                        p=p, 
                        brightness_limit=brightness_limit, 
                        contrast_limit=contrast_limit)
                )
        if "Superpixels".lower() in transform_settings:
            p = transform_settings["superpixels"]["p"]
            if p > 0 and p <= 1:
                transform_list.append(
                    A.Superpixels(
                        p=p,
                        p_replace=0.1,
                        n_segments=200,
                        max_size=int(input_shape / 2),
                    )
                )
        if "ZoomBlur".lower() in transform_settings:
            p = transform_settings["zoomblur"]["p"]
            if p > 0 and p <= 1:
                transform_list.append(A.ZoomBlur(p=p, max_factor=1.05))
        # No RandomSizedCrop / ElasticTransform: cells must stay aligned with the precomputed teacher features

        if backbone in ["virchow2", "uni2h", "hoptimus0", "hoptimus1"]:

            resize_size = backbone_transform["resize_size"]
            crop_size = backbone_transform["crop_size"]
            interp = backbone_transform["interpolation"]
            mean, std = backbone_transform["mean"], backbone_transform["std"]

            transform_list.append(A.Resize(height=resize_size, width=resize_size, interpolation=interp, mask_interpolation=cv2.INTER_NEAREST))
            transform_list.append(A.CenterCrop(height=crop_size, width=crop_size))
            transform_list.append(A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0))
            transform_list.append(ToTensorV2())

            train_transform = A.Compose(transform_list, additional_targets={"mask": "mask"}, keypoint_params=A.KeypointParams(format="xy", remove_invisible=True, label_fields=["keypoint_ids"]))
            val_transform = A.Compose([
                A.Resize(height=resize_size, width=resize_size, interpolation=interp, mask_interpolation=cv2.INTER_NEAREST),
                A.CenterCrop(height=crop_size, width=crop_size),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2()],
                additional_targets={"mask": "mask"},
                keypoint_params=A.KeypointParams(format="xy", remove_invisible=True, label_fields=["keypoint_ids"])
            )
        
        else:
            raise ValueError(f"Invalid backbone_img_encoder: {backbone}")
        
        return train_transform, val_transform


        

    def get_dataset(self, fold: str, he_transform: Callable, mode: str) -> HistoStDataset:
        """
        Instantiates the HistoStDataset for the given fold (train/valid), with the precomputed st_feat cache as ST-graph teacher.
        """
        transform_settings = cfg_get(self.config, "he_transformations")
        if "hed_augmentation" in transform_settings:
            hed_augm_p = transform_settings["hed_augmentation"]["p"]
            hed_augm_sigma = transform_settings["hed_augmentation"]["sigma"]
        else:
            hed_augm_p = None
            hed_augm_sigma = None

        return HistoStDataset(
            dataset_path=self.config["data"]["dataset_path"],
            fold=fold,
            training_id=self.config["logging"]["training_id_histost"],
            transforms=he_transform,
            n_min_cells=cfg_get(self.config, "data_select", "n_min_cells"),
            mode=mode,
            hed_augm_p=hed_augm_p,
            hed_augm_sigma=hed_augm_sigma,
            cached_dataset=cfg_get(self.config, "data", "cached_dataset"),
            use_scconcept_embeddings=float(cfg_get(self.config, "model", "clip", "lambda_token")) > 0.0,
            scconcept_embeddings_dir=cfg_get(self.config, "model", "st", "scconcept_embeddings_dir"),
            scconcept_max_cells=cfg_get(self.config, "data_select", "scconcept_max_cells"),
            novae_teacher_cache_dir=cfg_get(self.config, "model", "st", "novae_teacher_cache_dir"),
            novae_teacher_fingerprint=cfg_get(self.config, "model", "st", "novae_teacher_fingerprint"),
        )



    def get_train_sampler(
        self, train_dataset: HistoStDataset) -> Sampler:
        """Return the sampler (Random/DistributedSampler/WeightedSampler) for training.

        Args:
            train_dataset (HistoStDataset): Dataset for training

        Raises:
            NotImplementedError: Not implemented sampler is selected

        Returns:
            Sampler: Sampler for training
        """

        # batch slide mode (either 'mixed' or 'single_slide' or 'alternate')
        batch_slide_mode = _normalize_batch_slide_mode(
            cfg_get(self.config, "training", "batch_slide_mode")
        )

        # ---------- Build the single-slide sampler if needed
        need_single = (batch_slide_mode in ("single_slide", "alternate"))
        sampler_single = None
        if need_single:
            seed = int(cfg_get(self.config, "seed"))
            bs = int(cfg_get(self.config, "training", "batch_size"))
            if cfg_get(self.config, "training", "sampling_strategy") != "random":
                logger.warning(
                    "batch_slide_mode='single_slide' overrides sampling_strategy (cell/tissue/slide_headcap balancing is disabled in this mode)."
                )
            sampler_single = DistributedSingleSlideUniformGlobalBatchSampler(
                dataset=train_dataset,
                batch_size=bs,
                num_replicas=(self.world_size if self.distributed else 1),
                rank=(self.rank if self.distributed else 0),
                seed=seed,
                drop_last=True,
            )
            logger.info(
                "Using DistributedSingleSlideUniformGlobalBatchSampler for training "
                f"(global_bs={bs * max(1, self.world_size)}; slide-pure batches across all ranks)."
            )
            if batch_slide_mode == "single_slide":
                return sampler_single

        # ---------- Build the mixed sampler (several slide IDs per batch)
        strategy = str(cfg_get(self.config, "training", "sampling_strategy")).lower()
        gamma = float(cfg_get(self.config, "training", "sampling_gamma"))  # Gamma scaling factor, between 0 and 1. 1 means total balancing, 0 means original weights
        seed = int(cfg_get(self.config, "seed"))
        ignore_cat = cfg_get(self.config, "training", "ignore_cat")
        if ignore_cat is None:
            ignore_cat_list = None
        elif isinstance(ignore_cat, (list, tuple)):
            ignore_cat_list = list(ignore_cat)
        else:
            # allow YAML strings like "A,B,C"
            ignore_cat_list = [s.strip() for s in str(ignore_cat).split(",") if s.strip()]

        def _log_weight_stats(w: torch.Tensor, tag: str):
            w = w.detach().cpu()
            q = torch.quantile(w, torch.tensor([0.0, 0.5, 0.9, 0.99, 1.0]))
            logger.info(
                f"[sampler:{tag}] weights stats: "
                f"min={q[0]:.6g} med={q[1]:.6g} p90={q[2]:.6g} p99={q[3]:.6g} max={q[4]:.6g} "
                f"mean={w.mean():.6g}"
            )

        if strategy == "random":
            if self.distributed:
                bs = int(cfg_get(self.config, "training", "batch_size"))
                sampler = DistributedBatchAlignedSampler(
                    train_dataset,
                    batch_size=bs,
                    num_replicas=self.world_size,
                    rank=self.rank,
                    shuffle=True,
                    seed=seed,
                    drop_last=True,
                )
                logger.info("Using DistributedBatchAlignedSampler (mixed slideIDs and random) for DDP (shuffle=True, batch-aligned) for train_dataset.")
            else:
                bs = int(cfg_get(self.config, "training", "batch_size"))
                sampler = DistributedBatchAlignedSampler(
                    train_dataset,
                    batch_size=bs,
                    num_replicas=1,
                    rank=0,
                    shuffle=True,
                    seed=seed,
                    drop_last=True,
                )
                logger.info("Using DistributedBatchAlignedSampler (mixed slideIDs and random) (single GPU, shuffle=True, batch-aligned) for train_dataset.")
        
        else:
            ds = train_dataset
            # only load cell_count if needed
            if strategy in ("cell", "cell+tissue"):
                ds.load_cell_count()

            if strategy == "cell":
                logger.info(f"Using Weighted Sampling with strategy: cell with gamma: {gamma} and ignore_cat: {ignore_cat}")
                weights = ds.get_sampling_weights_cell(gamma, ignore_cat=ignore_cat_list)
            elif strategy == "tissue":
                logger.info(f"Using Weighted Sampling with strategy: tissue with gamma: {gamma}")
                weights = ds.get_sampling_weights_tissue(gamma)
            elif strategy == "cell+tissue":
                logger.info(f"Using Weighted Sampling with strategy: cell+tissue with gamma: {gamma} and ignore_cat: {ignore_cat}")
                weights = ds.get_sampling_weights_cell_tissue(gamma, ignore_cat=ignore_cat_list)
            elif strategy == "slide_headcap":
                # Per-slide head-cap: cap dominant slides to C=min(1,C/count) (C derived at runtime from
                # headcap_max_oversampling), exempting single-slide tissues. gamma is unused here.
                max_os = float(cfg_get(self.config, "training", "headcap_max_oversampling"))
                headcap_C = cfg_get(self.config, "training", "headcap_C")  # None -> derive at runtime
                headcap_C = None if headcap_C is None else float(headcap_C)
                logger.info(
                    f"Using Weighted Sampling with strategy: slide_headcap "
                    f"(headcap_max_oversampling={max_os}, headcap_C={headcap_C})"
                )
                weights = ds.get_sampling_weights_slide_headcap(
                    max_os, headcap_C=headcap_C, exempt_single_slide=True
                )
            else:
                raise NotImplementedError(
                    "Unknown sampling strategy - Implemented are cell, tissue, cell+tissue and slide_headcap"
                )

            if not isinstance(weights, torch.Tensor):
                weights = torch.tensor(weights, dtype=torch.float32)
            if weights.numel() != len(train_dataset):
                raise ValueError(f"weights length {weights.numel()} != dataset length {len(train_dataset)}")
            if not torch.isfinite(weights).all():
                raise ValueError("weights contain NaN/Inf")
            if (weights < 0).any():
                raise ValueError("weights contain negative values")
            if float(weights.sum()) <= 0.0:
                raise ValueError("weights sum to zero")

            _log_weight_stats(weights, strategy)

            if self.distributed:
                bs = int(cfg_get(self.config, "training", "batch_size"))
                sampler = DistributedWeightedGlobalBatchUniqueSampler(
                    weights=weights,
                    batch_size=bs,
                    num_replicas=self.world_size,
                    rank=self.rank,
                    seed=seed,
                    drop_last=True,
                )
                logger.info("Using DistributedWeightedGlobalBatchUniqueSampler (mixed slideIDs and balancing) (no duplicates per global batch) for DDP.\n")
            else:
                bs = int(cfg_get(self.config, "training", "batch_size"))
                sampler = DistributedWeightedGlobalBatchUniqueSampler(
                    weights=weights,
                    batch_size=bs,
                    num_replicas=1,
                    rank=0,
                    seed=seed,
                    drop_last=True,
                )
                logger.info(f"Using WeightedGlobalBatchUniqueSampler (mixed slideIDs and balancing) (no duplicates per batch) with strategy: {strategy}\n")

        sampler_mixed = sampler

        #  ---------- Alternate every epoch between mixed and single_slide options
        if batch_slide_mode == "alternate":
            if sampler_single is None:
                raise RuntimeError("Internal error: sampler_single is None in batch_slide_mode='alternate'.")
            start_with = str(cfg_get(self.config, "training", "alternate_start_with"))
            sampler_alt = DistributedAlternatingSampler(
                sampler_mixed=sampler_mixed,
                sampler_single=sampler_single,
                start_with=start_with,
            )
            logger.info(
                "Using DistributedAlternatingSampler for training: "
                f"start_with={start_with} | even/odd epoch alternation | "
                "mixed epochs respect sampling_strategy; single-slide epochs use uniform slide sampling."
            )
            return sampler_alt

        return sampler_mixed



    @staticmethod
    def seed_worker(worker_id: int):
        """
        Make NumPy & Python RNGs match the torch seed that PyTorch already gave
        this worker.  Keeps run-to-run determinism but still diverges across
        workers and across epochs.
        """
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)



    def get_dataloader(self, name_loader, dataset, sampler, shuffle, drop_last, batch_sampler=None):
        """
        Returns a PyG DataLoader that merges each batch into a single `Batch`.
        """
        batch_size = int(cfg_get(self.config, "training", "batch_size"))
        # Do not use shuffle=True if already using a sampler or a batch_sampler
        if sampler is not None or batch_sampler is not None:
            shuffle = False

        if self.config["device_type"] == "mps":
            multiprocessing_context = 'fork' if torch.backends.mps.is_available() else None
        else:
            # Use default values for multiprocessing_context
            multiprocessing_context = None
        
        persistent_workers = False   # keep False with this code design
        
        logger.info(f"For dataset {name_loader}:")
        logger.info(f"Using shuffle: {shuffle} - always False if sampler is not None")
        logger.info(f"Using multiprocessing_context: {multiprocessing_context}")
        logger.info(f"Using persistent_workers: {persistent_workers}")
        logger.info(f"Using drop_last: {drop_last} in the dataloader (NB: still True inside the sampler if DistributedSampler)")

        # batch_sampler is mutually exclusive with batch_size/sampler/shuffle/drop_last
        if batch_sampler is not None:
            logger.info("Using batch_sampler (slide-pure batches); DataLoader will NOT use batch_size/sampler/drop_last.")
            return PyGDataLoader(
                dataset,
                batch_sampler=batch_sampler,
                num_workers=cfg_get(self.config, "training", "num_workers"),
                pin_memory=(self.config["device"].type == "cuda"),
                persistent_workers=persistent_workers,
                multiprocessing_context=multiprocessing_context,
                worker_init_fn=Experiment.seed_worker,
            )

        return PyGDataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=shuffle,
            num_workers=cfg_get(self.config, "training", "num_workers"),
            pin_memory=(self.config["device"].type == "cuda"),
            drop_last=drop_last,
            persistent_workers=persistent_workers,
            multiprocessing_context=multiprocessing_context,
            worker_init_fn=Experiment.seed_worker,
        )



    def get_model(self, resume_run: bool = False):
        """
        Instantiates and returns the CLIP model.
        The ST-graph teacher is the precomputed st_feat cache (no ST encoder submodule).
        """

        pretrained_histost = cfg_get(self.config, "model", "pretrained_histost")
        # If resume_run is True the model is initialized later, from the resume checkpoint
        if resume_run:
            pretrained_histost = None
        
        # 1. Check for pretrained HistoST checkpoint first

        if pretrained_histost is not None:
            logger.info(f"Loading full pretrained HistoST model from {pretrained_histost}")
            checkpoint = loading_checkpoint_cpu(pretrained_histost, weights_only=False)
            checkpoint_config = checkpoint.get("config", None)
            checkpoint_weights = checkpoint["model_state_dict"]
            if checkpoint_config is not None:
                logger.info("Checking compatibility between pretrained HistoST checkpoint config and current config.")
                check_pretrained_histost_config_compatibility(
                    current_config=self.config,
                    checkpoint_config=checkpoint_config,
                    checkpoint_path=pretrained_histost,
                    logger=logger,
                )
            else:
                raise ValueError("No configuration found in the checkpoint. Please provide a valid HistoST checkpoint.")

        # 2. Build the image encoder

        backbone = self.config["model"]["he"]["backbone_img_encoder"].lower()
        logger.info(f"Building image encoder with backbone: {backbone}")
        if backbone in ["virchow2", "uni2h", "hoptimus0", "hoptimus1"]:
            ckpt_path = os.path.join(self.config["model"]["foundation_models_folder"], f"{backbone}/pytorch_model.bin")
            model_timm_encoder, transform_timm_encoder, extract_embedding_timm_encoder = get_timm_he_encoder(ckpt_path=ckpt_path, model_name=backbone, map_location="cpu")
            image_encoder = ImageEncoderTimm(vision_encoder=model_timm_encoder, extract_embedding=extract_embedding_timm_encoder, model_name=backbone)
            image_feature_dim = image_encoder.output_dim
        else:
            raise ValueError(f"Invalid backbone_img_encoder: {backbone}")
        
        # Optional tiny MLP head on top of the H&E backbone
        if cfg_get(self.config, "model", "he", "use_mlp_head"):
            he_mlp_hidden_dim = cfg_get(self.config, "model", "he", "he_mlp_hidden_dim")
            he_mlp_out_dim = cfg_get(self.config, "model", "he", "he_mlp_out_dim")
            he_mlp_dropout = cfg_get(self.config, "model", "he", "he_mlp_dropout")
            logger.info(f"Adding MLP head to image encoder (in={image_feature_dim} -> hidden={he_mlp_hidden_dim} -> out={he_mlp_out_dim}) with dropout={he_mlp_dropout}")
            image_encoder = HEEncoderWithMLP(
                backbone=image_encoder,
                in_dim=image_feature_dim,
                hidden_dim=he_mlp_hidden_dim,
                out_dim=he_mlp_out_dim,
                dropout=he_mlp_dropout,
            )
            image_feature_dim = he_mlp_out_dim  # downstream CLIP uses this

        # 3. Spatial-transcriptomics teacher (precomputed)
        # st_feat is loaded by the dataset from the Novae-teacher cache; only the st_proj head is trained
        st_feature_dim = int(cfg_get(self.config, "model", "st", "st_feature_dim"))
        logger.info(
            f"ST-graph teacher: precomputed st_feat cache (dim={st_feature_dim}) at "
            f"{cfg_get(self.config, 'model', 'st', 'novae_teacher_cache_dir')} "
            f"[novae_training_id={cfg_get(self.config, 'model', 'st', 'novae_training_id')}]"
        )

        # 4. Token/node channel sizes for the (optional) token branch
        img_token_dim = getattr(image_encoder, "token_dim", None)
        # Token-level node embeddings come from scConcept (precomputed)
        st_node_dim = None
        if float(cfg_get(self.config, "model", "clip", "lambda_token")) > 0.0:
            sc_dir = Path(self.config["model"]["st"]["scconcept_embeddings_dir"])
            first_h5 = next((p for p in sorted(sc_dir.glob("*.h5")) if not p.name.startswith("._")), None)
            if first_h5 is None:
                raise FileNotFoundError(f"No scConcept .h5 file found in {sc_dir}; cannot infer token embedding dimension.")
            with h5py.File(first_h5, "r") as f:
                if "cls_emb" not in f:
                    raise KeyError(f"'cls_emb' dataset not found in {first_h5}")
                st_node_dim = int(f.attrs.get("embedding_dim", f["cls_emb"].shape[1]))
        logger.info(f"Token dims (only if lambda_token>0): img_token_dim={img_token_dim}, st_node_dim={st_node_dim}")

        # 5. Create the CLIP model

        patch_local_loss = bool(cfg_get(self.config, "model", "clip", "patch_local_loss"))
        patch_gather_with_grad = bool(cfg_get(self.config, "model", "clip", "patch_gather_with_grad"))
        patch_reinsert_local = bool(cfg_get(self.config, "model", "clip", "patch_reinsert_local"))

        token_local_loss = cfg_get(self.config, "model", "clip", "token_local_loss")
        token_gather_with_grad = cfg_get(self.config, "model", "clip", "token_gather_with_grad")
        token_reinsert_local = cfg_get(self.config, "model", "clip", "token_reinsert_local")

        # Teacher-centering means: load and check the precomputed means of the active soft-target branches
        center_teacher_target = bool(cfg_get(self.config, "model", "clip", "center_teacher_target"))
        novae_center_mean = None
        scconcept_center_mean = None
        if center_teacher_target:
            _patch_soft = (str(cfg_get(self.config, "model", "clip", "loss_type")).lower()
                           in ("soft_targets", "soft_targets_st")
                           and float(cfg_get(self.config, "model", "clip", "lambda_patch")) > 0.0)
            _token_soft = (str(cfg_get(self.config, "model", "clip", "token_loss_type")).lower()
                           in ("soft_targets", "soft_targets_st")
                           and float(cfg_get(self.config, "model", "clip", "lambda_token")) > 0.0)
            if _patch_soft or _token_soft:
                means_path = cfg_get(self.config, "model", "st", "teacher_center_means_path")
                if not means_path or not os.path.isfile(means_path):
                    raise FileNotFoundError(
                        f"center_teacher_target=True but teacher-centering means artifact is missing: {means_path}. "
                        "Run src/dataset/precompute_teacher_mean.py."
                    )
                _tm = getattr(self, "_novae_teacher_manifest", {}) or {}
                with h5py.File(means_path, "r") as _mf:
                    if int(_mf.attrs.get("schema_version", -1)) != TEACHER_CENTER_MEANS_SCHEMA_VERSION:
                        raise ValueError(
                            f"teacher_center_means schema_version={_mf.attrs.get('schema_version')} != "
                            f"{TEACHER_CENTER_MEANS_SCHEMA_VERSION}; re-run precompute_teacher_mean.py."
                        )
                    if _patch_soft:
                        _exp_fp = cfg_get(self.config, "model", "st", "novae_teacher_fingerprint")
                        if _mf.attrs.get("novae_teacher_fingerprint") != _exp_fp:
                            raise ValueError(
                                f"teacher_center_means novae_teacher_fingerprint="
                                f"{_mf.attrs.get('novae_teacher_fingerprint')!r} != configured teacher cache "
                                f"fingerprint={_exp_fp!r}. Re-run precompute_teacher_mean.py."
                            )
                        _exp_nmc = int(cfg_get(self.config, "data_select", "n_min_cells"))
                        if int(_mf.attrs.get("novae_n_min_cells", -1)) != _exp_nmc:
                            raise ValueError(
                                f"teacher_center_means novae_n_min_cells={_mf.attrs.get('novae_n_min_cells')} != "
                                f"this run's data_select.n_min_cells={_exp_nmc}. The Novae mean is "
                                f"n_min_cells-dependent; re-run: precompute_teacher_mean.py --n_min_cells {_exp_nmc}."
                            )
                        if "novae_mean" not in _mf:
                            raise KeyError(
                                f"teacher_center_means.h5 has no 'novae_mean' dataset: {means_path} "
                                "(run precompute_teacher_mean.py --which both)."
                            )
                        novae_center_mean = torch.from_numpy(_mf["novae_mean"][:]).float()
                    if _token_soft:
                        for _k, _want in (("scconcept_training_id", _tm.get("scconcept_training_id")),
                                          ("scconcept_model_name", _tm.get("scconcept_model_name")),
                                          ("scconcept_model_version", _tm.get("scconcept_model_version")),
                                          ("training_dataset_id", self.config["data"]["training_dataset_id"]),
                                          ("split", "train")):
                            _got = _mf.attrs.get(_k)
                            if _want is not None and _got != _want:
                                raise ValueError(
                                    f"teacher_center_means '{_k}'={_got!r} != expected {_want!r}. "
                                    "Re-run precompute_teacher_mean.py for this teacher / training dataset."
                                )
                        if "scconcept_mean" not in _mf:
                            raise KeyError(
                                f"teacher_center_means.h5 has no 'scconcept_mean' dataset: {means_path} "
                                "(run precompute_teacher_mean.py --which both)."
                            )
                        scconcept_center_mean = torch.from_numpy(_mf["scconcept_mean"][:]).float()
                logger.info(
                    f"Teacher-centering ON: novae_mean={'loaded' if novae_center_mean is not None else 'n/a'}, "
                    f"scconcept_mean={'loaded' if scconcept_center_mean is not None else 'n/a'} (from {means_path})"
                )

        clip_model = CLIPModel(
            image_encoder=image_encoder,
            image_feature_dim=image_feature_dim,
            st_feature_dim=st_feature_dim,
            embed_dim=cfg_get(self.config, "model", "clip", "clip_embed_dim"),
            img_token_dim=img_token_dim,
            st_node_dim=st_node_dim,
            patch_local_loss=patch_local_loss,
            patch_gather_with_grad=patch_gather_with_grad,
            patch_reinsert_local=patch_reinsert_local,
            loss_type=cfg_get(self.config, "model", "clip", "loss_type"),
            target_temperature=cfg_get(self.config, "model", "clip", "target_temperature"),
            lambda_patch=cfg_get(self.config, "model", "clip", "lambda_patch"),
            lambda_token=cfg_get(self.config, "model", "clip", "lambda_token"),
            token_embed_dim=cfg_get(self.config, "model", "clip", "token_embed_dim"),
            token_loss_type=cfg_get(self.config, "model", "clip", "token_loss_type"),
            token_target_temperature=cfg_get(self.config, "model", "clip", "token_target_temperature"),
            token_local_loss=token_local_loss,
            token_gather_with_grad=token_gather_with_grad,
            token_reinsert_local=token_reinsert_local,
            center_teacher_target=center_teacher_target,
            novae_center_mean=novae_center_mean,
            scconcept_center_mean=scconcept_center_mean,
        )
        logger.info(f"CLIP model initialized with :"
                    f"\n- image_feature_dim={clip_model.image_feature_dim}"
                    f"\n- st_feature_dim={clip_model.st_feature_dim}"
                    f"\n- embed_dim={clip_model.embed_dim}"
                    f"\n- loss_type={clip_model.loss_type}"
                    f"\n- target_temperature={clip_model.target_temperature}"
                    f"\n- lambda_patch={clip_model.lambda_patch}"
                    f"\n- lambda_token={clip_model.lambda_token}"
                    f"\n- token_embed_dim={clip_model.token_embed_dim}"
                    f"\n- token_loss_type={clip_model.token_loss_type}"
                    f"\n- token_target_temperature={clip_model.token_target_temperature}"
                    f"\n- patch_local_loss={clip_model.patch_local_loss}"
                    f"\n- patch_gather_with_grad={clip_model.patch_gather_with_grad}"
                    f"\n- patch_reinsert_local={clip_model.patch_reinsert_local}"
                    f"\n- token_local_loss={clip_model.token_local_loss}"
                    f"\n- token_gather_with_grad={clip_model.token_gather_with_grad}"
                    f"\n- token_reinsert_local={clip_model.token_reinsert_local}"
                    f"\n- center_teacher_target={clip_model.center_teacher_target}"
        )

        # Log token-loss warmup configuration (actual scheduling happens inside Trainer.fit)
        lambda_token_warmup_epochs = int(cfg_get(self.config, "model", "clip", "lambda_token_warmup_epochs"))
        if clip_model.lambda_token > 0.0 and lambda_token_warmup_epochs > 0:
            logger.info(
                f"Token loss warmup enabled: lambda_token will be linearly ramped "
                f"from 0 to {clip_model.lambda_token} over {lambda_token_warmup_epochs} epoch(s)."
            )
        else:
            logger.info("Token loss warmup disabled (either lambda_token == 0 or lambda_token_warmup_epochs == 0).")

        # 6. Insert LoRa adapter if specified
        if cfg_get(self.config, "training", "HE_training_option") == "lora":
            logger.info("Inserting LoRA adapter into the image encoder.")
            insert_lora(
                image_encoder,
                cfg_get(self.config, "training", "lora_rank"),
                cfg_get(self.config, "training", "lora_alpha"),
            )


        # 7. Load pretrained weights for histost model from previous training if provided / at the end so that they override any pretrained image encoder weights
        if pretrained_histost is not None:
            checkpoint_weights = reconcile_state_dict(checkpoint_weights, clip_model)
            assert_pretrained_state_dict_compatible(clip_model, checkpoint_weights, pretrained_histost)
            load_result = clip_model.load_state_dict(checkpoint_weights, strict=True)
            logger.info(f"Full pretrained HistoST checkpoint loaded successfully: {str(load_result)}\n")
        else:
            logger.info("No previous HistoST pretrained weights provided.\n")

        return clip_model



    def print_clip_model_parameters(self, clip_model):
        """
        Logs the parameter counts of the CLIP model, the image encoder and the st_proj head.
        """

        # Under DDP, unwrap the underlying "real" model
        if isinstance(clip_model, DDP):
            clip_model_print = clip_model.module
        else:
            clip_model_print = clip_model

        # Entire CLIP model
        total_params_clip = sum(p.numel() for p in clip_model_print.parameters())
        trainable_params_clip = sum(p.numel() for p in clip_model_print.parameters() if p.requires_grad)

        # Image encoder
        total_params_img = sum(p.numel() for p in clip_model_print.image_encoder.parameters())
        trainable_params_img = sum(p.numel() for p in clip_model_print.image_encoder.parameters() if p.requires_grad)

        # ST projection head st_proj (the ST teacher is precomputed)
        total_params_stproj = sum(p.numel() for p in clip_model_print.st_proj.parameters())
        trainable_params_stproj = sum(p.numel() for p in clip_model_print.st_proj.parameters() if p.requires_grad)

        logger.info("======== CLIP Model Parameters ========")
        logger.info(f"Total parameters in CLIP model:            {total_params_clip:,}")
        logger.info(f"Trainable parameters in CLIP model:        {trainable_params_clip:,}\n")
        logger.info("======== Image Encoder Parameters =======")
        logger.info(f"Total parameters in Image Encoder:         {total_params_img:,}")
        logger.info(f"Trainable parameters in Image Encoder:     {trainable_params_img:,}\n")
        logger.info("===== ST projection head (st_proj; ST teacher is precomputed) ======")
        logger.info(f"Total parameters in st_proj:              {total_params_stproj:,}")
        logger.info(f"Trainable parameters in st_proj:          {trainable_params_stproj:,}\n\n")

    

    def multi_gpu(self, model):
        """
        Wraps the model in DistributedDataParallel (DDP) if using multiple GPUs.
        """
        if self.distributed:
            logger.info(f"Wrapping model in DDP on GPU rank= {self.rank} and local rank= {self.local_rank}")
            model = DDP(model, device_ids=[self.local_rank], output_device=self.local_rank, find_unused_parameters=True)
        else:
            logger.info("No multi-GPU distributed training detected.\n")
        return model
    


    def apply_model_freezing(self, model):

        # Apply model freezing based on HE_training_option (st_proj stays trainable in every option)

        option = cfg_get(self.config, "training", "HE_training_option")

        if option == "train":
            logger.info("Training the entire model (image encoder + projection heads).")
            for name, param in model.named_parameters():
                param.requires_grad = True

        elif option == "freeze":
            logger.info("Freezing the image encoder except mlp_head; projection heads (incl. st_proj) stay trainable.")
            for name, param in model.named_parameters():
                param.requires_grad = not ("image_encoder" in name and not any(layer in name for layer in ["mlp_head"]))

        elif option == "lora":
            logger.info("Freezing the image encoder except LoRA adapters + MLP head; projection heads (incl. st_proj) stay trainable.")
            for name, param in model.named_parameters():
                param.requires_grad = not ("image_encoder" in name and not any(layer in name for layer in ["adapter", "mlp_head"]))

        else:
            raise ValueError("Invalid HE_training_option. Choose from 'freeze', 'train' or 'lora'.")



    def _adv_sched_cfg(self) -> dict:
        cfg = cfg_get(self.config, "advanced_scheduling")
        return cfg if isinstance(cfg, dict) else {}

    def _adv_sched_enabled(self) -> bool:
        return bool(cfg_get(self.config, "advanced_scheduling", "enabled"))

    def _validate_advanced_scheduling_config(self) -> None:
        """
        Check the config when advanced scheduling is enabled: H&E trained with LoRA only, no MLP head, precomputed ST teacher.
        """
        if not self._adv_sched_enabled():
            return

        warm = int(cfg_get(self.config, "advanced_scheduling", "clip_head_warmup_epoch"))
        if warm < 0:
            raise ValueError("advanced_scheduling.clip_head_warmup_epoch must be >= 0")

        # Enforce H&E side: LoRA only (no tiny MLP head)
        if bool(cfg_get(self.config, "model", "he", "use_mlp_head")):
            raise ValueError(
                "advanced_scheduling.enabled=True requires model.he.use_mlp_head=False."
            )
        if str(cfg_get(self.config, "training", "HE_training_option")).lower() != "lora":
            raise ValueError(
                "advanced_scheduling.enabled=True requires training.HE_training_option='lora'."
            )

        # ST side: the precomputed Novae-teacher cache must be resolved
        if not cfg_get(self.config, "model", "st", "novae_teacher_cache_dir"):
            raise ValueError(
                "advanced_scheduling.enabled=True requires the precomputed Novae-teacher cache "
                "(model.st.novae_training_id resolved to novae_teacher_cache_dir)."
            )



    def _warn_soft_target_teacher_config(self) -> None:
        """
        Log warnings for the mixed 'soft_targets' losses and for a center_teacher_target without effect.
        """
        patch_loss = str(cfg_get(self.config, "model", "clip", "loss_type")).lower()
        token_loss = str(cfg_get(self.config, "model", "clip", "token_loss_type")).lower()
        lambda_patch = float(cfg_get(self.config, "model", "clip", "lambda_patch"))
        lambda_token = float(cfg_get(self.config, "model", "clip", "lambda_token"))

        # ---- Patch branch (if lambda_patch > 0) ----
        if lambda_patch > 0.0 and patch_loss == "soft_targets":
            logger.warning(
                "model.clip.loss_type='soft_targets' (mixed H&E+ST): the target uses the H&E encoder being "
                "trained, on augmented views. Prefer 'soft_targets_st' or 'classic'."
            )

        # ---- Token branch (only active when lambda_token>0) ----
        if lambda_token > 0.0 and token_loss in ("soft_targets", "soft_targets_st"):
            if token_loss == "soft_targets":
                logger.warning(
                    "model.clip.token_loss_type='soft_targets' (mixed): the target uses the H&E encoder being "
                    "trained, on augmented views. Prefer 'soft_targets_st' or 'classic'."
                )
            # token 'soft_targets_st' uses the precomputed scConcept teacher (always frozen) => no warning.

        # ---- Teacher-centering no-op warnings ----
        # center_teacher_target only affects soft-target branches
        if bool(cfg_get(self.config, "model", "clip", "center_teacher_target")):
            if lambda_patch > 0.0 and patch_loss == "classic":
                logger.warning(
                    "center_teacher_target=True has no effect on the patch branch (loss_type='classic' has no "
                    "soft target to center)."
                )
            if lambda_token > 0.0 and token_loss == "classic":
                logger.warning(
                    "center_teacher_target=True has no effect on the token branch (token_loss_type='classic')."
                )
            _p_off = (lambda_patch <= 0.0 or patch_loss == "classic")
            _t_off = (lambda_token <= 0.0 or token_loss == "classic")
            if _p_off and _t_off:
                logger.warning("center_teacher_target=True is a full no-op (no active soft-target loss).")



    def _warn_inert_weight_decay_lora(self) -> None:
        """
        Warn if advanced_scheduling.weight_decay_lora is set while advanced scheduling is off (LoRA factors are then not decayed).
        """
        if self._adv_sched_enabled():
            return  # applied to the "lora" group

        try:
            wd_lora = float(cfg_get(self.config, "advanced_scheduling", "weight_decay_lora"))
        except (TypeError, ValueError):
            # Unparsable value (e.g. empty YAML value)
            return

        if wd_lora != 0.0:
            logger.warning(
                f"advanced_scheduling.weight_decay_lora={wd_lora} has no effect: with "
                "advanced_scheduling.enabled=False, LoRA factors are in the no-decay group (wd=0). "
                "Set advanced_scheduling.enabled=True to apply it, or weight_decay_lora=0.0."
            )



    def get_optimizer(self, model):
        """
        Returns the optimizer for the model.
        """
        # Using AdamW optimizer with pytorch defaults values if not specified

        base_lr = float(cfg_get(self.config, "training", "optimizer_hyperparameters", "lr"))
        betas = cfg_get(self.config, "training", "optimizer_hyperparameters", "betas")
        eps = cfg_get(self.config, "training", "optimizer_hyperparameters", "eps")
        wd_default = float(cfg_get(self.config, "training", "optimizer_hyperparameters", "weight_decay"))

        adv_enabled = self._adv_sched_enabled()

        # Unwrap DDP so names are stable for head matching (no "module." prefix).
        backbone = model.module if isinstance(model, DDP) else model

        # --------- Case A: advanced scheduling OFF 
        
        if not adv_enabled:
            decay, no_decay = [], []
            for n, p in model.named_parameters():
                # Skip frozen parameters
                if not p.requires_grad:
                    continue

                n_low = n.lower()
                is_lora = ("adapter" in n_low)  # adapter_Q_*, adapter_V_*, etc.

                # LoRA is not decayed here (advanced scheduling applies weight_decay_lora to the "lora" group)
                if is_lora:
                    no_decay.append(p)
                    continue

                if p.ndim < 2 or "norm" in n_low or "bias" in n_low or "logit_scale" in n_low:
                    no_decay.append(p)         # γ=0
                else:
                    decay.append(p)            # γ>0

            param_groups = [
                {"params": no_decay, "weight_decay": 0.0},
                {"params": decay, "weight_decay": wd_default},
            ]

            optimizer = torch.optim.AdamW(
                param_groups,
                lr=base_lr,
                betas=betas,
                eps=eps,
            )
            logger.info(
                f"Using AdamW optimizer (advanced_scheduling=OFF) with lr={base_lr}, betas={betas}, eps={eps}, "
                f"weight_decay={wd_default} (LoRA wd=0)."
            )
            return optimizer

        # --------- Case B: advanced scheduling ON 
        # -> named param groups so Trainer can freeze heads after warmup without rebuilding optimizer

        adv = self._adv_sched_cfg()
        lr_lora  = float(adv.get("lr_lora", base_lr))
        lr_heads = float(adv.get("lr_heads", base_lr))
        lr_logit = float(adv.get("lr_logit", base_lr * 0.1))
        wd_lora  = float(adv.get("weight_decay_lora", 0.0))
        wd_heads = float(adv.get("weight_decay_heads", wd_default))
        wd_other = float(adv.get("weight_decay_other", wd_default))

        # Collect params (keep stable groups; empty groups are OK)
        lora_params = []
        head_decay = []
        head_no_decay = []
        logit_params = []
        other_no_decay = []
        other_decay = []

        # Heads frozen after warmup
        head_prefixes = ("image_proj.", "st_proj.", "img_tok_proj.", "st_node_proj.")

        for n, p in backbone.named_parameters():
            n_low = n.lower()

            # With advanced scheduling, only LoRA, CLIP heads (until warmup ends) and logit_scale are trained
            is_lora = ("adapter" in n_low)  # adapter_Q_*, adapter_V_*, etc.
            is_head = n.startswith(head_prefixes)
            is_logit = ("logit_scale" in n_low)

            # Keep heads/logits in optimizer even if later frozen, to preserve group structure on resume
            if is_logit:
                logit_params.append(p)
                continue
            if is_head:
                # No WD on bias/norm/1D params
                if (p.ndim < 2) or ("bias" in n_low) or ("norm" in n_low):
                    head_no_decay.append(p)
                else:
                    head_decay.append(p)
                continue
            if is_lora:
                lora_params.append(p)
                continue

            # Any remaining trainable params (should be empty given the enforced config)
            if not p.requires_grad:
                continue
            if p.ndim < 2 or "norm" in n_low or "bias" in n_low:
                other_no_decay.append(p)
            else:
                other_decay.append(p)

        param_groups = [
            {"group_name": "lora", "params": lora_params, "lr": lr_lora,  "weight_decay": wd_lora},
            {"group_name": "heads_decay", "params": head_decay, "lr": lr_heads, "weight_decay": wd_heads},
            {"group_name": "heads_no_decay", "params": head_no_decay, "lr": lr_heads, "weight_decay": 0.0},
            {"group_name": "logit", "params": logit_params, "lr": lr_logit, "weight_decay": 0.0},
            {"group_name": "other_no_decay", "params": other_no_decay, "lr": base_lr, "weight_decay": 0.0},
            {"group_name": "other_decay", "params": other_decay, "lr": base_lr, "weight_decay": wd_other},
        ]

        optimizer = torch.optim.AdamW(
            param_groups,
            betas=betas,
            eps=eps,
        )

        logger.info(
            "Using AdamW optimizer (advanced_scheduling=ON) with param groups:\n"
            f"  - lora:  lr={lr_lora} wd={wd_lora}\n"
            f"  - heads_decay:    lr={lr_heads} wd={wd_heads}\n"
            f"  - heads_no_decay: lr={lr_heads} wd=0\n"
            f"  - logit: lr={lr_logit} wd=0\n"
            f"  - other: lr={base_lr} wd={wd_other} (expected empty)\n"
            "Note: projection/token heads will be frozen after warmup by Trainer without rebuilding optimizer."
        )
        return optimizer



    def get_scheduler(self, optimizer):
        """
        Returns a learning rate scheduler.
        Supports "constant" and "cosine" schedulers.
        """
        scheduler_type = cfg_get(self.config, "training", "scheduler", "scheduler_type")
        warmup_steps = cfg_get(self.config, "training", "scheduler", "warmup_steps")
        num_train_samples = self.config["data_select"]["num_train_samples"]
        global_bs = cfg_get(self.config, "training", "batch_size") * max(1, self.world_size)
        steps_per_epoch = math.ceil(num_train_samples / global_bs)
        total_steps = steps_per_epoch * self.config["training"]["epochs"]
        
        if scheduler_type == "constant":
            if warmup_steps > 0:
                def lr_lambda(step):
                    return (step + 1) / warmup_steps if step < warmup_steps else 1.0
                scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
                logger.info(f"Using constant LR with warm-up ({warmup_steps} updates, {steps_per_epoch} steps/epoch)")
            else:
                scheduler = None
                logger.info("Using constant learning rate scheduler")

        elif scheduler_type == "cosine":
            def lr_lambda(step):
                if step < warmup_steps:
                    return (step + 1) / warmup_steps
                progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
                return 0.5 * (1.0 + math.cos(math.pi * progress))
            scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
            logger.info(f"Using cosine LR (update-wise), warm-up={warmup_steps}, total_steps={total_steps} by using steps/epoch={steps_per_epoch}")
        
        else:
            scheduler = None
            logger.warning("No valid scheduler type specified. Using constant learning rate scheduler.")
        
        return scheduler, steps_per_epoch



    def get_trainer(self, model, optimizer, scheduler):
        """
        Instantiates and returns a Trainer instance.
        """
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            device=self.config["device"],
            scheduler=scheduler,
            use_amp=cfg_get(self.config, "training", "mixed_precision"),
            amp_dtype=cfg_get(self.config, "training", "amp_dtype"),
        )
        logger.info(f"Using mixed precision training: {cfg_get(self.config, 'training', 'mixed_precision')}")
        logger.info(f"Using AMP dtype: {cfg_get(self.config, 'training', 'amp_dtype')}")
        logger.info(f"Using device: {self.config['device']}\n\n")
        return trainer




    def run_experiment(self):
        """
        Main entry point for running the experiment.
        Initializes wandb, sets up datasets, model, optimizer, scheduler, trainer,
        and runs the training/validation loop with logging.
        """
        
        resume_checkpoint_path = self.config.get("checkpoint_start_path", None)
        resume_run = (resume_checkpoint_path is not None and os.path.isfile(resume_checkpoint_path))
        if resume_run:
            logger.info(f"Will resume training from checkpoint: {resume_checkpoint_path}")

        # 0. Cross-run full-state resume: check config compatibility with the source run (incl. steps per epoch)
        if self.cross_run_resume and resume_run:
            src_ckpt = loading_checkpoint_cpu(resume_checkpoint_path, weights_only=False)
            source_config = src_ckpt.get("config", None)
            if source_config is None:
                raise ValueError(f"Cross-run resume source checkpoint has no embedded config: {resume_checkpoint_path}")
            # steps/epoch: new from the config (as in get_scheduler), source from the checkpoint global_step / epochs
            global_bs = cfg_get(self.config, "training", "batch_size") * max(1, self.world_size)
            new_step_per_epoch = math.ceil(self.config["data_select"]["num_train_samples"] / global_bs)
            src_global_step = src_ckpt.get("global_step", None)
            src_epoch = src_ckpt.get("epoch", None)
            source_step_per_epoch = (
                src_global_step / (src_epoch + 1)
                if (isinstance(src_global_step, (int, float)) and src_global_step > 0
                    and isinstance(src_epoch, int) and src_epoch >= 0)
                else None
            )
            check_cross_run_resume_compatibility(
                new_config=self.config,
                source_config=source_config,
                checkpoint_path=resume_checkpoint_path,
                source_step_per_epoch=source_step_per_epoch,
                new_step_per_epoch=new_step_per_epoch,
                allow_schedule_mismatch=self.allow_schedule_mismatch,
                logger=logger,
            )
            self._source_training_id = cfg_get(source_config, "logging", "training_id_histost")
            logger.info(
                f"Cross-run resume validated. Source training_id={self._source_training_id}; "
                f"source steps/epoch~={source_step_per_epoch}; new steps/epoch={new_step_per_epoch}."
            )
            del src_ckpt  # free the CPU copy; trainer.load_checkpoint reloads for the actual restore

        # 1. Setup or resume wandb
        if (not self.distributed) or (self.rank == 0):
            if resume_run and self.wandb_run_id:
                wandb.init(
                    project=self.config["logging"]["project"],
                    config=self.config,
                    tags=self.config["logging"]["tags"],
                    notes=self.config["logging"]["notes"],
                    mode=self.config["logging"]["mode"],
                    dir=self.config["logging"]["wandb_dir"],
                    group=self.config["logging"].get("group", None),
                    resume="must",
                    id=self.wandb_run_id
                )
                logger.info(f"Resumed W&B run with id: {self.wandb_run_id}\n")
            elif resume_run:
                wandb.init(
                    project=self.config["logging"]["project"],
                    config=self.config,
                    tags=self.config["logging"]["tags"],
                    notes=self.config["logging"]["notes"],
                    mode=self.config["logging"]["mode"],
                    dir=self.config["logging"]["wandb_dir"],
                    group=self.config["logging"].get("group", None),
                )
                logger.info("Resumed training but no W&B run ID found; starting new run.\n")
            else:
                wandb.init(
                    project=self.config["logging"]["project"],
                    config=self.config,
                    tags=self.config["logging"]["tags"],
                    notes=self.config["logging"]["notes"],
                    mode=self.config["logging"]["mode"],
                    dir=self.config["logging"]["wandb_dir"],
                    group=self.config["logging"].get("group", None),
                )
                logger.info("Initialized new wandb run.\n")

            wandb.run.name = self.config["logging"]["log_comment"]
            wandb.run.save()

        # 2. Instantiate model
        model = self.get_model(resume_run=resume_run)
        # move model to device
        model = model.to(self.config["device"])
        logger.info(f"Model architecture: {model}\n\n")

        # 3. Check for multi-GPU and wrap the model if needed
        model = self.multi_gpu(model)

        # 4. Freezing options for the image encoder
        self.apply_model_freezing(model)

        # Validate advanced scheduling constraints
        self._validate_advanced_scheduling_config()

        # Warn about soft-target loss settings
        self._warn_soft_target_teacher_config()

        # Warn if weight_decay_lora is set while advanced scheduling is off
        self._warn_inert_weight_decay_lora()

        # 5. Create optimizer, scheduler and trainer
        logger.info("Creating optimizer, scheduler and trainer...")
        optimizer = self.get_optimizer(model)
        scheduler, steps_per_epoch = self.get_scheduler(optimizer)
        trainer = self.get_trainer(model, optimizer, scheduler)

        # 6. If resuming, load the checkpoint
        if resume_run:
            start_epoch = trainer.load_checkpoint(checkpoint_path=resume_checkpoint_path)
        else:
            start_epoch = 0
        logger.info(f"Starting from epoch = {start_epoch}")

        # 6b. Cross-run resume: copy the source run's epochs [0, start_epoch) into this run's JSONL and W&B run (rank 0)
        if self.cross_run_resume and ((not self.distributed) or (self.rank == 0)):
            self._seed_cross_run_history(start_epoch)

        # 7. Check model parameters and their gradients
        logger.info(f"\n\n")
        logger.info("Checking model parameters and their gradients...")
        for name, param in model.named_parameters():
            logger.info(f"{name}: {param.requires_grad}")
        logger.info(f"\n\n")

        self.print_clip_model_parameters(model)

        # 8. Check inside the optimizer that weight decay is still 0 for bias, norm and logit_scale
        logger.info("Checking optimizer parameter groups and weight decay settings...")
        param_to_name = {p: n for n, p in model.named_parameters()}
        for gi, group in enumerate(optimizer.param_groups):
            wd = group.get("weight_decay", 0.0)
            for p in group["params"]:
                if not p.requires_grad:
                    continue
                name = param_to_name.get(p, "<unnamed-param>")
                logger.info(f"Optimizer group {gi}: {name} - weight decay: {wd}")
        logger.info(f"\n\n")

        # 9. Set up H&E augmentation transforms
        train_he_transforms, val_he_transforms = self.get_transforms_he(
            cfg_get(self.config, "he_transformations"),
            input_shape=self.config["data"]["img_size"]
        )

        # 10. Create datasets
        logger.info("Creating datasets...\n")
        train_dataset = self.get_dataset(fold=self.config["data"]["train_fold"], he_transform=train_he_transforms, mode="train")
        val_dataset = self.get_dataset(fold=self.config["data"]["val_fold"], he_transform=val_he_transforms, mode="valid")
        
        if self.distributed:
            dist.barrier()
        
        # Get slide_idx mapping for validation dataset class to compute metrics in trainer later
        slide_id2idx_val = val_dataset.slide_id_to_idx
        logger.info(f"Validation dataset slide_id2idx mapping: {slide_id2idx_val}\n")

        # Build INT/EXT partitions for validation (from slide_set_scope in cell_count_valid.csv)
        slide_partitions_val = None
        try:
            valid_fold_name = self.config["data"]["val_fold"]  # usually "valid"
            if cfg_get(self.config, "data_select", "n_min_cells") > 0:
                n_min_cells = cfg_get(self.config, "data_select", "n_min_cells")
                cc_path = Path(self.config["data"]["dataset_path"]) / f"cell_count_{valid_fold_name}_{n_min_cells}.csv"
            else:
                cc_path = Path(self.config["data"]["dataset_path"]) / f"cell_count_{valid_fold_name}.csv"
            if cc_path.is_file():
                df_cc = pd.read_csv(cc_path, usecols=["patch_id", "slide_set_scope"])
                df_cc["slide_id"] = df_cc["patch_id"].str.rsplit("_", n=1).str[0]
                # 'specific' => slide appears only in this set -> EXT
                # 'shared'   => slide appears in valid and train sets   -> INT
                ext_slides = set(df_cc.loc[df_cc["slide_set_scope"] == "specific", "slide_id"].unique().tolist())
                int_slides = set(df_cc.loc[df_cc["slide_set_scope"] == "shared", "slide_id"].unique().tolist())
                # keep only slides actually present in the validation dataset
                ext_idx = sorted([slide_id2idx_val[s] for s in ext_slides if s in slide_id2idx_val])
                int_idx = sorted([slide_id2idx_val[s] for s in int_slides if s in slide_id2idx_val])
                slide_partitions_val = {"INT": int_idx, "EXT": ext_idx}
                logger.info(f"Validation partitions -> INT slides: {len(int_idx)} | EXT slides: {len(ext_idx)}")
            else:
                raise FileNotFoundError(f"Cell count file not found for validation partition: {cc_path}")
        except Exception as e:
            raise RuntimeError(f"Error while building validation slide partitions: {e}")

        # 12. Get samplers and dataloaders
        logger.info("Creating samplers and dataloaders...")
        
        # training sampler
        train_sampler = self.get_train_sampler(train_dataset)
        seed = int(cfg_get(self.config, "seed"))

        # Step val (mixed) sampler (keep ranks batch-aligned to be safe for all_gather)
        if self.distributed:
            val_sampler_step_mixed = DistributedSampler(
                val_dataset,
                shuffle=False,
                drop_last=True,
                num_replicas=self.world_size,
                rank=self.rank,
                seed=seed,
            )
            logger.info("Using DistributedSampler(drop_last=True) for DDP val STEP mixed loader (batch-aligned).\n")
        else:
            val_sampler_step_mixed = None
            logger.info("Not using sampler for val STEP mixed loader (single GPU).\n")

        # Global val sampler (no drop, no pad; uneven lengths are allowed because variable lengths are gathered to rank0)
        if self.distributed:
            val_sampler_global = DistributedStridedEvalSampler(
                val_dataset,
                num_replicas=self.world_size,
                rank=self.rank,
            )
            logger.info("Using DistributedStridedEvalSampler for DDP val GLOBAL loader (no drop, no padding).\n")
        else:
            val_sampler_global = None
            logger.info("Not using sampler for val GLOBAL loader (single GPU).\n")

        # Optional single-slide batches for comparability
        train_batch_slide_mode = _normalize_batch_slide_mode(cfg_get(self.config, "training", "batch_slide_mode"))
        val_step_mode_raw = cfg_get(self.config, "training", "val_step_slide_mode")
        val_step_mode = train_batch_slide_mode if str(val_step_mode_raw).strip().lower() in ("inherit", "same") else _normalize_batch_slide_mode(val_step_mode_raw)

        # Training loader
        train_loader = self.get_dataloader("train_loader", train_dataset, train_sampler, shuffle=True, drop_last=True)

        # Global val loader (full set, embeddings-only pass)
        val_loader_global = self.get_dataloader(
            "val_loader_global",
            val_dataset,
            val_sampler_global,
            shuffle=False,
            drop_last=False,
        )

        # Step val mixed loader (batch-aligned to be safe for validation_step() all_gather)
        if self.distributed:
            val_loader_step_mixed = self.get_dataloader(
                "val_loader_step_mixed",
                val_dataset,
                val_sampler_step_mixed,
                shuffle=False,
                drop_last=False,
            )
        else:
            # single GPU: Step val mixed loader can just reuse Global val loader
            val_loader_step_mixed = val_loader_global

        # STEP val loader (optional slide-pure / optional alternating)
        val_loader_step_single = None

        if val_step_mode in ("single_slide", "alternate"):
            bs = int(cfg_get(self.config, "training", "batch_size"))
            bsampler = DistributedSingleSlideBatchSampler(
                dataset=val_dataset,
                batch_size=bs,
                num_replicas=(self.world_size if self.distributed else 1),
                rank=(self.rank if self.distributed else 0),
                seed=seed,
                drop_last=True,                 # per-slide DDP drop_last
                shuffle_slides=False,
                shuffle_within_slide=False,
            )
            val_loader_step_single = self.get_dataloader(
                "val_loader_step_single_slide",
                val_dataset,
                sampler=None,
                batch_sampler=bsampler,
                shuffle=False,
                drop_last=False,
            )
            logger.info("Built Validation STEP single-slide loader (BatchSampler). Global metrics still use mixed-slide loader.")
        
        if val_step_mode == "single_slide":
            val_loader_step = val_loader_step_single
            logger.info("Validation STEP loader will use single-slide batches.")
        elif val_step_mode == "alternate":
            # epoch-level alternation for STEP metrics: mixed loader is val_loader_global, single is val_loader_step_single
            if val_loader_step_single is None:
                raise RuntimeError("val_step_mode='alternate' requires val_loader_step_single to be built.")
            val_loader_step = val_loader_step_mixed  # mixed-by-default; Trainer will select per-epoch
            logger.info("Validation STEP loader will ALTERNATE per-epoch between mixed (val_loader_global) and single-slide (val_loader_step_single).")
        else:
            val_loader_step = val_loader_step_mixed
            logger.info("Validation STEP loader will use mixed-slide batches (STEP loader, batch-aligned).")

        # 13. Some checkings

        # Check steps_per_epoch approximatively correct
        real_steps_per_epoch = len(train_loader)
        logger.info(f"Real steps per epoch: {real_steps_per_epoch} and config steps per epoch used for warmup: {steps_per_epoch}\n")
        if abs(real_steps_per_epoch - steps_per_epoch) / real_steps_per_epoch > 0.05:
            raise ValueError(f"Steps per epoch mismatch: real = {real_steps_per_epoch} vs config = {steps_per_epoch}. Please check config num_train_samples.")

        # Sanity check: number of epochs must be >= start_epoch
        if self.config["training"]["epochs"] < start_epoch:
            raise ValueError(f"Number of epochs {self.config['training']['epochs']} should be greater than start_epoch {start_epoch}.")
        
        # Sanity check: model in trainer and experiment are the same object
        model_id_experiment = id(model)
        trainer_model_id = id(trainer.model)
        if model_id_experiment != trainer_model_id:
            raise ValueError(f"Model in trainer and experiment are not the same object. Experiment model ID: {model_id_experiment}, Trainer model ID: {trainer_model_id}")
        
        # Uncomment to stop after building the lmdb cache (no training)
        # raise Exception("Stop here after computing lmdb")

        
        # 14. Run training
        logger.info("Starting training...")
        checkpoint_path = os.path.join(self.config["logging"]["log_dir"], "checkpoints", "checkpoint_epoch_{epoch}.pth")
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        history = trainer.fit(
            train_loader,
            val_loader_step,
            val_loader_global=val_loader_global,
            slide_id2idx_val=slide_id2idx_val,
            num_epochs=self.config["training"]["epochs"],
            config=self.config,
            checkpoint_path=checkpoint_path if (not self.distributed or self.rank == 0) else None,
            start_epoch=start_epoch,    # from checkpoint if resuming
            slide_partitions_val=slide_partitions_val,
            val_loader_step_single=val_loader_step_single,
        )
        
        # If rank == 0, finish wandb
        if (not self.distributed) or (self.rank == 0):
            wandb.finish()
        
        logger.info("Experiment completed.")
        return history