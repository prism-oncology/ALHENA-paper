#!/usr/bin/env python
"""
Export the H&E and ST encoders of a HistoST CLIP checkpoint as torch.save bundles (config + state_dict).
If checkpoint_epoch < 0, only the pretrained H&E backbone is exported.

Saved in <ROOT_HistoST> / train / trainings / trainings_HistoST / <training_id> / trained_encoders /
    - HEencoder_t<n>e<c>.pt: H&E encoder (expects preprocessed input; no L2-normalization inside)
    - STencoderGraph_t<n>e<c>.pt: ST graph projection head st_proj (only if checkpoint_epoch >= 0)
    - STencoderCell_t<n>e<c>.pt: ST cell encoder (only if checkpoint_epoch >= 0)
"""

import argparse
import os
import sys
from typing import Tuple, Dict, Any, Optional, Sequence
import torch
import torch.nn as nn
from pathlib import Path
import cv2
import h5py


# Add the project root to sys.path so that 'src' can be imported
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.constants import get_full_path
from src.models.timmEncoders_histo import get_timm_he_encoder, ImageEncoderTimm
from src.models.mlp_head_histo import HEEncoderWithMLP
from src.models.lora import insert_lora
from src.utils.models_settings import BACKBONE_TRANSFORM



# --------------------------------------------------------------------------- H&E encoder


class _BaseHEClipEncoder(nn.Module):
    """
    H&E encoder for inference: image_encoder (backbone + optional MLP head), image_proj (CLIP projection)
    and optional img_tok_proj (token projection). Expects preprocessed input; no L2-normalization inside.
    """

    def __init__(
        self,
        image_encoder: nn.Module,
        image_feature_dim: int,
        embed_dim: int,
        has_image_proj: bool = True,
        image_proj_bias: bool = False,
        img_token_dim: Optional[int] = None,
        token_embed_dim: Optional[int] = None,
        backbone: Optional[str] = None,
        resize_size: int = 0,
        crop_size: int = 0,
        mean: Optional[torch.Tensor] = None,
        std: Optional[torch.Tensor] = None,
        interpolation_code: int = cv2.INTER_LINEAR,
    ) -> None:
        
        super().__init__()
        self.image_encoder = image_encoder
        self.image_feature_dim = int(image_feature_dim)
        self.embed_dim = int(embed_dim)
        self.has_image_proj = bool(has_image_proj)
        self.has_image_proj_bias = bool(image_proj_bias) and self.has_image_proj

        # Preprocessing is external; only its metadata is stored here.
        self.expects_preprocessed: bool = True
        self.backbone: str = str(backbone).lower() if backbone is not None else ""
        self.resize_size: int = int(resize_size)
        self.crop_size: int = int(crop_size)
        self.interpolation_code: int = int(interpolation_code)
        if mean is None or std is None:
            # Keep buffers always present for easy downstream access
            mean = torch.zeros((1, 3, 1, 1), dtype=torch.float32)
            std = torch.ones((1, 3, 1, 1), dtype=torch.float32)
        self.register_buffer("mean", mean.to(dtype=torch.float32))
        self.register_buffer("std", std.to(dtype=torch.float32))

        # Projection to CLIP space: no bias in training, bias for external encoders (e.g. SEAL).
        # Absent for backbone-only exports (checkpoint_epoch < 0).
        if self.has_image_proj:
            self.image_proj = nn.Linear(self.image_feature_dim, self.embed_dim, bias=self.has_image_proj_bias)
            nn.init.normal_(self.image_proj.weight, std=self.image_feature_dim ** -0.5)
            if self.has_image_proj_bias:
                nn.init.zeros_(self.image_proj.bias)
        else:
            self.image_proj = None

        # Optional token head
        self.has_img_tok_proj = (img_token_dim is not None and token_embed_dim is not None)
        if img_token_dim is not None and token_embed_dim is not None:
            # Matches training: bias=False
            self.img_tok_proj = nn.Linear(int(img_token_dim), int(token_embed_dim), bias=False)
            nn.init.normal_(self.img_tok_proj.weight, std=int(img_token_dim) ** -0.5)
        else:
            self.img_tok_proj = None

    def forward(
        self,
        x: torch.Tensor,
        project: bool = True,
    ) -> torch.Tensor:
        """
        Encode H&E patches.

        Args:
            x: input tensor [B, 3, H, W]
            project: if True (default), apply CLIP projection (`image_proj`)
                     if False, return features before the CLIP projection (backbone + optional MLP head).

        Returns:
            Tensor of shape:
              - [B, embed_dim] if project=True
              - [B, image_feature_dim] if project=False
        """
        feats = self.image_encoder(x)  # [B, image_feature_dim]
        if project:
            if self.image_proj is None:
                raise RuntimeError(
                    "This exported H&E encoder has no `image_proj` (checkpoint_epoch < 0 or "
                    "lambda_patch=0). Call encode_image(..., project=False)."
                )
            feats = self.image_proj(feats)
        return feats

    def encode_image(
        self,
        x: torch.Tensor,
        project: bool = True,
    ) -> torch.Tensor:
        """Alias of forward(x, project=...)"""
        return self.forward(x, project=project)


class HEClipEncoderNoTokens(_BaseHEClipEncoder):
    @torch.no_grad()
    def encode_with_tokens(
        self,
        x: torch.Tensor,
        project_img: bool = True,
        project_tokens: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Tuple[int, int]]:
        raise RuntimeError("Underlying image_encoder has no `encode_with_tokens` method.")


class HEClipEncoderWithTokens(_BaseHEClipEncoder):
    @torch.no_grad()
    def encode_with_tokens(
        self,
        x: torch.Tensor,
        project_img: bool = True,
        project_tokens: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Tuple[int, int]]:
        """
        Returns (img_embedding, token_embeddings, grid_hw) if the backbone exposes encode_with_tokens.

        Args:
            x: input tensor [B, 3, H, W]
            project_img: apply the CLIP projection (image_proj) to image-level features
            project_tokens: apply the token projection (img_tok_proj) to token features

        Returns:
            - img_embedding: [B, embed_dim] if project_img else [B, image_feature_dim]
            - token_embeddings: [B, Nt, token_embed_dim] if project_tokens else [B, Nt, img_token_dim],
              or None if no token head was exported
            - grid_hw: (Ht, Wt) token grid size
        """
        feats, tokens, grid_hw = self.image_encoder.encode_with_tokens(x)  # feats: [B,C], tokens: [B,Nt,Ct]
        
        img_emb = feats
        if project_img:
            if self.image_proj is None:
                raise RuntimeError(
                    "This exported H&E encoder has no `image_proj` (checkpoint_epoch < 0 or "
                    "lambda_patch=0). Call encode_with_tokens(..., project_img=False, ...)."
                )
            img_emb = self.image_proj(img_emb)

        tok_emb: Optional[torch.Tensor] = None
        if tokens is not None:
            tok_emb = tokens
            if project_tokens:
                if self.img_tok_proj is None:
                    raise RuntimeError(
                        "This exported H&E encoder does not contain `img_tok_proj` "
                        "(pretrained backbone-only export, or token CLIP head absent). "
                        "Call encode_with_tokens(..., project_tokens=False)."
                    )
                tok_emb = self.img_tok_proj(tok_emb)

        return img_emb, tok_emb, grid_hw



def _is_nullish(x: Any) -> bool:
    if x is None:
        return True
    if isinstance(x, str):
        return x.strip().lower() in {"", "none", "null"}
    return False



def _validate_clean_foundation_he_export(
    config: Dict[str, Any],
    full_state_dict: Optional[Dict[str, torch.Tensor]] = None,
) -> None:
    """
    Check that the H&E export is the raw foundation backbone (checkpoint_epoch < 0):
    no H&E MLP head, no LoRA, no pretrained HistoST checkpoint.
    """
    errs = []

    he_cfg = config["model"]["he"]
    training_cfg = config["training"]
    model_cfg = config["model"]

    if bool(he_cfg.get("use_mlp_head", False)):
        errs.append("model.he.use_mlp_head=True (would add an untrained H&E MLP head).")

    he_training_option = str(training_cfg.get("HE_training_option", "")).strip().lower()
    if he_training_option == "lora":
        errs.append("training.HE_training_option='lora', which would insert LoRA modules into the H&E encoder.")
    elif he_training_option not in {"freeze", "train"}:
        errs.append(f"training.HE_training_option={he_training_option!r} is not supported for a clean foundational export. Use 'freeze' or 'train'.")

    if not _is_nullish(model_cfg.get("pretrained_histost", None)):
        errs.append("model.pretrained_histost is set, so this is not a raw foundational H&E baseline.")

    if full_state_dict is not None:
        lora_keys = sorted(k for k in full_state_dict.keys() if "lora" in k.lower())
        if lora_keys:
            preview = ", ".join(lora_keys[:8])
            if len(lora_keys) > 8:
                preview += ", ..."
            errs.append(
                "model_state_dict already contains LoRA tensors, which is incompatible with a strict foundational "
                f"H&E export. Example keys: {preview}"
            )

    if errs:
        raise RuntimeError(
            "checkpoint_epoch < 0 is reserved for exporting the raw foundation H&E backbone.\n"
            + "\n".join(f"  - {e}" for e in errs)
        )



def _assert_he_backbone_matches_fresh_foundation(
    exported_image_encoder: nn.Module,
    config: Dict[str, Any],
) -> None:
    """Check that the exported H&E backbone is identical to a freshly built foundation backbone (checkpoint_epoch < 0)."""
    reference_encoder, _ = build_image_encoder_from_config(
        config,
        force_foundation_baseline=True,
    )

    reference_sd = reference_encoder.state_dict()
    exported_sd = exported_image_encoder.state_dict()

    if set(reference_sd.keys()) != set(exported_sd.keys()):
        missing = sorted(set(reference_sd.keys()) - set(exported_sd.keys()))
        extra = sorted(set(exported_sd.keys()) - set(reference_sd.keys()))
        raise RuntimeError(
            "Foundational H&E export sanity-check failed: exported backbone structure differs from the raw "
            "foundation backbone.\n"
            f"  Missing keys: {missing[:20]}\n"
            f"  Extra keys: {extra[:20]}"
        )

    mismatched = []
    for k in reference_sd.keys():
        ref_v = reference_sd[k].detach().cpu()
        got_v = exported_sd[k].detach().cpu()
        if ref_v.shape != got_v.shape or not torch.equal(ref_v, got_v):
            mismatched.append(k)

    if mismatched:
        preview = ", ".join(mismatched[:20])
        if len(mismatched) > 20:
            preview += ", ..."
        raise RuntimeError(
            "Foundational H&E export sanity-check failed: checkpoint_epoch < 0 did not reconstruct the exact raw "
            f"foundation backbone. First mismatched tensors: {preview}"
        )




def build_image_encoder_from_config(
    config: Dict[str, Any],
    force_foundation_baseline: bool = False,
) -> Tuple[nn.Module, int]:
    """
    Rebuild the H&E backbone + optional MLP head (+ LoRA) from the checkpoint config, as in Experiment.get_model.
    With force_foundation_baseline, only the raw backbone is built.

    Returns:
        image_encoder: nn.Module
        image_feature_dim: int (output dim before the CLIP projection)
    """
    he_cfg = config["model"]["he"]
    training_cfg = config["training"]

    if force_foundation_baseline:
        _validate_clean_foundation_he_export(config)

    backbone = he_cfg["backbone_img_encoder"].lower()
    print(f"Rebuilding H&E backbone: {backbone}")

    # --- Backbone -----------------------------------------------------------
    if backbone in ["virchow2", "uni2h", "hoptimus0", "hoptimus1"]:
        ckpt_folder = get_full_path('train', 'pretrained_foundation_models', as_str=True)
        ckpt_path = os.path.join(ckpt_folder, f"{backbone}/pytorch_model.bin")
        model_timm, _, extract_embedding = get_timm_he_encoder(
            ckpt_path=ckpt_path,
            model_name=backbone,
            map_location="cpu",
        )
        image_encoder = ImageEncoderTimm(
            vision_encoder=model_timm,
            extract_embedding=extract_embedding,
            model_name=backbone,
        )
        image_feature_dim = int(image_encoder.output_dim)

    else:
        raise ValueError(f"Unknown backbone_img_encoder='{backbone}' in config.")

    # --- Optional H&E MLP head
    if he_cfg["use_mlp_head"] and not force_foundation_baseline:
        he_mlp_hidden_dim = he_cfg["he_mlp_hidden_dim"]
        he_mlp_out_dim = he_cfg["he_mlp_out_dim"]
        he_mlp_dropout = he_cfg["he_mlp_dropout"]
        print(
            f"Adding H&E MLP head "
            f"(in={image_feature_dim} -> hidden={he_mlp_hidden_dim} -> out={he_mlp_out_dim}, "
            f"dropout={he_mlp_dropout})"
        )
        image_encoder = HEEncoderWithMLP(
            backbone=image_encoder,
            in_dim=image_feature_dim,
            hidden_dim=he_mlp_hidden_dim,
            out_dim=he_mlp_out_dim,
            dropout=he_mlp_dropout,
        )
        image_feature_dim = int(he_mlp_out_dim)

    # --- LoRA (if used) after optional MLP head, to mirror Experiment.get_model
    he_training_option = training_cfg["HE_training_option"]
    if he_training_option == "lora":
        if force_foundation_baseline:
            raise RuntimeError("Internal error: force_foundation_baseline=True should have blocked LoRA already.")

        rank = training_cfg["lora_rank"]
        alpha = training_cfg["lora_alpha"]
        print(f"Inserting LoRA adapters (rank={rank}, alpha={alpha}) into image encoder.")
        insert_lora(image_encoder, rank, alpha)

    return image_encoder, image_feature_dim



def patch_projections_are_untrained(clip_cfg: Dict[str, Any]) -> bool:
    """
    True if lambda_patch == 0: the patch projections (image_proj, st_proj) are then untrained and not
    exported. A missing lambda_patch counts as False.
    """
    lambda_patch = clip_cfg.get("lambda_patch", None)
    if lambda_patch is None:
        return False
    return float(lambda_patch) == 0.0



def build_he_clip_encoder_from_loaded(
    config: Dict[str, Any],
    full_state_dict: Dict[str, torch.Tensor],
    device: torch.device,
    force_no_clip_projections: bool = False,
    projection_flags: Optional[Dict[str, bool]] = None,
    force_foundation_baseline: bool = False,
) -> nn.Module:

    # --- Build H&E backbone (+ MLP, LoRA)
    he_cfg = config["model"]["he"]
    if force_foundation_baseline:
        print(
            "Foundation H&E export: rebuilding the backbone only "
            "(no H&E MLP head, no LoRA, no CLIP projections)."
        )
        _validate_clean_foundation_he_export(config, full_state_dict=full_state_dict)

    image_encoder, image_feature_dim = build_image_encoder_from_config(
        config,
        force_foundation_baseline=force_foundation_baseline,
    )

    # --- CLIP embedding dims
    clip_cfg = config["model"]["clip"]
    embed_dim = int(clip_cfg["clip_embed_dim"])

    lambda_token = float(clip_cfg["lambda_token"])
    if lambda_token == 0.0:
        print("WARNING: lambda_token=0.0 in config, so no token head will be exported.")

    # Projections: from the bundle flags, else none if checkpoint_epoch < 0, else from the state_dict tensors
    if projection_flags is not None:
        has_image_proj = bool(projection_flags.get("has_image_proj", False))
        has_img_tok_proj = bool(projection_flags.get("has_img_tok_proj", False))
    elif force_no_clip_projections:
        has_image_proj = False
        has_img_tok_proj = False
    else:
        has_image_proj = _has_ckpt_tensor(full_state_dict, "image_proj.weight")
        has_img_tok_proj = _has_ckpt_tensor(full_state_dict, "img_tok_proj.weight")

    # lambda_patch == 0: image_proj is untrained and dropped, at export and at reload (img_tok_proj is kept)
    if has_image_proj and patch_projections_are_untrained(clip_cfg):
        print(
            "lambda_patch=0.0: `image_proj` was not trained and is dropped (no `clip` space for "
            "this H&E encoder; the `he` space is unaffected)."
        )
        has_image_proj = False

    if not has_image_proj:
        print("INFO: exporting/loading H&E encoder without image_proj.")
    if not has_img_tok_proj:
        print("INFO: exporting/loading H&E encoder without img_tok_proj.")

    # Bias of image_proj: projection_flags["image_proj_has_bias"] (e.g. SEAL), else inferred from the state_dict
    if projection_flags is not None and "image_proj_has_bias" in projection_flags:
        image_proj_bias = bool(projection_flags.get("image_proj_has_bias", False))
    else:
        image_proj_bias = _has_ckpt_tensor(full_state_dict, "image_proj.bias")
    image_proj_bias = bool(image_proj_bias and has_image_proj)
    if image_proj_bias:
        print("INFO: patch image_proj has a bias (affine projection).")

    token_embed_dim = clip_cfg["token_embed_dim"]
    if token_embed_dim is None:
        token_embed_dim = embed_dim
    token_embed_dim = int(token_embed_dim)

    # Token dim from backbone only if a token projection head is actually exported.
    img_token_dim = getattr(image_encoder, "token_dim", None) if has_img_tok_proj else None
    if not has_img_tok_proj:
        token_embed_dim = None

    # --- Store backbone-specific preprocessing metadata (same as validation pipeline)
    backbone = he_cfg["backbone_img_encoder"].lower()
    if backbone not in BACKBONE_TRANSFORM:
        raise ValueError(f"BACKBONE_TRANSFORM does not define preprocessing for backbone '{backbone}'.")
    transform_cfg = BACKBONE_TRANSFORM[backbone]

    resize_size = transform_cfg["resize_size"]
    crop_size = transform_cfg["crop_size"]
    resize_size_i = int(resize_size) if resize_size is not None else 0
    crop_size_i = int(crop_size) if crop_size is not None else 0
    mean = torch.tensor(transform_cfg["mean"], dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor(transform_cfg["std"], dtype=torch.float32).view(1, 3, 1, 1)
    interpolation_code = int(transform_cfg["interpolation"])

    # --- Build compact H&E encoder module
    he_cls = HEClipEncoderWithTokens if hasattr(image_encoder, "encode_with_tokens") else HEClipEncoderNoTokens

    he_model = he_cls(
        image_encoder=image_encoder,
        image_feature_dim=image_feature_dim,
        embed_dim=embed_dim,
        has_image_proj=has_image_proj,
        image_proj_bias=image_proj_bias,
        img_token_dim=img_token_dim,
        token_embed_dim=token_embed_dim,
        backbone=backbone,
        resize_size=resize_size_i,
        crop_size=crop_size_i,
        mean=mean,
        std=std,
        interpolation_code=interpolation_code,
    )

    print(f"HE model architecture built: \n{he_model}")

    # --- Load weights (only the relevant subset)
    print("Loading H&E encoder weights from CLIP checkpoint...")
    he_sd = he_model.state_dict()
    loaded_keys = 0
    skipped_keys = []
    ckpt_keys = list(full_state_dict.keys())

    for k in he_sd.keys():
        ck = _match_ckpt_key(k, ckpt_keys)
        if ck is None:
            skipped_keys.append(k)
            continue
        if he_sd[k].shape != full_state_dict[ck].shape:
            skipped_keys.append(k)
            continue
        he_sd[k] = full_state_dict[ck]
        loaded_keys += 1

    he_model.load_state_dict(he_sd, strict=True)

    print(f"Loaded {loaded_keys} parameter tensors into HE encoder.")
    if skipped_keys:
        print(f"Warning: {len(skipped_keys)} parameters were not found in checkpoint (expected for non-image parts).")
        for k in skipped_keys:
            print("   -", k)

    # Refuse the export if fewer than 90% of the tensors were loaded
    total = len(list(he_model.state_dict().keys()))
    frac = loaded_keys / max(1, total)
    print(f"Load coverage: {loaded_keys}/{total} = {frac:.3f}")
    if frac < 0.90:
        raise RuntimeError(
            "Too few H&E parameters matched the checkpoint (<90%): "
            "key prefixes probably differ from what the exporter expects."
        )

    if force_foundation_baseline:
        _assert_he_backbone_matches_fresh_foundation(he_model.image_encoder, config)
        print("Verified: exported H&E backbone exactly matches the raw foundational model.")

    he_model.to(device)
    he_model.eval()

    return he_model


# --------------------------------------------------------------------------- ST encoders


class STGraphProjEncoder(nn.Module):
    """
    ST graph projection head st_proj (the ST-graph teacher is the precomputed frozen Novae encoder).
    Projects the precomputed st_feat [B, st_feature_dim] to CLIP space (no L2-normalization inside).
    """
    def __init__(self, st_feature_dim: int, embed_dim: int, has_proj: bool = True) -> None:
        super().__init__()
        self.st_feature_dim = int(st_feature_dim)
        self.embed_dim = int(embed_dim)
        self.has_proj = bool(has_proj)

        if self.has_proj:
            self.st_proj = nn.Linear(self.st_feature_dim, self.embed_dim, bias=False)
            nn.init.normal_(self.st_proj.weight, std=self.st_feature_dim ** -0.5)
        else:
            self.st_proj = None

    def forward(self, st_feat: torch.Tensor, project: bool = True) -> torch.Tensor:
        # st_feat: [B, st_feature_dim] precomputed frozen-Novae per-patch teacher output.
        if project:
            if self.st_proj is None:
                raise RuntimeError(
                    "This exported ST graph encoder has no `st_proj` (checkpoint_epoch < 0 or "
                    "lambda_patch=0). Call encode_graph(..., project=False)."
                )
            return self.st_proj(st_feat)
        return st_feat

    def encode_graph(self, st_feat: torch.Tensor, project: bool = True) -> torch.Tensor:
        return self.forward(st_feat, project=project)



class STCellClipEncoder(nn.Module):
    """
    ST cell encoder of the token branch: takes scConcept cell embeddings and optionally applies
    the CLIP token projection st_node_proj (bias=False). No lookup and no preprocessing inside.
    """
    def __init__(self, in_dim: int, out_dim: int, has_proj: bool) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.has_proj = bool(has_proj)
        if self.has_proj:
            self.st_node_proj = nn.Linear(self.in_dim, self.out_dim, bias=False)
            nn.init.normal_(self.st_node_proj.weight, std=self.in_dim ** -0.5)
        else:
            self.st_node_proj = None

    def forward(self, x: torch.Tensor, project: bool = True) -> torch.Tensor:
        # x: [..., in_dim] (e.g. [B,K,in_dim] or [N,in_dim])
        if project:
            if not self.has_proj or self.st_node_proj is None:
                raise RuntimeError(
                    "This exported ST cell encoder has no `st_node_proj` (checkpoint_epoch < 0 or "
                    "lambda_token=0). Call encode_cell(..., project=False)."
                )
            return self.st_node_proj(x)
        return x

    def encode_cell(self, x: torch.Tensor, project: bool = True) -> torch.Tensor:
        return self.forward(x, project=project)
    


def _infer_st_feature_dim(config: Dict[str, Any], full_state_dict: Dict[str, torch.Tensor]) -> int:
    """ST graph teacher feature dim (st_proj input): from st_proj.weight, else model.st.st_feature_dim, else 128."""
    wk = _match_ckpt_key("st_proj.weight", list(full_state_dict.keys()))
    if wk is not None:
        return int(full_state_dict[wk].shape[1])
    cfg_dim = config.get("model", {}).get("st", {}).get("st_feature_dim", None)
    return int(cfg_dim) if cfg_dim else 128


def build_st_graph_clip_encoder_from_loaded(
    config: Dict[str, Any],
    full_state_dict: Dict[str, torch.Tensor],
    device: torch.device,
    force_no_clip_projections: bool = False,
    projection_flags: Optional[Dict[str, bool]] = None,
) -> STGraphProjEncoder:
    """Build the ST graph projection head (st_proj only) and load its weight from the CLIP checkpoint."""
    clip_cfg = config["model"]["clip"]
    embed_dim = int(clip_cfg["clip_embed_dim"])
    st_feature_dim = _infer_st_feature_dim(config, full_state_dict)

    if projection_flags is not None:
        has_proj = bool(projection_flags.get("has_st_proj", False))
    elif force_no_clip_projections:
        has_proj = False
    else:
        has_proj = _has_ckpt_tensor(full_state_dict, "st_proj.weight")

    # lambda_patch == 0: st_proj is untrained and not exported
    if has_proj and patch_projections_are_untrained(clip_cfg):
        print(
            "lambda_patch=0.0: `st_proj` was not trained and is not exported "
            "(the ST graph bundle has no projection)."
        )
        has_proj = False

    st_graph_model = STGraphProjEncoder(
        st_feature_dim=st_feature_dim,
        embed_dim=embed_dim,
        has_proj=has_proj,
    )
    print(f"ST Graph projection head built (st_feature_dim={st_feature_dim}, embed_dim={embed_dim}):\n{st_graph_model}")

    if has_proj:
        print("Loading st_proj weight from CLIP checkpoint...")
        sd = st_graph_model.state_dict()
        ckpt_keys = list(full_state_dict.keys())
        loaded_keys = 0
        skipped_keys = []
        for k in sd.keys():
            ck = _match_ckpt_key(k, ckpt_keys)
            if ck is None or sd[k].shape != full_state_dict[ck].shape:
                skipped_keys.append(k)
                continue
            sd[k] = full_state_dict[ck]
            loaded_keys += 1
        st_graph_model.load_state_dict(sd, strict=True)

        total = len(list(st_graph_model.state_dict().keys()))
        frac = loaded_keys / max(1, total)
        print(f"Loaded {loaded_keys}/{total} tensor(s) into ST graph projection head (coverage={frac:.3f}).")
        if frac < 0.99:
            raise RuntimeError(
                "st_proj did not fully match the checkpoint; export aborted."
            )
    else:
        print("INFO: exporting/loading ST graph encoder without st_proj (identity-only head).")

    st_graph_model.to(device)
    st_graph_model.eval()
    return st_graph_model



def build_st_cell_clip_encoder_from_loaded(
    config: Dict[str, Any],
    full_state_dict: Dict[str, torch.Tensor],
    device: torch.device,
    force_no_clip_projections: bool = False,
    projection_flags: Optional[Dict[str, bool]] = None,
) -> STCellClipEncoder:
    """
    Build ST cell encoder for token branch:
      - identity on pretrained scConcept embeddings
      - optional learned st_node_proj (only if lambda_token>0 during training)
    """
    clip_cfg = config["model"]["clip"]
    embed_dim = int(clip_cfg["clip_embed_dim"])
    lambda_token = float(clip_cfg["lambda_token"])
    token_embed_dim = clip_cfg["token_embed_dim"]
    if token_embed_dim is None:
        token_embed_dim = embed_dim
    token_embed_dim = int(token_embed_dim)

    # In training, st_node_dim is scConcept embedding dim (panel-independent)
    in_dim, in_dim_source = infer_scconcept_embedding_dim(config, full_state_dict)
    print(f"Inferred scConcept embedding dim = {in_dim} from {in_dim_source}.")
    if projection_flags is not None:
        has_proj = bool(projection_flags.get("has_st_node_proj", False))
    elif force_no_clip_projections:
        has_proj = False
    else:
        has_proj = _has_ckpt_tensor(full_state_dict, "st_node_proj.weight")
    
    # lambda_token=0.0: the checkpoint should have no token head
    if lambda_token == 0.0 and has_proj:
        raise RuntimeError(
            "lambda_token=0.0 in the config, but st_node_proj weights were found in the checkpoint: "
            "the checkpoint does not match the config."
        )

    st_cell_model = STCellClipEncoder(in_dim=in_dim, out_dim=token_embed_dim, has_proj=has_proj)
    print(f"ST Cell model architecture built:\n{st_cell_model}")

    if has_proj:
        print("Loading ST cell token-projection weights (st_node_proj) from CLIP checkpoint...")
        sd = st_cell_model.state_dict()
        ckpt_keys = list(full_state_dict.keys())
        loaded_keys = 0
        skipped_keys = []
        for k in sd.keys():
            ck = _match_ckpt_key(k, ckpt_keys)
            if ck is None:
                skipped_keys.append(k)
                continue
            if sd[k].shape != full_state_dict[ck].shape:
                skipped_keys.append(k)
                continue
            sd[k] = full_state_dict[ck]
            loaded_keys += 1
        st_cell_model.load_state_dict(sd, strict=True)

        total = len(list(st_cell_model.state_dict().keys()))
        frac = loaded_keys / max(1, total)
        print(f"Loaded {loaded_keys}/{total} tensors into ST cell encoder (coverage={frac:.3f}).")
        if frac < 0.99:
            raise RuntimeError(
                "ST cell projection head did not fully match the checkpoint; export aborted."
            )
    else:
        print("INFO: exporting/loading ST cell encoder without st_node_proj (identity-only encoder).")

    st_cell_model.to(device)
    st_cell_model.eval()
    return st_cell_model



# --------------------------------------------------------------------------- Main pipeline


def load_checkpoint(ckpt_path: str, device: torch.device) -> Tuple[Dict[str, Any], Dict[str, torch.Tensor]]:
    """
    Load checkpoint, return (config, model_state_dict).
    """
    print(f"Loading checkpoint from: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    if "config" not in checkpoint:
        raise KeyError("Checkpoint does not contain 'config'")
    if "model_state_dict" not in checkpoint:
        raise KeyError("Checkpoint does not contain 'model_state_dict'")

    config = checkpoint["config"]
    state_dict = checkpoint["model_state_dict"]
    return config, state_dict



def get_scconcept_training_id_from_config(config: Dict[str, Any]) -> Optional[str]:
    """scConcept training id of the checkpoint config (locates train/trainings/trainings_scConcept/<training_id>/embeddings), or None."""
    st_cfg = config["model"]["st"]
    scconcept_training_id = st_cfg.get("scconcept_training_id", None)
    if scconcept_training_id is None:
        return None
    scconcept_training_id = str(scconcept_training_id).strip()
    if scconcept_training_id == "" or scconcept_training_id.lower() in ("none", "null"):
        return None
    return scconcept_training_id


def get_novae_training_id_from_config(config: Dict[str, Any]) -> Optional[str]:
    """Novae ST-graph teacher training id of the checkpoint config (model.st.novae_training_id), or None."""
    novae_training_id = config.get("model", {}).get("st", {}).get("novae_training_id", None)
    if novae_training_id is None:
        return None
    novae_training_id = str(novae_training_id).strip()
    if novae_training_id == "" or novae_training_id.lower() in ("none", "null"):
        return None
    return novae_training_id



def _resolve_scconcept_embeddings_dir_from_config(config: Dict[str, Any]) -> Optional[Path]:
    """scConcept embeddings dir: model.st.scconcept_embeddings_dir, else resolved from model.st.scconcept_training_id."""
    st_cfg = config["model"]["st"]

    sc_dir = st_cfg.get("scconcept_embeddings_dir", None)
    if sc_dir is not None:
        sc_dir = str(sc_dir).strip()
        if sc_dir and sc_dir.lower() not in ("none", "null"):
            return Path(sc_dir)

    scconcept_training_id = get_scconcept_training_id_from_config(config)
    if scconcept_training_id is None:
        return None

    return Path(
        get_full_path(
            "train",
            "trainings",
            "trainings_scConcept",
            scconcept_training_id,
            "embeddings",
            as_str=True,
        )
    )



def infer_scconcept_embedding_dim(
    config: Dict[str, Any],
    full_state_dict: Dict[str, torch.Tensor],
) -> Tuple[int, str]:
    """
    Infer the scConcept embedding dim (training-time st_node_dim), from model.clip.scconcept_embedding_dim,
    else st_node_proj.weight, else the scConcept .h5 files.
    """
    clip_cfg = config["model"]["clip"]

    # 1) explicit metadata if present
    explicit_dim = clip_cfg.get("scconcept_embedding_dim", None)
    if explicit_dim is not None:
        return int(explicit_dim), "config['model']['clip']['scconcept_embedding_dim']"

    # 2) authoritative source when token head exists: trained projection weight
    ckpt_keys = list(full_state_dict.keys())
    proj_key = _match_ckpt_key("st_node_proj.weight", ckpt_keys)
    if proj_key is not None:
        w = full_state_dict[proj_key]
        if w.ndim != 2:
            raise ValueError(f"Expected 2D tensor for '{proj_key}', got shape {tuple(w.shape)}")
        return int(w.shape[1]), f"checkpoint tensor '{proj_key}'"

    # 3) fallback: reproduce training-time inference from scConcept .h5 files
    sc_dir = _resolve_scconcept_embeddings_dir_from_config(config)
    if sc_dir is None:
        raise KeyError(
            "Could not infer scConcept embedding dim: missing "
            "config['model']['clip']['scconcept_embedding_dim'], missing "
            "st_node_proj.weight in checkpoint, and no "
            "model.st.scconcept_embeddings_dir / model.st.scconcept_training_id "
            "available in config."
        )
    if not sc_dir.is_dir():
        raise FileNotFoundError(f"Resolved scConcept embeddings directory does not exist: {sc_dir}")

    first_h5 = next(sc_dir.glob("*.h5"), None)
    if first_h5 is None:
        raise FileNotFoundError(f"No scConcept .h5 file found in {sc_dir}; cannot infer token embedding dimension.")

    with h5py.File(first_h5, "r") as f:
        if "cls_emb" not in f:
            raise KeyError(f"'cls_emb' dataset not found in {first_h5}")
        dim = int(f.attrs.get("embedding_dim", f["cls_emb"].shape[1]))

    return dim, f"scConcept file '{first_h5.name}'"



def _match_ckpt_key(model_key: str, ckpt_keys: Sequence[str]) -> Optional[str]:
    """Checkpoint key matching an export model key: exact match, else with a common prefix, else shortest suffix match."""
    if model_key in ckpt_keys:
        return model_key

    prefixes = [
        "he_encoder.",
        "model.he_encoder.",
        "module.he_encoder.",
        "model.",
        "module.",
    ]
    for p in prefixes:
        cand = p + model_key
        if cand in ckpt_keys:
            return cand

    # fallback: suffix match (common in Trainer wrappers)
    matches = [k for k in ckpt_keys if k.endswith(model_key)]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        # pick the "closest" (shortest extra prefix)
        matches = sorted(matches, key=len)
        return matches[0]
    return None



def _has_ckpt_tensor(state_dict: Dict[str, torch.Tensor], model_key: str) -> bool:
    """True if state_dict holds a tensor matching model_key (see _match_ckpt_key)."""
    return _match_ckpt_key(model_key, list(state_dict.keys())) is not None



def load_state_dict_bundle(bundle_path: str, device: torch.device) -> Dict[str, Any]:
    """Load an encoder bundle saved by this script (dict with config, state_dict, export_type="state_dict_bundle")."""
    bundle = torch.load(bundle_path, map_location=device, weights_only=False)
    if not isinstance(bundle, dict):
        raise TypeError(f"Expected a dict bundle at '{bundle_path}', got {type(bundle).__name__}.")

    if bundle.get("export_type") != "state_dict_bundle":
        raise ValueError(
            f"File '{bundle_path}' is not a state_dict bundle "
            f"(export_type={bundle.get('export_type', None)!r})."
        )

    missing = [k for k in ("config", "state_dict") if k not in bundle]
    if missing:
        raise KeyError(f"Bundle '{bundle_path}' is missing required keys: {missing}")

    return bundle


def load_exported_he_encoder(bundle_path: str, device: torch.device) -> nn.Module:
    bundle = load_state_dict_bundle(bundle_path, device=device)

    force_foundation_baseline = bool(bundle.get("foundation_baseline_only", False))
    if not force_foundation_baseline:
        ckpt_epoch = bundle.get("checkpoint_epoch", None)
        try:
            force_foundation_baseline = (int(ckpt_epoch) < 0)
        except (TypeError, ValueError):
            force_foundation_baseline = False

    return build_he_clip_encoder_from_loaded(
        bundle["config"],
        bundle["state_dict"],
        device=device,
        projection_flags=bundle.get("projection_flags"),
        force_foundation_baseline=force_foundation_baseline,
    )




def main(args):

    tnum = args.training_id.split('_')[-1]
    enum = args.checkpoint_name.split('_')[-1].split('.')[0]
    checkpoint_epoch = int(enum)
    force_no_clip_projections = (checkpoint_epoch < 0)
    force_foundation_baseline = (checkpoint_epoch < 0)

    if force_no_clip_projections:
        print(f"Checkpoint epoch = {checkpoint_epoch} < 0 -> exporting only the pretrained H&E backbone (ST graph / ST cell encoders are skipped).")
        print("H&E encoder = raw foundation backbone (no CLIP projection head, no H&E MLP head, no LoRA, no pretrained histost checkpoint).")

    # Checkpoint path
    checkpoint_path = get_full_path('train', 'trainings', 'trainings_HistoST', args.training_id, 'training', 'log', 'checkpoints', args.checkpoint_name, as_str=True)
    print(f"Checkpoint path: {checkpoint_path}")

    # Shared load (single checkpoint read)
    config, full_state_dict = load_checkpoint(checkpoint_path, device=torch.device("cpu"))

    # H&E encoder output path
    # training_46, checkpoint_epoch_13.pth -> output: HEencoder_t46e13.pt
    output_HEencoder_path = get_full_path('train', 'trainings', 'trainings_HistoST', args.training_id, 'trained_encoders', f"HEencoder_t{tnum}e{enum}.pt", as_str=True)
    os.makedirs(os.path.dirname(output_HEencoder_path), exist_ok=True)

    # 1. H&E ENCODER EXPORT ----------------
    print("\nBuilding H&E encoder (backbone + optional MLP head + optional CLIP projection) ...")

    he_model = build_he_clip_encoder_from_loaded(
        config,
        full_state_dict,
        device=torch.device("cpu"),
        force_no_clip_projections=force_no_clip_projections,
        force_foundation_baseline=force_foundation_baseline,
    )
    he_model.to(torch.device("cpu"))

    # Save as bundle (config + state_dict)
    he_bundle = {
        "config": config,
        "state_dict": he_model.state_dict(),
        "export_type": "state_dict_bundle",
        "checkpoint_epoch": checkpoint_epoch,
        "projection_flags": {
            "has_image_proj": he_model.image_proj is not None,
            "has_img_tok_proj": getattr(he_model, "img_tok_proj", None) is not None,
            "image_proj_has_bias": bool(getattr(he_model, "has_image_proj_bias", False)),
        },
        "foundation_baseline_only": force_foundation_baseline,
        "notes": "Rebuild with HEClipEncoderWithTokens/HEClipEncoderNoTokens "
                 "+ backbone/MLP/LoRA as specified in config, then load state_dict. "
                 "No preprocessing and no L2-normalization inside. "
                 "If checkpoint_epoch < 0, no CLIP projection heads are exported and the H&E export is forced to be "
                 "the strict clean foundational baseline (backbone only; no H&E MLP head; no LoRA; no pretrained histost checkpoint).",
    }
    torch.save(he_bundle, output_HEencoder_path)
    print(f"Saved HE encoder bundle to: {output_HEencoder_path}")

    # checkpoint_epoch < 0: no trained ST projection, so the ST encoders are not exported
    if force_no_clip_projections:
        print(
            f"\ncheckpoint_epoch = {checkpoint_epoch} < 0 -> skipping ST graph and ST cell encoder exports. "
            "Only the foundational H&E encoder was exported."
        )
        return

    # scConcept provenance for the ST cell branch
    scconcept_training_id = get_scconcept_training_id_from_config(config)
    if scconcept_training_id is not None:
        print(f"scConcept training_id recorded for export: {scconcept_training_id}")
    else:
        print(
            "No scConcept training_id stored in checkpoint config; "
            "export will rely on model.st.scconcept_embeddings_dir or embedded export metadata when needed."
        )

    # Output paths for ST encoders (same folder as HE)
    out_dir = os.path.dirname(output_HEencoder_path)
    output_STgraph_path = os.path.join(out_dir, f"STencoderGraph_t{tnum}e{enum}.pt")
    output_STcell_path  = os.path.join(out_dir, f"STencoderCell_t{tnum}e{enum}.pt")

    # Novae teacher training id (locates novae_teacher_cache/<novae_training_id>/)
    novae_training_id = get_novae_training_id_from_config(config)
    if novae_training_id is not None:
        print(f"Novae training_id recorded for STencoderGraph export: {novae_training_id}")
    else:
        print(
            "WARNING: no model.st.novae_training_id in checkpoint config; the STencoderGraph bundle "
            "will not carry the teacher-cache pointer (_10_compute_STembs.py then needs --novae_training_id)."
        )

    # 2. ST GRAPH ENCODER EXPORT (projection head st_proj only) ----------------
    print("\nBuilding ST graph projection head (st_proj) ...")

    st_graph_model = build_st_graph_clip_encoder_from_loaded(
        config,
        full_state_dict,
        device=torch.device("cpu"),
        force_no_clip_projections=force_no_clip_projections,
    )
    st_graph_model.to(torch.device("cpu"))

    # Save as bundle (config + state_dict + teacher training id)
    st_graph_bundle = {
        "config": config,
        "state_dict": st_graph_model.state_dict(),
        "export_type": "state_dict_bundle",
        "checkpoint_epoch": checkpoint_epoch,
        "projection_flags": {"has_st_proj": st_graph_model.st_proj is not None},
        "scconcept_training_id": scconcept_training_id,
        "novae_training_id": novae_training_id,
        "st_feature_dim": int(st_graph_model.st_feature_dim),
        "embed_dim": int(st_graph_model.embed_dim),
        "notes": "ST-graph teacher is the precomputed frozen Novae v1.1.1 encoder (scConcept node "
                 "features). This bundle carries only the trained st_proj head "
                 "(st_feature_dim=128 -> embed_dim, bias=False) + the novae_training_id pointer. "
                 "The consumer reads per-patch st_feat from novae_teacher_cache/{novae_training_id}/ "
                 "and applies st_proj. No GAT, no L2-normalization inside. "
                 "If checkpoint_epoch < 0, st_proj is not exported.",
    }
    torch.save(st_graph_bundle, output_STgraph_path)
    print(f"Saved ST graph encoder bundle to: {output_STgraph_path}")

    # 3. ST CELL ENCODER EXPORT ----------------
    print("\nBuilding ST cell encoder (scConcept embeddings + optional st_node_proj) ...")
    
    st_cell_model = build_st_cell_clip_encoder_from_loaded(
        config,
        full_state_dict,
        device=torch.device("cpu"),
        force_no_clip_projections=force_no_clip_projections,
    )
    st_cell_model.to(torch.device("cpu"))
    
    st_cell_bundle = {
        "config": config,
        "state_dict": st_cell_model.state_dict(),
        "export_type": "state_dict_bundle",
        "checkpoint_epoch": checkpoint_epoch,
        "projection_flags": {"has_st_node_proj": st_cell_model.has_proj},
        "scconcept_training_id": scconcept_training_id,
        "notes": "This encoder expects pretrained scConcept embeddings as input. "
                 "Resolve them dynamically from train/trainings/trainings_scConcept/{scconcept_training_id}/embeddings. "
                 "Optional projection matches training st_node_proj. No L2-normalization inside. "
                 "If checkpoint_epoch < 0, st_node_proj is not exported.",
    }
    torch.save(st_cell_bundle, output_STcell_path)
    print(f"Saved ST cell encoder bundle to: {output_STcell_path}")



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Export H&E and ST encoders from a CLIP training checkpoint.")
    
    parser.add_argument(
        "--checkpoint_name", type=str, 
        default="checkpoint_epoch_13.pth",
        help="Checkpoint filename, formatted as 'checkpoint_epoch_{epoch}.pth' "
             "(epoch -1: export of the pretrained H&E backbone only)."
    )
    parser.add_argument(
        "--training_id", type=str,
        default="training_46",
        help="HistoST training ID corresponding to the checkpoint (used to locate the checkpoint folder), "
             "formatted as 'training_{number}', for instance 'training_46'",
    )
    
    args = parser.parse_args()
    main(args)
