#!/usr/bin/env python
"""
Build the SEAL UNI2-h encoder (https://huggingface.co/MahmoodLab/SEAL) as a HistoST H&E encoder bundle:
SEAL LoRA merged into the UNI2-h backbone, SEAL projection_head as image_proj (clip space).

Saved in <ROOT_HistoST> / train / trainings / trainings_HistoST / training_<training_num> / trained_encoders /
    - HEencoder_t<training_num>e<checkpoint_num>.pt: H&E encoder bundle (default tag t9001e0)
"""

import argparse
import hashlib
import logging
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch

# Add the project root to sys.path so that 'src' can be imported
project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.constants import get_full_path
from src.utils.default_histost_config import merge_with_defaults
from src.evaluation.extract_encoders import (
    build_image_encoder_from_config,
    build_he_clip_encoder_from_loaded,
    load_exported_he_encoder,
)

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- constants

# SEAL LoRA hyperparameters (MahmoodLab/SEAL conf/config.yaml): scaling = lora_alpha / r = 1.0
SEAL_LORA_RANK = 8
SEAL_LORA_ALPHA = 8

# The SEAL projection maps the 1536-d UNI2-h backbone feature to a 1536-d aligned space.
SEAL_CLIP_EMBED_DIM = 1536

# Keys inside the SEAL delta (under a "state_dict" wrapper).
_LORA_A_RE = re.compile(r"(?:^|\.)base_model\.model\.(.+?)\.lora_A\.[^.]+\.weight$")
_LORA_B_RE = re.compile(r"(?:^|\.)base_model\.model\.(.+?)\.lora_B\.[^.]+\.weight$")
_PROJ_W_KEY_RE = re.compile(r"(?:^|\.)projection_head\.weight$")
_PROJ_B_KEY_RE = re.compile(r"(?:^|\.)projection_head\.bias$")


# --------------------------------------------------------------------------- helpers


def _sha256_file(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _load_seal_state_dict(delta_path: str) -> Dict[str, torch.Tensor]:
    """Load the SEAL delta and return its inner tensor state_dict (unwrapping {"state_dict": ...})."""
    obj = torch.load(delta_path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        sd = obj["state_dict"]
    elif isinstance(obj, dict):
        sd = obj
    else:
        raise TypeError(f"Unexpected SEAL delta object type: {type(obj).__name__}")
    sd = {str(k): v for k, v in sd.items() if hasattr(v, "shape")}
    if not sd:
        raise RuntimeError(f"No tensors found in SEAL delta at {delta_path}")
    return sd


def discover_seal_delta(sd: Dict[str, torch.Tensor]) -> Tuple[Dict[str, Dict[str, torch.Tensor]], torch.Tensor, torch.Tensor]:
    """
    Parse the SEAL delta into:
      - lora_targets: {target_module_path -> {"A": (r,in), "B": (out,r)}}, e.g. "blocks.21.attn.qkv"
      - proj_weight: SEAL projection_head weight (out, in)
      - proj_bias: SEAL projection_head bias (out,)
    """
    a_mats: Dict[str, torch.Tensor] = {}
    b_mats: Dict[str, torch.Tensor] = {}
    proj_weight = None
    proj_bias = None

    for k, v in sd.items():
        m = _LORA_A_RE.search(k)
        if m:
            a_mats[m.group(1)] = v.to(torch.float32)
            continue
        m = _LORA_B_RE.search(k)
        if m:
            b_mats[m.group(1)] = v.to(torch.float32)
            continue
        if _PROJ_W_KEY_RE.search(k):
            proj_weight = v.to(torch.float32)
            continue
        if _PROJ_B_KEY_RE.search(k):
            proj_bias = v.to(torch.float32)
            continue
        # other keys (decoder.*, ...) are ignored

    if set(a_mats) != set(b_mats):
        raise RuntimeError(
            "SEAL LoRA A/B target mismatch:\n"
            f"  A-only: {sorted(set(a_mats) - set(b_mats))}\n"
            f"  B-only: {sorted(set(b_mats) - set(a_mats))}"
        )
    if not a_mats:
        raise RuntimeError("No LoRA A/B pairs discovered in SEAL delta (unexpected).")
    if proj_weight is None:
        raise RuntimeError("SEAL delta has no projection_head.weight (unexpected).")

    lora_targets = {t: {"A": a_mats[t], "B": b_mats[t]} for t in sorted(a_mats)}
    return lora_targets, proj_weight, proj_bias


def merge_seal_lora_into_backbone(
    image_encoder: torch.nn.Module,
    lora_targets: Dict[str, Dict[str, torch.Tensor]],
    scaling: float,
    expected_rank: int,
) -> List[Dict[str, Any]]:
    """
    Fold each SEAL LoRA update into the corresponding base UNI2-h Linear weight, in place:
        W_effective = W_base + scaling * (B @ A)
    Returns a per-target report (shapes, rank, delta norm) for logging / provenance.
    """
    vit = image_encoder.vision_encoder  # timm VisionTransformer
    report: List[Dict[str, Any]] = []

    with torch.no_grad():
        for target, mats in lora_targets.items():
            A = mats["A"]  # (r, in)
            B = mats["B"]  # (out, r)
            r = int(A.shape[0])
            if r != int(B.shape[1]):
                raise RuntimeError(f"{target}: inconsistent LoRA rank A={tuple(A.shape)} B={tuple(B.shape)}")
            if r != int(expected_rank):
                raise RuntimeError(
                    f"{target}: discovered LoRA rank {r} != expected {expected_rank}. "
                    "Refuse to merge with a mismatched rank/scaling (pass --lora_rank to override)."
                )

            try:
                module = vit.get_submodule(target)
            except AttributeError as e:
                raise RuntimeError(
                    f"SEAL LoRA target '{target}' not found in the instantiated uni2h backbone. "
                    "Key-alignment failed -- the SEAL delta and this repo's uni2h do not match."
                ) from e

            weight = getattr(module, "weight", None)
            if weight is None:
                raise RuntimeError(f"SEAL LoRA target '{target}' has no .weight (not a Linear?).")

            delta_w = scaling * (B @ A)  # (out, in)
            if tuple(delta_w.shape) != tuple(weight.shape):
                raise RuntimeError(
                    f"{target}: LoRA delta shape {tuple(delta_w.shape)} != base weight {tuple(weight.shape)}."
                )

            delta_norm = float(torch.linalg.norm(delta_w).item())
            if not (delta_norm > 0.0):
                raise RuntimeError(f"{target}: zero LoRA delta norm -- nothing would be merged.")

            weight.add_(delta_w.to(weight.dtype))
            report.append({
                "target": target,
                "weight_shape": tuple(int(s) for s in weight.shape),
                "rank": r,
                "delta_frobenius_norm": delta_norm,
            })

    return report


def build_seal_he_config() -> Dict[str, Any]:
    """
    HistoST config of the SEAL encoder: uni2h backbone, no H&E MLP head, HE_training_option="freeze"
    (LoRA merged), clip_embed_dim = 1536; other keys from DEFAULT_HISTOST_CONFIG.
    """
    overrides = {
        "model": {
            "he": {
                "backbone_img_encoder": "uni2h",
                "use_mlp_head": False,
            },
            "clip": {
                "clip_embed_dim": SEAL_CLIP_EMBED_DIM,
            },
            "pretrained_histost": None,
        },
        "training": {
            "HE_training_option": "freeze",
        },
    }
    return merge_with_defaults(overrides)


def _embeddings(model: torch.nn.Module, x: torch.Tensor) -> Dict[str, torch.Tensor]:
    model.eval()
    with torch.no_grad():
        out = {"he": model.encode_image(x, project=False)}
        if getattr(model, "image_proj", None) is not None:
            out["clip"] = model.encode_image(x, project=True)
    return out


# --------------------------------------------------------------------------- main


def main(args: argparse.Namespace) -> None:
    training_num = int(args.training_num)
    checkpoint_num = int(args.checkpoint_num)
    if checkpoint_num < 0:
        raise SystemExit(
            f"checkpoint_num must be >= 0 (got {checkpoint_num}): a negative checkpoint is reserved "
            "for the raw UNI2-h backbone."
        )

    scaling = float(args.lora_alpha) / (
        (float(args.lora_rank) ** 0.5) if args.use_rslora else float(args.lora_rank)
    )

    delta_path = args.seal_delta_path or get_full_path(
        "train", "pretrained_foundation_models", "seal", "seal_univ2_vision.pth", as_str=True
    )
    if not os.path.exists(delta_path):
        raise SystemExit(f"SEAL delta not found: {delta_path}")

    logger.info("SEAL delta: %s", delta_path)
    logger.info("LoRA scaling = alpha/%s = %.6g (rank=%d, alpha=%d, use_rslora=%s)",
                "sqrt(r)" if args.use_rslora else "r", scaling, args.lora_rank, args.lora_alpha, args.use_rslora)

    # --- 1. Parse the SEAL delta -------------------------------------------------------
    seal_sd = _load_seal_state_dict(delta_path)
    lora_targets, proj_weight, proj_bias = discover_seal_delta(seal_sd)
    logger.info("Discovered %d LoRA targets: %s", len(lora_targets), ", ".join(lora_targets))
    logger.info("SEAL projection_head weight %s; bias %s (affine `clip` projection -- weight and bias applied).",
                tuple(proj_weight.shape), tuple(proj_bias.shape) if proj_bias is not None else None)
    if tuple(proj_weight.shape) != (SEAL_CLIP_EMBED_DIM, SEAL_CLIP_EMBED_DIM):
        raise SystemExit(f"Unexpected SEAL projection_head weight shape {tuple(proj_weight.shape)}.")

    # --- 2. Base uni2h backbone, then merge the LoRA -----------------------------------
    config = build_seal_he_config()
    logger.info("Rebuilding base uni2h backbone (freeze; no live LoRA) ...")
    image_encoder, image_feature_dim = build_image_encoder_from_config(config, force_foundation_baseline=False)
    if int(image_feature_dim) != SEAL_CLIP_EMBED_DIM:
        raise SystemExit(f"Base uni2h feature dim {image_feature_dim} != {SEAL_CLIP_EMBED_DIM}.")

    logger.info("Merging SEAL LoRA into the backbone weights ...")
    merge_report = merge_seal_lora_into_backbone(image_encoder, lora_targets, scaling, args.lora_rank)
    for r in merge_report:
        logger.info("  merged %-24s W=%s  ||dW||=%.4g", r["target"], r["weight_shape"], r["delta_frobenius_norm"])

    # State dict loaded by build_he_clip_encoder_from_loaded: merged backbone ("image_encoder." prefix) + projection
    full_state_dict: Dict[str, torch.Tensor] = {
        f"image_encoder.{k}": v for k, v in image_encoder.state_dict().items()
    }
    full_state_dict["image_proj.weight"] = proj_weight
    if proj_bias is not None:
        full_state_dict["image_proj.bias"] = proj_bias  # SEAL's affine projection bias -> exact `clip`
    del image_encoder  # free the merged backbone; build_he_clip_encoder_from_loaded rebuilds+loads a fresh one

    # --- 3. Wrap as a normal export -----------------------------------------------------
    logger.info("Wrapping into the HistoST H&E encoder (image_proj from SEAL projection; no token proj) ...")
    he_model = build_he_clip_encoder_from_loaded(
        config,
        full_state_dict,
        device=torch.device("cpu"),
        projection_flags={"has_image_proj": True, "has_img_tok_proj": False, "image_proj_has_bias": proj_bias is not None},
        force_foundation_baseline=False,
    )
    he_model.to("cpu").eval()

    # Structural assertions on the encoder that was just built.
    assert he_model.image_proj is not None, "image_proj must exist (clip space)."
    assert getattr(he_model, "img_tok_proj", None) is None, "token-branch projection must be absent."
    assert he_model.backbone == "uni2h", f"backbone must be uni2h, got {he_model.backbone!r}."
    if proj_bias is not None:
        assert getattr(he_model, "has_image_proj_bias", False) and he_model.image_proj.bias is not None, \
            "clip projection must carry SEAL's projection bias (affine image_proj)."
        assert torch.allclose(he_model.image_proj.bias.detach().cpu(), proj_bias.detach().cpu(), atol=1e-6), \
            "image_proj.bias must equal SEAL's projection_head.bias (loaded, not zero-init)."

    # --- 4. Build the bundle -----------------------------------------------------------
    bundle = {
        "config": config,
        "state_dict": he_model.state_dict(),
        "export_type": "state_dict_bundle",
        "checkpoint_epoch": checkpoint_num,
        "projection_flags": {"has_image_proj": True, "has_img_tok_proj": False, "image_proj_has_bias": proj_bias is not None},
        "foundation_baseline_only": False,
        "notes": (
            "SEAL (MahmoodLab) UNI2-h ST-pretrained baseline, tag t{tn}e{cn}. Built by "
            "src/evaluation/seal_comparison/build_seal_he_encoder.py -- NOT from a HistoST checkpoint. "
            "Base uni2h with SEAL LoRA MERGED into the backbone (no residual LoRA tensors); "
            "image_proj = SEAL projection_head (affine: weight + bias) so `clip` = SEAL's aligned space; no token projection. "
            "Treated as a finetuned encoder so both `he` (project=False) and `clip` (project=True) spaces exist."
        ).format(tn=training_num, cn=checkpoint_num),
        "seal_provenance": {
            "source_repo": "MahmoodLab/SEAL",
            "delta_file": os.path.basename(delta_path),
            "delta_sha256": _sha256_file(delta_path),
            "lora_rank": int(args.lora_rank),
            "lora_alpha": int(args.lora_alpha),
            "use_rslora": bool(args.use_rslora),
            "lora_scaling": scaling,
            "merged_targets": merge_report,
            "projection_head_weight_shape": tuple(int(s) for s in proj_weight.shape),
            "projection_head_bias_applied": proj_bias is not None,
            "projection_head_bias_shape": (tuple(int(s) for s in proj_bias.shape) if proj_bias is not None else None),
        },
    }

    # --- 5. Save + reload round-trip verification --------------------------------------
    if args.dry_run:
        out_dir = tempfile.mkdtemp(prefix="seal_he_dryrun_")
        output_path = os.path.join(out_dir, f"HEencoder_t{training_num}e{checkpoint_num}.pt")
        logger.info("[dry-run] writing to temp path (not ROOT_PATH): %s", output_path)
    else:
        output_path = get_full_path(
            "train", "trainings", "trainings_HistoST", f"training_{training_num}",
            "trained_encoders", f"HEencoder_t{training_num}e{checkpoint_num}.pt", as_str=True,
        )
        if os.path.exists(output_path) and not args.overwrite:
            raise SystemExit(f"Refusing to overwrite existing bundle (pass --overwrite): {output_path}")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    torch.save(bundle, output_path)
    logger.info("Saved SEAL H&E encoder bundle to: %s", output_path)

    # Reload with load_exported_he_encoder and check that the embeddings are identical
    logger.info("Reloading via load_exported_he_encoder and verifying round-trip ...")
    reloaded = load_exported_he_encoder(output_path, device=torch.device("cpu")).eval()
    x = torch.randn(3, 3, int(he_model.crop_size or 224), int(he_model.crop_size or 224), dtype=torch.float32)
    emb_before = _embeddings(he_model, x)
    emb_after = _embeddings(reloaded, x)
    for space in emb_before:
        a, b = emb_before[space], emb_after[space]
        if not torch.allclose(a, b, atol=1e-6, rtol=0):
            raise RuntimeError(f"Round-trip mismatch in `{space}` space (max abs diff {float((a-b).abs().max()):.3e}).")
        if not torch.isfinite(a).all():
            raise RuntimeError(f"Non-finite embeddings in `{space}` space.")
        logger.info("  `%s` space OK: dim=%d, round-trip identical.", space, int(a.shape[1]))

    logger.info(
        "DONE. Tag t%de%d ready. Next: Task B _0 (--training_num %d --checkpoint_num %d), "
        "then _2 --space {he,clip}.",
        training_num, checkpoint_num, training_num, checkpoint_num,
    )
    if args.dry_run:
        logger.info("[dry-run] Nothing was written under ROOT_PATH. Re-run without --dry_run to export for real.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="Export the SEAL UNI2-h ST-pretrained baseline as a HistoST H&E encoder bundle (HEencoder_t9001e0.pt).",
    )
    parser.add_argument(
        "--seal_delta_path", type=str, default=None,
        help="Path to seal_univ2_vision.pth. Default: "
             "<ROOT>/train/pretrained_foundation_models/seal/seal_univ2_vision.pth (via get_full_path).",
    )
    parser.add_argument(
        "--training_num", type=int, default=9001,
        help="training_num of the SEAL encoder tag (default 9001).",
    )
    parser.add_argument(
        "--checkpoint_num", type=int, default=0,
        help="Checkpoint number of the tag (default 0, must be >= 0).",
    )
    parser.add_argument(
        "--lora_rank", type=int, default=SEAL_LORA_RANK,
        help=f"SEAL LoRA rank r (default {SEAL_LORA_RANK}; asserted against the delta's A/B shapes).",
    )
    parser.add_argument(
        "--lora_alpha", type=int, default=SEAL_LORA_ALPHA,
        help=f"SEAL LoRA alpha (default {SEAL_LORA_ALPHA}). Scaling = alpha/r (or alpha/sqrt(r) with --use_rslora).",
    )
    parser.add_argument(
        "--use_rslora", action="store_true",
        help="Use rank-stabilized LoRA scaling alpha/sqrt(r). SEAL uses use_rslora=False, so leave unset.",
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Build and verify into a temp dir (nothing is written under ROOT_PATH).",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Allow overwriting an existing HEencoder bundle at the target path.",
    )

    main(parser.parse_args())
