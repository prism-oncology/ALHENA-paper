#!/usr/bin/env python
"""
Rebuild the encoder bundles of the released ALHENA model from its Hugging Face release
(https://huggingface.co/prism-oncology/ALHENA), so that the evaluation scripts run on it without the
training checkpoint.

The release holds the LoRA adapters and the projection heads, not the UNI2-h weights. They are combined
with the local UNI2-h weights (train/pretrained_foundation_models/uni2h/pytorch_model.bin) and the run
configuration configs/runs/training_46_histost.yaml, through the builders of extract_encoders.py, so the
bundles have the layout extract_encoders.py writes.

Saved in <ROOT_HistoST> / train / trainings / trainings_HistoST / <training_id> / trained_encoders /
    - HEencoder_t<n>e<c>.pt: H&E encoder
    - STencoderCell_t<n>e<c>.pt: ST cell encoder
The projection head of the Novae branch is not released, so no STencoderGraph bundle is written.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict

import torch
import yaml
from safetensors.torch import load_file

# Add the project root to sys.path so that 'src' can be imported
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.constants import get_full_path
from src.evaluation.extract_encoders import (
    build_he_clip_encoder_from_loaded,
    build_image_encoder_from_config,
    build_st_cell_clip_encoder_from_loaded,
    get_scconcept_training_id_from_config,
)

HE_HEAD_KEYS = ("image_proj.weight", "img_tok_proj.weight")
ST_CELL_HEAD_KEY = "st_node_proj.weight"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def backbone_weights_digest(vision_encoder: torch.nn.Module) -> str:
    """
    SHA-256 of the UNI2-h weights, computed as the release declares it: tensors sorted by name, each
    entered as its name, its shape and its values in float32. The LoRA adapters are left out and the
    wrapped qkv layers keep their original names.
    """
    tensors: Dict[str, torch.Tensor] = {}
    for name, value in vision_encoder.state_dict().items():
        if ".adapter_" in name:
            continue
        tensors[name.replace(".attn.qkv.linear_layer.", ".attn.qkv.")] = value
    digest = hashlib.sha256()
    for name in sorted(tensors):
        values = tensors[name].detach().to(device="cpu", dtype=torch.float32).contiguous()
        digest.update(f"{name}|{tuple(values.shape)}|".encode())
        digest.update(values.numpy().reshape(-1).data)
    return digest.hexdigest()


def main(args):
    release_dir = Path(args.release_dir)
    release = json.loads((release_dir / "config.json").read_text())

    # 1. Release files, checked against the digests the release declares
    for name, expected in release["sha256"].items():
        found = sha256_file(release_dir / name)
        if found != expected:
            raise ValueError(f"{release_dir / name} does not match the release (sha256 {found}, expected {expected}).")
    adapters = load_file(str(release_dir / "adapters.safetensors"))
    heads = load_file(str(release_dir / "heads.safetensors"))
    print(f"Release: {len(adapters)} adapter tensors, heads {sorted(heads)}")

    # 2. Configuration of the released run
    config = yaml.safe_load(Path(args.training_config).read_text())
    rank = int(config["training"]["lora_rank"])
    alpha = float(config["training"]["lora_alpha"])
    if (rank, alpha) != (int(release["adapters"]["rank"]), float(release["adapters"]["alpha"])):
        raise ValueError(
            f"{args.training_config} sets LoRA rank {rank} and alpha {alpha}, the release "
            f"rank {release['adapters']['rank']} and alpha {release['adapters']['alpha']}."
        )

    # 3. Output paths
    tnum = args.training_id.split("_")[-1]
    out_dir = get_full_path("train", "trainings", "trainings_HistoST", args.training_id, "trained_encoders")
    he_path = out_dir / f"HEencoder_t{tnum}e{args.checkpoint_num}.pt"
    st_cell_path = out_dir / f"STencoderCell_t{tnum}e{args.checkpoint_num}.pt"
    for path in (he_path, st_cell_path):
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"{path} exists. Pass --overwrite to replace it.")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 4. UNI2-h with the LoRA modules, checked against the weights ALHENA was trained on
    image_encoder, _ = build_image_encoder_from_config(config)
    digest = backbone_weights_digest(image_encoder.vision_encoder)
    if digest != release["backbone"]["weights_digest"]:
        raise ValueError(
            "The local UNI2-h weights differ from those ALHENA was trained on "
            f"(digest {digest}, expected {release['backbone']['weights_digest']})."
        )
    print("Local UNI2-h weights match the release.")

    # 5. Full state dict in the layout of a training checkpoint, with the released tensors
    state = {f"image_encoder.{key}": value for key, value in image_encoder.state_dict().items()}
    for key, value in adapters.items():
        if key not in state or state[key].shape != value.shape:
            raise KeyError(f"Released adapter {key} {tuple(value.shape)} has no matching parameter.")
        state[key] = value
    for key in HE_HEAD_KEYS:
        state[key] = heads[key]
    del image_encoder

    # 6. H&E encoder bundle
    he_model = build_he_clip_encoder_from_loaded(
        config,
        state,
        device=torch.device("cpu"),
        projection_flags={"has_image_proj": True, "has_img_tok_proj": True, "image_proj_has_bias": False},
    )
    he_state = he_model.state_dict()
    for key, value in [*adapters.items(), *((key, heads[key]) for key in HE_HEAD_KEYS)]:
        if not torch.equal(he_state[key], value):
            raise RuntimeError(f"{key} did not load into the H&E encoder.")
    he_bundle = {
        "config": config,
        "state_dict": he_state,
        "export_type": "state_dict_bundle",
        "checkpoint_epoch": args.checkpoint_num,
        "projection_flags": {
            "has_image_proj": he_model.image_proj is not None,
            "has_img_tok_proj": getattr(he_model, "img_tok_proj", None) is not None,
            "image_proj_has_bias": bool(getattr(he_model, "has_image_proj_bias", False)),
        },
        "foundation_baseline_only": False,
        "release_sha256": release["sha256"],
        "notes": "Rebuilt from the Hugging Face release with the local UNI2-h weights. Same layout as "
                 "extract_encoders.py: rebuild with HEClipEncoderWithTokens/HEClipEncoderNoTokens "
                 "+ backbone/LoRA as specified in config, then load state_dict. "
                 "No preprocessing and no L2-normalization inside.",
    }
    torch.save(he_bundle, he_path)
    print(f"Saved HE encoder bundle to: {he_path}")
    del he_model, he_state, he_bundle, state

    # 7. ST cell encoder bundle
    st_cell_model = build_st_cell_clip_encoder_from_loaded(
        config,
        {ST_CELL_HEAD_KEY: heads[ST_CELL_HEAD_KEY]},
        device=torch.device("cpu"),
        projection_flags={"has_st_node_proj": True},
    )
    st_cell_bundle = {
        "config": config,
        "state_dict": st_cell_model.state_dict(),
        "export_type": "state_dict_bundle",
        "checkpoint_epoch": args.checkpoint_num,
        "projection_flags": {"has_st_node_proj": st_cell_model.has_proj},
        "scconcept_training_id": get_scconcept_training_id_from_config(config),
        "release_sha256": release["sha256"],
        "notes": "Rebuilt from the Hugging Face release. Same layout as extract_encoders.py: expects "
                 "pretrained scConcept embeddings as input, then applies st_node_proj. "
                 "No L2-normalization inside.",
    }
    torch.save(st_cell_bundle, st_cell_path)
    print(f"Saved ST cell encoder bundle to: {st_cell_path}")
    print(
        "No STencoderGraph bundle: the projection head of the Novae branch is not released. "
        "Run _10_compute_STembs.py with --do_cell."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Rebuild the encoder bundles of the released ALHENA model from its Hugging Face release."
    )
    parser.add_argument(
        "--release_dir", type=str, required=True,
        help="Folder holding config.json, adapters.safetensors and heads.safetensors, "
             "downloaded from huggingface.co/prism-oncology/ALHENA.",
    )
    parser.add_argument(
        "--training_config", type=str,
        default=str(project_root / "configs" / "runs" / "training_46_histost.yaml"),
        help="Configuration of the released run.",
    )
    parser.add_argument("--training_id", type=str, default="training_46",
                        help="Run folder the bundles are written to.")
    parser.add_argument("--checkpoint_num", type=int, default=13,
                        help="Epoch of the released checkpoint, used in the bundle names.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing bundles.")
    main(parser.parse_args())
