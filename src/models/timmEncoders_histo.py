import logging
logger = logging.getLogger(__name__)

import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from timm.models.vision_transformer import VisionTransformer
import torch
from torch import nn
from typing import Callable, Tuple
import math

from src.utils.models_settings import MODEL_TO_ARGS
from src.utils.utils import loading_checkpoint_cpu



def get_timm_he_encoder(ckpt_path: str, model_name: str, map_location: str) -> tuple:
    """
    Return the timm model, its transforms and the embedding function.
    Adapted from:
    - https://huggingface.co/paige-ai/Virchow2
    - https://huggingface.co/MahmoodLab/UNI2-h
    - https://huggingface.co/bioptimus/H-optimus-0
    - https://huggingface.co/bioptimus/H-optimus-1

    :param ckpt_path: path to the pretrained ckpt.
    :param model_name: name of the model to load.
    :param map_location: location to map the model.
    """

    # Args
    timm_kwargs = MODEL_TO_ARGS[model_name]["timm_kwargs"]
    architecture = MODEL_TO_ARGS[model_name]["architecture"]
    pretrained_cfg = MODEL_TO_ARGS[model_name]["pretrained_cfg"]

    # Model
    model = timm.create_model(architecture, pretrained_cfg=pretrained_cfg, pretrained=False, **timm_kwargs)
    # assert map_location is cpu
    assert map_location == "cpu", "Use map_location='cpu' to load the timm encoder."
    model.load_state_dict(loading_checkpoint_cpu(ckpt_path, weights_only=True), strict=True)

    # Transform
    transform = create_transform(
        **resolve_data_config(model.pretrained_cfg, model=model)
    )

    if model_name == "virchow2":

        def extract_embedding(
            src: torch.Tensor, vision_encoder: VisionTransformer
        ) -> torch.Tensor:
            # Use the unpooled token sequence explicitly
            x_features = vision_encoder.forward_features(src)  # [B, N_all, C]
            num_prefix = getattr(vision_encoder, "num_prefix_tokens", 5) # tokens 1-4 are register tokens and are ignored
            class_token = x_features[:, 0, :]
            patch_tokens = x_features[:, num_prefix:, :]
            emb = torch.cat([class_token, patch_tokens.mean(1)], dim=-1)
            return emb
        
        # For Virchow2, the final embedding is 2×model.embed_dim:
        model.info_final_dim = model.embed_dim * 2

    else:

        def extract_embedding(
            src: torch.Tensor, vision_encoder: VisionTransformer
        ) -> torch.Tensor:
            # Explicit timm feature-extraction path:
            # unpooled tokens -> model-specific pooling/pre_logits
            x_features = vision_encoder.forward_features(src)
            emb = vision_encoder.forward_head(x_features, pre_logits=True)
            return emb
        
        # For UNI2-h, H-optimus-0, and H-optimus-1 the final embedding is model.embed_dim:
        model.info_final_dim = model.embed_dim

    return model, transform, extract_embedding



class ImageEncoderTimm(nn.Module):
    """
    HE encoder using timm models.
    """

    def __init__(
        self, vision_encoder: VisionTransformer, extract_embedding: Callable, model_name: str
    ) -> None:
        """
        Initializing the encoder.
        :param vision_encoder: vision encoder (Virchow2 or UNI2-h or H-optimus-0)
        :param extract_embedding: function to compute an embedding
        from the input image using the vision encoder.
        """
        super().__init__()
        self.vision_encoder = vision_encoder
        self.extract_embedding = extract_embedding
        self.model_name = model_name

        # Expected number of prefix tokens (CLS + register tokens) of each model
        self.expected_num_prefix_tokens = {
            "virchow2": 5,   # 1 CLS + 4 register tokens
            "uni2h": 9,      # 1 CLS + 8 register tokens
            "hoptimus0": 5,  # 1 CLS + 4 register tokens
            "hoptimus1": 5,  # 1 CLS + 4 register tokens
        }

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        :param src: input embeddings (bs, 3, h, w).
        :return: output embedding (bs, output_dim).
        """
        return self.extract_embedding(src, self.vision_encoder)

    @property
    def output_dim(self) -> int:
        """
        Get the output dimension of the encoder.
        :return: Size of the embedding vector.
        """
        return self.vision_encoder.info_final_dim
    

    @property
    def token_dim(self) -> int:
        """
        Channel dimension of patch tokens returned by encode_with_tokens().
        For ViTs this is the model's embed_dim (note: pooled Virchow2 is 2x for CLS||mean).
        """
        return int(getattr(self.vision_encoder, "embed_dim"))


    def encode_with_tokens(self, src: torch.Tensor):
        """
        Single pass via timm’s public API:
          - tokens = v.forward_features(x): unpooled (up to final norm)
          - pooled = model-specific head / rule (Virchow2 uses CLS||mean(patches))
          - patch tokens = drop prefix via v.num_prefix_tokens
        """
        v: VisionTransformer = self.vision_encoder
        x_features = v.forward_features(src)              # [B, N_all, C] (unpooled tokens)

        # Slice patch tokens robustly (handles cls/dist/reg prefixes)
        num_prefix = int(getattr(v, "num_prefix_tokens", 1))
        expected_prefix = self.expected_num_prefix_tokens.get(self.model_name, None)
        if expected_prefix is not None and num_prefix != expected_prefix:
            raise RuntimeError(
                f"{self.model_name}: unexpected num_prefix_tokens={num_prefix}, "
                f"expected {expected_prefix}. This usually means the instantiated "
                f"timm model args/checkpoint do not match the intended backbone."
            )

        patch_tokens = x_features[:, num_prefix:, :]      # [B, N_patch, C]

        # Pooling (same as extract_embedding)
        if self.model_name == "virchow2":
            cls_token = x_features[:, 0, :]
            pooled = torch.cat([cls_token, patch_tokens.mean(1)], dim=-1)  # 2*C per model card
        else:
            # Use timm’s official pooling/token-selection
            pooled = v.forward_head(x_features, pre_logits=True)           # honours global_pool, fc_norm, etc.

        # Token grid size, from the number of patch tokens
        n_patch = int(patch_tokens.shape[1])
        side = int(math.isqrt(n_patch))
        if side * side != n_patch:
            raise ValueError(
                f"{self.model_name}: patch token count {n_patch} is not a perfect square. "
                "Expected square token grid for 224x224 histology tiles."
            )
        Ht, Wt = side, side

        # Sanity check on number of patch tokens
        expected = int(Ht) * int(Wt)
        if patch_tokens.shape[1] != expected:
            raise ValueError(
                f"Unexpected number of patch tokens: got {patch_tokens.shape[1]}, "
                f"expected {expected} ({Ht}x{Wt})"
            )

        return pooled, patch_tokens, (int(Ht), int(Wt))