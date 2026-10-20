import torch
import torch.nn as nn


class LoRA(nn.Module):
    def __init__(self, linear_layer, rank, alpha):
        super(LoRA, self).__init__()
        
        self.linear_layer = linear_layer
        self.in_dim = self.linear_layer.in_features
        out_dim = self.linear_layer.out_features

        # Some sanity checks
        assert isinstance(linear_layer, nn.Linear), f"LoRA expects nn.Linear, got {type(linear_layer)}"
        assert out_dim % 3 == 0, f"Expected fused qkv with out_features divisible by 3, got {out_dim}"
        d_attn = out_dim // 3
        assert d_attn == self.in_dim, f"This LoRA variant assumes head_dim == embed_dim (d_attn == in_dim). Got d_attn={d_attn}, in_dim={self.in_dim}."
        
        std = 1 / torch.sqrt(torch.tensor(rank).float())
        self.adapter_Q_downsample = nn.Parameter(torch.randn(self.in_dim, rank) * std)
        self.adapter_Q_upsample = nn.Parameter(torch.zeros(rank, self.in_dim))
        self.adapter_V_downsample = nn.Parameter(torch.randn(self.in_dim, rank) * std)
        self.adapter_V_upsample = nn.Parameter(torch.zeros(rank, self.in_dim))
        self.adapter_alpha = alpha / rank
    
    def forward(self, x):
        x_q = self.adapter_alpha * (x @ self.adapter_Q_downsample @ self.adapter_Q_upsample)
        x_v = self.adapter_alpha * (x @ self.adapter_V_downsample @ self.adapter_V_upsample)
        x_lora = torch.cat([x_q, torch.zeros_like(x_v), x_v], dim=-1)
        x = self.linear_layer(x) + x_lora
        return x



def insert_lora(model, rank, alpha):
    """
    Recursively unwraps common wrapper layers until it reaches the Vision Transformer (object exposing blocks) 
    and replaces each qkv projection with a LoRA-augmented version.
    arguments:
        model [nn.Module]: model to insert LoRA into
        rank [int]: rank of LoRA
        alpha [float]: alpha parameter scaling the adaptation
    returns:
        None
    """
    # Find the ViT
    if hasattr(model, "vision_encoder"):                 # timm wrapper
        vit = model.vision_encoder
    elif hasattr(model, "backbone"):                     # e.g. HEEncoderWithMLP
        return insert_lora(model.backbone, rank, alpha)  # recurse down one level
    elif hasattr(model, "blocks"):                       # already at ViT
        vit = model
    else:
        raise AttributeError(
            f"insert_lora: cannot locate ViT inside object of type {type(model)}; "
            "looked for attributes 'vision_encoder', 'backbone', 'blocks'."
        )

    # Patch each encoder block
    for block in vit.blocks:
        block.attn.qkv = LoRA(linear_layer=block.attn.qkv,
                              rank=rank,
                              alpha=alpha)