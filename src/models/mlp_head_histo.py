import torch.nn as nn


class HEEncoderWithMLP(nn.Module):
    """
    Thin wrapper that adds a 2-layer MLP (GELU) on top of an existing H&E encoder.
    """
    def __init__(
        self,
        backbone: nn.Module,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.backbone = backbone
        self.mlp_head = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

        self.output_dim = out_dim

    def forward(self, x):
        x = self.backbone(x)
        return self.mlp_head(x)
    

    def encode_with_tokens(self, x):
        """
        Delegate to backbone to get (pooled_pre_mlp, tokens, grid_hw),
        then apply MLP only to the pooled part.
        """
        if not hasattr(self.backbone, "encode_with_tokens"):
            # Fallback: no tokens exposed; behave like plain forward
            pooled = self.backbone(x)
            return self.mlp_head(pooled), None, None
        pooled, tokens, grid_hw = self.backbone.encode_with_tokens(x)
        pooled = self.mlp_head(pooled)
        return pooled, tokens, grid_hw
    

    @property
    def token_dim(self):
        """
        Token channel size of the backbone (unchanged by the MLP head).
        """
        return getattr(self.backbone, "token_dim", None)


    def __getattr__(self, item):
        """
        Delegate any attribute not defined here to self.backbone (e.g. blocks or vision_encoder, used for LoRA insertion).
        """
        try:
            return super().__getattr__(item)
        except AttributeError:
            return getattr(self.backbone, item)