import logging
logger = logging.getLogger(__name__)

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.distributed as dist
import torch.distributed.nn
from typing import Optional, Tuple, Dict

from src.utils.utils import get_dist_rank, get_dist_world_size, safe_normalize



def gather_features(
    img_local: torch.Tensor,
    st_local: torch.Tensor,
    gather_with_grad: bool,
    rank: int,
    world_size: int,
    reinsert_local: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Gather features from all GPUs: with grad (torch.distributed.nn.all_gather) if gather_with_grad, else without
    grad (dist.all_gather); with reinsert_local, the local chunk is the original tensor so gradients flow into it.
    """

    if gather_with_grad:
        img_global = torch.cat(torch.distributed.nn.all_gather(img_local), dim=0)
        st_global = torch.cat(torch.distributed.nn.all_gather(st_local), dim=0)
    else:
        gathered_img_local = [torch.zeros_like(img_local) for _ in range(world_size)]
        gathered_st_local = [torch.zeros_like(st_local) for _ in range(world_size)]
        dist.all_gather(gathered_img_local, img_local)
        dist.all_gather(gathered_st_local, st_local)
        if reinsert_local:
            # ensure grads for local rank when gathered tensors don't have a gradient
            gathered_img_local[rank] = img_local
            gathered_st_local[rank] = st_local
        img_global = torch.cat(gathered_img_local, dim=0)
        st_global = torch.cat(gathered_st_local, dim=0)

    return img_global, st_global



def gather_tensor(
    x_local: torch.Tensor,
    rank: int,
    world_size: int,
) -> torch.Tensor:
    """
    all_gather for non-grad tensors (e.g. ids).
    """
    if world_size <= 1:
        return x_local
    gathered = [torch.zeros_like(x_local) for _ in range(world_size)]
    dist.all_gather(gathered, x_local)
    return torch.cat(gathered, dim=0)



def _frozen_teacher_global(
    x_local: torch.Tensor,
    rank: int,
    world_size: int,
    center_mean: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    L2-normalize (fp32) a frozen pre-projection teacher tensor and gather it across ranks in rank order (as
    gather_features), so that target column j matches logits column j. If center_mean is given, rows are centered:
    c = normalize(n - center_mean) (all-zero rows stay zero). No grad. Returns (local_n, global_n).
    """
    with torch.no_grad():
        x_local_n = safe_normalize(x_local.float(), dim=-1)
        if center_mean is not None:
            pre_norm = x_local_n.norm(dim=-1, keepdim=True)   # ~1 for real rows, ~0 for degenerate
            x_centered = safe_normalize(x_local_n - center_mean.to(x_local_n.dtype), dim=-1, eps=1e-8)
            x_local_n = torch.where(pre_norm >= 0.5, x_centered, torch.zeros_like(x_centered))
        if world_size > 1:
            x_global_n = gather_tensor(x_local_n, rank=rank, world_size=world_size)
        else:
            x_global_n = x_local_n
    return x_local_n, x_global_n



def _multi_positive_infonce(logits: torch.Tensor, pos_mask: torch.Tensor) -> torch.Tensor:
    """
    Multi-positive InfoNCE:
    loss_i = -log( sum_{p in P(i)} exp(logits_i,p) / sum_j exp(logits_i,j) )
            = logsumexp(all) - logsumexp(positives)

    logits:   [N_local, N_global] (float)
    pos_mask: [N_local, N_global] (bool) where True indicates a positive.
    """
    lse_all = torch.logsumexp(logits, dim=-1)  # [N]
    logits_pos = logits.masked_fill(~pos_mask, -torch.inf)
    lse_pos = torch.logsumexp(logits_pos, dim=-1)  # [N]
    valid_rows = torch.isfinite(lse_pos)  # rows that have >=1 positive
    if not bool(valid_rows.any()):
        return logits.new_tensor(0.0)
    return (lse_all[valid_rows] - lse_pos[valid_rows]).mean()



def _soft_cross_entropy(logits: torch.Tensor,
                        targets: torch.Tensor,
                        reduction: str = "mean") -> torch.Tensor:
    """
    Cross-entropy with soft targets (targets are probabilities per row).
    Assumes logits are unnormalized scores. No grad flows into targets.
    """
    # logits are typically float32 outside AMP in this codebase
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    loss = -(targets * log_probs).sum(dim=-1)
    if reduction == "mean":  return loss.mean()
    if reduction == "sum":   return loss.sum()
    return loss





def pool_scconcept_per_patch(scconcept_emb: torch.Tensor,
                             scconcept_cell_id: torch.Tensor) -> torch.Tensor:
    """
    Masked mean of the scConcept cell embeddings over the valid cells (cell_id > 0) of each patch, in float32
    (zero row if no valid cell).

    Args:
        scconcept_emb:     [B, K, D_scconcept] cell embeddings (zero-padded slots).
        scconcept_cell_id: [B, K] cell ids (-1 = padding, 0 = background, >0 = valid cell).

    Returns:
        [B, D_scconcept] float32 patch-level aggregate scConcept embedding.
    """
    emb = scconcept_emb.float()                                 # [B, K, D]
    valid = (scconcept_cell_id > 0).unsqueeze(-1).float()       # [B, K, 1]
    summ = (emb * valid).sum(dim=1)                             # [B, D]
    cnt = valid.sum(dim=1).clamp_min(1.0)                       # [B, 1]
    return summ / cnt


class CLIPModel(nn.Module):
    """
    CLIP model that aligns image and spatial transcriptomics modalities.
    The ST teacher vector ``st_feat`` (precomputed by the frozen Novae encoder) is read from the batch and projected by ``st_proj``.

    Args:
        image_encoder (nn.Module): Image encoder model.
        image_feature_dim (int): Dimension of the image encoder output features.
        st_feature_dim (int): Dimension of the (cached) ST teacher output features (= 128).
        embed_dim (int): Dimension of the common embedding space.
    """
    def __init__(
        self,
        image_encoder: nn.Module,
        # Patch-level branch
        image_feature_dim: int,
        st_feature_dim: int,
        embed_dim: int,
        img_token_dim: Optional[int] = None,
        st_node_dim: Optional[int] = None,
        patch_local_loss: bool = True,
        patch_gather_with_grad: bool = False,
        patch_reinsert_local: bool = True,
        loss_type: str = "classic",      # "classic" | "soft_targets" | "soft_targets_st"
        target_temperature: Optional[float] = 0.1,  # fixed teacher-target temperature T (targets = softmax(sim/T)); None => tied to learnable logit_scale
        cache_labels: bool = False,
        # Token-level branch
        lambda_patch: float = 1.0,
        lambda_token: float = 0.0,
        token_embed_dim: Optional[int] = None,  # defaults to embed_dim
        token_loss_type: str = "classic", # "classic" | "soft_targets" | "soft_targets_st"
        token_target_temperature: Optional[float] = 0.1,  # fixed teacher-target temperature for the token branch; None => tied to learnable logit_scale_token
        token_local_loss: Optional[bool] = None,  # default to patch if None
        token_gather_with_grad: Optional[bool] = None,  # default to patch if None
        token_reinsert_local: Optional[bool] = None,  # default to patch if None
        # Teacher-centering of the soft target (see _frozen_teacher_global / precompute_teacher_mean.py)
        center_teacher_target: bool = False,
        novae_center_mean: Optional[torch.Tensor] = None,      # [st_feature_dim] frozen global mean (patch teacher)
        scconcept_center_mean: Optional[torch.Tensor] = None,  # [st_node_dim] frozen global mean (token teacher)
    ):
        super().__init__()
        self.image_encoder = image_encoder
        self.image_feature_dim = image_feature_dim
        self.st_feature_dim = st_feature_dim
        self.embed_dim = embed_dim
        self.patch_local_loss = bool(patch_local_loss)
        self.patch_gather_with_grad = bool(patch_gather_with_grad)
        self.patch_reinsert_local = bool(patch_reinsert_local)
        self.token_local_loss = self.patch_local_loss if token_local_loss is None else bool(token_local_loss)
        self.token_gather_with_grad = self.patch_gather_with_grad if token_gather_with_grad is None else bool(token_gather_with_grad)
        self.token_reinsert_local = self.patch_reinsert_local if token_reinsert_local is None else bool(token_reinsert_local)
        self.cache_labels = cache_labels
        self.loss_type = loss_type.lower()
        self.token_loss_type = token_loss_type.lower()
        # Validate loss types
        _allowed_loss_types = {"classic", "soft_targets", "soft_targets_st"}
        if self.loss_type not in _allowed_loss_types:
            raise ValueError(f"Unknown loss_type='{self.loss_type}'. Use one of {sorted(_allowed_loss_types)}.")
        if self.token_loss_type not in _allowed_loss_types:
            raise ValueError(f"Unknown token_loss_type='{self.token_loss_type}'. Use one of {sorted(_allowed_loss_types)}.")
        # Teacher-target temperature(s): targets = softmax(sim / T); None ties T to the learnable logit scale
        self.target_temperature = None if target_temperature is None else float(target_temperature)
        self.token_target_temperature = None if token_target_temperature is None else float(token_target_temperature)
        for _tt_name, _tt_val in (("target_temperature", self.target_temperature),
                                  ("token_target_temperature", self.token_target_temperature)):
            if _tt_val is not None and not (_tt_val > 0.0):
                raise ValueError(f"{_tt_name} must be a positive float or None (tied to logit_scale); got {_tt_val}.")
        self.lambda_patch = float(lambda_patch)
        self.lambda_token = float(lambda_token)
        self.token_embed_dim = int(token_embed_dim) if token_embed_dim is not None else int(embed_dim)

        # Teacher-centering means (from src/dataset/precompute_teacher_mean.py, loaded by Experiment.get_model),
        # registered as non-persistent buffers (not saved in the state_dict)
        self.center_teacher_target = bool(center_teacher_target)
        if self.center_teacher_target:
            if (self.loss_type in ("soft_targets", "soft_targets_st") and self.lambda_patch > 0.0
                    and novae_center_mean is None):
                raise ValueError(
                    "center_teacher_target=True with a patch soft-target loss (lambda_patch>0) but "
                    "novae_center_mean is None. Run src/dataset/precompute_teacher_mean.py and pass the mean."
                )
            if (self.token_loss_type in ("soft_targets", "soft_targets_st") and self.lambda_token > 0.0
                    and scconcept_center_mean is None):
                raise ValueError(
                    "center_teacher_target=True with a token soft-target loss (lambda_token>0) but "
                    "scconcept_center_mean is None. Run src/dataset/precompute_teacher_mean.py and pass the mean."
                )
        if novae_center_mean is not None and tuple(novae_center_mean.shape) != (int(st_feature_dim),):
            raise ValueError(
                f"novae_center_mean shape {tuple(novae_center_mean.shape)} != (st_feature_dim={int(st_feature_dim)},)."
            )
        if scconcept_center_mean is not None:
            if st_node_dim is None:
                raise ValueError("scconcept_center_mean provided but st_node_dim is None (token branch disabled).")
            if tuple(scconcept_center_mean.shape) != (int(st_node_dim),):
                raise ValueError(
                    f"scconcept_center_mean shape {tuple(scconcept_center_mean.shape)} != "
                    f"(st_node_dim={int(st_node_dim)},)."
                )
        self.register_buffer(
            "novae_center_mean",
            None if novae_center_mean is None else novae_center_mean.detach().float(),
            persistent=False,
        )
        self.register_buffer(
            "scconcept_center_mean",
            None if scconcept_center_mean is None else scconcept_center_mean.detach().float(),
            persistent=False,
        )

        # will be refreshed in forward()
        self.rank = 0
        self.world_size = 1

        # cache labels keyed by (device_str, num_logits, local_loss, rank, world_size)
        self.labels: Dict[Tuple[str, int, bool, int, int], torch.Tensor] = {}

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1/0.07))
        # token head gets its own scale; harmless if lambda_token == 0
        self.logit_scale_token = nn.Parameter(torch.ones([]) * np.log(1/0.07))

        # Projection heads mapping modality features to a common embedding space.
        self.image_proj = nn.Linear(image_feature_dim, embed_dim, bias=False)
        self.st_proj = nn.Linear(st_feature_dim, embed_dim, bias=False)
        
        nn.init.normal_(self.image_proj.weight, std=image_feature_dim ** -0.5)
        nn.init.normal_(self.st_proj.weight, std=st_feature_dim ** -0.5)

        # cache holders (automatically overwritten every forward)
        self._last_img_feat: torch.Tensor | None = None
        self._last_st_feat: torch.Tensor | None = None

        # Token embedding projections
        if self.lambda_token > 0.0:
            if img_token_dim is None or st_node_dim is None:
                raise ValueError("lambda_token>0 requires img_token_dim and st_node_dim to be provided by the encoders.")
            self.img_tok_proj = nn.Linear(int(img_token_dim), self.token_embed_dim, bias=False)
            self.st_node_proj = nn.Linear(int(st_node_dim),  self.token_embed_dim, bias=False)
            nn.init.normal_(self.img_tok_proj.weight, std=(int(img_token_dim) ** -0.5))
            nn.init.normal_(self.st_node_proj.weight, std=(int(st_node_dim) ** -0.5))
        else:
            self.img_tok_proj = None
            self.st_node_proj = None



    def _validate_ddp_settings(self) -> None:
        """
        Raise if, in DDP with local_loss=False, neither gather_with_grad nor reinsert_local is set (no gradient).
        """
        if self.world_size <= 1:
            return
        if (not self.patch_local_loss) and (not self.patch_gather_with_grad) and (not self.patch_reinsert_local):
            raise ValueError(
                "Invalid patch distributed config: patch_local_loss=False with "
                "patch_gather_with_grad=False and patch_reinsert_local=False would make "
                "patch logits fully detached (no gradients). Set patch_gather_with_grad=True "
                "or patch_reinsert_local=True."
            )
        if self.lambda_token > 0.0:
            if (not self.token_local_loss) and (not self.token_gather_with_grad) and (not self.token_reinsert_local):
                raise ValueError(
                    "Invalid token distributed config: token_local_loss=False with "
                    "token_gather_with_grad=False and token_reinsert_local=False would make "
                    "token logits fully detached (no gradients). Set token_gather_with_grad=True "
                    "or token_reinsert_local=True."
                )



    def _ddp_loss_scale(
        self,
        local_rows: int,
        global_rows: int,
        local_loss: bool,
        gather_with_grad: bool,
    ) -> float:
        """
        Loss scale so that, after DDP gradient averaging, the gradient matches a global mean over `global_rows` rows:
        world_size * local_rows / global_rows if local_loss, else 1 (gather_with_grad) or world_size.
        """
        if self.world_size <= 1 or global_rows <= 0:
            return 1.0
        if local_loss:
            return float(self.world_size) * (float(local_rows) / float(global_rows))
        return 1.0 if gather_with_grad else float(self.world_size)



    def encode_image(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode H&E image patches.
        
        Args:
            x: Tensor of shape [B, 3, H, W]
        Returns:
            Normalized image embeddings of shape [B, embed_dim]
        """
        features = self.image_encoder(x)
        self._last_img_feat = features   # cache for CKA
        proj = self.image_proj(features)
        return safe_normalize(proj, dim=-1)



    def encode_st(self, st_data) -> torch.Tensor:
        """
        Project the precomputed ST teacher vector ``st_data.st_feat`` into the CLIP space (st_proj + L2 normalization).

        Args:
            st_data: PyTorch Geometric Batch carrying ``st_feat`` [B, st_feature_dim].
        Returns:
            Normalized ST embeddings of shape [B, embed_dim].
        """
        features = st_data.st_feat                # [B, st_feature_dim] cached frozen-teacher output
        self._last_st_feat = features             # cache for CKA (pre-projection)
        proj = self.st_proj(features)
        return safe_normalize(proj, dim=-1)



    def get_logits(self,
                   img_local: torch.Tensor,
                   st_local: torch.Tensor,
                   logit_scale: float,
                   local_loss: bool,
                   gather_with_grad: bool,
                   reinsert_local: bool,
                   return_global: bool = False
                   ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Compute logits. Optionally also return (img_global, st_global) so callers can
        reuse the gathered features (avoids double all_gather).
        """
        img_global = None
        st_global = None
        
        if self.world_size > 1:
            img_global, st_global = gather_features(
                img_local,
                st_local,
                gather_with_grad=gather_with_grad,
                rank=self.rank,
                world_size=self.world_size,
                reinsert_local=reinsert_local,
            )

            if local_loss:
                logits_i2st = logit_scale * img_local @ st_global.T
                logits_st2i = logit_scale * st_local @ img_global.T
            else:
                logits_i2st = logit_scale * img_global @ st_global.T
                logits_st2i = logits_i2st.T
        else:
            # single-GPU path
            img_global, st_global = img_local, st_local
            logits_i2st = logit_scale * img_local @ st_local.T
            logits_st2i = logit_scale * st_local @ img_local.T

        if return_global:
            return logits_i2st, logits_st2i, img_global, st_global
        return logits_i2st, logits_st2i, None, None
    


    def get_ground_truth(self, device: torch.device, num_logits: int, local_loss: bool) -> torch.Tensor:
        """
        Ground-truth indices for hard-positive CE.

        - local_loss=True  => logits are [B_local, N_global], positives are offset by rank.
        - local_loss=False => logits are [N_global, N_global], positives are diagonal (no offset).
        """
        if not self.cache_labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and local_loss:
                labels = labels + num_logits * self.rank
            return labels

        key = (str(device), int(num_logits), bool(local_loss), int(self.rank), int(self.world_size))
        labels = self.labels.get(key, None)
        if labels is None:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and local_loss:
                labels = labels + num_logits * self.rank
            self.labels[key] = labels
        return labels 



    def forward(self, data):
        """
        Forward pass returning both modality embeddings.
        """

        # Refresh distributed context
        if dist.is_available() and dist.is_initialized():
            self.rank = get_dist_rank()
            self.world_size = get_dist_world_size()

        self._validate_ddp_settings()

        # Local embeddings (encoded + projected + normalized)
        B_local = data.img.size(0)
        token_heads_exist = (self.img_tok_proj is not None) and (self.st_node_proj is not None)
        need_tokens = bool(token_heads_exist and (self.lambda_token > 0.0))

        if need_tokens:
            if not hasattr(self.image_encoder, "encode_with_tokens"):
                raise ValueError("Token loss is enabled (lambda_token>0) but image_encoder has no encode_with_tokens().")
            img_feat, img_tokens, grid_hw = self.image_encoder.encode_with_tokens(data.img)  # pooled, [B,Nt,Ct], (Ht,Wt)
        else:
            img_feat = self.image_encoder(data.img)
            img_tokens, grid_hw = None, None

        self._last_img_feat = img_feat

        # Patch-level ST teacher: precomputed Novae st_feat (src/dataset/precompute_novae_teacher.py)
        st_feat = data.st_feat   # [B, st_feature_dim] (= 128)

        # Token-level node embeddings come from precomputed scConcept embeddings (panel-independent)
        if need_tokens:
            if not hasattr(data, "scconcept_emb"):
                raise ValueError("Token loss is enabled (lambda_token>0) but batch is missing 'scconcept_emb'.")
            st_nodes = data.scconcept_emb  # expected [B, K, D_scconcept] for the cell-level objective
        else:
            st_nodes = None
        
        self._last_st_feat = st_feat

        img_local = safe_normalize(self.image_proj(img_feat), dim=-1)   # [local B,D]
        st_local  = safe_normalize(self.st_proj(st_feat), dim=-1)       # [local B,D]

        # Compute logit scale
        logit_scale = self.logit_scale.exp()
        logit_scale_token = self.logit_scale_token.exp()
        # Teacher-target softmax scale: 1/T if a target temperature is set, else the learnable logit scale
        target_scale = logit_scale if self.target_temperature is None else (1.0 / self.target_temperature)
        token_target_scale = logit_scale_token if self.token_target_temperature is None else (1.0 / self.token_target_temperature)
        
        # Compute global logits (+ optionally retrieve gathered features once)
        # Always return gathered globals so step metrics can be global w/o extra comms
        want_global = True
        logits_i2st_full, logits_st2i_full, img_global, st_global = self.get_logits(
            img_local,
            st_local,
            logit_scale,
            local_loss=self.patch_local_loss,
            gather_with_grad=self.patch_gather_with_grad,
            reinsert_local=self.patch_reinsert_local,
            return_global=want_global,
        )

        # Compute labels
        device = img_local.device
        labels_full = self.get_ground_truth(device, logits_i2st_full.shape[0], local_loss=self.patch_local_loss)

        # Loss logits (FP32)
        # For patch_local_loss=False, logits_st2i is exactly logits_i2st.T
        # => avoid a second FP32 allocation
        logits_i2st = logits_i2st_full.float()
        if (self.world_size > 1 or True) and (not self.patch_local_loss):
            logits_st2i = logits_i2st.T  # view, no extra FP32 copy
        else:
            logits_st2i = logits_st2i_full.float()

        # Output logits/labels for step metrics
        logits_i2st_out = logits_i2st
        logits_st2i_out = logits_st2i
        labels_out = labels_full

        # If patch_local_loss=False in DDP, step metrics only need local query rows
        # Slice here so a [B_global,B_global] tensor is not returned to the Trainer
        if (self.world_size > 1) and (not self.patch_local_loss):
            bs_local = int(data.img.size(0))
            start = int(self.rank) * bs_local
            end = start + bs_local
            logits_i2st_out = logits_i2st[start:end, :]
            logits_st2i_out = logits_st2i[start:end, :]
            labels_out = labels_full[start:end]

        # Frozen pre-projection teacher embeddings for the patch soft targets --------------------------------------
        st_feat_local_n = st_feat_global_n = None
        img_feat_local_n = img_feat_global_n = None
        if self.loss_type in ("soft_targets", "soft_targets_st"):
            st_feat_local_n, st_feat_global_n = _frozen_teacher_global(
                st_feat, self.rank, self.world_size,
                center_mean=(self.novae_center_mean if self.center_teacher_target else None),
            )
            if self.loss_type == "soft_targets":
                # "soft_targets" only: H&E teacher half (never centered)
                img_feat_local_n, img_feat_global_n = _frozen_teacher_global(
                    img_feat, self.rank, self.world_size, center_mean=None,
                )

        # Patch-level loss ---------------------------------------------------------------------------------------

        if self.loss_type == "classic":
            # Standard CLIP loss with hard positives
            loss_i2st = F.cross_entropy(logits_i2st, labels_full)
            loss_st2i = F.cross_entropy(logits_st2i, labels_full)
            loss = (loss_i2st + loss_st2i) / 2
        
        elif self.loss_type in ("soft_targets", "soft_targets_st"):
            # Similarity-adjusted soft targets from the frozen pre-projection teacher vectors (no grad).
            # Rows: local pairs if patch_local_loss, else global pairs; columns: global teacher tensor.
            st_src_anchor = st_feat_local_n if self.patch_local_loss else st_feat_global_n
            with torch.no_grad():
                # Cosine sims in the frozen teacher space; result: [N_rows, N_global].
                sim_src = (st_src_anchor @ st_feat_global_n.T)        # st <-> st (frozen teacher)
                if self.loss_type == "soft_targets":
                    # symmetric variant: average in the H&E <-> H&E similarity (pre-projection)
                    img_src_anchor = img_feat_local_n if self.patch_local_loss else img_feat_global_n
                    sim_src = 0.5 * (sim_src + (img_src_anchor @ img_feat_global_n.T))
                # "soft_targets_st": teacher-only similarity (H&E view ignored)
                targets = torch.softmax(sim_src * target_scale, dim=-1).float()
            loss_i2st = _soft_cross_entropy(logits_i2st, targets, reduction="mean")
            loss_st2i = _soft_cross_entropy(logits_st2i, targets, reduction="mean")
            loss = 0.5 * (loss_i2st + loss_st2i)
        
        else:
            raise ValueError(f"Unknown loss_type='{self.loss_type}'. Use 'classic', 'soft_targets' or 'soft_targets_st'.")

        # patch-level loss (raw, for logging/metrics)
        patch_loss = loss
        token_loss: Optional[torch.Tensor] = None

        # Scale only the backward objective to be a consistent global-mean across configs
        # global_rows = B_global; local_rows = B_local if local_loss else B_global
        B_local = int(data.img.size(0))
        B_global = int(img_global.size(0)) if img_global is not None else B_local
        patch_local_rows = B_local if self.patch_local_loss else B_global
        patch_scale = self._ddp_loss_scale(
            local_rows=patch_local_rows,
            global_rows=B_global,
            local_loss=self.patch_local_loss,
            gather_with_grad=self.patch_gather_with_grad,
        )
        total_loss = (self.lambda_patch * patch_loss) * patch_scale

        # DDP keepalive: add 0 * token-head weights so their params take part in autograd when the token loss is skipped
        if self.world_size > 1 and token_heads_exist and (not need_tokens):
            total_loss = total_loss + 0.0 * (self.img_tok_proj.weight.sum() + self.st_node_proj.weight.sum())


        # Token-level loss -------------------------------------------------------------------------------------------
        # cell <-> nucleus-mask–pooled image feature (weighted mean of image-token features assigned to each cell)
        
        # token_metrics is always returned when need_tokens (same collective calls on all ranks)
        token_metrics: Optional[Dict[str, int]] = None
        if need_tokens:
            token_metrics = {
                "tok_total": 0,
                "tok_top1_correct_sum": 0,
                "tok_topK_correct_sum": 0,
                "tok_K": 5,
            }

        if need_tokens and (img_tokens is not None) and (st_nodes is not None):

            # Skip the token branch if no rank has a valid cell id
            if not hasattr(data, "scconcept_cell_id"):
                raise ValueError("Token loss enabled but batch is missing 'scconcept_cell_id' (int32 [B,K]).")
            cell_ids = data.scconcept_cell_id.to(torch.long)  # [B,K], pad=-1
            any_local = (cell_ids > 0).any().to(dtype=torch.int32)
            if self.world_size > 1 and dist.is_available() and dist.is_initialized():
                dist.all_reduce(any_local, op=dist.ReduceOp.MAX)
            if int(any_local.item()) == 0:
                # DDP keepalive (token heads exist & enabled): make sure token-head params get a (zero) grad
                if self.world_size > 1 and token_heads_exist:
                    total_loss = total_loss + 0.0 * (self.img_tok_proj.weight.sum() + self.st_node_proj.weight.sum())
                # token_metrics already valid zeros
                return (
                    total_loss,
                    logits_i2st_out,
                    logits_st2i_out,
                    labels_out,
                    img_local,
                    st_local,
                    logit_scale,
                    logit_scale_token,
                    img_global.detach() if img_global is not None else None,
                    st_global.detach()  if st_global  is not None else None,
                    patch_loss.detach() if patch_loss is not None else None,
                    None,
                    token_metrics,
                )

            if not hasattr(data, "cell_id_map"):
                raise ValueError("Token loss enabled but batch is missing 'cell_id_map' (int32 [B,H,W] or [B,1,H,W]).")

            cell_id_map = data.cell_id_map                   # [B,H,W] or [B,1,H,W]

            if st_nodes.dim() != 3:
                raise ValueError(
                    f"Expected data.scconcept_emb to have shape [B,K,D], got {tuple(st_nodes.shape)}"
                )
            if cell_ids.dim() != 2:
                raise ValueError(
                    f"Expected data.scconcept_cell_id to have shape [B,K], got {tuple(cell_ids.shape)}"
                )
            # normalize cell_id_map shape
            if cell_id_map.dim() == 4 and cell_id_map.size(1) == 1:
                cell_id_map = cell_id_map[:, 0]  # -> [B,H,W]
            if cell_id_map.dim() != 3:
                raise ValueError(f"Expected data.cell_id_map to have shape [B,H,W] (or [B,1,H,W]), got {tuple(cell_id_map.shape)}")

            Btok, Kcells, _ = st_nodes.shape
            if Btok != B_local:
                raise ValueError(f"Mismatch: scconcept_emb batch={Btok} but img batch={B_local}.")
            if cell_ids.shape[0] != Btok or cell_ids.shape[1] != Kcells:
                raise ValueError("Mismatch between scconcept_emb [B,K,*] and scconcept_cell_id [B,K].")
            if cell_id_map.shape[0] != Btok:
                raise ValueError("Mismatch between scconcept_emb batch and cell_id_map batch.")

            # Per-cell nucleus pooling weights, computed in chunks over K (no full [B,K,H,W] float tensor)
            valid_id = (cell_ids > 0)  # [B,K]  (0=background, -1=pad)
            cell_ids_map_dtype = cell_ids.to(dtype=cell_id_map.dtype)  # match map dtype for == compare

            Ht, Wt = grid_hw
            Hpx, Wpx = int(cell_id_map.shape[-2]), int(cell_id_map.shape[-1])
            aligned = (Hpx % Ht == 0) and (Wpx % Wt == 0)
            if aligned:
                kh, kw = (Hpx // Ht), (Wpx // Wt)

            # Projected output dim
            Dtok = int(self.img_tok_proj.out_features)
            Ct = int(img_tokens.size(-1))
            img_tokens = img_tokens.contiguous()  # [B,Nt,Ct]

            # Soft-target flags (token branch): targets from the frozen pre-projection teachers
            tok_soft = self.token_loss_type in ("soft_targets", "soft_targets_st")
            tok_need_img = (self.token_loss_type == "soft_targets")

            # Holders (full [B,K,*] to keep all_gather shapes identical)
            img_cell_feat = img_tokens.new_zeros((Btok, Kcells, Dtok))
            valid_cell = torch.zeros((Btok, Kcells), device=img_tokens.device, dtype=torch.bool)
            # Pre-projection pooled image-cell features (H&E teacher of "soft_targets")
            img_cell_raw_full = img_tokens.new_zeros((Btok, Kcells, Ct)) if tok_need_img else None

            # Chunk size over K (8 is a good default for K<=32)
            K_CHUNK = 8
            for k0 in range(0, Kcells, K_CHUNK):
                k1 = min(k0 + K_CHUNK, Kcells)
                ids_chunk = cell_ids_map_dtype[:, k0:k1]         # [B,c]
                valid_chunk = valid_id[:, k0:k1]                 # [B,c]

                # bool mask [B,c,H,W] then pool in float32 (no grad anyway)
                m = (cell_id_map.unsqueeze(1) == ids_chunk.unsqueeze(-1).unsqueeze(-1))
                m = m & valid_chunk.unsqueeze(-1).unsqueeze(-1)
                m = m.to(dtype=torch.float32)

                if aligned:
                    m_tok = F.avg_pool2d(m, kernel_size=(kh, kw), stride=(kh, kw))
                else:
                    m_tok = F.interpolate(m, size=(Ht, Wt), mode="area")

                w_sum = m_tok.sum(dim=(-2, -1), keepdim=True)    # [B,c,1,1]
                v_cell = (w_sum.squeeze(-1).squeeze(-1) > 0.0) & valid_chunk
                valid_cell[:, k0:k1] = v_cell

                w = m_tok / (w_sum + 1e-6)                       # [B,c,Ht,Wt]
                w_flat = w.flatten(2).to(dtype=img_tokens.dtype) # [B,c,Nt] cast for fast bmm

                img_cell_raw = torch.bmm(w_flat, img_tokens)     # [B,c,Ct]
                if tok_need_img:
                    img_cell_raw_full[:, k0:k1] = img_cell_raw   # store pre-proj frozen-teacher H&E
                img_cell_feat[:, k0:k1] = self.img_tok_proj(img_cell_raw)  # [B,c,Dtok]

            # Project ST scConcept embeddings to token embedding dim
            # Let AMP handle dtype; just ensure numeric type matches projection
            st_cell_e = self.st_node_proj(st_nodes.to(dtype=img_cell_feat.dtype))  # [B,K,Dtok]

            # Normalize
            img_cell_z = safe_normalize(img_cell_feat, dim=-1)
            st_cell_z  = safe_normalize(st_cell_e,   dim=-1)

            # all_gather requires identical shapes across ranks
            # The gathered tensors are therefore the fixed [B*K, Dtok] ones (not filtered)
            # Masking/filtering for the loss then happens locally
            Dtok = img_cell_z.size(-1)
            img_all = img_cell_z.reshape(-1, Dtok)            # [B*K, Dtok]
            st_all = st_cell_z.reshape(-1, Dtok)             # [B*K, Dtok]
            ids_all = cell_ids.reshape(-1).to(torch.long)      # [B*K], pad=-1
            valid_all = valid_cell.reshape(-1)                 # [B*K]

            # Ensure invalid cells can never appear as candidates on any GPU
            ids_all = ids_all.masked_fill(~valid_all, -1)

            # Build globally-unique cell ids to avoid false positives across slides
            # Use (slide_idx << 32) | cell_id_int. Keep pad as -1.
            slide_idx = data.slide_idx.to(torch.long)
            if slide_idx.dim() == 0:
                slide_idx = slide_idx.unsqueeze(0)
            slide_flat = slide_idx.view(-1, 1).expand(-1, Kcells).reshape(-1)  # [B*K]
            ids_uid = torch.where(ids_all > 0, (slide_flat << 32) | ids_all, ids_all)  # [B*K]

            # Frozen pre-projection teacher tensors for the token soft targets, flattened to [B*K, *] like st_all/img_all (no grad)
            st_src_all = img_src_all = None
            if tok_soft:
                with torch.no_grad():
                    # st_nodes is the frozen pre-projection scConcept teacher ([B,K,D], fp16).
                    st_src_z = safe_normalize(st_nodes.float(), dim=-1)             # [B,K,D_scconcept]
                    if self.center_teacher_target and (self.scconcept_center_mean is not None):
                        # Center the ST-cell teacher by its global mean (padding rows are removed later by anchor_mask/cand_mask)
                        st_src_z = safe_normalize(
                            st_src_z - self.scconcept_center_mean.to(st_src_z.dtype), dim=-1, eps=1e-8
                        )
                    st_src_all = st_src_z.reshape(-1, st_src_z.size(-1))            # [B*K, D_scconcept]
                    if tok_need_img:
                        img_src_z = safe_normalize(img_cell_raw_full.float(), dim=-1)  # [B,K,Ct]
                        img_src_all = img_src_z.reshape(-1, img_src_z.size(-1))        # [B*K, Ct]

            # Gather global candidates across GPUs (cross-patch + cross-GPU negatives)
            st_src_global_all = img_src_global_all = None
            if self.world_size > 1:
                img_global_all, st_global_all = gather_features(
                    img_all,
                    st_all,
                    gather_with_grad=self.token_gather_with_grad,
                    rank=self.rank,
                    world_size=self.world_size,
                    reinsert_local=self.token_reinsert_local,
                )
                ids_global_all = gather_tensor(ids_uid, rank=self.rank, world_size=self.world_size)
                if tok_soft:
                    # Detached gather (teacher is frozen); same rank order as the projected gather.
                    st_src_global_all = gather_tensor(st_src_all, rank=self.rank, world_size=self.world_size)
                    if tok_need_img:
                        img_src_global_all = gather_tensor(img_src_all, rank=self.rank, world_size=self.world_size)
            else:
                img_global_all, st_global_all = img_all, st_all
                ids_global_all = ids_uid
                if tok_soft:
                    st_src_global_all = st_src_all
                    img_src_global_all = img_src_all

            # Filter anchors (local) and candidates (global) after the gather
            anchor_mask = valid_all & (ids_uid > 0)       # local valid anchors
            cand_mask = (ids_global_all > 0)              # global valid candidates

            img_cell_local = img_all[anchor_mask]         # [N_local, Dtok] (may be empty)
            st_cell_local = st_all[anchor_mask]           # [N_local, Dtok] (may be empty)
            ids_local = ids_uid[anchor_mask]              # [N_local] (may be empty)

            img_cell_global = img_global_all[cand_mask]   # [N_global, Dtok]
            st_cell_global = st_global_all[cand_mask]     # [N_global, Dtok]
            ids_global = ids_global_all[cand_mask]        # [N_global]

            # Frozen teacher anchors/candidates, filtered with the same masks (aligned with st_cell_local/st_cell_global)
            st_src_local = st_src_global = None
            img_src_local = img_src_global = None
            if tok_soft:
                st_src_local = st_src_all[anchor_mask]            # [N_local, D_scconcept]
                st_src_global = st_src_global_all[cand_mask]      # [N_global, D_scconcept]
                if tok_need_img:
                    img_src_local = img_src_all[anchor_mask]      # [N_local, Ct]
                    img_src_global = img_src_global_all[cand_mask]  # [N_global, Ct]

            N_global = int(ids_global.numel())
            N_local = int(ids_local.numel())

            # token_metrics is always returned in the token path (same all_reduce on all ranks)
            assert token_metrics is not None
            K_tok = int(min(5, N_global))
            token_metrics.update({
                "tok_total": N_local,
                "tok_top1_correct_sum": 0,
                "tok_topK_correct_sum": 0,
                "tok_K": K_tok,
            })

            # If there are no valid cells anywhere in the global pool, skip token loss on all ranks.
            if N_global == 0:
                token_loss = None
                if self.world_size > 1 and token_heads_exist:
                    total_loss = total_loss + 0.0 * (self.img_tok_proj.weight.sum() + self.st_node_proj.weight.sum())
                # total_loss unchanged; token_metrics already valid
                return (
                    total_loss,
                    logits_i2st_out,
                    logits_st2i_out,
                    labels_out,
                    img_local,
                    st_local,
                    logit_scale,
                    logit_scale_token,
                    img_global.detach() if img_global is not None else None,
                    st_global.detach()  if st_global  is not None else None,
                    patch_loss.detach() if patch_loss is not None else None,
                    None,
                    token_metrics,
                )

            if self.token_local_loss:
                # --------- Token logits: local x global
                if N_local == 0:
                    token_loss = None
                    if self.world_size > 1 and token_heads_exist:
                        total_loss = total_loss + 0.0 * (self.img_tok_proj.weight.sum() + self.st_node_proj.weight.sum())
                else:
                    logits_i2st_cell = (logit_scale_token * (img_cell_local @ st_cell_global.T)).float()  # [N_local,N_global]
                    logits_st2i_cell = (logit_scale_token * (st_cell_local  @ img_cell_global.T)).float()  # [N_local,N_global]

                    # Loss
                    if self.token_loss_type in ("soft_targets", "soft_targets_st"):
                        # Similarity-adjusted soft targets from the frozen pre-projection teacher (no grad)
                        with torch.no_grad():
                            sim_src = (st_src_local @ st_src_global.T)   # [N_local,N_global] st <-> st (teacher)
                            if self.token_loss_type == "soft_targets":
                                # symmetric variant: average in the image-cell <-> image-cell similarity (pre-proj)
                                sim_src = 0.5 * (sim_src + (img_src_local @ img_src_global.T))
                            targets = torch.softmax(sim_src * token_target_scale, dim=-1).float()
                        loss_i = _soft_cross_entropy(logits_i2st_cell, targets, reduction="mean")
                        loss_j = _soft_cross_entropy(logits_st2i_cell, targets, reduction="mean")
                        token_loss = 0.5 * (loss_i + loss_j)
                    else:
                        # classic: multi-positive InfoNCE on cell_id (multi-positive because one cell_id can appear in several patches)
                        pos = (ids_local[:, None] == ids_global[None, :])
                        loss_i = _multi_positive_infonce(logits_i2st_cell, pos)
                        loss_j = _multi_positive_infonce(logits_st2i_cell, pos)
                        token_loss = 0.5 * (loss_i + loss_j)

            else:
                # --------- Token logits: global x global (expensive)
                # Each rank computes the same loss; gradients flow only if token_gather_with_grad or token_reinsert_local
                img_cell_rows = img_cell_global      # [N_global,Dtok]
                st_cell_rows = st_cell_global       # [N_global,Dtok]
                ids_rows = ids_global           # [N_global]

                logits_i2st_cell = (logit_scale_token * (img_cell_rows @ st_cell_global.T)).float()  # [N_global,N_global]
                logits_st2i_cell = logits_i2st_cell.T

                if self.token_loss_type in ("soft_targets", "soft_targets_st"):
                    # Rows are global: teacher rows = global teacher tensor
                    with torch.no_grad():
                        sim_src = (st_src_global @ st_src_global.T)    # [N_global,N_global] st <-> st (teacher)
                        if self.token_loss_type == "soft_targets":
                            # symmetric variant: average in the image-cell <-> image-cell similarity (pre-proj)
                            sim_src = 0.5 * (sim_src + (img_src_global @ img_src_global.T))
                        targets = torch.softmax(sim_src * token_target_scale, dim=-1).float()
                    loss_i = _soft_cross_entropy(logits_i2st_cell, targets, reduction="mean")
                    loss_j = _soft_cross_entropy(logits_st2i_cell, targets, reduction="mean")
                    token_loss = 0.5 * (loss_i + loss_j)
                else:
                    pos = (ids_rows[:, None] == ids_global[None, :])  # [N_global,N_global]
                    loss_i = _multi_positive_infonce(logits_i2st_cell, pos)
                    loss_j = _multi_positive_infonce(logits_st2i_cell, pos)
                    token_loss = 0.5 * (loss_i + loss_j)

            # Apply scaling only if token_loss was computed
            if token_loss is not None:
                local_rows = (N_local if self.token_local_loss else N_global)
                token_scale = self._ddp_loss_scale(
                    local_rows=local_rows,
                    global_rows=N_global,
                    local_loss=self.token_local_loss,
                    gather_with_grad=self.token_gather_with_grad,
                )
                total_loss = total_loss + (self.lambda_token * token_loss) * token_scale

            # Token retrieval metrics (always computed on local anchors; i2st direction)
            if N_local > 0 and N_global > 0:
                with torch.no_grad():
                    # metrics logits are local x global regardless of token_local_loss choice (avoid double-counting)
                    logits_i2st_m = (logit_scale_token * (img_cell_local @ st_cell_global.T)).float()
                    pred_top1 = logits_i2st_m.argmax(dim=1)  # [N_local]
                    correct_top1_sum = (ids_global[pred_top1] == ids_local).sum().item()
                    K_tok = min(5, logits_i2st_m.size(1))
                    topk_idx = logits_i2st_m.topk(K_tok, dim=1).indices  # [N_local,K]
                    topk_ids = ids_global[topk_idx]                      # [N_local,K]
                    correct_topK_sum = (topk_ids == ids_local.unsqueeze(1)).any(dim=1).sum().item()
                    token_metrics["tok_top1_correct_sum"] = int(correct_top1_sum)
                    token_metrics["tok_topK_correct_sum"] = int(correct_topK_sum)
                    token_metrics["tok_K"] = int(K_tok)

        return (
            total_loss,
            logits_i2st_out,
            logits_st2i_out,
            labels_out,
            img_local,
            st_local,
            logit_scale,
            logit_scale_token,
            img_global.detach() if img_global is not None else None,
            st_global.detach()  if st_global  is not None else None,
            patch_loss.detach() if patch_loss is not None else None,
            token_loss.detach() if token_loss is not None else None,
            token_metrics,
        )
