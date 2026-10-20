import logging
import contextlib
logger = logging.getLogger(__name__)

import math
import numpy as np
import torch
import torch.distributed as dist
from typing import Dict, List, Optional, Tuple

from src.utils.utils import get_dist_world_size, get_dist_rank


# -------------------------------------------------------------------------------------------------------------- DDP helpers


def gather_to_rank0(tensor: torch.Tensor):
    """
    Gather a variable-length tensor from every rank onto rank-0 (CPU) with dist.gather_object and
    return the concatenated result on rank-0 (None on other ranks; the input on a single GPU).
    """
    if not (dist.is_available() and dist.is_initialized()):
        return tensor  # single-GPU case
    
    world_size = get_dist_world_size()
    rank = get_dist_rank()

    tensor_cpu = tensor.detach().to("cpu")

    if rank == 0:
        gathered = [None for _ in range(world_size)]
        dist.gather_object(tensor_cpu, gathered, dst=0)
        tensor_global = torch.cat(gathered, dim=0)
        del gathered
        return tensor_global

    dist.gather_object(tensor_cpu, None, dst=0)
    return None



def broadcast_scalars_from_rank0(values: List[float], device: torch.device) -> List[float]:
    """
    Broadcasts a list of floats from rank-0 to all ranks and returns them on every rank.
    """
    tensor = torch.tensor(values, dtype=torch.float32, device=device)
    if dist.is_available() and dist.is_initialized():
        dist.broadcast(tensor, src=0)
    return tensor.cpu().tolist()


def _retrieval_topk_mrr_counts(
    logits: torch.Tensor,
    labels: torch.Tensor,
    topk_all: torch.Tensor,
    idx_local: torch.Tensor,
    acc_top_k_list: List[int],
) -> Tuple[int, Dict[int, int], float]:
    """
    Cross-modal top-k accuracy + MRR over one [R, N] logits block (used by the global and within-slide metrics).

    Args:
        logits:    [R, N] similarity logits (query rows x gallery cols).
        labels:    [R] gold column index for each query row.
        topk_all:  [R, K_all] descending top-index block, K_all >= max(acc_top_k_list).
        idx_local: [R] row arange (== torch.arange(R)) on the same device as ``logits``.
        acc_top_k_list: k values for top-k accuracy (top-1 is always computed separately).
    Returns:
        (correct_top1, {k: correct_topk for k in acc_top_k_list}, rr_sum) as plain Python
        counts / sum, to be accumulated by the caller.
    """
    correct_top1 = (logits.argmax(1) == labels).sum().item()
    correct_topk = {
        k: (topk_all[:, :k] == labels.unsqueeze(1)).any(1).sum().item()
        for k in acc_top_k_list
    }
    pos_scores = logits[idx_local, labels]
    ranks = (logits >= pos_scores.unsqueeze(1)).sum(1)
    rr_sum = (1.0 / ranks).sum().item()
    return correct_top1, correct_topk, rr_sum




# -------------------------------------------------------------------------------------------------------------- 1. Step metrics (train/val step)
# => Step metrics are collected for each training/validation step.
# => For the following metrics: Accuracies top1, MRR, Cosine similarities, Graded nDCG@K, Spearman's rho

@torch.no_grad()
def collect_step_counts(
    logits_i2st: torch.Tensor,
    logits_st2i: torch.Tensor,
    labels: torch.Tensor,
    logit_scale: torch.Tensor,
    img_local: Optional[torch.Tensor] = None,
    st_local: Optional[torch.Tensor] = None,
    img_global: Optional[torch.Tensor] = None,
    st_global: Optional[torch.Tensor] = None,
    topk_list_step: Tuple[int, ...] = (10, 20, 100),
) -> Dict[str, float]:
    """
    Return counts (correct, total) for both directions, sums for: accuracies top1, MRR, cosine similarities, graded nDCG@K and Spearman's rho.
    Supports both local x global logits (local_loss=True) and global x global logits (local_loss=False) in DDP.
    DDP all-reduces are performed inside to keep ranks in sync.
    """

    # In DDP, logits can be either:
    #  - local x global:  [B_local, B_global]  (patch_local_loss=True)
    #  - global x global: [B_global, B_global] (patch_local_loss=False)
    # Only the local query rows are evaluated on each rank, then the sums are all-reduced.

    if img_local is not None:
        bs_local = int(img_local.size(0))
    else:
        bs_local = int(logits_i2st.size(0))

    if dist.is_available() and dist.is_initialized():
        world_size = get_dist_world_size()
        rank = get_dist_rank()
    else:
        world_size = 1
        rank = 0

    # Detect global x global logits and slice to local rows for this rank.
    # (This assumes rank-major concatenation, consistent with gather_features() and get_ground_truth().)
    if (
        world_size > 1
        and logits_i2st.dim() == 2
        and logits_i2st.size(0) == logits_i2st.size(1)
        and logits_i2st.size(0) == bs_local * world_size
        and labels.numel() == logits_i2st.size(0)
    ):
        start = rank * bs_local
        end = start + bs_local
        logits_i2st = logits_i2st[start:end, :]
        logits_st2i = logits_st2i[start:end, :]
        labels = labels[start:end]

    # Batch size = number of query rows evaluated on this rank
    bs = logits_i2st.size(0)

    # Total number of samples
    total_samples = torch.tensor(bs, device=logits_i2st.device)
    # Row indices
    row_idx = torch.arange(bs, device=logits_i2st.device)

    # --------- Top-1 retrieval correctness
    correct_i2st_sum = (logits_i2st.argmax(dim=1) == labels).sum()
    correct_st2i_sum = (logits_st2i.argmax(dim=1) == labels).sum()

    # --------- MRR
    # rank_i2st
    pos_i2st = logits_i2st[row_idx, labels]
    rank_i2st = (logits_i2st >= pos_i2st.unsqueeze(1)).sum(1)
    rr_i2st_sum = (1.0 / rank_i2st).sum()
    # rank_st2i
    pos_st2i = logits_st2i[row_idx, labels]
    rank_st2i = (logits_st2i >= pos_st2i.unsqueeze(1)).sum(1)
    rr_st2i_sum = (1.0 / rank_st2i).sum()

    # --------- Cosine similarity (i->st), positives/negatives
    # positive sims
    pos_sim = logits_i2st[row_idx, labels] / logit_scale  
    # hardest negative sims (max over non-matching columns) 
    pos_sim_sum = pos_sim.sum()
    if logits_i2st.size(1) > 1:
        sim_mat = (logits_i2st / logit_scale).clone()
        sim_mat[row_idx, labels] = -1e9   # robust exclusion of positives
        neg_sim, _ = sim_mat.max(1)
        neg_sim_sum = neg_sim.sum()
    else:
        neg_sim_sum = torch.zeros((), device=logits_i2st.device, dtype=pos_sim.dtype)

    # All reduce counts
    if dist.is_available() and dist.is_initialized():
        for t in (correct_i2st_sum, correct_st2i_sum, total_samples, pos_sim_sum, neg_sim_sum, rr_i2st_sum, rr_st2i_sum):
            dist.all_reduce(t, op=dist.ReduceOp.SUM)

    # Output dictionary
    out = {
        "correct_i2st_sum": float(correct_i2st_sum.item()),
        "correct_st2i_sum": float(correct_st2i_sum.item()),
        "total_samples": float(total_samples.item()),
        "pos_sim_sum": float(pos_sim_sum.item()),
        "neg_sim_sum": float(neg_sim_sum.item()),
        "rr_i2st_sum": float(rr_i2st_sum.item()),
        "rr_st2i_sum": float(rr_st2i_sum.item()),
    }

    # --------- Batch-level graded nDCG@K and Spearman's ρ
    if (img_local is not None) and (st_local is not None) and (img_global is not None) and (st_global is not None):   # always true (clip.py sets want_global=True)
        
        # gains (force FP32 for metric math stability)
        gains = 0.5 * (img_local @ img_global.t() + st_local @ st_global.t())      # [B,N] in [-1,1]
        gains = ((gains + 1.0) * 0.5).float()                                      # [B,N] in [0,1], FP32
        # scores using cross-modal logits (divided by logit_scale to get cosine similarities). Cast to FP32.
        scores_i2st = ((logits_i2st / logit_scale).float())                        # [B,N]
        scores_st2i = ((logits_st2i / logit_scale).float())                        # [B,N]

        # nDCG@K
        def _ndcg(scores: torch.Tensor, gains_row: torch.Tensor, K: int) -> torch.Tensor:
            K = min(K, scores.size(1))
            topk_idx = scores.topk(K, dim=1).indices     # [B,K]
            ar = torch.arange(2, K + 2, device=scores.device, dtype=torch.float32)  # FP32 discount for stability even under AMP
            disc = (1.0 / torch.log2(ar)).unsqueeze(0)   # [1,K]
            # DCG@K: gather gains at predicted order
            r = torch.gather(gains_row, 1, topk_idx)     # [B,K]
            dcg = (r.to(torch.float32) * disc).sum(dim=1)    # [B]
            # IDCG@K: sort gains_row per query
            ideal = gains_row.topk(K, dim=1).values.to(torch.float32)      # [B,K]
            idcg = (ideal * disc).sum(dim=1) + 1e-8      # [B]
            return (dcg / idcg)                          # [B]

        # Spearman's ρ
        def _spearman(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            a = a.float()
            b = b.float()
            # ranks
            ra_idx = a.argsort(dim=1)
            rb_idx = b.argsort(dim=1)
            ra = torch.empty_like(ra_idx, dtype=torch.float32)
            rb = torch.empty_like(rb_idx, dtype=torch.float32)
            ra.scatter_(1, ra_idx, torch.arange(a.size(1), device=a.device, dtype=torch.float32).unsqueeze(0).expand_as(a))
            rb.scatter_(1, rb_idx, torch.arange(b.size(1), device=b.device, dtype=torch.float32).unsqueeze(0).expand_as(b))
            ra = ra - ra.mean(dim=1, keepdim=True)
            rb = rb - rb.mean(dim=1, keepdim=True)
            num = (ra * rb).sum(dim=1)
            den = torch.sqrt((ra.pow(2).sum(dim=1) + 1e-8) * (rb.pow(2).sum(dim=1) + 1e-8))
            return (num / den)  # [B]

        # accumulate
        ndcg_sums = {}
        # nDCG@K sums
        for K in topk_list_step:
            v_i_sum = _ndcg(scores_i2st, gains, K).sum()
            v_s_sum = _ndcg(scores_st2i, gains, K).sum()
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(v_i_sum, op=dist.ReduceOp.SUM)
                dist.all_reduce(v_s_sum, op=dist.ReduceOp.SUM)
            ndcg_sums[f"ndcg_i2st@{K}_sum"] = float(v_i_sum.item())
            ndcg_sums[f"ndcg_st2i@{K}_sum"] = float(v_s_sum.item())
        out.update(ndcg_sums)
        # Spearman ρ sums
        sp_i_sum = _spearman(scores_i2st, gains).sum()
        sp_s_sum = _spearman(scores_st2i, gains).sum()
        # all-reduce
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(sp_i_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(sp_s_sum, op=dist.ReduceOp.SUM)
        out["spearman_i2st_sum"] = float(sp_i_sum.item())
        out["spearman_st2i_sum"] = float(sp_s_sum.item())

    return out



# -------------------------------------------------------------------------------------------------------------- 2. Global metrics (full valid set)
# => Global metrics are computed on the full validation set
# => For the following metrics: 
#          - Retrieval metrics: Accuracies topK, MRR, Cosine similarities, Graded nDCG@K, 
#          - Other metrics: Hubness, ECE, kNN-agreement, CKA
#          - Geometric metrics: Alignment, Spectrum, Uniformity, MMD



def global_metrics_computation(
    all_img_embeds: torch.Tensor,
    all_st_embeds: torch.Tensor,
    logit_scale: torch.Tensor,
    mode: str,
    acc_top_k_list: List[int],
    ndcg_top_k_list: List[int],
    chunk_size: int,
    device: torch.device,
    slide_idx: torch.Tensor,
    slide_id2idx: Dict[str, int],
    compute_per_slide_ndcg: Optional[bool] = None,
    subset_slide_idx: Optional[List[int]] = None,
    all_img_feats: Optional[torch.Tensor] = None,   # pre-CLIP H&E features (for CKA)
    all_st_feats: Optional[torch.Tensor] = None,    # pre-CLIP ST features (for CKA)
    all_scconcept_feats: Optional[torch.Tensor] = None,  # pre-CLIP scConcept patch-pooled features (for CKA)
) -> Dict[str, float]:
    """
    Computes global metrics (optionally incl. CKA) on the whole dataset.
    Rank-0 keeps the full matrices; results are broadcast to all ranks.
    """
    # If not specified, compute per-slide accuracies only for ALL (subset_slide_idx is None).
    if compute_per_slide_ndcg is None:
        compute_per_slide_ndcg = (subset_slide_idx is None)

    # Gather partial tensors -> rank-0 (CPU)
    imgs_cpu = gather_to_rank0(all_img_embeds)
    sts_cpu = gather_to_rank0(all_st_embeds)
    slide_idx = gather_to_rank0(slide_idx)
    # Optional: gather raw features for CKA
    has_cka = (all_img_feats is not None) and (all_st_feats is not None)
    # scConcept CKA (against the H&E features): only if has_cka and scConcept features are given
    has_scconcept_cka = has_cka and (all_scconcept_feats is not None)
    if has_cka:
        img_feats_cpu = gather_to_rank0(all_img_feats)
        st_feats_cpu  = gather_to_rank0(all_st_feats)
    if has_scconcept_cka:
        scconcept_feats_cpu = gather_to_rank0(all_scconcept_feats)

    # Num slides
    num_slides = len(slide_id2idx)
    # Sanity check: mapping must be a contiguous permutation 0..num_slides-1
    _idx_vals = np.array(list(slide_id2idx.values()), dtype=np.int64)
    assert _idx_vals.min() == 0 and _idx_vals.max() == num_slides - 1 and np.unique(_idx_vals).size == num_slides, "slide_id2idx must map to contiguous indices 0..num_slides-1"

    # Ordered list of the metric keys (order and length of the broadcast payload)
    def _metric_keys() -> List[str]:
        # 2 (acc top1) + 2*|accK| + (2*num_slides if per-slide, K=100 only) + 2 (MRR)
        # + 3 (cos stats) + 2*|ndcgK| + 4 (hubness) + 2 (ECE) + 2 (kNN agreement)
        # + (3 if CKA, +1 if scConcept CKA) + 1 (alignment) + 6 (spectrum effrank/partratio img/st/both)
        # + 3 (uniformity img/st/both) + 1 (mmd)
        keys: List[str] = [
            f"{mode}_global_acc_i2st",
            f"{mode}_global_acc_st2i",
        ]
        for k in acc_top_k_list:
            keys.append(f"{mode}_global_acc_i2st_top{k}")
            keys.append(f"{mode}_global_acc_st2i_top{k}")
        if compute_per_slide_ndcg:
            sorted_ids = [sid for sid, _ in sorted(slide_id2idx.items(), key=lambda kv: kv[1])]
            for sid in sorted_ids:
                keys.append(f"{mode}_{sid}_ndcg_i2st_top100")
                keys.append(f"{mode}_{sid}_ndcg_st2i_top100")
        keys += [
            f"{mode}_global_mrr_i2st",
            f"{mode}_global_mrr_st2i",
            f"{mode}_global_pos_sim",
            f"{mode}_global_neg_sim",
            f"{mode}_global_pos_neg_gap",
        ]
        for k in ndcg_top_k_list:
            keys.append(f"{mode}_global_ndcg_i2st_top{k}")
            keys.append(f"{mode}_global_ndcg_st2i_top{k}")
        keys += [
            f"{mode}_global_hub_gini_i2st@10",
            f"{mode}_global_hub_top1p_i2st@10",
            f"{mode}_global_hub_gini_st2i@10",
            f"{mode}_global_hub_top1p_st2i@10",
            f"{mode}_global_ece_i2st_bins15",
            f"{mode}_global_ece_st2i_bins15",
            f"{mode}_global_knn_overlap@10_i_anchor",
            f"{mode}_global_knn_jaccard@10_i_anchor",
        ]
        if has_cka:
            keys += [
                f"{mode}_global_cka_he",
                f"{mode}_global_cka_st",
                f"{mode}_global_cka_he_st_pre",
            ]
            if has_scconcept_cka:
                keys.append(f"{mode}_global_cka_he_scconcept_pre")
        keys += [
            f"{mode}_global_align_mse",
            f"{mode}_global_effrank_img",
            f"{mode}_global_effrank_st",
            f"{mode}_global_effrank_both",
            f"{mode}_global_partratio_img",
            f"{mode}_global_partratio_st",
            f"{mode}_global_partratio_both",
            f"{mode}_global_uni_img",
            f"{mode}_global_uni_st",
            f"{mode}_global_uni_both",
            f"{mode}_global_mmd_rbf",
        ]
        return keys

    # Same on every rank (same broadcast payload shape)
    metric_keys = _metric_keys()

    # Move logit_scale to GPU
    if device.type == "cuda":
        logit_scale = logit_scale.to(device)

    rank = get_dist_rank() if (dist.is_available() and dist.is_initialized()) else 0

    #######################################
    # #### ON RANK 0: full computation ####
    #######################################
    if rank == 0:
        # Fully chunked transfer; never move the full matrices at once

        # Hubness / ECE / kNN-agreement / Uniformity settings
        hub_k = 10
        ece_bins = 15
        knn_k = 10
        uni_t = 2.0

        # Optional subset restriction (INT / EXT): select rows/cols by slide indices mask ----------------------------------------
        if subset_slide_idx is not None:
            slide_idx_cpu = slide_idx.to(torch.long).cpu().numpy()
            mask = np.isin(slide_idx_cpu, np.array(subset_slide_idx, dtype=np.int64))
            sel = np.nonzero(mask)[0]
            if sel.size == 0:
                # Build a zero payload matching the normal layout exactly
                out = {k: 0.0 for k in metric_keys}
                n_working = 0
            else:
                sel_t = torch.from_numpy(sel).to(torch.long)
                imgs_cpu = imgs_cpu.index_select(0, sel_t)
                sts_cpu = sts_cpu.index_select(0, sel_t)
                slide_idx_cpu = slide_idx_cpu[sel]
                if has_cka:
                    img_feats_cpu = img_feats_cpu.index_select(0, sel_t)
                    st_feats_cpu  = st_feats_cpu.index_select(0, sel_t)
                if has_scconcept_cka:
                    scconcept_feats_cpu = scconcept_feats_cpu.index_select(0, sel_t)
                n_working = imgs_cpu.size(0)
        
        else:
            slide_idx_cpu = slide_idx.to(torch.long).cpu().numpy()
            n_working = imgs_cpu.size(0)


        # If nothing to do after (maybe empty dataset), emit a fully shaped zero payload
        if n_working == 0:
            if 'out' not in locals():
                out = {k: 0.0 for k in metric_keys}
            # fall through to broadcast at the end of the function scope
        else:

            # Effective Ks (robust to small datasets) ----------------------------------------
            # Effective k for kNN agreement: at most n-1 neighbors, 0 if n<2
            kk_eff = (min(knn_k, max(0, n_working - 1)))
            # Effective max-K for accuracy & nDCG
            max_k_acc_raw = max(acc_top_k_list + [1])
            max_k_acc_eff = min(n_working, max_k_acc_raw)
            maxK_ndcg_raw = max(ndcg_top_k_list) if len(ndcg_top_k_list) > 0 else 0
            maxK_ndcg_eff = min(n_working, maxK_ndcg_raw)
            # Hubness@k should not exceed available candidates
            hub_k_eff = min(hub_k, n_working) if n_working > 0 else 0

            # Accumulators initialization ----------------------------------------
            # Accuracy counters
            max_k_acc = max_k_acc_eff
            correct_i2st1 = 0
            correct_st2i1 = 0
            correct_i2st = {k: 0 for k in acc_top_k_list}
            correct_st2i = {k: 0 for k in acc_top_k_list}
            # Per-slide nDCG@K counters (K=100 is the exposed one)
            # Queries retrieve over the whole candidate pool; only the aggregation is per slide
            if compute_per_slide_ndcg:
                ndcg_i2st_slide_sum = {K: np.zeros(num_slides, dtype=np.float64) for K in ndcg_top_k_list}
                ndcg_st2i_slide_sum = {K: np.zeros(num_slides, dtype=np.float64) for K in ndcg_top_k_list}
                # same denominator for both directions: number of queries per slide
                ndcg_slide_count = np.zeros(num_slides, dtype=np.int64)
            # MRR counters
            rr_sum_i2st = rr_sum_st2i = 0.0
            # Cosine similarity counters
            inv_scale = 1.0 / (logit_scale.item() if torch.is_tensor(logit_scale) else float(logit_scale))
            pos_sim_sum = 0.0
            neg_sim_sum = 0.0
            # nDCG streaming state
            maxK_ndcg = maxK_ndcg_eff
            disc_cache = {
                K: (1.0 / torch.log2(torch.arange(2, K + 2, dtype=torch.float32))).view(1, K)
                for K in ndcg_top_k_list
            }
            idcg_rows = {K: torch.empty(n_working, dtype=torch.float32) for K in ndcg_top_k_list} if maxK_ndcg > 0 else {}
            ndcg_sum_i2st = {K: 0.0 for K in ndcg_top_k_list}
            ndcg_sum_st2i = {K: 0.0 for K in ndcg_top_k_list}
            # Hubness counters (columns voted in top-k)
            hub_counts_i2st = torch.zeros(n_working, dtype=torch.long)
            hub_counts_st2i = torch.zeros(n_working, dtype=torch.long)
            # ECE accumulators
            ece_cnt_i2st = torch.zeros(ece_bins, dtype=torch.float64)
            ece_cor_i2st = torch.zeros(ece_bins, dtype=torch.float64)
            ece_conf_i2st = torch.zeros(ece_bins, dtype=torch.float64)
            ece_cnt_st2i = torch.zeros(ece_bins, dtype=torch.float64)
            ece_cor_st2i = torch.zeros(ece_bins, dtype=torch.float64)
            ece_conf_st2i = torch.zeros(ece_bins, dtype=torch.float64)
            # kNN agreement holders (image-anchored)
            knn_i2st_all = torch.empty((n_working, kk_eff), dtype=torch.long) if kk_eff > 0 else None
            knn_imgimg_all = torch.empty((n_working, kk_eff), dtype=torch.long) if kk_eff > 0 else None
            # NB: CKA, Alignment, Spectrum do not need chunking; will be done later if requested
            # Uniformity accumulators
            exp_minus_2t = math.exp(-2.0 * uni_t)
            sum_exp_imgimg = 0.0   # \sum exp(2t * cos) over image-image (includes diag)
            sum_exp_stst  = 0.0    # \sum exp(2t * cos) over st-st     (includes diag)
            sum_exp_imgst = 0.0    # \sum exp(2t * cos) over image-st  (one direction)
            # RBF MMD accumulators
            both_pool = torch.cat([imgs_cpu, sts_cpu], dim=0)    # [2n, d], CPU
            Npool = both_pool.size(0)
            svec = both_pool.sum(dim=0, dtype=torch.float64)     # sum of vectors
            mean_cos_pool = float(((svec @ svec).item() - Npool) / max(1, (Npool * (Npool - 1))))
            mean_cos_pool = max(-1.0, min(1.0, mean_cos_pool))
            sigma = max(1e-6, (2.0 - 2.0 * mean_cos_pool))
            gamma = 1.0 / (2.0 * sigma)
            exp_minus_2g = math.exp(-2.0 * gamma)
            sum_k_imgimg = 0.0     # \sum k(x,x') incl. diag (k=1 on diag)
            sum_k_stst  = 0.0      # \sum k(y,y') incl. diag
            sum_k_imgst = 0.0      # \sum k(x,y)

            def _gather_topk_pair_gains(
                q_img: torch.Tensor,
                q_st: torch.Tensor,
                topk_idx: torch.Tensor,
            ) -> torch.Tensor:
                """
                Exact gain retrieval for the selected top-k candidates without materializing
                huge [R, K, D] tensors.

                gain(i, j) = 0.5 * (img_i·img_j + st_i·st_j), then mapped to [0, 1].

                Args:
                    q_img:    [R, D] query-side image embeddings (device tensor)
                    q_st:     [R, D] query-side ST embeddings    (device tensor)
                    topk_idx: [R, K] candidate indices (CPU long tensor)
                Returns:
                    [R, K] gains in [0, 1] on CPU.
                """
                R, K = topk_idx.shape
                out = torch.empty((R, K), dtype=torch.float32)

                for j_start in range(0, n_working, chunk_size):
                    j_end = min(j_start + chunk_size, n_working)
                    hit = (topk_idx >= j_start) & (topk_idx < j_end)
                    if not bool(hit.any()):
                        continue

                    imgs_c = imgs_cpu[j_start:j_end]
                    sts_c = sts_cpu[j_start:j_end]
                    if device.type == "cuda":
                        imgs_c = imgs_c.to(device)
                        sts_c = sts_c.to(device)

                    sim_img = (q_img @ imgs_c.t()).float().cpu()   # [R, J]
                    sim_st  = (q_st  @ sts_c.t()).float().cpu()    # [R, J]

                    rows, cols = hit.nonzero(as_tuple=True)
                    local_cols = (topk_idx[rows, cols] - j_start).to(torch.long)
                    out[rows, cols] = 0.5 * (
                        sim_img[rows, local_cols] + sim_st[rows, local_cols]
                    )

                    del imgs_c, sts_c, sim_img, sim_st, rows, cols, local_cols

                return (out + 1.0) * 0.5

            # image -> st computation ----------------------------------------
            for i_start in range(0, n_working, chunk_size):
                
                # Chunk img rows
                i_end = min(i_start + chunk_size, n_working)
                img_chunk = imgs_cpu[i_start:i_end]
                if device.type == "cuda":
                    img_chunk = img_chunk.to(device)
                R = i_end - i_start  # rows in this chunk

                # Paired ST rows (for gains in IDCG to ultimately compute nDCG)
                st_rows = sts_cpu[i_start:i_end]
                if device.type == "cuda":
                    st_rows = st_rows.to(device)
                
                # Streaming top-K indices/values for image-image similarity for knn agreement
                if kk_eff > 0:
                    best_vals = torch.full((R, kk_eff), -1e9, dtype=torch.float32)   # best values for image-image sim 
                    best_idx = torch.full((R, kk_eff), -1, dtype=torch.long)

                # Streaming top-K gains for IDCG
                best_gain = torch.full((R, maxK_ndcg), -1e9, dtype=torch.float32) if maxK_ndcg > 0 else None

                # Build logit for this image-row-chunk across all ST embeddings, in column ST chunks
                # Take also corresponding images for image-image sim block 
                row_logits_parts = []
                for j_start in range(0, n_working, chunk_size):
                    j_end = min(j_start + chunk_size, n_working)
                    st_chunk = sts_cpu[j_start:j_end]
                    imgs_c = imgs_cpu[j_start:j_end]
                    if device.type == "cuda":
                        st_chunk = st_chunk.to(device)
                        imgs_c = imgs_c.to(device)
                    
                    # Partial logits
                    part = logit_scale * img_chunk @ st_chunk.t()
                    part_cpu = part.detach().cpu()
                    row_logits_parts.append(part_cpu)

                    # Image-image sim block (used in both kNN and IDCG; keep an unmasked copy!)
                    s_img_block_raw = (img_chunk @ imgs_c.t()).float().cpu()  # [R, J]
                    # kNN agreement (mask self only for kNN), leave raw block intact for IDCG
                    if kk_eff > 0:
                        s_knn_block = s_img_block_raw.clone()
                        gi = torch.arange(i_start, i_end)   # global row indices (CPU)
                        mask_rows = (gi >= j_start) & (gi < j_end)
                        if mask_rows.any():
                            pos = (gi[mask_rows] - j_start).to(torch.long)
                            s_knn_block[mask_rows.nonzero(as_tuple=False).squeeze(1), pos] = -1e9
                        # update best values/indices
                        cand_vals = torch.cat([best_vals, s_knn_block], dim=1)
                        cand_idx = torch.cat([best_idx, torch.arange(j_start, j_end).view(1, -1).expand(R, -1)], dim=1)
                        new_vals, new_pos = cand_vals.topk(kk_eff, dim=1)
                        best_vals = new_vals
                        best_idx = torch.gather(cand_idx, 1, new_pos)

                    # Intra-modal st-st block (always compute; used for IDCG & exact geom)
                    s_st_block = (st_rows @ st_chunk.t()).float().cpu()      # [R, J]
                    # IDCG streaming: average intra-modal gains = 0.5*(img·img + st·st)
                    if maxK_ndcg > 0:
                        g_block = 0.5 * (s_img_block_raw + s_st_block)        # [-1,1], CPU (Unmasked)
                        cand_gain = torch.cat([best_gain, g_block], dim=1)
                        best_gain = cand_gain.topk(maxK_ndcg, dim=1).values

                    # Uniformity: accumulate exp(2t*cos) in each region
                    sum_exp_imgimg += torch.exp((2.0 * uni_t) * s_img_block_raw).double().sum().item()
                    sum_exp_stst += torch.exp((2.0 * uni_t) * s_st_block).double().sum().item()
                    sum_exp_imgst += torch.exp((2.0 * uni_t) * (part_cpu * inv_scale)).double().sum().item()
                    
                    # Exact RBF kernel sums for MMD
                    sum_k_imgimg += (exp_minus_2g * torch.exp((2.0 * gamma) * s_img_block_raw)).double().sum().item()
                    sum_k_stst += (exp_minus_2g * torch.exp((2.0 * gamma) * s_st_block)).double().sum().item()
                    sum_k_imgst += (exp_minus_2g * torch.exp((2.0 * gamma) * (part_cpu * inv_scale))).double().sum().item()

                    # Free GPU mem immediately
                    del st_chunk, imgs_c, part, s_img_block_raw, s_st_block, part_cpu
                    if kk_eff > 0:
                        del cand_vals, cand_idx, new_vals, new_pos, s_knn_block
                    if maxK_ndcg > 0:
                        del g_block, cand_gain
                
                # Concat to full [i_end-i_start, n]
                logits_row = torch.cat(row_logits_parts, dim=1)
                idx_local = torch.arange(logits_row.size(0), device=logits_row.device)
                cos_row = logits_row * inv_scale
                labels_row = torch.arange(i_start, i_end, device=logits_row.device)

                # Get top-K_all indices for all needed K once
                K_all = max(1, max(max_k_acc, kk_eff, hub_k_eff, (maxK_ndcg if maxK_ndcg > 0 else 1)))
                topk_all = logits_row.topk(K_all, 1).indices

                # Accuracy + MRR (topk_all is reused for nDCG / hubness below)
                c1, ck, rr = _retrieval_topk_mrr_counts(
                    logits_row, labels_row, topk_all, idx_local, acc_top_k_list
                )
                correct_i2st1 += c1
                for k in acc_top_k_list:
                    correct_i2st[k] += ck[k]
                rr_sum_i2st += rr

                # Cosine stats (only img->st direction to avoid double-counting)
                labels_cpu = labels_row.cpu()
                pos_sim_sum += cos_row[idx_local, labels_cpu].sum().item()
                if n_working > 1:   # if only one example exists, there is no valid negative
                    cos_row[idx_local, labels_cpu] = -float("inf")  # robust exclusion of positives
                    neg_sim_sum += cos_row.max(1)[0].sum().item()

                # nDCG: compute DCG at predicted indices, finalize and store IDCG rows
                if maxK_ndcg > 0:
                    sel = topk_all[:, :maxK_ndcg].cpu()            # [R, maxK]
                    g_pred = _gather_topk_pair_gains(img_chunk, st_rows, sel)   # [R, maxK] on CPU
                    best01 = (best_gain + 1.0) * 0.5               # [R,maxK]
                    for K in ndcg_top_k_list:
                        K_eff = min(K, maxK_ndcg)
                        if K_eff == 0:
                            continue
                        disc = disc_cache[K][:, :K_eff]
                        idcg_block = (best01[:, :K_eff] * disc).sum(1)      # [R]
                        idcg_rows[K][i_start:i_end] = idcg_block            # save for reuse
                        dcg_block = (g_pred[:, :K_eff]  * disc).sum(1)      # [R]
                        ndcg_block = dcg_block / (idcg_block + 1e-8)        # [R]
                        ndcg_sum_i2st[K] += ndcg_block.sum().item()

                        # Per-slide nDCG@K (only K=100 is logged)
                        if compute_per_slide_ndcg and K == 100:
                            slides_chunk = slide_idx_cpu[i_start:i_end]
                            ndcg_vals = ndcg_block.numpy()
                            np.add.at(ndcg_i2st_slide_sum[K], slides_chunk, ndcg_vals)

                # Per-slide nDCG: count queries per slide once (same for both directions)
                if compute_per_slide_ndcg:
                    slides_chunk = slide_idx_cpu[i_start:i_end]
                    ndcg_slide_count += np.bincount(slides_chunk, minlength=num_slides)

                # Hubness@k (columns frequency)
                cols = topk_all[:, :hub_k_eff].reshape(-1)
                hub_counts_i2st += torch.bincount(cols, minlength=n_working)
                
                # ECE bins: softmax on the logits
                probs = torch.softmax(logits_row, dim=1)
                conf, pred = probs.max(dim=1)
                corr = (pred == labels_row).to(torch.float32)
                bins = torch.clamp((conf * ece_bins).long(), min=0, max=ece_bins - 1)
                ece_cnt_i2st += torch.bincount(bins, minlength=ece_bins).to(torch.float64)
                ece_cor_i2st += torch.bincount(bins, weights=corr, minlength=ece_bins).to(torch.float64)
                ece_conf_i2st += torch.bincount(bins, weights=conf, minlength=ece_bins).to(torch.float64)

                # kNN-agreement using cache paired indices (use top-k); image-anchored only, never st-anchored
                if kk_eff > 0:
                    # Exclude the trivial self-match to keep parity with image→image kNN (which is self-masked).
                    logits_row_mask = logits_row.clone()
                    logits_row_mask[idx_local, labels_row] = -1e9
                    knn_i2st_all[i_start:i_end]  = logits_row_mask.topk(kk_eff, 1).indices
                    knn_imgimg_all[i_start:i_end] = best_idx  # image-image top-k streamed as best_idx (already self-masked)

                # Free GPU mem immediately
                del img_chunk, logits_row, row_logits_parts
                if maxK_ndcg > 0:
                    del best_gain, sel, g_pred, best01

            # st -> image  ----------------------------------------
            for i_start in range(0, n_working, chunk_size):

                # Chunk st rows
                i_end = min(i_start + chunk_size, n_working)
                st_chunk = sts_cpu[i_start:i_end]
                if device.type == "cuda":
                    st_chunk = st_chunk.to(device)

                # Paired img rows (for gains in IDCG to ultimately compute nDCG)
                img_rows = imgs_cpu[i_start:i_end]
                if device.type == "cuda":
                    img_rows = img_rows.to(device)

                # Build logit for this st-row-chunk across all img embeddings, in col img chunks
                row_logits_parts2 = []
                for j_start in range(0, n_working, chunk_size):
                    j_end = min(j_start + chunk_size, n_working)
                    img_chunk = imgs_cpu[j_start:j_end]
                    if device.type == "cuda":
                        img_chunk = img_chunk.to(device)

                    # Partial logits
                    part = logit_scale * st_chunk @ img_chunk.t()
                    row_logits_parts2.append(part.cpu())

                    # Free GPU mem immediately
                    del img_chunk, part

                # Concat to full [i_end-i_start, n]
                logits_row2 = torch.cat(row_logits_parts2, dim=1)
                idx_local = torch.arange(logits_row2.size(0), device=logits_row2.device)
                labels_row2 = torch.arange(i_start, i_end, device=logits_row2.device)

                # Get top-K_all indices for all needed K once
                K_all2 = max(1, max(max_k_acc, hub_k_eff, (maxK_ndcg if maxK_ndcg > 0 else 1)))
                topk_all2 = logits_row2.topk(K_all2, 1).indices

                # Accuracy + MRR (topk_all2 is reused for nDCG / hubness below)
                c1, ck, rr = _retrieval_topk_mrr_counts(
                    logits_row2, labels_row2, topk_all2, idx_local, acc_top_k_list
                )
                correct_st2i1 += c1
                for k in acc_top_k_list:
                    correct_st2i[k] += ck[k]
                rr_sum_st2i += rr

                # Cosine stats -> not done here to avoid double-counting (i->st direction only)

                # nDCG: DCG at predicted indices divided by stored IDCG rows
                if maxK_ndcg > 0:
                    sel = topk_all2[:, :maxK_ndcg].cpu()
                    g_pred = _gather_topk_pair_gains(img_rows, st_chunk, sel)
                    for K in ndcg_top_k_list:
                        K_eff = min(K, maxK_ndcg)
                        if K_eff == 0:
                            continue
                        disc = disc_cache[K][:, :K_eff]
                        dcg_block  = (g_pred[:, :K_eff] * disc).sum(1)
                        idcg_block = idcg_rows[K][i_start:i_end]  # reuse stored IDCG rows
                        ndcg_block = dcg_block / (idcg_block + 1e-8)
                        ndcg_sum_st2i[K] += ndcg_block.sum().item()

                        # Per-slide nDCG@K (only K=100 is logged)
                        if compute_per_slide_ndcg and K == 100:
                            slides_chunk = slide_idx_cpu[i_start:i_end]
                            ndcg_vals = ndcg_block.numpy()
                            np.add.at(ndcg_st2i_slide_sum[K], slides_chunk, ndcg_vals)

                # Hubness@k (now columns are images)
                cols2 = topk_all2[:, :hub_k_eff].reshape(-1)
                hub_counts_st2i += torch.bincount(cols2, minlength=n_working)

                # ECE bins: softmax on the logits
                probs_c = torch.softmax(logits_row2, dim=1)
                conf_c, pred_c = probs_c.max(dim=1)
                corr_c = (pred_c == labels_row2).to(torch.float32)
                bins_c = torch.clamp((conf_c * ece_bins).long(), min=0, max=ece_bins - 1)
                ece_cnt_st2i += torch.bincount(bins_c, minlength=ece_bins).to(torch.float64)
                ece_cor_st2i += torch.bincount(bins_c, weights=corr_c, minlength=ece_bins).to(torch.float64)
                ece_conf_st2i += torch.bincount(bins_c, weights=conf_c, minlength=ece_bins).to(torch.float64)

                # kNN-agreement -> not done here (image-anchored only)

                # Free memory
                del st_chunk, logits_row2, row_logits_parts2
                if maxK_ndcg > 0:
                    del img_rows, sel, g_pred
        

            # Finalize all metrics, add metrics which are not using chunks, and store in output ----------------------------------------

            # Average accuracies and store (keyed; serialized in _metric_keys() order below)
            out = {
                f"{mode}_global_acc_i2st": correct_i2st1 / n_working,
                f"{mode}_global_acc_st2i": correct_st2i1 / n_working,
            }
            for k in acc_top_k_list:
                out[f"{mode}_global_acc_i2st_top{k}"] = correct_i2st[k] / n_working
                out[f"{mode}_global_acc_st2i_top{k}"] = correct_st2i[k] / n_working

            # If per-slide nDCG@100 was computed, average and store
            if compute_per_slide_ndcg:
                # K=100 is the aggregated and exposed one; it is assumed to be in ndcg_top_k_list
                K_slide = 100
                if K_slide not in ndcg_i2st_slide_sum:
                    raise ValueError(f"Per-slide nDCG requested for K={K_slide} but K not in ndcg_top_k_list={ndcg_top_k_list}")
                i2st_sum = ndcg_i2st_slide_sum[K_slide]
                st2i_sum = ndcg_st2i_slide_sum[K_slide]
                with np.errstate(divide='ignore', invalid='ignore'):
                    ndcg_i2st_slide = np.zeros(num_slides, dtype=np.float64)
                    np.divide(i2st_sum, ndcg_slide_count, out=ndcg_i2st_slide, where=ndcg_slide_count > 0)
                    ndcg_st2i_slide = np.zeros(num_slides, dtype=np.float64)
                    np.divide(st2i_sum, ndcg_slide_count, out=ndcg_st2i_slide, where=ndcg_slide_count > 0)
                # Per-slide values keyed by slide_id (slide index = position in the index-sorted id list)
                sorted_ids = [sid for sid, _ in sorted(slide_id2idx.items(), key=lambda kv: kv[1])]
                for sidx, sid in enumerate(sorted_ids):
                    out[f"{mode}_{sid}_ndcg_i2st_top100"] = float(ndcg_i2st_slide[sidx])
                    out[f"{mode}_{sid}_ndcg_st2i_top100"] = float(ndcg_st2i_slide[sidx])
            
            # Average MRR and store
            out[f"{mode}_global_mrr_i2st"] = rr_sum_i2st / n_working
            out[f"{mode}_global_mrr_st2i"] = rr_sum_st2i / n_working

            # Average cosine similarities and store
            pos_sim = pos_sim_sum / n_working
            neg_sim = (neg_sim_sum / n_working) if n_working > 1 else 0.0
            gap = pos_sim - neg_sim
            out[f"{mode}_global_pos_sim"] = pos_sim
            out[f"{mode}_global_neg_sim"] = neg_sim
            out[f"{mode}_global_pos_neg_gap"] = gap

            # Average graded nDCG@K and store
            ndcg_i2st_means = {K: (ndcg_sum_i2st[K] / n_working) if min(K, maxK_ndcg) > 0 else 0.0 for K in ndcg_top_k_list}
            ndcg_st2i_means = {K: (ndcg_sum_st2i[K] / n_working) if min(K, maxK_ndcg) > 0 else 0.0 for K in ndcg_top_k_list}
            for k in ndcg_top_k_list:
                out[f"{mode}_global_ndcg_i2st_top{k}"] = ndcg_i2st_means[k]
                out[f"{mode}_global_ndcg_st2i_top{k}"] = ndcg_st2i_means[k]

            # Compute Hubness@10 (Gini + Top1% mass) and store
            def _gini(x: torch.Tensor) -> float:
                x = x.to(torch.float64)
                S = x.sum().item()
                nG = x.numel()
                if S <= 0 or nG <= 1:
                    return 0.0
                xs, _ = torch.sort(x)
                idx = torch.arange(1, nG + 1, dtype=torch.float64)
                g = (2.0 * (idx * xs).sum() / (nG * S)) - ((nG + 1.0) / nG)
                return float(g.item())

            def _top1p_mass(x: torch.Tensor) -> float:
                x = x.to(torch.float64)
                S = x.sum().item()
                if S <= 0:
                    return 0.0
                nG = x.numel()
                m = max(1, int(math.ceil(0.01 * nG)))
                top = torch.topk(x, k=m).values.sum().item()
                return float(top / S)

            g_i2s = _gini(hub_counts_i2st)
            g_s2i = _gini(hub_counts_st2i)
            m_i2s = _top1p_mass(hub_counts_i2st)
            m_s2i = _top1p_mass(hub_counts_st2i)
            out[f"{mode}_global_hub_gini_i2st@10"] = g_i2s
            out[f"{mode}_global_hub_top1p_i2st@10"] = m_i2s
            out[f"{mode}_global_hub_gini_st2i@10"] = g_s2i
            out[f"{mode}_global_hub_top1p_st2i@10"] = m_s2i

            # Compute ECE (with ece_bins number of bins) (=calibration) and store
            N = float(n_working)
            def _ece(cnt, cor, csum) -> float:
                cnt = cnt.double()
                cor = cor.double()
                csum = csum.double()
                mask = cnt > 0
                acc_b = torch.zeros_like(cnt)
                acc_b[mask] = (cor[mask] / cnt[mask])
                conf_b = torch.zeros_like(cnt)
                conf_b[mask] = (csum[mask] / cnt[mask])
                e = (cnt / max(N, 1.0)) * (acc_b - conf_b).abs()
                return float(e.sum().item())
            
            ece_i2s = _ece(ece_cnt_i2st, ece_cor_i2st, ece_conf_i2st)
            ece_s2i = _ece(ece_cnt_st2i, ece_cor_st2i, ece_conf_st2i)
            out[f"{mode}_global_ece_i2st_bins15"] = ece_i2s
            out[f"{mode}_global_ece_st2i_bins15"] = ece_s2i

            # Compute kNN-agreement@10 (only image-anchored) and store
            # Mean Overlap@k and Jaccard@k between image–image vs paired image indices from i->st top-k
            if kk_eff > 0:
                kk = kk_eff
                knn_i = knn_imgimg_all.numpy()   # [n, k], int
                knn_s = knn_i2st_all.numpy()     # [n, k], int
                # Vectorized rowwise intersection counts, processed in chunks to bound memory
                max_bytes = 64 * 1024 * 1024  # ~64MB workspace cap for the (chunk,k,k) boolean tensor
                rows_per_chunk = max(1, max_bytes // max(1, (kk * kk)))  # guard division
                inter_counts = np.empty(n_working, dtype=np.float64)
                for i0 in range(0, n_working, rows_per_chunk):
                    i1 = min(i0 + rows_per_chunk, n_working)
                    M = (knn_i[i0:i1, :, None] == knn_s[i0:i1, None, :])   # shape: (R, k, k)
                    inter_counts[i0:i1] = M.any(axis=2).sum(axis=1)        # |A_r ∩ B_r| per row
                ov = inter_counts / float(kk)
                jc = inter_counts / (float(2 * kk) - inter_counts + 1e-12)
                out[f"{mode}_global_knn_overlap@10_i_anchor"] = float(ov.mean())
                out[f"{mode}_global_knn_jaccard@10_i_anchor"] = float(jc.mean())
            else:
                # No valid neighbors -> define overlap/jaccard as 0.0
                out[f"{mode}_global_knn_overlap@10_i_anchor"] = 0.0
                out[f"{mode}_global_knn_jaccard@10_i_anchor"] = 0.0

            # Add CKA (optional) (no chunking needed)
            #  - cka_he:  linear CKA between pre-CLIP H&E feats and CLIP-space H&E embeds
            #  - cka_st:  linear CKA between pre-CLIP ST  feats and CLIP-space ST  embeds
            #  - cka_he_st_pre: linear CKA between pre-CLIP H&E feats and pre-CLIP ST feats
            if has_cka:
                if n_working >= 2:
                    cka_he = linear_cka(img_feats_cpu, imgs_cpu)
                    cka_st = linear_cka(st_feats_cpu,  sts_cpu)
                    cka_he_st_pre = linear_cka(img_feats_cpu, st_feats_cpu)
                else:
                    cka_he = cka_st = cka_he_st_pre = 0.0
                out[f"{mode}_global_cka_he"] = cka_he
                out[f"{mode}_global_cka_st"] = cka_st
                out[f"{mode}_global_cka_he_st_pre"] = cka_he_st_pre

            # cka_he_scconcept_pre: linear CKA between pre-CLIP H&E feats and patch-pooled scConcept feats
            if has_scconcept_cka:
                if n_working >= 2:
                    cka_he_scconcept_pre = linear_cka(img_feats_cpu, scconcept_feats_cpu)
                else:
                    cka_he_scconcept_pre = 0.0
                out[f"{mode}_global_cka_he_scconcept_pre"] = cka_he_scconcept_pre

            # Add Alignment & Spectrum (no chunking needed)
            # Compute on CPU tensors for the current (possibly subset) selection
            # Alignment: paired L2-mean on normalized embeddings
            align_mse_val = alignment_mse(imgs_cpu, sts_cpu)
            # Spectrum stats: eff. rank & participation ratio for each set and pooled
            eff_i, pr_i = spectrum_stats(imgs_cpu)
            eff_s, pr_s = spectrum_stats(sts_cpu)
            eff_b, pr_b = spectrum_stats(torch.cat([imgs_cpu, sts_cpu], dim=0))
            out[f"{mode}_global_align_mse"] = align_mse_val
            out[f"{mode}_global_effrank_img"] = eff_i
            out[f"{mode}_global_effrank_st"] = eff_s
            out[f"{mode}_global_effrank_both"] = eff_b
            out[f"{mode}_global_partratio_img"] = pr_i
            out[f"{mode}_global_partratio_st"] = pr_s
            out[f"{mode}_global_partratio_both"] = pr_b

            # Finalize uniformity & RBF MMD and store
            if n_working >= 2:
                n = float(n_working)
                # remove diagonal contributions (each diag term = exp(2t) exactly)
                diag_e2t = math.exp(2.0 * uni_t)
                sum_imgimg_no_diag = max(0.0, sum_exp_imgimg - diag_e2t * n_working)
                sum_stst_no_diag   = max(0.0, sum_exp_stst  - diag_e2t * n_working)
                # "both" is on the pooled set of size 2n. Ordered pairs include both cross directions.
                N = float(2 * n_working)
                cross_ordered = 2.0 * sum_exp_imgst  # no diagonals in cross-modal
                uni_img = math.log(exp_minus_2t * (sum_imgimg_no_diag / (n * (n - 1.0) + 1e-12)) + 1e-30)
                uni_st  = math.log(exp_minus_2t * (sum_stst_no_diag   / (n * (n - 1.0) + 1e-12)) + 1e-30)
                uni_both = math.log(
                    exp_minus_2t * ((sum_imgimg_no_diag + sum_stst_no_diag + cross_ordered) / (N * (N - 1.0) + 1e-12))
                    + 1e-30
                )
                # exact unbiased MMD^2 with pooled-bandwidth gamma
                kxx = (max(0.0, sum_k_imgimg - n_working)) / (n * (n - 1.0) + 1e-12)
                kyy = (max(0.0, sum_k_stst  - n_working)) / (n * (n - 1.0) + 1e-12)
                kxy =  sum_k_imgst / (n * n + 1e-12)   # one direction
                mmd2_exact = kxx + kyy - 2.0 * kxy
            else:
                uni_img = uni_st = uni_both = 0.0
                mmd2_exact = 0.0
            out[f"{mode}_global_uni_img"] = uni_img
            out[f"{mode}_global_uni_st"] = uni_st
            out[f"{mode}_global_uni_both"] = uni_both
            out[f"{mode}_global_mmd_rbf"] = mmd2_exact

    #############################################################
    # #### ON NON-RANK 0: nothing computed locally            ####
    #############################################################
    else:
        # Non-zero ranks compute no payload; the broadcast below provides every value.
        out = None


    #######################################
    # #### BROADCAST AND STORE RESULTS ####
    #######################################

    # Serialize the rank-0 results in _metric_keys() order (other ranks send zeros of the same length)
    if rank == 0:
        key_set = set(metric_keys)
        missing = [k for k in metric_keys if k not in out]
        extra = [k for k in out if k not in key_set]
        assert not missing and not extra, (
            f"global_metric payload key mismatch (rank-0 packing vs schema): "
            f"missing={missing}, extra={extra}"
        )
        out_list = [float(out[k]) for k in metric_keys]
    else:
        out_list = [0.0] * len(metric_keys)

    # Send scalar metrics to every rank so code stays symmetric
    values = broadcast_scalars_from_rank0(out_list, device)
    assert len(values) == len(metric_keys), (
        f"broadcast length {len(values)} != schema length {len(metric_keys)}"
    )

    # Re-attach the metric names (same order on every rank)
    results = dict(zip(metric_keys, values))

    return results


# -------------------------------------------------------------------------------------------------------------- 2bis. Within-slide retrieval (batch-controlled)
# => Restricts the retrieval gallery to each query's own slide, then macro-averages over slides.
def within_slide_metrics_computation(
    all_img_embeds: torch.Tensor,
    all_st_embeds: torch.Tensor,
    logit_scale: torch.Tensor,
    mode: str,
    acc_top_k_list: List[int],
    min_patches: int,
    chunk_size: int,
    device: torch.device,
    slide_idx: torch.Tensor,
    slide_id2idx: Dict[str, int],
) -> Dict[str, float]:
    """
    Within-slide retrieval: for each validation slide, i2st and st2i top-k accuracy + MRR with the gallery
    restricted to that slide's patches, then macro-averaged over slides.
    Slides with n_s < max(min_patches, max(acc_top_k_list) + 1) are skipped (NaN per-slide values).
    Rank-0 keeps the full matrices; results are broadcast to all ranks.
    """
    # Gather partial tensors -> rank-0 (CPU)
    imgs_cpu = gather_to_rank0(all_img_embeds)
    sts_cpu = gather_to_rank0(all_st_embeds)
    slide_idx = gather_to_rank0(slide_idx)

    num_slides = len(slide_id2idx)
    _idx_vals = np.array(list(slide_id2idx.values()), dtype=np.int64)
    assert _idx_vals.min() == 0 and _idx_vals.max() == num_slides - 1 and np.unique(_idx_vals).size == num_slides, \
        "slide_id2idx must map to contiguous indices 0..num_slides-1"

    # Slide ids in index order (matches the ordering used by global_metrics_computation).
    sorted_ids = [sid for sid, _ in sorted(slide_id2idx.items(), key=lambda kv: kv[1])]
    # All accuracy ks: top-1 (always, via argmax) + the requested ks.
    all_acc_ks = [1] + list(acc_top_k_list)
    # Per-slide values: top-1, top-5
    per_slide_acc_ks = [k for k in (1, 5) if (k == 1 or k in acc_top_k_list)]

    # Ordered list of the metric keys (order and length of the broadcast payload, same on every rank)
    def _within_keys() -> List[str]:
        keys: List[str] = []
        for k in all_acc_ks:
            keys.append(f"{mode}_within_slide_acc_i2st_top{k}")
            keys.append(f"{mode}_within_slide_acc_st2i_top{k}")
        keys.append(f"{mode}_within_slide_mrr_i2st")
        keys.append(f"{mode}_within_slide_mrr_st2i")
        keys.append(f"{mode}_within_slide_n_slides_used")
        keys.append(f"{mode}_within_slide_n_slides_skipped")
        for sid in sorted_ids:
            for k in per_slide_acc_ks:
                keys.append(f"{mode}_within_slide_{sid}_acc_i2st_top{k}")
                keys.append(f"{mode}_within_slide_{sid}_acc_st2i_top{k}")
            keys.append(f"{mode}_within_slide_{sid}_mrr_i2st")
            keys.append(f"{mode}_within_slide_{sid}_mrr_st2i")
        return keys

    metric_keys = _within_keys()

    # Move logit_scale to GPU
    if device.type == "cuda":
        logit_scale = logit_scale.to(device)

    rank = get_dist_rank() if (dist.is_available() and dist.is_initialized()) else 0

    #######################################
    # #### ON RANK 0: full computation ####
    #######################################
    if rank == 0:
        slide_idx_cpu = slide_idx.to(torch.long).cpu().numpy()
        topk_K = max(all_acc_ks)                                  # >= max(acc_top_k_list)
        effective_min = max(int(min_patches), max(acc_top_k_list) + 1)
        nan = float("nan")

        out: Dict[str, float] = {}
        # Macro accumulators over included slides only.
        macro_acc_i2st = {k: 0.0 for k in all_acc_ks}
        macro_acc_st2i = {k: 0.0 for k in all_acc_ks}
        macro_mrr_i2st = 0.0
        macro_mrr_st2i = 0.0
        n_used = 0
        n_skipped = 0

        for sidx, sid in enumerate(sorted_ids):
            rows = np.nonzero(slide_idx_cpu == sidx)[0]
            n_s = int(rows.size)

            if n_s < effective_min:
                # Skipped slide: NaN per-slide values, excluded from the macro average.
                n_skipped += 1
                for k in per_slide_acc_ks:
                    out[f"{mode}_within_slide_{sid}_acc_i2st_top{k}"] = nan
                    out[f"{mode}_within_slide_{sid}_acc_st2i_top{k}"] = nan
                out[f"{mode}_within_slide_{sid}_mrr_i2st"] = nan
                out[f"{mode}_within_slide_{sid}_mrr_st2i"] = nan
                continue

            sel_t = torch.from_numpy(rows).to(torch.long)
            img_s = imgs_cpu.index_select(0, sel_t)   # [n_s, D]
            st_s = sts_cpu.index_select(0, sel_t)     # [n_s, D]
            if device.type == "cuda":
                img_s = img_s.to(device)
                st_s = st_s.to(device)

            # i2st: query = img rows, gallery = st_s ; st2i: query = st rows, gallery = img_s.
            cnt_i = {k: 0 for k in all_acc_ks}; rr_i = 0.0
            cnt_s = {k: 0 for k in all_acc_ks}; rr_s = 0.0
            for q_start in range(0, n_s, chunk_size):
                q_end = min(q_start + chunk_size, n_s)
                labels = torch.arange(q_start, q_end)            # CPU (gallery row positions)
                idx_local = torch.arange(q_end - q_start)        # CPU

                # GPU matmul -> CPU reduce (as in global_metrics_computation)
                logits_i = (logit_scale * (img_s[q_start:q_end] @ st_s.t())).detach().cpu()
                topk_i = logits_i.topk(topk_K, 1).indices
                a1, ak, rra = _retrieval_topk_mrr_counts(logits_i, labels, topk_i, idx_local, acc_top_k_list)
                cnt_i[1] += a1
                for k in acc_top_k_list:
                    cnt_i[k] += ak[k]
                rr_i += rra
                del logits_i, topk_i

                logits_s = (logit_scale * (st_s[q_start:q_end] @ img_s.t())).detach().cpu()
                topk_s = logits_s.topk(topk_K, 1).indices
                b1, bk, rrb = _retrieval_topk_mrr_counts(logits_s, labels, topk_s, idx_local, acc_top_k_list)
                cnt_s[1] += b1
                for k in acc_top_k_list:
                    cnt_s[k] += bk[k]
                rr_s += rrb
                del logits_s, topk_s

            acc_i = {k: cnt_i[k] / n_s for k in all_acc_ks}
            acc_s = {k: cnt_s[k] / n_s for k in all_acc_ks}
            mrr_i = rr_i / n_s
            mrr_s = rr_s / n_s

            # Per-slide exposed leaves.
            for k in per_slide_acc_ks:
                out[f"{mode}_within_slide_{sid}_acc_i2st_top{k}"] = float(acc_i[k])
                out[f"{mode}_within_slide_{sid}_acc_st2i_top{k}"] = float(acc_s[k])
            out[f"{mode}_within_slide_{sid}_mrr_i2st"] = float(mrr_i)
            out[f"{mode}_within_slide_{sid}_mrr_st2i"] = float(mrr_s)

            # Macro accumulation (equal weight per slide).
            for k in all_acc_ks:
                macro_acc_i2st[k] += acc_i[k]
                macro_acc_st2i[k] += acc_s[k]
            macro_mrr_i2st += mrr_i
            macro_mrr_st2i += mrr_s
            n_used += 1

        # Macro means (NaN if no slide met the threshold this epoch).
        for k in all_acc_ks:
            out[f"{mode}_within_slide_acc_i2st_top{k}"] = float(macro_acc_i2st[k] / n_used) if n_used > 0 else nan
            out[f"{mode}_within_slide_acc_st2i_top{k}"] = float(macro_acc_st2i[k] / n_used) if n_used > 0 else nan
        out[f"{mode}_within_slide_mrr_i2st"] = float(macro_mrr_i2st / n_used) if n_used > 0 else nan
        out[f"{mode}_within_slide_mrr_st2i"] = float(macro_mrr_st2i / n_used) if n_used > 0 else nan
        out[f"{mode}_within_slide_n_slides_used"] = float(n_used)
        out[f"{mode}_within_slide_n_slides_skipped"] = float(n_skipped)
    else:
        out = None

    #######################################
    # #### BROADCAST AND STORE RESULTS ####
    #######################################
    if rank == 0:
        key_set = set(metric_keys)
        missing = [k for k in metric_keys if k not in out]
        extra = [k for k in out if k not in key_set]
        assert not missing and not extra, (
            f"within_slide payload key mismatch (rank-0 packing vs schema): "
            f"missing={missing}, extra={extra}"
        )
        out_list = [float(out[k]) for k in metric_keys]
    else:
        out_list = [0.0] * len(metric_keys)

    values = broadcast_scalars_from_rank0(out_list, device)
    assert len(values) == len(metric_keys), (
        f"broadcast length {len(values)} != schema length {len(metric_keys)}"
    )
    results = dict(zip(metric_keys, values))

    return results



def linear_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    """
    Kornblith et al. (2019) linear CKA, CPU implementation.
    X : (n, d_x), Y : (n, d_y)  – already on rank-0 CPU.
    """
    X = X.float()
    Y = Y.float()
    Xc = X - X.mean(0, keepdim=True)
    Yc = Y - Y.mean(0, keepdim=True)
    # || XᵀY ||_F²
    dot = torch.linalg.matrix_norm(Xc.T @ Yc, ord="fro") ** 2
    # || XᵀX ||_F · || YᵀY ||_F
    norm_x = torch.linalg.matrix_norm(Xc.T @ Xc, ord="fro")
    norm_y = torch.linalg.matrix_norm(Yc.T @ Yc, ord="fro")
    # || XᵀY ||_F² / (|| XᵀX ||_F · || YᵀY ||_F)
    return (dot / (norm_x * norm_y + 1e-8)).item()




def alignment_mse(img: torch.Tensor, st: torch.Tensor) -> float:
    """
    Paired closeness on L2-normalized embeddings:
    mean ||z_img - z_st||^2 = 2 * mean(1 - cos).
    """
    img = img.float()
    st = st.float()
    # assume already normalized; still safe even if slightly off
    cos = (img * st).sum(dim=1).clamp(-1, 1)
    return (2.0 * (1.0 - cos)).mean().item()


def spectrum_stats(z: torch.Tensor) -> Tuple[float, float]:
    """
    Effective rank & Participation ratio from spectrum of centered features.
        eff_rank = exp(H(p)),  p = λ / sum λ
        PR       = (sum λ)^2 / sum λ^2
    """
    if z.size(0) <= 1:
        return 0.0, 0.0
    Z = z.float()
    Zc = Z - Z.mean(dim=0, keepdim=True)
    # covariance in feature space (D x D)
    C = (Zc.T @ Zc) / max(1, (Zc.size(0) - 1))
    # sym PSD -> use eigvalsh for stability
    lam = torch.linalg.eigvalsh(C.to(torch.float64)).clamp(min=0).to(torch.float64)
    s = lam.sum()
    if s <= 0:
        return 0.0, 0.0
    p = (lam / s).clamp(min=1e-12)
    H = -(p * p.log()).sum().item()
    eff_rank = math.exp(H)
    pr = float((s * s) / (lam.pow(2).sum() + 1e-12))
    return float(eff_rank), float(pr)