from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple
import numpy as np

from src_A2.common import softmax_numpy


@dataclass
class RetrievalResult:
    indices: np.ndarray  # [Nq, K]
    similarities: np.ndarray  # [Nq, K]



def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return x / norms



def _sort_topk_row(sim_row: np.ndarray, idx_row: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    order = np.lexsort((idx_row, -sim_row))
    return sim_row[order], idx_row[order]



def _topk_block_deterministic(
    sim: np.ndarray,
    global_start_idx: int,
    K: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Deterministic top-K inside one reference block with tie-break:
      larger cosine first, then smaller global reference row index.
    Returns:
      block_idx_topk: [B, K] int32
      block_sim_topk: [B, K] float32
    """
    B, R = sim.shape
    K = int(min(K, R))
    global_idx = np.arange(global_start_idx, global_start_idx + R, dtype=np.int32)

    if R == K:
        idx_full = np.broadcast_to(global_idx[None, :], (B, R))
        order = np.lexsort((idx_full, -sim), axis=1)
        return (
            np.take_along_axis(idx_full, order[:, :K], axis=1).astype(np.int32, copy=False),
            np.take_along_axis(sim, order[:, :K], axis=1).astype(np.float32, copy=False),
        )

    # Top-(K+1) by similarity: only rows tied at the K-th similarity need the correction below
    select = int(min(K + 1, R))
    kth = R - select
    part = np.argpartition(sim, kth=kth, axis=1)[:, kth:]
    part_sim = np.take_along_axis(sim, part, axis=1)
    part_idx = (global_start_idx + part).astype(np.int32, copy=False)

    order = np.lexsort((part_idx, -part_sim), axis=1)
    sorted_idx = np.take_along_axis(part_idx, order, axis=1).astype(np.int32, copy=False)
    sorted_sim = np.take_along_axis(part_sim, order, axis=1).astype(np.float32, copy=False)

    out_idx = sorted_idx[:, :K].astype(np.int32, copy=False)
    out_sim = sorted_sim[:, :K].astype(np.float32, copy=False)

    # Exact boundary-tie correction: among tied candidates, keep the smallest global reference indices
    if select > K:
        fix_rows = np.flatnonzero(sorted_sim[:, K - 1] == sorted_sim[:, K])
    else:
        fix_rows = np.empty((0,), dtype=np.int64)

    for i in fix_rows.tolist():
        row = sim[i]
        threshold = sorted_sim[i, K - 1]
        higher_local = np.flatnonzero(row > threshold)
        equal_local = np.flatnonzero(row == threshold)
        need_equal = K - higher_local.size
        selected_local = np.concatenate([higher_local, equal_local[:need_equal]], axis=0)

        selected_idx = (global_start_idx + selected_local).astype(np.int32, copy=False)
        selected_sim = row[selected_local].astype(np.float32, copy=False)
        row_order = np.lexsort((selected_idx, -selected_sim))
        out_idx[i] = selected_idx[row_order]
        out_sim[i] = selected_sim[row_order]

    return out_idx, out_sim




def _merge_topk_deterministic(
    best_idx: Optional[np.ndarray],
    best_sim: Optional[np.ndarray],
    new_idx: np.ndarray,
    new_sim: np.ndarray,
    K: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Deterministically merge two already top-K-sorted candidate sets.
    """
    if best_idx is None or best_sim is None:
        return new_idx, new_sim

    idx = np.concatenate([best_idx.astype(np.int32, copy=False), new_idx.astype(np.int32, copy=False)], axis=1)
    sim = np.concatenate([best_sim.astype(np.float32, copy=False), new_sim.astype(np.float32, copy=False)], axis=1)
    K = int(min(K, idx.shape[1]))

    # Full lexsort over the 2K merged candidates
    order = np.lexsort((idx, -sim), axis=1)[:, :K]
    return (
        np.take_along_axis(idx, order, axis=1).astype(np.int32, copy=False),
        np.take_along_axis(sim, order, axis=1).astype(np.float32, copy=False),
    )



def exact_topk_cosine(
    queries: np.ndarray,
    references: np.ndarray,
    K: int,
    query_chunk_size: int = 512,
    reference_chunk_size: int = 16384,
    queries_normalized: bool = False,
    references_normalized: bool = False,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> RetrievalResult:
    q = queries if queries_normalized else l2_normalize(queries)
    r = references if references_normalized else l2_normalize(references)
    q = np.ascontiguousarray(q, dtype=np.float32)
    r = np.ascontiguousarray(r, dtype=np.float32)

    nq = q.shape[0]
    nr = r.shape[0]
    K = int(min(K, nr))
    if K <= 0:
        raise ValueError("K must be > 0 and reference matrix must be non-empty")
    sims_out = np.empty((nq, K), dtype=np.float32)
    idx_out = np.empty((nq, K), dtype=np.int64)

    for start in range(0, nq, query_chunk_size):
        stop = min(start + query_chunk_size, nq)
        q_chunk = q[start:stop]
        bsz = q_chunk.shape[0]
        best_sim = None
        best_idx = None

        for rstart in range(0, nr, reference_chunk_size):
            rstop = min(rstart + reference_chunk_size, nr)
            sim = q_chunk @ r[rstart:rstop].T  # [B, Rchunk]
            block_idx, block_sim = _topk_block_deterministic(sim=sim, global_start_idx=rstart, K=K)
            best_idx, best_sim = _merge_topk_deterministic(
                best_idx=best_idx,
                best_sim=best_sim,
                new_idx=block_idx,
                new_sim=block_sim,
                K=K,
            )

        sims_out[start:stop] = best_sim.astype(np.float32, copy=False)
        idx_out[start:stop] = best_idx.astype(np.int64, copy=False)
        if progress_callback is not None:
            progress_callback(stop, nq)

    return RetrievalResult(indices=idx_out, similarities=sims_out)



def faiss_topk_cosine(
    queries: np.ndarray,
    references: np.ndarray,
    K: int,
    queries_normalized: bool = False,
    references_normalized: bool = False,
) -> RetrievalResult:
    import faiss

    q = queries if queries_normalized else l2_normalize(queries)
    r = references if references_normalized else l2_normalize(references)
    q = np.ascontiguousarray(q, dtype=np.float32)
    r = np.ascontiguousarray(r, dtype=np.float32)
    index = faiss.IndexFlatIP(r.shape[1])
    index.add(r)
    sims, idx = index.search(q, int(min(K, r.shape[0])))
    if idx.shape[1] > 1:
        order = np.lexsort((idx, -sims), axis=1)
        sims = np.take_along_axis(sims, order, axis=1)
        idx = np.take_along_axis(idx, order, axis=1)

    return RetrievalResult(indices=idx.astype(np.int64, copy=False), similarities=sims.astype(np.float32, copy=False))



def retrieve_topk(
    queries: np.ndarray,
    references: np.ndarray,
    K: int,
    backend: str = "auto",
    query_chunk_size: int = 2048,
    reference_chunk_size: int = 16384,
    queries_normalized: bool = False,
    references_normalized: bool = False,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> RetrievalResult:
    if queries.shape[0] == 0:
        return RetrievalResult(
            indices=np.zeros((0, int(K)), dtype=np.int64),
            similarities=np.zeros((0, int(K)), dtype=np.float32),
        )
    if references.shape[0] == 0:
        raise ValueError("Reference embedding matrix is empty")
    if backend == "auto":
        # FAISS IndexFlatIP (exact inner-product search) if installed, else NumPy exact search
        try:
            import faiss  # noqa: F401
            backend = "faiss"
        except Exception:
            backend = "exact"

    if backend == "faiss":
        return faiss_topk_cosine(
            queries,
            references,
            K=K,
            queries_normalized=queries_normalized,
            references_normalized=references_normalized,
        )
    if backend != "exact":
        raise ValueError(f"Unsupported retrieval backend '{backend}'. Use one of {'auto','exact','faiss'}.")
    return exact_topk_cosine(
        queries,
        references,
        K=K,
        query_chunk_size=query_chunk_size,
        reference_chunk_size=reference_chunk_size,
        queries_normalized=queries_normalized,
        references_normalized=references_normalized,
        progress_callback=progress_callback,
    )



def weights_from_similarities(similarities: np.ndarray, tau: float) -> np.ndarray:
    return softmax_numpy(np.asarray(similarities, dtype=np.float32) / float(tau), axis=1)



def mapped_class_mass(
    neighbor_indices: np.ndarray,
    weights: np.ndarray,
    ref_label_ids: np.ndarray,
    gt_label_ids: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    neigh_label_ids = ref_label_ids[neighbor_indices]
    gt_expand = gt_label_ids[:, None]
    mask = neigh_label_ids == gt_expand
    mass = np.sum(weights * mask.astype(np.float32), axis=1)
    present = np.any(mask, axis=1)
    return mass.astype(np.float32), present



def _jaccard_chunk(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Vectorized set Jaccard for a chunk of paired neighbor rows."""
    m, k = a.shape
    if m == 0 or k == 0:
        return np.zeros(m, dtype=np.float32)

    a_sorted = np.sort(a, axis=1)
    a_first = np.empty_like(a_sorted, dtype=bool)
    a_first[:, 0] = True
    if k > 1:
        a_first[:, 1:] = a_sorted[:, 1:] != a_sorted[:, :-1]
    n_a_unique = a_first.sum(axis=1)

    b_sort_idx = np.argsort(b, axis=1, kind="stable")
    b_sorted = np.take_along_axis(b, b_sort_idx, axis=1)
    b_first = np.empty_like(b_sorted, dtype=bool)
    b_first[:, 0] = True
    if k > 1:
        b_first[:, 1:] = b_sorted[:, 1:] != b_sorted[:, :-1]
    n_b_unique = b_first.sum(axis=1)

    # in_a_for_b[i, j] = (b[i, j] appears anywhere in a[i, :])
    in_a_for_b = (b[:, :, None] == a[:, None, :]).any(axis=2)
    in_a_for_b_sorted = np.take_along_axis(in_a_for_b, b_sort_idx, axis=1)
    intersect = (in_a_for_b_sorted & b_first).sum(axis=1)

    union = n_a_unique + n_b_unique - intersect
    out = np.zeros(m, dtype=np.float32)
    nz = union > 0
    out[nz] = intersect[nz].astype(np.float32) / union[nz].astype(np.float32)
    return out



def jaccard_at_k(a_idx: np.ndarray, b_idx: np.ndarray, k: int, chunk_size: int = 2048) -> np.ndarray:
    a = np.asarray(a_idx)[:, :k]
    b = np.asarray(b_idx)[:, :k]
    n = a.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    if a.shape != b.shape:
        raise ValueError(f"a_idx and b_idx shapes differ after k slice: {a.shape} vs {b.shape}")
    chunk_size = max(1, int(chunk_size))

    out = np.empty(n, dtype=np.float32)
    for start in range(0, n, chunk_size):
        stop = min(n, start + chunk_size)
        out[start:stop] = _jaccard_chunk(a[start:stop], b[start:stop])
    return out



def _jaccard_at_k_reference(a_idx: np.ndarray, b_idx: np.ndarray, k: int) -> np.ndarray:
    """Python-set reference implementation of jaccard_at_k (not used by the pipeline)."""
    a = a_idx[:, :k]
    b = b_idx[:, :k]
    out = np.zeros(a.shape[0], dtype=np.float32)
    for i in range(a.shape[0]):
        sa = set(a[i].tolist())
        sb = set(b[i].tolist())
        denom = len(sa | sb)
        out[i] = 0.0 if denom == 0 else float(len(sa & sb) / denom)
    return out
