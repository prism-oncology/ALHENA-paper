"""Samplers for training and validation (batch-aligned, alternating, single-slide, weighted, strided eval)."""

import logging
logger = logging.getLogger(__name__)

from typing import Iterator, List, Optional
import logging
import torch
from torch.utils.data import Sampler
import torch
import math
import itertools
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

from src.dataset.histost_dataset import HistoStDataset



def _normalize_batch_slide_mode(x: str) -> str:
    """
    Normalize user-configured batch slide mode.
    Returns: "mixed" | "single_slide" | "alternate"
    """
    s = str(x).strip().lower()
    if s in ("mixed", "mix"):
        return "mixed"
    if s in ("single", "single_slide", "per_slide"):
        return "single_slide"
    if s in ("alternate", "alternating", "alt"):
        return "alternate"
    raise ValueError(f"Unknown batch_slide_mode='{x}'. Use 'mixed', 'single_slide' or 'alternate'.")




class DistributedBatchAlignedSampler(DistributedSampler):
    """
    Torch DistributedSampler whose per-rank index stream is truncated to a multiple of batch_size
    (same number of steps across epochs and modes; no padding; deterministic shuffling via set_epoch(epoch)).
    """
    def __init__(
        self,
        dataset,
        batch_size: int,
        num_replicas: int,
        rank: int,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = True,
    ) -> None:
        super().__init__(
            dataset,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
            seed=seed,
            drop_last=drop_last,
        )
        self.batch_size = int(batch_size)
        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        # Truncate per-rank samples to a whole number of local batches
        self._aligned_num_samples = (int(self.num_samples) // self.batch_size) * self.batch_size
        if self._aligned_num_samples <= 0:
            raise ValueError(
                f"Aligned num_samples is 0 (num_samples={self.num_samples}, batch_size={self.batch_size})."
            )

    def __iter__(self) -> Iterator[int]:
        # super().__iter__ already uses (seed + epoch) internally
        return iter(itertools.islice(super().__iter__(), self._aligned_num_samples))

    def __len__(self) -> int:
        return int(self._aligned_num_samples)





class DistributedAlternatingSampler(Sampler[int]):
    """
    Epoch-level alternation between sampler_mixed and sampler_single
    (default: even epochs -> mixed, odd epochs -> single_slide; all ranks must call set_epoch(epoch)).
    Requires len(sampler_mixed) == len(sampler_single).
    """
    def __init__(
        self,
        sampler_mixed: Sampler[int],
        sampler_single: Sampler[int],
        start_with: str = "mixed",
    ) -> None:
        self.sampler_mixed = sampler_mixed
        self.sampler_single = sampler_single
        sw = _normalize_batch_slide_mode(start_with)
        if sw not in ("mixed", "single_slide"):
            raise ValueError(f"start_with must be 'mixed' or 'single_slide', got '{start_with}'")
        self.start_with = sw
        self.epoch = 0

        lm = len(self.sampler_mixed)
        ls = len(self.sampler_single)
        if lm != ls:
            raise ValueError(
                "DistributedAlternatingSampler requires identical per-rank sampler lengths to keep "
                f"the same #steps/epoch, got len(mixed)={lm} vs len(single)={ls}."
            )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        for s in (self.sampler_mixed, self.sampler_single):
            if hasattr(s, "set_epoch"):
                s.set_epoch(epoch)

    def mode_for_epoch(self, epoch: int) -> str:
        # even/odd alternation with configurable starting mode
        e = int(epoch)
        if self.start_with == "mixed":
            return "mixed" if (e % 2 == 0) else "single_slide"
        return "single_slide" if (e % 2 == 0) else "mixed"

    @property
    def current_mode(self) -> str:
        return self.mode_for_epoch(self.epoch)

    def __iter__(self) -> Iterator[int]:
        active = self.sampler_mixed if (self.current_mode == "mixed") else self.sampler_single
        return iter(active)

    def __len__(self) -> int:
        return len(self.sampler_mixed)





class DistributedSingleSlideUniformGlobalBatchSampler(Sampler[int]):
    """
    DDP-safe sampler that enforces, for each global step:
      - one single slide_id across all ranks (global batch is slide-pure)
      - no duplicate dataset indices inside the global batch (avoid CLIP false negatives)

    Slide selection:
      - approximately uniform / balanced per epoch: each slide appears (almost) equally often.

    Patch selection within slide:
      - per-slide shuffled list of indices
      - take consecutive chunks of size global_batch_size
      - wrap by reshuffling; ensure uniqueness within the wrapping batch.
    """
    def __init__(
        self,
        dataset: "HistoStDataset",
        batch_size: int,
        num_replicas: int,
        rank: int,
        seed: int = 0,
        drop_last: bool = True,
    ) -> None:
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.epoch = 0

        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        self.global_batch_size = self.batch_size * self.num_replicas

        # Requires dataset precomputed slide grouping
        if not hasattr(dataset, "indices_by_slide_idx"):
            raise ValueError("Dataset must expose indices_by_slide_idx for single-slide batching.")

        self.slide_ids: List[str] = list(getattr(dataset, "slide_id_list"))
        self.slide_indices: List[torch.Tensor] = list(getattr(dataset, "indices_by_slide_idx"))
        if len(self.slide_indices) == 0:
            raise ValueError("No slides found in dataset.")

        # Hard constraint: each slide must be able to provide one full global batch without duplicates.
        bad = []
        for sidx, idxs in enumerate(self.slide_indices):
            n = int(idxs.numel())
            if n < self.global_batch_size:
                bad.append((self.slide_ids[sidx], n))
        if bad:
            msg = ", ".join([f"{sid}: {n}" for sid, n in bad])
            raise ValueError(
                f"Some slides have < global_batch_size patches (global_bs={self.global_batch_size}). "
                f"Cannot enforce global-batch uniqueness. Offending slides: {msg}"
            )

        # Sampler length drives DataLoader length.
        n_total = len(dataset)
        if self.drop_last:
            local_n = n_total // self.num_replicas
        else:
            local_n = int(math.ceil(n_total / self.num_replicas))
        # Align local stream to full local batches (training DataLoader drop_last=True anyway)
        local_n = (local_n // self.batch_size) * self.batch_size
        self.num_samples = local_n
        self.num_global_batches = self.num_samples // self.batch_size
        self.total_size = self.num_global_batches * self.global_batch_size

        # Per-epoch mutable state (created fresh in __iter__)
        self._perms: List[torch.Tensor] = []
        self._ptrs: List[int] = []

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _reshuffle_slide(self, slide_idx: int, g: torch.Generator) -> None:
        base = self.slide_indices[slide_idx]
        n = int(base.numel())
        perm = base[torch.randperm(n, generator=g)]
        self._perms[slide_idx] = perm
        self._ptrs[slide_idx] = 0

    def _take_global_batch_from_slide(self, slide_idx: int, g: torch.Generator) -> torch.Tensor:
        """
        Return a CPU LongTensor of length global_batch_size with unique indices.
        Wrap by reshuffling when necessary; ensure uniqueness inside the wrapping batch.
        """
        perm = self._perms[slide_idx]
        ptr = self._ptrs[slide_idx]
        n = int(perm.numel())
        k = self.global_batch_size

        # Fast path: enough remaining
        if ptr + k <= n:
            out = perm[ptr:ptr + k]
            self._ptrs[slide_idx] = ptr + k
            return out

        # Wrap path: take tail + refill from fresh shuffle, ensuring uniqueness inside the global batch.
        tail = perm[ptr:]  # tail_len < k (since n >= k)
        tail_list = tail.tolist()  # small: < k
        seen = set(tail_list)

        # reshuffle for new cycle
        self._reshuffle_slide(slide_idx, g)
        perm2 = self._perms[slide_idx]

        need = k - len(tail_list)
        refill: List[int] = []
        i = 0
        # deterministically scan perm2 until the batch is filled with new unique indices
        while len(refill) < need:
            cand = int(perm2[i].item())
            i += 1
            if cand not in seen:
                refill.append(cand)
                seen.add(cand)

        # advance pointer in the new perm by the number of candidates consumed (including skipped)
        self._ptrs[slide_idx] = i
        out = torch.tensor(tail_list + refill, dtype=torch.long)
        return out

    def __iter__(self) -> Iterator[int]:
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        # Initialize per-slide permutations each epoch
        self._perms = []
        self._ptrs = []
        for sidx, base in enumerate(self.slide_indices):
            n = int(base.numel())
            self._perms.append(base[torch.randperm(n, generator=g)])
            self._ptrs.append(0)

        num_slides = len(self.slide_indices)

        # Build a near-perfectly balanced slide schedule per epoch
        reps = int(math.ceil(self.num_global_batches / num_slides))
        slide_sched = torch.arange(num_slides, dtype=torch.long).repeat(reps)
        slide_sched = slide_sched[torch.randperm(slide_sched.numel(), generator=g)]
        slide_sched = slide_sched[: self.num_global_batches]

        # Build the global index matrix: [num_batches, global_bs]
        batches = torch.empty((self.num_global_batches, self.global_batch_size), dtype=torch.long)
        for b in range(self.num_global_batches):
            sidx = int(slide_sched[b].item())
            batches[b] = self._take_global_batch_from_slide(sidx, g)

        # Slice this rank's chunk per global batch
        start = self.rank * self.batch_size
        end = start + self.batch_size
        local = batches[:, start:end].reshape(-1)
        return iter(local.tolist())

    def __len__(self) -> int:
        return self.num_samples





class DistributedSingleSlideBatchSampler(Sampler[List[int]]):
    """
    BatchSampler for the validation step metrics: yields per-rank batches (lists of indices)
    such that each global step uses a single slide across all ranks.
    """
    def __init__(
        self,
        dataset: "HistoStDataset",
        batch_size: int,
        num_replicas: int,
        rank: int,
        seed: int = 0,
        drop_last: bool = True,
        shuffle_slides: bool = False,
        shuffle_within_slide: bool = False,
    ) -> None:
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.shuffle_slides = bool(shuffle_slides)
        self.shuffle_within_slide = bool(shuffle_within_slide)
        self.epoch = 0

        if not hasattr(dataset, "indices_by_slide_idx"):
            raise ValueError("Dataset must expose indices_by_slide_idx for single-slide batching.")
        self.slide_ids: List[str] = list(getattr(dataset, "slide_id_list"))
        self.slide_indices: List[torch.Tensor] = list(getattr(dataset, "indices_by_slide_idx"))

        # Precompute per-slide per-rank counts with DDP drop_last semantics (per slide)
        self._per_rank_counts: List[int] = []
        self._num_batches: int = 0
        for idxs in self.slide_indices:
            n = int(idxs.numel())
            if self.drop_last:
                n = (n // self.num_replicas) * self.num_replicas
            else:
                # padding in validation is not desired here
                raise ValueError("drop_last=False is not supported for DistributedSingleSlideBatchSampler (would require padding).")
            per_rank = n // self.num_replicas
            self._per_rank_counts.append(per_rank)
            self._num_batches += int(math.ceil(per_rank / self.batch_size)) if per_rank > 0 else 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self._num_batches

    def __iter__(self) -> Iterator[List[int]]:
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        slide_order = torch.arange(len(self.slide_indices), dtype=torch.long)
        if self.shuffle_slides:
            slide_order = slide_order[torch.randperm(slide_order.numel(), generator=g)]

        for sidx_t in slide_order:
            sidx = int(sidx_t.item())
            idxs = self.slide_indices[sidx]
            n = int(idxs.numel())
            if self.drop_last:
                n = (n // self.num_replicas) * self.num_replicas
            if n <= 0:
                continue
            idxs = idxs[:n]
            if self.shuffle_within_slide:
                idxs = idxs[torch.randperm(int(idxs.numel()), generator=g)]

            per_rank = n // self.num_replicas
            start = self.rank * per_rank
            end = start + per_rank
            local = idxs[start:end]

            # Yield true per-slide batches without crossing boundary
            for i in range(0, int(local.numel()), self.batch_size):
                yield local[i:i + self.batch_size].tolist()




class DistributedWeightedGlobalBatchUniqueSampler(Sampler[int]):
    """
    Weighted sampler that enforces:
      - replacement across batches: yes (balanced sampling)
      - replacement within a *global* batch: no (avoid CLIP false negatives)

    Implementation:
      - Sample an epoch-length index stream with replacement (one multinomial call)
      - View it as [num_global_batches, global_batch_size]
      - For each global batch row, replace duplicate indices using a small pre-sampled pool
      - In DDP, all ranks get consistent slices of each global batch
    """
    def __init__(
        self,
        weights: torch.Tensor,
        batch_size: int,
        num_replicas: int,
        rank: int,
        seed: int = 0,
        drop_last: bool = True,
        repair_pool_factor: float = 0.01,
        repair_pool_min: int = 4096,
        repair_refill: int = 4096,
    ) -> None:
        if weights.ndim != 1:
            raise ValueError(f"weights must be 1D, got shape={tuple(weights.shape)}")
        if not torch.isfinite(weights).all():
            raise ValueError("weights contain NaN/Inf")
        if (weights < 0).any():
            raise ValueError("weights contain negative values")
        if float(weights.sum()) <= 0.0:
            raise ValueError("weights sum to zero")

        self.weights = weights.detach().cpu().to(torch.double)
        self.batch_size = int(batch_size)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)

        n = int(self.weights.numel())
        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        self.global_batch_size = self.batch_size * self.num_replicas
        if n < self.global_batch_size:
            raise ValueError(f"Dataset too small for global-batch uniqueness: n={n} < global_bs={self.global_batch_size}")

        # If too many weights are zero, multinomial can only draw from a small support.
        # Then global-batch uniqueness may be impossible and the dedup loop can hang.
        support = int((self.weights > 0).sum().item())
        if support < self.global_batch_size:
            raise ValueError(
                f"Cannot enforce global-batch uniqueness with current weights: support={support} < global_bs={self.global_batch_size}. "
                "Reduce batch size/world size, or ensure more non-zero weights (e.g., avoid hard-zeroing too many samples)."
            )

        # Stream truncated to full local batches
        if self.drop_last:
            local_n = n // self.num_replicas
        else:
            local_n = int(math.ceil(n / self.num_replicas))
        local_n = (local_n // self.batch_size) * self.batch_size
        self.num_samples = local_n
        self.num_global_batches = self.num_samples // self.batch_size
        self.total_size = self.num_global_batches * self.global_batch_size

        self._pool_factor = float(repair_pool_factor)
        self._pool_min = int(repair_pool_min)
        self._refill = int(repair_refill)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _dedup_global_batches_inplace(self, batches: torch.Tensor, g: torch.Generator) -> None:
        """
        batches: LongTensor [num_batches, global_bs], sampled with replacement.
        Repairs duplicates in each row by drawing extra candidates from a pool.
        """
        # pre-sampled pool to avoid calling multinomial per batch
        pool_size = max(self._pool_min, int(self._pool_factor * self.total_size))
        pool = torch.multinomial(self.weights, pool_size, replacement=True, generator=g).tolist()
        pool_ptr = 0

        for b in range(batches.size(0)):
            row = batches[b].tolist()
            seen = set()
            dup_pos: List[int] = []
            for j, idx in enumerate(row):
                if idx in seen:
                    dup_pos.append(j)
                else:
                    seen.add(idx)
            if not dup_pos:
                continue

            needed = len(dup_pos)
            repl: List[int] = []
            while len(repl) < needed:
                if pool_ptr >= len(pool):
                    # refill pool in coarse chunks (still rare)
                    pool.extend(
                        torch.multinomial(self.weights, self._refill, replacement=True, generator=g).tolist()
                    )
                cand = pool[pool_ptr]
                pool_ptr += 1
                if cand not in seen:
                    repl.append(cand)
                    seen.add(cand)

            # in-place write replacements
            batches[b, torch.tensor(dup_pos, dtype=torch.long)] = torch.tensor(repl, dtype=torch.long)

    def __iter__(self) -> Iterator[int]:
        g = torch.Generator()
        # Important: same seed on all ranks so global batches are consistent
        g.manual_seed(self.seed + self.epoch)

        # Build the same global index matrix on every rank (no distributed collectives here)
        idx = torch.multinomial(self.weights, self.total_size, replacement=True, generator=g).to(torch.long)
        batches = idx.view(self.num_global_batches, self.global_batch_size)
        self._dedup_global_batches_inplace(batches, g)

        # Slice out this rank's portion of each global batch
        start = self.rank * self.batch_size
        end = start + self.batch_size
        local = batches[:, start:end].reshape(-1)
        return iter(local.tolist())

    def __len__(self) -> int:
        return self.num_samples
    



class DistributedStridedEvalSampler(Sampler[int]):
    """
    Deterministic distributed sampler for evaluation that:
      - does not drop samples
      - does not pad/duplicate samples
      - allows uneven per-rank lengths (at most 1 apart)

    Each rank gets indices: rank, rank+world_size, rank+2*world_size, ...
    This suits computing embeddings locally and gathering them to rank 0 afterwards
    with variable lengths (e.g., via all_gather_object).
    """
    def __init__(
        self,
        dataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
    ):
        self.dataset = dataset
        if num_replicas is None:
            num_replicas = dist.get_world_size() if (dist.is_available() and dist.is_initialized()) else 1
        if rank is None:
            rank = dist.get_rank() if (dist.is_available() and dist.is_initialized()) else 0
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.dataset_len = int(len(self.dataset))

    def __iter__(self) -> Iterator[int]:
        return iter(range(self.rank, self.dataset_len, self.num_replicas))

    def __len__(self) -> int:
        # number of indices in range(rank, N, world_size)
        return max(0, (self.dataset_len - self.rank + self.num_replicas - 1) // self.num_replicas)

    def set_epoch(self, epoch: int) -> None:
        # API compatibility with DistributedSampler; no-op (deterministic, shuffle-free)
        return