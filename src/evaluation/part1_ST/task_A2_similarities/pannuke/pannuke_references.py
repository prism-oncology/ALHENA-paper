"""scRNA-seq reference pools of the PanNuke retrieval (row filter, coarse-v2 labels, CLIP projection, coverage)."""

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# Project imports
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src_A2.common import ensure_dir, file_info, hash_mapping, read_json, src_import, write_json
from src_A2.coverage import (
    COVERAGE_DEFAULTS,
    STAGE_RUN,
    SlideCoverageReport,
    assert_coverage_contract,
    build_slide_coverage_report,
)
from src_A2.embeddings import (
    project_reference_embeddings_if_needed,
    resolve_raw_reference_embedding_h5,
)
from src_A2.label_granularity import apply_label_map, resolve_label_map
from src_A2.metrics import labels_to_ids
from src_A2.references import (
    ReferenceSpec,
    build_label_counts,
    drop_unscorable_reference_rows,
    filter_reference_rows,
    load_reference_mapping,
    read_obs_columns,
    resolve_reference_for_tissue,
)
from pannuke.pannuke_common import (
    A2_EVAL_TASK_DIRNAME,
    COARSE_V2_SCORABLE_CLASSES,
    COARSE_V2_UNSCORABLE_LABEL,
    PANNUKE_DIRNAME,
    PANNUKE_TISSUE_TO_HISTOST,
    RESULTS_DIRNAME,
    pannuke_root,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------------------------
# Protocol values, from eval_2/settings/eval_config.yaml
# ---------------------------------------------------------------------------------------------

CT_CAT_ID: str = "ct_1"
LABEL_GRANULARITY: str = "coarse"
COARSE_MAPPING_VERSION: str = "v2"

# `references.mapped_label_column_template` and `references.fine_label_column`
MAPPED_LABEL_COLUMN_TEMPLATE: str = "{ct_cat_id}_cell_type"
MAPPED_LABEL_COLUMN: str = MAPPED_LABEL_COLUMN_TEMPLATE.format(ct_cat_id=CT_CAT_ID)
FINE_LABEL_COLUMN: str = "original_cell_type"

# `references.filter.require_nonempty_expression` (streams the full expression matrix; the result is cached)
REQUIRE_NONEMPTY_EXPRESSION: bool = True

# `data.gt_mapped_unscorable_labels`, dropped from both the scored GT cells and the reference pool
UNSCORABLE_LABELS: Tuple[str, ...] = (COARSE_V2_UNSCORABLE_LABEL,)

# Label vocabulary in the A.2 order (`Unknown` included, although unscorable)
LABEL_ORDER: Tuple[str, ...] = tuple(COARSE_V2_SCORABLE_CLASSES) + (COARSE_V2_UNSCORABLE_LABEL,)
LABEL_TO_INDEX: Dict[str, int] = {label: i for i, label in enumerate(LABEL_ORDER)}

# `references.raw_reference_embedding_subdir`.
RAW_REFERENCE_EMBEDDING_SUBDIR: str = "scConcept_embs"

# `references.coverage` floors (A.2 defaults, no declared exceptions)
MIN_REF_CELLS_PER_GT_CLASS: int = int(COVERAGE_DEFAULTS["min_ref_cells_per_gt_class"])
MIN_SCORED_CELLS_PER_SLIDE: int = int(COVERAGE_DEFAULTS["min_scored_cells_per_slide"])

# Version of the reference-pool cache format (part of its signature)
REFERENCE_POOL_SCHEMA_VERSION: int = 1

# Batch size of the projection forward pass
PROJECTION_BATCH_SIZE: int = 4096


# ---------------------------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------------------------
# Inputs are read under data_root(), outputs written under results_root() (same folder with --data_dir)

_DATA_ROOT_OVERRIDE: Optional[Path] = None
_SHARED_ROOT_OVERRIDE: Optional[Path] = None


def set_data_root(path: Optional[Path]) -> None:
    """Use one folder for the inputs (prepared/) and outputs (results/) of the run; None restores the defaults."""
    global _DATA_ROOT_OVERRIDE
    _DATA_ROOT_OVERRIDE = Path(path).expanduser().resolve() if path is not None else None


def set_shared_root(path: Optional[Path]) -> None:
    """
    Redirect the files read outside the PanNuke tree (references, scConcept embeddings, A.2 filter
    cache, encoder bundles) to another root.
    """
    global _SHARED_ROOT_OVERRIDE
    _SHARED_ROOT_OVERRIDE = Path(path).expanduser().resolve() if path is not None else None


def data_root() -> Path:
    """Root of the PanNuke dataset tree (holding prepared/): pannuke_root() unless overridden."""
    return _DATA_ROOT_OVERRIDE if _DATA_ROOT_OVERRIDE is not None else pannuke_root()


def _full_path(*parts: str) -> Path:
    """`get_full_path(*parts)`, or the same parts under the root set by set_shared_root."""
    if _SHARED_ROOT_OVERRIDE is not None:
        out = Path(_SHARED_ROOT_OVERRIDE)
        for part in parts:
            out = out / str(part)
        return out
    return Path(src_import("get_full_path")(*parts))


def results_root() -> Path:
    """Root of the retrieval outputs: <ROOT_PATH>/eval/A-2_eval_st_predictions/pannuke/results/ (or <data_dir>/results)."""
    if _DATA_ROOT_OVERRIDE is not None:
        return _DATA_ROOT_OVERRIDE / RESULTS_DIRNAME
    return _full_path("eval", A2_EVAL_TASK_DIRNAME, PANNUKE_DIRNAME, RESULTS_DIRNAME)


def reference_pool_dir() -> Path:
    """PanNuke row-filter cache (shared by the recipes)."""
    return results_root() / "_reference_pool"


def projected_reference_cache_dir() -> Path:
    """Projected CLIP-space reference cache (shared by the recipes)."""
    return results_root() / "_projected_reference_cache"


def adatas_dir() -> Path:
    """Where the reference `.h5ad` files live under ROOT_PATH."""
    return _full_path("eval", "eval_datasets", "scRNAseq_references", "adatas")


def raw_reference_embedding_dir(scconcept_training_id: str) -> Path:
    """Raw scConcept reference embeddings of scconcept_training_id (read from the ST cell bundle)."""
    return _full_path(
        "eval",
        "eval_datasets",
        "scRNAseq_references",
        RAW_REFERENCE_EMBEDDING_SUBDIR,
        str(scconcept_training_id),
    )


def shipped_filter_cache_dir() -> Path:
    """A.2 row-filter cache (only read here)."""
    return _full_path("eval", "eval_datasets", "scRNAseq_references", "_a2_reference_filter_cache")


def st_cell_bundle_path(training_num: str, checkpoint_num: str) -> Path:
    """Exported ST cell bundle (with st_node_proj) of a checkpoint."""
    return _full_path(
        "train",
        "trainings",
        "trainings_HistoST",
        f"training_{training_num}",
        "trained_encoders",
        f"STencoderCell_t{training_num}e{checkpoint_num}.pt",
    )


def repo_ref_mapping_path() -> Path:
    """Path of data/eval_scRNAseq_ref/ref_tissue_files.json in the repo (or under the set_shared_root root)."""
    if _SHARED_ROOT_OVERRIDE is not None:
        return Path(_SHARED_ROOT_OVERRIDE) / "data" / "eval_scRNAseq_ref" / "ref_tissue_files.json"
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "src" / "utils" / "constants.py").exists():
            return parent / "data" / "eval_scRNAseq_ref" / "ref_tissue_files.json"
    return here.parents[5] / "data" / "eval_scRNAseq_ref" / "ref_tissue_files.json"


# ---------------------------------------------------------------------------------------------
# Config for the src_A2 modules
# ---------------------------------------------------------------------------------------------

def coverage_cfg() -> Dict[str, Any]:
    """Minimal `cfg` read by src_A2/coverage.py (references.coverage only)."""
    return {
        "references": {
            "coverage": {
                "enabled": True,
                "min_ref_cells_per_gt_class": MIN_REF_CELLS_PER_GT_CLASS,
                "min_scored_cells_per_slide": MIN_SCORED_CELLS_PER_SLIDE,
                "require_runtime_match": True,
                # No declared exceptions
                "declared_exceptions": {},
            }
        }
    }


def coarse_label_map() -> Dict[str, str]:
    """Fine -> coarse-v2 label map (src_A2 resolve_label_map)."""
    mapping = resolve_label_map(CT_CAT_ID, LABEL_GRANULARITY, COARSE_MAPPING_VERSION)
    if mapping is None:
        raise RuntimeError(
            "resolve_label_map returned None (the `fine` identity) for a coarse granularity."
        )
    observed = sorted(set(mapping.values()))
    expected = sorted(set(LABEL_ORDER))
    if observed != expected:
        raise ValueError(
            f"coarse_ct_mapping_{COARSE_MAPPING_VERSION}.json produces classes {observed}, but "
            f"LABEL_ORDER is {expected}."
        )
    return mapping


# ---------------------------------------------------------------------------------------------
# Row filter and its caches
# ---------------------------------------------------------------------------------------------

def root_independent_filter_signature(reference_h5ad_path: Path) -> Dict[str, Any]:
    """Signature of a row-filter result that does not depend on ROOT_PATH (file name, size, mtime, filter parameters)."""
    st = Path(reference_h5ad_path).stat()
    return {
        "reference_h5ad_name": Path(reference_h5ad_path).name,
        "reference_h5ad_size_bytes": int(st.st_size),
        "reference_h5ad_mtime_ns": int(st.st_mtime_ns),
        "mapped_label_column": MAPPED_LABEL_COLUMN,
        "fine_label_column": FINE_LABEL_COLUMN,
        "require_nonempty_expression": bool(REQUIRE_NONEMPTY_EXPRESSION),
        "schema_version": int(REFERENCE_POOL_SCHEMA_VERSION),
    }


def _shipped_signature_matches(shipped: Mapping[str, Any], ours: Mapping[str, Any]) -> bool:
    """Whether an A.2 filter-cache signature matches this run's signature (same fields, absolute path ignored)."""
    try:
        return (
            Path(str(shipped.get("reference_h5ad", ""))).name == ours["reference_h5ad_name"]
            and int(shipped.get("reference_h5ad_size_bytes", -1)) == ours["reference_h5ad_size_bytes"]
            and int(shipped.get("reference_h5ad_mtime_ns", -1)) == ours["reference_h5ad_mtime_ns"]
            and str(shipped.get("mapped_label_column", "")) == ours["mapped_label_column"]
            and str(shipped.get("fine_label_column", "")) == ours["fine_label_column"]
            and bool(shipped.get("require_nonempty_expression")) == ours["require_nonempty_expression"]
        )
    except (TypeError, ValueError):
        return False


def _verify_adopted_rows(
    filtered_rows: np.ndarray,
    fine_rows: np.ndarray,
    obs: pd.DataFrame,
    source: str,
) -> Optional[str]:
    """Check a cached row set against obs. Returns a reason string on failure, None if the rows are usable."""
    n_obs = int(obs.shape[0])
    if filtered_rows.ndim != 1 or fine_rows.ndim != 1:
        return f"{source}: cached row arrays are not 1-D"
    if filtered_rows.size == 0 or fine_rows.size == 0:
        return f"{source}: cached row arrays are empty"
    if filtered_rows.size != fine_rows.size:
        return (
            f"{source}: filtered_rows ({filtered_rows.size:,}) and fine_rows "
            f"({fine_rows.size:,}) differ in size"
        )
    if int(filtered_rows.min()) < 0 or int(filtered_rows.max()) >= n_obs:
        return f"{source}: cached rows fall outside [0, {n_obs:,})"
    if np.any(np.diff(filtered_rows) <= 0):
        return f"{source}: cached filtered_rows are not strictly increasing"
    if not np.isin(fine_rows, filtered_rows).all():
        return f"{source}: fine_rows is not a subset of filtered_rows"

    mapped = obs[MAPPED_LABEL_COLUMN]
    mapped_ok = mapped.notna().to_numpy() & (
        mapped.astype(str).str.strip().str.len().to_numpy() > 0
    )
    if not mapped_ok[filtered_rows].all():
        n_bad = int((~mapped_ok[filtered_rows]).sum())
        return (
            f"{source}: {n_bad:,} cached row(s) have a null or empty "
            f"'{MAPPED_LABEL_COLUMN}'"
        )
    return None


def _load_pannuke_filter_cache(stem: str, signature: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """Read the PanNuke-local cache blob for a reference, or None on any mismatch."""
    npz_path = reference_pool_dir() / f"{stem}.npz"
    json_path = reference_pool_dir() / f"{stem}.json"
    if not (npz_path.is_file() and json_path.is_file()):
        return None
    try:
        record = read_json(json_path)
        if record.get("signature") != dict(signature):
            return None
        with np.load(npz_path) as npz:
            return {
                "filtered_rows": np.asarray(npz["filtered_rows"], dtype=np.int64),
                "fine_rows": np.asarray(npz["fine_rows"], dtype=np.int64),
                "record": record,
            }
    except Exception:  # noqa: BLE001 - an unreadable cache is a miss, never a failure
        return None


def _write_pannuke_filter_cache(
    stem: str,
    signature: Mapping[str, Any],
    filtered_rows: np.ndarray,
    fine_rows: np.ndarray,
    provenance: Mapping[str, Any],
) -> Path:
    """Save the rows (before the unscorable drop) with their signature, atomically."""
    ensure_dir(reference_pool_dir())
    npz_path = reference_pool_dir() / f"{stem}.npz"
    json_path = reference_pool_dir() / f"{stem}.json"
    tmp = npz_path.with_name(npz_path.name + ".tmp.npz")
    np.savez_compressed(tmp, filtered_rows=filtered_rows, fine_rows=fine_rows)
    tmp.replace(npz_path)
    write_json(
        json_path,
        {
            "signature": dict(signature),
            "cache_file": npz_path.name,
            "n_filtered_rows": int(filtered_rows.size),
            "n_fine_rows": int(fine_rows.size),
            **dict(provenance),
        },
    )
    return npz_path


def _find_shipped_cache_entry(
    stem: str, signature: Mapping[str, Any]
) -> Optional[Tuple[Path, Dict[str, Any]]]:
    """Find an A.2 cache entry `_a2_reference_filter_cache/{stem}_{digest}.json` whose signature matches this run's."""
    cache_dir = shipped_filter_cache_dir()
    if not Path(cache_dir).is_dir():
        return None
    for json_path in sorted(Path(cache_dir).glob(f"{stem}_*.json")):
        if json_path.name.startswith("."):
            continue
        try:
            shipped = read_json(json_path).get("signature") or {}
        except Exception:  # noqa: BLE001
            continue
        if _shipped_signature_matches(shipped, signature):
            npz_path = json_path.with_suffix(".npz")
            if npz_path.is_file():
                return npz_path, dict(shipped)
    return None


def adopt_or_compute_filter(
    reference_h5ad_path: Path,
    *,
    allow_adoption: bool = True,
    log: Optional[logging.Logger] = None,
) -> Tuple[Any, Dict[str, Any]]:
    """
    A.2 row filter of a reference: from the PanNuke cache, else from a matching A.2 cache entry, else computed
    (then saved in the PanNuke cache). Returns (FilteredReference, record).
    """
    log = log or logger
    reference_h5ad_path = Path(reference_h5ad_path)
    stem = reference_h5ad_path.stem
    signature = root_independent_filter_signature(reference_h5ad_path)

    # obs is needed on every path
    obs = read_obs_columns(reference_h5ad_path, [MAPPED_LABEL_COLUMN, FINE_LABEL_COLUMN])

    from src_A2.references import FilteredReference

    def _as_reference(filtered_rows: np.ndarray, fine_rows: np.ndarray) -> Any:
        return FilteredReference(
            obs=obs,
            filtered_rows=np.asarray(filtered_rows, dtype=np.int64),
            fine_rows=np.asarray(fine_rows, dtype=np.int64),
            mapped_label_column=MAPPED_LABEL_COLUMN,
            fine_label_column=FINE_LABEL_COLUMN,
        )

    def _record(source: str, filtered_rows: np.ndarray, **extra: Any) -> Dict[str, Any]:
        return {
            "reference_stem": stem,
            "source": source,
            "n_obs": int(obs.shape[0]),
            "n_filtered_rows": int(np.asarray(filtered_rows).size),
            "retention_ratio": float(np.asarray(filtered_rows).size) / float(max(1, obs.shape[0])),
            "signature": dict(signature),
            **extra,
        }

    local = _load_pannuke_filter_cache(stem, signature)
    if local is not None and allow_adoption:
        reason = _verify_adopted_rows(local["filtered_rows"], local["fine_rows"], obs, "pannuke_cache")
        if reason is None:
            log.info(
                "Reference '%s': row filter served from the PanNuke-local cache (%d/%d rows).",
                stem,
                int(local["filtered_rows"].size),
                int(obs.shape[0]),
            )
            return _as_reference(local["filtered_rows"], local["fine_rows"]), _record(
                "pannuke_cache", local["filtered_rows"]
            )
        log.warning("Ignoring the PanNuke-local row-filter cache for '%s': %s", stem, reason)

    if allow_adoption:
        found = _find_shipped_cache_entry(stem, signature)
        if found is not None:
            npz_path, shipped_signature = found
            try:
                with np.load(npz_path) as npz:
                    filtered_rows = np.asarray(npz["filtered_rows"], dtype=np.int64)
                    fine_rows = np.asarray(npz["fine_rows"], dtype=np.int64)
            except Exception as exc:  # noqa: BLE001
                log.warning("Shipped row-filter cache %s is unreadable (%s); recomputing.", npz_path, exc)
            else:
                reason = _verify_adopted_rows(filtered_rows, fine_rows, obs, "shipped_cache")
                if reason is None:
                    log.info(
                        "Reference '%s': row filter adopted from the A.2 cache %s "
                        "(%d/%d rows; recorded root %r).",
                        stem,
                        npz_path.name,
                        int(filtered_rows.size),
                        int(obs.shape[0]),
                        str(Path(str(shipped_signature.get("reference_h5ad", ""))).parent),
                    )
                    _write_pannuke_filter_cache(
                        stem,
                        signature,
                        filtered_rows,
                        fine_rows,
                        {
                            "source": "shipped_cache",
                            "adopted_from": str(npz_path),
                            "shipped_signature": shipped_signature,
                        },
                    )
                    return _as_reference(filtered_rows, fine_rows), _record(
                        "shipped_cache", filtered_rows, adopted_from=str(npz_path)
                    )
                log.warning("Not adopting shipped cache %s: %s", npz_path.name, reason)

    log.info(
        "Reference '%s': computing the row filter (require_nonempty_expression=%s).",
        stem,
        REQUIRE_NONEMPTY_EXPRESSION,
    )
    ref_filtered = filter_reference_rows(
        reference_h5ad_path=reference_h5ad_path,
        mapped_label_column=MAPPED_LABEL_COLUMN,
        fine_label_column=FINE_LABEL_COLUMN,
        require_nonempty_expression=REQUIRE_NONEMPTY_EXPRESSION,
    )
    _write_pannuke_filter_cache(
        stem,
        signature,
        ref_filtered.filtered_rows,
        ref_filtered.fine_rows,
        {"source": "computed"},
    )
    return ref_filtered, _record("computed", ref_filtered.filtered_rows)


# ---------------------------------------------------------------------------------------------
# The retrieval pool
# ---------------------------------------------------------------------------------------------

@dataclass
class ReferencePool:
    """One tissue's retrieval pool, in the checkpoint's CLIP cell space."""

    tissue_histost: str
    reference_stem: str
    reference_h5ad: Path
    raw_embedding_h5: Path
    projected_cache: Path
    z_ref: np.ndarray = field(repr=False)
    labels: np.ndarray = field(repr=False)
    label_ids: np.ndarray = field(repr=False)
    label_counts: Dict[str, int] = field(default_factory=dict)
    audit: Dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def n_rows(self) -> int:
        return int(self.z_ref.shape[0])

    @property
    def embedding_dim(self) -> int:
        return int(self.z_ref.shape[1])


def safe_embedding_validity(z: np.ndarray) -> np.ndarray:
    """Mask of the finite rows with a non-zero norm."""
    if z.size == 0:
        return np.zeros((0,), dtype=bool)
    finite = np.all(np.isfinite(z), axis=1)
    norms = np.linalg.norm(z.astype(np.float32), axis=1)
    return finite & (norms > 0)


def resolve_reference_specs(tissues_histost: Sequence[str]) -> Dict[str, ReferenceSpec]:
    """tissue_HistoST -> reference .h5ad (ReferenceSpec), from the repo mapping made absolute under adatas/."""
    mapping_path = repo_ref_mapping_path()
    raw_mapping = load_reference_mapping(mapping_path)
    adatas = Path(adatas_dir())
    absolute = {
        str(tissue): str(
            (adatas / Path(str(value))).resolve()
            if not Path(str(value)).is_absolute()
            else Path(str(value)).resolve()
        )
        for tissue, value in raw_mapping.items()
    }
    out: Dict[str, ReferenceSpec] = {}
    for tissue in tissues_histost:
        out[str(tissue)] = resolve_reference_for_tissue(str(tissue), absolute, mapping_path)
    return out


def read_st_cell_bundle_facts(bundle_path: Path) -> Dict[str, Any]:
    """scConcept training_id, lambda_token and dimensions of the exported ST cell bundle (raises if lambda_token <= 0)."""
    import torch

    bundle_path = Path(bundle_path)
    if not bundle_path.is_file():
        raise FileNotFoundError(
            f"Exported ST cell bundle not found: {bundle_path}. "
            "Export it with src/evaluation/extract_encoders.py."
        )
    bundle = torch.load(bundle_path, map_location="cpu", weights_only=False)
    config = bundle["config"]
    state_dict = bundle["state_dict"]
    clip_cfg = config["model"]["clip"]
    lambda_token = float(clip_cfg["lambda_token"])
    if lambda_token <= 0.0:
        raise ValueError(
            f"{bundle_path} was trained with lambda_token={lambda_token}: no trained `st_node_proj` "
            "to project the reference."
        )
    from src_A2.embeddings import resolve_st_cell_bundle_dims

    in_dim, token_embed_dim = resolve_st_cell_bundle_dims(config, state_dict)
    scconcept_training_id = str(bundle.get("scconcept_training_id", "")).strip()
    if not scconcept_training_id:
        raise RuntimeError(
            f"Missing scconcept_training_id in {bundle_path}."
        )
    return {
        "st_cell_bundle": str(bundle_path),
        "scconcept_training_id": scconcept_training_id,
        "lambda_token": lambda_token,
        "in_dim": int(in_dim),
        "token_embed_dim": int(token_embed_dim),
        "training_num": str(bundle.get("training_num", "")),
        "checkpoint_num": str(bundle.get("checkpoint_num", "")),
    }


def _projected_cache_name(
    raw_ref_h5: Path, reference_h5ad: Path, st_cell_bundle_path_: Path
) -> str:
    """Name of a projected-reference file: reference stem + digest of its three inputs (size and mtime, not paths)."""
    payload = {
        "raw_reference_embedding": {
            k: v for k, v in file_info(Path(raw_ref_h5), compute_sha256=False).items() if k != "path"
        },
        "reference_h5ad": {
            k: v for k, v in file_info(Path(reference_h5ad), compute_sha256=False).items() if k != "path"
        },
        "st_cell_bundle": {
            k: v
            for k, v in file_info(Path(st_cell_bundle_path_), compute_sha256=False).items()
            if k != "path"
        },
        "schema_version": REFERENCE_POOL_SCHEMA_VERSION,
    }
    return f"{Path(reference_h5ad).stem}_{hash_mapping(payload)[:16]}"


def build_reference_pool(
    tissue_histost: str,
    spec: ReferenceSpec,
    *,
    bundle_facts: Mapping[str, Any],
    label_map: Mapping[str, str],
    device: str,
    allow_adoption: bool = True,
    log: Optional[logging.Logger] = None,
) -> ReferencePool:
    """Build one tissue's retrieval pool: row filter, coarse collapse, unscorable drop, projection, QC, label counts."""
    log = log or logger

    ref_filtered, filter_record = adopt_or_compute_filter(
        spec.path, allow_adoption=allow_adoption, log=log
    )
    n_before_drop = int(ref_filtered.filtered_rows.size)

    # Coarse collapse then unscorable drop (drop_unscorable_reference_rows applies label_map)
    ref_filtered, drops = drop_unscorable_reference_rows(
        ref_filtered,
        label_map=label_map,
        unscorable_labels=UNSCORABLE_LABELS,
        reference_h5ad_path=spec.path,
    )
    filtered_rows = np.asarray(ref_filtered.filtered_rows, dtype=np.int64)
    n_dropped = n_before_drop - int(filtered_rows.size)
    if n_dropped:
        log.info(
            "Reference '%s': dropped %d/%d filtered cell(s) (%.2f%%) whose coarse-%s label is "
            "unscorable %s. Per label: %s.",
            spec.stem,
            n_dropped,
            n_before_drop,
            100.0 * n_dropped / float(max(1, n_before_drop)),
            COARSE_MAPPING_VERSION,
            sorted(UNSCORABLE_LABELS),
            drops,
        )

    raw_ref_h5 = resolve_raw_reference_embedding_h5(
        Path(raw_reference_embedding_dir(bundle_facts["scconcept_training_id"])), spec.path
    )
    bundle_path = Path(bundle_facts["st_cell_bundle"])
    cache_name = _projected_cache_name(raw_ref_h5, spec.path, bundle_path)
    projected = project_reference_embeddings_if_needed(
        raw_ref_h5=raw_ref_h5,
        reference_h5ad_path=spec.path,
        projected_cache_path=projected_reference_cache_dir() / f"{cache_name}.h5",
        st_cell_bundle_path=bundle_path,
        device=device,
        batch_size=PROJECTION_BATCH_SIZE,
    )
    if int(projected.embedding_dim) != int(bundle_facts["token_embed_dim"]):
        raise RuntimeError(
            f"Projected reference for '{spec.stem}' has width {projected.embedding_dim}, but the "
            f"ST cell bundle declares token_embed_dim={bundle_facts['token_embed_dim']}."
        )

    z_all = projected.z_ref[filtered_rows]
    labels_all = np.asarray(
        apply_label_map(
            ref_filtered.obs.iloc[filtered_rows][MAPPED_LABEL_COLUMN].astype(str).to_numpy(),
            label_map,
        ),
        dtype=object,
    )

    valid = safe_embedding_validity(z_all)
    if not np.any(valid):
        raise RuntimeError(
            f"Every projected reference embedding is invalid (non-finite or zero-norm) for "
            f"'{spec.stem}'. The retrieval pool would be empty."
        )
    if not np.all(valid):
        log.warning(
            "Dropping %d/%d projected reference embedding(s) failing the finite/non-zero-norm QC "
            "for '%s'.",
            int((~valid).sum()),
            int(valid.size),
            spec.stem,
        )

    z_ref = np.ascontiguousarray(z_all[valid], dtype=np.float32)
    labels = labels_all[valid]
    label_ids = labels_to_ids(labels, LABEL_TO_INDEX)
    if np.any(label_ids < 0):
        missing = sorted(set(labels[label_ids < 0].tolist()))
        raise ValueError(
            f"Reference labels absent from LABEL_ORDER {list(LABEL_ORDER)} for "
            f"'{spec.stem}': {missing}."
        )
    unscorable_present = sorted(set(labels.tolist()) & set(UNSCORABLE_LABELS))
    if unscorable_present:
        raise RuntimeError(
            f"'{spec.stem}': the unscorable label(s) {unscorable_present} are still in the retrieval pool."
        )

    label_counts = build_label_counts(
        pd.DataFrame({MAPPED_LABEL_COLUMN: labels}),
        np.arange(labels.size, dtype=np.int64),
        MAPPED_LABEL_COLUMN,
    )

    log.info(
        "Reference pool '%s' (%s): %d cells, %d-d, classes %s",
        tissue_histost,
        spec.stem,
        int(z_ref.shape[0]),
        int(z_ref.shape[1]),
        {k: int(v) for k, v in sorted(label_counts.items())},
    )

    audit = {
        "tissue_HistoST": tissue_histost,
        "reference_stem": spec.stem,
        # Provenance only
        "reference_h5ad": str(spec.path),
        "raw_reference_embedding_h5": str(raw_ref_h5),
        "projected_cache": str(projected.path),
        "row_filter": filter_record,
        "n_rows_after_filter": n_before_drop,
        "n_rows_dropped_unscorable": int(n_dropped),
        "unscorable_drops": {str(k): int(v) for k, v in dict(drops).items()},
        "n_rows_after_unscorable_drop": int(filtered_rows.size),
        "n_rows_failing_embedding_qc": int((~valid).sum()),
        "n_rows_in_pool": int(z_ref.shape[0]),
        "embedding_dim": int(z_ref.shape[1]),
        "label_counts": {str(k): int(v) for k, v in sorted(label_counts.items())},
        "min_class_support": int(min(label_counts.values())) if label_counts else 0,
        "min_class_support_margin_factor": (
            float(min(label_counts.values())) / float(MIN_REF_CELLS_PER_GT_CLASS)
            if label_counts
            else 0.0
        ),
    }
    return ReferencePool(
        tissue_histost=tissue_histost,
        reference_stem=spec.stem,
        reference_h5ad=spec.path,
        raw_embedding_h5=raw_ref_h5,
        projected_cache=Path(projected.path),
        z_ref=z_ref,
        labels=labels,
        label_ids=label_ids,
        label_counts={str(k): int(v) for k, v in label_counts.items()},
        audit=audit,
    )


# ---------------------------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------------------------

def coverage_report_for_slide(
    slide_id: str,
    tissue_histost: str,
    pool: ReferencePool,
    gt_labels: Sequence[str],
    gt_scorable_mask: Sequence[bool],
) -> SlideCoverageReport:
    """A.2 coverage report of one slide, on the pool label counts (after the unscorable drop and the embedding QC)."""
    return build_slide_coverage_report(
        slide_id=str(slide_id),
        tissue=str(tissue_histost),
        reference_stem=pool.reference_stem,
        label_granularity=LABEL_GRANULARITY,
        stage=STAGE_RUN,
        gt_labels=gt_labels,
        gt_scorable_mask=gt_scorable_mask,
        ref_label_counts=pool.label_counts,
        cfg=coverage_cfg(),
    )


def assert_coverage(reports: Sequence[SlideCoverageReport], log: Optional[logging.Logger] = None) -> None:
    """Check the coverage contract over all slides at once (check_declared_bookkeeping=True)."""
    assert_coverage_contract(
        list(reports), coverage_cfg(), stage=STAGE_RUN, log=log or logger,
        check_declared_bookkeeping=True,
    )


def tissues_histost_for_pannuke() -> Dict[str, str]:
    """PanNuke tissue -> `tissue_HistoST` of the retained tissues."""
    return dict(PANNUKE_TISSUE_TO_HISTOST)


def protocol_record() -> Dict[str, Any]:
    """Reference-side protocol recorded in the run manifest."""
    return {
        "ct_cat_id": CT_CAT_ID,
        "label_granularity": LABEL_GRANULARITY,
        "coarse_mapping_version": COARSE_MAPPING_VERSION,
        "label_order": list(LABEL_ORDER),
        "scorable_classes": list(COARSE_V2_SCORABLE_CLASSES),
        "gt_mapped_unscorable_labels": list(UNSCORABLE_LABELS),
        "unscorable_drop_is_two_sided": True,
        "reference_mapped_label_column": MAPPED_LABEL_COLUMN,
        "reference_fine_label_column": FINE_LABEL_COLUMN,
        "require_nonempty_expression": REQUIRE_NONEMPTY_EXPRESSION,
        "coverage": coverage_cfg()["references"]["coverage"],
        "raw_reference_embedding_subdir": RAW_REFERENCE_EMBEDDING_SUBDIR,
        "reference_pool_schema_version": REFERENCE_POOL_SCHEMA_VERSION,
        "source": (
            "Values of the A.2 eval_2 protocol "
            "(eval/A-2_eval_st_predictions/eval_2/settings/eval_config.yaml)."
        ),
    }
