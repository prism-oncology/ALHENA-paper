"""Data loading for Task A.3: cell-level loaders of the shared ST data layer and path helpers."""

from pathlib import Path

from src_A3.common import ensure_imports

ensure_imports()

# Imports after ensure_imports() (sys.path setup)
from src_A3.common import (  # noqa: E402
    resolve_eval_dataset_root,
    resolve_training_dataset_dir,
)
from src.evaluation.part1_ST._shared_st.st_data import (  # noqa: E402
    SlideRowBundle,
    build_row_manifest_for_slide,
    build_split_specific_cell_embedding_index,
    load_cell_embeddings_via_main_assignment,
    load_scconcept_cell_embeddings,
    load_slide_records,
    load_split_specific_cell_embeddings,
    resolve_scconcept_cell_embeddings_dir,
)


def scconcept_slide_h5_path(scconcept_training_id: str, slide_id: str) -> Path:
    """Path to a slide's per-cell scConcept embedding H5 (keys cell_id_int / cls_emb)."""
    return resolve_scconcept_cell_embeddings_dir(scconcept_training_id) / f"{slide_id}.h5"


def resolve_main_assign_path(embeddings_dir: Path, slide_id: str) -> Path:
    """Path to a slide's main H&E-embedding-per-cell assignment parquet (<embeddings_dir>/main_HEemb_cellID/)."""
    return Path(embeddings_dir) / "main_HEemb_cellID" / f"{slide_id}.parquet"


def centroid_parquet_path(dataset_id: str, training_dataset_id: str, slide_id: str) -> Path:
    """Path to a slide's nucleus-centroid parquet in the training-dataset dir."""
    training_dataset_dir = resolve_training_dataset_dir(dataset_id, training_dataset_id)
    return training_dataset_dir / "nucleus_centroids" / f"nucleus_centroids_{slide_id}.parquet"


def slide_patch_h5_path(training_dataset_id: str, slide_id: str) -> Path:
    """Path to a slide's patch H5 in the eval dataset."""
    eval_dataset_root = resolve_eval_dataset_root(training_dataset_id)
    return eval_dataset_root / "patches" / f"{slide_id}.h5"


__all__ = [
    # Shared ST data layer (_shared_st.st_data)
    "SlideRowBundle",
    "build_row_manifest_for_slide",
    "load_cell_embeddings_via_main_assignment",
    "build_split_specific_cell_embedding_index",
    "load_split_specific_cell_embeddings",
    "load_slide_records",
    # scConcept embedding loaders (shared layer)
    "load_scconcept_cell_embeddings",
    "resolve_scconcept_cell_embeddings_dir",
    # Task A.3 path helpers
    "resolve_main_assign_path",
    "centroid_parquet_path",
    "slide_patch_h5_path",
    "scconcept_slide_h5_path",
]
