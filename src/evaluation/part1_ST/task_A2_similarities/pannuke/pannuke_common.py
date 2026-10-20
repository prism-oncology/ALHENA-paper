"""Vocabularies, label/tissue maps, geometry recipes, cell-id and path helpers shared by the PanNuke scripts."""

import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np


# ---------------------------------------------------------------------------------------------
# Nucleus classes
# ---------------------------------------------------------------------------------------------

# Canonical PanNuke nucleus-class names, in the release's own channel/index order (0..4).
PANNUKE_CLASS_NAMES: Tuple[str, ...] = (
    "Neoplastic",
    "Inflammatory",
    "Connective",
    "Dead",
    "Epithelial",
)

PANNUKE_CLASS_NAME_TO_INDEX: Dict[str, int] = {n: i for i, n in enumerate(PANNUKE_CLASS_NAMES)}


# PanNuke class -> A.2 coarse `v2` class (`Dead` is absent: dropped on both sides)
PANNUKE_CLASS_TO_COARSE_V2: Dict[str, str] = {
    "Neoplastic": "Epithelial_Melanocyte",
    "Epithelial": "Epithelial_Melanocyte",
    "Inflammatory": "Immune",
    "Connective": "Stromal_Vascular",
}

# Classes that leave the scored population entirely (two-sided drop).
PANNUKE_UNSCORABLE_CLASSES: Set[str] = {"Dead"}

# A.2 coarse-v2 scorable classes (coarse_ct_mapping_v2.json without `Unknown`)
COARSE_V2_SCORABLE_CLASSES: Tuple[str, ...] = (
    "Immune",
    "Stromal_Vascular",
    "Epithelial_Melanocyte",
)

# Label of the unscorable v2 classes (`Specialized` / `Unknown`), dropped on both sides
COARSE_V2_UNSCORABLE_LABEL: str = "Unknown"


# ---------------------------------------------------------------------------------------------
# Tissues
# ---------------------------------------------------------------------------------------------

# PanNuke tissues in the release's index order, in the CellViT spelling of `_4_opt_setup_patch_eval.py`
PANNUKE_TISSUES: Tuple[str, ...] = (
    "Adrenal_gland",
    "Bile-duct",
    "Bladder",
    "Breast",
    "Cervix",
    "Colon",
    "Esophagus",
    "HeadNeck",
    "Kidney",
    "Liver",
    "Lung",
    "Ovarian",
    "Pancreatic",
    "Prostate",
    "Skin",
    "Stomach",
    "Testis",
    "Thyroid",
    "Uterus",
)

PANNUKE_TISSUE_TO_INDEX: Dict[str, int] = {t: i for i, t in enumerate(PANNUKE_TISSUES)}


# Alternative tissue spellings (e.g. from HuggingFace RationAI/PanNuke) -> canonical tokens
_TISSUE_ALIASES: Dict[str, str] = {
    "adrenalgland": "Adrenal_gland",
    "adrenal gland": "Adrenal_gland",
    "adrenal_gland": "Adrenal_gland",
    "bileduct": "Bile-duct",
    "bile duct": "Bile-duct",
    "bile-duct": "Bile-duct",
    "bile_duct": "Bile-duct",
    "headneck": "HeadNeck",
    "head neck": "HeadNeck",
    "head & neck": "HeadNeck",
    "head and neck": "HeadNeck",
    "head-neck": "HeadNeck",
    "head_neck": "HeadNeck",
}


def normalize_tissue_name(value: str) -> str:
    """Map a PanNuke tissue spelling onto the canonical CellViT token (unknown strings are returned unchanged)."""
    raw = re.sub(r"\s+", " ", str(value).strip())
    key = raw.lower()
    if key in _TISSUE_ALIASES:
        return _TISSUE_ALIASES[key]
    for canonical in PANNUKE_TISSUES:
        if key == canonical.lower():
            return canonical
    return raw


# PanNuke tissue -> `tissue_HistoST` key of `data/eval_scRNAseq_ref/ref_tissue_files.json` (exact-name matches only)
PANNUKE_TISSUE_TO_HISTOST: Dict[str, str] = {
    "Breast": "Breast",
    "Cervix": "Cervix",
    "Colon": "Colon",
    "Kidney": "Kidney",
    "Liver": "Liver",
    "Lung": "Lung",
    "Ovarian": "Ovary",
    "Pancreatic": "Pancreatic",
    "Prostate": "Prostate",
    "Skin": "Skin",
    "Uterus": "Uterus",
}

PANNUKE_TISSUES_WITHOUT_REFERENCE: Tuple[str, ...] = (
    "Adrenal_gland",
    "Bile-duct",
    "Bladder",
    "Esophagus",
    "HeadNeck",
    "Stomach",
    "Testis",
    "Thyroid",
)


# ---------------------------------------------------------------------------------------------
# Published PanNuke figures (checked by _2_characterize_pannuke.py)
# ---------------------------------------------------------------------------------------------

# From the PanNuke extension paper / PLOS One 10.1371/journal.pone.0322749
PUBLISHED_N_TILES: int = 7901
PUBLISHED_FOLD_SIZES: Dict[str, int] = {"fold1": 2656, "fold2": 2523, "fold3": 2722}
PUBLISHED_N_NUCLEI: int = 189744
PUBLISHED_CLASS_COUNTS: Dict[str, int] = {
    "Neoplastic": 77403,
    "Inflammatory": 32276,
    "Connective": 50585,
    "Dead": 2908,
    "Epithelial": 26572,
}

# Tile size of the release (px)
PANNUKE_TILE_PX: int = 256

# Nominal magnification and resolution of the release (resampled, see PANNUKE_RESOLUTION_CAVEAT)
PANNUKE_NOMINAL_MAGNIFICATION: str = "40x (resampled, not native)"
PANNUKE_NOMINAL_MPP_UM_PER_PX: float = 0.25
PANNUKE_RESOLUTION_CAVEAT: str = (
    "PanNuke's '40x / 0.25 um/px' is a resampled nominal figure, not a native optical one. The "
    "extension paper (arXiv 2003.10778) states the source WSIs were scanned at 'either 20x or "
    "40x' and that the authors 're-sized the selected visual fields so that all images were at "
    "40x resolution'. An unmarked subset of tiles is therefore upsampled from 20x scans, and "
    "neither paper states um/px. The release does not mark which tiles."
)

PANNUKE_LICENSE: str = "CC-BY-NC-SA-4.0"
PANNUKE_CITATIONS: Tuple[str, ...] = (
    "Gamper et al., PanNuke: an open pan-cancer histology dataset for nuclei instance "
    "segmentation and classification, ECDP 2019.",
    "Gamper et al., PanNuke Dataset Extension, Insights and Baselines, arXiv:2003.10778, 2020.",
)


# ---------------------------------------------------------------------------------------------
# Vocabulary checks
# ---------------------------------------------------------------------------------------------

class PanNukeVocabularyError(ValueError):
    """Raised when observed PanNuke class ids / tissue strings leave the declared vocabulary."""


def assert_known_class_indices(indices: Iterable[int], context: str) -> None:
    """Every observed class index must be a valid PANNUKE_CLASS_NAMES position. Raises otherwise."""
    observed = {int(i) for i in indices}
    valid = set(range(len(PANNUKE_CLASS_NAMES)))
    unknown = sorted(observed - valid)
    if unknown:
        raise PanNukeVocabularyError(
            f"{context}: observed PanNuke class indices outside the declared vocabulary: "
            f"{unknown}. Declared order is {list(PANNUKE_CLASS_NAMES)} (indices "
            f"{sorted(valid)}). Update PANNUKE_CLASS_NAMES if the release changed."
        )


def assert_known_tissues(tissues: Iterable[str], context: str) -> None:
    """Raise if an observed tissue does not normalize onto a declared PanNuke tissue."""
    observed = {normalize_tissue_name(t) for t in tissues}
    unknown = sorted(observed - set(PANNUKE_TISSUES))
    if unknown:
        raise PanNukeVocabularyError(
            f"{context}: observed PanNuke tissue strings outside the declared vocabulary: "
            f"{unknown}. Declared tissues are {list(PANNUKE_TISSUES)}. Add the spelling to "
            "_TISSUE_ALIASES in pannuke_common.py if it is a known alias."
        )


# ---------------------------------------------------------------------------------------------
# Label mapping
# ---------------------------------------------------------------------------------------------

def is_scorable_class(class_name: str) -> bool:
    """False for the classes dropped on both sides (`Dead`)."""
    return str(class_name) not in PANNUKE_UNSCORABLE_CLASSES


def map_pannuke_class_to_v2(class_name: str) -> str:
    """Map one PanNuke class onto the A.2 coarse-v2 classes (raises on `Dead` and on unknown classes)."""
    name = str(class_name).strip()
    if name in PANNUKE_UNSCORABLE_CLASSES:
        raise PanNukeVocabularyError(
            f"PanNuke class '{name}' is dropped two-sided and has no coarse-v2 label. Filter "
            "unscorable classes out with `is_scorable_class` before mapping."
        )
    if name not in PANNUKE_CLASS_TO_COARSE_V2:
        raise PanNukeVocabularyError(
            f"PanNuke class '{name}' is not in the declared class->coarse_v2 map "
            f"{PANNUKE_CLASS_TO_COARSE_V2}."
        )
    return PANNUKE_CLASS_TO_COARSE_V2[name]


def map_pannuke_classes_to_v2(class_names: Sequence[str]) -> List[str]:
    """`map_pannuke_class_to_v2` applied to a sequence."""
    return [map_pannuke_class_to_v2(c) for c in class_names]


# ---------------------------------------------------------------------------------------------
# Per-nucleus geometry (pure helpers)
# ---------------------------------------------------------------------------------------------

def nucleus_area_px(instance_mask: np.ndarray) -> int:
    """Number of foreground pixels of one binary nucleus mask."""
    return int(np.count_nonzero(np.asarray(instance_mask)))


def touches_border(instance_mask: np.ndarray) -> bool:
    """True if any foreground pixel of the nucleus lies on the outermost row/column of the tile."""
    m = np.asarray(instance_mask)
    if m.ndim != 2:
        raise ValueError(f"Expected a 2-D instance mask, got shape {m.shape}.")
    if m.size == 0:
        return False
    return bool(
        np.any(m[0, :]) or np.any(m[-1, :]) or np.any(m[:, 0]) or np.any(m[:, -1])
    )


def instance_map_from_masks(
    masks: Sequence[np.ndarray], shape: Tuple[int, int]
) -> Tuple[np.ndarray, int]:
    """
    Flatten per-nucleus binary masks into one int32 instance map (0 = background, nucleus i -> id i + 1).
    Returns (instance_map, n_overlap_px); later masks overwrite earlier ones where they overlap.
    """
    h, w = int(shape[0]), int(shape[1])
    out = np.zeros((h, w), dtype=np.int32)
    n_overlap_px = 0
    for i, m in enumerate(masks):
        arr = np.asarray(m).astype(bool)
        if arr.shape != (h, w):
            raise ValueError(
                f"Instance {i} has shape {arr.shape}, expected {(h, w)}."
            )
        n_overlap_px += int(np.count_nonzero(out[arr]))
        out[arr] = np.int32(i + 1)
    return out, n_overlap_px


# ---------------------------------------------------------------------------------------------
# Cell-id budget
# ---------------------------------------------------------------------------------------------

# cell_id_int is int32 in the membership parquet (0 = background): usable range [1, 2**31)
CELL_ID_MAX_EXCLUSIVE: int = 2 ** 31

# Spacing between tiles in the composite cell id (must exceed the max number of nuclei per tile)
DEFAULT_TILE_ID_STRIDE: int = 512


def composite_cell_id(tile_index: int, instance_ordinal: int, stride: int = DEFAULT_TILE_ID_STRIDE) -> int:
    """Cell id from a tile index and a 1-based instance ordinal: tile_index * stride + instance_ordinal."""
    t, k = int(tile_index), int(instance_ordinal)
    if t < 0:
        raise ValueError(f"tile_index must be >= 0, got {t}.")
    if not (1 <= k <= stride):
        raise ValueError(
            f"instance_ordinal must be in [1, stride={stride}], got {k} (use a larger stride)."
        )
    return t * int(stride) + k


def assert_cell_id_budget(n_tiles: int, max_instances_per_tile: int, stride: int = DEFAULT_TILE_ID_STRIDE) -> Dict[str, int]:
    """Check that the composite cell ids of this cohort fit in int32. Returns the budget arithmetic."""
    n_tiles = int(n_tiles)
    max_inst = int(max_instances_per_tile)
    stride = int(stride)
    if max_inst > stride:
        raise ValueError(
            f"max_instances_per_tile={max_inst} exceeds stride={stride}; ids would collide "
            "across tiles (raise DEFAULT_TILE_ID_STRIDE)."
        )
    largest = composite_cell_id(max(n_tiles - 1, 0), max(max_inst, 1), stride=stride)
    if largest >= CELL_ID_MAX_EXCLUSIVE:
        raise ValueError(
            f"Largest composite cell id {largest} does not fit the int32 cell-id range "
            f"[1, {CELL_ID_MAX_EXCLUSIVE}). n_tiles={n_tiles}, stride={stride}."
        )
    return {
        "n_tiles": n_tiles,
        "max_instances_per_tile": max_inst,
        "stride": stride,
        "largest_cell_id": largest,
        "cell_id_max_exclusive": CELL_ID_MAX_EXCLUSIVE,
        "headroom_factor": int(CELL_ID_MAX_EXCLUSIVE // max(largest, 1)),
    }


# ---------------------------------------------------------------------------------------------
# The STHELAR calibrator for the empirical resolution check
# ---------------------------------------------------------------------------------------------

# Resolution of the STHELAR H&E images (um/px)
STHELAR_MPP_UM_PER_PX: float = 0.425


def per_cell_nucleus_area_px(membership: "pd.DataFrame") -> np.ndarray:
    """
    Per-cell nucleus area (px^2) from a STHELAR cell-token-membership table: nuc_px summed within
    each patch, then max over patches (patches overlap).
    """
    import pandas as pd  # lazy import

    required = {"cell_id_int", "patch_id", "nuc_px"}
    missing = required - set(membership.columns)
    if missing:
        raise KeyError(f"membership table is missing columns {sorted(missing)}")
    if membership.empty:
        return np.empty(0, dtype=np.int64)
    per_patch = membership.groupby(["cell_id_int", "patch_id"], sort=False)["nuc_px"].sum()
    return per_patch.groupby(level="cell_id_int", sort=False).max().to_numpy()


def estimate_mpp_from_area_ratio(
    median_area_px_query: float,
    median_area_px_reference: float,
    reference_mpp_um_per_px: float = STHELAR_MPP_UM_PER_PX,
) -> float:
    """
    Estimate a dataset's um/px from the ratio of median nucleus areas against a calibrator:
    mpp_query = mpp_ref * sqrt(area_ref / area_query).
    """
    if median_area_px_query <= 0 or median_area_px_reference <= 0:
        raise ValueError(
            "Median nucleus areas must be positive to estimate a resolution ratio; got "
            f"query={median_area_px_query}, reference={median_area_px_reference}."
        )
    return float(reference_mpp_um_per_px * np.sqrt(median_area_px_reference / median_area_px_query))


def bimodality_coefficient(values: Sequence[float]) -> float:
    """
    Sarle's bimodality coefficient of a sample: (skew^2 + 1) / (kurtosis + 3*(n-1)^2/((n-2)(n-3))).
    Values above ~0.555 suggest bimodality. Returns NaN for n < 4.
    """
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 4:
        return float("nan")
    mean = x.mean()
    sd = x.std(ddof=1)
    if sd <= 0:
        return float("nan")
    z = (x - mean) / sd
    skew = float(np.mean(z ** 3))
    kurt = float(np.mean(z ** 4)) - 3.0
    denom = kurt + 3.0 * ((n - 1) ** 2) / ((n - 2) * (n - 3))
    if denom <= 0:
        return float("nan")
    return float((skew ** 2 + 1.0) / denom)


# =============================================================================================
# Used by _4_prepare_patches.py: pseudo-slides, geometry recipes, instance-map compositing, admission floors
# =============================================================================================

# ---------------------------------------------------------------------------------------------
# Pseudo-slides
# ---------------------------------------------------------------------------------------------

# One pseudo-slide per PanNuke tissue (id = prefix + lowercase tissue)
PSEUDO_SLIDE_PREFIX: str = "pannuke_"


def pseudo_slide_id_for_tissue(tissue: str) -> str:
    """`Breast` -> `pannuke_breast` (lowercase canonical tissue token)."""
    canonical = normalize_tissue_name(tissue)
    assert_known_tissues([canonical], context=f"pseudo_slide_id_for_tissue({tissue!r})")
    return f"{PSEUDO_SLIDE_PREFIX}{canonical.lower()}"


def tissue_for_pseudo_slide_id(pseudo_slide_id: str) -> str:
    """Inverse of `pseudo_slide_id_for_tissue`, returning the canonical PanNuke tissue token."""
    text = str(pseudo_slide_id)
    if not text.startswith(PSEUDO_SLIDE_PREFIX):
        raise PanNukeVocabularyError(
            f"'{text}' is not a PanNuke pseudo-slide id: it must start with "
            f"'{PSEUDO_SLIDE_PREFIX}'."
        )
    canonical = normalize_tissue_name(text[len(PSEUDO_SLIDE_PREFIX):])
    assert_known_tissues([canonical], context=f"tissue_for_pseudo_slide_id({text!r})")
    return canonical


# ---------------------------------------------------------------------------------------------
# Geometry recipes
# ---------------------------------------------------------------------------------------------

# Model input size of the backbones and ViT token size (px)
MODEL_INPUT_PX: int = 224
VIT_TOKEN_PX_EXPECTED: int = 14

# Interpolation name -> cv2 flag name
INTERPOLATION_BY_NAME: Dict[str, str] = {
    "nearest": "INTER_NEAREST",
    "linear": "INTER_LINEAR",
    "cubic": "INTER_CUBIC",
    "area": "INTER_AREA",
}

# Sentinel for `native224`, whose image resize is left to `_8_compute_HEembs.py`.
INTERPOLATION_DEFERRED: str = "deferred"

RECIPE_NATIVE_224: str = "native224"
RECIPE_SCALE_224: str = "scale224"


@dataclass(frozen=True)
class GeometryRecipe:
    """
    One way of turning a 256 px PanNuke tile into a 224 px model input (fields in the model-input frame).

      `model_tissue_px`  number of the 224 model-input px covered by the tile
      `stored_px`        size of the canvas stored in the patch h5
      `defer_image_resize`
                         True  -> the h5 holds the raw 256 px tile, resized by `_8` with each
                                  backbone's own interpolation
                         False -> the stored canvas is already the 224 px model input
    """

    name: str
    model_tissue_px: int
    stored_px: int
    defer_image_resize: bool
    image_interpolation: str
    pad_value: int
    rationale: str

    @property
    def model_pad_each(self) -> int:
        """Padding on each side of the tissue within the 224 model input."""
        return (MODEL_INPUT_PX - int(self.model_tissue_px)) // 2

    @property
    def source_to_model_scale(self) -> float:
        """Linear factor taking a source (256 px) coordinate to the model-input frame."""
        return float(self.model_tissue_px) / float(PANNUKE_TILE_PX)

    @property
    def model_area_scale(self) -> float:
        """Area factor, i.e. what a source nucleus area is multiplied by in the model frame."""
        return self.source_to_model_scale ** 2

    @property
    def field_of_view_um(self) -> float:
        """Physical width of the tile (um), from the nominal resolution."""
        return float(PANNUKE_TILE_PX) * PANNUKE_NOMINAL_MPP_UM_PER_PX

    @property
    def mpp_um_per_px(self) -> float:
        """Effective um/px at the model input, from the nominal 0.25 um/px."""
        return self.field_of_view_um / float(self.model_tissue_px)

    @property
    def mpp_error_vs_sthelar(self) -> float:
        """Signed relative deviation from the STHELAR um/px."""
        return (self.mpp_um_per_px - STHELAR_MPP_UM_PER_PX) / STHELAR_MPP_UM_PER_PX

    def um_per_token(self, token_px: int = VIT_TOKEN_PX_EXPECTED) -> float:
        """Physical size of one ViT token (um)."""
        return float(token_px) * self.mpp_um_per_px

    def source_point_to_stored(self, coord: np.ndarray) -> np.ndarray:
        """
        Map a source (256 px) pixel coordinate onto the stored canvas (OpenCV half-pixel convention;
        identity for native224).
        """
        c = np.asarray(coord, dtype=np.float64)
        if self.defer_image_resize:
            return c
        scale = self.source_to_model_scale
        return (c + 0.5) * scale - 0.5 + float(self.model_pad_each)

    def as_dict(self, token_px: int = VIT_TOKEN_PX_EXPECTED) -> Dict[str, Any]:
        """JSON-ready record for `prepare_manifest.json`."""
        record = {
            "name": self.name,
            "model_input_px": MODEL_INPUT_PX,
            "model_tissue_px": int(self.model_tissue_px),
            "model_pad_each": int(self.model_pad_each),
            "stored_px": int(self.stored_px),
            "defer_image_resize": bool(self.defer_image_resize),
            "image_interpolation": self.image_interpolation,
            "mask_interpolation": "nearest",
            "pad_value": int(self.pad_value),
            "source_tile_px": int(PANNUKE_TILE_PX),
            "source_to_model_scale": self.source_to_model_scale,
            "model_area_scale": self.model_area_scale,
            "field_of_view_um": self.field_of_view_um,
            "mpp_um_per_px": self.mpp_um_per_px,
            "mpp_error_vs_sthelar": self.mpp_error_vs_sthelar,
            "sthelar_mpp_um_per_px": STHELAR_MPP_UM_PER_PX,
            "um_per_token": self.um_per_token(token_px),
            "nominal_mpp_source": "PANNUKE_NOMINAL_MPP_UM_PER_PX = "
                                  f"{PANNUKE_NOMINAL_MPP_UM_PER_PX}",
            "empirical_mpp_estimate_used": False,
            "resolution_caveat": PANNUKE_RESOLUTION_CAVEAT,
            "rationale": self.rationale,
        }
        record.update(token_composition(self, token_px=token_px))
        return record


# Default model_tissue_px of scale224 (token-aligned, close to the STHELAR um/px)
DEFAULT_SCALE_TISSUE_PX: int = 140

# Padding value (white)
DEFAULT_PAD_VALUE: int = 255

# Image interpolation of scale224 (the mask always uses nearest)
DEFAULT_SCALE_IMAGE_INTERPOLATION: str = "area"


GEOMETRY_RECIPE_NAMES: Tuple[str, ...] = (RECIPE_NATIVE_224, RECIPE_SCALE_224)


def build_geometry_recipe(
    name: str,
    *,
    scale_tissue_px: Optional[int] = None,
    image_interpolation: Optional[str] = None,
    pad_value: Optional[int] = None,
) -> GeometryRecipe:
    """Build one of the two recipes; the scale224 parameters can be overridden (not allowed for native224)."""
    name = str(name)
    if name not in GEOMETRY_RECIPE_NAMES:
        raise ValueError(
            f"Unknown geometry recipe '{name}'. Declared recipes: {list(GEOMETRY_RECIPE_NAMES)}."
        )

    if name == RECIPE_NATIVE_224:
        for label, value in (
            ("scale_tissue_px", scale_tissue_px),
            ("image_interpolation", image_interpolation),
            ("pad_value", pad_value),
        ):
            if value is not None:
                raise ValueError(
                    f"'{label}' does not apply to recipe '{RECIPE_NATIVE_224}' (raw "
                    f"{PANNUKE_TILE_PX} px tile, no resampling, no padding)."
                )
        recipe = GeometryRecipe(
            name=RECIPE_NATIVE_224,
            model_tissue_px=MODEL_INPUT_PX,
            stored_px=PANNUKE_TILE_PX,
            defer_image_resize=True,
            image_interpolation=INTERPOLATION_DEFERRED,
            pad_value=DEFAULT_PAD_VALUE,
            rationale=(
                "Store the raw 256 px tile so `_8_compute_HEembs.py` applies each backbone's own "
                "interpolation (uni2h INTER_LINEAR vs INTER_CUBIC for the other three). Full frame, "
                "no padding, no baked filter."
            ),
        )
    else:
        recipe = GeometryRecipe(
            name=RECIPE_SCALE_224,
            model_tissue_px=int(DEFAULT_SCALE_TISSUE_PX if scale_tissue_px is None else scale_tissue_px),
            stored_px=MODEL_INPUT_PX,
            defer_image_resize=False,
            image_interpolation=str(
                DEFAULT_SCALE_IMAGE_INTERPOLATION if image_interpolation is None else image_interpolation
            ),
            pad_value=int(DEFAULT_PAD_VALUE if pad_value is None else pad_value),
            rationale=(
                "Resample the tile towards STHELAR's ~0.425 um/px and pad to the model input "
                "(a 64 um PanNuke tile does not fill the 95 um field of 224 px at 0.425 um/px). The "
                "stored canvas is the model input."
            ),
        )

    assert_recipe_geometry(recipe)
    return recipe


def assert_recipe_geometry(
    recipe: GeometryRecipe,
    model_input_px: int = MODEL_INPUT_PX,
    token_px: int = VIT_TOKEN_PX_EXPECTED,
) -> Dict[str, Any]:
    """Raise if a recipe cannot produce an aligned image/mask pair. Returns the derived geometry."""
    model_input_px = int(model_input_px)
    token_px = int(token_px)
    tissue = int(recipe.model_tissue_px)

    if not (1 <= tissue <= model_input_px):
        raise ValueError(
            f"Recipe '{recipe.name}': model_tissue_px={tissue} must lie in [1, {model_input_px}]."
        )
    if (model_input_px - tissue) % 2 != 0:
        raise ValueError(
            f"Recipe '{recipe.name}': model_input_px - model_tissue_px = "
            f"{model_input_px - tissue} is odd (the tile cannot be centred with integer padding)."
        )
    if model_input_px % token_px != 0:
        raise ValueError(
            f"Model input {model_input_px} is not divisible by the ViT token size {token_px}."
        )
    if int(recipe.pad_value) < 0 or int(recipe.pad_value) > 255:
        raise ValueError(f"Recipe '{recipe.name}': pad_value must be a uint8, got {recipe.pad_value}.")

    if recipe.defer_image_resize:
        if int(recipe.stored_px) != PANNUKE_TILE_PX:
            raise ValueError(
                f"Recipe '{recipe.name}' defers the image resize, so it must store the raw "
                f"{PANNUKE_TILE_PX} px tile, not {recipe.stored_px}."
            )
        if recipe.model_pad_each != 0:
            raise ValueError(
                f"Recipe '{recipe.name}' defers the image resize, so it cannot pad."
            )
        if recipe.image_interpolation != INTERPOLATION_DEFERRED:
            raise ValueError(
                f"Recipe '{recipe.name}' defers the image resize, so its interpolation must be "
                f"'{INTERPOLATION_DEFERRED}' (each backbone applies its own), got "
                f"'{recipe.image_interpolation}'."
            )
    else:
        if int(recipe.stored_px) != model_input_px:
            raise ValueError(
                f"Recipe '{recipe.name}' bakes its geometry, so the stored canvas must be "
                f"{model_input_px} px. Got {recipe.stored_px}."
            )
        if recipe.image_interpolation not in INTERPOLATION_BY_NAME:
            raise ValueError(
                f"Recipe '{recipe.name}': unknown image interpolation "
                f"'{recipe.image_interpolation}'. Declared: {sorted(INTERPOLATION_BY_NAME)}."
            )

    derived = recipe.as_dict(token_px=token_px)
    derived["model_input_px"] = model_input_px
    return derived


def token_composition(
    recipe: GeometryRecipe,
    model_input_px: int = MODEL_INPUT_PX,
    token_px: int = VIT_TOKEN_PX_EXPECTED,
) -> Dict[str, Any]:
    """Count the tokens of the model input that are pure tissue, pure padding or mixed."""
    model_input_px = int(model_input_px)
    token_px = int(token_px)
    tokens_per_dim = model_input_px // token_px
    lo = recipe.model_pad_each
    hi = lo + int(recipe.model_tissue_px)

    # Tissue pixels contributed by each token position along one axis.
    starts = np.arange(tokens_per_dim, dtype=np.int64) * token_px
    ends = starts + token_px
    tissue_px_1d = np.clip(np.minimum(ends, hi) - np.maximum(starts, lo), 0, token_px)

    # A 2-D token is pure tissue only when both axes are, and pure padding when either axis is.
    pure_1d = tissue_px_1d == token_px
    empty_1d = tissue_px_1d == 0
    pure_tissue = int(np.count_nonzero(pure_1d) ** 2)
    pure_pad = int(np.count_nonzero(np.outer(empty_1d, np.ones(tokens_per_dim, bool)) |
                                    np.outer(np.ones(tokens_per_dim, bool), empty_1d)))
    total = tokens_per_dim * tokens_per_dim
    mixed = total - pure_tissue - pure_pad
    return {
        "tokens_per_dim": tokens_per_dim,
        "tokens_per_patch": total,
        "n_pure_tissue_tokens": pure_tissue,
        "n_pure_pad_tokens": pure_pad,
        "n_mixed_tokens": int(mixed),
        "token_aligned": bool(mixed == 0),
    }


def _cv2_flag(name: str) -> int:
    import cv2

    if name not in INTERPOLATION_BY_NAME:
        raise ValueError(f"Unknown interpolation '{name}'. Declared: {sorted(INTERPOLATION_BY_NAME)}.")
    return int(getattr(cv2, INTERPOLATION_BY_NAME[name]))


def _pad_to_canvas(array: np.ndarray, canvas_px: int, pad_each: int, pad_value: int) -> np.ndarray:
    """Centre `array` on a `canvas_px` square filled with `pad_value`, preserving dtype."""
    h, w = int(array.shape[0]), int(array.shape[1])
    if h != w:
        raise ValueError(f"Expected a square array to pad, got {array.shape}.")
    if h + 2 * pad_each != canvas_px:
        raise ValueError(
            f"Padding {pad_each} on each side of a {h} px array gives {h + 2 * pad_each} px, not "
            f"the declared canvas {canvas_px} px."
        )
    shape = (canvas_px, canvas_px) + tuple(array.shape[2:])
    out = np.full(shape, pad_value, dtype=array.dtype)
    out[pad_each:pad_each + h, pad_each:pad_each + w, ...] = array
    return out


def apply_recipe_to_image(image: np.ndarray, recipe: GeometryRecipe) -> np.ndarray:
    """Transform a raw 256x256x3 uint8 tile into the stored canvas (same geometry as apply_recipe_to_instance_map)."""
    img = np.asarray(image)
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"Expected an HxWx3 RGB tile, got shape {img.shape}.")
    if img.dtype != np.uint8:
        raise ValueError(f"Expected a uint8 tile, got dtype {img.dtype}.")
    if img.shape[0] != PANNUKE_TILE_PX or img.shape[1] != PANNUKE_TILE_PX:
        raise ValueError(
            f"Expected a {PANNUKE_TILE_PX}x{PANNUKE_TILE_PX} tile, got {img.shape[0]}x{img.shape[1]}."
        )

    if recipe.defer_image_resize:
        return img

    import cv2

    tissue = int(recipe.model_tissue_px)
    resized = cv2.resize(img, (tissue, tissue), interpolation=_cv2_flag(recipe.image_interpolation))
    return _pad_to_canvas(resized, int(recipe.stored_px), recipe.model_pad_each, int(recipe.pad_value))


def apply_recipe_to_instance_map(instance_map: np.ndarray, recipe: GeometryRecipe) -> np.ndarray:
    """Transform a raw 256x256 int32 instance map into the stored canvas (nearest-neighbour)."""
    m = np.asarray(instance_map)
    if m.ndim != 2:
        raise ValueError(f"Expected a 2-D instance map, got shape {m.shape}.")
    if m.dtype != np.int32:
        raise ValueError(
            f"Expected an int32 instance map, got dtype {m.dtype}."
        )
    if m.shape[0] != PANNUKE_TILE_PX or m.shape[1] != PANNUKE_TILE_PX:
        raise ValueError(
            f"Expected a {PANNUKE_TILE_PX}x{PANNUKE_TILE_PX} instance map, got {m.shape}."
        )

    if recipe.defer_image_resize:
        return m

    import cv2

    tissue = int(recipe.model_tissue_px)
    resized = cv2.resize(m, (tissue, tissue), interpolation=cv2.INTER_NEAREST)
    resized = np.asarray(resized, dtype=np.int32)
    return _pad_to_canvas(resized, int(recipe.stored_px), recipe.model_pad_each, 0)


# ---------------------------------------------------------------------------------------------
# Instance-map compositing (cell_id_int values)
# ---------------------------------------------------------------------------------------------

OVERLAP_LAST_WRITE_WINS: str = "last_write_wins"
OVERLAP_DROP_CONTESTED: str = "drop_contested"
OVERLAP_POLICIES: Tuple[str, ...] = (OVERLAP_LAST_WRITE_WINS, OVERLAP_DROP_CONTESTED)


@dataclass(frozen=True)
class CompositeInstanceMap:
    """Flattened instance map of one tile, with the realized pixel count and centroid of each nucleus."""

    instance_map: np.ndarray      # int32, values are `cell_id_int`, 0 = background
    n_overlap_px: int             # overwritten pixels (as in instance_map_from_masks)
    n_contested_px: int           # distinct pixels claimed by more than one nucleus
    realized_px: np.ndarray       # int64, per nucleus, after overlap resolution
    centroid_y: np.ndarray        # float64, source frame, NaN where `realized_px == 0`
    centroid_x: np.ndarray


def composite_instance_map(
    masks: Sequence[np.ndarray],
    cell_ids: Sequence[int],
    shape: Tuple[int, int],
    policy: str = OVERLAP_LAST_WRITE_WINS,
) -> CompositeInstanceMap:
    """
    Flatten per-nucleus binary masks into one int32 map whose values are cell_id_int.
    Overlaps: 'last_write_wins' (later masks overwrite earlier ones) or 'drop_contested' (contested
    pixels removed from every nucleus). realized_px[i] = 0 means nucleus i is lost at flatten.
    """
    policy = str(policy)
    if policy not in OVERLAP_POLICIES:
        raise ValueError(f"Unknown overlap policy '{policy}'. Declared: {list(OVERLAP_POLICIES)}.")

    h, w = int(shape[0]), int(shape[1])
    ids = np.asarray(list(cell_ids), dtype=np.int64)
    n = ids.shape[0]
    if len(masks) != n:
        raise ValueError(f"Got {len(masks)} masks but {n} cell ids; they must be parallel.")
    if n and (ids.min() < 1 or ids.max() >= CELL_ID_MAX_EXCLUSIVE):
        raise ValueError(
            f"cell ids must lie in [1, {CELL_ID_MAX_EXCLUSIVE}); got min={int(ids.min())}, "
            f"max={int(ids.max())}. 0 is reserved for background and the column is int32."
        )
    if n and np.unique(ids).size != n:
        raise ValueError("cell ids must be unique within a tile; got duplicates.")

    # Work in ordinal space (realized_px is one bincount)
    ordinal_map = np.zeros((h, w), dtype=np.int32)
    coverage = np.zeros((h, w), dtype=np.int32)
    for i, m in enumerate(masks):
        arr = np.asarray(m).astype(bool)
        if arr.shape != (h, w):
            raise ValueError(
                f"Instance {i} has shape {arr.shape}, expected {(h, w)}."
            )
        coverage[arr] += 1
        if policy == OVERLAP_LAST_WRITE_WINS:
            ordinal_map[arr] = np.int32(i + 1)

    contested = coverage > 1
    n_overlap_px = int((coverage[contested] - 1).sum()) if contested.any() else 0

    if policy == OVERLAP_DROP_CONTESTED:
        for i, m in enumerate(masks):
            arr = np.asarray(m).astype(bool) & ~contested
            ordinal_map[arr] = np.int32(i + 1)

    flat = ordinal_map.ravel()
    counts = np.bincount(flat, minlength=n + 1)
    realized_px = counts[1:n + 1].astype(np.int64, copy=False)

    # Centroids in one pass over the ordinal map
    rows = np.repeat(np.arange(h, dtype=np.float64), w)
    cols = np.tile(np.arange(w, dtype=np.float64), h)
    sum_y = np.bincount(flat, weights=rows, minlength=n + 1)[1:n + 1]
    sum_x = np.bincount(flat, weights=cols, minlength=n + 1)[1:n + 1]
    with np.errstate(invalid="ignore", divide="ignore"):
        denom = realized_px.astype(np.float64)
        centroid_y = np.where(denom > 0, sum_y / denom, np.nan)
        centroid_x = np.where(denom > 0, sum_x / denom, np.nan)

    lut = np.zeros(n + 1, dtype=np.int32)
    if n:
        lut[1:] = ids.astype(np.int32, copy=False)
    instance_map = lut[ordinal_map]
    return CompositeInstanceMap(
        instance_map=instance_map,
        n_overlap_px=n_overlap_px,
        n_contested_px=int(np.count_nonzero(contested)),
        realized_px=realized_px,
        centroid_y=centroid_y,
        centroid_x=centroid_x,
    )


# ---------------------------------------------------------------------------------------------
# A.2 slide-admission floors
# ---------------------------------------------------------------------------------------------

# Copy of the admission defaults of src_A2/protocol.py (keep in sync)
ADMISSION_MIN_SCORABLE_CELLS_PER_SLIDE: int = 1000
ADMISSION_MIN_CLASSES_PER_SLIDE: int = 3
ADMISSION_MIN_CLASS_SUPPORT: int = 50


def admission_verdict(
    class_counts: Mapping[str, int],
    scorable_classes: Sequence[str] = COARSE_V2_SCORABLE_CLASSES,
) -> Dict[str, Any]:
    """Whether A.2 would admit a pseudo-slide, given its token-eligible cell counts per class."""
    counts = {str(c): int(class_counts.get(c, 0)) for c in scorable_classes}
    unexpected = sorted(set(map(str, class_counts)) - set(counts))
    if unexpected:
        raise PanNukeVocabularyError(
            f"Class counts carry labels outside the scorable coarse-v2 axis: {unexpected}. "
            f"Declared: {list(scorable_classes)}."
        )

    n_cells = int(sum(counts.values()))
    at_support = [c for c, k in counts.items() if k >= ADMISSION_MIN_CLASS_SUPPORT]

    failures: List[str] = []
    if n_cells < ADMISSION_MIN_SCORABLE_CELLS_PER_SLIDE:
        failures.append(
            f"n_scorable_cells={n_cells} < min_scorable_gt_cells_per_slide="
            f"{ADMISSION_MIN_SCORABLE_CELLS_PER_SLIDE}"
        )
    if len(at_support) < ADMISSION_MIN_CLASSES_PER_SLIDE:
        thin = {c: k for c, k in counts.items() if k < ADMISSION_MIN_CLASS_SUPPORT}
        failures.append(
            f"only {len(at_support)} of {len(counts)} classes carry >= "
            f"{ADMISSION_MIN_CLASS_SUPPORT} cells (need {ADMISSION_MIN_CLASSES_PER_SLIDE}); "
            f"below floor: {thin}"
        )

    return {
        "n_scorable_cells": n_cells,
        "class_counts": counts,
        "class_margins": {c: k - ADMISSION_MIN_CLASS_SUPPORT for c, k in counts.items()},
        "n_classes_at_support": len(at_support),
        "min_scorable_gt_cells_per_slide": ADMISSION_MIN_SCORABLE_CELLS_PER_SLIDE,
        "min_classes_per_slide": ADMISSION_MIN_CLASSES_PER_SLIDE,
        "min_class_support_for_admission": ADMISSION_MIN_CLASS_SUPPORT,
        "passes": not failures,
        "failures": failures,
    }


# ---------------------------------------------------------------------------------------------
# Paths and provenance
# ---------------------------------------------------------------------------------------------

# Results of _6_run_retrieval.py: <ROOT_PATH>/eval/A-2_eval_st_predictions/pannuke/results/
A2_EVAL_TASK_DIRNAME: str = "A-2_eval_st_predictions"
PANNUKE_DIRNAME: str = "pannuke"
RESULTS_DIRNAME: str = "results"


def _src_import(name: str):
    """`src_A2.common.src_import`, imported lazily."""
    package_root = Path(__file__).resolve().parents[1]
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))
    from src_A2.common import src_import

    return src_import(name)


def pannuke_root() -> Path:
    """Root of the PanNuke dataset tree under ROOT_PATH (raw data, characterisation, prepared trees)."""
    return _src_import("get_full_path")("eval", "eval_datasets", "pannuke_eval_data")


def prepared_root(data_root: Path, recipe_name: str) -> Path:
    """Prepared tree of one geometry recipe: <data_root>/prepared/eval_st_pannuke_<recipe>."""
    return Path(data_root) / "prepared" / f"eval_st_pannuke_{recipe_name}"


def git_head() -> Optional[str]:
    """Current repo commit, recorded in every manifest. Returns None outside a git checkout."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(Path(__file__).resolve().parents[4]),
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except Exception:
        return None
