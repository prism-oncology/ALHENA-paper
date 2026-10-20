from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional
import logging
import math

from src_A2.common import read_yaml
from src_A2.coverage import COVERAGE_DEFAULTS, validate_coverage_config
from src_A2.label_granularity import (
    VALID_A2_LABEL_GRANULARITY,
    normalize_coarse_mapping_version,
)


logger = logging.getLogger(__name__)


# Task A.3 default of macro_min_class_support_per_slide (a different A.2 value is logged as a warning)
A3_MACRO_MIN_CLASS_SUPPORT_PER_SLIDE = 50


@dataclass
class ProtocolPaths:
    eval_root: Path
    settings_dir: Path
    shared_dir: Path
    protocol_yaml: Path
    metadata_snapshot_csv: Path
    ref_mapping_snapshot_json: Path
    metric_semantics_json: Path


REQUIRED_ROOT_KEYS = [
    "identity",
    "data",
    "references",
    "genes",
    "expression",
    "gene_panels",
    "retrieval",
    "tuning",
    "umap",
    "runtime",
    "metrics",
    "reporting",
]


def get_protocol_paths(eval_root: Path) -> ProtocolPaths:
    settings_dir = eval_root / "settings"
    return ProtocolPaths(
        eval_root=eval_root,
        settings_dir=settings_dir,
        shared_dir=eval_root / "shared",
        protocol_yaml=settings_dir / "eval_config.yaml",
        metadata_snapshot_csv=settings_dir / "metadata_snapshot.csv",
        ref_mapping_snapshot_json=settings_dir / "ref_tissue_files_snapshot.json",
        metric_semantics_json=settings_dir / "metric_semantics.json",
    )


DEFAULTS: Dict[str, Any] = {
    "identity": {
        "protocol_name": "A-2_cell_level_clip_space_biological_transfer",
        # fine: mapped {ct_cat_id}_cell_type labels; coarse: labels collapsed with the coarse mapping below
        "label_granularity": "fine",
        # Version of data/annots/annot_dicts_{ct_cat_id}/coarse_ct_mapping_{version}.json (v1: 5 classes,
        # v2: 3 classes); required when label_granularity is "coarse"
        "coarse_mapping_version": None,
    },
    "data": {
        "source": "STHELAR",
        "official_subsets": ["STHELAR_train", "STHELAR_test"],
        "subset_for_tuning": "STHELAR_train",
        "gt_label_column": "cell_type",
        "gt_fine_label_column": None,
        # Mapped labels removed from both the scored GT population and the reference retrieval pool ([] disables)
        "gt_mapped_unscorable_labels": ["Unknown"],
        # Slide admission: minimum number of scorable GT cells per slide (0 disables)
        "min_scorable_gt_cells_per_slide": 1000,
        # Slide admission: minimum number of scorable classes per slide (0 disables)
        "min_classes_per_slide": 3,
        # Minimum number of cells for a class to count in min_classes_per_slide (0: at least 1 cell)
        "min_class_support_for_admission": 50,
        "tissue_column": "tissue_HistoST",
        "source_column": "source",
        "subset_column": "subset",
        "slide_id_column": "slide_id",
    },
    "references": {
        "mapped_label_column_template": "{ct_cat_id}_cell_type",
        "fine_label_column": "original_cell_type",
        "filter": {
            "require_mapped_label": True,
            "require_nonempty_expression": True,
            "minimum_reference_cells": 200,
            # Minimum support over the GT classes present in the reference (missing classes: see coverage)
            "minimum_class_support": 5,
        },
        # Reference-coverage contract (defaults in coverage.COVERAGE_DEFAULTS)
        "coverage": dict(COVERAGE_DEFAULTS),
        "raw_reference_embedding_subdir": "scConcept_embs",
    },
    "genes": {
        "reference_gene_id_column": "gene_ids",
        "gt_gene_id_column": "gene_ids",
        "gt_gene_name_source": "var_index",
        "canonicalization": {
            "strip_whitespace": True,
            "uppercase": False,
            "strip_ensembl_version": False,
        },
    },
    "expression": {
        # False skips the expression transfer (the cell-type path is unaffected; incompatible
        # with the composite tuning objective)
        "enabled": True,
        "normalize_target_sum": 10000.0,
        "min_aligned_genes_official": 50,
        "variance_eps": 1e-12,
        "min_valid_cells_for_cellwise_corr": 10,
        "min_valid_cells_for_genewise_corr": 20,
        "min_valid_genes_for_slide_summary": 20,
        "g_base_id": None,
        "g_base_match_field": "gt_gene_name",
        "g_base_match_canonicalization": {
            "strip_whitespace": True,
            "uppercase": False,
            "strip_ensembl_version": False,
        },
    },
    "gene_panels": {
        "enabled": [
            "G_s_qc",
            "G_base_qc",
            "G_eval_slide_hvg50",
            "G_eval_slide_hvg100",
        ],
        "support_filters": {
            "st_min_nonzero_cells_abs": 20,
            "st_min_nonzero_rate": 0.005,
            "ref_min_nonzero_cells_abs": 50,
            "ref_min_nonzero_rate": 0.001,
            "use_st_variance_filter": True,
            "st_variance_eps": None,
        },
        "slide_hvg": {
            "Ns": [50, 100],
            "hvg_flavor": "seurat",
            "fallback_to_variance": True,
        },
        "tuning": {
            "expression_gene_panel": "G_base_qc",
            "min_genes_for_G_base_qc_tuning": 30,
        },
    },
    "retrieval": {
        "backend": "auto",
        "similarity": "cosine_after_l2",
        "k_grid": [10, 30, 50, 100, 200],
        "tau_grid": [0.01, 0.03, 0.05, 0.1, 0.2, 0.5],
        "Kmax": 200,
    },
    "tuning": {
        "strategy": "pooled_train_slide_macro_grid_search",
        "objective": "mapped_macro_f1_matched_covered",
        # composite_weights are used only when objective == composite_rank_matched_covered_he
        "composite_weights": {
            "mapped_macro_f1_matched_covered": 0.40,
            "gene_wise_pearson": 0.40,
            "mean_cellwise_pearson": 0.20,
        },
        "tie_breaks": ["smaller_k", "smaller_tau"],
        "fallback": {"k": 10, "tau": 0.1},
    },
    "umap": {
        "enabled": True,
        "reference_max_cells_per_slide": 5000,
        "reference_min_cells_per_class": 25,
        "query_max_cells_per_slide": 3000,
        "use_pca": True,
        "pca_dim": 50,
        "n_neighbors": 15,
        "min_dist": 0.3,
        "metric": "cosine",
        "random_state": 13,
        "randomize_query_draw_order": True,
        "plot_reference_alpha": 0.25,
        "plot_query_alpha": 0.75,
        "plot_reference_point_size": 3.0,
        "plot_query_point_size": 4.0,
    },
    "runtime": {
        "device": "auto",
        "seed": 13,
        "expression_query_batch_size": 512,
        "reference_expression_cache_row_block_size": 65536,
        "reference_expression_cache_target_block_mb": 128.0,
        "reference_expression_cache_compression": None,
        "reference_expression_cache_compression_opts": None,
        "retrieval_query_chunk_size": 2048,
        "retrieval_reference_chunk_size": 32768,
        "tuning_ref_expr_max_memory_gb": 8.0,
        "he_support": {
            "min_selected_alpha": 0.0,
        },
    },
    "metrics": {
        # Minimum per-slide class support to enter the slide's macro average (0 disables); applied to the
        # headline, the (k, tau) tuning objective and the chance floor, not to the pooled subset metrics
        "macro_min_class_support_per_slide": 50,
    },
    "reporting": {
        "prioritize_subsets": ["STHELAR_test", "STHELAR_train"],
        "save_debug_tables": True,
        "save_prediction_arrays": False,
        "save_selected_k_retrieval": False,
        # Write metrics/subset_pooled_expression_metrics.csv (expression pooled over cells within a subset)
        "save_subset_pooled_expression_metrics": True,
        "primary_metrics": [
            "mapped_macro_f1_matched_covered",
            "mapped_macro_f1_matched_all",
            "gene_wise_pearson",
            "gene_wise_spearman",
        ],
    },
}


def _deep_merge(base: Dict[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, Mapping) and isinstance(out.get(k), Mapping):
            out[k] = _deep_merge(dict(out[k]), v)
        else:
            out[k] = v
    return out


class ProtocolError(ValueError):
    pass


def load_protocol(eval_root: Path) -> Dict[str, Any]:
    paths = get_protocol_paths(eval_root)
    if not paths.protocol_yaml.exists():
        raise FileNotFoundError(paths.protocol_yaml)
    cfg = _deep_merge(DEFAULTS, read_yaml(paths.protocol_yaml))
    validate_protocol(cfg)
    return cfg



def validate_protocol(cfg: Mapping[str, Any]) -> None:
    missing = [k for k in REQUIRED_ROOT_KEYS if k not in cfg]
    if missing:
        raise ProtocolError(f"Protocol config missing required top-level keys: {missing}")

    k_grid = list(cfg["retrieval"]["k_grid"])
    Kmax = int(cfg["retrieval"]["Kmax"])
    if len(k_grid) == 0:
        raise ProtocolError("retrieval.k_grid cannot be empty")
    if Kmax < 1:
        raise ProtocolError("retrieval.Kmax must be >= 1")
    if any(int(x) < 1 for x in k_grid):
        raise ProtocolError("All retrieval.k_grid values must be positive integers")
    if max(int(x) for x in k_grid) > Kmax:
        raise ProtocolError("retrieval.Kmax must be >= max(k_grid)")

    tau_grid = list(cfg["retrieval"]["tau_grid"])
    if len(tau_grid) == 0:
        raise ProtocolError("retrieval.tau_grid cannot be empty")
    if any((not math.isfinite(float(t))) or float(t) <= 0 for t in tau_grid):
        raise ProtocolError("All retrieval.tau_grid values must be finite and > 0")

    similarity = str(cfg["retrieval"].get("similarity", "cosine_after_l2"))
    if similarity != "cosine_after_l2":
        raise ProtocolError("This A-2 implementation currently supports only retrieval.similarity == 'cosine_after_l2'.")

    label_granularity = str(cfg["identity"].get("label_granularity", "fine")).strip().lower()
    if label_granularity not in VALID_A2_LABEL_GRANULARITY:
        raise ProtocolError(
            f"Invalid identity.label_granularity={cfg['identity'].get('label_granularity')!r}. "
            f"Supported values are {sorted(VALID_A2_LABEL_GRANULARITY)}."
        )

    coarse_mapping_version = cfg["identity"].get("coarse_mapping_version", None)
    if label_granularity == "coarse":
        if coarse_mapping_version is None or not str(coarse_mapping_version).strip():
            raise ProtocolError(
                "identity.coarse_mapping_version is required when label_granularity='coarse' "
                "(e.g. 'v1' for the 5-class axis, 'v2' for the 3-class axis)."
            )
        try:
            normalize_coarse_mapping_version(coarse_mapping_version)
        except ValueError as exc:
            raise ProtocolError(str(exc)) from exc
    elif coarse_mapping_version is not None and str(coarse_mapping_version).strip():
        # Not used under fine, but still validated
        try:
            normalize_coarse_mapping_version(coarse_mapping_version)
        except ValueError as exc:
            raise ProtocolError(str(exc)) from exc
        logger.warning(
            "identity.coarse_mapping_version=%r is set but label_granularity='fine'; it has no effect.",
            coarse_mapping_version,
        )

    official_subsets = [str(x) for x in cfg["data"]["official_subsets"]]
    subset_for_tuning = str(cfg["data"]["subset_for_tuning"])
    if subset_for_tuning not in official_subsets:
        raise ProtocolError("data.subset_for_tuning must belong to data.official_subsets")

    # Mapped labels are required on the reference side
    if not bool(cfg["references"]["filter"].get("require_mapped_label", True)):
        raise ProtocolError("This A-2 implementation requires references.filter.require_mapped_label == true")

    if int(cfg["references"]["filter"]["minimum_reference_cells"]) < 1:
        raise ProtocolError("references.filter.minimum_reference_cells must be >= 1")
    if int(cfg["references"]["filter"]["minimum_class_support"]) < 1:
        raise ProtocolError("references.filter.minimum_class_support must be >= 1")

    # Reference-coverage contract (src_A2/coverage.py)
    try:
        validate_coverage_config(cfg)
    except ValueError as exc:
        raise ProtocolError(str(exc)) from exc
    if int(cfg["expression"]["min_aligned_genes_official"]) < 1:
        raise ProtocolError("expression.min_aligned_genes_official must be >= 1")
    if int(cfg["expression"]["min_valid_cells_for_cellwise_corr"]) < 1:
        raise ProtocolError("expression.min_valid_cells_for_cellwise_corr must be >= 1")
    if int(cfg["expression"]["min_valid_cells_for_genewise_corr"]) < 1:
        raise ProtocolError("expression.min_valid_cells_for_genewise_corr must be >= 1")
    if int(cfg["expression"]["min_valid_genes_for_slide_summary"]) < 1:
        raise ProtocolError("expression.min_valid_genes_for_slide_summary must be >= 1")

    gt_gene_name_source = str(cfg["genes"].get("gt_gene_name_source", "var_index"))
    if gt_gene_name_source != "var_index":
        raise ProtocolError(
            "This A-2 implementation currently supports only "
            "genes.gt_gene_name_source == 'var_index'. "
            "Gene names are report-only metadata read from STHELAR GT adata.var_names."
        )

    gene_panels = cfg.get("gene_panels", {})
    enabled_panels = [str(x) for x in gene_panels.get("enabled", [])]
    if not enabled_panels:
        raise ProtocolError("gene_panels.enabled cannot be empty")

    hvg_Ns = [int(x) for x in gene_panels.get("slide_hvg", {}).get("Ns", [])]
    if any(n <= 0 for n in hvg_Ns):
        raise ProtocolError("All gene_panels.slide_hvg.Ns values must be positive integers")

    allowed_fixed_panels = {"G_s_raw", "G_s_qc", "G_base_qc"}
    allowed_hvg_panels = {f"G_eval_slide_hvg{n}" for n in hvg_Ns}
    bad_panels = sorted(set(enabled_panels) - allowed_fixed_panels - allowed_hvg_panels)
    if bad_panels:
        raise ProtocolError(
            "Unsupported gene_panels.enabled entries. "
            f"Bad={bad_panels}; allowed fixed={sorted(allowed_fixed_panels)}, "
            f"allowed HVG={sorted(allowed_hvg_panels)}."
        )

    required_panels = {"G_s_qc", "G_base_qc"}
    missing_required_panels = sorted(required_panels - set(enabled_panels))
    if missing_required_panels:
        raise ProtocolError(
            "This A2 protocol requires gene_panels.enabled to include "
            f"{sorted(required_panels)}. Missing={missing_required_panels}."
        )

    tuning_panel = str(gene_panels.get("tuning", {}).get("expression_gene_panel", "G_base_qc"))
    if tuning_panel not in enabled_panels:
        raise ProtocolError("gene_panels.tuning.expression_gene_panel must be present in gene_panels.enabled")
    if tuning_panel != "G_base_qc":
        raise ProtocolError("A2 tuning must use gene_panels.tuning.expression_gene_panel == G_base_qc")

    min_tuning_genes = int(gene_panels.get("tuning", {}).get("min_genes_for_G_base_qc_tuning", 30))
    if min_tuning_genes < 1:
        raise ProtocolError("gene_panels.tuning.min_genes_for_G_base_qc_tuning must be >= 1")

    support = gene_panels.get("support_filters", {})
    if int(support.get("st_min_nonzero_cells_abs", 20)) < 0:
        raise ProtocolError("gene_panels.support_filters.st_min_nonzero_cells_abs must be >= 0")
    if int(support.get("ref_min_nonzero_cells_abs", 50)) < 0:
        raise ProtocolError("gene_panels.support_filters.ref_min_nonzero_cells_abs must be >= 0")
    if float(support.get("st_min_nonzero_rate", 0.005)) < 0:
        raise ProtocolError("gene_panels.support_filters.st_min_nonzero_rate must be >= 0")
    if float(support.get("ref_min_nonzero_rate", 0.001)) < 0:
        raise ProtocolError("gene_panels.support_filters.ref_min_nonzero_rate must be >= 0")

    if "G_base_qc" in enabled_panels and not cfg["expression"].get("g_base_id"):
        raise ProtocolError("expression.g_base_id must be set when G_base_qc is enabled")

    g_base_match_field = str(cfg["expression"].get("g_base_match_field", "gt_gene_name"))
    allowed_g_base_match_fields = {"gene", "gt_gene_name"}
    if g_base_match_field not in allowed_g_base_match_fields:
        raise ProtocolError(
            "expression.g_base_match_field must be one of "
            f"{sorted(allowed_g_base_match_fields)}. "
            "'gene' means canonical aligned gene IDs; 'gt_gene_name' means STHELAR GT var_names."
        )

    gbase_canon = cfg["expression"].get("g_base_match_canonicalization", {})
    if not isinstance(gbase_canon, Mapping):
        raise ProtocolError("expression.g_base_match_canonicalization must be a mapping.")
    for k in ["strip_whitespace", "uppercase", "strip_ensembl_version"]:
        if k in gbase_canon and not isinstance(gbase_canon[k], bool):
            raise ProtocolError(f"expression.g_base_match_canonicalization.{k} must be boolean.")

    objective = str(cfg["tuning"].get("objective", "mapped_macro_f1_matched_covered"))
    allowed_objectives = {
        "mapped_macro_f1_matched_covered",
        "mapped_macro_f1_matched_all",
        "composite_rank_matched_covered_he",
    }
    if objective not in allowed_objectives:
        raise ProtocolError(f"Unsupported tuning.objective. Supported values are {sorted(allowed_objectives)}.")

    strategy = str(cfg["tuning"].get("strategy", "pooled_train_slide_macro_grid_search"))
    allowed_strategies = {"pooled_train_slide_macro_grid_search"}
    if strategy not in allowed_strategies:
        raise ProtocolError("Unsupported tuning.strategy. Supported values are {'pooled_train_slide_macro_grid_search'}.")

    if objective == "composite_rank_matched_covered_he":
        weights = cfg["tuning"].get("composite_weights", {})
        required_weights = {
            "mapped_macro_f1_matched_covered",
            "gene_wise_pearson",
            "mean_cellwise_pearson",
        }
        if not isinstance(weights, Mapping):
            raise ProtocolError("tuning.composite_weights must be a mapping for composite_rank_matched_covered_he.")

        missing_weights = sorted(required_weights - set(map(str, weights.keys())))
        extra_weights = sorted(set(map(str, weights.keys())) - required_weights)
        if missing_weights or extra_weights:
            raise ProtocolError(
                "tuning.composite_weights must contain exactly "
                f"{sorted(required_weights)}. Missing={missing_weights}, extra={extra_weights}."
            )

        weight_values = {k: float(weights[k]) for k in required_weights}
        if any((not math.isfinite(v)) or v < 0.0 for v in weight_values.values()):
            raise ProtocolError("All tuning.composite_weights values must be finite and >= 0.")
        if not math.isclose(sum(weight_values.values()), 1.0, rel_tol=0.0, abs_tol=1e-6):
            raise ProtocolError("tuning.composite_weights must sum to 1.0.")

    tie_breaks = [str(x) for x in cfg["tuning"].get("tie_breaks", [])]
    allowed_tie_breaks = {
        "gene_wise_pearson",
        "mapped_macro_f1_matched_covered",
        "mean_cellwise_pearson",
        "smaller_k",
        "smaller_tau",
    }
    bad_ties = [x for x in tie_breaks if x not in allowed_tie_breaks]
    if bad_ties:
        raise ProtocolError(f"Unsupported tuning.tie_breaks entries: {bad_ties}. Allowed values are {sorted(allowed_tie_breaks)}.")

    # The composite objective requires expression
    expression_enabled = bool(cfg["expression"].get("enabled", True))
    if not expression_enabled:
        if objective == "composite_rank_matched_covered_he":
            raise ProtocolError(
                "expression.enabled is False but tuning.objective == "
                "'composite_rank_matched_covered_he', which requires expression-transfer "
                "metrics on every train slide. Use the default cell-type objective "
                "'mapped_macro_f1_matched_covered' or re-enable expression."
            )
        expr_tie_breaks = [t for t in tie_breaks if t in {"gene_wise_pearson", "mean_cellwise_pearson"}]
        if expr_tie_breaks:
            logger.warning(
                "expression.enabled is False: expression-based tuning.tie_breaks %s have no effect "
                "(only smaller_k and smaller_tau apply).",
                expr_tie_breaks,
            )

    source = str(cfg["data"]["source"])
    if source != "STHELAR":
        raise ProtocolError("This A-2 cell evaluation currently supports only source == 'STHELAR'")

    unscorable = cfg["data"].get("gt_mapped_unscorable_labels", [])
    if unscorable is None:
        cfg["data"]["gt_mapped_unscorable_labels"] = []
    elif not isinstance(unscorable, list):
        raise ProtocolError("data.gt_mapped_unscorable_labels must be a list of exact GT mapped labels.")

    macro_floor = cfg.get("metrics", {}).get("macro_min_class_support_per_slide", 0)
    if isinstance(macro_floor, bool) or not isinstance(macro_floor, int):
        raise ProtocolError(
            "metrics.macro_min_class_support_per_slide must be an integer (0 disables the "
            f"per-slide macro support floor); got {macro_floor!r}."
        )
    if macro_floor < 0:
        raise ProtocolError(
            f"metrics.macro_min_class_support_per_slide must be >= 0 (got {macro_floor})."
        )
    # Compare with the Task A.3 default
    if macro_floor == A3_MACRO_MIN_CLASS_SUPPORT_PER_SLIDE:
        logger.info(
            "metrics.macro_min_class_support_per_slide=%d (same as Task A.3), applied to the per-slide "
            "headline, the (k, tau) tuning objective and the chance floor.",
            macro_floor,
        )
    else:
        logger.warning(
            "metrics.macro_min_class_support_per_slide=%d differs from Task A.3's %d: the per-slide "
            "macro-F1 of the two tasks are not the same statistic.",
            macro_floor,
            A3_MACRO_MIN_CLASS_SUPPORT_PER_SLIDE,
        )

    for admission_key in (
        "min_scorable_gt_cells_per_slide",
        "min_classes_per_slide",
        "min_class_support_for_admission",
    ):
        value = cfg["data"].get(admission_key, 0)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ProtocolError(
                f"data.{admission_key} must be an integer (0 disables that slide-admission "
                f"floor); got {value!r}."
            )
        if value < 0:
            raise ProtocolError(f"data.{admission_key} must be >= 0 (got {value}).")

    # Log whether the class-admission support matches the macro floor
    admission_support = int(cfg["data"].get("min_class_support_for_admission", 0))
    if int(cfg["data"].get("min_classes_per_slide", 0)) > 0:
        if admission_support == macro_floor:
            logger.info(
                "data.min_class_support_for_admission=%d matches "
                "metrics.macro_min_class_support_per_slide (the residual is reported per slide as "
                "n_classes_below_macro_floor).",
                admission_support,
            )
        else:
            logger.warning(
                "data.min_class_support_for_admission=%d differs from "
                "metrics.macro_min_class_support_per_slide=%d, so data.min_classes_per_slide=%s counts "
                "classes on a different support scale than the macro. Check n_classes_below_macro_floor "
                "in the results.",
                admission_support,
                macro_floor,
                cfg["data"].get("min_classes_per_slide"),
            )
