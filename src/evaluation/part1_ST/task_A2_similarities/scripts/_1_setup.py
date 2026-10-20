#!/usr/bin/env python
"""
Set up the evaluation folder of a Task A.2 eval_id, with a default eval_config.yaml.

Saved in <ROOT_HistoST> / eval / A-2_eval_st_predictions / <eval_id> / settings /
    - eval_config.yaml: default evaluation protocol
    - metadata_snapshot.csv, ref_tissue_files_snapshot.json, ref_mapping_source.json, metric_semantics.json, sthelar_summary.json
"""

import argparse
import json
from pathlib import Path
import pandas as pd
import sys

# Project imports
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src_A2.common import ensure_dir, now_tag, write_json, write_text
from src_A2.label_granularity import VALID_A2_LABEL_GRANULARITY

from src_A2.common import src_import
get_full_path = src_import("get_full_path")



FOLDER_TREE = [
    "settings",
    "shared/manifests",
    "shared/reference_audits",
    "shared/gene_universes",
    "shared/gene_qc",
    "shared/umap_sampling",
    "shared/logs",
]


METRIC_SEMANTICS = {
    "_class_axis_note": "Every mapped-label metric below is computed over the class axis frozen in shared/manifests/label_vocabulary.json, i.e. at this eval_id's identity.label_granularity (fine or coarse). The formulas are granularity-agnostic; only the class axis changes.",
    "mapped_macro_f1_matched_covered": "Mapped-label macro F1 on matched cells whose GT mapped label is classification-scorable and present in the reference.",
    "mapped_macro_f1_matched_all": "Mapped-label macro F1 on matched cells whose GT mapped label is classification-scorable, without requiring reference coverage.",
    "gene_wise_pearson": "Average of per-gene Pearson correlations across cells within slide, reported separately by A2 gene_panel.",
    "gene_wise_spearman": "Average of per-gene Spearman correlations across cells within slide.",
    "mean_cellwise_pearson": "Average of per-cell Pearson correlations across shared genes within slide.",
    "mean_cellwise_spearman": "Average of per-cell Spearman correlations across shared genes within slide.",
    "composite_rank_matched_covered_he": "Rank-based tuning objective on STHELAR_train: 40% mapped macro-F1 on matched-covered H&E-query cells, 40% gene-wise Pearson on G_base_qc, 20% mean cell-wise Pearson on G_base_qc.",
}



def _make_eval_id(training_dataset_id: str, ct_cat_id: str, label_granularity: str,
                  coarse_mapping_version: str, modelgroup: str) -> str:
    clean_modelgroup = str(modelgroup).replace("/", "-")
    # The coarse mapping version is part of the generated id
    axis = label_granularity if not coarse_mapping_version else f"{label_granularity}-{coarse_mapping_version}"
    return f"eval_A2_{training_dataset_id}_{ct_cat_id}_{axis}_{clean_modelgroup}_{now_tag()}"




def _protocol_template(training_dataset_id: str, ct_cat_id: str, label_granularity: str,
                       coarse_mapping_version_yaml: str, modelgroup: str, G_base_id: str, device: str) -> str:
    return f"""# A-2 evaluation protocol for cell-level checkpoint-specific biological transfer
# in the learned CLIP cell space.
#
# - Only source == STHELAR is evaluated.
# - The H&E encoder must belong to the modelgroup of this eval_id.
# - device: auto means cuda if available, otherwise mps if available, otherwise cpu.
#
# Edit values if needed, but keep the YAML structure valid.

identity:
  protocol_name: A-2_cell_level_clip_space_biological_transfer  # Fixed protocol identifier for this evaluation family.
  training_dataset_id: {training_dataset_id}  # Training dataset ID whose eval dataset / shared artifacts are used.
  ct_cat_id: "{ct_cat_id}"  # Cell-type category ID used to resolve mapped reference labels via {{ct_cat_id}}_cell_type.
  label_granularity: "{label_granularity}"  # fine | coarse. Cell-type label granularity scored by this eval_id.
  # 'fine' scores the mapped {{ct_cat_id}}_cell_type labels as they are.
  # 'coarse' collapses them through
  # data/annots/annot_dicts_{{ct_cat_id}}/coarse_ct_mapping_{{coarse_mapping_version}}.json
  # before the label vocabulary, coverage, class support, the neighbour vote, (k, tau) tuning,
  # the chance floor and the GT-anchor ceiling are built.
  # references.fine_label_column (the scRNA reference's own annotation) is never collapsed.
  coarse_mapping_version: {coarse_mapping_version_yaml}  # Which versioned coarse mapping defines the class axis.
  # v1 -> 5 classes: T_NK, B_Plasma, Myeloid, Stromal_Vascular, Epithelial_Melanocyte
  # v2 -> 3 classes: Immune, Stromal_Vascular, Epithelial_Melanocyte
  # Required when label_granularity is 'coarse'.
  # Both versions map Specialized -> Unknown, so those cells leave the scored GT population and
  # the reference retrieval pool (see data.gt_mapped_unscorable_labels).
  modelgroup: "{modelgroup}"  # Frozen H&E preprocess/modelgroup; evaluated checkpoints must resolve to this exact group.

# Data semantics and required metadata / label column names
data:
  source: STHELAR  # Only rows with metadata[source_column] == this value are evaluated.
  official_subsets: [STHELAR_train, STHELAR_test]  # Subsets considered part of the official A-2 evaluation.
  subset_for_tuning: STHELAR_train  # Subset used to tune retrieval hyperparameters (k, tau).
  source_column: source  # Column name in metadata_snapshot.csv that stores the data source.
  subset_column: subset  # Column name in metadata_snapshot.csv that stores the evaluation subset.
  slide_id_column: slide_id  # Column name in metadata_snapshot.csv that stores the slide identifier.
  tissue_column: tissue_HistoST  # Column name in metadata_snapshot.csv used to choose the tissue-matched reference.
  gt_label_column: cell_type  # Ground-truth mapped label column in the STHELAR eval .h5ad cell rows.
  gt_fine_label_column: null  # Optional GT fine-label column in STHELAR eval data; keep null if not available / unused.
  gt_mapped_unscorable_labels: ["Unknown"]  # Mapped labels dropped from both the scored GT population and the reference retrieval pool (after the label_granularity collapse).
  # Slide admission: a slide with fewer classification-scorable GT cells than this (counted after the
  # label_granularity collapse) is excluded from the cohort (no scoring, no (k, tau) tuning). Applied before
  # the coverage contract; references.coverage.declared_exceptions does not apply to it. 0 disables it.
  min_scorable_gt_cells_per_slide: 1000
  # Slide admission: a slide with fewer scorable classes than this is excluded from the cohort. Counted on the
  # class axis of this eval_id (3 means 3 of 5 classes under coarse v1, all 3 under v2). 0 disables it.
  min_classes_per_slide: 3
  # Minimum number of scorable cells for a class to count in min_classes_per_slide (0: at least 1 cell).
  # Classes below metrics.macro_min_class_support_per_slide on the scored population are reported per slide
  # as n_classes_below_macro_floor.
  min_class_support_for_admission: 50

# Tissue-matched scRNA references
references:
  mapped_label_column_template: "{{ct_cat_id}}_cell_type"  # Reference obs column template for mapped labels; {{ct_cat_id}} is replaced from identity.ct_cat_id.
  fine_label_column: original_cell_type  # Reference obs column used for fine-label agreement diagnostics.
  filter:
    require_mapped_label: true  # Keep only reference cells with a non-empty mapped label.
    require_nonempty_expression: true  # Keep only reference cells with non-zero total expression.
    minimum_reference_cells: 200  # Minimum number of filtered reference cells required for official classification on a slide.
    minimum_class_support: 5  # Minimum support among covered mapped classes (GT classes present in the reference). GT classes absent from the reference are checked by references.coverage.
  # Reference-coverage contract, checked in _2 and again in _3 (after the projected-embedding QC).
  # A slide that violates it raises an error unless it is listed in declared_exceptions.
  coverage:
    enabled: true  # Master switch; false disables the contract.
    min_ref_cells_per_gt_class: 200  # Reference cells required for every classification-scorable GT class present on a slide (an absent class has 0).
    min_scored_cells_per_slide: 200  # Covered, scorable GT cells a slide must retain to be scored.
    require_runtime_match: true  # Check that the per-cell coverage of _3 equals the coverage frozen by _2.
    # {{slide_id: reason}}. A listed slide may violate the floors above; the reason is mandatory and
    # logged at WARNING on every run. A declared slide is scored and reported, but is excluded from the
    # (k, tau) tuning pool (recorded under `tuning_pool` in the run manifest). An exception only applies
    # when the slide violates the contract on this eval_id's class axis; otherwise the run warns.
    declared_exceptions:
      tonsil_s0: "Tonsil is mapped to tonsil.h5ad, which contains 0 Blood_vessel and 0 Fibroblast_Myofibroblast cells. Under both coarse versions these two classes merge into Stromal_Vascular, which therefore has 0 reference cells: every Stromal_Vascular GT cell of this slide is uncovered and the slide is scored on a reduced class axis. The uncovered fraction and class count are in the per-run coverage report (shared/reference_audits/)."
      tonsil_s1: "Tonsil is mapped to tonsil.h5ad, which contains 0 Blood_vessel and 0 Fibroblast_Myofibroblast cells. Under both coarse versions these two classes merge into Stromal_Vascular, which therefore has 0 reference cells: every Stromal_Vascular GT cell of this slide is uncovered and the slide is scored on a reduced class axis. The uncovered fraction and class count are in the per-run coverage report (shared/reference_audits/)."
      # lymph_node_s0 needs no entry: it is excluded by data.min_scorable_gt_cells_per_slide.
      ovary_s0: "Every GT class of this slide is present in ovary.h5ad, but it holds only 182 B_Plasma reference cells, below references.coverage.min_ref_cells_per_gt_class=200. Under coarse v2, B_Plasma merges into Immune (6965 reference cells) and this entry has no effect."
  raw_reference_embedding_subdir: scConcept_embs  # Folder name under eval/eval_datasets/scRNAseq_references/adatas/ that stores raw scConcept reference embeddings.

# Gene alignment rules between STHELAR GT expression and the scRNA reference
genes:
  reference_gene_id_column: gene_ids  # Column in reference .var that contains gene identifiers used for alignment.
  gt_gene_id_column: gene_ids  # Column in STHELAR GT .var that contains gene identifiers used for alignment.
  gt_gene_name_source: var_index  # Report-only gene names are read from STHELAR GT adata.var_names / adata.var index. Never used for alignment.
  canonicalization:
    strip_whitespace: true  # Strip leading/trailing whitespace before comparing gene IDs.
    uppercase: false  # Convert gene IDs to uppercase before comparing if true.
    strip_ensembl_version: false  # Remove Ensembl version suffixes like '.12' before comparing if true.

# Expression-transfer evaluation settings (diagnostic).
# With enabled: false, the GT expression matrix is not loaded, no metrics/*_gene_metrics.csv is written,
# and reporting.save_subset_pooled_expression_metrics, the runtime.*expression* knobs and the gene_panels
# block have no effect. _2 builds the gene-QC artifacts either way.
expression:
  enabled: false  # Master switch. false skips the (expensive) reference-expression transfer entirely; the cell-type path is unaffected. Incompatible with tuning.objective=composite_rank_matched_covered_he.
  normalize_target_sum: 10000.0  # Library-size normalization target applied once on G_s_raw before log1p for GT and reference expression comparison.
  min_aligned_genes_official: 50  # Minimum number of genes required in the primary official expression panel G_base_qc.
  variance_eps: 1.0e-12  # Small numerical floor used to reject nearly constant vectors in correlation computations.
  min_valid_cells_for_cellwise_corr: 10  # Minimum number of valid cells required to report mean cell-wise correlations.
  min_valid_cells_for_genewise_corr: 20  # Minimum number of valid cells required before attempting gene-wise correlations.
  min_valid_genes_for_slide_summary: 20  # Minimum number of valid genes required to report slide-level gene-wise summary metrics.
  g_base_id: "{G_base_id}"  # G_base ID used to define the fixed train-derived primary panel G_base_qc.
  g_base_match_field: gt_gene_name  # Use only for G_base membership. 'gt_gene_name' means STHELAR GT var_names; expression alignment still uses gene IDs.
  g_base_match_canonicalization:
    strip_whitespace: true
    uppercase: false
    strip_ensembl_version: false

# Gene-panel-aware expression evaluation.
# - G_s_raw is the raw shared gene universe between each STHELAR slide and its matched scRNA reference.
# - ST/reference expression is normalized/log1p once on G_s_raw.
# - Every panel below is a mask over G_s_raw (no per-panel renormalization).
gene_panels:
  enabled:
    - G_s_qc
    - G_base_qc
    - G_eval_slide_hvg50
    - G_eval_slide_hvg100
    #- G_s_raw  # Optional debug/stress-test panel

  support_filters:
    st_min_nonzero_cells_abs: 20
    st_min_nonzero_rate: 0.005
    ref_min_nonzero_cells_abs: 50
    ref_min_nonzero_rate: 0.001
    use_st_variance_filter: true
    st_variance_eps: null  # null means use expression.variance_eps.

  slide_hvg:
    Ns: [50, 100]
    hvg_flavor: seurat
    fallback_to_variance: true

  tuning:
    expression_gene_panel: G_base_qc
    min_genes_for_G_base_qc_tuning: 30

# Retrieval hyperparameters inside the projected CLIP cell space
retrieval:
  backend: auto  # auto -> FAISS exact inner-product search if available, otherwise deterministic NumPy exact search.
  similarity: cosine_after_l2  # Similarity definition used for retrieval. Current implementation supports only this value.
  k_grid: [10, 30, 50, 100, 200]  # Candidate top-k neighbor counts evaluated during tuning.
  tau_grid: [0.01, 0.03, 0.05, 0.1, 0.2, 0.5]  # Candidate softmax temperatures used to turn similarities into neighbor weights.
  Kmax: 200  # Maximum retrieval depth precomputed per slide; must be >= max(k_grid).

# Global hyperparameter selection on STHELAR_train only
# (pooled train-slide macro grid search; no model is refit).
tuning:
  strategy: pooled_train_slide_macro_grid_search  # Hyperparameter-selection strategy used by the current implementation.
  objective: mapped_macro_f1_matched_covered  # Select (k, tau) on cell-type macro-F1 over matched-covered H&E-query cells. composite_rank_matched_covered_he also weights expression transfer (requires expression.enabled).
  # composite_weights are used only when objective is composite_rank_matched_covered_he.
  composite_weights:
    mapped_macro_f1_matched_covered: 0.40
    gene_wise_pearson: 0.40
    mean_cellwise_pearson: 0.20
  tie_breaks: [smaller_k, smaller_tau]  # Ordered tie-break rules after the objective; parsimony only (no expression term under the cell-type default).
  fallback:
    k: 10  # Fallback k if tuning cannot be run.
    tau: 0.1  # Fallback tau if tuning cannot be run.

# UMAP diagnostics
# The same sampled reference cells and the same fitted UMAP space are reused
# for H&E and GT queries within each slide.
umap:
  enabled: true  # If true, generate per-slide UMAP diagnostic plots and coordinate tables.
  reference_max_cells_per_slide: 5000  # Maximum number of reference cells sampled for one slide's UMAP fit.
  reference_min_cells_per_class: 25  # Target minimum number of sampled reference cells per mapped class when possible.
  query_max_cells_per_slide: 3000  # Maximum number of H&E or GT query cells shown per slide in UMAP diagnostics.
  use_pca: true  # If true, fit PCA before UMAP.
  pca_dim: 50  # Number of PCA components used before UMAP when use_pca is true.
  n_neighbors: 15  # UMAP n_neighbors parameter.
  min_dist: 0.3  # UMAP min_dist parameter.
  metric: cosine  # UMAP distance metric.
  random_state: 13  # Random seed used for UMAP and UMAP sampling.
  # PNG rendering only. These do not affect UMAP coordinates, sampled cells,
  # labels, metrics, retrieval, or saved coordinate tables.
  randomize_query_draw_order: true  # Randomize H&E/GT query draw order only during PNG rendering.
  plot_reference_alpha: 0.25  # Lower alpha for reference/background points.
  plot_query_alpha: 0.75  # Alpha for H&E/GT query points.
  plot_reference_point_size: 3.0  # Point size for reference/background points.
  plot_query_point_size: 4.0  # Point size for H&E/GT query points.

# Runtime behavior
runtime:
  device: {device}  # Device string: auto, cuda, cuda:0, mps, cpu, etc.
  seed: 13  # Global random seed used for deterministic evaluation steps.
  expression_query_batch_size: 512  # Batch size used when predicting expression from retrieved reference neighbors.
  retrieval_query_chunk_size: 2048  # Query chunk size for NumPy exact retrieval fallback.
  retrieval_reference_chunk_size: 32768  # Reference chunk size for NumPy exact retrieval fallback / lower if RAM is limited.
  reference_expression_cache_row_block_size: 65536  # Max contiguous reference-row scan block; memory target below still caps wide gene universes.
  reference_expression_cache_target_block_mb: 128.0  # Approximate dense temporary memory target while building expression caches.
  reference_expression_cache_compression: null  # null is fastest; set to gzip only if disk space is more important than runtime.
  reference_expression_cache_compression_opts: null  # e.g. 1 or 4 when compression is gzip.
  tuning_ref_expr_max_memory_gb: 8.0  # Load tuning expression cache into RAM when smaller than this.
  he_support:
    min_selected_alpha: 0.0  # H&E rows require selected_alpha strictly greater than this value to be considered usable.

# Metric definitions
metrics:
  # Minimum per-slide cell support for a class to enter that slide's macro average (0 disables it).
  # It changes the denominator only: no cell is dropped. Applied to the per-slide macro-F1, the (k, tau)
  # tuning objective and the chance floor, not to the pooled subset metrics.
  macro_min_class_support_per_slide: 50

# Reporting priorities
reporting:
  prioritize_subsets: [STHELAR_test, STHELAR_train]  # Preferred subset order in summaries and reports.
  save_debug_tables: true  # If true, save per-class, confusion, per-gene, and per-cell detail tables (only the classification tables when expression.enabled is false).
  save_prediction_arrays: false  # If true, save large .npz arrays containing predicted expression matrices.
  save_selected_k_retrieval: false  # If true, save selected-k retrieval files in addition to topKmax retrieval caches.
  # If true, write metrics/subset_pooled_expression_metrics.csv (expression pooled over cells within a subset).
  # Independent of save_debug_tables. No effect when expression.enabled is false.
  save_subset_pooled_expression_metrics: true
  primary_metrics:  # Main metrics to highlight first in reports / tables.
    - mapped_macro_f1_matched_covered  # Primary mapped-label macro-F1 on matched covered cells.
    - mapped_macro_f1_matched_all  # Mapped-label macro-F1 on all matched cells.
    - gene_wise_pearson  # Mean gene-wise Pearson correlation over valid genes.
    - gene_wise_spearman  # Mean gene-wise Spearman correlation over valid genes.
"""



def main(args: argparse.Namespace) -> None:

    if args.label_granularity == "coarse" and not args.coarse_mapping_version:
        raise SystemExit(
            "--coarse_mapping_version is required when --label_granularity coarse "
            "(e.g. 'v1' for the 5-class axis, 'v2' for the 3-class axis)."
        )

    eval_id = args.eval_id or _make_eval_id(
        args.training_dataset_id,
        args.ct_cat_id,
        args.label_granularity,
        args.coarse_mapping_version,
        args.modelgroup,
    )
    eval_root = get_full_path("eval", "A-2_eval_st_predictions", eval_id)
    if eval_root.exists() and any(eval_root.iterdir()):
        raise FileExistsError(
            f"Refusing to reuse existing eval_id '{eval_id}' at {eval_root}. "
            "Create a new eval_id."
        )
    ensure_dir(eval_root)
    for rel in FOLDER_TREE:
        ensure_dir(eval_root / rel)
    
    print(f"Setting up A2 evaluation protocol at {eval_root} with eval_id {eval_id}...")

    metadata_csv = get_full_path("eval", "eval_datasets", "st_eval_data", f"eval_st_{args.training_dataset_id}", f"metadata_eval_st_{args.training_dataset_id}.csv")
    ref_mapping = Path(args.ref_mapping_json).expanduser().resolve()   # path in the code repository (not under the data root)
    scRNAseq_ref_folder = get_full_path("eval", "eval_datasets", "scRNAseq_references", "adatas")

    if not metadata_csv.exists():
        raise FileNotFoundError(metadata_csv)
    if not ref_mapping.exists():
        raise FileNotFoundError(ref_mapping)
    mapping_raw = json.loads(ref_mapping.read_text())
    if not isinstance(mapping_raw, dict) or len(mapping_raw) == 0:
        raise ValueError(f"Invalid reference mapping JSON: {ref_mapping}")

    # Store absolute reference paths in the snapshot
    mapping_payload = {}
    for tissue, ref_value in mapping_raw.items():
        p = Path(str(ref_value))
        if not p.is_absolute():
            p = (scRNAseq_ref_folder / p).resolve()
        else:
            p = p.resolve()
        if not p.exists():
            raise FileNotFoundError(f"Reference file for tissue '{tissue}' does not exist after resolution: {p}")
        mapping_payload[str(tissue)] = str(p)

    md = pd.read_csv(metadata_csv)
    sthelar = md.loc[md["source"].astype(str) == "STHELAR"].copy()
    if sthelar.empty:
        raise ValueError("No STHELAR rows found in metadata; this protocol requires STHELAR cell data.")
    
    for col in ["slide_id", "subset", "tissue_HistoST"]:
        if col not in sthelar.columns:
            raise ValueError(f"Column {col} not found in metadata.")

    missing_tissues = sorted(set(sthelar["tissue_HistoST"].astype(str).unique().tolist()) - set(mapping_payload.keys()))
    if missing_tissues:
        raise ValueError(
            "The reference mapping does not cover all STHELAR tissues found in metadata. "
            f"Missing mapping keys: {missing_tissues}"
        )

    sthelar_summary = {
        "n_slides_total": int(sthelar["slide_id"].nunique()),
        "slides_by_subset": sthelar.groupby("subset")["slide_id"].nunique().to_dict(),
        "slides_by_tissue": sthelar.groupby("tissue_HistoST")["slide_id"].nunique().to_dict(),
    }

    (eval_root / "settings" / "metadata_snapshot.csv").write_text(md.to_csv(index=False))
    write_json(eval_root / "settings" / "ref_tissue_files_snapshot.json", mapping_payload)
    write_json(
        eval_root / "settings" / "ref_mapping_source.json",
        {
            "original_mapping_json": str(ref_mapping),
            "snapshot_contains_absolute_paths": True,
        },
    )
    write_json(eval_root / "settings" / "metric_semantics.json", METRIC_SEMANTICS)
    write_json(eval_root / "settings" / "sthelar_summary.json", sthelar_summary)
    # YAML null when no coarse mapping version is given
    coarse_mapping_version_yaml = (
        f'"{args.coarse_mapping_version}"' if args.coarse_mapping_version else "null"
    )
    write_text(
        eval_root / "settings" / "eval_config.yaml",
        _protocol_template(
            args.training_dataset_id,
            args.ct_cat_id,
            args.label_granularity,
            coarse_mapping_version_yaml,
            args.modelgroup,
            args.G_base_id,
            args.device,
        ),
    )

    print(f"Created eval_id: {eval_id}")
    print(f"Root: {eval_root}")
    print(f"Label granularity: {args.label_granularity}"
          + (f" (coarse_mapping_version={args.coarse_mapping_version})"
             if args.coarse_mapping_version else ""))
    print(f"STHELAR slide count: {sthelar_summary['n_slides_total']}")
    print("You can now edit settings/eval_config.yaml before running _2_prepare_shared_artifacts.py and _3_run_similarities.py.")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create an A-2 evaluation protocol folder.")
    
    parser.add_argument("--training_dataset_id", type=str, 
                        required=True, 
                        help="Training dataset ID, e.g. ds_ct1_20x_224_o112_1")
    
    parser.add_argument("--ct_cat_id", type=str,
                        required=True,
                        help="Cell-type category ID used for mapped reference labels.")

    parser.add_argument("--label_granularity", type=str,
                        required=True,
                        choices=sorted(VALID_A2_LABEL_GRANULARITY),
                        help=("Cell-type label granularity scored by this eval_id: 'fine' "
                              "({ct_cat_id}_cell_type labels) or 'coarse' (labels collapsed with the "
                              "mapping selected by --coarse_mapping_version)."))

    parser.add_argument("--coarse_mapping_version", type=str,
                        default=None,
                        help=("Version of the coarse mapping "
                              "data/annots/annot_dicts_{ct_cat_id}/coarse_ct_mapping_{version}.json: "
                              "'v1' (5 classes) or 'v2' (3 classes). Required with --label_granularity coarse."))

    parser.add_argument("--modelgroup", type=str,
                        required=True, 
                        help="Frozen modelgroup, e.g. hoptimus0_hoptimus1_uni2h_virchow2")
    
    parser.add_argument("--eval_id", type=str, 
                        default=None, 
                        help="Optional explicit eval_id. If omitted, one is generated.")
    
    parser.add_argument("--G_base_id", type=str,
                        required=True,
                        help="G_base_id for the secondary G_s ∩ G_base evaluation")
    
    parser.add_argument("--ref_mapping_json", type=str, 
                        default="/path/to/ALHENA-paper/data/eval_scRNAseq_ref/ref_tissue_files.json",
                        help="Path to ref_tissue_files.json to do tissue_HistoST -> scRNAseq reference filename mapping.")
    
    parser.add_argument("--device", type=str,
                        default="auto",
                        help="Device, choices are auto, cuda, mps, cpu. "
                             "auto means cuda if available, otherwise mps if available, otherwise cpu.")
    
    args = parser.parse_args()
    main(args)
