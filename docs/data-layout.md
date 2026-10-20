# Data layout

Data, models, checkpoints and results are not stored in this repository. They live in a separate
external tree, whose location you choose. This page describes how paths into that tree are built,
the identifiers used in its folder names, and its organisation.

## How paths are built

Every path into the external tree is built by `get_full_path()` in
[`src/utils/constants.py`](../src/utils/constants.py):

```python
from src.utils.constants import get_full_path

get_full_path("train", "datasets", "ds_ct1_20x_224_o112")               # -> Path
get_full_path("train", "datasets", "ds_ct1_20x_224_o112", as_str=True)  # -> str
get_full_path("train", "training_datasets", scratch=True)                # cluster scratch root
```

The parts are joined onto one of three roots, declared in the same file. The shipped values are
placeholders: set them before running anything.

| Root | Used |
|---|---|
| `ROOT_PATH` | on any machine other than the Jean Zay cluster |
| `ROOT_PATH_JeanZay` | on the cluster (default) |
| `ROOT_PATH_JeanZay_scratch` | on the cluster, when `scratch=True` |

The cluster is detected automatically (`on_jeanzay()` in the same file), so the same relative path
resolves correctly on both machines. Parts must be relative; an absolute part raises a
`ValueError`.

On the cluster, large data that can be rebuilt is read from the scratch root: the training-dataset
folder (with the Novae teacher cache) during training, the ST evaluation dataset during embedding
extraction, the single-cell reference embeddings, and the whole-slide images of Tasks B and D
(`--wsi_on_scratch`). Off the cluster, `scratch=True` has no effect.

## Identifiers

These identifiers appear in folder names throughout the tree.

| Identifier | Example | Meaning |
|---|---|---|
| `slide_id` | `breast_s0` | one Xenium slide |
| `ct_cat_id` | `ct_1` | one set of cell-type categories |
| `dataset_id` | `ds_ct1_20x_224_o112` | a patch dataset: cell-type categories, H&E resolution, patch size and overlap |
| `training_dataset_id` | `ds_ct1_20x_224_o112_1` | a `dataset_id` with fixed train/validation/test splits and a gene selection |
| `training_id` | `training_46` | one training run, numbered separately for scConcept, Novae and ALHENA |
| `checkpoint_name` | `checkpoint_epoch_13.pth` | one saved training checkpoint |
| `eval_id` | `eval_1` | one frozen evaluation protocol, numbered per task |
| `space` | `he`, `he_l2`, `clip` | the embedding space an encoder is evaluated in |
| `cohort_id` | `human_cellbased_nobrain_scc40M30M_1` | the public slides used for Novae pretraining |
| `benchmark_id` | `tcga_benchmark_1` | one frozen Task B (TCGA) benchmark |
| `G_base_id` | `G_base_1` | a frozen gene panel |
| `gene_property_id` | `gene_property_1` | a frozen table of per-gene properties |
| `eval_signatures_id` | `eval_signatures_1` | a frozen panel of external gene signatures |
| `eval_signatures_INT_id` | `eval_signatures_INT_1` | a frozen panel of co-expression modules found in the training cells |
| `investigation_id` | `preclip_investigation_1` | one run of the teacher-geometry study |
| `geometry_recipe` | `native224`, `scale224` | how a PanNuke tile is presented to a 224-pixel encoder input |

### Encoder tags

An exported encoder, and every result derived from it, is named with the tag
`t{training_num}e{checkpoint_num}`:

```
HEencoder_t46e13.pt         H&E encoder from training_46, checkpoint_epoch_13.pth
STencoderGraph_t46e13.pt    trained projection head of the neighbourhood (Novae) branch
STencoderCell_t46e13.pt     trained projection head of the cell (scConcept) branch
```

A `checkpoint_num` of `-1` denotes a pretrained backbone without ALHENA training (`t-1e-1`,
`t-2e-1`, `t-3e-1`); only its H&E encoder is exported, since it has no trained projection head.
SEAL is packaged under the tag `t9001e0`. The tags used in the study are listed in
[evaluation.md](evaluation.md#encoders).

## The tree

```
<data root>/
├── STHELAR_HistoST/    source Xenium slides, aligned H&E, cell-type maps
├── train/              patch datasets, splits, pretrained models, ST teachers, training runs
└── eval/               evaluation datasets and results
```

### `STHELAR_HistoST/`

Written once per slide during preprocessing.

```
STHELAR_HistoST/
├── data_xenium10X/{slide_id}/                  raw Xenium output, H&E image, alignment matrix
├── sdata_processed/sdata_{slide_id}.zarr       SpatialData object, H&E aligned to ST
├── adata_annots_cells/adata_annots_cells_{slide_id}.h5ad   annotated cells
├── sdata_final/sdata_{slide_id}.zarr           sdata_processed plus the standardised cell type
├── adata_graph/{slide_id}.h5ad                 cells with their Delaunay graph, raw counts in .X
└── cID2cT/{ct_cat_id}/cID2cT_{slide_id}.parquet   cell id to cell type
```

### `train/`

```
train/
├── datasets/{dataset_id}/                      patch dataset
│   ├── images.zip                              H&E patches
│   ├── masks_cell_ids_nuclei.zip               nucleus masks labelled with cell ids
│   ├── cell_count.csv, types.csv, patch_metrics.csv, slide_ids.txt
│   ├── nucleus_centroids/                      nucleus centroids in H&E coordinates, per slide
│   ├── patch_bboxes.parquet                    patch bounding boxes in H&E coordinates
│   └── do_patch_eval/{run_number}/             optional CellViT patch metrics
│
├── training_datasets/{dataset_id}/{training_dataset_id}/
│   ├── cell_count_{split}.csv                  patches of each split
│   ├── cell_count_{split}_{n_min_cells}.csv    the same, after filtering by cell count
│   ├── informations/, informations_{n_min_cells}/   plots describing the splits
│   ├── slide_ids.txt, cell_ids_sets.npz        slides, and cell ids of each split
│   ├── genes_training_{training_dataset_id}.json    gene selection
│   ├── adatas/, cID2cT/, images.zip, ...       symbolic links to the files above
│   ├── .lmdb_cache/                            optional LMDB cache
│   ├── novae_teacher_cache/{novae_training_id}/     precomputed Novae teacher output, centring means
│   └── checking/                               optional connectivity and augmentation checks
│
├── pretraining_datasets/novae_public/{cohort_id}/   public slides for Novae pretraining
├── pretrained_foundation_models/               downloaded H&E and ST foundation models
├── preclip_investigation/{training_dataset_id}/{investigation_id}/   teacher-geometry study
│
└── trainings/
    ├── trainings_scConcept/{training_id}/      manifest.json, embeddings/{slide_id}.h5
    ├── trainings_Novae/{training_id}/          {training_id}_novae.yaml, checkpoints/, validation/, figures/
    └── trainings_HistoST/
        ├── {training_id}/
        │   ├── training/                       {training_id}_histost.yaml, log/, wandb/, training report
        │   └── trained_encoders/               HEencoder_*.pt, STencoderGraph_*.pt, STencoderCell_*.pt
        └── analysis/lora_drift/                adapter-drift tables, across runs
```

A training run reads its configuration from
`trainings_HistoST/{training_id}/training/{training_id}_histost.yaml`, which must exist before the
run starts. Its checkpoints are written to `training/log/checkpoints/`.

### `eval/`

Evaluation inputs are under `eval_datasets/` and are shared by every protocol that reads them.
Results are under one folder per task.

```
eval/
├── eval_datasets/
│   ├── st_eval_data/
│   │   ├── hest/                               HEST slides, downloaded once
│   │   └── eval_st_{training_dataset_id}/      Task A evaluation dataset
│   │       ├── metadata_eval_st_{training_dataset_id}.csv   slide list
│   │       ├── patches/, st/                   H&E patches and expression, per slide
│   │       ├── cell_token_membership_{modelgroup}/          patch and token of each cell
│   │       ├── novae_domains_HEpatches/{training_id}/       Novae domain of each patch
│   │       ├── he_embeddings/HEembs_t{n}e{c}/  H&E embeddings, per slide
│   │       ├── st_sthelar_embeddings_clip/     projected ST embeddings (GRAPHembs_*, CELLembs_*)
│   │       ├── embeddings_analysis/            Task A.0 outputs
│   │       ├── G_base/{G_base_id}/
│   │       └── gene_properties/{gene_property_id}/
│   ├── scRNAseq_references/                    single-cell references (adatas/) and their embeddings
│   ├── eval_signatures/{eval_signatures_id}/
│   ├── eval_signatures_INT/{eval_signatures_INT_id}/
│   ├── pannuke_eval_data/                      PanNuke download, characterisation, prepared trees
│   ├── TCGA_eval_data/                         Task B data (see below)
│   ├── thunder_eval_data/                      Task C datasets
│   └── pathobench_eval_data/                   Task D data (see below)
│
├── A-1_eval_st_predictions/{eval_id}/
├── A-2_eval_st_predictions/{eval_id}/          plus pannuke/ for the PanNuke evaluation
├── A-3_eval_celltype_predictions/{eval_id}/
├── B_eval_case/{eval_id}/
├── C_eval_thunder/                             no eval_id
└── D_eval_pathobench/{eval_id}/
```

A results folder has this shape:

```
{eval_id}/
├── settings/                         the frozen protocol: eval_config.yaml and its frozen inputs
├── HEencoder_t{n}e{c}_{space}/       results of one encoder in one space
└── COORDS_BASELINE/, SCCONCEPT_CEILING/   reference arms (Tasks A.1 and A.3)
```

Task A.2 also has a `shared/` folder (artifacts shared by all checkpoints) and evaluates the `clip`
space only. Task B also has `baselines/{baseline_id}/`. Task C results are directly under
`C_eval_thunder/HEencoder_t{n}e{c}_{space}/`. Task D stores its protocol in `manifest.json` and one
record per run in `arms/`.

Task B data:

```
TCGA_eval_data/
├── refs/hgnc/                        gene-symbol alias map
├── source_releases/{tcga_source_id}/     frozen snapshot of the TCGA metadata
├── processed_assets/                 RNA tables, HistoQC masks and tiles, shared across benchmarks
├── selected_slide_cache/             downloaded slides, shared across benchmarks
└── benchmarks/{benchmark_id}/        settings/, shared/ (cohort, tiles, targets, folds, encoder features), reports/
```

Task D data:

```
pathobench_eval_data/
├── slide_store/{cohort}/             wsi/ (slides) and manifests/
├── trident_features/{cohort}/        _tiling/ (shared by all encoders) and HEencoder_t{n}e{c}/{he,clip}/
├── pooled_embeddings/                per-case pooled features, rebuilt when missing
└── pathobench_assets/                PathoBench splits and labels
```

## Files shipped in the repository

| Path | Contents |
|---|---|
| `configs/template_train_histost_config.yaml` | template of a training configuration, with every key documented |
| `configs/template_novae_config.yaml` | template of a Novae pretraining configuration |
| `data/annots/` | cell-type annotation dictionaries per `ct_cat_id`, including the coarse mappings `coarse_ct_mapping_v1.json` and `coarse_ct_mapping_v2.json` |
| `data/eval_signatures/` | the external gene-signature panel (`signatures.gmt`) and its sources (`sources.json`) |
| `data/eval_scRNAseq_ref/ref_tissue_files.json` | the single-cell reference used for each tissue |
| `data/slide_status.json` | status of each slide |
| `data/data_trainings_recap.xlsx` | summary of the slides, datasets, training runs and evaluation runs |

A training configuration is merged onto the Python defaults in
`src/utils/default_histost_config.py`.
