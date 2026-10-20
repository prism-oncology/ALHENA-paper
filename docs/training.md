# Training

This guide covers everything before evaluation: preprocessing, the two spatial-transcriptomics (ST)
teachers, training, encoder export, and the training ablations reported in the paper. The
evaluation is described in [evaluation.md](evaluation.md), the environments in
[installation.md](installation.md) and the data tree in [data-layout.md](data-layout.md).

Commands are run from the repository root. Paths written `train/...` or `STHELAR_HistoST/...` are
relative to the data root. The examples use the identifiers of the paper's runs:

| | Identifier |
|---|---|
| Patch dataset | `ds_ct1_20x_224_o112` |
| Training dataset (splits and gene selection) | `ds_ct1_20x_224_o112_1` |
| scConcept teacher run | `training_7` |
| Novae teacher run | `training_4` |
| ALHENA run | `training_46`, checkpoint `checkpoint_epoch_13.pth` (encoder tag `t46e13`) |

## 1. Before you start

### 1.1 Data root and cluster scripts

Set the data root in `src/utils/constants.py` (see [data-layout.md](data-layout.md)).

The GPU steps were run on a SLURM cluster (Jean Zay) with the scripts in [`scripts/`](../scripts).
Before submitting one, edit its `#SBATCH --account` and `--mail-user` lines, the path to this
repository on the cluster (`/linkhome/rech/<home_group>/<login>/ST_HE/HistoST/`) and, if needed,
the run identifiers it passes. The scripts load the `miniforge/24.11.3` and `cuda/12.8.0` modules
and write their logs to `./output/`, relative to the folder they are submitted from.

### 1.2 Working on two machines

Each step reads and writes under the data root of the machine it runs on. When two consecutive
steps run on different machines, copy the outputs of the first to the second, keeping the same
relative paths. Three scripts do this for the main transfers (they use the SSH host alias
`jeanzay`):

- `src/preprocessing/_4_finalize/_3_send_data_to_JeanZay.sh`: the training dataset;
- `scripts/train/set_up_train_histost.sh`: training configurations and training runs;
- `src/evaluation/part1_ST/_1_build_dataset/_7_send_data_to_jeanzay.sh`: the ST evaluation dataset.

Symbolic links are not copied: recreate them on the cluster with the script that created them.

On the cluster, training and the scripts of sections 4.3 to 4.5 and 8.3 read the training-dataset
folder, including the Novae teacher cache, from the scratch root (`ROOT_PATH_JeanZay_scratch`).
Copy it there first, following the symbolic links:

```bash
rsync -avhL <ROOT_PATH_JeanZay>/train/training_datasets/ds_ct1_20x_224_o112/ds_ct1_20x_224_o112_1 \
    <ROOT_PATH_JeanZay_scratch>/train/training_datasets/ds_ct1_20x_224_o112/
```

## 2. Overview

```
Preprocessing (3)         slides, cell types, patches, splits, cell ids, ST graphs, gene selection
      |
scConcept (4.1)           one embedding per cell                     trainings_scConcept/training_7
      |
Novae (4.2)               pretrained with scConcept node features    trainings_Novae/training_4
      |
Teacher cache (4.3)       Novae output for every patch               novae_teacher_cache/training_4/
      |
Centring means (4.5)      mean embedding of each teacher             teacher_center_means.h5
      |
LMDB cache (5, optional)
      |
Training (6)              run_experiment.py --training_id training_46
      |
Export (7)                HEencoder_t46e13.pt, STencoderGraph_t46e13.pt, STencoderCell_t46e13.pt
```

Each step needs the outputs of the previous ones. The minimum number of cells per patch
(`n_min_cells`, 5 in the paper) must be the same in the patch filter (section 3.4), the centring
means (section 4.5), the training configuration (`data_select.n_min_cells`) and the control run
(section 8.3).

## 3. Preprocessing

*Environment: `process_slide`, local.* Notebooks and scripts are under `src/preprocessing/`.

### 3.1 Slides and cell annotations

The slides come from the STHELAR dataset. For each slide:

1. Download the Xenium section from the [10x Genomics datasets page](https://www.10xgenomics.com/datasets)
   (human, FFPE, Xenium) into `STHELAR_HistoST/data_xenium10X/{slide_id}/`.
2. Create the H&E-to-ST alignment matrix with
   [Xenium Explorer](https://www.10xgenomics.com/support/software/xenium-explorer/latest/tutorials/xe-image-alignment)
   and save it in the same folder.
3. Build the aligned SpatialData object `STHELAR_HistoST/sdata_processed/sdata_{slide_id}.zarr`:

   ```bash
   python3 src/preprocessing/_1_ct_ds/_1_process_slide.py \
       --slide_id {slide_id} \
       --st_dir <ST output folder> \
       --he_file <H&E .ome.tif file> \
       --align_matrix_path <relative path to the alignment CSV, or None>
   ```

4. Check that the H&E pixel size is 0.2125 µm (40x) with `_1_ct_ds/_2_check_resolution_HE.ipynb`.
5. Annotate the cells with a notebook in `_1_ct_ds/_3_annotations/` (example: `kidney_s2.ipynb`). It
   writes `STHELAR_HistoST/adata_annots_cells/adata_annots_cells_{slide_id}.h5ad`.
6. Build the final SpatialData object with a notebook in `_1_ct_ds/_4_sdata_final/` (example:
   `kidney_s2.ipynb`). It writes `STHELAR_HistoST/sdata_final/sdata_{slide_id}.zarr`.

Then, for all slides:

7. Harmonise the cell labels across slides with `_1_ct_ds/_5_add_final_cell_annots.ipynb` (column
   `final_label`).
8. Choose the cell-type categories with `_1_ct_ds/_6_ct_cat_id.ipynb`. A set of categories is a
   `ct_cat_id`; its dictionaries are written to `data/annots/annot_dicts_{ct_cat_id}/`.
9. Add the stain matrix of each slide to `M_MATRICES` in `src/dataset/hed_augmentation.py`, for the
   HED colour augmentation. It is estimated in QuPath (Analyze > Estimate stain vectors).

### 3.2 Patches, splits and cell ids

The commands below use the slides of the paper's patch dataset:

```bash
SLIDES=(breast_s0 breast_s1 breast_s3 breast_s6 lung_s1 lung_s3 skin_s1 skin_s2 skin_s3 skin_s4
        pancreatic_s0 pancreatic_s1 pancreatic_s2 heart_s0 colon_s1 colon_s2 kidney_s0 kidney_s1
        kidney_s2 liver_s0 liver_s1 tonsil_s0 tonsil_s1 lymph_node_s0 ovary_s0 ovary_s1 prostate_s0
        cervix_s0)
```

Extract the H&E patches into `train/datasets/ds_ct1_20x_224_o112/`. `--slide_tissues` follows the
order of `--slides`.

```bash
python3 src/preprocessing/_2_histo/_1_get_patches.py \
    --dataset_id ds_ct1_20x_224_o112 \
    --slides "${SLIDES[@]}" \
    --slide_tissues Breast Breast Breast Breast Lung Lung Skin Skin Skin Skin Pancreatic Pancreatic \
        Pancreatic Heart Colon Colon Kidney Kidney Kidney Liver Liver Tonsil Tonsil LymphNode Ovarian \
        Ovarian Prostate Cervix \
    --dict_dir data/annots/annot_dicts_ct_1 \
    --he_resolution 20 \
    --he_patch_width 224 \
    --he_patch_overlap 112 \
    --no_inst_type_maps
```

Write the cell-type map of each slide to `STHELAR_HistoST/cID2cT/ct_1/`, then inspect the patches
with `_2_histo/_3_check_data.ipynb`:

```bash
for slide_id in "${SLIDES[@]}"; do
    python3 src/preprocessing/_2_histo/_2_cID2cT.py \
        --cat_dict_dir data/annots/annot_dicts_ct_1 --slide_id "$slide_id"
done
```

**Optional: CellViT patch metrics.** These add segmentation-quality metrics to
`patch_metrics.csv`, which the split step can filter on. They apply only to datasets with 256-pixel
patches and were not used for the paper's 224-pixel dataset.

1. Prepare the inputs in `train/datasets/{dataset_id}/do_patch_eval/{run_number}/`:

   ```bash
   python3 src/preprocessing/_2_histo/_4_opt_setup_patch_eval.py \
       --dataset_id ds_ct1_20x_256_o112 --resolution 20 --slide_ids "${SLIDES[@]}" --run_number run1
   ```

2. Run CellViT with [CellViT_for_STHELAR](https://github.com/MICS-Lab/CellViT_for_STHELAR), after
   uncommenting `save the instance_map for each patch` in
   `inference/inference_cellvit_experiment_pannuke.py` and `type_map[type_map != 0] = 1` at the end
   of `load_maskfile` in `datasets/pannuke.py`. On the cluster, put a CellViT checkpoint
   (`CellViT-SAM-H-x20.pth` or `CellViT-SAM-H-x40.pth`, matching the H&E resolution) in
   `do_patch_eval/run1/checkpoints/`, link the dataset's `images.zip` into `do_patch_eval/run1/`,
   and submit `scripts/train/slurm_check_align_jeanzay.sh` (environment `cellvit_env`, one GPU; edit
   the path to your CellViT_for_STHELAR checkout).
3. Copy back `inference_instance_map_predictions.h5` and `inference_results.json`, add the metrics
   to `patch_metrics.csv`, and review them with `_2_histo/_6_opt_patch_eval_analyze.ipynb`:

   ```bash
   python3 src/preprocessing/_2_histo/_5_opt_patch_eval_results.py \
       --dataset_id ds_ct1_20x_256_o112 --run_id run1
   ```

**Splits.** This step defines the `training_dataset_id`. The validation and test slides are held out
whole; the other slides are split 77/9/14 into non-overlapping spatial regions, with cuts at 77%
and 86% of the longer axis. The three `*_selection` intervals filter on the CellViT metrics
(`0 1` keeps every patch).

```bash
python3 src/preprocessing/_2_histo/_7_make_folds.py \
    --dataset_id ds_ct1_20x_224_o112 \
    --training_dataset_id ds_ct1_20x_224_o112_1 \
    --dice_selection 0 1 --jaccard_selection 0 1 --bpq_selection 0 1 \
    --val_slide_ids breast_s3 \
    --test_slide_ids kidney_s0 pancreatic_s2 \
    --border_margin_px 112
```

The splits and their description are written to
`train/training_datasets/ds_ct1_20x_224_o112/ds_ct1_20x_224_o112_1/`. Review them with
`_2_histo/_8_analyze_training_datasets.ipynb`.

**Cell ids of each split.** The script reads the nucleus masks from the training-dataset folder, so
link them there first. It writes `cell_ids_sets.npz`, used by both teachers and by the centring
means.

```bash
ln -s /path/to/ALHENA_data/train/datasets/ds_ct1_20x_224_o112/masks_cell_ids_nuclei.zip \
      /path/to/ALHENA_data/train/training_datasets/ds_ct1_20x_224_o112/ds_ct1_20x_224_o112_1/
python3 src/preprocessing/_2_histo/_9_cell_ids_sets.py \
    --dataset_id ds_ct1_20x_224_o112 --training_dataset_id ds_ct1_20x_224_o112_1
```

### 3.3 ST graphs, geometry and gene selection

Build the Delaunay graph of each slide (`STHELAR_HistoST/adata_graph/{slide_id}.h5ad`):

```bash
for slide_id in "${SLIDES[@]}"; do
    python3 src/preprocessing/_3_st/_1_graph_st.py --slide_id "$slide_id"
done
```

Compute the nucleus centroids and the patch bounding boxes in H&E coordinates:

```bash
python3 src/preprocessing/_3_st/_2_nucleus_centroids.py --dataset_id ds_ct1_20x_224_o112 --resolution 20
python3 src/preprocessing/_3_st/_3_patch_coords.py --dataset_id ds_ct1_20x_224_o112
```

Select the genes. `--min_disp` can be chosen with `_3_st/overview_disp_norm_genes.ipynb`. The
selection is written to `genes_training_ds_ct1_20x_224_o112_1.json` in the training-dataset folder.

```bash
python3 src/preprocessing/_3_st/_4_gene_selection.py \
    --dataset_id ds_ct1_20x_224_o112 --training_dataset_id ds_ct1_20x_224_o112_1 \
    --min_disp -1.5 --min_smallslide_support 5 --top_high_expr 0
```

### 3.4 Finalise

Create the symbolic links inside the training-dataset folder (arguments: `dataset_id`,
`training_dataset_id`, `ct_cat_id`), then check that every cell id fits in `int32`:

```bash
./src/preprocessing/_4_finalize/_1_make_symb_links.sh ds_ct1_20x_224_o112 ds_ct1_20x_224_o112_1 ct_1
python3 src/preprocessing/_4_finalize/_2_check_int32_cID.py
```

Keep only the patches with at least 5 cells. This writes `cell_count_train_5.csv`,
`cell_count_valid_5.csv` and the plots in `informations_5/`:

```bash
python3 src/preprocessing/_5_final_choices/_1_filter_patches_by_min_cells.py \
    --dataset_id ds_ct1_20x_224_o112 --training_dataset_id ds_ct1_20x_224_o112_1 --n_min_cells 5
```

Download the pretrained models into `train/pretrained_foundation_models/`, one folder each:

| Model | Folder |
|---|---|
| UNI2-h, the backbone ALHENA adapts | `uni2h` |
| Virchow2 | `virchow2` |
| H-optimus-0, H-optimus-1 | `hoptimus0`, `hoptimus1` |
| SEAL (`seal_univ2_vision.pth`; gated on Hugging Face) | `seal` |
| scGPT human, scGPT pan-cancer | `scGPT_human`, `scGPT_pancancer` |
| scConcept version 1, version 2 | `scConcept_Corpus-30M`, `scConcept_corpus40M-model30M` |

For example, for SEAL (after requesting access and running `huggingface-cli login`) and for
scConcept version 2:

```bash
python3 - <<'PY'
from huggingface_hub import snapshot_download
root = "/path/to/ALHENA_data/train/pretrained_foundation_models"
snapshot_download(repo_id="MahmoodLab/SEAL", allow_patterns=["seal_univ2_vision.pth"],
                  local_dir=f"{root}/seal")
snapshot_download(repo_id="theislab/scConcept",
                  allow_patterns=["corpus40M-model30M/config.yaml", "corpus40M-model30M/model.ckpt",
                                  "corpus40M-model30M/gene_mappings/**", "corpus40M-model30M/panels/**",
                                  "corpus40M-model30M/pretrained_vocabulary/**"],
                  local_dir=f"{root}/scConcept_corpus40M-model30M")
PY
```

For scConcept version 1, download `Corpus-30M/config.yaml`, `Corpus-30M/model.ckpt`,
`Corpus-30M/pc_gene_token_mapping.pkl` and `Corpus-30M/panels/**` into `scConcept_Corpus-30M`.

Finally, copy the training dataset and this repository to the cluster, and recreate the symbolic
links there:

```bash
./src/preprocessing/_4_finalize/_3_send_data_to_JeanZay.sh ds_ct1_20x_224_o112 ds_ct1_20x_224_o112_1 ct_1
# then, on the cluster:
./src/preprocessing/_4_finalize/_1_make_symb_links.sh ds_ct1_20x_224_o112 ds_ct1_20x_224_o112_1 ct_1
```

## 4. The ST teachers

ALHENA is supervised by two frozen ST models: scConcept at the cell level and Novae at the
neighbourhood level. Novae uses the scConcept embeddings as node features, so scConcept comes first.

### 4.1 scConcept cell embeddings

*Environment: `sc_concept_env2`, cluster, one GPU.*

```bash
sbatch scripts/train/slurm_scConcept_v2_jeanzay.sh
```

The script runs `src/pretraining_ST/scConcept_pretraining_st/v2_scConcept_cellEmb.py` as scConcept
run `training_7`, with the pretrained model `corpus40M-model30M` and no adaptation
(`--adapt_steps 0`). It writes one embedding per cell of every slide to
`train/trainings/trainings_scConcept/training_7/embeddings/{slide_id}.h5`, and the model name and
version to `manifest.json`. The embeddings can be inspected with `v2_visualize_scConcept_cellEmb.ipynb`
in the same folder.

The repository also contains scConcept version 1 (`Corpus-30M`, environment `sc_concept_env`,
`slurm_scConcept_v1_jeanzay.sh`). Never mix embeddings from the two versions.

### 4.2 Novae

Novae is pretrained on the training cells of the project's slides and on public human MERSCOPE and
CosMx slides (brain excluded) from the `prism-oncology/novae` Hugging Face dataset.

1. Download the public slides, locally (environment `novae`). `--dry_run` only lists them.

   ```bash
   python3 src/pretraining_ST/novae_pretraining_st/_0_download_novae_public_cohort.py --dry_run
   python3 src/pretraining_ST/novae_pretraining_st/_0_download_novae_public_cohort.py \
       --cohort_id human_cellbased_nobrain_scc40M30M_1
   ```

   They are written to `train/pretraining_datasets/novae_public/human_cellbased_nobrain_scc40M30M_1/`.

2. Compute the scConcept embeddings of the public cells with scConcept run `training_7` (environment
   `sc_concept_env2`, cluster, one GPU). They are stored in `obsm["X_scConcept"]` of each public
   slide; the project's own cells reuse the embeddings of section 4.1.

   ```bash
   sbatch scripts/train/slurm_scConcept_embs_novae_public_jeanzay.sh
   ```

3. Create the run configuration from `configs/runs/training_4_novae.yaml`, the configuration of the
   teacher used for the paper, which holds the Novae hyperparameters:

   ```bash
   mkdir -p /path/to/ALHENA_data/train/trainings/trainings_Novae/training_4
   cp configs/runs/training_4_novae.yaml \
      /path/to/ALHENA_data/train/trainings/trainings_Novae/training_4/training_4_novae.yaml
   ```

4. Pretrain Novae (environment `novae_env`, cluster, one GPU):

   ```bash
   sbatch scripts/train/slurm_novae_jeanzay.sh
   ```

   The script runs `src/pretraining_ST/novae_pretraining_st/_2_novae_pretraining.py` with
   `--novae_training_id training_4 --own_scconcept_training_id training_7 --val_slide_id skin_s4
   --radius 100`; slide `skin_s4` is used for Novae validation. Results are written to
   `train/trainings/trainings_Novae/training_4/`. If the run stops after training (for example
   because it runs out of memory), rerun only the post-training step with `--post_only --ckpt
   <checkpoint file>`.

5. Check the teacher: read `validation/validation_report.md` (FIDE, JSD and MNE at 15, 20 and 25
   domains) and the domain maps `figures/domains_n{15,20,25}.png`.

### 4.3 Novae teacher cache

*Environment: `novae_env`, cluster, one GPU.*

Training does not run Novae: its output for every patch (`st_feat`, 128 dimensions) is computed
once and cached.

```bash
sbatch scripts/train/slurm_precompute_novae_teacher_jeanzay.sh
```

This runs:

```bash
python3 src/dataset/precompute_novae_teacher.py \
    --novae_training_id training_4 \
    --dataset_id ds_ct1_20x_224_o112 --training_dataset_id ds_ct1_20x_224_o112_1 \
    --folds train,valid,test --device cuda
```

The cache is written to `novae_teacher_cache/training_4/` in the training-dataset folder. Keep the
test fold: the evaluation uses the same cache. The training configuration then only needs
`model.st.novae_training_id: training_4`.

### 4.4 Teacher-geometry study (optional)

*Environment: `histost`. No GPU.*

This offline study of the two frozen teachers measures whether their embeddings form groups that
recur across slides, and the effect of subtracting their mean. A second script then suggests the
per-slide cap used when sampling training patches and the temperatures of the soft targets. The
paper uses `training.sampling_strategy: slide_headcap` and a temperature of 0.1 for both teachers.

```bash
python3 src/preclip_investigation/run_teacher_recurrence.py \
    --dataset_id ds_ct1_20x_224_o112 --training_dataset_id ds_ct1_20x_224_o112_1 \
    --novae_training_id training_4 --investigation_id preclip_investigation_1 --slide_probe

python3 src/preclip_investigation/select_training_params.py \
    --dataset_id ds_ct1_20x_224_o112 --training_dataset_id ds_ct1_20x_224_o112_1 \
    --novae_training_id training_4 --investigation_id preclip_investigation_1
```

Results, starting with `report.md`, are written to
`train/preclip_investigation/ds_ct1_20x_224_o112_1/preclip_investigation_1/`, and those of the second
script to its `param_selection/` subfolder.

### 4.5 Teacher-centring means

*Environment: `histost` (`histost_env` on the cluster). No GPU.*

Computes the mean embedding of each teacher over the training fold. Training subtracts it from the
teacher embeddings before building its targets (`model.clip.center_teacher_target: true`) and stops
if the file is missing.

```bash
python3 src/dataset/precompute_teacher_mean.py \
    --dataset_id ds_ct1_20x_224_o112 --training_dataset_id ds_ct1_20x_224_o112_1 \
    --novae_training_id training_4 --n_min_cells 5
```

`--n_min_cells` must equal `data_select.n_min_cells` in the training configuration. The means are
written to `teacher_center_means.h5` in the Novae teacher cache.

## 5. LMDB cache (optional)

*Environment: `histost`. No GPU. Long.*

The LMDB cache stores the H&E patches, nucleus masks, cell ids and nucleus centroids of the training
dataset, which speeds up data loading. It holds no teacher data, so it does not depend on the
teachers or on the training settings. Training uses it when `data.cached_dataset` is `"lmdb_init"`
(`"no"` disables it).

To build it:

1. Create the configuration
   `train/trainings/trainings_HistoST/compute_lmdb/training/compute_lmdb_histost.yaml` from the
   template, with `data_select.n_min_cells: 0` (required, so that the cache holds every patch) and
   `model.clip.center_teacher_target: false`.
2. Uncomment the line `raise Exception("Stop here after computing lmdb")` in
   `src/training/experiment.py`.
3. Run `python3 src/training/run_experiment.py --training_id compute_lmdb`. The cache is written to
   `.lmdb_cache/` in the training-dataset folder.
4. Comment the line again, and copy `.lmdb_cache/` to the same folder on the cluster.

## 6. Training

### 6.1 Optional checks

*Environment: `histost`.*

Check the ST graphs inside the patches, as the Novae teacher sees them (written to
`checking/connectivity_r100/` in the training-dataset folder):

```bash
python3 src/preprocessing/_5_final_choices/_2_check_connectivity_in_patches.py \
    --dataset_id ds_ct1_20x_224_o112 --training_dataset_id ds_ct1_20x_224_o112_1 \
    --edge_distance_threshold 100 --folds train,valid,test --iso_th 0.4 --deg_th 2.0
```

Render the H&E augmentations on a few patches (written to `checking/HE_augmentations/`):

```bash
python3 src/preprocessing/_5_final_choices/_3_check_HE_augmentations.py \
    --dataset_id ds_ct1_20x_224_o112 --training_dataset_id ds_ct1_20x_224_o112_1 \
    --num_patches 30 --max_per_png 5 --backbone uni2h \
    --aug_list rotate90,hflip,vflip,downscale,blur,gaussnoise,brightnesscontrast,superpixels,zoomblur,hed \
    --downscale_min 0.88 --downscale_max 0.97 --blur_limit 3 5 \
    --brightness_limit 0.12 --contrast_limit 0.12 --hed_sigma 0.2
```

### 6.2 Configuration

A run is configured by `train/trainings/trainings_HistoST/{training_id}/training/{training_id}_histost.yaml`,
which must exist before the run starts. Start from
[`configs/template_train_histost_config.yaml`](../configs/template_train_histost_config.yaml), where
every key is documented. The exact configuration of the released run is
[`configs/runs/training_46_histost.yaml`](../configs/runs/training_46_histost.yaml): copy it to the
path above to rerun it.

The keys that connect a run to the steps above:

| Key | Value |
|---|---|
| `data.dataset_id`, `data.training_dataset_id` | the dataset and splits to train on |
| `data.cached_dataset` | `"lmdb_init"` to use the LMDB cache (section 5), `"no"` otherwise |
| `data_select.n_min_cells` | the same value as in sections 3.4 and 4.5 |
| `data_select.num_train_samples` | number of training patches, from `src/preprocessing/_5_final_choices/_4_num_train_samples.ipynb`; the run stops at startup if it is off by more than 5% |
| `model.st.novae_training_id` | the Novae teacher cache (section 4.3) |
| `model.st.scconcept_training_id` | the scConcept embeddings (section 4.1) |
| `model.clip.center_teacher_target` | `true` requires the centring means (section 4.5) |
| `model.clip.target_temperature`, `model.clip.token_target_temperature` | soft-target temperatures of the Novae and scConcept targets |
| `training.sampling_strategy` | `slide_headcap` to cap the sampling weight of the largest slides; the template default is `random` |
| `training.lora_rank`, `training.lora_alpha` | rank and scaling of the low-rank adapters |
| `logging.training_id_histost` | must equal the `training_id` of the run folder |

### 6.3 Launch

*Environment: `histost_env`, cluster, one node with four GPUs.*

```bash
./scripts/train/set_up_train_histost.sh send_data training_46   # copy the configuration to the cluster
sbatch scripts/train/slurm_train_histost_jeanzay.sh              # on the cluster; set --training_id in the script
```

The script trains on one node with four GPUs (one process per GPU) and a batch size of 128 per GPU,
512 in total. A checkpoint is saved at every epoch.

To resume, `--checkpoint_name checkpoint_epoch_{e}.pth` continues the same run with the
configuration stored in the checkpoint, while `--resume_state_from_checkpoint <checkpoint path>`
starts a new `training_id` from the state of another run, with the new run's configuration.

### 6.4 Results

```bash
./scripts/train/set_up_train_histost.sh get_data exclude training_46    # "noexclude" also copies the checkpoints
./scripts/train/set_up_train_histost.sh sync_wandb training_46 group1   # upload the offline W&B run
python3 src/training/training_report.py --training_id training_46
```

Logs, checkpoints and the W&B run are in `train/trainings/trainings_HistoST/training_46/training/`.
The report, `report_training_46.html` in the same folder, shows the losses and validation metrics
per epoch. `--standalone` embeds its figures in a single file, and `--generate_md` also writes a
Markdown version to `access_reports/` in the repository.

The released encoder is `training_46` at `checkpoint_epoch_13.pth`.

## 7. Export the encoders

*Environment: `histost`. No GPU.*

The evaluations use exported encoder files, not training checkpoints:

```bash
python3 src/evaluation/extract_encoders.py --training_id training_46 --checkpoint_name checkpoint_epoch_13.pth
```

The files are written to `train/trainings/trainings_HistoST/training_46/trained_encoders/`:

| File | Contents |
|---|---|
| `HEencoder_t46e13.pt` | the H&E encoder: backbone with its adapters, and the H&E projection heads |
| `STencoderGraph_t46e13.pt` | the projection head applied to the Novae teacher features, and the Novae run it expects |
| `STencoderCell_t46e13.pt` | the projection head applied to the scConcept cell embeddings |

The H&E encoder expects tiles resized, cropped and normalised as during validation (the
preprocessing parameters are stored in the file) and does not L2-normalise its outputs.

**Pretrained backbones without ALHENA training** are exported the same way, with checkpoint number
`-1`: `training_-1` for UNI2-h, `training_-2` for Virchow2 and `training_-3` for H-optimus-1 (tags
`t-1e-1`, `t-2e-1`, `t-3e-1`).

1. Create the run configuration with the backbone in `model.he.backbone_img_encoder`,
   `model.he.use_mlp_head: False`, `training.HE_training_option: freeze`,
   `model.pretrained_histost: null` and `advanced_scheduling.enabled: False`.
2. Uncomment the block "Uncomment to save the pretrained model without training" in
   `src/training/trainer.py`, run `python3 src/training/run_experiment.py --training_id training_-1`
   (it saves `checkpoint_epoch_-1.pth` and stops), then comment the block again.
3. Export with `python3 src/evaluation/extract_encoders.py --training_id training_-1
   --checkpoint_name checkpoint_epoch_-1.pth`. Only `HEencoder_t-1e-1.pt` is written, since an
   untrained backbone has no projection head.

## 8. Training ablations

### 8.1 Adapter rank

These runs differ from the released run only in the rank and scaling of the adapters. They all use
20 epochs and save a checkpoint at every epoch.

| `training_id` | `lora_rank` | `lora_alpha` |
|---|---|---|
| `training_41` | 4 | 4 |
| `training_40` | 8 | 8 |
| `training_42` | 16 | 16 |
| `training_45` | 8 | 4 |
| `training_44` | 16 | 8 |
| `training_43` | 32 | 16 |
| `training_46` (released) | 64 | 32 |

The three runs with `lora_alpha = lora_rank` are generated from the template by
`gen_sweep_configs.py` and launched as one SLURM array (keep the `TIDS` list and the `--array` range
of the array script consistent with the generator's `RUNS`):

```bash
python3 scripts/train/gen_sweep_configs.py     # --dry_run to preview
for tid in training_40 training_41 training_42; do ./scripts/train/set_up_train_histost.sh send_data "$tid"; done
sbatch scripts/train/slurm_sweep_histost_jeanzay.sh
```

For the four other runs, copy a generated configuration, change `training.lora_rank`,
`training.lora_alpha` and the `logging` fields, and launch each run with
`slurm_train_histost_jeanzay.sh` (section 6.3).

### 8.2 Adapter drift

*Environment: `histost` or `histost_env`. No GPU.*

`lora_drift.py` measures how much the adapters change the backbone weights, from the saved
checkpoints: the norms and singular values of the low-rank update of each adapted matrix, and its
intruder dimensions ([Shuttleworth et al., 2024](https://arxiv.org/abs/2410.21228)).

```bash
python3 src/training/lora_drift.py \
    --training_ids training_41 training_40 training_42 training_45 training_44 training_43 training_46 \
    --last_only --markdown --strict_low_memory
```

The tables (`lora_drift_*.csv`, and a Markdown summary with `--markdown`) are written to
`train/trainings/trainings_HistoST/analysis/lora_drift/`. Replace `--last_only` by
`--epochs 9 13 19` to measure several epochs. On the cluster, use
`scripts/train/slurm_lora_drift_jeanzay.sh`.

### 8.3 Shuffled-pairing control and loss ablations

Three runs start from the configuration of `training_46` and change only the keys below (and the
`logging` fields). Their encoders are exported at epoch 13, and their configurations are in
[`configs/runs/`](../configs/runs/).

| Run | Changed keys |
|---|---|
| `training_50`: H&E and ST pairing shuffled | `model.st.novae_training_id: training_4_SHUFFLED_seed1234`<br>`model.st.scconcept_training_id: training_7_SHUFFLED_seed1234` |
| `training_51`: no cell-level loss | `model.clip.lambda_token: 0.0` |
| `training_52`: no patch-level loss | `model.clip.lambda_patch: 0.0`<br>`model.clip.lambda_token_warmup_epochs: 0` |

**Shuffled teacher caches for the control run** (environment `histost`, no GPU). The script
shuffles the teacher embeddings of the training rows across slides, with a fixed seed; the
validation rows are left unchanged. `--dry_run` shows the plan without writing anything.

```bash
python3 src/dataset/permute_teacher_caches.py \
    --dataset_id ds_ct1_20x_224_o112 --training_dataset_id ds_ct1_20x_224_o112_1 \
    --novae_training_id training_4 --n_min_cells 5 --seed 1234
```

This writes `novae_teacher_cache/training_4_SHUFFLED_seed1234/` and
`train/trainings/trainings_scConcept/training_7_SHUFFLED_seed1234/`. Then compute the centring
means of the shuffled cache, and check that they are identical to those of the original cache
(shuffling does not change the set of training embeddings; the check exits with an error if they
differ):

```bash
python3 src/dataset/precompute_teacher_mean.py \
    --dataset_id ds_ct1_20x_224_o112 --training_dataset_id ds_ct1_20x_224_o112_1 \
    --novae_training_id training_4_SHUFFLED_seed1234 --n_min_cells 5

python3 src/dataset/permute_teacher_caches.py --mode verify_means \
    --dataset_id ds_ct1_20x_224_o112 --training_dataset_id ds_ct1_20x_224_o112_1 \
    --novae_training_id training_4 --n_min_cells 5 --seed 1234
```

**Training and export.** Launch the three runs as in section 6.3, then export them:

```bash
for tid in training_50 training_51 training_52; do
    python3 src/evaluation/extract_encoders.py --training_id "$tid" --checkpoint_name checkpoint_epoch_13.pth
done
```

This gives the encoders `t50e13`, `t51e13` and `t52e13`. `training_51` has no cell-level projection
head, so its `STencoderCell` file cannot be used by Task A.2.
