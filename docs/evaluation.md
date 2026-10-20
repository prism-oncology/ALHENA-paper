# Evaluation

ALHENA is evaluated by seven tasks:

| Task | Question | Unit | Results under `eval/` |
|---|---|---|---|
| **A.0** | What is the geometry of the embedding space? (descriptive) | cell, tile | `eval_datasets/st_eval_data/eval_st_{training_dataset_id}/embeddings_analysis/` |
| **A.1** | Can gene and gene-signature expression be predicted from the image? | tile | `A-1_eval_st_predictions/` |
| **A.2** | Can cells be typed against a single-cell reference, with nothing fitted? | cell | `A-2_eval_st_predictions/` |
| **A.3** | Can a linear probe on the frozen backbone predict the cell type? | cell | `A-3_eval_celltype_predictions/` |
| **B** | Can case-level transcriptomic signatures be predicted on TCGA? | case | `B_eval_case/` |
| **C** | Is general pathology performance preserved? (Thunder benchmark) | tile | `C_eval_thunder/` |
| **D** | How does it perform on slide-level CPTAC tasks? (PathoBench) | slide, case | `D_eval_pathobench/` |

The paper indexes its ten evaluations as E1 to E10 (Table 1). They map onto the tasks as follows:

| Paper | Evaluation | Task | Protocol |
|---|---|---|---|
| E1 | Transcriptomic signatures, TCGA | B | `eval_1`, `mean_std` pooling |
| E2 | Clinical and molecular labels, CPTAC | D | `eval_1`, linear probe |
| E3 | General pathology performance, Thunder | C | no `eval_id` |
| E4 | Local gene abundance | A.1 | `eval_1` (test stripes); `eval_3` and `eval_4` (fully held-out sections, transferred and refitted) |
| E5 | Gene-level transfer to HEST-1k | A.1 | `eval_5` |
| E6 | Tile-to-cell alignment | A.0 | no `eval_id` |
| E7 | Linear cell-type separability | A.3 | `eval_1` |
| E8 | Atlas-based typing, five cell types | A.2 | `eval_1` (`v1` classes) |
| E9 | Atlas-based typing, three broad classes | A.2 | `eval_2` (`v2` classes) |
| E10 | Transfer to PanNuke | A.2 | PanNuke scripts, with the `eval_2` settings |

The paper's permuted-target control, neighbourhood-only and cell-only models are `training_50`,
`training_51` and `training_52` ([training.md](training.md#83-shuffled-pairing-control-and-loss-ablations)).

Commands are run from the repository root. Encoders must be exported first (see
[training.md](training.md#7-export-the-encoders)).

## Concepts

### Frozen protocols (`eval_id`)

An `eval_id` fixes everything a task measures except the encoder: the cohort, the splits, the
targets or cell-type classes, the probe and its hyperparameter grid, and the metrics. It is written
by the task's setup step and then only read, so that every encoder is evaluated under the same
protocol. The encoder and the embedding space are not part of the `eval_id`: evaluating another
encoder adds a result folder to the same `eval_id`. Tasks A.0 and C have no `eval_id`.

The setup commands below regenerate the protocols used in the paper.

### Encoders

| Tag | Encoder |
|---|---|
| `t46e13` | ALHENA (`training_46`, `checkpoint_epoch_13.pth`) |
| `t-1e-1` | UNI2-h, the backbone ALHENA adapts |
| `t-2e-1` | Virchow2 |
| `t-3e-1` | H-optimus-1 |
| `t9001e0` | SEAL, another adaptation of UNI2-h with spatial transcriptomics |
| `t50e13`, `t51e13`, `t52e13` | the shuffled-pairing control and the two loss ablations (Task A.3) |

Scripts take an encoder as `--training_num` and `--checkpoint_num`, for example
`--training_num 46 --checkpoint_num 13`. To evaluate the released model without the training
checkpoint, download it from [Hugging Face](https://huggingface.co/prism-oncology/ALHENA) and run
`python3 src/evaluation/rebuild_released_encoders.py --release_dir <download folder>` (environment
`histost`), which writes `HEencoder_t46e13.pt` and `STencoderCell_t46e13.pt` with your copy of
UNI2-h. Its Novae-side head is not released, so run step 12 below with `--do_cell`.

The embedding **spaces** are `he` (the backbone output), `he_l2` (the same, L2-normalised) and
`clip` (after the projection head; only for encoders trained with ALHENA).

Two **reference arms** are evaluated under the same protocol as the encoders:
`COORDS_BASELINE` (Tasks A.1 and A.3), the same probe using only the nucleus coordinates, and
`SCCONCEPT_CEILING` (Task A.3), the same probe using the scConcept embedding of each cell.

### The SEAL encoder

SEAL is packaged as an ordinary encoder file by merging its adapter weights (downloaded in
[training.md](training.md#34-finalise)) into UNI2-h.

```bash
conda activate histost
python3 src/evaluation/seal_comparison/build_seal_he_encoder.py --dry_run   # build and check, write nothing
python3 src/evaluation/seal_comparison/build_seal_he_encoder.py
```

This writes `train/trainings/trainings_HistoST/training_9001/trained_encoders/HEencoder_t9001e0.pt`.
SEAL was trained on HEST data, so in Task A it is evaluated on the STHELAR slides only.

## Environments

| Step | Environment | Machine |
|---|---|---|
| Task A dataset: download and setup (steps 1 to 9) | `process_slide` | local |
| Task A dataset: Novae domains (step 3) | `novae` | local |
| H&E embeddings (Task A, PanNuke) | `histost_env` | cluster |
| scConcept embeddings of the single-cell references | `sc_concept_env2` | cluster |
| Task A dataset: steps 11 to 16; Tasks A.0, A.1, A.3 | `histost` | local |
| Task A.2, PanNuke download and retrieval | `histost_eval_a2` | local |
| PanNuke patch preparation | `process_slide` | local |
| Task B, except HistoQC and feature extraction | `histost` | local |
| Task B HistoQC | `histoqc` | local |
| Task B feature extraction | `histost_env` | cluster |
| Task C dataset download | `thunder` | local |
| Task C benchmark | `thunder_env` | cluster |
| Task D slide download | `pathobench` | local |
| Task D tiling and features | `trident_env` | cluster |
| Task D probes | `pathobench_env` | cluster |

## The Task A evaluation dataset

Tasks A.0 to A.3 use one evaluation dataset per training dataset, built once. It combines the
held-out slides and regions of the project's Xenium cohort (STHELAR) with external slides from the
HEST collection. For another encoder, only steps 10 to 12 are needed.

1. **Choose the evaluation slides** with
   `src/evaluation/part1_ST/_1_build_dataset/_0_choose_data/ds_ct1_20x_224_o112_1.ipynb`. It writes
   the slide list `metadata_eval_st_ds_ct1_20x_224_o112_1.csv`. Its `source` and `subset` columns
   identify the STHELAR slides (`STHELAR_train`: held-out regions of training slides;
   `STHELAR_test`: test slides) and the HEST subsets (`other_technologies`, `other_preservations`,
   `other_magnifications`, `other_species`, `other_organs`).

2. **Download HEST and set up the STHELAR slides.** For each slide, this writes the H&E patches
   (`patches/{slide_id}.h5`), the expression (`st/{slide_id}.h5ad`) and, for each group of encoders
   sharing the same preprocessing, the patch and token of each cell (`cell_token_membership_*`).

   ```bash
   python3 src/evaluation/part1_ST/_1_build_dataset/_1_download_eval_data.py \
       --dataset_id ds_ct1_20x_224_o112 --training_dataset_id ds_ct1_20x_224_o112_1 --he_resolution 20
   ```

3. **Novae domain of each patch** (environment `novae`), written to
   `novae_domains_HEpatches/training_4/`:

   ```bash
   python3 src/evaluation/part1_ST/_1_build_dataset/_2_novae_domains_HEpatches.py \
       --dataset_id ds_ct1_20x_224_o112 --training_dataset_id ds_ct1_20x_224_o112_1 \
       --training_id training_4 --ckpt last.ckpt
   ```

4. **Check the gene names of the HEST slides** (malformed files are corrected):

   ```bash
   python3 src/evaluation/part1_ST/_1_build_dataset/_3_format_hest.py --training_dataset_id ds_ct1_20x_224_o112_1
   ```

5. **Map the genes of the mouse HEST slides to human genes:**

   ```bash
   python3 src/evaluation/part1_ST/_1_build_dataset/_4_mouse_human_mapping.py \
       --training_dataset_id ds_ct1_20x_224_o112_1 --allow_live_rest
   ```

6. **Link the HEST slides into the dataset** (HEST is downloaded once, in `st_eval_data/hest/`):

   ```bash
   ./src/evaluation/part1_ST/_1_build_dataset/_5_symbolic_links_hest.sh ds_ct1_20x_224_o112_1
   ```

7. **Inspect the dataset** with `_1_build_dataset/_6_check_eval_data.ipynb`, and check the format of
   the expression data (report in `st/check_eval_data.txt`):

   ```bash
   python3 src/evaluation/part1_ST/_1_build_dataset/_6_check_eval_data_2.py --training_dataset_id ds_ct1_20x_224_o112_1
   ```

8. **Genes shared across slides:** see `_1_build_dataset/_0_choose_data/commun_genes.ipynb`.

9. **Copy the dataset to the cluster,** recreate the HEST links there, and copy the dataset to the
   scratch root, where the H&E embedding script reads it:

   ```bash
   ./src/evaluation/part1_ST/_1_build_dataset/_7_send_data_to_jeanzay.sh ds_ct1_20x_224_o112_1
   # then, on the cluster:
   ./src/evaluation/part1_ST/_1_build_dataset/_5_symbolic_links_hest.sh ds_ct1_20x_224_o112_1 restrict
   rsync -avhL <ROOT_PATH_JeanZay>/eval/eval_datasets/st_eval_data/eval_st_ds_ct1_20x_224_o112_1 \
       <ROOT_PATH_JeanZay_scratch>/eval/eval_datasets/st_eval_data/
   ```

10. **H&E embeddings** (environment `histost_env`, cluster, one GPU):

    ```bash
    sbatch scripts/eval/slurm_e1_HEembs_jeanzay.sh
    ```

    The script runs `_1_build_dataset/_8_compute_HEembs.py` for `t46e13`; edit `--training_num` and
    `--checkpoint_num` for another encoder. For each slide, it writes the patch embeddings and the
    embeddings of each cell in each patch, with and without the projection head, to
    `he_embeddings/HEembs_t{n}e{c}/{slide_id}.h5`. `--sets` restricts the slides; for SEAL, use
    `--sets STHELAR_train STHELAR_test`. Copy `he_embeddings/HEembs_t{n}e{c}/` back to the local
    machine.

11. **One H&E embedding per cell,** taken from the most relevant patch when a cell appears in
    several (environment `histost`):

    ```bash
    python3 src/evaluation/part1_ST/_1_build_dataset/_9_main_HEemb_cellID.py \
        --dataset_id ds_ct1_20x_224_o112 --training_dataset_id ds_ct1_20x_224_o112_1 \
        --training_num 46 --checkpoint_num 13 --overwrite
    ```

12. **Projected ST embeddings** of the STHELAR slides, for encoders trained with ALHENA
    (environment `histost`, or `scripts/eval/slurm_e1_STembs_jeanzay.sh` on the cluster):

    ```bash
    python3 src/evaluation/part1_ST/_1_build_dataset/_10_compute_STembs.py \
        --dataset_id ds_ct1_20x_224_o112 --training_dataset_id ds_ct1_20x_224_o112_1 \
        --training_id training_46 --checkpoint_num 13 \
        --graph_batch_size 256 --cell_batch_size 8192
    ```

    The projection heads are applied to the precomputed teacher embeddings (the Novae teacher cache
    must include the test fold). The results are written to `st_sthelar_embeddings_clip/GRAPHembs_t46e13/`
    and `CELLembs_t46e13/`.

Steps 13 to 16 build the gene panels (environment `histost`). They use training cells only.

13. **Gene panel** `G_base_1`:

    ```bash
    python3 src/evaluation/part1_ST/_1_build_dataset/_11_Gbase.py \
        --dataset_id ds_ct1_20x_224_o112 --training_dataset_id ds_ct1_20x_224_o112_1 \
        --G_base_id G_base_1 --scGPT_model_name scGPT_human --k_selected 256 \
        --min_shared_slides 5 --min_nonzero_rate 0.05 --min_support_slides 5 --hvg_top_k 100 \
        --dispersion_q_low 0.10 --dispersion_q_high 0.90 \
        --w_shared_slides 2.0 --w_hvg 2.0 --w_nonzero 1.0 --w_dispersion 1.0
    ```

14. **Gene properties** (cell-type specificity and spatial autocorrelation of each panel gene):

    ```bash
    python3 src/evaluation/part1_ST/_1_build_dataset/_12_gene_properties.py \
        --dataset_id ds_ct1_20x_224_o112 --training_dataset_id ds_ct1_20x_224_o112_1 \
        --G_base_id G_base_1 --gene_property_id gene_property_1 --ct_cat_id ct_1 \
        --min_cells_per_cell_type 50 --min_train_slides 3 --min_train_cells_per_slide_moran 200
    ```

15. **External gene signatures,** from `data/eval_signatures/` (MSigDB Hallmark and curated immune,
    niche and stromal sets), written to `eval/eval_datasets/eval_signatures/eval_signatures_1/`:

    ```bash
    python3 src/evaluation/part1_ST/_1_build_dataset/_13_eval_signatures.py --eval_signatures_id eval_signatures_1
    ```

16. **Internal gene signatures:** co-expression modules found among the panel genes in the training
    cells, written to `eval/eval_datasets/eval_signatures_INT/eval_signatures_INT_1/`:

    ```bash
    python3 src/evaluation/part1_ST/_1_build_dataset/_14_eval_signatures_INT.py \
        --dataset_id ds_ct1_20x_224_o112 --training_dataset_id ds_ct1_20x_224_o112_1 \
        --G_base_id G_base_1 --eval_signatures_INT_id eval_signatures_INT_1 \
        --n_modules 20 --min_module_size 5 --weight_scheme eigengene_kme --cell_cap 200000 --seed 42
    ```

## Task A.0: embedding geometry

*Environment: `histost`.* Three descriptive analyses, run for one encoder at a time. Their outputs
are written to `embeddings_analysis/` in the Task A evaluation dataset.

**Cell-view consistency.** A cell can appear in several overlapping patches: this compares its
embeddings across patches, with random-pair and other-cell baselines, and describes the anisotropy
and effective rank of the space. Output: `cell_view_consistency_HEembs/HEembs_t46e13/`.

```bash
python3 src/evaluation/part1_ST/_2_analyze_embs/_1_cell_view_consistency_HEembs.py \
    --training_dataset_id ds_ct1_20x_224_o112_1 --training_num 46 --checkpoint_num 13 \
    --spaces he he_clip --ct_cat_id ct_1 --celltype_axes fine coarse --coarse_mapping_version v1 \
    --min_views 2 --min_cells_per_group 30 --min_rows_for_corr 100 --seed 42 \
    --n_random_pairs_per_slide 100000 --anisotropy_sample 200000 --n_pairs_per_type_pair 5000
```

**Patch-versus-cell similarity.** Compares each patch embedding with the embeddings of the cells it
contains, with shuffled baselines, and maps the scores onto the patch tokens. Only `--space he` is
supported. Output: `similarity_heatmaps_HEembs/HEembs_t46e13_he/`. `--resume` skips the slides
already done.

```bash
python3 src/evaluation/part1_ST/_2_analyze_embs/_2_similarity_heatmaps_HEembs.py \
    --training_dataset_id ds_ct1_20x_224_o112_1 --training_num 46 --checkpoint_num 13 \
    --output_level both --space he --similarity cosine --tau_mode scale_free --tau_scale 1.0 \
    --max_median_entropy_norm 0.99 --ct_cat_id ct_1 --celltype_axes fine coarse \
    --coarse_mapping_version v1 --topk 10 --seed 42 --n_shuffled_pairs_per_group 4 \
    --max_in_memory_gb 6.0 --headline_min_cells 5
```

**UMAPs and structure metrics.** UMAP figures, and nearest-neighbour label purity (by slide, tissue,
modality, source and cell type) computed on the embeddings themselves. `--training_id_novae` must be
the Novae teacher used in training. Output: `umap_embeddings/t46e13/`.

```bash
python3 src/evaluation/part1_ST/_2_analyze_embs/_3_embs_umaps.py \
    --training_dataset_id ds_ct1_20x_224_o112_1 --training_num 46 --checkpoint_num 13 \
    --training_id_novae training_4 --Np 1500 --Nc 3000 --pca_dim 50 --umap_metric cosine \
    --umap_n_neighbors 30 --umap_min_dist 0.15 --umap_spread 1.0 --ct_cat_id ct_1 \
    --celltype_axes fine coarse --coarse_mapping_version v1 --metrics_knn_k 15 \
    --min_points_per_label 30 --no_keep_intermediate_parquet
```

## Task A.1: gene and signature prediction

*Environment: `histost`.* A ridge probe, fitted in closed form on the frozen H&E embedding of each
patch, predicts the expression of each panel gene in the patch, or the score of each gene
signature. The regularisation is selected by cross-validation over training slides. Each run is
deterministic and runs on CPU.

1. **Create the protocol** (the values for all thirteen `eval_id`s are in the table below):

   ```bash
   python3 src/evaluation/part1_ST/task_A1_gene_predict/scripts/_1_setup.py \
       --eval_id eval_1 \
       --dataset_id ds_ct1_20x_224_o112 --training_dataset_id ds_ct1_20x_224_o112_1 \
       --eval_configuration STHELAR_TRAIN --sub_eval_configuration STHELAR_TRAIN_CV \
       --level patch_all \
       --G_base_id G_base_1 --gene_property_id gene_property_1 \
       --eval_signatures_id eval_signatures_1 \
       --target_basis genes \
       --modelgroup hoptimus0_hoptimus1_uni2h_virchow2 \
       --novae_training_id training_4 --novae_domain_n 15
   ```

   It writes `A-1_eval_st_predictions/eval_1/settings/eval_config.yaml`, which may be edited
   before step 2.

2. **Freeze the protocol:**

   ```bash
   python3 src/evaluation/part1_ST/task_A1_gene_predict/scripts/_2_fill_eval_settings.py --eval_id eval_1
   ```

3. **Run the probe** for each encoder and space:

   ```bash
   python3 src/evaluation/part1_ST/task_A1_gene_predict/scripts/_3_run_eval.py \
       --eval_id eval_1 --training_num 46 --checkpoint_num 13 --space he --skip_reports --skip_predictions
   ```

   Results are written to `A-1_eval_st_predictions/eval_1/HEencoder_t46e13_he/`.
   `--skip_reports` and `--skip_predictions` skip optional figures and prediction files; they do not
   change the metrics. `scripts/eval/sync_wandb_eval_A1.py --eval_id eval_1` uploads the offline
   W&B runs.

**The thirteen protocols.** Every `eval_id` uses `--dataset_id ds_ct1_20x_224_o112
--training_dataset_id ds_ct1_20x_224_o112_1 --level patch_all --gene_property_id gene_property_1
--modelgroup hoptimus0_hoptimus1_uni2h_virchow2 --eval_signatures_id eval_signatures_1`. In
addition:

| `eval_id` | `--eval_configuration` | `--sub_eval_configuration` | `--G_base_id` | `--target_basis` |
|---|---|---|---|---|
| `eval_1` | `STHELAR_TRAIN` | `STHELAR_TRAIN_CV` | `G_base_1` | `genes` |
| `eval_2` | `STHELAR_TRAIN` | `STHELAR_TRAIN_COORDS` | `G_base_1` | (from `eval_1`) |
| `eval_3` | `STHELAR_TEST` | `STHELAR_TEST_ONE_SHOT` | `G_base_1` | `genes` |
| `eval_4` | `STHELAR_TEST` | `STHELAR_TEST_ADAPTED` | `G_base_1` | `genes` |
| `eval_5` | `HEST` | `HEST_CV_CLOSE` | `auto` | `genes` |
| `eval_6` | `HEST` | `HEST_CV_MAGNIFICATION` | `auto` | `genes` |
| `eval_7` | `HEST` | `HEST_CV_SPECIES` | `auto` | `genes` |
| `eval_8` | `STHELAR_TRAIN` | `STHELAR_TRAIN_CV` | `G_base_1` | `signatures` |
| `eval_9` | `STHELAR_TEST` | `STHELAR_TEST_ONE_SHOT` | `G_base_1` | `signatures` |
| `eval_10` | `STHELAR_TEST` | `STHELAR_TEST_ADAPTED` | `G_base_1` | `signatures` |
| `eval_11` | `HEST` | `HEST_CV_CLOSE` | `auto` | `signatures` |
| `eval_12` | `HEST` | `HEST_CV_MAGNIFICATION` | `auto` | `signatures` |
| `eval_13` | `HEST` | `HEST_CV_SPECIES` | `auto` | `signatures` |

- The STHELAR protocols, except `eval_2`, also pass `--novae_training_id training_4 --novae_domain_n 15`.
- The signature protocols also pass `--signature_sets internal external --eval_signatures_INT_id eval_signatures_INT_1`.
- `eval_2` is the coordinate control: it passes `--reference_eval_id eval_1`, takes no encoder and
  no `--space` at step 3, and writes `COORDS_BASELINE/`.
- Run steps 1 and 2 for each `eval_id` in numerical order, since `eval_2` reads the frozen
  `eval_1`.
- SEAL is evaluated on the STHELAR protocols only (`eval_1`, `3`, `4`, `8`, `9`, `10`).

## Cell-type classes of Tasks A.2 and A.3

Tasks A.2 and A.3 score coarse cell types, set by `--label_granularity coarse` and
`--coarse_mapping_version`, which maps the cell types through
`data/annots/annot_dicts_ct_1/coarse_ct_mapping_{version}.json`:

| Version | Classes |
|---|---|
| `v1` | `T_NK`, `B_Plasma`, `Myeloid`, `Stromal_Vascular`, `Epithelial_Melanocyte` |
| `v2` | `Immune`, `Stromal_Vascular`, `Epithelial_Melanocyte` |

Both versions map `Specialized` to `Unknown`, and `Unknown` cells are not scored. Each version is a
separate `eval_id`.

## Task A.2: cell typing against a single-cell reference

*Environment: `histost_eval_a2`.* Task A.2 runs on the STHELAR slides, which have paired cell
types. The projected H&E embedding of each cell is used as a query against a single-cell RNA
reference of the same tissue, projected into the same space; the cell type is transferred from the
nearest reference cells and compared with the cell type from the ST data. The retrieval
hyperparameters are tuned on the training slides. Task A.2 needs the encoder's cell projection head,
so it applies only to encoders trained with ALHENA and the cell-level loss.

**Single-cell references.**

1. Build one reference per tissue with `src/evaluation/part1_ST/scRNAseq_ref/_1_scRNAseq_ref.ipynb`.
   It writes `eval/eval_datasets/scRNAseq_references/adatas/{tissue}.h5ad`. The reference used for
   each tissue is listed in `data/eval_scRNAseq_ref/ref_tissue_files.json`.
2. Compute their scConcept embeddings with the scConcept run of the teacher, `training_7`
   (environment `sc_concept_env2`, cluster, one GPU):

   ```bash
   sbatch scripts/eval/slurm_e1_scRNAseq_embs_v2_jeanzay.sh
   ```

   The embeddings are written to `scRNAseq_references/scConcept_embs/training_7/{tissue}.h5`. The
   projection is applied later, during the evaluation.

**Protocol.** The paper uses two `eval_id`s: `eval_1` with the `v1` classes and `eval_2` with the
`v2` classes.

3. Create the protocol (for `eval_2`, use `--coarse_mapping_version v2 --eval_id eval_2`). It writes
   `settings/eval_config.yaml`, which may be edited before step 4.

   ```bash
   python3 src/evaluation/part1_ST/task_A2_similarities/scripts/_1_setup.py \
       --training_dataset_id ds_ct1_20x_224_o112_1 --ct_cat_id ct_1 \
       --label_granularity coarse --coarse_mapping_version v1 \
       --modelgroup hoptimus0_hoptimus1_uni2h_virchow2 \
       --eval_id eval_1 --G_base_id G_base_1 \
       --ref_mapping_json data/eval_scRNAseq_ref/ref_tissue_files.json \
       --device cpu
   ```

4. Build the artifacts shared by all encoders (cohort, reference filters, gene sets):

   ```bash
   python3 src/evaluation/part1_ST/task_A2_similarities/scripts/_2_prepare_shared_artifacts.py --eval_id eval_1
   ```

5. Run the evaluation of one encoder. Results are written to
   `A-2_eval_st_predictions/eval_1/HEencoder_t46e13_clip/`, and
   `task_A2_similarities/scripts/umaps_slide_explorer.ipynb` helps explore the UMAPs.

   ```bash
   python3 src/evaluation/part1_ST/task_A2_similarities/scripts/_3_run_similarities.py \
       --eval_id eval_1 --training_num 46 --checkpoint_num 13
   ```

### PanNuke

[PanNuke](https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke) is an external test of the same
cell-typing system: its nuclei go through the same query path, with the same references, projection
and retrieval hyperparameters as `eval_2` (the `v2` classes), and the transferred cell types are
compared with PanNuke's pathologist labels (`Neoplastic` and `Epithelial` as
`Epithelial_Melanocyte`, `Inflammatory` as `Immune`, `Connective` as `Stromal_Vascular`; `Dead`
nuclei are not scored). Each tissue is treated as one slide. Only `t46e13` is evaluated. PanNuke is
released under CC BY-NC-SA 4.0.

The scripts are in `src/evaluation/part1_ST/task_A2_similarities/pannuke/`, and run in four steps:

| Step | Scripts | Environment |
|---|---|---|
| 1. Download and describe the data | `_1_download_pannuke.py`, `_2_characterize_pannuke.py`, `_3_check_reference_support.py` | `histost_eval_a2` |
| 2. Prepare the patches | `_4_prepare_patches.py` | `process_slide` |
| 3. Compute the embeddings | `_5_compute_HEembs.py` | `histost_env` (cluster), `histost` (check) |
| 4. Retrieval and scoring | `_6_run_retrieval.py` | `histost_eval_a2` |

**Step 1.** Download the dataset (`--probe_only` lists the files without downloading), describe it,
and select the tissues that have a single-cell reference:

```bash
P=src/evaluation/part1_ST/task_A2_similarities/pannuke
python3 $P/_1_download_pannuke.py --probe_only
python3 $P/_1_download_pannuke.py --revision 1f498f7bd6a85ef5f204c592b41ac881eab61005
python3 $P/_2_characterize_pannuke.py
python3 $P/_3_check_reference_support.py
```

The data and its description are written to `eval/eval_datasets/pannuke_eval_data/`.

**Step 2.** Prepare the patches in the same layout as the Task A evaluation dataset, for two
geometries: `native224` keeps the 256-pixel tile, resized by each encoder's own preprocessing, and
`scale224` resizes the tissue to 140 pixels and pads it with white to 224 pixels, closer to the
pixel size of the STHELAR slides.

```bash
python3 $P/_4_prepare_patches.py --geometry_recipe native224
python3 $P/_4_prepare_patches.py --geometry_recipe scale224
```

**Step 3.** Copy `pannuke_eval_data/prepared/` to the cluster (without the `overlays/` folders, and
to the default root, not the scratch root), submit the embedding job, copy the `he_embeddings/`
folders back, and check them locally:

```bash
sbatch scripts/eval/slurm_e1_pannuke_HEembs_jeanzay.sh     # on the cluster
python3 $P/_5_compute_HEembs.py --geometry_recipe native224 scale224 \
    --training_num 46 --checkpoint_num 13 --stages verify    # locally, environment histost
```

**Step 4.** Run the retrieval and the scoring. The results are written to
`eval/A-2_eval_st_predictions/pannuke/results/`.

```bash
python3 $P/_6_run_retrieval.py --geometry_recipe native224 scale224 \
    --training_num 46 --checkpoint_num 13 --stages references score cross_recipe verify
```

## Task A.3: cell-type classification

*Environment: `histost`.* A multinomial logistic regression, trained in each cross-validation fold
on the frozen H&E embedding of each cell, predicts its cell type. Its regularisation is selected by
cross-validation over the training slides. The paper uses one protocol, `eval_1`, with the `v1`
classes.

1. **Create the protocol** (its `settings/eval_config.yaml` may be edited before step 2):

   ```bash
   python3 src/evaluation/part1_ST/task_A3_celltype_predict/scripts/_1_setup.py \
       --eval_id eval_1 \
       --dataset_id ds_ct1_20x_224_o112 --training_dataset_id ds_ct1_20x_224_o112_1 --ct_cat_id ct_1 \
       --eval_configuration STHELAR_TRAIN --sub_eval_configuration STHELAR_TRAIN_CV \
       --level cell --label_granularity coarse --coarse_mapping_version v1 \
       --head_type linear_softmax --probe_C 1.0 \
       --scconcept_training_id training_7 \
       --modelgroup hoptimus0_hoptimus1_uni2h_virchow2
   ```

2. **Freeze the protocol:**

   ```bash
   python3 src/evaluation/part1_ST/task_A3_celltype_predict/scripts/_2_fill_eval_settings.py --eval_id eval_1
   ```

3. **Run the probe** for each encoder, and for the two reference arms. Results are written to
   `A-3_eval_celltype_predictions/eval_1/`, in one folder per arm.

   ```bash
   python3 src/evaluation/part1_ST/task_A3_celltype_predict/scripts/_3_run_eval.py \
       --eval_id eval_1 --arm he_encoder --training_num 46 --checkpoint_num 13 --space he --probe_seed all
   python3 src/evaluation/part1_ST/task_A3_celltype_predict/scripts/_3_run_eval.py --eval_id eval_1 --arm coords
   python3 src/evaluation/part1_ST/task_A3_celltype_predict/scripts/_3_run_eval.py --eval_id eval_1 --arm scconcept_ceiling
   ```

`_3_run_eval.py` deletes an existing result before recomputing it. The loop below runs all the arms
of the paper and skips those already done:

```bash
A3=src/evaluation/part1_ST/task_A3_celltype_predict/scripts
RESULTS=/path/to/ALHENA_data/eval/A-3_eval_celltype_predictions/eval_1
done_run () { [ -f "$RESULTS/$1/runs/seed_00011/benchmark/benchmark_summary.json" ]; }

for enc in 46:13 -1:-1 -2:-1 -3:-1 9001:0 50:13 51:13 52:13; do
    tn="${enc%%:*}"; cn="${enc##*:}"
    done_run "HEencoder_t${tn}e${cn}_he" || python3 "$A3/_3_run_eval.py" --eval_id eval_1 \
        --training_num "$tn" --checkpoint_num "$cn" --space he --probe_seed all
done
done_run HEencoder_t46e13_clip || python3 "$A3/_3_run_eval.py" --eval_id eval_1 \
    --training_num 46 --checkpoint_num 13 --space clip --probe_seed all
done_run COORDS_BASELINE || python3 "$A3/_3_run_eval.py" --eval_id eval_1 --arm coords
done_run SCCONCEPT_CEILING || python3 "$A3/_3_run_eval.py" --eval_id eval_1 --arm scconcept_ceiling
```

## Task B: transcriptomic signatures on TCGA

Task B has two phases. First, a frozen benchmark is built once (`tcga_benchmark_1`): a cohort of
TCGA cases with their diagnostic slides, tiles, RNA-derived signature scores and cross-validation
folds. Then each encoder is evaluated on it with a ridge probe per project and signature.

*Environment: `histost`, locally, except HistoQC (`histoqc`) and feature extraction (`histost_env`,
cluster).* The scripts are in `src/evaluation/part2_TCGA/tcga_benchmark/` (building) and
`src/evaluation/part2_TCGA/taskB_tcga_eval/` (evaluation).

### B.1 Build the benchmark

```bash
T=src/evaluation/part2_TCGA/tcga_benchmark
```

1. **Gene-symbol alias map,** from the HGNC table:

   ```bash
   mkdir -p /path/to/ALHENA_data/eval/eval_datasets/TCGA_eval_data/refs/hgnc
   curl -L "https://storage.googleapis.com/public-download-files/hgnc/tsv/tsv/hgnc_complete_set.txt" \
       -o /path/to/ALHENA_data/eval/eval_datasets/TCGA_eval_data/refs/hgnc/hgnc_complete_set_20260519.txt
   python3 $T/_0_make_hgnc_alias_map.py \
       --input hgnc_complete_set_20260519.txt --output gene_alias_map_hgnc_20260519.tsv
   ```

2. **Source release:** a frozen snapshot of the TCGA RNA-seq and diagnostic-slide metadata, from the
   GDC API (`--tcga_projects` restricts the projects):

   ```bash
   python3 $T/_1_build_source_release.py \
       --tcga_source_id gdc_tcga_20260519_starcounts_dxslides_v1 \
       --hgnc_alias_map gene_alias_map_hgnc_20260519.tsv
   ```

3. **Label each TCGA project** as `in_domain`, `out_of_domain` or `excluded` with
   `$T/_2_build_project_domain_map.ipynb`.

4. **Initialise the benchmark:**

   ```bash
   python3 $T/_3_init_benchmark.py \
       --benchmark_id tcga_benchmark_1 \
       --tcga_source_id gdc_tcga_20260519_starcounts_dxslides_v1 \
       --rna_asset_id rna_tcga_20260519_starcounts_v1 \
       --dataset_id ds_ct1_20x_224_o112 --training_dataset_id ds_ct1_20x_224_o112_1 \
       --modelgroup_id 224_112um_nooverlap_noups_qcv1
   ```

   The model group `224_112um_nooverlap_noups_qcv1` defines the tiles: 224 pixels at 0.5 µm per
   pixel, the native resolution of the foundation models. Slides are never upsampled, so slides
   coarser than 0.5 µm per pixel are excluded. The cohort settings are also fixed here: at most 75
   cases per project (`--max_cases_per_project`) and one slide per case (`--max_slides_per_case`).

5. **Signature set,** from the external signature panel (the script checks the result and stops on
   an error):

   ```bash
   python3 $T/_4_build_signature_set.py --benchmark_id tcga_benchmark_1
   python3 $T/_5_build_signature_support_training.py --benchmark_id tcga_benchmark_1
   ```

   The second command records which signatures the training genes support.

6. **Select the cohort** (one RNA file and one diagnostic slide per case; metadata only), then read
   the technical metadata of the selected slides:

   ```bash
   python3 $T/_6_resolve_rna_and_slides.py --benchmark_id tcga_benchmark_1
   python3 $T/_7_prefetch_slide_technical_metadata.py --benchmark_id tcga_benchmark_1
   ```

   `$T/check_slides.ipynb` shows the selected slides.

7. **Download the slides.** This needs the
   [GDC Data Transfer Tool](https://gdc.cancer.gov/access-data/gdc-data-transfer-tool) (`gdc-client`,
   version 2.3.0) on the `PATH`, or its location in `--gdc_client_bin`. The slides are open access.
   The download can be resumed.

   ```bash
   python3 $T/_8_download_selected_slides.py --benchmark_id tcga_benchmark_1 --n_processes 8
   ```

8. **HistoQC masks** (environment `histoqc`). The script can be split into shards run in parallel,
   followed by a final step that merges them:

   ```bash
   conda activate histoqc
   export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
   N=8
   for i in $(seq 0 $((N-1))); do
       python $T/_9_run_histoqc.py --benchmark_id tcga_benchmark_1 \
           --shard_index "$i" --total_shards "$N" > "histoqc_shard_$i.log" 2>&1 &
   done
   wait
   python $T/_9_run_histoqc.py --benchmark_id tcga_benchmark_1 --finalize_only --total_shards "$N"
   ```

   Without sharding, run `python $T/_9_run_histoqc.py --benchmark_id tcga_benchmark_1`. The final
   step fails if a slide failed; `--allow_histoqc_failures` drops those slides instead.
   `$T/watch_histoqc_progress.py --benchmark_id tcga_benchmark_1` shows the progress.

9. **Tiles:** tiling and tile quality control, using the HistoQC masks (`--resume` after an
   interruption):

   ```bash
   python3 $T/_10_build_tile_asset.py --benchmark_id tcga_benchmark_1
   ```

10. **Final cohort, RNA data, targets and folds:**

    ```bash
    python3 $T/_11_build_case_tile_index.py --benchmark_id tcga_benchmark_1
    python3 $T/_12_build_rna_asset.py --benchmark_id tcga_benchmark_1
    python3 $T/_13_build_targets_and_folds.py --benchmark_id tcga_benchmark_1
    ```

    The targets are signature scores computed from the RNA data of each case. The folds are grouped
    by tissue source site, or by case when a project has too few sites.

11. **Validate the benchmark,** which must pass before any encoder is evaluated:

    ```bash
    python3 $T/_14_validate_benchmark.py --benchmark_id tcga_benchmark_1
    ```

Downloaded slides, HistoQC masks and tiles are stored once and shared by all benchmarks (see
[data-layout.md](data-layout.md#eval)). An optional cache of the tile pixels
(`_15_build_tile_rgb_cache.py`) is not needed: feature extraction reads the tiles from the slides.

### B.2 Evaluate an encoder

1. **Extract the features** on the cluster (environment `histost_env`, one GPU). Copy to the cluster
   `TCGA_eval_data/benchmarks/tcga_benchmark_1/`, `TCGA_eval_data/selected_slide_cache/tcga_benchmark_1/manifests/`
   and the encoder file. With `--wsi_on_scratch`, the slides are read from the scratch root: copy
   there the slides of the final cohort (listed in `shared/tiles/case_tile_index.parquet`), keeping
   the layout of `selected_slide_cache/_store/slide_files/`.

   ```bash
   sbatch scripts/eval/slurm_e2_TCGAfeatures_jeanzay.sh
   ```

   The script runs `taskB_tcga_eval/_0_extract_encoder_features.py` for `t46e13`. A single pass
   computes both the `he` and `clip` spaces, and the mean, standard deviation and maximum of the
   tile embeddings of each case. The features are written to
   `benchmarks/tcga_benchmark_1/shared/encoder_features/HEencoder_t46e13/`; copy them back.

2. **Create the protocol** of the ridge probe, including how tile embeddings are pooled per case
   (`mean_std`: mean and standard deviation; also `mean` and `mean_std_max`). Each pooling needs its
   own `eval_id`.

   ```bash
   python3 src/evaluation/part2_TCGA/taskB_tcga_eval/_1_setup.py \
       --eval_id eval_1 --benchmark_id tcga_benchmark_1 --pooling mean_std
   ```

3. **Fit the ridge probes** of one encoder and space. Results are written to
   `B_eval_case/eval_1/HEencoder_t46e13_he/`.

   ```bash
   python3 src/evaluation/part2_TCGA/taskB_tcga_eval/_2_fit_ridge_probe.py \
       --eval_id eval_1 --benchmark_id tcga_benchmark_1 --training_num 46 --checkpoint_num 13 --space he
   ```

4. **Run the baselines,** which use no image features (`mean_predictor`, `tile_count`,
   `tissue_area`, `batch_stain_diagnostic`, `random_features`, `target_permutation`). Results are
   written to `B_eval_case/eval_1/baselines/`.

   ```bash
   python3 src/evaluation/part2_TCGA/taskB_tcga_eval/_3_run_baselines.py \
       --eval_id eval_1 --benchmark_id tcga_benchmark_1 --baselines all \
       --reference_run_id HEencoder_t46e13 --reference_space he
   ```

Repeat step 3 for each encoder and space. Encoders without a projection head support `--space he`
only.

## Task C: Thunder

*Environments: `thunder` (download, local) and `thunder_env` (benchmark, cluster).* Task C runs the
[Thunder](https://mics-lab.github.io/thunder/) benchmark: k-nearest-neighbour classification, linear
probing, few-shot classification, image retrieval, transformation invariance, adversarial attacks
and segmentation, on public pathology datasets. It needs the two corrections to `thunder-bench`
described in [installation.md](installation.md#thunder-and-thunder_env).

1. **Download the datasets** and copy them to the cluster:

   ```bash
   export THUNDER_BASE_DATA_FOLDER=/path/to/ALHENA_data/eval/eval_datasets/thunder_eval_data/
   thunder download-datasets all --make-splits
   ```

2. **Run the benchmark** for each encoder:

   ```bash
   sbatch scripts/eval/slurm_e3_thunder_jeanzay.sh
   ```

   The script runs `src/evaluation/part3_Thunder/_1_run_thunder_benchmark.py --training_num 46
   --checkpoint_num 13 --space he --datasets all --tasks all_non_vlm`. Results are written to
   `eval/C_eval_thunder/HEencoder_t46e13_he/`. Completed (dataset, task) pairs are recorded in
   `summary/_done/` and skipped when the script is run again.

3. **Copy the results back** (`results/`, `summary/`, `manifest.json` and `run.log`).

`thunder_anisotropy.py` and `thunder_geometry.py`, in the same folder, compute the geometry of the
Thunder tile embeddings (pairwise cosines, effective dimensionality) for each encoder and dataset.

## Task D: PathoBench

*Environments: `pathobench` (download, local), `trident_env` (features, cluster) and
`pathobench_env` (probes, cluster, CUDA GPU).* Task D runs 24
[PathoBench](https://github.com/mahmoodlab/patho-bench) tasks on five CPTAC cohorts (BRCA, CCRCC,
COAD, HNSC, LUAD), in four families (`molecular_status`, `molecular_status_ext`, `tme`, `grading`),
with a linear probe. [Trident](https://github.com/mahmoodlab/trident) tiles the slides and extracts
the patch features. The scripts are in `src/evaluation/part4_PathoBench/`.

1. **Download the slides** from the NCI Imaging Data Commons (local; `--dry_run` shows the slide
   counts and the download size first):

   ```bash
   python3 src/evaluation/part4_PathoBench/_0_download_cohorts.py --dry_run
   python3 src/evaluation/part4_PathoBench/_0_download_cohorts.py
   ```

   The slides are written to `pathobench_eval_data/slide_store/{cohort}/`, and the PathoBench
   splits to `pathobench_eval_data/pathobench_assets/`. Slides that cannot be matched automatically
   are listed for manual placement. The download can be resumed.

2. **Copy the data to the cluster:** the slides (`slide_store/{cohort}/wsi/`) to the scratch root,
   and `slide_store/{cohort}/manifests/` and `pathobench_assets/` to the default root, so that the
   probes can run without internet access.

3. **Extract the features** (environment `trident_env`), after preparing Trident's segmentation
   model ([installation.md](installation.md#tridents-segmentation-model-on-offline-compute-nodes)):

   ```bash
   sbatch scripts/eval/slurm_e4_pathobench_features_jeanzay.sh
   ```

   The script runs `_1_extract_features.py` for the encoder given in `--encoders` (one encoder per
   job). Resubmit it until every slide is done: it resumes where it stopped. The first job also
   tiles the slides, and later encoders reuse the same tiles. Do not run two jobs on the same cohort
   at the same time. The features are written to
   `trident_features/{cohort}/HEencoder_t{n}e{c}/{he,clip}/`, and the slides kept or rejected (with
   the reason) to `slide_store/{cohort}/manifests/slide_inventory.tsv`.

4. **Run the probes** (environment `pathobench_env`, one GPU). Edit `--encoders` in the script for
   each submission:

   ```bash
   sbatch scripts/eval/slurm_e4_pathobench_probe_jeanzay.sh
   ```

   or directly:

   ```bash
   python3 src/evaluation/part4_PathoBench/_2_run_pathobench.py \
       --eval_id eval_1 --encoders t46e13 --spaces he --heads linprobe --abmil_seeds 0 \
       --gpu 0 --pathobench_version 0.0.1 --trident_version 0.3.0
   ```

   `eval_1` fixes the task list and the probe settings; encoders, spaces and heads can be added to it
   later. Results are written to `eval/D_eval_pathobench/eval_1/`. A few slides whose image data
   cannot be decoded are excluded through a fixed list (`KNOWN_EXCLUDED_SLIDES` in
   `_pathobench_common.py`).
   `python3 -m src.evaluation.part4_PathoBench._pathobench_common --audit` checks the task list
   against the PathoBench catalogue.

5. **Copy the results back.**
