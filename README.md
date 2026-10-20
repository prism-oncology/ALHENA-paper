<p align="center">
  <img src="docs/assets/alhena_banner.svg" alt="ALHENA" width="100%" height="168">
</p>

<p align="center">
  <b>ALHENA, an H&amp;E encoder distilled from transcriptomic foundation models for multi-scale slide analysis.</b>
</p>

<p align="center">
  <b>Preprint</b> [TODO: add link to preprint] |
  <a href="https://github.com/prism-oncology/ALHENA"><b>ALHENA package</b></a> |
  <a href="https://prism-oncology.github.io/ALHENA/"><b>ALHENA documentation</b></a> |
  <a href="https://huggingface.co/prism-oncology/ALHENA"><b>Weights on Hugging Face</b></a> |
  <a href="https://huggingface.co/prism-oncology/ALHENA-references"><b>References on Hugging Face</b></a>
</p>

<p align="center">
  <img alt="Licence: BSD-3-Clause" src="https://img.shields.io/badge/licence-BSD--3--Clause-blue.svg">
  <img alt="Python 3.12" src="https://img.shields.io/badge/python-3.12-blue.svg">
</p>

---

This repository holds the code for the ALHENA training and evaluation: the preprocessing pipeline, the
distillation training loop, the export of the adapted encoder and its projection heads, and the
seven evaluation tasks.

ALHENA adapts the pathology foundation encoder UNI2-h with low-rank adapters, supervised by two
frozen spatial transcriptomics foundation models: scConcept at the cell level and Novae at the
neighbourhood level.

ALHENA is gamma Geminorum, from the Arabic *al-han'ah*, a brand mark placed on something to make it readable. To the eye, it appears as a single star, though it is in fact a binary system. The light we see comes from a bright A-type subgiant. Its companion has never been imaged and reveals itself only through the faint wobble it leaves in the motion of the visible star. The H&E section is that visible star, inexpensive, familiar, and collected everywhere. The spatial transcriptome is the unseen companion, physically present, shaping what we observe, yet never directly accessible at inference.

## Which repository do you want?

| You want to | Go to |
|---|---|
| **Use the model** — embed tiles or cells, type cells against a single-cell atlas | the `alhena` package: [github.com/prism-oncology/ALHENA](https://github.com/prism-oncology/ALHENA) (code) and [prism-oncology.github.io/ALHENA](https://prism-oncology.github.io/ALHENA/) (documentation) |
| **Get the weights** — adapter tensors, the projection heads, the embedded references | [huggingface.co/prism-oncology/ALHENA](https://huggingface.co/prism-oncology/ALHENA) (adapters and heads) and [huggingface.co/prism-oncology/ALHENA-references](https://huggingface.co/prism-oncology/ALHENA-references) (embedded references) |
| **Read or re-run the study** — training and the evaluation protocols | **this repository** |

> This repository contains the code used to train and evaluate the model. It can also be used to apply the same training methodology to other datasets or foundation models.
> For practical use of the trained model, please use ALHENA.

## Documentation

| Guide | Contents |
|---|---|
| [docs/installation.md](docs/installation.md) | The seventeen conda environments and how to build them |
| [docs/data-layout.md](docs/data-layout.md) | The external data and results tree, and how paths are resolved |
| [docs/training.md](docs/training.md) | Preprocessing, teacher pretraining, training, export and the training ablations, step by step |
| [docs/evaluation.md](docs/evaluation.md) | How to run the seven evaluation tasks |

## Repository layout

```
src/
  preprocessing/          raw Xenium sections -> aligned slides, H&E tiles, ST graphs, splits
  pretraining_ST/         the two frozen ST teachers (Novae, scConcept)
  dataset/                the training dataset, teacher caches and centring statistics
  models/                 the encoder, the low-rank adapters and the contrastive heads
  training/               the training loop, entry point run_experiment.py
  evaluation/             the seven evaluation tasks and encoder export
  utils/                  paths, configuration defaults and shared helpers
  preclip_investigation/  the offline teacher-geometry study behind the training settings

configs/                  configuration templates
data/                     small metadata: annotation dictionaries, gene signatures
envs/                     seventeen conda environment specifications
scripts/                  SLURM submission scripts for cluster runs
docs/                     the guides listed above
```

Code lives here; **data, models, checkpoints and results do not**. They live in a separate external
tree whose location you set yourself.

The code uses the internal project name `HistoST` in module paths, environment names,
configuration keys and folder names. It refers to the same model as ALHENA.

## Before you run anything: set the data root

Every path into the external tree is built by `get_full_path()` in
[src/utils/constants.py](src/utils/constants.py). That file ships with **placeholder roots**, and
they must point at your own storage before any script will run:

```python
ROOT_PATH: Path = Path("/path/to/ALHENA_data")                                                     # local
ROOT_PATH_JeanZay: Path = Path("/lustre/fswork/projects/rech/<group>/<login>/ST_HE/HistoST")        # cluster
ROOT_PATH_JeanZay_scratch: Path = Path("/lustre/fsn1/projects/rech/<group>/<login>/ST_HE/HistoST")  # cluster scratch
```

Never hard-code a root anywhere else; always build paths with `get_full_path()`. The expected tree
is described in [docs/data-layout.md](docs/data-layout.md).

The SLURM scripts under `scripts/` likewise carry placeholders for the compute allocation, the
notification address and the location of this repository on the cluster (`--account=<group>@a100`,
`your.email@institution.org`, `/linkhome/rech/<home_group>/<login>/...`). Replace them with your own
before submitting.

## The pipeline

1. **Preprocessing** — raw Xenium sections become aligned slides, H&E tiles, ST graphs, cell-type
   maps and train/validation/test folds.
2. **ST teacher pretraining** — Novae at the neighbourhood level and scConcept at the cell level;
   both are then frozen.
3. **Teacher precomputation** — the per-tile supervision targets and their centring statistics are
   computed once and cached.
4. **LMDB cache** *(optional)* — a one-time, geometry-only cache of a training dataset.
5. **Training** — low-rank adapters are fitted inside the frozen backbone against the two frozen
   teachers. Entry point: `src/training/run_experiment.py`.
6. **Export and evaluation** — the adapted encoder and its heads are exported, then graded by the
   seven tasks below.

Full commands in [docs/training.md](docs/training.md).

## The evaluations

| Task | Question it answers |
|---|---|
| **A.0** | What geometry does the adapted embedding space have? *(descriptive)* |
| **A.1** | Can a gene's expression be read from the image, at the neighbourhood? |
| **A.2** | Can cells be typed against a single-cell atlas with nothing fitted? |
| **A.3** | Is the cell-level knowledge in the backbone rather than the projection? |
| **B** | Does it transfer to case-level molecular signatures on TCGA? |
| **C** | Does general pathology capability survive? |
| **D** | Does it transfer to slide-level tasks on CPTAC? |

The paper's ten evaluations, E1 to E10 in its Table 1, are run by these seven tasks;
[docs/evaluation.md](docs/evaluation.md) maps one onto the other. Tasks A.1, A.2, A.3, B and D
freeze each protocol under an identifier (`eval_id`) that fixes the cohort, the splits, the targets
or cell-type classes, the probe and its hyperparameter grid, and the metrics. The setup commands in
[docs/evaluation.md](docs/evaluation.md) regenerate these protocols.

To evaluate the released model rather than a model you trained, rebuild its encoder files from the
weights on Hugging Face with `src/evaluation/rebuild_released_encoders.py` (see
[docs/evaluation.md](docs/evaluation.md)).

## Licence

The source code, the evaluation harness and the environment specifications are released under the
**BSD 3-Clause License** — see [LICENSE](LICENSE).

The ALHENA model weights, comprising the LoRA adapters and the projection heads used at inference,
are released under the Creative Commons Attribution-NonCommercial 4.0 International License
(CC BY-NC 4.0). ALHENA requires the UNI2-h weights, which are not redistributed and must be
obtained from [MahmoodLab/UNI2-h](https://huggingface.co/MahmoodLab/UNI2-h) under the licence and
terms of use of their authors.

The precomputed single-cell reference representations are released under the CC BY-NC 4.0
licence. They remain subject to the terms of their source atlases, and those derived from DISCO
are provided for academic research only, in accordance with the DISCO terms of use. No expression
values, gene names or cell barcodes from the source atlases are redistributed. The
[references repository](https://huggingface.co/prism-oncology/ALHENA-references) gives the source,
terms and credit of each.

## Citation

[TODO: add link to preprint]

If you use this code or ALHENA, please cite the preprint:

```bibtex
@article{alhena,
  title   = {{ALHENA}, an {H\&E} encoder distilled from transcriptomic foundation models
             for multi-scale slide analysis},
  author  = {Giraud-Sauveur, F{\'e}licie and Blampey, Quentin and Marza, Pierre and
             Benkirane, Hakim and Vakalopoulou, Maria and Courn{\`e}de, Paul-Henry and
             Christodoulidis, Stergios},
  journal = {TODO: preprint server},
  year    = {TODO},
  doi     = {TODO}
}
```
