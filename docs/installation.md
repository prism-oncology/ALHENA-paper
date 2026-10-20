# Installation

The study uses seventeen conda environments, specified under [`envs/`](../envs). They are kept
separate because the benchmark tools (HistoQC, Thunder, Trident, PathoBench, CellViT) pin package
versions that conflict with each other and with the training stack.

The code was run on two machines, and most environments exist in two versions:

| Machine | Used for | File suffix |
|---|---|---|
| Local workstation (macOS, Apple silicon) | preprocessing, downloads, CPU-bound evaluation steps | `_local.yml` |
| Linux GPU cluster with SLURM (Jean Zay) | teacher pretraining, training, GPU embedding extraction, GPU benchmark steps | `_jeanzay.yml` |

The two versions of an environment differ mainly in the PyTorch build: CPU/MPS on the workstation,
CUDA on the cluster (12.8 for most environments, 11.8 for CellViT, 12.4 for scConcept version 1,
13.0 for Novae and Thunder). A single Linux machine with a CUDA GPU can host all of them.

Before running anything, set the data root in `src/utils/constants.py`; see the
[README](../README.md#before-you-run-anything-set-the-data-root) and [data-layout.md](data-layout.md).

## The environments

| Environment | File | Used for |
|---|---|---|
| `process_slide` | `environment_process_slide_local.yml` | Preprocessing (`src/preprocessing/`), downloading the evaluation data and the pretrained models, PanNuke patch preparation |
| `histost` | `environment_HistoST_local.yml` | Main local environment: LMDB cache, teacher statistics, teacher-geometry study, encoder export, most evaluation steps |
| `histost_env` | `environment_HistoST_jeanzay.yml` | Cluster version of `histost`: training and GPU embedding extraction |
| `histost_eval_a2` | `environment_eval_a2_local.yml` | Task A.2 and the PanNuke retrieval (the only environment with FAISS) |
| `histoqc` | `environment_histoqc_local.yml` | Task B tissue and artefact masks (HistoQC) |
| `sc_concept_env` | `environment_scConcept_v1_jeanzay.yml` | scConcept version 1 (`Corpus-30M`) |
| `sc_concept_env2` | `environment_scConcept_v2_jeanzay.yml` | scConcept version 2 (`corpus40M-model30M`), the version used for the released model |
| `novae` | `environment_Novae_local.yml` | Novae public-cohort download, Novae domains on evaluation patches |
| `novae_env` | `environment_Novae_jeanzay.yml` | Novae pretraining and the Novae teacher precomputation |
| `thunder` | `environment_Thunder_local.yml` | Task C dataset download |
| `thunder_env` | `environment_Thunder_jeanzay.yml` | Task C benchmark |
| `trident` | `environment_trident_local.yml` | Task D tiling and patch features, local |
| `trident_env` | `environment_trident_jeanzay.yml` | Task D tiling and patch features, cluster |
| `pathobench` | `environment_pathobench_local.yml` | Task D slide download |
| `pathobench_env` | `environment_pathobench_jeanzay.yml` | Task D slide-level probes (needs a CUDA GPU) |
| `cellvit` | `environment_CellViT_local.yml` | Optional CellViT patch metrics during preprocessing |
| `cellvit_env` | `environment_CellViT_jeanzay.yml` | Optional CellViT inference on the cluster |

Each step in [training.md](training.md) and [evaluation.md](evaluation.md) states the
environment it runs in.

## Building an environment

Each file under `envs/` is a full `conda env export` of the environment as it was used. Build from
it directly:

```bash
conda env create -f envs/environment_HistoST_local.yml
```

If a file does not solve on your platform (another operating system or CUDA version), build the
environment from its recipe below. On the cluster, every recipe starts with:

```bash
module purge
module load miniforge/24.11.3
module load cuda/12.8.0
```

On a local machine, skip these lines and install PyTorch without the CUDA index URL.

### `process_slide`

```bash
conda create -n process_slide python=3.12
conda activate process_slide
pip install sopa
pip install git+https://github.com/scverse/spatialdata-io.git@main
pip install tiffslide decoupler leidenalg huggingface-hub opencv-python timm
```

### `histost` and `histost_env`

Python 3.12 is required. On the cluster:

```bash
conda create -n histost_env python=3.12
conda activate histost_env
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install pyg_lib torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-2.8.0+cu128.html
pip install anndata==0.12.3 wandb==0.19.11
pip install numpy pandas scipy pyyaml pillow tqdm natsort albumentations lightning pyarrow fastparquet timm lmdb scanpy
pip install idr_torch    # multi-GPU helper used on Jean Zay
conda install -c conda-forge openslide-python
```

On the local machine:

```bash
conda create -n histost python=3.12
conda activate histost
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.8.0+cpu.html
pip install torch_geometric
pip install anndata==0.12.3 wandb==0.19.11
pip install numpy pandas scipy pyyaml pillow tqdm natsort albumentations lightning pyarrow fastparquet timm lmdb scanpy
pip install nbformat jupyter papermill beautifulsoup4
conda install -c conda-forge openslide-python
```

`openslide-python` comes from conda-forge so that it is built against the conda-forge OpenSlide
library.

### `histost_eval_a2`

```bash
conda create -n histost_eval_a2 python=3.11 -y
conda activate histost_eval_a2
conda install -c conda-forge -y numpy scipy pandas h5py anndata pyarrow pyyaml matplotlib \
    scikit-learn numba umap-learn pillow tqdm natsort opencv timm faiss-cpu llvm-openmp
conda install -c conda-forge -y pytorch torchvision
```

FAISS and its OpenMP runtime must come from conda-forge, not pip, so that the native libraries stay
consistent with the rest of the environment.

### `histoqc`

```bash
conda create -n histoqc python=3.10 -y
conda activate histoqc
conda install -c conda-forge openslide -y
pip install "git+https://github.com/choosehappy/HistoQC.git@b79561292e49d153e2e4c1401c9c48d3d5f925e1"
pip install pyyaml pandas
python -m histoqc --help    # check the install
```

HistoQC is not on PyPI. It is installed from a pinned commit because the configuration shipped with
this repository, `src/evaluation/part2_TCGA/tcga_benchmark/histoqc_configs/tcga_taskb_v1.ini` (a
copy of HistoQC's `config_v2.1.ini`), calls functions that moved in later versions. Do not edit that
file: its hash and the HistoQC version are part of the Task B benchmark identifiers.
`_9_run_histoqc.py` runs HistoQC with the interpreter of this environment, which must therefore also
be able to import the repository.

### `sc_concept_env` (scConcept version 1)

```bash
conda create -n sc_concept_env python=3.12
conda activate sc_concept_env
pip install git+https://github.com/theislab/scConcept.git@main
pip install --no-cache-dir https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp312-cp312-linux_x86_64.whl
pip install git+https://github.com/theislab/lamin_dataloader.git
```

The flash-attn wheel must match the Python, CUDA and PyTorch versions. Alternatively, build it with
`pip install flash-attn==2.7.* --no-build-isolation`.

### `sc_concept_env2` (scConcept version 2)

```bash
conda create -n sc_concept_env2 python=3.12
conda activate sc_concept_env2
pip install sc-concept==0.2.4 --extra-index-url https://download.pytorch.org/whl/cu128
pip install --no-cache-dir --no-build-isolation "flash-attn @ https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
```

The extra index is needed to get the CUDA 12.8 build of PyTorch. flash-attn is optional: without
it, scConcept uses standard attention. The two scConcept versions produce different embeddings; never
mix them.

### `novae` and `novae_env`

Novae is used unmodified at tag `v1.1.1`, installed from a source checkout:

```bash
git clone https://github.com/prism-oncology/novae.git
cd novae
git checkout v1.1.1
conda create -n novae python=3.12 -y    # novae_env on the cluster
conda activate novae
pip install -e .
pip install fast-array-utils wandb==0.19.11 pyarrow
python -c "import novae; print(novae.__version__)"    # 1.1.1
```

### `thunder` and `thunder_env`

```bash
conda create -n thunder_env python=3.10    # thunder on the local machine
conda activate thunder_env
pip install thunder-bench==0.2.2
```

On the cluster, load only the `miniforge` module; this environment needs no CUDA module.

`thunder-bench==0.2.2` needs two corrections, to be made by hand in the installed package, whose
location is printed by:

```bash
python -c "import thunder, os; print(os.path.dirname(thunder.__file__))"
```

The paths in the diffs below are relative to the Thunder source repository (`src/thunder/...`).

**Correction 1.** A custom encoder is not passed to the transformation-invariance task, which then
silently evaluates the default model:

```diff
diff --git a/src/thunder/benchmark.py b/src/thunder/benchmark.py
--- a/src/thunder/benchmark.py
+++ b/src/thunder/benchmark.py
@@ -427,6 +427,7 @@ def run_benchmark(cfg: DictConfig, model_cls: Callable = None) -> None:
             base_data_folder,
             wandb_base_folder,
             res_folder,
+            model_cls,
         )
     elif task_type == "adversarial_attack":
         adversarial_attack(
diff --git a/src/thunder/tasks/transformation_invariance.py b/src/thunder/tasks/transformation_invariance.py
--- a/src/thunder/tasks/transformation_invariance.py
+++ b/src/thunder/tasks/transformation_invariance.py
@@ -9,7 +9,8 @@ import os
 import random
 from collections import defaultdict
-from typing import Dict, List
+from collections.abc import Callable
+from typing import Dict, List
 
 import hydra
 import numpy as np
@@ -112,6 +113,7 @@ def transformation_invariance(
     base_data_folder: str,
     wandb_base_folder: str,
     res_folder: str,
+    model_cls: Callable = None,
 ) -> dict:
     """
     Computing transformation invariance metrics for a given model and dataset.
@@ -124,6 +126,7 @@ def transformation_invariance(
     :param base_data_folder: base path to the data.
     :param wandb_base_folder: folder name for wandb logging.
     :param res_folder: path where to save outputs.
+    :param model_cls: optional custom PretrainedModel instance.
     :return: dictionary with invariance metrics.
     """
 
@@ -159,9 +162,18 @@ def transformation_invariance(
     # Loading pretrained foundation model (FM) and preprocessing
     # ---------------------------------------------------------------------
 
-    fm_model, preprocess, extract_embedding = load_pretrained_model(
-        cfg, adaptation_type, device
-    )
+    if model_cls is not None:
+        fm_model = model_cls
+        preprocess = model_cls.get_transform()
+        extract_embedding = model_cls.get_embeddings
+
+        for param in fm_model.parameters():
+            param.requires_grad = False
+
+        fm_model.to(device)
+    else:
+        fm_model, preprocess, extract_embedding = load_pretrained_model(
+            cfg, adaptation_type, device
+        )
 
     fm_model.eval()
```

**Correction 2.** At evaluation time the segmentation decoder is rebuilt with the wrong feature
dimension, which crashes encoders whose token width differs from their pooled width (for example
Virchow2):

```diff
diff --git a/src/thunder/tasks/train_eval_probe.py b/src/thunder/tasks/train_eval_probe.py
--- a/src/thunder/tasks/train_eval_probe.py
+++ b/src/thunder/tasks/train_eval_probe.py
@@ -359,6 +359,8 @@ def eval_probe(
     if model_cls is not None:
         OmegaConf.set_struct(cfg, False)
         cfg.pretrained_model = {"emb_dim": model_cls.emb_dim}
+        if hasattr(model_cls, "emb_dim_seg"):
+            cfg.pretrained_model["emb_dim_seg"] = model_cls.emb_dim_seg
 
     # Task-specific model
     if task_type == "linear_probing":
```

### `trident`, `trident_env`, `pathobench` and `pathobench_env`

Task D uses [Trident](https://github.com/mahmoodlab/trident) for tiling and patch features and
[PathoBench](https://github.com/mahmoodlab/patho-bench) for the slide-level probes. Both are
licensed for non-commercial academic use only. Install PyTorch first, so that the CUDA build is
used, and keep `timm==0.9.16`: timm 1.x breaks the Trident patch encoders.

Trident, on the cluster:

```bash
conda create -n trident_env python=3.10
conda activate trident_env
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
pip install timm==0.9.16 openslide-python opencv-python h5py pandas
git clone https://github.com/mahmoodlab/trident && cd trident && pip install -e ".[patch-encoders]"
```

PathoBench, on the cluster:

```bash
conda create -n pathobench_env python=3.10
conda activate pathobench_env
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
pip install scikit-learn scikit-survival datasets h5py pandas requests pyyaml
git clone https://github.com/mahmoodlab/patho-bench && cd patho-bench && pip install -e .
pip install "timm==0.9.16" ipython lovely-tensors git+https://github.com/mahmoodlab/trident.git
```

The last line must come after the PathoBench install: PathoBench imports Trident and IPython but
does not install them.

The local environments are called `trident` and `pathobench`: drop the CUDA module and install
PyTorch with `pip install torch==2.8.0 torchvision==0.23.0`. The local `pathobench` environment also
needs `idc-index`, which downloads the CPTAC slides from the NCI Imaging Data Commons:

```bash
conda activate pathobench
pip install idc-index
```

#### Trident's segmentation model on offline compute nodes

Trident's tissue segmenter downloads two checkpoints the first time it runs. If the cluster's
compute nodes have no internet access, prepare both beforehand, on a node that has it:

1. The torchvision ResNet-50 ImageNet weights: place
   [`resnet50-0676ba61.pth`](https://download.pytorch.org/models/resnet50-0676ba61.pth) in
   `~/.cache/torch/hub/checkpoints/`.
2. The HEST tissue-segmentation checkpoint. It must be downloaded **and** registered in Trident's
   `local_ckpts.json`, because Trident only looks for it there. In the `trident_env` environment:

```bash
python3 - <<'PY'
import json, os, trident
from huggingface_hub import snapshot_download
seg_dir = os.path.join(os.path.dirname(trident.__file__), "segmentation_models")
model_zoo = os.path.join(seg_dir, "model_zoo")
os.makedirs(model_zoo, exist_ok=True)
snapshot_download(repo_id="MahmoodLab/hest-tissue-seg", repo_type="model",
                  allow_patterns=["deeplabv3_seg_v4.ckpt"], local_dir=model_zoo)
registry_path = os.path.join(seg_dir, "local_ckpts.json")
registry = json.load(open(registry_path))
registry["hest"] = "deeplabv3_seg_v4.ckpt"
json.dump(registry, open(registry_path, "w"), indent=4)
PY

# Prints the checkpoint path; an empty line means the registration did not work
python3 -c "from trident.IO import get_weights_path; print(get_weights_path('seg', 'hest'))"
```

To confirm that it works offline, run this on a compute node:

```bash
python -c "from trident.segmentation_models.load import segmentation_model_factory as f; f('hest'); print('OK')"
```

### `cellvit` and `cellvit_env`

These are only needed for the optional CellViT patch metrics during preprocessing. Build them from
their files (Python 3.9.7, `torch==2.2.1`):

```bash
conda env create -f envs/environment_CellViT_local.yml
conda env create -f envs/environment_CellViT_jeanzay.yml
```

## Pinned versions

| Package | Version |
|---|---|
| HistoQC | commit `b79561292e49d153e2e4c1401c9c48d3d5f925e1` (reports version 2.0) |
| thunder-bench | 0.2.2, with the two corrections above |
| Trident | 0.3.0, with `timm==0.9.16` |
| patho-bench | 0.0.1, with `timm==0.9.16` |
| Novae | 1.1.1 |
| scConcept | 0.0.1 (version 1), `sc-concept` 0.2.4 (version 2) |
| anndata, wandb | 0.12.3 and 0.19.11 in `histost` and `histost_env` |

The exact version of every other package is in the environment files.

## Checking an install

Run from the repository root, in any environment that imports the repository:

```bash
python -c "from src.utils.constants import get_full_path; print(get_full_path('train'))"
```

This prints the resolved path under the data root, which also confirms that the root was set.

For the benchmark environments, check that the tools import:

```bash
python -c "import thunder; print(thunder.__file__)"                           # thunder, thunder_env
python -c "import trident, timm; print(trident.__version__, timm.__version__)"  # trident, trident_env
python -c "import patho_bench, trident, IPython; print('OK')"                  # pathobench, pathobench_env
```
