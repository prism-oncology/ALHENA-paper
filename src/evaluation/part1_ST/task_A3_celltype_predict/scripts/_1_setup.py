#!/usr/bin/env python
"""
Create the Task A.3 evaluation folder and its initial (editable) eval_config.yaml for a given eval_id.

Saved in <ROOT_HistoST> / eval / A-3_eval_celltype_predictions / <eval_id> / settings / eval_config.yaml
"""

import argparse
from pathlib import Path
import sys

# Add the task_A3 dir to sys.path so that 'src_A3' can be imported
# (ensure_imports() then adds the repo root and the task_A1 dir)
task_A3_dir = Path(__file__).resolve().parents[1]
if str(task_A3_dir) not in sys.path:
    sys.path.insert(0, str(task_A3_dir))

from src_A3.common import ensure_imports

ensure_imports()

from src_A3.common import (
    assert_setup_granularity_allowed,
    VALID_A3_EVAL_CONFIGS,
    VALID_A3_HEAD_TYPES,
    VALID_A3_LABEL_GRANULARITY,
    VALID_A3_SUB_EVAL_CONFIGS,
    VALID_LEVELS,
    mkdir,
    resolve_a3_eval_task_root,
    rmtree_appledouble_safe,
)
from src_A3.protocols import create_initial_a3_eval_folder


# Folders under an eval_id that hold results
_RESULT_FOLDER_PREFIXES = ("HEencoder_",)
_RESULT_FOLDER_NAMES = ("COORDS_BASELINE", "SCCONCEPT_CEILING")


def _existing_result_folders(eval_dir: Path) -> list:
    """Sorted names of the result folders under eval_dir (dot-prefixed entries skipped)."""
    if not eval_dir.exists():
        return []
    found = []
    for child in eval_dir.iterdir():
        name = child.name
        if name.startswith(".") or not child.is_dir():
            continue
        if name in _RESULT_FOLDER_NAMES or name.startswith(_RESULT_FOLDER_PREFIXES):
            found.append(name)
    return sorted(found)


def main(args: argparse.Namespace) -> None:
    """Create the A.3 eval folder and write its initial eval_config.yaml for one eval_id."""

    # Check the class axis (fine / coarse mapping version) before writing anything
    assert_setup_granularity_allowed(
        label_granularity=args.label_granularity,
        coarse_mapping_version=args.coarse_mapping_version,
        allow_retired_fine_axis=bool(args.allow_retired_fine_axis),
    )

    eval_root = mkdir(resolve_a3_eval_task_root())
    eval_dir = eval_root / args.eval_id

    print(f"Setting up Task A.3 evaluation folder for eval_id: {args.eval_id}")

    overwrite = bool(args.overwrite or args.overwrite_with_results)

    if eval_dir.exists() and not overwrite:
        raise FileExistsError(
            f"Eval folder already exists: {eval_dir}. Use --overwrite to refresh settings."
        )

    if overwrite and eval_dir.exists():
        # Refuse to delete existing results without --overwrite_with_results
        existing_results = _existing_result_folders(eval_dir)
        if existing_results and not args.overwrite_with_results:
            raise FileExistsError(
                f"Refusing to --overwrite {eval_dir}: it holds {len(existing_results)} result "
                f"folder(s) that would be deleted: {existing_results}. Use "
                "--overwrite_with_results to delete them, or pick a new --eval_id."
            )
        if existing_results:
            print(
                f"--overwrite_with_results: deleting {len(existing_results)} completed result "
                f"folder(s): {existing_results}"
            )
        rmtree_appledouble_safe(eval_dir)

    eval_dir.mkdir(parents=True, exist_ok=True)

    # --level is not forwarded (Task A.3 is cell-level only); protocol_seed=None applies the default seed
    create_initial_a3_eval_folder(
        eval_dir=eval_dir,
        eval_id=args.eval_id,
        dataset_id=args.dataset_id,
        training_dataset_id=args.training_dataset_id,
        ct_cat_id=args.ct_cat_id,
        eval_configuration=args.eval_configuration,
        sub_eval_configuration=args.sub_eval_configuration,
        modelgroup=args.modelgroup,
        label_granularity=args.label_granularity,
        coarse_mapping_version=args.coarse_mapping_version,
        head_type=args.head_type,
        probe_C=args.probe_C,
        probe_c_min=args.probe_c_min,
        probe_c_max=args.probe_c_max,
        probe_c_points=args.probe_c_points,
        min_selected_alpha=args.min_selected_alpha,
        scconcept_training_id=args.scconcept_training_id,
        protocol_seed=None,
    )

    print(f"Created eval settings folder: {eval_dir / 'settings'}")
    print(f"Main config: {eval_dir / 'settings' / 'eval_config.yaml'}")
    print("You can edit eval_config.yaml before running the settings fill step. "
          "Do not modify this config after running the `_2_fill_eval_settings.py` step")




if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create the Task A.3 evaluation folder and initial eval_config.yaml.")

    parser.add_argument("--eval_id", type=str,
                        required=True,
                        help="Evaluation ID, e.g. 'eval_1'")

    parser.add_argument("--dataset_id", type=str,
                        required=True,
                        help="Dataset ID, e.g. 'ds_ct1_20x_224_o112'")

    parser.add_argument("--training_dataset_id", type=str,
                        required=True,
                        help="Training dataset ID, e.g. 'ds_ct1_20x_224_o112_1'")

    parser.add_argument("--ct_cat_id", type=str,
                        default="ct_1",
                        help=("Cell-type category ID, e.g. 'ct_1' (label dictionaries in "
                              "data/annots/annot_dicts_<ct_cat_id>)."))

    parser.add_argument("--eval_configuration", type=str,
                        required=True,
                        choices=sorted(VALID_A3_EVAL_CONFIGS),
                        help="Evaluation configuration (STHELAR_TRAIN or STHELAR_TEST).")

    parser.add_argument("--sub_eval_configuration", type=str,
                        required=True,
                        choices=sorted(VALID_A3_SUB_EVAL_CONFIGS),
                        help=("Sub-evaluation configuration, containing eval_configuration "
                              "(e.g. STHELAR_TRAIN -> STHELAR_TRAIN_CV)."))

    parser.add_argument("--level", type=str,
                        default="cell",
                        choices=sorted(VALID_LEVELS),
                        help="Evaluation level. Task A.3 is cell-level only ('cell').")

    parser.add_argument("--label_granularity", type=str,
                        required=True,
                        choices=sorted(VALID_A3_LABEL_GRANULARITY),
                        help=("Cell-type label granularity: 'coarse' (cat2idx categories grouped by the "
                              "mapping of --coarse_mapping_version) or 'fine' (cat2idx categories, "
                              "requires --allow_retired_fine_axis)."))

    parser.add_argument("--coarse_mapping_version", type=str,
                        default=None,
                        help=("Coarse mapping version (coarse_ct_mapping_<version>.json): 'v1' = 5 classes, "
                              "'v2' = 3 classes. Required with --label_granularity coarse; optional with "
                              "'fine' (adds a coarse-by-aggregation view)."))

    parser.add_argument("--allow_retired_fine_axis",
                        action="store_true",
                        help=("Allow --label_granularity fine (8-class axis, refused otherwise)."))

    parser.add_argument("--modelgroup", type=str,
                        required=True,
                        help="Model group, e.g. 'hoptimus0_hoptimus1_uni2h_virchow2'.")

    parser.add_argument("--head_type", type=str,
                        default="linear_softmax",
                        choices=sorted(VALID_A3_HEAD_TYPES),
                        help=("Probe head type: 'linear_softmax' (sklearn multinomial logistic "
                              "regression) or 'mlp_softmax' (not implemented)."))

    parser.add_argument("--probe_C", type=float,
                        default=1.0,
                        help=("Fixed inverse L2 regularization strength C of the logistic probe, used "
                              "only when protocol.probe.inner_cv_select is False (otherwise C is "
                              "selected by inner CV)."))

    # Inner-CV C search grid
    parser.add_argument("--probe_c_min", type=float,
                        default=1e-6,
                        help=("Lower end of the inner-CV C search grid (protocol.probe.C_grid.c_min). "
                              "Default 1e-6."))

    parser.add_argument("--probe_c_max", type=float,
                        default=1e0,
                        help=("Upper end of the inner-CV C search grid (protocol.probe.C_grid.c_max). "
                              "Default 1e0."))

    parser.add_argument("--probe_c_points", type=int,
                        default=7,
                        help=("Number of log-spaced points in the inner-CV C search grid "
                              "(protocol.probe.C_grid.n_points). Default 7."))

    parser.add_argument("--min_selected_alpha", type=float,
                        default=None,
                        help=("Keep only cells whose main-assignment selected_alpha is >= this "
                              "threshold. Default None = no filter."))

    parser.add_argument("--scconcept_training_id", type=str,
                        default=None,
                        help=("scConcept training_id of the per-cell embeddings used by the "
                              "scconcept_ceiling arm (requires embeddings for every eval slide). "
                              "Default None = arm not configured."))

    parser.add_argument("--overwrite",
                        action="store_true",
                        help=("Delete and recreate the evaluation folder of this eval_id. Refused if "
                              "it holds results (HEencoder_*/, COORDS_BASELINE/, SCCONCEPT_CEILING/): "
                              "see --overwrite_with_results."))

    parser.add_argument("--overwrite_with_results",
                        action="store_true",
                        help=("Same as --overwrite, and delete the existing results."))

    args = parser.parse_args()
    main(args)
