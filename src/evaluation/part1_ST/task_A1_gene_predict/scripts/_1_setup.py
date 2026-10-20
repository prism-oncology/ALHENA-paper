#!/usr/bin/env python
"""
Create the Task A.1 evaluation folder and initial eval_config.yaml for a given eval_id.

Saved in <ROOT_HistoST> / eval / A-1_eval_st_predictions / <eval_id> / settings / eval_config.yaml
"""

import argparse
from pathlib import Path
import sys

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src_A1.common import (
    CONTROL_SUB_EVAL_CONFIGS,
    mkdir,
    resolve_eval_task_root,
    rmtree_appledouble_safe,
    validate_G_base_id_usage,
)
from src_A1.protocols import create_initial_eval_folder




def main(args: argparse.Namespace) -> None:

    # Check the signature-target options
    if args.target_basis == "signatures":
        if not args.signature_sets:
            raise ValueError(
                "--target_basis signatures requires at least one --signature_sets "
                "(external and/or internal)."
            )
        if "external" in args.signature_sets and not args.eval_signatures_id:
            raise ValueError("--signature_sets external requires --eval_signatures_id.")
        if "internal" in args.signature_sets and not args.eval_signatures_INT_id:
            raise ValueError("--signature_sets internal requires --eval_signatures_INT_id.")


    eval_root = mkdir(resolve_eval_task_root())
    eval_dir = eval_root / args.eval_id

    print(f"Setting up evaluation folder for eval_id: {args.eval_id}")

    validate_G_base_id_usage(
        eval_configuration=args.eval_configuration,
        sub_eval_configuration=args.sub_eval_configuration,
        G_base_id=args.G_base_id,
    )

    if args.sub_eval_configuration in CONTROL_SUB_EVAL_CONFIGS:
        if args.reference_eval_id is None or str(args.reference_eval_id).strip() == "":
            raise ValueError(
                f"{args.sub_eval_configuration} is a control protocol and requires "
                "--reference_eval_id pointing to an existing STHELAR_TRAIN_CV eval_id."
            )
        if str(args.reference_eval_id) == str(args.eval_id):
            raise ValueError(
                "A control child eval_id cannot use itself as reference_eval_id."
            )
    
    if eval_dir.exists() and not args.overwrite:
        raise FileExistsError(f"Eval folder already exists: {eval_dir}. Use --overwrite to refresh settings.")
    
    if args.overwrite and eval_dir.exists():
        rmtree_appledouble_safe(eval_dir)
    
    eval_dir.mkdir(parents=True, exist_ok=True)
    cfg = create_initial_eval_folder(
        eval_dir=eval_dir,
        eval_id=args.eval_id,
        dataset_id=args.dataset_id,
        training_dataset_id=args.training_dataset_id,
        eval_configuration=args.eval_configuration,
        sub_eval_configuration=args.sub_eval_configuration,
        level=args.level,
        G_base_id=args.G_base_id,
        gene_property_id=args.gene_property_id,
        eval_signatures_id=args.eval_signatures_id,
        eval_signatures_INT_id=args.eval_signatures_INT_id,
        target_basis=args.target_basis,
        signature_sets=args.signature_sets,
        modelgroup=args.modelgroup,
        scgpt_model_name=args.scgpt_model_name,
        novae_training_id=args.novae_training_id,
        novae_domain_n=args.novae_domain_n,
        protocol_seed=None,
        reference_eval_id=args.reference_eval_id,
    )
    
    print(f"Created eval settings folder: {eval_dir / 'settings'}")
    print(f"Main config: {eval_dir / 'settings' / 'eval_config.yaml'}")
    print("eval_config.yaml can be edited before running _2_fill_eval_settings.py, "
          "not after.")




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create the Task A.1 evaluation folder and initial eval_config.yaml.")
    
    parser.add_argument("--eval_id", type=str,
                        required=True,
                        help="Evaluation ID, e.g. 'eval_1'")
    
    parser.add_argument("--dataset_id", type=str,
                        required=True,
                        help="Dataset ID, e.g. 'ds_ct1_20x_224_o112'")
    
    parser.add_argument("--training_dataset_id", type=str,
                        required=True,
                        help="Training dataset ID, e.g. 'ds_ct1_20x_224_o112_1'")
    
    parser.add_argument("--eval_configuration", type=str,
                        required=True,
                        choices=["STHELAR_TRAIN", "STHELAR_TEST", "HEST"],
                        help="Evaluation configuration")
    
    parser.add_argument("--sub_eval_configuration", type=str,
                        required=True,
                        choices=["STHELAR_TRAIN_CV", "STHELAR_TRAIN_COORDS",
                                 "STHELAR_TEST_ONE_SHOT", "STHELAR_TEST_ADAPTED",
                                 "HEST_CV_CLOSE", "HEST_CV_MAGNIFICATION", "HEST_CV_SPECIES", "HEST_ORGANS"],
                        help="Sub-evaluation configuration")
    
    parser.add_argument("--level", type=str,
                        default="patch_all",
                        choices=["patch_all"],
                        help=("Supervision level (only patch_all: H&E patch vs the full ST content of the patch)."))
    
    parser.add_argument("--G_base_id", type=str,
                        required=True,
                        help=("Gene panel: a G_base ID (e.g. 'G_base_1'), or 'auto' "
                              "(only for HEST_CV_* and STHELAR_TEST_ADAPTED)."))

    parser.add_argument("--gene_property_id", type=str,
                        required=True,
                        help=("Gene-property table ID built by _12_gene_properties.py "
                              "(e.g. 'gene_property_1'), used for the gene strata."))

    parser.add_argument("--eval_signatures_id", type=str,
                        default=None,
                        help=("External gene-signature panel ID built by _13_eval_signatures.py "
                              "(e.g. 'eval_signatures_1'). Default: None (no signature stratum)."))

    parser.add_argument("--eval_signatures_INT_id", type=str,
                        default=None,
                        help=("Internal gene-signature panel ID built by _14_eval_signatures_INT.py "
                              "(e.g. 'eval_signatures_INT_1'), used when --signature_sets includes 'internal'."))

    parser.add_argument("--target_basis", type=str,
                        default="genes",
                        choices=["genes", "signatures"],
                        help=("Prediction target: 'genes' (default) or 'signatures' "
                              "(per-signature scores over --signature_sets)."))

    parser.add_argument("--signature_sets", type=str,
                        nargs="*",
                        default=None,
                        choices=["external", "internal"],
                        help=("Signature sets predicted with --target_basis signatures: 'external' "
                              "(--eval_signatures_id) and/or 'internal' (--eval_signatures_INT_id)."))

    # The ridge alpha grid is set in eval_config.yaml (protocol.ridge.alpha_grid)

    parser.add_argument("--modelgroup", type=str,
                        required=True,
                        help=("Model group (only 'hoptimus0_hoptimus1_uni2h_virchow2')."))
    
    parser.add_argument("--scgpt_model_name", type=str,
                        default=None,
                        choices=["scGPT_human", "scGPT_pancancer"],
                        help="scGPT model name (default: scGPT_human; control evals inherit the reference value).")
    
    parser.add_argument("--novae_training_id", type=str,
                        default=None,
                        help="Novae training ID whose patch-domain labels are added to the STHELAR "
                             "row manifests, e.g. 'training_1'.")

    parser.add_argument("--novae_domain_n", type=int,
                        default=None,
                        help="Novae domain resolution, e.g. 15.")

    parser.add_argument("--reference_eval_id", type=str,
                        default=None,
                        help=(
                            "Required only for the STHELAR_TRAIN_COORDS control protocol. "
                            "Must point to an existing STHELAR_TRAIN/STHELAR_TRAIN_CV eval_id."
                        ))
    
    parser.add_argument("--overwrite", 
                        action="store_true",
                        help="Overwrite existing evaluation folder")
    
    args = parser.parse_args()
    main(args)
