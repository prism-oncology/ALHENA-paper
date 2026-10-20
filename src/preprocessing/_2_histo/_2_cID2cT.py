"""
Create one Parquet file per SpatialData slide, mapping every `cell_id` to its annotated cell-type for the given <ct_cat_id>.
Saved in <ROOT_HistoST> / STHELAR_HistoST / cID2cT / <ct_cat_id> / cID2cT_<slide_id>.parquet, where <ct_cat_id> is the cell type category (e.g. ct_1)
"""

import sys
import argparse
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
import spatialdata as sd
import gc
import json

# Add the project root to sys.path so that 'src' can be imported
project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.constants import get_full_path


def main(args):
    """
    Extract <cell_id, cell_type> from one sdata slide
    and write it as a compressed Parquet file.
    """

    # Check that args.annots_table == "table_cells" and args.annots_column == "final_label"
    if args.annots_table != "table_cells":
        raise ValueError(f"annots_table should be 'table_cells'. Got {args.annots_table}")
    if args.annots_column != "final_label": 
        raise ValueError(f"annots_column should be 'final_label'. Got {args.annots_column}")

    # ct_cat_id
    # Check that the cat_dict_dir name follows the expected format "annot_dicts_<ct_cat_id>", e.g. annot_dicts_ct_1 for ct_1
    # and then extract the ct_cat_id from the folder name
    if not args.cat_dict_dir.name.startswith("annot_dicts_"):
        raise ValueError(f"cat_dict_dir name should formatted as 'annot_dicts_<ct_cat_id>', e.g. annot_dicts_ct_1 for ct_1. Got {args.cat_dict_dir.name}")
    ct_cat_id = args.cat_dict_dir.name.replace("annot_dicts_", "")
    print(f"Processing cell type category '{ct_cat_id}'")

    # Paths
    sdata_dir = get_full_path("STHELAR_HistoST", "sdata_final")
    output_dir = get_full_path("STHELAR_HistoST", f"cID2cT/{ct_cat_id}")

    print(f"\n==== Processing slide {args.slide_id} ====")
    
    # Load sdata
    print("\n* Loading sdata...")
    sdata_path = sdata_dir / f"sdata_{args.slide_id}.zarr"
    sdata = sd.read_zarr(sdata_path, selection=("tables",))
    for t in list(sdata.tables):
        if t != args.annots_table:
            del sdata.tables[t]
    gc.collect()
    print(sdata)

    # Load category dictionary
    print("\n* Loading cat_dict...")
    with open(args.cat_dict_dir / 'label2cat.json') as f: 
        label2cat = json.load(f)
    print(label2cat)

    # Assign cat to each cell type
    print("\n* Assigning cell types...")
    ad_obs = sdata.tables[args.annots_table].obs[["cell_id", args.annots_column]].copy()
    ad_obs['cell_type'] = ad_obs[args.annots_column].map(label2cat)
    ad_obs.drop(columns=[args.annots_column], inplace=True)

    # Cast to Categorical -> dictionary-encoded in Parquet
    ad_obs["cell_type"] = ad_obs["cell_type"].astype("category")
    print(ad_obs)

    # Checking results using the count for each cell type (including NaN)
    print("\n* Checking results...")
    print(ad_obs["cell_type"].value_counts(dropna=False))

    # Write to Parquet
    print("\n* Writing Parquet...")
    table = pa.Table.from_pandas(ad_obs, preserve_index=False)
    slide_id = sdata_path.stem.replace("sdata_", "")        # sdata_<ID>.zarr -> <ID>
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"cID2cT_{slide_id}.parquet"
    pq.write_table(
        table,
        out_path,
        compression="zstd",
        use_dictionary=True,
        data_page_size=1 << 20,
    )
    print("Done.")




if __name__ == "__main__":

    p = argparse.ArgumentParser(description="Save <cell_id> -> <cell_type> mapping from SpatialData slide for a given ct_cat_id")
  
    p.add_argument("--annots_table", 
                   default="table_cells", 
                   help="Annotations table to read (always using table_cells)")
    
    p.add_argument("--annots_column", 
                   default="final_label", 
                   help="Column holding the cell type (always using final_label)")
    
    p.add_argument('--cat_dict_dir', type=Path, 
                   default=Path('/path/to/ALHENA-paper/data/annots/annot_dicts_ct_1'), 
                   help='Directory with annot dictionaries for given ct dataset, annot_dicts_<ct_cat_id>, e.g. annot_dicts_ct_1 for ct_1')
    
    p.add_argument("--slide_id", type=str, 
                   required=True, 
                   help="Slide ID to process")

    args = p.parse_args()
    main(args)