"""
Get sdata .zarr from Xenium data.
Built for the STHELAR dataset: opens the Xenium slides and keeps only the data the HistoST project needs.
Save in <ROOT_HistoST> / 'STHELAR_HistoST' / 'sdata_processed' / 'sdata_{slide_id}.zarr'.
H&E and ST are aligned using the provided alignment matrix.
"""

from pathlib import Path
import sys
import argparse
import os
import spatialdata_io
import sopa
import spatialdata as sd
from typing import Dict

# Add the project root to sys.path so that 'src' can be imported
project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.constants import get_full_path
from src.utils.cell_ids import str_cell_id_init



def relabel_cell_ids(sdata: sd.SpatialData) -> sd.SpatialData:
    """
    Create a stable 1:1 mapping from the current table_cells order to new
    Xenium Explorer-style IDs and apply it consistently to:
      - tables['table_cells'].obs['cell_id'] (column)
      - shapes['nucleus_boundaries'].index
      - shapes['cell_boundaries'].index
    """
    # Pull elements
    cells = sdata.tables["table_cells"]              # AnnData
    nuc   = sdata.shapes["nucleus_boundaries"]       # GeoDataFrame, index = cell_id
    cell  = sdata.shapes["cell_boundaries"]          # GeoDataFrame, index = cell_id

    # Build mapping based on the CURRENT row order of table_cells
    old_ids_table = cells.obs["cell_id"].astype(str).tolist()
    new_ids_table = [str_cell_id_init(i) for i in range(cells.n_obs)]
    id_map: Dict[str, str] = dict(zip(old_ids_table, new_ids_table))

    # Safety: every shape id must be present in the table (should hold after filtering)
    missing_nuc  = set(nuc.index.astype(str))  - set(id_map.keys())
    missing_cell = set(cell.index.astype(str)) - set(id_map.keys())
    if missing_nuc or missing_cell:
        raise ValueError(
            f"Relabel failure: shapes contain IDs not present in table_cells.\n"
            f"  nucleus_boundaries missing: {len(missing_nuc)}\n"
            f"  cell_boundaries    missing: {len(missing_cell)}"
        )

    # Apply mapping (preserve row order; ensure dtype=str)
    cells.obs["cell_id"] = cells.obs["cell_id"].astype(str).map(id_map)
    nuc.index  = nuc.index.astype(str).map(id_map)
    cell.index = cell.index.astype(str).map(id_map)

    # Sanity checks
    assert cells.obs["cell_id"].is_unique, "New table cell_ids are not unique."
    assert nuc.index.is_unique, "New nucleus_boundaries index is not unique."
    assert cell.index.is_unique, "New cell_boundaries index is not unique."

    # Write back (objects are mutated in place, but keep symmetry)
    sdata.tables["table_cells"] = cells
    sdata.shapes["nucleus_boundaries"] = nuc
    sdata.shapes["cell_boundaries"]    = cell
    print(f"Relabeled cell_ids -> new format (a..p strings + '-1') for {cells.n_obs:,} cells.")
    return sdata



def filter_sdata_cells(sdata: sd.SpatialData) -> sd.SpatialData:
    """Filter SpatialData object to keep only cells that have entries in all three elements:
    - table_cells
    - nucleus_boundaries
    - cell_boundaries
    Args:
        sdata (sd.SpatialData): Input SpatialData object.
    Returns:
        sd.SpatialData: Filtered SpatialData object.
    """

    # Pull elements
    cells = sdata.tables["table_cells"]              # AnnData
    nuc   = sdata.shapes["nucleus_boundaries"]       # GeoDataFrame, index = cell_id
    cell  = sdata.shapes["cell_boundaries"]          # GeoDataFrame, index = cell_id

    # Ensure unique cell_ids in the table
    if not cells.obs["cell_id"].is_unique:
        cells = cells[~cells.obs.duplicated("cell_id", keep="first")].copy()

    # Build intersection (use strings to avoid dtype mismatches)
    ids_table = set(cells.obs["cell_id"].astype(str).values)
    ids_nuc   = set(nuc.index.astype(str).values)
    ids_cell  = set(cell.index.astype(str).values)

    common_ids = ids_table & ids_nuc & ids_cell
    print(f"Common cell_ids: {len(common_ids):,}")

    # Filter each element (preserve original dtypes/order with boolean masks)
    mask_nuc  = nuc.index.astype(str).isin(common_ids)
    mask_cell = cell.index.astype(str).isin(common_ids)
    mask_tab  = cells.obs["cell_id"].astype(str).isin(common_ids)

    nuc_f   = nuc.loc[mask_nuc].copy()
    cell_f  = cell.loc[mask_cell].copy()
    cells_f = cells[mask_tab].copy()

    print(
        f"nucleus_boundaries: {len(nuc)} -> {len(nuc_f)}\n"
        f"cell_boundaries:    {len(cell)} -> {len(cell_f)}\n"
        f"table_cells:        {cells.n_obs} -> {cells_f.n_obs}"
    )

    # Write back into the SpatialData object
    sdata.shapes["nucleus_boundaries"] = nuc_f
    sdata.shapes["cell_boundaries"]    = cell_f
    sdata.tables["table_cells"]        = cells_f

    return sdata



def main(args):

    # Get full paths
    slide_id = args.slide_id
    st_dir = get_full_path("STHELAR_HistoST", "data_xenium10X", slide_id, args.st_dir, as_str=True)
    he_path = get_full_path("STHELAR_HistoST", "data_xenium10X", slide_id, args.he_file, as_str=True)
    align_matrix_path = get_full_path("STHELAR_HistoST", "data_xenium10X", slide_id, args.align_matrix_path, as_str=True) if args.align_matrix_path is not None else None
    
    # Load ST data and rename
    print("\n## Loading data ##")
    sdata = spatialdata_io.xenium(path=st_dir,
                                    n_jobs=1, transcripts=True, nucleus_boundaries=True, cells_boundaries=True,
                                    cells_table=True, aligned_images=False, nucleus_labels=False, cells_labels=False, cells_as_circles=False)
    sdata.images['morpho'] = sdata.images.pop("morphology_focus")
    sdata.points['st'] = sdata.points.pop("transcripts")
    print(sdata)

    # Rename table as table_cells
    sdata.tables['table_cells'] = sdata.tables.pop('table')

    # Filter sdata to keep only cells with all 3 elements
    print("\n## Filtering sdata to keep only complete cells ##")
    sdata = filter_sdata_cells(sdata)
    print(sdata)

    # Relabel cell_id everywhere (table + shapes) using Xenium Explorer-style labels
    print("\n## Relabeling cell_ids to Xenium Explorer style ##")
    sdata = relabel_cell_ids(sdata)
    print(sdata)

    # Save sdata
    print("\n## Saving initial sdata ##")
    output_dir = get_full_path("STHELAR_HistoST", "sdata_processed", as_str=True)
    output_zarr_path = os.path.join(output_dir, f"sdata_{slide_id}.zarr")
    os.makedirs(output_dir, exist_ok=True)
    sdata.write(output_zarr_path)

    # Loading H&E image
    print("\n## Loading H&E image ##")
    he = sopa.io.wsi(he_path, as_image=True)
    print(he)
    
    # Aligning H&E to sdata
    print("\n## Aligning H&E image to sdata and saving in the zarr ##")
    sopa.io.explorer.align(sdata, he,
                           transformation_matrix_path=align_matrix_path,
                           key_added='he',
                           image_key="morpho")

    # Checking sdata
    print("\n## Checking sdata ##")
    print(sdata)

    print("\nDone.")
    print(f"sdata saved at: {output_zarr_path}")




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process sdata")

    parser.add_argument("--slide_id", type=str, 
                        default='kidney_s2', 
                        help="Slide ID")
    
    parser.add_argument("--st_dir", type=str, 
                        default="Xenium_V1_Human_Kidney_FFPE_Protein_outs",
                        help="Name of ST data directory")
    
    parser.add_argument("--he_file", type=str, 
                        default="Xenium_V1_Human_Kidney_FFPE_Protein_he_image.ome.tif",
                        help="Name of H&E image ome.tif file")
    
    parser.add_argument("--align_matrix_path", type=lambda x : None if x == 'None' else str(x),
                        default="Xenium_V1_Human_Kidney_FFPE_Protein_he_image_alignment_files_me/matrix.csv",
                        help="Relative path to alignment matrix csv file, or 'None' if no alignment should be applied")

    args = parser.parse_args()
    main(args)