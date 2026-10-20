"""
Copied and updated from Novae, itself inspired from squidpy.
Building Delaunay graph from spatial coordinates in anndata object.
Radius and percentile are not used, because subgraphs are extracted later from the cell IDs inside H&E patches.

Saved in <ROOT_HistoST> / STHELAR_HistoST / adata_graph / {slide_id}.h5ad.
Anndata coming from sdata_processed, table_cells => no layers, and .X with RNA counts directly.
See also plots in connectivities folder.
"""

import sys
from pathlib import Path
import os
import gc
import warnings
from enum import Enum
from functools import partial
from itertools import chain
from typing import Iterable, Literal, get_args
import argparse
import numpy as np
from anndata import AnnData
from scipy.sparse import SparseEfficiencyWarning, csr_matrix
from scipy.spatial import Delaunay
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.neighbors import NearestNeighbors
import spatialdata as sd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

# Add the project root to sys.path so that 'src' can be imported
project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.constants import get_full_path


__all__ = ["spatial_neighbors"]

SpatialTechnology = Literal["cosmx", "merscope", "xenium", "visium", "visium_hd"]



class CoordType(Enum):
    GRID = "grid"
    GENERIC = "generic"



def spatial_neighbors(
    adata: AnnData | list[AnnData],
    slide_key: str | None = None,
    radius: tuple[float, float] | float | None = None,
    pixel_size: float | None = None,
    technology: str | SpatialTechnology | None = None,
    coord_type: str | CoordType | None = None,
    n_neighs: int | None = None,
    delaunay: bool | None = None,
    n_rings: int = 1,
    percentile: float | None = None,
    set_diag: bool = False,
    adj_key: str = 'spatial_distances',
    max_mean_distance_ratio: float = 8,
    mean_ngh_th_warning: float = 3.5,
):
    """Create a Delaunay graph from the spatial coordinates of the cells.
    The graph is stored in `adata.obsp['spatial_connectivities']` and `adata.obsp['spatial_distances']`. The long edges
    are removed from the graph according to the `radius` argument (if provided).

    Info:
        This function was updated from [squidpy](https://squidpy.readthedocs.io/en/latest/api/squidpy.gr.spatial_neighbors.html#squidpy.gr.spatial_neighbors).

    Args:
        adata: An `AnnData` object, or a list of `AnnData` objects.
        slide_key: Optional key in `adata.obs` indicating the slide ID of each cell. If provided, the graph is computed for each slide separately.
        radius: `tuple` that prunes the final graph to only contain edges in interval `[min(radius), max(radius)]`. If `float`, uses `[0, radius]`. If `None`, all edges are kept.
        technology: Technology or machine used to generate the spatial data. One of `"cosmx", "merscope", "xenium", "visium", "visium_hd"`. If `None`, uses `adata.obsm["spatial"]`.
        coord_type: Either `"grid"` or `"generic"`. If `"grid"`, the graph is built on a grid. If `"generic"`, the graph is built using the coordinates as they are. By default, uses `"grid"` for Visium/VisiumHD and `"generic"` for other technologies.
        n_neighs: Number of neighbors to consider. If `None`, uses `6` for Visium, `4` for Visium HD, and `None` for generic graphs.
        delaunay: Whether to use Delaunay triangulation to build the graph. If `None`, uses `False` for grid-based graphs and `True` for generic graphs.
        n_rings: See `squidpy.gr.spatial_neighbors` documentation.
        percentile: See `squidpy.gr.spatial_neighbors` documentation.
        set_diag: See `squidpy.gr.spatial_neighbors` documentation.
        adj_key: Key in `adata.obsp` where to store the adjacency matrix.
        max_mean_distance_ratio: Threshold for maximum mean distance ratio between neighbors.
        mean_ngh_th_warning: Threshold for mean number of neighbors.
    """

    if isinstance(adata, list):
        for adata_ in adata:
            spatial_neighbors(
                adata_,
                slide_key=slide_key,
                radius=radius,
                pixel_size=pixel_size,
                technology=technology,
                coord_type=coord_type,
                n_neighs=n_neighs,
                delaunay=delaunay,
                n_rings=n_rings,
                percentile=percentile,
                set_diag=set_diag,
                adj_key=adj_key,
            )
        return

    if isinstance(radius, float) or isinstance(radius, int):
        radius = [0.0, float(radius)]

    assert radius is None or len(radius) == 2, "Radius is expected to be a tuple (min_radius, max_radius)"

    assert pixel_size is None or technology is None, "You must choose argument between `pixel_size` and `technology`"

    if technology == "visium":
        n_neighs = 6 if n_neighs is None else n_neighs
        coord_type, delaunay = CoordType.GRID, False
    elif technology == "visium_hd":
        n_neighs = 8 if n_neighs is None else n_neighs
        coord_type, delaunay = CoordType.GRID, False
    elif technology is not None:
        if "spatial" not in adata.obsm:
            adata.obsm["spatial"] = _technology_coords(adata, technology)
        else:
            print("[WARNING] adata.obsm['spatial'] already exists. Skipping technology coordinates computation.")

    assert ("spatial" in adata.obsm), "Key 'spatial' not found in adata.obsm. This should contain the 2D spatial coordinates of the cells"

    coord_type = CoordType(coord_type or "generic")
    delaunay = True if delaunay is None else delaunay
    n_neighs = 6 if (n_neighs is None and not delaunay) else n_neighs

    print(
        f"Computing graph on {adata.n_obs:,} cells (coord_type={coord_type.value}, {delaunay=}, {radius=}, {n_neighs=})"
    )

    _build_fun = partial(
        _spatial_neighbor,
        coord_type=coord_type,
        n_neighs=n_neighs,
        radius=radius,
        delaunay=delaunay,
        n_rings=n_rings,
        set_diag=set_diag,
        percentile=percentile,
    )

    Adj, Dst = _build_fun(adata)

    adata.obsp["spatial_connectivities"] = Adj
    adata.obsp["spatial_distances"] = Dst

    adata.uns["spatial_neighbors"] = {
        "connectivities_key": "spatial_connectivities",
        "distances_key": "spatial_distances",
        "params": {"radius": radius, "set_diag": set_diag, "n_neighbors": n_neighs, "coord_type": coord_type.value},
    }

    _sanity_check_spatial_neighbors(adata, adj_key, max_mean_distance_ratio, mean_ngh_th_warning)



def _spatial_neighbor(
    adata: AnnData,
    spatial_key: str = "spatial",
    coord_type: str | CoordType | None = None,
    n_neighs: int = 6,
    radius: float | tuple[float, float] | None = None,
    delaunay: bool = False,
    n_rings: int = 1,
    set_diag: bool = False,
    percentile: float | None = None,
) -> tuple[csr_matrix, csr_matrix]:
    coords = adata.obsm[spatial_key]
    assert coords.shape[1] == 2, "Spatial coordinates must be 2D."

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SparseEfficiencyWarning)
        if coord_type == CoordType.GRID:
            Adj, Dst = _build_grid(coords, n_neighs=n_neighs, n_rings=n_rings, delaunay=delaunay, set_diag=set_diag)
        elif coord_type == CoordType.GENERIC:
            Adj, Dst = _build_connectivity(
                coords, n_neighs=n_neighs, radius=radius, delaunay=delaunay, return_distance=True, set_diag=set_diag
            )
        else:
            raise NotImplementedError(f"Coordinate type `{coord_type}` is not yet implemented.")

    if coord_type == CoordType.GENERIC and isinstance(radius, Iterable):
        minn, maxx = sorted(radius)[:2]
        mask = (Dst.data < minn) | (Dst.data > maxx)
        a_diag = Adj.diagonal()

        Dst.data[mask] = 0.0
        Adj.data[mask] = 0.0
        Adj.setdiag(a_diag)

    if percentile is not None and coord_type == CoordType.GENERIC:
        threshold = np.percentile(Dst.data, percentile)
        Adj[Dst > threshold] = 0.0
        Dst[Dst > threshold] = 0.0

    Adj.eliminate_zeros()
    Dst.eliminate_zeros()

    return Adj, Dst



def _build_grid(
    coords: np.ndarray, n_neighs: int, n_rings: int, delaunay: bool = False, set_diag: bool = False
) -> tuple[csr_matrix, csr_matrix]:
    if n_rings > 1:
        Adj: csr_matrix = _build_connectivity(
            coords,
            n_neighs=n_neighs,
            neigh_correct=True,
            set_diag=True,
            delaunay=delaunay,
            return_distance=False,
        )
        Res, Walk = Adj, Adj
        for i in range(n_rings - 1):
            Walk = Walk @ Adj
            Walk[Res.nonzero()] = 0.0
            Walk.eliminate_zeros()
            Walk.data[:] = i + 2.0
            Res = Res + Walk
        Adj = Res
        Adj.setdiag(float(set_diag))
        Adj.eliminate_zeros()

        Dst = Adj.copy()
        Adj.data[:] = 1.0
    else:
        Adj = _build_connectivity(coords, n_neighs=n_neighs, neigh_correct=True, delaunay=delaunay, set_diag=set_diag)
        Dst = Adj.copy()

    Dst.setdiag(0.0)

    return Adj, Dst



def _build_connectivity(
    coords: np.ndarray,
    n_neighs: int,
    radius: float | tuple[float, float] | None = None,
    delaunay: bool = False,
    neigh_correct: bool = False,
    set_diag: bool = False,
    return_distance: bool = False,
) -> csr_matrix | tuple[csr_matrix, csr_matrix]:
    N = coords.shape[0]
    if delaunay:
        tri = Delaunay(coords)
        indptr, indices = tri.vertex_neighbor_vertices
        Adj = csr_matrix((np.ones_like(indices, dtype=np.float64), indices, indptr), shape=(N, N))

        if return_distance:
            # fmt: off
            dists = np.array(list(chain(*(
                euclidean_distances(coords[indices[indptr[i] : indptr[i + 1]], :], coords[np.newaxis, i, :])
                for i in range(N)
                if len(indices[indptr[i] : indptr[i + 1]])
            )))).squeeze()
            Dst = csr_matrix((dists, indices, indptr), shape=(N, N))
            # fmt: on
    else:
        r = 1 if radius is None else radius if isinstance(radius, (int, float)) else max(radius)
        tree = NearestNeighbors(n_neighbors=n_neighs, radius=r, metric="euclidean")
        tree.fit(coords)

        if radius is None:
            dists, col_indices = tree.kneighbors()
            dists, col_indices = dists.reshape(-1), col_indices.reshape(-1)
            row_indices = np.repeat(np.arange(N), n_neighs)
            if neigh_correct:
                dist_cutoff = np.median(dists) * 1.3  # there's a small amount of sway
                mask = dists < dist_cutoff
                row_indices, col_indices, dists = row_indices[mask], col_indices[mask], dists[mask]
        else:
            dists, col_indices = tree.radius_neighbors()
            row_indices = np.repeat(np.arange(N), [len(x) for x in col_indices])
            dists = np.concatenate(dists)
            col_indices = np.concatenate(col_indices)

        Adj = csr_matrix((np.ones_like(row_indices, dtype=np.float64), (row_indices, col_indices)), shape=(N, N))
        if return_distance:
            Dst = csr_matrix((dists, (row_indices, col_indices)), shape=(N, N))

    # radius-based filtering needs same indices/indptr: do not remove 0s
    Adj.setdiag(1.0 if set_diag else Adj.diagonal())
    if return_distance:
        Dst.setdiag(0.0)
        return Adj, Dst

    return Adj



def _technology_coords(adata: AnnData, technology: str) -> np.ndarray:
    VALID_TECHNOLOGIES = list(get_args(SpatialTechnology))
    factor: float = 1.0

    assert technology in VALID_TECHNOLOGIES, f"Invalid `technology` argument. Choose one of {VALID_TECHNOLOGIES}"

    assert (
        "spatial" not in adata.obsm
    ), "Running `spatial_neighbors` with `technology` but `adata.obsm['spatial']` already exists."

    if technology == "cosmx":
        columns = ["CenterX_global_px", "CenterY_global_px"]
        factor = 0.120280945

    if technology == "merscope":
        columns = ["center_x", "center_y"]

    if technology == "xenium":
        columns = ["x_centroid", "y_centroid"]

    assert all(column in adata.obs for column in columns), f"For {technology=}, you must have {columns} in `adata.obs`"

    return adata.obs[columns].values * factor



def _sanity_check_spatial_neighbors(adata: AnnData, adj_key: str, max_mean_distance_ratio: float, mean_ngh_th_warning: float):
    assert adata.obsp[adj_key].getnnz() > 0, "No neighbors found. Please check your `radius` parameter."

    mean_distance = adata.obsp[adj_key].data.mean()
    max_distance = adata.obsp[adj_key].data.max()

    if max_distance / mean_distance > max_mean_distance_ratio:
        print(
            f"[WARNING] The maximum distance between neighbors is {max_distance:.1f}, which is very high compared "
            f"to the mean distance of {mean_distance:.1f}.\n Consider re-running `spatial_neighbors` with a different `radius` threshold."
        )

    mean_ngh = adata.obsp[adj_key].getnnz(axis=1).mean()

    if mean_ngh <= mean_ngh_th_warning:
        print(
            f"[WARNING] The mean number of neighbors is {mean_ngh}, which is very low. Consider re-running `spatial_neighbors` with a different `radius` threshold."
        )




def connectivities(
    slide_id: str,
    adata: AnnData,
    adj_key: str = 'spatial_distances',
    ngh_threshold: int | None = 2,
    cell_size: int | None = 0.2,
    fig_size: tuple[int, int] = (5, 5),
    linewidths: float = 0.1,
    line_color: str = "#333",
    cmap: str = "viridis",
    color_isolated_cells: str = "orangered",
    save_folder: str = None,
    name_png: str = None,
):
    """Show the graph of the spatial connectivities between cells. By default,
    the cells which have a number of neighbors inferior to `ngh_threshold` are shown
    in red. If `ngh_threshold` is `None`, the cells are colored by the number of neighbors.

    !!! info "Quality control"
        This plot is useful to check the quality of the spatial connectivities obtained via [spatial_neighbors][].
        Make sure few cells (e.g., less than 5%) have a number of neighbors below `ngh_threshold`.
        If too many cells are isolated, increase the `radius` parameter in [spatial_neighbors][].
        Conversely, if some cells that are really **far from each other** are still connected, decrease the `radius` parameter to **disconnect** them.

    Args:
        slide_id: Slide ID.
        adata: An AnnData object.
        adj_key: Key in `adata.obsp` where the adjacency matrix is stored.
        ngh_threshold: Only cells with a number of neighbors below this threshold are shown (with color `color_isolated_cells`). If `None`, cells are colored by the number of neighbors.
        cell_size: Size of the dots for each cell. By default, it uses the median distance between neighbor cells.
        fig_size: Size of the figure.
        linewidths: Width of the lines/edges connecting the cells.
        line_color: Color of the lines/edges.
        cmap: Name of the colormap to use for the number of neighbors.
        color_isolated_cells: Color for the cells with a number of neighbors below `ngh_threshold` (if not `None`).
        save_folder: Path to save the plot. If `None`, the plot is not saved.
        name_png: Name of the png file to save the plot. If `None`, the plot is not saved.
    """

    if cell_size is None:
        cell_size = np.median(adata.obsp[adj_key].data)

    fig, ax = plt.subplots(figsize=fig_size)

    assert "spatial" in adata.obsm, "No spatial coordinates found in `adata.obsm['spatial']`"
    assert adj_key in adata.obsp, ("No spatial adjacency found in `adata.obsp['spatial_distances']`.")

    X, A = adata.obsm["spatial"], adata.obsp[adj_key]

    ax.invert_yaxis()
    ax.set_aspect("equal")

    rows, cols = A.nonzero()
    mask = rows < cols
    rows, cols = rows[mask], cols[mask]
    edge_segments = np.stack([X[rows], X[cols]], axis=1)
    edges = LineCollection(edge_segments, color=line_color, linewidths=linewidths, zorder=1)
    ax.add_collection(edges)

    n_neighbors = (A > 0).sum(1).A1

    if ngh_threshold is None:
        _ = ax.scatter(X[:, 0], X[:, 1], c=n_neighbors, s=cell_size, zorder=2, cmap=cmap)
        plt.colorbar(_, ax=ax)
    else:
        isolated_cells = n_neighbors < ngh_threshold
        ax.scatter(X[isolated_cells, 0], X[isolated_cells, 1], color=color_isolated_cells, s=cell_size, zorder=2)

    ax.set_title(f"{slide_id} / Node connectivities" + (f" (threshold={ngh_threshold} neighbors)" if ngh_threshold is not None else ""))

    if save_folder is not None and name_png is not None:
        os.makedirs(save_folder, exist_ok=True)
        plt.savefig(os.path.join(save_folder, name_png))




def main(args):

    # Check that args.table_name == "table_cells" to be compatible with other steps
    if args.table_name != "table_cells":
        raise ValueError(f"`table_name` argument must be 'table_cells' to be compatible with other steps. Got {args.table_name} instead.")

    print(f"\n==== Proccessing {args.slide_id} ====")

    # Paths
    sdata_folder = get_full_path('STHELAR_HistoST', 'sdata_processed', as_str=True)
    output_folder = get_full_path('STHELAR_HistoST', 'adata_graph', as_str=True)

    # Load sdata and get adata
    print(f"\n** Loading sdata from {sdata_folder}...")
    sdata_path = os.path.join(sdata_folder, f"sdata_{args.slide_id}.zarr")
    sdata = sd.read_zarr(sdata_path, selection=('tables',))
    adata = sdata.tables[args.table_name].copy()
    del sdata
    gc.collect()

    # Put cell_id as index
    adata.obs.set_index('cell_id', inplace=True)
    adata.obs.index.name = None
    adata.obs['cell_id'] = adata.obs.index
    print(adata)

    # Compute spatial neighbors
    print("\n** Computing spatial neighbors...")
    spatial_neighbors(adata, technology="xenium", radius=None, percentile=None, max_mean_distance_ratio=args.max_mean_distance_ratio, mean_ngh_th_warning=args.mean_ngh_th_warning)
    print(adata)

    # Plot connectivities
    print("\n** Plotting connectivities...")
    plot_folder = os.path.join(output_folder, "connectivities")
    connectivities(slide_id=args.slide_id, adata=adata, ngh_threshold=args.ngh_threshold, save_folder=plot_folder, name_png=f"{args.slide_id}_connectivities_threshold.png")
    connectivities(slide_id=args.slide_id, adata=adata, ngh_threshold=None, save_folder=plot_folder, name_png=f"{args.slide_id}_connectivities.png")

    # Save adata
    print(f"\n** Saving adata to {output_folder}...")
    os.makedirs(output_folder, exist_ok=True)
    adata.write_h5ad(os.path.join(output_folder, f"{args.slide_id}.h5ad"))

    print("\n==== Done ====")




if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Build graph from spatial coordinates")
    
    parser.add_argument('--slide_id', type=str, default="heart_s0", help="Slide ID")
    parser.add_argument('--table_name', type=str, default="table_cells", help="Name of the table in the sdata zarr, choose table_cell everytime to be compatible with other steps")

    parser.add_argument('--max_mean_distance_ratio', type=float, default=8, help="Threshold for maximum mean distance ratio between neighbors, default is 8")
    parser.add_argument('--mean_ngh_th_warning', type=float, default=3.5, help="Threshold for mean number of neighbors, default is 3.5")
    parser.add_argument('--ngh_threshold', type=int, default=4, help="For connectivity plot / Only cells with a number of neighbors below this threshold are shown (with color `color_isolated_cells`)")

    args = parser.parse_args()
    main(args)

