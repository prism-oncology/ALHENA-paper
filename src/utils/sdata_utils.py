"""
Shared SpatialData helpers.
"""


def promote_20x_to_default(sdata, key: str = "he") -> None:
    """
    Replace `sdata.images[key]` in memory by a tree whose 'scale0' is the original 'scale1' (20x instead of 40x),
    so that SOPA's `_return_element(sdata, key, ..., as_spatial_image=True)` returns the 20x image (no data copy).
    """
    from xarray import DataTree  # lazy: only the sdata-handling envs need xarray

    img_tree: DataTree = sdata.images[key]              # original multiscale tree

    # Build a wrapper whose only child is 'scale0' -> original 'scale1'
    wrapper = DataTree(name=key)
    wrapper["scale0"] = img_tree["scale1"]              # just a pointer, no data copy
    wrapper.attrs.update(img_tree.attrs)                # keep transforms & metadata

    # Overwrite the entry with the wrapper
    sdata.images[key] = wrapper
