"""IO helpers of the Task A-0 embedding analyses (atomic writes, row reads, streaming parquet writer)."""

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


# HDF5 chunk cache (size, number of slots) used when the patch-cell matrix is not loaded in RAM
FALLBACK_RDCC_NBYTES = 512 * 1024 ** 2
FALLBACK_RDCC_NSLOTS = 100_003


@contextmanager
def atomic_output(path: Path):
    """
    Yield a temporary sibling path, moved onto `path` on clean exit (os.replace) and removed if the body raises.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        yield tmp
        os.replace(tmp, path)
    except BaseException:
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass
        raise


def take_rows(source, row_indices) -> np.ndarray:
    """
    Read `row_indices` from a 2-D matrix held in RAM or in an open h5py Dataset, in the caller's order.
    Returns a float32 array of shape (len(row_indices), D).
    """
    idx = np.asarray(row_indices)
    if idx.ndim != 1:
        raise ValueError(f"row_indices must be 1-D, got shape {idx.shape}.")
    if idx.size == 0:
        return np.zeros((0, int(source.shape[1])), dtype=np.float32)
    if isinstance(source, np.ndarray):
        return np.asarray(source[idx], dtype=np.float32)
    # h5py Dataset: needs a plain sorted list of Python ints.
    order = np.argsort(idx, kind="mergesort")
    sorted_idx = idx[order]
    block = np.asarray(source[sorted_idx.tolist()], dtype=np.float32)
    if np.array_equal(order, np.arange(idx.size)):
        return block
    inverse = np.empty_like(order)
    inverse[order] = np.arange(order.size)
    return block[inverse]


class StreamingParquetWriter:
    """
    Append per-slide frames to one parquet file, holding at most one frame in memory.
    The schema is set by the first frame; the file is moved into place on `close`.
    """

    def __init__(self, path: Path, enabled: bool = True) -> None:
        self.path = Path(path)
        self.enabled = bool(enabled)
        self.n_rows = 0
        self._writer: Optional[pq.ParquetWriter] = None
        self._tmp: Optional[Path] = None
        self._schema: Optional[pa.Schema] = None

    def write(self, frame: pd.DataFrame) -> None:
        """Append one slide's rows. Empty frames and a disabled writer are no-ops."""
        if not self.enabled or frame is None or frame.empty:
            return
        table = pa.Table.from_pandas(frame, preserve_index=False)
        if self._writer is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._tmp = self.path.with_suffix(self.path.suffix + ".tmp")
            self._schema = table.schema
            self._writer = pq.ParquetWriter(str(self._tmp), self._schema, compression="zstd")
        elif not table.schema.equals(self._schema):
            # Cast to the first schema (raises on a real mismatch)
            table = table.cast(self._schema)
        self._writer.write_table(table)
        self.n_rows += int(frame.shape[0])

    def close(self) -> Optional[Path]:
        """Finalise and atomically move into place. Returns the path, or None if nothing was written."""
        if self._writer is None:
            return None
        self._writer.close()
        self._writer = None
        os.replace(self._tmp, self.path)
        self._tmp = None
        return self.path

    def abort(self) -> None:
        """Drop a partially written file (used when the run fails before `close`)."""
        if self._writer is not None:
            try:
                self._writer.close()
            except Exception:
                pass
            self._writer = None
        if self._tmp is not None:
            try:
                Path(self._tmp).unlink()
            except OSError:
                pass
            self._tmp = None


def matrix_fits_in_memory(dataset, max_bytes: Union[int, float]) -> bool:
    """True when a 2-D h5py Dataset can be materialised under a declared byte budget (as float32)."""
    if int(max_bytes) <= 0:
        return False
    n_rows = int(dataset.shape[0])
    n_cols = int(dataset.shape[1]) if len(dataset.shape) > 1 else 1
    return (n_rows * n_cols * 4) <= int(max_bytes)
