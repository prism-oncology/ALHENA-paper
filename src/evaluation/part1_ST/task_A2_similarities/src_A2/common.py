import hashlib
import json
import logging
import os
import random
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence
import numpy as np
import pandas as pd
import yaml


DEFAULT_CHUNK_SIZE = 1024 * 1024



def _find_project_root() -> Path | None:
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        if (parent / "src" / "utils" / "constants.py").exists():
            return parent
    return None

PROJECT_ROOT = _find_project_root()

def ensure_project_import() -> None:
    if PROJECT_ROOT is None:
        return
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

def src_import(name_function: str):
    ensure_project_import()
    if name_function == "get_full_path":
        from src.utils.constants import get_full_path
        return get_full_path
    elif name_function == "BACKBONE_TRANSFORM":
        from src.utils.models_settings import BACKBONE_TRANSFORM
        return BACKBONE_TRANSFORM
    elif name_function == "load_exported_he_encoder":
        from src.evaluation.extract_encoders import load_exported_he_encoder
        return load_exported_he_encoder
    elif name_function == "STCellClipEncoder":
        # ST cell token-branch encoder (shared with _10_compute_STembs.py)
        from src.evaluation.extract_encoders import STCellClipEncoder
        return STCellClipEncoder
    elif name_function == "infer_scconcept_embedding_dim":
        # scConcept input width (shared with the exporter)
        from src.evaluation.extract_encoders import infer_scconcept_embedding_dim
        return infer_scconcept_embedding_dim
    else:
        raise ValueError(f"Unknown function to import: {name_function}")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def resolve_device(device: Optional[str]) -> str:
    if device is None or str(device).lower() == "auto":
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except Exception:
            pass
        return "cpu"
    return str(device)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


class JsonNumpyEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        if isinstance(obj, Path):
            return str(obj)
        try:
            import torch
            if isinstance(obj, torch.device):
                return str(obj)
        except ImportError:
            pass
        return super().default(obj)


def write_json(path: Path, payload: Mapping[str, Any], indent: int = 2) -> None:
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(payload, f, indent=indent, cls=JsonNumpyEncoder)
    os.replace(tmp, path)


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r") as f:
        return json.load(f)


def write_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        yaml.safe_dump(dict(payload), f, sort_keys=False)
    os.replace(tmp, path)


def read_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r") as f:
        out = yaml.safe_load(f)
    return {} if out is None else out


def write_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def write_dataframe(df: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    if path.suffix == ".csv":
        tmp = path.with_suffix(path.suffix + ".tmp")
        df.to_csv(tmp, index=False)
        os.replace(tmp, path)
    elif path.suffix == ".parquet":
        tmp = path.with_suffix(path.suffix + ".tmp")
        df.to_parquet(tmp, index=False)
        os.replace(tmp, path)
    else:
        raise ValueError(f"Unsupported dataframe output suffix: {path.suffix}")


def sha256_file(path: Path, chunk_size: int = DEFAULT_CHUNK_SIZE) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def hash_mapping(payload: Mapping[str, Any]) -> str:
    data = json.dumps(payload, sort_keys=True, cls=JsonNumpyEncoder).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def copy_if_missing(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(src)
    if not dst.exists():
        ensure_dir(dst.parent)
        shutil.copy2(src, dst)


def configure_logger(log_path: Path, name: str = "A2_eval", level: int = logging.INFO) -> logging.Logger:
    ensure_dir(log_path.parent)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    fh = logging.FileHandler(log_path)
    fh.setLevel(level)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(level)
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger


@dataclass(frozen=True)
class CanonicalGeneRules:
    strip_whitespace: bool = True
    uppercase: bool = False
    strip_ensembl_version: bool = False


def canonicalize_gene_ids(values: Sequence[Any], rules: CanonicalGeneRules) -> List[str]:
    out: List[str] = []
    for v in values:
        s = str(v)
        if rules.strip_whitespace:
            s = s.strip()
        if rules.strip_ensembl_version and s.startswith("ENSG") and "." in s:
            s = s.split(".", 1)[0]
        if rules.uppercase:
            s = s.upper()
        out.append(s)
    return out


def assert_no_duplicates(values: Sequence[str], context: str) -> None:
    idx = pd.Index(values)
    if idx.has_duplicates:
        dup = idx[idx.duplicated()].unique().tolist()[:20]
        raise ValueError(f"Duplicate values after canonicalization in {context}. Examples: {dup}")


def file_info(path: Path, compute_sha256: bool = True) -> Dict[str, Any]:
    st = path.stat()
    return {
        "path": str(path),
        "size_bytes": int(st.st_size),
        "mtime": float(st.st_mtime),
        "mtime_ns": int(st.st_mtime_ns),
        "sha256": sha256_file(path) if compute_sha256 else None,
    }


def summarize_counts(series: pd.Series) -> Dict[str, int]:
    vc = series.astype(str).value_counts(dropna=False).sort_index()
    return {str(k): int(v) for k, v in vc.items()}


def coerce_bool_series(series: pd.Series, context: str) -> pd.Series:
    """Parse a boolean AnnData obs column (strings such as "False" or "0" are parsed as False)."""
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)

    if pd.api.types.is_numeric_dtype(series):
        bad = ~series.isin([0, 1, 0.0, 1.0]) & series.notna()
        if bad.any():
            examples = series.loc[bad].head(10).tolist()
            raise ValueError(f"{context} contains non-boolean numeric values. Examples: {examples}")
        return series.fillna(0).astype(int).astype(bool)

    s = series.astype(str).str.strip().str.lower()
    true_values = {"true", "t", "1", "yes", "y"}
    false_values = {"false", "f", "0", "no", "n", "", "nan", "none", "null"}
    ok = s.isin(true_values | false_values)
    if not ok.all():
        examples = series.loc[~ok].head(10).tolist()
        raise ValueError(f"{context} contains unparseable boolean values. Examples: {examples}")
    return s.isin(true_values)


def softmax_numpy(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    x = x - np.max(x, axis=axis, keepdims=True)
    np.exp(x, out=x)
    denom = np.sum(x, axis=axis, keepdims=True)
    denom = np.maximum(denom, 1e-12)
    x /= denom
    return x


def entropy_from_probs(p: np.ndarray, axis: int = -1) -> np.ndarray:
    p = np.asarray(p, dtype=np.float32)
    safe = np.clip(p, 1e-12, 1.0)
    return -np.sum(safe * np.log(safe), axis=axis)
