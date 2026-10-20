import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

from .common import get_full_path


def _load_vocab(vocab_path: Path) -> Dict[str, int]:
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    return {
        str(k): int(v)
        for k, v in vocab.items()
        if not (str(k).startswith("<") and str(k).endswith(">"))
    }


def _scgpt_vocab_path(model_name: str) -> Path:
    model_dir = Path(get_full_path("train", "pretrained_foundation_models", model_name, as_str=True))
    return model_dir / "vocab.json"


@lru_cache(maxsize=4)
def load_scgpt_vocab(model_name: str) -> tuple:
    """Sorted scGPT gene vocabulary without special tokens (used only to filter the gene panel)."""
    vocab = _load_vocab(_scgpt_vocab_path(model_name))
    return tuple(sorted(vocab.keys()))


def scgpt_vocab_manifest(model_name: str) -> Dict[str, Any]:
    """Provenance record for the scGPT vocabulary used as a gene-name filter."""
    genes: List[str] = list(load_scgpt_vocab(model_name))
    blob = "\n".join(genes).encode("utf-8")
    return {
        "model_name": str(model_name),
        "usage": "gene_vocabulary_filtering_only",
        "vocab_path": str(_scgpt_vocab_path(model_name)),
        "n_vocab_genes": int(len(genes)),
        "vocab_sha256": hashlib.sha256(blob).hexdigest(),
    }
