#!/usr/bin/env python
"""
Fix malformed HEST .h5ad gene metadata for the HEST slides listed in metadata_eval_st_<training_dataset_id>.csv: prefix cleanup,
Ensembl-ID var_names replaced by gene symbols (symbol column or Ensembl REST), duplicated symbols summed, X stored as CSR.

Saved in <ROOT_HistoST> / eval / eval_datasets / st_eval_data / hest /
    - st / <slide_id>.h5ad: corrected files
    - OLD / <slide_id>.h5ad: original files
"""

import argparse
import math
import re
import shutil
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
import requests
from anndata import read_h5ad
from scipy import sparse
from tqdm import tqdm

# Add the project root to sys.path so that 'src' can be imported
project_root = Path(__file__).resolve().parents[4]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.constants import get_full_path


# -----------------------------------------------------------------------------
# Regex / constants
# -----------------------------------------------------------------------------

# Ensembl gene id, optionally versioned (e.g. ENSG00000141510, ENSMUSG00000051951.2)
ENSEMBL_GENE_ID_RE = re.compile(r"^ENS[A-Z0-9]*G\d+(?:\.\d+)?$", re.IGNORECASE)

# "prefix + separator + rest" patterns of malformed HEST gene names
PREFIX_PATTERNS = [
    re.compile(r"^(?P<token>[A-Za-z0-9._-]+)(?P<sep>_{2,})(?P<rest>.+)$"),
    re.compile(r"^(?P<token>[A-Za-z0-9._-]+)(?P<sep>[|:])(?P<rest>.+)$"),
]

MISSING_STRINGS = {"", "na", "n/a", "nan", "none", "null", "<na>"}

# Genome/build prefix tokens to strip (in addition to .var["genome"])
FALLBACK_PREFIX_TOKENS = {
    "mm10", "mm39",
    "hg19", "hg38",
    "grch37", "grch38",
    "grcm38", "grcm39",
    "rn6", "rn7",
    "susscrofa11.1", "susscr11",
    "canfam3", "canfam4",
    "danrer10", "danrer11",
    "galgal6", "galgal7",
}

# Ordered candidates for gene-symbol columns to promote into adata.var_names
# when the current index is Ensembl-like
SYMBOL_COLUMN_CANDIDATES = (
    "SYMBOL",
    "symbol",
    "gene_symbol",
    "gene_symbols",
    "gene_name",
    "genes",
) 

# Ensembl REST fallback used when var_names are Ensembl-like and no usable
# symbol column exists in .var
ENSEMBL_REST_BASE_URL = "https://rest.ensembl.org"
ENSEMBL_LOOKUP_BATCH_SIZE = 500
ENSEMBL_REST_TIMEOUT = (10, 60)  # (connect_timeout, read_timeout)
ENSEMBL_REST_MAX_RETRIES = 3
# If Ensembl REST resolves a reasonable fraction of genes, keep unresolved ones
# as their original Ensembl IDs instead of failing
ENSEMBL_REST_MIN_RESOLVED_FRACTION = 0.70

METADATA_SPECIES_TO_ENSEMBL = {
    "homo sapiens": "homo_sapiens",
    "homo_sapiens": "homo_sapiens",
    "human": "homo_sapiens",
    "mus musculus": "mus_musculus",
    "mus_musculus": "mus_musculus",
    "mouse": "mus_musculus",
}

# Cache symbols across slides to avoid repeated REST calls for the same gene IDs
ENSEMBL_SYMBOL_CACHE: dict[tuple[str, str], Optional[str]] = {}

# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------

def plural(n: int, singular: str, plural_form: Optional[str] = None) -> str:
    if n == 1:
        return singular
    return plural_form if plural_form is not None else singular + "s"


def safe_to_string(x) -> str:
    """
    Convert to string in a predictable way.
    """
    if isinstance(x, bytes):
        return x.decode("utf-8", errors="replace")
    return str(x)


def normalize_text_value(x) -> Tuple[str, bool, bool]:
    """
    Returns:
        normalized_string, decoded_bytes, trimmed_whitespace
    """
    decoded_bytes = isinstance(x, bytes)
    s = safe_to_string(x)
    stripped = s.strip()
    trimmed_whitespace = stripped != s
    return stripped, decoded_bytes, trimmed_whitespace


def looks_missing_string(s: str) -> bool:
    return s.strip().lower() in MISSING_STRINGS


def choose_backup_path(dst: Path) -> Path:
    """
    Backup path that does not overwrite an existing backup (adds .bak<i>).
    """
    if not dst.exists():
        return dst

    stem = dst.stem
    suffix = dst.suffix
    parent = dst.parent
    i = 1
    while True:
        candidate = parent / f"{stem}.bak{i}{suffix}"
        if not candidate.exists():
            return candidate
        i += 1


def make_h5ad_safe_nullable_string_series(series: pd.Series) -> pd.Series:
    """Convert a string column with missing values to Categorical (writable to .h5ad)."""
    out = series.copy()
    out = out.astype("object")
    out = out.map(lambda x: pd.NA if pd.isna(x) else str(x))
    return out.astype("category")


def normalize_nullable_string_series_for_compare(series: pd.Series) -> pd.Series:
    """Values as a nullable object Series, for dtype-independent comparisons (e.g. category vs object after a rerun)."""
    return pd.Series(
        [pd.NA if pd.isna(x) else str(x) for x in series.tolist()],
        index=series.index,
        dtype="object",
    )


def matrix_looks_like_raw_counts(matrix, atol: float = 1e-8) -> bool:
    """True if the matrix looks like raw counts (finite, non-negative, integer-valued), required to sum duplicated genes."""
    if sparse.issparse(matrix):
        data = matrix.data
    else:
        data = np.asarray(matrix)

    if data.size == 0:
        return True

    if not np.isfinite(data).all():
        return False

    if (data < 0).any():
        return False

    if np.issubdtype(data.dtype, np.integer):
        return True

    return np.allclose(data, np.round(data), atol=atol, rtol=0.0)


def is_likely_same_custom_feature(gene_id: str, var_name: str) -> bool:
    """
    Accept non-Ensembl custom features that clearly belong to the same
    accession/namespace, e.g.

        gene_id  = NC_045512.2_gene-GU280_gp01
        var_name = NC_045512.2_ORF1ab_1
    """
    gene_id = safe_to_string(gene_id).strip()
    var_name = safe_to_string(var_name).strip()

    if gene_id == var_name:
        return True

    if ENSEMBL_GENE_ID_RE.fullmatch(gene_id):
        return False
    if ENSEMBL_GENE_ID_RE.fullmatch(var_name):
        return False

    # Shared accession prefix like "NC_045512.2_"
    m = re.match(r"^(?P<prefix>[A-Za-z]{1,8}_[A-Za-z0-9.]+)_", gene_id)
    if m:
        prefix = m.group("prefix")
        if var_name.startswith(prefix + "_"):
            return True

    return False



# -----------------------------------------------------------------------------
# Change recording
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class ChangeKey:
    field: str   # "var_names" or "gene_ids"
    kind: str    # e.g. "removed_prefix", "trimmed_whitespace"
    detail: str  # e.g. "mm10___________"


class ChangeRecorder:
    def __init__(self) -> None:
        self.counts: Counter[ChangeKey] = Counter()
        self.examples: defaultdict[ChangeKey, List[str]] = defaultdict(list)

    def add(
        self,
        field: str,
        kind: str,
        detail: str = "",
        before: Optional[str] = None,
        after: Optional[str] = None,
    ) -> None:
        key = ChangeKey(field=field, kind=kind, detail=detail)
        self.counts[key] += 1

        if before is not None and len(self.examples[key]) < 3:
            if after is None:
                ex = f"{before}"
            else:
                ex = f"{before} -> {after}"
            if ex not in self.examples[key]:
                self.examples[key].append(ex)

    def merge(self, other: "ChangeRecorder") -> None:
        for key, n in other.counts.items():
            self.counts[key] += n
            for ex in other.examples.get(key, []):
                if len(self.examples[key]) < 3 and ex not in self.examples[key]:
                    self.examples[key].append(ex)

    def is_empty(self) -> bool:
        return len(self.counts) == 0

    def items(self):
        return self.counts.items()

    def get_examples(self, key: ChangeKey) -> List[str]:
        return self.examples.get(key, [])


# -----------------------------------------------------------------------------
# Prefix inference / cleanup
# -----------------------------------------------------------------------------

def collect_allowed_prefix_tokens(var_df: pd.DataFrame) -> set[str]:
    """
    Allowed prefix tokens:
      - values from .var['genome'] if present
      - fallback set above
    """
    tokens = set(FALLBACK_PREFIX_TOKENS)
    if "genome" in var_df.columns:
        series = var_df["genome"]
        for x in series.dropna().tolist():
            s, _, _ = normalize_text_value(x)
            if s:
                tokens.add(s.lower())
    return tokens


def infer_strip_prefixes(values: Sequence, allowed_tokens: set[str]) -> set[str]:
    """Prefixes to strip: token in allowed_tokens, or exact prefix of >= 80% of the non-missing values (e.g. "mm10___", "hg38|")."""
    normalized: List[str] = []
    for x in values:
        if pd.isna(x):
            continue
        s, _, _ = normalize_text_value(x)
        if s:
            normalized.append(s)

    if not normalized:
        return set()

    exact_prefix_counts: Counter[str] = Counter()
    prefix_token: dict[str, str] = {}

    for s in normalized:
        for pat in PREFIX_PATTERNS:
            m = pat.match(s)
            if m:
                prefix = f"{m.group('token')}{m.group('sep')}"
                token = m.group("token").lower()
                exact_prefix_counts[prefix] += 1
                prefix_token[prefix] = token
                break

    systematic_threshold = max(3, math.ceil(0.80 * len(normalized)))
    prefixes_to_strip = set()

    for prefix, count in exact_prefix_counts.items():
        token = prefix_token[prefix]
        if token in allowed_tokens or count >= systematic_threshold:
            prefixes_to_strip.add(prefix)

    return prefixes_to_strip


def strip_known_prefixes(
    s: str,
    prefixes_to_strip: set[str],
    recorder: ChangeRecorder,
    field: str,
) -> str:
    """
    Repeatedly strip any known exact prefix from the beginning of s.
    """
    if not prefixes_to_strip:
        return s

    ordered_prefixes = sorted(prefixes_to_strip, key=len, reverse=True)

    out = s
    while True:
        matched = None
        for prefix in ordered_prefixes:
            if out.startswith(prefix) and len(out) > len(prefix):
                matched = prefix
                break

        if matched is None:
            break

        new_out = out[len(matched):].strip()
        recorder.add(
            field=field,
            kind="removed_prefix",
            detail=matched,
            before=out,
            after=new_out,
        )
        out = new_out

    return out

 
def ensembl_like_fraction(values: Sequence[str]) -> float:
    names = [safe_to_string(x).strip() for x in values if not pd.isna(x)]
    if not names:
        return 0.0
    return sum(bool(ENSEMBL_GENE_ID_RE.fullmatch(x)) for x in names) / len(names)


def is_ensembl_like_majority(values: Sequence[str], threshold: float = 0.80) -> bool:
    return ensembl_like_fraction(values) > threshold


def normalize_species_for_ensembl(species: str) -> str:
    """
    Normalize metadata species labels to Ensembl REST species aliases.
    """
    if pd.isna(species):
        raise ValueError("Species is missing in metadata.")

    raw = safe_to_string(species).strip()
    normalized = METADATA_SPECIES_TO_ENSEMBL.get(raw.lower())
    if normalized is None:
        raise ValueError(
            "Unsupported species value in metadata. "
            f"Got '{raw}', expected one of: 'Homo sapiens' or 'Mus musculus'."
        )
    return normalized


def strip_ensembl_version(gene_id: str) -> str:
    """Remove the version suffix of an Ensembl id (ENSG....12 -> ENSG...), for the REST lookup."""
    gene_id = safe_to_string(gene_id).strip()
    return re.sub(r"\.\d+$", "", gene_id)


def batched(values: Sequence[str], batch_size: int) -> Iterable[List[str]]:
    for i in range(0, len(values), batch_size):
        yield list(values[i:i + batch_size])


def fetch_ensembl_lookup_batch(
    ensembl_ids: Sequence[str],
    species: str,
) -> dict:
    """
    POST /lookup/id for a batch of Ensembl stable IDs.
    """
    if not ensembl_ids:
        return {}

    url = f"{ENSEMBL_REST_BASE_URL}/lookup/id"
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    params = {
        "species": species,
    }
    payload = {
        "ids": list(ensembl_ids),
    }

    last_error = None
    for attempt in range(1, ENSEMBL_REST_MAX_RETRIES + 1):
        try:
            response = requests.post(
                url,
                params=params,
                json=payload,
                headers=headers,
                timeout=ENSEMBL_REST_TIMEOUT,
            )

            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After")
                try:
                    sleep_s = float(retry_after) if retry_after is not None else float(attempt)
                except ValueError:
                    sleep_s = float(attempt)
                time.sleep(sleep_s)
                continue

            response.raise_for_status()
            decoded = response.json()
            if not isinstance(decoded, dict):
                raise ValueError(f"Unexpected response type from Ensembl REST: {type(decoded).__name__}")
            return decoded

        except (requests.RequestException, ValueError) as e:
            last_error = e
            if attempt == ENSEMBL_REST_MAX_RETRIES:
                break
            time.sleep(float(attempt))

    raise RuntimeError(
        "Ensembl REST lookup/id failed after "
        f"{ENSEMBL_REST_MAX_RETRIES} attempts for species='{species}'. "
        f"Last error: {last_error}"
    )


def resolve_symbol_names_via_ensembl_rest(
    ensembl_ids: Sequence[str],
    species: str,
) -> Tuple[pd.Index, ChangeRecorder]:
    """
    Resolve Ensembl gene IDs to gene symbols via Ensembl REST. Unresolved IDs are kept as is; raises if the
    resolved fraction is below ENSEMBL_REST_MIN_RESOLVED_FRACTION.
    """
    recorder = ChangeRecorder()
    species_alias = normalize_species_for_ensembl(species)
    print(f"  [Ensembl REST] resolving gene symbols for species='{species_alias}'")

    cleaned_ids = [safe_to_string(x).strip() for x in ensembl_ids]
    invalid_ids = [x for x in cleaned_ids if not ENSEMBL_GENE_ID_RE.fullmatch(x)]
    if invalid_ids:
        raise ValueError(
            "Cannot resolve symbols via Ensembl REST because some cleaned var_names "
            f"are not valid Ensembl gene IDs. Examples: {invalid_ids[:10]}"
        )

    base_ids = [strip_ensembl_version(x) for x in cleaned_ids]
    unresolved_base_ids = sorted(
        {x for x in base_ids if (species_alias, x) not in ENSEMBL_SYMBOL_CACHE}
    )
    print(
        f"  [Ensembl REST] total ids={len(base_ids)}, "
        f"cached={len(base_ids) - len(unresolved_base_ids)}, "
        f"to_fetch={len(unresolved_base_ids)}"
    )

    for batch_ids in tqdm(
        batched(unresolved_base_ids, ENSEMBL_LOOKUP_BATCH_SIZE),
        total=math.ceil(len(unresolved_base_ids) / ENSEMBL_LOOKUP_BATCH_SIZE) if unresolved_base_ids else 0,
        desc=f"Ensembl REST [{species_alias}]",
        leave=False,
    ):
        decoded = fetch_ensembl_lookup_batch(batch_ids, species_alias)

        for base_id in batch_ids:
            entry = decoded.get(base_id)
            symbol = None

            if isinstance(entry, dict):
                entry_species = safe_to_string(entry.get("species", "")).strip().lower()
                raw_symbol = entry.get("display_name")

                if raw_symbol is not None and (not entry_species or entry_species == species_alias):
                    symbol = safe_to_string(raw_symbol).strip()
                    if looks_missing_string(symbol):
                        symbol = None

            ENSEMBL_SYMBOL_CACHE[(species_alias, base_id)] = symbol

    resolved_symbols: List[str] = []
    unresolved_original_ids: List[str] = []

    detail = f"Ensembl REST lookup [{species_alias}]"
    for original_id, base_id in zip(cleaned_ids, base_ids):
        symbol = ENSEMBL_SYMBOL_CACHE.get((species_alias, base_id))
        if symbol is None:
            # Keep the original Ensembl ID when REST cannot resolve it
            resolved_symbols.append(original_id)
            unresolved_original_ids.append(original_id)
            continue

        resolved_symbols.append(symbol)
        if original_id != symbol:
            recorder.add(
                field="var_names",
                kind="replaced_ensembl_index_with_symbol",
                detail=detail,
                before=original_id,
                after=symbol,
            )

    n_total = len(resolved_symbols)
    n_unresolved = len(unresolved_original_ids)
    n_resolved = n_total - n_unresolved
    resolved_fraction = (n_resolved / n_total) if n_total else 0.0

    print(
        f"  [Ensembl REST] resolved={n_resolved}, "
        f"unresolved={n_unresolved}, "
        f"resolved_fraction={resolved_fraction:.1%}"
    )

    if resolved_fraction < ENSEMBL_REST_MIN_RESOLVED_FRACTION:
        raise ValueError(
            "var_names look like Ensembl IDs and no usable symbol column was found, "
            "and Ensembl REST mapping quality is too low to trust. "
            f"Species='{species_alias}', resolved={n_resolved}/{n_total} "
            f"({resolved_fraction:.1%}), minimum_required={ENSEMBL_REST_MIN_RESOLVED_FRACTION:.1%}, "
            f"unresolved examples={unresolved_original_ids[:10]}"
        )

    resolved_index = pd.Index(resolved_symbols, dtype="object", name=None)
    validate_gene_names(resolved_index, allow_duplicates=True)
    return resolved_index, recorder


# -----------------------------------------------------------------------------
# Validation
# -----------------------------------------------------------------------------

def validate_gene_names(
    var_names: Sequence[str],
    allow_duplicates: bool = False,
) -> None:
    """
    Raise if the gene names do not look like gene symbols (empty, duplicated, Ensembl-like, numeric or prefixed).
    """
    names = [safe_to_string(x).strip() for x in var_names]
    n = len(names)

    if n == 0:
        raise ValueError("adata.var_names is empty.")

    empty_idx = [i for i, x in enumerate(names) if x == ""]
    if empty_idx:
        raise ValueError(
            f"Found {len(empty_idx)} empty gene {plural(len(empty_idx), 'name')} in var_names "
            f"(first indices: {empty_idx[:10]})."
        )

    idx = pd.Index(names)
    duplicated = idx[idx.duplicated()].unique().tolist()
    if duplicated and not allow_duplicates:
        raise ValueError(
            "Duplicate gene names detected in var_names after cleanup. "
            f"Examples: {duplicated[:10]}"
        )

    ensembl_like = sum(bool(ENSEMBL_GENE_ID_RE.fullmatch(x)) for x in names)
    numeric_like = sum(x.isdigit() for x in names)
    suspicious_prefixed = sum(bool(re.match(r"^[A-Za-z0-9._-]+_{2,}.+$", x)) for x in names)

    if ensembl_like / n > 0.80:
        examples = [x for x in names if ENSEMBL_GENE_ID_RE.fullmatch(x)][:5]
        raise ValueError(
            "var_names still look like Ensembl IDs rather than gene symbols. "
            f"Examples: {examples}"
        )

    if numeric_like / n > 0.80:
        examples = [x for x in names if x.isdigit()][:5]
        raise ValueError(
            "var_names still look numeric rather than gene symbols. "
            f"Examples: {examples}"
        )

    if suspicious_prefixed / n > 0.50:
        examples = [x for x in names if re.match(r"^[A-Za-z0-9._-]+_{2,}.+$", x)][:5]
        raise ValueError(
            "var_names still look systematically prefixed after cleanup. "
            f"Examples: {examples}"
        )
 
 
def clean_symbol_names_from_column(
    var_df: pd.DataFrame,
    column_name: str,
) -> Tuple[pd.Index, ChangeRecorder]:
    """
    Clean a candidate gene-symbol column so it can be promoted to adata.var_names.
    Raises if symbols are missing.
    """
    recorder = ChangeRecorder()
    allowed_tokens = collect_allowed_prefix_tokens(var_df)
    prefixes_to_strip = infer_strip_prefixes(var_df[column_name].tolist(), allowed_tokens)

    cleaned: List[str] = []
    missing_rows: List[int] = []

    for i, raw in enumerate(var_df[column_name].tolist()):
        if pd.isna(raw):
            cleaned.append("")
            missing_rows.append(i)
            continue

        s, decoded_bytes, trimmed_whitespace = normalize_text_value(raw)

        if decoded_bytes:
            recorder.add("var_names", "decoded_bytes", before=s)
        if trimmed_whitespace:
            recorder.add("var_names", "trimmed_whitespace", before=safe_to_string(raw), after=s)

        s = strip_known_prefixes(
            s=s,
            prefixes_to_strip=prefixes_to_strip,
            recorder=recorder,
            field="var_names",
        )

        if looks_missing_string(s):
            cleaned.append("")
            missing_rows.append(i)
            continue

        cleaned.append(s)

    if missing_rows:
        raise ValueError(
            f"Column '{column_name}' cannot be used to replace Ensembl-like var_names "
            f"because it contains {len(missing_rows)} missing/empty symbol values "
            f"(first row indices: {missing_rows[:10]})."
        )

    return pd.Index(cleaned, dtype="object", name=None), recorder


def choose_symbol_names_for_var_index(
    var_df: pd.DataFrame,
    cleaned_ensembl_index: pd.Index,
    species: str,
) -> Tuple[pd.Index, str, ChangeRecorder]:
    present = [c for c in SYMBOL_COLUMN_CANDIDATES if c in var_df.columns]

    errors: List[str] = []
    for col in present:
        try:
            cleaned_symbols, recorder = clean_symbol_names_from_column(var_df, col)
            validate_gene_names(cleaned_symbols, allow_duplicates=True)
            return cleaned_symbols, col, recorder
        except ValueError as e:
            errors.append(f"{col}: {e}")

    # Fallback: recover symbols directly from Ensembl gene IDs using metadata species
    try:
        cleaned_symbols, recorder = resolve_symbol_names_via_ensembl_rest(
            ensembl_ids=cleaned_ensembl_index.tolist(),
            species=species,
        )
        return cleaned_symbols, f"Ensembl REST lookup [{normalize_species_for_ensembl(species)}]", recorder
    except (ValueError, RuntimeError) as e:
        errors.append(f"Ensembl REST: {e}")

    if not present:
        raise ValueError(
            "var_names look like Ensembl IDs, but no symbol column is available to replace them. "
            f"Looked for: {list(SYMBOL_COLUMN_CANDIDATES)} | "
            + " | ".join(errors)
        )

    raise ValueError(
        "var_names look like Ensembl IDs, but no usable symbol column was found. "
        + " | ".join(errors)
    )



# -----------------------------------------------------------------------------
# Cleaning logic
# -----------------------------------------------------------------------------

def clean_var_names(
    var_df: pd.DataFrame,
    species: str,
) -> Tuple[pd.Index, pd.Index, bool, ChangeRecorder]:
    recorder = ChangeRecorder()
    allowed_tokens = collect_allowed_prefix_tokens(var_df)
    prefixes_to_strip = infer_strip_prefixes(var_df.index.tolist(), allowed_tokens)

    cleaned: List[str] = []
    for raw in var_df.index.tolist():
        if pd.isna(raw):
            s = ""
            cleaned.append(s)
            continue

        s, decoded_bytes, trimmed_whitespace = normalize_text_value(raw)

        if decoded_bytes:
            recorder.add("var_names", "decoded_bytes", before=s)
        if trimmed_whitespace:
            recorder.add("var_names", "trimmed_whitespace", before=safe_to_string(raw), after=s)

        s2 = strip_known_prefixes(
            s=s,
            prefixes_to_strip=prefixes_to_strip,
            recorder=recorder,
            field="var_names",
        )
        cleaned.append(s2)

    cleaned_index = pd.Index(cleaned, dtype="object", name=None)

    # Ensembl-like index: replace it with gene symbols
    if is_ensembl_like_majority(cleaned_index):
        symbol_names, symbol_source, symbol_recorder = choose_symbol_names_for_var_index(
            var_df=var_df,
            cleaned_ensembl_index=cleaned_index,
            species=species,
        )
        recorder.merge(symbol_recorder)

        for before, after in zip(cleaned_index.tolist(), symbol_names.tolist()):
            if before != after:
                recorder.add(
                    "var_names",
                    "replaced_ensembl_index_with_symbol",
                    detail=symbol_source,
                    before=before,
                    after=after,
                )

        # Duplicates allowed here (collapsed later)
        validate_gene_names(symbol_names, allow_duplicates=True)
        return symbol_names, cleaned_index, True, recorder

    # Duplicates allowed here as well (collapsed later)
    validate_gene_names(cleaned_index, allow_duplicates=True)
    return cleaned_index, cleaned_index, False, recorder


def clean_gene_ids(
    var_df: pd.DataFrame,
    cleaned_var_names: Sequence[str],
) -> Tuple[pd.Series, ChangeRecorder]:
    recorder = ChangeRecorder()
    allowed_tokens = collect_allowed_prefix_tokens(var_df)
    prefixes_to_strip = infer_strip_prefixes(var_df["gene_ids"].tolist(), allowed_tokens)

    cleaned: List[object] = []

    for raw, clean_var_name in zip(var_df["gene_ids"].tolist(), cleaned_var_names):
        if pd.isna(raw):
            cleaned.append(pd.NA)
            continue

        s, decoded_bytes, trimmed_whitespace = normalize_text_value(raw)

        if decoded_bytes:
            recorder.add("gene_ids", "decoded_bytes", before=s)
        if trimmed_whitespace:
            recorder.add("gene_ids", "trimmed_whitespace", before=safe_to_string(raw), after=s)

        s = strip_known_prefixes(
            s=s,
            prefixes_to_strip=prefixes_to_strip,
            recorder=recorder,
            field="gene_ids",
        )

        if looks_missing_string(s):
            recorder.add("gene_ids", "missing_string_to_na", before=s, after="<NA>")
            cleaned.append(pd.NA)
            continue

        if not ENSEMBL_GENE_ID_RE.fullmatch(s):
            if s == "Unspecified_gene_id":
                recorder.add(
                    "gene_ids",
                    "non_ensembl_unspecified_to_na",
                    before=s,
                    after="<NA>",
                )
                cleaned.append(pd.NA)
                continue

            # Custom (non-Ensembl) features: gene_ids set to NA
            if is_likely_same_custom_feature(s, clean_var_name):
                recorder.add(
                    "gene_ids",
                    "non_ensembl_custom_feature_to_na",
                    before=s,
                    after="<NA>",
                )
                cleaned.append(pd.NA)
                continue

            raise ValueError(
                "Invalid non-Ensembl gene_id after cleanup that does not match "
                "a recognized custom feature pattern: "
                f"gene_id='{s}', var_name='{clean_var_name}'"
            )

        cleaned.append(s)

    return pd.Series(cleaned, index=var_df.index, dtype="object"), recorder


def collapse_duplicate_symbol_features(
    adata,
    target_var_names: pd.Index,
    cleaned_gene_ids: Optional[pd.Series],
) -> Tuple[object, ChangeRecorder]:
    """Collapse duplicated gene symbols by summing their columns in adata.X (raw counts, no layers/raw/varm/varp)."""
    recorder = ChangeRecorder()

    target_index = pd.Index(
        [safe_to_string(x).strip() for x in target_var_names.tolist()],
        dtype="object",
        name=None,
    )

    if not target_index.has_duplicates:
        return adata, recorder

    duplicated_examples = target_index[target_index.duplicated()].unique().tolist()[:10]

    if list(adata.layers.keys()):
        raise ValueError(
            "Duplicate gene symbols are present after cleanup, but adata.layers is not "
            "empty. Refusing to collapse only .X because that would leave layers "
            f"inconsistent. Examples: {duplicated_examples}"
        )

    if adata.raw is not None:
        raise ValueError(
            "Duplicate gene symbols are present after cleanup, but adata.raw is present. "
            "Refusing to collapse only .X because that would leave adata.raw "
            f"inconsistent. Examples: {duplicated_examples}"
        )

    if list(adata.varm.keys()) or list(adata.varp.keys()):
        raise ValueError(
            "Duplicate gene symbols are present after cleanup, but adata.varm or "
            "adata.varp is present. Refusing to collapse duplicated features without "
            f"var-axis-aware aggregation for those annotations. Examples: {duplicated_examples}"
        )

    if not matrix_looks_like_raw_counts(adata.X):
        raise ValueError(
            "Duplicate gene symbols are present after cleanup, but adata.X does not "
            "look like raw integer counts. Refusing to sum duplicated features because "
            "that would be unsafe for normalized or log-transformed values. "
            f"Examples: {duplicated_examples}"
        )

    codes, unique_symbols = pd.factorize(target_index, sort=False)
    first_mask = ~target_index.duplicated(keep="first")

    # Sum duplicated columns in X
    if sparse.issparse(adata.X):
        x_csr = adata.X.tocsr()
        assignment = sparse.csr_matrix(
            (np.ones(adata.n_vars, dtype=np.int64), (np.arange(adata.n_vars), codes)),
            shape=(adata.n_vars, len(unique_symbols)),
        )
        collapsed_X = x_csr @ assignment
    else:
        x_arr = np.asarray(adata.X)
        collapsed_dtype = np.result_type(x_arr.dtype, np.int64)
        collapsed_X = np.zeros((adata.n_obs, len(unique_symbols)), dtype=collapsed_dtype)
        for old_idx, new_idx in enumerate(codes):
            collapsed_X[:, new_idx] += x_arr[:, old_idx]

    grouped_var = (
        adata.var.copy()
        .assign(__target_var_name=target_index)
        .groupby("__target_var_name", sort=False, observed=True)
    )

    collapsed_var = grouped_var.first()
    collapsed_var.index = pd.Index(unique_symbols, dtype="object", name=None)
    collapsed_var["n_source_features"] = grouped_var.size().astype("int64")

    if cleaned_gene_ids is not None:
        gene_ids_by_symbol = (
            pd.Series(cleaned_gene_ids.tolist(), index=target_index, dtype="object")
            .groupby(level=0, sort=False)
            .agg(list)
        )

        def _normalize_gene_id_list(values: Sequence[object]) -> List[str]:
            out: List[str] = []
            seen: set[str] = set()
            for x in values:
                if pd.isna(x):
                    continue
                s = safe_to_string(x).strip()
                if not s or looks_missing_string(s):
                    continue
                if s not in seen:
                    seen.add(s)
                    out.append(s)
            return out

        normalized_gene_ids = gene_ids_by_symbol.map(_normalize_gene_id_list)

        # All source Ensembl IDs, joined by ';'
        collapsed_var["source_gene_ids"] = make_h5ad_safe_nullable_string_series(
            normalized_gene_ids.map(lambda xs: ";".join(xs) if xs else pd.NA)
        )

        # gene_ids: the Ensembl ID if unique, else <NA>
        collapsed_var["gene_ids"] = make_h5ad_safe_nullable_string_series(
            normalized_gene_ids.map(lambda xs: xs[0] if len(xs) == 1 else pd.NA)
        )

        group_sizes = grouped_var.size()
        for symbol, group_n in group_sizes.items():
            if group_n <= 1:
                continue
            ids = normalized_gene_ids.loc[symbol]
            before = ";".join(ids) if ids else f"{group_n} source features"
            recorder.add(
                field="var_names",
                kind="collapsed_duplicate_symbols",
                detail="summed_raw_counts",
                before=before,
                after=symbol,
            )
            if len(ids) > 1:
                recorder.add(
                    field="gene_ids",
                    kind="multi_ensembl_symbol_to_na",
                    before=";".join(ids),
                    after=symbol,
                )
    else:
        group_sizes = grouped_var.size()
        for symbol, group_n in group_sizes.items():
            if group_n > 1:
                recorder.add(
                    field="var_names",
                    kind="collapsed_duplicate_symbols",
                    detail="summed_raw_counts",
                    before=f"{group_n} source features",
                    after=symbol,
                )

    collapsed = adata[:, first_mask].copy()
    collapsed.X = collapsed_X
    collapsed.var = collapsed_var
    collapsed.var_names = collapsed_var.index

    validate_gene_names(collapsed.var_names)
    return collapsed, recorder


def normalize_x_to_sparse_csr_inplace(adata) -> Tuple[bool, ChangeRecorder]:
    """Convert adata.X to CSR sparse if needed. Returns (changed, recorder)."""
    recorder = ChangeRecorder()
    X = adata.X

    if sparse.isspmatrix_csr(X):
        return False, recorder

    if sparse.issparse(X):
        before_type = type(X).__name__
        X_csr = X.tocsr()
        X_csr.sum_duplicates()
        X_csr.eliminate_zeros()
        adata.X = X_csr
        recorder.add(
            field="X",
            kind="converted_sparse_to_csr",
            detail=before_type,
            before=before_type,
            after="csr_matrix",
        )
        return True, recorder

    x_arr = np.asarray(X)
    if x_arr.ndim != 2:
        raise ValueError(f"adata.X is dense but not 2D: shape={x_arr.shape}")
    if not np.issubdtype(x_arr.dtype, np.number):
        raise ValueError(f"adata.X is dense but not numeric: dtype={x_arr.dtype}")

    X_csr = sparse.csr_matrix(x_arr)
    X_csr.sum_duplicates()
    X_csr.eliminate_zeros()
    adata.X = X_csr
    recorder.add(
        field="X",
        kind="converted_dense_to_csr",
        detail=str(x_arr.dtype),
        before=f"ndarray[{x_arr.dtype}]",
        after="csr_matrix",
    )
    return True, recorder


# -----------------------------------------------------------------------------
# I/O helpers
# -----------------------------------------------------------------------------

def replace_with_backup(src_h5ad: Path, corrected_adata, old_dir: Path) -> Path:
    """
    Replace src_h5ad by the corrected AnnData (written to a temp file first); the original is moved to old_dir
    and restored on failure. Returns the backup path.
    """
    old_dir.mkdir(parents=True, exist_ok=True)

    backup_path = choose_backup_path(old_dir / src_h5ad.name)
    tmp_path = src_h5ad.with_name(f".{src_h5ad.stem}.tmp_fixed.h5ad")

    if tmp_path.exists():
        tmp_path.unlink()

    corrected_adata.write_h5ad(tmp_path)

    moved_original = False
    try:
        shutil.move(str(src_h5ad), str(backup_path))
        moved_original = True
        shutil.move(str(tmp_path), str(src_h5ad))
    except Exception:
        if moved_original and backup_path.exists() and not src_h5ad.exists():
            shutil.move(str(backup_path), str(src_h5ad))
        raise
    finally:
        if tmp_path.exists():
            tmp_path.unlink()

    return backup_path


# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------

def format_change_line(key: ChangeKey, n: int, examples: Sequence[str]) -> str:
    if key.kind == "removed_prefix":
        msg = f"{key.field}: removed prefix '{key.detail}' from {n} {plural(n, 'entry', 'entries')}"
    elif key.kind == "trimmed_whitespace":
        msg = f"{key.field}: trimmed whitespace in {n} {plural(n, 'entry', 'entries')}"
    elif key.kind == "decoded_bytes":
        msg = f"{key.field}: decoded bytes in {n} {plural(n, 'entry', 'entries')}"
    elif key.kind == "missing_string_to_na":
        msg = f"{key.field}: turned {n} missing-like string {plural(n, 'value')} into <NA>"
    elif key.kind == "non_ensembl_custom_feature_to_na":
        msg = (
            f"{key.field}: turned {n} non-Ensembl custom feature "
            f"{plural(n, 'identifier')} into <NA>"
        )
    elif key.kind == "non_ensembl_unspecified_to_na":
        msg = (
            f"{key.field}: turned {n} 'Unspecified_gene_id' string "
            f"{plural(n, 'value')} into <NA>"
        )
    elif key.kind == "replaced_ensembl_index_with_symbol":
        if key.detail.startswith("Ensembl REST lookup"):
            msg = (
                f"{key.field}: replaced Ensembl-like index with gene symbols via "
                f"{key.detail} in {n} {plural(n, 'entry', 'entries')}"
            )
        else:
            msg = (
                f"{key.field}: replaced Ensembl-like index with gene symbols from column "
                f"'{key.detail}' in {n} {plural(n, 'entry', 'entries')}"
            )
    elif key.kind == "created_from_ensembl_var_names":
        msg = (
            f"{key.field}: created from original Ensembl-like var_names for "
            f"{n} {plural(n, 'entry', 'entries')}"
        )
    elif key.kind == "collapsed_duplicate_symbols":
        msg = (
            f"{key.field}: collapsed {n} duplicated gene-symbol "
            f"{plural(n, 'group')} by summing raw counts in .X"
        )
    elif key.kind == "multi_ensembl_symbol_to_na":
        msg = (
            f"{key.field}: set gene_ids to <NA> for {n} collapsed symbol "
            f"{plural(n, 'group')} backed by multiple Ensembl IDs"
        )
    elif key.kind == "converted_dense_to_csr":
        msg = (
            f"{key.field}: converted dense matrix to CSR sparse format in "
            f"{n} {plural(n, 'slide')}"
        )
    elif key.kind == "converted_sparse_to_csr":
        msg = (
            f"{key.field}: normalized non-CSR sparse matrix to CSR format in "
            f"{n} {plural(n, 'slide')}"
        )
    else:
        msg = f"{key.field}: {key.kind} ({key.detail}) x {n}"

    if examples:
        msg += f" | examples: {examples}"
    return msg


# -----------------------------------------------------------------------------
# Main per-slide processing
# -----------------------------------------------------------------------------

def process_one_slide(
    slide_id: str,
    species: str,
    st_dir: Path,
    old_dir: Path,
    global_counts: Counter[ChangeKey],
    global_slides: defaultdict[ChangeKey, set[str]],
) -> str:
    """Fix one HEST slide in place. Returns "unchanged" or "modified"."""
    src_h5ad = st_dir / f"{slide_id}.h5ad"
    if not src_h5ad.exists():
        raise FileNotFoundError(f"Expected HEST slide file not found: {src_h5ad}")

    print(f"\n--- Slide {slide_id} ---")
    adata = read_h5ad(src_h5ad)

    if adata.var is None or adata.var.shape[0] == 0:
        raise ValueError(f"{slide_id}: adata.var is empty.")

    slide_recorder = ChangeRecorder()

    # Clean var_names
    cleaned_var_names, original_cleaned_var_names, original_index_was_ensembl_like, var_recorder = clean_var_names(
        adata.var,
        species=species,
    )
    slide_recorder.merge(var_recorder)

    # Clean gene_ids if present
    cleaned_gene_ids = None
    if "gene_ids" in adata.var.columns:
        cleaned_gene_ids, gene_ids_recorder = clean_gene_ids(
            adata.var,
            cleaned_var_names=cleaned_var_names,
        )
        slide_recorder.merge(gene_ids_recorder)
    else:
        if original_index_was_ensembl_like:
            invalid_examples = [
                x for x in original_cleaned_var_names.tolist()
                if not ENSEMBL_GENE_ID_RE.fullmatch(x)
            ][:5]
            if invalid_examples:
                raise ValueError(
                    "var_names look Ensembl-like and gene_ids is absent, but the original "
                    f"var_names are not all valid Ensembl gene IDs. Examples: {invalid_examples}"
                )

            cleaned_gene_ids = pd.Series(
                original_cleaned_var_names.tolist(),
                index=adata.var.index,
                dtype="object",
            )
            for gene_id in original_cleaned_var_names.tolist():
                slide_recorder.add(
                    "gene_ids",
                    "created_from_ensembl_var_names",
                    before=gene_id,
                )
        else:
            print("  gene_ids column absent -> left unchanged.")

    duplicate_target_symbols = pd.Index(cleaned_var_names).has_duplicates

    if duplicate_target_symbols:
        # Collapse duplicated gene symbols (sum of raw counts)

        adata, collapse_recorder = collapse_duplicate_symbol_features(
            adata=adata,
            target_var_names=cleaned_var_names,
            cleaned_gene_ids=cleaned_gene_ids,
        )
        slide_recorder.merge(collapse_recorder)
        var_names_changed = True
        gene_ids_changed = True

    else:
        var_names_changed = not cleaned_var_names.equals(adata.var_names)
        gene_ids_changed = False

        if cleaned_gene_ids is not None:
            if "gene_ids" in adata.var.columns:
                # Compare normalized values (dtype may differ on reruns)
                existing_gene_ids_norm = normalize_nullable_string_series_for_compare(
                    adata.var["gene_ids"]
                )
                cleaned_gene_ids_norm = normalize_nullable_string_series_for_compare(
                    cleaned_gene_ids
                )
                gene_ids_changed = not cleaned_gene_ids_norm.equals(existing_gene_ids_norm)
            else:
                gene_ids_changed = True

        # Apply simple non-collapsing changes
        if var_names_changed:
            adata.var_names = cleaned_var_names
        if gene_ids_changed and cleaned_gene_ids is not None:
            # Series rebuilt on the current var index (var_names may have changed above)
            gene_ids_for_write = pd.Series(
                cleaned_gene_ids.array,
                index=adata.var.index,
                dtype="object",
            )
            adata.var["gene_ids"] = make_h5ad_safe_nullable_string_series(gene_ids_for_write)

    # Store X as CSR sparse
    x_changed, x_recorder = normalize_x_to_sparse_csr_inplace(adata)
    slide_recorder.merge(x_recorder)

    if not var_names_changed and not gene_ids_changed and not x_changed:
        print("  [OK] no change needed.")
        return "unchanged"

    # Print detailed local changes
    print("  [MODIFIED]")
    for key, n in sorted(
        slide_recorder.items(),
        key=lambda x: (x[0].field, x[0].kind, x[0].detail)
    ):
        examples = slide_recorder.get_examples(key)
        print("   - " + format_change_line(key, n, examples))
        global_counts[key] += n
        global_slides[key].add(slide_id)

    backup_path = replace_with_backup(src_h5ad=src_h5ad, corrected_adata=adata, old_dir=old_dir)
    print(f"  original moved to: {backup_path}")
    print(f"  corrected file written to: {src_h5ad}")

    return "modified"


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main(args) -> None:

    training_dataset_id = args.training_dataset_id

    metadata_csv = get_full_path("eval", "eval_datasets", "st_eval_data", f"eval_st_{training_dataset_id}", f"metadata_eval_st_{training_dataset_id}.csv")
    hest_root = get_full_path("eval", "eval_datasets", "st_eval_data", "hest")
    st_dir = hest_root / "st"
    old_dir = hest_root / "OLD"

    if not metadata_csv.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {metadata_csv}")
    if not st_dir.exists():
        raise FileNotFoundError(f"HEST ST directory not found: {st_dir}")

    print(f"Reading metadata: {metadata_csv}")
    metadata = pd.read_csv(metadata_csv)

    required_cols = {"slide_id", "source", "species"}
    missing_cols = required_cols - set(metadata.columns)
    if missing_cols:
        raise KeyError(f"Metadata CSV is missing required columns: {sorted(missing_cols)}")

    hest_metadata = metadata.loc[
        metadata["source"].astype(str).str.upper() == "HEST",
        ["slide_id", "species"],
    ].copy()

    hest_metadata["slide_id"] = hest_metadata["slide_id"].astype(str)

    species_by_slide: dict[str, str] = {}
    for slide_id, species_values in hest_metadata.groupby("slide_id")["species"]:
        unique_species = {
            safe_to_string(x).strip()
            for x in species_values.tolist()
            if not pd.isna(x) and safe_to_string(x).strip() != ""
        }

        if len(unique_species) != 1:
            raise ValueError(
                f"Slide '{slide_id}' has ambiguous or missing species values in metadata: "
                f"{sorted(unique_species)}"
            )

        species_value = next(iter(unique_species))
        normalize_species_for_ensembl(species_value)  # fail early on unexpected labels
        species_by_slide[slide_id] = species_value

    hest_slides = sorted(species_by_slide.keys())

    if not hest_slides:
        print("No HEST slides found in metadata.")
        return

    print(f"Found {len(hest_slides)} HEST {plural(len(hest_slides), 'slide')} in metadata.")
    print(f"HEST ST dir: {st_dir}")
    print(f"Backup dir : {old_dir}")

    unchanged = 0
    modified = 0

    global_counts: Counter[ChangeKey] = Counter()
    global_slides: defaultdict[ChangeKey, set[str]] = defaultdict(set)

    for slide_id in tqdm(hest_slides, desc="Processing slides"):
        status = process_one_slide(
            slide_id=slide_id,
            species=species_by_slide[slide_id],
            st_dir=st_dir,
            old_dir=old_dir,
            global_counts=global_counts,
            global_slides=global_slides,
        )
        if status == "unchanged":
            unchanged += 1
        elif status == "modified":
            modified += 1
        else:
            raise RuntimeError(f"Unexpected slide status: {status}")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Slides not changed : {unchanged}")
    print(f"Slides modified    : {modified}")

    if global_counts:
        print("\nUnique corrected issue types:")
        for key, total_n in sorted(global_counts.items(), key=lambda x: (x[0].field, x[0].kind, x[0].detail)):
            slide_list = sorted(global_slides[key])
            slide_preview = ", ".join(slide_list[:10])
            if len(slide_list) > 10:
                slide_preview += ", ..."
            examples = []
            # summary kept compact: no examples here
            line = format_change_line(key, total_n, examples)
            print(f" - {line} | slides: {len(slide_list)} [{slide_preview}]")
    else:
        print("\nNo formatting issue was corrected.")

    print("\nDone.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=("Fix malformed HEST .h5ad gene metadata for the HEST slides listed in metadata_eval_st_{training_dataset_id}.csv."))
    
    parser.add_argument("--training_dataset_id", type=str,
                        required=True,
                        help="Training dataset ID, e.g. ds_ct1_20x_224_o112_1")
    
    args = parser.parse_args()
    main(args)