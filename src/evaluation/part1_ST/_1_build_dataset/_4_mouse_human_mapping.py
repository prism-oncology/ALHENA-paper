#!/usr/bin/env python
"""
Build per-slide mouse -> human gene-name mapping files for the Mus musculus HEST eval slides: each var_name (mouse symbol
or Ensembl ID) is mapped to its human one-to-one ortholog via Ensembl REST, or kept unchanged if there is none.
Cached mappings are reused when they match the slide; live Ensembl queries require --allow_live_rest.

Saved in <ROOT_HistoST> / eval / eval_datasets / st_eval_data / hest /
    - <slide_id>_mouse2human_mapping.parquet: one row per var_name (target_gene_name: human symbol, or original name)
    - mouse2human_mapping_cache / <slide_id>_mouse2human_mapping.parquet and .manifest.json: cached mapping and its input signature
"""

import sys
import io
import re
import time
import json
import shutil
import hashlib
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
import requests
from anndata import read_h5ad
from requests.adapters import HTTPAdapter
from tqdm.auto import tqdm
from urllib3.util.retry import Retry


# Add the project root to sys.path so that 'src' can be imported
project_root = Path(__file__).resolve().parents[4]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.constants import get_full_path  # noqa: E402


# -----------------------------
# Config
# -----------------------------

ENSEMBL_REST_BASE_URL = "https://rest.ensembl.org"

# Keep lookup POST payloads reasonably small
REST_LOOKUP_ID_BATCH_SIZE = 500
REST_LOOKUP_SYMBOL_BATCH_SIZE = 500

# Homology is only available as single-ID GET, so use limited concurrency
REST_HOMOLOGY_MAX_WORKERS = 6

MAPPING_CACHE_DIRNAME = "mouse2human_mapping_cache"
MAPPING_CACHE_VERSION = "mouse2human_hest_cache_v1"

REQUEST_TIMEOUT = (20, 120)
SLEEP_BETWEEN_CHUNKS_SEC = 0.05

MOUSE_ENSEMBL_GENE_RE = re.compile(r"^ENSMUSG\d+$")
HUMAN_ENSEMBL_GENE_RE = re.compile(r"^ENSG\d+$")


FINAL_MAPPING_COLUMNS = [
    "slide_id",
    "var_index",
    "original_var_name",
    "input_identifier_type",
    "mouse_symbol_input",
    "mouse_ensembl_gene_id_input",
    "mapping_input_used",
    "ortholog_mapping_status",
    "orthology_type",
    "mouse_gene_name_ensembl",
    "human_ensembl_gene_id",
    "human_gene_name_ensembl",
    "human_gene_name_final",
    "human_gene_name_source",
    "converted_to_human",
    "keep_mouse_name_reason",
    "target_gene_name",
]

SIGNATURE_INPUT_COLUMNS = [
    "var_index",
    "original_var_name",
    "mouse_symbol_input",
    "mouse_ensembl_gene_id_input",
]


# -----------------------------
# Small utils
# -----------------------------

def chunked(seq: Sequence[str], n: int) -> Iterable[List[str]]:
    seq = list(seq)
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def normalize_gene_id(x: object) -> Optional[str]:
    """
    Strip Ensembl version suffix if present.
    Example: ENSMUSG00000012345.7 -> ENSMUSG00000012345
    """
    if x is None:
        return None
    if pd.isna(x):
        return None
    s = str(x).strip()
    if s == "" or s.lower() == "nan":
        return None
    return s.split(".", 1)[0]


def is_mouse_ensembl_gene_id(x: Optional[str]) -> bool:
    return x is not None and bool(MOUSE_ENSEMBL_GENE_RE.match(x))


def is_human_ensembl_gene_id(x: Optional[str]) -> bool:
    return x is not None and bool(HUMAN_ENSEMBL_GENE_RE.match(x))


class EnsemblRestError(RuntimeError):
    pass


def first_nonempty_strict(values: Iterable[Optional[str]]) -> Optional[str]:
    for v in values:
        if v is None:
            continue
        s = str(v).strip()
        if s:
            return s
    return None


def utc_now_iso_z() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def compute_mapping_input_signature(gene_df: pd.DataFrame) -> str:
    """sha256 of the per-gene mapping inputs (SIGNATURE_INPUT_COLUMNS), not of the whole h5ad file."""
    sig_df = gene_df[SIGNATURE_INPUT_COLUMNS].copy()
    sig_df = sig_df.fillna("")

    buf = io.StringIO()
    sig_df.to_csv(
        buf,
        sep="\t",
        index=False,
        header=True,
        lineterminator="\n",
    )
    payload = buf.getvalue().encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def get_cache_paths(hest_root: Path, slide_id: str) -> Tuple[Path, Path]:
    cache_dir = hest_root / MAPPING_CACHE_DIRNAME
    cache_path = cache_dir / f"{slide_id}_mouse2human_mapping.parquet"
    manifest_path = cache_dir / f"{slide_id}_mouse2human_mapping.manifest.json"
    return cache_path, manifest_path


def validate_mapping_df_structure(mapping_df: pd.DataFrame, slide_id: str) -> pd.DataFrame:
    missing = [c for c in FINAL_MAPPING_COLUMNS if c not in mapping_df.columns]
    if missing:
        raise KeyError(
            f"Cached mapping for slide '{slide_id}' is missing required columns: {missing}"
        )

    mapping_df = mapping_df[FINAL_MAPPING_COLUMNS].copy()

    if mapping_df.empty:
        raise ValueError(f"Cached mapping for slide '{slide_id}' is empty.")

    slide_ids = sorted(mapping_df["slide_id"].astype(str).unique().tolist())
    if slide_ids != [slide_id]:
        raise ValueError(
            f"Cached mapping slide_id mismatch for slide '{slide_id}'. "
            f"Found slide_id values: {slide_ids}"
        )

    var_index = mapping_df["var_index"].astype(np.int64).to_numpy()
    expected_var_index = np.arange(mapping_df.shape[0], dtype=np.int64)
    if not np.array_equal(var_index, expected_var_index):
        raise ValueError(f"Cached mapping for slide '{slide_id}' has invalid var_index ordering.")

    return mapping_df


def count_converted_genes(mapping_df: pd.DataFrame, slide_id: str) -> int:
    """Number of genes of a slide converted to human."""
    if "converted_to_human" not in mapping_df.columns:
        raise KeyError(f"Mapping for slide '{slide_id}' is missing required column 'converted_to_human'.")

    converted = mapping_df["converted_to_human"]

    invalid_mask = converted.notna() & ~converted.isin([True, False])
    if invalid_mask.any():
        bad_values = converted.loc[invalid_mask].drop_duplicates().tolist()
        raise ValueError(f"Mapping for slide '{slide_id}' has invalid values in 'converted_to_human': {bad_values}")

    return int(converted.fillna(False).astype(bool).sum())


def write_cache_manifest(
    manifest_path: Path,
    slide_id: str,
    input_signature: str,
    n_rows: int,
) -> None:
    manifest = {
        "slide_id": slide_id,
        "mapping_cache_version": MAPPING_CACHE_VERSION,
        "source": "ensembl_rest_live_query",
        "ensembl_rest_base_url": ENSEMBL_REST_BASE_URL,
        "created_utc": utc_now_iso_z(),
        "input_signature_sha256": input_signature,
        "n_rows": int(n_rows),
        "mapping_columns": FINAL_MAPPING_COLUMNS,
        "signature_input_columns": SIGNATURE_INPUT_COLUMNS,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)


def load_cached_mapping_for_slide(
    slide_id: str,
    slide_h5ad_path: Path,
    cache_path: Path,
    manifest_path: Path,
) -> Optional[pd.DataFrame]:
    """
    Load a canonical cached mapping if present and validate that it matches the
    current slide var inputs exactly.
    """
    cache_exists = cache_path.exists()
    manifest_exists = manifest_path.exists()

    if not cache_exists and not manifest_exists:
        return None

    if cache_exists != manifest_exists:
        raise RuntimeError(
            f"Partial cache state for slide '{slide_id}'. "
            f"cache_exists={cache_exists}, manifest_exists={manifest_exists}. "
            f"Expected both files or neither:\n"
            f"  cache   : {cache_path}\n"
            f"  manifest: {manifest_path}"
        )

    print(f"  [cache-check] Reading current var inputs for slide '{slide_id}' to validate cache...")
    current_gene_df = read_slide_var_table(slide_h5ad_path)
    expected_signature = compute_mapping_input_signature(current_gene_df)
    expected_n_rows = int(current_gene_df.shape[0])

    print(f"  [cache-check] Loading manifest: {manifest_path}")
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    required_manifest_keys = {
        "slide_id",
        "mapping_cache_version",
        "input_signature_sha256",
        "n_rows",
        "mapping_columns",
        "signature_input_columns",
    }
    missing_manifest_keys = sorted(required_manifest_keys - set(manifest.keys()))
    if missing_manifest_keys:
        raise KeyError(
            f"Manifest for slide '{slide_id}' is missing keys: {missing_manifest_keys}"
        )

    if str(manifest["slide_id"]) != slide_id:
        raise ValueError(
            f"Manifest slide_id mismatch for slide '{slide_id}': {manifest['slide_id']}"
        )
    if str(manifest["mapping_cache_version"]) != MAPPING_CACHE_VERSION:
        raise ValueError(
            f"Cache version mismatch for slide '{slide_id}'. "
            f"Expected '{MAPPING_CACHE_VERSION}', found '{manifest['mapping_cache_version']}'."
        )
    if list(manifest["mapping_columns"]) != FINAL_MAPPING_COLUMNS:
        raise ValueError(
            f"Manifest mapping_columns mismatch for slide '{slide_id}'."
        )
    if list(manifest["signature_input_columns"]) != SIGNATURE_INPUT_COLUMNS:
        raise ValueError(
            f"Manifest signature_input_columns mismatch for slide '{slide_id}'."
        )
    if str(manifest["input_signature_sha256"]) != expected_signature:
        raise ValueError(
            f"Input signature mismatch for slide '{slide_id}'. "
            f"The cached mapping does not match the current slide var inputs."
        )
    if int(manifest["n_rows"]) != expected_n_rows:
        raise ValueError(
            f"Row-count mismatch for slide '{slide_id}'. "
            f"Manifest has {manifest['n_rows']}, current slide has {expected_n_rows}."
        )

    print(f"  [cache-check] Loading cached parquet: {cache_path}")
    mapping_df = pd.read_parquet(cache_path)
    mapping_df = validate_mapping_df_structure(mapping_df, slide_id=slide_id)

    if int(mapping_df.shape[0]) != expected_n_rows:
        raise ValueError(
            f"Cached mapping row-count mismatch for slide '{slide_id}'. "
            f"Cache has {mapping_df.shape[0]}, current slide has {expected_n_rows}."
        )

    cache_var_index = mapping_df["var_index"].astype(np.int64).to_numpy()
    current_var_index = current_gene_df["var_index"].astype(np.int64).to_numpy()
    if not np.array_equal(cache_var_index, current_var_index):
        raise ValueError(
            f"Cached mapping var_index mismatch for slide '{slide_id}'."
        )

    cache_var_names = mapping_df["original_var_name"].astype(str).tolist()
    current_var_names = current_gene_df["original_var_name"].astype(str).tolist()
    if cache_var_names != current_var_names:
        raise ValueError(
            f"Cached mapping original_var_name mismatch for slide '{slide_id}'."
        )

    print(f"  [cache-check] Cache validated successfully for slide '{slide_id}' ({expected_n_rows} rows).")
    return mapping_df


def make_retry_session() -> requests.Session:
    print("Creating HTTP session with retry policy for Ensembl REST...")
    session = requests.Session()
    retry = Retry(
        total=5,
        read=5,
        connect=5,
        backoff_factor=1.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET", "POST"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({"User-Agent": "HistoST-mouse-human-gene-mapper/1.1"})
    return session


# -----------------------------
# Ensembl REST client
# -----------------------------

class EnsemblRestClient:
    """
    Ensembl REST client:
      - POST /lookup/id               for batched Ensembl ID -> gene display_name
      - POST /lookup/symbol/:species  for batched symbol -> Ensembl gene ID
      - GET  /homology/id/:species/:id for mouse -> human orthology
    """

    def __init__(self, session: requests.Session, base_url: str = ENSEMBL_REST_BASE_URL):
        self.session = session
        self.base_url = base_url.rstrip("/")

    def _headers(self) -> Dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def _get_json(self, path: str, params: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        resp = self.session.get(
            self.base_url + path,
            params=params,
            headers=self._headers(),
            timeout=REQUEST_TIMEOUT,
        )
        if resp.status_code == 404:
            return None
        try:
            resp.raise_for_status()
        except requests.HTTPError as e:
            raise EnsemblRestError(
                f"Ensembl REST GET failed.\n"
                f"path={path}\n"
                f"params={params}\n"
                f"http_status={resp.status_code}\n"
                f"server_reply={resp.text[:1000]}"
            ) from e
        return resp.json()

    def _post_json(
        self,
        path: str,
        payload: Dict[str, Any],
        params: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        resp = self.session.post(
            self.base_url + path,
            params=params,
            json=payload,
            headers=self._headers(),
            timeout=REQUEST_TIMEOUT,
        )
        if resp.status_code == 404:
            return None
        try:
            resp.raise_for_status()
        except requests.HTTPError as e:
            raise EnsemblRestError(
                f"Ensembl REST POST failed.\n"
                f"path={path}\n"
                f"params={params}\n"
                f"payload_preview={json.dumps(payload)[:1000]}\n"
                f"http_status={resp.status_code}\n"
                f"server_reply={resp.text[:1000]}"
            ) from e
        return resp.json()

    def lookup_ids(
        self,
        ids: Sequence[str],
        species: Optional[str] = None,
    ) -> Dict[str, Dict[str, Any]]:
        ids = [normalize_gene_id(x) for x in ids]
        ids = [x for x in ids if x]
        if not ids:
            return {}
        params: Dict[str, Any] = {"object_type": "gene"}
        if species:
            params["species"] = species
        out = self._post_json("/lookup/id", {"ids": ids}, params=params)
        return {} if out is None else out

    def lookup_symbols(
        self,
        species: str,
        symbols: Sequence[str],
    ) -> Dict[str, Dict[str, Any]]:
        symbols = [str(x).strip() for x in symbols if str(x).strip()]
        if not symbols:
            return {}
        out = self._post_json(f"/lookup/symbol/{species}", {"symbols": symbols})
        return {} if out is None else out

    def homology_by_id(
        self,
        species: str,
        gene_id: str,
        target_species: str,
    ) -> Optional[Dict[str, Any]]:
        gene_id = normalize_gene_id(gene_id)
        if gene_id is None:
            return None
        return self._get_json(
            f"/homology/id/{species}/{gene_id}",
            params={
                "target_species": target_species,
                "type": "orthologues",
                "format": "condensed",
                "sequence": "none",
            },
        )


# -----------------------------
# Resolver with in-memory cache
# -----------------------------

class OrthologResolver:
    """
    Caches all Ensembl REST lookups within a run.
    """

    def __init__(self, rest_client: EnsemblRestClient):
        self.rest = rest_client

        # Ensembl gene lookup caches
        self.mouse_lookup_by_id_cache: Dict[str, Dict[str, Optional[str]]] = {}
        self.mouse_lookup_by_symbol_cache: Dict[str, Dict[str, Optional[str]]] = {}
        self.human_lookup_by_id_cache: Dict[str, Dict[str, Optional[str]]] = {}

        # Mouse Ensembl gene ID -> chosen human ortholog mapping
        self.mouse_homology_cache: Dict[str, Dict[str, Optional[str]]] = {}

    @staticmethod
    def _empty_mapping_record(status: str) -> Dict[str, Optional[str]]:
        return {
            "status": status,
            "mouse_ensembl_gene_id": None,
            "mouse_gene_name_ensembl": None,
            "human_ensembl_gene_id": None,
            "human_gene_name_ensembl": None,
            "orthology_type": None,
        }

    @staticmethod
    def _empty_lookup_record(status: str) -> Dict[str, Optional[str]]:
        return {
            "status": status,
            "ensembl_gene_id": None,
            "display_name": None,
        }

    @staticmethod
    def _parse_lookup_obj(obj: Optional[Dict[str, Any]]) -> Dict[str, Optional[str]]:
        if not obj:
            return OrthologResolver._empty_lookup_record("not_found")
        ensembl_gene_id = normalize_gene_id(obj.get("id"))
        if ensembl_gene_id is None:
            return OrthologResolver._empty_lookup_record("not_found")
        display_name = first_nonempty_strict([
            obj.get("display_name"),
            obj.get("id"),
        ])
        return {
            "status": "mapped",
            "ensembl_gene_id": ensembl_gene_id,
            "display_name": display_name,
        }

    def preload_mouse_lookup_from_ids(self, mouse_gene_ids: Sequence[str]) -> None:
        mouse_gene_ids = sorted({gid for gid in mouse_gene_ids if is_mouse_ensembl_gene_id(gid)})
        to_query = [gid for gid in mouse_gene_ids if gid not in self.mouse_lookup_by_id_cache]
        if not to_query:
            print("    [resolver] Mouse ID lookup cache already warm: nothing to query.")
            return

        n_chunks = (len(to_query) + REST_LOOKUP_ID_BATCH_SIZE - 1) // REST_LOOKUP_ID_BATCH_SIZE
        print(f"    [resolver] Querying mouse ID lookup for {len(to_query)} IDs in {n_chunks} chunk(s)...")
        for chunk in tqdm(
            chunked(to_query, REST_LOOKUP_ID_BATCH_SIZE),
            total=n_chunks,
            desc="    mouse lookup by ID",
            leave=False,
        ):
            raw = self.rest.lookup_ids(chunk, species="mus_musculus")
            for gid in chunk:
                self.mouse_lookup_by_id_cache[gid] = self._parse_lookup_obj(raw.get(gid))
            time.sleep(SLEEP_BETWEEN_CHUNKS_SEC)
        print(f"    [resolver] Mouse ID lookup cache size: {len(self.mouse_lookup_by_id_cache)}")

    def preload_mouse_lookup_from_symbols(self, mouse_symbols: Sequence[str]) -> None:
        mouse_symbols = sorted({str(s).strip() for s in mouse_symbols if str(s).strip() != ""})
        to_query = [s for s in mouse_symbols if s not in self.mouse_lookup_by_symbol_cache]
        if not to_query:
            print("    [resolver] Mouse symbol lookup cache already warm: nothing to query.")
            return

        n_chunks = (len(to_query) + REST_LOOKUP_SYMBOL_BATCH_SIZE - 1) // REST_LOOKUP_SYMBOL_BATCH_SIZE
        print(f"    [resolver] Querying mouse symbol lookup for {len(to_query)} symbols in {n_chunks} chunk(s)...")
        for chunk in tqdm(
            chunked(to_query, REST_LOOKUP_SYMBOL_BATCH_SIZE),
            total=n_chunks,
            desc="    mouse lookup by symbol",
            leave=False,
        ):
            raw = self.rest.lookup_symbols("mus_musculus", chunk)
            for symbol in chunk:
                self.mouse_lookup_by_symbol_cache[symbol] = self._parse_lookup_obj(raw.get(symbol))
            time.sleep(SLEEP_BETWEEN_CHUNKS_SEC)
        print(f"    [resolver] Mouse symbol lookup cache size: {len(self.mouse_lookup_by_symbol_cache)}")

    def _resolve_homology_for_mouse_id(self, mouse_gene_id: str) -> Dict[str, Optional[str]]:
        mouse_gene_id = normalize_gene_id(mouse_gene_id)
        if not is_mouse_ensembl_gene_id(mouse_gene_id):
            return self._empty_mapping_record("invalid_mouse_gene_id")

        mouse_lookup = self.mouse_lookup_by_id_cache.get(mouse_gene_id)
        if mouse_lookup is None or mouse_lookup["status"] != "mapped":
            return self._empty_mapping_record("not_found")

        raw = self.rest.homology_by_id(
            species="mus_musculus",
            gene_id=mouse_gene_id,
            target_species="human",
        )
        if not raw:
            return self._empty_mapping_record("homology_response_empty")

        data = raw.get("data") or []
        if not data:
            return self._empty_mapping_record("homology_data_empty")

        entry = data[0] if isinstance(data, list) else data
        homologies = entry.get("homologies") or []
        if not homologies:
            return self._empty_mapping_record("no_homologies_returned")

        valid = []
        saw_human_like_target = False

        for h in homologies:

            orthology_type = str(h.get("type") or "").strip()

            target = h.get("target") if isinstance(h.get("target"), dict) else {}
            human_gene_id = normalize_gene_id(target.get("id") or h.get("id"))

            if human_gene_id is not None:
                saw_human_like_target = True

            if orthology_type != "ortholog_one2one":
                continue
            if not is_human_ensembl_gene_id(human_gene_id):
                continue

            valid.append(
                {
                    "status": "mapped",
                    "mouse_ensembl_gene_id": mouse_gene_id,
                    "mouse_gene_name_ensembl": mouse_lookup["display_name"],
                    "human_ensembl_gene_id": human_gene_id,
                    "human_gene_name_ensembl": None,
                    "orthology_type": orthology_type,
                }
            )

        if len(valid) == 0:
            return self._empty_mapping_record("no_one2one" if saw_human_like_target else "no_valid_human_target_in_response")

        unique_human_ids = {rec["human_ensembl_gene_id"] for rec in valid}
        if len(unique_human_ids) > 1:
            return self._empty_mapping_record("ambiguous_one_to_one_multiple_targets")

        return valid[0]

    def preload_mouse_homologies_from_ids(self, mouse_gene_ids: Sequence[str]) -> None:
        mouse_gene_ids = sorted({gid for gid in mouse_gene_ids if is_mouse_ensembl_gene_id(gid)})
        to_query = [gid for gid in mouse_gene_ids if gid not in self.mouse_homology_cache]
        if not to_query:
            print("    [resolver] Mouse homology cache already warm: nothing to query.")
            return

        print(
            f"    [resolver] Querying mouse->human homologies for {len(to_query)} IDs "
            f"with {REST_HOMOLOGY_MAX_WORKERS} workers..."
        )

        # Make sure display_name is available for source gene IDs
        self.preload_mouse_lookup_from_ids(to_query)

        with ThreadPoolExecutor(max_workers=REST_HOMOLOGY_MAX_WORKERS) as ex:
            futures = {
                ex.submit(self._resolve_homology_for_mouse_id, gid): gid
                for gid in to_query
            }
            for fut in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="    mouse->human homology",
                leave=False,
            ):
                gid = futures[fut]
                self.mouse_homology_cache[gid] = fut.result()

        print(f"    [resolver] Mouse homology cache size: {len(self.mouse_homology_cache)}")

    def preload_human_lookup_from_ids(self, human_gene_ids: Sequence[str]) -> None:
        human_gene_ids = sorted({gid for gid in human_gene_ids if is_human_ensembl_gene_id(gid)})
        to_query = [gid for gid in human_gene_ids if gid not in self.human_lookup_by_id_cache]
        if not to_query:
            print("    [resolver] Human ID lookup cache already warm: nothing to query.")
            return

        n_chunks = (len(to_query) + REST_LOOKUP_ID_BATCH_SIZE - 1) // REST_LOOKUP_ID_BATCH_SIZE
        print(f"    [resolver] Querying human ID lookup for {len(to_query)} IDs in {n_chunks} chunk(s)...")
        for chunk in tqdm(
            chunked(to_query, REST_LOOKUP_ID_BATCH_SIZE),
            total=n_chunks,
            desc="    human lookup by ID",
            leave=False,
        ):
            raw = self.rest.lookup_ids(chunk, species="homo_sapiens")
            for gid in chunk:
                self.human_lookup_by_id_cache[gid] = self._parse_lookup_obj(raw.get(gid))
            time.sleep(SLEEP_BETWEEN_CHUNKS_SEC)
        print(f"    [resolver] Human ID lookup cache size: {len(self.human_lookup_by_id_cache)}")

    def choose_best_mouse_to_human_mapping(
        self,
        mouse_symbol_input: Optional[str],
        mouse_ensembl_gene_id_input: Optional[str],
    ) -> Tuple[Optional[str], Dict[str, Optional[str]]]:
        """
        Mapping of one gene: from the mouse Ensembl ID if available, else (or if the ID is not found) from the symbol.
        Returns (mapping_input_used, mapping_record).
        """
        id_rec = None
        if is_mouse_ensembl_gene_id(mouse_ensembl_gene_id_input):
            id_rec = self.mouse_homology_cache.get(mouse_ensembl_gene_id_input)
            if id_rec is None:
                raise KeyError(f"Missing cached homology mapping for {mouse_ensembl_gene_id_input}")

            if id_rec["status"] == "mapped":
                return "mouse_ensembl_gene_id", id_rec

            if id_rec["status"] in {"no_one2one", "ambiguous_one_to_one_multiple_targets"}:
                return "mouse_ensembl_gene_id", id_rec

            if id_rec["status"] == "not_found":
                pass
            else:
                return "mouse_ensembl_gene_id", id_rec

        sym_key = None if mouse_symbol_input is None else str(mouse_symbol_input).strip()
        if sym_key:
            sym_lookup = self.mouse_lookup_by_symbol_cache.get(sym_key)
            if sym_lookup is None:
                raise KeyError(f"Missing cached mouse symbol lookup for {sym_key}")

            if sym_lookup["status"] == "mapped" and is_mouse_ensembl_gene_id(sym_lookup["ensembl_gene_id"]):
                sym_gid = sym_lookup["ensembl_gene_id"]
                sym_hrec = self.mouse_homology_cache.get(sym_gid)
                if sym_hrec is None:
                    raise KeyError(f"Missing cached homology mapping for symbol-derived ID {sym_gid}")
                if sym_hrec["status"] == "mapped":
                    return "mouse_gene_symbol", sym_hrec
                if id_rec is None:
                    return "mouse_gene_symbol", sym_hrec

            if id_rec is not None:
                return "mouse_ensembl_gene_id", id_rec

            return "mouse_gene_symbol", self._empty_mapping_record("not_found")

        if id_rec is not None:
            return "mouse_ensembl_gene_id", id_rec

        return None, self._empty_mapping_record("not_found")


# -----------------------------
# Slide processing
# -----------------------------


def get_mouse_hest_slide_ids(metadata: pd.DataFrame) -> List[str]:
    if "species" not in metadata.columns:
        raise KeyError("Metadata CSV must contain a 'species' column.")

    slide_col = 'slide_id'
    df = metadata.copy()
    df["species"] = df["species"].astype(str)

    df = df[df["species"] == "Mus musculus"]

    if "source" in df.columns:
        df = df[df["source"].astype(str) == "HEST"]

    slide_ids = sorted(df[slide_col].astype(str).unique().tolist())
    return slide_ids


def read_slide_var_table(slide_h5ad_path: Path) -> pd.DataFrame:
    """
    Read the var table of a mouse HEST slide: mouse Ensembl ID from var_name (ENSMUSG...) or adata.var["gene_ids"],
    mouse symbol from var_name otherwise.
    """
    print(f"  Opening h5ad in backed mode: {slide_h5ad_path}")
    adata = read_h5ad(slide_h5ad_path, backed="r")
    try:
        var_names = [str(x) for x in adata.var_names.tolist()]
        n = len(var_names)
        print(f"  Loaded var_names: {n} genes")

        df = pd.DataFrame({
            "var_index": np.arange(n, dtype=np.int64),
            "original_var_name": var_names,
        })

        # Optional helper column from adata.var["gene_ids"]
        gene_ids_helper: List[Optional[str]]
        if "gene_ids" in adata.var.columns:
            print("  Found helper column adata.var['gene_ids']; using it when useful.")
            gene_ids_raw = adata.var["gene_ids"].tolist()
            gene_ids_helper = [normalize_gene_id(x) for x in gene_ids_raw]
        else:
            print("  No helper column adata.var['gene_ids']; mapping will rely on var_names only.")
            gene_ids_helper = [None] * n

        mouse_symbol_input = []
        mouse_ensembl_gene_id_input = []
        input_identifier_type = []

        for var_name, helper_gid in tqdm(
            zip(var_names, gene_ids_helper),
            total=n,
            desc="  parsing var inputs",
            leave=False,
        ):
            var_name_norm = normalize_gene_id(var_name)

            if is_mouse_ensembl_gene_id(var_name_norm):
                symbol_in = None
                ens_in = var_name_norm
                id_type = "mouse_ensembl_gene_id_in_var_name"
            else:
                symbol_in = var_name
                if is_mouse_ensembl_gene_id(helper_gid):
                    ens_in = helper_gid
                    id_type = "mouse_symbol_in_var_name_plus_mouse_ensembl_gene_id_in_gene_ids"
                else:
                    ens_in = None
                    id_type = "mouse_symbol_in_var_name"

            mouse_symbol_input.append(symbol_in)
            mouse_ensembl_gene_id_input.append(ens_in)
            input_identifier_type.append(id_type)

        df["mouse_symbol_input"] = mouse_symbol_input
        df["mouse_ensembl_gene_id_input"] = mouse_ensembl_gene_id_input
        df["input_identifier_type"] = input_identifier_type

        n_mouse_ids = int(df["mouse_ensembl_gene_id_input"].notna().sum())
        n_mouse_symbols = int(df["mouse_symbol_input"].notna().sum())
        print(f"  Parsed inputs summary: {n_mouse_ids} with mouse Ensembl IDs, {n_mouse_symbols} with mouse symbols")

        return df
    finally:
        if getattr(adata, "file", None) is not None:
            adata.file.close()
            print("  Closed backed h5ad file handle.")


def build_mapping_for_slide(
    slide_id: str,
    slide_h5ad_path: Path,
    resolver: OrthologResolver,
) -> Tuple[pd.DataFrame, str]:
    slide_start_time = time.time()
    print(f"\nProcessing slide '{slide_id}'")
    print(f"  Reading var table from: {slide_h5ad_path}")

    gene_df = read_slide_var_table(slide_h5ad_path)
    input_signature = compute_mapping_input_signature(gene_df)
    gene_df["slide_id"] = slide_id

    # 1) Preload ID-based mappings
    print("  Step 1/5: preparing mouse Ensembl-ID based mappings...")
    mouse_ids = sorted({
        gid for gid in gene_df["mouse_ensembl_gene_id_input"].dropna().astype(str).tolist()
        if is_mouse_ensembl_gene_id(gid)
    })
    print(f"  Mouse genes with valid Ensembl IDs: {len(mouse_ids)}")
    resolver.preload_mouse_lookup_from_ids(mouse_ids)
    resolver.preload_mouse_homologies_from_ids(mouse_ids)

    # Quick sanity print on a few known inputs
    print("  Sanity check: first 5 mouse ID homology results")
    for gid in mouse_ids[:5]:
        rec = resolver.mouse_homology_cache.get(gid, {})
        print(
            f"    {gid}: status={rec.get('status')}, "
            f"human_ensembl_gene_id={rec.get('human_ensembl_gene_id')}, "
            f"orthology_type={rec.get('orthology_type')}"
        )

    # 2) Preload symbol-based fallback mappings only for genes with:
    #    - no valid mouse Ensembl ID, or
    #    - valid ID but Ensembl ID lookup was not found
    print("  Step 2/5: identifying genes that need symbol fallback...")
    symbol_fallback_names: List[str] = []
    for row in tqdm(
        gene_df.itertuples(index=False),
        total=gene_df.shape[0],
        desc="  scanning fallback candidates",
        leave=False,
    ):
        mouse_symbol_input = row.mouse_symbol_input
        mouse_gene_id = row.mouse_ensembl_gene_id_input

        use_symbol_fallback = False
        if not is_mouse_ensembl_gene_id(mouse_gene_id):
            use_symbol_fallback = True
        else:
            id_rec = resolver.mouse_homology_cache.get(mouse_gene_id)
            if id_rec is None:
                raise KeyError(f"Internal error: no cached homology mapping for {mouse_gene_id}")
            if id_rec["status"] == "not_found":
                use_symbol_fallback = True

        if use_symbol_fallback and isinstance(mouse_symbol_input, str) and mouse_symbol_input.strip() != "":
            symbol_fallback_names.append(mouse_symbol_input.strip())

    symbol_fallback_names = sorted(set(symbol_fallback_names))
    print(f"  Mouse genes needing symbol fallback: {len(symbol_fallback_names)}")
    resolver.preload_mouse_lookup_from_symbols(symbol_fallback_names)

    symbol_fallback_ids = sorted({
        rec["ensembl_gene_id"]
        for rec in resolver.mouse_lookup_by_symbol_cache.values()
        if rec["status"] == "mapped" and is_mouse_ensembl_gene_id(rec["ensembl_gene_id"])
    })
    print(f"  Symbol fallback produced {len(symbol_fallback_ids)} unique mouse Ensembl IDs to resolve.")
    resolver.preload_mouse_lookup_from_ids(symbol_fallback_ids)
    resolver.preload_mouse_homologies_from_ids(symbol_fallback_ids)

    # 3) Decide mapping source for each gene
    print("  Step 3/5: choosing best mapping source for each gene...")
    chosen_mapping_inputs: List[Optional[str]] = []
    chosen_mapping_records: List[Dict[str, Optional[str]]] = []

    for row in tqdm(
        gene_df.itertuples(index=False),
        total=gene_df.shape[0],
        desc="  choosing mapping source",
        leave=False,
    ):
        mapping_input_used, mapping_rec = resolver.choose_best_mouse_to_human_mapping(
            mouse_symbol_input=row.mouse_symbol_input,
            mouse_ensembl_gene_id_input=row.mouse_ensembl_gene_id_input,
        )
        chosen_mapping_inputs.append(mapping_input_used)
        chosen_mapping_records.append(mapping_rec)

    gene_df["mapping_input_used"] = chosen_mapping_inputs
    gene_df["ortholog_mapping_status"] = [rec["status"] for rec in chosen_mapping_records]
    gene_df["mouse_gene_name_ensembl"] = [rec["mouse_gene_name_ensembl"] for rec in chosen_mapping_records]
    gene_df["human_ensembl_gene_id"] = [rec["human_ensembl_gene_id"] for rec in chosen_mapping_records]
    gene_df["human_gene_name_ensembl"] = [rec["human_gene_name_ensembl"] for rec in chosen_mapping_records]
    gene_df["orthology_type"] = [rec["orthology_type"] for rec in chosen_mapping_records]

    status_counts = gene_df["ortholog_mapping_status"].fillna("unknown").value_counts(dropna=False).sort_index()
    print("  Ortholog mapping status breakdown:")
    for status, count in status_counts.items():
        print(f"    - {status}: {int(count)}")

    # 4) Preload human gene display names for all mapped human Ensembl IDs
    print("  Step 4/5: loading human gene display names...")
    human_ids = sorted({
        gid for gid in gene_df["human_ensembl_gene_id"].dropna().astype(str).tolist()
        if is_human_ensembl_gene_id(gid)
    })
    print(f"  Unique mapped human Ensembl IDs: {len(human_ids)}")
    resolver.preload_human_lookup_from_ids(human_ids)

    # 5) Finalize names
    print("  Step 5/5: finalizing output gene names...")
    human_gene_name_final = []
    human_gene_name_source = []
    converted_to_human = []
    keep_mouse_name_reason = []
    target_gene_name = []
    resolved_human_gene_name_ensembl = []

    for row in tqdm(
        gene_df.itertuples(index=False),
        total=gene_df.shape[0],
        desc="  finalizing names",
        leave=False,
    ):
        original_var_name = str(row.original_var_name)
        ortholog_status = row.ortholog_mapping_status
        human_id = row.human_ensembl_gene_id

        hrec = None
        if is_human_ensembl_gene_id(human_id):
            hrec = resolver.human_lookup_by_id_cache.get(human_id)
            if hrec is None:
                raise KeyError(f"Internal error: no cached human lookup mapping for {human_id}")

        final_human_name = hrec["display_name"] if (hrec is not None and hrec["status"] == "mapped") else None
        final_human_source = "ensembl_lookup_display_name" if final_human_name is not None else None

        converted = bool(
            ortholog_status == "mapped"
            and is_human_ensembl_gene_id(human_id)
            and final_human_name is not None
        )

        if converted:
            reason = None
            final_target_name = final_human_name
        else:
            if ortholog_status == "no_one2one":
                reason = "no_human_one_to_one_ortholog"
            elif ortholog_status == "ambiguous_one_to_one_multiple_targets":
                reason = "ambiguous_one_to_one_multiple_targets"
            elif ortholog_status == "not_found":
                reason = "mouse_gene_not_found_in_ensembl"
            elif ortholog_status == "mouse_lookup_not_found":
                reason = "mouse_gene_not_found_in_ensembl"
            elif ortholog_status == "invalid_mouse_gene_id":
                reason = "invalid_mouse_gene_id"
            elif ortholog_status == "homology_response_empty":
                reason = "empty_homology_response_from_ensembl"
            elif ortholog_status == "homology_data_empty":
                reason = "empty_homology_data_from_ensembl"
            elif ortholog_status == "no_homologies_returned":
                reason = "no_human_homology_returned"
            elif ortholog_status == "no_valid_human_target_in_response":
                reason = "invalid_human_target_in_homology_response"
            elif ortholog_status == "mapped" and final_human_name is None:
                reason = "human_ortholog_found_but_no_human_gene_symbol"
            else:
                reason = ortholog_status if pd.notna(ortholog_status) else "unknown"

            # Keep the original mouse var_name
            final_target_name = original_var_name

        resolved_human_gene_name_ensembl.append(final_human_name)
        human_gene_name_final.append(final_human_name)
        human_gene_name_source.append(final_human_source)
        converted_to_human.append(converted)
        keep_mouse_name_reason.append(reason)
        target_gene_name.append(final_target_name)

    gene_df["human_gene_name_ensembl"] = resolved_human_gene_name_ensembl
    gene_df["human_gene_name_final"] = human_gene_name_final
    gene_df["human_gene_name_source"] = human_gene_name_source
    gene_df["converted_to_human"] = np.asarray(converted_to_human, dtype=bool)
    gene_df["keep_mouse_name_reason"] = keep_mouse_name_reason
    gene_df["target_gene_name"] = target_gene_name

    gene_df = gene_df[FINAL_MAPPING_COLUMNS].copy()

    n_total = gene_df.shape[0]
    n_converted = int(gene_df["converted_to_human"].sum())
    n_not_converted = int((~gene_df["converted_to_human"]).sum())

    print(f"  Total genes           : {n_total}")
    print(f"  Converted to human    : {n_converted}")
    print(f"  NOT converted         : {n_not_converted}")

    reason_counts = (
        gene_df.loc[~gene_df["converted_to_human"], "keep_mouse_name_reason"]
        .fillna("unknown")
        .value_counts(dropna=False)
        .sort_index()
    )
    if len(reason_counts) > 0:
        print("  Not converted breakdown:")
        for reason, count in reason_counts.items():
            print(f"    - {reason}: {int(count)}")

    elapsed = time.time() - slide_start_time
    print(f"  Finished slide '{slide_id}' in {elapsed:.1f}s")

    return gene_df, input_signature


# -----------------------------
# Main
# -----------------------------

def main(args) -> None:
    global_start_time = time.time()

    print("=" * 100)
    print("Starting mouse -> human HEST mapping build")
    print("=" * 100)
    print(f"Arguments:")
    print(f"  training_dataset_id      : {args.training_dataset_id}")
    print(f"  allow_live_rest       : {args.allow_live_rest}")
    print(f"  refresh_cached_mappings  : {args.refresh_cached_mappings}")

    if args.refresh_cached_mappings and not args.allow_live_rest:
        raise ValueError("--refresh_cached_mappings requires --allow_live_rest.")

    # Paths
    print("\nResolving input/output paths...")
    training_dataset_id = args.training_dataset_id
    metadata_csv = get_full_path("eval", "eval_datasets", "st_eval_data", f"eval_st_{training_dataset_id}", f"metadata_eval_st_{training_dataset_id}.csv")
    hest_root = get_full_path("eval", "eval_datasets", "st_eval_data", "hest")
    hest_st_dir = hest_root / "st"

    print(f"  metadata_csv : {metadata_csv}")
    print(f"  hest_root    : {hest_root}")
    print(f"  hest_st_dir  : {hest_st_dir}")

    if not metadata_csv.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {metadata_csv}")
    if not hest_root.exists():
        raise FileNotFoundError(f"HEST root folder not found: {hest_root}")
    if not hest_st_dir.exists():
        raise FileNotFoundError(f"HEST ST folder not found: {hest_st_dir}")

    # Metadata with species and slide IDs information
    print(f"\nReading metadata: {metadata_csv}")
    metadata = pd.read_csv(metadata_csv)
    print(f"Metadata shape: {metadata.shape}")

    # Get list of mouse HEST slide IDs from metadata
    mouse_hest_slide_ids = get_mouse_hest_slide_ids(metadata)

    if len(mouse_hest_slide_ids) == 0:
        print("No 'Mus musculus' HEST slides found in metadata. Nothing to do.")
        return

    print(f"\nMouse HEST slides found: {len(mouse_hest_slide_ids)}")
    for sid in mouse_hest_slide_ids:
        print(f"  - {sid}")

    resolver: Optional[OrthologResolver] = None
    cache_dir = hest_root / MAPPING_CACHE_DIRNAME
    print(f"\nCanonical cache directory: {cache_dir}")

    n_loaded_from_cache = 0
    n_computed = 0
    failures: List[Tuple[str, str]] = []
    zero_mapping_slides: List[str] = []

    # Process each mouse HEST slide
    print("\nBeginning per-slide processing...")
    for slide_id in tqdm(mouse_hest_slide_ids, desc="slides", unit="slide"):
        print("\n" + "-" * 100)
        print(f"Slide: {slide_id}")
        print("-" * 100)

        out_path = hest_root / f"{slide_id}_mouse2human_mapping.parquet"
        cache_path, manifest_path = get_cache_paths(hest_root=hest_root, slide_id=slide_id)

        print(f"  Output parquet : {out_path}")
        print(f"  Cache parquet  : {cache_path}")
        print(f"  Cache manifest : {manifest_path}")

        slide_h5ad_path = hest_st_dir / f"{slide_id}.h5ad"
        if not slide_h5ad_path.exists():
            failures.append((slide_id, f"Missing HEST slide h5ad: {slide_h5ad_path}"))
            print(f"\nERROR for slide '{slide_id}': missing file {slide_h5ad_path}")
            continue

        try:
            if not args.refresh_cached_mappings:
                print(f"  Checking whether canonical cache can be reused for slide '{slide_id}'...")
                cached_df = load_cached_mapping_for_slide(
                    slide_id=slide_id,
                    slide_h5ad_path=slide_h5ad_path,
                    cache_path=cache_path,
                    manifest_path=manifest_path,
                )
                if cached_df is not None:
                    print(f"\nUsing canonical cached mapping for slide '{slide_id}':")
                    print(f"  cache   : {cache_path}")
                    print(f"  manifest: {manifest_path}")

                    n_converted_cached = count_converted_genes(cached_df, slide_id=slide_id)
                    print(f"  Converted genes in cached mapping: {n_converted_cached}")
                    if n_converted_cached == 0:
                        zero_mapping_slides.append(slide_id)
                        print(f"  WARNING: slide '{slide_id}' has 0 successfully mapped genes.")

                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    if cache_path != out_path:
                        print("  Copying cached parquet to materialized output location...")
                        shutil.copy2(cache_path, out_path)
                    print(f"  Materialized mapping file to: {out_path}")
                    n_loaded_from_cache += 1
                    continue
                else:
                    print(f"  No valid canonical cache found for slide '{slide_id}'.")

            if not args.allow_live_rest:
                raise RuntimeError(
                    f"Missing canonical cached mapping for slide '{slide_id}'.\n"
                    f"Expected:\n"
                    f"  cache   : {cache_path}\n"
                    f"  manifest: {manifest_path}\n"
                    f"Re-run with --allow_live_rest to create the cache from Ensembl REST."
                )

            if resolver is None:
                print("\nInitializing shared Ensembl REST resolver for live queries...")
                session = make_retry_session()
                rest_client = EnsemblRestClient(session=session)
                resolver = OrthologResolver(rest_client=rest_client)
                print("Shared resolver initialized.")

            mapping_df, input_signature = build_mapping_for_slide(
                slide_id=slide_id,
                slide_h5ad_path=slide_h5ad_path,
                resolver=resolver,
            )

            print("  Validating final mapping dataframe structure...")
            mapping_df = validate_mapping_df_structure(mapping_df, slide_id=slide_id)

            n_converted_current = count_converted_genes(mapping_df, slide_id=slide_id)
            print(f"  Converted genes in final mapping: {n_converted_current}")
            if n_converted_current == 0:
                zero_mapping_slides.append(slide_id)
                print(f"  WARNING: slide '{slide_id}' has 0 successfully mapped genes.")

            cache_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.parent.mkdir(parents=True, exist_ok=True)

            print(f"  Writing canonical cache parquet: {cache_path}")
            mapping_df.to_parquet(cache_path, index=False)

            print(f"  Writing cache manifest JSON: {manifest_path}")
            write_cache_manifest(
                manifest_path=manifest_path,
                slide_id=slide_id,
                input_signature=input_signature,
                n_rows=mapping_df.shape[0],
            )

            if cache_path != out_path:
                print(f"  Copying canonical cache parquet to materialized output: {out_path}")
                shutil.copy2(cache_path, out_path)

            print(f"  Saved canonical cache to: {cache_path}")
            print(f"  Saved cache manifest to : {manifest_path}")
            print(f"  Materialized mapping to : {out_path}")
            n_computed += 1
        except Exception as e:
            failures.append((slide_id, str(e)))
            print(f"\nERROR while processing slide '{slide_id}': {e}")

    total_elapsed = time.time() - global_start_time
    print("\n" + "=" * 100)
    print("Done.")
    print("=" * 100)
    print(f"Computed mappings : {n_computed}")
    print(f"Loaded from cache : {n_loaded_from_cache}")
    print(f"Failures          : {len(failures)}")
    print(f"Slides with 0 mapped genes : {len(zero_mapping_slides)}")
    if zero_mapping_slides:
        print("Slides with 0 mapped genes:")
        for slide_id in zero_mapping_slides:
            print(f"  - {slide_id}")
    print(f"Total elapsed     : {total_elapsed:.1f}s")

    if failures:
        print("\nFailure details:")
        for slide_id, err in failures:
            print(f"  - {slide_id}: {err}")
        raise RuntimeError("Some slides failed. See messages above.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build per-slide mouse->human gene-name mapping files for Mus musculus HEST eval slides."
    )
    parser.add_argument("--training_dataset_id", type=str,
                        required=True,
                        help="Training dataset ID, e.g. ds_ct1_20x_224_o112_1")

    parser.add_argument("--allow_live_rest",
                        action="store_true",
                        help=("Allow live Ensembl REST queries to create missing cache files (default: cached mappings only)."))

    parser.add_argument("--refresh_cached_mappings",
                        action="store_true",
                        help=("Recompute and overwrite canonical cached mappings. "
                              "Requires --allow_live_rest."))

    args = parser.parse_args()
    main(args)