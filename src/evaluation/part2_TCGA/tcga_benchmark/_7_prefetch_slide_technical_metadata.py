#!/usr/bin/env python3
"""
Estimate the MPP and geometry of each provisional slide selected by _6_resolve_rna_and_slides.py before the full
download, from a few HTTP Range reads of its TIFF header and first IFD (inconclusive slides stay eligible).

Saved in <ROOT_HistoST> / eval / eval_datasets / TCGA_eval_data /
    - benchmarks/<benchmark_id>/shared/cohort/: slide_prefetch_manifest.parquet, slide_prefetch_manifest.json,
      slide_prefetch_audit.csv, slide_prefetch_exclusion_log.csv
    - benchmarks/<benchmark_id>/reports/slide_prefetch_report.md
    - selected_slide_cache/<benchmark_id>/prefetch/by_uuid/<uuid>/header_parse.json
    - selected_slide_cache/_store/prefetch/by_uuid/<uuid>/technical_metadata.json: cache shared by all benchmarks
"""

import argparse
import hashlib
import json
import math
import os
import re
import struct
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import pandas as pd
import requests
import sys
from tqdm import tqdm

# Add the project root to sys.path so that 'src' can be imported
project_root = Path(__file__).resolve().parents[4]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.constants import get_full_path

from _taskb_common import (
    DEFAULT_GDC_API_BASE,
    collect_output_hashes,
    jsonish,
    manifest_header,
    read_json,
    read_yaml,
    sha256_obj,
    sha256_path,
    shared_prefetch_metadata_path,
    utc_now_iso,
    write_csv_atomic,
    write_json_atomic,
    write_parquet_atomic,
    write_text_atomic
)

# IFD-chase settings (a non-TIFF verdict needs >= 4096 bytes read, see _status_from_prefetch)
DEFAULT_HEADER_PROBE_BYTES = 4096         # bytes 0..N: TIFF/BigTIFF header + bad-magic conclusiveness
DEFAULT_IFD_WINDOW_BYTES = 256 * 1024     # window read at the IFD0 offset (entries + Aperio desc)
DEFAULT_MAX_METADATA_BYTES = 8 * 1024 * 1024   # hard ceiling on total bytes read per slide
DEFAULT_MAX_WORKERS = 16                   # concurrent slides; the work is latency-bound
PARSER_ENGINE = "tiff_svs_ifd_chase_v2"
# Defaults of the unused --prefetch_bytes_* args
DEFAULT_PREFETCH_BYTES_INITIAL = 4 * 1024 * 1024
DEFAULT_PREFETCH_BYTES_MAX = 64 * 1024 * 1024
# MPP acceptance bounds come from the benchmark's model group (--mpp_* overrides for debugging)
WSI_EXTENSIONS = {".svs", ".tif", ".tiff", ".ndpi", ".mrxs", ".scn", ".vms", ".vmu", ".bif"}

REQUIRED_OUTPUT_COLUMNS = [
    "slide_file_uuid", "case_id", "case_uuid", "project_id", "slide_submitter_id",
    "biospecimen_match_level", "match_tier_best_case", "official_primary_match",
    "file_name", "slide_file_name", "file_size", "md5sum",
    "requested_range", "http_status", "accept_ranges_header", "content_range_header",
    "content_length_header", "etag", "last_modified", "bytes_received", "partial_sha256",
    "range_supported", "server_response_headers_hash", "request_timestamp", "retry_count", "failure_reason",
    "partial_parse_success", "partial_parse_engine", "partial_parse_error",
    "prefetch_openslide_readable", "prefetch_tiff_readable", "prefetch_mpp_x", "prefetch_mpp_y",
    "prefetch_objective_power", "prefetch_slide_width_l0", "prefetch_slide_height_l0",
    "prefetch_vendor", "prefetch_openslide_properties", "prefetch_image_description",
    "mpp_source", "mpp_confidence", "prefetch_mpp_valid_range", "prefetch_mpp_anisotropy_ok",
    "technical_prefetch_status", "prefetch_exclusion_reason", "eligible_for_full_download",
    "is_bigtiff", "ifd0_offset", "prefetch_requests_count", "prefetch_bytes_total",
    "prefetch_served_from_cache",
    "benchmark_id", "modelgroup_id",
]


def _load_mapping(path: Path) -> Dict[str, Any]:
    obj = read_json(path) if path.suffix.lower() == ".json" else read_yaml(path)
    if not isinstance(obj, Mapping):
        raise RuntimeError(f"Expected mapping at {path}")
    return dict(obj)


def _nonempty(x: Any) -> bool:
    if x is None:
        return False
    try:
        if pd.isna(x):
            return False
    except Exception:
        pass
    s = str(x).strip()
    return bool(s and s.lower() not in {"nan", "none", "null", "na", "n/a", "<na>"})


def _boolish(x: Any) -> bool:
    if isinstance(x, bool):
        return x
    if not _nonempty(x):
        return False
    return str(x).strip().lower() in {"true", "t", "1", "yes", "y"}


def _float_or_none(x: Any) -> Optional[float]:
    if not _nonempty(x):
        return None
    try:
        v = float(str(x).strip())
    except Exception:
        return None
    if not math.isfinite(v):
        return None
    return v


def _headers_hash(headers: Mapping[str, Any]) -> str:
    return sha256_obj({str(k).lower(): str(v) for k, v in sorted(headers.items(), key=lambda kv: str(kv[0]).lower())})


def _read_stream_limited(resp: requests.Response, max_bytes: int) -> bytes:
    chunks: List[bytes] = []
    total = 0
    for chunk in resp.iter_content(chunk_size=256 * 1024):
        if not chunk:
            continue
        remaining = max_bytes - total
        if remaining <= 0:
            break
        if len(chunk) > remaining:
            chunk = chunk[:remaining]
        chunks.append(chunk)
        total += len(chunk)
        if total >= max_bytes:
            break
    return b"".join(chunks)


def _range_valid(content_range: Any) -> bool:
    if not _nonempty(content_range):
        return False
    # The "bytes " unit prefix is optional (the GDC API omits it)
    return bool(re.match(r"^(?:bytes\s+)?\d+-\d+/(\d+|\*)$", str(content_range).strip(), flags=re.IGNORECASE))


class _RangeFetcher:
    """Fetch byte ranges of a GDC /data/{uuid} URL with HTTP Range requests and record the per-slide HTTP metadata.

    After the first failure (including a 200 or 416 answer, or max_total_bytes reached), fetch() returns None.
    """

    def __init__(self, session: requests.Session, url: str, *, timeout: int, retries: int,
                 backoff: float, max_total_bytes: int) -> None:
        self.session = session
        self.url = url
        self.timeout = int(timeout)
        self.retries = int(retries)
        self.backoff = float(backoff)
        self.max_total_bytes = int(max_total_bytes)
        self.meta: Dict[str, Any] = {
            "requested_range": None,
            "http_status": None,
            "accept_ranges_header": None,
            "content_range_header": None,
            "content_length_header": None,
            "etag": None,
            "last_modified": None,
            "bytes_received": 0,
            "partial_sha256": None,
            "range_supported": False,
            "server_response_headers_hash": None,
            "request_timestamp": None,
            "retry_count": 0,
            "failure_reason": "",
            "requests_count": 0,
            "prefetch_bytes_total": 0,
        }
        self._first_recorded = False
        self._failed = False
        self._sha = hashlib.sha256()

    def failed(self) -> bool:
        return self._failed

    def _set_failure(self, reason: str) -> None:
        if not self.meta["failure_reason"]:
            self.meta["failure_reason"] = reason
        self._failed = True

    def _record_first(self, resp: requests.Response, requested_range: str) -> None:
        if self._first_recorded:
            return
        self.meta["requested_range"] = requested_range
        self.meta["http_status"] = int(resp.status_code)
        self.meta["accept_ranges_header"] = resp.headers.get("Accept-Ranges")
        self.meta["content_range_header"] = resp.headers.get("Content-Range")
        self.meta["content_length_header"] = resp.headers.get("Content-Length")
        self.meta["etag"] = resp.headers.get("ETag")
        self.meta["last_modified"] = resp.headers.get("Last-Modified")
        self.meta["server_response_headers_hash"] = _headers_hash(resp.headers)
        self._first_recorded = True

    def _account(self, data: bytes) -> None:
        n = len(data)
        self.meta["bytes_received"] = int(self.meta["bytes_received"]) + n
        self.meta["prefetch_bytes_total"] = int(self.meta["prefetch_bytes_total"]) + n
        self._sha.update(data)
        self.meta["partial_sha256"] = self._sha.hexdigest()

    def fetch(self, start: int, length: int) -> Optional[bytes]:
        """Return up to `length` bytes starting at absolute offset `start`, or None if a prior
        read already failed, the per-slide byte ceiling is hit, or this read fails."""
        if self._failed or start < 0 or length <= 0:
            return None
        remaining_budget = self.max_total_bytes - int(self.meta["prefetch_bytes_total"])
        if remaining_budget <= 0:
            self._set_failure("max_metadata_bytes_exceeded")
            return None
        length = min(length, remaining_budget)
        requested_range = f"bytes={start}-{start + length - 1}"
        last_error: Optional[str] = None
        for attempt in range(self.retries + 1):
            self.meta["requests_count"] = int(self.meta["requests_count"]) + 1
            self.meta["retry_count"] = attempt
            self.meta["request_timestamp"] = utc_now_iso()
            try:
                resp = self.session.get(self.url, headers={"Range": requested_range}, stream=True,
                                        timeout=self.timeout, allow_redirects=True)
                self._record_first(resp, requested_range)
                if resp.status_code == 206 and _range_valid(resp.headers.get("Content-Range")):
                    self.meta["range_supported"] = True
                    data = _read_stream_limited(resp, length)
                    self._account(data)
                    return data
                if resp.status_code == 200:
                    # Range ignored. Never read a whole WSI; treat as range-not-supported.
                    resp.close()
                    self._set_failure("server_returned_200_ok_to_range_request")
                    return None
                if resp.status_code == 416:
                    resp.close()
                    self._set_failure("http_416_range_not_satisfiable")
                    return None
                if resp.status_code in {429, 500, 502, 503, 504} and attempt < self.retries:
                    resp.close()
                    time.sleep(self.backoff ** attempt)
                    continue
                resp.close()
                self._set_failure(f"http_status_{resp.status_code}")
                return None
            except Exception as e:
                last_error = f"{type(e).__name__}: {e}"
                if attempt < self.retries:
                    time.sleep(self.backoff ** attempt)
                    continue
                self._set_failure(last_error)
                return None
        return None


def _safe_decode_ascii(data: bytes, limit: int = 1024 * 1024) -> str:
    # TIFF image descriptions are ASCII-ish; latin-1 preserves byte positions and never fails.
    return data[:limit].decode("latin-1", errors="ignore")


def _read_num(buf: bytes, offset: int, size: int, endian: str) -> Optional[int]:
    if offset < 0 or offset + size > len(buf):
        return None
    fmt = {("<", 2): "<H", (">", 2): ">H", ("<", 4): "<I", (">", 4): ">I", ("<", 8): "<Q", (">", 8): ">Q"}.get((endian, size))
    if fmt is None:
        return None
    return int(struct.unpack(fmt, buf[offset : offset + size])[0])


# TIFF field type -> byte size of one element. Covers classic + BigTIFF types
# (16/17/18 = LONG8/SLONG8/IFD8, BigTIFF only).
_TIFF_TYPE_SIZES = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 6: 1, 7: 1, 8: 2, 9: 4, 10: 8, 11: 4, 12: 8, 16: 8, 17: 8, 18: 8}


def _first_float_regex(patterns: Iterable[str], text: str) -> Optional[float]:
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            v = _float_or_none(m.group(1))
            if v is not None:
                return v
    return None


def _extract_mpp_and_objective(text: str) -> Dict[str, Any]:
    # Common SVS/Aperio: "MPP = 0.2520" and "AppMag = 40".
    mpp_x = _first_float_regex([
        r"openslide\.mpp-x\s*[=:]\s*([0-9]*\.?[0-9]+)",
        r"mpp[_\s-]*x\s*[=:]\s*([0-9]*\.?[0-9]+)",
        r"microns?\s+per\s+pixel\s*x?\s*[=:]\s*([0-9]*\.?[0-9]+)",
    ], text)
    mpp_y = _first_float_regex([
        r"openslide\.mpp-y\s*[=:]\s*([0-9]*\.?[0-9]+)",
        r"mpp[_\s-]*y\s*[=:]\s*([0-9]*\.?[0-9]+)",
    ], text)
    mpp_both = _first_float_regex([
        r"(?:^|[|;\n\r\t ])MPP\s*[=:]\s*([0-9]*\.?[0-9]+)",
        r"aperio\.MPP\s*[=:]\s*([0-9]*\.?[0-9]+)",
    ], text)
    source = None
    if mpp_both is not None:
        mpp_x = mpp_x if mpp_x is not None else mpp_both
        mpp_y = mpp_y if mpp_y is not None else mpp_both
        source = "image_description_MPP"
    elif mpp_x is not None or mpp_y is not None:
        source = "image_description_mpp_x_y"

    objective = _first_float_regex([
        r"AppMag\s*[=:]\s*([0-9]*\.?[0-9]+)",
        r"objective[-_\s]*power\s*[=:]\s*([0-9]*\.?[0-9]+)",
        r"openslide\.objective-power\s*[=:]\s*([0-9]*\.?[0-9]+)",
    ], text)
    vendor = None
    low = text.lower()
    for candidate in ["Aperio", "Hamamatsu", "Leica", "Philips", "Ventana", "Mirax", "Sakura"]:
        if candidate.lower() in low:
            vendor = candidate
            break
    return {
        "prefetch_mpp_x": mpp_x,
        "prefetch_mpp_y": mpp_y,
        "prefetch_objective_power": objective,
        "prefetch_vendor": vendor,
        "mpp_source": source,
        "mpp_confidence": "high" if mpp_x is not None and mpp_y is not None else "missing",
    }


def _empty_parse(error: str, *, tiff_readable: bool = False) -> Dict[str, Any]:
    return {
        "partial_parse_success": False,
        "partial_parse_engine": PARSER_ENGINE,
        "partial_parse_error": error,
        "prefetch_tiff_readable": bool(tiff_readable),
        "prefetch_mpp_x": None,
        "prefetch_mpp_y": None,
        "prefetch_objective_power": None,
        "prefetch_slide_width_l0": None,
        "prefetch_slide_height_l0": None,
        "prefetch_vendor": None,
        "prefetch_image_description": None,
        "mpp_source": None,
        "mpp_confidence": "missing",
        "is_bigtiff": None,
        "ifd0_offset": None,
    }


def _parse_ifd(win: bytes, *, base: int, endian: str, bigtiff: bool) -> Tuple[Dict[int, Dict[str, Any]], str]:
    """Parse the IFD entries of a window read at base. Returns ({tag: {type, count, raw}}, error).
    Classic TIFF: 12-byte entries, 2-byte count; BigTIFF: 20-byte entries, 8-byte count and offsets."""
    entries: Dict[int, Dict[str, Any]] = {}
    off_size = 8 if bigtiff else 4
    if bigtiff:
        n = _read_num(win, 0, 8, endian); entry_start = 8
    else:
        n = _read_num(win, 0, 2, endian); entry_start = 2
    if n is None:
        return entries, "cannot_read_ifd_entry_count"
    entry_size = 4 + 2 * off_size
    max_in_window = max(0, (len(win) - entry_start) // entry_size)
    err = ""
    if n > max_in_window:
        err = "ifd_entries_truncated_in_window"
        n = max_in_window
    for i in range(int(n)):
        o = entry_start + i * entry_size
        tag = _read_num(win, o, 2, endian)
        type_id = _read_num(win, o + 2, 2, endian)
        count = _read_num(win, o + 4, off_size, endian)
        if tag is None or type_id is None or count is None:
            continue
        entries[int(tag)] = {"type": int(type_id), "count": int(count), "raw": win[o + 4 + off_size: o + entry_size]}
    return entries, err


def _ifd_scalar(entry: Optional[Mapping[str, Any]], endian: str) -> Optional[int]:
    """Read a single inline integer (SHORT/LONG/LONG8) from an IFD entry's value field."""
    if not entry:
        return None
    raw = entry.get("raw") or b""
    t = entry.get("type")
    try:
        if t == 3 and len(raw) >= 2:
            return _read_num(raw, 0, 2, endian)
        if t == 4 and len(raw) >= 4:
            return _read_num(raw, 0, 4, endian)
        if t == 16 and len(raw) >= 8:
            return _read_num(raw, 0, 8, endian)
    except Exception:
        return None
    return None


def _field_bytes(fetcher: "_RangeFetcher", entry: Optional[Mapping[str, Any]], win: bytes, base: int,
                 endian: str, bigtiff: bool) -> Optional[bytes]:
    """Resolve an IFD entry's value bytes: inline if they fit the value slot, else from the
    already-read window when they fall inside it, else one more targeted Range read."""
    if not entry:
        return None
    type_id = entry.get("type"); count = entry.get("count"); raw = entry.get("raw") or b""
    unit = _TIFF_TYPE_SIZES.get(type_id)
    if unit is None or count is None or count < 0:
        return None
    n = unit * count
    if n == 0:
        return b""
    off_size = 8 if bigtiff else 4
    if n <= off_size:
        return raw[:n]
    off = _read_num(raw, 0, off_size, endian)
    if off is None or off < 0:
        return None
    lo = off - base
    if lo >= 0 and lo + n <= len(win):
        return win[lo:lo + n]
    return fetcher.fetch(off, n)


def _mpp_from_resolution(fetcher: "_RangeFetcher", entries: Mapping[int, Dict[str, Any]], win: bytes,
                         base: int, endian: str, bigtiff: bool) -> Optional[Dict[str, Any]]:
    """Fallback MPP from XResolution/YResolution + ResolutionUnit (2=inch, 3=cm); returns only the MPP keys."""
    def _rational(tag: int) -> Optional[float]:
        b = _field_bytes(fetcher, entries.get(tag), win, base, endian, bigtiff)
        if not b or len(b) < 8:
            return None
        num = _read_num(b, 0, 4, endian); den = _read_num(b, 4, 4, endian)
        if not num or not den:
            return None
        return float(num) / float(den)

    xres = _rational(282); yres = _rational(283)
    if xres is None or yres is None:
        return None
    unit_entry = entries.get(296)
    unit = _ifd_scalar(unit_entry, endian) if unit_entry else 2
    if unit not in (2, 3):  # 1=none is not convertible to microns
        return None
    per_unit_um = 25400.0 if unit == 2 else 10000.0
    mpp_x = per_unit_um / xres
    mpp_y = per_unit_um / yres
    if not (0 < mpp_x < 100 and 0 < mpp_y < 100):  # sanity guard against garbage resolutions
        return None
    return {"prefetch_mpp_x": mpp_x, "prefetch_mpp_y": mpp_y,
            "mpp_source": "resolution_tags", "mpp_confidence": "high"}


def _chase_tiff_metadata(fetcher: "_RangeFetcher", file_name: str, *, header_probe_bytes: int,
                         ifd_window_bytes: int) -> Dict[str, Any]:
    """Recover MPP/geometry with a few Range reads: TIFF header, a window at IFD0 (entries + ImageDescription),
    and, if needed, one read of a value stored outside that window."""
    head = fetcher.fetch(0, int(header_probe_bytes))
    if not head or len(head) < 8:
        return _empty_parse(fetcher.meta.get("failure_reason") or "too_few_bytes_for_tiff_header")
    if head[:2] == b"II":
        endian = "<"
    elif head[:2] == b"MM":
        endian = ">"
    else:
        out = _empty_parse("not_tiff_magic")
        out.update(_extract_mpp_and_objective(_safe_decode_ascii(head)))
        return out

    version = _read_num(head, 2, 2, endian)
    if version not in {42, 43}:
        return _empty_parse(f"unexpected_tiff_magic_{version}")
    bigtiff = (version == 43)
    out = _empty_parse("", tiff_readable=True)
    out["is_bigtiff"] = bool(bigtiff)

    if bigtiff:
        offset_bytesize = _read_num(head, 4, 2, endian)
        if offset_bytesize != 8:
            out["partial_parse_error"] = f"unsupported_bigtiff_offset_bytesize_{offset_bytesize}"
            out.update(_extract_mpp_and_objective(_safe_decode_ascii(head)))
            return out
        ifd0 = _read_num(head, 8, 8, endian)
    else:
        ifd0 = _read_num(head, 4, 4, endian)
    if ifd0 is None or ifd0 < 8:
        out["partial_parse_error"] = "invalid_ifd0_offset"
        return out
    out["ifd0_offset"] = int(ifd0)

    win = fetcher.fetch(int(ifd0), int(ifd_window_bytes))
    if not win or len(win) < (8 if bigtiff else 2):
        out["partial_parse_error"] = fetcher.meta.get("failure_reason") or "could_not_read_ifd_window"
        return out

    entries, parse_err = _parse_ifd(win, base=int(ifd0), endian=endian, bigtiff=bigtiff)
    if parse_err:
        out["partial_parse_error"] = parse_err

    width = _ifd_scalar(entries.get(256), endian)
    height = _ifd_scalar(entries.get(257), endian)
    if width is not None:
        out["prefetch_slide_width_l0"] = int(width)
    if height is not None:
        out["prefetch_slide_height_l0"] = int(height)

    desc_bytes = _field_bytes(fetcher, entries.get(270), win, int(ifd0), endian, bigtiff)
    image_description = desc_bytes.decode("latin-1", errors="ignore").rstrip("\x00") if desc_bytes else None
    if image_description:
        out["prefetch_image_description"] = image_description[:20000]

    combined_text = "\n".join([t for t in [image_description, _safe_decode_ascii(win)] if t])
    out.update(_extract_mpp_and_objective(combined_text))

    if out["prefetch_mpp_x"] is None or out["prefetch_mpp_y"] is None:
        res = _mpp_from_resolution(fetcher, entries, win, int(ifd0), endian, bigtiff)
        if res:
            out.update(res)

    has_mpp = out["prefetch_mpp_x"] is not None and out["prefetch_mpp_y"] is not None
    out["partial_parse_success"] = bool(has_mpp or image_description or width is not None or height is not None)
    return out


def _mpp_flags(x: Optional[float], y: Optional[float], *, mpp_min: float, mpp_max: float, anisotropy_max: float) -> Tuple[Optional[bool], Optional[bool], Optional[str]]:
    if x is None or y is None:
        return None, None, None
    valid_range = (mpp_min <= x <= mpp_max) and (mpp_min <= y <= mpp_max)
    if x <= 0 or y <= 0:
        anis_ok = False
    else:
        anis_ok = max(x, y) / min(x, y) <= anisotropy_max
    reason = None
    if not valid_range:
        reason = f"mpp_outside_allowed_range_{mpp_min}_{mpp_max}"
    elif not anis_ok:
        reason = f"mpp_anisotropy_exceeds_{anisotropy_max}"
    return valid_range, anis_ok, reason


def _status_from_prefetch(row: Mapping[str, Any], parse: Mapping[str, Any], *, file_name: str, mpp_min: float, mpp_max: float, anisotropy_max: float) -> Tuple[str, str, bool, Optional[bool], Optional[bool]]:
    http_status = row.get("http_status")
    range_supported = bool(row.get("range_supported"))
    bytes_received = int(row.get("bytes_received") or 0)
    failure_reason = str(row.get("failure_reason") or "")
    x = _float_or_none(parse.get("prefetch_mpp_x"))
    y = _float_or_none(parse.get("prefetch_mpp_y"))
    valid_range, anis_ok, mpp_reason = _mpp_flags(x, y, mpp_min=mpp_min, mpp_max=mpp_max, anisotropy_max=anisotropy_max)

    if http_status is None:
        return "prefetch_error_final", failure_reason or "request_failed_without_http_status", True, valid_range, anis_ok
    if not range_supported:
        if int(http_status) == 200:
            return "range_not_supported", "server_returned_200_ok_to_range_request", True, valid_range, anis_ok
        if int(http_status) == 416:
            return "range_not_supported", "http_416_range_not_satisfiable", True, valid_range, anis_ok
        if int(http_status) in {429, 500, 502, 503, 504}:
            return "prefetch_error_retryable", failure_reason or f"http_status_{http_status}", True, valid_range, anis_ok
        return "prefetch_error_final", failure_reason or f"http_status_{http_status}", True, valid_range, anis_ok

    # From here on a valid 206 range response has been received.
    if bytes_received <= 0:
        return "inconclusive_prefetch", "range_supported_but_no_bytes_received", True, valid_range, anis_ok

    tiff_readable = parse.get("prefetch_tiff_readable") is True
    parse_error = str(parse.get("partial_parse_error") or "")
    suffix = Path(file_name or "").suffix.lower()
    if not tiff_readable and bytes_received >= 4096:
        # Invalid TIFF magic in a WSI file is conclusive
        if suffix in WSI_EXTENSIONS or not suffix:
            return "failed_prefetch_conclusive", "failed_unreadable_header_not_tiff_magic", False, valid_range, anis_ok
        return "failed_prefetch_conclusive", "failed_not_wsi_extension_and_header", False, valid_range, anis_ok

    if x is not None and y is not None:
        if valid_range and anis_ok:
            return "passed_prefetch", "", True, valid_range, anis_ok
        return "failed_prefetch_conclusive", mpp_reason or "mpp_invalid", False, valid_range, anis_ok

    if tiff_readable:
        return "inconclusive_prefetch", parse_error or "mpp_missing_from_partial_header", True, valid_range, anis_ok
    return "inconclusive_prefetch", parse_error or "partial_header_inconclusive", True, valid_range, anis_ok


def _ordered(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    out = df.copy()
    for c in columns:
        if c not in out.columns:
            out[c] = pd.NA
    extras = [c for c in out.columns if c not in columns]
    return out.loc[:, list(columns) + extras]


def _software_optional_versions() -> Dict[str, str]:
    out: Dict[str, str] = {}
    try:
        import openslide  # type: ignore
        out["openslide_python"] = getattr(openslide, "__version__", "unknown")
        try:
            out["openslide_library"] = openslide.__library_version__  # type: ignore[attr-defined]
        except Exception:
            out["openslide_library"] = "unknown"
    except Exception:
        out["openslide_python"] = "not_importable"
        out["openslide_library"] = "not_importable"
    try:
        import tifffile  # type: ignore
        out["tifffile"] = getattr(tifffile, "__version__", "unknown")
    except Exception:
        out["tifffile"] = "not_importable"
    out["tiff_parser"] = PARSER_ENGINE
    return out


def _write_report(path: Path, summary: Mapping[str, Any], *, overwrite: bool) -> None:
    lines = [
        "# Slide technical-prefetch report",
        "",
        f"- benchmark_id: `{summary.get('benchmark_id')}`",
        f"- modelgroup_id: `{summary.get('modelgroup_id')}`",
        f"- slides examined: `{summary.get('n_slides')}`",
        f"- passed prefetch: `{summary.get('n_passed_prefetch')}`",
        f"- conclusively excluded: `{summary.get('n_failed_prefetch_conclusive')}`",
        f"- inconclusive: `{summary.get('n_inconclusive_prefetch')}`",
        f"- range not supported: `{summary.get('n_range_not_supported')}`",
        "",
        "Inconclusive or unsupported-range slides remain eligible for full download by default.",
        "",
    ]
    write_text_atomic(path, "\n".join(lines), overwrite=overwrite)


def _preflight_gdc_reachable(session: requests.Session, gdc_api_base: str, *, timeout: int) -> None:
    """Raise if the GDC API status endpoint is unreachable (checked before prefetching)."""
    status_url = f"{gdc_api_base}/status"
    try:
        resp = session.get(status_url, timeout=timeout, allow_redirects=True)
    except Exception as e:
        raise RuntimeError(
            f"GDC preflight failed: cannot reach {status_url} ({type(e).__name__}: {e}). "
            "Check connectivity (offline, VPN, proxy, or DNS) before prefetching. "
            "Use --skip_preflight_check to bypass this guard."
        ) from e
    if resp.status_code != 200:
        raise RuntimeError(
            f"GDC preflight failed: {status_url} returned HTTP {resp.status_code}. "
            "The GDC API may be degraded. Use --skip_preflight_check to bypass this guard."
        )


def _process_slide(row: Mapping[str, Any], *, session: requests.Session, gdc_api_base: str,
                   data_root: Path, cache_dir: Path, modelgroup_id: str, args) -> Dict[str, Any]:
    """Prefetch and parse one slide's technical metadata; returns its row, audit and exclusion records (thread-safe)."""
    uuid = str(row["slide_file_uuid"]).strip()
    file_name = str(row.get("file_name") or row.get("slide_file_name") or "")
    url = f"{gdc_api_base}/data/{uuid}"
    # Parsed result cached in the shared store (--refresh_prefetch_cache forces a re-fetch)
    meta_path = shared_prefetch_metadata_path(data_root, uuid)

    http_meta: Optional[Dict[str, Any]] = None
    parse: Optional[Dict[str, Any]] = None
    served_from_cache = False
    if (not args.refresh_prefetch_cache) and meta_path.exists():
        try:
            cached = read_json(meta_path)
            if (isinstance(cached, Mapping) and isinstance(cached.get("http_meta"), Mapping)
                    and isinstance(cached.get("parse"), Mapping)):
                http_meta = dict(cached["http_meta"])
                parse = dict(cached["parse"])
                served_from_cache = True
        except Exception:
            http_meta, parse, served_from_cache = None, None, False

    if http_meta is None or parse is None:
        fetcher = _RangeFetcher(session, url, timeout=int(args.timeout), retries=int(args.retries),
                                backoff=float(args.backoff), max_total_bytes=int(args.max_metadata_bytes))
        parse = _chase_tiff_metadata(fetcher, file_name,
                                     header_probe_bytes=int(args.header_probe_bytes),
                                     ifd_window_bytes=int(args.ifd_window_bytes))
        http_meta = dict(fetcher.meta)
        served_from_cache = False
        # Cache only results with an HTTP answer and no fetch failure
        if http_meta.get("http_status") is not None and not fetcher.failed():
            try:
                write_json_atomic(meta_path, {"http_meta": http_meta, "parse": parse}, overwrite=True)
            except Exception:
                pass

    http_meta = dict(http_meta)
    http_meta["prefetch_served_from_cache"] = served_from_cache
    parse = dict(parse)
    # OpenSlide is not run on the prefetched bytes (checked by _10_build_tile_asset.py)
    parse.setdefault("prefetch_openslide_readable", None)
    parse.setdefault("prefetch_openslide_error", None)
    parse.setdefault("prefetch_openslide_properties", None)

    status, exclusion_reason, eligible, valid_range, anis_ok = _status_from_prefetch(
        http_meta, parse, file_name=file_name,
        mpp_min=float(args.mpp_min), mpp_max=float(args.mpp_max), anisotropy_max=float(args.mpp_anisotropy_max),
    )

    out = dict(row)
    if "slide_file_name" not in out or not _nonempty(out.get("slide_file_name")):
        out["slide_file_name"] = file_name
    out.update(http_meta)
    out.update(parse)
    slide_cache_dir = cache_dir / uuid
    out.update({
        "prefetch_mpp_valid_range": valid_range,
        "prefetch_mpp_anisotropy_ok": anis_ok,
        "technical_prefetch_status": status,
        "prefetch_exclusion_reason": exclusion_reason,
        "eligible_for_full_download": bool(eligible),
        "is_bigtiff": parse.get("is_bigtiff"),
        "ifd0_offset": parse.get("ifd0_offset"),
        "prefetch_requests_count": int(http_meta.get("requests_count") or 0),
        "prefetch_bytes_total": int(http_meta.get("prefetch_bytes_total") or http_meta.get("bytes_received") or 0),
        "benchmark_id": args.benchmark_id,
        "modelgroup_id": modelgroup_id,
        "gdc_data_url": url,
        "prefetch_metadata_cache_path": str(meta_path),
        "header_parse_cache_path": str(slide_cache_dir / "header_parse.json"),
    })
    if out.get("prefetch_openslide_properties") is not None and not isinstance(out.get("prefetch_openslide_properties"), str):
        out["prefetch_openslide_properties"] = jsonish(out["prefetch_openslide_properties"])
    slide_cache_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(slide_cache_dir / "header_parse.json", out, overwrite=True)

    audit = {
        "slide_file_uuid": uuid,
        "case_id": out.get("case_id"),
        "project_id": out.get("project_id"),
        "requested_range": out.get("requested_range"),
        "http_status": out.get("http_status"),
        "range_supported": out.get("range_supported"),
        "bytes_received": out.get("bytes_received"),
        "prefetch_bytes_total": out.get("prefetch_bytes_total"),
        "prefetch_requests_count": out.get("prefetch_requests_count"),
        "partial_sha256": out.get("partial_sha256"),
        "partial_parse_success": out.get("partial_parse_success"),
        "prefetch_mpp_x": out.get("prefetch_mpp_x"),
        "prefetch_mpp_y": out.get("prefetch_mpp_y"),
        "is_bigtiff": out.get("is_bigtiff"),
        "technical_prefetch_status": status,
        "prefetch_exclusion_reason": exclusion_reason,
        "eligible_for_full_download": bool(eligible),
        "server_response_headers_hash": out.get("server_response_headers_hash"),
        "benchmark_id": args.benchmark_id,
    }
    exclusion = None
    if not eligible:
        exclusion = {
            "stage": "slide_prefetch",
            "entity_type": "slide_file",
            "entity_id": uuid,
            "case_id": out.get("case_id"),
            "case_uuid": out.get("case_uuid"),
            "project_id": out.get("project_id"),
            "technical_prefetch_status": status,
            "reason": exclusion_reason or status,
            "detail": jsonish({
                "prefetch_mpp_x": out.get("prefetch_mpp_x"),
                "prefetch_mpp_y": out.get("prefetch_mpp_y"),
                "partial_parse_error": out.get("partial_parse_error"),
                "http_status": out.get("http_status"),
            }),
            "benchmark_id": args.benchmark_id,
        }
    return {"uuid": uuid, "status": status, "out": out, "audit": audit, "exclusion": exclusion}


def main(args) -> int:

    if int(args.max_workers) < 1:
        raise RuntimeError("--max_workers must be >= 1")
    if int(args.header_probe_bytes) < 16 or int(args.ifd_window_bytes) < 64:
        raise RuntimeError("--header_probe_bytes must be >= 16 and --ifd_window_bytes >= 64")
    if int(args.max_metadata_bytes) < int(args.ifd_window_bytes):
        raise RuntimeError("--max_metadata_bytes must be >= --ifd_window_bytes")

    data_root = get_full_path('eval', 'eval_datasets', 'TCGA_eval_data')
    benchmark_root = data_root / "benchmarks" / args.benchmark_id
    settings_dir = benchmark_root / "settings"
    cohort_dir = benchmark_root / "shared" / "cohort"
    reports_dir = benchmark_root / "reports"
    # Per-benchmark prefetch dir: header_parse.json per slide
    cache_dir = data_root / "selected_slide_cache" / args.benchmark_id / "prefetch" / "by_uuid"
    if not settings_dir.exists():
        raise FileNotFoundError(settings_dir)
    if not cohort_dir.exists():
        raise FileNotFoundError(cohort_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    output_paths = [
        cohort_dir / "slide_prefetch_manifest.parquet",
        cohort_dir / "slide_prefetch_audit.csv",
        cohort_dir / "slide_prefetch_exclusion_log.csv",
        cohort_dir / "slide_prefetch_manifest.json",
    ]
    if any(p.exists() for p in output_paths) and not args.force:
        existing = [str(p) for p in output_paths if p.exists()]
        raise FileExistsError(f"Prefetch outputs already exist; use --force to rebuild. Existing: {existing}")
    # Downstream outputs: the per-benchmark download manifest and the tile assets
    downstream = [data_root / "selected_slide_cache" / args.benchmark_id / "manifests" / "selected_slide_download_manifest.tsv", data_root / "processed_assets" / "tiles"]
    if args.force and any(p.exists() for p in downstream) and not args.force_cascade:
        raise RuntimeError(f"Refusing to rebuild prefetch outputs while downstream slide downloads/tile assets exist: {downstream}. Use --force_cascade if intentional.")

    cfg = _load_mapping(settings_dir / "benchmark_config.yaml")
    links_path = settings_dir / "benchmark_asset_links.json"
    links = read_json(links_path) if links_path.exists() else {}
    if not isinstance(links, Mapping):
        raise RuntimeError("benchmark_asset_links.json must be a mapping when present")
    modelgroup_id = str(cfg.get("modelgroup_id") or links.get("modelgroup_id"))
    # MPP acceptance bounds from the benchmark's model group (as in _10_build_tile_asset.py), unless --mpp_* are
    # given; the model group path is rebuilt under the current root (not read from benchmark_asset_links.json)
    modelgroup_path = settings_dir / "modelgroups" / f"{modelgroup_id}.yaml"
    if not modelgroup_path.exists():
        raise FileNotFoundError(modelgroup_path)
    modelgroup_cfg = _load_mapping(modelgroup_path)
    mpp_bounds_source = "cli_override" if any(
        v is not None for v in (args.mpp_min, args.mpp_max, args.mpp_anisotropy_max)
    ) else "model_group"
    args.mpp_min = float(args.mpp_min) if args.mpp_min is not None else float(modelgroup_cfg["mpp_min"])
    args.mpp_max = float(args.mpp_max) if args.mpp_max is not None else float(modelgroup_cfg["mpp_max"])
    args.mpp_anisotropy_max = (
        float(args.mpp_anisotropy_max) if args.mpp_anisotropy_max is not None
        else float(modelgroup_cfg["mpp_anisotropy_max"])
    )
    if args.mpp_min <= 0 or args.mpp_max <= args.mpp_min or args.mpp_anisotropy_max < 1.0:
        raise RuntimeError(
            f"Invalid resolved MPP bounds (source={mpp_bounds_source}): "
            f"min={args.mpp_min}, max={args.mpp_max}, anisotropy_max={args.mpp_anisotropy_max}"
        )
    source_root = data_root / "source_releases" / str(cfg.get("tcga_source_id"))  # under the current root
    source_config_path = source_root / "settings" / "source_config.yaml"
    source_config = _load_mapping(source_config_path) if source_config_path.exists() else {}
    gdc_api_base = (args.gdc_api_base or source_config.get("gdc_api_base_url") or DEFAULT_GDC_API_BASE).rstrip("/")

    slide_manifest_csv = cohort_dir / "slide_manifest.csv"
    if not slide_manifest_csv.exists():
        raise FileNotFoundError(f"Run _6_resolve_rna_and_slides.py first; missing {slide_manifest_csv}")
    slide_manifest = pd.read_csv(slide_manifest_csv, dtype=str, keep_default_na=False)
    if slide_manifest.empty:
        raise RuntimeError("slide_manifest.csv is empty; no provisional official-primary slides to prefetch")
    required = ["slide_file_uuid", "case_id", "case_uuid", "project_id", "file_name", "md5sum", "official_primary_match"]
    missing = [c for c in required if c not in slide_manifest.columns]
    if missing:
        raise RuntimeError(f"slide_manifest.csv missing required columns: {missing}")
    if args.limit is not None:
        slide_manifest = slide_manifest.head(args.limit).copy()

    source_slide_inv = source_root / "inventory" / "candidate_slide_files.parquet"
    source_versions = source_root / "inventory" / "file_versions.parquet"
    if not source_slide_inv.exists():
        raise FileNotFoundError(source_slide_inv)
    if not source_versions.exists():
        raise FileNotFoundError(source_versions)

    session = requests.Session()
    session.headers.update({"User-Agent": "HistoST-TCGA-TaskB-prefetch/2.0"})
    if not args.skip_preflight_check:
        _preflight_gdc_reachable(session, gdc_api_base, timeout=min(int(args.timeout), 30))

    rows: List[Dict[str, Any]] = []
    audit_rows: List[Dict[str, Any]] = []
    exclusion_rows: List[Dict[str, Any]] = []

    records = slide_manifest.sort_values(["project_id", "case_id", "slide_file_uuid"]).to_dict("records")
    n_records = len(records)
    max_workers = max(1, min(int(args.max_workers), n_records or 1))
    print(f"[prefetch] {n_records} slide(s) to prefetch for '{args.benchmark_id}' via {gdc_api_base} "
          f"(targeted IFD-chase, {max_workers} worker(s))", flush=True)
    status_tally: Dict[str, int] = {}
    abort_window = int(args.abort_after_consecutive_errors)
    recent_errors: "deque[bool]" = deque(maxlen=abort_window if abort_window > 0 else 1)
    abort_error: Optional[RuntimeError] = None

    def _safe_process(record: Mapping[str, Any]) -> Dict[str, Any]:
        # An exception in one slide becomes an error row
        try:
            return _process_slide(record, session=session, gdc_api_base=gdc_api_base, data_root=data_root,
                                  cache_dir=cache_dir, modelgroup_id=modelgroup_id, args=args)
        except Exception as e:
            uuid = str(record.get("slide_file_uuid") or "").strip()
            out = dict(record)
            out.update({
                "technical_prefetch_status": "prefetch_error_final",
                "prefetch_exclusion_reason": f"worker_exception: {type(e).__name__}: {e}",
                "failure_reason": f"worker_exception: {type(e).__name__}: {e}",
                "http_status": None,
                "eligible_for_full_download": True,
                "benchmark_id": args.benchmark_id,
                "modelgroup_id": modelgroup_id,
            })
            audit = {"slide_file_uuid": uuid, "case_id": record.get("case_id"), "project_id": record.get("project_id"),
                     "technical_prefetch_status": "prefetch_error_final", "eligible_for_full_download": True,
                     "benchmark_id": args.benchmark_id}
            return {"uuid": uuid, "status": "prefetch_error_final", "out": out, "audit": audit, "exclusion": None}

    # Process slides concurrently
    pbar = tqdm(total=n_records, desc="Prefetching slide headers", unit="slide")
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_safe_process, rec) for rec in records]
        try:
            for fut in as_completed(futures):
                res = fut.result()
                status = res["status"]
                status_tally[status] = status_tally.get(status, 0) + 1
                rows.append(res["out"])
                if res.get("audit"):
                    audit_rows.append(res["audit"])
                if res.get("exclusion"):
                    exclusion_rows.append(res["exclusion"])
                # Abort if the last N completed slides all failed
                recent_errors.append(status == "prefetch_error_final")
                pbar.update(1)
                pbar.set_postfix(
                    passed=status_tally.get("passed_prefetch", 0),
                    failed=status_tally.get("failed_prefetch_conclusive", 0),
                    inconcl=status_tally.get("inconclusive_prefetch", 0),
                    err=status_tally.get("prefetch_error_final", 0),
                )
                if abort_window > 0 and len(recent_errors) == abort_window and all(recent_errors):
                    abort_error = RuntimeError(
                        f"Aborting prefetch after {abort_window} consecutive 'prefetch_error_final' results. "
                        f"http_status=None means GDC at {gdc_api_base} was unreachable (offline, VPN, proxy, or DNS); "
                        "a repeated HTTP error code instead points to access/auth or stale file UUIDs. "
                        "Fix the cause and re-run; pass --abort_after_consecutive_errors 0 to disable this guard."
                    )
                    break
        finally:
            pbar.close()
            if abort_error is not None:
                ex.shutdown(wait=False, cancel_futures=True)
    if abort_error is not None:
        raise abort_error

    # Sort rows (completion order varies)
    rows.sort(key=lambda r: (str(r.get("project_id") or ""), str(r.get("case_id") or ""),
                             str(r.get("slide_file_uuid") or "")))

    manifest_df = _ordered(pd.DataFrame(rows), REQUIRED_OUTPUT_COLUMNS)
    audit_df = pd.DataFrame(audit_rows).sort_values(["project_id", "case_id", "slide_file_uuid"]).reset_index(drop=True) if audit_rows else pd.DataFrame()
    exclusion_df = pd.DataFrame(exclusion_rows)
    if not exclusion_df.empty:
        exclusion_df = exclusion_df.sort_values(["project_id", "case_id", "entity_id"]).reset_index(drop=True)
    else:
        exclusion_df = pd.DataFrame(columns=["stage", "entity_type", "entity_id", "case_id", "case_uuid", "project_id", "technical_prefetch_status", "reason", "detail", "benchmark_id"])

    write_parquet_atomic(cohort_dir / "slide_prefetch_manifest.parquet", manifest_df, overwrite=args.force)
    write_csv_atomic(cohort_dir / "slide_prefetch_audit.csv", audit_df, overwrite=args.force)
    write_csv_atomic(cohort_dir / "slide_prefetch_exclusion_log.csv", exclusion_df, overwrite=args.force)

    status_counts = manifest_df["technical_prefetch_status"].value_counts(dropna=False).to_dict()
    summary = {
        "benchmark_id": args.benchmark_id,
        "modelgroup_id": modelgroup_id,
        "n_slides": int(len(manifest_df)),
        "n_passed_prefetch": int(status_counts.get("passed_prefetch", 0)),
        "n_failed_prefetch_conclusive": int(status_counts.get("failed_prefetch_conclusive", 0)),
        "n_inconclusive_prefetch": int(status_counts.get("inconclusive_prefetch", 0)),
        "n_range_not_supported": int(status_counts.get("range_not_supported", 0)),
        "n_prefetch_error_retryable": int(status_counts.get("prefetch_error_retryable", 0)),
        "n_prefetch_error_final": int(status_counts.get("prefetch_error_final", 0)),
        "n_eligible_for_full_download": int(manifest_df["eligible_for_full_download"].map(_boolish).sum()),
        "status_counts": {str(k): int(v) for k, v in status_counts.items()},
    }

    manifest = manifest_header(Path(__file__).name, vars(args))
    manifest.update({
        "benchmark_id": args.benchmark_id,
        "benchmark_root": str(benchmark_root),
        "source_root": str(source_root),
        "gdc_api_base_url": gdc_api_base,
        "modelgroup_id": modelgroup_id,
        "input_paths": {
            "slide_manifest_csv": str(slide_manifest_csv),
            "benchmark_config": str(settings_dir / "benchmark_config.yaml"),
            "benchmark_asset_links": str(links_path) if links_path.exists() else None,
            "modelgroup_yaml": str(settings_dir / "modelgroups" / f"{modelgroup_id}.yaml"),
            "candidate_slide_files": str(source_slide_inv),
            "file_versions": str(source_versions),
        },
        "input_hashes": {
            "slide_manifest_csv": sha256_path(slide_manifest_csv),
            "benchmark_config": sha256_path(settings_dir / "benchmark_config.yaml"),
            "benchmark_asset_links": sha256_path(links_path) if links_path.exists() else None,
            "modelgroup_yaml": sha256_path(settings_dir / "modelgroups" / f"{modelgroup_id}.yaml") if (settings_dir / "modelgroups" / f"{modelgroup_id}.yaml").exists() else None,
            "candidate_slide_files": sha256_path(source_slide_inv),
            "file_versions": sha256_path(source_versions),
        },
        "prefetch_policy": {
            "strategy": "targeted_ifd_chase_v2",
            "header_probe_bytes": int(args.header_probe_bytes),
            "ifd_window_bytes": int(args.ifd_window_bytes),
            "max_metadata_bytes": int(args.max_metadata_bytes),
            "max_workers": int(args.max_workers),
            "range_request": "GET /data/{slide_file_uuid} with Range bytes=START-END following the file's IFD0 pointer",
            "range_supported_rule": "206 Partial Content with a valid Content-Range (leading 'bytes' unit optional; GDC omits it)",
            "server_200_policy": "treat_200_to_range_as_range_not_supported_never_read_full_WSI",
            "bigtiff_supported": True,
            "mpp_sources": ["image_description_MPP", "image_description_mpp_x_y", "resolution_tags"],
            "inconclusive_default": "eligible_for_full_download_true",
            "conclusive_exclusion_status": "failed_prefetch_conclusive",
        },
        "header_parser": {
            "engine": PARSER_ENGINE,
            "openslide_partial_tested": False,
            "software_versions": _software_optional_versions(),
        },
        "mpp_thresholds": {
            "mpp_min": float(args.mpp_min),
            "mpp_max": float(args.mpp_max),
            "anisotropy_max_ratio": float(args.mpp_anisotropy_max),
            "bounds_source": mpp_bounds_source,
        },
        "summary": summary,
        "request_response_hashes": manifest_df[["slide_file_uuid", "server_response_headers_hash", "partial_sha256"]].to_dict("records"),
        "output_hashes": collect_output_hashes(cohort_dir),
        "cache_hashes": collect_output_hashes(data_root / "selected_slide_cache" / args.benchmark_id / "prefetch"),
        "definitive_validation_note": "OpenSlide readability, full MPP, read windows and tile QC are checked by _10_build_tile_asset.py.",
    })
    write_json_atomic(cohort_dir / "slide_prefetch_manifest.json", manifest, overwrite=args.force)
    _write_report(reports_dir / "slide_prefetch_report.md", summary, overwrite=args.force)

    print(json.dumps({"status": "ok", **summary, "prefetch_manifest": str(cohort_dir / "slide_prefetch_manifest.parquet")}, indent=2))
    return 0


if __name__ == "__main__":

    ap = argparse.ArgumentParser(description="Prefetch TCGA slide technical metadata before full WSI download.")

    ap.add_argument("--benchmark_id", type=str,
                    required=True,
                    help="Benchmark ID whose provisional slides are prefetched.")

    ap.add_argument("--gdc_api_base", type=str,
                    default=None,
                    help="Base URL for the GDC API. Defaults to the project default.")

    ap.add_argument("--max_workers", type=int,
                    default=DEFAULT_MAX_WORKERS,
                    help="Number of slides prefetched concurrently.")

    ap.add_argument("--header_probe_bytes", type=int,
                    default=DEFAULT_HEADER_PROBE_BYTES,
                    help="Bytes read at offset 0 (TIFF/BigTIFF header, IFD0 offset); a non-TIFF verdict "
                         "needs >= 4096.")

    ap.add_argument("--ifd_window_bytes", type=int,
                    default=DEFAULT_IFD_WINDOW_BYTES,
                    help="Bytes read at the IFD0 offset (directory entries and ImageDescription).")

    ap.add_argument("--max_metadata_bytes", type=int,
                    default=DEFAULT_MAX_METADATA_BYTES,
                    help="Hard ceiling on total bytes read per slide across all targeted reads.")

    ap.add_argument("--prefetch_bytes_initial", type=int,
                    default=DEFAULT_PREFETCH_BYTES_INITIAL,
                    help="Unused (see --header_probe_bytes, --ifd_window_bytes).")

    ap.add_argument("--prefetch_bytes_max", type=int,
                    default=DEFAULT_PREFETCH_BYTES_MAX,
                    help="Unused (see --max_metadata_bytes).")

    ap.add_argument("--timeout", type=int,
                    default=120,
                    help="HTTP request timeout in seconds.")

    ap.add_argument("--retries", type=int,
                    default=3,
                    help="Number of retries per HTTP request.")

    ap.add_argument("--backoff", type=float,
                    default=1.5,
                    help="Exponential backoff factor between retries.")

    ap.add_argument("--skip_preflight_check", action="store_true",
                    help="Skip the one-shot GDC reachability check performed before prefetching.")

    ap.add_argument("--abort_after_consecutive_errors", type=int,
                    default=25,
                    help="Abort if this many consecutive slides end in 'prefetch_error_final' (0: disabled).")

    ap.add_argument("--mpp_min", type=float,
                    default=None,
                    help="Minimum accepted MPP (debug override of the model group's mpp_min).")

    ap.add_argument("--mpp_max", type=float,
                    default=None,
                    help="Maximum accepted MPP (debug override of the model group's mpp_max).")

    ap.add_argument("--mpp_anisotropy_max", type=float,
                    default=None,
                    help="Maximum accepted x/y MPP anisotropy ratio (debug override of the model group's "
                         "mpp_anisotropy_max).")

    ap.add_argument("--allow_200_read_cap", action="store_true",
                    help="Unused (a 200 answer to a Range request is treated as range not supported).")

    ap.add_argument("--try_openslide_partial", action="store_true",
                    help="Unused (OpenSlide is not run on the prefetched bytes).")

    ap.add_argument("--refresh_prefetch_cache", action="store_true",
                    help="Re-fetch the metadata even if it is cached in the shared store.")

    ap.add_argument("--limit", type=int,
                    default=None,
                    help="Debug only: prefetch only the first N manifest rows.")

    ap.add_argument("--force", action="store_true",
                    help="Overwrite an existing prefetch manifest.")

    ap.add_argument("--force_cascade", action="store_true",
                    help="Required with --force if downstream full slide downloads or tile assets already exist.")

    args = ap.parse_args()
    raise SystemExit(main(args))
