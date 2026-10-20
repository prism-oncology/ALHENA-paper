#!/usr/bin/env python
"""
Download the PanNuke nuclei dataset (HuggingFace RationAI/PanNuke, parquet) and write its provenance manifest.
Runs in the `histost_eval_a2` env.

Saved in <ROOT_HistoST> / eval / eval_datasets / pannuke_eval_data /
    - raw/: downloaded files (data/*.parquet, gen_script.py, README.md)
    - manifest.json: repo id, resolved revision, sha256 and size of each file, licence, citations
    - download_log.txt
"""

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Project imports
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src_A2.common import configure_logger, ensure_dir, sha256_file, write_json
from src_A2.common import src_import
from pannuke.pannuke_common import (
    PANNUKE_CITATIONS,
    PANNUKE_LICENSE,
    PANNUKE_NOMINAL_MAGNIFICATION,
    PANNUKE_NOMINAL_MPP_UM_PER_PX,
    PANNUKE_RESOLUTION_CAVEAT,
    PUBLISHED_CLASS_COUNTS,
    PUBLISHED_FOLD_SIZES,
    PUBLISHED_N_NUCLEI,
    PUBLISHED_N_TILES,
)

get_full_path = src_import("get_full_path")

logger = logging.getLogger(__name__)


HF_REPO_ID = "RationAI/PanNuke"
HF_ALLOW_PATTERNS = ["data/*.parquet", "gen_script.py", "README.md"]

WARWICK_BASE_URL = "https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke"

MANIFEST_SCHEMA_VERSION = 1


def pannuke_root() -> Path:
    """Root of the PanNuke acquisition under ROOT_PATH."""
    return get_full_path("eval", "eval_datasets", "pannuke_eval_data")


# ---------------------------------------------------------------------------------------------
# Probe (read-only, no bytes fetched)
# ---------------------------------------------------------------------------------------------

def probe_hf(revision: Optional[str]) -> Dict[str, Any]:
    """Metadata of the HF dataset repo (resolved revision, files, byte sizes), without downloading the files."""
    from huggingface_hub import HfApi

    api = HfApi()
    info = api.repo_info(
        repo_id=HF_REPO_ID, repo_type="dataset", revision=revision, files_metadata=True
    )
    files: List[Dict[str, Any]] = []
    for sibling in info.siblings or []:
        files.append(
            {
                "path": sibling.rfilename,
                "size_bytes": getattr(sibling, "size", None),
                "blob_sha": getattr(sibling, "blob_id", None),
                "lfs": bool(getattr(sibling, "lfs", None)),
            }
        )
    files.sort(key=lambda d: d["path"])
    known = [f for f in files if f["size_bytes"] is not None]
    return {
        "source": "hf",
        "repo_id": HF_REPO_ID,
        "requested_revision": revision,
        "resolved_revision": info.sha,
        "last_modified": str(getattr(info, "last_modified", "")),
        "n_files": len(files),
        "total_size_bytes_known": int(sum(f["size_bytes"] for f in known)),
        "n_files_with_unknown_size": len(files) - len(known),
        "files": files,
    }


def probe_warwick() -> Dict[str, Any]:
    """Metadata of the Warwick release: size of each fold zip (HTTP HEAD, nothing downloaded)."""
    import urllib.request

    files: List[Dict[str, Any]] = []
    for name in ("fold_1.zip", "fold_2.zip", "fold_3.zip"):
        url = f"{WARWICK_BASE_URL}/{name}"
        entry: Dict[str, Any] = {"path": name, "url": url, "size_bytes": None, "reachable": False}
        try:
            req = urllib.request.Request(url, method="HEAD")
            with urllib.request.urlopen(req, timeout=60) as resp:
                entry["reachable"] = True
                length = resp.headers.get("Content-Length")
                entry["size_bytes"] = int(length) if length is not None else None
        except Exception as exc:  # noqa: BLE001 - a probe reports reachability, it does not fail
            entry["error"] = f"{type(exc).__name__}: {exc}"
        files.append(entry)
    known = [f for f in files if f["size_bytes"] is not None]
    return {
        "source": "warwick",
        "base_url": WARWICK_BASE_URL,
        "n_files": len(files),
        "total_size_bytes_known": int(sum(f["size_bytes"] for f in known)),
        "files": files,
    }


# ---------------------------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------------------------

def _tool_versions() -> Dict[str, str]:
    versions = {"python": sys.version.split()[0]}
    for mod in ("huggingface_hub", "pyarrow", "pandas", "numpy"):
        try:
            versions[mod] = __import__(mod).__version__
        except Exception:  # noqa: BLE001 - a missing optional tool is recorded, not fatal here
            versions[mod] = "not installed"
    return versions


def build_manifest(probe: Dict[str, Any], raw_dir: Path, file_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Provenance record written to manifest.json."""
    return {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "dataset": "PanNuke",
        "source": probe.get("source"),
        "repo_id": probe.get("repo_id"),
        "base_url": probe.get("base_url"),
        "requested_revision": probe.get("requested_revision"),
        "resolved_revision": probe.get("resolved_revision"),
        "fetched_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "raw_dir": str(raw_dir),
        "files": file_records,
        "n_files": len(file_records),
        "total_size_bytes": int(sum(int(f.get("size_bytes") or 0) for f in file_records)),
        "tool_versions": _tool_versions(),
        "license": PANNUKE_LICENSE,
        "license_note": "Non-commercial research use. Both citations below are required.",
        "citations": list(PANNUKE_CITATIONS),
        "resolution": {
            "nominal_magnification": PANNUKE_NOMINAL_MAGNIFICATION,
            "nominal_mpp_um_per_px": PANNUKE_NOMINAL_MPP_UM_PER_PX,
            "caveat": PANNUKE_RESOLUTION_CAVEAT,
            "per_tile_resolution_metadata_available": False,
        },
        "published_figures_for_acceptance_gate": {
            "n_tiles": PUBLISHED_N_TILES,
            "fold_sizes": dict(PUBLISHED_FOLD_SIZES),
            "n_nuclei": PUBLISHED_N_NUCLEI,
            "class_counts": dict(PUBLISHED_CLASS_COUNTS),
            "note": (
                "Checked by _2_characterize_pannuke.py against the downloaded arrays (full runs); "
                "a mismatch raises an error."
            ),
        },
    }


def is_payload_file(path: Path, raw_dir: Path) -> bool:
    """False for dot-files and files inside dot-directories (e.g. .cache/huggingface), True otherwise."""
    return not any(part.startswith(".") for part in path.relative_to(raw_dir).parts)


def _record_files(raw_dir: Path) -> List[Dict[str, Any]]:
    """sha256 + size for every downloaded payload file, sorted, relative to raw_dir."""
    records: List[Dict[str, Any]] = []
    for path in sorted(p for p in raw_dir.rglob("*") if p.is_file()):
        if not is_payload_file(path, raw_dir):
            continue
        records.append(
            {
                "path": str(path.relative_to(raw_dir)),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return records


# ---------------------------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------------------------

def download_hf(raw_dir: Path, revision: str, force: bool) -> None:
    """Download the parquet folds with snapshot_download, then check that parquet files are present."""
    from huggingface_hub import snapshot_download

    ensure_dir(raw_dir)
    logger.info("Downloading %s @ %s -> %s", HF_REPO_ID, revision, raw_dir)
    snapshot_download(
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        revision=revision,
        allow_patterns=HF_ALLOW_PATTERNS,
        local_dir=str(raw_dir),
        force_download=bool(force),
    )

    parquets = sorted(raw_dir.rglob("*.parquet"))
    if not parquets:
        raise FileNotFoundError(
            f"No parquet files under {raw_dir} after snapshot_download of {HF_REPO_ID}. "
            f"Expected patterns {HF_ALLOW_PATTERNS}."
        )
    logger.info("Downloaded %d parquet file(s): %s", len(parquets), [p.name for p in parquets])


def verify_against_existing_manifest(raw_dir: Path, manifest_path: Path) -> None:
    """Raise if a file already present does not match the sha256 recorded in the existing manifest."""
    if not manifest_path.is_file():
        return
    import json

    with manifest_path.open("r") as f:
        previous = json.load(f)
    mismatches = []
    for record in previous.get("files", []):
        path = raw_dir / record["path"]
        if not path.is_file():
            continue
        if not is_payload_file(path, raw_dir):
            # Skip non-payload entries (e.g. .cache/huggingface files, rewritten by huggingface_hub)
            continue
        actual = sha256_file(path)
        if actual != record.get("sha256"):
            mismatches.append({"path": record["path"], "expected": record.get("sha256"), "actual": actual})
    if mismatches:
        raise RuntimeError(
            f"Existing PanNuke files do not match the sha256 recorded in the manifest: {mismatches}."
        )


def main(args: argparse.Namespace) -> int:
    out_root = Path(args.out_dir).expanduser().resolve() if args.out_dir else pannuke_root()
    raw_dir = out_root / "raw"
    manifest_path = out_root / "manifest.json"

    if args.probe_only:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
        probe = probe_hf(args.revision) if args.source == "hf" else probe_warwick()
        logger.info("PROBE (%s): no bytes fetched, nothing written.", args.source)
        logger.info("  resolved_revision : %s", probe.get("resolved_revision"))
        logger.info("  n_files           : %s", probe.get("n_files"))
        logger.info(
            "  total_size        : %s bytes (%.1f MiB)",
            probe.get("total_size_bytes_known"),
            (probe.get("total_size_bytes_known") or 0) / (1024 ** 2),
        )
        for f in probe.get("files", []):
            logger.info("    %-48s %s", f["path"], f.get("size_bytes"))
        if args.probe_json:
            write_json(Path(args.probe_json).expanduser().resolve(), probe)
            logger.info("Probe payload written to %s", args.probe_json)
        return 0

    if args.dry_run:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
        probe = probe_hf(args.revision) if args.source == "hf" else probe_warwick()
        logger.info("DRY RUN: nothing will be written under %s", out_root)
        logger.info("  would download   : %s files, %s bytes", probe.get("n_files"), probe.get("total_size_bytes_known"))
        logger.info("  would write raw  : %s", raw_dir)
        logger.info("  would write mfst : %s", manifest_path)
        logger.info("  resolved revision: %s", probe.get("resolved_revision"))
        return 0

    if args.source != "hf":
        raise NotImplementedError(
            "Only --source hf is implemented."
        )

    ensure_dir(out_root)
    log = configure_logger(out_root / "download_log.txt", name="A2_pannuke_download")
    log.info("PanNuke acquisition starting. out_root=%s", out_root)

    verify_against_existing_manifest(raw_dir, manifest_path)

    probe = probe_hf(args.revision)
    revision = probe["resolved_revision"]
    log.info("Resolved revision %s (requested %s)", revision, args.revision)

    # Provisional manifest (no files yet), rewritten after the download
    write_json(manifest_path, build_manifest(probe, raw_dir, []))

    download_hf(raw_dir, revision=revision, force=args.force)

    file_records = _record_files(raw_dir)
    write_json(manifest_path, build_manifest(probe, raw_dir, file_records))
    log.info(
        "Done. %d files, %d bytes. Manifest -> %s",
        len(file_records),
        sum(int(f["size_bytes"]) for f in file_records),
        manifest_path,
    )
    log.info("License: %s -- non-commercial research use, both citations required.", PANNUKE_LICENSE)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Download the PanNuke nuclei dataset and write its provenance manifest."
        )
    )

    parser.add_argument("--source", type=str, default="hf", choices=["hf", "warwick"],
                        help="Source: hf (RationAI/PanNuke) or warwick (probe only, download not implemented).")

    parser.add_argument("--revision", type=str, default=None,
                        help="HuggingFace revision (branch, tag or commit sha; default: head of main). "
                             "The resolved sha is recorded in manifest.json.")

    parser.add_argument("--out_dir", type=str, default=None,
                        help="Output root (default: <ROOT_PATH>/eval/eval_datasets/pannuke_eval_data).")

    parser.add_argument("--probe_only", action="store_true",
                        help="List the files and sizes at the resolved revision, without downloading or writing.")

    parser.add_argument("--probe_json", type=str, default=None,
                        help="With --probe_only, also write the probe result to this JSON path.")

    parser.add_argument("--dry_run", action="store_true",
                        help="Resolve the revision and report what would be written, without writing anything.")

    parser.add_argument("--force", action="store_true",
                        help="Re-download files that are already present.")

    args = parser.parse_args()
    sys.exit(main(args))
