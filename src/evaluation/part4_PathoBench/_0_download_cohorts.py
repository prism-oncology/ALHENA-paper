#!/usr/bin/env python
"""
Download the CPTAC WSIs of the Task D cohorts from the NCI Imaging Data Commons (IDC) and cache the PathoBench splits.

Saved in <ROOT_HistoST> / eval / eval_datasets / pathobench_eval_data /
    - slide_store/<cohort>/manifests/download_manifest.tsv: slides of the cohort
    - slide_store/<cohort>/wsi/<slide_id>/<slide_id>.dcm: WSIs (IDC DICOM series)
    - pathobench_assets/: PathoBench splits and labels
"""

import argparse
import csv
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Add the project root to sys.path so that 'src' can be imported
project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.evaluation.part4_PathoBench._pathobench_common import (
    cohorts_in_registry,
    idc_collection_id,
    install_appledouble_tolerant_fs_ops,
    pathobench_assets_root,
    cohort_manifest_dir,
    cohort_wsi_dir,
    slide_wsi_path,
    split_task_id,
    tasks_for_cohort,
)

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- manifests


def cohort_manifest_path(cohort: str) -> Path:
    return cohort_manifest_dir(cohort) / "download_manifest.tsv"


def write_cohort_manifest(cohort: str, rows: List[Dict[str, str]], *, dry_run: bool) -> None:
    """Per-cohort slide download manifest consumed by ``_1_extract_features.py``."""
    path = cohort_manifest_path(cohort)
    logger.info("Cohort %s: %d unique slides -> %s", cohort, len(rows), path)
    if dry_run:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["slide_id", "source", "uri", "md5", "size_mb", "local_path", "label_hint"]
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, delimiter="\t")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


# --------------------------------------------------------------------------- external: splits


def load_pathobench_split_slides(cohort: str) -> List[Dict[str, str]]:
    """
    Unique slides of the PathoBench splits of a cohort, as rows {slide_id, source, uri, md5, size_mb, label_hint}
    (uri and size_mb are filled later for CPTAC).
    """
    try:
        from patho_bench.SplitFactory import SplitFactory  # type: ignore  # lazy: pathobench env
    except Exception as exc:  # pragma: no cover - env-dependent
        raise SystemExit(
            "PathoBench is not importable in this env. Activate the `pathobench` env and install "
            f"patho-bench (+ trident + ipython; see docs/installation.md). Original error: {exc}"
        )
    # PathoBench cache cleanup can fail on macOS ._* files
    install_appledouble_tolerant_fs_ops()

    seen: Dict[str, Dict[str, str]] = {}
    for task in tasks_for_cohort(cohort):
        source, task_name = split_task_id(task.task_id)
        split_path, _config_path = SplitFactory.from_hf(str(pathobench_assets_root()), source, task_name)
        for slide_id in _read_split_slide_ids(Path(split_path)):
            if slide_id not in seen:
                seen[slide_id] = {
                    "slide_id": slide_id,
                    "source": _source_for_cohort(cohort),
                    "uri": "",
                    "md5": "",
                    "size_mb": "",
                    "label_hint": "",
                }
    if not seen:
        raise SystemExit(
            f"No slides resolved for cohort {cohort!r} from PathoBench splits — check the task ids "
            f"({[t.task_id for t in tasks_for_cohort(cohort)]}) against the installed PathoBench."
        )
    return list(seen.values())


def _read_split_slide_ids(split_path: Path) -> List[str]:
    """Unique slide_id values of a PathoBench split TSV, in order."""
    ids: List[str] = []
    with split_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        if "slide_id" not in (reader.fieldnames or []):
            raise SystemExit(f"Split {split_path} has no 'slide_id' column (columns={reader.fieldnames}).")
        for row in reader:
            sid = str(row.get("slide_id", "")).strip()
            if sid and sid.lower() != "nan":
                ids.append(sid)
    return list(dict.fromkeys(ids))  # unique, order-preserving


def _source_for_cohort(cohort: str) -> str:
    if cohort.startswith("cptac"):
        return "idc"          # CPTAC pathology WSIs are on the NCI Imaging Data Commons (IDC)
    return cohort             # bcnb / nadt / mbc_ : cohort-specific portal (manual)


def _is_idc_cohort(cohort: str) -> bool:
    return _source_for_cohort(cohort) == "idc"


# --------------------------------------------------------------------------- external: IDC resolution


def _idc_client():
    """idc-index client with the slide-microscopy (SM) index fetched (lazy import)."""
    try:
        from idc_index import IDCClient  # type: ignore  # lazy: pathobench env
    except Exception as exc:  # pragma: no cover - env-dependent
        raise SystemExit(
            "idc-index is not importable in this env. Activate the `pathobench` env and "
            f"`pip install idc-index` (see docs/installation.md). Original error: {exc}"
        )
    client = IDCClient()
    client.fetch_index("sm_index")  # ContainerIdentifier <-> SeriesInstanceUID map (cached after first fetch)
    return client


def resolve_cptac_slides_idc(rows: List[Dict[str, str]], cohort: str, *, client: Optional[object] = None) -> None:
    """
    Fill row['uri'] (IDC SeriesInstanceUID) and row['size_mb'] of CPTAC slides, by matching each slide_id with the
    ContainerIdentifier (or <PatientID>-<ContainerIdentifier>) of the SM series of the cohort's IDC collection.
    Slides with 0 or > 1 matching series are left unresolved (manual placement).
    """
    client = client or _idc_client()
    collection = idc_collection_id(cohort)
    want = {r["slide_id"]: r for r in rows}

    main = client.index                                                     # type: ignore[attr-defined]
    sm = client.sm_index                                                    # type: ignore[attr-defined]
    main_c = main[(main["collection_id"] == collection) & (main["Modality"] == "SM")][
        ["SeriesInstanceUID", "PatientID", "series_size_MB"]
    ].copy()
    main_c["SeriesInstanceUID"] = main_c["SeriesInstanceUID"].astype(str)
    sm_slim = sm[["SeriesInstanceUID", "ContainerIdentifier"]].copy()
    sm_slim["SeriesInstanceUID"] = sm_slim["SeriesInstanceUID"].astype(str)
    joined = main_c.merge(sm_slim, on="SeriesInstanceUID", how="inner")

    by_key: Dict[str, set] = {}          # ContainerIdentifier | {PatientID}-{ContainerIdentifier} -> {series}
    size_by_series: Dict[str, float] = {}
    for suid, pid, size, cid in zip(joined["SeriesInstanceUID"], joined["PatientID"].astype(str),
                                    joined["series_size_MB"], joined["ContainerIdentifier"].astype(str)):
        size_by_series[suid] = float(size or 0.0)
        for key in (cid, f"{pid}-{cid}"):
            by_key.setdefault(key, set()).add(suid)

    n_ok = 0
    for sid, r in want.items():
        series = sorted(by_key.get(sid, set()))
        if len(series) == 1:
            r["uri"] = series[0]
            r["size_mb"] = f"{size_by_series.get(series[0], 0.0):.1f}"
            n_ok += 1
        else:
            r["uri"], r["size_mb"] = "", ""  # 0 matches or ambiguous -> manual
            logger.warning("CPTAC slide %s: %d IDC SM series match(es) in %s (need exactly 1) -> manual placement.",
                           sid, len(series), collection)
    logger.info("IDC resolution (%s -> %s): %d/%d CPTAC slides resolved to a unique SM series.",
                cohort, collection, n_ok, len(want))


# --------------------------------------------------------------------------- external: download


def _finalize_idc_series(slide_dir: Path, slide_id: str) -> Optional[Path]:
    """
    Rename one downloaded .dcm instance of the series to <slide_id>.dcm (so Trident writes <slide_id>.h5).
    Returns the entry path, or None if there is no .dcm.
    """
    entry = slide_dir / f"{slide_id}.dcm"
    if entry.exists():
        return entry
    dcms = sorted(p for p in slide_dir.glob("*.dcm") if not p.name.startswith("."))
    if not dcms:
        return None
    dcms[0].rename(entry)
    return entry


def download_wsi(row: Dict[str, str], cohort: str, *, dry_run: bool, client: Optional[object] = None) -> Optional[Path]:
    """
    Download one CPTAC WSI (IDC DICOM series) into slide_store/<cohort>/wsi/<slide_id>/.
    Returns its entry path (existing entries are kept), or None if the slide must be placed manually.
    """
    slide_id = row["slide_id"]
    slide_dir = slide_wsi_path(cohort, slide_id).parent  # slide_store/{cohort}/wsi/{slide_id}/
    entry = slide_dir / f"{slide_id}.dcm"
    if entry.exists():
        return entry  # resume: already downloaded / placed
    resolvable_idc = row.get("source") == "idc" and bool(row.get("uri"))
    if dry_run:
        return entry if resolvable_idc else None
    if resolvable_idc:
        slide_dir.mkdir(parents=True, exist_ok=True)
        series_uid = row["uri"]
        logger.info("IDC download %s (series %s) -> %s", slide_id, series_uid, slide_dir)
        client.download_from_selection(  # type: ignore[union-attr]
            seriesInstanceUID=series_uid, downloadDir=str(slide_dir), dirTemplate=None, quiet=True,
        )
        finalized = _finalize_idc_series(slide_dir, slide_id)
        if finalized is None:
            logger.warning("IDC download for %s produced no .dcm in %s -> manual placement.", slide_id, slide_dir)
        return finalized
    return None  # an unresolved CPTAC slide -> manual placement


# --------------------------------------------------------------------------- main


def main(args: argparse.Namespace) -> None:
    cohorts = args.cohorts or cohorts_in_registry()
    unknown = [c for c in cohorts if c not in cohorts_in_registry()]
    if unknown:
        raise SystemExit(f"Unknown cohort(s) {unknown}; registry cohorts are {cohorts_in_registry()}.")

    idc_client = None  # one shared idc-index client (SM index fetched once) across all IDC cohorts
    for cohort in cohorts:
        rows = load_pathobench_split_slides(cohort)
        # WSIs are downloaded under the default root (not $SCRATCH)
        wsi_dir = cohort_wsi_dir(cohort)

        if _is_idc_cohort(cohort):
            if idc_client is None:
                idc_client = _idc_client()
            resolve_cptac_slides_idc(rows, cohort, client=idc_client)  # fills uri + size_mb where resolvable

        unresolved: List[str] = []
        for r in rows:
            path = download_wsi(r, cohort, dry_run=args.dry_run, client=idc_client)
            if path is None:
                unresolved.append(r["slide_id"])
            else:
                r["local_path"] = str(path)
        write_cohort_manifest(cohort, rows, dry_run=args.dry_run)

        resolved_gb = sum(float(r.get("size_mb") or 0.0) for r in rows if r.get("uri")) / 1024.0
        logger.info("cohort %s: %d resolved (~%.1f GB), %d need manual placement.",
                    cohort, len(rows) - len(unresolved), resolved_gb, len(unresolved))
        if unresolved:
            shown = ", ".join(unresolved[:10]) + (" …" if len(unresolved) > 10 else "")
            logger.warning(
                "cohort %s: %d/%d slide(s) need manual placement — drop each as "
                "%s/{slide_id}/{slide_id}.dcm and re-run (resumes). Unresolved: %s",
                cohort, len(unresolved), len(rows), wsi_dir, shown,
            )

    logger.info("Done (dataset built). Next: _1_extract_features.py --encoders <tags> [--cohorts %s]", " ".join(cohorts))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Task D step 0: build the dataset (download benchmark WSIs + cache splits).")
    parser.add_argument("--cohorts", type=str, nargs="*", default=None,
                        help="Cohort subset to download (default: all registry cohorts).")
    parser.add_argument("--dry_run", action="store_true", help="Resolve + write manifests but download nothing.")
    main(parser.parse_args())
