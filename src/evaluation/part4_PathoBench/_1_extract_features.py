#!/usr/bin/env python
"""
Trident tiling and he patch-feature extraction of the Task D slides, then clip features derived from he
(clip = L2(image_proj(he))).

Saved in <ROOT_HistoST> / eval / eval_datasets / pathobench_eval_data /
    - trident_features/<cohort>/_tiling/: segmentation masks and patch coords (shared by all encoders)
    - trident_features/<cohort>/HEencoder_t<n>e<c>/he/<slide_id>.h5: he patch features (features, coords)
    - trident_features/<cohort>/HEencoder_t<n>e<c>/clip/<slide_id>.h5: clip patch features (not for t-1e-1)
    - slide_store/<cohort>/manifests/slide_inventory.tsv: kept / rejected slides, with reason and native MPP
    - slide_store/<cohort>/manifests/slide_inventory_summary.json: slide counts
"""

import argparse
import csv
import json
import logging
import os
import shutil
import socket
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Add the project root to sys.path so that 'src' can be imported
project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.evaluation.part4_PathoBench._pathobench_common import (
    MPP_MAX_UM_PER_PX,
    MPP_MIN_UM_PER_PX,
    PATCH_SIZE_PX,
    TARGET_MAG,
    bundle_path,
    cohort_manifest_dir,
    cohort_tiling_dir,
    cohort_wsi_dir,
    cohorts_in_registry,
    encoder_features_run_id,
    encoder_tag,
    expects_clip,
    list_h5,
    parse_encoder_tag,
    slide_inventory_path,
    slide_wsi_path,
    trident_features_root,
)

logger = logging.getLogger(__name__)

# Trident coords dir (relative to the per-cohort tiling dir): 20x, 224 px (0.5 µm/px, 112 µm)
COORDS_REL = f"{TARGET_MAG}x_{PATCH_SIZE_PX}px_0px_overlap"

# Number of coarsest pyramid levels decoded by the readability check (probe_slide)
PREFLIGHT_LEVELS = 2


def coords_h5_path(tiling_dir: Path, slide_id: str) -> Path:
    """Per-slide Trident coords file (Trident layout: <COORDS_REL>/patches/<slide_id>_patches.h5)."""
    return tiling_dir / COORDS_REL / "patches" / f"{slide_id}_patches.h5"


# --------------------------------------------------------------------------- manifest / slides


def read_cohort_manifest(cohort: str) -> List[Dict[str, str]]:
    path = cohort_manifest_dir(cohort) / "download_manifest.tsv"
    if not path.exists():
        raise SystemExit(f"Missing cohort manifest {path} — run _0_download_cohorts.py --cohorts {cohort} first.")
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def _probe_slide_once(slide_path: Path) -> Tuple[Optional[float], bool, str]:
    """
    Open a WSI once: read its MPP, then decode its coarsest levels. Returns (mpp, pixels_ok, failure_detail).
    The MPP and level dimensions are read first (OpenSlide re-raises a previous error on every later call).
    """
    import openslide  # trident env
    mpp: Optional[float] = None
    try:
        with openslide.OpenSlide(str(slide_path)) as s:
            mpp_x = s.properties.get(openslide.PROPERTY_NAME_MPP_X)
            mpp = float(mpp_x) if mpp_x is not None else None
            dims = s.level_dimensions  # geometry first: a failed decode would poison this handle
            for lvl in range(len(dims) - 1, max(len(dims) - 1 - PREFLIGHT_LEVELS, -1), -1):
                s.read_region((0, 0), lvl, dims[lvl])
        return mpp, True, ""
    except Exception as exc:  # pragma: no cover - env/file dependent
        return mpp, False, f"{type(exc).__name__}: {exc}"


def probe_slide(slide_path: Path) -> Tuple[Optional[float], bool, str]:
    """
    Return (native_mpp, pixels_ok, failure_detail) of a WSI. pixels_ok: the PREFLIGHT_LEVELS coarsest levels
    decode (IDC CPTAC series with JP2-boxed tiles do not). Retried once on failure.
    """
    mpp, ok, detail = _probe_slide_once(slide_path)
    if ok:
        return mpp, True, ""
    mpp_retry, ok_retry, detail_retry = _probe_slide_once(slide_path)
    if ok_retry:
        logger.warning("Slide %s failed the readability preflight once (%s) but passed on retry — "
                       "treating the first failure as transient I/O and keeping the slide.",
                       slide_path, detail)
        return mpp_retry, True, ""
    return mpp_retry, False, detail_retry


def _resolve_slide_path(cohort: str, row: Dict[str, str], *, wsi_on_scratch: bool = False) -> Path:
    """WSI entry of a manifest row: slide_store/<cohort>/wsi/<slide_id>/<slide_id>.dcm if it exists, else the recorded local_path."""
    canonical = slide_wsi_path(cohort, row.get("slide_id", ""), scratch=wsi_on_scratch)
    if canonical.exists():
        return canonical
    recorded = row.get("local_path") or ""
    return Path(recorded) if recorded else canonical


def filter_no_upsampling(
    cohort: str, rows: Sequence[Dict[str, str]], *, wsi_on_scratch: bool = False
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """
    Keep the slides with MPP in [MPP_MIN_UM_PER_PX, MPP_MAX_UM_PER_PX] whose pixels decode; reject the others
    (wsi_not_found, unreadable_mpp, coarser_than_target_would_upsample, unreadable_pixels).
    Returns (kept, rejected) rows with the slide_inventory.tsv fields.
    """
    kept, rejected = [], []
    for r in rows:
        p = _resolve_slide_path(cohort, r, wsi_on_scratch=wsi_on_scratch)
        found = p.exists()
        mpp, pixels_ok, detail = probe_slide(p) if found else (None, False, "")
        mpp_ok = mpp is not None and MPP_MIN_UM_PER_PX <= mpp <= MPP_MAX_UM_PER_PX
        if mpp_ok and pixels_ok:
            direction = "none" if mpp >= 0.5 * (1 - 1e-9) else "downsample"
            kept.append({**r, "local_path": str(p), "native_mpp": f"{mpp:.4f}",
                         "resample_direction": direction, "slide_status": "kept",
                         "failure_reason": "", "failure_detail": ""})
        else:
            if not found:
                reason = "wsi_not_found"
            elif mpp is None:
                reason = "unreadable_mpp"
            elif not mpp_ok:
                # The MPP reason takes precedence over pixel decoding
                reason = "coarser_than_target_would_upsample"
            else:
                reason = "unreadable_pixels"
                logger.warning("cohort %s slide %s: headers parse (mpp=%.4f) but the pixels do not "
                               "decode — rejected as unreadable_pixels. %s",
                               cohort, r.get("slide_id", ""), mpp, detail)
            rejected.append({**r, "local_path": (str(p) if found else (r.get("local_path") or "")),
                             "native_mpp": ("" if mpp is None else f"{mpp:.4f}"),
                             "resample_direction": "", "slide_status": "rejected",
                             "failure_reason": reason,
                             "failure_detail": (detail if reason in ("unreadable_mpp", "unreadable_pixels") else "")})
    n_missing = sum(1 for r in rejected if r["failure_reason"] == "wsi_not_found")
    if n_missing:
        logger.warning(
            "cohort %s: %d/%d WSIs are not on disk under %s (unresolved slides awaiting manual "
            "placement, or WSIs not staged to this root; see --wsi_on_scratch). Reading WSIs from: %s.",
            cohort, n_missing, len(rows), cohort_wsi_dir(cohort, scratch=wsi_on_scratch),
            "$SCRATCH root (--wsi_on_scratch)" if wsi_on_scratch else "the default root",
        )
    if rejected:
        logger.warning("cohort %s slide gate: rejected %d/%d slides (%d not found, %d unreadable MPP, "
                       "%d coarser than %.3f µm/px, %d undecodable pixels).", cohort, len(rejected),
                       len(rows), n_missing,
                       sum(1 for r in rejected if r["failure_reason"] == "unreadable_mpp"),
                       sum(1 for r in rejected if r["failure_reason"] == "coarser_than_target_would_upsample"),
                       MPP_MAX_UM_PER_PX,
                       sum(1 for r in rejected if r["failure_reason"] == "unreadable_pixels"))
    return kept, rejected


# --------------------------------------------------------------------------- inventory (provenance)


def write_slide_inventory(cohort: str, kept: Sequence[Dict[str, str]], rejected: Sequence[Dict[str, str]]) -> None:
    """
    Write slide_inventory.tsv (slide status, failure reason and detail, native MPP, resample direction) and
    slide_inventory_summary.json of a cohort.
    """
    man_dir = cohort_manifest_dir(cohort)  # default root, never $SCRATCH
    man_dir.mkdir(parents=True, exist_ok=True)
    fields = ["slide_id", "cohort", "slide_status", "failure_reason", "failure_detail",
              "native_mpp", "resample_direction", "local_path"]
    inv_path = slide_inventory_path(cohort)  # read by _2_run_pathobench.py
    with inv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, delimiter="\t")
        w.writeheader()
        for r in [*kept, *rejected]:
            row = {k: r.get(k, "") for k in fields}
            row["cohort"] = cohort
            w.writerow(row)

    n_downsample = sum(1 for r in kept if r.get("resample_direction") == "downsample")
    n_native = sum(1 for r in kept if r.get("resample_direction") == "none")
    by_reason: Dict[str, int] = {}
    for r in rejected:
        reason = str(r.get("failure_reason") or "unknown")
        by_reason[reason] = by_reason.get(reason, 0) + 1
    summary = {
        "cohort": cohort,
        "n_total": len(kept) + len(rejected),
        "n_kept": len(kept),
        "n_rejected": len(rejected),
        # Number of rejected slides per reason
        "n_rejected_by_reason": by_reason,
        "n_slides_native_resolution": n_native,
        "n_slides_downsampled": n_downsample,
        "n_slides_upsampled": 0,  # coarser slides are rejected
        "mpp_min_um_per_px": MPP_MIN_UM_PER_PX,
        "mpp_max_um_per_px": MPP_MAX_UM_PER_PX,
        "preflight_levels_decoded": PREFLIGHT_LEVELS,
    }
    (man_dir / "slide_inventory_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    logger.info("cohort=%s inventory: %d kept (%d native, %d downsample), %d rejected%s -> %s",
                cohort, len(kept), n_native, n_downsample, len(rejected),
                (" " + json.dumps(by_reason, sort_keys=True)) if by_reason else "", inv_path)


# --------------------------------------------------------------------------- Trident (external)


def _lock_is_live(lock_fp: Path) -> bool:
    """True if a Trident .lock is held by a running process on this host (locks from other hosts are not live)."""
    try:
        data = json.loads(lock_fp.read_text() or "{}")
        host, pid = data.get("hostname"), int(data["pid"])
    except Exception:
        return False  # empty / unreadable / pid-less lock
    if host != socket.gethostname():
        return False
    try:
        os.kill(pid, 0)
    except PermissionError:
        return True   # the PID exists, it is simply not signallable from here
    except Exception:
        return False  # ProcessLookupError => the owner is gone
    return True


def clear_stale_locks(tiling_dir: Path) -> None:
    """
    Delete the Trident .lock files left by an interrupted run and the (possibly partial) outputs they guard,
    so that these slides are recomputed. Locks held by a live process on this host are kept.
    Only one _1_extract_features.py run per cohort at a time.
    """
    # ._* files are macOS AppleDouble files, not locks
    locks = [p for p in sorted(tiling_dir.rglob("*.lock")) if not p.name.startswith(".")]
    n_locks = n_artifacts = 0
    for lock_fp in locks:
        if _lock_is_live(lock_fp):
            logger.warning("lock %s is held by a live PID on this host — another `_1` is running on this "
                           "cohort. Leaving it and its slide untouched.", lock_fp)
            continue
        guarded = Path(str(lock_fp)[: -len(".lock")])
        if guarded.exists():
            guarded.unlink(missing_ok=True)  # missing_ok: an AppleDouble sidecar can vanish mid-walk
            n_artifacts += 1
        lock_fp.unlink(missing_ok=True)
        n_locks += 1
    if n_locks or n_artifacts:
        logger.warning(
            "tiling_dir=%s: reclaimed %d stale lock(s) and deleted %d half-written artifact(s) from an "
            "interrupted run — those slides will be re-tiled / re-extracted.",
            tiling_dir, n_locks, n_artifacts,
        )


def _make_processor(tiling_dir: Path, cohort_wsi_dir: Path, slide_paths: Sequence[Path], *, max_workers: int):
    """Trident Processor on the per-cohort tiling dir, restricted to slide_paths (<slide_id>/<slide_id>.dcm entries)."""
    from trident import Processor  # type: ignore  # lazy: trident env
    return Processor(
        job_dir=str(tiling_dir),
        wsi_source=str(cohort_wsi_dir),
        wsi_ext=[".dcm"],
        search_nested=True,   # DICOM entries live in per-slide subdirs (ignored when selected_wsi_paths is set)
        selected_wsi_paths=[str(p) for p in slide_paths],
        # DataLoader worker cap (otherwise Trident may use 2*batch_limit workers); does not change the features
        max_workers=max_workers,
        # A slide that errors aborts the job (every stage is resumable per slide)
        skip_errors=False,
    )


def run_trident_tiling(slide_paths: Sequence[Path], cohort_wsi_dir: Path, tiling_dir: Path, *, device: str,
                       max_workers: int) -> Path:
    """Run Trident segmentation ('hest' model) and patching of a cohort; coords are written under tiling_dir/COORDS_REL."""
    from trident.segmentation_models.load import segmentation_model_factory  # type: ignore  # lazy: trident env

    processor = _make_processor(tiling_dir, cohort_wsi_dir, slide_paths, max_workers=max_workers)
    seg_model = segmentation_model_factory("hest")
    processor.run_segmentation_job(
        segmentation_model=seg_model, seg_mag=getattr(seg_model, "target_mag", 10), device=device,
    )
    processor.run_patching_job(
        target_magnification=TARGET_MAG, patch_size=PATCH_SIZE_PX, overlap=0, saveto=COORDS_REL,
    )
    coords_dir = tiling_dir / COORDS_REL
    if not coords_dir.exists():
        raise SystemExit(f"Trident produced no coords dir {coords_dir} (check the Processor API / output layout).")
    return coords_dir


def run_trident_feature_extraction(
    slide_paths: Sequence[Path],
    cohort_wsi_dir: Path,
    tiling_dir: Path,
    bundle: Path,
    enc_name: str,
    *,
    device: str,
    batch_limit: int,
    max_workers: int,
) -> Path:
    """
    Run Trident patch-feature extraction (fp32) for one encoder on the coords of run_trident_tiling.
    Writes tiling_dir/COORDS_REL/features_<enc_name>/<slide>.h5.
    """
    import torch
    from src.evaluation.part4_PathoBench.encoder_adapter import build_trident_custom_encoder

    custom_encoder = build_trident_custom_encoder(
        bundle_path=bundle, enc_name=enc_name, device=torch.device(device), precision=torch.float32,
    )
    # Feature extraction only (coords from run_trident_tiling)
    processor = _make_processor(tiling_dir, cohort_wsi_dir, slide_paths, max_workers=max_workers)
    processor.run_patch_feature_extraction_job(
        coords_dir=COORDS_REL, patch_encoder=custom_encoder, device=device, saveas="h5", batch_limit=batch_limit,
    )
    features_dir = tiling_dir / COORDS_REL / f"features_{enc_name}"
    if not features_dir.exists():
        raise SystemExit(f"Trident did not produce {features_dir} (check the Processor API / output layout).")
    return features_dir


def derive_clip_h5(he_h5: Path, clip_h5: Path, encoder: Any) -> None:
    """Write a clip feature h5 from a he feature h5: clip = L2(image_proj(he)), same coords (encoder already loaded)."""
    import h5py
    from src.evaluation.part4_PathoBench.encoder_adapter import derive_clip_from_he

    with h5py.File(he_h5, "r") as f:
        he = f["features"][:]
        coords = f["coords"][:]
        coord_attrs = dict(f["coords"].attrs)  # coords always present in a Trident feature h5
    clip = derive_clip_from_he(he, encoder)
    clip_h5.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(clip_h5, "w") as f:
        f.create_dataset("features", data=clip)
        d = f.create_dataset("coords", data=coords)
        for k, v in coord_attrs.items():
            d.attrs[k] = v


def load_clip_encoder(bundle: Path, tag: str) -> Any:
    """Load a bundle on CPU for the clip derivation; raises if it has no image_proj."""
    import torch
    from src.evaluation.part4_PathoBench.encoder_adapter import load_bundle_encoder, encoder_has_clip

    encoder = load_bundle_encoder(bundle, device=torch.device("cpu"))
    if not encoder_has_clip(encoder):
        raise SystemExit(
            f"Encoder {tag} is expected to carry a `clip` space (checkpoint_num >= 0) but its bundle "
            f"{bundle} exports no `image_proj`."
        )
    return encoder


# --------------------------------------------------------------------------- per cohort / encoder


def prepare_cohort_tiling(cohort: str, *, device: str, max_workers: int,
                          wsi_on_scratch: bool = False) -> List[Dict[str, str]]:
    """
    Read the cohort manifest, filter the slides, write the slide inventory, and run Trident segmentation + patching
    for the slides without coords (after clear_stale_locks). Returns the kept rows.
    """
    rows = read_cohort_manifest(cohort)
    kept, rejected = filter_no_upsampling(cohort, rows, wsi_on_scratch=wsi_on_scratch)
    write_slide_inventory(cohort, kept, rejected)
    if not kept:
        raise SystemExit(
            f"cohort {cohort}: no slides survived the no-upsampling gate ({len(rows)} in the manifest, "
            f"{sum(1 for r in rejected if r['failure_reason'] == 'wsi_not_found')} not found on disk). "
            f"WSIs were read from {cohort_wsi_dir(cohort, scratch=wsi_on_scratch)} (see --wsi_on_scratch). "
            f"See {slide_inventory_path(cohort)} for the per-slide reasons."
        )

    tiling_dir = cohort_tiling_dir(cohort)
    tiling_dir.mkdir(parents=True, exist_ok=True)
    clear_stale_locks(tiling_dir)
    wsi_dir = cohort_wsi_dir(cohort, scratch=wsi_on_scratch)
    # Resume: only tile slides whose coords are missing
    todo = [r for r in kept if not coords_h5_path(tiling_dir, r["slide_id"]).exists()]
    if todo:
        logger.info("cohort=%s: tiling %d/%d slides (%d cached coords).", cohort, len(todo), len(kept), len(kept) - len(todo))
        run_trident_tiling([Path(r["local_path"]) for r in todo], wsi_dir, tiling_dir, device=device,
                           max_workers=max_workers)
    else:
        logger.info("cohort=%s: coords cached for all %d slides (seg+patch skipped).", cohort, len(kept))
    return kept


def report_missing_features(
    cohort: str,
    tag: str,
    tiling_dir: Path,
    enc_name: str,
    missing: Sequence[str],
    n_todo: int,
    *,
    allow_missing: bool,
) -> None:
    """
    Raise (or warn if allow_missing) when Trident produced no he features for slides it was asked to extract.
    Coords present: the feature stage skipped the slide (locked by a live process); coords missing: segmentation
    or patching skipped it (lock, or no tissue detected).
    """
    detail = "\n  ".join(
        f"{sid} (coords {'present' if coords_h5_path(tiling_dir, sid).exists() else 'MISSING'})"
        for sid in missing
    )
    msg = (
        f"cohort={cohort} encoder={tag}: Trident produced no `he` features for {len(missing)}/{n_todo} "
        f"slide(s) it was asked to extract:\n  {detail}\n"
        f"Per-slide reasons are in Trident's logs:\n"
        f"  {tiling_dir / '_logs_segmentation.txt'}\n"
        f"  {tiling_dir / COORDS_REL / '_logs_coords.txt'}\n"
        f"  {tiling_dir / COORDS_REL / f'_logs_feats_{enc_name}.txt'}\n"
        f"'locked' => another _1_extract_features.py run is extracting this cohort: let it finish, then re-run. "
        f"'Empty GeoDataFrame' => no tissue detected: pass --allow_missing_features to proceed."
    )
    if allow_missing:
        logger.warning("%s", msg)
    else:
        raise SystemExit(msg)


def extract_cohort_encoder(
    cohort: str,
    kept: Sequence[Dict[str, str]],
    tiling_dir: Path,
    training_num: int,
    checkpoint_num: int,
    *,
    device: str,
    batch_limit: int,
    max_workers: int,
    wsi_on_scratch: bool = False,
    allow_missing_features: bool = False,
) -> None:
    tag = encoder_tag(training_num, checkpoint_num)
    bundle = bundle_path(training_num, checkpoint_num)
    if not bundle.exists():
        raise SystemExit(f"Encoder bundle not found: {bundle} (export it first).")

    out_root = trident_features_root(cohort, training_num, checkpoint_num)
    he_out = out_root / "he"
    clip_out = out_root / "clip"
    he_out.mkdir(parents=True, exist_ok=True)

    # Resume: only extract slides missing a `he` h5 for this encoder.
    todo = [r for r in kept if not (he_out / f"{r['slide_id']}.h5").exists()]
    logger.info("cohort=%s encoder=%s: %d slides, %d to extract (%d cached).",
                cohort, tag, len(kept), len(todo), len(kept) - len(todo))

    if todo:
        wsi_dir = cohort_wsi_dir(cohort, scratch=wsi_on_scratch)
        enc_name = encoder_features_run_id(training_num, checkpoint_num)
        features_dir = run_trident_feature_extraction(
            [Path(r["local_path"]) for r in todo], wsi_dir, tiling_dir, bundle,
            enc_name, device=device, batch_limit=batch_limit, max_workers=max_workers,
        )
        # Move Trident's he h5 files into he_out (used by the resume check above)
        missing: List[str] = []
        for r in todo:
            src = features_dir / f"{r['slide_id']}.h5"
            if not src.exists():
                missing.append(r["slide_id"])
                continue
            shutil.move(str(src), str(he_out / f"{r['slide_id']}.h5"))
        # Report the missing slides after moving the produced ones
        if missing:
            report_missing_features(cohort, tag, tiling_dir, enc_name, missing, len(todo),
                                    allow_missing=allow_missing_features)

    # Derive clip h5 for every he h5 still missing one (encoders with a clip space only)
    if expects_clip(training_num, checkpoint_num):
        pending = [h for h in list_h5(he_out) if not (clip_out / h.name).exists()]
        if pending:
            encoder = load_clip_encoder(bundle, tag)  # once per encoder, not once per slide
            for he_h5 in pending:
                derive_clip_h5(he_h5, clip_out / he_h5.name, encoder)

    logger.info("cohort=%s encoder=%s: he=%d, clip=%d feature files.",
                cohort, tag, len(list_h5(he_out)), len(list_h5(clip_out)))


# --------------------------------------------------------------------------- main


def main(args: argparse.Namespace) -> None:
    encoders = [parse_encoder_tag(t) for t in args.encoders]
    cohorts = args.cohorts or cohorts_in_registry()
    unknown = [c for c in cohorts if c not in cohorts_in_registry()]
    if unknown:
        raise SystemExit(f"Unknown cohort(s) {unknown}; registry cohorts are {cohorts_in_registry()}.")

    for cohort in cohorts:
        # Seg+patch once + inventory (encoder-independent)
        kept = prepare_cohort_tiling(cohort, device=args.device, max_workers=args.max_workers,
                                     wsi_on_scratch=args.wsi_on_scratch)
        tiling_dir = cohort_tiling_dir(cohort)
        for tn, cn in encoders:
            extract_cohort_encoder(cohort, kept, tiling_dir, tn, cn, device=args.device,
                                   batch_limit=args.batch_limit, max_workers=args.max_workers,
                                   wsi_on_scratch=args.wsi_on_scratch,
                                   allow_missing_features=args.allow_missing_features)

    logger.info("Done (features extracted). Next: _2_run_pathobench.py --eval_id <id> --encoders %s", " ".join(args.encoders))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Task D step 1: Trident he feature extraction (shared per-cohort tiling) + clip derivation.")
    parser.add_argument("--encoders", type=str, nargs="+", required=True, help="Encoder tags, e.g. t46e13 t9001e0 t-1e-1.")
    parser.add_argument("--cohorts", type=str, nargs="*", default=None, help="Cohort subset (default: all in the registry).")
    parser.add_argument("--device", type=str, default="cuda", help="Torch device for Trident (cuda / cpu).")
    parser.add_argument("--batch_limit", type=int, default=32, help="Trident patch-encoder batch size (GPU knob; independent of --max_workers).")
    parser.add_argument("--max_workers", type=int, default=16,
                        help=(
                            "Cap on Trident's DataLoader workers (tile decode). Keep it a few below the "
                            "job's --cpus-per-task. Does not change the extracted features."
                        ))
    parser.add_argument("--wsi_on_scratch", action="store_true",
                        help=(
                            "Read the WSI store under the $SCRATCH root (get_full_path(..., scratch=True)); "
                            "manifests, inventory, tiling and features stay under the default root."
                        ))
    parser.add_argument("--allow_missing_features", action="store_true",
                        help=(
                            "Warn instead of failing when Trident produces no `he` features for a slide "
                            "(e.g. no tissue detected)."
                        ))
    main(parser.parse_args())
