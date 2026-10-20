"""Output writers (predictions, metric tables) and across-seed aggregation for Task A.3."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Mapping

# sys.path setup (as in every src_A3 module)
from src_A3.common import ensure_imports

ensure_imports()

import h5py
import numpy as np
import pandas as pd

from src_A3.common import load_json, mkdir, rmtree_appledouble_safe, save_json, save_text

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# HDF5 string coercion helper
# -----------------------------------------------------------------------------

_H5_NULL_STRINGS = {"", "na", "nan", "none", "null", "<na>"}


def _as_h5_utf8_strings(
    values,
    *,
    name: str,
    n_expected: "int | None" = None,
    allow_missing: bool = False,
    missing_value: str = "unknown",
) -> np.ndarray:
    """
    Convert a 1D string-like sequence to an object array of str for h5py UTF-8 datasets.
    Missing values raise, or are replaced by missing_value if allow_missing.
    """
    s = pd.Series(list(values), dtype="string")

    missing = s.isna() | s.str.strip().str.casefold().isin(_H5_NULL_STRINGS)
    if bool(missing.any()):
        if not allow_missing:
            examples = pd.Series(list(values), dtype=object).loc[missing].head(10).tolist()
            raise ValueError(
                f"Cannot write HDF5 string dataset '{name}' because it contains "
                f"{int(missing.sum())} missing/null values. Examples: {examples}"
            )
        s = s.mask(missing, str(missing_value))

    s = s.astype("string").str.strip().fillna(str(missing_value))
    out = s.astype(object).to_numpy()

    if n_expected is not None and out.shape[0] != int(n_expected):
        raise ValueError(
            f"HDF5 string dataset '{name}' has length {out.shape[0]}, "
            f"expected {int(n_expected)}."
        )
    return out


# -----------------------------------------------------------------------------
# Prediction bundle writer (A.3 classification schema)
# -----------------------------------------------------------------------------


def save_a3_prediction_bundle(preds: Mapping[str, Mapping], out_dir: Path) -> None:
    """
    Write the per-slide class predictions as out_dir/<slide_id>.h5 (row_keys, class_names, coords_xy, y_true,
    y_proba, true_class_covered, optional fold_names / cell_types), optional row_metadata/<slide_id>.parquet,
    and the index out_dir/prediction_files.csv.
    """
    mkdir(out_dir)
    row_meta_dir = mkdir(out_dir / "row_metadata")
    meta_rows = []
    for slide_id, bundle in preds.items():
        out_path = out_dir / f"{slide_id}.h5"
        obs_path = ""
        y_true = np.asarray(bundle["y_true"], dtype=np.int32).reshape(-1)
        y_proba = np.asarray(bundle["y_proba"], dtype=np.float32)
        if y_proba.ndim != 2:
            raise ValueError(
                f"Bundle '{slide_id}' y_proba must be 2-D [N, C]; got shape {y_proba.shape}."
            )
        n_rows = int(y_true.shape[0])
        n_classes = int(y_proba.shape[1])

        with h5py.File(out_path, "w") as f:
            str_dt = h5py.string_dtype(encoding="utf-8")

            f.create_dataset(
                "row_keys",
                data=_as_h5_utf8_strings(bundle["row_keys"], name="row_keys", n_expected=n_rows),
                dtype=str_dt,
            )
            f.create_dataset(
                "class_names",
                data=_as_h5_utf8_strings(
                    bundle["class_names"], name="class_names", n_expected=n_classes
                ),
                dtype=str_dt,
            )
            f.create_dataset("coords_xy", data=np.asarray(bundle["coords_xy"], dtype=np.float32))
            f.create_dataset("y_true", data=y_true)
            f.create_dataset("y_proba", data=y_proba, compression="gzip", compression_opts=4)
            f.create_dataset(
                "true_class_covered",
                data=np.asarray(bundle["true_class_covered"], dtype=bool).reshape(-1),
            )
            if bundle.get("fold_names") is not None:
                # Per-row fold names
                f.create_dataset(
                    "fold_names",
                    data=_as_h5_utf8_strings(
                        bundle["fold_names"], name="fold_names", n_expected=n_rows
                    ),
                    dtype=str_dt,
                )
            if bundle.get("cell_types") is not None:
                f.create_dataset(
                    "cell_types",
                    data=_as_h5_utf8_strings(
                        bundle["cell_types"],
                        name="cell_types",
                        n_expected=n_rows,
                        allow_missing=True,
                        missing_value="unknown",
                    ),
                    dtype=str_dt,
                )
            f.attrs["slide_id"] = str(bundle["slide_id"])
            f.attrs["source"] = str(bundle["source"])
            f.attrs["subset"] = str(bundle["subset"])

        if bundle.get("obs_df") is not None:
            obs_path = str(row_meta_dir / f"{slide_id}.parquet")
            bundle["obs_df"].reset_index(drop=True).to_parquet(obs_path, index=False)

        meta_rows.append(
            {
                "slide_id": slide_id,
                "source": str(bundle["source"]),
                "subset": str(bundle["subset"]),
                "tissue": str(bundle["tissue"]),
                "n_rows": n_rows,
                "n_classes": n_classes,
                "file": str(out_path),
                "row_metadata_file": obs_path,
            }
        )
    pd.DataFrame(meta_rows).to_csv(out_dir / "prediction_files.csv", index=False)


# -----------------------------------------------------------------------------
# Metric table writers
# -----------------------------------------------------------------------------


def _write_table(df: object, out_path: Path, *, index: bool = False) -> None:
    """Write a DataFrame to CSV (nothing is written for None)."""
    if isinstance(df, pd.DataFrame):
        df.to_csv(out_path, index=index)
    elif df is None:
        logger.debug("A.3 metrics table absent (None); not writing %s.", out_path.name)
    else:
        try:
            pd.DataFrame(df).to_csv(out_path, index=index)
        except Exception as exc:  # pragma: no cover
            logger.warning("Could not coerce/write A.3 metrics table %s: %s", out_path.name, exc)


def _write_confusion_matrix_csv(confusion: object, out_path: Path) -> None:
    """
    Write a ``{"labels": [...], "matrix": [C, C]}`` confusion dict as a labelled CSV
    (rows = true label, columns = predicted label). Tolerates a None/degenerate payload.
    """
    if confusion is None:
        logger.debug("A.3 confusion matrix absent (None); not writing %s.", out_path.name)
        return
    if not isinstance(confusion, Mapping):
        logger.warning(
            "A.3 confusion payload for %s is not a mapping (%s); skipping.",
            out_path.name,
            type(confusion).__name__,
        )
        return
    labels = [str(x) for x in list(confusion.get("labels", []))]
    matrix = np.asarray(confusion.get("matrix", []))
    if matrix.ndim != 2:
        matrix = np.zeros((len(labels), len(labels)), dtype=np.int64)
    df = pd.DataFrame(matrix)
    if labels and len(labels) == df.shape[0]:
        df.index = pd.Index(labels, name="true_label")
    if labels and len(labels) == df.shape[1]:
        df.columns = pd.Index(labels, name="pred_label")
    df.to_csv(out_path)


def _write_metric_block(block: Mapping[str, object], out_dir: Path, *, prefix: str = "") -> None:
    """Write the classification block (``benchmark_summary`` + CSV tables)."""
    save_json(out_dir / f"{prefix}benchmark_summary.json", block.get("benchmark_summary", {}))
    _write_table(block.get("per_slide"), out_dir / f"{prefix}per_slide.csv")
    # Per-(slide, class) F1
    _write_table(block.get("per_slide_per_class"), out_dir / f"{prefix}per_slide_per_class.csv")
    _write_table(block.get("per_class"), out_dir / f"{prefix}per_class.csv")
    _write_confusion_matrix_csv(block.get("confusion"), out_dir / f"{prefix}confusion_matrix.csv")
    _write_table(block.get("fold_coverage"), out_dir / f"{prefix}fold_coverage.csv")
    _write_table(block.get("per_slide_restricted"), out_dir / f"{prefix}per_slide_restricted.csv")


def save_a3_metrics_outputs(metrics: Mapping[str, object], out_dir: Path) -> None:
    """
    Write the metric tables of compute_classification_metrics (benchmark_summary.json, per_slide.csv,
    per_slide_per_class.csv, per_class.csv, confusion_matrix.csv, fold_coverage.csv, per_slide_restricted.csv),
    and their coarse_* versions when metrics["coarse"] is set.
    """
    mkdir(out_dir)
    _write_metric_block(metrics, out_dir, prefix="")
    coarse = metrics.get("coarse")
    if isinstance(coarse, Mapping) and coarse:
        _write_metric_block(coarse, out_dir, prefix="coarse_")


def save_a3_seed_summary(seed_dir: Path, metrics: Mapping[str, object]) -> None:
    """Write ``seed_dir/seed_summary.txt`` — the benchmark-summary JSON dump plus row counts."""
    benchmark_summary = metrics.get("benchmark_summary", {})
    per_slide = metrics.get("per_slide")
    per_class = metrics.get("per_class")
    n_per_slide = int(len(per_slide)) if isinstance(per_slide, pd.DataFrame) else 0
    n_per_class = int(len(per_class)) if isinstance(per_class, pd.DataFrame) else 0
    summary_txt = [
        "Seed-level summary (Task A.3 cell-type classification)",
        json.dumps(benchmark_summary, indent=2, default=str),
        f"n_per_slide_rows: {n_per_slide}",
        f"n_per_class_rows: {n_per_class}",
    ]
    save_text(seed_dir / "seed_summary.txt", "\n".join(summary_txt) + "\n")


# -----------------------------------------------------------------------------
# Across-seed aggregation
# -----------------------------------------------------------------------------


def aggregate_a3_seed_outputs(encoder_dir: Path) -> None:
    """
    Aggregate the runs/seed_*/ outputs into encoder_dir/aggregate_across_seeds/ (rebuilt at each call):
    per_seed_benchmark.csv, aggregated_benchmark.csv (mean/std/min/max over seeds), per_seed_fold_benchmark.csv,
    per_fold_benchmark_mean_over_seeds.csv and fold_variability_benchmark.csv (std across the fold means).
    """
    encoder_dir = Path(encoder_dir)
    runs_dir = encoder_dir / "runs"
    if not runs_dir.exists():
        logger.warning("aggregate_a3_seed_outputs: no runs/ directory under %s.", encoder_dir)
        return

    seed_dirs = sorted(
        p for p in runs_dir.iterdir() if p.is_dir() and p.name.startswith("seed_")
    )
    agg_dir = encoder_dir / "aggregate_across_seeds"
    if agg_dir.exists():
        rmtree_appledouble_safe(agg_dir)
    agg_dir.mkdir(parents=True, exist_ok=True)

    def _aggregate_summary(subdir: str, filename: str, stem: str) -> None:
        rows = []
        for seed_dir in seed_dirs:
            path = seed_dir / subdir / filename
            if not path.exists():
                continue
            payload = load_json(path)
            rec = {"seed": int(seed_dir.name.split("_")[-1])}
            rec.update(payload)
            rows.append(rec)
        if not rows:
            return
        df = pd.DataFrame(rows).sort_values("seed")
        df.to_csv(agg_dir / f"per_seed_{stem}.csv", index=False)
        numeric_cols = [
            c for c in df.columns if c != "seed" and pd.api.types.is_numeric_dtype(df[c])
        ]
        if numeric_cols:
            agg = (
                df[numeric_cols]
                .agg(["mean", "std", "min", "max"])
                .T.reset_index()
                .rename(columns={"index": "metric"})
            )
            agg.to_csv(agg_dir / f"aggregated_{stem}.csv", index=False)

    def _aggregate_fold_metrics() -> None:
        """
        Aggregate fold_metrics.csv across probe seeds: all seed/fold rows, per-fold means over seeds,
        and fold_std across the per-fold means.
        """
        rows = []
        for seed_dir in seed_dirs:
            path = seed_dir / "fold_metrics.csv"
            if not path.exists():
                continue
            df = pd.read_csv(path)
            if df.empty or "fold_name" not in df.columns:
                continue
            df = df.copy()
            df.insert(0, "seed", int(seed_dir.name.split("_")[-1]))
            rows.append(df)

        if not rows:
            return

        df = pd.concat(rows, axis=0, ignore_index=True)
        df = df.sort_values(["fold_name", "seed"]).reset_index(drop=True)
        df.to_csv(agg_dir / "per_seed_fold_benchmark.csv", index=False)

        numeric_cols = [
            c
            for c in df.columns
            if c not in {"seed", "fold_name"} and pd.api.types.is_numeric_dtype(df[c])
        ]
        if not numeric_cols:
            return

        long_df = df.melt(
            id_vars=["seed", "fold_name"],
            value_vars=numeric_cols,
            var_name="metric",
            value_name="value",
        )
        long_df["value"] = pd.to_numeric(long_df["value"], errors="coerce")
        long_df = long_df.loc[np.isfinite(long_df["value"])].copy()
        if long_df.empty:
            return

        per_fold = (
            long_df.groupby(["fold_name", "metric"], dropna=False)["value"]
            .agg(
                mean_over_seeds="mean",
                seed_std_within_fold="std",
                min_over_seeds="min",
                max_over_seeds="max",
                n_seeds="count",
            )
            .reset_index()
        )
        per_fold.to_csv(agg_dir / "per_fold_benchmark_mean_over_seeds.csv", index=False)

        fold_var = (
            per_fold.groupby("metric", dropna=False)["mean_over_seeds"]
            .agg(
                mean_over_folds="mean",
                fold_std="std",
                min_over_folds="min",
                max_over_folds="max",
                n_folds="count",
            )
            .reset_index()
        )
        seed_noise = (
            per_fold.groupby("metric", dropna=False)["seed_std_within_fold"]
            .mean()
            .reset_index(name="mean_seed_std_within_fold")
        )
        fold_var = fold_var.merge(seed_noise, on="metric", how="left")
        fold_var["std_semantics"] = (
            "fold_std is std across fold means after averaging probe seeds within each fold; "
            "mean_seed_std_within_fold is the average probe-seed std inside a fixed fold"
        )
        fold_var.to_csv(agg_dir / "fold_variability_benchmark.csv", index=False)

    _aggregate_summary("benchmark", "benchmark_summary.json", "benchmark")
    _aggregate_fold_metrics()
