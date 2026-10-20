#!/usr/bin/env python
"""
Generate an HTML report of the HistoST CLIP training dynamics of one training_id, from the per-epoch
metrics (training/log/backup_training_metrics.jsonl) and the run config (<training_id>_histost.yaml).

Saved in <ROOT_HistoST> / train / trainings / trainings_HistoST / <training_id> / training /
    - report_training_<num>.html and report_training_<num>_assets/ (figures)
    - report_training_<num>_standalone.html: figures embedded (with --standalone)
Markdown copy (with --generate_md): access_reports/training_dynamics_t<num>.md in the repo.
"""

import argparse
import json
import logging
import math
import re
import sys
from datetime import datetime
from html import escape
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: must be set before importing pyplot
import matplotlib.pyplot as plt
import pandas as pd
import yaml

# Add the project root to sys.path so that 'src' can be imported.
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.constants import get_full_path
from src.utils.report_assets import (
    image_src,
    is_standalone,
    set_generate_md,
    set_standalone,
    should_generate_md,
    standalone_output_path,
    write_markdown_copy,
)

# Optional import (config reads use the local _cfg walker)
try:
    from src.utils.default_histost_config import merge_with_defaults
except Exception:  # pragma: no cover
    merge_with_defaults = None

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Small formatting helpers
# -----------------------------------------------------------------------------


def e(value):
    """HTML-escape a value (None -> '')."""
    return escape("" if value is None else str(value))


def fmt_val(v):
    """Compact display of a numeric value; None/NaN/non-numeric handled."""
    if v is None:
        return "—"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if math.isnan(f):
        return "—"
    if math.isinf(f):
        return "∞" if f > 0 else "-∞"
    if f == int(f) and abs(f) < 1e15:
        return str(int(f))
    return f"{f:.4g}"


def disp(v):
    """Display helper for config values (numbers via fmt_val, else str/—)."""
    if v is None:
        return "—"
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, (int, float)):
        return fmt_val(v)
    return str(v)


# -----------------------------------------------------------------------------
# Path helpers (all under ROOT_PATH via get_full_path)
# -----------------------------------------------------------------------------


def training_num_from_id(training_id):
    """'training_1' -> '1' (numeric part); falls back to a slug."""
    m = re.search(r"(-?\d+)", str(training_id))
    if m:
        return m.group(1)
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(training_id)).strip("_") or "x"


def training_dir(training_id):
    return get_full_path("train", "trainings", "trainings_HistoST", training_id, "training")


def jsonl_path(training_id):
    return training_dir(training_id) / "log" / "backup_training_metrics.jsonl"


def yaml_path(training_id):
    return training_dir(training_id) / f"{training_id}_histost.yaml"


def output_path_for(training_id):
    num = training_num_from_id(training_id)
    return training_dir(training_id) / f"report_training_{num}.html"


def assets_dir_for(base_output_html):
    return base_output_html.with_name(f"{base_output_html.stem}_assets")


# -----------------------------------------------------------------------------
# Data loading / access
# -----------------------------------------------------------------------------


def load_metrics_df(path):
    """One row per epoch. Tolerates missing file, blank lines and bad JSON."""
    if not path.exists() or not path.is_file():
        logger.warning("No training metrics JSONL at %s", path)
        return pd.DataFrame()
    rows, n_bad = [], 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                n_bad += 1
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    if n_bad:
        logger.warning("Skipped %d malformed JSONL line(s) in %s", n_bad, path)
    if not rows:
        logger.warning("No usable rows in %s", path)
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if "epoch" in df.columns:
        df["epoch"] = pd.to_numeric(df["epoch"], errors="coerce")
        df = df.sort_values("epoch").reset_index(drop=True)
        # On a resumed run the JSONL is appended; keep the last write per epoch.
        df = df.drop_duplicates(subset="epoch", keep="last").reset_index(drop=True)
    logger.info("Loaded %d epoch row(s) and %d metric column(s) from %s", df.shape[0], df.shape[1], path)
    return df


def series(df, key):
    """NaN-safe numeric Series for a metric column, or None if absent."""
    if df is None or df.empty or key not in df.columns:
        return None
    return pd.to_numeric(df[key], errors="coerce")


def has_any(df, key):
    s = series(df, key)
    return s is not None and bool(s.notna().any())


def last_epoch(df):
    if df is None or df.empty or "epoch" not in df.columns:
        return None
    eps = pd.to_numeric(df["epoch"], errors="coerce").dropna()
    return int(eps.iloc[-1]) if len(eps) else None


def value_at(df, key, epoch):
    s = series(df, key)
    if s is None or epoch is None or "epoch" not in df.columns:
        return None
    eps = pd.to_numeric(df["epoch"], errors="coerce")
    sel = s[eps == epoch].dropna()
    return float(sel.iloc[-1]) if len(sel) else None


def first_valid(df, key):
    s = series(df, key)
    if s is None:
        return None
    s = s.dropna()
    return float(s.iloc[0]) if len(s) else None


def last_valid(df, key):
    s = series(df, key)
    if s is None:
        return None
    s = s.dropna()
    return float(s.iloc[-1]) if len(s) else None


def first_valid_at_or_after(df, key, epoch_floor):
    """First non-NaN value of ``key`` at an epoch >= ``epoch_floor`` (warmup-aware baseline)."""
    s = series(df, key)
    if s is None or df is None or "epoch" not in df.columns:
        return None
    eps = pd.to_numeric(df["epoch"], errors="coerce")
    try:
        floor = float(epoch_floor)
    except (TypeError, ValueError):
        floor = 0.0
    sel = s[eps >= floor].dropna()
    return float(sel.iloc[0]) if len(sel) else None


def best_epoch_from_df(df):
    if df is None or df.empty or "epoch" not in df.columns:
        return None
    if "early_stopping/is_best_epoch" in df.columns:
        flags = pd.to_numeric(df["early_stopping/is_best_epoch"], errors="coerce")
        eps = pd.to_numeric(df["epoch"], errors="coerce")
        best = eps[flags == 1].dropna()
        if len(best):
            return int(best.iloc[-1])
    if "early_stopping/best_epoch" in df.columns:
        s = pd.to_numeric(df["early_stopping/best_epoch"], errors="coerce").dropna()
        if len(s):
            return int(s.iloc[-1])
    return None


def present_partitions(df):
    parts = ["all"]
    if df is None or df.empty:
        return parts
    for p in ("INT", "EXT"):
        prefix = f"global/{p}/"
        if any(c.startswith(prefix) and has_any(df, c) for c in df.columns):
            parts.append(p)
    return parts


def dynamic_topk_token_cols(df):
    # Dynamic top-K token columns (e.g. token/train_tok_top5); top1 is excluded (plotted in render_token_section)
    if df is None or df.empty:
        return []
    return sorted(
        c for c in df.columns
        if re.fullmatch(r"token/(train|val)_tok_top\d+", c) and not c.endswith("_top1")
    )


def lr_group_cols(df):
    if df is None or df.empty:
        return []
    return sorted(c for c in df.columns if c.startswith("lr/"))


def per_slide_ndcg_cols(df, part):
    if df is None or df.empty:
        return []
    return sorted(c for c in df.columns if c.startswith(f"global/{part}/global_slide_ndcg/"))


def within_slide_present(df):
    """True iff this run logged any within-slide retrieval metric."""
    if df is None or df.empty:
        return False
    return any(c.startswith("global/within_slide/") and has_any(df, c) for c in df.columns)


def within_slide_per_slide_cols(df, suffix):
    """Per-slide within-slide columns for a given leaf suffix, e.g. 'acc_i2st_top5' or 'mrr_i2st'."""
    if df is None or df.empty:
        return []
    grp = "global_slide_mrr" if suffix.startswith("mrr") else "global_slide_acc"
    pref = f"global/within_slide/{grp}/"
    return sorted(c for c in df.columns
                  if c.startswith(pref) and c.endswith(f"_{suffix}") and has_any(df, c))


# -----------------------------------------------------------------------------
# Config loading
# -----------------------------------------------------------------------------


def _cfg(d, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def warmup_epochs(cfg):
    """Token-loss warmup length in epochs from the run config; 0 when disabled/missing."""
    v = _cfg(cfg, "model", "clip", "lambda_token_warmup_epochs", default=0)
    try:
        return max(0, int(v or 0))
    except (TypeError, ValueError):
        return 0


def load_config(path):
    if not path.exists():
        logger.warning("No training config YAML at %s", path)
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
    except Exception as exc:  # pragma: no cover
        logger.warning("Could not parse config %s (%s)", path, exc)
        return {}
    if merge_with_defaults is not None:
        try:
            return merge_with_defaults(raw)
        except Exception as exc:  # pragma: no cover
            logger.warning("merge_with_defaults failed (%s); using raw config.", exc)
    return raw


def config_summary_cards(cfg):
    g = lambda *k: _cfg(cfg, *k, default=None)
    rows = [
        ("training_id", g("logging", "training_id_histost")),
        ("dataset_id", g("data", "dataset_id")),
        ("training_dataset_id", g("data", "training_dataset_id")),
        ("H&E backbone", g("model", "he", "backbone_img_encoder")),
        ("epochs (config)", g("training", "epochs")),
        ("batch_size", g("training", "batch_size")),
        ("optimizer lr", g("training", "optimizer_hyperparameters", "lr")),
        ("eval_every", g("training", "eval_every")),
        ("CLIP embed dim", g("model", "clip", "clip_embed_dim")),
        ("lambda_patch", g("model", "clip", "lambda_patch")),
        ("lambda_token", g("model", "clip", "lambda_token")),
        ("lambda_token warmup", g("model", "clip", "lambda_token_warmup_epochs")),
        ("HE training", g("training", "HE_training_option")),
        ("early-stop monitor", g("training", "early_stopping", "monitor")),
    ]
    return [{"label": k, "value": disp(v)} for k, v in rows]


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------


def save_line_plot(df, series_keys, *, assets_dir, fname, title, ylabel="",
                   best_epoch=None, hline=None, logy=False, warmup=None):
    """Draw epoch-indexed series; skip all-NaN series; return None if nothing to draw."""
    if df is None or df.empty or "epoch" not in df.columns:
        return None
    x = pd.to_numeric(df["epoch"], errors="coerce")
    drawable = []
    for key, label in series_keys:
        s = series(df, key)
        if s is not None and s.notna().any():
            drawable.append((s, label))
    if not drawable:
        return None
    assets_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    if warmup:
        try:
            w = float(warmup)
        except (TypeError, ValueError):
            w = 0.0
        if w > 0:
            # Shade the token-loss warmup [0, warmup)
            ax.axvspan(0, w, color="#f59e0b", alpha=0.08, linewidth=0,
                       label=f"token-loss warmup (0-{int(w)})")
    for s, label in drawable:
        m = s.notna() & x.notna()
        ax.plot(x[m], s[m], marker="o", markersize=3, linewidth=1.6, label=str(label))
    if best_epoch is not None:
        try:
            ax.axvline(float(best_epoch), color="#9a3412", linestyle="--", linewidth=1.2,
                       alpha=0.85, label=f"best epoch ({best_epoch})")
        except (TypeError, ValueError):
            pass
    if hline is not None:
        ax.axhline(hline, color="#667085", linestyle=":", linewidth=1.0, alpha=0.7)
    if logy:
        ax.set_yscale("log")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("epoch")
    ax.set_ylabel(ylabel)
    ax.grid(True, color="#e6ebf2", linewidth=0.8)
    if len(drawable) <= 14:  # avoid an unreadable legend for per-slide plots
        ax.legend(fontsize=8, frameon=False, ncol=2)
    fig.tight_layout()
    out = assets_dir / fname
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return out


def _distinct_colors(n):
    """n visually distinct RGBA colors: tab20 for n <= 20, else evenly sampled from gist_ncar."""
    if n <= 0:
        return []
    if n <= 20:
        cmap = plt.get_cmap("tab20")
        return [cmap(i % 20) for i in range(n)]
    cmap = plt.get_cmap("gist_ncar")
    if n == 1:
        return [cmap(0.5)]
    # Sample within [0.02, 0.98] to avoid the near-white / near-black ends.
    return [cmap(0.02 + 0.96 * (i / (n - 1))) for i in range(n)]


def save_per_slide_plot(df, series_keys, *, assets_dir, fname, title, ylabel=""):
    """Per-slide curves (one color per slide, legend outside the axes). Returns the PNG path or None."""
    if df is None or df.empty or "epoch" not in df.columns:
        return None
    x = pd.to_numeric(df["epoch"], errors="coerce")
    drawable = []
    for key, label in series_keys:
        s = series(df, key)
        if s is not None and s.notna().any():
            drawable.append((s, label))
    if not drawable:
        return None
    assets_dir.mkdir(parents=True, exist_ok=True)
    colors = _distinct_colors(len(drawable))
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    for (s, label), color in zip(drawable, colors):
        m = s.notna() & x.notna()
        ax.plot(x[m], s[m], marker="o", markersize=2.5, linewidth=1.3, label=str(label), color=color)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("epoch")
    ax.set_ylabel(ylabel)
    ax.grid(True, color="#e6ebf2", linewidth=0.8)
    n = len(drawable)
    ncol = 1 if n <= 16 else (2 if n <= 36 else 3)
    ax.legend(fontsize=7, frameon=False, ncol=ncol, loc="upper left",
              bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    fig.tight_layout()
    out = assets_dir / fname
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return out


# -----------------------------------------------------------------------------
# HTML builders
# -----------------------------------------------------------------------------


def section(title, anchor, body):
    return f'<section id="{e(anchor)}"><h2>{e(title)}</h2>{body}</section>'


def explanation(text=None, *, lead=None, bullets=None):
    """Explanatory callout box.

    text    : optional prose; blank-line-separated paragraphs.
    lead    : optional short intro sentence rendered as the first paragraph.
    bullets : optional iterable of points, each a string or a (term, description) pair.
    All text is HTML-escaped.
    """
    parts = []
    if lead:
        parts.append(f"<p>{e(lead)}</p>")
    if text:
        parts.extend(f"<p>{e(p)}</p>" for p in str(text).strip().split("\n\n") if p.strip())
    if bullets:
        lis = []
        for b in bullets:
            if isinstance(b, (tuple, list)) and len(b) == 2:
                lis.append(f"<li><b>{e(b[0])}</b> — {e(b[1])}</li>")
            else:
                lis.append(f"<li>{e(b)}</li>")
        parts.append('<ul class="explain-list">' + "".join(lis) + "</ul>")
    return f'<div class="explain">{"".join(parts)}</div>'


def note(text, missing=False):
    cls = "note missing" if missing else "note"
    return f'<div class="{cls}">{e(text)}</div>'


def callout(tone, text, label=None):
    tone = tone if tone in {"good", "warn", "bad", "neutral"} else "neutral"
    tag = f'<span class="verdict-tag">{e(label)}</span>' if label else ""
    return f'<div class="callout {tone}">{tag}{e(text)}</div>'


def details(title, body, open_=False):
    attr = " open" if open_ else ""
    return f"<details{attr}><summary>{e(title)}</summary>{body}</details>"


def variable_definitions(items, title="Show metric glossary", open_=False):
    lis = "".join(f"<li><b>{e(k)}</b> — {e(v)}</li>" for k, v in items)
    return details(title, f'<ul class="var-defs">{lis}</ul>', open_=open_)


def card_grid(cards):
    out = []
    for c in cards:
        trend = c.get("trend", "")
        cls = " good" if trend == "good" else (" bad" if trend == "bad" else "")
        sub = f'<div class="card-subtitle">{e(c.get("subtitle", ""))}</div>' if c.get("subtitle") else ""
        out.append(
            f'<div class="card{cls}"><div class="card-label">{e(c.get("label", ""))}</div>'
            f'<div class="card-value">{e(c.get("value", ""))}</div>{sub}</div>'
        )
    return '<div class="cards">' + "".join(out) + "</div>"


def button_link(label, href, external=True):
    tgt = ' target="_blank" rel="noopener noreferrer"' if external else ""
    return f'<a class="button-link" href="{e(href)}"{tgt}>{e(label)}</a>'


def image_html(path, output_html, caption=""):
    if path is None:
        return ""
    p = Path(path)
    if not p.exists():
        return note(f"Missing figure: {p.name}", missing=True)
    src = e(image_src(p, output_html))
    cap = f'<figcaption class="caption">{e(caption)}</figcaption>' if caption else ""
    return (
        f'<figure class="plot-figure"><a href="{src}">'
        f'<img class="plot-img" src="{src}" alt="{e(caption)}" loading="lazy"></a>{cap}</figure>'
    )


def html_table(df, table_id, searchable=False, small=False):
    if df is None or df.empty:
        return note("No metrics rows to display.", missing=True)
    cols = list(df.columns)
    numeric = {c: pd.api.types.is_numeric_dtype(df[c]) for c in cols}
    thead = "".join(
        f'<th class="sortable{" right" if numeric[c] else ""}" '
        f"onclick=\"sortTable('{table_id}',{i})\">{e(c)}</th>"
        for i, c in enumerate(cols)
    )
    body_rows = []
    for _, row in df.iterrows():
        tds = []
        for c in cols:
            cls = ' class="right"' if numeric[c] else ""
            tds.append(f"<td{cls}>{e(fmt_val(row[c]))}</td>")
        body_rows.append("<tr>" + "".join(tds) + "</tr>")
    search = (
        f'<input class="table-search" type="text" placeholder="Filter rows…" '
        f"oninput=\"filterTable('{table_id}',this.value)\">"
        if searchable
        else ""
    )
    small_cls = " small" if small else ""
    return (
        search
        + f'<div class="table-wrap"><table id="{e(table_id)}" class="data-table{small_cls}">'
        + f"<thead><tr>{thead}</tr></thead><tbody>{''.join(body_rows)}</tbody></table></div>"
    )


def css():
    return """
:root {
  --bg: #f6f8fb; --card: #ffffff; --text: #18202b; --muted: #667085;
  --line: #d9e0ea; --soft: #eef3f8; --accent: #2b6cb0; --accent-soft: #e7f0fb;
  --ok: #176c3a; --warn: #9a3412; --danger: #991b1b;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--text);
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  line-height: 1.45; }
header { padding: 34px 42px 24px; background: linear-gradient(135deg, #14213d, #275783); color: white; }
header h1 { margin: 0 0 8px; font-size: 30px; }
header p { margin: 0; color: rgba(255,255,255,0.84); }
.kpi-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; margin-top: 16px; }
.kpi { background: rgba(255,255,255,0.12); border-radius: 12px; padding: 10px 12px; }
.kpi .k { font-size: 11px; text-transform: uppercase; letter-spacing: 0.04em; color: rgba(255,255,255,0.78); }
.kpi .v { font-size: 18px; font-weight: 700; margin-top: 3px; }
.layout { display: grid; grid-template-columns: 278px minmax(0, 1fr); gap: 24px; padding: 24px; }
nav { position: sticky; top: 18px; align-self: start; background: var(--card); border: 1px solid var(--line);
  border-radius: 16px; padding: 16px; box-shadow: 0 8px 24px rgba(20,33,61,0.06);
  max-height: calc(100vh - 36px); overflow: auto; }
nav strong { display: block; margin-bottom: 8px; }
nav a { display: block; color: var(--accent); text-decoration: none; padding: 6px 0; font-size: 13px; }
nav a:hover { text-decoration: underline; }
main { min-width: 0; }
section { background: var(--card); border: 1px solid var(--line); border-radius: 18px; padding: 22px;
  margin-bottom: 22px; box-shadow: 0 8px 24px rgba(20,33,61,0.05); }
h2 { margin: 0 0 16px; font-size: 23px; }
h3 { margin: 24px 0 10px; font-size: 18px; }
h4 { margin: 18px 0 8px; font-size: 15px; color: var(--muted); }
.explain { background: var(--accent-soft); border-left: 4px solid var(--accent); border-radius: 12px;
  padding: 12px 14px; margin: 14px 0; color: #21354d; }
.explain p { margin: 0 0 9px; }
.explain p:last-child { margin-bottom: 0; }
.explain ul.explain-list { margin: 8px 0 0; padding-left: 20px; }
.explain ul.explain-list li { margin: 4px 0; line-height: 1.5; }
.explain ul.explain-list li b { color: #1c2c41; font-weight: 650; }
.note { background: var(--soft); border: 1px solid var(--line); border-radius: 12px; padding: 10px 12px;
  margin: 12px 0; color: var(--muted); }
.note.missing { background: #fff7ed; color: var(--warn); border-color: #fed7aa; }
.callout { border-radius: 12px; border-left: 4px solid; padding: 11px 14px; margin: 14px 0; font-size: 14px; }
.callout .verdict-tag { font-weight: 700; text-transform: uppercase; letter-spacing: 0.04em; font-size: 11px; margin-right: 8px; }
.callout.good { background: #f0fdf4; border-color: var(--ok); color: #14532d; }
.callout.warn { background: #fff7ed; border-color: var(--warn); color: #7c2d12; }
.callout.bad { background: #fef2f2; border-color: var(--danger); color: #7f1d1d; }
.callout.neutral { background: var(--accent-soft); border-color: var(--accent); color: #21354d; }
.muted { color: var(--muted); }
.button-link { display: inline-block; margin-top: 4px; padding: 8px 14px; background: var(--accent);
  color: white; text-decoration: none; border-radius: 10px; font-size: 13px; font-weight: 650; }
.button-link:hover { background: #1f5288; }
.cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 12px; margin: 10px 0 18px; }
.card { background: #fbfdff; border: 1px solid var(--line); border-radius: 16px; padding: 16px; }
.card.good { border-color: #bbf7d0; background: #f0fdf4; }
.card.bad { border-color: #fecaca; background: #fef2f2; }
.card-label { color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: 0.04em; }
.card-value { font-size: 24px; font-weight: 700; margin: 6px 0; }
.card-subtitle { color: var(--muted); font-size: 13px; }
.plot-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(330px, 1fr)); gap: 16px; margin: 12px 0; }
.plot-figure { margin: 0; }
.plot-img { max-width: 100%; height: auto; border: 1px solid var(--line); border-radius: 12px; background: white; }
.caption { color: var(--muted); font-size: 12px; margin-top: 6px; }
.tv-figure { margin: 0; }
.tv-pane { margin: 0; }
.curve-toggle { display: inline-flex; gap: 3px; margin: 0 0 8px; padding: 3px;
  background: var(--soft); border: 1px solid var(--line); border-radius: 10px; }
.tv-btn { border: 0; background: transparent; color: var(--muted); font-size: 12px;
  font-weight: 650; padding: 4px 12px; border-radius: 8px; cursor: pointer; }
.tv-btn:hover { color: var(--accent); }
.tv-btn.active { background: var(--accent); color: #fff; }
.table-search { width: min(420px, 100%); padding: 9px 11px; border: 1px solid var(--line); border-radius: 10px; margin: 8px 0 10px; }
.table-wrap { overflow: auto; border: 1px solid var(--line); border-radius: 12px; margin: 10px 0; max-height: 70vh; }
table.data-table { border-collapse: collapse; width: 100%; font-size: 13px; background: white; }
table.data-table.small { font-size: 12px; }
th, td { border-bottom: 1px solid var(--line); padding: 8px 10px; vertical-align: top; white-space: nowrap; }
th { position: sticky; top: 0; background: #f8fafc; z-index: 1; text-align: left; color: #334155; }
th.sortable { cursor: pointer; }
th.sortable:hover { color: var(--accent); }
td.right, th.right { text-align: right; font-variant-numeric: tabular-nums; }
tr:hover td { background: #f8fbff; }
details { border: 1px solid var(--line); border-radius: 12px; padding: 10px 12px; margin: 12px 0; background: #fcfdff; }
summary { cursor: pointer; color: var(--accent); font-weight: 650; }
.var-defs { margin: 6px 0 0; padding-left: 22px; color: #344054; }
.var-defs li { margin: 4px 0; line-height: 1.5; }
.var-defs b { color: #273549; font-weight: 650; }
.json-block { white-space: pre-wrap; background: #0f172a; color: #e2e8f0; border-radius: 12px; padding: 14px; overflow: auto; font-size: 12px; }
@media (max-width: 980px) {
  .layout { grid-template-columns: 1fr; padding: 14px; }
  nav { position: relative; top: 0; }
  header { padding: 24px 18px; }
}
"""


def js():
    return r"""
function sortTable(tableId, colIndex) {
  const table = document.getElementById(tableId);
  if (!table || !table.tBodies.length) return;
  const tbody = table.tBodies[0];
  const rows = Array.from(tbody.rows);
  const prev = table.getAttribute('data-sort-col');
  const prevDir = table.getAttribute('data-sort-dir') || 'asc';
  const dir = (prev === String(colIndex) && prevDir === 'asc') ? 'desc' : 'asc';
  table.setAttribute('data-sort-col', String(colIndex));
  table.setAttribute('data-sort-dir', dir);
  function parseCell(row) {
    let text = row.cells[colIndex].innerText.trim().replace(/,/g, '');
    if (text === '—' || text === '') return dir === 'asc' ? Infinity : -Infinity;
    let match = text.match(/-?\d+(\.\d+)?/);
    if (match && /^-?\d+(\.\d+)?/.test(text)) return parseFloat(match[0]);
    return text.toLowerCase();
  }
  rows.sort((a, b) => {
    const av = parseCell(a), bv = parseCell(b);
    if (typeof av === 'number' && typeof bv === 'number') return dir === 'asc' ? av - bv : bv - av;
    if (av < bv) return dir === 'asc' ? -1 : 1;
    if (av > bv) return dir === 'asc' ? 1 : -1;
    return 0;
  });
  rows.forEach(r => tbody.appendChild(r));
}
function filterTable(tableId, query) {
  const table = document.getElementById(tableId);
  if (!table || !table.tBodies.length) return;
  const q = query.toLowerCase();
  Array.from(table.tBodies[0].rows).forEach(row => {
    row.style.display = row.innerText.toLowerCase().includes(q) ? '' : 'none';
  });
}
function toggleCurves(groupId, which, btn) {
  const group = document.getElementById(groupId);
  if (!group) return;
  group.querySelectorAll('.tv-pane').forEach(function (p) {
    p.style.display = (p.getAttribute('data-which') === which) ? '' : 'none';
  });
  group.querySelectorAll('.tv-btn').forEach(function (b) { b.classList.remove('active'); });
  if (btn) btn.classList.add('active');
}
"""


# -----------------------------------------------------------------------------
# Section renderers
# -----------------------------------------------------------------------------

NAV = [
    ("run-summary", "Run summary"),
    ("scorecards", "Headline scorecards"),
    ("losses-schedules", "Losses & schedules"),
    ("token-head", "Token head"),
    ("step-retrieval", "Step retrieval"),
    ("global-performance", "Global - Performance"),
    ("global-within-slide", "Global - Within-slide"),
    ("global-geometry-health", "Global - Geometry health"),
    ("global-coherence", "Global - Coherence"),
    ("global-trustworthiness", "Global - Trustworthiness"),
    ("data-quality", "Data quality"),
    ("early-stopping", "Early stopping"),
    ("metrics-table", "Per-epoch table"),
    ("glossary", "Metric glossary"),
    ("reproducibility", "Reproducibility"),
]

GLOSSARY = [
    ("Loss (total/patch/token)", "CLIP contrastive objective; total = lambda_patch*patch + lambda_token*token. Lower is better."),
    ("Top-1/Top-k accuracy", "Is the true pair retrieved at rank 1 / within top-k? Higher is better."),
    ("MRR", "Mean reciprocal rank of the true pair. Higher is better (1 = always rank 1)."),
    ("nDCG@K", "Graded ranking quality using soft H&E/ST similarity gains. Higher is better."),
    ("Spearman rho", "Agreement between predicted ranking and the soft-gain ranking. Higher is better."),
    ("Cosine pos/neg/gap", "Mean cosine of matched pairs vs hardest negative; the gap should widen."),
    ("ECE", "Expected calibration error (15 bins). Lower is better (0 = perfectly calibrated)."),
    ("Hubness Gini / Top-1% mass", "Concentration of retrievals on a few 'hub' items. Lower is better."),
    ("kNN agreement@10", "Do image-image neighbours survive cross-modal retrieval? Higher is better."),
    ("CKA", "Representation similarity (e.g. H&E pre vs shared). Context-dependent, not a quality score."),
    ("Alignment MSE", "Mean 2(1-cos) over true pairs. Lower is better (pairs co-located)."),
    ("Spectrum (EffRank / PartRatio)", "Soft dimensionality (effective rank / participation ratio) of the embedding cloud. Higher = richer (avoid collapse)."),
    ("Uniformity", "Spread on the unit sphere. More negative is better; near 0 = collapse."),
    ("MMD (RBF)", "Distribution gap between H&E and ST clouds. Lower is better."),
    ("logit scale", "Learnable CLIP temperature; rising sharpens confidence (can worsen ECE)."),
    ("lambda_patch / lambda_token", "Loss weights; lambda_token may warm up over the first epochs."),
]

# Headline metrics for the scorecards: (key, label, direction, hint).
# direction: 'up' = higher better, 'down'/'neg' = lower/more-negative better.
HEADLINE = [
    ("losses/val_loss", "Val loss", "down", "lower is better"),
    ("global/all/global_acc/val_global_acc_i2st_top1", "Global acc@1 (i2st)", "up", "higher is better"),
    ("global/within_slide/global_acc/val_within_slide_acc_i2st_top1", "Within-slide acc@1 (i2st)", "up", "batch-controlled; higher is better"),
    ("global/within_slide/global_mrr/val_within_slide_mrr_i2st", "Within-slide MRR (i2st)", "up", "batch-controlled; higher is better"),
    ("global/all/global_mrr/val_global_mrr_i2st", "Global MRR (i2st)", "up", "higher is better"),
    ("global/all/global_ndcg/val_global_ndcg_i2st_top10", "Global nDCG@10 (i2st)", "up", "higher is better"),
    ("global/all/global_geom/val_global_align_mse", "Alignment MSE", "down", "lower is better"),
    ("global/all/global_calib/val_ece_i2st_bins15", "ECE (i2st)", "down", "lower is better"),
    ("global/all/global_hub/val_gini_i2st@10", "Hubness Gini@10 (i2st)", "down", "lower is better"),
    ("global/all/global_geom/val_global_uni_both", "Uniformity (both)", "neg", "more negative is better"),
]


def _fig(df, series_keys, fname, title, assets_dir, output_html, ylabel="", **kw):
    p = save_line_plot(df, series_keys, assets_dir=assets_dir, fname=fname, title=title, ylabel=ylabel, **kw)
    return image_html(p, output_html, caption=title) if p is not None else ""


def _classify_tv(series_keys):
    """Split (key, label) pairs into (train, val, other) by matching a train/val token in the key."""
    train, val, other = [], [], []
    for key, label in series_keys:
        k = str(key).lower()
        if re.search(r"(?:^|[/_])train(?:[/_]|$)", k):
            train.append((key, label))
        elif re.search(r"(?:^|[/_])val(?:[/_]|$)", k):
            val.append((key, label))
        else:
            other.append((key, label))
    return train, val, other


def toggle_figure_html(base, figs):
    """Wrap the three pre-rendered curve views (both/train/val) in a Both/Train/Val toggle.
    The button bar and the hidden train/val panes carry class ``md-skip`` so the Markdown copy
    keeps just one figure per chart."""
    gid = "tv_" + re.sub(r"[^A-Za-z0-9_-]+", "_", str(base))
    btns = "".join(
        f'<button type="button" class="tv-btn{" active" if w == "both" else ""}" '
        f"onclick=\"toggleCurves('{gid}','{w}',this)\">{lbl}</button>"
        for w, lbl in (("both", "Both"), ("train", "Train"), ("val", "Val"))
    )
    panes = []
    for w in ("both", "train", "val"):
        style = "" if w == "both" else ' style="display:none"'
        skip = "" if w == "both" else " md-skip"
        panes.append(f'<div class="tv-pane{skip}" data-which="{w}"{style}>{figs.get(w, "")}</div>')
    return (f'<div class="tv-figure" id="{gid}">'
            f'<div class="curve-toggle md-skip" role="group" aria-label="curve filter">{btns}</div>'
            + "".join(panes) + "</div>")


def _fig_tv(df, series_keys, fname, title, assets_dir, output_html, ylabel="", **kw):
    """Like :func:`_fig`, but when the chart has both train and val series it emits a
    Both/Train/Val toggle over three separately rendered, independently y-scaled PNGs (so each
    view rescales the axis to the curves shown). Falls back to a plain :func:`_fig` otherwise."""
    train, val, other = _classify_tv(series_keys)
    if not (train and val):
        return _fig(df, series_keys, fname, title, assets_dir, output_html, ylabel=ylabel, **kw)
    base = fname[:-4] if fname.lower().endswith(".png") else fname
    views = {"both": series_keys, "train": train + other, "val": val + other}
    figs = {}
    for which, sk in views.items():
        p = save_line_plot(df, sk, assets_dir=assets_dir, fname=f"{base}_{which}.png",
                           title=title, ylabel=ylabel, **kw)
        figs[which] = image_html(p, output_html, caption=title) if p is not None else ""
    if not figs.get("both"):
        return ""
    return toggle_figure_html(base, figs)


def _grid(figs):
    figs = [f for f in figs if f]
    return ('<div class="plot-grid">' + "".join(figs) + "</div>") if figs else ""


def header_kpis(df):
    n_epochs = int(df.shape[0]) if (df is not None and not df.empty) else 0
    be = best_epoch_from_df(df)
    nval = None
    if df is not None and not df.empty and "data/val_epoch_ran" in df.columns:
        nval = int(pd.to_numeric(df["data/val_epoch_ran"], errors="coerce").fillna(0).sum())
    final_loss = last_valid(df, "losses/val_loss")
    if final_loss is None:
        final_loss = last_valid(df, "losses/train_loss")
    final_acc = last_valid(df, "global/all/global_acc/val_global_acc_i2st_top1")
    return [
        ("epochs logged", str(n_epochs)),
        ("best epoch", str(be) if be is not None else "—"),
        ("val epochs", str(nval) if nval is not None else "—"),
        ("final loss", disp(final_loss)),
        ("final global acc@1", disp(final_acc)),
    ]


def render_header(args, cfg, df):
    chips = "".join(
        f'<div class="kpi"><div class="k">{e(k)}</div><div class="v">{e(v)}</div></div>'
        for k, v in header_kpis(df)
    )
    wandb = button_link("Open W&B project", args.wandb_clip_url) if args.wandb_clip_url else ""
    return f"""<header>
  <h1>HistoST training report — {e(args.training_id)}</h1>
  <p>Dual-level CLIP training dynamics (H&amp;E &harr; ST), built from backup_training_metrics.jsonl.</p>
  <div class="kpi-grid">{chips}</div>
  <div style="margin-top:14px;">{wandb}</div>
</header>"""


def make_nav():
    items = "".join(f'<a href="#{e(a)}">{e(t)}</a>' for a, t in NAV)
    return f"<nav><strong>Contents</strong>{items}</nav>"


def _good(direction, v0, v1):
    if v0 is None or v1 is None:
        return ""
    if direction == "up":
        return "good" if v1 > v0 else ("bad" if v1 < v0 else "")
    return "good" if v1 < v0 else ("bad" if v1 > v0 else "")  # down / neg


def scorecards(df, epoch, warmup=0):
    cards = []
    for key, label, direction, hint in HEADLINE:
        v1 = value_at(df, key, epoch)
        if v1 is None:
            v1 = last_valid(df, key)
        # Baseline: first value at/after the end of the token-loss warmup (else the first valid value)
        v0 = first_valid_at_or_after(df, key, warmup)
        if v0 is None:
            v0 = first_valid(df, key)
        cards.append({
            "label": label,
            "value": fmt_val(v1) if v1 is not None else "—",
            "subtitle": hint,
            "trend": _good(direction, v0, v1),
        })
    return cards


def verdict_callout(cards, warmup=0):
    goods = sum(1 for c in cards if c["trend"] == "good")
    bads = sum(1 for c in cards if c["trend"] == "bad")
    n = goods + bads
    base = "first post-warmup" if warmup and warmup > 0 else "first"
    if n == 0:
        return note("Not enough validation history yet to assess the training direction.")
    if goods >= bads and goods >= max(1, math.ceil(0.6 * n)):
        return callout("good", f"{goods}/{n} headline metrics improved from the {base} to the selected epoch.", label="Healthy")
    if bads > goods:
        return callout("bad", f"Only {goods}/{n} headline metrics improved; {bads} regressed. "
                              "Inspect the curves below.", label="Check")
    return callout("warn", f"{goods}/{n} headline metrics improved — mixed signal. Read the curves below.", label="Mixed")


def render_run_summary(cfg):
    body = explanation(
        "Configuration and identifiers for this training_id, read from the run YAML. "
        "Missing entries show as —."
    )
    body += card_grid(config_summary_cards(cfg))
    return section("Run summary", "run-summary", body)


def render_scorecards(df, be, cfg):
    if df is None or df.empty:
        return section("Headline scorecards", "scorecards", note("No metrics available.", missing=True))
    le = last_epoch(df)
    sel = be if be is not None else le
    wu = warmup_epochs(cfg)
    best_cards = scorecards(df, sel, wu)
    final_cards = scorecards(df, le, wu)
    baseline = (f"the first post-warmup validation epoch (token-loss warmup = {wu} epochs)"
                if wu > 0 else "the first validation epoch")
    bullets = [("Colour", f"green = improved, red = regressed, vs {baseline}.")]
    if wu > 0:
        bullets.append(("Warmup", "the total loss ramps up during the token-loss warmup epochs."))
    bullets.append(("Directions", "the preferred direction of each metric is given under its value."))
    body = explanation(
        lead="Headline metrics at the best epoch (by the early-stopping monitor) and at the final logged epoch.",
        bullets=bullets,
    )
    body += verdict_callout(best_cards, wu)
    body += f"<h3>Best epoch ({e(sel)})</h3>" + card_grid(best_cards)
    body += f"<h3>Final epoch ({e(le)})</h3>" + card_grid(final_cards)
    return section("Headline scorecards", "scorecards", body)


def render_losses_section(df, assets_dir, output_html, be, warmup=0):
    F = lambda s, fname, title, **kw: _fig(df, s, fname, title, assets_dir, output_html, **kw)
    FT = lambda s, fname, title, **kw: _fig_tv(df, s, fname, title, assets_dir, output_html, **kw)
    figs = [
        FT([("losses/train_loss", "train"), ("losses/val_loss", "val")], "loss_total.png", "Total CLIP loss", ylabel="loss", best_epoch=be, warmup=warmup),
        FT([("losses/train_patch_loss", "train"), ("losses/val_patch_loss", "val")], "loss_patch.png", "Patch contrastive loss", ylabel="loss", best_epoch=be, warmup=warmup),
        FT([("losses/train_token_loss", "train"), ("losses/val_token_loss", "val")], "loss_token.png", "Token contrastive loss", ylabel="loss", best_epoch=be, warmup=warmup),
        F([("lambda/lambda_patch", "lambda patch"), ("lambda/lambda_token", "lambda token")], "lambdas.png", "Loss weights (lambda) schedule", ylabel="lambda"),
        F([("last_logit_scale", "patch"), ("last_logit_scale_token", "token")], "logit_scale.png", "Logit scale (inverse temperature)", ylabel="logit scale"),
        F([("lr", "lr")] + [(c, c.split("/", 1)[1]) for c in lr_group_cols(df)], "lr.png", "Learning-rate schedule", ylabel="lr", logy=True),
    ]
    body = explanation(
        lead="Optimization signals over training. Use the Both/Train/Val toggle on each loss chart.",
        bullets=[
            ("Total / patch / token loss", "total = lambda_patch·patch + lambda_token·token; lower is better."),
            ("Lambda schedule", "loss weights over epochs; lambda_token warms up linearly (shaded band on the loss charts)."),
            ("Logit scale", "learnable CLIP inverse temperature(s), 1/T; rising = lower temperature, sharper confidence."),
            ("Learning rate", "LR schedule on a log axis, including per-parameter-group LRs when enabled."),
        ],
    )
    grid = _grid(figs)
    body += grid if grid else note("No loss/schedule metrics in this run.", missing=True)
    return section("Losses & schedules", "losses-schedules", body)


def render_token_section(df, assets_dir, output_html, be, warmup=0):
    sk = [("token/train_tok_top1", "train top1"), ("token/val_tok_top1", "val top1")]
    for c in dynamic_topk_token_cols(df):
        mode = "train" if "train" in c else "val"
        k = c.rsplit("_top", 1)[-1]
        sk.append((c, f"{mode} top{k}"))
    fig = _fig_tv(df, sk, "token_head.png", "Token-retrieval head accuracy", assets_dir, output_html, ylabel="accuracy", best_epoch=be, warmup=warmup)
    body = explanation(
        "Accuracy of the auxiliary token-retrieval head (fine-grained cell-token alignment within a patch). "
        "Present only when the token loss is enabled (lambda_token > 0). Higher is better."
    )
    body += _grid([fig]) if fig else note("No token-head metrics (token loss likely disabled).", missing=True)
    return section("Token head", "token-head", body)


def render_step_section(df, assets_dir, output_html, be):
    FT = lambda s, fname, title, **kw: _fig_tv(df, s, fname, title, assets_dir, output_html, **kw)
    figs = [
        FT([("step/step_acc/train_acc_i2st", "train i2st"), ("step/step_acc/val_acc_i2st", "val i2st"),
           ("step/step_acc/train_acc_st2i", "train st2i"), ("step/step_acc/val_acc_st2i", "val st2i")],
          "step_acc.png", "Step Top-1 accuracy", ylabel="accuracy", best_epoch=be),
        FT([("step/step_mrr/train_mrr_i2st", "train i2st"), ("step/step_mrr/val_mrr_i2st", "val i2st"),
           ("step/step_mrr/train_mrr_st2i", "train st2i"), ("step/step_mrr/val_mrr_st2i", "val st2i")],
          "step_mrr.png", "Step MRR", ylabel="MRR", best_epoch=be),
        FT([("step/step_sim_i2st/train_pos_sim", "train pos"), ("step/step_sim_i2st/val_pos_sim", "val pos"),
           ("step/step_sim_i2st/train_neg_sim", "train neg"), ("step/step_sim_i2st/val_neg_sim", "val neg"),
           ("step/step_sim_i2st/train_pos_neg_gap", "train gap"), ("step/step_sim_i2st/val_pos_neg_gap", "val gap")],
          "step_sim.png", "Step cosine similarity (i2st)", ylabel="cosine"),
        FT([("step/step_ndcg/train_ndcg_i2st@10", "train i2st"), ("step/step_ndcg/val_ndcg_i2st@10", "val i2st"),
           ("step/step_ndcg/train_ndcg_st2i@10", "train st2i"), ("step/step_ndcg/val_ndcg_st2i@10", "val st2i")],
          "step_ndcg10.png", "Step nDCG@10", ylabel="nDCG", best_epoch=be),
        FT([("step/step_ndcg/train_ndcg_i2st@20", "train i2st"), ("step/step_ndcg/val_ndcg_i2st@20", "val i2st"),
           ("step/step_ndcg/train_ndcg_st2i@20", "train st2i"), ("step/step_ndcg/val_ndcg_st2i@20", "val st2i")],
          "step_ndcg20.png", "Step nDCG@20", ylabel="nDCG"),
        FT([("step/step_ndcg/train_ndcg_i2st@100", "train i2st"), ("step/step_ndcg/val_ndcg_i2st@100", "val i2st"),
           ("step/step_ndcg/train_ndcg_st2i@100", "train st2i"), ("step/step_ndcg/val_ndcg_st2i@100", "val st2i")],
          "step_ndcg100.png", "Step nDCG@100", ylabel="nDCG"),
        FT([("step/step_spearman/train_spearman_i2st", "train i2st"), ("step/step_spearman/val_spearman_i2st", "val i2st"),
           ("step/step_spearman/train_spearman_st2i", "train st2i"), ("step/step_spearman/val_spearman_st2i", "val st2i")],
          "step_spearman.png", "Step Spearman rho", ylabel="rho"),
    ]
    body = explanation(
        lead="Per-step retrieval quality averaged over each epoch, both directions (image→ST and ST→image), "
             "on train and validation batches. Use the Both/Train/Val toggle on each chart.",
        bullets=[
            ("Accuracy / MRR", "rank of the true pair; higher is better."),
            ("nDCG@K / Spearman", "graded ranking vs the soft-similarity target; higher is better."),
            ("Cosine pos / neg / gap", "matched vs hardest-negative similarity; the positive−negative gap should widen."),
        ],
    )
    grid = _grid(figs)
    body += grid if grid else note("No step-level retrieval metrics in this run.", missing=True)
    return section("Step retrieval (train & val)", "step-retrieval", body)


def render_global_section(df, assets_dir, output_html, be):
    parts = present_partitions(df)
    extra = ", plus INT/EXT internal/external folds" if len(parts) > 1 else ""
    body = explanation(
        lead=("Validation-set retrieval over the whole validation pool (not per batch), reported for each "
              f"available data partition (all{extra})."),
        bullets=[
            ("Accuracy@K / MRR", "is the true pair retrieved at rank 1 / within top-K? Higher is better."),
            ("nDCG@K", "graded ranking quality using soft H&E/ST similarity gains; higher is better."),
            ("Cosine pos / neg / gap", "matched vs hardest-negative similarity; the gap should widen."),
            ("Per-slide nDCG@100", "the same ranking quality per validation slide (collapsible; one colour per slide)."),
        ],
    )
    any_fig = False
    for part in parts:
        F = lambda s, fname, title, **kw: _fig(df, s, fname, title, assets_dir, output_html, **kw)
        pfx = f"global/{part}/"
        figs = [
            F([(f"{pfx}global_acc/val_global_acc_i2st_top1", "top1"), (f"{pfx}global_acc/val_global_acc_i2st_top3", "top3"),
               (f"{pfx}global_acc/val_global_acc_i2st_top5", "top5"), (f"{pfx}global_acc/val_global_acc_i2st_top10", "top10")],
              f"global_acc_i2st_{part}.png", f"Global accuracy@K — {part} (i2st)", ylabel="accuracy", best_epoch=be),
            F([(f"{pfx}global_acc/val_global_acc_st2i_top1", "top1"), (f"{pfx}global_acc/val_global_acc_st2i_top3", "top3"),
               (f"{pfx}global_acc/val_global_acc_st2i_top5", "top5"), (f"{pfx}global_acc/val_global_acc_st2i_top10", "top10")],
              f"global_acc_st2i_{part}.png", f"Global accuracy@K — {part} (st2i)", ylabel="accuracy"),
            F([(f"{pfx}global_mrr/val_global_mrr_i2st", "i2st"), (f"{pfx}global_mrr/val_global_mrr_st2i", "st2i")],
              f"global_mrr_{part}.png", f"Global MRR — {part}", ylabel="MRR", best_epoch=be),
            F([(f"{pfx}global_ndcg/val_global_ndcg_i2st_top10", "@10"), (f"{pfx}global_ndcg/val_global_ndcg_i2st_top20", "@20"),
               (f"{pfx}global_ndcg/val_global_ndcg_i2st_top100", "@100")],
              f"global_ndcg_i2st_{part}.png", f"Global nDCG@K — {part} (i2st)", ylabel="nDCG"),
            F([(f"{pfx}global_ndcg/val_global_ndcg_st2i_top10", "@10"), (f"{pfx}global_ndcg/val_global_ndcg_st2i_top20", "@20"),
               (f"{pfx}global_ndcg/val_global_ndcg_st2i_top100", "@100")],
              f"global_ndcg_st2i_{part}.png", f"Global nDCG@K — {part} (st2i)", ylabel="nDCG"),
            F([(f"{pfx}global_sim_i2st/val_global_pos_sim", "pos"), (f"{pfx}global_sim_i2st/val_global_neg_sim", "neg"),
               (f"{pfx}global_sim_i2st/val_global_pos_neg_gap", "gap")],
              f"global_sim_{part}.png", f"Global cosine similarity — {part}", ylabel="cosine"),
        ]
        grid = _grid(figs)
        slide_cols = per_slide_ndcg_cols(df, part)
        slide_block = ""
        if slide_cols:
            sl_series = []
            for c in slide_cols:
                leaf = c.split("/")[-1]
                lab = leaf[4:] if leaf.startswith("val_") else leaf
                sl_series.append((c, lab.replace("_top100", "")))
            sp = save_per_slide_plot(df, sl_series, assets_dir=assets_dir, fname=f"global_slide_ndcg_{part}.png",
                                     title=f"Per-slide nDCG@100 — {part}", ylabel="nDCG")
            if sp is not None:
                slide_block = details(f"Per-slide nDCG@100 — {part} ({len(slide_cols)} series)",
                                      image_html(sp, output_html, caption=f"Per-slide nDCG@100 — {part}"))
        if grid or slide_block:
            any_fig = True
            if len(parts) > 1:
                body += f"<h3>Partition: {e(part)}</h3>"
            body += (grid or "") + slide_block
    if not any_fig:
        body += note("No global validation retrieval metrics in this run (validation may not have run).", missing=True)
    return section("Global - Performance (Retrieval & Ranking)", "global-performance", body)


def render_within_slide_section(df, assets_dir, output_html, be):
    body = explanation(
        lead=("Batch-controlled retrieval: for each validation slide the gallery is restricted to "
              "that slide's own patches, then scores are macro-averaged across slides. This holds "
              "slide / staining / scanner constant, isolating within-slide molecular discrimination "
              "with no batch shortcut (unlike INT, which pools many slides into one gallery). EXT is "
              "exactly this for one slide; here it is averaged over every validation slide."),
        bullets=[
            ("Accuracy@K / MRR", "macro-averaged within-slide retrieval quality; higher is better."),
            ("Slides used / skipped", "slides with too few patches (< min_patches, or ≤ max top-k) are skipped and excluded from the average."),
            ("Per-slide", "the same scores per validation slide (collapsible; one colour per slide)."),
        ],
    )
    if not within_slide_present(df):
        body += note("No within-slide retrieval metrics in this run (not logged, or validation did not run).", missing=True)
        return section("Global - Within-slide retrieval (batch-controlled)", "global-within-slide", body)

    F = lambda s, fname, title, **kw: _fig(df, s, fname, title, assets_dir, output_html, **kw)
    wpfx = "global/within_slide/"
    figs = [
        F([(f"{wpfx}global_acc/val_within_slide_acc_i2st_top1", "top1"), (f"{wpfx}global_acc/val_within_slide_acc_i2st_top3", "top3"),
           (f"{wpfx}global_acc/val_within_slide_acc_i2st_top5", "top5"), (f"{wpfx}global_acc/val_within_slide_acc_i2st_top10", "top10")],
          "within_slide_acc_i2st.png", "Within-slide accuracy@K (i2st)", ylabel="accuracy", best_epoch=be),
        F([(f"{wpfx}global_acc/val_within_slide_acc_st2i_top1", "top1"), (f"{wpfx}global_acc/val_within_slide_acc_st2i_top3", "top3"),
           (f"{wpfx}global_acc/val_within_slide_acc_st2i_top5", "top5"), (f"{wpfx}global_acc/val_within_slide_acc_st2i_top10", "top10")],
          "within_slide_acc_st2i.png", "Within-slide accuracy@K (st2i)", ylabel="accuracy"),
        F([(f"{wpfx}global_mrr/val_within_slide_mrr_i2st", "i2st"), (f"{wpfx}global_mrr/val_within_slide_mrr_st2i", "st2i")],
          "within_slide_mrr.png", "Within-slide MRR", ylabel="MRR", best_epoch=be),
        F([(f"{wpfx}global_count/val_within_slide_n_slides_used", "used"), (f"{wpfx}global_count/val_within_slide_n_slides_skipped", "skipped")],
          "within_slide_nslides.png", "Within-slide slides used / skipped", ylabel="n slides"),
    ]
    grid = _grid(figs)

    # Per-slide panels (collapsible): acc@5 (i2st) and MRR (i2st), one series per slide.
    slide_block = ""
    for suffix, lab_title, ylab in (("acc_i2st_top5", "Per-slide within-slide accuracy@5 (i2st)", "accuracy"),
                                    ("mrr_i2st", "Per-slide within-slide MRR (i2st)", "MRR")):
        cols = within_slide_per_slide_cols(df, suffix)
        if not cols:
            continue
        sl_series = []
        for c in cols:
            leaf = c.split("/")[-1]                       # val_within_slide_{sid}_{suffix}
            sid = leaf[len("val_within_slide_"):]
            if sid.endswith("_" + suffix):
                sid = sid[: -(len(suffix) + 1)]
            sl_series.append((c, sid))
        sp = save_per_slide_plot(df, sl_series, assets_dir=assets_dir,
                                 fname=f"within_slide_{suffix}.png", title=lab_title, ylabel=ylab)
        if sp is not None:
            slide_block += details(f"{lab_title} ({len(cols)} slides)",
                                   image_html(sp, output_html, caption=lab_title))

    body += (grid or "") + slide_block
    if not (grid or slide_block):
        body += note("No within-slide retrieval figures could be drawn.", missing=True)
    return section("Global - Within-slide retrieval (batch-controlled)", "global-within-slide", body)


def render_geometry_health_section(df, assets_dir, output_html, be):
    parts = present_partitions(df)
    body = explanation(
        lead="Geometry of the shared embedding space on the validation set.",
        bullets=[
            ("Alignment (MSE)", "mean 2(1−cos) over true pairs; lower is better (pairs co-located)."),
            ("Uniformity", "spread on the unit sphere; more negative is better (the dotted line marks 0 = collapse)."),
            ("MMD (RBF)", "distribution gap between the H&E and ST clouds; lower is better."),
        ],
    )
    any_fig = False
    for part in parts:
        F = lambda s, fname, title, **kw: _fig(df, s, fname, title, assets_dir, output_html, **kw)
        pfx = f"global/{part}/global_geom/"
        figs = [
            F([(f"{pfx}val_global_align_mse", "alignment MSE")], f"geom_align_{part}.png", f"Alignment MSE — {part}", ylabel="MSE", best_epoch=be),
            F([(f"{pfx}val_global_uni_img", "img"), (f"{pfx}val_global_uni_st", "st"), (f"{pfx}val_global_uni_both", "both")],
              f"geom_uni_{part}.png", f"Uniformity — {part}", ylabel="uniformity", hline=0.0),
            F([(f"{pfx}val_global_mmd_rbf", "MMD rbf")], f"geom_mmd_{part}.png", f"MMD (RBF) — {part}", ylabel="MMD"),
        ]
        grid = _grid(figs)
        if grid:
            any_fig = True
            if len(parts) > 1:
                body += f"<h3>Partition: {e(part)}</h3>"
            body += grid
    if not any_fig:
        body += note("No geometry diagnostics in this run.", missing=True)
    return section("Global - Geometry health", "global-geometry-health", body)


def render_coherence_section(df, assets_dir, output_html):
    parts = present_partitions(df)
    body = explanation(
        lead="Structure- and topology-preservation diagnostics on the validation set.",
        bullets=[
            ("CKA", "representation similarity; context-dependent, not a quality score. H&E-Novae pre "
                    "(= val_global_cka_he_st_pre): linear CKA between the pre-CLIP H&E and Novae teacher features; "
                    "H&E-scConcept pre: the same with the patch-pooled scConcept features."),
            ("kNN agreement@10", "do image–image neighbours survive cross-modal retrieval? Higher is better."),
        ],
    )
    any_fig = False
    for part in parts:
        F = lambda s, fname, title, **kw: _fig(df, s, fname, title, assets_dir, output_html, **kw)
        pfx = f"global/{part}/"
        figs = [
            F([(f"{pfx}global_cka/val_global_cka_he", "CKA H&E"), (f"{pfx}global_cka/val_global_cka_st", "CKA ST"),
               (f"{pfx}global_cka/val_global_cka_he_st_pre", "CKA H&E-Novae pre"),
               (f"{pfx}global_cka/val_global_cka_he_scconcept_pre", "CKA H&E-scConcept pre")],
              f"cka_{part}.png", f"CKA — {part}", ylabel="CKA"),
            F([(f"{pfx}global_knn_agree_imgs/overlap@10_i", "overlap"), (f"{pfx}global_knn_agree_imgs/jaccard@10_i", "jaccard")],
              f"knn_{part}.png", f"kNN agreement@10 — {part}", ylabel="agreement"),
        ]
        grid = _grid(figs)
        if grid:
            any_fig = True
            if len(parts) > 1:
                body += f"<h3>Partition: {e(part)}</h3>"
            body += grid
    if not any_fig:
        body += note("No coherence diagnostics in this run.", missing=True)
    return section("Global - Coherence (Structure & Topology)", "global-coherence", body)


def render_trustworthiness_section(df, assets_dir, output_html):
    parts = present_partitions(df)
    body = explanation(
        lead="Retrieval-pathology and confidence diagnostics on the validation set.",
        bullets=[
            ("Hubness (Gini / Top-1%)", "concentration of retrievals on a few 'hub' items; lower is better."),
            ("Spectrum (EffRank / PartRatio)", "soft dimensionality of the embedding cloud; higher is better (avoid collapse)."),
            ("ECE (calibration)", "expected calibration error (15 bins); lower is better."),
        ],
    )
    any_fig = False
    for part in parts:
        F = lambda s, fname, title, **kw: _fig(df, s, fname, title, assets_dir, output_html, **kw)
        pfx = f"global/{part}/"
        gpfx = f"global/{part}/global_geom/"
        figs = [
            F([(f"{pfx}global_hub/val_gini_i2st@10", "i2st"), (f"{pfx}global_hub/val_gini_st2i@10", "st2i")],
              f"hub_gini_{part}.png", f"Hubness Gini@10 — {part}", ylabel="Gini"),
            F([(f"{pfx}global_hub/val_top1p_i2st@10", "i2st"), (f"{pfx}global_hub/val_top1p_st2i@10", "st2i")],
              f"hub_top1p_{part}.png", f"Hubness Top-1% mass@10 — {part}", ylabel="mass"),
            F([(f"{gpfx}val_global_effrank_img", "img"), (f"{gpfx}val_global_effrank_st", "st"), (f"{gpfx}val_global_effrank_both", "both")],
              f"geom_effrank_{part}.png", f"Spectrum (EffRank) — {part}", ylabel="eff. rank"),
            F([(f"{gpfx}val_global_partratio_img", "img"), (f"{gpfx}val_global_partratio_st", "st"), (f"{gpfx}val_global_partratio_both", "both")],
              f"geom_partratio_{part}.png", f"Spectrum (PartRatio) — {part}", ylabel="PR"),
            F([(f"{pfx}global_calib/val_ece_i2st_bins15", "i2st"), (f"{pfx}global_calib/val_ece_st2i_bins15", "st2i")],
              f"ece_{part}.png", f"ECE (calibration, 15 bins) — {part}", ylabel="ECE"),
        ]
        grid = _grid(figs)
        if grid:
            any_fig = True
            if len(parts) > 1:
                body += f"<h3>Partition: {e(part)}</h3>"
            body += grid
    if not any_fig:
        body += note("No trustworthiness diagnostics in this run.", missing=True)
    return section("Global - Trustworthiness", "global-trustworthiness", body)


def render_dataquality_section(df, assets_dir, output_html):
    F = lambda s, fname, title, **kw: _fig(df, s, fname, title, assets_dir, output_html, **kw)
    figs = [
        F([("data_quality/train_dummy_over_total", "dummy fraction")], "dq_frac.png", "Dummy ST fraction (train)", ylabel="fraction"),
        F([("data_quality/train_dummy_count", "dummy count")], "dq_count.png", "Dummy ST count (train)", ylabel="count"),
    ]
    body = explanation(
        "Fraction and count of dummy/placeholder ST graphs seen during training. High or rising values mean "
        "many patches lacked real ST supervision."
    )
    grid = _grid(figs)
    body += grid if grid else note("No data-quality metrics in this run.", missing=True)
    return section("Data quality", "data-quality", body)


def render_earlystopping_section(df, assets_dir, output_html, be):
    F = lambda s, fname, title, **kw: _fig(df, s, fname, title, assets_dir, output_html, **kw)
    figs = [
        F([("early_stopping/monitor_value", "monitor"), ("early_stopping/best_value", "best")],
          "es_monitor.png", "Early-stopping monitor", ylabel="value", best_epoch=be),
        F([("early_stopping/patience_counter", "patience")], "es_patience.png", "Early-stopping patience counter", ylabel="patience"),
    ]
    body = explanation(
        "The metric tracked for early stopping (its value vs. the best-so-far) and the patience counter. "
        "The dashed line marks the selected best epoch."
    )
    grid = _grid(figs)
    body += grid if grid else note("No early-stopping metrics in this run.", missing=True)
    return section("Early stopping", "early-stopping", body)


def render_metrics_table(df):
    if df is None or df.empty:
        return section("Per-epoch table", "metrics-table", note("No metrics rows.", missing=True))
    show = df.copy()
    drop = [c for c in show.columns if "/global_slide_ndcg/" in c]
    if drop:
        show = show.drop(columns=drop)
    show = show.dropna(axis=1, how="all")
    if "epoch" in show.columns:
        cols = ["epoch"] + [c for c in show.columns if c != "epoch"]
        show = show[cols]
    body = explanation(
        "Every logged metric, one row per epoch. Click a header to sort; type to filter rows. Per-slide nDCG "
        "columns are omitted here (shown as plots in Global retrieval)."
    )
    body += html_table(show, "metrics-per-epoch", searchable=True, small=True)
    return section("Per-epoch table", "metrics-table", body)


def render_glossary():
    body = variable_definitions(GLOSSARY, title="Show metric glossary", open_=False)
    body += note("Short definitions of the metrics in this report.")
    return section("Metric glossary", "glossary", body)


def render_repro(args, df, output_html):
    eps = pd.to_numeric(df["epoch"], errors="coerce") if (df is not None and not df.empty and "epoch" in df.columns) else None
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "training_id": args.training_id,
        "training_num": training_num_from_id(args.training_id),
        "jsonl_path": str(jsonl_path(args.training_id)),
        "yaml_path": str(yaml_path(args.training_id)),
        "output_html": str(output_html),
        "standalone": bool(is_standalone()),
        "n_epochs": int(df.shape[0]) if (df is not None and not df.empty) else 0,
        "epoch_range": ([int(eps.min()), int(eps.max())] if eps is not None and eps.notna().any() else None),
        "n_val_epochs": (int(pd.to_numeric(df["data/val_epoch_ran"], errors="coerce").fillna(0).sum())
                         if (df is not None and not df.empty and "data/val_epoch_ran" in df.columns) else None),
        "best_epoch": best_epoch_from_df(df),
        "partitions_present": present_partitions(df),
        "n_metric_columns": int(df.shape[1]) if (df is not None and not df.empty) else 0,
        "wandb_clip_url": args.wandb_clip_url,
    }
    body = explanation("Inputs and resolved paths for this report, for debugging and provenance.")
    body += details("Show reproducibility payload",
                    f'<pre class="json-block">{e(json.dumps(payload, indent=2, sort_keys=True, default=str))}</pre>')
    return section("Reproducibility", "reproducibility", body)


# -----------------------------------------------------------------------------
# Page assembly / writing
# -----------------------------------------------------------------------------


def build_html(args, cfg, df, output_html, assets_dir):
    be = best_epoch_from_df(df)
    wu = warmup_epochs(cfg)
    main_parts = [
        render_run_summary(cfg),
        render_scorecards(df, be, cfg),
        render_losses_section(df, assets_dir, output_html, be, wu),
        render_token_section(df, assets_dir, output_html, be, wu),
        render_step_section(df, assets_dir, output_html, be),
        render_global_section(df, assets_dir, output_html, be),
        render_within_slide_section(df, assets_dir, output_html, be),
        render_geometry_health_section(df, assets_dir, output_html, be),
        render_coherence_section(df, assets_dir, output_html),
        render_trustworthiness_section(df, assets_dir, output_html),
        render_dataquality_section(df, assets_dir, output_html),
        render_earlystopping_section(df, assets_dir, output_html, be),
        render_metrics_table(df),
        render_glossary(),
        render_repro(args, df, output_html),
    ]
    empty_banner = ""
    if df is None or df.empty:
        empty_banner = callout(
            "bad",
            f"No usable rows found in {jsonl_path(args.training_id)} — the training metrics log is missing or "
            "empty. The report structure is shown below but contains no data.",
            label="No data",
        )
    title = f"HistoST training report — {args.training_id}"
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{e(title)}</title>
  <style>{css()}</style>
</head>
<body>
{render_header(args, cfg, df)}
<div class="layout">
{make_nav()}
<main>
{empty_banner}
{''.join(main_parts)}
</main>
</div>
<script>{js()}</script>
</body>
</html>
"""


def write_report(args):
    tid = args.training_id
    df = load_metrics_df(jsonl_path(tid))
    cfg = load_config(yaml_path(tid))
    base_output = output_path_for(tid)
    output_html = standalone_output_path(base_output) if is_standalone() else base_output
    output_html.parent.mkdir(parents=True, exist_ok=True)
    assets_dir = assets_dir_for(base_output)
    html = build_html(args, cfg, df, output_html, assets_dir)
    output_html.write_text(html, encoding="utf-8")
    logger.info("Wrote training report: %s", output_html)
    if should_generate_md():
        write_markdown_copy(
            html,
            basename=f"training_dynamics_t{training_num_from_id(tid)}",
            source_html_path=output_html,
        )
    return output_html


def build_arg_parser():
    p = argparse.ArgumentParser(
        description="Generate an HTML report of HistoST CLIP training dynamics for one training_id."
    )
    p.add_argument("--training_id", type=str, required=True,
                   help="HistoST training id, e.g. training_1. Used to locate the run folder.")
    p.add_argument("--standalone", action="store_true",
                   help="Embed every figure as base64 and write a self-contained *_standalone.html.")
    p.add_argument("--generate_md", action="store_true",
                   help=("Also write a Markdown copy of the report (tables and numbers, no figures) "
                         "to the in-repo access_reports/ folder."))
    p.add_argument("--wandb_clip_url", type=str, default="",
                   help="Optional W&B project URL linked from the report header.")
    return p


def main(args):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    set_standalone(args.standalone)
    set_generate_md(args.generate_md)
    out = write_report(args)
    print(f"Wrote training report: {out}")
    return out


if __name__ == "__main__":
    main(build_arg_parser().parse_args())
