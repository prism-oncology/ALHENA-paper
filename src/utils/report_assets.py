"""
Helpers to embed the assets of the HistoST HTML reports (standalone mode) and to write Markdown copies of the reports.
"""

import base64
import logging
import os
import re
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

# MIME type per file extension, for the ``data:<mime>;base64,`` prefix.
_MIME_BY_SUFFIX = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".svg": "image/svg+xml",
    ".webp": "image/webp",
    ".pdf": "application/pdf",
    ".csv": "text/csv",
    ".tsv": "text/tab-separated-values",
    ".json": "application/json",
    ".txt": "text/plain",
}

# Warn once when the raw (pre-base64) embedded payload exceeds this size
_SIZE_WARN_THRESHOLD_BYTES = 20 * 1024 * 1024  # 20 MB

# --- process-global state (set once per run by the entry point) ---------------
_STANDALONE = False
_embedded_bytes = 0
_size_warned = False


def set_standalone(value: bool) -> None:
    """Enable or disable self-contained (base64 ``data:`` URI) asset embedding (``--standalone``); resets the size accounting."""
    global _STANDALONE, _embedded_bytes, _size_warned
    _STANDALONE = bool(value)
    _embedded_bytes = 0
    _size_warned = False
    if _STANDALONE:
        logger.info("Standalone report mode enabled: assets will be embedded as base64 data URIs.")


def is_standalone() -> bool:
    """Return whether standalone (embedded-asset) mode is active for this run."""
    return _STANDALONE


def rel_link(target: Path, html_path: Path) -> str:
    """Relative POSIX path from the report directory to ``target``."""
    try:
        return Path(os.path.relpath(target, start=Path(html_path).parent)).as_posix()
    except Exception:
        return Path(target).as_posix()


def _data_uri(path: Path) -> str:
    """Base64 ``data:`` URI for ``path``; bytes are embedded verbatim (no re-encode)."""
    path = Path(path)
    mime = _MIME_BY_SUFFIX.get(path.suffix.lower(), "application/octet-stream")
    raw = path.read_bytes()

    global _embedded_bytes, _size_warned
    _embedded_bytes += len(raw)
    if _embedded_bytes > _SIZE_WARN_THRESHOLD_BYTES and not _size_warned:
        _size_warned = True
        logger.warning(
            "Standalone report has embedded ~%.1f MB of assets so far (~33%% more after base64).",
            _embedded_bytes / 1024 / 1024,
        )

    encoded = base64.b64encode(raw).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def asset_href(target: Path, html_path: Path) -> str:
    """``src``/``href`` of a local asset: a ``data:`` URI in standalone mode (if readable), else a relative link."""
    target = Path(target)
    if _STANDALONE:
        try:
            if target.is_file():
                return _data_uri(target)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Could not embed %s inline (%s); falling back to a relative link.", target, exc)
    return rel_link(target, html_path)


# Aliases of asset_href
def image_src(target: Path, html_path: Path) -> str:
    """``src`` value for an ``<img>`` (and any link wrapping it)."""
    return asset_href(target, html_path)


def download_href(target: Path, html_path: Path) -> str:
    """``href`` value for a link to a small data file (CSV / JSON / TXT)."""
    return asset_href(target, html_path)


def standalone_output_path(output_html: Path) -> Path:
    """Sibling ``*_standalone.html`` path, so the self-contained file never clobbers the normal report."""
    output_html = Path(output_html)
    if output_html.stem.endswith("_standalone"):
        return output_html
    return output_html.with_name(f"{output_html.stem}_standalone{output_html.suffix}")


# =============================================================================
# Plain-text Markdown copies (the ``--generate_md`` flag)
# =============================================================================

_GENERATE_MD = False


def set_generate_md(value: bool) -> None:
    """Enable or disable writing a plain-text Markdown copy to ``access_reports/`` (``--generate_md``)."""
    global _GENERATE_MD
    _GENERATE_MD = bool(value)
    if _GENERATE_MD:
        logger.info(
            "Markdown report mode enabled: a plain-text .md rendering will be written to access_reports/."
        )


def should_generate_md() -> bool:
    """Return whether a plain-text Markdown copy should be written for this run."""
    return _GENERATE_MD


def access_reports_dir() -> Path:
    """``access_reports/`` folder at the repo root (destination of the Markdown copies)."""
    return Path(__file__).resolve().parents[2] / "access_reports"


def _md_collapse_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _md_inline(node) -> str:
    """Inline Markdown for a node and its descendants (whitespace collapsed later, by the caller)."""
    from bs4 import NavigableString, Tag

    if isinstance(node, NavigableString):
        return str(node)
    if not isinstance(node, Tag):
        return ""
    name = (node.name or "").lower()
    if name == "br":
        return " "
    if name == "img":
        return ""  # images are dropped from the Markdown
    inner = "".join(_md_inline(child) for child in node.children)
    stripped = inner.strip()
    if not stripped:
        return inner
    if name in ("strong", "b"):
        return f"**{stripped}**"
    if name in ("em", "i"):
        return f"*{stripped}*"
    if name == "code":
        return f"`{stripped}`"
    if name == "a":
        href = (node.get("href") or "").strip()
        if href and not href.startswith("data:"):
            return f"[{stripped}]({href})"
        return stripped
    return inner


def _md_inline_clean(node) -> str:
    return _md_collapse_ws(_md_inline(node))


def _md_figure_caption(node) -> str:
    """Caption text for a ``<figure>``/``<img>`` (figcaption, else img alt/title), collapsed."""
    cap = ""
    if node.name == "figure":
        figcap = node.find("figcaption")
        if figcap is not None:
            cap = figcap.get_text(" ")
    if not cap:
        img = node if node.name == "img" else node.find("img")
        if img is not None:
            cap = img.get("alt") or img.get("title") or ""
    return _md_collapse_ws(cap)


def _md_render_table(table, out: List[str]) -> None:
    grid = []
    for tr in table.find_all("tr"):
        cells = tr.find_all(["th", "td"])
        if cells:
            grid.append([_md_inline_clean(c).replace("|", "\\|") for c in cells])
    grid = [row for row in grid if any(cell for cell in row)]
    if not grid:
        return
    ncol = max(len(row) for row in grid)
    for row in grid:
        row.extend([""] * (ncol - len(row)))
    out.append("| " + " | ".join(grid[0]) + " |")
    out.append("| " + " | ".join(["---"] * ncol) + " |")
    for row in grid[1:]:
        out.append("| " + " | ".join(row) + " |")


def _md_render_list(lst, out: List[str], ordered: bool = False, depth: int = 0) -> None:
    index = 1
    for li in lst.find_all("li", recursive=False):
        nested = list(li.find_all(["ul", "ol"], recursive=False))
        for sub in nested:
            sub.extract()
        marker = f"{index}." if ordered else "-"
        text = _md_inline_clean(li)
        if text:
            out.append(f"{'  ' * depth}{marker} {text}")
        for sub in nested:
            _md_render_list(sub, out, ordered=(sub.name == "ol"), depth=depth + 1)
        index += 1


def _md_render_blocks(node, out: List[str]) -> None:
    from bs4 import NavigableString, Tag

    for child in node.children:
        if isinstance(child, NavigableString):
            text = _md_collapse_ws(str(child))
            if text:
                out.append(text)
                out.append("")
            continue
        if not isinstance(child, Tag):
            continue
        name = (child.name or "").lower()
        if name in ("script", "style", "head", "nav"):
            continue
        if "md-skip" in (child.get("class") or []):
            # Elements with class md-skip are excluded from the Markdown
            continue
        if name in ("h1", "h2", "h3", "h4", "h5", "h6"):
            out.append(f"{'#' * int(name[1])} {_md_inline_clean(child)}")
            out.append("")
        elif name == "p":
            text = _md_inline_clean(child)
            if text:
                out.append(text)
                out.append("")
        elif name in ("ul", "ol"):
            _md_render_list(child, out, ordered=(name == "ol"))
            out.append("")
        elif name == "table":
            _md_render_table(child, out)
            out.append("")
        elif name in ("figure", "img"):
            caption = _md_figure_caption(child)
            if caption:
                out.append(f"*(figure omitted: {caption})*")
                out.append("")
        elif name == "pre":
            out.append("```")
            out.append(child.get_text("\n").rstrip())
            out.append("```")
            out.append("")
        elif name == "blockquote":
            text = _md_inline_clean(child)
            if text:
                out.append(f"> {text}")
                out.append("")
        elif name == "hr":
            out.append("---")
            out.append("")
        elif name == "details":
            summary = child.find("summary", recursive=False)
            if summary is not None:
                label = _md_inline_clean(summary)
                if label:
                    out.append(f"**{label}**")
                    out.append("")
                summary.extract()
            _md_render_blocks(child, out)
        else:
            # generic container (div / section / span / header / main / article / ...): recurse
            _md_render_blocks(child, out)


def html_to_markdown(html: str) -> str:
    """Convert a report HTML string to Markdown: tables, headings, paragraphs, lists and links are kept; scripts,
    styles, images (a captioned figure leaves a one-line note) and ``md-skip`` elements are dropped. Requires ``bs4``.
    """
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "head", "nav"]):
        tag.decompose()
    root = soup.body or soup
    out: List[str] = []
    _md_render_blocks(root, out)
    md = "\n".join(out)
    md = re.sub(r"\n{3,}", "\n\n", md).strip()
    return md + "\n" if md else md


def write_markdown_copy(html: str, *, basename: str, source_html_path=None) -> Optional[Path]:
    """Write a plain-text Markdown copy of ``html`` to ``access_reports/{basename}.md`` (if ``--generate_md`` is set).

    Returns the path, or ``None`` if not written (failures are logged, not raised).
    """
    if not should_generate_md():
        return None
    # Sanitize the file name (it stays inside access_reports/)
    safe_basename = re.sub(r"[^A-Za-z0-9._-]+", "_", basename).strip("._-") or "report"
    try:
        body = html_to_markdown(html)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Could not build Markdown copy '%s' (%s); skipping.", safe_basename, exc)
        return None
    try:
        out_dir = access_reports_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{safe_basename}.md"
        front = ["---", f"report: {safe_basename}"]
        if source_html_path is not None:
            front.append(f"source_html: {Path(source_html_path)}")
        front.append("note: plain-text Markdown rendering of a HistoST report; figures omitted.")
        front.append("---")
        out_path.write_text("\n".join(front) + "\n\n" + body, encoding="utf-8")
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Could not write Markdown copy '%s' (%s); skipping.", safe_basename, exc)
        return None
    logger.info("Wrote Markdown report: %s", out_path)
    return out_path
