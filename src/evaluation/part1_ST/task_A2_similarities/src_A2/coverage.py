"""Task A.2 reference-coverage contract: every scorable GT class of a slide must be covered by its tissue reference."""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np


logger = logging.getLogger(__name__)


# Defaults of the references.coverage block (keep in sync with protocol.DEFAULTS and the _1_setup.py template)
COVERAGE_DEFAULTS: Dict[str, Any] = {
    "enabled": True,
    # Reference cells required for every scorable GT class present on the slide
    "min_ref_cells_per_gt_class": 200,
    # Covered, scorable GT cells a slide must retain to be scored
    "min_scored_cells_per_slide": 200,
    # Check that the run-time (_3) per-cell coverage equals the _2 coverage
    "require_runtime_match": True,
    # {slide_id: reason}: slides allowed to violate the floors above (reason required, logged on every run)
    "declared_exceptions": {},
}

STAGE_FREEZE = "_2_freeze"
STAGE_RUN = "_3_run"


class CoverageContractError(RuntimeError):
    """Raised when a slide violates the A.2 reference-coverage contract."""


@dataclass(frozen=True)
class ClassCoverage:
    """Coverage of one classification-scorable GT class on one slide."""

    label: str
    n_gt_cells: int
    n_ref_cells: int

    @property
    def covered(self) -> bool:
        return self.n_ref_cells > 0

    def meets_floor(self, min_ref_cells: int) -> bool:
        return self.n_ref_cells >= int(min_ref_cells)


@dataclass(frozen=True)
class SlideCoverageReport:
    """Per-slide coverage accounting + the contract verdict."""

    slide_id: str
    tissue: str
    reference_stem: str
    label_granularity: str
    stage: str
    classes: List[ClassCoverage]
    n_scorable_gt_cells: int
    n_covered_gt_cells: int
    violations: List[str] = field(default_factory=list)
    declared_reason: Optional[str] = None

    @property
    def n_gt_classes(self) -> int:
        return len(self.classes)

    @property
    def n_covered_classes(self) -> int:
        return sum(1 for c in self.classes if c.covered)

    @property
    def covered_cell_fraction(self) -> float:
        if self.n_scorable_gt_cells <= 0:
            return float("nan")
        return float(self.n_covered_gt_cells) / float(self.n_scorable_gt_cells)

    @property
    def ok(self) -> bool:
        return not self.violations

    @property
    def active_exception_reason(self) -> Optional[str]:
        """The declared reason if the slide violates the contract, else None."""
        return self.declared_reason if self.violations else None

    def as_dict(self) -> Dict[str, Any]:
        """Serializable record for the reference audit / run manifest."""
        return {
            "slide_id": self.slide_id,
            "tissue": self.tissue,
            "reference_stem": self.reference_stem,
            "label_granularity": self.label_granularity,
            "stage": self.stage,
            "n_gt_classes": self.n_gt_classes,
            "n_covered_classes": self.n_covered_classes,
            "n_scorable_gt_cells": int(self.n_scorable_gt_cells),
            "n_covered_gt_cells": int(self.n_covered_gt_cells),
            "covered_cell_fraction": self.covered_cell_fraction,
            "per_class": [
                {
                    "label": c.label,
                    "n_gt_cells": int(c.n_gt_cells),
                    "n_ref_cells": int(c.n_ref_cells),
                    "covered": bool(c.covered),
                }
                for c in self.classes
            ],
            "violations": list(self.violations),
            "declared_exception_reason": self.declared_reason,
            "contract_satisfied": bool(self.ok),
        }


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
def coverage_config(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    """Read the `references.coverage` block, filling any missing key from COVERAGE_DEFAULTS."""
    raw = dict(cfg.get("references", {}).get("coverage", {}) or {})
    out = dict(COVERAGE_DEFAULTS)
    out.update(raw)
    out["declared_exceptions"] = {
        str(k): str(v) for k, v in dict(out.get("declared_exceptions") or {}).items()
    }
    return out


def validate_coverage_config(cfg: Mapping[str, Any]) -> None:
    """Validate the references.coverage block (raises ValueError)."""
    cov = coverage_config(cfg)

    if not isinstance(cov["enabled"], bool):
        raise ValueError("references.coverage.enabled must be a boolean.")
    if not isinstance(cov["require_runtime_match"], bool):
        raise ValueError("references.coverage.require_runtime_match must be a boolean.")

    for key in ("min_ref_cells_per_gt_class", "min_scored_cells_per_slide"):
        value = int(cov[key])
        if value < 1:
            raise ValueError(f"references.coverage.{key} must be >= 1 (got {value}).")

    exceptions = cov["declared_exceptions"]
    if not isinstance(exceptions, Mapping):
        raise ValueError(
            "references.coverage.declared_exceptions must be a mapping {slide_id: reason}."
        )
    empty = sorted(sid for sid, reason in exceptions.items() if not str(reason).strip())
    if empty:
        raise ValueError(
            "references.coverage.declared_exceptions entries need a non-empty written reason. "
            f"Missing a reason for: {empty}."
        )

    if not cov["enabled"]:
        logger.warning(
            "references.coverage.enabled is False: the A-2 reference-coverage contract is "
            "disabled. Slides with uncovered GT classes are scored on a reduced class axis."
        )


# -----------------------------------------------------------------------------
# Per-slide coverage report
# -----------------------------------------------------------------------------
def build_slide_coverage_report(
    *,
    slide_id: str,
    tissue: str,
    reference_stem: str,
    label_granularity: str,
    stage: str,
    gt_labels: Sequence[str],
    gt_scorable_mask: Sequence[bool],
    ref_label_counts: Mapping[str, int],
    cfg: Mapping[str, Any],
) -> SlideCoverageReport:
    """
    Build the coverage report of one slide and check the contract floors.
    ref_label_counts is {label: n_cells} of the reference used at this stage (_2: filtered, _3: also after projected-embedding QC).
    """
    cov = coverage_config(cfg)
    min_ref = int(cov["min_ref_cells_per_gt_class"])
    min_cells = int(cov["min_scored_cells_per_slide"])

    labels = np.asarray(gt_labels, dtype=object).reshape(-1)
    scorable = np.asarray(gt_scorable_mask, dtype=bool).reshape(-1)
    if labels.shape[0] != scorable.shape[0]:
        raise ValueError(
            f"gt_labels ({labels.shape[0]}) and gt_scorable_mask ({scorable.shape[0]}) "
            f"differ in length for slide '{slide_id}'."
        )

    scorable_labels = labels[scorable]
    ref_counts = {str(k): int(v) for k, v in ref_label_counts.items()}

    uniq, counts = (
        np.unique(scorable_labels.astype(str), return_counts=True)
        if scorable_labels.size
        else (np.empty(0, dtype=str), np.empty(0, dtype=int))
    )
    classes = [
        ClassCoverage(label=str(u), n_gt_cells=int(n), n_ref_cells=int(ref_counts.get(str(u), 0)))
        for u, n in sorted(zip(uniq, counts), key=lambda kv: -int(kv[1]))
    ]

    n_scorable = int(scorable_labels.size)
    n_covered = int(sum(c.n_gt_cells for c in classes if c.covered))

    violations: List[str] = []
    below = [c for c in classes if not c.meets_floor(min_ref)]
    absent = [c for c in below if not c.covered]
    thin = [c for c in below if c.covered]
    if absent:
        violations.append(
            "GT class(es) absent from the reference: "
            + ", ".join(f"{c.label} ({c.n_gt_cells:,} GT cells, 0 reference cells)" for c in absent)
        )
    if thin:
        violations.append(
            f"GT class(es) with reference support < {min_ref}: "
            + ", ".join(
                f"{c.label} ({c.n_gt_cells:,} GT cells, {c.n_ref_cells:,} reference cells)"
                for c in thin
            )
        )
    if n_covered < min_cells:
        violations.append(
            f"only {n_covered:,} covered scorable GT cell(s) remain "
            f"(min_scored_cells_per_slide={min_cells:,})"
        )

    return SlideCoverageReport(
        slide_id=str(slide_id),
        tissue=str(tissue),
        reference_stem=str(reference_stem),
        label_granularity=str(label_granularity),
        stage=str(stage),
        classes=classes,
        n_scorable_gt_cells=n_scorable,
        n_covered_gt_cells=n_covered,
        violations=violations,
        declared_reason=cov["declared_exceptions"].get(str(slide_id)),
    )


# -----------------------------------------------------------------------------
# Enforcement
# -----------------------------------------------------------------------------
def _format_report(report: SlideCoverageReport) -> str:
    table = "\n".join(
        f"      {c.label:<28} GT {c.n_gt_cells:>9,}   reference {c.n_ref_cells:>9,}"
        + ("" if c.covered else "   <-- absent")
        for c in report.classes
    )
    return (
        f"  slide '{report.slide_id}' (tissue={report.tissue}, "
        f"reference='{report.reference_stem}', granularity={report.label_granularity})\n"
        f"    covered {report.n_covered_classes}/{report.n_gt_classes} GT classes, "
        f"{report.n_covered_gt_cells:,}/{report.n_scorable_gt_cells:,} scorable GT cells "
        f"({100.0 * report.covered_cell_fraction:.1f}%)\n"
        f"{table}\n"
        + "".join(f"    VIOLATION: {v}\n" for v in report.violations)
    )


def assert_coverage_contract(
    reports: Sequence[SlideCoverageReport],
    cfg: Mapping[str, Any],
    *,
    stage: str,
    log: Optional[logging.Logger] = None,
    check_declared_bookkeeping: bool = True,
) -> None:
    """
    Check the contract over the slide reports. Raises CoverageContractError on any violating slide not in
    declared_exceptions (declared slides are logged as warnings).
    check_declared_bookkeeping: warn about declared slides absent from reports (only if reports cover all slides).
    """
    log = log or logger
    cov = coverage_config(cfg)
    if not cov["enabled"]:
        log.warning(
            "[%s] A-2 reference-coverage contract is disabled (references.coverage.enabled=false); "
            "skipping %d slide check(s).",
            stage,
            len(reports),
        )
        return

    failing = [r for r in reports if not r.ok and r.declared_reason is None]
    declared = [r for r in reports if not r.ok and r.declared_reason is not None]

    for report in declared:
        log.warning(
            "[%s] Declared coverage exception for slide '%s': %s\n%s",
            stage,
            report.slide_id,
            report.declared_reason,
            _format_report(report),
        )

    if check_declared_bookkeeping:
        unused = sorted(
            set(cov["declared_exceptions"])
            - {r.slide_id for r in declared}
            - {r.slide_id for r in reports if r.ok}
        )
        if unused:
            log.warning(
                "[%s] references.coverage.declared_exceptions names slide(s) absent from this "
                "eval: %s. Remove them.",
                stage,
                unused,
            )
    satisfied_but_declared = sorted(
        {r.slide_id for r in reports if r.ok} & set(cov["declared_exceptions"])
    )
    if satisfied_but_declared:
        log.warning(
            "[%s] slide(s) %s satisfy the coverage contract but are still listed in "
            "references.coverage.declared_exceptions. Remove them.",
            stage,
            satisfied_but_declared,
        )

    if not failing:
        log.info(
            "[%s] A-2 reference-coverage contract satisfied: %d slide(s) checked, "
            "%d declared exception(s), min_ref_cells_per_gt_class=%d, min_scored_cells_per_slide=%d.",
            stage,
            len(reports),
            len(declared),
            int(cov["min_ref_cells_per_gt_class"]),
            int(cov["min_scored_cells_per_slide"]),
        )
        return

    blocks = "".join(_format_report(r) for r in failing)
    offenders = ", ".join(repr(r.slide_id) for r in failing)
    raise CoverageContractError(
        f"[{stage}] A-2 reference-coverage contract violated by {len(failing)} slide(s): "
        f"{offenders}.\n\n"
        f"{blocks}\n"
        "Either (1) supply a reference for this tissue that contains the missing class(es) and\n"
        "re-run the setup step, or (2) declare the exception in eval_config.yaml:\n\n"
        "        references:\n"
        "          coverage:\n"
        "            declared_exceptions:\n"
        + "".join(
            f'              {r.slide_id}: "<why this slide is still worth scoring '
            f'without {", ".join(c.label for c in r.classes if not c.covered) or "the missing support"}>"\n'
            for r in failing
        )
        + "\n"
        f"Current floors: min_ref_cells_per_gt_class={int(cov['min_ref_cells_per_gt_class'])}, "
        f"min_scored_cells_per_slide={int(cov['min_scored_cells_per_slide'])}."
    )


def assert_runtime_coverage_matches_frozen(
    *,
    slide_id: str,
    frozen_covered: np.ndarray,
    runtime_covered: np.ndarray,
    cfg: Mapping[str, Any],
    log: Optional[logging.Logger] = None,
) -> None:
    """Check that the run-time (_3) coverage mask equals the _2 one (raises CoverageContractError otherwise)."""
    log = log or logger
    cov = coverage_config(cfg)
    if not cov["enabled"] or not cov["require_runtime_match"]:
        return

    frozen = np.asarray(frozen_covered, dtype=bool).reshape(-1)
    runtime = np.asarray(runtime_covered, dtype=bool).reshape(-1)
    if frozen.shape != runtime.shape:
        raise CoverageContractError(
            f"[{STAGE_RUN}] slide '{slide_id}': frozen coverage mask has {frozen.size} rows but "
            f"the run-time mask has {runtime.size}. The row universe changed between _2 and _3."
        )
    lost = int(np.sum(frozen & ~runtime))
    gained = int(np.sum(runtime & ~frozen))
    if lost == 0 and gained == 0:
        return

    raise CoverageContractError(
        f"[{STAGE_RUN}] slide '{slide_id}': run-time reference coverage does not match the frozen "
        f"_2 coverage ({lost:,} cell(s) lost, {gained:,} gained out of {frozen.size:,}).\n"
        "Check the invalid projected reference embeddings of this slide's reference."
    )
