from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import math

from pses_chatbot.core.query_engine import QueryResult, YearlyMetric


@dataclass
class AuditTrendFact:
    """
    A single trend fact between two years for the MOST POSITIVE OR LEAST NEGATIVE metric.
    """
    year_start: int
    year_end: int
    value_start: float
    value_end: float
    delta: float
    direction: str  # 'increase', 'decrease', 'no_change'


@dataclass
class AuditSnapshot:
    """
    Canonical numerical facts derived from a QueryResult.

    These are the facts the AI is allowed to talk about. They can be shown
    to the user for manual audit, and later used for automatic consistency
    checks against AI-generated statements.
    """
    question_code: str
    question_label_en: str
    question_label_fr: str

    org_label_en: Optional[str]
    org_label_fr: Optional[str]

    demcode: str
    dem_label_en: Optional[str]
    dem_label_fr: Optional[str]

    metric_name_en: str
    metric_name_fr: str

    metrics_by_year: Dict[int, float]
    n_by_year: Dict[int, Optional[int]]

    overall_delta: Optional[float]
    overall_direction: Optional[str]

    trend_facts: List[AuditTrendFact]


def _direction_from_delta(delta: float, tolerance: float = 0.1) -> str:
    """
    Interpret a numeric delta as 'increase', 'decrease', or 'no_change'.

    Tolerance is used to treat very small changes as 'no_change' to avoid
    over-interpreting small fluctuations.
    """
    if math.isnan(delta):
        return "no_change"
    if delta > tolerance:
        return "increase"
    if delta < -tolerance:
        return "decrease"
    return "no_change"


def build_audit_snapshot(result: QueryResult, tolerance: float = 0.1) -> AuditSnapshot:
    """
    Build a canonical set of audit facts from a QueryResult.

    This function:
      - Extracts metric values and sample sizes by year
      - Computes overall delta (first -> last)
      - Computes pairwise year-over-year deltas and directions
      - Labels everything with question/org/dem metadata

    The AI narrative layer should:
      - Only make claims that can be traced back to these facts.
      - Never invent numbers not represented here.
    """
    # Metrics by year
    metrics_by_year: Dict[int, float] = {}
    n_by_year: Dict[int, Optional[int]] = {}

    for m in result.yearly_metrics:
        metrics_by_year[m.year] = m.value
        n_by_year[m.year] = m.n

    # Overall delta & direction
    overall_delta = result.overall_delta
    overall_direction: Optional[str] = None
    if overall_delta is not None and not math.isnan(overall_delta):
        overall_direction = _direction_from_delta(overall_delta, tolerance=tolerance)

    # Year-over-year trend facts
    trend_facts: List[AuditTrendFact] = []
    sorted_years = sorted(metrics_by_year.keys())

    prev_year: Optional[int] = None
    prev_value: Optional[float] = None

    for year in sorted_years:
        value = metrics_by_year[year]
        if prev_year is not None and prev_value is not None and not math.isnan(value) and not math.isnan(prev_value):
            delta = value - prev_value
            direction = _direction_from_delta(delta, tolerance=tolerance)
            trend_facts.append(
                AuditTrendFact(
                    year_start=prev_year,
                    year_end=year,
                    value_start=prev_value,
                    value_end=value,
                    delta=delta,
                    direction=direction,
                )
            )
        prev_year = year
        prev_value = value

    # Metric labels (for display / narrative)
    metric_name_en = "Most positive or least negative"
    metric_name_fr = "Réponses les plus positives ou les moins négatives"

    snapshot = AuditSnapshot(
        question_code=result.params.question_code,
        question_label_en=result.question_label_en,
        question_label_fr=result.question_label_fr,
        org_label_en=result.org_label_en,
        org_label_fr=result.org_label_fr,
        demcode=result.params.demcode,
        dem_label_en=result.dem_label_en,
        dem_label_fr=result.dem_label_fr,
        metric_name_en=metric_name_en,
        metric_name_fr=metric_name_fr,
        metrics_by_year=metrics_by_year,
        n_by_year=n_by_year,
        overall_delta=overall_delta,
        overall_direction=overall_direction,
        trend_facts=trend_facts,
    )

    return snapshot
