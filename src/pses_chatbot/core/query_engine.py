from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional

import logging
import pandas as pd

from pses_chatbot.core.data_loader import query_pses_results, get_available_survey_years

from pses_chatbot.core.metadata_loader import (
    load_questions_meta,
    load_org_meta,
    load_demographics_meta,
)

logger = logging.getLogger(__name__)

# Column names in the PSES DataStore
SURVEY_YEAR_COL = "SURVEYR"
QUESTION_COL = "QUESTION"
DEMCODE_COL = "DEMCODE"

# Business metric “Most positive or least negative” maps to POSITIVE in DataStore
PRIMARY_METRIC_COL = "POSITIVE"
ALT_METRIC_COLS = [
    "MOST POSITIVE OR LEAST NEGATIVE",
    "MOST_POSITIVE_OR_LEAST_NEGATIVE",
]

ANSCOUNT_COL = "ANSCOUNT"
LEVEL_COLS = ["LEVEL1ID", "LEVEL2ID", "LEVEL3ID", "LEVEL4ID", "LEVEL5ID"]


class QueryEngineError(Exception):
    """Custom exception for query engine failures."""


@dataclass
class QueryParameters:
    survey_years: List[int]
    question_code: str
    demcode: Optional[str]  # None => overall / no breakdown
    org_levels: Dict[str, int]  # LEVEL1ID..LEVEL5ID


@dataclass
class YearlyMetric:
    year: int
    value: float
    delta_vs_prev: float | None
    n: int | None


@dataclass
class QueryResult:
    params: QueryParameters
    raw_df: pd.DataFrame
    yearly_metrics: List[YearlyMetric]
    overall_delta: float | None
    question_label_en: str
    question_label_fr: str
    org_label_en: str | None
    org_label_fr: str | None
    dem_label_en: str | None
    dem_label_fr: str | None
    metric_col_used: str


# ---------------------------------------------------------------------------
# Metadata label helpers
# ---------------------------------------------------------------------------

def _lookup_question_labels(question_code: str) -> Tuple[str, str]:
    q_meta = load_questions_meta()
    code_norm = str(question_code).strip().upper()
    row = q_meta[q_meta["code"] == code_norm]
    if row.empty:
        return code_norm, code_norm
    r = row.iloc[0]
    return str(r.get("text_en", code_norm)), str(r.get("text_fr", code_norm))


def _lookup_org_labels(org_levels: Dict[str, int]) -> Tuple[str | None, str | None]:
    if not org_levels:
        raise QueryEngineError("org_levels is empty. LEVEL1ID..LEVEL5ID must be specified.")

    org_meta = load_org_meta()

    mask = pd.Series([True] * len(org_meta))
    for col in LEVEL_COLS:
        if col in org_levels:
            mask = mask & (org_meta[col] == int(org_levels[col]))

    matched = org_meta[mask]
    if matched.empty:
        return None, None

    row = matched.iloc[0]
    return (
        str(row.get("org_name_en", "")).strip() or None,
        str(row.get("org_name_fr", "")).strip() or None,
    )


def _lookup_dem_labels(demcode: Optional[str]) -> Tuple[str | None, str | None]:
    if demcode is None or str(demcode).strip() == "":
        return "Overall (no breakdown)", "Ensemble (sans ventilation)"

    code = str(demcode).strip()
    dem_meta = load_demographics_meta()

    col_name = "demcode" if "demcode" in dem_meta.columns else ("code" if "code" in dem_meta.columns else None)
    if col_name is None:
        return None, None

    row = dem_meta[dem_meta[col_name].astype(str) == code]
    if row.empty:
        return None, None

    r = row.iloc[0]
    en_col = "label_en" if "label_en" in r.index else ("DESCRIP_E" if "DESCRIP_E" in r.index else None)
    fr_col = "label_fr" if "label_fr" in r.index else ("DESCRIP_F" if "DESCRIP_F" in r.index else None)

    label_en = str(r.get(en_col, "")).strip() if en_col else ""
    label_fr = str(r.get(fr_col, "")).strip() if fr_col else ""
    return (label_en or None, label_fr or None)


# ---------------------------------------------------------------------------
# Years
# ---------------------------------------------------------------------------

def _validate_years(requested: List[int]) -> List[int]:
    cleaned: List[int] = []
    for y in requested:
        try:
            cleaned.append(int(y))
        except Exception:
            continue
    cleaned = sorted(set(cleaned))
    if not cleaned:
        raise QueryEngineError("SURVEYR list must contain at least one valid year.")

    # Prototype: rely on a fast, non-scanning list (no SQL DISTINCT)
    available = get_available_survey_years()
    avail_set = set(available)
    intersection = [y for y in cleaned if y in avail_set]
    return intersection if intersection else cleaned


# ---------------------------------------------------------------------------
# Metric column resolution
# ---------------------------------------------------------------------------

def _resolve_metric_column(df: pd.DataFrame) -> str:
    if PRIMARY_METRIC_COL in df.columns:
        return PRIMARY_METRIC_COL
    for c in ALT_METRIC_COLS:
        if c in df.columns:
            return c
    raise QueryEngineError(
        f"Required metric column not found. Expected '{PRIMARY_METRIC_COL}' "
        f"(or one of {ALT_METRIC_COLS}). Present columns: {list(df.columns)}"
    )


# ---------------------------------------------------------------------------
# CKAN filters + overall DEMCODE handling
# ---------------------------------------------------------------------------

def _build_filters_base(params: QueryParameters, year: int) -> Dict[str, Any]:
    if not params.question_code:
        raise QueryEngineError("QUESTION (question_code) must be specified.")
    if not params.org_levels:
        raise QueryEngineError("org_levels must always be specified (LEVEL1ID..LEVEL5ID).")

    filters: Dict[str, Any] = {
        QUESTION_COL: str(params.question_code).strip().upper(),
        SURVEY_YEAR_COL: int(year),
    }

    for col in LEVEL_COLS:
        if col in params.org_levels:
            filters[col] = int(params.org_levels[col])

    return filters


def _is_overall_demcode_value(x: Any) -> bool:
    if x is None:
        return True
    try:
        if pd.isna(x):
            return True
    except Exception:
        pass
    return str(x).strip() == ""


def _fetch_one_year(params: QueryParameters, year: int) -> pd.DataFrame:
    """
    Performance strategy:
      - For a single year slice, we only need ONE row.
      - Use stop_after_rows when filtering is reliable.
      - For overall (demcode=None), we cannot filter blank DEMCODE in CKAN reliably,
        so we:
          1) query without DEMCODE filter (small pages)
          2) filter locally for blank DEMCODE
          3) stop as soon as we found 1 row
    """
    base_filters = _build_filters_base(params, year)

    # Case A: specific demographic (reliable equality filter)
    if params.demcode is not None and str(params.demcode).strip() != "":
        base_filters[DEMCODE_COL] = str(params.demcode).strip()

        return query_pses_results(
            filters=base_filters,
            fields=None,
            sort=None,
            max_rows=5_000,
            page_size=1_000,
            timeout_seconds=120,
            include_total=False,
            stop_after_rows=1,
        )

    # Case B: overall (demcode=None) — do not filter DEMCODE in CKAN
    # We page manually in small chunks by repeatedly calling query_pses_results
    # with increasing offsets is NOT exposed; so we use max_rows + local filter
    # and rely on early-stop within one call by fetching small max_rows.
    #
    # Pragmatic approach:
    #   - fetch a modest slice (e.g., up to 10k) for that year/question/org
    #   - isolate overall rows (blank DEMCODE)
    #   - return that (should be 1 row)
    df_all = query_pses_results(
        filters=base_filters,
        fields=None,
        sort=None,
        max_rows=10_000,
        page_size=1_000,
        timeout_seconds=120,
        include_total=False,
        stop_after_rows=None,  # cannot early-stop before filtering locally
    )

    if df_all.empty:
        return df_all

    if DEMCODE_COL not in df_all.columns:
        raise QueryEngineError(
            f"Returned slice does not contain '{DEMCODE_COL}' column; cannot isolate overall row."
        )

    df_overall = df_all[df_all[DEMCODE_COL].apply(_is_overall_demcode_value)].copy()
    # If multiple somehow exist, keep them all and let the “no aggregation” checks catch it.
    return df_overall


def _fetch_raw_slice(params: QueryParameters) -> pd.DataFrame:
    if not params.survey_years:
        raise QueryEngineError("SURVEYR list must be specified (at least one year).")

    years = _validate_years(params.survey_years)

    frames: List[pd.DataFrame] = []
    for y in years:
        df_y = _fetch_one_year(params, year=y)
        frames.append(df_y)

    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if df.empty:
        raise QueryEngineError("Combined slice is empty after querying all years.")

    return df


def _extract_yearly_metrics(df: pd.DataFrame) -> Tuple[List[YearlyMetric], str]:
    if SURVEY_YEAR_COL not in df.columns:
        raise QueryEngineError(
            f"Required column missing from slice: {SURVEY_YEAR_COL}. Present: {list(df.columns)}"
        )

    metric_col = _resolve_metric_column(df)

    # Expect exactly 1 row per year (after DEMCODE handling)
    counts = df.groupby(SURVEY_YEAR_COL).size()
    problematic = counts[counts > 1]
    if not problematic.empty:
        raise QueryEngineError(
            "Multiple rows per year found for the given QUESTION/DEMCODE/org levels. "
            "Aggregation is not permitted. "
            f"Problematic years: {problematic.to_dict()}"
        )

    cols = [SURVEY_YEAR_COL, metric_col]
    if ANSCOUNT_COL in df.columns:
        cols.append(ANSCOUNT_COL)

    work = df[cols].copy().sort_values(SURVEY_YEAR_COL)

    metrics: List[YearlyMetric] = []
    prev_value: float | None = None

    for _, row in work.iterrows():
        year = int(row[SURVEY_YEAR_COL])

        try:
            value = float(row[metric_col])
        except Exception:
            value = float("nan")

        n: int | None = None
        if ANSCOUNT_COL in row.index:
            try:
                n_val = row.get(ANSCOUNT_COL)
                n = int(n_val) if pd.notna(n_val) else None
            except Exception:
                n = None

        delta = None
        if prev_value is not None and pd.notna(value):
            delta = value - prev_value

        metrics.append(YearlyMetric(year=year, value=value, delta_vs_prev=delta, n=n))
        if pd.notna(value):
            prev_value = value

    return metrics, metric_col


def _compute_overall_delta(metrics: List[YearlyMetric]) -> float | None:
    if len(metrics) < 2:
        return None
    valid = [m for m in metrics if pd.notna(m.value)]
    if len(valid) < 2:
        return None
    return valid[-1].value - valid[0].value


def run_analytical_query(params: QueryParameters) -> QueryResult:
    raw_df = _fetch_raw_slice(params)
    yearly_metrics, metric_col_used = _extract_yearly_metrics(raw_df)
    overall_delta = _compute_overall_delta(yearly_metrics)

    q_en, q_fr = _lookup_question_labels(params.question_code)
    org_en, org_fr = _lookup_org_labels(params.org_levels)
    dem_en, dem_fr = _lookup_dem_labels(params.demcode)

    return QueryResult(
        params=params,
        raw_df=raw_df,
        yearly_metrics=yearly_metrics,
        overall_delta=overall_delta,
        question_label_en=q_en,
        question_label_fr=q_fr,
        org_label_en=org_en,
        org_label_fr=org_fr,
        dem_label_en=dem_en,
        dem_label_fr=dem_fr,
        metric_col_used=metric_col_used,
    )
