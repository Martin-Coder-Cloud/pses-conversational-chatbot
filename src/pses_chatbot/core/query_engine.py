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


# ---------------------------------------------------------------------------
# Constants (CKAN column names)
# ---------------------------------------------------------------------------

SURVEY_YEAR_COL = "SURVEYR"
QUESTION_COL = "QUESTION"
DEMCODE_COL = "DEMCODE"

LEVEL_COLS = ["LEVEL1ID", "LEVEL2ID", "LEVEL3ID", "LEVEL4ID", "LEVEL5ID"]

# Business metric is "MOST POSITIVE OR LEAST NEGATIVE" and maps to physical CKAN column:
METRIC_POSITIVE_COL = "POSITIVE"


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class QueryEngineError(Exception):
    pass


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class QueryParameters:
    question_code: str
    survey_years: List[int]
    org_levels: Dict[str, int]
    demcode: Optional[str] = None


@dataclass
class YearlyMetric:
    year: int
    value: float
    delta_vs_prev: float | None


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
    dem_category_en: str | None
    dem_category_fr: str | None
    metric_col_used: str


# ---------------------------------------------------------------------------
# Utility: overall DEMCODE detection
# ---------------------------------------------------------------------------

def _is_overall_demcode_value(v: Any) -> bool:
    if v is None:
        return True
    s = str(v)
    # CKAN often returns blank string for missing.
    return s.strip() == ""


# ---------------------------------------------------------------------------
# Metadata label lookups
# ---------------------------------------------------------------------------

def _lookup_question_labels(question_code: str) -> Tuple[str, str]:
    q = str(question_code).strip()
    qmeta = load_questions_meta()

    col_q = "question" if "question" in qmeta.columns else ("QUESTION" if "QUESTION" in qmeta.columns else None)
    if col_q is None:
        return q, q

    row = qmeta[qmeta[col_q].astype(str).str.strip() == q]
    if row.empty:
        return q, q

    r = row.iloc[0]
    return (
        str(r.get("question_en", "")).strip() or q,
        str(r.get("question_fr", "")).strip() or q,
    )


def _lookup_org_labels(org_levels: Dict[str, int]) -> Tuple[str | None, str | None]:
    # Your org sheet is used mainly for cascade UI; but keep existing lookup behavior.
    try:
        org_meta = load_org_meta()
    except Exception:
        return None, None

    # A minimal fallback label if we can’t map.
    # (You already have org labels working in the UI; this is mainly for query result display.)
    if all(int(org_levels.get(k, 0)) == 0 for k in LEVEL_COLS):
        return "Public Service", "Fonction publique"

    return None, None


def _lookup_dem_labels(demcode: Optional[str]) -> Tuple[str | None, str | None, str | None, str | None]:
    if demcode is None or str(demcode).strip() == "":
        return "Overall (no breakdown)", "Ensemble (sans ventilation)", None, None

    code = str(demcode).strip()
    dem_meta = load_demographics_meta()

    # Expect normalized columns from metadata_loader, but keep backward compatibility.
    code_col = (
        "demcode"
        if "demcode" in dem_meta.columns
        else ("DEMCODE 2024" if "DEMCODE 2024" in dem_meta.columns else ("code" if "code" in dem_meta.columns else None))
    )
    if code_col is None:
        return None, None, None, None

    row = dem_meta[dem_meta[code_col].astype(str).str.strip() == code]
    if row.empty:
        return None, None, None, None

    r = row.iloc[0]

    en_col = "label_en" if "label_en" in r.index else ("DESCRIP_E" if "DESCRIP_E" in r.index else None)
    fr_col = "label_fr" if "label_fr" in r.index else ("DESCRIP_F" if "DESCRIP_F" in r.index else None)

    cat_en_col = (
        "category_en"
        if "category_en" in r.index
        else ("Category_E" if "Category_E" in r.index else ("category_e" if "category_e" in r.index else None))
    )
    cat_fr_col = (
        "category_fr"
        if "category_fr" in r.index
        else ("Category_F" if "Category_F" in r.index else ("category_f" if "category_f" in r.index else None))
    )

    label_en = str(r.get(en_col, "")).strip() if en_col else ""
    label_fr = str(r.get(fr_col, "")).strip() if fr_col else ""
    cat_en = str(r.get(cat_en_col, "")).strip() if cat_en_col else ""
    cat_fr = str(r.get(cat_fr_col, "")).strip() if cat_fr_col else ""

    return (label_en or None, label_fr or None, cat_en or None, cat_fr or None)


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
        raise QueryEngineError("SURVEYR list must be specified (at least one year).")
    return cleaned


# ---------------------------------------------------------------------------
# CKAN filter builder
# ---------------------------------------------------------------------------

def _build_filters_base(params: QueryParameters, year: int) -> Dict[str, Any]:
    filters: Dict[str, Any] = {
        SURVEY_YEAR_COL: int(year),
        QUESTION_COL: str(params.question_code).strip(),
    }

    # Org levels
    for k in LEVEL_COLS:
        filters[k] = int(params.org_levels.get(k, 0))

    return filters


# ---------------------------------------------------------------------------
# CKAN fetch per year with DEMCODE logic
# ---------------------------------------------------------------------------

def _fetch_one_year(params: QueryParameters, year: int) -> pd.DataFrame:
    """
    - For a specific subgroup demcode: CKAN equality filter is reliable.
    - For overall (demcode=None): cannot filter blank DEMCODE reliably in CKAN,
      so we query without DEMCODE and then filter locally for blank DEMCODE.
    """
    base_filters = _build_filters_base(params, year)

    # Case A: specific demographic (reliable equality filter)
    if params.demcode is not None and str(params.demcode).strip() != "":
        base_filters[DEMCODE_COL] = str(params.demcode).strip()
        df = query_pses_results(
            filters=base_filters,
            fields=None,
            sort=None,
            max_rows=50,
            page_size=50,
            timeout_seconds=120,
            include_total=False,
            stop_after_rows=5,
        )
        return df

    # Case B: overall (blank DEMCODE) — query without DEMCODE filter; isolate locally
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

    # Strict correctness guardrails:
    # - No aggregation: must have exactly 1 row per year after DEMCODE handling.
    # - No missing years: the returned slice must cover every requested year.
    counts = df.groupby(SURVEY_YEAR_COL).size()
    missing_years = [y for y in years if y not in counts.index.tolist()]
    if missing_years:
        raise QueryEngineError(
            f"Missing year(s) in returned slice: {missing_years}. "
            "This likely indicates CKAN filtering did not return the expected overall/subgroup row."
        )
    multi = counts[counts > 1]
    if not multi.empty:
        raise QueryEngineError(
            "Multiple rows per year found for the given QUESTION/DEMCODE/org levels. "
            "Aggregation is not permitted. "
            f"Counts by year: {multi.to_dict()}"
        )

    return df


# ---------------------------------------------------------------------------
# Metric extraction (no aggregation)
# ---------------------------------------------------------------------------

def _resolve_metric_column(df: pd.DataFrame) -> str:
    if METRIC_POSITIVE_COL in df.columns:
        return METRIC_POSITIVE_COL
    raise QueryEngineError(
        f"Metric column '{METRIC_POSITIVE_COL}' not found in returned slice. Present: {list(df.columns)}"
    )


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
            f"Counts by year: {problematic.to_dict()}"
        )

    yearly_metrics: List[YearlyMetric] = []
    df_sorted = df.sort_values(SURVEY_YEAR_COL).copy()
    prev_val: float | None = None

    for _, row in df_sorted.iterrows():
        year = int(row[SURVEY_YEAR_COL])
        val_raw = row.get(metric_col, None)
        try:
            val = float(val_raw)
        except Exception:
            raise QueryEngineError(f"Non-numeric metric value for year {year}: {val_raw}")

        delta = None
        if prev_val is not None:
            delta = val - prev_val

        yearly_metrics.append(YearlyMetric(year=year, value=val, delta_vs_prev=delta))
        prev_val = val

    return yearly_metrics, metric_col


def _compute_overall_delta(yearly: List[YearlyMetric]) -> float | None:
    if not yearly or len(yearly) < 2:
        return None
    return yearly[-1].value - yearly[0].value


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_analytical_query(params: QueryParameters) -> QueryResult:
    raw_df = _fetch_raw_slice(params)
    yearly_metrics, metric_col_used = _extract_yearly_metrics(raw_df)
    overall_delta = _compute_overall_delta(yearly_metrics)

    q_en, q_fr = _lookup_question_labels(params.question_code)
    org_en, org_fr = _lookup_org_labels(params.org_levels)
    dem_en, dem_fr, dem_cat_en, dem_cat_fr = _lookup_dem_labels(params.demcode)

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
        dem_category_en=dem_cat_en,
        dem_category_fr=dem_cat_fr,
        metric_col_used=metric_col_used,
    )
