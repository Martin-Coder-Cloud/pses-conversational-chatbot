from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional

import logging
import pandas as pd

from pses_chatbot.core.data_loader import query_pses_results
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
MOST_POSITIVE_COL = "MOST POSITIVE OR LEAST NEGATIVE"
ANSCOUNT_COL = "ANSCOUNT"

LEVEL_COLS = ["LEVEL1ID", "LEVEL2ID", "LEVEL3ID", "LEVEL4ID", "LEVEL5ID"]


class QueryEngineError(Exception):
    """Custom exception for query engine failures."""


@dataclass
class QueryParameters:
    """
    Structured parameters required to query the PSES DataStore.

    Rules:
      - survey_years: explicit list (UI defaults to all cycles)
      - question_code: required (e.g., 'Q08')
      - demcode:
          * None => overall / no breakdown.
            IMPORTANT: CKAN datastore_search cannot reliably filter for blank cells.
            Therefore overall is implemented as:
              - query WITHOUT DEMCODE filter
              - then filter locally where DEMCODE is blank/NULL/whitespace
          * str  => specific demographic code (filter equality at CKAN level)
      - org_levels: required explicit LEVEL1ID..LEVEL5ID integers
    """
    survey_years: List[int]
    question_code: str
    demcode: Optional[str]  # None => overall baseline
    org_levels: Dict[str, int]


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


# ---------------------------------------------------------------------------
# Metadata label helpers
# ---------------------------------------------------------------------------

def _lookup_question_labels(question_code: str) -> Tuple[str, str]:
    q_meta = load_questions_meta()
    code_norm = str(question_code).strip().upper()
    row = q_meta[q_meta["code"] == code_norm]
    if row.empty:
        logger.warning("Question %s not found in QUESTIONS metadata.", question_code)
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
        logger.warning("Org levels %s did not match any row in org metadata.", org_levels)
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
        logger.warning("Demographics metadata has no demcode/code column. Cannot label DEMCODE=%s.", code)
        return None, None

    row = dem_meta[dem_meta[col_name].astype(str) == code]
    if row.empty:
        logger.warning("DEMCODE %s not found in DEMCODE metadata.", code)
        return None, None

    r = row.iloc[0]
    en_col = "label_en" if "label_en" in r.index else ("DESCRIP_E" if "DESCRIP_E" in r.index else None)
    fr_col = "label_fr" if "label_fr" in r.index else ("DESCRIP_F" if "DESCRIP_F" in r.index else None)

    label_en = str(r.get(en_col, "")).strip() if en_col else ""
    label_fr = str(r.get(fr_col, "")).strip() if fr_col else ""
    return (label_en or None, label_fr or None)


# ---------------------------------------------------------------------------
# Years (local only)
# ---------------------------------------------------------------------------

def _known_survey_years() -> List[int]:
    """
    Local known cycles (no network calls). If you later add
    PSES_KNOWN_SURVEY_YEARS in config.py, it will be used automatically.
    """
    try:
        from pses_chatbot.config import PSES_KNOWN_SURVEY_YEARS  # type: ignore
        years = [int(y) for y in list(PSES_KNOWN_SURVEY_YEARS)]
        return sorted(set(years))
    except Exception:
        # Current known cycles in your open.canada.ca table
        return [2019, 2020, 2022, 2024]


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

    known = _known_survey_years()
    if known:
        known_set = set(known)
        intersection = [y for y in cleaned if y in known_set]
        if intersection:
            return intersection

    return cleaned


# ---------------------------------------------------------------------------
# CKAN query helpers
# ---------------------------------------------------------------------------

def _build_filters_base(params: QueryParameters, year: int) -> Dict[str, Any]:
    if not params.question_code:
        raise QueryEngineError("QUESTION (question_code) must be specified.")
    if not params.org_levels:
        raise QueryEngineError("org_levels must always be specified (LEVEL1ID..LEVEL5ID).")

    filters: Dict[str, Any] = {
        QUESTION_COL: str(params.question_code).strip(),
        SURVEY_YEAR_COL: int(year),
    }

    for col in LEVEL_COLS:
        if col in params.org_levels:
            filters[col] = int(params.org_levels[col])

    return filters


def _is_overall_demcode_value(x: Any) -> bool:
    """
    Overall/no breakdown row is represented by DEMCODE blank.
    Treat None, NaN, "", and whitespace-only as overall.
    """
    if x is None:
        return True
    try:
        if pd.isna(x):
            return True
    except Exception:
        pass
    return str(x).strip() == ""


def _fetch_one_year(params: QueryParameters, year: int, max_rows_per_year: int) -> pd.DataFrame:
    """
    Fetch one year slice.

    - If demcode is specified: filter DEMCODE at CKAN equality level.
    - If demcode is None (overall): do NOT filter DEMCODE in CKAN;
      fetch the slice and filter locally to DEMCODE blank/NULL.
    """
    base_filters = _build_filters_base(params, year)

    # Case A: specific demographic requested
    if params.demcode is not None and str(params.demcode).strip() != "":
        base_filters[DEMCODE_COL] = str(params.demcode).strip()

        return query_pses_results(
            filters=base_filters,
            fields=None,
            sort=None,
            max_rows=max_rows_per_year,
            page_size=min(max_rows_per_year, 10_000),
            timeout_seconds=120,
            include_total=False,
        )

    # Case B: overall requested (demcode None)
    # Fetch without DEMCODE filter, then filter locally to DEMCODE blank.
    df_all = query_pses_results(
        filters=base_filters,
        fields=None,
        sort=None,
        max_rows=max_rows_per_year,
        page_size=min(max_rows_per_year, 10_000),
        timeout_seconds=120,
        include_total=False,
    )

    if df_all.empty:
        return df_all

    if DEMCODE_COL not in df_all.columns:
        raise QueryEngineError(
            f"Returned slice does not contain '{DEMCODE_COL}' column; cannot isolate overall row."
        )

    df_overall = df_all[df_all[DEMCODE_COL].apply(_is_overall_demcode_value)].copy()
    return df_overall


def _fetch_raw_slice(params: QueryParameters, max_rows_per_year: int = 50_000) -> pd.DataFrame:
    if not params.survey_years:
        raise QueryEngineError("SURVEYR list must be specified (at least one year).")

    years = _validate_years(params.survey_years)

    frames: List[pd.DataFrame] = []
    for y in years:
        logger.info(
            "Querying PSES DataStore for year=%s (question=%s, demcode=%s, org=%s)",
            y, params.question_code, params.demcode, params.org_levels
        )

        df_y = _fetch_one_year(params, year=y, max_rows_per_year=max_rows_per_year)

        if df_y.empty:
            logger.warning("No records returned for year %s (after DEMCODE handling).", y)

        frames.append(df_y)

    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if df.empty:
        raise QueryEngineError("Combined slice is empty after querying all years.")

    return df


def _extract_yearly_metrics(df: pd.DataFrame) -> List[YearlyMetric]:
    required_cols = {SURVEY_YEAR_COL, MOST_POSITIVE_COL}
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise QueryEngineError(
            f"Required columns missing from slice: {missing}. Present columns: {list(df.columns)}"
        )

    # Expect exactly 1 row per year (after DEMCODE handling)
    counts = df.groupby(SURVEY_YEAR_COL).size()
    problematic = counts[counts > 1]
    if not problematic.empty:
        raise QueryEngineError(
            "Multiple rows per year found for the given QUESTION/DEMCODE/org levels. "
            "Aggregation is not permitted. "
            f"Problematic years: {problematic.to_dict()}"
        )

    work_cols = [SURVEY_YEAR_COL, MOST_POSITIVE_COL]
    if ANSCOUNT_COL in df.columns:
        work_cols.append(ANSCOUNT_COL)

    work = df[work_cols].copy().sort_values(SURVEY_YEAR_COL)

    metrics: List[YearlyMetric] = []
    prev_value: float | None = None

    for _, row in work.iterrows():
        year = int(row[SURVEY_YEAR_COL])

        try:
            value = float(row[MOST_POSITIVE_COL])
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

    return metrics


def _compute_overall_delta(metrics: List[YearlyMetric]) -> float | None:
    if len(metrics) < 2:
        return None
    valid = [m for m in metrics if pd.notna(m.value)]
    if len(valid) < 2:
        return None
    return valid[-1].value - valid[0].value


def run_analytical_query(params: QueryParameters) -> QueryResult:
    logger.info("Running analytical query with params=%s", params)

    raw_df = _fetch_raw_slice(params)
    yearly_metrics = _extract_yearly_metrics(raw_df)
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
    )
