from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List, Tuple

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

    Semantic rules (aligned with your design):

      - survey_years:
          * Always explicitly specified by the caller.
          * If the user does not mention years, the conversational layer
            should pass "all available cycles" for this question/org/dem.

      - question_code:
          * Always required (e.g., 'Q08').

      - demcode:
          * Always required and always a string.
          * Special case: demcode == "" (empty string) means "no breakdown"
            / "overall" (all respondents). This is a valid value and must be
            passed explicitly when the user does not request a demographic.

      - org_levels:
          * Always required. Caller must pass explicit LEVEL1ID..LEVEL5ID
            integers for the organizational scope (e.g., PS-wide, a department,
            or a specific unit).
    """
    survey_years: List[int]                    # e.g. [2019, 2020, 2021, 2022, 2023, 2024]
    question_code: str                         # e.g. 'Q08'
    demcode: str                               # "" for overall; "1001" for a specific demo
    org_levels: Dict[str, int]                 # keys: LEVEL1ID..LEVEL5ID (int IDs)


@dataclass
class YearlyMetric:
    year: int
    value: float
    delta_vs_prev: float | None  # difference vs previous year, if any
    n: int | None


@dataclass
class QueryResult:
    """
    Structured result of an analytical query.

    Contains:
      - The raw slice returned from CKAN (no aggregation).
      - Metric series by year (MOST POSITIVE OR LEAST NEGATIVE).
      - Overall change from first to last year.
      - Basic labels for question, org, and demographic.
    """
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
# Helpers for labels from metadata
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
    """
    Resolve an org label (EN/FR) from LEVEL1ID..LEVEL5ID.

    We use the LEVEL1ID_LEVEL5ID metadata. For now we do a simple match:
      - Match on all level IDs passed in org_levels.
      - If multiple rows match, we pick the first.
    """
    if not org_levels:
        raise QueryEngineError(
            "org_levels is empty. LEVEL1ID..LEVEL5ID must be specified "
            "for every analytical query."
        )

    org_meta = load_org_meta()

    mask = pd.Series([True] * len(org_meta))
    for col in LEVEL_COLS:
        if col in org_levels:
            val = org_levels[col]
            mask = mask & (org_meta[col] == int(val))

    matched = org_meta[mask]
    if matched.empty:
        logger.warning("Org levels %s did not match any row in org metadata.", org_levels)
        return None, None

    row = matched.iloc[0]
    return (
        str(row.get("org_name_en", "")).strip() or None,
        str(row.get("org_name_fr", "")).strip() or None,
    )


def _lookup_dem_labels(demcode: str) -> Tuple[str | None, str | None]:
    """
    Resolve demographic labels from DEMCODE metadata.

    Special rule:
      - demcode == "" (empty string) means "no breakdown / overall".
        In that case, we return a generic label.
    """
    if demcode.strip() == "":
        # No breakdown – overall respondents
        return "Overall (all respondents)", "Ensemble des répondants"

    dem_meta = load_demographics_meta()

    # The workbook-specific loader defines a 'demcode' column.
    # Fallback to 'code' if needed for robustness.
    col_name = None
    for cand in ("demcode", "code"):
        if cand in dem_meta.columns:
            col_name = cand
            break

    if col_name is None:
        logger.warning(
            "Demographics metadata has no 'demcode' or 'code' column. "
            "Cannot look up label for DEMCODE=%s.",
            demcode,
        )
        return None, None

    code_norm = str(demcode).strip()
    row = dem_meta[dem_meta[col_name] == code_norm]
    if row.empty:
        logger.warning("DEMCODE %s not found in DEMCODE metadata.", demcode)
        return None, None

    r = row.iloc[0]
    # For the workbook-specific loader, we expect 'label_en'/'label_fr'
    en_col = "label_en" if "label_en" in r.index else "DESCRIP_E"
    fr_col = "label_fr" if "label_fr" in r.index else "DESCRIP_F"

    label_en = str(r.get(en_col, "")).strip() or None
    label_fr = str(r.get(fr_col, "")).strip() or None
    return label_en, label_fr


# ---------------------------------------------------------------------------
# Core query & analysis functions
# ---------------------------------------------------------------------------

def _build_filters(params: QueryParameters, year: int | None = None) -> Dict[str, Any]:
    """
    Build CKAN filters dict for a single request.

    Note: CKAN's 'filters' supports equality conditions only. To query multiple
    years, we call the API multiple times (once per year) and concatenate.
    """
    if not params.question_code:
        raise QueryEngineError("QUESTION (question_code) must be specified.")
    if params.demcode is None:
        # By design, this should never happen if the conversational layer
        # is implemented correctly (empty string "" is used for 'overall').
        raise QueryEngineError(
            "DEMCODE must always be specified. Use empty string ('') for "
            "overall / no breakdown."
        )
    if not params.org_levels:
        raise QueryEngineError(
            "org_levels must always be specified (LEVEL1ID..LEVEL5ID)."
        )

    filters: Dict[str, Any] = {
        QUESTION_COL: str(params.question_code).strip(),
        DEMCODE_COL: str(params.demcode),  # empty string "" is valid (overall)
    }

    # Year
    if year is not None:
        filters[SURVEY_YEAR_COL] = int(year)

    # Organization levels
    for col in LEVEL_COLS:
        if col in params.org_levels:
            filters[col] = int(params.org_levels[col])

    return filters


def _fetch_raw_slice(params: QueryParameters, max_rows_per_year: int = 50_000) -> pd.DataFrame:
    """
    Fetch the raw PSES records for the given query parameters.

    This uses multiple CKAN calls if multiple survey years are requested.
    No aggregation is performed; this returns the published rows as-is.
    """
    if not params.survey_years:
        raise QueryEngineError("SURVEYR list must be specified (at least one year).")

    frames: List[pd.DataFrame] = []
    for y in params.survey_years:
        filters = _build_filters(params, year=y)
        logger.info("Querying PSES DataStore with filters=%s", filters)
        df_y = query_pses_results(
            filters=filters,
            fields=None,  # We keep all columns for now (safe slices).
            sort=None,
            max_rows=max_rows_per_year,
            page_size=min(max_rows_per_year, 50_000),
        )
        if df_y.empty:
            logger.warning("No records returned for year %s and filters %s", y, filters)
        frames.append(df_y)

    if not frames:
        raise QueryEngineError("No records returned for any requested year.")

    df = pd.concat(frames, ignore_index=True)
    if df.empty:
        raise QueryEngineError("Combined slice is empty after querying all years.")

    return df


def _extract_yearly_metrics(df: pd.DataFrame) -> List[YearlyMetric]:
    """
    Extract the MOST POSITIVE OR LEAST NEGATIVE metric by year, and
    compute simple differences vs previous year.

    This assumes that for a given (QUESTION, DEMCODE, org_levels, SURVEYR)
    there is exactly one row. If multiple rows per year are found, we raise
    an error rather than aggregating.
    """
    required_cols = {SURVEY_YEAR_COL, MOST_POSITIVE_COL}
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise QueryEngineError(
            f"Required columns missing from slice: {missing}. "
            f"Present columns: {list(df.columns)}"
        )

    # We expect 1 row per year for this combination of parameters.
    counts = df.groupby(SURVEY_YEAR_COL).size()
    problematic = counts[counts > 1]
    if not problematic.empty:
        raise QueryEngineError(
            "Multiple rows per year found for the given QUESTION/DEMCODE/org levels. "
            "Aggregation is not permitted, so the engine cannot safely summarize. "
            f"Problematic years: {problematic.to_dict()}"
        )

    work = df[[SURVEY_YEAR_COL, MOST_POSITIVE_COL, ANSCOUNT_COL]].copy()
    work = work.sort_values(SURVEY_YEAR_COL)

    metrics: List[YearlyMetric] = []
    prev_value: float | None = None

    for _, row in work.iterrows():
        year = int(row[SURVEY_YEAR_COL])
        val_raw = row[MOST_POSITIVE_COL]
        try:
            value = float(val_raw)
        except Exception:
            value = float("nan")

        n_val = row.get(ANSCOUNT_COL)
        try:
            n = int(n_val) if pd.notna(n_val) else None
        except Exception:
            n = None

        delta = None
        if prev_value is not None and pd.notna(value):
            delta = value - prev_value

        metrics.append(
            YearlyMetric(
                year=year,
                value=value,
                delta_vs_prev=delta,
                n=n,
            )
        )
        if pd.notna(value):
            prev_value = value

    return metrics


def _compute_overall_delta(metrics: List[YearlyMetric]) -> float | None:
    """
    Compute change from first to last year in MOST POSITIVE OR LEAST NEGATIVE.

    Returns None if fewer than 2 valid values are present.
    """
    if len(metrics) < 2:
        return None

    valid = [m for m in metrics if pd.notna(m.value)]
    if len(valid) < 2:
        return None

    first = valid[0]
    last = valid[-1]
    return last.value - first.value


def run_analytical_query(params: QueryParameters) -> QueryResult:
    """
    Main entrypoint for the analytical query engine.

    Steps:
      1. Fetch raw slice from the PSES DataStore using the mandatory parameters:
         SURVEYR(s), QUESTION, DEMCODE (including "" for overall),
         LEVEL1ID..LEVEL5ID (as provided).
      2. Extract MOST POSITIVE OR LEAST NEGATIVE by year (no aggregation).
      3. Compute year-to-year differences and overall change.
      4. Attach human-readable labels from metadata (question, org, dem).

    This function does NOT:
      - perform any aggregation
      - decide on default years, demcodes, or org levels

    Those decisions belong to the conversational layer, which must always
    resolve and pass explicit parameters aligned to your rules.
    """
    logger.info("Running analytical query with params=%s", params)

    # Fetch slice (no aggregation)
    raw_df = _fetch_raw_slice(params)

    # Extract metric per year
    yearly_metrics = _extract_yearly_metrics(raw_df)
    overall_delta = _compute_overall_delta(yearly_metrics)

    # Labels from metadata
    q_en, q_fr = _lookup_question_labels(params.question_code)
    org_en, org_fr = _lookup_org_labels(params.org_levels)
    dem_en, dem_fr = _lookup_dem_labels(params.demcode)

    result = QueryResult(
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

    return result
