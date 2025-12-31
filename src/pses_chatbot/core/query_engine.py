from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd

from pses_chatbot.core.data_loader import query_pses_results
from pses_chatbot.core.metadata_loader import (
    load_demographics_meta,
    load_org_meta,
    load_questions_meta,
)

logger = logging.getLogger(__name__)


class QueryEngineError(RuntimeError):
    pass


@dataclass
class QueryParameters:
    """
    Canonical analytical query parameters.

    - survey_years: list of years (2019/2020/2022/2024)
    - question_code: e.g. "Q08" (matches QUESTION in CKAN)
    - demcode:
        * None => overall baseline
            CKAN cannot reliably filter on empty DEMCODE.
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
    dem_category_en: str | None
    dem_category_fr: str | None
    metric_col_used: str  # NEW: which metric column we used (POSITIVE, etc.)


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
    org_meta = load_org_meta()
    if org_meta is None or org_meta.empty:
        return None, None

    df = org_meta.copy()
    for c in ["LEVEL1ID", "LEVEL2ID", "LEVEL3ID", "LEVEL4ID", "LEVEL5ID"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

    mask = pd.Series([True] * len(df))
    for c in ["LEVEL1ID", "LEVEL2ID", "LEVEL3ID", "LEVEL4ID", "LEVEL5ID"]:
        if c in df.columns:
            mask = mask & (df[c] == int(org_levels.get(c, 0)))

    row = df[mask]
    if row.empty:
        return None, None

    r = row.iloc[0]
    return (
        str(r.get("org_name_en", "")).strip() or None,
        str(r.get("org_name_fr", "")).strip() or None,
    )


def _lookup_dem_labels(demcode: Optional[str]) -> Tuple[str | None, str | None, str | None, str | None]:
    if demcode is None or str(demcode).strip() == "":
        return "Overall (no breakdown)", "Ensemble (sans ventilation)", None, None

    code = str(demcode).strip()
    dem_meta = load_demographics_meta()

    col_name = "demcode" if "demcode" in dem_meta.columns else ("code" if "code" in dem_meta.columns else None)
    if col_name is None:
        logger.warning("Demographics metadata has no demcode/code column. Cannot label DEMCODE=%s.", code)
        return None, None, None, None

    row = dem_meta[dem_meta[col_name].astype(str) == code]
    if row.empty:
        logger.warning("DEMCODE %s not found in DEMCODE metadata.", code)
        return None, None, None, None

    r = row.iloc[0]
    en_col = "label_en" if "label_en" in r.index else ("DESCRIP_E" if "DESCRIP_E" in r.index else None)
    fr_col = "label_fr" if "label_fr" in r.index else ("DESCRIP_F" if "DESCRIP_F" in r.index else None)

    label_en = str(r.get(en_col, "")).strip() if en_col else ""
    label_fr = str(r.get(fr_col, "")).strip() if fr_col else ""

    cat_en = str(r.get("category_en", "")).strip() if "category_en" in r.index else ""
    cat_fr = str(r.get("category_fr", "")).strip() if "category_fr" in r.index else ""
    return (label_en or None, label_fr or None, cat_en or None, cat_fr or None)


# ---------------------------------------------------------------------------
# Years (local only)
# ---------------------------------------------------------------------------

def _known_survey_years() -> List[int]:
    """
    Local known cycles (no network calls).
    """
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
        raise QueryEngineError("No valid survey years provided.")
    unknown = [y for y in cleaned if y not in _known_survey_years()]
    if unknown:
        raise QueryEngineError(f"Unknown survey years requested: {unknown}. Known: {_known_survey_years()}")
    return cleaned


# ---------------------------------------------------------------------------
# CKAN query + analytical logic
# ---------------------------------------------------------------------------

def run_analytical_query(params: QueryParameters) -> QueryResult:
    """
    Enforces:
      - no aggregation
      - expects exactly one row per year for overall (blank DEMCODE) OR specific subgroup DEMCODE
      - uses POSITIVE as the "most positive / least negative" metric
    """
    t0 = time.perf_counter()

    years = _validate_years(params.survey_years)
    question_code = str(params.question_code).strip().upper()
    demcode = None if params.demcode is None or str(params.demcode).strip() == "" else str(params.demcode).strip()

    # Build CKAN filters (do NOT include DEMCODE when overall)
    filters: Dict[str, Any] = {
        "QUESTION": question_code,
    }
    for c in ["LEVEL1ID", "LEVEL2ID", "LEVEL3ID", "LEVEL4ID", "LEVEL5ID"]:
        filters[c] = int(params.org_levels.get(c, 0))

    if demcode is not None:
        filters["DEMCODE"] = demcode

    # Pull a slice (never the whole dataset)
    df = query_pses_results(
        filters=filters,
        fields=None,
        sort=None,
        max_rows=5000,
        page_size=1000,
        timeout_seconds=120,
        include_total=False,
    )

    if df is None or df.empty:
        raise QueryEngineError("No rows returned from CKAN for the requested parameters.")

    # Local filter for overall rows (blank DEMCODE)
    if demcode is None:
        if "DEMCODE" not in df.columns:
            raise QueryEngineError("Returned data is missing DEMCODE column (required for overall filtering).")
        df["__dem_blank__"] = df["DEMCODE"].astype(str).fillna("").str.strip()
        df = df[df["__dem_blank__"] == ""].copy()
        df.drop(columns=["__dem_blank__"], inplace=True, errors="ignore")

    # Filter years locally (CKAN cannot do IN reliably with datastore_search)
    if "SURVEY_YEAR" not in df.columns:
        raise QueryEngineError("Returned data is missing SURVEY_YEAR column.")
    df["SURVEY_YEAR"] = pd.to_numeric(df["SURVEY_YEAR"], errors="coerce").astype("Int64")
    df = df[df["SURVEY_YEAR"].isin(years)].copy()

    if df.empty:
        raise QueryEngineError("No rows remain after local filtering for years/overall-demcode.")

    # Core metric column
    metric_col_used = "POSITIVE"
    if metric_col_used not in df.columns:
        raise QueryEngineError("Returned data is missing POSITIVE column (required).")

    # Enforce one row per year
    counts = df.groupby("SURVEY_YEAR").size().to_dict()
    bad = {int(k): int(v) for k, v in counts.items() if int(v) != 1}
    if bad:
        raise QueryEngineError(
            "No aggregation permitted, but multiple rows found for some years. "
            f"Expected 1 row per year; got: {bad}"
        )

    df = df.sort_values("SURVEY_YEAR").reset_index(drop=True)

    # Build yearly metrics (no aggregation; one row per year)
    yearly_metrics: List[YearlyMetric] = []
    prev_val: Optional[float] = None
    for _, row in df.iterrows():
        year = int(row["SURVEY_YEAR"])
        try:
            val = float(row[metric_col_used])
        except Exception:
            raise QueryEngineError(f"POSITIVE is not numeric for year {year}.")

        n_val = None
        if "ANSCOUNT" in df.columns:
            try:
                n_val = int(row["ANSCOUNT"])
            except Exception:
                n_val = None

        delta = None
        if prev_val is not None:
            delta = val - prev_val

        yearly_metrics.append(YearlyMetric(year=year, value=val, delta_vs_prev=delta, n=n_val))
        prev_val = val

    overall_delta = None
    if len(yearly_metrics) >= 2:
        overall_delta = yearly_metrics[-1].value - yearly_metrics[0].value

    q_en, q_fr = _lookup_question_labels(question_code)
    org_en, org_fr = _lookup_org_labels(params.org_levels)
    dem_en, dem_fr, dem_cat_en, dem_cat_fr = _lookup_dem_labels(params.demcode)

    t1 = time.perf_counter()
    logger.info("Analytical query completed in %.2fs", t1 - t0)

    return QueryResult(
        params=params,
        raw_df=df,
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
