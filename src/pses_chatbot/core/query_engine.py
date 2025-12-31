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

# Business rule "MOST POSITIVE OR LEAST NEGATIVE" corresponds to POSITIVE in CKAN
PRIMARY_METRIC_COL = "POSITIVE"
ALT_METRIC_COLS = [
    "MOST POSITIVE OR LEAST NEGATIVE",  # legacy/local naming
    "MOST_POSITIVE_OR_LEAST_NEGATIVE",
]

ANSCOUNT_COL = "ANSCOUNT"
LEVEL_COLS = ["LEVEL1ID", "LEVEL2ID", "LEVEL3ID", "LEVEL4ID", "LEVEL5ID"]


class QueryEngineError(Exception):
    """Custom exception for query engine failures."""


@dataclass
class QueryParameters:
    """
    Structured parameters required to query the PSES DataStore.

    Demographic selection rules:
      - Overall (no breakdown): demcode is None and demcodes is None
        -> query without DEMCODE filter and locally keep blank DEMCODE rows only
      - Single subgroup: demcode is set, demcodes is None
        -> filter DEMCODE equality in CKAN
      - Category (all subgroups): demcodes is a list, demcode is None
        -> loop through each DEMCODE and run equality filters in CKAN (no IN)
    """
    survey_years: List[int]
    question_code: str
    demcode: Optional[str]            # single subgroup only
    demcodes: Optional[List[str]]     # category query (multiple subgroups)
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

    # For single-subgroup and overall queries, this is populated.
    yearly_metrics: List[YearlyMetric]
    overall_delta: float | None

    # For category (multi-demcode) queries, this is populated.
    metrics_by_demcode: Dict[str, List[YearlyMetric]] | None

    question_label_en: str
    question_label_fr: str
    org_label_en: str | None
    org_label_fr: str | None

    # If single subgroup (or overall), these can be populated.
    dem_label_en: str | None
    dem_label_fr: str | None
    dem_category_en: str | None
    dem_category_fr: str | None

    metric_col_used: str


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


def _lookup_dem_label_and_category(demcode: Optional[str]) -> Tuple[str | None, str | None, str | None, str | None]:
    """
    Returns: (label_en, label_fr, category_en, category_fr)
    """
    if demcode is None or str(demcode).strip() == "":
        return "Overall (no breakdown)", "Ensemble (sans ventilation)", None, None

    code = str(demcode).strip()
    dem_meta = load_demographics_meta()
    if dem_meta is None or dem_meta.empty:
        return None, None, None, None

    if "demcode" not in dem_meta.columns:
        logger.warning("Demographics metadata missing normalized 'demcode' column.")
        return None, None, None, None

    row = dem_meta[dem_meta["demcode"].astype(str) == code]
    if row.empty:
        logger.warning("DEMCODE %s not found in DEMCODE metadata.", code)
        return None, None, None, None

    r = row.iloc[0]
    label_en = str(r.get("label_en", "")).strip() or None
    label_fr = str(r.get("label_fr", "")).strip() or None
    cat_en = str(r.get("category_en", "")).strip() or None
    cat_fr = str(r.get("category_fr", "")).strip() or None
    return label_en, label_fr, cat_en, cat_fr


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
        filters[col] = int(params.org_levels.get(col, 0))

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


def _fetch_one_year_one_demcode(
    params: QueryParameters,
    year: int,
    demcode: Optional[str],
    max_rows_per_year: int,
) -> pd.DataFrame:
    filters = _build_filters_base(params, year)

    # Specific subgroup => filter in CKAN
    if demcode is not None and str(demcode).strip() != "":
        filters[DEMCODE_COL] = str(demcode).strip()
        return query_pses_results(
            filters=filters,
            fields=None,
            sort=None,
            max_rows=max_rows_per_year,
            page_size=min(max_rows_per_year, 10_000),
            timeout_seconds=120,
            include_total=False,
        )

    # Overall => do NOT filter on DEMCODE in CKAN; filter locally for blank
    df_all = query_pses_results(
        filters=filters,
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
        raise QueryEngineError(f"Returned slice does not contain '{DEMCODE_COL}' column; cannot isolate overall row.")

    return df_all[df_all[DEMCODE_COL].apply(_is_overall_demcode_value)].copy()


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
    return cleaned


def _fetch_raw_slice(params: QueryParameters, max_rows_per_year: int = 50_000) -> pd.DataFrame:
    years = _validate_years(params.survey_years)

    # Determine demographic targets
    # - Overall: demcode None and demcodes None -> one pass with demcode=None
    # - Single subgroup: demcode set -> one pass with that demcode
    # - Category: demcodes list -> loop demcodes
    dem_targets: List[Optional[str]] = []
    if params.demcodes and len(params.demcodes) > 0:
        dem_targets = [str(d).strip() for d in params.demcodes if str(d).strip() != ""]
        if not dem_targets:
            dem_targets = [None]
    elif params.demcode is not None and str(params.demcode).strip() != "":
        dem_targets = [str(params.demcode).strip()]
    else:
        dem_targets = [None]

    frames: List[pd.DataFrame] = []
    for d in dem_targets:
        for y in years:
            logger.info(
                "Querying year=%s (question=%s, demcode=%s, org=%s)",
                y, params.question_code, d, params.org_levels
            )
            df_y = _fetch_one_year_one_demcode(params, year=y, demcode=d, max_rows_per_year=max_rows_per_year)
            frames.append(df_y)

    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if df.empty:
        raise QueryEngineError("Combined slice is empty after querying requested years/demographics.")
    return df


def _extract_metrics_single(df: pd.DataFrame, metric_col: str) -> List[YearlyMetric]:
    if SURVEY_YEAR_COL not in df.columns:
        raise QueryEngineError(f"Required column missing from slice: {SURVEY_YEAR_COL}. Present: {list(df.columns)}")

    counts = df.groupby(SURVEY_YEAR_COL).size()
    problematic = counts[counts > 1]
    if not problematic.empty:
        raise QueryEngineError(
            "Multiple rows per year found for the given QUESTION/DEMCODE/org levels. Aggregation is not permitted. "
            f"Problematic years: {problematic.to_dict()}"
        )

    work_cols = [SURVEY_YEAR_COL, metric_col]
    if ANSCOUNT_COL in df.columns:
        work_cols.append(ANSCOUNT_COL)

    work = df[work_cols].copy().sort_values(SURVEY_YEAR_COL)

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

    return metrics


def _extract_metrics_by_demcode(df: pd.DataFrame, metric_col: str) -> Dict[str, List[YearlyMetric]]:
    if DEMCODE_COL not in df.columns:
        raise QueryEngineError(f"Returned slice missing '{DEMCODE_COL}' required for multi-demcode analysis.")

    out: Dict[str, List[YearlyMetric]] = {}
    # DEMCODE may be numeric or blank; normalize to string key
    df = df.copy()
    df["__demcode_key__"] = df[DEMCODE_COL].astype(str).fillna("").str.strip()

    # In category mode, all demcodes are non-blank (we filtered with CKAN equality),
    # but keep safety anyway.
    for dem_key, grp in df.groupby("__demcode_key__"):
        if dem_key == "":
            continue
        counts = grp.groupby(SURVEY_YEAR_COL).size()
        problematic = counts[counts > 1]
        if not problematic.empty:
            raise QueryEngineError(
                "Multiple rows per year found for a subgroup. Aggregation is not permitted. "
                f"DEMCODE={dem_key}, problematic years: {problematic.to_dict()}"
            )
        out[dem_key] = _extract_metrics_single(grp, metric_col)

    df.drop(columns=["__demcode_key__"], inplace=True, errors="ignore")
    return out


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
    metric_col_used = _resolve_metric_column(raw_df)

    # Determine whether this is multi-demcode mode
    is_multi = bool(params.demcodes and len(params.demcodes) > 0)

    if is_multi:
        metrics_by_dem = _extract_metrics_by_demcode(raw_df, metric_col_used)
        yearly_metrics: List[YearlyMetric] = []
        overall_delta = None
        dem_label_en = dem_label_fr = None
        dem_cat_en = dem_cat_fr = None
    else:
        yearly_metrics = _extract_metrics_single(raw_df, metric_col_used)
        overall_delta = _compute_overall_delta(yearly_metrics)
        metrics_by_dem = None
        dem_label_en, dem_label_fr, dem_cat_en, dem_cat_fr = _lookup_dem_label_and_category(params.demcode)

    q_en, q_fr = _lookup_question_labels(params.question_code)
    org_en, org_fr = _lookup_org_labels(params.org_levels)

    return QueryResult(
        params=params,
        raw_df=raw_df,
        yearly_metrics=yearly_metrics,
        overall_delta=overall_delta,
        metrics_by_demcode=metrics_by_dem,
        question_label_en=q_en,
        question_label_fr=q_fr,
        org_label_en=org_en,
        org_label_fr=org_fr,
        dem_label_en=dem_label_en,
        dem_label_fr=dem_label_fr,
        dem_category_en=dem_cat_en,
        dem_category_fr=dem_cat_fr,
        metric_col_used=metric_col_used,
    )
