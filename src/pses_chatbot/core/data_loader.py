from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from pses_chatbot.config import (
    CKAN_DATASTORE_SEARCH_URL,
    PSES_DATASTORE_RESOURCE_ID,
)

# Derived endpoint (keeps config minimal)
# Example: https://open.canada.ca/data/en/api/3/action/datastore_search_sql
CKAN_DATASTORE_SEARCH_SQL_URL = CKAN_DATASTORE_SEARCH_URL.replace(
    "datastore_search", "datastore_search_sql"
)


class DataLoaderError(Exception):
    """Raised when CKAN/DataStore calls fail or return invalid responses."""


@dataclass(frozen=True)
class _DataStorePage:
    records: List[Dict[str, Any]]
    total: Optional[int]


def _make_session() -> requests.Session:
    """
    Requests session with retries/backoff for transient CKAN issues (429/5xx).
    This reduces “silent hangs” caused by intermittent portal instability.
    """
    session = requests.Session()

    retry = Retry(
        total=3,
        connect=3,
        read=3,
        status=3,
        backoff_factor=0.8,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET", "POST"}),
        raise_on_status=False,
    )

    adapter = HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=10)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


_SESSION = _make_session()


def _resolve_resource_id(resource_id: Optional[str] = None) -> str:
    rid = (resource_id or "").strip() or (PSES_DATASTORE_RESOURCE_ID or "").strip()
    if not rid:
        raise DataLoaderError(
            "Missing CKAN resource id. Set PSES_DATASTORE_RESOURCE_ID in config.py "
            "or via environment variable."
        )
    return rid


def _safe_sql_literal(value: str) -> str:
    """Escape single quotes for SQL literals."""
    return value.replace("'", "''")


def _ckan_post_json(
    url: str,
    payload: Dict[str, Any],
    *,
    timeout_seconds: int,
) -> Dict[str, Any]:
    try:
        resp = _SESSION.post(url, json=payload, timeout=timeout_seconds)
    except requests.exceptions.RequestException as exc:
        raise DataLoaderError(f"HTTP error while calling CKAN: {exc}") from exc

    try:
        data = resp.json()
    except Exception as exc:
        raise DataLoaderError(
            f"Non-JSON response from CKAN (status={resp.status_code})."
        ) from exc

    if not data.get("success", False):
        err = data.get("error", {}) or {}
        raise DataLoaderError(
            f"CKAN call failed (status={resp.status_code}, success=false): {err}"
        )

    return data


def _datastore_search(
    *,
    resource_id: str,
    filters: Optional[Dict[str, Any]],
    fields: Optional[List[str]],
    sort: Optional[str],
    limit: int,
    offset: int,
    include_total: bool,
    timeout_seconds: int,
) -> _DataStorePage:
    """
    CKAN datastore_search via POST JSON (more reliable than long GET querystrings).
    """
    payload: Dict[str, Any] = {
        "resource_id": resource_id,
        "limit": int(limit),
        "offset": int(offset),
        "include_total": bool(include_total),
    }
    if filters is not None:
        payload["filters"] = filters
    if fields:
        payload["fields"] = fields
    if sort:
        payload["sort"] = sort

    data = _ckan_post_json(CKAN_DATASTORE_SEARCH_URL, payload, timeout_seconds=timeout_seconds)
    result = data.get("result", {}) or {}
    records = result.get("records", []) or []
    total = result.get("total", None)

    if not isinstance(records, list):
        raise DataLoaderError("Unexpected CKAN response: result.records is not a list.")

    return _DataStorePage(records=records, total=total if include_total else None)


def query_pses_results(
    *,
    filters: Optional[Dict[str, Any]],
    fields: Optional[List[str]] = None,
    sort: Optional[str] = None,
    max_rows: int = 5_000,
    page_size: int = 1_000,
    timeout_seconds: int = 90,
    include_total: bool = False,
    resource_id: Optional[str] = None,
) -> pd.DataFrame:
    """
    Query PSES DataStore using equality filters (datastore_search) and paging.

    Notes:
      - No aggregation is performed.
      - For “overall/no breakdown” DEMCODE, use query_pses_results_overall_sql()
        because CKAN filters cannot express (DEMCODE IS NULL OR DEMCODE='').
    """
    rid = _resolve_resource_id(resource_id)

    if max_rows <= 0:
        return pd.DataFrame()

    # Prevent accidental huge pulls if filters are missing
    if filters is None and max_rows > 500:
        raise DataLoaderError(
            "Refusing an unfiltered datastore_search with max_rows > 500. "
            "Provide filters or reduce max_rows."
        )

    page_size = max(1, min(int(page_size), 10_000))
    max_rows = int(max_rows)

    records: List[Dict[str, Any]] = []
    offset = 0

    while offset < max_rows:
        limit = min(page_size, max_rows - offset)

        page = _datastore_search(
            resource_id=rid,
            filters=filters,
            fields=fields,
            sort=sort,
            limit=limit,
            offset=offset,
            include_total=include_total,
            timeout_seconds=timeout_seconds,
        )

        if not page.records:
            break

        records.extend(page.records)

        if len(page.records) < limit:
            break

        offset += limit
        time.sleep(0.05)

    if not records:
        return pd.DataFrame()

    return pd.DataFrame.from_records(records)


def query_pses_results_overall_sql(
    *,
    question_code: str,
    survey_year: int,
    org_levels: Dict[str, int],
    fields: Optional[List[str]] = None,
    timeout_seconds: int = 90,
    limit: int = 50_000,
    resource_id: Optional[str] = None,
) -> pd.DataFrame:
    """
    Query “overall/no breakdown” rows where DEMCODE is NULL or '' using datastore_search_sql.

    This is the canonical way to fetch baseline rows from this DataStore.
    """
    rid = _resolve_resource_id(resource_id)

    q = _safe_sql_literal(str(question_code).strip())
    y = int(survey_year)

    where_parts: List[str] = [
        f"\"QUESTION\" = '{q}'",
        f"\"SURVEYR\" = {y}",
        "(\"DEMCODE\" IS NULL OR \"DEMCODE\" = '')",
    ]

    for col in ("LEVEL1ID", "LEVEL2ID", "LEVEL3ID", "LEVEL4ID", "LEVEL5ID"):
        if col in org_levels:
            where_parts.append(f"\"{col}\" = {int(org_levels[col])}")

    select_clause = "*"
    if fields and len(fields) > 0:
        safe_fields = [f"\"{f.replace('\"', '')}\"" for f in fields]
        select_clause = ", ".join(safe_fields)

    sql = f'SELECT {select_clause} FROM "{rid}" WHERE ' + " AND ".join(where_parts) + f" LIMIT {int(limit)}"
    payload = {"sql": sql}

    data = _ckan_post_json(CKAN_DATASTORE_SEARCH_SQL_URL, payload, timeout_seconds=timeout_seconds)
    result = data.get("result", {}) or {}
    records = result.get("records", []) or []

    if not records:
        return pd.DataFrame()

    return pd.DataFrame.from_records(records)


def get_available_survey_years(
    *,
    timeout_seconds: int = 30,
    resource_id: Optional[str] = None,
) -> List[int]:
    """
    Discover available SURVEYR values using datastore_search_sql.

    IMPORTANT: do not call this automatically on app boot; call it from a button.
    """
    rid = _resolve_resource_id(resource_id)

    sql = f'SELECT DISTINCT "SURVEYR" FROM "{rid}" ORDER BY "SURVEYR"'
    payload = {"sql": sql}

    data = _ckan_post_json(CKAN_DATASTORE_SEARCH_SQL_URL, payload, timeout_seconds=timeout_seconds)
    result = data.get("result", {}) or {}
    records = result.get("records", []) or []

    years: List[int] = []
    for r in records:
        v = r.get("SURVEYR")
        try:
            years.append(int(v))
        except Exception:
            continue

    return sorted(set(years))
