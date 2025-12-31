from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from pses_chatbot.config import (
    CKAN_DATASTORE_SEARCH_URL,
    PSES_DATASTORE_RESOURCE_ID,
)


class DataLoaderError(Exception):
    """Raised when CKAN DataStore calls fail or return unexpected shapes."""


@dataclass
class CKANSearchResult:
    records: List[Dict[str, Any]]
    total: Optional[int] = None


def _build_retry_session() -> requests.Session:
    """
    Build a requests Session with conservative retries.
    Streamlit Cloud + open.canada.ca can be slow or transiently flaky.
    """
    session = requests.Session()

    retry = Retry(
        total=5,
        connect=5,
        read=5,
        status=5,
        backoff_factor=0.6,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        raise_on_status=False,
    )

    adapter = HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=10)
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    return session


_SESSION: Optional[requests.Session] = None


def _get_session() -> requests.Session:
    global _SESSION
    if _SESSION is None:
        _SESSION = _build_retry_session()
    return _SESSION


def _ckan_datastore_search(
    *,
    resource_id: str,
    filters: Optional[Dict[str, Any]],
    fields: Optional[List[str]],
    sort: Optional[str],
    offset: int,
    limit: int,
    timeout_seconds: int,
    include_total: bool,
) -> CKANSearchResult:
    """
    Calls CKAN datastore_search (GET).
    open.canada.ca supports datastore_search (not datastore_search_sql).
    """
    params: Dict[str, Any] = {
        "resource_id": resource_id,
        "offset": int(offset),
        "limit": int(limit),
    }

    # CKAN expects filters as a JSON string
    if filters:
        params["filters"] = json.dumps(filters, ensure_ascii=False)

    # 'fields' can reduce payload size if you only need a few columns
    if fields:
        params["fields"] = ",".join(fields)

    if sort:
        params["sort"] = sort

    if include_total:
        params["include_total"] = "true"

    try:
        resp = _get_session().get(CKAN_DATASTORE_SEARCH_URL, params=params, timeout=timeout_seconds)
    except Exception as exc:
        raise DataLoaderError(f"HTTP error while calling datastore_search: {exc}") from exc

    # CKAN sometimes returns 200 with success=false inside JSON
    try:
        data = resp.json()
    except Exception as exc:
        preview = (resp.text or "")[:200]
        raise DataLoaderError(f"Non-JSON response from CKAN (status={resp.status_code}). Preview: {preview}") from exc

    if not isinstance(data, dict):
        raise DataLoaderError(f"Unexpected CKAN response type: {type(data)}")

    if not data.get("success", False):
        # include any CKAN message if present
        msg = data.get("error") or data.get("help") or data
        raise DataLoaderError(f"CKAN datastore_search returned success=false. Status={resp.status_code}. Detail={msg}")

    result = data.get("result") or {}
    records = result.get("records") or []
    total = result.get("total") if include_total else None

    if not isinstance(records, list):
        raise DataLoaderError("CKAN result.records is not a list")

    return CKANSearchResult(records=records, total=total)


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
    stop_after_rows: Optional[int] = None,
) -> pd.DataFrame:
    """
    Query PSES DataStore using equality filters (datastore_search) with paging.

    Key behavior:
      - No aggregation is performed.
      - 'filters' is equality-match only (CKAN DataStore).
      - stop_after_rows can be used to exit early once enough rows are found.

    Parameters:
      - max_rows: absolute safety cap on rows returned
      - page_size: per-page limit (keep modest; big pages can time out)
      - stop_after_rows: if set (e.g., 1), stop paging once >= that many rows collected
    """
    rid = (resource_id or PSES_DATASTORE_RESOURCE_ID or "").strip()
    if not rid:
        raise DataLoaderError(
            "Missing CKAN resource id in config.py. Expected PSES_DATASTORE_RESOURCE_ID to be set."
        )

    max_rows = int(max_rows)
    page_size = int(page_size)
    if max_rows <= 0:
        return pd.DataFrame()
    if page_size <= 0:
        page_size = 1_000

    target = int(stop_after_rows) if stop_after_rows is not None else None
    if target is not None and target <= 0:
        target = None

    all_records: List[Dict[str, Any]] = []
    offset = 0

    # Hard cap to avoid runaway loops in case CKAN misbehaves
    # (keeps the app responsive even if total is huge)
    hard_page_cap = max(1, (max_rows // page_size) + 5)

    pages = 0
    while True:
        pages += 1
        if pages > hard_page_cap:
            break

        limit = min(page_size, max_rows - len(all_records))
        if limit <= 0:
            break

        page = _ckan_datastore_search(
            resource_id=rid,
            filters=filters,
            fields=fields,
            sort=sort,
            offset=offset,
            limit=limit,
            timeout_seconds=timeout_seconds,
            include_total=include_total,
        )

        recs = page.records
        if not recs:
            break

        all_records.extend(recs)

        # EARLY STOP (performance): quit as soon as we have enough
        if target is not None and len(all_records) >= target:
            break

        if len(all_records) >= max_rows:
            break

        offset += len(recs)

        # If CKAN returns fewer than requested, we reached the end
        if len(recs) < limit:
            break

    if not all_records:
        return pd.DataFrame()

    df = pd.DataFrame.from_records(all_records)
    return df


def get_available_survey_years() -> List[int]:
    """
    IMPORTANT: We do NOT scan the full dataset (15M+ rows) to discover years.
    open.canada.ca does not provide datastore_search_sql, so DISTINCT queries are not available.

    For prototype correctness + speed, keep the known cycles here.
    If cycles change, update this list (or later wire a lightweight metadata source).
    """
    return [2019, 2020, 2022, 2024]


def timed_ckan_query(
    *,
    filters: Optional[Dict[str, Any]],
    fields: Optional[List[str]] = None,
    sort: Optional[str] = None,
    max_rows: int = 5_000,
    page_size: int = 1_000,
    timeout_seconds: int = 90,
    include_total: bool = False,
    resource_id: Optional[str] = None,
    stop_after_rows: Optional[int] = None,
) -> Tuple[pd.DataFrame, float]:
    """
    Convenience helper for UI timing logs.
    """
    t0 = time.perf_counter()
    df = query_pses_results(
        filters=filters,
        fields=fields,
        sort=sort,
        max_rows=max_rows,
        page_size=page_size,
        timeout_seconds=timeout_seconds,
        include_total=include_total,
        resource_id=resource_id,
        stop_after_rows=stop_after_rows,
    )
    return df, (time.perf_counter() - t0)
