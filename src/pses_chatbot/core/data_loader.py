from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from pses_chatbot.config import CKAN_DATASTORE_SEARCH_URL, PSES_DATASTORE_RESOURCE_ID


class DataLoaderError(Exception):
    """Raised when CKAN/DataStore calls fail or return invalid responses."""


@dataclass(frozen=True)
class _DataStorePage:
    records: List[Dict[str, Any]]
    total: Optional[int]


def _make_session() -> requests.Session:
    """
    Requests session with retries/backoff for transient CKAN issues (429/5xx).
    """
    session = requests.Session()

    retry = Retry(
        total=3,
        connect=3,
        read=3,
        status=3,
        backoff_factor=0.8,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),  # open.canada.ca: GET is the safe default
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


def _encode_filters(filters: Optional[Dict[str, Any]]) -> Optional[str]:
    """
    For datastore_search via GET, CKAN expects `filters` as a JSON string.

    IMPORTANT: we must preserve empty strings exactly (e.g., DEMCODE="").
    """
    if filters is None:
        return None
    try:
        return json.dumps(filters, ensure_ascii=False)
    except Exception as exc:
        raise DataLoaderError(f"Could not JSON-encode filters for CKAN: {exc}") from exc


def _ckan_get_json(url: str, params: Dict[str, Any], *, timeout_seconds: int) -> Dict[str, Any]:
    try:
        resp = _SESSION.get(url, params=params, timeout=timeout_seconds)
    except requests.exceptions.RequestException as exc:
        raise DataLoaderError(f"HTTP error while calling CKAN: {exc}") from exc

    # Try JSON first
    try:
        data: Any = resp.json()
    except Exception as exc:
        preview = (resp.text or "")[:800]
        raise DataLoaderError(
            f"Non-JSON response from CKAN (status={resp.status_code}). "
            f"Response preview: {preview}"
        ) from exc

    # Some CKAN responses can be stringified JSON; decode once.
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except Exception:
            preview = (data or "")[:800]
            raise DataLoaderError(
                "CKAN returned JSON as a string but it could not be parsed. "
                f"Response string preview: {preview}"
            )

    if not isinstance(data, dict):
        preview = str(data)[:800]
        raise DataLoaderError(
            "Unexpected CKAN response type after JSON decoding. "
            f"Expected dict, got {type(data).__name__}. Preview: {preview}"
        )

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
    CKAN datastore_search via GET.
    """
    params: Dict[str, Any] = {
        "resource_id": resource_id,
        "limit": int(limit),
        "offset": int(offset),
    }

    if include_total:
        params["include_total"] = "true"

    filt_str = _encode_filters(filters)
    if filt_str is not None:
        params["filters"] = filt_str

    if fields:
        # CKAN accepts comma-separated fields in many installs
        params["fields"] = ",".join(fields)

    if sort:
        params["sort"] = sort

    data = _ckan_get_json(CKAN_DATASTORE_SEARCH_URL, params, timeout_seconds=timeout_seconds)
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

    IMPORTANT:
      - Overall/no breakdown must be expressed as DEMCODE == "" (empty string),
        because open.canada.ca does not expose datastore_search_sql and we cannot
        use (IS NULL OR '') logic.
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


def get_available_survey_years(
    *,
    timeout_seconds: int = 60,
    resource_id: Optional[str] = None,
    max_scan_rows: int = 50_000,
    page_size: int = 5_000,
) -> List[int]:
    """
    Discover available SURVEYR values WITHOUT SQL.

    We sample a limited number of rows requesting only SURVEYR and return distinct years.
    This avoids scanning the full 15M+ table.
    """
    rid = _resolve_resource_id(resource_id)

    years: set[int] = set()
    offset = 0
    page_size = max(100, min(int(page_size), 10_000))
    max_scan_rows = max(1_000, int(max_scan_rows))

    while offset < max_scan_rows:
        page = _datastore_search(
            resource_id=rid,
            filters=None,
            fields=["SURVEYR"],
            sort=None,
            limit=page_size,
            offset=offset,
            include_total=False,
            timeout_seconds=timeout_seconds,
        )

        if not page.records:
            break

        for r in page.records:
            v = r.get("SURVEYR")
            try:
                years.add(int(v))
            except Exception:
                continue

        # If years stabilize quickly, stop early (heuristic)
        if len(years) >= 4:
            # For current dataset this is typically enough; keep it conservative.
            pass

        offset += page_size

    return sorted(years)
