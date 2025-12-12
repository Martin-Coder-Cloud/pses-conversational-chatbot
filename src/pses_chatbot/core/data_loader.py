from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set

import pandas as pd
import requests

import pses_chatbot.config as cfg


class DataLoaderError(Exception):
    """Raised when the CKAN datastore cannot be queried reliably."""


@dataclass(frozen=True)
class DataStoreResponse:
    records: List[Dict[str, Any]]
    total: int
    fields: Optional[List[Dict[str, Any]]] = None


def _resolve_ckan_search_url() -> str:
    url = getattr(cfg, "CKAN_DATASTORE_SEARCH_URL", "").strip()
    if not url:
        return "https://open.canada.ca/data/en/api/3/action/datastore_search"
    return url


def _resolve_ckan_sql_url() -> str:
    base = getattr(cfg, "CKAN_BASE_URL", "").strip()
    if base:
        return f"{base}/datastore_search_sql"

    search = _resolve_ckan_search_url()
    if search.endswith("/datastore_search"):
        return search[: -len("/datastore_search")] + "/datastore_search_sql"
    return "https://open.canada.ca/data/en/api/3/action/datastore_search_sql"


def _resolve_resource_id() -> str:
    rid = getattr(cfg, "PSES_DATASTORE_RESOURCE_ID", "").strip()
    if not rid:
        raise DataLoaderError("Missing CKAN resource id in config.py. Expected: PSES_DATASTORE_RESOURCE_ID")
    return rid


CKAN_DATASTORE_SEARCH_URL = _resolve_ckan_search_url()
CKAN_DATASTORE_SEARCH_SQL_URL = _resolve_ckan_sql_url()


def _clean_filters(filters: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    CKAN datastore_search expects `filters` as a JSON object (equality only).

    Preserve empty-string values intentionally.
    Remove only keys whose value is None (caller may use None as sentinel).
    """
    if not filters:
        return None
    out: Dict[str, Any] = {}
    for k, v in filters.items():
        if v is None:
            continue
        out[k] = v
    return out if out else None


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update(
        {
            "Accept": "application/json",
            "User-Agent": "pses-conversational-chatbot/0.1.0",
        }
    )
    return s


def _datastore_search(
    *,
    resource_id: str,
    filters: Optional[Dict[str, Any]] = None,
    fields: Optional[List[str]] = None,
    sort: Optional[str] = None,
    offset: int = 0,
    limit: int = 5_000,
    include_total: bool = False,
    timeout_seconds: int = 90,
    max_retries: int = 4,
    backoff_seconds: float = 1.0,
    adaptive_limit_floor: int = 500,
) -> DataStoreResponse:
    """
    Calls CKAN datastore_search reliably.

    - Sends filters as JSON object (CKAN-correct).
    - Retries on timeouts/transient errors with exponential backoff.
    - On timeout, automatically reduces `limit` and retries.

    NOTE: include_total defaults to False for analytical queries (faster/safer).
    """
    sess = _session()
    cleaned_filters = _clean_filters(filters)

    params: Dict[str, Any] = {
        "resource_id": resource_id,
        "offset": int(offset),
        "limit": int(limit),
    }
    if cleaned_filters is not None:
        params["filters"] = json.dumps(cleaned_filters, ensure_ascii=False)
    if fields:
        params["fields"] = ",".join(fields)
    if sort:
        params["sort"] = sort
    if include_total:
        params["include_total"] = "true"

    attempt = 0
    cur_limit = int(limit)

    try:
        while True:
            attempt += 1
            params["limit"] = cur_limit

            try:
                resp = sess.get(CKAN_DATASTORE_SEARCH_URL, params=params, timeout=timeout_seconds)
                resp.raise_for_status()
                payload = resp.json()

                if not isinstance(payload, dict) or not payload.get("success", False):
                    raise DataLoaderError(f"CKAN datastore_search returned success=false: {payload}")

                result = payload.get("result", {}) or {}
                records = result.get("records", []) or []
                total = int(result.get("total", len(records)) or 0)
                fields_meta = result.get("fields")

                return DataStoreResponse(records=records, total=total, fields=fields_meta)

            except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectTimeout) as exc:
                if attempt > max_retries:
                    raise DataLoaderError(
                        f"HTTP timeout calling datastore_search after {max_retries} retries: {exc}"
                    ) from exc
                cur_limit = max(adaptive_limit_floor, cur_limit // 2)
                time.sleep(backoff_seconds * (2 ** (attempt - 1)))
                continue

            except requests.exceptions.HTTPError as exc:
                status = getattr(exc.response, "status_code", None)
                if status in (429, 500, 502, 503, 504) and attempt <= max_retries:
                    cur_limit = max(adaptive_limit_floor, int(cur_limit * 0.7))
                    time.sleep(backoff_seconds * (2 ** (attempt - 1)))
                    continue
                raise DataLoaderError(f"HTTP error while calling datastore_search: {exc}") from exc

            except requests.exceptions.RequestException as exc:
                if attempt <= max_retries:
                    time.sleep(backoff_seconds * (2 ** (attempt - 1)))
                    continue
                raise DataLoaderError(f"Network error while calling datastore_search: {exc}") from exc
    finally:
        try:
            sess.close()
        except Exception:
            pass


def _datastore_search_sql(
    *,
    sql: str,
    timeout_seconds: int = 90,
    max_retries: int = 4,
    backoff_seconds: float = 1.0,
) -> List[Dict[str, Any]]:
    """
    Calls CKAN datastore_search_sql with retries and returns records.
    """
    sess = _session()
    params = {"sql": sql}

    attempt = 0
    try:
        while True:
            attempt += 1
            try:
                resp = sess.get(CKAN_DATASTORE_SEARCH_SQL_URL, params=params, timeout=timeout_seconds)
                resp.raise_for_status()
                payload = resp.json()

                if not isinstance(payload, dict) or not payload.get("success", False):
                    raise DataLoaderError(f"CKAN datastore_search_sql returned success=false: {payload}")

                result = payload.get("result", {}) or {}
                return result.get("records", []) or []

            except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectTimeout) as exc:
                if attempt > max_retries:
                    raise DataLoaderError(
                        f"HTTP timeout calling datastore_search_sql after {max_retries} retries: {exc}"
                    ) from exc
                time.sleep(backoff_seconds * (2 ** (attempt - 1)))
                continue

            except requests.exceptions.HTTPError as exc:
                status = getattr(exc.response, "status_code", None)
                if status in (429, 500, 502, 503, 504) and attempt <= max_retries:
                    time.sleep(backoff_seconds * (2 ** (attempt - 1)))
                    continue
                raise DataLoaderError(f"HTTP error while calling datastore_search_sql: {exc}") from exc

            except requests.exceptions.RequestException as exc:
                if attempt <= max_retries:
                    time.sleep(backoff_seconds * (2 ** (attempt - 1)))
                    continue
                raise DataLoaderError(f"Network error while calling datastore_search_sql: {exc}") from exc
    finally:
        try:
            sess.close()
        except Exception:
            pass


def get_available_survey_years(
    *,
    resource_id: Optional[str] = None,
    timeout_seconds: int = 90,
) -> List[int]:
    """
    Discover available SURVEYR values in the DataStore resource.

    Preferred: SQL DISTINCT (fast).
    Fallback: paged datastore_search for SURVEYR only.
    """
    rid = (resource_id or _resolve_resource_id()).strip()

    # Preferred: SQL DISTINCT
    try:
        sql = f'SELECT DISTINCT "SURVEYR" AS year FROM "{rid}" WHERE "SURVEYR" IS NOT NULL ORDER BY "SURVEYR"'
        records = _datastore_search_sql(sql=sql, timeout_seconds=timeout_seconds)
        years: List[int] = []
        for r in records:
            v = r.get("year", r.get("SURVEYR"))
            if v is None:
                continue
            try:
                years.append(int(v))
            except Exception:
                continue
        years = sorted(set(years))
        if years:
            return years
    except Exception:
        pass

    # Fallback scan (fields=["SURVEYR"]) with no totals
    years_set: Set[int] = set()
    offset = 0
    page = 10_000

    while True:
        resp = _datastore_search(
            resource_id=rid,
            filters=None,
            fields=["SURVEYR"],
            sort=None,
            offset=offset,
            limit=page,
            include_total=False,
            timeout_seconds=timeout_seconds,
        )
        if not resp.records:
            break

        for r in resp.records:
            v = r.get("SURVEYR")
            if v is None:
                continue
            try:
                years_set.add(int(v))
            except Exception:
                continue

        offset += len(resp.records)
        if len(resp.records) < page:
            break
        if len(years_set) >= 25:
            break

    return sorted(years_set)


def query_pses_results(
    *,
    filters: Optional[Dict[str, Any]] = None,
    fields: Optional[List[str]] = None,
    sort: Optional[str] = None,
    max_rows: int = 50_000,
    page_size: int = 5_000,
    resource_id: Optional[str] = None,
    timeout_seconds: int = 90,
    include_total: bool = False,
) -> pd.DataFrame:
    """
    Fetch a slice from CKAN datastore_search with pagination.

    - No aggregation is performed.
    - include_total defaults to False (better for large tables).
    """
    rid = (resource_id or _resolve_resource_id()).strip()

    page_limit = int(max(200, min(int(page_size), 10_000)))

    all_records: List[Dict[str, Any]] = []
    offset = 0
    total: Optional[int] = None

    while True:
        remaining = max_rows - len(all_records)
        if remaining <= 0:
            break

        limit = min(page_limit, remaining)

        page = _datastore_search(
            resource_id=rid,
            filters=filters,
            fields=fields,
            sort=sort,
            offset=offset,
            limit=limit,
            include_total=include_total,
            timeout_seconds=timeout_seconds,
            max_retries=4,
            backoff_seconds=1.0,
            adaptive_limit_floor=200,
        )

        if include_total and total is None:
            total = page.total

        if not page.records:
            break

        all_records.extend(page.records)
        offset += len(page.records)

        if include_total and total is not None and offset >= total:
            break

        if len(page.records) < limit:
            break

    if not all_records:
        return pd.DataFrame()

    return pd.DataFrame.from_records(all_records)


def query_pses_results_overall_sql(
    *,
    question_code: str,
    survey_year: int,
    org_levels: Dict[str, int],
    fields: Optional[List[str]] = None,
    resource_id: Optional[str] = None,
    timeout_seconds: int = 90,
) -> pd.DataFrame:
    """
    Special-case query for OVERALL (no breakdown) rows when DEMCODE may be NULL or ''.

    Uses datastore_search_sql:
      WHERE QUESTION = 'Q08'
        AND SURVEYR = 2018
        AND LEVEL1ID..LEVEL5ID = ...
        AND (DEMCODE IS NULL OR DEMCODE = '')

    No aggregation. Returns rows as-is.
    """
    rid = (resource_id or _resolve_resource_id()).strip()

    q = str(question_code).strip().replace("'", "''")
    year = int(survey_year)

    where_parts = [
        f"\"QUESTION\" = '{q}'",
        f"\"SURVEYR\" = {year}",
        "(\"DEMCODE\" IS NULL OR \"DEMCODE\" = '')",
    ]

    for col, val in org_levels.items():
        if col in ("LEVEL1ID", "LEVEL2ID", "LEVEL3ID", "LEVEL4ID", "LEVEL5ID"):
            where_parts.append(f"\"{col}\" = {int(val)}")

    cols = "*"
    if fields:
        safe_cols = [f"\"{c}\"" for c in fields]
        cols = ", ".join(safe_cols)

    sql = f"SELECT {cols} FROM \"{rid}\" WHERE " + " AND ".join(where_parts)

    records = _datastore_search_sql(sql=sql, timeout_seconds=timeout_seconds)
    if not records:
        return pd.DataFrame()
    return pd.DataFrame.from_records(records)
