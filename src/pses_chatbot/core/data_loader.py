from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Union

import pandas as pd
import requests

from pses_chatbot.config import CKAN_DATASTORE_SEARCH_URL, PSES_RESOURCE_ID


class DataLoaderError(Exception):
    """Raised when the CKAN datastore cannot be queried reliably."""


@dataclass(frozen=True)
class DataStoreResponse:
    records: List[Dict[str, Any]]
    total: int
    fields: Optional[List[Dict[str, Any]]] = None


def _clean_filters(filters: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    CKAN datastore_search expects a JSON object for `filters`.
    We also preserve empty-string values (e.g., DEMCODE = "") intentionally.
    We remove only keys whose value is None.
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
    """
    Build a session with sane defaults. We keep it simple (no urllib3 Retry object)
    and implement retries ourselves so we can adapt page sizes on timeouts.
    """
    s = requests.Session()
    # helpful headers; CKAN doesn't require them, but they can improve behavior
    s.headers.update(
        {
            "Accept": "application/json",
            "User-Agent": "pses-conversational-chatbot/1.0",
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
    include_total: bool = True,
    timeout_seconds: int = 90,
    max_retries: int = 4,
    backoff_seconds: float = 1.0,
    adaptive_limit_floor: int = 500,
) -> DataStoreResponse:
    """
    Calls CKAN datastore_search reliably.

    Key behaviors:
      - Sends filters as JSON object (CKAN-correct).
      - Retries on timeouts / transient errors with exponential backoff.
      - If a timeout occurs, automatically reduces `limit` and retries.
    """
    sess = _session()

    cleaned_filters = _clean_filters(filters)

    # CKAN params
    params: Dict[str, Any] = {
        "resource_id": resource_id,
        "offset": int(offset),
        "limit": int(limit),
    }
    if cleaned_filters is not None:
        params["filters"] = json.dumps(cleaned_filters, ensure_ascii=False)
    if fields:
        # CKAN supports "fields" as comma-separated list or repeated param.
        # Use comma-separated for consistency.
        params["fields"] = ",".join(fields)
    if sort:
        params["sort"] = sort
    if include_total:
        params["include_total"] = "true"

    attempt = 0
    cur_limit = int(limit)

    while True:
        attempt += 1
        params["limit"] = cur_limit

        try:
            resp = sess.get(CKAN_DATASTORE_SEARCH_URL, params=params, timeout=timeout_seconds)
            resp.raise_for_status()
            payload = resp.json()

            if not isinstance(payload, dict) or not payload.get("success", False):
                raise DataLoaderError(f"CKAN datastore_search returned success=false: {payload}")

            result = payload.get("result", {})
            records = result.get("records", []) or []
            total = int(result.get("total", len(records)) or 0)
            fields_meta = result.get("fields")

            return DataStoreResponse(records=records, total=total, fields=fields_meta)

        except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectTimeout) as exc:
            # Timeout: reduce page size and retry
            if attempt > max_retries:
                raise DataLoaderError(
                    f"HTTP timeout calling datastore_search after {max_retries} retries: {exc}"
                ) from exc

            # Adaptive paging: halve the limit down to a floor.
            new_limit = max(adaptive_limit_floor, cur_limit // 2)
            cur_limit = new_limit

            sleep_for = backoff_seconds * (2 ** (attempt - 1))
            time.sleep(sleep_for)
            continue

        except requests.exceptions.HTTPError as exc:
            # CKAN sometimes throws 500/502 transiently
            status = getattr(exc.response, "status_code", None)
            if status in (429, 500, 502, 503, 504) and attempt <= max_retries:
                # On server error, also reduce limit a bit (helps large queries)
                cur_limit = max(adaptive_limit_floor, int(cur_limit * 0.7))
                sleep_for = backoff_seconds * (2 ** (attempt - 1))
                time.sleep(sleep_for)
                continue

            raise DataLoaderError(f"HTTP error while calling datastore_search: {exc}") from exc

        except requests.exceptions.RequestException as exc:
            # Other transient network issues
            if attempt <= max_retries:
                sleep_for = backoff_seconds * (2 ** (attempt - 1))
                time.sleep(sleep_for)
                continue
            raise DataLoaderError(f"Network error while calling datastore_search: {exc}") from exc

        finally:
            try:
                sess.close()
            except Exception:
                pass


def query_pses_results(
    *,
    filters: Optional[Dict[str, Any]] = None,
    fields: Optional[List[str]] = None,
    sort: Optional[str] = None,
    max_rows: int = 50_000,
    page_size: int = 5_000,
    resource_id: Optional[str] = None,
    timeout_seconds: int = 90,
) -> pd.DataFrame:
    """
    Fetch a slice of PSES results from CKAN DataStore, using pagination.

    Notes:
      - This is NOT aggregation: the datastore is already aggregated.
      - We paginate and concatenate up to max_rows.
      - Filters must include the required query keys at the call site
        (SURVEYR, QUESTION, DEMCODE, LEVEL1ID..LEVEL5ID).

    Args:
      filters: CKAN filters dict (sent as JSON). Keep DEMCODE="" when overall.
      fields: optional list of fields to return.
      sort: optional CKAN sort string.
      max_rows: hard cap to prevent loading too much.
      page_size: requested per page; will be adaptively reduced on timeout.
      resource_id: defaults to configured PSES_RESOURCE_ID.
      timeout_seconds: per-request timeout.

    Returns:
      DataFrame of records (may be empty).
    """
    rid = resource_id or PSES_RESOURCE_ID

    # Keep page size reasonable; large limits can trigger slow responses.
    # Your query_engine currently uses up to 50,000; that is often too big for CKAN reliably.
    page_limit = int(max(500, min(page_size, 10_000)))

    all_records: List[Dict[str, Any]] = []
    offset = 0
    total: Optional[int] = None

    while True:
        remaining = max_rows - len(all_records)
        if remaining <= 0:
            break

        # Never request more than remaining rows
        limit = min(page_limit, remaining)

        page = _datastore_search(
            resource_id=rid,
            filters=filters,
            fields=fields,
            sort=sort,
            offset=offset,
            limit=limit,
            include_total=True,
            timeout_seconds=timeout_seconds,
            max_retries=4,
            backoff_seconds=1.0,
            adaptive_limit_floor=500,
        )

        if total is None:
            total = page.total

        if not page.records:
            break

        all_records.extend(page.records)
        offset += len(page.records)

        # Stop if we reached total (when CKAN provides it)
        if total is not None and offset >= total:
            break

        # Safety stop: if CKAN returns fewer than requested, we may be at the end
        if len(page.records) < limit:
            break

    if not all_records:
        return pd.DataFrame()

    return pd.DataFrame.from_records(all_records)
