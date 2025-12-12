from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

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


def _get_config_value(*names: str, default: Any = None) -> Any:
    """
    Return the first attribute found in pses_chatbot.config among `names`.
    """
    for n in names:
        if hasattr(cfg, n):
            return getattr(cfg, n)
    return default


def _resolve_ckan_search_url() -> str:
    # Prefer existing config names, but default to the canonical CKAN endpoint.
    return str(
        _get_config_value(
            "CKAN_DATASTORE_SEARCH_URL",
            "DATASTORE_SEARCH_URL",
            "CKAN_SEARCH_URL",
            "CKAN_API_DATASTORE_SEARCH_URL",
            default="https://open.canada.ca/data/en/api/3/action/datastore_search",
        )
    )


def _resolve_resource_id() -> str:
    rid = _get_config_value(
        # common names people use
        "PSES_RESOURCE_ID",
        "CKAN_RESOURCE_ID",
        "DATASTORE_RESOURCE_ID",
        "RESOURCE_ID",
        # if you stored it under a nested config object/dict, adjust later
        default=None,
    )
    if rid is None or str(rid).strip() == "":
        raise DataLoaderError(
            "Missing CKAN resource id in config.py. "
            "Expected one of: PSES_RESOURCE_ID, CKAN_RESOURCE_ID, DATASTORE_RESOURCE_ID, RESOURCE_ID."
        )
    return str(rid).strip()


CKAN_DATASTORE_SEARCH_URL = _resolve_ckan_search_url()


def _clean_filters(filters: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    CKAN datastore_search expects a JSON object for `filters`.
    We preserve empty-string values intentionally (e.g., DEMCODE="").
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
    s = requests.Session()
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

    - Sends filters as JSON object (CKAN-correct).
    - Retries on timeouts/transient errors with exponential backoff.
    - On timeout, automatically reduces `limit` and retries.
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


def query_pses_results(
    *,
    filters: Optional[Dict[str, Any]] = None,
    fields: Optional[List[str]] = None,
    sort: Optional[str] = None,
    max_rows: int = 50_000,
    page_size: int = 5_000,
    resource_id: Optional[str] = None,
    timeout_seconds: int = 90,
    include_total: bool = True,
) -> pd.DataFrame:
    """
    Fetch a slice of PSES results from CKAN DataStore, using pagination.

    Important:
      - Preserve DEMCODE="" when you want “no breakdown”.
      - This does not aggregate (data is already aggregated).
    """
    rid = (resource_id or _resolve_resource_id()).strip()

    page_limit = int(max(500, min(int(page_size), 10_000)))

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
            adaptive_limit_floor=500,
        )

        if total is None:
            total = page.total if include_total else None

        if not page.records:
            break

        all_records.extend(page.records)
        offset += len(page.records)

        if total is not None and offset >= total:
            break

        if len(page.records) < limit:
            break

    if not all_records:
        return pd.DataFrame()

    return pd.DataFrame.from_records(all_records)
