from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

import pandas as pd
import requests

from pses_chatbot.config import PSES_DATASTORE_RESOURCE_ID

logger = logging.getLogger(__name__)

CKAN_DATASTORE_SEARCH_URL = (
    "https://open.canada.ca/data/en/api/3/action/datastore_search"
)


class DataLoaderError(Exception):
    """Custom exception for DataStore query failures."""


def _datastore_search(
    resource_id: str,
    filters: Optional[Dict[str, Any]] = None,
    fields: Optional[List[str]] = None,
    sort: Optional[str] = None,
    offset: int = 0,
    limit: int = 1000,
) -> Dict[str, Any]:
    """
    Low-level wrapper around CKAN's datastore_search.

    IMPORTANT:
      - CKAN expects 'filters' as a JSON-encoded string.
      - If we pass a dict directly, requests will serialize it incorrectly
        (e.g., filters=QUESTION&filters=DEMCODE...), causing 500 errors.

    Parameters
    ----------
    resource_id : str
        CKAN resource ID (PSES main table).
    filters : dict, optional
        Dict of equality filters, e.g. {"QUESTION": "Q08", "SURVEYR": 2024}.
    fields : list[str], optional
        Subset of columns to return.
    sort : str, optional
        Sort expression (CKAN style), e.g. "SURVEYR asc".
    offset : int
        Starting offset for paging.
    limit : int
        Max number of rows to return in this call.

    Returns
    -------
    dict
        The 'result' object from CKAN's JSON response.
    """
    params: Dict[str, Any] = {
        "resource_id": resource_id,
        "offset": offset,
        "limit": limit,
    }

    # Correct handling of filters: must be JSON-encoded
    if filters is not None:
        params["filters"] = json.dumps(filters)

    if fields:
        # CKAN allows comma-separated list of field names
        params["fields"] = ",".join(fields)
    if sort:
        params["sort"] = sort

    logger.info(
        "Calling CKAN datastore_search with params (offset=%s, limit=%s, filters=%s)",
        offset,
        limit,
        filters,
    )

    try:
        resp = requests.get(CKAN_DATASTORE_SEARCH_URL, params=params, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as exc:
        logger.error("HTTP error while calling datastore_search: %s", exc)
        raise DataLoaderError(f"HTTP error while calling datastore_search: {exc}") from exc

    try:
        payload = resp.json()
    except ValueError as exc:
        logger.error("Invalid JSON from datastore_search: %s", exc)
        raise DataLoaderError(f"Invalid JSON from datastore_search: {exc}") from exc

    if not payload.get("success", False):
        logger.error("CKAN reported failure: %s", payload)
        raise DataLoaderError(f"CKAN reported failure: {payload}")

    result = payload.get("result", {})
    return result


def query_pses_results(
    resource_id: Optional[str] = None,
    filters: Optional[Dict[str, Any]] = None,
    fields: Optional[List[str]] = None,
    sort: Optional[str] = None,
    max_rows: int = 50_000,
    page_size: int = 10_000,
) -> pd.DataFrame:
    """
    Higher-level helper to fetch PSES records from the DataStore.

    This function:
      - Uses the configured PSES_DATASTORE_RESOURCE_ID by default.
      - Applies equality filters if provided.
      - Pages through the DataStore until:
          * we have fetched 'max_rows' records, OR
          * no more records are returned.
      - Returns a pandas DataFrame (may be empty).

    Parameters
    ----------
    resource_id : str, optional
        CKAN resource ID. If None, uses PSES_DATASTORE_RESOURCE_ID from config.
    filters : dict, optional
        Dict of equality filters, e.g. {"QUESTION": "Q08", "SURVEYR": 2024}.
    fields : list[str], optional
        Limit returned columns to this subset.
    sort : str, optional
        CKAN sort expression, e.g. "SURVEYR asc".
    max_rows : int
        Hard cap on total rows to fetch.
    page_size : int
        Number of rows to request per call to datastore_search.

    Returns
    -------
    pd.DataFrame
        DataFrame with the collected records.
    """
    if resource_id is None:
        resource_id = PSES_DATASTORE_RESOURCE_ID

    if not resource_id:
        raise DataLoaderError("PSES_DATASTORE_RESOURCE_ID is not configured.")

    if page_size <= 0:
        raise ValueError("page_size must be positive.")
    if max_rows <= 0:
        raise ValueError("max_rows must be positive.")

    all_records: List[Dict[str, Any]] = []
    offset = 0

    while len(all_records) < max_rows:
        remaining = max_rows - len(all_records)
        page_limit = min(page_size, remaining)

        result = _datastore_search(
            resource_id=resource_id,
            filters=filters,
            fields=fields,
            sort=sort,
            offset=offset,
            limit=page_limit,
        )

        records = result.get("records", [])
        if not records:
            # No more data available
            break

        all_records.extend(records)
        offset += len(records)

        # If returned fewer than requested, we've hit the end
        if len(records) < page_limit:
            break

    if not all_records:
        return pd.DataFrame()

    df = pd.DataFrame.from_records(all_records)
    return df
