from __future__ import annotations

from typing import Any, Dict, List, Optional

import logging

import pandas as pd
import requests

from pses_chatbot.config import (
    CKAN_DATASTORE_SEARCH_URL,
    PSES_DATASTORE_RESOURCE_ID,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Low-level CKAN DataStore client
# ---------------------------------------------------------------------------

def _datastore_search(
    resource_id: str,
    filters: Optional[Dict[str, Any]] = None,
    fields: Optional[List[str]] = None,
    q: Optional[str] = None,
    sort: Optional[str] = None,
    offset: int = 0,
    limit: int = 1_000,
) -> Dict[str, Any]:
    """
    Call CKAN datastore_search for a single page of results.

    Docs:
      https://open.canada.ca/data/en/api/3/action/datastore_search

    Parameters
    ----------
    resource_id : str
        CKAN resource_id (the table ID in the DataStore).
    filters : dict, optional
        CKAN 'filters' parameter, e.g. {"SURVEY_YEAR": 2024, "QSTCD": "Q08"}.
    fields : list of str, optional
        Restrict returned fields/columns to this subset (saves bandwidth).
    q : str, optional
        Full-text search query (not used in our core analytics, but available).
    sort : str, optional
        Sort expression, e.g. "SURVEY_YEAR asc".
    offset : int
        Row offset for pagination.
    limit : int
        Page size (max rows to return in this call).

    Returns
    -------
    dict
        Parsed JSON response from CKAN.
    """
    if not resource_id:
        raise RuntimeError(
            "PSES_DATASTORE_RESOURCE_ID is not configured. "
            "Set it as an environment variable or Streamlit secret."
        )

    params: Dict[str, Any] = {
        "resource_id": resource_id,
        "offset": offset,
        "limit": limit,
    }

    if filters:
        params["filters"] = filters
    if fields:
        # CKAN allows a comma-separated string for the 'fields' param
        params["fields"] = ",".join(fields)
    if q:
        params["q"] = q
    if sort:
        params["sort"] = sort

    logger.info(
        "CKAN datastore_search: resource=%s offset=%d limit=%d filters=%s",
        resource_id,
        offset,
        limit,
        filters,
    )

    resp = requests.get(CKAN_DATASTORE_SEARCH_URL, params=params, timeout=120)
    resp.raise_for_status()
    data = resp.json()

    if not data.get("success", False):
        raise RuntimeError(
            f"CKAN datastore_search failed: {data.get('error') or 'Unknown error'}"
        )

    return data


# ---------------------------------------------------------------------------
# Public API: query PSES results in slices
# ---------------------------------------------------------------------------

def query_pses_results(
    filters: Optional[Dict[str, Any]] = None,
    fields: Optional[List[str]] = None,
    sort: Optional[str] = None,
    max_rows: int = 200_000,
    page_size: int = 50_000,
) -> pd.DataFrame:
    """
    Query the PSES DataStore and return a DataFrame of matching rows.

    IMPORTANT:
    - This function is *not* meant to load the full dataset (15M+ rows).
    - It will paginate through datastore_search until:
        * it has collected `max_rows` records, OR
        * it has reached the server-reported total, whichever comes first.

    Parameters
    ----------
    filters : dict, optional
        CKAN 'filters' parameter, e.g. {"SURVEY_YEAR": 2024, "QSTCD": "Q08"}.
        The higher-level queries layer will build these filters from
        question codes, years, organization IDs (LEVEL1/2/3), and DEMCODEs.
    fields : list of str, optional
        Restrict returned columns to a subset (saves bandwidth and memory).
        Example: ["SURVEY_YEAR", "QSTCD", "LEVEL1ID", "DEMCODE", "PERCENT_POSITIVE", "N"]
    sort : str, optional
        Sort expression (rarely needed for our analytics use cases).
    max_rows : int
        Safety cap: maximum number of rows to fetch across all pages.
    page_size : int
        Number of rows per datastore_search call (server limit is typically 100_000).

    Returns
    -------
    pandas.DataFrame
        DataFrame with up to `max_rows` records matching the filters.
    """
    resource_id = PSES_DATASTORE_RESOURCE_ID
    if not resource_id:
        raise RuntimeError(
            "PSES_DATASTORE_RESOURCE_ID is not configured. "
            "Set it as an environment variable or Streamlit secret."
        )

    all_records: List[Dict[str, Any]] = []
    offset = 0
    total: Optional[int] = None

    while True:
        remaining = max_rows - len(all_records)
        if remaining <= 0:
            logger.info(
                "Reached max_rows=%d; stopping CKAN paging.", max_rows
            )
            break

        page_limit = min(page_size, remaining)

        page = _datastore_search(
            resource_id=resource_id,
            filters=filters,
            fields=fields,
            sort=sort,
            offset=offset,
            limit=page_limit,
        )

        result = page.get("result", {})
        if total is None:
            total = int(result.get("total", 0))

        records = result.get("records", [])
        if not records:
            logger.info(
                "No more records returned from CKAN (offset=%d); stopping.", offset
            )
            break

        all_records.extend(records)
        offset += len(records)

        logger.info(
            "Fetched %d records (cumulative=%d, total=%s)",
            len(records),
            len(all_records),
            total,
        )

        # Extra guard: if we've consumed all records up to total, stop
        if total is not None and offset >= total:
            break

    df = pd.DataFrame.from_records(all_records)
    logger.info(
        "query_pses_results: final DataFrame shape %s (filters=%s)",
        df.shape,
        filters,
    )
    return df
