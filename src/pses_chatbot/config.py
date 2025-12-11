from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

# Root of the project (repo root)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = DATA_DIR / "pses_results"   # for any local result files if needed
METADATA_DIR = DATA_DIR / "metadata"      # local metadata files
CACHE_DIR = DATA_DIR / "cache"            # cache directory (for small artifacts if needed)

CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# App identity
# ---------------------------------------------------------------------------

APP_NAME = "PSES Conversational Analytics Chatbot"
APP_VERSION = "0.1.0"

# ---------------------------------------------------------------------------
# Open Data API configuration (for metadata or alternate CSV endpoints)
#
# These are optional and mainly intended for future use with metadata_loader.
# ---------------------------------------------------------------------------

# Optional CSV endpoints (if you ever expose them directly)
PSES_RESULTS_URL = os.getenv("PSES_RESULTS_URL", "").strip()

# Optional metadata endpoints (questions, scales, demographics, org structure)
PSES_QUESTIONS_URL = os.getenv("PSES_QUESTIONS_URL", "").strip()
PSES_SCALES_URL = os.getenv("PSES_SCALES_URL", "").strip()
PSES_DEMOGRAPHICS_URL = os.getenv("PSES_DEMOGRAPHICS_URL", "").strip()
PSES_ORG_URL = os.getenv("PSES_ORG_URL", "").strip()

# ---------------------------------------------------------------------------
# CKAN / DataStore configuration for PSES main dataset
#
# We will use the CKAN DataStore API on open.canada.ca:
#   https://open.canada.ca/data/en/api/3/action/datastore_search
#
# IMPORTANT:
#   - You must supply the *resource_id* (NOT the package/dataset ID).
#   - For your dataset:
#       package (dataset) ID: 7f625e97-9d02-4c12-a756-1ddebb50e69f
#       resource (table) ID:  7c89b939-d99a-4bd1-b7d3-caefa61db84e  (main results table)
#   - We only ever query slices by filters; we never load the full 15M+ rows.
# ---------------------------------------------------------------------------

CKAN_BASE_URL = "https://open.canada.ca/data/en/api/3/action"
CKAN_DATASTORE_SEARCH_URL = f"{CKAN_BASE_URL}/datastore_search"

# Resource ID for the PSES 2024 results table (DataStore-enabled resource)
# You can override this via environment variable or Streamlit secrets.
PSES_DATASTORE_RESOURCE_ID = os.getenv(
    "PSES_DATASTORE_RESOURCE_ID",
    "7c89b939-d99a-4bd1-b7d3-caefa61db84e",  # default to the main resource you provided
).strip()
