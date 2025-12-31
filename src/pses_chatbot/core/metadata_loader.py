from __future__ import annotations

from pathlib import Path
from typing import Optional, List, Dict, Any

import logging

import pandas as pd

from pses_chatbot.config import METADATA_DIR

logger = logging.getLogger(__name__)

# Name we expect for the metadata workbook.
# Place your Excel file here:
#   data/metadata/2024_PSES_Metadata.xlsx
METADATA_WORKBOOK_NAME = "2024_PSES_Metadata.xlsx"

# In-memory caches
_METADATA_XLS: Optional[pd.ExcelFile] = None
_QUESTIONS_CACHE: Optional[pd.DataFrame]
_SCALES_CACHE: Optional[p.DataFrame]
_DEM_CACHE: Optional[pd.DataFrame]
_ORG_CACHE: Optional[pd.DataFrame]
_POSNEG_CACHE: Optional[pd.DataFrame]

_QUESTIONS_CACHE = None
_SCALES_CACHE = None
_DEM_CACHE = None
_ORG_CACHE = None
_POSNEG_CACHE = None


def _metadata_workbook_path() -> Path:
    return Path(METADATA_DIR) / METADATA_WORKBOOK_NAME


def _get_metadata_workbook(refresh: bool = False) -> pd.ExcelFile:
    global _METADATA_XLS

    if _METADATA_XLS is not None and not refresh:
        return _METADATA_XLS

    path = _metadata_workbook_path()
    if not path.exists():
        raise FileNotFoundError(
            f"Metadata workbook not found: {path}. "
            f"Expected at: data/metadata/{METADATA_WORKBOOK_NAME}"
        )

    _METADATA_XLS = pd.ExcelFile(path)
    return _METADATA_XLS


# ---------------------------------------------------------------------------
# Questions metadata (QUESTIONS sheet)
# ---------------------------------------------------------------------------

def load_questions_meta(refresh: bool = False) -> pd.DataFrame:
    global _QUESTIONS_CACHE
    if _QUESTIONS_CACHE is not None and not refresh:
        return _QUESTIONS_CACHE

    xls = _get_metadata_workbook(refresh=refresh)
    df = xls.parse("QUESTIONS", header=0)

    # Normalize columns to expected
    lower = {c.lower(): c for c in df.columns}
    q_col = lower.get("question")
    en_col = lower.get("question_en")
    fr_col = lower.get("question_fr")

    if not (q_col and en_col and fr_col):
        raise ValueError("QUESTIONS sheet does not contain expected columns (QUESTION, QUESTION_EN, QUESTION_FR).")

    out = pd.DataFrame()
    out["question"] = df[q_col].astype(str).str.strip()
    out["question_en"] = df[en_col].astype(str).str.strip()
    out["question_fr"] = df[fr_col].astype(str).str.strip()

    _QUESTIONS_CACHE = out
    return out


# ---------------------------------------------------------------------------
# Scales metadata (SCALES sheet)
# ---------------------------------------------------------------------------

def load_scales_meta(refresh: bool = False) -> pd.DataFrame:
    global _SCALES_CACHE
    if _SCALES_CACHE is not None and not refresh:
        return _SCALES_CACHE

    xls = _get_metadata_workbook(refresh=refresh)
    df = xls.parse("SCALES", header=0)

    lower = {c.lower(): c for c in df.columns}
    scale_col = lower.get("scale")
    name_en_col = lower.get("name_en")
    name_fr_col = lower.get("name_fr")
    min_col = lower.get("min")
    max_col = lower.get("max")

    if not (scale_col and name_en_col and name_fr_col):
        raise ValueError("SCALES sheet does not contain expected columns (SCALE, NAME_EN, NAME_FR).")

    out = pd.DataFrame()
    out["scale"] = df[scale_col].astype(str).str.strip()
    out["name_en"] = df[name_en_col].astype(str).str.strip()
    out["name_fr"] = df[name_fr_col].astype(str).str.strip()

    if min_col:
        out["min"] = df[min_col]
    if max_col:
        out["max"] = df[max_col]

    _SCALES_CACHE = out
    return out


# ---------------------------------------------------------------------------
# Demographics metadata (DEMCODE sheet)
# ---------------------------------------------------------------------------

def load_demographics_meta(refresh: bool = False) -> pd.DataFrame:
    """
    Load demographics metadata from the 'DEMCODE' sheet.

    Expected columns (as in your workbook):
      - 'DEMCODE 2024'  (numeric code used in the dataset)
      - 'BYCOND'        (definition, e.g. 'Q82 = 1')
      - 'DESCRIP_E'     (label EN)
      - 'DESCRIP_F'     (label FR)
      - 'Category_E'    (category EN, e.g. 'Gender')
      - 'Category_F'    (category FR, e.g. 'Genre')

    Normalized columns:
      - demcode             (string, e.g., '1001')
      - bycond              (string definition)
      - label_en            (English label)
      - label_fr            (French label)
      - category_en         (English category)
      - category_fr         (French category)
      - dimension_question  (e.g., 'Q82', extracted from BYCOND if present)
...
    """
    global _DEM_CACHE
    if _DEM_CACHE is not None and not refresh:
        return _DEM_CACHE

    xls = _get_metadata_workbook(refresh=refresh)
    df = xls.parse("DEMCODE", header=0)

    # Normalize columns
    lower = {c.lower(): c for c in df.columns}
    dem_col = lower.get("demcode 2024")
    bycond_col = lower.get("bycond")
    en_col = lower.get("descrip_e")
    fr_col = lower.get("descrip_f")
    cat_en_col = lower.get("category_e")
    cat_fr_col = lower.get("category_f")

    if not (dem_col and bycond_col and en_col and fr_col):
        raise ValueError(
            "DEMCODE sheet does not contain the expected columns "
            "('DEMCODE 2024', 'BYCOND', 'DESCRIP_E', 'DESCRIP_F')."
        )

    out = pd.DataFrame()
    out["demcode"] = df[dem_col].astype(str).str.strip()
    out["bycond"] = df[bycond_col].astype(str).str.strip()
    out["label_en"] = df[en_col].astype(str).str.strip()
    out["label_fr"] = df[fr_col].astype(str).str.strip()

    # Category (demographic variable name) is used for UI grouping and audit.
    # Keep as optional to tolerate workbook variations.
    out["category_en"] = df[cat_en_col].astype(str).str.strip() if cat_en_col else ""
    out["category_fr"] = df[cat_fr_col].astype(str).str.strip() if cat_fr_col else ""

    # Extract base question (dimension) from BYCOND, e.g., 'Q82' from 'Q82 = 1'
    out["dimension_question"] = (
        out["bycond"].str.extract(r"(Q\d+)", expand=False).astype(str).str.strip()
    )

    _DEM_CACHE = out
    return out


# ---------------------------------------------------------------------------
# Organization metadata (LEVEL1ID_LEVEL5ID sheet)
# ---------------------------------------------------------------------------

def load_org_meta(refresh: bool = False) -> pd.DataFrame:
    global _ORG_CACHE
    if _ORG_CACHE is not None and not refresh:
        return _ORG_CACHE

    xls = _get_metadata_workbook(refresh=refresh)
    df = xls.parse("LEVEL1ID_LEVEL5ID", header=0)

    # Leave as-is; app handles cascade labels.
    _ORG_CACHE = df
    return df


# ---------------------------------------------------------------------------
# POS/NEG metadata (POSNEG sheet)
# ---------------------------------------------------------------------------

def load_posneg_meta(refresh: bool = False) -> pd.DataFrame:
    global _POSNEG_CACHE
    if _POSNEG_CACHE is not None and not refresh:
        return _POSNEG_CACHE

    xls = _get_metadata_workbook(refresh=refresh)
    df = xls.parse("POSNEG", header=0)

    _POSNEG_CACHE = df
    return df
