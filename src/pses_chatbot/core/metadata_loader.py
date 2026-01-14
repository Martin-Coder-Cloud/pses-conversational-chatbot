from __future__ import annotations

from pathlib import Path
from typing import Optional
import logging

import pandas as pd

from pses_chatbot.config import METADATA_DIR

logger = logging.getLogger(__name__)

METADATA_WORKBOOK_NAME = "2024_PSES_Metadata.xlsx"

_METADATA_XLS: Optional[pd.ExcelFile] = None
_QUESTIONS_CACHE: Optional[pd.DataFrame] = None
_SCALES_CACHE: Optional[pd.DataFrame] = None
_DEM_CACHE: Optional[pd.DataFrame] = None
_ORG_CACHE: Optional[pd.DataFrame] = None
_POSNEG_CACHE: Optional[pd.DataFrame] = None


def _metadata_workbook_path() -> Path:
    return Path(METADATA_DIR) / METADATA_WORKBOOK_NAME


def _get_metadata_workbook(refresh: bool = False) -> pd.ExcelFile:
    global _METADATA_XLS
    if _METADATA_XLS is not None and not refresh:
        return _METADATA_XLS

    path = _metadata_workbook_path()
    if not path.exists():
        raise FileNotFoundError(f"Metadata workbook not found: {path}")

    _METADATA_XLS = pd.ExcelFile(path)
    return _METADATA_XLS


def _norm_text(x: object) -> str:
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    return str(x).replace("\u00A0", " ").strip()


def _strip_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Excel sometimes includes trailing spaces or blank columns (e.g., 'Unnamed: 5').
    We normalize by stripping whitespace from column names only.
    """
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


# ------------------------------------------------------------------
# QUESTIONS (approved)
# ------------------------------------------------------------------

def load_questions_meta(refresh: bool = False) -> pd.DataFrame:
    """
    Sheet: QUESTIONS
    Headers (confirmed):
      - 'Question number / numéro de la question'
      - 'English'
      - 'Français'

    Output:
      - code
      - text_en
      - text_fr
    """
    global _QUESTIONS_CACHE
    if _QUESTIONS_CACHE is not None and not refresh:
        return _QUESTIONS_CACHE

    xls = _get_metadata_workbook(refresh=refresh)
    df = _strip_column_names(xls.parse("QUESTIONS", header=0))

    code_col = "Question number / numéro de la question"
    en_col = "English"
    fr_col = "Français"

    missing = [c for c in [code_col, en_col, fr_col] if c not in df.columns]
    if missing:
        raise ValueError(
            f"QUESTIONS sheet missing required columns: {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    out = pd.DataFrame()
    out["code"] = df[code_col].apply(_norm_text).str.upper()
    out["text_en"] = df[en_col].apply(_norm_text)
    out["text_fr"] = df[fr_col].apply(_norm_text)

    _QUESTIONS_CACHE = out
    return out


# ------------------------------------------------------------------
# SCALES (pass-through)
# ------------------------------------------------------------------

def load_scales_meta(refresh: bool = False) -> pd.DataFrame:
    global _SCALES_CACHE
    if _SCALES_CACHE is not None and not refresh:
        return _SCALES_CACHE

    xls = _get_metadata_workbook(refresh=refresh)
    df = _strip_column_names(xls.parse("RESPONSE OPTIONS DE RÉPONSES", header=0))

    _SCALES_CACHE = df
    return df


# ------------------------------------------------------------------
# DEMCODE (normalized)
# ------------------------------------------------------------------

def load_demographics_meta(refresh: bool = False) -> pd.DataFrame:
    """
    Sheet: DEMCODE
    Headers (confirmed):
      - 'DEMCODE 2024'
      - 'BYCOND'
      - 'DESCRIP_E'
      - 'DESCRIP_F'
      - 'Category_E'
      - (blank column may exist)
      - 'Category_F'

    Output (adds normalized columns, while keeping original columns too):
      - demcode
      - bycond
      - label_en
      - label_fr
      - category_en
      - category_fr
    """
    global _DEM_CACHE
    if _DEM_CACHE is not None and not refresh:
        return _DEM_CACHE

    xls = _get_metadata_workbook(refresh=refresh)
    df = _strip_column_names(xls.parse("DEMCODE", header=0))

    required = ["DEMCODE 2024", "BYCOND", "DESCRIP_E", "DESCRIP_F", "Category_E", "Category_F"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"DEMCODE sheet missing required columns: {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    out = df.copy()

    out["demcode"] = out["DEMCODE 2024"].apply(_norm_text)
    out["bycond"] = out["BYCOND"].apply(_norm_text)
    out["label_en"] = out["DESCRIP_E"].apply(_norm_text)
    out["label_fr"] = out["DESCRIP_F"].apply(_norm_text)
    out["category_en"] = out["Category_E"].apply(_norm_text)
    out["category_fr"] = out["Category_F"].apply(_norm_text)

    _DEM_CACHE = out
    return out


# ------------------------------------------------------------------
# ORG (normalized)
# ------------------------------------------------------------------

def load_org_meta(refresh: bool = False) -> pd.DataFrame:
    """
    Sheet: LEVEL1ID_LEVEL5ID
    Headers (confirmed):
      LEVEL1ID LEVEL2ID LEVEL3ID LEVEL4ID LEVEL5ID UNITID DESCRIP_E DESCRIP_F DEPT

    Output (adds normalized columns, while keeping original columns too):
      - org_name_en (from DESCRIP_E)
      - org_name_fr (from DESCRIP_F)
      - dept_code   (from DEPT)
    Also coerces LEVEL1ID..LEVEL5ID and UNITID to int (NaN -> 0).
    """
    global _ORG_CACHE
    if _ORG_CACHE is not None and not refresh:
        return _ORG_CACHE

    xls = _get_metadata_workbook(refresh=refresh)
    df = _strip_column_names(xls.parse("LEVEL1ID_LEVEL5ID", header=0))

    required = ["LEVEL1ID", "LEVEL2ID", "LEVEL3ID", "LEVEL4ID", "LEVEL5ID", "UNITID", "DESCRIP_E", "DESCRIP_F", "DEPT"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"LEVEL1ID_LEVEL5ID sheet missing required columns: {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    out = df.copy()

    # Coerce IDs to int to match cascade expectations
    for c in ["LEVEL1ID", "LEVEL2ID", "LEVEL3ID", "LEVEL4ID", "LEVEL5ID", "UNITID"]:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0).astype(int)

    # Normalized labels expected by UI/query layer
    out["org_name_en"] = out["DESCRIP_E"].apply(_norm_text)
    out["org_name_fr"] = out["DESCRIP_F"].apply(_norm_text)
    out["dept_code"] = out["DEPT"].apply(_norm_text)

    _ORG_CACHE = out
    return out


# ------------------------------------------------------------------
# POSNEG (pass-through)
# ------------------------------------------------------------------

def load_posneg_meta(refresh: bool = False) -> pd.DataFrame:
    global _POSNEG_CACHE
    if _POSNEG_CACHE is not None and not refresh:
        return _POSNEG_CACHE

    xls = _get_metadata_workbook(refresh=refresh)
    df = _strip_column_names(xls.parse("POSITIVE_NEUTRAL_NEGATIVE_AGREE", header=0))

    _POSNEG_CACHE = df
    return df
