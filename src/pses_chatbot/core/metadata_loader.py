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


# ------------------------------------------------------------------
# QUESTIONS  ✅ SURGICAL FIX APPLIED HERE ONLY
# ------------------------------------------------------------------

def load_questions_meta(refresh: bool = False) -> pd.DataFrame:
    """
    QUESTIONS sheet expected headers:
      - 'Question number / numéro de la question'
      - 'English'
      - 'Français'

    Normalized output:
      - code
      - text_en
      - text_fr
    """
    global _QUESTIONS_CACHE
    if _QUESTIONS_CACHE is not None and not refresh:
        return _QUESTIONS_CACHE

    xls = _get_metadata_workbook(refresh=refresh)
    df = xls.parse("QUESTIONS", header=0)

    # Exact header names as confirmed
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
    out["code"] = df[code_col].astype(str).str.strip().str.upper()
    out["text_en"] = df[en_col].astype(str).str.strip()
    out["text_fr"] = df[fr_col].astype(str).str.strip()

    _QUESTIONS_CACHE = out
    return out


# ------------------------------------------------------------------
# EVERYTHING BELOW IS UNCHANGED
# ------------------------------------------------------------------

def load_scales_meta(refresh: bool = False) -> pd.DataFrame:
    global _SCALES_CACHE
    if _SCALES_CACHE is not None and not refresh:
        return _SCALES_CACHE

    xls = _get_metadata_workbook(refresh=refresh)
    df = xls.parse("SCALES", header=0)

    _SCALES_CACHE = df
    return df


def load_demographics_meta(refresh: bool = False) -> pd.DataFrame:
    global _DEM_CACHE
    if _DEM_CACHE is not None and not refresh:
        return _DEM_CACHE

    xls = _get_metadata_workbook(refresh=refresh)
    df = xls.parse("DEMCODE", header=0)

    _DEM_CACHE = df
    return df


def load_org_meta(refresh: bool = False) -> pd.DataFrame:
    global _ORG_CACHE
    if _ORG_CACHE is not None and not refresh:
        return _ORG_CACHE

    xls = _get_metadata_workbook(refresh=refresh)
    df = xls.parse("LEVEL1ID_LEVEL5ID", header=0)

    _ORG_CACHE = df
    return df


def load_posneg_meta(refresh: bool = False) -> pd.DataFrame:
    global _POSNEG_CACHE
    if _POSNEG_CACHE is not None and not refresh:
        return _POSNEG_CACHE

    xls = _get_metadata_workbook(refresh=refresh)
    df = xls.parse("POSNEG", header=0)

    _POSNEG_CACHE = df
    return df
