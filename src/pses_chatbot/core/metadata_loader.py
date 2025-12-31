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


def _find_metadata_workbook() -> Path:
    return Path(METADATA_DIR) / METADATA_WORKBOOK_NAME


def _get_metadata_workbook(refresh: bool = False) -> pd.ExcelFile:
    global _METADATA_XLS

    if _METADATA_XLS is not None and not refresh:
        return _METADATA_XLS

    path = _find_metadata_workbook()
    if not path.exists():
        raise FileNotFoundError(
            f"Metadata workbook not found: {path}. "
            f"Expected at: {Path(METADATA_DIR) / METADATA_WORKBOOK_NAME}"
        )

    _METADATA_XLS = pd.ExcelFile(path)
    return _METADATA_XLS


def load_questions_meta(refresh: bool = False) -> pd.DataFrame:
    global _QUESTIONS_CACHE
    if _QUESTIONS_CACHE is not None and not refresh:
        return _QUESTIONS_CACHE

    xls = _get_metadata_workbook(refresh=refresh)
    df = xls.parse("QUESTIONS", header=0)

    # Normalize columns
    lower = {c.lower(): c for c in df.columns}
    code_col = lower.get("question")
    en_col = lower.get("question_en")
    fr_col = lower.get("question_fr")

    if not (code_col and en_col and fr_col):
        raise ValueError("QUESTIONS sheet missing expected columns (QUESTION, QUESTION_EN, QUESTION_FR).")

    out = pd.DataFrame()
    out["code"] = df[code_col].astype(str).str.strip().str.upper()
    out["text_en"] = df[en_col].astype(str).str.strip()
    out["text_fr"] = df[fr_col].astype(str).str.strip()

    _QUESTIONS_CACHE = out
    return out


def load_scales_meta(refresh: bool = False) -> pd.DataFrame:
    global _SCALES_CACHE
    if _SCALES_CACHE is not None and not refresh:
        return _SCALES_CACHE

    xls = _get_metadata_workbook(refresh=refresh)
    df = xls.parse("SCALES", header=0)

    # Normalize columns
    lower = {c.lower(): c for c in df.columns}
    scale_col = lower.get("scale")
    name_en_col = lower.get("name_en")
    name_fr_col = lower.get("name_fr")

    if not (scale_col and name_en_col and name_fr_col):
        raise ValueError("SCALES sheet missing expected columns (SCALE, NAME_EN, NAME_FR).")

    out = pd.DataFrame()
    out["scale"] = df[scale_col].astype(str).str.strip()
    out["name_en"] = df[name_en_col].astype(str).str.strip()
    out["name_fr"] = df[name_fr_col].astype(str).str.strip()

    _SCALES_CACHE = out
    return out


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

    # Optional: Category labels used for UI grouping (e.g., Gender, Age group)
    # Keep blank strings if columns are missing in the workbook.
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

    lower = {c.lower(): c for c in df.columns}
    level_cols = ["level1id", "level2id", "level3id", "level4id", "level5id"]
    missing_levels = [c for c in level_cols if c not in lower]

    if missing_levels:
        raise ValueError(f"Org sheet missing required columns: {missing_levels}")

    out = pd.DataFrame()
    for lvl in level_cols:
        out[lvl.upper()] = pd.to_numeric(df[lower[lvl]], errors="coerce").fillna(0).astype(int)

    unitid_col = lower.get("unitid")
    if unitid_col:
        out["UNITID"] = pd.to_numeric(df[unitid_col], errors="coerce").fillna(0).astype(int)
    else:
        out["UNITID"] = 0

    en_col = lower.get("descrip_e")
    fr_col = lower.get("descrip_f")
    dept_col = lower.get("dept")

    out["org_name_en"] = df[en_col].astype(str).str.strip() if en_col else ""
    out["org_name_fr"] = df[fr_col].astype(str).str.strip() if fr_col else ""
    out["dept_code"] = df[dept_col].astype(str).str.strip() if dept_col else ""

    _ORG_CACHE = out
    return out


def load_posneg_meta(refresh: bool = False) -> pd.DataFrame:
    global _POSNEG_CACHE
    if _POSNEG_CACHE is not None and not refresh:
        return _POSNEG_CACHE

    xls = _get_metadata_workbook(refresh=refresh)
    df = xls.parse("POSNEG", header=0)

    _POSNEG_CACHE = df
    return df
