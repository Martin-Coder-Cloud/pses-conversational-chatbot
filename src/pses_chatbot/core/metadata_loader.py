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
_QUESTIONS_CACHE: Optional[pd.DataFrame] = None
_SCALES_CACHE: Optional[pd.DataFrame] = None
_DEM_CACHE: Optional[pd.DataFrame] = None
_ORG_CACHE: Optional[pd.DataFrame] = None
_POSNEG_CACHE: Optional[pd.DataFrame] = None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _find_metadata_workbook() -> Path:
    """
    Locate the metadata workbook under data/metadata.

    Primary expectation:
      data/metadata/2024_PSES_Metadata.xlsx

    If that file is not found, we fall back to:
      - any .xlsx or .xls file in METADATA_DIR (first one found)

    This keeps things robust if the filename changes, but for clarity
    it is recommended to rename your file to METADATA_WORKBOOK_NAME.
    """
    primary = METADATA_DIR / METADATA_WORKBOOK_NAME
    if primary.exists():
        return primary

    # Fallback: search for any Excel file
    candidates: List[Path] = list(METADATA_DIR.glob("*.xlsx")) + list(METADATA_DIR.glob("*.xls"))
    if not candidates:
        raise FileNotFoundError(
            f"No metadata workbook found. Expected at least {primary}, "
            f"or any .xls/.xlsx file under {METADATA_DIR}."
        )

    chosen = candidates[0]
    logger.warning(
        "Using metadata workbook %s (could not find %s). "
        "Consider renaming your file for clarity.",
        chosen,
        primary,
    )
    return chosen


def _get_metadata_workbook(refresh: bool = False) -> pd.ExcelFile:
    """
    Return a pd.ExcelFile for the metadata workbook, cached in memory.
    """
    global _METADATA_XLS
    if _METADATA_XLS is not None and not refresh:
        return _METADATA_XLS

    path = _find_metadata_workbook()
    logger.info("Loading metadata workbook: %s", path)
    _METADATA_XLS = pd.ExcelFile(path)
    logger.info("Metadata workbook sheets: %s", _METADATA_XLS.sheet_names)
    return _METADATA_XLS


# ---------------------------------------------------------------------------
# Questions metadata (QUESTIONS sheet)
# ---------------------------------------------------------------------------

def load_questions_meta(refresh: bool = False) -> pd.DataFrame:
    """
    Load questionnaire metadata from the 'QUESTIONS' sheet.

    Expected sheet structure (as in your workbook):
      - Column: 'Question number / numéro de la question'  (e.g., 'Q01')
      - Column: 'English'                                 (question text EN)
      - Column: 'Français'                                (question text FR)

    Normalized columns returned:
      - code      (e.g., 'Q01')
      - text_en   (English question text)
      - text_fr   (French question text)
      - polarity  (default 'POS' for now; we can refine later if needed)
    """
    global _QUESTIONS_CACHE
    if _QUESTIONS_CACHE is not None and not refresh:
        return _QUESTIONS_CACHE

    xls = _get_metadata_workbook(refresh=refresh)
    df = xls.parse("QUESTIONS")

    # Normalize columns
    cols = {c: c for c in df.columns}
    # Make a lower-cased mapping to be robust
    lower = {c.lower(): c for c in df.columns}

    code_col = lower.get("question number / numéro de la question".lower())
    en_col = lower.get("english")
    fr_col = lower.get("français")

    if not (code_col and en_col and fr_col):
        raise ValueError(
            "QUESTIONS sheet does not contain the expected columns "
            "('Question number / numéro de la question', 'English', 'Français')."
        )

    out = pd.DataFrame()
    out["code"] = (
        df[code_col]
        .astype(str)
        .str.strip()
        .str.upper()
    )
    out["text_en"] = df[en_col].astype(str).str.strip()
    out["text_fr"] = df[fr_col].astype(str).str.strip()
    out["polarity"] = "POS"  # default; we can refine via POSITIVE_NEUTRAL sheet if needed

    _QUESTIONS_CACHE = out
    return out


# ---------------------------------------------------------------------------
# Scales / answer options (RESPONSE OPTIONS DE RÉPONSES sheet)
# ---------------------------------------------------------------------------

def load_scales_meta(refresh: bool = False) -> pd.DataFrame:
    """
    Load answer options per question from 'RESPONSE OPTIONS DE RÉPONSES'.

    Expected columns (as in your workbook):
      - 'Question'   (e.g., 'Q01')
      - 'Answer1ENG' ... 'Answer7ENG'
      - 'Answer1FRA' ... 'Answer7FRA'
      - plus some summary columns (PositiveENG, etc.) we ignore here.

    Normalized tidy structure:
      - question_code  (e.g., 'Q01')
      - option_index   (1..7)
      - label_en       (e.g., 'Strongly agree')
      - label_fr       (e.g., 'Tout à fait d'accord')

    Only rows where at least one of (label_en, label_fr) is non-empty are kept.
    """
    global _SCALES_CACHE
    if _SCALES_CACHE is not None and not refresh:
        return _SCALES_CACHE

    xls = _get_metadata_workbook(refresh=refresh)
    df = xls.parse("RESPONSE OPTIONS DE RÉPONSES")

    # Basic validation
    expected_prefixes_eng = [f"Answer{i}ENG" for i in range(1, 8)]
    expected_prefixes_fra = [f"Answer{i}FRA" for i in range(1, 8)]

    if "Question" not in df.columns:
        raise ValueError("Scales sheet missing 'Question' column.")

    for col in expected_prefixes_eng:
        if col not in df.columns:
            logger.warning("Scales sheet missing column %s (ENG).", col)
    for col in expected_prefixes_fra:
        if col not in df.columns:
            logger.warning("Scales sheet missing column %s (FRA).", col)

    records: List[Dict[str, Any]] = []

    for _, row in df.iterrows():
        qcode = str(row["Question"]).strip().upper()
        if not qcode:
            continue

        for idx in range(1, 8):
            col_en = f"Answer{idx}ENG"
            col_fr = f"Answer{idx}FRA"

            label_en = str(row[col_en]).strip() if col_en in df.columns and pd.notna(row[col_en]) else ""
            label_fr = str(row[col_fr]).strip() if col_fr in df.columns and pd.notna(row[col_fr]) else ""

            if not label_en and not label_fr:
                continue

            records.append(
                {
                    "question_code": qcode,
                    "option_index": idx,
                    "label_en": label_en,
                    "label_fr": label_fr,
                }
            )

    out = pd.DataFrame.from_records(records)
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

    Normalized columns:
      - demcode             (string, e.g., '1001')
      - bycond              (string definition)
      - label_en            (English label)
      - label_fr            (French label)
      - dimension_question  (e.g., 'Q82', extracted from BYCOND if present)
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
    """
    Load organization hierarchy from the 'LEVEL1ID_LEVEL5ID' sheet.

    Expected columns (as in your workbook):
      - 'LEVEL1ID', 'LEVEL2ID', 'LEVEL3ID', 'LEVEL4ID', 'LEVEL5ID', 'UNITID'
      - 'DESCRIP_E', 'DESCRIP_F' (EN/FR names)
      - 'DEPT' (department code)

    Normalized columns:
      - LEVEL1ID ... LEVEL5ID (Int64, NaN filled as 0)
      - UNITID                (Int64, NaN filled as 0)
      - org_name_en           (DESCRIP_E)
      - org_name_fr           (DESCRIP_F)
      - dept_code             (DEPT as string)
    """
    global _ORG_CACHE
    if _ORG_CACHE is not None and not refresh:
        return _ORG_CACHE

    xls = _get_metadata_workbook(refresh=refresh)
    df = xls.parse("LEVEL1ID_LEVEL5ID")

    lower = {c.lower(): c for c in df.columns}
    l1 = lower.get("level1id")
    l2 = lower.get("level2id")
    l3 = lower.get("level3id")
    l4 = lower.get("level4id")
    l5 = lower.get("level5id")
    unit = lower.get("unitid")
    descr_e = lower.get("descrip_e")
    descr_f = lower.get("descrip_f")
    dept = lower.get("dept")

    required_cols = [l1, l2, l3, l4, l5, unit, descr_e, descr_f, dept]
    if any(c is None for c in required_cols):
        raise ValueError(
            "LEVEL1ID_LEVEL5ID sheet does not contain the expected columns "
            "('LEVEL1ID', 'LEVEL2ID', 'LEVEL3ID', 'LEVEL4ID', 'LEVEL5ID', "
            "'UNITID', 'DESCRIP_E', 'DESCRIP_F', 'DEPT')."
        )

    out = pd.DataFrame()
    for col_name, new_name in [
        (l1, "LEVEL1ID"),
        (l2, "LEVEL2ID"),
        (l3, "LEVEL3ID"),
        (l4, "LEVEL4ID"),
        (l5, "LEVEL5ID"),
        (unit, "UNITID"),
    ]:
        out[new_name] = pd.to_numeric(df[col_name], errors="coerce").fillna(0).astype("Int64")

    out["org_name_en"] = df[descr_e].astype(str).str.strip()
    out["org_name_fr"] = df[descr_f].astype(str).str.strip()
    out["dept_code"] = df[dept].astype(str).str.strip()

    _ORG_CACHE = out
    return out


# ---------------------------------------------------------------------------
# Positive / Neutral / Negative / Agree mapping
# (POSITIVE_NEUTRAL_NEGATIVE_AGREE sheet)
# ---------------------------------------------------------------------------

def load_posneg_meta(refresh: bool = False) -> pd.DataFrame:
    """
    Load the mapping of answer positions to positive/neutral/negative/agree
    categories from the 'POSITIVE_NEUTRAL_NEGATIVE_AGREE' sheet.

    Expected columns:
      - 'QUESTION'              (e.g., 'Q01')
      - 'TITLE_E'               (EN question text)
      - 'TITLE_F'               (FR question text)
      - 'POSITIVE / POSITIF'    (e.g., '1,2')
      - 'NEUTRAL/NEUTRE'        (e.g., '3')
      - 'NEGATIVE / NÉGATIF'    (e.g., '4,5')
      - 'AGREE / ACCORD'        (e.g., '1,2')

    Normalized columns:
      - question_code
      - title_en
      - title_fr
      - positive_positions   (list[int])
      - neutral_positions    (list[int])
      - negative_positions   (list[int])
      - agree_positions      (list[int])
    """
    global _POSNEG_CACHE
    if _POSNEG_CACHE is not None and not refresh:
        return _POSNEG_CACHE

    xls = _get_metadata_workbook(refresh=refresh)
    df = xls.parse("POSITIVE_NEUTRAL_NEGATIVE_AGREE")

    lower = {c.lower(): c for c in df.columns}
    q_col = lower.get("question")
    te_col = lower.get("title_e")
    tf_col = lower.get("title_f")
    pos_col = lower.get("positive / positif")
    neu_col = lower.get("neutral/neutre")
    neg_col = lower.get("negative / négatif")
    agr_col = lower.get("agree / accord")

    if not (q_col and te_col and tf_col and pos_col and neu_col and neg_col and agr_col):
        raise ValueError(
            "POSITIVE_NEUTRAL_NEGATIVE_AGREE sheet does not contain the expected columns."
        )

    def parse_positions(val: Any) -> List[int]:
        if pd.isna(val):
            return []
        text = str(val)
        if not text.strip():
            return []
        parts = [p.strip() for p in text.replace(" ", "").split(",") if p.strip()]
        positions: List[int] = []
        for p in parts:
            try:
                positions.append(int(float(p)))
            except Exception:
                continue
        return positions

    out = pd.DataFrame()
    out["question_code"] = df[q_col].astype(str).str.strip().str.upper()
    out["title_en"] = df[te_col].astype(str).str.strip()
    out["title_fr"] = df[tf_col].astype(str).str.strip()
    out["positive_positions"] = df[pos_col].apply(parse_positions)
    out["neutral_positions"] = df[neu_col].apply(parse_positions)
    out["negative_positions"] = df[neg_col].apply(parse_positions)
    out["agree_positions"] = df[agr_col].apply(parse_positions)

    _POSNEG_CACHE = out
    return out
