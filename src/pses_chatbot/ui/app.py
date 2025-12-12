from __future__ import annotations

import traceback
from typing import Any, Dict, List

import pandas as pd
import streamlit as st

from pses_chatbot.config import APP_NAME, APP_VERSION
from pses_chatbot.core.data_loader import query_pses_results
from pses_chatbot.core.metadata_loader import (
    load_questions_meta,
    load_scales_meta,
    load_demographics_meta,
    load_org_meta,
    load_posneg_meta,
)
from pses_chatbot.core.query_engine import (
    QueryParameters,
    run_analytical_query,
    QueryEngineError,
)
from pses_chatbot.core.audit import build_audit_snapshot


LEVEL_COLS = ["LEVEL1ID", "LEVEL2ID", "LEVEL3ID", "LEVEL4ID", "LEVEL5ID"]
REQ_ORG_COLS = set(LEVEL_COLS + ["UNITID", "DESCRIP_E", "DESCRIP_F", "DEPT"])


# ---------------------------------------------------------------------
# Robust org metadata normalization (fixes hidden spaces / casing issues)
# ---------------------------------------------------------------------

def _canon_colname(c: str) -> str:
    # Normalize common Excel weirdness: NBSP, tabs, double spaces, etc.
    s = str(c).replace("\u00A0", " ").replace("\t", " ").strip()
    s = " ".join(s.split())
    return s.upper()


def _normalize_org_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize org metadata columns so our UI can reliably reference:
      LEVEL1ID..LEVEL5ID, UNITID, DESCRIP_E, DESCRIP_F, DEPT

    If the loader produces e.g. "DESCRIP_E " (trailing space) or different casing,
    this will fix it.
    """
    out = df.copy()

    # Build mapping from raw -> canonical
    raw_cols = list(out.columns)
    canon_map = {_canon_colname(c): c for c in raw_cols}

    # Required columns (canonical -> existing raw)
    rename: Dict[str, str] = {}
    for canonical in [
        "LEVEL1ID", "LEVEL2ID", "LEVEL3ID", "LEVEL4ID", "LEVEL5ID",
        "UNITID", "DESCRIP_E", "DESCRIP_F", "DEPT",
    ]:
        raw = canon_map.get(canonical)
        if raw is not None and raw != canonical:
            rename[raw] = canonical

    if rename:
        out = out.rename(columns=rename)

    # If some required columns are still not present, we leave as-is;
    # the diagnostic expander will make it obvious.
    return out


def _normalize_text(x: Any) -> str:
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    s = str(x).replace("\u00A0", " ").strip()
    if s.lower() in {"nan", "none", "null"}:
        return ""
    return s


def _coerce_org_df(org_df: pd.DataFrame) -> pd.DataFrame:
    df = _normalize_org_columns(org_df)

    # Numeric cols
    for c in LEVEL_COLS + ["UNITID"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

    # Text cols
    for c in ["DESCRIP_E", "DESCRIP_F", "DEPT"]:
        if c in df.columns:
            df[c] = df[c].apply(_normalize_text)

    return df


def _label_row(row: pd.Series, lang: str = "EN", show_ids: bool = True) -> str:
    en = _normalize_text(row.get("DESCRIP_E", ""))
    fr = _normalize_text(row.get("DESCRIP_F", ""))
    dept = _normalize_text(row.get("DEPT", ""))

    if lang.upper() == "FR":
        name = fr or en or dept
    else:
        name = en or fr or dept

    # If DESCRIP_E truly exists, we should basically never hit this.
    if not name:
        unitid = int(row.get("UNITID", 0))
        name = f"Unit {unitid}" if unitid else "Unit"

    if not show_ids:
        return name

    ids = ".".join(str(int(row.get(c, 0))) for c in LEVEL_COLS)
    unitid = int(row.get("UNITID", 0))
    dept_suffix = f", DEPT={dept}" if dept else ""
    return f"{name}  (IDs {ids}, UNITID={unitid}{dept_suffix})"


def _level_candidates(df: pd.DataFrame, level_idx: int, selected: Dict[str, int]) -> pd.DataFrame:
    work = df.copy()

    # prior levels must match
    for i in range(level_idx):
        col = LEVEL_COLS[i]
        if col in work.columns:
            work = work[work[col] == int(selected[col])]

    # current level > 0
    cur_col = LEVEL_COLS[level_idx]
    if cur_col in work.columns:
        work = work[work[cur_col] > 0]

    # deeper levels == 0
    for j in range(level_idx + 1, len(LEVEL_COLS)):
        dcol = LEVEL_COLS[j]
        if dcol in work.columns:
            work = work[work[dcol] == 0]

    sort_cols = [cur_col] if cur_col in work.columns else []
    if "UNITID" in work.columns:
        sort_cols.append("UNITID")
    if sort_cols:
        work = work.sort_values(sort_cols)

    return work


def _find_rollup_name(df: pd.DataFrame, selected: Dict[str, int], lang: str) -> str:
    work = df.copy()
    for c in LEVEL_COLS:
        if c in work.columns:
            work = work[work[c] == int(selected.get(c, 0))]
    if work.empty:
        ids = ".".join(str(selected.get(c, 0)) for c in LEVEL_COLS)
        return f"(IDs {ids})"
    return _label_row(work.iloc[0], lang=lang, show_ids=False)


def render_org_cascade(org_df_raw: pd.DataFrame, lang: str = "EN") -> Dict[str, int]:
    if org_df_raw.empty:
        st.warning("Org metadata is empty; defaulting org levels to PS-wide (all zeros).")
        return {c: 0 for c in LEVEL_COLS}

    df = _coerce_org_df(org_df_raw)

    sel: Dict[str, int] = {c: 0 for c in LEVEL_COLS}

    # Level1 options: rows where LEVEL2-5 == 0
    lvl1 = df.copy()
    for c in LEVEL_COLS[1:]:
        if c in lvl1.columns:
            lvl1 = lvl1[lvl1[c] == 0]

    sort_cols = ["LEVEL1ID"] if "LEVEL1ID" in lvl1.columns else []
    if "UNITID" in lvl1.columns:
        sort_cols.append("UNITID")
    if sort_cols:
        lvl1 = lvl1.sort_values(sort_cols)

    lvl1_map: Dict[int, str] = {0: "Public Service-wide (roll-up) — LEVEL1ID=0"}
    if "LEVEL1ID" in lvl1.columns:
        for _, row in lvl1.iterrows():
            l1 = int(row.get("LEVEL1ID", 0))
            if l1 <= 0:
                continue
            if l1 not in lvl1_map:
                lvl1_map[l1] = _label_row(row, lang=lang)

    l1_choice = st.selectbox(
        "Organization – Level 1",
        options=list(lvl1_map.keys()),
        format_func=lambda k: lvl1_map.get(k, str(k)),
        key="org_lvl1",
    )
    sel["LEVEL1ID"] = int(l1_choice)

    if sel["LEVEL1ID"] == 0:
        return sel

    # Levels 2..5
    for level_idx in range(1, 5):
        col = LEVEL_COLS[level_idx]

        parent_for_label = {**sel}
        for deeper in LEVEL_COLS[level_idx:]:
            parent_for_label[deeper] = 0

        parent_name = _find_rollup_name(df, parent_for_label, lang=lang)
        rollup_label = f"All respondents (roll-up at {col}=0) — {parent_name}"

        candidates = _level_candidates(df, level_idx=level_idx, selected=sel)

        options_map: Dict[int, str] = {0: rollup_label}
        if not candidates.empty and col in candidates.columns:
            for _, row in candidates.iterrows():
                lv = int(row.get(col, 0))
                if lv <= 0:
                    continue
                if lv not in options_map:
                    options_map[lv] = _label_row(row, lang=lang)

        if len(options_map) == 1:
            sel[col] = 0
            break

        choice = st.selectbox(
            f"Organization – {col}",
            options=list(options_map.keys()),
            format_func=lambda k, m=options_map: m.get(k, str(k)),
            key=f"org_{col}",
        )
        sel[col] = int(choice)

        if sel[col] == 0:
            break

        for deeper in LEVEL_COLS[level_idx + 1:]:
            sel[deeper] = 0

    return sel


# ---------------------------------------------------------------------
# UI sections
# ---------------------------------------------------------------------

def _render_chat_area() -> None:
    st.subheader("Prototype chat area")
    st.write("Conversational engine not wired yet. Use the developer panels below.")


def _render_backend_status() -> None:
    with st.expander("Backend status (developer view)", expanded=False):
        if st.button("Run test query (max 1,000 rows)"):
            try:
                with st.spinner("Querying PSES DataStore..."):
                    df = query_pses_results(
                        filters=None,
                        fields=None,
                        sort=None,
                        max_rows=1_000,
                        page_size=1_000,
                    )
                st.success("PSES DataStore query succeeded.")
                st.write(f"Returned: {df.shape[0]} rows × {df.shape[1]} columns")
                if df.shape[1] > 0:
                    st.write("First columns:", list(df.columns)[:20])
            except Exception as e:
                st.error("Error while querying the PSES DataStore.")
                st.code(repr(e))
                st.text_area("Traceback", value=traceback.format_exc(), height=220)


def _render_metadata_status() -> None:
    with st.expander("Metadata status (developer view)", expanded=False):
        if st.button("Load metadata and show summary"):
            try:
                with st.spinner("Loading questions..."):
                    q = load_questions_meta(refresh=True)
                with st.spinner("Loading scales..."):
                    s = load_scales_meta(refresh=True)
                with st.spinner("Loading demographics..."):
                    d = load_demographics_meta(refresh=True)
                with st.spinner("Loading org hierarchy..."):
                    o = load_org_meta(refresh=True)
                with st.spinner("Loading pos/neg/agree..."):
                    p = load_posneg_meta(refresh=True)

                st.success("Metadata loaded successfully.")
                st.write(f"Questions: {len(q)}")
                st.write(f"Scales: {len(s)}")
                st.write(f"Demographics: {len(d)}")
                st.write(f"Org rows: {len(o)}")
                st.write(f"Pos/Neg mappings: {len(p)}")
            except Exception as e:
                st.error("Error while loading metadata.")
                st.code(repr(e))
                st.text_area("Traceback", value=traceback.format_exc(), height=260)


def _render_analytical_query_tester() -> None:
    with st.expander("Analytical query test (developer view)", expanded=True):
        col1, col2 = st.columns(2)

        with col1:
            question_code = st.text_input("Question code (QUESTION, e.g. Q08):", value="Q08")
            years_str = st.text_input(
                "Survey years (comma-separated, e.g. 2019,2020,2021,2022,2023,2024):",
                value="2019,2020,2021,2022,2023,2024",
            )
            demcode_input = st.text_input(
                'DEMCODE (leave blank for overall/no breakdown = empty string ""):',
                value="",
            )

        with col2:
            try:
                org_raw = load_org_meta(refresh=False)
            except Exception:
                org_raw = pd.DataFrame()

            # Diagnostic: show columns pre/post normalization and a sample of DESCRIP_E
            with st.expander("Org metadata diagnostics", expanded=False):
                st.write("Columns as loaded:", list(org_raw.columns) if not org_raw.empty else [])
                org_norm = _normalize_org_columns(org_raw) if not org_raw.empty else pd.DataFrame()
                st.write("Columns after normalization:", list(org_norm.columns) if not org_norm.empty else [])
                if not org_norm.empty:
                    cols = [c for c in ["LEVEL1ID", "UNITID", "DESCRIP_E", "DESCRIP_F", "DEPT"] if c in org_norm.columns]
                    st.dataframe(org_norm[cols].head(20), use_container_width=True)

            st.write("Organization (cascading selection):")
            org_levels = render_org_cascade(org_raw, lang="EN")

        if st.button("Run analytical query", key="run_analytical_query_btn"):
            try:
                years: List[int] = []
                for part in years_str.split(","):
                    p = part.strip()
                    if p:
                        years.append(int(p))
                if not years:
                    st.error("Please specify at least one survey year.")
                    return

                params = QueryParameters(
                    survey_years=years,
                    question_code=question_code.strip(),
                    demcode=demcode_input,  # may be ""
                    org_levels=org_levels,
                )

                with st.spinner("Running analytical query + audit snapshot..."):
                    result = run_analytical_query(params)
                    snapshot = build_audit_snapshot(result)

                st.success("Analytical query succeeded.")
                st.write(f"Question: {snapshot.question_code} — {snapshot.question_label_en}")
                st.write(f"Organization: {snapshot.org_label_en or '(label not found)'}")
                st.write(f"Demographic: {snapshot.dem_label_en or 'Overall (all respondents)'}")
                st.write(f"Metric: {snapshot.metric_name_en}")

                rows = []
                for m in result.yearly_metrics:
                    rows.append(
                        {
                            "Year": m.year,
                            "Value (Most positive / least negative)": m.value,
                            "Δ vs previous year": m.delta_vs_prev,
                            "N (ANSCOUNT)": m.n,
                        }
                    )
                st.dataframe(pd.DataFrame(rows), use_container_width=True)

                st.write("Supporting table (raw slice):")
                st.dataframe(result.raw_df, use_container_width=True)

            except QueryEngineError as qerr:
                st.error(f"Analytical query failed: {qerr}")
            except Exception as e:
                st.error("Unexpected error while running analytical query.")
                st.code(repr(e))
                st.text_area("Traceback", value=traceback.format_exc(), height=280)


def run_app() -> None:
    st.set_page_config(page_title=APP_NAME, page_icon="📊", layout="wide")
    st.title(APP_NAME)
    st.caption(f"Prototype version {APP_VERSION}")

    _render_chat_area()
    _render_backend_status()
    _render_metadata_status()
    _render_analytical_query_tester()
