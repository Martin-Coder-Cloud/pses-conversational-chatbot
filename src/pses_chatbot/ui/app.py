from __future__ import annotations

import traceback
from typing import Any, Dict, List, Optional

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


# ---------------------------------------------------------------------
# Org cascade helpers (tailored to your org metadata columns)
# Columns: LEVEL1ID..LEVEL5ID, UNITID, DESCRIP_E, DESCRIP_F, DEPT
# ---------------------------------------------------------------------

def _coerce_org_df(org_df: pd.DataFrame) -> pd.DataFrame:
    df = org_df.copy()
    for c in LEVEL_COLS + ["UNITID"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    # Keep labels as strings
    for c in ["DESCRIP_E", "DESCRIP_F", "DEPT"]:
        if c in df.columns:
            df[c] = df[c].astype(str)
    return df


def _label_row(row: pd.Series, lang: str = "EN", show_ids: bool = True) -> str:
    name = ""
    if lang.upper() == "FR":
        name = str(row.get("DESCRIP_F", "")).strip()
    else:
        name = str(row.get("DESCRIP_E", "")).strip()

    if not name:
        name = "(Unnamed unit)"

    if not show_ids:
        return name

    ids = ".".join(str(int(row.get(c, 0))) for c in LEVEL_COLS)
    unitid = int(row.get("UNITID", 0))
    dept = str(row.get("DEPT", "")).strip()
    dept_suffix = f", DEPT={dept}" if dept and dept.lower() != "nan" else ""
    return f"{name}  (IDs {ids}, UNITID={unitid}{dept_suffix})"


def _level_candidates(
    df: pd.DataFrame,
    level_idx: int,
    selected_levels: Dict[str, int],
) -> pd.DataFrame:
    """
    Return candidate rows for LEVEL{level_idx+1} under the already selected path.

    Pattern:
      - Prior levels match selected values.
      - Current level > 0 (real units, not roll-up).
      - Deeper levels == 0 (roll-up at that current level).
    """
    work = df.copy()

    # Prior levels must match
    for i in range(level_idx):
        col = LEVEL_COLS[i]
        work = work[work[col] == int(selected_levels[col])]

    # Current level must be > 0
    cur_col = LEVEL_COLS[level_idx]
    work = work[work[cur_col] > 0]

    # Deeper levels must be == 0
    for j in range(level_idx + 1, len(LEVEL_COLS)):
        dcol = LEVEL_COLS[j]
        work = work[work[dcol] == 0]

    # Dedupe on the current level value (keep first, but sorting below helps)
    work = work.sort_values([cur_col, "UNITID"] if "UNITID" in work.columns else [cur_col])
    return work


def _find_rollup_name(df: pd.DataFrame, selected_levels: Dict[str, int], lang: str) -> str:
    """
    Find the best label for the current selected roll-up node (with deeper levels = 0).
    If not found, fallback to a simple ID representation.
    """
    work = df.copy()
    for c in LEVEL_COLS:
        work = work[work[c] == int(selected_levels.get(c, 0))]
    if work.empty:
        ids = ".".join(str(selected_levels.get(c, 0)) for c in LEVEL_COLS)
        return f"(IDs {ids})"
    return _label_row(work.iloc[0], lang=lang, show_ids=False)


def render_org_cascade(org_df: pd.DataFrame, lang: str = "EN") -> Dict[str, int]:
    """
    Cascading org selection based on LEVEL1ID..LEVEL5ID.

    Rules:
      - Always returns a dict with all five levels defined.
      - Selecting 0 at any level stops deeper selection (remaining are 0).
      - Level 1 list includes PS-wide (0) + departments (LEVEL1ID>0 with LEVEL2-5==0).
    """
    if org_df.empty:
        st.warning("Org metadata is empty; defaulting org levels to PS-wide (all zeros).")
        return {c: 0 for c in LEVEL_COLS}

    df = _coerce_org_df(org_df)

    # Initialize selection
    sel: Dict[str, int] = {c: 0 for c in LEVEL_COLS}

    # LEVEL1 options: rows where LEVEL2-5 == 0
    lvl1 = df.copy()
    for c in LEVEL_COLS[1:]:
        lvl1 = lvl1[lvl1[c] == 0]
    lvl1 = lvl1.sort_values(["LEVEL1ID", "UNITID"] if "UNITID" in lvl1.columns else ["LEVEL1ID"])

    lvl1_map: Dict[int, str] = {0: "Public Service-wide (roll-up) — LEVEL1ID=0"}
    for _, row in lvl1.iterrows():
        l1 = int(row["LEVEL1ID"])
        if l1 <= 0:
            continue
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

        parent_name = _find_rollup_name(df, {**sel, **{c: 0 for c in LEVEL_COLS[level_idx:]}}, lang=lang)
        rollup_label = f"All respondents (roll-up at {col}=0) — {parent_name}"

        candidates = _level_candidates(df, level_idx=level_idx, selected_levels=sel)

        options_map: Dict[int, str] = {0: rollup_label}
        if not candidates.empty:
            for _, row in candidates.iterrows():
                lv = int(row[col])
                if lv <= 0:
                    continue
                # Prefer not to overwrite if duplicated key; first sorted row wins
                if lv not in options_map:
                    options_map[lv] = _label_row(row, lang=lang)

        # If no candidates besides roll-up, stop cascading
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

        # Ensure deeper levels stay at 0 unless selected later
        for deeper in LEVEL_COLS[level_idx + 1 :]:
            sel[deeper] = 0

    return sel


# ---------------------------------------------------------------------
# UI sections
# ---------------------------------------------------------------------

def _render_chat_area() -> None:
    st.subheader("Prototype chat area")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for entry in st.session_state.chat_history:
        role = entry.get("role", "user")
        content = entry.get("content", "")
        if role == "user":
            st.write(f"You: {content}")
        else:
            st.write(f"Chatbot: {content}")

    user_input = st.text_input(
        "Ask a question about PSES results:",
        "",
        placeholder="Example: How has work satisfaction changed since 2019 for CRA?",
        key="chat_input",
    )

    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": "Conversational engine not wired yet. Use the developer panels below to test queries + audit facts.",
            }
        )
        st.experimental_rerun()


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
        st.write(
            "This panel runs the analytical query engine and shows audit facts.\n"
            "Organization selection uses cascading dropdowns based on LEVEL1ID–LEVEL5ID metadata."
        )

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
                org_df = load_org_meta(refresh=False)
            except Exception:
                org_df = pd.DataFrame()

            st.write("Organization (cascading selection):")
            org_levels = render_org_cascade(org_df, lang="EN")

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

                demcode = demcode_input  # may be "" (overall)

                params = QueryParameters(
                    survey_years=years,
                    question_code=question_code.strip(),
                    demcode=demcode,
                    org_levels=org_levels,
                )

                with st.spinner("Running analytical query + audit snapshot..."):
                    result = run_analytical_query(params)
                    snapshot = build_audit_snapshot(result)

                st.success("Analytical query succeeded.")

                st.write("Labels:")
                st.write(f"Question: {snapshot.question_code} — {snapshot.question_label_en}")
                st.write(f"Organization: {snapshot.org_label_en or '(label not found)'}")
                st.write(f"Demographic: {snapshot.dem_label_en or 'Overall (all respondents)'}")
                st.write(f"Metric: {snapshot.metric_name_en}")

                st.write("Yearly metrics:")
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

                st.write("Audit facts (canonical):")
                for year in sorted(snapshot.metrics_by_year.keys()):
                    val = snapshot.metrics_by_year[year]
                    n = snapshot.n_by_year.get(year)
                    st.write(f"{year}: {val:.1f} (N={n if n is not None else 'N/A'})")

                if snapshot.trend_facts:
                    st.write("Year-over-year deltas:")
                    for tf in snapshot.trend_facts:
                        st.write(
                            f"{tf.year_start}→{tf.year_end}: {tf.value_start:.1f}→{tf.value_end:.1f} "
                            f"({tf.delta:+.1f}, {tf.direction})"
                        )

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

    st.write(
        "PSES Conversational Analytics Chatbot (prototype).\n"
        "Use the developer panels to test: data access, metadata loading, org cascading selection, analytical query, and audit facts."
    )

    _render_chat_area()
    _render_backend_status()
    _render_metadata_status()
    _render_analytical_query_tester()
