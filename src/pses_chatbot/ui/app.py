from __future__ import annotations

import traceback
from typing import List

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


def _render_chat_area() -> None:
    st.subheader("Prototype chat area")
    st.write("Conversational engine not wired yet. Use the developer panels below.")


def _render_backend_status() -> None:
    with st.expander("Backend status (developer view)", expanded=False):
        st.write(
            "This is a quick connectivity test to the CKAN DataStore resource configured in your app."
        )
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
    with st.expander("Metadata status (developer view)", expanded=True):
        st.write(
            "This panel loads metadata from the Excel workbook and shows both counts and a diagnostic preview.\n"
            "Focus: confirming the organization metadata columns and DESCRIP_E values."
        )

        # Controls
        colA, colB, colC = st.columns([1, 1, 2])
        with colA:
            refresh = st.checkbox("Force refresh metadata loaders", value=True)
        with colB:
            preview_rows = st.number_input("Preview rows", min_value=5, max_value=200, value=25, step=5)
        with colC:
            st.write("Org lookup test (LEVEL1ID..LEVEL5ID)")
            default_path = "1,0,0,0,0"
            level_path_str = st.text_input(
                "Enter 5 comma-separated ints (e.g., 1,0,0,0,0):",
                value=default_path,
            )

        if st.button("Load metadata and show summary + diagnostics"):
            try:
                with st.spinner("Loading questions..."):
                    q = load_questions_meta(refresh=bool(refresh))
                with st.spinner("Loading scales..."):
                    s = load_scales_meta(refresh=bool(refresh))
                with st.spinner("Loading demographics..."):
                    d = load_demographics_meta(refresh=bool(refresh))
                with st.spinner("Loading org hierarchy..."):
                    o = load_org_meta(refresh=bool(refresh))
                with st.spinner("Loading pos/neg/agree..."):
                    p = load_posneg_meta(refresh=bool(refresh))

                st.success("Metadata loaded successfully.")

                # Counts
                st.write("Counts")
                st.write(f"Questions: {len(q)}")
                st.write(f"Scales: {len(s)}")
                st.write(f"Demographics: {len(d)}")
                st.write(f"Org rows: {len(o)}")
                st.write(f"Pos/Neg mappings: {len(p)}")

                # --- ORG DIAGNOSTICS ---
                st.write("---")
                st.write("Organization metadata diagnostics")

                if o is None or len(o) == 0:
                    st.warning("Org metadata dataframe is empty.")
                    return

                # Show raw columns (exact) and repr columns (reveals hidden chars/spaces)
                st.write("Org columns (as loaded)")
                st.write(list(o.columns))

                st.write("Org columns (repr view — reveals hidden spaces/characters)")
                st.write([repr(c) for c in list(o.columns)])

                # Try to select key columns safely (without assuming exact names)
                # We intentionally do NOT rename columns here; we want to see what's actually loaded.
                wanted = [
                    "LEVEL1ID",
                    "LEVEL2ID",
                    "LEVEL3ID",
                    "LEVEL4ID",
                    "LEVEL5ID",
                    "UNITID",
                    "DESCRIP_E",
                    "DESCRIP_F",
                    "DEPT",
                ]
                available = [c for c in wanted if c in o.columns]

                if not available:
                    st.error(
                        "None of the expected org columns were found by exact match. "
                        "This strongly suggests hidden spaces or different header names."
                    )
                else:
                    st.write("Org preview (available key columns)")
                    st.dataframe(o[available].head(int(preview_rows)), use_container_width=True)

                # Parse the user-provided LEVEL path and filter
                st.write("---")
                st.write("Org lookup test: filter by LEVEL1ID..LEVEL5ID")

                # Parse 5 integers
                levels: List[int] = []
                try:
                    parts = [p.strip() for p in level_path_str.split(",")]
                    levels = [int(x) for x in parts if x != ""]
                except Exception:
                    levels = []

                if len(levels) != 5:
                    st.warning(
                        "Please enter exactly 5 integers for LEVEL1ID..LEVEL5ID "
                        "(example: 1,0,0,0,0)."
                    )
                    return

                # Only attempt filtering if the needed columns exist exactly
                needed_cols = ["LEVEL1ID", "LEVEL2ID", "LEVEL3ID", "LEVEL4ID", "LEVEL5ID"]
                if not all(c in o.columns for c in needed_cols):
                    st.error(
                        "Cannot run the LEVEL filter because one or more LEVEL columns "
                        "do not match expected names exactly. Check the repr column list above."
                    )
                    return

                f = o.copy()

                # Coerce LEVEL columns to numeric safely for matching
                for c in needed_cols:
                    f[c] = pd.to_numeric(f[c], errors="coerce")

                mask = (
                    (f["LEVEL1ID"] == levels[0])
                    & (f["LEVEL2ID"] == levels[1])
                    & (f["LEVEL3ID"] == levels[2])
                    & (f["LEVEL4ID"] == levels[3])
                    & (f["LEVEL5ID"] == levels[4])
                )
                matches = f[mask]

                st.write(f"Matches found: {len(matches)} for LEVEL path = {levels}")

                show_cols = []
                for c in ["LEVEL1ID", "LEVEL2ID", "LEVEL3ID", "LEVEL4ID", "LEVEL5ID", "UNITID", "DESCRIP_E", "DESCRIP_F", "DEPT"]:
                    if c in f.columns:
                        show_cols.append(c)

                if len(matches) == 0:
                    st.warning(
                        "No matching org rows found for that LEVEL path. "
                        "Either the path doesn’t exist in metadata, or types/values differ."
                    )
                else:
                    st.dataframe(matches[show_cols].head(50), use_container_width=True)

                    # Quick check: show DESCRIP_E length summary if available
                    if "DESCRIP_E" in matches.columns:
                        # Normalize to string for the check
                        tmp = matches["DESCRIP_E"].astype(str)
                        blank_count = (tmp.str.strip() == "").sum()
                        st.write(
                            f'DESCRIP_E blanks among matches: {blank_count} / {len(matches)}'
                        )

            except Exception as e:
                st.error("Error while loading metadata or running diagnostics.")
                st.code(repr(e))
                st.text_area("Traceback", value=traceback.format_exc(), height=260)


def _render_analytical_query_tester() -> None:
    with st.expander("Analytical query test (developer view)", expanded=False):
        st.write(
            "This panel runs the analytical query engine and shows audit facts.\n"
            "Use it once org metadata loading/labels are confirmed."
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
            st.write("Org levels (temporary manual entry; will be replaced by cascade)")
            level1 = st.text_input("LEVEL1ID", value="0")
            level2 = st.text_input("LEVEL2ID", value="0")
            level3 = st.text_input("LEVEL3ID", value="0")
            level4 = st.text_input("LEVEL4ID", value="0")
            level5 = st.text_input("LEVEL5ID", value="0")

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

                org_levels = {
                    "LEVEL1ID": int(level1.strip() or "0"),
                    "LEVEL2ID": int(level2.strip() or "0"),
                    "LEVEL3ID": int(level3.strip() or "0"),
                    "LEVEL4ID": int(level4.strip() or "0"),
                    "LEVEL5ID": int(level5.strip() or "0"),
                }

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
