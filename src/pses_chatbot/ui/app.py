from __future__ import annotations

import traceback

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
    """
    Very simple placeholder chat area.

    Later, this will be wired to the full conversational pipeline:
      - intent parsing
      - parameter inference
      - data retrieval
      - narrative generation
      - audit of statements vs data
    """
    st.subheader("Prototype chat area")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display past messages
    for entry in st.session_state.chat_history:
        role = entry.get("role", "user")
        content = entry.get("content", "")
        if role == "user":
            st.markdown(f"**You:** {content}")
        else:
            st.markdown(f"**Chatbot:** {content}")

    user_input = st.text_input(
        "Ask a question about PSES results:",
        "",
        placeholder="Example: How has work satisfaction changed since 2019 for CRA?",
        key="chat_input",
    )

    if user_input:
        # Append user message
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        # Placeholder bot response – will be replaced by real orchestration later
        bot_reply = (
            "Thanks for your question! The conversational analytics engine is not implemented yet.\n\n"
            "In the next steps, this chatbot will:\n"
            "- Infer the PSES question, years, demographics, organization and level from your text\n"
            "- Query the PSES open data API for the corresponding slice of the dataset\n"
            "- Generate a short, validated analytical summary\n"
            "- Show you the data table used for the answer\n"
            "- Provide an audit view so you can verify that every statement matches the data"
        )
        st.session_state.chat_history.append({"role": "assistant", "content": bot_reply})

        # Force rerun to show the new messages immediately
        st.experimental_rerun()


def _render_backend_status() -> None:
    """
    Developer-only backend status panel.

    Tests:
      - Connectivity to the CKAN DataStore API
      - That PSES_DATASTORE_RESOURCE_ID is valid
      - The basic schema (columns) of the PSES results table
    """
    with st.expander("Backend status (developer view)", expanded=False):
        st.markdown(
            """
            This section is for development and diagnostics.

            When you click the button below, the app will:

            1. Call the CKAN DataStore API for the configured PSES resource  
            2. Request up to **1,000 rows** (no filters)  
            3. Display the resulting DataFrame shape and the first few column names  

            This is just to confirm that:
            - The resource ID is correct  
            - The DataStore API is reachable from Streamlit Cloud  
            - Our `query_pses_results` wrapper is working
            """
        )

        if st.button("Run test query (max 1,000 rows)"):
            from pses_chatbot.core.data_loader import query_pses_results

            try:
                with st.spinner("Querying PSES DataStore (this may take a few seconds)..."):
                    df = query_pses_results(
                        filters=None,
                        fields=None,      # all columns (we only show names)
                        sort=None,
                        max_rows=1_000,
                        page_size=1_000,
                    )

                st.success("PSES DataStore query succeeded.")
                st.markdown(f"- Returned shape: **{df.shape[0]} rows × {df.shape[1]} columns**")

                if df.shape[1] > 0:
                    cols = list(df.columns)
                    preview_cols = cols[:10]
                    st.markdown("First few column names:")
                    for c in preview_cols:
                        st.markdown(f"- `{c}`")
                    if len(cols) > len(preview_cols):
                        st.markdown(f"... and {len(cols) - len(preview_cols)} more columns.")
                else:
                    st.warning("The DataFrame has zero columns. Check the resource/schema configuration.")

            except Exception as e:
                st.error("Error while querying the PSES DataStore.")
                st.code(repr(e))
                st.text_area(
                    "Traceback (for debugging)",
                    value=traceback.format_exc(),
                    height=200,
                )


def _render_metadata_status() -> None:
    """
    Developer-only metadata status panel.

    Loads metadata from the Excel workbook:
      - Questions
      - Scales (answer options)
      - Demographics
      - Organization hierarchy
      - Positive/Neutral/Negative/Agree mappings
    """
    with st.expander("Metadata status (developer view)", expanded=False):
        st.markdown(
            """
            This section loads metadata from the Excel workbook under `data/metadata/`.

            It will attempt to read:
            - **QUESTIONS** sheet  
            - **RESPONSE OPTIONS DE RÉPONSES** sheet  
            - **DEMCODE** sheet  
            - **LEVEL1ID_LEVEL5ID** sheet  
            - **POSITIVE_NEUTRAL_NEGATIVE_AGREE** sheet  

            and summarize how many records are found in each.
            """
        )

        if st.button("Load metadata and show summary"):
            try:
                with st.spinner("Loading questions metadata..."):
                    q = load_questions_meta(refresh=True)
                with st.spinner("Loading scales metadata..."):
                    s = load_scales_meta(refresh=True)
                with st.spinner("Loading demographics metadata..."):
                    d = load_demographics_meta(refresh=True)
                with st.spinner("Loading organization metadata..."):
                    o = load_org_meta(refresh=True)
                with st.spinner("Loading pos/neg/agree metadata..."):
                    p = load_posneg_meta(refresh=True)

                st.success("Metadata loaded successfully.")

                st.markdown("**Questions**")
                st.markdown(f"- Questions: **{len(q)}**")
                if not q.empty:
                    sample_q = q.iloc[0]
                    st.markdown(
                        f"  - Example: `{sample_q['code']}` – EN: *{sample_q['text_en'][:80]}*"
                    )

                st.markdown("---")
                st.markdown("**Scales / answer options**")
                st.markdown(f"- Question–option records: **{len(s)}**")
                if not s.empty:
                    sample_s = s.iloc[0]
                    st.markdown(
                        "  - Example: "
                        f"question `{sample_s['question_code']}`, "
                        f"option {sample_s['option_index']} – "
                        f"EN: *{sample_s['label_en'][:60]}*"
                    )

                st.markdown("---")
                st.markdown("**Demographics**")
                st.markdown(f"- Demographic codes: **{len(d)}**")
                if not d.empty:
                    sample_d = d.iloc[0]
                    st.markdown(
                        "  - Example: "
                        f"DEMCODE `{sample_d.get('code', sample_d.get('demcode', ''))}` – "
                        f"EN: *{sample_d['label_en'][:60]}* "
                        f"(BYCOND: {sample_d.get('bycond', '')})"
                    )

                st.markdown("---")
                st.markdown("**Organization hierarchy**")
                st.markdown(f"- Org rows: **{len(o)}**")
                if not o.empty:
                    sample_o = o.iloc[0]
                    st.markdown(
                        "  - Example: "
                        f"LEVEL1ID={sample_o['LEVEL1ID']}, "
                        f"LEVEL2ID={sample_o['LEVEL2ID']}, "
                        f"UNITID={sample_o['UNITID']} – "
                        f"EN: *{sample_o['org_name_en'][:60]}* "
                        f"(DEPT code: {sample_o['dept_code']})"
                    )

                st.markdown("---")
                st.markdown("**Positive / Neutral / Negative / Agree**")
                st.markdown(f"- Question mappings: **{len(p)}**")
                if not p.empty:
                    sample_p = p.iloc[0]
                    st.markdown(
                        "  - Example: "
                        f"question `{sample_p['question_code']}` – "
                        f"positive positions: {sample_p['positive_positions']}, "
                        f"agree positions: {sample_p['agree_positions']}"
                    )

            except Exception as e:
                st.error("Error while loading metadata.")
                st.code(repr(e))
                st.text_area(
                    "Traceback (for debugging)",
                    value=traceback.format_exc(),
                    height=260,
                )


def _render_analytical_query_tester() -> None:
    """
    Developer-only analytical query tester.

    This lets you manually specify:
      - SURVEYR list
      - QUESTION
      - DEMCODE (including "" for overall / no breakdown)
      - LEVEL1ID..LEVEL5ID

    It then:
      - Calls run_analytical_query(...)
      - Shows yearly metrics (MOST POSITIVE OR LEAST NEGATIVE)
      - Builds and displays audit facts (AuditSnapshot)
      - Shows the raw slice in a table

    This is the core of the future conversational engine + audit layer.
    """
    with st.expander("Analytical query test (developer view)", expanded=False):
        st.markdown(
            """
            Use this panel to exercise the analytical query engine and audit layer.

            All four pillars must be specified:
            - **SURVEYR** (list of survey years)  
            - **QUESTION** (e.g., `Q08`)  
            - **DEMCODE** (e.g., `1001` for a demographic, or empty `\"\"` for no breakdown)  
            - **LEVEL1ID–LEVEL5ID** (organization scope)  
            """
        )

        col1, col2 = st.columns(2)

        with col1:
            question_code = st.text_input(
                "Question code (QUESTION, e.g. Q08):",
                value="Q08",
            )
            years_str = st.text_input(
                "Survey years (comma-separated, e.g. 2019,2020,2021,2022,2023,2024):",
                value="2019,2020,2021,2022,2023,2024",
            )
            demcode_input = st.text_input(
                "DEMCODE (empty string \"\" means no breakdown / overall):",
                value="",
            )

        with col2:
            st.markdown("**Organization levels** (integers; leave blank if not used at that level)")
            level1id_str = st.text_input("LEVEL1ID:", value="")
            level2id_str = st.text_input("LEVEL2ID:", value="")
            level3id_str = st.text_input("LEVEL3ID:", value="")
            level4id_str = st.text_input("LEVEL4ID:", value="")
            level5id_str = st.text_input("LEVEL5ID:", value="")

        if st.button("Run analytical query"):
            try:
                # Parse years
                years = []
                for part in years_str.split(","):
                    p = part.strip()
                    if not p:
                        continue
                    years.append(int(p))

                if not years:
                    st.error("Please specify at least one survey year.")
                    return

                # DEMCODE: empty input means no breakdown
                demcode = demcode_input  # may be ""

                # Org levels: build dict only with filled values
                org_levels: dict[str, int] = {}
                if level1id_str.strip():
                    org_levels["LEVEL1ID"] = int(level1id_str.strip())
                if level2id_str.strip():
                    org_levels["LEVEL2ID"] = int(level2id_str.strip())
                if level3id_str.strip():
                    org_levels["LEVEL3ID"] = int(level3id_str.strip())
                if level4id_str.strip():
                    org_levels["LEVEL4ID"] = int(level4id_str.strip())
                if level5id_str.strip():
                    org_levels["LEVEL5ID"] = int(level5id_str.strip())

                if not org_levels:
                    st.error("Please specify at least LEVEL1ID for this test.")
                    return

                params = QueryParameters(
                    survey_years=years,
                    question_code=question_code,
                    demcode=demcode,
                    org_levels=org_levels,
                )

                with st.spinner("Running analytical query and building audit snapshot..."):
                    result = run_analytical_query(params)
                    snapshot = build_audit_snapshot(result)

                st.success("Analytical query succeeded.")

                # Labels
                st.markdown("### Labels")
                st.markdown(
                    f"- **Question:** `{snapshot.question_code}` – EN: *{snapshot.question_label_en}*"
                )
                if snapshot.org_label_en:
                    st.markdown(f"- **Organization:** {snapshot.org_label_en}")
                if snapshot.dem_label_en:
                    st.markdown(f"- **Demographic:** {snapshot.dem_label_en}")
                else:
                    st.markdown(f"- **Demographic:** {snapshot.dem_label_en or 'Overall (all respondents)'}")

                st.markdown(f"- **Metric:** {snapshot.metric_name_en}")

                # Yearly metrics table
                st.markdown("### Yearly metrics (MOST POSITIVE OR LEAST NEGATIVE)")
                rows = []
                for m in result.yearly_metrics:
                    rows.append(
                        {
                            "Year": m.year,
                            "Value": m.value,
                            "Δ vs previous year": m.delta_vs_prev,
                            "N (ANSCOUNT)": m.n,
                        }
                    )
                import pandas as pd

                metrics_df = pd.DataFrame(rows)
                st.dataframe(metrics_df, use_container_width=True)

                # Overall change
                st.markdown("### Overall change")
                if snapshot.overall_delta is not None:
                    st.markdown(
                        f"- Overall change ({min(snapshot.metrics_by_year.keys())} → "
                        f"{max(snapshot.metrics_by_year.keys())}): "
                        f"{snapshot.overall_delta:+.1f} points "
                        f"({snapshot.overall_direction})"
                    )
                else:
                    st.markdown("- Overall change: not available (insufficient data).")

                # Audit facts
                st.markdown("### Audit facts (for AI and human validation)")
                st.markdown(
                    "These are the canonical numerical facts that any AI-generated "
                    "statement must be consistent with."
                )

                st.markdown("**Metrics by year**")
                for year in sorted(snapshot.metrics_by_year.keys()):
                    val = snapshot.metrics_by_year[year]
                    n = snapshot.n_by_year.get(year)
                    st.markdown(
                        f"- {year}: {val:.1f} (N={n if n is not None else 'N/A'})"
                    )

                st.markdown("**Year-over-year trends**")
                if not snapshot.trend_facts:
                    st.markdown("- No year-over-year comparisons available.")
                else:
                    for fact in snapshot.trend_facts:
                        st.markdown(
                            f"- {fact.year_start} → {fact.year_end}: "
                            f"{fact.value_start:.1f} → {fact.value_end:.1f} "
                            f"({fact.delta:+.1f} pts, {fact.direction})"
                        )

                # Raw slice
                st.markdown("### Raw data slice (supporting table)")
                st.dataframe(result.raw_df, use_container_width=True)

            except QueryEngineError as qerr:
                st.error(f"Analytical query failed: {qerr}")
            except Exception as e:
                st.error("Unexpected error while running analytical query.")
                st.code(repr(e))
                st.text_area(
                    "Traceback (for debugging)",
                    value=traceback.format_exc(),
                    height=260,
                )


def run_app() -> None:
    """
    Main UI entrypoint for the PSES Conversational Analytics Chatbot.

    Currently includes:
      - App header and description
      - Placeholder chat area
      - Backend status panel for data loader (CKAN)
      - Metadata status panel for Excel-based metadata
      - Analytical query tester + audit view (developer-only)
    """
    st.set_page_config(
        page_title=APP_NAME,
        page_icon="📊",
        layout="wide",
    )

    st.title(APP_NAME)
    st.caption(f"Prototype version {APP_VERSION}")

    st.markdown(
        """
        This prototype will let you ask natural-language questions about the Public Service 
        Employee Survey (PSES) and receive:
        - A precise, data-grounded answer  
        - A supporting table with the data used  
        - A short narrative summary  
        - Clarifying questions only when strictly necessary  

        The current build is a skeleton. The conversational engine and data pipeline 
        are being implemented step by step, with a focus on strict, auditable use of data.
        """
    )

    st.markdown("---")

    _render_chat_area()
    st.markdown("---")
    _render_backend_status()
    st.markdown("---")
    _render_metadata_status()
    st.markdown("---")
    _render_analytical_query_tester()
