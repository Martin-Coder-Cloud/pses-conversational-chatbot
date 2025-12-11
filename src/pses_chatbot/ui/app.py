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


def _render_chat_area() -> None:
    """
    Very simple placeholder chat area.

    Later, this will be wired to the full conversational pipeline:
      - intent parsing
      - parameter inference
      - data retrieval
      - narrative generation
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
            "- Show you the data table used for the answer"
        )
        st.session_state.chat_history.append({"role": "assistant", "content": bot_reply})

        # Force rerun to show the new messages immediately
        st.experimental_rerun()


def _render_backend_status() -> None:
    """
    Developer-only backend status panel.

    This lets you quickly test:
      - Connectivity to the CKAN DataStore API
      - That PSES_DATASTORE_RESOURCE_ID is valid
      - The basic schema (columns) of the PSES results table

    It does NOT run automatically; it only runs when you click the button.
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

    This loads all key metadata from the Excel workbook:
      - Questions
      - Scales (answer options)
      - Demographics
      - Organization hierarchy
      - Positive/Neutral/Negative/Agree mappings

    and displays basic counts so you can confirm everything is wired correctly.
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
                        f"DEMCODE `{sample_d['demcode']}` – "
                        f"EN: *{sample_d['label_en'][:60]}* "
                        f"(BYCOND: {sample_d['bycond']})"
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


def run_app() -> None:
    """
    Main UI entrypoint for the PSES Conversational Analytics Chatbot.

    Currently includes:
      - App header and description
      - Placeholder chat area
      - Backend status panel for data loader (CKAN)
      - Metadata status panel for Excel-based metadata
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
        will be implemented step by step.
        """
    )

    st.markdown("---")

    _render_chat_area()
    st.markdown("---")
    _render_backend_status()
    st.markdown("---")
    _render_metadata_status()
