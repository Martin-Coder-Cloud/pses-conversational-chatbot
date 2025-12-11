from __future__ import annotations

import traceback

import streamlit as st

from pses_chatbot.config import APP_NAME, APP_VERSION
from pses_chatbot.core.data_loader import query_pses_results


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


def run_app() -> None:
    """
    Main UI entrypoint for the PSES Conversational Analytics Chatbot.

    Currently includes:
      - App header and description
      - Placeholder chat area
      - Backend status panel for testing the data loader
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
