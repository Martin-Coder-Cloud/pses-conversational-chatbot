from __future__ import annotations
import streamlit as st

from pses_chatbot.config import APP_NAME, APP_VERSION


def run_app() -> None:
    """
    Main UI entrypoint for the PSES Conversational Analytics Chatbot.

    For now, this is a simple placeholder page. We'll gradually wire in:
    - Core data loading and metadata checks
    - Conversational orchestration
    - Parameter inference and result rendering
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

    st.subheader("Prototype chat area")

    # Very simple chat-like interaction for now
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

        # Placeholder bot response – we'll replace this with real orchestration later
        bot_reply = (
            "Thanks for your question! The conversational analytics engine is not implemented yet.\n\n"
            "In the next steps, this chatbot will:\n"
            "- Infer the PSES question, years, demographics, organization and level from your text\n"
            "- Retrieve the corresponding data from the full PSES dataset (PS-wide + departments)\n"
            "- Generate a short, validated analytical summary\n"
            "- Show you the data table used for the answer"
        )
        st.session_state.chat_history.append({"role": "assistant", "content": bot_reply})

        # Force rerun to show the new messages immediately
        st.experimental_rerun()
