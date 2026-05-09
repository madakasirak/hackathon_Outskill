import os
from dotenv import load_dotenv
import tempfile
from typing import Optional

import streamlit as st

from graph.state import ResearchState
from graph.workflow import run_workflow
from services.llm import OpenRouterLLM


load_dotenv()

st.set_page_config(page_title="Multi-Agent Deep Researcher", layout="wide")


def main() -> None:
    st.title("Multi-Agent AI Deep Researcher — MVP")

    # Sidebar configuration
    with st.sidebar:
        st.header("Settings")
        api_key = st.text_input("OpenRouter API Key", type="password", value=os.getenv("OPENROUTER_API_KEY", ""))
        model = st.selectbox("Model", options=["openai/gpt-4.1-mini", "anthropic/claude-3.7-sonnet"], index=0)
        temperature = st.slider("Temperature", 0.0, 1.0, 0.2)
        st.markdown("---")
        st.write("Workflow: Retriever → Analysis → Insights → Report")

    query = st.text_input("Research question / topic", placeholder="Impact of AI agents on software engineering")
    uploaded_file = st.file_uploader("Upload PDF (required for RAG)", type=["pdf"])

    run_button = st.button("Run Research")

    status_placeholder = st.empty()

    if run_button and query:
        if not api_key:
            st.error("Please provide an OpenRouter API key in the sidebar.")
            return
        if not uploaded_file:
            st.error("Please upload a PDF to run FAISS RAG retrieval.")
            return

        # Save uploaded PDF to temp path
        tmp_dir = tempfile.gettempdir()
        pdf_path = os.path.join(tmp_dir, uploaded_file.name)
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Initialize state and services
        state = ResearchState(query=query)
        llm = OpenRouterLLM(api_key=api_key)

        # Status callback to update UI
        def status_cb(stage: str, msg: Optional[str] = None) -> None:
            status_placeholder.markdown(f"**{stage}** — {msg or ''}")

        with st.spinner("Running research workflow..."):
            try:
                run_workflow(state=state, llm=llm, model=model, temperature=temperature, status_callback=status_cb, pdf_path=pdf_path)
            except Exception as e:
                st.error(f"Workflow failed: {e}")
                return

        st.success("Research complete")

        # Expanders for results
        with st.expander("Retrieved Documents", expanded=False):
            for d in state.retrieved_docs:
                st.markdown(f"**[{d.get('title', 'source')}]({d.get('url')})**")
                st.write(d.get("snippet"))

        with st.expander("Analysis", expanded=False):
            st.markdown(state.analyzed_findings)

        with st.expander("Insights", expanded=False):
            st.markdown(state.insights)

        with st.expander("Final Report (Markdown)", expanded=True):
            st.markdown(state.final_report)
            st.download_button("Download Report", state.final_report, file_name="research_report.md")


if __name__ == "__main__":
    main()
