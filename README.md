# Multi-Agent AI Deep Researcher — MVP

Minimal hackathon-friendly POC demonstrating a multi-agent deep research workflow.

Features
- Multi-agent pipeline: Retriever → Analysis → Insight → Report
- Lightweight web retrieval (DuckDuckGo + page scraping)
- LangGraph-friendly linear workflow orchestration
- OpenRouter LLM integration (chat-style)
- Streamlit UI with status updates and downloadable report

Architecture (ASCII)

User -> Streamlit UI -> Workflow Orchestrator -> [Retriever, Analysis, Insight, Report] -> Output

Project structure

deep_researcher/
  - app.py
  - services/llm.py
  - tools/web_search.py
  - agents/
    - retriever_agent.py
    - analysis_agent.py
    - insight_agent.py
    - report_agent.py
  - graph/
    - state.py
    - workflow.py
  - utils/prompts.py
  - requirements.txt
  - .env.example

Quickstart

1. Create and activate a Python 3.11+ environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Copy `.env.example` to `.env` and add your `OPENROUTER_API_KEY`.

4. Run the Streamlit app:

```bash
streamlit run app.py
```

macOS notes
- Installing `faiss-cpu` via `pip` can be problematic on macOS. If you encounter issues when running the FAISS RAG flow, install FAISS via conda:

```bash
conda install -c conda-forge faiss-cpu
```

Then install the remaining Python deps:

```bash
pip install -r requirements.txt
```

Usage
- Enter a research query in the main UI.
- Choose the model and temperature in the sidebar.
- Click "Run Research" and watch stage updates.
- Expand sections for Retrieved Docs, Analysis, Insights, and the Final Report.

Notes & Design Choices
- The workflow is intentionally linear and minimal for clarity.
- `services/llm.py` wraps OpenRouter's chat completions endpoint directly via HTTP.
- `tools/web_search.py` uses DuckDuckGo's HTML interface and fetches lightweight page content.
- Pydantic `ResearchState` holds shared state across agents.

Future Improvements
- Replace simple retrieval with a proper RAG/vector DB.
- Add streaming token-level UI updates.
- Add simple source reliability scoring and provenance.
- Add small local cache to avoid repeated fetching in a session.

License
- MIT (for demo purposes)

**Architecture**

This project is intentionally simple and linear to make the multi-agent flow clear during a demo.

ASCII overview

User -> Streamlit UI -> Workflow Orchestrator (LangGraph-friendly) -> Agents -> Output

- Retriever (FAISS RAG on uploaded PDF)
- Analysis (LLM-based evidence summary)
- Insight (LLM-based strategic synthesis)
- Report (LLM-based polished markdown report)

Mermaid diagram (conceptual)

```mermaid
flowchart LR
  U[User]
  UI[Streamlit UI]
  WF[Workflow Orchestrator]
  R[Retriever Agent\n(FAISS RAG)]
  A[Analysis Agent]
  I[Insight Agent]
  P[Report Agent]
  O[Output\n(Markdown Report + Sources)]

  U --> UI --> WF
  WF --> R --> A --> I --> P --> O
```

Notes on architecture
- The `Workflow Orchestrator` is a minimal linear controller implemented in `graph/workflow.py` that wires agent calls and shares a `ResearchState` Pydantic model.
- Retrieval uses a local FAISS vector store built from an uploaded PDF via `langchain` loaders and `sentence-transformers` embeddings.
- All LLM calls are routed through `services/llm.py`, which hits the OpenRouter API; the chosen model and temperature are configurable from the Streamlit sidebar.
- The UI focuses on clarity for demos: a single query input, PDF upload, and stage-by-stage status updates.

