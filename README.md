# 🔬 Multi-Agent AI Deep Researcher

A streamlined, high-performance 4-agent research assistant built on LangGraph. It produces fact-checked, source-rated research reports and features a Model Council for multi-perspective synthesis, robust token/billing tracking, and a premium Streamlit UI. Submitted for the Engineering Accelerator hackathon.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/built%20with-Streamlit-FF4B4B)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 🌟 What it does

Type a research question or upload documents. Four specialized agents collaborate to produce a structured, highly analytical report with:

- **Dynamic Parallel Retrieval**: Automatically selects the best tools (Tavily, ArXiv, Wikipedia, DuckDuckGo) and fetches data concurrently.
- **Local RAG (FAISS)**: Upload PDFs or TXT files directly in the UI. They are instantly indexed and searched.
- **The Model Council**: The Analyzer agent consults multiple distinct LLMs (e.g., GPT-4o-mini and Claude 3 Haiku) to gain diverse perspectives, synthesizing them into profound insights.
- **Self-Correction via Reflection**: The system automatically detects coverage gaps or contradictions in the research and loops back to find missing information.
- **Stats & Billing Dashboard**: A beautiful analytics dashboard tracks every API call, providing a detailed breakdown of input/output tokens and costs **per-model** (e.g., GPT vs Claude) for every query.
- **Contextual Follow-up Chat**: After a report is generated, you can ask follow-up questions in a chat interface that strictly uses the retrieved evidence as context.

## 🏗️ Architecture

```mermaid
graph TD
    START((START)) --> Retriever[🔍 Retriever Agent]
    Retriever --> Analyzer[🔬 Analyzer Agent]
    Analyzer --> Reflection[🤔 Reflection Agent]
    
    Reflection -- Needs More Data --> Retriever
    Reflection -- Research Complete --> ReportBuilder[📝 Report Builder]
    
    ReportBuilder --> END((END))
```

## 🎬 Recommended Demo Query

> *"Is nuclear energy net-positive for global climate goals when accounting for waste, costs, and accidents?"*

This query reliably exercises the full pipeline:
1. Retriever fires across 4 tools in parallel
2. Analyzer's Model Council surfaces consensus + unique perspectives
3. Reflection identifies contradictions (climate benefit vs. waste/safety)
4. *Loop fires* — Retriever runs again with the gap as new query
5. Final report shows insights from both rounds with contradictions flagged

Backup queries (if the first one's API hits rate limits):
- "How do GLP-1 drugs reshape food and beverage company strategy?"
- "Can open-source LLMs replace enterprise SaaS copilots for sensitive workloads?"

### The Four Agents (Mapped to Hackathon PDF Requirements)

| PDF Requirement | Our Agent | What it does |
|---|---|---|
| *Contextual Retriever Agent* | `retriever_agent` | Dynamic tool routing across Tavily, ArXiv, Wikipedia, DuckDuckGo, and local FAISS — runs in parallel via ThreadPoolExecutor |
| *Critical Analysis* + *Insight Generation* | `analyzer_agent` | Embeds findings into FAISS for grounding, runs Model Council (GPT-4o-mini + Claude Haiku), synthesizes cross-model consensus into insights |
| *Report Builder Agent* | `report_builder_agent` | Compiles structured markdown report with citations, contradictions section |
| Extra agent (per PDF: "any more agents you want to add") | `reflection_agent` | Identifies coverage gaps and contradictions; drives the conditional loop back to Retriever |

*Note:* The `analyzer_agent` deliberately combines two PDF roles — Critical Analysis (RAG-grounded evidence assessment) and Insight Generation (Model Council synthesis) — because the Council pattern is most powerful when both responsibilities feed the same multi-model reasoning step.

## 💻 Tech Stack

- **Orchestration**: LangGraph (conditional routing + persistent memory)
- **LLM Gateway**: OpenRouter (one key, all providers)
- **RAG**: FAISS + HuggingFace (`all-MiniLM-L6-v2`)
- **Retrieval**: Tavily, ArXiv, Wikipedia, DuckDuckGo
- **State & Memory**: LangGraph SqliteSaver checkpointer (`checkpoints.db`)
- **Telemetry**: Custom SQLite Database (`stats.db`) for Token & Cost Tracking
- **UI**: Streamlit with Custom CSS (Glassmorphism & glowing agent widgets)

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- `uv` package manager (recommended)
- API keys (both have free tiers):
  - **OpenRouter** — https://openrouter.ai/keys
  - **Tavily** — https://tavily.com

### Setup

```bash
# Clone and enter the repo
git clone https://github.com/your-org/deep-researcher
cd deep-researcher

# Copy the env template and fill in your keys
cp .env.example .env
# Edit .env with your real OPENROUTER_API_KEY and TAVILY_API_KEY

# Set up the virtual environment and install dependencies using uv
uv venv
source .venv/bin/activate
uv pip sync requirements.txt
# Alternatively, use 'uv sync' if relying on pyproject.toml

# Launch the Streamlit app
streamlit run app.py
```

The app opens at `http://localhost:8501`.

## 📁 Folder Structure

```
deep-researcher/
├── README.md                            # You are here
├── pyproject.toml                       # Project metadata
├── requirements.txt                     # Pinned runtime dependencies
├── .env.example                         # Template for required env vars
├── .gitignore                           # Standard ignores
├── app.py                               # Premium Streamlit UI & Entry Point
│
├── agents/                              # The 4 Core LangGraph Agents
│   ├── retriever_agent.py               # Parallel retrieval + FAISS
│   ├── analyzer_agent.py                # Model Council logic
│   ├── reflection_agent.py              # Gap detection loop
│   └── report_agent.py                  # Final markdown assembly
│
├── graph/                               # LangGraph Orchestration
│   ├── state.py                         # TypedDict for agent memory
│   └── workflow.py                      # Node and Edge assembly
│
├── services/                            # Utilities & Integrations
│   ├── callbacks.py                     # Token tracking and cost calculation
│   ├── db.py                            # SQLite stats database
│   └── llm.py                           # OpenRouter LLM configurations
│
└── tools/                               # Custom Tools
    ├── utilities.py                     # PDF parsing and REPL
    └── __init__.py
```

## 📊 The Dashboard

This project includes a dedicated **Stats Dashboard** tab inside the Streamlit UI. It intercepts every LangChain model call to log `input_tokens` and `output_tokens`, matches them against a pricing matrix, and displays your lifetime API costs alongside a history of all your queries. 

Each past query is displayed as an interactive, expandable row. Clicking on a query reveals a **Model Breakdown** table, showing you exactly how many tokens and fractional cents were spent on the different models used during that run (e.g., how much the `gpt-4o-mini` Retriever cost versus the `claude-3-haiku` Model Council synthesis).

## 🏆 Hackathon Scorecard

| Criterion | How we hit it |
|---|---|
| **LangGraph Architecture** | ✅ 4 distinct nodes wired with conditional edges |
| **Conditional Routing** | ✅ Reflection node routes back to Retriever if gaps are found |
| **Iterative Loop** | ✅ Gap-filling loop capped by `MAX_ITERATIONS` |
| **RAG Vector Store** | ✅ FAISS + HuggingFace Embeddings for local document QA |
| **Multi-source Retrieval** | ✅ Parallel execution of Tavily, ArXiv, Wikipedia, DDG |
| **Memory** | ✅ SqliteSaver checkpointer for state persistence |
| **Production UI** | ✅ Premium Streamlit interface with glowing animations & stats tracking |
| **LLM Independence** | ✅ OpenRouter configuration allows instant swapping between OpenAI, Anthropic, and Google models |
| **Follow-up Chat** | ✅ Context-aware follow-up chat post-generation |

## 📜 License

MIT — see `LICENSE` file.
