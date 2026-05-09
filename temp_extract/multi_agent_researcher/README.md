# Multi-Agent Deep Researcher 🔬

A production-ready **LangGraph** implementation of a multi-agent research pipeline that decomposes complex queries, retrieves and critically analyses evidence, synthesises insights via Chain-of-Thought reasoning, and delivers a professional cited Markdown report — with a QA loop that triggers targeted re-search when coverage is insufficient.

---

## Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────┐
│  🧭 Lead Strategist              │  Decomposes query → 4–5 sub-questions
│     (Router / Query Decomposer)  │  + strategic research plan
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  🔍 Contextual Retriever  ◄─────┼──────────────────────────┐
│  Tavily · DDG · ArXiv · PDF     │                          │
└──────────────┬──────────────────┘                          │
               │                                             │
               ▼                                             │ needs_more_data
┌─────────────────────────────────┐                          │
│  🔬 Critical Analyst            │──── insufficient ────────┘
│  Bias · Contradictions · Gaps   │
└──────────────┬──────────────────┘
               │ sufficient evidence
               ▼
┌─────────────────────────────────┐
│  💡 Insight Generator            │  Chain-of-Thought synthesis
│  Trends · Hypotheses · Insights  │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  📝 Report Builder               │  Markdown · Citations · Confidence Bars
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  ✅ QA Agent                    │──── qa failed, loops left ──► Retriever
│  Coverage · Logic · Caveats     │
└──────────────┬──────────────────┘
               │ qa passed
               ▼
             END
```

---

## Agent Roles

| Agent | Role | Key Output |
|---|---|---|
| **Lead Strategist** | Breaks complex queries into 4–5 focused sub-questions | `sub_questions`, `research_plan` |
| **Contextual Retriever** | Fetches evidence via Tavily, DuckDuckGo, ArXiv, PDF | `raw_sources` |
| **Critical Analyst** | Audits sources for bias, contradictions, and gaps; scores confidence | `findings`, `needs_more_data` |
| **Insight Generator** | CoT synthesis — connects dots, surfaces trends, forms hypotheses | `insights`, `emerging_trends`, `hypotheses` |
| **Report Builder** | Formats Markdown report with executive summary, citations, confidence bars | `final_report` |
| **QA Agent** | Checks coverage, logic, and citation integrity; triggers loop-back | `qa_passed`, `qa_feedback` |

---

## Project Structure

```
multi_agent_researcher/
├── __init__.py           # Public API
├── config.py             # ResearcherConfig dataclass
├── state.py              # Shared ResearchState TypedDict
├── graph.py              # LangGraph DAG with conditional edges
├── main.py               # run_research() + CLI entry point
├── requirements.txt      # Dependencies
├── agents/
│   ├── lead_strategist.py
│   ├── retriever.py
│   ├── critical_analyst.py
│   ├── insight_generator.py
│   ├── report_builder.py
│   └── qa_agent.py
└── tools/
    └── search_tools.py   # Tavily, DDG, ArXiv, Python REPL, PDF parser
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r multi_agent_researcher/requirements.txt
```

### 2. Set environment variables

```bash
export OPENAI_API_KEY=sk-...
export TAVILY_API_KEY=tvly-...   # optional but recommended
```

Or create a `.env` file:

```
OPENAI_API_KEY=sk-...
TAVILY_API_KEY=tvly-...
```

### 3. Run from Python

```python
from multi_agent_researcher import run_research

report = run_research(
    "How will the 2026 interest rate shift affect fintech liquidity in SE Asia?"
)
print(report)
```

### 4. Run from CLI

```bash
python -m multi_agent_researcher.main \
    --query "How will the 2026 interest rate shift affect fintech liquidity in SE Asia?" \
    --output-dir reports/
```

CLI options:

| Flag | Description |
|---|---|
| `--query` / `-q` | Research question (required) |
| `--output-dir` / `-o` | Directory for saved reports |
| `--no-save` | Skip saving to disk |
| `--quiet` | Suppress live progress output |
| `--model` | OpenAI model (default: gpt-4o) |

---

## Configuration

All settings live in `config.py` as a `ResearcherConfig` dataclass:

```python
from multi_agent_researcher.config import config

config.max_sub_questions = 5         # sub-questions per query
config.max_retrieval_rounds = 3      # max analyst→retriever loops
config.max_qa_loops = 2              # max QA→retriever loops
config.confidence_threshold = 0.70   # below this → request more evidence
config.llm_model = "gpt-4o"
```

---

## State Schema

The shared `ResearchState` TypedDict carries all data between agents:

```python
{
    # Input
    "original_query": str,

    # Lead Strategist
    "sub_questions": list[str],
    "research_plan": str,

    # Retriever
    "raw_sources": list[Source],       # appended each round
    "retrieval_round": int,

    # Critical Analyst
    "findings": list[Finding],         # per sub-question
    "needs_more_data": bool,
    "data_gaps": list[str],

    # Insight Generator
    "insights": list[str],
    "hypotheses": list[str],
    "emerging_trends": list[str],

    # Report Builder
    "final_report": str,               # Markdown

    # QA Agent
    "qa_passed": bool,
    "qa_feedback": list[QAFeedback],
    "qa_loop_count": int,

    # Orchestration
    "current_agent": str,
    "completed": bool,
    "error_log": list[str],
}
```

---

## Conditional Routing Logic

```
Critical Analyst → needs_more_data AND rounds_remaining?
    YES → Retriever (targeted re-search for data_gaps)
    NO  → Insight Generator

QA Agent → qa_passed?
    NO AND loops_remaining → Retriever (narrowed sub_questions from issues)
    YES OR no loops left   → END
```

---

## Extending the Pipeline

### Add a new agent

1. Create `agents/my_agent.py` with a `my_agent_node(state) -> dict` function.
2. Register it in `agents/__init__.py`.
3. Add it to the graph in `graph.py`:
   ```python
   graph.add_node("my_agent", my_agent_node)
   graph.add_edge("insight_generator", "my_agent")
   graph.add_edge("my_agent", "report_builder")
   ```

### Add a new tool

Add a `@tool`-decorated function in `tools/search_tools.py` and include it in `RETRIEVAL_TOOLS` or `CALCULATION_TOOLS`.

---

## Example Output Structure

```markdown
# Impact of 2026 Interest Rate Shifts on Fintech Liquidity in SE Asia

## Executive Summary
...

## Research Findings

### Central bank rate trajectory in ASEAN economies (2025–2026)
**Confidence Score: 0.8 / 1.0** ⬛⬛⬛⬛⬜

[Evidence-based answer with inline citations [1][2]...]

> ⚠️ **Analyst Notes:** IMF projections contradict ADB forecasts on timing...

## Insights & Strategic Implications
...

## References
[1] ADB Outlook 2026 (https://...) — web
[2] IMF Regional Economic Outlook (https://...) — web
```

---

## License

MIT — use freely, attribution appreciated.
