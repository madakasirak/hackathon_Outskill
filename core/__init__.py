"""Deep Researcher core package.

YOUR TODO: refactor the notebook into modular files in this package:

  core/
  ├── __init__.py    (this file)
  ├── llm.py         # OpenRouter LLM factory (reasoning_llm, fast_llm, vision_llm)
  ├── schemas.py     # All Pydantic models (ResearchPlan, FactCheckResult, etc.)
  ├── agents.py      # All 9 agent functions + should_continue routing
  └── graph.py       # build_app() that compiles the StateGraph with checkpointer

Once those exist:
  - app.py imports `from core.graph import build_app`
  - tests/ imports `from core import agents` and `from core.schemas import ...`
  - Everything in the notebook still works because notebook cells can do
    `from core.agents import planner_agent, ...`

The notebook is a working reference; this package is the production layout.
"""
