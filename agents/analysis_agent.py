from typing import Any
from graph.state import ResearchState
from services.llm import OpenRouterLLM
from utils.prompts import ANALYSIS_PROMPT


def run(state: ResearchState, llm: OpenRouterLLM, model: str, temperature: float = 0.2) -> None:
    """Analyze retrieved documents and populate state.analyzed_findings."""
    # Build context from retrieved docs
    pieces = []
    for d in state.retrieved_docs:
        pieces.append(f"URL: {d.get('url')}\nSnippet: {d.get('snippet')}\n")
    context = "\n---\n".join(pieces) or "(no results)"

    prompt = ANALYSIS_PROMPT.format(context=context)
    messages = [
        {"role": "system", "content": "You are an expert research analyst."},
        {"role": "user", "content": prompt},
    ]

    resp = llm.call_chat(messages=messages, model=model, temperature=temperature)
    state.analyzed_findings = resp
