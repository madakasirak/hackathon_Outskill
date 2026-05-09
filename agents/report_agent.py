from graph.state import ResearchState
from services.llm import OpenRouterLLM
from utils.prompts import REPORT_PROMPT


def run(state: ResearchState, llm: OpenRouterLLM, model: str, temperature: float = 0.2) -> None:
    """Generate the final markdown report and store it in state.final_report."""
    query = state.query
    analysis = state.analyzed_findings or ""
    insights = state.insights or ""
    sources = "\n".join(state.sources or [])

    prompt = REPORT_PROMPT.format(query=query, analysis=analysis, insights=insights, sources=sources)
    messages = [
        {"role": "system", "content": "You are a clear, crisp technical writer."},
        {"role": "user", "content": prompt},
    ]
    resp = llm.call_chat(messages=messages, model=model, temperature=temperature)
    state.final_report = resp
