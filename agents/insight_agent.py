from graph.state import ResearchState
from services.llm import OpenRouterLLM
from utils.prompts import INSIGHT_PROMPT


def run(state: ResearchState, llm: OpenRouterLLM, model: str, temperature: float = 0.2) -> None:
    """Generate strategic insights from analysis."""
    analysis = state.analyzed_findings or "(no analysis)"
    prompt = INSIGHT_PROMPT.format(analysis=analysis)
    messages = [
        {"role": "system", "content": "You are a strategic research synthesizer."},
        {"role": "user", "content": prompt},
    ]
    resp = llm.call_chat(messages=messages, model=model, temperature=temperature)
    state.insights = resp
