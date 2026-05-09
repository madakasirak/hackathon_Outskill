import os
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI

OPENROUTER_BASE = "https://openrouter.ai/api/v1"
OPENROUTER_HEADERS = {
    "HTTP-Referer": "https://hackathon.local",
    "X-Title": "Deep Researcher Hackathon",
}

def _resolve_model():
    """Resolve the selected model from the Streamlit dropdown. Falls back to gpt-4o-mini."""
    model = os.environ.get("LLM_MODEL", "openai/gpt-4o-mini")
    # If it's the auto-select string from the dropdown, default to gpt-4o-mini
    if "auto" in model.lower() or "Auto-select" in model:
        return "openai/gpt-4o-mini"
    return model

def get_fast_llm():
    """Factory for the fast LLM used for extractions and tool selection. Always routed through OpenRouter."""
    return ChatOpenAI(
        model=_resolve_model(),
        openai_api_key=os.environ.get("OPENROUTER_API_KEY", "dummy"),
        openai_api_base=OPENROUTER_BASE, 
        temperature=0.1,
        default_headers=OPENROUTER_HEADERS,
    )

def get_reasoning_llm():
    """Factory for the reasoning LLM used for reflection and report building. Always routed through OpenRouter."""
    return ChatOpenAI(
        model=_resolve_model(),
        openai_api_key=os.environ.get("OPENROUTER_API_KEY", "dummy"),
        openai_api_base=OPENROUTER_BASE, 
        temperature=0.2,
        default_headers=OPENROUTER_HEADERS,
    )

def get_council_llm(model_name: str, temperature: float = 0.5):
    """Factory for Model Council secondary members. Always routed through OpenRouter."""
    return ChatOpenAI(
        model=model_name,
        openai_api_key=os.environ.get("OPENROUTER_API_KEY", "dummy"),
        openai_api_base=OPENROUTER_BASE,
        temperature=temperature,
        default_headers=OPENROUTER_HEADERS,
    )
