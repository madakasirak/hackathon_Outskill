import os
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI

OPENROUTER_BASE = "https://openrouter.ai/api/v1"
OPENROUTER_HEADERS = {
    "HTTP-Referer": "https://hackathon.local",
    "X-Title": "Deep Researcher Hackathon",
}

# The user has their OPENROUTER_API_KEY in the environment
# Fast LLM for standard extractions
fast_llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    openai_api_key=os.environ.get("OPENROUTER_API_KEY", "dummy"),
    openai_api_base=OPENROUTER_BASE, 
    temperature=0.1,
    default_headers=OPENROUTER_HEADERS,
)

# Reasoning LLM for reflection and report building
reasoning_llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    openai_api_key=os.environ.get("OPENROUTER_API_KEY", "dummy"),
    openai_api_base=OPENROUTER_BASE, 
    temperature=0.2,
    default_headers=OPENROUTER_HEADERS,
)
