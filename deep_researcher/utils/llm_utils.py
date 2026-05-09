import os
from langchain_openai import ChatOpenAI

def get_llm():
    return ChatOpenAI(
        model=os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o-mini"),
        api_key=os.environ.get("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
        streaming=True
    )

def get_council_llm(model_name: str):
    return ChatOpenAI(
        model=model_name,
        api_key=os.environ.get("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
        streaming=False
    )
