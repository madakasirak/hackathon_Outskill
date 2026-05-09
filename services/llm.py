import os
import requests
from typing import List, Dict, Any


class OpenRouterLLM:
    """Simple OpenRouter HTTP wrapper compatible with Chat-style messages.

    It uses the OpenRouter public endpoint to produce completions.
    """

    def __init__(self, api_key: str, base_url: str = "https://openrouter.ai/api/v1") -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")

    def call_chat(self, messages: List[Dict[str, str]], model: str = "openai/gpt-4.1-mini", temperature: float = 0.2) -> str:
        url = f"{self.base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {"model": model, "messages": messages, "temperature": temperature}
        resp = requests.post(url, json=payload, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        # Attempt to extract assistant text from common shapes
        if "choices" in data and len(data["choices"]) > 0:
            content = data["choices"][0].get("message", {}).get("content") or data["choices"][0].get("text")
            return content or ""
        # Fallback
        return data.get("result", "") or ""
