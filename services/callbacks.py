from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from typing import Any

# Pricing per 1M tokens (in USD)
MODEL_PRICING = {
    "openai/gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "anthropic/claude-3-haiku": {"input": 0.25, "output": 1.25},
    "anthropic/claude-3-5-sonnet": {"input": 3.00, "output": 15.00},
    "openai/gpt-4o": {"input": 2.50, "output": 10.00},
    "google/gemini-flash-1.5": {"input": 0.075, "output": 0.30},
}

class TokenTrackingCallback(BaseCallbackHandler):
    def __init__(self):
        super().__init__()
        self.input_tokens = 0
        self.output_tokens = 0
        self.models_used = set()
        self.estimated_cost = 0.0
        self.model_breakdown = {}

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> Any:
        # Check if token usage exists in the llm_output
        if response.llm_output and "token_usage" in response.llm_output:
            usage = response.llm_output["token_usage"]
            
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            
            self.input_tokens += prompt_tokens
            self.output_tokens += completion_tokens
            
            model_name = response.llm_output.get("model_name", "unknown")
            self.models_used.add(model_name)
            
            # Calculate cost
            pricing = MODEL_PRICING.get(model_name, None)
            if pricing:
                cost = (prompt_tokens / 1_000_000) * pricing["input"] + (completion_tokens / 1_000_000) * pricing["output"]
            else:
                # Default to gpt-4o-mini pricing as a rough fallback if model isn't mapped
                cost = (prompt_tokens / 1_000_000) * 0.15 + (completion_tokens / 1_000_000) * 0.60
                
            self.estimated_cost += cost
            
            if model_name not in self.model_breakdown:
                self.model_breakdown[model_name] = {"input_tokens": 0, "output_tokens": 0, "cost": 0.0}
                
            self.model_breakdown[model_name]["input_tokens"] += prompt_tokens
            self.model_breakdown[model_name]["output_tokens"] += completion_tokens
            self.model_breakdown[model_name]["cost"] += cost
