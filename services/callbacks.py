from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from typing import Any, Dict, List, Optional
from uuid import UUID

# Pricing per 1M tokens (in USD)
MODEL_PRICING = {
    "openai/gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "anthropic/claude-3-haiku": {"input": 0.25, "output": 1.25},
    "anthropic/claude-3.5-sonnet": {"input": 3.00, "output": 15.00},
    "openai/gpt-4o": {"input": 2.50, "output": 10.00},
    "google/gemini-flash-1.5": {"input": 0.075, "output": 0.30},
    "google/gemini-2.0-flash-001": {"input": 0.10, "output": 0.40},
    "google/gemini-1.5-pro": {"input": 1.25, "output": 5.00},
}

# Known agent stages — used to filter out LangChain's internal tags like "seq:step:1"
KNOWN_STAGES = {"Retriever", "Analyzer", "Analyzer Synthesis", "Reflection", "Report Builder", "Follow-up Chat"}

class TokenTrackingCallback(BaseCallbackHandler):
    def __init__(self):
        super().__init__()
        self.input_tokens = 0
        self.output_tokens = 0
        self.models_used = set()
        self.estimated_cost = 0.0
        self.model_breakdown = {}
        # Nested stage→model breakdown: { "Analyzer": { "openai/gpt-4o-mini": { tokens, cost } } }
        self.stage_breakdown = {}
        self.run_id_to_tag = {}

    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], *, run_id: UUID, parent_run_id: Optional[UUID] = None, tags: Optional[List[str]] = None, metadata: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Any:
        stage_tag = "Other"
        if tags:
            for t in tags:
                if t in KNOWN_STAGES:
                    stage_tag = t
                    break
        self.run_id_to_tag[run_id] = stage_tag

    def on_llm_end(self, response: LLMResult, *, run_id: UUID, parent_run_id: Optional[UUID] = None, **kwargs: Any) -> Any:
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
            
            # Flat model breakdown (kept for backwards compatibility)
            if model_name not in self.model_breakdown:
                self.model_breakdown[model_name] = {"input_tokens": 0, "output_tokens": 0, "cost": 0.0}
            self.model_breakdown[model_name]["input_tokens"] += prompt_tokens
            self.model_breakdown[model_name]["output_tokens"] += completion_tokens
            self.model_breakdown[model_name]["cost"] += cost
            
            # Nested stage→model breakdown
            stage_tag = self.run_id_to_tag.get(run_id, "Other")
            if stage_tag not in self.stage_breakdown:
                self.stage_breakdown[stage_tag] = {}
            if model_name not in self.stage_breakdown[stage_tag]:
                self.stage_breakdown[stage_tag][model_name] = {"input_tokens": 0, "output_tokens": 0, "cost": 0.0}
            self.stage_breakdown[stage_tag][model_name]["input_tokens"] += prompt_tokens
            self.stage_breakdown[stage_tag][model_name]["output_tokens"] += completion_tokens
            self.stage_breakdown[stage_tag][model_name]["cost"] += cost
