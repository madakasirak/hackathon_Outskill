from pydantic import BaseModel
from typing import List, Any


class ResearchState(BaseModel):
    query: str = ""
    retrieved_docs: List[dict] = []
    analyzed_findings: str = ""
    insights: str = ""
    final_report: str = ""
    sources: List[str] = []
    messages: List[str] = []
