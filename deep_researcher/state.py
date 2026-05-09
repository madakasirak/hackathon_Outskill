from typing import TypedDict, List, Dict, Any

class ResearchState(TypedDict, total=False):
    topic: str
    raw_data: List[str]
    analysis: str
    insights: str
    draft_report: str
    final_report: str
    citations: List[str]
    errors: List[str]
    local_vector_store: str
