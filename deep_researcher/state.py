from typing import TypedDict, List, Dict, Any

class ResearchState(TypedDict):
    topic: str
    raw_data: List[str]
    analysis: str
    insights: str
    draft_report: str
    final_report: str
    citations: List[str]
    errors: List[str]
