from typing import TypedDict, List, Annotated, Optional
from operator import add
from pydantic import BaseModel, Field

class ReflectionResult(BaseModel):
    coverage_gaps: List[str] = Field(description="Areas where evidence is lacking")
    contradictions: List[str] = Field(description="Claims that disagree across sources")
    needs_more_research: bool = Field(description="True if we must loop back to gather more info")

class ResearchState(TypedDict):
    query: str
    # Using a list of dicts for documents to avoid importing langchain classes here
    documents: Annotated[List[dict], add]
    insights: List[str]
    reflection: Optional[ReflectionResult]
    iteration: int
    final_report: str

MAX_ITERATIONS = 2
