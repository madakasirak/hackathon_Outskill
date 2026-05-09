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
    insights: Annotated[List[str], add]
    reflection: Optional[ReflectionResult]
    iteration: int
    final_report: str

# Total retrieval rounds: initial pass + (MAX_ITERATIONS - 1) reflection loops.
# Currently: 1 initial pass + 1 loop = 2 total rounds.
# Set to 3 to allow 2 loops (more thorough, ~2x API cost).
MAX_ITERATIONS = 2
