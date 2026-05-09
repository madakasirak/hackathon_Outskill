from typing import TypedDict, List, Dict, Any

class ResearchState(TypedDict, total=False):
    topic: str
    plan: str
    
    raw_data: List[str]
    citations: List[str]
    
    reliability_scores: str
    multimodal_data: str
    
    analysis: str
    fact_checking_results: str
    reflection_feedback: str
    
    is_satisfied: bool
    loop_count: int
    
    visualizations: str
    
    final_report: str
    errors: List[str]
    local_vector_store: str

