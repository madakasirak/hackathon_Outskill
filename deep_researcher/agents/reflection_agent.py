from typing import Dict, Any
from langchain_core.messages import HumanMessage
from utils.llm_utils import get_llm

def reflection_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """Reflects on the current state and decides whether to loop back."""
    if "errors" in state and state["errors"]:
        return state
        
    analysis = state.get("analysis", "")
    fact_checking = state.get("fact_checking_results", "")
    loop_count = state.get("loop_count", 0)
    
    if loop_count >= 1:
        # Prevent infinite loops, force sequence out
        return {"reflection_feedback": "Max loops reached. Proceeding to report.", "is_satisfied": True, "loop_count": loop_count + 1}
        
    llm = get_llm()
    prompt = f"""
    Review the analysis and fact-checking results. Are there significant gaps, missing information, or contradictions that require more research?
    
    Analysis: {analysis}
    Fact Checks: {fact_checking}
    
    Reply with "SATISFIED" if the research is complete and robust.
    Otherwise, explain what is missing or contradicts, starting your response with "NEEDS_MORE_RESEARCH:".
    """
    
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        content = response.content.strip()
        if content.startswith("SATISFIED") or "SATISFIED" in content.upper()[:20]:
            return {"is_satisfied": True, "reflection_feedback": content, "loop_count": loop_count + 1}
        else:
            return {"is_satisfied": False, "reflection_feedback": content, "loop_count": loop_count + 1}
    except Exception as e:
        error_list = state.get("errors", [])
        return {"errors": error_list + [str(e)], "is_satisfied": True}
