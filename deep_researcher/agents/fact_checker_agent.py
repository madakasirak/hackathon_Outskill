from typing import Dict, Any
from langchain_core.messages import HumanMessage
from utils.llm_utils import get_llm

def fact_checker_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """Fact checks the analysis."""
    if "errors" in state and state["errors"]:
        return state
    analysis = state.get("analysis", "")
    llm = get_llm()
    
    prompt = f"Fact-check the following analysis for accuracy and identify any unsupported claims:\n\n{analysis}\n\nProvide your fact-checking results briefly."
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        return {"fact_checking_results": response.content}
    except Exception as e:
        error_list = state.get("errors", [])
        return {"errors": error_list + [str(e)]}
