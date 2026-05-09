from typing import Dict, Any
from langchain_core.messages import HumanMessage
from utils.llm_utils import get_llm

def source_reliability_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """Checks the reliability of the retrieved sources."""
    if "errors" in state and state["errors"]:
        return state
    raw_data = state.get("raw_data", [])
    topic = state.get("topic")
    llm = get_llm()
    
    prompt = f"Assess the reliability of the following sources retrieved for '{topic}':\n\n{raw_data}\n\nProvide a very brief reliability score and summary."
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        return {"reliability_scores": response.content}
    except Exception as e:
        error_list = state.get("errors", [])
        return {"errors": error_list + [str(e)]}
