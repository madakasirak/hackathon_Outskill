from typing import Dict, Any
from langchain_core.messages import HumanMessage
from utils.llm_utils import get_llm

def visualization_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """Prepares or decides on visualizations for the report."""
    if "errors" in state and state["errors"]:
        return state
        
    topic = state.get("topic", "")
    llm = get_llm()
    prompt = f"Suggest 1-2 types of charts or visual diagrams that would enhance a report on '{topic}'. Briefly describe them."
    
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        return {"visualizations": response.content}
    except Exception as e:
        error_list = state.get("errors", [])
        return {"errors": error_list + [str(e)]}
