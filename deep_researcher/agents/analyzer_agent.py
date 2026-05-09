from typing import Dict, Any
from langchain_core.messages import HumanMessage
from utils.llm_utils import get_llm

def analyzer_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """Analyzes the retrieved data."""
    if "errors" in state and state["errors"]:
        return state
        
    raw_data = state.get("raw_data", [])
    topic = state.get("topic")
    llm = get_llm()
    
    prompt = f"Analyze the following raw data extracted for the topic '{topic}':\n\n{raw_data}\n\nIdentify the core themes, trends, and any contradictory points."
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        return {"analysis": response.content}
    except Exception as e:
        return {"errors": [state.get("errors", []) + [str(e)]]}
