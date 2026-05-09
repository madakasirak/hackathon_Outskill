from typing import Dict, Any
from langchain_core.messages import HumanMessage
from utils.llm_utils import get_llm

def planner_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """Plans the research based on the topic."""
    if "errors" in state and state["errors"]:
        return state
    topic = state.get("topic")
    llm = get_llm()
    prompt = f"Create a brief research plan for the topic: '{topic}'. Keep it under 3 bullet points."
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        return {"plan": response.content, "loop_count": state.get("loop_count", 0)}
    except Exception as e:
        error_list = state.get("errors", [])
        return {"errors": error_list + [str(e)], "loop_count": 0}
