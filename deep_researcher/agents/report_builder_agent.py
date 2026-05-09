from typing import Dict, Any
from langchain_core.messages import HumanMessage
from utils.llm_utils import get_llm

def report_builder_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """Drafts the final markdown report."""
    if "errors" in state and state["errors"]:
        error_msg = "\n".join(state["errors"])
        return {"final_report": f"## Research Failed\n\nErrors encountered:\n{error_msg}"}
        
    topic = state.get("topic")
    analysis = state.get("analysis", "")
    fact_checking = state.get("fact_checking_results", "")
    visualizations = state.get("visualizations", "")
    citations = state.get("citations", [])
    
    llm = get_llm()
    
    prompt = f"""
    You are an expert researcher. Write a comprehensive, cohesive, and easily readable final research report on the topic: '{topic}'.
    Structure it with professional headers, bullet points where appropriate, and an executive summary.

    Synthesize the following analysis:
    {analysis}
    
    Address fact checking constraints:
    {fact_checking}
    
    Recommended Visualizations (describe them in text):
    {visualizations}

    Include a "Sources & Citations" section at the end utilizing: {citations}
    """
    try:
        response = llm.invoke([HumanMessage(content=prompt)], config={"tags": ["report_agent"]})
        return {"final_report": response.content}
    except Exception as e:
        return {"final_report": f"## Report Generation Failed\n\nError: {str(e)}"}
