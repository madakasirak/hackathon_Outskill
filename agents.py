import os
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from typing import Dict, Any

def get_llm():
    return ChatOpenAI(
        model=os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o-mini"),
        api_key=os.environ.get("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1"
    )

def retriever_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """Simulates retrieving information from the web/documents."""
    topic = state.get("topic")
    # In a real app, this would use a Web Search API like Tavily or Exa.
    # For a minimal POC without external dependencies, we use the LLM to generate simulated search context.
    llm = get_llm()
    prompt = f"Act as a retriever. Recall and provide 5 factual, distinct bullet points of raw context about the topic: '{topic}'."
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        raw_data = [line.strip() for line in response.content.split("\n") if line.strip()]
        return {"raw_data": raw_data, "citations": ["Simulated Domain Knowledge Retrieval"]}
    except Exception as e:
        return {"errors": [str(e)]}

def analysis_agent(state: Dict[str, Any]) -> Dict[str, Any]:
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

def insight_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """Extracts deep insights based on the analysis."""
    if "errors" in state and state["errors"]:
        return state
        
    analysis = state.get("analysis", "")
    topic = state.get("topic")
    llm = get_llm()
    
    prompt = f"Based on the following analysis of '{topic}', what are the 3 most profound, forward-looking insights or non-obvious conclusions?\n\n{analysis}"
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        return {"insights": response.content}
    except Exception as e:
        return {"errors": [state.get("errors", []) + [str(e)]]}

def report_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """Drafts the final markdown report."""
    if "errors" in state and state["errors"]:
        error_msg = "\n".join(state["errors"])
        return {"final_report": f"## Research Failed\n\nErrors encountered:\n{error_msg}"}
        
    topic = state.get("topic")
    analysis = state.get("analysis", "")
    insights = state.get("insights", "")
    citations = state.get("citations", [])
    
    llm = get_llm()
    
    prompt = f"""
You are an expert researcher. Write a comprehensive, cohesive, and easily readable final research report on the topic: '{topic}'.
Structure it with professional headers, bullet points where appropriate, and an executive summary.

Synthesize the following analysis:
{analysis}

Feature these key insights prominently:
{insights}

Include a "Sources & Citations" section at the end utilizing: {citations}
"""
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        return {"final_report": response.content}
    except Exception as e:
        return {"final_report": f"## Report Generation Failed\n\nError: {str(e)}"}
