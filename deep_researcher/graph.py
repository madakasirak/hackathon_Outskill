from langgraph.graph import StateGraph, START, END
from deep_researcher.state import ResearchState
from deep_researcher.agents import retriever_agent, analysis_agent, insight_agent, report_agent

def create_research_graph():
    # Initialize the graph
    graph = StateGraph(ResearchState)
    
    # Add our nodes (the specialized agents)
    graph.add_node("retriever", retriever_agent)
    graph.add_node("analysis", analysis_agent)
    graph.add_node("insight", insight_agent)
    graph.add_node("report", report_agent)
    
    # Define the sequential workflow
    graph.add_edge(START, "retriever")
    graph.add_edge("retriever", "analysis")
    graph.add_edge("analysis", "insight")
    graph.add_edge("insight", "report")
    graph.add_edge("report", END)
    
    # Compile the graph into runnable workflow
    return graph.compile()
