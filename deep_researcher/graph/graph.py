from langgraph.graph import StateGraph, START, END
from graph.state import ResearchState
from agents.planner_agent import planner_agent
from agents.retriever_agent import retriever_agent
from agents.source_reliability_agent import source_reliability_agent
from agents.multimodal_agent import multimodal_agent
from agents.analyzer_agent import analyzer_agent
from agents.fact_checker_agent import fact_checker_agent
from agents.reflection_agent import reflection_agent
from agents.visualization_agent import visualization_agent
from agents.report_builder_agent import report_builder_agent

def should_continue(state: ResearchState):
    """Conditional edge from reflection to either visualization or retriever."""
    if state.get("is_satisfied", False):
        return "visualization"
    else:
        return "retriever"

def create_research_graph():
    # Initialize the graph
    graph = StateGraph(ResearchState)
    
    # Add our nodes
    graph.add_node("planner", planner_agent)
    graph.add_node("retriever", retriever_agent)
    graph.add_node("source_reliability", source_reliability_agent)
    graph.add_node("multimodal", multimodal_agent)
    graph.add_node("analyzer", analyzer_agent)
    graph.add_node("fact_checker", fact_checker_agent)
    graph.add_node("reflection", reflection_agent)
    graph.add_node("visualization", visualization_agent)
    graph.add_node("report_builder", report_builder_agent)
    
    # Define the sequential workflow with loop
    graph.add_edge(START, "planner")
    graph.add_edge("planner", "retriever")
    graph.add_edge("retriever", "source_reliability")
    graph.add_edge("source_reliability", "multimodal")
    graph.add_edge("multimodal", "analyzer")
    graph.add_edge("analyzer", "fact_checker")
    graph.add_edge("fact_checker", "reflection")
    
    # Add conditional edge from reflection
    graph.add_conditional_edges(
        "reflection",
        should_continue,
        {
            "visualization": "visualization",
            "retriever": "retriever"
        }
    )
    
    graph.add_edge("visualization", "report_builder")
    graph.add_edge("report_builder", END)
    
    # Compile the graph into runnable workflow
    return graph.compile()

