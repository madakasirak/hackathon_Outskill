from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from typing import Literal

from graph.state import ResearchState, MAX_ITERATIONS
from agents.retriever_agent import retriever_agent
from agents.analyzer_agent import analyzer_agent
from agents.reflection_agent import reflection_agent
from agents.report_agent import report_builder_agent

def should_continue(state: ResearchState) -> Literal["retriever", "report_builder"]:
    r = state.get('reflection')
    if r and r.needs_more_research and state['iteration'] < MAX_ITERATIONS:
        return "retriever"
    return "report_builder"

import sqlite3

def build_app():
    conn = sqlite3.connect("./checkpoints.db", check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    
    workflow = StateGraph(ResearchState)
    
    workflow.add_node("retriever", retriever_agent)
    workflow.add_node("analyzer", analyzer_agent)
    workflow.add_node("reflection", reflection_agent)
    workflow.add_node("report_builder", report_builder_agent)
    
    workflow.add_edge(START, "retriever")
    workflow.add_edge("retriever", "analyzer")
    workflow.add_edge("analyzer", "reflection")
    
    workflow.add_conditional_edges(
        "reflection",
        should_continue,
        {
            "retriever": "retriever",
            "report_builder": "report_builder"
        }
    )
    
    workflow.add_edge("report_builder", END)
    
    app = workflow.compile(checkpointer=checkpointer)
    return app
