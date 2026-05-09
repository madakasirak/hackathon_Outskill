from typing import Callable

from graph.state import ResearchState
from services.llm import OpenRouterLLM

from agents.retriever_agent import run as retriever_run
from agents.analysis_agent import run as analysis_run
from agents.insight_agent import run as insight_run
from agents.report_agent import run as report_run


def run_workflow(state: ResearchState, llm: OpenRouterLLM, model: str, temperature: float = 0.2,
                 status_callback: Callable[[str, str], None] = None, pdf_path: str = None) -> ResearchState:
    """Run the linear workflow: retriever -> analysis -> insight -> report.

    status_callback(stage, message) is called when stages start/finish.
    """

    def cb(stage: str, msg: str = ""):
        if status_callback:
            status_callback(stage, msg)

    cb("Retriever", "starting FAISS RAG retrieval")
    retriever_run(state=state, pdf_path=pdf_path)
    cb("Retriever", f"found {len(state.retrieved_docs)} docs")

    cb("Analysis", "summarizing and evaluating evidence")
    analysis_run(state=state, llm=llm, model=model, temperature=temperature)
    cb("Analysis", "complete")

    cb("Insight", "deriving strategic insights")
    insight_run(state=state, llm=llm, model=model, temperature=temperature)
    cb("Insight", "complete")

    cb("Report", "generating final report")
    report_run(state=state, llm=llm, model=model, temperature=temperature)
    cb("Report", "complete")

    return state
