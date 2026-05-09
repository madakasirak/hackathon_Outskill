import os
import concurrent.futures
from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import HumanMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import ArxivAPIWrapper
from langchain_huggingface import HuggingFaceEmbeddings
from services.rag import load_user_faiss

from graph.state import ResearchState
from services.llm import fast_llm
from tools.utilities import parse_pdf

# Initialize Tools Once
tavily_tool = TavilySearchResults(max_results=3)
wiki_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=1500))
ddg_tool = DuckDuckGoSearchRun()
arxiv_tool = ArxivAPIWrapper(top_k_results=3, load_max_docs=3)

def _run_faiss(query: str):
    vectorstore = load_user_faiss()
    if vectorstore is None:
        return []
    try:
        docs = vectorstore.similarity_search(query, k=5)
        return [{"page_content": d.page_content, "metadata": {"source": "local_rag"}} for d in docs]
    except Exception as e:
        print(f"FAISS query error: {e}")
        return []

def _run_tavily(query: str):
    try:
        results = tavily_tool.invoke(query)
        if isinstance(results, list):
            return [{"page_content": r.get('content', ''), "metadata": {"source": "tavily", "url": r.get('url', '')}} for r in results]
        return [{"page_content": str(results), "metadata": {"source": "tavily"}}]
    except Exception as e:
        print(f"Tavily error: {e}")
        return []

def _run_wikipedia(query: str):
    try:
        res = wiki_tool.invoke(query)
        return [{"page_content": res, "metadata": {"source": "wikipedia"}}]
    except Exception as e:
        print(f"Wiki error: {e}")
        return []

def _run_ddg(query: str):
    try:
        res = ddg_tool.invoke(query)
        return [{"page_content": res, "metadata": {"source": "duckduckgo"}}]
    except Exception as e:
        print(f"DDG error: {e}")
        return []

def _run_arxiv(query: str):
    try:
        res = arxiv_tool.run(query)
        if res and "No good Arxiv Result was found" not in res:
            return [{"page_content": res, "metadata": {"source": "arxiv"}}]
    except Exception as e:
        print(f"Arxiv error: {e}")
    return []

def retriever_agent(state: ResearchState) -> dict:
    """Fetches documents using multiple tools dynamically chosen and executed in parallel."""
    query = state.get("query", "")
    iteration = state.get("iteration", 0)
    
    # If looping back, target the specific gaps found by Reflection
    reflection = state.get("reflection")
    if iteration > 0 and reflection and reflection.coverage_gaps:
        search_query = reflection.coverage_gaps[0]
    else:
        search_query = query
        
    print(f"\n--- Retriever (Iter {iteration}): Analyzing query to select tools ---")
    
    decision_prompt = f"""
    You are a research planning agent. Topic: '{search_query}'.
    Decide which of the following tools should be used.
    Available tools:
    - LOCAL RAG: Always include if the user might be referring to uploaded documents.
    - ARXIV: For academic, scientific, or highly technical papers.
    - WIKIPEDIA: For general knowledge, history, and broad facts.
    - DUCKDUCKGO: For general web search.
    - TAVILY: For deep AI web research and specific up-to-date facts.
    - PDF: If a specific PDF file is requested (rare).
    
    Return a comma-separated list of the EXACT tool names you want to use.
    ONLY return the tool names, no markdown. Example: TAVILY, WIKIPEDIA
    """
    
    try:
        decision = fast_llm.invoke([HumanMessage(content=decision_prompt)])
        selected_tools = [t.strip().upper() for t in decision.content.split(",")]
    except Exception as e:
        print(f"Tool selection failed, defaulting to all: {e}")
        selected_tools = ["TAVILY", "WIKIPEDIA", "DUCKDUCKGO", "ARXIV"]

    print(f"Selected Tools: {selected_tools}")
    
    all_docs = []
    
    # Run tools in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = []
        if "LOCAL RAG" in selected_tools:
            futures.append(executor.submit(_run_faiss, search_query))
        if "TAVILY" in selected_tools:
            futures.append(executor.submit(_run_tavily, search_query))
        if "WIKIPEDIA" in selected_tools:
            futures.append(executor.submit(_run_wikipedia, search_query))
        if "DUCKDUCKGO" in selected_tools:
            futures.append(executor.submit(_run_ddg, search_query))
        if "ARXIV" in selected_tools:
            futures.append(executor.submit(_run_arxiv, search_query))
            
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            if res:
                all_docs.extend(res)
                
    print(f"--- Retrieved {len(all_docs)} document chunks ---")
    return {"documents": all_docs}
