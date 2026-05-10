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
from services.llm import get_fast_llm
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
    
    decision_prompt = f"""You are a research planning agent. Your job is to select the RIGHT tools for the query — not all of them.

QUERY: '{search_query}'

AVAILABLE TOOLS:
- LOCAL RAG: Search user's uploaded documents (PDF/TXT files indexed in FAISS)
- ARXIV: Search academic research papers (math, physics, CS, biology, economics papers)
- WIKIPEDIA: Look up established facts, definitions, history, general knowledge
- DUCKDUCKGO: General web search for news, companies, products, current events
- TAVILY: AI-powered deep web search — best for recent data, specific facts, industry analysis

ROUTING RULES (follow strictly):

1. DOCUMENT QUERIES (user asks about their uploaded files, "my document", "the PDF", course material):
   → LOCAL RAG, TAVILY
   Example: "Summarize the uploaded report" → LOCAL RAG
   Example: "What does the document say about X?" → LOCAL RAG, TAVILY

2. ACADEMIC/SCIENTIFIC RESEARCH (research papers, algorithms, scientific methods, theoretical topics):
   → ARXIV, WIKIPEDIA, TAVILY
   Example: "Transformer attention mechanisms" → ARXIV, WIKIPEDIA, TAVILY
   Example: "Latest research on GLP-1 drugs" → ARXIV, TAVILY

3. GENERAL KNOWLEDGE (history, definitions, well-known topics, "what is X"):
   → WIKIPEDIA, TAVILY
   Example: "What is nuclear energy?" → WIKIPEDIA, TAVILY

4. CURRENT EVENTS / INDUSTRY (companies, products, market trends, recent news):
   → TAVILY, DUCKDUCKGO
   Example: "Tesla stock performance 2024" → TAVILY, DUCKDUCKGO
   Example: "Outskill GenAI program review" → TAVILY, DUCKDUCKGO

5. COMPREHENSIVE RESEARCH (broad multi-faceted questions, pros vs cons, policy debates):
   → TAVILY, WIKIPEDIA, DUCKDUCKGO, ARXIV
   Example: "Is nuclear energy net-positive for climate goals?" → TAVILY, WIKIPEDIA, DUCKDUCKGO, ARXIV

IMPORTANT: If the user uploaded documents AND is asking a question, ALWAYS include LOCAL RAG alongside other tools.
Select 2-4 tools maximum. Do NOT select all tools unless the query is truly comprehensive.

Return ONLY a comma-separated list of tool names, nothing else.
Example output: TAVILY, WIKIPEDIA"""
    
    fast_llm = get_fast_llm()
    try:
        decision = fast_llm.invoke([HumanMessage(content=decision_prompt)], config={"tags": ["Retriever"]})
        selected_tools = [t.strip().upper() for t in decision.content.split(",")]
    except Exception as e:
        print(f"Tool selection failed, defaulting to all: {e}")
        selected_tools = ["TAVILY", "WIKIPEDIA", "DUCKDUCKGO", "ARXIV"]

    print(f"Selected Tools: {selected_tools}")
    
    # Auto-include LOCAL RAG if user has uploaded documents (FAISS index exists)
    if load_user_faiss() is not None and "LOCAL RAG" not in selected_tools:
        selected_tools.append("LOCAL RAG")
        print("Auto-added LOCAL RAG (user documents detected)")
    
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
