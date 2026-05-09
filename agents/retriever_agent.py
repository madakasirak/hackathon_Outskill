import os
from dotenv import load_dotenv
load_dotenv()

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from graph.state import ResearchState

# Tools
tavily_tool = TavilySearchResults(max_results=3)
wiki_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=1500))

def retriever_agent(state: ResearchState) -> dict:
    """Fetches documents from Tavily and Wikipedia based on the query or gaps."""
    query = state.get("query", "")
    iteration = state.get("iteration", 0)
    
    # If looping back, target the specific gaps found by Reflection
    reflection = state.get("reflection")
    if iteration > 0 and reflection and reflection.coverage_gaps:
        search_query = reflection.coverage_gaps[0]
    else:
        search_query = query
        
    print(f"--- Retriever (Iter {iteration}): Searching for '{search_query}' ---")
    
    new_docs = []
    
    # 1. Tavily Search
    try:
        tavily_results = tavily_tool.invoke(search_query)
        for r in tavily_results:
            new_docs.append({
                "page_content": r.get('content', ''),
                "metadata": {"source": "tavily", "url": r.get('url', '')}
            })
    except Exception as e:
        print(f"Tavily search failed: {e}")
        
    # 2. Wikipedia Search
    try:
        wiki_res = wiki_tool.invoke(search_query)
        new_docs.append({
            "page_content": wiki_res,
            "metadata": {"source": "wikipedia", "url": "wikipedia"}
        })
    except Exception as e:
        print(f"Wikipedia search failed: {e}")
        
    return {"documents": new_docs}
