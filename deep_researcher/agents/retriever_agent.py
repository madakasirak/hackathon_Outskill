import os
import concurrent.futures
from typing import Dict, Any
from langchain_core.messages import HumanMessage
from utils.llm_utils import get_llm

def retriever_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """Retrieves information from the web/documents using APIs."""
    topic = state.get("topic")
    plan = state.get("plan", "")
    reflection = state.get("reflection_feedback", "")
    raw_data = []
    citations = []
    errors = state.get("errors", [])
    vector_store_path = state.get("local_vector_store")
    
    # Decide which tools to run using an LLM
    decision_prompt = f"""
    You are a research planning agent. Given the user's research topic: '{topic}', and plan: '{plan}', decide which of the following information retrieval tools should be used.
    If you received previous feedback: '{reflection}', adjust your strategy accordingly.
    
    Available tools:
    - LOCAL RAG: Use if the topic implies referring to user's uploaded documents, local context, or specific provided data.
    - ARXIV: Use ONLY for highly academic, scientific, mathematical, or physics research papers.
    - WIKIPEDIA: Use for general knowledge, historical events, entities, and broader facts.
    - DUCKDUCKGO: Use for current events, web searches, news, recent updates, or specific company/course information.
    - TAVILY: Best for comprehensive AI web research and specific company/course information (requires API key).
    - SERPAPI: Use for Google search results (requires API key).
    
    Return a comma-separated list of the EXACT tool names you want to use (from the capitalized names above). 
    You MUST return ONLY the comma-separated tool names, without any other text, quotes, or markdown.
    Example: DUCKDUCKGO, WIKIPEDIA, LOCAL RAG
    """
    
    try:
        decision_llm = get_llm()
        decision_resp = decision_llm.invoke([HumanMessage(content=decision_prompt)])
        selected_tools = [t.strip().upper() for t in decision_resp.content.split(",")]
    except Exception as e:
        # Fallback to run all if decision parsing fails
        selected_tools = ["LOCAL RAG", "ARXIV", "WIKIPEDIA", "DUCKDUCKGO", "TAVILY", "SERPAPI"]

    def run_local_rag():
        if "LOCAL RAG" in selected_tools and vector_store_path and os.path.exists(vector_store_path):
            try:
                from langchain_community.vectorstores import FAISS
                from langchain_huggingface import HuggingFaceEmbeddings
                embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                vectorstore = FAISS.load_local(vector_store_path, embeddings, allow_dangerous_deserialization=True)
                docs = vectorstore.similarity_search(topic, k=8)
                if docs:
                    rag_content = "\n\n".join([d.page_content for d in docs])
                    return (f"--- Local Documents Context ---\n{rag_content}", "Local Uploaded Documents", None)
            except Exception as e:
                return (None, None, f"Local RAG Error: {str(e)}")
        return (None, None, None)

    def run_arxiv():
        if "ARXIV" in selected_tools:
            try:
                from langchain_community.utilities import ArxivAPIWrapper
                arxiv = ArxivAPIWrapper(top_k_results=3, load_max_docs=3)
                arxiv_results = arxiv.run(topic)
                if arxiv_results and "No good Arxiv Result was found" not in arxiv_results:
                    return (f"--- Arxiv Results ---\n{arxiv_results}", "Arxiv", None)
            except Exception as e:
                pass # occasionally fails, ignore
        return (None, None, None)

    def run_tavily():
        if "TAVILY" in selected_tools and os.environ.get("TAVILY_API_KEY"):
            try:
                try:
                    from langchain_tavily import TavilySearch as TavilySearchResults
                except ImportError:
                    try:
                        from langchain_tavily import TavilySearchResults
                    except ImportError:
                        from langchain_community.tools.tavily_search import TavilySearchResults
                    
                tavily = TavilySearchResults(max_results=3)
                tavily_results = tavily.invoke({"query": topic})
                if tavily_results:
                    return (f"--- Tavily Results ---\n{str(tavily_results)}", "Tavily", None)
            except Exception as e:
                return (None, None, f"Tavily Error: {str(e)}")
        return (None, None, None)
            
    def run_serpapi():
        if "SERPAPI" in selected_tools and os.environ.get("SERPAPI_API_KEY"):
            try:
                from langchain_community.utilities import SerpAPIWrapper
                serpapi = SerpAPIWrapper()
                serp_results = serpapi.run(topic)
                if serp_results:
                    return (f"--- SerpAPI Results ---\n{serp_results}", "SerpAPI", None)
            except Exception as e:
                return (None, None, f"SerpAPI Error: {str(e)}")
        return (None, None, None)

    def run_wikipedia():
        if "WIKIPEDIA" in selected_tools:
            try:
                from langchain_community.utilities import WikipediaAPIWrapper
                wikipedia = WikipediaAPIWrapper(top_k_results=3, doc_content_chars_max=1000)
                wiki_results = wikipedia.run(topic)
                if wiki_results and "No good Wikipedia Result was found" not in wiki_results:
                    return (f"--- Wikipedia Results ---\n{wiki_results}", "Wikipedia", None)
            except Exception as e:
                return (None, None, f"Wikipedia Error: {str(e)}")
        return (None, None, None)

    def run_duckduckgo():
        if "DUCKDUCKGO" in selected_tools:
            try:
                from langchain_community.tools import DuckDuckGoSearchRun
                ddg = DuckDuckGoSearchRun()
                ddg_results = ddg.run(topic)
                if ddg_results:
                    return (f"--- DuckDuckGo Results ---\n{ddg_results}", "DuckDuckGo", None)
            except Exception as e:
                return (None, None, f"DuckDuckGo Error: {str(e)}")
        return (None, None, None)

    # Run tools in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        futures = [
            executor.submit(run_local_rag),
            executor.submit(run_arxiv),
            executor.submit(run_tavily),
            executor.submit(run_serpapi),
            executor.submit(run_wikipedia),
            executor.submit(run_duckduckgo),
        ]
        
        for future in concurrent.futures.as_completed(futures):
            res_data, res_citation, res_error = future.result()
            if res_data:
                raw_data.append(res_data)
            if res_citation:
                citations.append(res_citation)
            if res_error:
                errors.append(res_error)

    # Add info about logic
    citations.append(f"Tools Chosen by Agent: {', '.join(selected_tools)}")

    # Fallback to LLM if no external results
    if not raw_data:
        try:
            llm = get_llm()
            prompt = f"Act as a retriever. Recall and provide 5 factual, distinct bullet points of raw context about the topic: '{topic}'."
            response = llm.invoke([HumanMessage(content=prompt)])
            raw_data.append(f"--- LLM Internal Knowledge ---\n{response.content}")
            citations.append("LLM Internal Knowledge")
        except Exception as e:
            errors.append(f"LLM Retrieval Error: {str(e)}")

    return {"raw_data": raw_data, "citations": citations, "errors": errors}
