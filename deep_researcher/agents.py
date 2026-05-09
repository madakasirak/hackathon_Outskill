import os
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from typing import Dict, Any

def get_llm():
    return ChatOpenAI(
        model=os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o-mini"),
        api_key=os.environ.get("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
        streaming=True
    )

def retriever_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """Retrieves information from the web/documents using APIs."""
    import concurrent.futures
    topic = state.get("topic")
    raw_data = []
    citations = []
    errors = state.get("errors", [])
    vector_store_path = state.get("local_vector_store")
    
    # Decide which tools to run using an LLM
    decision_prompt = f"""
    You are a research planning agent. Given the user's research topic: '{topic}', decide which of the following information retrieval tools should be used.
    
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

def get_council_llm(model_name: str):
    return ChatOpenAI(
        model=model_name,
        api_key=os.environ.get("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
        streaming=False
    )

def insight_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """Extracts deep insights using a Model Council approach."""
    if "errors" in state and state["errors"]:
        return state
        
    analysis = state.get("analysis", "")
    topic = state.get("topic")
    
    prompt = f"Based on the following analysis of '{topic}', what are the 3 most profound, forward-looking insights or non-obvious conclusions?\n\n{analysis}"
    try:
        # Prompt two different models for insights
        llm_1 = get_council_llm("openai/gpt-4o-mini")
        resp_1 = llm_1.invoke([HumanMessage(content=prompt)])
        
        llm_2 = get_council_llm("anthropic/claude-sonnet-4.5")
        resp_2 = llm_2.invoke([HumanMessage(content=prompt)])
        
        # Council synthesis using the primary configured LLM
        synthesis_llm = get_llm()
        synthesis_prompt = f"""
We asked two different AI models for their top insights on the topic '{topic}' based on the same analysis.

Model 1 (OpenAI GPT-4o-mini) insights:
{resp_1.content}

Model 2 (Anthropic Claude 4.5 Sonnet) insights:
{resp_2.content}

Your task as the Council President is to compare their insights. 
Provide a consolidated view highlighting:
1. Similar points (Consensus)
2. Dissimilar or distinct points (Divergence/Unique perspectives)
3. Final synthesized profound insights.
"""
        synthesis_resp = synthesis_llm.invoke([HumanMessage(content=synthesis_prompt)])
        return {"insights": synthesis_resp.content}
    except Exception as e:
        error_list = state.get("errors", [])
        error_list.append(f"Model Council Error: {str(e)}")
        return {"errors": error_list}

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
        response = llm.invoke([HumanMessage(content=prompt)], config={"tags": ["report_agent"]})
        return {"final_report": response.content}
    except Exception as e:
        return {"final_report": f"## Report Generation Failed\n\nError: {str(e)}"}
