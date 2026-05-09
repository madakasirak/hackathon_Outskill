import os
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from graph.state import ResearchState
from services.llm import fast_llm, OPENROUTER_BASE, OPENROUTER_HEADERS

embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")

def _get_council_llm(model_name: str, temperature: float = 0.5):
    return ChatOpenAI(
        model=model_name,
        openai_api_key=os.environ.get("OPENROUTER_API_KEY", "dummy"),
        openai_api_base=OPENROUTER_BASE,
        temperature=temperature,
        default_headers=OPENROUTER_HEADERS,
    )

def analyzer_agent(state: ResearchState) -> dict:
    """Embeds retrieved docs into FAISS, does similarity search, and synthesizes insights using a Model Council."""
    docs = state.get("documents", [])
    query = state.get("query", "")
    
    if not docs:
        return {"insights": ["No documents found to analyze."]}
        
    print(f"--- Analyzer: Embedding {len(docs)} documents into FAISS ---")
    
    # Convert dicts back to Document objects for FAISS
    lc_docs = [Document(page_content=d["page_content"], metadata=d.get("metadata", {})) for d in docs if d.get("page_content")]
    
    if not lc_docs:
        return {"insights": ["No valid text content found in documents."]}
        
    # RAG: Create FAISS index and perform similarity search against the query
    vector_store = FAISS.from_documents(lc_docs, embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    
    relevant_docs = retriever.invoke(query)
    evidence = "\n---\n".join([d.page_content[:600] for d in relevant_docs])
    
    prompt = f"""You are a research analyst. Extract the most profound, non-obvious insights to answer the user's query based ONLY on the evidence below.
    
QUERY: {query}

EVIDENCE FROM RAG:
{evidence}
"""
    print(f"--- Analyzer: Consulting Model Council for '{query}' ---")
    
    try:
        # Council Member 1: Default fast model (GPT-4o-mini)
        member1_resp = fast_llm.invoke([HumanMessage(content=prompt)])
        
        # Council Member 2: Alternative model (e.g., Claude Haiku or higher temp GPT)
        member2_llm = _get_council_llm("anthropic/claude-3-haiku")
        member2_resp = member2_llm.invoke([HumanMessage(content=prompt)])
        
        # Synthesis by Council President
        synthesis_prompt = f"""
        We asked two different AI models for their top insights on the query '{query}'.
        
        Model 1 Insights:
        {member1_resp.content}
        
        Model 2 Insights:
        {member2_resp.content}
        
        Your task as the Council President is to compare their insights based on the same evidence. 
        Provide a consolidated, highly refined list of 3-5 distinct insights.
        Highlight consensus and any unique, profound perspectives.
        Return as a clear numbered list.
        """
        
        synthesis_resp = fast_llm.invoke([HumanMessage(content=synthesis_prompt)])
        insights = [i.strip() for i in synthesis_resp.content.split('\n') if i.strip()]
    except Exception as e:
        print(f"Model Council failed: {e}. Falling back to single model.")
        response = fast_llm.invoke([HumanMessage(content=prompt)])
        insights = [i.strip() for i in response.content.split('\n') if i.strip()]

    print(f"--- Analyzer: Council synthesized {len(insights)} insights ---")
    
    # Append to existing insights if we are in a reflection loop
    iteration = state.get("iteration", 0)
    if iteration > 0:
        insights = ["--- New Insights from Reflection Loop ---"] + insights
    return {"insights": insights}
