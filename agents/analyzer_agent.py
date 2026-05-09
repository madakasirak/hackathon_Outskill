import os
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import FastEmbedEmbeddings
from graph.state import ResearchState
from services.llm import fast_llm

embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")

def analyzer_agent(state: ResearchState) -> dict:
    """Embeds retrieved docs into FAISS, does similarity search, and synthesizes insights."""
    docs = state.get("documents", [])
    query = state.get("query", "")
    
    if not docs:
        return {"insights": ["No documents found to analyze."]}
        
    print(f"--- Analyzer: Embedding {len(docs)} documents into FAISS ---")
    
    # Convert dicts back to Document objects for FAISS
    lc_docs = [Document(page_content=d["page_content"], metadata=d["metadata"]) for d in docs if d["page_content"]]
    
    if not lc_docs:
        return {"insights": ["No valid text content found in documents."]}
        
    # RAG: Create FAISS index and perform similarity search against the query
    vector_store = FAISS.from_documents(lc_docs, embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    
    relevant_docs = retriever.invoke(query)
    evidence = "\n---\n".join([d.page_content[:500] for d in relevant_docs])
    
    prompt = f"""You are a research analyst. Extract 3-5 key insights to answer the user's query based ONLY on the evidence below.
    
QUERY: {query}

EVIDENCE FROM RAG (FAISS):
{evidence}

Return as a clear numbered list, one insight per line."""
    
    response = fast_llm.invoke(prompt)
    insights = [i.strip() for i in response.content.split('\n') if i.strip()]
    
    print(f"--- Analyzer: Generated {len(insights)} insights ---")
    
    # Append to existing insights if we are in a reflection loop
    existing_insights = state.get("insights", [])
    if existing_insights:
        existing_insights.append("\n--- New Insights from Reflection Loop ---")
        
    return {"insights": existing_insights + insights}
