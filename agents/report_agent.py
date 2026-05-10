from graph.state import ResearchState
from services.llm import get_reasoning_llm

def report_builder_agent(state: ResearchState) -> dict:
    """Takes the synthesized insights and formats them into a final markdown report with inline source citations."""
    query = state.get("query", "")
    insights = "\n".join([f"- {i}" for i in state.get("insights", [])])
    
    r = state.get("reflection")
    contradictions = "\n".join([f"- {c}" for c in (r.contradictions if r else [])]) or "_None identified_"
    
    # Build a sources reference list from retrieved documents
    docs = state.get("documents", [])
    sources_list = []
    seen = set()
    for d in docs:
        meta = d.get("metadata", {})
        source = meta.get("source", "unknown")
        url = meta.get("url", "")
        label = f"{source}: {url}" if url else source
        if label not in seen:
            seen.add(label)
            sources_list.append(label)
    
    sources_text = "\n".join([f"[{i+1}] {s}" for i, s in enumerate(sources_list[:15])])
    
    print("--- ReportBuilder: Formatting final report with citations ---")
    
    prompt = f"""Write a comprehensive research report on: {query}
    
KEY INSIGHTS:
{insights}

CONTRADICTIONS FOUND:
{contradictions}

AVAILABLE SOURCES:
{sources_text}

Format as markdown with: 
## Executive Summary
## Key Findings (cite sources inline using [Source Name] notation)
## Contradictions & Open Questions
## Recommendations
## Sources

IMPORTANT: For each key finding, add an inline citation referencing which source it came from, e.g. (Tavily: url) or (ArXiv) or (Wikipedia). 
At the end, include a ## Sources section listing all sources used.

Be concise but rigorous."""
    
    reasoning_llm = get_reasoning_llm()
    response = reasoning_llm.invoke(prompt, config={"tags": ["Report Builder"]})
    print("--- ReportBuilder: Report ready ---")
    return {"final_report": response.content}
