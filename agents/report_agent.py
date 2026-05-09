from graph.state import ResearchState
from services.llm import get_reasoning_llm

def report_builder_agent(state: ResearchState) -> dict:
    """Takes the synthesized insights and formats them into a final markdown report."""
    query = state.get("query", "")
    insights = "\n".join([f"- {i}" for i in state.get("insights", [])])
    
    r = state.get("reflection")
    contradictions = "\n".join([f"- {c}" for c in (r.contradictions if r else [])]) or "_None identified_"
    
    print("--- ReportBuilder: Formatting final report ---")
    
    prompt = f"""Write a comprehensive research report on: {query}
    
KEY INSIGHTS:
{insights}

CONTRADICTIONS FOUND:
{contradictions}

Format as markdown with: 
## Executive Summary
## Key Findings
## Contradictions & Open Questions
## Recommendations

Be concise but rigorous."""
    
    reasoning_llm = get_reasoning_llm()
    response = reasoning_llm.invoke(prompt, config={"tags": ["Report Builder"]})
    print("--- ReportBuilder: Report ready ---")
    return {"final_report": response.content}
