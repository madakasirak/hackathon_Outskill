from graph.state import ResearchState
from services.llm import get_reasoning_llm

def report_builder_agent(state: ResearchState) -> dict:
    """Takes the synthesized insights, raw evidence, and source metadata to produce a detailed, cited research report."""
    query = state.get("query", "")
    insights = "\n".join([f"- {i}" for i in state.get("insights", [])])
    
    r = state.get("reflection")
    contradictions = "\n".join([f"- {c}" for c in (r.contradictions if r else [])]) or "_None identified_"
    
    # Include raw evidence snippets so the report can be detailed
    docs = state.get("documents", [])
    evidence_chunks = []
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
        # Include evidence text (truncated per chunk to stay within context limits)
        content = d.get("page_content", "")[:800]
        if content.strip():
            evidence_chunks.append(f"[{source}] {content}")
    
    # Limit to top 20 evidence chunks to avoid token overflow
    evidence_text = "\n---\n".join(evidence_chunks[:20])
    sources_text = "\n".join([f"[{i+1}] {s}" for i, s in enumerate(sources_list[:15])])
    
    print(f"--- ReportBuilder: Formatting detailed report with {len(evidence_chunks)} evidence chunks ---")
    
    prompt = f"""You are a senior research analyst producing a comprehensive, publication-quality research report.

RESEARCH QUESTION: {query}

SYNTHESIZED INSIGHTS FROM MODEL COUNCIL:
{insights}

RAW EVIDENCE FROM MULTIPLE SOURCES:
{evidence_text}

CONTRADICTIONS IDENTIFIED:
{contradictions}

AVAILABLE SOURCES:
{sources_text}

Write a DETAILED, in-depth research report using the following structure. Each section should be thorough — aim for 3-5 paragraphs per major section. Use specific data points, quotes, and examples from the evidence.

## Executive Summary
Provide a comprehensive overview of the research question, methodology, and key conclusions. (2-3 paragraphs)

## Background & Context
Set the stage — why does this topic matter? What is the current landscape? Reference relevant trends and data.

## Key Findings
Present 5-8 detailed findings, each with:
- A clear, bold heading
- 2-3 sentences of analysis with specific evidence
- Inline source citations using (Source Name) notation, e.g. (Tavily: url) or (ArXiv) or (Wikipedia)

## Analysis & Discussion
Synthesize the findings into broader themes. What patterns emerge? What are the implications? Compare different perspectives from the evidence.

## Contradictions & Open Questions
Discuss conflicting claims found across sources. What remains unresolved? Why do these disagreements exist?

## Recommendations
Provide actionable, evidence-based recommendations. Be specific and practical.

## Sources
List all sources referenced in the report with their full citations.

IMPORTANT GUIDELINES:
- Be DETAILED and thorough — this is a professional research report, not a summary
- Use specific numbers, quotes, and data points from the evidence
- Every major claim should have an inline citation
- Use markdown formatting: bold for key terms, bullet points, numbered lists
- Minimum 800 words
"""
    
    reasoning_llm = get_reasoning_llm()
    response = reasoning_llm.invoke(prompt, config={"tags": ["Report Builder"]})
    print("--- ReportBuilder: Report ready ---")
    return {"final_report": response.content}
