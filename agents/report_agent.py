from graph.state import ResearchState
from services.llm import get_reasoning_llm

def report_builder_agent(state: ResearchState) -> dict:
    """Takes the synthesized insights, raw evidence, and source metadata to produce a detailed, critically-balanced research report."""
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
        content = d.get("page_content", "")[:800]
        if content.strip():
            evidence_chunks.append(f"[{source}] {content}")
    
    evidence_text = "\n---\n".join(evidence_chunks[:20])
    sources_text = "\n".join([f"[{i+1}] {s}" for i, s in enumerate(sources_list[:15])])
    
    print(f"--- ReportBuilder: Formatting detailed report with {len(evidence_chunks)} evidence chunks ---")
    
    prompt = f"""You are a senior investigative research analyst. Your job is to produce a CRITICAL, BALANCED, and THOROUGH research report — not a summary or marketing copy.

RESEARCH QUESTION: {query}

SYNTHESIZED INSIGHTS FROM MODEL COUNCIL:
{insights}

RAW EVIDENCE FROM MULTIPLE SOURCES:
{evidence_text}

CONTRADICTIONS IDENTIFIED:
{contradictions}

AVAILABLE SOURCES:
{sources_text}

Write an in-depth, critically-balanced research report using the structure below. You MUST be analytical, not promotional. Challenge assumptions. Identify what is NOT said. Highlight risks, limitations, and unknowns.

---

## Executive Summary
A comprehensive overview of findings, methodology, and key conclusions. State the bottom-line answer to the research question upfront. (2-3 paragraphs)

## Background & Context
Why does this topic matter? What is the current landscape? Set the stage with relevant trends, market dynamics, or historical context.

## Key Findings
Present 5-8 detailed findings. For each:
- **Bold heading** summarizing the finding
- 2-4 sentences of analysis with specific data points, numbers, or quotes from the evidence
- Inline source citation: (Source Name) or (Tavily: url)

## Strengths & Advantages
What are the clear positives? What works well? Support each point with evidence.

## Weaknesses, Risks & Limitations
What are the downsides, risks, or gaps? What could go wrong? What is missing from the available information? Be brutally honest — do NOT whitewash.

## Grey Areas & Uncertainties
What remains unclear or debatable? Where do experts disagree? What claims lack sufficient evidence? What assumptions are being made?

## Comparative Analysis
If applicable, how does this compare to alternatives or competitors? What are the trade-offs?

## Contradictions & Open Questions
Discuss conflicting claims across sources. Why do these disagreements exist? What follow-up research is needed?

## Recommendations & Next Steps
Actionable, evidence-based recommendations. Include what specific follow-up actions or research the reader should pursue.

## Sources
List all sources referenced with full citations.

---

CRITICAL GUIDELINES:
- Be INVESTIGATIVE, not promotional. A good report challenges the subject, not just describes it.
- Include PROS AND CONS for every major topic
- Identify WHAT IS NOT SAID — gaps in available information are as important as what is found
- Use specific numbers, data points, quotes, and examples from the evidence
- Every major claim must have an inline citation
- Use rich markdown: bold, bullet points, numbered lists, tables where appropriate
- If the evidence is insufficient to draw a conclusion, SAY SO explicitly
- Minimum 1000 words — this is a professional research deliverable
"""
    
    reasoning_llm = get_reasoning_llm()
    response = reasoning_llm.invoke(prompt, config={"tags": ["Report Builder"]})
    print("--- ReportBuilder: Report ready ---")
    return {"final_report": response.content}
