from graph.state import ResearchState
from services.llm import get_reasoning_llm
from langchain_core.messages import SystemMessage, HumanMessage

SYSTEM_PROMPT = """You are a senior investigative research analyst known for producing brutally honest, critically-balanced reports. You NEVER write marketing copy or one-sided summaries.

Your reports are valued because they:
- Present BOTH sides of every issue (strengths AND weaknesses)
- Identify what the evidence does NOT say (information gaps)
- Challenge assumptions rather than accept claims at face value
- Provide specific data points, not vague generalities
- Always include a "devil's advocate" perspective

You will be PENALIZED if your report:
- Only presents positive findings without criticism
- Says "None identified" for contradictions (there are ALWAYS uncertainties)
- Reads like promotional material
- Lacks specific numbers, comparisons, or data points
- Is shorter than 1000 words"""

def report_builder_agent(state: ResearchState) -> dict:
    """Produces a detailed, critically-balanced research report with inline citations."""
    query = state.get("query", "")
    insights = "\n".join([f"- {i}" for i in state.get("insights", [])])
    
    r = state.get("reflection")
    contradictions = "\n".join([f"- {c}" for c in (r.contradictions if r else [])]) or "_None identified by reflection agent_"
    
    # Include raw evidence
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
    
    user_prompt = f"""Write an investigative research report for: {query}

SYNTHESIZED INSIGHTS:
{insights}

RAW EVIDENCE:
{evidence_text}

CONTRADICTIONS FROM REFLECTION AGENT:
{contradictions}

SOURCES:
{sources_text}

REQUIRED STRUCTURE (you MUST include ALL of these sections — do NOT skip any):

## Executive Summary
State the bottom-line answer to the research question. Include the scope of evidence reviewed (how many sources, what types). 2-3 substantive paragraphs.

## Background & Context
Why this topic matters now. Current landscape, market size, trends, or historical context. Use specific data.

## Key Findings
5-8 findings, each with: **Bold title**, 2-4 sentences with specific evidence, inline citation (Source Name).

## Strengths & Advantages
What genuinely works well? What are the clear positives backed by evidence? Be specific with data.

## Weaknesses, Risks & Limitations
THIS SECTION IS MANDATORY AND MUST BE SUBSTANTIVE.
What are the downsides? What could go wrong? What is overpriced, overpromised, or unproven?
If the evidence doesn't mention weaknesses, explicitly state: "The available evidence does not address [X], which itself is a red flag because..."
Consider: cost concerns, time investment, opportunity cost, unverified claims, lack of independent reviews, market alternatives.

## Grey Areas & Uncertainties
THIS SECTION IS MANDATORY AND MUST BE SUBSTANTIVE.
What remains unclear? Where is the evidence insufficient to draw conclusions?
What assumptions are being made? What would a skeptic ask?
Consider: long-term outcomes, ROI data, comparison with free alternatives, survivorship bias, selection bias in testimonials.

## Comparative Analysis
How does this compare to alternatives? What are the trade-offs?
If no direct comparisons exist in the evidence, state what the ideal comparison would be and why it's missing.

## Contradictions & Open Questions
Even if the reflection agent found none, YOU must identify at least 2-3 open questions that a thorough researcher would want answered. Nothing is ever 100% contradiction-free.

## Recommendations & Next Steps
Actionable recommendations with caveats. What specific follow-up research should the reader do? What questions should they ask before acting?

## Sources
Full list of all cited sources.

ABSOLUTE REQUIREMENTS:
1. Minimum 1000 words
2. Weaknesses section MUST be at least 3 bullet points (find them or infer them)
3. Grey Areas section MUST be at least 3 bullet points
4. Every section must have content — no empty sections
5. Use markdown: **bold**, bullet points, numbered lists
6. Cite sources inline: (Tavily: url) or (Wikipedia) or (ArXiv)
7. If evidence is one-sided, explicitly flag it: "⚠️ Note: Available evidence is predominantly positive, suggesting potential selection bias"
"""
    
    reasoning_llm = get_reasoning_llm()
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_prompt)
    ]
    response = reasoning_llm.invoke(messages, config={"tags": ["Report Builder"]})
    print("--- ReportBuilder: Report ready ---")
    return {"final_report": response.content}
