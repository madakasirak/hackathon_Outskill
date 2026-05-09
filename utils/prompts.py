ANALYSIS_PROMPT = """
You are a critical research analyst. Given the search results below, produce a concise, evidence-based summary.
- Only use the provided excerpts.
- Highlight points of agreement and contradictions.
- Call out uncertainty and possible gaps in evidence.

Context:
{context}

Respond in markdown, 3-6 short paragraphs.
"""

INSIGHT_PROMPT = """
You are a strategic research synthesizer. Based on the analysis below, produce 6 concise insights, each 1-2 sentences.
- Identify trends, implications, and recommendations for stakeholders.
- Be evidence-driven and avoid hallucinations.

Analysis:
{analysis}

Respond as a bullet list in markdown.
"""

REPORT_PROMPT = """
You are an expert technical writer. Produce a polished research report in markdown with the following sections:
# Title
## Executive Summary
## Key Findings
## Contradictions & Risks
## Trends & Insights
## Conclusion
## Sources

Use the inputs below (query, analysis, insights, sources). Keep the report ~600-1200 words and cite sources by URL in the Sources section.

Query: {query}
Analysis: {analysis}
Insights: {insights}
Sources:
{sources}

Respond in markdown.
"""
