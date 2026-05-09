from graph.state import ResearchState, ReflectionResult, MAX_ITERATIONS
from services.llm import reasoning_llm

def reflection_agent(state: ResearchState) -> dict:
    """Reviews the generated insights. If there are gaps, triggers a self-correction loop back to Retriever."""
    query = state.get("query", "")
    insights = "\n".join([f"- {i}" for i in state.get("insights", [])])
    iteration = state.get("iteration", 0)
    
    print(f"--- Reflection (Iter {iteration}): Reviewing insights against original query ---")
    
    structured_llm = reasoning_llm.with_structured_output(ReflectionResult)
    
    prompt = f"""You are a critical reviewer evaluating a research report draft.
    
ORIGINAL QUERY: {query}

CURRENT INSIGHTS:
{insights}

Identify:
1. coverage_gaps — Is there a specific part of the original query that is not fully answered? Be specific.
2. contradictions — Are there conflicting claims?
3. needs_more_research — Set to True if we must loop back to gather more info because of major gaps. 

Iteration {iteration}/{MAX_ITERATIONS}. Set needs_more_research=True ONLY if concrete gaps remain."""
    
    try:
        r = structured_llm.invoke(prompt)
    except Exception as e:
        print(f"Reflection failed, defaulting to no more research: {e}")
        r = ReflectionResult(coverage_gaps=[], contradictions=[], needs_more_research=False)
    
    # Force stop if we hit max iterations
    # iteration is incremented to iteration+1 after this function returns.
    # If that would equal MAX_ITERATIONS, we've used all rounds — force-stop.
    if iteration + 1 >= MAX_ITERATIONS:
        r.needs_more_research = False
        
    print(f"--- Reflection: gaps={len(r.coverage_gaps)}, loop_triggered={r.needs_more_research} ---")
    
    return {"reflection": r, "iteration": iteration + 1}
