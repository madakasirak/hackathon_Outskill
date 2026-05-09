import streamlit as st
import os
from dotenv import load_dotenv
load_dotenv()

from graph.workflow import build_app

st.set_page_config(page_title="Deep Researcher", page_icon="🔬", layout="wide")

st.title("🔬 Multi-Agent AI Deep Researcher")
st.markdown("A 4-agent LangGraph system with RAG (FAISS), Conditional Routing, and Self-Correction.")

# Load the compiled graph
@st.cache_resource
def load_app():
    return build_app()

st.subheader("Research Query")
query = st.text_area(
    "What do you want to research?",
    value="How are GLP-1 drugs reshaping food and beverage company strategy?",
    height=80,
)

if st.button("🔍 Research", type="primary"):
    app = load_app()
    
    # Checkpointer thread for persistent memory
    thread_id = "streamlit_session_1"
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 30}
    
    initial_state = {
        "query": query,
        "documents": [],
        "insights": [],
        "iteration": 0,
        "final_report": ""
    }

    log_container = st.empty()
    logs = ""
    
    with st.spinner("Agents researching..."):
        # We run the graph and stream updates
        for event in app.stream(initial_state, config=config):
            for node, state in event.items():
                logs += f"**[{node}]** Finished processing.\n"
                
                # If reflection node just ran, show the self-correction logic
                if node == "reflection":
                    r = state.get("reflection")
                    if r:
                        logs += f" > Gaps found: {len(r.coverage_gaps)}. Looping back to Retriever: {r.needs_more_research}\n\n"
                else:
                    logs += "\n"
                    
                log_container.markdown(logs)
                
        # Final output
        final_state = app.get_state(config).values
        st.success("Research Complete!")
        
        st.markdown("---")
        st.markdown(final_state.get("final_report", "No report generated."))
