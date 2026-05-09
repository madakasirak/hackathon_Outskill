import streamlit as st
import os
import sys

# Ensure local imports work when run from outside the deep_researcher directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from graph import create_research_graph

st.set_page_config(page_title="Deep Researcher", page_icon="🔍", layout="wide")

st.title("🤖 Multi-Agent AI Deep Researcher")
st.markdown("A LangGraph-powered research system using OpenRouter.")

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input("OpenRouter API Key *", type="password")
    model = st.selectbox(
        "Model",
        ["openai/gpt-4o-mini", "anthropic/claude-3.5-sonnet"],
        index=0
    )
    
    st.header("🔌 External Tools (Optional)")
    tavily_key = st.text_input("Tavily API Key", type="password")
    serpapi_key = st.text_input("SerpAPI Key", type="password")
    st.caption("Arxiv is enabled by default.")
    
    st.markdown("---")
    st.markdown("""
    **Agents in this workflow:**
    1. **Retriever Agent:** Gathers context (Arxiv, Tavily, SerpAPI)
    2. **Analysis Agent:** Synthesizes & spots trends
    3. **Insight Agent:** Extracts profound conclusions
    4. **Report Agent:** Writes the structured markdown report
    """)
    
    if api_key:
        os.environ["OPENROUTER_API_KEY"] = api_key
        os.environ["OPENROUTER_MODEL"] = model
    if tavily_key:
        os.environ["TAVILY_API_KEY"] = tavily_key
    if serpapi_key:
        os.environ["SERPAPI_API_KEY"] = serpapi_key

# Main content
topic = st.text_input("Enter a research topic:", placeholder="e.g. Impact of AI agents on software engineering")

workflow = create_research_graph()

with st.expander("🗺️ View LangGraph Architecture", expanded=False):
    try:
        # Generate the graph visualization using Mermaid
        graph_png = workflow.get_graph().draw_mermaid_png()
        st.image(graph_png, caption="Multi-Agent Workflow Architecture", use_container_width=True)
    except Exception as e:
        st.error(f"Could not generate graph image: {e}")

if st.button("Start Deep Research", type="primary"):
    if not os.environ.get("OPENROUTER_API_KEY"):
        st.error("Please enter your OpenRouter API Key in the sidebar.")
    elif not topic:
        st.warning("Please enter a research topic to proceed.")
    else:
        with st.spinner("Initializing Deep Research Agents..."):
            workflow = create_research_graph()
            
            st.info(f"Initiating research protocol for: **{topic}**")
            progress_container = st.container()
            
            initial_state = {"topic": topic, "errors": []}
            
            try:
                with progress_container:
                    stages = ["Retriever", "Analysis", "Insight", "Report"]
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    final_state = {}
                    step = 0
                    
                    # Stream state updates straight from the LangGraph workflow
                    for output in workflow.stream(initial_state):
                        for node_name, state_update in output.items():
                            step += 1
                            progress = min(step / len(stages), 1.0)
                            progress_bar.progress(progress)
                            status_text.markdown(f"🟢 **Agent Active:** `{node_name.upper()}`")
                            
                            # Merge state updates dictionary
                            if isinstance(state_update, dict):
                                final_state.update(state_update)
                
                st.success("Research Protocol Complete!")
                
                # Render results
                st.markdown("---")
                st.markdown("## 📊 Final Research Report")
                st.markdown(final_state.get("final_report", "Report generation failed. Please check logs."))
                
                with st.expander("🔍 View Agent Internal Thoughts & Intermediary Steps"):
                    st.markdown("### 1. Retriever Agent Data")
                    raw_data = final_state.get("raw_data", [])
                    if isinstance(raw_data, list):
                        for i, data_chunk in enumerate(raw_data):
                            st.text_area(f"Raw Data Context {i+1}", value=data_chunk, height=150)
                    else:
                        st.write(raw_data)
                    
                    st.markdown(f"**Citations:** {', '.join(final_state.get('citations', []))}")
                    
                    st.markdown("### 2. Analysis Agent Thoughts")
                    st.write(final_state.get("analysis", ""))
                    
                    st.markdown("### 3. Insight Agent Thoughts")
                    st.write(final_state.get("insights", ""))
                    
                    errors = final_state.get("errors", [])
                    if errors:
                        st.error("Errors during execution:")
                        for err in errors:
                            st.write(err)
                    
            except Exception as e:
                st.error(f"An unexpected error occurred during execution: {str(e)}")
