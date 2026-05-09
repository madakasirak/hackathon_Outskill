import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import time
import streamlit as st
from dotenv import load_dotenv
load_dotenv()

from graph.workflow import build_app
from services.db import log_stats, get_aggregate_stats, get_stats_history
from services.callbacks import TokenTrackingCallback

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Deep Researcher",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom Advanced CSS ────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Hide header decoration but keep sidebar toggle button visible */
    [data-testid="stHeader"] { background: transparent !important; border-bottom: none !important; }
    [data-testid="stDecoration"] { display: none !important; }
    #MainMenu { visibility: hidden !important; }
    footer { visibility: hidden !important; }

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    [data-testid="stAppViewContainer"] {
        background: radial-gradient(circle at top left, #1a1b26 0%, #0d0f17 100%);
    }
    [data-testid="stSidebar"] {
        background: rgba(22, 25, 43, 0.6) !important;
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    h1, h2, h3 { color: #f8fafc !important; font-weight: 600 !important; }
    p, li, span { color: #cbd5e1; }

    .app-logo-header {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 0px;
        padding-bottom: 20px;
    }
    .app-logo-header svg { width: 42px; height: 42px; fill: url(#grad1); }
    .app-logo-header h1 {
        margin: 0; font-size: 2.2rem;
        background: -webkit-linear-gradient(45deg, #4ade80, #3b82f6);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }

    .agent-widget {
        background: rgba(30, 34, 53, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px; padding: 14px 20px; margin: 8px 0;
        color: #f1f5f9; font-size: 0.95rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }
    
    @keyframes pulse-border {
        0%   { box-shadow: 0 0 0 0 rgba(56, 189, 248, 0.4); border-color: #38bdf8; }
        70%  { box-shadow: 0 0 0 8px rgba(56, 189, 248, 0); border-color: #0284c7; }
        100% { box-shadow: 0 0 0 0 rgba(56, 189, 248, 0); border-color: #38bdf8; }
    }
    .agent-widget.active { background: rgba(14, 34, 53, 0.8); animation: pulse-border 2s infinite; }
    .agent-widget.done { border-left: 4px solid #10b981; background: rgba(16, 185, 129, 0.05); }
    .agent-widget.error { border-left: 4px solid #ef4444; background: rgba(239, 68, 68, 0.05); }

    .metric-panel {
        background: rgba(255, 255, 255, 0.02); border-radius: 12px; padding: 16px 12px;
        text-align: center; color: #94a3b8; border: 1px solid rgba(255,255,255,0.05);
        backdrop-filter: blur(10px); box-shadow: inset 0 1px 0 rgba(255,255,255,0.05);
    }
    .metric-panel h3 { color: #e2e8f0 !important; font-size: 1.8rem !important; margin: 0 0 6px 0 !important; font-weight: 700 !important; }

    .stTextArea textarea {
        background: rgba(0,0,0,0.2) !important; border: 1px solid rgba(255,255,255,0.1) !important;
        border-radius: 12px !important; color: white !important; padding: 16px !important; font-size: 1rem !important;
    }
    .stTextArea textarea:focus { border-color: #3b82f6 !important; box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.3) !important; }
    
    .stButton>button { border-radius: 12px !important; font-weight: 600 !important; padding: 6px 24px !important; transition: transform 0.1s !important; }
    .stButton>button:active { transform: scale(0.98); }
</style>
""", unsafe_allow_html=True)

AGENT_INFO = {
    "retriever":         ("🔍", "Retriever Agent", "[gpt-4o-mini] Searching web & documents"),
    "analyzer":          ("🔬", "Analysis Agent", "[gpt-4o-mini + claude-haiku] Extracting insights"),
    "reflection":        ("🤔", "Reflection Agent", "[gpt-4o-mini] Checking for gaps"),
    "report_builder":    ("📝", "Report Builder", "[gpt-4o-mini] Drafting final report"),
}

def process_uploaded_files(files):
    from langchain_community.document_loaders import PyPDFLoader, TextLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    import tempfile
    
    docs = []
    for file in files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file.name.split('.')[-1]}") as f:
            f.write(file.getvalue())
            temp_path = f.name
            
        try:
            if file.name.lower().endswith(".pdf"):
                loader = PyPDFLoader(temp_path)
                docs.extend(loader.load())
            elif file.name.lower().endswith(".txt"):
                loader = TextLoader(temp_path, encoding="utf-8")
                docs.extend(loader.load())
        except Exception as e:
            st.error(f"Error loading {file.name}: {e}")
        finally:
            os.remove(temp_path)
            
    if not docs:
        return None
        
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = text_splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(splits, embeddings)
    index_path = "temp_faiss_index"
    vectorstore.save_local(index_path)
    return index_path

# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔑 Enter your Key here")
    
    st.markdown("<h5 style='color: #f8fafc; margin-top: 20px; margin-bottom: 5px; font-size: 14px;'>🤖 AI Provider</h5>", unsafe_allow_html=True)
    provider = st.selectbox("AI Provider", ["OpenRouter (All Models)", "Google Gemini", "OpenAI", "Anthropic"], index=0, label_visibility="collapsed")
    
    st.markdown("<h5 style='color: #f8fafc; margin-top: 15px; margin-bottom: 5px; font-size: 14px;'>⚙️ Model Selection</h5>", unsafe_allow_html=True)
    # All models routed through OpenRouter — the prefix tells OpenRouter which provider to use
    if "Gemini" in provider:
        models = ["google/gemini-2.0-flash-001", "google/gemini-1.5-pro"]
    elif "OpenAI" in provider:
        models = ["openai/gpt-4o", "openai/gpt-4o-mini"]
    elif "Anthropic" in provider:
        models = ["anthropic/claude-3.5-sonnet", "anthropic/claude-3-haiku"]
    else:
        # OpenRouter auto: show all popular models
        models = ["Auto-select (gpt-4o-mini + claude-3-haiku)", "openai/gpt-4o-mini", "openai/gpt-4o", "google/gemini-2.0-flash-001", "anthropic/claude-3.5-sonnet", "anthropic/claude-3-haiku"]
    selected_model = st.selectbox("Model Selection", models, label_visibility="collapsed")
    
    st.markdown("<h5 style='color: #f8fafc; margin-top: 25px; margin-bottom: 5px; font-size: 14px;'>🔑 Authentication</h5>", unsafe_allow_html=True)
    env_or_key = os.getenv("OPENROUTER_API_KEY", "")
    col_input, col_status = st.columns([3, 1])
    
    with col_input:
        api_key = st.text_input("Authentication", type="password", placeholder="OpenRouter API Key...", label_visibility="collapsed")
        if not api_key: api_key = env_or_key
    with col_status:
        if env_or_key or api_key:
            st.markdown("<div style='color: #4ade80; font-size: 12px; margin-top: 10px;'>✅ Loaded</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div style='color: #ef4444; font-size: 12px; margin-top: 10px;'>❌ Required</div>", unsafe_allow_html=True)
            
    st.markdown("<a href='https://openrouter.ai/keys' style='font-size: 12px; color: #3b82f6; text-decoration: none;'>🔗 Get API Key</a>", unsafe_allow_html=True)
    
    st.markdown("<h5 style='color: #f8fafc; margin-top: 25px; margin-bottom: 5px; font-size: 14px;'>🌐 Web Search API</h5>", unsafe_allow_html=True)
    env_tavily = os.getenv("TAVILY_API_KEY", "")
    col_tav_input, col_tav_status = st.columns([3, 1])
    
    with col_tav_input:
        tavily_key = st.text_input("Web Search API", type="password", placeholder="Tavily Key (Optional)...", label_visibility="collapsed")
        if not tavily_key: tavily_key = env_tavily
    with col_tav_status:
        if env_tavily or tavily_key:
            st.markdown("<div style='color: #4ade80; font-size: 12px; margin-top: 10px;'>✅ Loaded</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div style='color: #94a3b8; font-size: 12px; margin-top: 10px;'>Optional</div>", unsafe_allow_html=True)
    
    # Everything routes through OpenRouter
    if api_key: os.environ["OPENROUTER_API_KEY"] = api_key
    os.environ["LLM_MODEL"] = selected_model
    if tavily_key: os.environ["TAVILY_API_KEY"] = tavily_key

@st.cache_resource
def load_app():
    return build_app()

st.markdown("""
<div class="app-logo-header" style="flex-direction: column; align-items: flex-start; gap: 4px;">
    <div style="display: flex; align-items: center; gap: 12px;">
        <svg viewBox="0 0 24 24" style="width: 42px; height: 42px; fill: url(#grad1);">
            <defs><linearGradient id="grad1" x1="0%" y1="0%" x2="100%" y2="100%"><stop offset="0%" style="stop-color:#4ade80;stop-opacity:1" /><stop offset="100%" style="stop-color:#3b82f6;stop-opacity:1" /></linearGradient></defs>
            <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5" stroke="url(#grad1)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" fill="none"/>
        </svg>
        <h1 style="margin: 0; font-size: 2.5rem; background: -webkit-linear-gradient(45deg, #4ade80, #3b82f6); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">Multi-Agent Researcher</h1>
    </div>
    <div style="font-size: 0.9rem; color: #94a3b8; font-weight: 500; margin-left: 54px; margin-top: -6px;">by Group 12</div>
</div>
<div style="margin-top: 15px; margin-bottom: 25px; font-size: 1.1rem; color: #cbd5e1;">An elite, multi-agent AI framework for comprehensive, multi-source research investigations.</div>
""", unsafe_allow_html=True)

# ── Tabs ────────────────────────────────────────────────────────────────
tab_research, tab_stats = st.tabs(["🔍 Research Studio", "📊 Stats Dashboard"])

with tab_research:
    query = st.text_area("Research Query", placeholder="Enter your research Query or upload documents (PDF, DOCX, PPTX, TXT, MD)...", height=80, label_visibility="collapsed")
    col_upload, col_space, col_run = st.columns([1.5, 3, 1])
    
    with col_upload:
        with st.popover("📎 Attach Documents", use_container_width=True):
            uploaded_files = st.file_uploader("Upload Documents to index", accept_multiple_files=True, type=["pdf", "txt"], label_visibility="collapsed")
            
    with col_run:
        run_btn = st.button("Run Analysis", type="primary", use_container_width=True)

    st.markdown("---")

    if run_btn:
        if uploaded_files:
            with st.spinner("Indexing uploaded documents via FAISS..."):
                process_uploaded_files(uploaded_files)

        app = load_app()
        tracker = TokenTrackingCallback()
        config = {"configurable": {"thread_id": "session_" + str(time.time())}, "recursion_limit": 10, "callbacks": [tracker]}
        
        initial_state = {"query": query, "documents": [], "insights": [], "iteration": 0, "final_report": ""}

        m1, m2, m3 = st.columns(3)
        sources_met  = m1.empty()
        agent_met    = m2.empty()
        elapsed_met  = m3.empty()

        def update_metrics(sources=0, agent="—", elapsed=0.0):
            sources_met.markdown(f'<div class="metric-panel"><h3>{sources}</h3><small>Sources Extracted</small></div>', unsafe_allow_html=True)
            agent_met.markdown(f'<div class="metric-panel"><h3 style="font-size:1.1rem !important; padding-top:8px;">{agent}</h3><small>Active Process</small></div>', unsafe_allow_html=True)
            elapsed_met.markdown(f'<div class="metric-panel"><h3>{elapsed:.0f}s</h3><small>Time Elapsed</small></div>', unsafe_allow_html=True)

        update_metrics(0, "Booting...")
        
        st.markdown("### <br>Live Execution Graph", unsafe_allow_html=True)
        agent_placeholder = st.empty()
        agent_log = []

        def render_agents():
            html = ""
            for icon, name, detail, status in agent_log:
                css = {"active": "active", "done": "done", "error": "error"}.get(status, "")
                badge = {"active": "⏳", "done": "✅", "error": "❌"}.get(status, "")
                html += f'<div class="agent-widget {css}"><div style="font-weight:600; font-size:1.05rem; margin-bottom:2px;">{icon} {name} <span style="float:right">{badge}</span></div><div style="color:#94a3b8;">{detail}</div></div>'
            agent_placeholder.markdown(html, unsafe_allow_html=True)

        start = time.time()
        total_sources = 0

        try:
            for event in app.stream(initial_state, config=config):
                for node_name, partial_state in event.items():
                    icon, name, base_detail = AGENT_INFO.get(node_name, ("⚙️", node_name, "Processing task..."))
                    detail = base_detail

                    if node_name == "retriever":
                        docs = partial_state.get("documents", [])
                        total_sources = len(docs)
                        detail = f"Gathered {total_sources} document chunks."
                    elif node_name == "analyzer":
                        insights = partial_state.get("insights", [])
                        detail = f"[gpt-4o-mini + claude-haiku] Council synthesized {len(insights)} key insights."
                    elif node_name == "reflection":
                        r = partial_state.get("reflection")
                        if r: detail = f"Gaps found: {len(r.coverage_gaps)}. Needs more research: {r.needs_more_research}"
                    
                    if agent_log:
                        p = agent_log[-1]
                        agent_log[-1] = (p[0], p[1], p[2], "done")

                    agent_log.append((icon, name, detail, "active"))
                    render_agents()

                    elapsed = time.time() - start
                    update_metrics(total_sources, name, elapsed)

            if agent_log:
                p = agent_log[-1]
                agent_log[-1] = (p[0], p[1], p[2], "done")
                render_agents()

            elapsed = time.time() - start
            update_metrics(total_sources, "Complete 🎉", elapsed)

            st.session_state.full_state = app.get_state(config).values
            st.session_state.chat_history = []
            
            import json
            # Log metrics to DB
            log_stats(
                query=query, 
                models_used=list(tracker.models_used), 
                input_tokens=tracker.input_tokens, 
                output_tokens=tracker.output_tokens, 
                estimated_cost=tracker.estimated_cost,
                model_breakdown=json.dumps(tracker.model_breakdown)
            )
            
            sources_met.empty()
            agent_met.empty()
            elapsed_met.empty()
            agent_placeholder.empty()

        except Exception as e:
            st.error(f"❌ Execution Failure: {str(e)}")
            if agent_log:
                p = agent_log[-1]
                agent_log[-1] = (p[0], p[1], f"Error: {str(e)[:100]}", "error")
                render_agents()

    if "full_state" in st.session_state:
        full_state = st.session_state.full_state
        if full_state and full_state.get("final_report"):
            st.success("✅ Research Protocol Completed")
            st.markdown("## 📄 Comprehensive Report")
            st.markdown(full_state["final_report"])
            
            with st.expander("🔍 View Agent Internal Thoughts & Citations"):
                docs = full_state.get("documents", [])
                for i, d in enumerate(docs[:5]):
                    st.text_area(f"Source {i+1} ({d.get('metadata', {}).get('source', 'unknown')})", d.get('page_content', ''), height=100)
                st.markdown("### 2. Model Council Insights")
                for insight in full_state.get("insights", []):
                    st.markdown(f"- {insight}")
                    
            st.markdown("---")
            st.markdown("### 💬 Follow-up Conversation")
            for msg in st.session_state.chat_history:
                with st.chat_message(msg["role"]): st.markdown(msg["content"])
                    
            if chat_q := st.chat_input("Ask a question about the findings..."):
                st.session_state.chat_history.append({"role": "user", "content": chat_q})
                with st.chat_message("user"): st.markdown(chat_q)
                with st.chat_message("assistant"):
                    with st.spinner("Analyzing context..."):
                        from services.llm import fast_llm
                        from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
                        sys_prompt = f"Answer based strictly on the report and evidence:\n\nREPORT:\n{full_state['final_report']}\n\nEVIDENCE:\n"
                        for d in full_state.get('documents', [])[:15]:
                            sys_prompt += f"{d.get('page_content', '')}\n---\n"
                            
                        messages = [SystemMessage(content=sys_prompt)]
                        for msg in st.session_state.chat_history[:-1]:
                            messages.append(HumanMessage(content=msg["content"]) if msg["role"] == "user" else AIMessage(content=msg["content"]))
                        messages.append(HumanMessage(content=chat_q))
                        
                        try:
                            response = fast_llm.invoke(messages)
                            reply = response.content
                        except Exception as e:
                            reply = f"Error: {e}"
                            
                        st.markdown(reply)
                        st.session_state.chat_history.append({"role": "assistant", "content": reply})

with tab_stats:
    st.markdown("## 📊 Usage Metrics & Cost Dashboard")
    st.markdown("Track your queries, API token usage, and estimated billing.")
    
    agg = get_aggregate_stats()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Queries", agg["total_queries"])
    c2.metric("Total Input Tokens", f"{agg['total_input_tokens']:,}")
    c3.metric("Total Output Tokens", f"{agg['total_output_tokens']:,}")
    c4.metric("Estimated API Cost", f"${agg['total_cost']:.4f}")
    
    st.markdown("---")
    st.markdown("### 🕒 Conversation History")
    df = get_stats_history()
    
    if not df.empty:
        import json
        for _, row in df.iterrows():
            title_query = row['query'][:80] + "..." if len(row['query']) > 80 else row['query']
            with st.expander(f"📅 **{row['timestamp']}** | 🔍 {title_query} | 💰 Total Cost: ${row['estimated_cost']:.4f}"):
                st.markdown(f"**Full Query**: {row['query']}")
                
                # Safely parse model breakdown
                try:
                    breakdown_str = row.get("model_breakdown", "{}")
                    # If it's a float (NaN) or None, set it to "{}"
                    if not isinstance(breakdown_str, str):
                        breakdown_str = "{}"
                    
                    breakdown = json.loads(breakdown_str)
                    
                    if breakdown:
                        st.markdown("### Model Breakdown")
                        for model, stats in breakdown.items():
                            st.markdown(f"🔹 **{model}**")
                            col1, col2, col3 = st.columns(3)
                            col1.metric("Input Tokens", f"{stats['input_tokens']:,}")
                            col2.metric("Output Tokens", f"{stats['output_tokens']:,}")
                            col3.metric("Cost", f"${stats['cost']:.4f}")
                    else:
                        st.markdown(f"**Models Used**: {row['models_used']}")
                except Exception as e:
                    st.markdown(f"**Models Used**: {row['models_used']}")
    else:
        st.info("No queries have been run yet. Run a research query to see stats here!")
