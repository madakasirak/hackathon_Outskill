import streamlit as st
import os
import sys

# Ensure local imports work when run from outside the deep_researcher directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from graph import create_research_graph

def create_pdf(text_md):
    from fpdf import FPDF
    
    class PDF(FPDF):
        def header(self):
            self.set_font('helvetica', 'B', 15)
            self.cell(0, 10, 'Deep Research Report', border=False, ln=1, align='C')
            self.ln(5)
            
        def footer(self):
            self.set_y(-15)
            self.set_font('helvetica', 'I', 8)
            self.cell(0, 10, f'Page {self.page_no()}', border=False, ln=0, align='C')

    pdf = PDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # Replace smart quotes and common unicode characters
    replacements = {
        '“': '"', '”': '"', "‘": "'", "’": "'",
        '–': '-', '—': '-', '…': '...',
        '\u200b': '', '\u202f': ' ', '\u00a0': ' ',
        '•': '-', '·': '-',
    }
    for orig, rep in replacements.items():
        text_md = text_md.replace(orig, rep)
        
    # Remove all remaining non-latin1 characters (like emojis) to avoid encoding errors
    text_cleaned = ''.join(c if ord(c) < 256 else '' for c in text_md)
    
    # Use multi_cell for text
    pdf.set_font("helvetica", size=11)
    pdf.multi_cell(w=0, h=7, text=text_cleaned)
    return pdf.output()

st.set_page_config(page_title="Deep Researcher", page_icon="🔍", layout="wide")

st.title("🤖 Multi-Agent AI Deep Researcher")
st.markdown("A LangGraph-powered research system using OpenRouter.")

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input("OpenRouter API Key *", type="password")
    model = st.selectbox(
        "Model",
        ["openai/gpt-4o-mini", "anthropic/claude-4.5-sonnet"],
        index=0
    )
    
    st.header("🔌 External Tools (Optional)")
    tavily_key = st.text_input("Tavily API Key", type="password")
    serpapi_key = st.text_input("SerpAPI Key", type="password")
    st.caption("Arxiv is enabled by default.")
    
    st.markdown("---")
    st.header("📄 Local Knowledge (RAG)")
    uploaded_files = st.file_uploader(
        "Upload Documents (PDF, TXT)",
        accept_multiple_files=True,
        type=["pdf", "txt"]
    )
    
    st.markdown("---")
    st.header("🛠️ Chat Actions")
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
        
    if "messages" in st.session_state and len(st.session_state.messages) > 0:
        last_report = None
        for msg in reversed(st.session_state.messages):
            if msg["role"] == "assistant":
                last_report = msg["content"]
                break
        
        if last_report:
            try:
                pdf_bytes = create_pdf(last_report)
                st.download_button(
                    label="📥 Download Last Report (PDF)",
                    data=bytes(pdf_bytes),
                    file_name="deep_research_report.pdf",
                    mime="application/pdf"
                )
            except Exception as e:
                st.error(f"Could not generate PDF: {e}")
    
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

workflow = create_research_graph()

with st.expander("🗺️ View LangGraph Architecture", expanded=False):
    try:
        # Generate the graph visualization using Mermaid
        graph_png = workflow.get_graph().draw_mermaid_png()
        st.image(graph_png, caption="Multi-Agent Workflow Architecture", use_container_width=True)
    except Exception as e:
        st.error(f"Could not generate graph image: {e}")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Enter a research topic, ask a question, or query your uploaded documents..."):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        if not os.environ.get("OPENROUTER_API_KEY"):
            st.error("Please enter your OpenRouter API Key in the sidebar.")
        else:
            with st.spinner("Initializing Deep Research Agents..."):
                local_index_path = None
                if uploaded_files:
                    st.info("Embedding uploaded documents (this may take a moment)...")
                    local_index_path = process_uploaded_files(uploaded_files)

                # Initialize state
                initial_state = {"topic": prompt, "errors": []}
                if local_index_path:
                    initial_state["local_vector_store"] = local_index_path
                
                progress_container = st.container()
                
                try:
                    with progress_container:
                        st.info(f"Initiating research protocol for: **{prompt}**")
                        stages = ["Retriever", "Analysis", "Insight", "Report"]
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        from langchain_core.callbacks import BaseCallbackHandler
                        
                        class ReportStreamHandler(BaseCallbackHandler):
                            def __init__(self, container):
                                self.container = container
                                self.text = ""
                                self.is_reporting = False

                            def on_chat_model_start(self, serialized, messages, **kwargs):
                                tags = kwargs.get("tags", [])
                                if "report_agent" in tags:
                                    self.is_reporting = True
                                    self.text = ""
                                else:
                                    self.is_reporting = False

                            def on_llm_new_token(self, token: str, **kwargs) -> None:
                                if self.is_reporting:
                                    self.text += token
                                    self.container.markdown(self.text)
                        
                        report_container = st.empty()
                        stream_handler = ReportStreamHandler(report_container)
                        
                        final_state = {}
                        step = 0
                        
                        # Stream state updates straight from the LangGraph workflow
                        for output in workflow.stream(initial_state, config={"callbacks": [stream_handler]}):
                            for node_name, state_update in output.items():
                                step += 1
                                progress = min(step / len(stages), 1.0)
                                progress_bar.progress(progress)
                                status_text.markdown(f"🟢 **Agent Active:** `{node_name.upper()}`")
                                
                                # Merge state updates dictionary
                                if isinstance(state_update, dict):
                                    final_state.update(state_update)
                    
                    st.success("Research Protocol Complete!")
                    
                    # Store the final generated report in chat history
                    report_content = final_state.get("final_report", "Report generation failed. Please check logs.")
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": report_content})
                    
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
