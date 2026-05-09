from typing import Any, Optional
from pathlib import Path

from graph.state import ResearchState

# LangChain imports for RAG
try:
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
except Exception as e:
    raise ImportError(
        "langchain and related packages are required for FAISS RAG.\n"
        "Install dependencies with: `pip install -r requirements.txt`.\n"
        "On macOS, installing `faiss-cpu` via pip may fail; consider using conda:\n"
        "  conda install -c conda-forge faiss-cpu\n"
        f"Underlying error: {e}"
    ) from e


def run(state: ResearchState, pdf_path: Optional[str] = None, limit: int = 5) -> None:
    """Populate `state.retrieved_docs` and `state.sources` using FAISS RAG.

    - `pdf_path` must be a path to a local PDF file.
    - `state.query` is used as the retrieval question against the FAISS index.
    """
    if not pdf_path:
        raise ValueError("pdf_path is required for FAISS RAG retrieval")

    path = Path(pdf_path)
    if not path.exists() or path.suffix.lower() != ".pdf":
        raise ValueError(f"pdf_path not found or not a PDF: {pdf_path}")

    # Load and split PDF
    docs = PyPDFLoader(str(path)).load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(docs)

    # Embeddings and FAISS
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.from_documents(chunks, embeddings)
    retriever = db.as_retriever(search_kwargs={"k": limit})

    # Use the user's research query to fetch relevant chunks
    query = state.query or ""
    # Call retriever with compatibility fallbacks for different langchain versions
    try:
        hits = retriever.get_relevant_documents(query)
    except AttributeError:
        # Some Retriever implementations expose a protected `_get_relevant_documents`
        # or only async variants. Try falling back gracefully and supply `run_manager`
        # as a keyword-only argument when required by some langchain versions.
        if hasattr(retriever, "_get_relevant_documents"):
            try:
                hits = retriever._get_relevant_documents(query, run_manager=None)
            except TypeError:
                hits = retriever._get_relevant_documents(query)
        elif hasattr(retriever, "aget_relevant_documents"):
            import asyncio

            try:
                hits = asyncio.run(retriever.aget_relevant_documents(query, run_manager=None))
            except TypeError:
                hits = asyncio.run(retriever.aget_relevant_documents(query))
        else:
            raise

    state.retrieved_docs = []
    state.sources = []
    for i, d in enumerate(hits):
        src = d.metadata.get("source") if isinstance(d.metadata, dict) else None
        text = getattr(d, "page_content", str(d))
        state.retrieved_docs.append({
            "title": src or f"chunk_{i}",
            "url": src or "",
            "snippet": text[:400],
            "text": text,
        })
        if src:
            state.sources.append(src)
