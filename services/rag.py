"""Shared RAG infrastructure — single embedding model."""
import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
USER_FAISS_PATH = "temp_faiss_index"

def load_user_faiss() -> FAISS | None:
    """Load uploaded-document FAISS index if it exists."""
    if not os.path.exists(USER_FAISS_PATH):
        return None
    try:
        return FAISS.load_local(
            USER_FAISS_PATH, embeddings, allow_dangerous_deserialization=True
        )
    except Exception as e:
        print(f"FAISS load error: {e}")
        return None
