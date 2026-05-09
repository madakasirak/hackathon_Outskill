import os
import io
import contextlib
import math
import statistics
import json

try:
    import pypdf
    _PYPDF_AVAILABLE = True
except ImportError:
    _PYPDF_AVAILABLE = False

def parse_pdf(file_path: str, max_pages: int = 20) -> list[dict]:
    """
    Extract text from a local PDF file and return it as source dicts
    (one dict per page).
    """
    if not _PYPDF_AVAILABLE:
        print("pypdf not installed. Skipping PDF parse.")
        return [{"page_content": "", "metadata": {"source": "pdf", "error": "pypdf not installed"}}]

    if not os.path.exists(file_path):
        return [{"page_content": "", "metadata": {"source": "pdf", "error": f"File not found: {file_path}"}}]

    reader = pypdf.PdfReader(file_path)
    results = []
    for i, page in enumerate(reader.pages[:max_pages]):
        text = page.extract_text() or ""
        results.append({
            "page_content": text[:800],
            "metadata": {
                "source": "pdf", 
                "title": f"{os.path.basename(file_path)} – Page {i + 1}",
                "url": f"file://{os.path.abspath(file_path)}#page={i + 1}"
            }
        })
    return results

def python_repl(code: str) -> str:
    """
    Execute Python code for numerical calculations or data processing.
    """
    allowed_globals = {
        "__builtins__": {
            k: __builtins__[k] if isinstance(__builtins__, dict) else getattr(__builtins__, k)
            for k in [
                "print", "len", "range", "sum", "min", "max",
                "abs", "round", "sorted", "enumerate", "zip",
                "list", "dict", "str", "int", "float", "bool",
            ] if (isinstance(__builtins__, dict) and k in __builtins__) or (not isinstance(__builtins__, dict) and hasattr(__builtins__, k))
        },
        "math": math,
        "statistics": statistics,
        "json": json,
    }

    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            exec(code, allowed_globals)
    except Exception as exc:
        return f"ERROR: {exc}"

    return buf.getvalue().strip() or "(no output)"
