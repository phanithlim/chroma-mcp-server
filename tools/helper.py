import os
import chromadb

def get_chroma_client():
    """Get the Chroma client."""
    if os.environ.get("CHROMA_HOST") is None or os.environ.get("CHROMA_PORT") is None:
        return None
    chroma_client = chromadb.HttpClient(
        host=os.getenv("CHROMA_HOST", "localhost"),
        port=int(os.getenv("CHROMA_PORT", 8000)),
    )
    return chroma_client