from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from fastmcp import FastMCP
from .helper import get_chroma_client
from .model import CollectionModel, DocumentModel
import os

mcp = FastMCP(
    name="Chroma DB Document Storage",
    instructions="""
        You are a document storage and retrieval system using Chroma DB.
        - Your tasks include listing collections and retrieving documents based on query similarity.
        - Always respond with structured JSON. For errors, use {"error": "<error message>"}.
        - Avoid explanations or extra text; return only JSON objects representing results.
    """
)

@mcp.tool(name="get_all_collections", description="List all collections with pagination support.")
def get_all_collections(page: int = 1, page_size: int = 10): # type: ignore
    '''
    Arguments:
    page: Page number for pagination (default is 1).
    page_size: Number of collections per page (default is 10).
    Returns a list of CollectionModel objects containing collection names and metadata.
    '''
    client = get_chroma_client()
    if not client:
        return {"error": "Chroma client unavailable."}
    try:
        collections = client.list_collections()
        start = (page - 1) * page_size
        end = start + page_size
        paginated = collections[start:end]
        return [
            CollectionModel(name=col.name, meta=col.metadata) for col in paginated
        ]
    except Exception as e:
        return {"error": f"Failed to list collections: {str(e)}"}

@mcp.tool(name="get_collection_info", description="Get detailed info about a Chroma collection.")
def get_collection_info(name: str):
    '''
    Arguments:
    name: Name of the collection to retrieve information for.
    Returns a CollectionModel object containing the collection name and metadata.
    '''
    client = get_chroma_client()
    if not client:
        return {"error": "Chroma client unavailable."}
    try:
        collection = client.get_collection(name=name)
        return CollectionModel(
            name=collection.name,
            meta=collection.metadata if collection.metadata else {}
        )
    except Exception as e:
        return {"error": f"Failed to get collection info: {str(e)}"}
    
@mcp.tool(name="get_collection_count", description="Get document count in a collection.")
def get_collection_count(name: str):
    '''
    Arguments:
    name: Name of the collection to get the document count for.
    Returns the count of documents in the specified collection.
    '''
    client = get_chroma_client()
    if not client:
        return {"error": "Chroma client unavailable."}
    try:
        collection = client.get_collection(name=name)
        count = collection.count()
        return {"count": count}
    except Exception as e:
        return {"error": f"Failed to get document count: {str(e)}"}

@mcp.tool(name="query_documents", description="Query documents using semantic search and filters.")
def query_documents(collection_name: str, query: str, k: int = 3, filter: dict | None = None): # type: ignore
    '''
    Arguments:
    collection_name: Name of the collection to query.
    query: The search query string.
    k: Number of top results to return (default is 3).
    filter: Optional filter to apply to the search results.(dict from metadata)
    Returns a list of DocumentModel objects containing page content and metadata..
    '''
    client = get_chroma_client()
    if not client:
        return {"error": "Chroma client unavailable."}
    try:
        embeddings = OllamaEmbeddings(model=os.environ.get("OLLAMA_EMBEDDING", "nomic-embed-text"))
        vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            client=client,
        )
        results = vector_store.similarity_search(query, k=k, filter=filter)
        return [
            DocumentModel(page_content=doc.page_content, metadata=doc.metadata)
            for doc in results
        ]
    except Exception as e:
        return {"error": f"Failed to query documents: {str(e)}"}

# if __name__ == "__main__":
#     mcp.run()