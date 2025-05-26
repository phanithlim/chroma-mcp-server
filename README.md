# Chroma MCP Server
Template for a FastAPI application that serves as a server inspector for ChromaDB, providing endpoints to interact with collections and documents.

## Prerequisites
Assume that you have ChromaDB in docker or host model on any Cloud provider. If you don't have it, you can run it locally using Docker:

```bash
docker run -v ./chroma-data:/data -p 8000:8000 chromadb/chroma
```
And for text embedding, we use `nomic-embed-text` from Ollama. But can be replaced with any other embedding model from any provider. If you don't have Ollama installed, you can install and pull text embedding model with the following commands:

```bash
ollama pull nomic-embed-text
```

## Installation
Install UV package manager:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
or pipx
```bash
pip install uv
```

## Install Dependencies
```bash
uv sync
```

## Usage
Run MCP Server Inspector:
```bash
make mcp-dev
```
Run the FastAPI application:
```bash
make dev
```
## Project Structure

```
├── app.py            # FastAPI application entry point
├── Dockerfile        # Containerization support
├── pyproject.toml    # Project metadata and dependencies
├── uv.lock           # uv dependency lock file
├── resource/
│   ├── __init__.py  # Resource contain static files
├── tools/
│   ├── server.py     # Server logic for FastMCP
│   ├── model.py      # Response models for FastMCP
│   ├── helpers.py    # Helper functions for FastMCP
│   └── __init__.py
└── README.md          # Project documentation
```

## Features
- `get_all_collections`: Retrieve all collections from the database.
- `get_collection_info`: Get information about a specific collection.
- `get_collection_items`: Fetch items from a specific collection.
- `get_collection_count`: Count items in a specific collection.
- `query_documents`: Query documents across collections with optional filters.

## Deployment
### Docker
Build the Docker image:
```bash
docker build -t mcp-server .
```
### Gradio UI
To run the Gradio UI, you can use the following command:
```bash
uv run gradio_ui.py
```