import gradio as gr
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from uuid import uuid4
import os
import chromadb

# Initialize Chroma client and embeddings
chroma_client = chromadb.HttpClient(
    host=os.environ.get("CHROMA_HOST", "localhost"),
    port=int(os.environ.get("CHROMA_PORT", 8000))
)
embeddings = OllamaEmbeddings(model="nomic-embed-text")

def process_pdfs(pdf_files, collection_name, description, source, language, doc_type, chunk_size, chunk_overlap):
    pages = []
    for pdf_file in pdf_files:
        loader = PyPDFLoader(pdf_file.name)
        pages.extend(loader.load())
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=int(chunk_size),
        chunk_overlap=int(chunk_overlap)
    )
    docs = text_splitter.split_documents(pages)
    
    for doc in docs:
        doc.id = str(uuid4())
    
    # Create vector store with user inputs
    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        client=chroma_client,
        collection_metadata={
            "description": description,
            "source": source,
            "language": language,
            "type": doc_type,
        },
    )
    
    vector_store.add_documents(documents=docs, ids=[doc.id for doc in docs])
    
    return f"✅ Successfully added {len(docs)} documents to the vector store '{collection_name}'."

# Gradio Interface with Two Columns
with gr.Blocks() as demo:
    gr.Markdown("# 📚 Khmer Press Release PDF Processor")
    
    with gr.Row():
        # Left side: File upload and Text Splitter Config
        with gr.Column(scale=1):
            gr.Markdown("### 📂 File Upload")
            pdf_input = gr.File(file_types=['.pdf'], label="Upload PDF Files", file_count="multiple")
            
            gr.Markdown("### ✂️ Text Splitter Configuration")
            chunk_size = gr.Number(label="Chunk Size", value=500, precision=0)
            chunk_overlap = gr.Number(label="Chunk Overlap", value=100, precision=0)
        
        # Right side: Collection Config and Output
        with gr.Column(scale=1):
            gr.Markdown("### 📑 Collection Configuration")
            collection_name = gr.Textbox(label="Collection Name", value="khmer_press_release")
            description = gr.Textbox(label="Description", value="This is a collection of Khmer press release documents.")
            source = gr.Textbox(label="Source", value="Khmer Press Release")
            language = gr.Textbox(label="Language", value="English")
            doc_type = gr.Textbox(label="Document Type", value="PDF")
            
            process_button = gr.Button("🚀 Process and Add to Vector Store")
            output = gr.Textbox(label="Output", lines=4)
    
    # Button Click Handler
    process_button.click(
        fn=process_pdfs,
        inputs=[pdf_input, collection_name, description, source, language, doc_type, chunk_size, chunk_overlap],
        outputs=[output]
    )

demo.launch()
