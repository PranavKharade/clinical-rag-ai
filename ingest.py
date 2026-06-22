import os
from dotenv import load_dotenv
from llama_index.core import Document, Settings, SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.readers.papers import PubmedReader
from llama_index.readers.file import PyMuPDFReader
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from qdrant_client import QdrantClient
from llama_index.vector_stores.qdrant import QdrantVectorStore

# Load the secret keys from the .env file
load_dotenv()

def main():
    print("1. Loading Local Embedding Model (BAAI/bge-base-en-v1.5)...")
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
    Settings.embed_model = embed_model

    documents = []

    print("2. Fetching recent papers from PubMed API...")
    pubmed_reader = PubmedReader()
    pubmed_docs = pubmed_reader.load_data(
        search_query="myocardial infarction treatment guidelines", 
        max_results=3
    )
    documents.extend(pubmed_docs)
    print(f"   -> Fetched {len(pubmed_docs)} papers from PubMed.")

    print("3. Checking for local clinical PDFs in /data folder...")
    os.makedirs("data", exist_ok=True)
    pdf_reader = PyMuPDFReader()
    
    for file in os.listdir("data"):
        if file.endswith(".pdf"):
            pdf_path = os.path.join("data", file)
            pdf_docs = pdf_reader.load_data(file_path=pdf_path)
            documents.extend(pdf_docs)
            print(f"   -> Parsed local PDF: {file}")

    if not documents:
        print("Error: No documents loaded. Exiting.")
        return

    print("4. Applying Semantic Chunking Strategy...")
    splitter = SemanticSplitterNodeParser(
        buffer_size=1,
        breakpoint_percentile_threshold=90, 
        embed_model=embed_model
    )
    nodes = splitter.get_nodes_from_documents(documents)
    print(f"   -> Generated {len(nodes)} semantic chunks.")

    print("5. Connecting to Qdrant Cloud Database...")
    # Get keys securely from the .env file
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")

    # Connect to your cloud cluster
    client = QdrantClient(
        url=qdrant_url, 
        api_key=qdrant_api_key,
        timeout=120.0,
    )

    print("6. Uploading chunks to the Vector Database (this may take a minute)...")
    # Create a specific collection (folder) in your database named "medical_guidelines"
    vector_store = QdrantVectorStore(client=client, collection_name="medical_guidelines")
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # This takes your 167 chunks, runs them through the embedding model, 
    # and pushes them straight into the cloud!
    index = VectorStoreIndex(
        nodes=nodes,
        storage_context=storage_context,
    )

    print("\n✅ Phase 2 Complete! Your medical library is now live in the cloud!")

if __name__ == "__main__":
    main()