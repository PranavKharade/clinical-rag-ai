import os
from dotenv import load_dotenv
from llama_index.core import Settings, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from qdrant_client import QdrantClient
from llama_index.vector_stores.qdrant import QdrantVectorStore

# 1. Unlock our secure passwords
load_dotenv()

def main():
    print("1. Waking up the Embedding Model (The Math Translator)...")
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
    
    print("2. Waking up the LLM (The Brain - Llama 3 via Groq)...")
    # We are using Llama 3 (8 billion parameters) which is incredibly fast for medical summaries
    Settings.llm = Groq(model="llama-3.3-70b-versatile", api_key=os.getenv("GROQ_API_KEY"))
    
    print("3. Connecting to Qdrant Database in Brazil...")
    client = QdrantClient(
        url=os.getenv("QDRANT_URL"), 
        api_key=os.getenv("QDRANT_API_KEY"),
        timeout=120.0
    )
    
    # Connect to the exact folder we uploaded to yesterday
    vector_store = QdrantVectorStore(client=client, collection_name="medical_guidelines")
    
    print("4. Assembling the Librarian (RAG Pipeline)...")
    # Tell LlamaIndex to look at our existing cloud database
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
    
    # Create the search engine and tell it to grab the Top 5 best flashcards
    query_engine = index.as_query_engine(similarity_top_k=5)
    
    print("\n" + "="*50)
    print("🤖 THE MEDICAL AI IS READY")
    print("="*50)
    
    # 5. The Chat Loop!
    while True:
        question = input("\nDoctor's Question (or type 'quit' to exit): ")
        if question.lower() == 'quit':
            break
            
        print("\nThinking... (Searching library & reading flashcards)...")
        # This one line does the entire retrieval and LLM generation process!
        response = query_engine.query(question)
        
        print("\n🩺 Answer:")
        print(response.response)
        
        print("\n📚 Sources used (To prevent hallucination):")
        # Let's print out the exact flashcards it used so the doctor can verify
        for i, node in enumerate(response.source_nodes):
            print(f"  {i+1}. {node.text[:100]}...")

if __name__ == "__main__":
    main()