import streamlit as st
import os
from dotenv import load_dotenv
from llama_index.core import Settings, VectorStoreIndex
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.llms.groq import Groq
from qdrant_client import QdrantClient
from llama_index.vector_stores.qdrant import QdrantVectorStore

# Load environment variables (for local testing)
load_dotenv()

# Configure the web page
st.set_page_config(page_title="Medical AI Assistant", page_icon="🩺", layout="centered")

# --- CACHING THE ENGINE ---
# We use @st.cache_resource so the app doesn't reconnect to the database every time you type a letter
@st.cache_resource
def initialize_rag():
    # 1. Math Translator (Lightweight FastEmbed Model)
    Settings.embed_model = FastEmbedEmbedding(model_name="BAAI/bge-base-en-v1.5")
    
    # 2. The Brain (Groq Qwen 3.6 - Updated to replace deprecated Llama 3.3)
    # We use st.secrets here for when we deploy to the cloud!
    groq_key = os.environ.get("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")
    Settings.llm = Groq(model="qwen-3.6-27b", api_key=groq_key)
    
    # 3. The Database Connection (Qdrant)
    qdrant_url = os.environ.get("QDRANT_URL") or st.secrets.get("QDRANT_URL")
    qdrant_key = os.environ.get("QDRANT_API_KEY") or st.secrets.get("QDRANT_API_KEY")
    
    client = QdrantClient(url=qdrant_url, api_key=qdrant_key, timeout=120.0)
    vector_store = QdrantVectorStore(client=client, collection_name="medical_guidelines")
    
    # 4. The Librarian (Query Engine)
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
    return index.as_query_engine(similarity_top_k=5)

# --- UI DESIGN ---
st.title("🩺 Clinical RAG Assistant")
st.markdown("Ask clinical questions based on **AHA Guidelines** & **PubMed Trials**.")

# Initialize chat history so the conversation stays on screen
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Load the engine silently in the background
try:
    query_engine = initialize_rag()
except Exception as e:
    st.error(f"Error starting AI Engine: {e}")
    st.stop()

# --- CHAT INPUT ---
if prompt := st.chat_input("E.g., What is the recommended therapy after STEMI?"):
    
    # Add user message to history and display it
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate AI response
    with st.chat_message("assistant"):
        with st.spinner("Searching clinical literature..."):
            response = query_engine.query(prompt)
            answer = response.response
            
            # Show the answer
            st.markdown(answer)
            
            # Show the sources in a neat, clickable dropdown expander
            with st.expander("📚 View Clinical Sources"):
                for i, node in enumerate(response.source_nodes):
                    st.write(f"**Source {i+1}:**")
                    st.caption(f"{node.text[:300]}...")
                    st.divider()
            
    # Add assistant response to history
    st.session_state.messages.append({"role": "assistant", "content": answer})