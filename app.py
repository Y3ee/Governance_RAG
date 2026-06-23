import os
import streamlit as st
import chromadb
from config import INDEX_DIR, DATA_DIR
from retrieve import get_hybrid_query_engine
from generate import generate_grounded_answer
from ingest import ingest_documents

# Page configuration
st.set_page_config(
    page_title="Governance RAG Portal",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium Styling (Apple-inspired minimalist design)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Core Font and Background overrides */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, sans-serif;
    }
    
    .stApp {
        background-color: #FFFFFF !important;
        color: #111111 !important;
    }
    
    /* Sidebar override using F5EBE0 as background accent */
    [data-testid="stSidebar"] {
        background-color: #F5EBE0 !important;
        border-right: 1px solid #E5E5E5;
    }
    
    [data-testid="stSidebar"] .stMarkdown h1, 
    [data-testid="stSidebar"] .stMarkdown h2, 
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: #111111 !important;
    }
    
    /* Title styling - clean, bold, black (Apple-style) */
    .title-text {
        color: #111111;
        font-weight: 700;
        font-size: 2.75rem;
        letter-spacing: -0.03em;
        margin-bottom: 0.25rem;
    }
    
    /* Subtle subtitle */
    .subtitle-text {
        color: #666666;
        font-size: 1.1rem;
        font-weight: 400;
        margin-bottom: 1.5rem;
    }
    
    /* Custom cards for sources - minimalist with clean borders */
    .source-card {
        background: #FFFFFF;
        border-radius: 6px;
        padding: 18px;
        margin-bottom: 12px;
        border: 1px solid #E5E5E5;
        color: #111111;
        transition: transform 0.2s ease-in-out, box-shadow 0.2s ease-in-out, border-color 0.2s ease-in-out;
    }
    .source-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        border-color: #111111;
    }
    
    /* Grounded Badge style */
    .badge-grounded {
        background-color: #F5EBE0;
        color: #111111;
        border: 1px solid #111111;
        padding: 5px 12px;
        border-radius: 16px;
        font-size: 0.8rem;
        font-weight: 500;
        display: inline-block;
        margin-bottom: 12px;
    }
    
    /* Warning Badge style */
    .badge-warning {
        background-color: #FFFFFF;
        color: #B22222;
        border: 1px solid #B22222;
        padding: 5px 12px;
        border-radius: 16px;
        font-size: 0.8rem;
        font-weight: 500;
        display: inline-block;
        margin-bottom: 12px;
    }
    
    /* Apple-like clean divider */
    .clean-divider {
        height: 1px;
        background-color: #E5E5E5;
        margin-bottom: 25px;
    }

    /* Streamlit button custom styling */
    .stButton>button {
        background-color: #111111 !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 6px !important;
        font-weight: 500 !important;
        padding: 8px 16px !important;
        transition: background-color 0.2s ease !important;
    }
    .stButton>button:hover {
        background-color: #333333 !important;
        color: #FFFFFF !important;
    }
</style>
""", unsafe_allow_html=True)

# Helper to get uniquely indexed files from ChromaDB
def get_indexed_files():
    try:
        chroma_client = chromadb.PersistentClient(path=INDEX_DIR)
        chroma_collection = chroma_client.get_collection("governance_docs")
        results = chroma_collection.get(include=["metadatas"])
        if results and results["metadatas"]:
            file_names = set()
            for meta in results["metadatas"]:
                if meta and "file_name" in meta:
                    file_names.add(meta["file_name"])
            return sorted(list(file_names)), chroma_collection.count()
    except Exception:
        pass
    return [], 0

# header
st.markdown("<div class='title-text'>Governance Policy RAG Portal</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle-text'>Hybrid Semantic Search and Automated Compliance Grounding for Governance Documentation</div>", unsafe_allow_html=True)
st.markdown("<div class='clean-divider'></div>", unsafe_allow_html=True)

# Fetch indexed files for filtering
available_files, chunk_count = get_indexed_files()

# Sidebar Configuration
st.sidebar.title("System Configuration")
st.sidebar.markdown("---")

# Display status of the local vector database
if chunk_count > 0:
    st.sidebar.success(f"Database Active: {chunk_count} chunks indexed from {len(available_files)} files.")
else:
    st.sidebar.warning("Database Empty. Please run ingestion to index governance documents.")

st.sidebar.markdown("### Document Filter")

# Let the user filter by document name
selected_files = st.sidebar.multiselect(
    "Search only in:",
    options=available_files,
    default=None,
    help="Leave empty to search across all governance policies."
)

st.sidebar.markdown("### Search Settings")
rerank_active = st.sidebar.toggle(
    "Enable Reranking", 
    value=True, 
    help="Reranks final context search using Cohere API if available."
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Upload Documents")
uploaded_files = st.sidebar.file_uploader(
    "Upload policy documents (PDF, TXT, CSV, MD):",
    accept_multiple_files=True,
    type=["pdf", "txt", "csv", "md"]
)

if uploaded_files:
    for uploaded_file in uploaded_files:
        file_path = os.path.join(DATA_DIR, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
    st.sidebar.success(f"Saved {len(uploaded_files)} files to data folder!")

if st.sidebar.button("Sync and Index Documents"):
    with st.spinner("Processing files and indexing embeddings..."):
        success = ingest_documents()
        if success:
            st.sidebar.success("Ingestion successful! Refreshing directory filters...")
            st.rerun()
        else:
            st.sidebar.error("Ingestion failed. Please verify files are placed inside data folder.")

# Main Application Logic
if chunk_count == 0:
    st.info("Welcome! It looks like your vector database is empty. Put some governance CSV files in the data folder and click 'Sync and Index Documents' on the left sidebar to start.")
else:
    # Initialize query engine and cache it inside session state to prevent reload lag
    if "query_engine" not in st.session_state:
        with st.spinner("Loading AI query engine..."):
            st.session_state.query_engine = get_hybrid_query_engine(similarity_top_k=6, rerank=rerank_active)

    # Simple chat input
    query = st.chat_input("Ask a question about policy terms, guidelines, or procedures:")

    if query:
        # User message display
        with st.chat_message("user"):
            st.write(query)

        # Assistant generation display
        with st.chat_message("assistant"):
            with st.spinner("Analyzing context & generating grounded answer..."):
                try:
                    # Run search and generation
                    result = generate_grounded_answer(query, st.session_state.query_engine)
                    
                    # Apply manual frontend filter if documents were selected
                    sources_to_show = result["sources"]
                    if selected_files:
                        sources_to_show = [s for s in result["sources"] if s["file_name"] in selected_files]

                    # Show compliance grounding audit badge
                    if result["is_grounded"]:
                        st.markdown("<span class='badge-grounded'>Grounding Audited and Verified</span>", unsafe_allow_html=True)
                    else:
                        st.markdown("<span class='badge-warning'>Grounding Warning: Potential Extrapolation</span>", unsafe_allow_html=True)
                    
                    # Print generated response
                    st.write(result["answer"])
                    
                    # Show sources and citations in accordion tabs
                    st.markdown("#### Verified Source References")
                    
                    if not sources_to_show:
                        st.caption("No matching sources found after applying selected filters.")
                    else:
                        cols = st.columns(min(len(sources_to_show), 3))
                        for idx, source in enumerate(sources_to_show):
                            col_idx = idx % 3
                            with cols[col_idx]:
                                st.markdown(f"""
                                <div class="source-card">
                                    <strong>Document:</strong> {source['file_name']}<br/>
                                    <strong>Page:</strong> {source['page']}<br/>
                                    <strong>Search Match Score:</strong> {source['score']:.4f if source['score'] is not None else 'N/A'}<br/>
                                    <hr style="margin: 8px 0; border: 0; border-top: 1px solid rgba(255,255,255,0.1);"/>
                                    <em>"{source['text'][:150]}..."</em>
                                </div>
                                """, unsafe_allow_html=True)
                
                except Exception as e:
                    st.error(f"Error querying RAG system: {e}")
                    st.info("If you just updated database configs, try running the ingestion indexer again.")
