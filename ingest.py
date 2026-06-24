import os
import sys
import glob
from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex, Settings
from llama_index.core.schema import TextNode
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

#import configuration first
from config import DATA_DIR, INDEX_DIR, CHUNK_SIZE, CHUNK_OVERLAP, LLM_MODEL, EMBED_MODEL, GEMINI_API_KEY

def initialize_settings():
    """
    Initializes LlamaIndex global settings (LLM and Embedding model).
    This ensures that all index and query operations use Gemini.
    """
    if not GEMINI_API_KEY:
        print("[ERROR] GEMINI_API_KEY is not set. Please create a .env file containing: GEMINI_API_KEY=your_key")
        sys.exit(1)
        
    # Configure Gemini Embeddings with a larger batch size to minimize API requests
    Settings.embed_model = GoogleGenAIEmbedding(
        model_name=EMBED_MODEL,
        api_key=GEMINI_API_KEY,
        embed_batch_size=100
    )
    
    # Configure Gemini LLM
    # Model: gemini-1.5-flash (Fast, accurate, large context window)
    Settings.llm = GoogleGenAI(
        model_name=LLM_MODEL,
        api_key=GEMINI_API_KEY
    )
    
    # Configure Default Chunker Settings
    Settings.node_parser = SentenceSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

def ingest_documents():
    """
    Loads PDFs, CSVs, TXT, or MD documents from data directory, chunks them,
    extracts metadata, and saves embeddings into ChromaDB vector store.
    """
    print(f"Checking for documents in: {DATA_DIR}")
    
    # Check if there are any supported files in data folder
    supported_files = []
    for ext in ['*.pdf', '*.txt', '*.md', '*.csv']:
        supported_files.extend(glob.glob(os.path.join(DATA_DIR, ext)))
        
    if not supported_files:
        print(f"\n[WARNING] No documents found in '{DATA_DIR}'.")
        print("Please place your governance CSV, PDF, TXT, or MD files inside the 'data' directory and run this script again.")
        return False

    print(f"Found {len(supported_files)} files to ingest. Starting parsing...")

    # Initialize LlamaIndex Settings
    initialize_settings()

    nodes = []

    # Process files one by one to handle CSVs specifically
    for file_path in supported_files:
        file_name = os.path.basename(file_path)
        print(f"Processing: {file_name}...")
        
        if file_path.endswith('.csv'):
            # Custom CSV Parser to turn rows into structured nodes
            import csv
            # We limit rows to 10000 per CSV to read all data (since total rows across all CSVs is 6188)
            MAX_ROWS = 10000
            try:
                with open(file_path, mode='r', encoding='utf-8-sig') as f:
                    reader = csv.DictReader(f)
                    row_idx = 0
                    for row in reader:
                        if row_idx >= MAX_ROWS:
                            print(f"  -> [RATE LIMIT PROTECTION] Capped parsing at {MAX_ROWS} rows for {file_name} to avoid Gemini free-tier 429 quota errors.")
                            break
                            
                        # Format row content nicely: "ColumnHeader: Value"
                        row_content_parts = []
                        metadata = {
                            "file_name": file_name,
                            "page_label": f"Row {row_idx + 1}"
                        }
                        
                        # Extract row columns
                        for col_name, col_val in row.items():
                            if col_val and col_val.strip():
                                row_content_parts.append(f"{col_name}: {col_val.strip()}")
                                
                                # Promote common column fields to metadata for filtering/search
                                clean_col = col_name.lower().strip()
                                if clean_col in ["policy", "policy_name", "title", "policy name"]:
                                    metadata["policy_name"] = col_val.strip()
                                elif clean_col in ["section", "category", "type"]:
                                    metadata["section"] = col_val.strip()
                        
                        row_text = "\n".join(row_content_parts)
                        
                        # Skip empty rows
                        if not row_text.strip():
                            continue
                            
                        # Create LlamaIndex TextNode
                        node = TextNode(
                            text=row_text,
                            metadata=metadata
                        )
                        nodes.append(node)
                        row_idx += 1
                print(f"  -> Parsed {row_idx} rows from CSV.")
            except Exception as e:
                print(f"[ERROR] Failed to parse CSV {file_name}: {e}")
                
        else:
            # Standard PDF/TXT/MD Reader
            try:
                reader = SimpleDirectoryReader(input_files=[file_path])
                documents = reader.load_data()
                # Parse into chunks (nodes)
                file_nodes = Settings.node_parser.get_nodes_from_documents(documents)
                
                # Standardize metadata
                for node in file_nodes:
                    node.metadata["file_name"] = file_name
                    if "page_label" not in node.metadata:
                        node.metadata["page_label"] = "1"
                        
                nodes.extend(file_nodes)
                print(f"  -> Parsed {len(file_nodes)} chunks from document.")
            except Exception as e:
                print(f"[ERROR] Failed to parse file {file_name}: {e}")

    if not nodes:
        print("[WARNING] No text nodes created. Ingestion stopped.")
        return False

    print(f"Total chunks created across all files: {len(nodes)}")

    # 3. Setup ChromaDB Vector Store
    print(f"Connecting to ChromaDB at: {INDEX_DIR}")
    chroma_client = chromadb.PersistentClient(path=INDEX_DIR)
    
    # Get or create collection
    chroma_collection = chroma_client.get_or_create_collection("governance_docs")
    
    # Connect LlamaIndex ChromaVectorStore wrapper to the Chroma Collection
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    
    # Storage context tells LlamaIndex where to store vectors
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # 4. Generate embeddings in batches manually to bypass LlamaIndex's one-by-one loop
    print("Generating embeddings in batches via Gemini API (Rate Limit Protection)...")
    
    node_texts = [node.get_content(metadata_mode="embed") for node in nodes]
    batch_size = 100  # Max batch size allowed by Google API
    embeddings = []
    
    import time
    total_batches = ((len(node_texts) - 1) // batch_size) + 1
    for i in range(0, len(node_texts), batch_size):
        batch = node_texts[i:i + batch_size]
        print(f"  -> Processing embedding batch {i // batch_size + 1} of {total_batches} ({len(batch)} nodes)...")
        # Generate batch embeddings (only 1 API request per batch)
        batch_embeddings = Settings.embed_model.get_text_embedding_batch(batch)
        embeddings.extend(batch_embeddings)
        # Sleep for 2.0 seconds between batches to be absolutely safe on Free Tier (100 RPM limit)
        time.sleep(2.0)
        
    # Assign embeddings back to nodes
    for node, embedding in zip(nodes, embeddings):
        node.embedding = embedding
        
    print("All embeddings generated. Registering nodes in ChromaDB...")
    index = VectorStoreIndex(
        nodes,
        storage_context=storage_context,
        show_progress=True
    )
    
    print("\n[SUCCESS] Ingestion completed successfully!")
    print(f"ChromaDB collection 'governance_docs' currently has {chroma_collection.count()} chunks.")
    return True

if __name__ == "__main__":
    ingest_documents()
