import os
import sys
import math
import re
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.retrievers import BaseRetriever, QueryFusionRetriever
from llama_index.core.schema import TextNode, NodeWithScore
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
import chromadb

from config import INDEX_DIR, GEMINI_API_KEY
from ingest import initialize_settings

class PurePythonBM25Retriever(BaseRetriever):
    """
    A pure-Python implementation of a BM25 retriever that inherits from LlamaIndex's BaseRetriever.
    This eliminates the need for C-based compilation (like pystemmer) on Windows.
    """
    def __init__(self, nodes, similarity_top_k=5):
        super().__init__()
        self.nodes = nodes
        self.similarity_top_k = similarity_top_k
        self.k1 = 1.5
        self.b = 0.75
        
        # Tokenize doc corpus
        self.corpus = []
        self.doc_lengths = []
        for node in nodes:
            tokens = self._tokenize(node.text)
            self.corpus.append(tokens)
            self.doc_lengths.append(len(tokens))
            
        self.avg_doc_len = sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 1
        self.doc_count = len(nodes)
        
        # Document frequencies for term matching
        self.df = {}
        for doc in self.corpus:
            for term in set(doc):
                self.df[term] = self.df.get(term, 0) + 1
                
        # Calculate IDF values
        self.idf = {}
        for term, freq in self.df.items():
            self.idf[term] = math.log((self.doc_count - freq + 0.5) / (freq + 0.5) + 1.0)
            
    def _tokenize(self, text):
        return re.findall(r'\b\w+\b', text.lower())
        
    def _retrieve(self, query_bundle):
        query_str = query_bundle.query_str
        query_tokens = self._tokenize(query_str)
        
        scores = []
        for idx, doc_tokens in enumerate(self.corpus):
            score = 0.0
            doc_len = self.doc_lengths[idx]
            
            # Term Frequency in current doc
            tf = {}
            for term in doc_tokens:
                tf[term] = tf.get(term, 0) + 1
                
            for term in query_tokens:
                if term in tf:
                    term_tf = tf[term]
                    numerator = term_tf * (self.k1 + 1)
                    denominator = term_tf + self.k1 * (1 - self.b + self.b * (doc_len / self.avg_doc_len))
                    score += self.idf.get(term, 0.0) * (numerator / denominator)
            
            scores.append((score, self.nodes[idx]))
            
        # Sort and take top K
        scores.sort(key=lambda x: x[0], reverse=True)
        top_k = scores[:self.similarity_top_k]
        
        return [NodeWithScore(node=node, score=score) for score, node in top_k]

def get_nodes_from_chroma(chroma_collection):
    """
    Helper to reconstruct LlamaIndex TextNode objects from raw ChromaDB data.
    This allows us to run BM25 search dynamically on our indexed documents.
    """
    results = chroma_collection.get()
    
    if not results or not results["ids"]:
        return []
        
    nodes = []
    for i in range(len(results["ids"])):
        node_id = results["ids"][i]
        text = results["documents"][i]
        metadata = results["metadatas"][i] if results["metadatas"] else {}
        
        node = TextNode(
            id_=node_id,
            text=text,
            metadata=metadata
        )
        nodes.append(node)
    return nodes

def get_hybrid_query_engine(similarity_top_k=6, rerank=True):
    """
    Constructs a hybrid (Dense + Sparse) search query engine with optional reranking.
    """
    # 1. Initialize Gemini settings
    initialize_settings()
    
    # 2. Connect to local ChromaDB
    chroma_client = chromadb.PersistentClient(path=INDEX_DIR)
    chroma_collection = chroma_client.get_collection("governance_docs")
    
    if chroma_collection.count() == 0:
        print("[ERROR] ChromaDB collection is empty. Run 'python ingest.py' first.")
        sys.exit(1)
        
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    index = VectorStoreIndex.from_vector_store(vector_store)
    
    # 3. Create Dense Vector Retriever
    vector_retriever = index.as_retriever(similarity_top_k=similarity_top_k * 2)
    
    # 4. Create Sparse BM25 Retriever
    # Retrieve nodes from Chroma to initialize BM25
    nodes = get_nodes_from_chroma(chroma_collection)
    bm25_retriever = PurePythonBM25Retriever(
        nodes=nodes,
        similarity_top_k=similarity_top_k * 2
    )
    
    # 5. Build Hybrid Fusion Retriever
    # mode="reciprocal_rerank" merges BM25 and Vector search results using Reciprocal Rank Fusion (RRF)
    hybrid_retriever = QueryFusionRetriever(
        [vector_retriever, bm25_retriever],
        similarity_top_k=similarity_top_k,
        num_queries=1,              # Set to 1 to skip expensive query expansion
        mode="reciprocal_rerank",
        use_async=False
    )
    
    # 6. Node Postprocessors (Reranker & Similarity Filters)
    node_postprocessors = []
    
    # Add Cohere Reranker if key is available in environment
    cohere_api_key = os.getenv("COHERE_API_KEY")
    if rerank and cohere_api_key:
        print("[INFO] Cohere API Key found. Enabling Cohere Reranker...")
        from llama_index.postprocessor.cohere_rerank import CohereRerank
        # Rerank to find the top 4 most matching chunks from the fusion set
        cohere_rerank = CohereRerank(
            api_key=cohere_api_key,
            top_n=4
        )
        node_postprocessors.append(cohere_rerank)
    else:
        if rerank:
            print("[INFO] No COHERE_API_KEY found. Falling back to default RRF ranking.")
            
        # Standard filter: removes chunks with similarity score below threshold (optional)
        # We'll use a mild similarity threshold to remove irrelevant chunks
        similarity_filter = SimilarityPostprocessor(similarity_cutoff=0.0)
        node_postprocessors.append(similarity_filter)
        
    # 7. Construct final query engine
    query_engine = RetrieverQueryEngine.from_args(
        retriever=hybrid_retriever,
        node_postprocessors=node_postprocessors
    )
    
    return query_engine

if __name__ == "__main__":
    # Test retrieval logic
    print("Testing hybrid query engine...")
    try:
        engine = get_hybrid_query_engine(similarity_top_k=4, rerank=False)
        print("[SUCCESS] Hybrid retriever constructed successfully!")
    except Exception as e:
        print(f"[ERROR] Failed to construct retriever: {e}")
        print("Make sure you have indexed documents using 'python ingest.py' first.")
