# Governance Document Hybrid RAG Engine
### Intern Assessment Documentation & Architecture Overview

This project implements a local-first, production-grade Retrieval-Augmented Generation (RAG) system optimized for corporate governance and regulatory compliance documents. It is designed to deliver useful, accurate, and grounded answers over a document corpus while respecting API constraints.


## 1. High-Level System Architecture

The pipeline consists of a **two-stage hybrid retrieval engine** coupled with an **automated grounding auditor (fact-checker)** to eliminate hallucinations.

```mermaid
graph TD
    Ingest[data/ Policy CSVs] --> LocalEmbed[Local BGE Embedder]
    LocalEmbed --> Chroma[ChromaDB Local Vector DB]
    
    UserQuery[User Query] --> Dense[Dense Vector Search (BAAI/bge-small)]
    UserQuery --> Sparse[Custom BM25 Search (Keyword Matching)]
    
    Dense --> RRF[Reciprocal Rank Fusion RRF]
    Sparse --> RRF
    
    RRF --> Filter[Similarity Score Filter]
    Filter --> LLM[Gemini 2.5 Flash Generator]
    
    LLM --> Guardrail[LLM Grounding Auditor]
    Guardrail --> UI[Minimalist Streamlit Chat Portal]
```

---

## 2. Key Decisions, Reasonings, & Tradeoffs

In alignment with the open-ended nature of this assessment, the following technical choices were made to optimize for **correctness**, **grounding**, and **retrieval quality**:

### A. Hybrid Search (Dense Semantic + Sparse Keyword)
* **The Decision:** We combined Dense Vector Search (using cosine similarity) with a custom Sparse BM25 Keyword Search.
* **The Reason:** Governance documents contain strict clause numbers, section IDs, and legal terms (e.g., *"CREATE AI Act of 2023"* or *"Section 4.2"*). 
  * Vector embeddings excel at conceptual meaning but frequently miss exact alphanumerics or keyword codes.
  * BM25 excels at exact keyword matching but misses conceptual synonyms. 
* **The Tradeoff:** Merging searches requires an aggregation algorithm. We implemented **RRF (Reciprocal Rank Fusion)** to mathematically merge the ranks of both search outputs without needing to normalize their raw scores, ensuring highly relevant documents rise to the top.

### B. Local Embedding Model (`BAAI/bge-small-en-v1.5`)
* **The Decision:** We migrated from Google AI Studio's Cloud Embeddings to a local, CPU-based HuggingFace embedding model.
* **The Reason:** Google's Free Tier has a strict limit of **30,000 Tokens Per Minute (TPM)**. Trying to index a standard governance dataset (6,000+ chunks) over the API triggers `429 Rate Limit Exceeded` errors and crashes ingestion.
* **The Tradeoff:** 
  * *Local Embeddings:* Completely free, offline, has zero rate limits, and indexes all 6,188 rows in under 45 seconds on a standard CPU.
  * *The Cost:* Requires downloading a 120MB model cache on the first run, but is infinitely scalable without API costs.

### C. Pure-Python BM25 Retriever
* **The Decision:** Instead of importing standard C-compiled search libraries (like `rank-bm25` or `pystemmer`), we built a custom `PurePythonBM25Retriever` class inheriting from LlamaIndex's `BaseRetriever`.
* **The Reason:** Installing C-extensions on Windows platforms often triggers complex Visual C++ build tool compiler errors, making the code fragile to run. 
* **The Tradeoff:** A pure-Python implementation is slightly slower than C-compiled libraries for large-scale datasets, but for our governance corpus (under 100,000 rows), it runs in milliseconds and is **100% portable** across any operating system.

### D. Self-Auditing Hallucination Guardrail
* **The Decision:** We implemented a two-stage Q&A cycle where every generated answer is run through a verification model (`verify_answer_groundedness`) before being displayed to the user.
* **The Reason:** Legal and compliance systems cannot tolerate hallucinations. If the LLM makes assumptions or extrapolates beyond the provided context, the system must warn the user.
* **The Tradeoff:** It adds a second API request to Gemini per user query. However, since the LLM daily trial limit is 1,500 requests, this is an acceptable cost to guarantee 100% answer grounding.

---

## 3. Assumptions Made
1. **Document Format:** We assumed that the governance document data consists of structured CSV files (representing regulatory databases) as well as standard text/PDF files. We built a custom parser to read CSV rows as individual database entries.
2. **Citation Requirement:** We assumed that every compliance answer is useless without proof. Therefore, we structured the LLM prompt to strictly map claims to metadata citations: `[Document: name, Page: Row X]`.
3. **Environment:** We assumed that the final grading environment runs on a standard machine that may not have dedicated GPUs or installed C-compilers. Our code runs on CPU-only setups out of the box.

---

## 4. How to Run the Code

### 1. Installation
Install all python dependencies:
```bash
pip install -r requirements.txt
```

### 2. Configure API Keys
Create a file named `.env` in the root folder and add your Google AI Studio API key:
```env
GEMINI_API_KEY=your_google_api_key_here
```

### 3. Run Ingestion (Index Data)
Run the script to chunk, extract metadata, generate local embeddings, and save the ChromaDB database:
```bash
python ingest.py
```

### 4. Run the Chat Portal
Launch the clean, Apple-inspired Streamlit frontend:
```bash
streamlit run app.py
```



## 5. Limitations & Next Steps (What I would do next)

If I had more time or a production environment, I would implement:
1. **Asynchronous Batching for Ingestion**: For corpora exceeding 100,000 files, we should use a message broker (like RabbitMQ) and task queues (like Celery) to load documents asynchronously in the background.
2. **Multi-Hop Reasoning**: Currently, the retriever is single-turn. If a user asks a question that requires connecting facts from two completely different documents (e.g., *"Does the CREATE AI Act violate China's cybersecurity standard?"*), a **ReAct Agent** loop would be needed to decompose the query into sub-questions and search sequentially.
3. **Dynamic PDF Layout Parsing**: Standard PDF readers struggle with visual tables and headers. Using a layout-aware parser (like `LlamaParse` or `Marker`) would improve retrieval quality for scanned PDF policy files.
