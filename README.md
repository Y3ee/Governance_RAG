# Governance Document Hybrid RAG & Auditing Portal

This repository contains a production-grade, highly grounded Question & Answering (RAG) system built over governance and policy documents. It combines dense semantic retrieval with sparse keyword searches, incorporates reranking, dynamically maps citations, and utilizes a self-compliance guardrail to prevent hallucinations.

## Key Architectural Decisions

1. **Hybrid Retrieval (Dense + Sparse)**:
   - **Dense (ChromaDB + Gemini `text-embedding-004`)**: Captures semantic concepts and contextual meaning.
   - **Sparse (BM25)**: Ensures exact keyword matches (essential for specific policy clauses, IDs, and section numbers).
   - Combined using **Reciprocal Rank Fusion (RRF)** to get the best of both.
2. **Metadata Enrichment & Filtering**:
   - Chunks are enriched with source document names and page numbers.
   - Allows users to narrow down searches (e.g., searching only within a specific HR policy).
3. **Automated Citation Enforcer**:
   - The LLM prompt forces citation tracking. Factual assertions must map to a `[Document: <name>, Page: <number>]` citation.
4. **Self-Auditing Grounding Guardrail**:
   - An LLM-as-a-judge validator reviews the answer against raw context. If it detects outside assumptions or hallucinations, it flags a warning badge in the UI.
5. **Reranking Fallback**:
   - Integrates Cohere Reranking if an API key is available. Falls back to default RRF ranking dynamically without crashing.

---

## Folder Structure

```text
governance_rag/
├── data/                  # Place PDF/TXT/MD governance documents here
├── index/                 # Local ChromaDB vector database directory
├── requirements.txt       # Python package dependencies
├── config.py              # Configurations (hyperparameters, model configurations)
├── ingest.py              # Loads, chunks, extracts metadata, and indexes documents
├── retrieve.py            # Custom Hybrid + RRF + Rerank retriever
├── generate.py            # Gemini text synthesis and compliance guardrail checks
├── app.py                 # Streamlit UI dashboard
└── evaluate.py            # Automated evaluation script using a Golden Dataset
```

---

## Setup & Running Guide

### 1. Prerequisites
Ensure you have Python 3.10+ installed.

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure API Credentials
Create a `.env` file in the root directory:
```env
GEMINI_API_KEY=your_google_ai_studio_api_key
COHERE_API_KEY=your_cohere_api_key_here (optional, for reranking)
```

### 4. Load Governance Documents
Place your governance documents (PDFs, TXTs, or MDs) inside the `data/` directory.

### 5. Index the Documents
Run the ingestion script to parse your documents and save embeddings locally:
```bash
python ingest.py
```

### 6. Run the Streamlit Interface
Launch the interactive web portal:
```bash
streamlit run app.py
```

### 7. Run Compliance Evaluation
Evaluate the pipeline against a preset test set:
```bash
python evaluate.py
```
This runs your Golden Dataset and outputs an auditing score and rationale to `eval_report.md`.
