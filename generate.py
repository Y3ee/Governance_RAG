import os
import sys
from llama_index.core import PromptTemplate, Settings
from llama_index.llms.google_genai import GoogleGenAI

# Import configuration and retrieve engine
from config import GEMINI_API_KEY, LLM_MODEL
from retrieve import get_hybrid_query_engine

# Custom text QA template to enforce formatting and citations
QA_PROMPT_TMPL = (
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context information and not prior knowledge, answer the query.\n"
    "Strict Guidelines:\n"
    "1. Base your answer ONLY on the provided context. If the answer cannot be found in the context, output: "
    "'I cannot find the answer in the provided documents.'\n"
    "2. For every factual claim, rule, or direct quote you write, you MUST cite the source document and page number in brackets. "
    "Use the exact format: [Document: <file_name>, Page: <page_label>].\n"
    "3. Keep your response clear, structured, and professional.\n\n"
    "Query: {query_str}\n"
    "Answer: "
)

def verify_answer_groundedness(context_str, query, answer):
    """
    An LLM-as-a-judge hallucination guardrail.
    Asks Gemini to check if the generated answer is strictly grounded in the retrieved chunks.
    """
    # Check if we defaulted to 'cannot find answer'
    if "cannot find the answer" in answer.lower():
        return True # Safe, no hallucination risk
        
    guardrail_prompt = f"""
You are a strict compliance auditor. Your job is to verify if a Q&A assistant's answer is 100% grounded in the provided context, without any assumptions or outside knowledge.

Context:
{context_str}

Query: {query}
Answer: {answer}

Verify: Does the Answer contain any statements, names, numbers, or facts that are NOT directly supported by the Context?
If the answer is fully supported, reply with exactly: YES
If the answer contains any hallucination, speculation, or unsupported claims, reply with exactly: NO

Verdict (YES or NO):"""
    
    try:
        # Use the global Gemini model instance
        response = Settings.llm.complete(guardrail_prompt)
        verdict = response.text.strip().upper()
        return "YES" in verdict
    except Exception as e:
        print(f"[WARNING] Guardrail check failed to execute: {e}")
        return True # Fallback to true if API errors out

def generate_grounded_answer(query, query_engine):
    """
    Queries the hybrid retrieval engine, generates the answer with citations,
    and runs a compliance guardrail check to prevent hallucination.
    """
    # Set the custom QA prompt template on the query engine
    qa_prompt = PromptTemplate(QA_PROMPT_TMPL)
    query_engine.update_prompts({"response_synthesizer:text_qa_template": qa_prompt})
    
    # Run the query
    print(f"Retrieving and generating for query: '{query}'...")
    response = query_engine.query(query)
    
    # Reconstruct context string from retrieved chunks for the guardrail
    context_chunks = []
    for source in response.source_nodes:
        file_name = source.node.metadata.get("file_name", "Unknown")
        page = source.node.metadata.get("page_label", "Unknown")
        text = source.node.text
        context_chunks.append(f"--- [Source Document: {file_name}, Page: {page}] ---\n{text}")
    
    full_context_str = "\n\n".join(context_chunks)
    
    # Run guardrail verification
    is_grounded = verify_answer_groundedness(full_context_str, query, response.response)
    
    return {
        "answer": response.response,
        "is_grounded": is_grounded,
        "sources": [
            {
                "file_name": node.node.metadata.get("file_name", "Unknown"),
                "page": node.node.metadata.get("page_label", "Unknown"),
                "text": node.node.text,
                "score": node.score
            }
            for node in response.source_nodes
        ]
    }

if __name__ == "__main__":
    # Test generation logic
    engine = get_hybrid_query_engine(similarity_top_k=3, rerank=False)
    test_query = "What is the policy?"
    
    try:
        result = generate_grounded_answer(test_query, engine)
        print("\n=== Answer ===")
        print(result["answer"])
        print(f"Is Grounded: {result['is_grounded']}")
    except Exception as e:
        print(f"[ERROR] Ingestion index empty or API key missing. Test failed: {e}")
