import os
import sys
import json
from llama_index.core import Settings

# Import configuration, retrieval, and generation code
from config import GEMINI_API_KEY
from retrieve import get_hybrid_query_engine
from generate import generate_grounded_answer

# Define a small Golden Dataset of mock/real governance questions.
# You should customize this with questions and ground truths specific to the documents you load!
GOLDEN_DATASET = [
    {
        "question": "What is the primary purpose of the CREATE AI Act of 2023?",
        "expected_criteria": "The answer should state that it establishes the National Artificial Intelligence Research Resource (NAIRR) to democratize access to AI resources and enhance U.S. AI research capacity."
    },
    {
        "question": "Who is responsible for overseeing the NAIRR governance according to the CREATE AI Act of 2023?",
        "expected_criteria": "The answer should mention the NAIRR Steering Subcommittee chaired by the Director of the Office of Science and Technology Policy."
    },
    {
        "question": "What areas does the Chinese Cybersecurity Technology Draft Standard for Generative AI Services cover?",
        "expected_criteria": "The answer should mention that it covers safety requirements for training data security, model safety, safety measures, and personal information protection."
    }
]

def evaluate_with_llm_judge(question, expected_criteria, generated_answer):
    """
    Uses Gemini LLM as a judge to grade the generated answer based on grounding and citation presence.
    """
    judge_prompt = f"""
You are an expert compliance and RAG quality auditor. Your job is to grade the quality and accuracy of a RAG system's generated answer against a set of expected criteria.

Question asked by user:
{question}

Expected answer criteria:
{expected_criteria}

RAG system generated answer:
{generated_answer}

Rate the generated answer on a scale from 1 to 5 based on these rules:
- 5 (Excellent): The answer is fully correct, aligns with the criteria, contains explicit document/page citations, and contains zero hallucinations.
- 4 (Good): The answer is correct and cited, but is missing minor context or could be structured better.
- 3 (Average): The answer is partially correct, but lacks references/citations, or contains slightly ambiguous phrasing.
- 2 (Poor): The answer is mostly incorrect, misses critical criteria points, or fails to provide citations.
- 1 (Critical Failure): The answer is completely wrong, hallucinated, or states it cannot find the answer when it should have.

Provide your feedback in the following JSON format:
{{
  "score": <integer from 1 to 5>,
  "reason": "<detailed explanation of why this score was given, highlighting strengths and weaknesses>"
}}

JSON response:"""
    
    try:
        response = Settings.llm.complete(judge_prompt)
        # Parse the JSON response
        text = response.text.strip()
        # Clean markdown code block wraps if LLM adds them
        if text.startswith("```json"):
            text = text[7:]
        if text.endswith("```"):
            text = text[:-3]
        
        data = json.loads(text.strip())
        return data["score"], data["reason"]
    except Exception as e:
        print(f"[WARNING] Could not parse LLM judge response: {e}. Raw: {response.text if 'response' in locals() else 'None'}")
        return 3, "Failed to run LLM judge cleanly; default score of 3 applied."

def run_evaluation_suite():
    """
    Iterates through the Golden Dataset, executes the RAG pipeline,
    and runs the judge LLM to generate an audit report.
    """
    if not GEMINI_API_KEY:
        print("[ERROR] GEMINI_API_KEY is not set. Cannot run evaluation.")
        sys.exit(1)

    print("Initializing Hybrid Retrieval engine...")
    try:
        query_engine = get_hybrid_query_engine(similarity_top_k=4, rerank=False)
    except Exception as e:
        print(f"[ERROR] Could not load database index: {e}")
        print("Please load documents into the 'data' directory and run 'python ingest.py' first.")
        return

    print(f"\nStarting evaluation on {len(GOLDEN_DATASET)} golden questions...")
    
    total_score = 0
    results = []
    
    for idx, item in enumerate(GOLDEN_DATASET):
        question = item["question"]
        criteria = item["expected_criteria"]
        
        print(f"\n[{idx+1}/{len(GOLDEN_DATASET)}] Querying: '{question}'...")
        rag_output = generate_grounded_answer(question, query_engine)
        
        print(f"[{idx+1}/{len(GOLDEN_DATASET)}] Running LLM compliance judge...")
        score, reason = evaluate_with_llm_judge(question, criteria, rag_output["answer"])
        
        total_score += score
        
        results.append({
            "question": question,
            "expected_criteria": criteria,
            "answer": rag_output["answer"],
            "is_grounded": rag_output["is_grounded"],
            "score": score,
            "reason": reason,
            "num_sources": len(rag_output["sources"])
        })
        
        print(f"Result: Score = {score}/5 | Grounded = {rag_output['is_grounded']}")
        
    avg_score = total_score / len(GOLDEN_DATASET)
    print(f"\n======================================")
    print(f"Evaluation Complete! Average Score: {avg_score:.2f} / 5.0")
    print(f"======================================")
    
    # Save the report as markdown
    report_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_report.md")
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"# RAG Compliance Evaluation Report\n\n")
        f.write(f"### Performance Metric Summary\n")
        f.write(f"- **Average LLM Judge Score:** `{avg_score:.2f} / 5.0`\n")
        f.write(f"- **Total Queries Checked:** `{len(GOLDEN_DATASET)}`\n\n")
        f.write(f"--- \n\n")
        
        for idx, res in enumerate(results):
            f.write(f"## {idx+1}. {res['question']}\n\n")
            f.write(f"**Target Evaluation Criteria:**\n> {res['expected_criteria']}\n\n")
            f.write(f"**Generated Answer:**\n{res['answer']}\n\n")
            f.write(f"**Grounding Verification Verdict:** `{'PASS' if res['is_grounded'] else 'FAIL'}`\n\n")
            f.write(f"**Audit Score:** `{res['score']} / 5`\n\n")
            f.write(f"**Auditor Rationale:**\n{res['reason']}\n\n")
            f.write(f"**Sources retrieved:** {res['num_sources']} chunks\n\n")
            f.write(f"--- \n\n")
            
    print(f"Saved audit report to: {report_path}")

if __name__ == "__main__":
    run_evaluation_suite()
