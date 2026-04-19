"""
GoldyLoopAI - RAG Q&A Pipeline
A simple retrieval-augmented generation pipeline for customer support Q&A.
"""

import os
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.env"))
import json
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def build_rag_prompt(question: str, context: str) -> str:
    """Build a RAG prompt from question and retrieved context."""
    return f"""You are a helpful customer support assistant. Answer the customer's question using ONLY the information provided in the context below. If the context doesn't contain enough information to answer, say so clearly.

Context:
{context}

Customer Question:
{question}

Answer:"""


def run_rag_pipeline(question: str, context: str, model: str = "gpt-4o-mini") -> str:
    """
    Run the RAG pipeline: given a question and context, generate an answer.
    In a real system, context would come from a vector DB (FAISS/Chroma).
    """
    prompt = build_rag_prompt(question, context)

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=300,
    )
    return response.choices[0].message.content.strip()


def run_on_golden_dataset(dataset_path: str, model: str = "gpt-4o-mini") -> list[dict]:
    """
    Run the RAG pipeline on every example in the golden dataset.
    Returns the dataset with actual_output added to each record.
    """
    with open(dataset_path, "r") as f:
        dataset = json.load(f)

    results = []
    for i, item in enumerate(dataset):
        print(f"Running [{i+1}/{len(dataset)}] {item['id']}...")
        actual_output = run_rag_pipeline(item["input"], item["context"], model)
        results.append({**item, "actual_output": actual_output, "model": model})

    return results


if __name__ == "__main__":
    dataset_path = os.path.join(os.path.dirname(__file__), "../data/golden_dataset.json")
    results = run_on_golden_dataset(dataset_path)

    output_path = os.path.join(os.path.dirname(__file__), "../data/pipeline_outputs.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Pipeline outputs saved to {output_path}")
