"""
GoldyLoopAI - Golden Dataset Builder
Utilities to:
  1. Generate synthetic "silver" Q&A pairs from a knowledge base
  2. Validate dataset quality (check for duplicates, missing fields)
  3. Promote silver → gold with human review simulation
"""

import os
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.env"))
import json
import uuid
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

REQUIRED_FIELDS = ["id", "input", "context", "expected_output", "metadata"]
REQUIRED_METADATA = ["difficulty", "scenario_tag", "source", "risk_level"]


SILVER_GENERATION_PROMPT = """You are building a golden evaluation dataset for a customer support AI.

Given the following knowledge base snippet, generate {num_pairs} diverse Q&A pairs.
Include a mix of easy, medium, and hard questions. Hard questions should be edge cases or adversarial.

Knowledge Base:
{knowledge_base}

Return a JSON array of objects with this exact structure:
[
  {{
    "input": "<customer question>",
    "context": "<relevant excerpt from knowledge base>",
    "expected_output": "<ideal support agent answer>",
    "metadata": {{
      "difficulty": "<easy|medium|hard>",
      "scenario_tag": "<topic tag>",
      "source": "synthetic",
      "risk_level": "<low|medium|high>"
    }}
  }}
]

Return ONLY the JSON array, no markdown."""


def generate_silver_pairs(knowledge_base: str, num_pairs: int = 5, model: str = "gpt-4o") -> list[dict]:
    """Generate synthetic Q&A pairs from a knowledge base snippet."""
    prompt = SILVER_GENERATION_PROMPT.format(
        knowledge_base=knowledge_base,
        num_pairs=num_pairs
    )
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=2000,
    )
    raw = response.choices[0].message.content.strip()
    pairs = json.loads(raw)

    # Assign unique IDs
    for pair in pairs:
        pair["id"] = f"SYN-{str(uuid.uuid4())[:8].upper()}"

    return pairs


def validate_dataset(dataset: list[dict]) -> dict:
    """
    Validate the golden dataset for required fields, duplicates, and coverage.
    Returns a validation report.
    """
    errors = []
    warnings = []

    # Check required fields
    for i, item in enumerate(dataset):
        for field in REQUIRED_FIELDS:
            if field not in item:
                errors.append(f"Item {i} missing required field: '{field}'")
        if "metadata" in item:
            for mf in REQUIRED_METADATA:
                if mf not in item.get("metadata", {}):
                    errors.append(f"Item {i} missing metadata field: '{mf}'")

    # Check for duplicate inputs
    inputs = [item.get("input", "") for item in dataset]
    seen = set()
    for inp in inputs:
        if inp in seen:
            warnings.append(f"Duplicate input detected: '{inp[:60]}...'")
        seen.add(inp)

    # Check difficulty distribution
    difficulties = [item.get("metadata", {}).get("difficulty") for item in dataset]
    dist = {d: difficulties.count(d) for d in ["easy", "medium", "hard"]}
    if dist.get("hard", 0) == 0:
        warnings.append("No 'hard' examples — consider adding adversarial edge cases")
    if dist.get("easy", 0) < 2:
        warnings.append("Very few 'easy' examples — add baseline sanity checks")

    # Check scenario coverage
    tags = set(item.get("metadata", {}).get("scenario_tag") for item in dataset)

    return {
        "total_items": len(dataset),
        "errors": errors,
        "warnings": warnings,
        "is_valid": len(errors) == 0,
        "difficulty_distribution": dist,
        "scenario_tags": sorted(list(tags)),
        "source_types": list(set(
            item.get("metadata", {}).get("source") for item in dataset
        )),
    }


def promote_to_gold(silver_pairs: list[dict], existing_dataset_path: str) -> list[dict]:
    """
    Merge validated silver pairs into the existing golden dataset.
    In production, this step would involve human review.
    """
    with open(existing_dataset_path, "r") as f:
        golden = json.load(f)

    existing_inputs = {item["input"] for item in golden}
    added = 0
    for pair in silver_pairs:
        if pair["input"] not in existing_inputs:
            golden.append(pair)
            added += 1

    print(f"✅ Promoted {added} new examples to gold dataset (skipped {len(silver_pairs) - added} duplicates)")
    return golden


if __name__ == "__main__":
    # Example: validate the existing golden dataset
    dataset_path = os.path.join(os.path.dirname(__file__), "../data/golden_dataset.json")
    with open(dataset_path, "r") as f:
        dataset = json.load(f)

    report = validate_dataset(dataset)
    print("\n📋 Dataset Validation Report")
    print(f"   Total items   : {report['total_items']}")
    print(f"   Valid         : {report['is_valid']}")
    print(f"   Difficulties  : {report['difficulty_distribution']}")
    print(f"   Scenario tags : {report['scenario_tags']}")
    if report["errors"]:
        print(f"   ❌ Errors: {report['errors']}")
    if report["warnings"]:
        print(f"   ⚠️  Warnings: {report['warnings']}")

    report_path = os.path.join(os.path.dirname(__file__), "../data/validation_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n   Report saved to {report_path}")
