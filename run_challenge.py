#!/usr/bin/env python3
"""Automated provisioning and Final Challenge evaluation script for Code Assistant Agent."""
import json
import logging
import os
import subprocess
import time
import warnings

warnings.filterwarnings("ignore")
os.environ["TQDM_DISABLE"] = "1"
logging.getLogger("google_adk").setLevel(logging.CRITICAL)

PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT")
if not PROJECT:
    PROJECT = subprocess.check_output(
        ["gcloud", "config", "get-value", "project"], text=True
    ).strip()

LOCATION = "global"
EVAL_LOCATION = "us-central1"
os.environ.update(
    GOOGLE_GENAI_USE_VERTEXAI="TRUE",
    GOOGLE_CLOUD_PROJECT=PROJECT,
    GOOGLE_CLOUD_LOCATION=LOCATION,
)

from code_assistant import vais
from code_assistant.agents import answer_query, build_code_assistant
from code_assistant.data import load_jsonl
from code_assistant.ledger import UsageLedger
from code_assistant.quality import run_quality_eval
from code_assistant.retrieval import build_index
from google.cloud import storage

PRICING = {
    "gemini-3.1-pro-preview": {"input": 2.00, "cached": 0.20, "output": 12.00},
    "gemini-3.5-flash":       {"input": 1.50, "cached": 0.15, "output":  9.00},
    "gemini-3.1-flash-lite":  {"input": 0.25, "cached": 0.025, "output": 1.50},
    "gemini-2.5-pro":         {"input": 1.25, "cached": 0.13, "output": 10.00},
    "gemini-2.5-flash":       {"input": 0.30, "cached": 0.03, "output":  2.50},
    "embedding":  {"per_1m_tokens": 0.15},
    "vais_query": {"per_1k_queries": 1.50},
}


def cost_of(record):
    """USD for one billable event. Thought (reasoning) tokens bill at the output rate."""
    ct = record.get("call_type")
    if ct == "model":
        p = PRICING[record["model"]]
        out = record.get("output_tokens", 0) + record.get("thought_tokens", 0)
        return round(
            (
                record.get("input_tokens", 0) * p["input"]
                + record.get("cached_tokens", 0) * p["cached"]
                + out * p["output"]
            )
            / 1e6,
            12,
        )
    if ct == "embedding":
        return round(
            record.get("embed_tokens", 0) * PRICING["embedding"]["per_1m_tokens"] / 1e6,
            12,
        )
    if ct == "retrieval":
        return round(
            record.get("retrieval_queries", 0)
            * PRICING["vais_query"]["per_1k_queries"]
            / 1e3,
            12,
        )
    return 0.0


def main():
    print(f"Using Google Cloud Project: {PROJECT}")

    # Task 4: Provision Agent Search resources
    vais_location = "global"
    ds_id = "code_assistant-docs"
    engine_id = "code_assistant-docs-engine"
    docs_sc = vais.ensure(
        PROJECT, vais_location, ds_id, engine_id, load_jsonl("assets/docs.jsonl")
    )
    print(f"DocsAgent backend provisioned: {docs_sc}")

    so = load_jsonl("assets/stackoverflow.jsonl")
    holdout = load_jsonl("assets/eval_holdout.jsonl")
    community = build_index(
        so, "gemini-embedding-001", PROJECT, LOCATION, name="community"
    )
    print(f"Indexed {len(so)} Stack Overflow posts into ChromaDB")

    # Task 5: Optimized Final Challenge configuration
    my_models = {
        "coordinator": "gemini-3.5-flash",
        "docs":        "gemini-3.1-flash-lite",
        "community":   "gemini-3.1-flash-lite",
        "embedding":   "gemini-embedding-001",
    }
    my_thinking = "LOW"
    my_topk = 2
    my_maxout = None

    qid = {"v": "_"}
    ledger = UsageLedger(cost_fn=cost_of)
    coord = build_code_assistant(
        models=my_models,
        ledger=ledger,
        qid_holder=qid,
        project=PROJECT,
        location=LOCATION,
        collections={"community": community},
        docs_serving_config=docs_sc,
        top_k=my_topk,
        max_output_tokens=my_maxout,
        thinking_level=my_thinking,
    )

    answers = []
    for item in holdout:
        for attempt in (1, 2):
            try:
                ans = answer_query(coord, item["query"], str(item["id"]), qid)
                if not ans.strip():
                    raise RuntimeError("agent returned an empty answer")
                break
            except Exception as e:
                if attempt == 2:
                    raise
                print(f"[run warning] query {item['id']}: {e}; retrying once", flush=True)
                time.sleep(5)
        answers.append(
            {
                "query": item["query"],
                "answer": ans,
                "reference": item.get("reference", ""),
            }
        )
        print(
            f"Completed query {item['id']} | running cost/query: ${ledger.cost_per_query():.6f}",
            flush=True,
        )

    quality = run_quality_eval(answers, PROJECT, EVAL_LOCATION)
    cost_per_query = ledger.cost_per_query()
    budget = 0.0135
    quality_floor = 0.90
    passed = (cost_per_query <= budget) and (quality >= quality_floor)

    results = {
        "cost_per_query": cost_per_query,
        "quality": quality,
        "budget": budget,
        "quality_floor": quality_floor,
        "passed": passed,
    }
    print("\nFINAL CHALLENGE RESULTS:")
    print(json.dumps(results, indent=2))

    with open("final_metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    if os.path.isdir("/home/jupyter"):
        with open("/home/jupyter/final_metrics.json", "w") as f:
            json.dump(results, f, indent=2)

    result_bucket = f"{PROJECT}-metrics-result"
    client = storage.Client(project=PROJECT)
    bucket = client.bucket(result_bucket)
    blob = bucket.blob("results/final_metrics.json")
    blob.upload_from_filename("final_metrics.json")
    print(f"Uploaded metrics to gs://{result_bucket}/results/final_metrics.json")


if __name__ == "__main__":
    main()
