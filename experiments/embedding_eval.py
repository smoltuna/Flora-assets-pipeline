"""MLflow experiment: compare embedding models on Flora botanical corpus.

Evaluates recall@5 for candidate embedding models against a hand-labeled query set.
Results are logged to the 'embedding-model-eval' MLflow experiment.

Usage:
    uv run python experiments/embedding_eval.py

Prerequisites:
    - MLflow server running (docker-compose up mlflow)
    - Ollama running with candidate models pulled
    - PostgreSQL with embeddings populated (run_batch.py first)
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import mlflow
import numpy as np
from sqlalchemy import create_engine, text

MLFLOW_URI = "http://localhost:5000"
DB_URL = "postgresql://flora:flora@localhost:5432/flora"

# Hand-labeled evaluation queries with expected relevant latin names
EVAL_QUERIES = [
    {
        "query": "hedgerow shrub with red hips used in herbal medicine",
        "relevant": {"Rosa canina", "Rosa rubiginosa"},
    },
    {
        "query": "alpine meadow wildflower with yellow petals",
        "relevant": {"Ranunculus acris", "Helianthus annuus"},
    },
    {
        "query": "spring bulb white nodding flowers toxic to livestock",
        "relevant": {"Galanthus nivalis", "Narcissus pseudonarcissus"},
    },
    {
        "query": "aromatic Mediterranean herb purple flowers drought tolerant",
        "relevant": {"Lavandula angustifolia", "Rosmarinus officinalis"},
    },
    {
        "query": "tall biennial purple spikes cardiac glycoside toxic",
        "relevant": {"Digitalis purpurea"},
    },
]

CANDIDATE_MODELS = [
    "nomic-embed-text",
    "mxbai-embed-large",
]


@dataclass
class RetrievalResult:
    model: str
    query: str
    retrieved: list[str]
    relevant: set[str]

    @property
    def recall_at_5(self) -> float:
        hits = sum(1 for r in self.retrieved[:5] if r in self.relevant)
        return hits / len(self.relevant) if self.relevant else 0.0

    @property
    def precision_at_5(self) -> float:
        hits = sum(1 for r in self.retrieved[:5] if r in self.relevant)
        return hits / 5


def embed_query(model: str, query: str) -> list[float]:
    """Embed a query string using Ollama."""
    import httpx
    resp = httpx.post(
        "http://localhost:11434/api/embeddings",
        json={"model": model, "prompt": query},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["embedding"]


def retrieve_top_k(engine, query_embedding: list[float], k: int = 5) -> list[str]:
    """Return top-k latin names by cosine similarity."""
    emb_str = "[" + ",".join(str(x) for x in query_embedding) + "]"
    sql = text("""
        SELECT f.latin_name
        FROM source_embeddings se
        JOIN flowers f ON se.flower_id = f.id
        ORDER BY se.embedding <=> CAST(:emb AS vector)
        LIMIT :k
    """)
    with engine.connect() as conn:
        rows = conn.execute(sql, {"emb": emb_str, "k": k}).fetchall()
    seen: list[str] = []
    for row in rows:
        if row[0] not in seen:
            seen.append(row[0])
    return seen


def run_eval(model: str, engine) -> list[RetrievalResult]:
    results = []
    for item in EVAL_QUERIES:
        embedding = embed_query(model, item["query"])
        retrieved = retrieve_top_k(engine, embedding)
        results.append(RetrievalResult(
            model=model,
            query=item["query"],
            retrieved=retrieved,
            relevant=item["relevant"],
        ))
    return results


def main():
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("embedding-model-eval")

    engine = create_engine(DB_URL)

    for model in CANDIDATE_MODELS:
        print(f"\nEvaluating model: {model}")
        with mlflow.start_run(run_name=model):
            mlflow.log_param("model", model)
            mlflow.log_param("eval_queries", len(EVAL_QUERIES))
            mlflow.log_param("k", 5)

            t0 = time.perf_counter()
            results = run_eval(model, engine)
            elapsed = time.perf_counter() - t0

            recall_scores = [r.recall_at_5 for r in results]
            precision_scores = [r.precision_at_5 for r in results]

            mlflow.log_metrics({
                "mean_recall_at_5": float(np.mean(recall_scores)),
                "mean_precision_at_5": float(np.mean(precision_scores)),
                "eval_duration_s": elapsed,
            })

            for i, result in enumerate(results):
                mlflow.log_metric(f"recall_q{i}", result.recall_at_5)
                print(f"  [{result.recall_at_5:.2f}] {result.query[:60]}")

            print(f"  → mean recall@5: {np.mean(recall_scores):.3f}")


if __name__ == "__main__":
    main()
