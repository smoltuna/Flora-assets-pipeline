"""MLflow experiment: compare chunking strategies for botanical source text.

Tests three strategies for splitting raw scraped content before embedding:
  - whole_doc: embed entire source as one chunk
  - sentence: split on sentence boundaries
  - paragraph: split on blank-line paragraph boundaries

Measures retrieval recall@5 on the hand-labeled query set from embedding_eval.py.
Results are logged to the 'chunk-strategy-eval' MLflow experiment.

Usage:
    uv run python experiments/chunk_strategy.py

Prerequisites:
    - MLflow server running (docker-compose up mlflow)
    - Ollama running with nomic-embed-text pulled
    - raw_sources table populated (seed + scrape first)
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Callable

import mlflow
import numpy as np
from sqlalchemy import create_engine, text

MLFLOW_URI = "http://localhost:5000"
DB_URL = "postgresql://flora:flora@localhost:5432/flora"
EMBED_MODEL = "nomic-embed-text"

EVAL_QUERIES = [
    {
        "query": "hedgerow shrub with red hips used in herbal medicine",
        "relevant": {"Rosa canina"},
    },
    {
        "query": "aromatic Mediterranean herb purple flowers drought tolerant",
        "relevant": {"Lavandula angustifolia"},
    },
    {
        "query": "tall biennial purple spikes cardiac glycoside toxic",
        "relevant": {"Digitalis purpurea"},
    },
    {
        "query": "spring ephemeral woodland bulb white flowers",
        "relevant": {"Galanthus nivalis", "Anemone nemorosa"},
    },
    {
        "query": "traditional daisy lawn weed anti-inflammatory",
        "relevant": {"Bellis perennis"},
    },
]


# ── Chunking strategies ────────────────────────────────────────────────────────

def chunk_whole_doc(text: str) -> list[str]:
    return [text.strip()] if text.strip() else []


def chunk_paragraph(text: str, min_chars: int = 80) -> list[str]:
    paras = re.split(r"\n{2,}", text)
    return [p.strip() for p in paras if len(p.strip()) >= min_chars]


def chunk_sentence(text: str, min_chars: int = 40) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks, current = [], ""
    for sent in sentences:
        current = (current + " " + sent).strip()
        if len(current) >= min_chars:
            chunks.append(current)
            current = ""
    if current:
        chunks.append(current)
    return chunks


STRATEGIES: dict[str, Callable[[str], list[str]]] = {
    "whole_doc": chunk_whole_doc,
    "paragraph": chunk_paragraph,
    "sentence": chunk_sentence,
}


# ── Embedding helpers ──────────────────────────────────────────────────────────

def embed(text: str) -> list[float]:
    import httpx
    resp = httpx.post(
        "http://localhost:11434/api/embeddings",
        json={"model": EMBED_MODEL, "prompt": text},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["embedding"]


def cosine_sim(a: list[float], b: list[float]) -> float:
    va, vb = np.array(a), np.array(b)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    return float(np.dot(va, vb) / denom) if denom > 0 else 0.0


# ── Retrieval ──────────────────────────────────────────────────────────────────

@dataclass
class IndexedChunk:
    latin_name: str
    chunk_text: str
    embedding: list[float] = field(default_factory=list)


def build_index(engine, chunker: Callable[[str], list[str]]) -> list[IndexedChunk]:
    """Chunk all raw sources with the given strategy and embed them in-memory."""
    sql = text("SELECT f.latin_name, rs.raw_content FROM raw_sources rs JOIN flowers f ON rs.flower_id = f.id WHERE rs.raw_content IS NOT NULL")
    with engine.connect() as conn:
        rows = conn.execute(sql).fetchall()

    index: list[IndexedChunk] = []
    for latin_name, raw_content in rows:
        chunks = chunker(raw_content)
        for chunk_text in chunks:
            emb = embed(chunk_text)
            index.append(IndexedChunk(latin_name=latin_name, chunk_text=chunk_text, embedding=emb))
    return index


def retrieve(query_emb: list[float], index: list[IndexedChunk], k: int = 5) -> list[str]:
    scored = [(cosine_sim(query_emb, c.embedding), c.latin_name) for c in index]
    scored.sort(key=lambda x: x[0], reverse=True)
    seen, result = set(), []
    for _, name in scored:
        if name not in seen:
            seen.add(name)
            result.append(name)
        if len(result) >= k:
            break
    return result


def recall_at_k(retrieved: list[str], relevant: set[str], k: int = 5) -> float:
    hits = sum(1 for r in retrieved[:k] if r in relevant)
    return hits / len(relevant) if relevant else 0.0


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("chunk-strategy-eval")
    engine = create_engine(DB_URL)

    for strategy_name, chunker in STRATEGIES.items():
        print(f"\nStrategy: {strategy_name}")
        with mlflow.start_run(run_name=strategy_name):
            mlflow.log_param("strategy", strategy_name)
            mlflow.log_param("embed_model", EMBED_MODEL)

            t0 = time.perf_counter()
            index = build_index(engine, chunker)
            build_time = time.perf_counter() - t0

            mlflow.log_metric("index_size", len(index))
            mlflow.log_metric("index_build_s", build_time)
            print(f"  Index size: {len(index)} chunks, built in {build_time:.1f}s")

            recalls = []
            for i, item in enumerate(EVAL_QUERIES):
                query_emb = embed(item["query"])
                retrieved = retrieve(query_emb, index)
                r = recall_at_k(retrieved, item["relevant"])
                recalls.append(r)
                mlflow.log_metric(f"recall_q{i}", r)
                print(f"  [{r:.2f}] {item['query'][:55]}")

            mlflow.log_metric("mean_recall_at_5", float(np.mean(recalls)))
            print(f"  → mean recall@5: {np.mean(recalls):.3f}")


if __name__ == "__main__":
    main()
