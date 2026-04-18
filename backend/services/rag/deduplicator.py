"""Semantic deduplication across scraped sources.

Clusters near-duplicate chunks (cosine similarity >= threshold) and keeps
the richest representative per cluster. Runs before RAG synthesis to keep
the context window lean and avoid the LLM over-weighting repeated claims.
"""
from __future__ import annotations

import numpy as np

from services.rag.retriever import RetrievedChunk

_STRUCTURED_SOURCES = {"pfaf", "wikidata", "gbif"}


def cosine_sim(a: list[float], b: list[float]) -> float:
    va, vb = np.array(a, dtype=float), np.array(b, dtype=float)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    if denom == 0:
        return 0.0
    return float(np.dot(va, vb) / denom)


def deduplicate_chunks(
    chunks: list[RetrievedChunk],
    similarity_threshold: float = 0.92,
) -> list[RetrievedChunk]:
    """Return deduplicated chunks, keeping richest representative per cluster.

    Richness heuristic: longer text wins; structured sources (pfaf, wikidata)
    break ties over prose sources (wikipedia).
    """
    if not chunks:
        return []

    kept: list[RetrievedChunk] = []
    used: set[int] = set()

    for i, chunk_a in enumerate(chunks):
        if i in used:
            continue
        cluster = [chunk_a]
        for j in range(i + 1, len(chunks)):
            if j in used:
                continue
            if not chunk_a.embedding or not chunks[j].embedding:
                continue
            sim = cosine_sim(chunk_a.embedding, chunks[j].embedding)
            if sim >= similarity_threshold:
                cluster.append(chunks[j])
                used.add(j)

        best = max(
            cluster,
            key=lambda c: (
                len(c.chunk_text),
                1 if c.source in _STRUCTURED_SOURCES else 0,
            ),
        )
        kept.append(best)

    return kept
