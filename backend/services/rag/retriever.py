"""Hybrid BM25 + dense vector retrieval using Reciprocal Rank Fusion.

The hybrid_search SQL function (defined in the DB migration) combines:
  - Dense: HNSW cosine similarity on pgvector embeddings
  - Sparse: tsvector BM25 via ts_rank
  - Fusion: Reciprocal Rank Fusion (RRF) merging both ranked lists
"""
from dataclasses import dataclass

from models import SourceEmbedding
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from services.llm.provider import LLMProvider


@dataclass
class RetrievedChunk:
    chunk_id: int
    chunk_text: str
    source: str
    rrf_score: float
    embedding: list[float]


async def retrieve_for_flower(
    flower_id: int,
    session: AsyncSession,
) -> list[RetrievedChunk]:
    """Return all embedded chunks for a specific flower (deterministic filter)."""
    result = await session.execute(
        select(SourceEmbedding).where(SourceEmbedding.flower_id == flower_id)
    )
    rows = result.scalars().all()
    return [
        RetrievedChunk(
            chunk_id=row.id,
            chunk_text=row.chunk_text,
            source=(row.metadata_ or {}).get("source", "unknown"),
            rrf_score=1.0,
            embedding=list(row.embedding) if row.embedding is not None else [],
        )
        for row in rows
    ]


async def hybrid_search(
    query_text: str,
    llm: LLMProvider,
    session: AsyncSession,
    match_count: int = 10,
    rrf_k: int = 60,
) -> list[RetrievedChunk]:
    """Cross-flower hybrid search — used for gap detection and cross-flower queries."""
    query_embedding = await llm.embed(query_text)

    # Call the SQL hybrid_search function (defined in migration)
    rows = await session.execute(
        text("""
            SELECT chunk_id, chunk_text, rrf_score
            FROM hybrid_search(:query_text, :query_embedding::vector, :match_count, :rrf_k)
        """),
        {
            "query_text": query_text,
            "query_embedding": str(query_embedding),
            "match_count": match_count,
            "rrf_k": rrf_k,
        },
    )

    chunks = []
    for row in rows:
        # Fetch embedding separately for gap detection
        emb_row = await session.execute(
            select(SourceEmbedding).where(SourceEmbedding.id == row.chunk_id)
        )
        emb_obj = emb_row.scalar_one_or_none()
        chunks.append(RetrievedChunk(
            chunk_id=row.chunk_id,
            chunk_text=row.chunk_text,
            source="unknown",
            rrf_score=row.rrf_score,
            embedding=list(emb_obj.embedding) if emb_obj and emb_obj.embedding is not None else [],
        ))
    return chunks
