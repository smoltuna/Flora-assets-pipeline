"""Embedding service — chunks scraped content and stores vector embeddings.

Uses nomic-embed-text-v1.5 via Ollama (768-dim, Matryoshka-capable).
Each source's full text for a flower becomes one chunk (botanical entries are short).
"""
from models import RawSource, SourceEmbedding
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from services.llm.provider import LLMProvider


async def embed_and_store(
    flower_id: int,
    raw_source: RawSource,
    llm: LLMProvider,
    session: AsyncSession,
) -> SourceEmbedding:
    """Embed a single raw source and upsert into source_embeddings."""
    chunk_text = _build_chunk_text(raw_source)
    embedding = await llm.embed(chunk_text)

    # Upsert: delete old embedding for this raw_source if it exists
    existing = await session.execute(
        select(SourceEmbedding).where(SourceEmbedding.raw_source_id == raw_source.id)
    )
    for row in existing.scalars().all():
        await session.delete(row)

    emb = SourceEmbedding(
        raw_source_id=raw_source.id,
        flower_id=flower_id,
        chunk_text=chunk_text,
        embedding=embedding,
        metadata_={"source": raw_source.source},
    )
    session.add(emb)
    await session.commit()
    await session.refresh(emb)
    return emb


async def embed_all_sources(
    flower_id: int,
    llm: LLMProvider,
    session: AsyncSession,
) -> list[SourceEmbedding]:
    """Embed all raw sources for a flower that don't yet have embeddings."""
    result = await session.execute(
        select(RawSource).where(RawSource.flower_id == flower_id)
    )
    sources = result.scalars().all()

    embeddings = []
    for src in sources:
        if not src.raw_content and not src.parsed_content:
            continue
        emb = await embed_and_store(flower_id, src, llm, session)
        embeddings.append(emb)
    return embeddings


def _build_chunk_text(raw_source: RawSource) -> str:
    """Concatenate available text from a raw source into a single chunk."""
    parts: list[str] = []

    if raw_source.raw_content:
        # Trim to first 3000 chars — botanical entries are short, this covers everything
        parts.append(raw_source.raw_content[:3000])

    if raw_source.parsed_content:
        pc = raw_source.parsed_content
        # Flatten relevant structured fields into readable text
        if isinstance(pc, dict):
            for key, val in pc.items():
                if isinstance(val, str) and val.strip():
                    parts.append(f"{key}: {val}")
                elif isinstance(val, list):
                    flat = ", ".join(str(v) for v in val if v)
                    if flat:
                        parts.append(f"{key}: {flat}")

    return "\n".join(parts).strip()
