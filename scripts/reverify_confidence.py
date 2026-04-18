"""Re-run Self-RAG verification on existing enriched flowers.

Uses already-stored source chunks and synthesized field values to compute
fresh confidence_scores — no scraping or re-synthesis needed.

Usage:
    DATABASE_URL=... OLLAMA_BASE_URL=... python scripts/reverify_confidence.py [flower_id ...]
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from sqlalchemy import select
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

from config import settings
from models import Flower, SourceEmbedding
from services.llm.provider import get_provider
from services.rag.synthesizer import NOT_AVAILABLE
from services.rag.verifier import verify_all_fields

TEXT_FIELDS = ["description", "fun_fact", "wiki_description", "habitat", "etymology", "cultural_info"]


async def reverify(flower_ids: list[int]) -> None:
    engine = create_async_engine(settings.database_url)
    Session = async_sessionmaker(engine, expire_on_commit=False)
    llm = get_provider()

    async with Session() as session:
        for fid in flower_ids:
            flower = await session.get(Flower, fid)
            if not flower:
                print(f"[{fid}] not found — skipping")
                continue

            # Gather source text from stored chunks
            result = await session.execute(
                select(SourceEmbedding.chunk_text).where(SourceEmbedding.flower_id == fid)
            )
            chunks = result.scalars().all()
            if not chunks:
                print(f"[{fid}] {flower.latin_name}: no source chunks — skipping")
                continue

            source_text = "\n\n".join(chunks)

            # Only verify fields that have real content
            fields_to_verify = {
                f: getattr(flower, f)
                for f in TEXT_FIELDS
                if getattr(flower, f) and getattr(flower, f) != NOT_AVAILABLE
            }

            if not fields_to_verify:
                print(f"[{fid}] {flower.latin_name}: no verifiable fields — skipping")
                continue

            print(f"[{fid}] {flower.latin_name}: verifying {list(fields_to_verify)} ...")
            verification_results = await verify_all_fields(fields_to_verify, source_text, llm)

            confidence_scores = {
                field: {"llm_score": res.confidence}
                for field, res in verification_results.items()
            }

            flower.confidence_scores = confidence_scores
            await session.commit()

            print(f"[{fid}] {flower.latin_name}: done")
            for field, scores in confidence_scores.items():
                print(f"       {field}: {scores['llm_score']:.2f}")

    await engine.dispose()


if __name__ == "__main__":
    ids = [int(x) for x in sys.argv[1:]] if len(sys.argv) > 1 else [1, 2, 3]
    asyncio.run(reverify(ids))
