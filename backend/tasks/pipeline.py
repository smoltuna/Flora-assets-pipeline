"""Sequential pipeline orchestration — runs full enrichment for one flower.

Pipeline stages (in order):
  1. Scrape all sources (PFAF, Wikipedia, Wikidata, GBIF)
  2. Embed all sources → store in source_embeddings
  3. Retrieve all chunks for this flower
  4. Semantic deduplication
  5. Adaptive RAG routing (full / sparse / minimal based on source coverage)
  6. Per-field Corrective RAG grading
  7. LLM synthesis
  8. Self-RAG verification
  9. Persist enriched flower + confidence scores
"""
from __future__ import annotations

import time
from contextlib import nullcontext
from datetime import date

import mlflow
import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from config import settings
from models import Flower, RawSource
from routers.scrape import _do_scrape
from services.llm.provider import LLMProvider, get_provider
from services.rag.deduplicator import deduplicate_chunks
from services.rag.embedder import embed_and_store
from services.rag.grader import grade_retrieval
from services.rag.retriever import RetrievedChunk, retrieve_for_flower
from services.rag.synthesizer import NOT_AVAILABLE, SynthesizedFlower, synthesize
from services.rag.verifier import verify_all_fields
from services.translation.translator import translate_flower

log = structlog.get_logger()

TEXT_FIELDS = ["description", "fun_fact", "wiki_description", "habitat", "etymology", "cultural_info"]


def _mlflow_context(latin_name: str, flower_id: int):
    """Return an MLflow run context manager, or a no-op if server is unavailable."""
    try:
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.set_experiment("flora-enrichment")
        return mlflow.start_run(run_name=f"{latin_name}-{flower_id}")
    except Exception:
        return nullcontext()


def _log_mlflow_metrics(confidence_scores: dict, n_chunks: int, n_deduped: int, elapsed: float) -> None:
    """Log metrics to the active MLflow run; silently skip if no run is active."""
    try:
        flat: dict[str, float] = {
            "pipeline_duration_s": elapsed,
            "chunks_retrieved": float(n_chunks),
            "chunks_after_dedup": float(n_deduped),
        }
        for field, scores in confidence_scores.items():
            flat[f"confidence_llm_{field}"] = scores.get("llm_score", 0.0)
        mlflow.log_metrics(flat)
    except Exception:
        pass


async def run_pipeline(flower_id: int, db: AsyncSession, feature_date: date | None = None) -> Flower:
    """Run the full enrichment pipeline for a single flower. Returns the updated Flower."""
    flower = await db.get(Flower, flower_id)
    if not flower:
        raise ValueError(f"Flower {flower_id} not found")

    log.info("pipeline.start", flower_id=flower_id, latin_name=flower.latin_name)
    start_time = time.perf_counter()

    with _mlflow_context(flower.latin_name, flower_id):
        try:
            mlflow.set_tags({
                "latin_name": flower.latin_name,
                "flower_id": str(flower_id),
                "llm_provider": settings.llm_provider,
            })
        except Exception:
            pass

        # Stage 1: Scrape
        flower.status = "scraping"
        await db.commit()
        scrape_result = await _do_scrape(flower_id, flower.latin_name, db)
        log.info("pipeline.scraped", sources=scrape_result.sources_scraped, failed=scrape_result.sources_failed)

        # Stage 2: Embed
        flower.status = "embedding"
        await db.commit()
        llm: LLMProvider = get_provider()
        await db.refresh(flower)

        sources_result = await db.execute(select(RawSource).where(RawSource.flower_id == flower_id))
        raw_sources = sources_result.scalars().all()
        pfaf_raw_care: dict | None = None
        for src in raw_sources:
            if src.source != "pfaf" or not src.parsed_content:
                continue
            care_info = src.parsed_content.get("care_info")
            if isinstance(care_info, dict) and care_info:
                # Keep original PFAF labels/values as the canonical flower care info.
                pfaf_raw_care = care_info
                break

        embed_success = 0
        embed_failed = 0
        for src in raw_sources:
            if src.raw_content or src.parsed_content:
                try:
                    await embed_and_store(flower_id, src, llm, db)
                    embed_success += 1
                except Exception as e:
                    embed_failed += 1
                    log.warning("pipeline.embed_failed", source=src.source, error=str(e))

        if raw_sources and embed_success == 0:
            flower.status = "failed"
            await db.commit()
            raise RuntimeError(
                "No embeddings were created. Ensure Ollama is running and "
                "OLLAMA_EMBED_MODEL is available."
            )

        if embed_failed:
            log.info("pipeline.embed_summary", succeeded=embed_success, failed=embed_failed)

        # Stage 3: Retrieve
        chunks = await retrieve_for_flower(flower_id, db)
        log.info("pipeline.retrieved", n_chunks=len(chunks))

        if not chunks:
            flower.status = "failed"
            await db.commit()
            raise RuntimeError(
                "No retrieved chunks found after embedding. Pipeline stopped to avoid "
                "saving low-confidence empty enrichment output."
            )

        # Stage 4: Semantic deduplication
        deduped = deduplicate_chunks(chunks)
        log.info("pipeline.deduped", before=len(chunks), after=len(deduped))

        # Stage 5: Adaptive RAG routing
        sources_present = {c.source for c in deduped}
        synthesis_result = await _adaptive_synthesize(flower, deduped, sources_present, llm)

        # Stage 6–8: CRAG grading + synthesis + Self-RAG verification
        # Wikipedia/Wikidata first: less boilerplate, more concise botanical content
        _source_order = {"wikipedia": 0, "wikidata": 1, "gbif": 2, "pfaf": 3}
        verification_chunks = sorted(deduped, key=lambda c: _source_order.get(c.source, 4))
        source_text = "\n\n".join(c.chunk_text for c in verification_chunks)
        generated_fields = {
            f: getattr(synthesis_result, f, NOT_AVAILABLE)
            for f in TEXT_FIELDS
            if getattr(synthesis_result, f, NOT_AVAILABLE) != NOT_AVAILABLE
        }

        for field_name in list(generated_fields.keys()):
            grade, _ = await grade_retrieval(field_name, flower.latin_name, deduped, llm)
            if grade == "insufficient":
                generated_fields[field_name] = NOT_AVAILABLE
            elif grade == "partial":
                log.info("pipeline.crag_partial", field=field_name)

        fields_to_verify = {f: v for f, v in generated_fields.items() if v != NOT_AVAILABLE}
        verification_results = await verify_all_fields(fields_to_verify, source_text, llm)

        confidence_scores = {
            field: {"llm_score": res.confidence}
            for field, res in verification_results.items()
        }

        # Stage 10: Persist
        flower.status = "enriched"
        flower.description = synthesis_result.description
        flower.fun_fact = synthesis_result.fun_fact
        flower.wiki_description = synthesis_result.wiki_description
        flower.habitat = synthesis_result.habitat
        flower.etymology = synthesis_result.etymology
        flower.cultural_info = synthesis_result.cultural_info
        flower.petal_color_hex = synthesis_result.petal_color_hex
        if pfaf_raw_care:
            flower.care_info = pfaf_raw_care
        elif synthesis_result.care_info:
            flower.care_info = synthesis_result.care_info
        flower.confidence_scores = confidence_scores

        if not flower.feature_month:
            d = feature_date or date(2026, 5, 1)
            flower.feature_year = d.year
            flower.feature_month = d.month
            flower.feature_day = d.day

        await db.commit()
        await db.refresh(flower)

        # Stage 10: Translate into all supported languages
        log.info("pipeline.translating", flower_id=flower_id)
        try:
            await translate_flower(flower_id, db)
            log.info("pipeline.translated", flower_id=flower_id)
        except Exception as e:
            log.warning("pipeline.translate_failed", flower_id=flower_id, error=str(e))

        elapsed = time.perf_counter() - start_time
        _log_mlflow_metrics(confidence_scores, len(chunks), len(deduped), elapsed)
        log.info("pipeline.complete", flower_id=flower_id, status=flower.status, elapsed_s=round(elapsed, 2))
        return flower


async def _adaptive_synthesize(
    flower: Flower,
    chunks: list[RetrievedChunk],
    sources_present: set[str],
    llm: LLMProvider,
) -> SynthesizedFlower:
    """Route synthesis based on source coverage."""
    if "pfaf" in sources_present and "wikipedia" in sources_present:
        return await synthesize(flower.latin_name, flower.common_name, chunks, llm)
    elif "wikidata" in sources_present or "gbif" in sources_present:
        return await synthesize(
            flower.latin_name, flower.common_name, chunks, llm,
            fields_to_skip={"fun_fact", "cultural_info"},
        )
    else:
        return await synthesize(
            flower.latin_name, flower.common_name, chunks, llm,
            fields_to_skip={"fun_fact", "cultural_info", "etymology"},
        )
