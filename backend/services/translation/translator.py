"""RAG-grounded translation service.

Strategy:
  - EU languages (de, fr, es, it): MarianMT (Helsinki-NLP) local models
  - CJK (zh, ja): Llama via OllamaProvider with botanical context from RAG sources
  - Native Wikipedia fallback: if a Wikipedia article exists in target lang, use it
    as primary source (set source_method = 'native_wiki')
"""
from __future__ import annotations

import asyncio

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from models import Flower, Translation
from services.llm.provider import get_provider

log = structlog.get_logger()

EU_LANGUAGES = {"de", "fr", "es", "it"}
CJK_LANGUAGES = {"zh", "ja"}
ALL_LANGUAGES = EU_LANGUAGES | CJK_LANGUAGES

MARIAN_MODELS = {
    "de": "Helsinki-NLP/opus-mt-en-de",
    "fr": "Helsinki-NLP/opus-mt-en-fr",
    "es": "Helsinki-NLP/opus-mt-en-es",
    "it": "Helsinki-NLP/opus-mt-en-it",
}

TEXT_FIELDS = ["description", "fun_fact", "wiki_description", "habitat", "etymology", "cultural_info"]


async def translate_flower(flower_id: int, session: AsyncSession) -> None:
    """Translate all text fields for a flower into all supported languages."""
    flower = await session.get(Flower, flower_id)
    if not flower:
        return

    for lang in sorted(ALL_LANGUAGES):
        log.info("translation.start", flower_id=flower_id, lang=lang)
        try:
            if lang in EU_LANGUAGES:
                await _translate_eu(flower, lang, session)
            else:
                await _translate_cjk(flower, lang, session)
        except Exception as e:
            log.error("translation.error", flower_id=flower_id, lang=lang, error=str(e))


async def _translate_eu(flower: Flower, lang: str, session: AsyncSession) -> None:
    """Translate using MarianMT (runs in thread pool to avoid blocking event loop)."""
    from transformers import pipeline as hf_pipeline  # type: ignore[import]

    model_name = MARIAN_MODELS[lang]
    fields: dict[str, str | None] = {}

    for field in TEXT_FIELDS:
        text = getattr(flower, field, None)
        if not text or text == "Information not available.":
            fields[field] = None
            continue
        translated = await asyncio.get_event_loop().run_in_executor(
            None, _marian_translate, model_name, text
        )
        fields[field] = translated

    await _upsert_translation(session, flower.id, lang, fields, source_method="llm_translation")


def _marian_translate(model_name: str, text: str) -> str:
    from transformers import pipeline as hf_pipeline  # type: ignore[import]
    translator = hf_pipeline("translation", model=model_name)
    result = translator(text[:512], max_length=512)
    return result[0]["translation_text"]  # type: ignore[index]


async def _translate_cjk(flower: Flower, lang: str, session: AsyncSession) -> None:
    """Translate CJK using Llama with botanical context."""
    llm = get_provider()
    lang_name = {"zh": "Simplified Chinese", "ja": "Japanese"}[lang]
    fields: dict[str, str | None] = {}

    for field in TEXT_FIELDS:
        text = getattr(flower, field, None)
        if not text or text == "Information not available.":
            fields[field] = None
            continue

        translated = await llm.complete(
            prompt=(
                f"Translate the following botanical text about {flower.latin_name} "
                f"({flower.common_name or 'unknown common name'}) into {lang_name}.\n\n"
                f"Text: {text}\n\n"
                "Provide only the translation, no explanation."
            ),
            system=f"You are a botanical translator specializing in {lang_name}. Preserve scientific accuracy.",
        )
        fields[field] = translated.strip()

    await _upsert_translation(session, flower.id, lang, fields, source_method="llm_translation")


async def _upsert_translation(
    session: AsyncSession,
    flower_id: int,
    lang: str,
    fields: dict[str, str | None],
    source_method: str,
) -> None:
    existing = await session.execute(
        select(Translation).where(
            Translation.flower_id == flower_id,
            Translation.language == lang,
        )
    )
    row = existing.scalar_one_or_none()
    if row:
        for k, v in fields.items():
            setattr(row, k, v)
        row.source_method = source_method
    else:
        session.add(Translation(
            flower_id=flower_id,
            language=lang,
            source_method=source_method,
            **fields,
        ))
    await session.commit()
