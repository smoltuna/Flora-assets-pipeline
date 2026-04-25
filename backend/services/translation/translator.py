"""RAG-grounded translation service.

Strategy:
  - Provider: uses settings.translation_provider (default: "ollama" — no rate limits).
    Set TRANSLATION_PROVIDER=groq only with a paid account; free Groq TPM (~14,400/min)
    is far too low for batch translation.
  - FORCE_FIELDWISE=True (default): each field is translated with a separate plain-text
    call. More reliable for small models (llama3.2:3b) at the cost of more LLM round trips.
    Set FORCE_FIELDWISE=False to try a single batched JSON call first, falling back to
    per-field calls if the batch response is too sparse (< MIN_BATCH_FIELDS fields).
  - Languages are processed sequentially to avoid bursting any rate limits.
  - Input fields are truncated to MAX_FIELD_CHARS to keep prompts manageable.
"""
from __future__ import annotations

import json
import re

import structlog
from models import Flower, Translation
from sqlalchemy.ext.asyncio import AsyncSession

from services.llm.provider import get_provider


def _get_translation_provider():
    from config import settings
    return get_provider(settings.translation_provider)

log = structlog.get_logger()

ALL_LANGUAGES = {"de", "fr", "es", "it", "zh", "ja"}

LANG_NAMES = {
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "it": "Italian",
    "zh": "Simplified Chinese",
    "ja": "Japanese",
}

TEXT_FIELDS = [
    "description", "fun_fact", "wiki_description",
    "habitat", "etymology", "cultural_info",
]

# camelCase keys used in LLM prompts (avoids underscore escaping issues in JSON)
_FIELD_PROMPT_KEY = {
    "description": "description",
    "fun_fact": "funFact",
    "wiki_description": "wikiDescription",
    "habitat": "habitat",
    "etymology": "etymology",
    "cultural_info": "culturalInfo",
}
_REVERSE_KEY = {v: k for k, v in _FIELD_PROMPT_KEY.items()}

# Truncate each field to this many chars before sending to LLM (controls token usage)
MAX_FIELD_CHARS = 600

# Minimum number of successfully translated fields to accept a batch response.
# Set high (5+) so that a near-empty batch response triggers the field-by-field fallback.
# llama3.2:3b often produces partial or malformed JSON in batch mode; field-by-field
# is slower but far more reliable for small models.
MIN_BATCH_FIELDS = 5

# Set to True to always use field-by-field mode (skip the batch attempt entirely).
# Recommended for llama3.2:3b and other small models.
FORCE_FIELDWISE = True


async def translate_flower(flower_id: int, session: AsyncSession) -> None:
    """Translate all text fields + common name for a flower into all supported languages.

    Languages are processed sequentially so we never burst against rate limits.
    """
    flower = await session.get(Flower, flower_id)
    if not flower:
        return

    for lang in sorted(ALL_LANGUAGES):
        try:
            await _translate_all(flower, lang, session)
        except Exception as e:
            log.error("translation.error", flower_id=flower_id, lang=lang, error=str(e))


async def _translate_all(flower: Flower, lang: str, session: AsyncSession) -> None:
    """Translate name + all text fields for one language.

    Tries a single batched JSON call first; falls back to per-field calls if the
    batch response can't be parsed or is too sparse.
    """
    log.info("translation.start", flower_id=flower.id, lang=lang)
    llm = _get_translation_provider()
    lang_name = LANG_NAMES[lang]

    # Collect non-empty source fields (truncated)
    source: dict[str, str] = {}
    for field in TEXT_FIELDS:
        text = getattr(flower, field, None)
        if text and text != "Information not available.":
            source[_FIELD_PROMPT_KEY[field]] = text[:MAX_FIELD_CHARS]

    common = flower.common_name or flower.latin_name

    # ── Attempt 1: batched JSON call (skipped when FORCE_FIELDWISE is set) ────
    if FORCE_FIELDWISE or not source:
        translated = {}
    else:
        translated = await _batch_translate(llm, flower.latin_name, common, lang_name, source)

    # ── Attempt 2: field-by-field fallback if batch produced too few results ──
    if len([v for v in translated.values() if v]) < MIN_BATCH_FIELDS and source:
        if not FORCE_FIELDWISE:
            log.info("translation.fallback", flower_id=flower.id, lang=lang)
        translated = await _fieldwise_translate(llm, flower.latin_name, common, lang_name, source)

    # Map camelCase prompt keys back to snake_case model fields
    fields: dict[str, str | None] = {"name": translated.get("name") or None}
    for prompt_key, field_name in _REVERSE_KEY.items():
        val = translated.get(prompt_key)
        fields[field_name] = val.strip() if val else None

    await _upsert_translation(session, flower.id, lang, fields, source_method="llm_translation")
    n = len([v for v in fields.values() if v])
    log.info("translation.done", flower_id=flower.id, lang=lang, n_fields=n)


async def _batch_translate(
    llm, latin_name: str, common_name: str, lang_name: str, source: dict[str, str]
) -> dict:
    """Single LLM call: translate all fields at once. Returns parsed dict (may be empty)."""
    source_json = json.dumps(source, ensure_ascii=False, indent=2)
    prompt = (
        f"Translate all JSON fields below into {lang_name} for the plant "
        f"{latin_name} (English common name: {common_name}).\n\n"
        f"Also add a \"name\" field with the proper {lang_name} botanical common name "
        f"(e.g. the actual local name, not a literal word-for-word translation).\n\n"
        f"Source (English):\n{source_json}\n\n"
        f'Return ONLY a valid JSON object with the same keys '
        f'plus "name". No markdown, no explanation.'
    )
    try:
        response = await llm.complete(
            prompt=prompt,
            system=(
                f"You are a precise botanical translator. "
                f"Output only valid JSON with {lang_name} values."
            ),
        )
        return _parse_json(response)
    except Exception as e:
        log.warning("translation.batch_failed", error=str(e))
        return {}


async def _fieldwise_translate(
    llm, latin_name: str, common_name: str, lang_name: str, source: dict[str, str]
) -> dict:
    """Translate each field with a simple plain-text prompt, then reassemble."""
    results: dict[str, str] = {}

    # Translate common name
    try:
        resp = await llm.complete(
            prompt=(
                f"What is the common name for {latin_name} ({common_name}) in {lang_name}? "
                f"Reply with only the {lang_name} name, nothing else."
            ),
            system=f"You are a botanical expert. Reply with only the plant name in {lang_name}.",
        )
        name = resp.strip().strip("\"'")
        if name:
            results["name"] = name
    except Exception as e:
        log.warning("translation.name_failed", lang=lang_name, error=str(e))

    # Translate each text field
    for prompt_key, text in source.items():
        try:
            resp = await llm.complete(
                prompt=(
                    f"Translate the following botanical text "
                    f"about {latin_name} into {lang_name}.\n\n"
                    f"{text}\n\n"
                    f"Reply with only the {lang_name} translation, nothing else."
                ),
                system=(
                    f"You are a botanical translator. "
                    f"Reply with only the translation in {lang_name}."
                ),
            )
            val = resp.strip()
            if val:
                results[prompt_key] = val
        except Exception as e:
            log.warning("translation.field_failed", field=prompt_key, lang=lang_name, error=str(e))

    return results


def _parse_json(response: str) -> dict:
    """Extract a JSON object from an LLM response, stripping markdown fences."""
    text = re.sub(r"```(?:json)?\s*", "", response).strip().rstrip("`").strip()
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end == 0:
        return {}
    try:
        data = json.loads(text[start:end])
        return {k: v for k, v in data.items() if isinstance(v, str) and v.strip()}
    except json.JSONDecodeError:
        return {}


async def _upsert_translation(
    session: AsyncSession,
    flower_id: int,
    lang: str,
    fields: dict[str, str | None],
    source_method: str,
) -> None:
    from sqlalchemy import select
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
