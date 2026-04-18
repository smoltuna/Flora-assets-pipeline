"""LLM synthesis — constructs prompts from retrieved sources and generates structured output.

Output is validated against Pydantic v2 schemas. Uses source attribution
(according to PFAF... / Wikipedia states...) so the LLM grounds claims.
"""
from __future__ import annotations

import json
import re

from pydantic import BaseModel, Field

from services.llm.provider import LLMProvider
from services.rag.retriever import RetrievedChunk

NOT_AVAILABLE = "Information not available."


class SynthesizedFlower(BaseModel):
    description: str = Field(default=NOT_AVAILABLE)
    fun_fact: str = Field(default=NOT_AVAILABLE)
    wiki_description: str = Field(default=NOT_AVAILABLE)
    habitat: str = Field(default=NOT_AVAILABLE)
    etymology: str = Field(default=NOT_AVAILABLE)
    cultural_info: str = Field(default=NOT_AVAILABLE)
    petal_color_hex: str | None = Field(default=None)
    care_info: dict = Field(default_factory=dict)


def _format_context(chunks: list[RetrievedChunk]) -> str:
    """Group chunks by source for attributed context."""
    by_source: dict[str, list[str]] = {}
    for chunk in chunks:
        by_source.setdefault(chunk.source, []).append(chunk.chunk_text)

    sections: list[str] = []
    source_labels = {
        "pfaf": "PFAF (Plants For A Future)",
        "wikipedia": "Wikipedia",
        "wikidata": "Wikidata",
        "gbif": "GBIF",
    }
    for source, texts in by_source.items():
        label = source_labels.get(source, source.upper())
        sections.append(f"[{label}]\n" + "\n---\n".join(texts))
    return "\n\n".join(sections)


async def synthesize(
    latin_name: str,
    common_name: str | None,
    chunks: list[RetrievedChunk],
    llm: LLMProvider,
    fields_to_skip: set[str] | None = None,
) -> SynthesizedFlower:
    """Generate all text fields for a flower from retrieved source chunks."""
    if not chunks:
        return SynthesizedFlower()

    skip = fields_to_skip or set()
    context = _format_context(chunks)
    display_name = common_name or latin_name

    prompt = f"""You are a botanical data writer. Using ONLY the source material below, generate JSON for the plant "{display_name}" ({latin_name}).

SOURCE MATERIAL:
{context}

Generate a JSON object with these fields (use exactly these keys):
- "description": 2-3 sentence engaging description for a general audience (skip if insufficient data: "{NOT_AVAILABLE}")
- "fun_fact": one surprising or delightful fact (skip if insufficient: "{NOT_AVAILABLE}")
- "wiki_description": concise encyclopedic summary, 1-2 sentences (skip if insufficient: "{NOT_AVAILABLE}")
- "habitat": native habitat and range description (skip if insufficient: "{NOT_AVAILABLE}")
- "etymology": meaning/origin of the latin name (skip if insufficient: "{NOT_AVAILABLE}")
- "cultural_info": historical or cultural significance (skip if insufficient: "{NOT_AVAILABLE}")
- "petal_color_hex": dominant petal color as hex code like "#FF6B6B", or null if unknown
- "care_info": object with keys like "water", "sun", "soil", "hardiness" — extract from PFAF data

Fields to skip (set to "{NOT_AVAILABLE}"): {list(skip) if skip else "none"}

Rules:
- Ground every claim in the provided sources. Do not invent facts.
- If a field cannot be answered from sources, use "{NOT_AVAILABLE}"
- Return only valid JSON, no markdown, no explanation."""

    response = await llm.complete(prompt=prompt, system="You are a precise botanical content writer. Output only valid JSON.")

    return _parse_response(response)


def _parse_response(response: str) -> SynthesizedFlower:
    """Extract and validate JSON from LLM response."""
    # Strip markdown code fences if present
    text = re.sub(r"```(?:json)?\s*", "", response).strip().rstrip("`").strip()

    # Find JSON object boundaries
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end == 0:
        return SynthesizedFlower()

    try:
        data = json.loads(text[start:end])
        return SynthesizedFlower(**{k: v for k, v in data.items() if k in SynthesizedFlower.model_fields})
    except (json.JSONDecodeError, Exception):
        return SynthesizedFlower()
