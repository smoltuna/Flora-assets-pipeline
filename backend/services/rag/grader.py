"""Corrective RAG (CRAG) — grades retrieval quality before synthesis.

For each output field, grades whether retrieved chunks provide sufficient
evidence. Falls back gracefully: 'insufficient' → skip field (no hallucination).
"""
from __future__ import annotations

from services.llm.provider import LLMProvider
from services.rag.retriever import RetrievedChunk

RetrievalGrade = str  # 'sufficient' | 'partial' | 'insufficient'


async def grade_retrieval(
    field_name: str,
    latin_name: str,
    chunks: list[RetrievedChunk],
    llm: LLMProvider,
) -> tuple[RetrievalGrade, list[RetrievedChunk]]:
    """Grade each chunk's relevance to the target field.

    Returns (grade, filtered_chunks) where grade determines synthesis strategy:
      sufficient  → standard RAG synthesis
      partial     → generate but mark confidence as low, flag for review
      insufficient → set field to 'Information not available.' — no hallucination
    """
    if not chunks:
        return "insufficient", []

    graded: list[RetrievedChunk] = []
    for chunk in chunks:
        snippet = chunk.chunk_text[:500]
        response = await llm.complete(
            prompt=(
                f"Is the following text relevant to generating the '{field_name}' field "
                f"for the plant {latin_name}?\n\n{snippet}\n\n"
                "Respond with exactly one word: relevant or irrelevant"
            ),
            system="You are a botanical data quality assessor. Be strict and concise.",
        )
        word = response.strip().lower().split()[0] if response.strip() else "irrelevant"
        if word == "relevant":
            graded.append(chunk)

    if len(graded) >= 2:
        return "sufficient", graded
    elif len(graded) == 1:
        return "partial", graded
    else:
        return "insufficient", []
