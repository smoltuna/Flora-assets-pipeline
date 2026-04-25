"""Extended deduplication tests with realistic embedding scenarios."""
from __future__ import annotations

import math

from services.rag.deduplicator import deduplicate_chunks
from services.rag.retriever import RetrievedChunk


def _unit_vec(angle_deg: float, dim: int = 768) -> list[float]:
    """2D unit vector padded to 768 dims."""
    rad = math.radians(angle_deg)
    v = [math.cos(rad), math.sin(rad)] + [0.0] * (dim - 2)
    return v


def _chunk(text: str, source: str, angle: float) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=0, chunk_text=text, source=source,
        rrf_score=1.0, embedding=_unit_vec(angle),
    )


def test_deduplicate_four_sources_keeps_distinct():
    chunks = [
        _chunk("Wikipedia article about Rosa canina, common name dog rose.", "wikipedia", 0),
        _chunk("Wikidata: Rosa canina, native to Europe.", "wikidata", 5),   # near-dup of 0
        _chunk("PFAF care info: Hardy to zone 4, full sun preferred.", "pfaf", 90),
        _chunk("GBIF distribution data: found in 40 countries.", "gbif", 180),
    ]
    # angle 0 and 5 have cosine ~0.9962, above 0.92 threshold
    result = deduplicate_chunks(chunks, similarity_threshold=0.92)
    assert len(result) == 3


def test_deduplicate_single_chunk_returns_it():
    chunks = [_chunk("Only one chunk.", "wikipedia", 45)]
    result = deduplicate_chunks(chunks)
    assert len(result) == 1
    assert result[0].chunk_text == "Only one chunk."


def test_deduplicate_prefers_longer_text():
    emb = _unit_vec(0)
    short = RetrievedChunk(
        chunk_id=1, chunk_text="Short.",
        source="wikipedia", rrf_score=1.0, embedding=emb,
    )
    long_ = RetrievedChunk(
        chunk_id=2,
        chunk_text="Much longer text with lots of botanical detail about the species.",
        source="wikipedia", rrf_score=1.0, embedding=emb,
    )
    result = deduplicate_chunks([short, long_], similarity_threshold=0.92)
    assert len(result) == 1
    assert "longer" in result[0].chunk_text
