"""Unit tests for RAG pipeline components — no LLM or DB calls."""
from __future__ import annotations

import pytest
from services.rag.deduplicator import cosine_sim, deduplicate_chunks
from services.rag.retriever import RetrievedChunk
from services.rag.synthesizer import NOT_AVAILABLE, _parse_response

# --- Deduplication tests ---

def _make_chunk(
    text: str, source: str = "wikipedia",
    emb: list[float] | None = None,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=0,
        chunk_text=text,
        source=source,
        rrf_score=1.0,
        embedding=emb or [0.0] * 768,
    )


def test_deduplicate_removes_near_duplicates():
    # Two chunks with identical embeddings → one survives
    emb = [1.0] + [0.0] * 767
    chunks = [
        _make_chunk("Rosa canina grows in hedgerows across Europe.", "wikipedia", emb),
        _make_chunk("Rosa canina is found in hedgerows throughout Europe.", "gbif", emb),
        _make_chunk(
            "The roots are used in traditional medicine.",
            "pfaf", [0.0, 1.0] + [0.0] * 766,
        ),
    ]
    result = deduplicate_chunks(chunks, similarity_threshold=0.92)
    assert len(result) == 2  # near-duplicates collapsed, distinct chunk kept


def test_deduplicate_prefers_structured_source():
    emb = [1.0] + [0.0] * 767
    chunks = [
        _make_chunk("Short wikipedia text.", "wikipedia", emb),
        _make_chunk("Longer pfaf text with more detail about the plant.", "pfaf", emb),
    ]
    result = deduplicate_chunks(chunks, similarity_threshold=0.90)
    assert len(result) == 1
    # pfaf wins: both longer AND structured
    assert result[0].source == "pfaf"


def test_deduplicate_empty_returns_empty():
    assert deduplicate_chunks([]) == []


def test_cosine_sim_identical_vectors():
    v = [1.0, 0.0, 0.0]
    assert cosine_sim(v, v) == pytest.approx(1.0)


def test_cosine_sim_orthogonal_vectors():
    a = [1.0, 0.0]
    b = [0.0, 1.0]
    assert cosine_sim(a, b) == pytest.approx(0.0)


# --- Synthesizer parsing tests ---

def test_parse_response_valid_json():
    json_str = (
        '{"description": "A lovely rose.", '
        '"fun_fact": "Used in jams.", '
        '"petal_color_hex": "#FF0000"}'
    )
    result = _parse_response(json_str)
    assert result.description == "A lovely rose."
    assert result.fun_fact == "Used in jams."
    assert result.petal_color_hex == "#FF0000"


def test_parse_response_with_markdown_fences():
    md = '```json\n{"description": "A lovely rose.", "habitat": "Hedgerows."}\n```'
    result = _parse_response(md)
    assert result.description == "A lovely rose."
    assert result.habitat == "Hedgerows."


def test_parse_response_invalid_json_returns_defaults():
    result = _parse_response("Sorry, I cannot help with that.")
    assert result.description == NOT_AVAILABLE


def test_parse_response_partial_json():
    result = _parse_response('{"description": "Nice flower.", "unknown_field": "ignored"}')
    assert result.description == "Nice flower."
    assert result.fun_fact == NOT_AVAILABLE
