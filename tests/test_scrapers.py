"""Unit tests for scrapers — mock HTTP responses, no real network calls."""
from __future__ import annotations

import pytest
import respx
import httpx

from services.scraper.wikipedia import fetch_wikipedia, _parse_taxobox
from services.scraper.gbif import fetch_gbif
from services.scraper.wikidata import fetch_wikidata


# --- Wikipedia tests ---

@pytest.fixture
def wiki_search_response():
    return {
        "query": {
            "search": [{"title": "Rosa canina", "snippet": "Dog rose..."}]
        }
    }


@pytest.fixture
def wiki_extract_response():
    return {
        "query": {
            "pages": {
                "12345": {
                    "title": "Rosa canina",
                    "canonicalurl": "https://en.wikipedia.org/wiki/Rosa_canina",
                    "extract": "Rosa canina, commonly known as the dog rose, is a variable climbing, rose species.\n\nIt is a deciduous shrub.",
                    "revisions": [{"slots": {"main": {"*": "{{Taxobox|kingdom=Plantae|order=Rosales|family=Rosaceae}}}"}}}],
                }
            }
        }
    }


@respx.mock
@pytest.mark.asyncio
async def test_fetch_wikipedia_returns_data(wiki_search_response, wiki_extract_response):
    respx.get("https://en.wikipedia.org/w/api.php").mock(
        side_effect=[
            httpx.Response(200, json=wiki_search_response),
            httpx.Response(200, json=wiki_extract_response),
        ]
    )
    result = await fetch_wikipedia("Rosa canina")
    assert result is not None
    assert result.page_title == "Rosa canina"
    assert "dog rose" in result.extract
    assert result.summary is not None
    assert "wikipedia.org" in result.url


@respx.mock
@pytest.mark.asyncio
async def test_fetch_wikipedia_no_results():
    respx.get("https://en.wikipedia.org/w/api.php").mock(
        return_value=httpx.Response(200, json={"query": {"search": []}})
    )
    result = await fetch_wikipedia("Fakeus nonexistens")
    assert result is None


def test_parse_taxobox_extracts_fields():
    wikitext = "{{Taxobox|kingdom=Plantae|order=Rosales|family=Rosaceae|genus=Rosa}}"
    taxonomy = _parse_taxobox(wikitext)
    assert taxonomy.get("kingdom") == "Plantae"
    assert taxonomy.get("family") == "Rosaceae"


# --- GBIF tests ---

@respx.mock
@pytest.mark.asyncio
async def test_fetch_gbif_returns_data():
    respx.get("https://api.gbif.org/v1/species/match").mock(
        return_value=httpx.Response(200, json={
            "usageKey": 5334357,
            "matchType": "EXACT",
            "kingdom": "Plantae",
            "family": "Rosaceae",
            "genus": "Rosa",
            "species": "Rosa canina",
            "status": "ACCEPTED",
        })
    )
    respx.get("https://api.gbif.org/v1/species/5334357/vernacularNames").mock(
        return_value=httpx.Response(200, json={"results": [
            {"language": "de", "vernacularName": "Hundsrose"},
            {"language": "fr", "vernacularName": "Rosier des chiens"},
        ]})
    )
    respx.get("https://api.gbif.org/v1/species/5334357/distributions").mock(
        return_value=httpx.Response(200, json={"results": [
            {"locationId": "ISO 3166-1:DE"},
            {"locationId": "ISO 3166-1:FR"},
        ]})
    )
    result = await fetch_gbif("Rosa canina")
    assert result is not None
    assert result.family == "Rosaceae"
    assert result.vernacular_names.get("de") == "Hundsrose"
    assert len(result.distributions) == 2


@respx.mock
@pytest.mark.asyncio
async def test_fetch_gbif_no_match():
    respx.get("https://api.gbif.org/v1/species/match").mock(
        return_value=httpx.Response(200, json={"matchType": "NONE"})
    )
    result = await fetch_gbif("Fakeus nonexistens")
    assert result is None
