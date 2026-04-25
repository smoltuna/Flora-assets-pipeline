"""GBIF API client — taxonomic and distribution data.

Uses the public GBIF REST API (no auth required for read-only lookups).
"""
from dataclasses import dataclass, field

import httpx


@dataclass
class GBIFData:
    latin_name: str
    usage_key: int | None = None
    kingdom: str | None = None
    phylum: str | None = None
    order: str | None = None
    family: str | None = None
    genus: str | None = None
    species: str | None = None
    taxonomic_status: str | None = None
    habitats: list[str] = field(default_factory=list)
    distributions: list[str] = field(default_factory=list)  # country codes / region names
    vernacular_names: dict[str, str] = field(default_factory=dict)  # lang → name


_SPECIES_API = "https://api.gbif.org/v1/species"
_HEADERS = {"User-Agent": "FloraRAGPipeline/1.0 (portfolio; contact: simone.84858@gmail.com)"}


async def fetch_gbif(latin_name: str) -> GBIFData | None:
    async with httpx.AsyncClient(timeout=20.0, headers=_HEADERS) as client:
        # GBIF handles hybrids best when the 'x' hybrid marker is stripped.
        # e.g. 'Crocosmia x crocosmiiflora' → 'Crocosmia crocosmiiflora'
        search_name = latin_name.replace(" x ", " ").replace(" × ", " ")

        # Match species by name
        match_resp = await client.get(
            f"{_SPECIES_API}/match",
            params={"name": search_name, "verbose": False},
        )
        match_resp.raise_for_status()
        match = match_resp.json()

        if match.get("matchType") == "NONE" or "usageKey" not in match:
            return None

        key = match["usageKey"]
        data = GBIFData(latin_name=latin_name, usage_key=key)
        data.kingdom = match.get("kingdom")
        data.phylum = match.get("phylum")
        data.order = match.get("order")
        data.family = match.get("family")
        data.genus = match.get("genus")
        data.species = match.get("species")
        data.taxonomic_status = match.get("status")

        # Fetch vernacular names
        vern_resp = await client.get(f"{_SPECIES_API}/{key}/vernacularNames", params={"limit": 50})
        vern_resp.raise_for_status()
        for entry in vern_resp.json().get("results", []):
            lang = entry.get("language", "")
            name = entry.get("vernacularName", "")
            if lang and name and lang not in data.vernacular_names:
                data.vernacular_names[lang] = name

        # Fetch distributions
        dist_resp = await client.get(f"{_SPECIES_API}/{key}/distributions", params={"limit": 100})
        dist_resp.raise_for_status()
        for entry in dist_resp.json().get("results", []):
            loc = entry.get("locationId") or entry.get("locality") or ""
            if loc and loc not in data.distributions:
                data.distributions.append(loc)

    return data
