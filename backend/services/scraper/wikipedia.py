"""Wikipedia API client — fetches article text + taxonomy infobox."""
import re
from dataclasses import dataclass

import httpx


@dataclass
class WikipediaData:
    latin_name: str
    page_title: str | None = None
    url: str | None = None
    extract: str | None = None          # Full article plain text
    summary: str | None = None          # First paragraph
    taxonomy: dict = None               # Parsed infobox key/value pairs

    def __post_init__(self) -> None:
        if self.taxonomy is None:
            self.taxonomy = {}


_API = "https://en.wikipedia.org/w/api.php"
_HEADERS = {"User-Agent": "FloraRAGPipeline/1.0 (portfolio; contact: simone.84858@gmail.com)"}


def _hybrid_variants(latin_name: str) -> list[str]:
    """Return alternate search strings for hybrid names containing ' x '.

    Wikipedia often uses the multiplication sign (×) instead of the letter x,
    or omits the hybrid marker entirely.  We try all three forms so that e.g.
    'Crocosmia x crocosmiiflora' still resolves to a valid article.
    """
    if " x " not in latin_name:
        return [latin_name]
    return [
        latin_name,                              # original: 'Crocosmia x crocosmiiflora'
        latin_name.replace(" x ", " × "),        # Unicode ×: 'Crocosmia × crocosmiiflora'
        latin_name.replace(" x ", " "),          # no marker:  'Crocosmia crocosmiiflora'
    ]


async def _try_fetch(latin_name: str, client: httpx.AsyncClient) -> WikipediaData | None:
    """Attempt to fetch a single Wikipedia article for the given name."""
    search_resp = await client.get(_API, params={
        "action": "query",
        "list": "search",
        "srsearch": latin_name,
        "srlimit": 3,
        "format": "json",
    })
    search_resp.raise_for_status()
    results = search_resp.json().get("query", {}).get("search", [])
    if not results:
        return None

    page_title = results[0]["title"]

    extract_resp = await client.get(_API, params={
        "action": "query",
        "titles": page_title,
        "prop": "extracts|info|revisions",
        "explaintext": True,
        "inprop": "url",
        "rvprop": "content",
        "rvslots": "main",
        "format": "json",
    })
    extract_resp.raise_for_status()
    pages = extract_resp.json().get("query", {}).get("pages", {})
    page = next(iter(pages.values()))

    if page.get("missing") is not None:
        return None

    data = WikipediaData(latin_name=latin_name, page_title=page_title)
    base = "https://en.wikipedia.org/wiki"
    slug = page_title.replace(" ", "_")
    data.url = page.get("canonicalurl") or f"{base}/{slug}"
    data.extract = page.get("extract", "")

    if data.extract:
        for para in data.extract.split("\n"):
            para = para.strip()
            if para and not para.startswith("="):
                data.summary = para
                break

    wikitext = ""
    revisions = page.get("revisions", [])
    if revisions:
        wikitext = revisions[0].get("slots", {}).get("main", {}).get("*", "")
    data.taxonomy = _parse_taxobox(wikitext)

    return data


async def fetch_wikipedia(latin_name: str) -> WikipediaData | None:
    async with httpx.AsyncClient(timeout=20.0, headers=_HEADERS) as client:
        for variant in _hybrid_variants(latin_name):
            result = await _try_fetch(variant, client)
            if result is not None:
                # Always tag the result with the original canonical name
                result.latin_name = latin_name
                return result
        return None


def _parse_taxobox(wikitext: str) -> dict[str, str]:
    """Extract key=value pairs from {{Taxobox|...}} or {{speciesbox|...}}."""
    taxonomy: dict[str, str] = {}
    m = re.search(r"\{\{(?:Taxobox|Automatic taxobox|Speciesbox)(.*?)\}\}", wikitext, re.S | re.I)
    if not m:
        return taxonomy
    for line in m.group(1).split("|"):
        if "=" in line:
            key, _, val = line.partition("=")
            key = re.sub(r"[\[\]{}']", "", key).strip().lower().replace(" ", "_")
            val = re.sub(r"\[\[([^\]|]+)(?:\|[^\]]*)?\]\]", r"\1", val)  # unwrap links
            val = re.sub(r"\{\{[^}]+\}\}", "", val).strip()
            if key and val:
                taxonomy[key] = val
    return taxonomy
