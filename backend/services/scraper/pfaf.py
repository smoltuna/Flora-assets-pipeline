"""PFAF.org scraper — plants for a future database.

PFAF robots.txt (checked 2026-04): no crawl restrictions for /database/ paths.
Rate limit: 2-second delay enforced by caller. One request per plant.
"""
import asyncio
import re
from dataclasses import dataclass, field

import httpx
from bs4 import BeautifulSoup


@dataclass
class PFAFData:
    latin_name: str
    common_name: str | None = None
    edibility_rating: int | None = None
    medicinal_rating: int | None = None
    other_uses_rating: int | None = None
    weed_potential: str | None = None
    habitat: str | None = None
    care_info: dict = field(default_factory=dict)
    raw_text: str = ""


_BASE = "https://pfaf.org/user/Plant.aspx"
_DELAY = 2.0


async def scrape_pfaf(latin_name: str) -> PFAFData | None:
    url = f"{_BASE}?Latin={latin_name.replace(' ', '+')}"
    async with httpx.AsyncClient(
        timeout=30.0,
        headers={"User-Agent": "FloraRAGPipeline/1.0 (portfolio; contact: simone.84858@gmail.com)"},
        follow_redirects=True,
    ) as client:
        try:
            resp = await client.get(url)
            resp.raise_for_status()
        except httpx.HTTPStatusError:
            return None

    await asyncio.sleep(_DELAY)

    soup = BeautifulSoup(resp.text, "html.parser")
    if "plant not found" in resp.text.lower():
        return None

    data = PFAFData(latin_name=latin_name)
    data.raw_text = soup.get_text(separator="\n", strip=True)

    # Common name
    cn_tag = soup.select_one("h2.plant-common-name, span#Label1, .common-name")
    if cn_tag:
        data.common_name = cn_tag.get_text(strip=True) or None

    # Star ratings (PFAF uses images with alt text "star" or numeric spans)
    def _parse_rating(label: str) -> int | None:
        row = soup.find(string=re.compile(label, re.I))
        if not row:
            return None
        parent = row.parent
        # Look for digit nearby
        text = parent.get_text()
        m = re.search(r"\b([0-5])\b", text)
        return int(m.group(1)) if m else None

    data.edibility_rating = _parse_rating("edib")
    data.medicinal_rating = _parse_rating("medic")
    data.other_uses_rating = _parse_rating("other use")

    # Weed potential
    weed_row = soup.find(string=re.compile(r"weed potential", re.I))
    if weed_row:
        data.weed_potential = weed_row.parent.get_text(strip=True).replace("Weed potential", "").strip() or None

    # Care info — harvest structured table rows
    care: dict[str, str] = {}
    for row in soup.select("table tr"):
        cells = row.find_all(["td", "th"])
        if len(cells) >= 2:
            key = cells[0].get_text(strip=True).rstrip(":")
            val = cells[1].get_text(strip=True)
            if key and val and len(key) < 50:
                care[key] = val
    data.care_info = care

    # Habitat — look for known section headers
    for candidate in ("Habitat", "habitat", "Ecology"):
        section = soup.find(string=re.compile(candidate, re.I))
        if section:
            p = section.find_next("p")
            if p:
                data.habitat = p.get_text(strip=True)
                break

    return data
