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


def _clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def _parse_int_rating(value: str | None) -> int | None:
    if not value:
        return None
    m = re.search(r"\b([0-5])\b", value)
    return int(m.group(1)) if m else None


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for v in values:
        k = v.lower()
        if k not in seen:
            seen.add(k)
            out.append(v)
    return out


def _extract_care_from_icons(care_cell: BeautifulSoup) -> dict[str, str]:
    labels: list[str] = []
    for img in care_cell.find_all("img"):
        label = _clean_text((img.get("title") or img.get("alt") or ""))
        if label:
            labels.append(label)

    if not labels:
        return {}

    labels = _dedupe(labels)
    hardiness: list[str] = []
    soil: list[str] = []
    sun: list[str] = []
    other: list[str] = []

    for label in labels:
        lower = label.lower()
        if any(k in lower for k in ("hardy", "frost", "tender", "warm", "temperature")):
            hardiness.append(label)
        elif any(k in lower for k in ("soil", "water", "moist", "wet", "drained", "aquatic")):
            soil.append(label)
        elif any(k in lower for k in ("sun", "shade")):
            sun.append(label)
        else:
            other.append(label)

    care: dict[str, str] = {}
    if sun:
        care["Sun"] = ", ".join(sun)
    if soil:
        care["Soil"] = ", ".join(soil)
    if hardiness:
        care["Hardiness"] = ", ".join(hardiness)
    if other:
        care["Other"] = ", ".join(other)
    return care


async def scrape_pfaf(latin_name: str) -> PFAFData | None:
    async with httpx.AsyncClient(
        timeout=30.0,
        headers={"User-Agent": "FloraRAGPipeline/1.0 (portfolio; contact: simone.84858@gmail.com)"},
        follow_redirects=True,
    ) as client:
        try:
            # Use LatinName to open the actual plant detail page.
            resp = await client.get(_BASE, params={"LatinName": latin_name})
            resp.raise_for_status()
        except httpx.HTTPStatusError:
            return None

    await asyncio.sleep(_DELAY)

    soup = BeautifulSoup(resp.text, "html.parser")
    if "plant not found" in resp.text.lower():
        return None

    data = PFAFData(latin_name=latin_name)
    data.raw_text = soup.get_text(separator="\n", strip=True)

    # Parse top-level key/value rows from plant detail tables.
    row_values: dict[str, str] = {}
    care_cell = None
    for row in soup.select("table tr"):
        cells = row.find_all(["td", "th"], recursive=False)
        if len(cells) < 2:
            continue
        key = _clean_text(cells[0].get_text(" ", strip=True)).rstrip(":")
        if not key or len(key) > 80:
            continue
        val = _clean_text(cells[1].get_text(" ", strip=True))
        row_values[key] = val
        if "care" in key.lower():
            care_cell = cells[1]

    # Common/detail fields
    data.common_name = row_values.get("Common Name") or None
    data.edibility_rating = _parse_int_rating(row_values.get("Edibility Rating"))
    data.medicinal_rating = _parse_int_rating(row_values.get("Medicinal Rating"))
    data.other_uses_rating = _parse_int_rating(row_values.get("Other Uses"))
    data.weed_potential = row_values.get("Weed Potential") or None
    data.habitat = row_values.get("Habitats") or None

    # Care info from PFAF care icons (exact labels preserved as values).
    if care_cell is not None:
        data.care_info = _extract_care_from_icons(care_cell)
    else:
        data.care_info = {}

    return data
