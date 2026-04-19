"""iNaturalist observation photo search — no API key required.

iNaturalist is the best free source for species-specific flower photographs:
  • Research-grade observations are verified by multiple identifiers
  • Photos are typically close-up shots of the plant/flower
  • CC-licensed photos are clearly tagged
  • Taxon search by Latin name is accurate

Strategy:
  1. Resolve latin name → iNaturalist taxon ID
  2. Fetch research-grade observations with photos
  3. Parse photo URLs, license, dimensions, and observer attribution
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import httpx

_TAXA_API = "https://api.inaturalist.org/v1/taxa"
_OBS_API = "https://api.inaturalist.org/v1/observations"
_HEADERS = {
    "User-Agent": "FloraRAGPipeline/1.0 (portfolio; contact: simone.84858@gmail.com)",
    "Accept": "application/json",
}

# iNaturalist CC licenses we accept (same spirit as Wikimedia whitelist)
_ALLOWED_LICENSES: frozenset[str] = frozenset({
    "cc0", "cc-by", "cc-by-sa",
    "cc-by-nc",       # iNaturalist-specific: many citizen-science photos
    "cc-by-nc-sa",
})

# Photo size suffixes on iNaturalist CDN
# original → full res, large → 1024px, medium → 500px
_SIZE_ORIGINAL = "original"
_SIZE_LARGE = "large"


@dataclass
class INatPhoto:
    """A single photo from an iNaturalist observation."""
    photo_id: int
    url_original: str      # full-resolution URL
    url_large: str         # 1024 px version
    attribution: str       # observer / photographer credit
    license_code: str      # e.g. "cc-by-nc"
    taxon_name: str        # confirmed species
    observation_id: int
    quality_grade: str     # "research" or "needs_id"

    # iNaturalist doesn't return dimensions in the API, but we know:
    # - original: variable (typically 1000–4000 px)
    # - large: 1024 px longest edge
    # We estimate conservatively for scoring.
    width: int = 1024
    height: int = 768

    @property
    def aspect(self) -> float:
        return self.width / self.height if self.height else 1.0

    @property
    def source(self) -> str:
        return "inaturalist"


async def _resolve_taxon(
    client: httpx.AsyncClient, latin_name: str,
) -> int | None:
    """Resolve a Latin species name to an iNaturalist taxon ID.

    Returns None if no match found.
    """
    resp = await client.get(_TAXA_API, params={
        "q": latin_name,
        "rank": "species,subspecies,variety",
        "is_active": "true",
        "per_page": 5,
    })
    resp.raise_for_status()
    results = resp.json().get("results", [])

    # Prefer exact match on the name
    latin_lower = latin_name.lower()
    for taxon in results:
        name = (taxon.get("name") or "").lower()
        if name == latin_lower:
            return taxon["id"]

    # Fall back to first result if it's close
    if results:
        return results[0]["id"]

    return None


async def search_inaturalist(
    latin_name: str, *, limit: int = 60,
) -> list[INatPhoto]:
    """Search iNaturalist for CC-licensed photos of a species.

    Returns up to `limit` photos, prioritizing research-grade observations.
    """
    async with httpx.AsyncClient(
        timeout=30.0, headers=_HEADERS,
    ) as client:
        taxon_id = await _resolve_taxon(client, latin_name)
        if taxon_id is None:
            return []

        photos: list[INatPhoto] = []
        seen_photo_ids: set[int] = set()

        # Fetch research-grade first, then needs_id if not enough
        for quality in ("research", "needs_id"):
            if len(photos) >= limit:
                break

            resp = await client.get(_OBS_API, params={
                "taxon_id": taxon_id,
                "quality_grade": quality,
                "photos": "true",
                "photo_licensed": "true",
                "per_page": min(limit, 50),
                "order_by": "votes",       # community-upvoted first
                "order": "desc",
                "locale": "en",
            })
            resp.raise_for_status()
            observations = resp.json().get("results", [])

            for obs in observations:
                if len(photos) >= limit:
                    break

                obs_photos = obs.get("photos", [])
                obs_taxon = obs.get("taxon", {})
                taxon_display = obs_taxon.get("name", latin_name)
                observer = obs.get("user", {}).get("login", "Unknown")

                for p in obs_photos:
                    if len(photos) >= limit:
                        break

                    pid = p.get("id")
                    if pid in seen_photo_ids:
                        continue
                    seen_photo_ids.add(pid)

                    # License check
                    lic = (p.get("license_code") or "").lower().replace("_", "-")
                    if lic not in _ALLOWED_LICENSES:
                        continue

                    # Build URLs — iNaturalist uses a suffix pattern
                    url_raw = p.get("url", "")
                    if not url_raw:
                        continue

                    # URL pattern: .../photos/{id}/{size}.{ext}
                    # The API returns "square" size by default
                    url_original = re.sub(
                        r"/square\.", "/original.", url_raw,
                    )
                    url_large = re.sub(
                        r"/square\.", "/large.", url_raw,
                    )

                    attribution = p.get("attribution", f"(c) {observer}")

                    photos.append(INatPhoto(
                        photo_id=pid,
                        url_original=url_original,
                        url_large=url_large,
                        attribution=attribution,
                        license_code=lic,
                        taxon_name=taxon_display,
                        observation_id=obs.get("id", 0),
                        quality_grade=quality,
                    ))

    return photos
