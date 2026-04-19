"""Unified multi-source image search orchestrator.

Combines candidates from Wikimedia Commons and iNaturalist, scores them
uniformly for two roles (info / blossom), and returns the best pair.

Output spec (matching Flora xcassets conventions):
  • name-info.jpg   — artistic photograph with author attribution
                      (~340 KB, JPEG, 640–1024 px)
  • name.png        — transparent-background blossom shot
                      (~300 KB, PNG, 400–500 px)
  • name-lock.png   — flat/minimalistic silhouette for lock screen widget
                      (~42 KB, PNG, 200×200 px)
"""
from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import structlog

from services.images.wikimedia import WikimediaImage, ImagePair, search_wikimedia
from services.images.inaturalist import INatPhoto, search_inaturalist

log = structlog.get_logger()

# ---------------------------------------------------------------------------
# Unified candidate wrapper — normalises Wikimedia and iNaturalist photos
# ---------------------------------------------------------------------------

@runtime_checkable
class ImageCandidate(Protocol):
    """Minimal interface both WikimediaImage and INatPhoto satisfy."""
    url: str
    author: str
    width: int
    height: int
    source: str

    @property
    def aspect(self) -> float: ...


@dataclass
class _UnifiedCandidate:
    """Normalised wrapper so scoring functions work on any source."""
    url: str               # download URL (original / full-res)
    author: str            # attribution string
    license: str
    width: int
    height: int
    size_bytes: int        # 0 if unknown (iNaturalist doesn't report)
    title: str             # filename or observation description
    description: str       # rich text / categories combined
    source: str            # "wikimedia" | "inaturalist"
    quality_badge: bool    # Wikimedia featured/quality image, or iNat research grade

    # Keep original object for downstream processing
    _original: WikimediaImage | INatPhoto | None = None

    @property
    def aspect(self) -> float:
        return self.width / self.height if self.height else 1.0

    @property
    def megapixels(self) -> float:
        return (self.width * self.height) / 1_000_000

    @property
    def short_side(self) -> int:
        return min(self.width, self.height)

    @property
    def text(self) -> str:
        return f"{self.title} {self.description}".lower()


def _wrap_wikimedia(img: WikimediaImage) -> _UnifiedCandidate:
    quality = bool(re.search(
        r"quality.?image|featured.?picture|valued.?image|wiki.?loves",
        img._text, re.IGNORECASE,
    ))
    return _UnifiedCandidate(
        url=img.url,
        author=img.author,
        license=img.license,
        width=img.width,
        height=img.height,
        size_bytes=img.size_bytes,
        title=img.title,
        description=f"{img.description} {img.categories}",
        source="wikimedia",
        quality_badge=quality,
        _original=img,
    )


def _wrap_inaturalist(p: INatPhoto) -> _UnifiedCandidate:
    return _UnifiedCandidate(
        url=p.url_original,
        author=p.attribution,
        license=p.license_code,
        width=p.width,
        height=p.height,
        size_bytes=0,  # unknown
        title=f"iNaturalist observation {p.observation_id} {p.taxon_name}",
        description=p.taxon_name,
        source="inaturalist",
        quality_badge=(p.quality_grade == "research"),
        _original=p,
    )


# ---------------------------------------------------------------------------
# Positive-signal patterns for scoring
# ---------------------------------------------------------------------------

_FLOWER_RE = re.compile(
    r"\bflower\b|\bbloom\b|\bblossom\b|\binflorescence\b|\bpetal\b|\bfloral\b",
    re.IGNORECASE,
)
_CLOSEUP_RE = re.compile(
    r"\bmacro\b|\bclose.?up\b|\bdetail\b|\bnahaufnahme\b|\bclose\b",
    re.IGNORECASE,
)
_SCENIC_RE = re.compile(
    r"\bgarden\b|\bfield\b|\bmeadow\b|\bhabitat\b|\blandscape\b|\bnature\b"
    r"|\bwild\b|\bhedgerow\b|\bpath\b|\bforest\b|\bwoodland\b",
    re.IGNORECASE,
)
_NONFLOWER_PARTS_RE = re.compile(
    r"\bleaf\b|\bleaves\b|\bstem\b|\bbranch\b|\broot\b"
    r"|\bfruit\b|\bseed\b|\bbark\b|\bthorn\b|\bhip\b|\bberry\b",
    re.IGNORECASE,
)
_GROUP_RE = re.compile(
    r"\bfield\b|\bmeadow\b|\bmany\b|\bgroup\b|\bmass\b|\bpatch\b|\bbunch\b",
    re.IGNORECASE,
)
_SINGLE_RE = re.compile(
    r"\bsingle\b|\bisolated\b|\bone\b|\bindividual\b", re.IGNORECASE,
)
_STUDIO_RE = re.compile(
    r"\bwhite.?background\b|\bisolated\b|\bstudio\b|\bcutout\b", re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Scoring — info role (artistic photograph for detail screen)
# ---------------------------------------------------------------------------

def _score_info(c: _UnifiedCandidate) -> float:
    """Score for the info/detail-screen role.

    Goal: artistic photograph showing the plant in context.
    Ideal: landscape or wide-square, high resolution, scenic setting,
    natural colours.  JPEG ~340 KB, 640–1024 px.
    """
    score = 0.0
    text = c.text

    # --- Source bonus: Wikimedia curated photos tend to be more artistic ---
    if c.source == "wikimedia":
        score += 2.0

    # --- Aspect ratio: landscape or wide-square is ideal ---
    ar = c.aspect
    if 1.2 <= ar <= 1.8:          # classic landscape
        score += 5.0
    elif 0.9 <= ar <= 2.2:        # acceptable range
        score += 3.0
    elif 0.7 <= ar <= 2.8:
        score += 1.0
    if ar > 2.8:                  # extreme panorama
        score -= 3.0

    # --- Resolution: reward high-res originals ---
    mp = c.megapixels
    if mp >= 4.0:
        score += 4.0
    elif mp >= 2.0:
        score += 3.0
    elif mp >= 1.0:
        score += 2.0
    else:
        score += 1.0

    # --- File size sweet spot (natural JPEG photo range) ---
    if c.size_bytes > 0:
        kb = c.size_bytes / 1024
        if 150 <= kb <= 2000:
            score += 2.0
        elif 80 <= kb <= 4000:
            score += 1.0

    # --- Content signals ---
    if _SCENIC_RE.search(text):
        score += 3.0
    if _FLOWER_RE.search(text):
        score += 2.0
    if c.quality_badge:
        score += 5.0   # stronger bonus for quality/featured images

    # --- Penalties ---
    if _CLOSEUP_RE.search(text):
        score -= 3.0
    if _NONFLOWER_PARTS_RE.search(text):
        score -= 2.0

    # JPEG preferred for natural photos
    if c.url.lower().endswith((".jpg", ".jpeg")):
        score += 1.0

    return score


# ---------------------------------------------------------------------------
# Scoring — blossom role (close-up for rembg → home.png)
# ---------------------------------------------------------------------------

def _score_blossom(c: _UnifiedCandidate) -> float:
    """Score for the blossom/home-png role.

    Goal: tight close-up of a single flower head, ideally filling the frame.
    Square or portrait orientation works best for rembg background removal.
    """
    score = 0.0
    text = c.text

    # --- Source bonus: iNaturalist photos are typically close-up shots ---
    if c.source == "inaturalist":
        score += 3.0

    # --- Aspect ratio: near-square/portrait = flower fills frame ---
    ar = c.aspect
    if 0.65 <= ar <= 1.25:        # ideal near-square or slight portrait
        score += 8.0
    elif ar < 0.65:               # tall/narrow portrait
        score += 2.0
    elif ar <= 1.5:               # mild landscape
        score -= 2.0
    else:                         # strong landscape — terrible for rembg
        score -= 8.0

    # --- Short-side resolution ---
    short = c.short_side
    if short >= 1500:
        score += 4.0
    elif short >= 1000:
        score += 3.0
    elif short >= 700:
        score += 2.0
    elif short >= 500:
        score += 1.0

    # --- Content signals: close-up, single flower ---
    if _CLOSEUP_RE.search(text):
        score += 5.0
    if _FLOWER_RE.search(text):
        score += 3.0
    if _SINGLE_RE.search(text):
        score += 3.0
    if _STUDIO_RE.search(text):
        score += 4.0   # white/studio background — rembg heaven
    if c.quality_badge:
        score += 3.0

    # --- Penalties ---
    if _NONFLOWER_PARTS_RE.search(text):
        score -= 5.0
    if _SCENIC_RE.search(text):
        score -= 3.0
    if _GROUP_RE.search(text):
        score -= 3.0

    # File size for close-ups
    if c.size_bytes > 0:
        kb = c.size_bytes / 1024
        if 100 <= kb <= 1500:
            score += 1.0

    return score


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def find_images(latin_name: str) -> ImagePair:
    """Return the best (info, blossom) image pair from all sources.

    Searches Wikimedia Commons and iNaturalist concurrently, merges
    candidates, scores for both roles, and picks the best pair.

    Raises ValueError when no usable images can be found.
    """
    # Run both searches concurrently
    wiki_results, inat_results = await asyncio.gather(
        search_wikimedia(latin_name),
        search_inaturalist(latin_name),
        return_exceptions=True,
    )

    # Handle individual source failures gracefully
    if isinstance(wiki_results, Exception):
        log.warning("search.wikimedia_failed", error=str(wiki_results))
        wiki_results = []
    if isinstance(inat_results, Exception):
        log.warning("search.inaturalist_failed", error=str(inat_results))
        inat_results = []

    # Wrap into unified candidates
    candidates: list[_UnifiedCandidate] = []
    for img in wiki_results:
        candidates.append(_wrap_wikimedia(img))
    for photo in inat_results:
        candidates.append(_wrap_inaturalist(photo))

    log.info(
        "search.candidates",
        latin_name=latin_name,
        wikimedia=len(wiki_results) if isinstance(wiki_results, list) else 0,
        inaturalist=len(inat_results) if isinstance(inat_results, list) else 0,
        total=len(candidates),
    )

    if not candidates:
        raise ValueError(
            f"No suitable images found for {latin_name!r} from any source"
        )

    # Score every candidate for both roles (species-in-title bonus applied to both)
    def _species_bonus(c: _UnifiedCandidate) -> float:
        title = c.title.lower()
        genus = latin_name.split()[0].lower()
        if latin_name.lower() in title:
            return 12.0
        if genus in title:
            return 6.0
        return -4.0  # no species/genus in title — deprioritise

    scored_info = sorted(
        candidates,
        key=lambda c: _score_info(c) + _species_bonus(c),
        reverse=True,
    )
    scored_blossom = sorted(
        candidates,
        key=lambda c: _score_blossom(c) + _species_bonus(c),
        reverse=True,
    )

    best_info_c = scored_info[0]
    # Pick a different image for blossom when possible
    best_blossom_c = next(
        (c for c in scored_blossom if c.url != best_info_c.url),
        scored_blossom[0],
    )

    log.info(
        "search.selected",
        info_source=best_info_c.source,
        info_score=round(_score_info(best_info_c), 1),
        info_url=best_info_c.url[:80],
        blossom_source=best_blossom_c.source,
        blossom_score=round(_score_blossom(best_blossom_c), 1),
        blossom_url=best_blossom_c.url[:80],
    )

    # Convert back to WikimediaImage for processor compatibility
    info_img = _to_wikimedia_image(best_info_c)
    blossom_img = _to_wikimedia_image(best_blossom_c)

    # Return top N blossom candidates for Gemini vision to re-rank
    _MAX_BLOSSOM_CANDIDATES = 4
    top_blossom = []
    seen_urls = set()
    for c in scored_blossom:
        if c.url not in seen_urls:
            seen_urls.add(c.url)
            top_blossom.append(_to_wikimedia_image(c))
        if len(top_blossom) >= _MAX_BLOSSOM_CANDIDATES:
            break

    return ImagePair(
        info=info_img,
        blossom=blossom_img,
        blossom_candidates=top_blossom,
    )


def _to_wikimedia_image(c: _UnifiedCandidate) -> WikimediaImage:
    """Convert a unified candidate back to WikimediaImage for processor.py."""
    if isinstance(c._original, WikimediaImage):
        return c._original
    # For iNaturalist photos, wrap in a WikimediaImage-compatible object
    return WikimediaImage(
        title=c.title,
        url=c.url,
        author=c.author,
        license=c.license,
        width=c.width,
        height=c.height,
        size_bytes=c.size_bytes if c.size_bytes > 0 else 500_000,
        description=c.description,
        categories="",
    )
