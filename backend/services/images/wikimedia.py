"""Wikimedia Commons image search for plant photographs.

Search strategy — cast a wide net, then score aggressively:
  1. Category:{latin_name}          (most precise; well-curated for many species)
  2. Category:{genus}               (broader genus-level category)
  3. Text search: "{latin_name}"    (catches mis-categorised files)
  4. Text search: "{genus} flower"  (last-resort fallback)

All queries run until we accumulate ≥ MIN_CANDIDATES (40) images, then stop.
Each candidate is scored independently for two roles:

  • info    — artistic / landscape photograph for the detail screen
  • blossom — tight close-up of the flower head for rembg → home.png

Scoring uses **all available metadata**: title, description, categories,
dimensions, file size — not just the title.  This dramatically improves
selection quality on Wikimedia where many excellent photos have generic
titles like "File:Rosa canina 003.jpg".
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import httpx

_API = "https://commons.wikimedia.org/w/api.php"
_HEADERS = {
    "User-Agent": "FloraRAGPipeline/1.0 (portfolio; contact: simone.84858@gmail.com)"
}

_ALLOWED_LICENSES = frozenset({
    "cc0", "cc-by", "cc by", "cc-by-sa", "cc by-sa",
    "public domain", "pd",
})

_ACCEPTED_MIME = frozenset({"image/jpeg", "image/png"})

# Min candidates before we stop issuing new search queries
_MIN_CANDIDATES = 40

# ---------------------------------------------------------------------------
# Skip patterns — non-photographic or wrong-subject content
# ---------------------------------------------------------------------------

_SKIP_RE = re.compile(
    r"illustration|drawing|painting|watercolor|lithograph|engraving|sketch"
    r"|herbarium|specimen|pressed|dried"
    r"|stamp|postage|colnect|rcin|barcode"
    r"|museum|naturalis"
    r"|distribution|range|\bmap\b"
    r"|\blogo\b|\bicon\b|\bclipart\b|\bdiagram\b"
    # Commercial / non-botanical contexts (shop displays, product labels, etc.)
    r"|\bposter\b|\bstore\b|\bshop\b|\bcollage\b|\bpanel\b|\bpackaging\b|\blabel\b"
    # Camera dump filenames ("Batch", "DSC_", "IMG_" with a number)
    r"|\bbatch\b"
    # Animals / insects — often photographed on flowers but wrong subject
    r"|bombus|apis|butterfly|moth|bee\b|bumblebee|insect|beetle|spider|bird"
    r"|caterpillar|larvae|larva|hymenoptera|lepidoptera|coleoptera|diptera"
    r"|pollinator|pollinat"
    # Common butterfly genera that appear on flower photos
    r"|\bvanessa\b|\bpapilio\b|\bpieris\b|\bgonepteryx\b|\baglais\b"
    # Person-centred photos (title is the primary clue — "girl", "woman", etc.)
    r"|\bgirl\b|\bwoman\b|\bman\b|\bboy\b|\bchild\b|\bperson\b|\bpeople\b"
    r"|\bdívka\b|\bchica\b|\bfrau\b|\bhomme\b|\bfemme\b"
    r"|\d{7,}",  # long numeric IDs (scan barcodes, Flickr IDs, etc.)
    re.IGNORECASE,
)

# Additional skip for description text (more lenient — descriptions are long)
_DESC_SKIP_RE = re.compile(
    r"herbarium sheet|pressed plant|dried specimen|botanical illustration"
    r"|line drawing|pen and ink|woodcut"
    # Commercial / display contexts
    r"|\bposter\b|\bstore\b|\bshop\b|\bcollage\b"
    r"|collection of (?:flowers|plants)|mixed flowers",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Positive-signal patterns for scoring
# ---------------------------------------------------------------------------

_FLOWER_RE = re.compile(
    r"\bflower|bloom|blossom|inflorescence|petal|floral\b", re.IGNORECASE
)
_CLOSEUP_RE = re.compile(
    r"\bmacro\b|\bclose.?up\b|\bdetail\b|\bnahaufnahme\b", re.IGNORECASE
)
_SCENIC_RE = re.compile(
    r"\bgarden|field|meadow|habitat|landscape|nature|wild|hedgerow|path\b",
    re.IGNORECASE,
)
_NONFLOWER_PARTS_RE = re.compile(
    r"\bleaf\b|\bleaves\b|\bstem\b|\bbranch\b|\broot\b"
    r"|\bfruit\b|\bseed\b|\bbark\b|\bthorn\b|\bhip\b|\bberry\b",
    re.IGNORECASE,
)
# Quality-of-Commons indicators — images from known quality reviews
_QUALITY_RE = re.compile(
    r"quality.?image|featured.?picture|valued.?image|wiki.?loves",
    re.IGNORECASE,
)


@dataclass
class WikimediaImage:
    title: str
    url: str
    author: str
    license: str
    width: int
    height: int
    size_bytes: int
    description: str = ""
    categories: str = ""

    @property
    def aspect(self) -> float:
        """width / height."""
        return self.width / self.height if self.height else 1.0

    @property
    def megapixels(self) -> float:
        return (self.width * self.height) / 1_000_000

    @property
    def short_side(self) -> int:
        return min(self.width, self.height)

    @property
    def _text(self) -> str:
        """Combined searchable text for scoring."""
        return f"{self.title} {self.description} {self.categories}"


@dataclass
class ImagePair:
    info: WikimediaImage      # artistic / landscape — for the detail/info screen
    blossom: WikimediaImage   # close-up flower head — for home.png after rembg


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def _parse_image(page: dict) -> WikimediaImage | None:
    """Validate and convert an API page dict to a WikimediaImage.

    Returns None on license, MIME, resolution, size, or title failures.
    """
    info_list = page.get("imageinfo", [])
    if not info_list:
        return None
    info = info_list[0]
    meta = info.get("extmetadata", {})

    # License
    raw_license = (
        meta.get("LicenseShortName", {}).get("value", "") or ""
    ).lower()
    if not any(lic in raw_license for lic in _ALLOWED_LICENSES):
        return None

    # MIME — raster only, no SVG/TIF
    mime = info.get("mime", "").lower()
    if mime not in _ACCEPTED_MIME:
        return None

    # Dimensions — need at least 500px on the short side for decent quality
    width = info.get("width", 0)
    height = info.get("height", 0)
    if min(width, height) < 500:
        return None

    # File size: skip stubs (<30 KB) and huge RAW scans (>15 MB)
    size = info.get("size", 0)
    if not (30_000 <= size <= 15_000_000):
        return None

    title = page.get("title", "")
    if _SKIP_RE.search(title):
        return None

    # Extract description and categories for richer scoring
    description = re.sub(
        r"<[^>]+>", "",
        meta.get("ImageDescription", {}).get("value", "") or "",
    ).strip()
    categories = (
        meta.get("Categories", {}).get("value", "") or ""
    ).replace("|", " ")

    # Skip if description clearly indicates non-photo content
    if _DESC_SKIP_RE.search(description):
        return None

    author_raw = (
        meta.get("Artist", {}).get("value")
        or meta.get("Credit", {}).get("value")
        or "Unknown"
    )
    author = re.sub(r"<[^>]+>", "", author_raw).strip() or "Unknown"

    return WikimediaImage(
        title=title,
        url=info.get("url", ""),
        author=author,
        license=raw_license,
        width=width,
        height=height,
        size_bytes=size,
        description=description,
        categories=categories,
    )


# ---------------------------------------------------------------------------
# Search helpers
# ---------------------------------------------------------------------------

_IMAGEINFO_PARAMS = {
    "prop": "imageinfo",
    "iiprop": "url|size|extmetadata|mime",
    "format": "json",
}


async def _category_search(
    client: httpx.AsyncClient, category: str, limit: int = 50,
) -> list[WikimediaImage]:
    """Fetch images from a Wikimedia Commons category."""
    resp = await client.get(_API, params={
        **_IMAGEINFO_PARAMS,
        "action": "query",
        "generator": "categorymembers",
        "gcmtitle": f"Category:{category}",
        "gcmnamespace": 6,
        "gcmlimit": limit,
    })
    resp.raise_for_status()
    pages = resp.json().get("query", {}).get("pages", {})
    return [img for p in pages.values() if (img := _parse_image(p))]


async def _text_search(
    client: httpx.AsyncClient, query: str, limit: int = 40,
) -> list[WikimediaImage]:
    """Full-text search for images on Wikimedia Commons."""
    resp = await client.get(_API, params={
        **_IMAGEINFO_PARAMS,
        "action": "query",
        "generator": "search",
        "gsrnamespace": 6,
        "gsrsearch": query,
        "gsrlimit": limit,
    })
    resp.raise_for_status()
    pages = resp.json().get("query", {}).get("pages", {})
    return [img for p in pages.values() if (img := _parse_image(p))]


# ---------------------------------------------------------------------------
# Scoring — info role (artistic photograph for detail screen)
# ---------------------------------------------------------------------------

def _score_info(img: WikimediaImage) -> float:
    """Score for the info/detail-screen role.

    Goal: artistic photograph showing the plant in context.
    Ideal: landscape or wide-square, high resolution, scenic setting,
    natural colours.  JPEG ~340 KB, 640–1024 px.
    """
    score = 0.0
    text = img._text.lower()

    # --- Aspect ratio: landscape or wide-square is ideal ---
    ar = img.aspect
    if 1.2 <= ar <= 1.8:          # classic landscape
        score += 5.0
    elif 0.9 <= ar <= 2.2:        # acceptable range
        score += 3.0
    elif 0.7 <= ar <= 2.8:
        score += 1.0
    if ar > 2.8:                  # extreme panorama
        score -= 3.0

    # --- Resolution: reward high-res originals ---
    mp = img.megapixels
    if mp >= 4.0:
        score += 4.0
    elif mp >= 2.0:
        score += 3.0
    elif mp >= 1.0:
        score += 2.0
    else:
        score += 1.0

    # --- File size sweet spot (natural JPEG photo range) ---
    kb = img.size_bytes / 1024
    if 150 <= kb <= 2000:
        score += 2.0
    elif 80 <= kb <= 4000:
        score += 1.0

    # --- Content signals from title + description + categories ---
    # Scenic / artistic shots
    if _SCENIC_RE.search(text):
        score += 3.0
    # Flower-related content (basic relevance)
    if _FLOWER_RE.search(text):
        score += 2.0
    # Quality badges on Commons
    if _QUALITY_RE.search(text):
        score += 4.0

    # --- Penalties ---
    # Close-ups belong in the blossom role, not info
    if _CLOSEUP_RE.search(text):
        score -= 3.0
    # Non-flower parts are less appealing for an artistic shot
    if _NONFLOWER_PARTS_RE.search(text):
        score -= 2.0

    # JPEG preferred (natural photos vs PNG diagrams/screenshots)
    if img.url.lower().endswith((".jpg", ".jpeg")):
        score += 1.0

    return score


# ---------------------------------------------------------------------------
# Scoring — blossom role (close-up for rembg → home.png)
# ---------------------------------------------------------------------------

def _score_blossom(img: WikimediaImage) -> float:
    """Score for the blossom/home-png role.

    Goal: tight close-up of flower head, ideally a single bloom filling
    the frame.  Square or portrait orientation works best for rembg
    background removal and subsequent cropping.
    Ideal: 400–500 px output after rembg, PNG with transparency.
    """
    score = 0.0
    text = img._text.lower()

    # --- Aspect ratio: single clear scale, no stacking ---
    # Near-square/portrait = flower fills the frame = rembg works well.
    # Landscape = flower is small against context = rembg fails.
    ar = img.aspect
    if 0.65 <= ar <= 1.25:        # ideal — near-square or slight portrait
        score += 8.0
    elif ar < 0.65:               # tall/narrow portrait — unusual but acceptable
        score += 2.0
    elif ar <= 1.5:               # mild landscape — below average
        score -= 2.0
    else:                         # strong landscape — very bad for rembg
        score -= 8.0

    # --- Short-side resolution (rembg quality scales with input) ---
    short = img.short_side
    if short >= 1500:
        score += 4.0
    elif short >= 1000:
        score += 3.0
    elif short >= 700:
        score += 2.0
    elif short >= 500:
        score += 1.0

    # --- Content signals: close-up / macro / single flower ---
    if _CLOSEUP_RE.search(text):
        score += 5.0
    if _FLOWER_RE.search(text):
        score += 3.0

    # Single-flower indicators
    if re.search(r"\bsingle\b|\bisolated\b|\bone\b", text):
        score += 3.0
    # White/plain background (great for rembg)
    if re.search(r"\bwhite.?background\b|\bisolated\b|\bstudio\b", text):
        score += 4.0

    # Quality badges
    if _QUALITY_RE.search(text):
        score += 3.0

    # --- Penalties ---
    # Non-flower parts → rembg gives poor masks
    if _NONFLOWER_PARTS_RE.search(text):
        score -= 5.0
    # Habitat / landscape → subject too small for rembg
    if _SCENIC_RE.search(text):
        score -= 3.0
    # Multiple plants / group shots → messy rembg output
    if re.search(r"\bfield\b|\bmeadow\b|\bmany\b|\bgroup\b|\bmass\b", text):
        score -= 3.0

    # File size: close-ups tend to be moderate-sized JPEGs
    kb = img.size_bytes / 1024
    if 100 <= kb <= 1500:
        score += 1.0

    return score


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def find_images(latin_name: str) -> ImagePair:
    """Return the best (info, blossom) image pair from Wikimedia Commons.

    Searches with multiple strategies to build a large candidate pool,
    then scores each candidate independently for both roles.

    Raises ValueError when no usable images can be found.
    """
    genus = latin_name.split()[0]

    async with httpx.AsyncClient(timeout=30.0, headers=_HEADERS) as client:
        candidates: list[WikimediaImage] = []
        seen: set[str] = set()

        def _add(imgs: list[WikimediaImage]) -> None:
            for img in imgs:
                if img.title not in seen and img.url:
                    seen.add(img.title)
                    candidates.append(img)

        # 1. Species category — most precise; well-curated for most species
        _add(await _category_search(client, latin_name, limit=50))

        # 2. Full-text search with exact latin name (catches mis-categorised files)
        if len(candidates) < _MIN_CANDIDATES:
            _add(await _text_search(client, f'"{latin_name}"', limit=40))

        # 3. Genus + "flower" text search — fills gaps without the noise of
        #    the broad genus category (Category:Rosa has thousands of unrelated images)
        if len(candidates) < _MIN_CANDIDATES:
            _add(await _text_search(client, f"{genus} flower", limit=30))

        # 4. Plain genus text search as last resort
        if len(candidates) < _MIN_CANDIDATES:
            _add(await _text_search(client, genus, limit=30))

    if not candidates:
        raise ValueError(
            f"No suitable Wikimedia Commons images found for {latin_name!r}"
        )

    # Score every candidate for both roles
    scored_info = sorted(candidates, key=_score_info, reverse=True)
    scored_blossom = sorted(candidates, key=_score_blossom, reverse=True)

    best_info = scored_info[0]

    # Pick a different image for blossom when possible
    best_blossom = next(
        (img for img in scored_blossom if img.title != best_info.title),
        scored_blossom[0],
    )

    return ImagePair(info=best_info, blossom=best_blossom)
