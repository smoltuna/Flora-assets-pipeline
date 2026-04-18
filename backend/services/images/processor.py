"""Image processing — download, resize, background removal via rembg.

Output specs (matching Flora xcassets conventions):
  • name-info.jpg  — artistic photograph, JPEG, 640–1024 px, quality 85, ~340 KB
  • name.png       — transparent-background blossom, PNG, 400–500 px, ~300 KB
  • name-lock.png  — flat silhouette (handled by lock_gen.py)
"""
from __future__ import annotations

import io
from pathlib import Path

import httpx
from PIL import Image

from services.images.wikimedia import WikimediaImage

_OUTPUT_DIR = Path("/tmp/flora_images")
_HEADERS = {
    "User-Agent": "FloraRAGPipeline/1.0 (portfolio; contact: simone.84858@gmail.com)"
}

# Info image: longest edge clamped to this range
_INFO_MAX_PX = 1024
_INFO_MIN_PX = 640

# Main/home image: longest edge after crop+pad, targeting 400–500 px
_MAIN_TARGET_PX = 492


async def _download(url: str) -> bytes:
    async with httpx.AsyncClient(
        timeout=60.0, headers=_HEADERS, follow_redirects=True,
    ) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        return resp.content


def _slug(latin_name: str) -> str:
    return latin_name.replace(" ", "_").lower()


async def process_info_image(
    img: WikimediaImage, latin_name: str,
) -> tuple[str, str]:
    """Download and prepare the info/detail-screen image.

    Output: JPEG, longest edge between 640–1024 px, quality 85.
    Target file size ~340 KB.
    Returns (file_path, author_string).
    """
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    raw = await _download(img.url)

    pil = Image.open(io.BytesIO(raw)).convert("RGB")

    # Only downscale if larger than max; never upscale
    if max(pil.size) > _INFO_MAX_PX:
        pil.thumbnail((_INFO_MAX_PX, _INFO_MAX_PX), Image.LANCZOS)

    # If the image is smaller than the min, it still passes — we accepted
    # it during search (min 500 px short side) and upscaling hurts quality.

    out_path = _OUTPUT_DIR / f"{_slug(latin_name)}-info.jpg"
    pil.save(str(out_path), "JPEG", quality=85, optimize=True)

    return str(out_path), img.author


async def process_main_image(img: WikimediaImage, latin_name: str) -> str:
    """Download, remove background, crop and resize the home / widget image.

    Output: PNG, transparent background, longest edge ~400–500 px.
    The image is the flower with its petals — not a rectangular photo.

    Quality gate: if rembg removes > 90 % of pixels (subject lost in a wide
    habitat shot), the image is used as-is without background removal so
    downstream lock_gen still gets usable input.
    """
    import numpy as np
    from rembg import remove

    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    raw = await _download(img.url)

    no_bg = remove(raw)
    result = Image.open(io.BytesIO(no_bg)).convert("RGBA")

    alpha = np.array(result)[:, :, 3]
    if (alpha < 10).mean() > 0.90:
        # rembg stripped the plant — use original without removal
        result = Image.open(io.BytesIO(raw)).convert("RGBA")

    # Crop to non-transparent bounding box
    bbox = result.getbbox()
    if bbox:
        result = result.crop(bbox)

    # Re-add 5 % padding on each side for breathing room
    w, h = result.size
    pad = max(int(min(w, h) * 0.05), 4)
    padded = Image.new("RGBA", (w + 2 * pad, h + 2 * pad), (0, 0, 0, 0))
    padded.paste(result, (pad, pad))

    # Final size cap: target 400–500 px (Flora xcassets convention)
    padded.thumbnail((_MAIN_TARGET_PX, _MAIN_TARGET_PX), Image.LANCZOS)

    out_path = _OUTPUT_DIR / f"{_slug(latin_name)}.png"
    padded.save(str(out_path), "PNG", optimize=True)

    return str(out_path)
