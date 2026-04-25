"""Image processing — download, fal.ai vision judging, rembg background removal.

Output specs (matching Flora xcassets conventions):
  • name-info.jpg  — artistic photograph, JPEG, 640–1024 px, quality 85, ~340 KB
  • name.png       — transparent-background blossom, PNG, 400–500 px, ~300 KB
  • name-lock.png  — fal.ai FLUX-generated flat icon (handled by lock_gen.py)

Key improvement over pure-metadata scoring: the blossom image is selected
by sending up to 4 candidate URLs to fal.ai Llava-Next, which scores each
on how well the petals fill the frame with the simplest background.
Falls back to the metadata-scored top pick if fal.ai is unavailable.
"""
from __future__ import annotations

import asyncio
import io
import re
from pathlib import Path

import httpx
import numpy as np
from PIL import Image, ImageOps

import structlog

from services.images.wikimedia import WikimediaImage

log = structlog.get_logger()

_OUTPUT_DIR = Path("/tmp/flora_images")
_XCASSETS_DIR = Path(__file__).parents[3] / "output" / "FlowerAssets.xcassets"
_HEADERS = {
    "User-Agent": "FloraRAGPipeline/1.0 (portfolio; contact: simone.84858@gmail.com)"
}

_INFO_MAX_PX = 1024
_MAIN_TARGET_PX = 600


_WIKIMEDIA_RE = re.compile(
    r"(https://upload\.wikimedia\.org/wikipedia/commons/)"
    r"([0-9a-f]/[0-9a-f]{2})/"
    r"(.+)$",
    re.IGNORECASE,
)


def _thumb_url(url: str, width: int) -> str:
    """Convert a Wikimedia full-res URL to a CDN thumbnail URL.

    https://upload.wikimedia.org/wikipedia/commons/X/XX/file.jpg
    → https://upload.wikimedia.org/wikipedia/commons/thumb/X/XX/file.jpg/{width}px-file.jpg

    Returns url unchanged for non-Wikimedia or already-thumb URLs.
    """
    if "/thumb/" in url:
        return url
    m = _WIKIMEDIA_RE.match(url)
    if not m:
        return url
    base, path, filename = m.groups()
    return f"{base}thumb/{path}/{filename}/{width}px-{filename}"


async def _download(url: str, *, retries: int = 3) -> bytes:
    """Download with exponential backoff on 429 rate-limit responses."""
    delay = 2.0
    async with httpx.AsyncClient(
        timeout=60.0, headers=_HEADERS, follow_redirects=True,
    ) as client:
        for attempt in range(retries):
            resp = await client.get(url)
            if resp.status_code == 429 and attempt < retries - 1:
                log.warning("download.rate_limited", url=url[:80], retry=attempt + 1, wait=delay)
                await asyncio.sleep(delay)
                delay *= 2
                continue
            resp.raise_for_status()
            return resp.content
    resp.raise_for_status()
    return resp.content  # type: ignore[return-value]


def _slug(latin_name: str) -> str:
    return latin_name.replace(" ", "-").lower()


def _write_imageset_contents(imageset_dir: Path, filename: str) -> None:
    """Write the Xcode imageset Contents.json (all 3 scales point to the same file)."""
    import json
    imageset_dir.mkdir(parents=True, exist_ok=True)
    contents = {
        "images": [
            {"filename": filename, "idiom": "universal", "scale": "1x"},
            {"filename": filename, "idiom": "universal", "scale": "2x"},
            {"filename": filename, "idiom": "universal", "scale": "3x"},
        ],
        "info": {"author": "xcode", "version": 1},
    }
    (imageset_dir / "Contents.json").write_text(
        json.dumps(contents, indent=2) + "\n", encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# Info image — artistic photo for the detail screen
# ---------------------------------------------------------------------------

async def process_info_image(
    img: WikimediaImage, latin_name: str,
) -> tuple[str, str]:
    """Download and prepare the info/detail-screen image.

    Output: JPEG, longest edge between 640–1024 px, quality 85.
    Returns (file_path, author_string).
    """
    slug = _slug(latin_name)
    fetch_url = img.thumb_url or _thumb_url(img.url, _INFO_MAX_PX)
    raw = await _download(fetch_url)

    pil = Image.open(io.BytesIO(raw)).convert("RGB")

    if max(pil.size) > _INFO_MAX_PX:
        pil.thumbnail((_INFO_MAX_PX, _INFO_MAX_PX), Image.LANCZOS)

    imageset_dir = _XCASSETS_DIR / f"{slug}-info.imageset"
    imageset_dir.mkdir(parents=True, exist_ok=True)
    out_path = imageset_dir / "info.jpg"
    pil.save(str(out_path), "JPEG", quality=85, optimize=True)
    _write_imageset_contents(imageset_dir, "info.jpg")

    return f"{slug}-info", img.author


# ---------------------------------------------------------------------------
# fal.ai vision — pick the best blossom photo from N candidates
# ---------------------------------------------------------------------------

def _make_thumb(raw: bytes, max_px: int = 800) -> bytes:
    """Shrink to max_px for vision scoring — saves bandwidth and latency."""
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    img.thumbnail((max_px, max_px), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=75)
    return buf.getvalue()


async def _fal_pick_best(
    candidate_bytes: list[bytes],
    latin_name: str,
    fal_key: str = "",
) -> int:
    """Use fal.ai OpenRouter vision to pick the best flower photo for background removal.

    Uploads thumbnails to fal.ai storage (Wikimedia URLs are blocked by fal),
    scores each candidate individually on a 1–10 scale, returns index of best.
    Falls back to 0 (metadata-scored top pick) on any error or missing key.
    """
    if not fal_key or len(candidate_bytes) <= 1:
        return 0

    import os
    import fal_client

    os.environ["FAL_KEY"] = fal_key

    prompt = (
        f"Rate this flower photo for use in a flower widget app showing {latin_name}.\n"
        "Score 1–10 based on:\n"
        "  + Petals clearly visible and filling most of the frame (most important)\n"
        "  + Background simple, plain, or blurred (easy to remove)\n"
        "  + Close-up or portrait shot of the flower head\n"
        "Deduct points for: insects visible, mainly leaves/buds/stem, "
        "wide landscape where the flower is tiny.\n"
        "Reply with ONLY a single integer 1–10."
    )

    scores: list[int] = []
    for raw in candidate_bytes:
        try:
            thumb = _make_thumb(raw)

            def _upload(data: bytes = thumb) -> str:
                return fal_client.upload(data, "image/jpeg")

            fal_url = await asyncio.to_thread(_upload)

            def _score(u: str = fal_url) -> dict:
                return fal_client.subscribe(
                    "openrouter/router/openai/v1/chat/completions",
                    arguments={
                        "model": "qwen/qwen3-vl-235b-a22b-instruct",
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "image_url", "image_url": {"url": u}},
                                    {"type": "text", "text": prompt},
                                ],
                            }
                        ],
                        "max_tokens": 10,
                    },
                )

            result = await asyncio.to_thread(_score)
            text = (
                (result.get("choices") or [{}])[0]
                .get("message", {})
                .get("content") or ""
            ).strip()
            m = re.search(r"\b(10|[1-9])\b", text)
            score = int(m.group()) if m else 1
            scores.append(score)
            log.info("fal.vision_score", score=score)
        except Exception as e:
            log.warning("fal.vision_failed", error=str(e))
            scores.append(0)

    if not scores or max(scores) == 0:
        return 0

    best = max(range(len(scores)), key=lambda i: scores[i])
    log.info("fal.vision_pick", chosen=best + 1, total=len(scores), scores=scores)
    return best


# ---------------------------------------------------------------------------
# Background removal quality gate
# ---------------------------------------------------------------------------

def _resize_fit_transparent(img: Image.Image, max_size: int) -> Image.Image:
    """Resize to fit within max_size × max_size, pad with transparency."""
    img = img.convert("RGBA")
    img.thumbnail((max_size, max_size), Image.LANCZOS)
    canvas = Image.new("RGBA", (max_size, max_size), (0, 0, 0, 0))
    offset = ((max_size - img.width) // 2, (max_size - img.height) // 2)
    canvas.paste(img, offset, img)
    return canvas


def _is_bad_mask(img_rgba: Image.Image, visible_pct: float) -> bool:
    """Return True if rembg produced a useless result (plant lost or dark blob)."""
    if visible_pct < 0.10:
        return True
    arr = np.array(img_rgba.convert("RGBA"), dtype=np.float32)
    visible = arr[arr[:, :, 3] > 10][:, :3]
    if len(visible) == 0:
        return True
    mean_brightness = visible.mean()
    return float(mean_brightness) < 30.0


# ---------------------------------------------------------------------------
# Main image — fal.ai-guided blossom selection + rembg cascade
# ---------------------------------------------------------------------------

async def process_main_image(
    img: WikimediaImage,
    latin_name: str,
    candidates: list[WikimediaImage] | None = None,
    fal_key: str = "",
) -> tuple[str, bytes | None]:
    """Download, pick best via fal.ai vision, remove background, crop, resize.

    Tries each candidate through rembg until one produces a clean mask.
    Uses birefnet-general model for better quality.

    Output: PNG, transparent background, max 492×492 px.
    Returns (file_path, raw_bytes_of_chosen_source).
    """
    from rembg import remove, new_session

    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_candidates = candidates or [img]
    if img not in all_candidates:
        all_candidates = [img] + all_candidates

    # Download up to 4 candidates — stagger requests to avoid Wikimedia 429
    downloaded: list[tuple[bytes, WikimediaImage]] = []
    for i, c in enumerate(all_candidates[:4]):
        if i > 0:
            await asyncio.sleep(0.5)
        try:
            fetch_url = c.thumb_url or _thumb_url(c.url, _MAIN_TARGET_PX * 2)
            raw = await _download(fetch_url)
            downloaded.append((raw, c))
        except Exception as e:
            log.warning("download.failed", url=c.url[:60], error=str(e))

    if not downloaded:
        raise ValueError(f"Could not download any blossom candidate for {latin_name}")

    # fal.ai vision re-ranks if we have multiple candidates
    if len(downloaded) > 1:
        bytes_list = [raw for raw, _ in downloaded]
        best_idx = await _fal_pick_best(bytes_list, latin_name, fal_key)
        chosen = downloaded[best_idx]
        ordered = [chosen] + [d for i, d in enumerate(downloaded) if i != best_idx]
    else:
        ordered = downloaded

    # Try rembg on each candidate until we get a clean mask
    try:
        session = new_session("birefnet-general")
    except Exception:
        log.warning("rembg.birefnet_unavailable")
        session = None

    for i, (raw_bytes, meta) in enumerate(ordered):
        try:
            raw_img = ImageOps.exif_transpose(Image.open(io.BytesIO(raw_bytes)))
            buf = io.BytesIO()
            raw_img.save(buf, format="JPEG")
            corrected = buf.getvalue()

            bg_removed = remove(corrected, session=session) if session else remove(corrected)
            img_rgba = Image.open(io.BytesIO(bg_removed)).convert("RGBA")

            alpha = np.array(img_rgba)[:, :, 3]
            visible_pct = float((alpha > 10).sum() / alpha.size)

            label = "fal pick" if i == 0 else f"fallback {i}"
            log.info(
                "rembg.result",
                label=label,
                visible_pct=f"{visible_pct:.1%}",
                bad=_is_bad_mask(img_rgba, visible_pct),
                title=meta.title[:50],
            )

            if not _is_bad_mask(img_rgba, visible_pct):
                result_img = _resize_fit_transparent(img_rgba, _MAIN_TARGET_PX)
                slug = _slug(latin_name)
                imageset_dir = _XCASSETS_DIR / f"{slug}.imageset"
                imageset_dir.mkdir(parents=True, exist_ok=True)
                out_path = imageset_dir / "home.png"
                result_img.save(str(out_path), "PNG", optimize=True)
                _write_imageset_contents(imageset_dir, "home.png")
                return str(out_path), raw_bytes

        except Exception as e:
            log.warning("rembg.failed", candidate=i, error=str(e))
            continue

    raise ValueError(
        f"No usable home image for {latin_name} — "
        f"rembg gave bad masks on all {len(ordered)} candidates"
    )
