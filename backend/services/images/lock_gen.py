"""Lock image generation — fal.ai FLUX Schnell flat botanical icon (200 × 200 px PNG).

Primary path: FLUX Schnell generates a purpose-built flat linocut icon using
the species name.  White background is threshold-removed to transparent.

Fallback: dominant-color silhouette from home.png if fal.ai is unavailable.

Target: ~42 KB, 200 × 200 px, transparent background.
"""
from __future__ import annotations

import asyncio
import io
import json as _json
from pathlib import Path

import httpx
import numpy as np
import structlog
from PIL import Image

log = structlog.get_logger()

_OUTPUT_DIR = Path("/tmp/flora_images")
_XCASSETS_DIR = Path(__file__).parents[3] / "output" / "FlowerAssets.xcassets"


# ---------------------------------------------------------------------------
# Dominant petal color — used for fallback silhouette
# ---------------------------------------------------------------------------

def _dominant_petal_color(img_rgba: Image.Image) -> tuple[int, int, int]:
    arr = np.array(img_rgba.convert("RGBA"), dtype=np.float32)

    visible_mask = arr[:, :, 3] > 200
    visible = arr[visible_mask][:, :3]
    if len(visible) == 0:
        visible_mask = arr[:, :, 3] > 50
        visible = arr[visible_mask][:, :3]
        if len(visible) == 0:
            return (200, 150, 180)

    pmax = visible.max(axis=1)
    pmin = visible.min(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        sat = np.where(pmax > 0, (pmax - pmin) / np.where(pmax > 0, pmax, 1.0), 0.0)
    colorful_mask = (sat > 0.12) & (pmax > 80) & (pmax < 240)
    colorful = visible[colorful_mask]
    if len(colorful) < 20:
        colorful = visible

    step = max(1, len(colorful) // 20_000)
    sample = colorful[::step].astype(np.uint8)
    swatch = Image.fromarray(sample.reshape(1, len(sample), 3), "RGB")
    quantized = swatch.quantize(colors=8, method=Image.Quantize.MEDIANCUT)
    raw_palette = quantized.getpalette() or []
    palette = np.array(raw_palette[:8 * 3], dtype=np.float32).reshape(8, 3)

    p_max = palette.max(axis=1)
    p_min = palette.min(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        p_sat = np.where(p_max > 0, (p_max - p_min) / np.where(p_max > 0, p_max, 1.0), 0.0)
    p_bright = p_max / 255.0
    valid_mask = (p_max > 60) & (p_max < 240)
    score = p_sat * p_bright * valid_mask

    best = int(score.argmax()) if score.max() > 0 else int(p_sat.argmax())
    r, g, b = palette[best]
    return (int(r), int(g), int(b))


# ---------------------------------------------------------------------------
# White-background removal — threshold + fringe propagation
# ---------------------------------------------------------------------------

def _remove_white_bg(img: Image.Image) -> Image.Image:
    img = img.convert("RGBA")
    arr = np.array(img, dtype=np.uint8)

    white = (arr[:, :, 0] > 240) & (arr[:, :, 1] > 240) & (arr[:, :, 2] > 240)
    arr[white, 3] = 0

    near_white = (arr[:, :, 0] > 220) & (arr[:, :, 1] > 220) & (arr[:, :, 2] > 220)
    for _ in range(6):
        transparent = arr[:, :, 3] == 0
        adj = (
            np.roll(transparent, 1, axis=0) | np.roll(transparent, -1, axis=0) |
            np.roll(transparent, 1, axis=1) | np.roll(transparent, -1, axis=1)
        )
        spill = near_white & adj
        if not spill.any():
            break
        arr[spill, 3] = 0
        near_white[spill] = False

    return Image.fromarray(arr)


# ---------------------------------------------------------------------------
# Resize helper — fit within square without stretching
# ---------------------------------------------------------------------------

def _fit_square(img: Image.Image, size: int = 200) -> Image.Image:
    img = img.convert("RGBA")
    img.thumbnail((size, size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    offset = ((size - img.width) // 2, (size - img.height) // 2)
    canvas.paste(img, offset, img)
    return canvas


# ---------------------------------------------------------------------------
# fal.ai FLUX Schnell generation
# ---------------------------------------------------------------------------

async def _flux_generate(
    latin_name: str, common_name: str, fal_key: str,
) -> bytes | None:
    """Generate a flat botanical icon via FLUX Schnell on fal.ai."""
    import os

    import fal_client

    os.environ["FAL_KEY"] = fal_key

    flower_label = f"{common_name} ({latin_name})" if common_name else latin_name

    prompt = (
        f"Flat botanical icon of a {flower_label}. "
        "Composition: one large flower head prominently at the top-center, "
        "a single straight stem, and 2-3 simple leaves below. "
        "The entire illustration — flower, petals, stem, leaves — is filled "
        "with one single solid black color. "
        "Pure white background. No gradients, no shading, no outlines, no second color anywhere. "
        "Bold graphic style like a linocut stamp or app icon. "
        "The flower must be clearly recognizable and fill most of the frame."
    )

    def _subscribe() -> dict:
        return fal_client.subscribe(
            "fal-ai/flux/schnell",
            arguments={
                "prompt": prompt,
                "image_size": "square_hd",
                "num_inference_steps": 4,
                "output_format": "png",
                "enable_safety_checker": False,
            },
        )

    try:
        result = await asyncio.to_thread(_subscribe)
        image_url = result["images"][0]["url"]
    except Exception as e:
        log.warning("flux.generation_failed", error=str(e))
        return None

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.get(image_url)
            resp.raise_for_status()
            return resp.content
    except Exception as e:
        log.warning("flux.download_failed", error=str(e))
        return None


# ---------------------------------------------------------------------------
# Fallback — dominant-color silhouette
# ---------------------------------------------------------------------------

def _fallback_silhouette(main_image_path: str) -> Image.Image:
    original = Image.open(main_image_path).convert("RGBA")
    color = _dominant_petal_color(original)
    arr = np.array(original)
    alpha = arr[:, :, 3].copy()
    flat = np.empty_like(arr)
    flat[:, :, 0] = color[0]
    flat[:, :, 1] = color[1]
    flat[:, :, 2] = color[2]
    flat[:, :, 3] = alpha
    return Image.fromarray(flat.astype(np.uint8), "RGBA")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def generate_lock_image(
    main_image_path: str,
    latin_name: str,
    *,
    common_name: str = "",
    fal_key: str = "",
) -> str:
    """Generate a flat botanical icon for the lock screen widget.

    1. FLUX Schnell (fal.ai) generates a black linocut icon on white background.
    2. White background is threshold-removed to transparent.
    3. Cropped and fit into 200×200 without stretching.
    Fallback: dominant-color silhouette from home.png.

    Output: PNG, transparent background, 200 × 200 px.
    """
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not common_name:
        common_name = ""

    if fal_key:
        generated = await _flux_generate(latin_name, common_name, fal_key)
        if generated:
            log.info("lock.flux_ok", latin_name=latin_name)
            img = Image.open(io.BytesIO(generated)).convert("RGBA")
            img = _remove_white_bg(img)
            bbox = img.getbbox()
            if bbox:
                img = img.crop(bbox)
            result = _fit_square(img, 200)
        else:
            log.info("lock.flux_failed_fallback", latin_name=latin_name)
            result = _fit_square(_fallback_silhouette(main_image_path), 200)
    else:
        log.info("lock.no_key_fallback", latin_name=latin_name)
        result = _fit_square(_fallback_silhouette(main_image_path), 200)

    slug = latin_name.replace(" ", "-").lower()
    imageset_dir = _XCASSETS_DIR / f"{slug}-lock.imageset"
    imageset_dir.mkdir(parents=True, exist_ok=True)
    out_path = imageset_dir / "lock.png"
    result.save(str(out_path), "PNG", optimize=True, compress_level=9)

    contents = {
        "images": [
            {"filename": "lock.png", "idiom": "universal", "scale": "1x"},
            {"filename": "lock.png", "idiom": "universal", "scale": "2x"},
            {"filename": "lock.png", "idiom": "universal", "scale": "3x"},
        ],
        "info": {"author": "xcode", "version": 1},
    }
    (imageset_dir / "Contents.json").write_text(
        _json.dumps(contents, indent=2) + "\n", encoding="utf-8"
    )

    return f"{slug}-lock"
