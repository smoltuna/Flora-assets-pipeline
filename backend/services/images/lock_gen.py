"""Lock image generation — flat single-color silhouette (200 × 200 px PNG).

Takes home.png (transparent-background flower) and fills every visible pixel
with the flower's dominant petal color, producing a flat iOS-widget icon.
The silhouette shape and alpha channel are preserved exactly; only the RGB
channels are replaced with a single solid color.

Target: ~42 KB, 200 × 200 px, transparent background.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

_OUTPUT_DIR = Path("/tmp/flora_images")


def _dominant_petal_color(img_rgba: Image.Image) -> tuple[int, int, int]:
    """Find the single most representative petal color.

    Strategy:
    1. Keep only non-transparent pixels (alpha > 128).
    2. Discard near-black, near-white, and near-gray pixels — those are
       shadows, highlights, and stems rather than petals.
    3. Quantize the remaining colorful pixels to 8 representative colors.
    4. Return the most saturated one that passes a brightness guard.

    Falls back to the mean of all visible pixels when the image is achromatic
    (e.g. a white flower against white bg after rembg — rare but handled).
    """
    arr = np.array(img_rgba.convert("RGBA"), dtype=np.float32)

    # 1. Non-transparent pixels only
    visible_mask = arr[:, :, 3] > 128
    visible = arr[visible_mask][:, :3]   # (N, 3) float32 RGB

    if len(visible) == 0:
        return (200, 150, 180)  # lavender placeholder

    # 2. Filter to colorful pixels: saturation > 0.15, mid-brightness
    pmax = visible.max(axis=1)
    pmin = visible.min(axis=1)
    sat = np.where(pmax > 0, (pmax - pmin) / pmax, 0.0)
    colorful_mask = (sat > 0.15) & (pmax > 45) & (pmax < 235)
    colorful = visible[colorful_mask]

    if len(colorful) < 20:
        colorful = visible  # all-achromatic fallback

    # 3. Quantize to 8 colors; sample at most 20 000 pixels for speed
    step = max(1, len(colorful) // 20_000)
    sample = colorful[::step].astype(np.uint8)
    # PIL quantize needs a proper 2-D image
    swatch = Image.fromarray(sample.reshape(1, len(sample), 3), "RGB")
    quantized = swatch.quantize(colors=8, method=Image.Quantize.MEDIANCUT)
    palette = np.array(quantized.getpalette()[:8 * 3], dtype=np.float32).reshape(8, 3)

    # 4. Pick the palette entry with highest saturation inside safe brightness range
    p_max = palette.max(axis=1)
    p_min = palette.min(axis=1)
    p_sat = np.where(p_max > 0, (p_max - p_min) / p_max, 0.0)
    brightness_ok = (p_max > 50) & (p_max < 230)
    valid = p_sat * brightness_ok  # zero out invalid entries

    best = int(valid.argmax()) if valid.max() > 0 else int(p_sat.argmax())
    r, g, b = palette[best]
    return (int(r), int(g), int(b))


async def generate_lock_image(main_image_path: str, latin_name: str) -> str:
    """Generate a flat single-color flower silhouette for the lock screen widget.

    The flower's shape (alpha channel) from home.png is preserved exactly, but
    all RGB channels are replaced with the dominant petal color — no gradients,
    no shadows, no texture.

    Output: PNG, transparent background, 200 × 200 px, ~42 KB target.
    """
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    original = Image.open(main_image_path).convert("RGBA")
    color = _dominant_petal_color(original)

    arr = np.array(original)
    alpha = arr[:, :, 3].copy()

    # Fill all RGB channels with the single dominant color
    flat = np.empty_like(arr)
    flat[:, :, 0] = color[0]
    flat[:, :, 1] = color[1]
    flat[:, :, 2] = color[2]
    flat[:, :, 3] = alpha          # preserve original alpha channel exactly

    result = Image.fromarray(flat.astype(np.uint8), "RGBA")
    result = result.resize((200, 200), Image.LANCZOS)

    slug = latin_name.replace(" ", "_").lower()
    out_path = _OUTPUT_DIR / f"{slug}-lock.png"
    result.save(str(out_path), "PNG", optimize=True, compress_level=9)

    return str(out_path)
