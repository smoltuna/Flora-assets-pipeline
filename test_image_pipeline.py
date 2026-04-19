"""Standalone test: run the full image pipeline for 2 flowers."""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "backend"))

# Load .env so settings picks up FAL_KEY
from dotenv import load_dotenv  # noqa: E402
load_dotenv(Path(__file__).parent / ".env")

FLOWERS = ["Hypericum perforatum", "Malva sylvestris", "Campanula rotundifolia", "Leucanthemum vulgare", "Ranunculus acris"]


async def run_flower(latin_name: str) -> dict:
    from config import settings
    from services.images.wikimedia import find_images
    from services.images.processor import process_info_image, process_main_image
    from services.images.lock_gen import generate_lock_image

    print(f"\n{'='*60}")
    print(f"  {latin_name}")
    print(f"{'='*60}")

    print("  [1/4] Searching Wikimedia Commons…")
    pair = await find_images(latin_name)
    print(f"        info:    {pair.info.title}")
    for i, c in enumerate(pair.blossom_candidates or [pair.blossom], 1):
        tag = "blossom" if i == 1 else f"       "
        print(f"        {tag}[{i}]: {c.title}")

    print("  [2/4] Processing info image…")
    info_path, author = await process_info_image(pair.info, latin_name)
    size_kb = Path(info_path).stat().st_size // 1024
    print(f"        → {info_path}  ({size_kb} KB, author: {author[:60]})")

    print("  [3/4] fal.ai vision judge + birefnet-general rembg cascade…")
    main_path, _ = await process_main_image(
        pair.blossom,
        latin_name,
        candidates=pair.blossom_candidates,
        fal_key=settings.fal_key,
    )
    size_kb = Path(main_path).stat().st_size // 1024
    print(f"        → {main_path}  ({size_kb} KB)")

    print("  [4/4] FLUX Schnell lock icon…")
    lock_path = await generate_lock_image(
        main_path, latin_name, fal_key=settings.fal_key,
    )
    size_kb = Path(lock_path).stat().st_size // 1024
    print(f"        → {lock_path}  ({size_kb} KB)")

    return {"latin_name": latin_name, "info": info_path, "main": main_path, "lock": lock_path}


async def main():
    results = []
    for flower in FLOWERS:
        result = await run_flower(flower)
        results.append(result)

    print(f"\n{'='*60}")
    print("  DONE — output files:")
    print(f"{'='*60}")
    for r in results:
        print(f"\n  {r['latin_name']}")
        for key in ("info", "main", "lock"):
            p = Path(r[key])
            print(f"    {key:5s}: {p}  ({p.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    asyncio.run(main())
