"""Run the image sub-pipeline only (Wikimedia → rembg → lock screen).

The flower must already be enriched (status: enriched / images_done / complete).
Edit the FLOWERS list below or pass overrides on the command line.

Usage:
  uv run python scripts/run_image_pipeline.py                  # process FLOWERS list
  uv run python scripts/run_image_pipeline.py --name "Rosa canina"
  uv run python scripts/run_image_pipeline.py --file flowers.txt
  uv run python scripts/run_image_pipeline.py --limit 5        # first N enriched flowers in DB
"""
from __future__ import annotations

# ── Flowers to process ───────────────────────────────────────────────────────
FLOWERS = [
    "Iris germanica",
    "Papaver orientale",
    "Nymphaea alba",
]
# ─────────────────────────────────────────────────────────────────────────────

import asyncio
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

import structlog  # noqa: E402
from sqlalchemy import select  # noqa: E402

from config import settings  # noqa: E402
from database import async_session_factory, create_tables  # noqa: E402
from log_config import configure_logging  # noqa: E402
from models import Flower  # noqa: E402
from services.images.wikimedia import find_images  # noqa: E402
from services.images.processor import process_info_image, process_main_image  # noqa: E402
from services.images.lock_gen import generate_lock_image  # noqa: E402

log = structlog.get_logger()

ENRICHED_STATUSES = ("enriched", "images_done", "complete")


async def _run_images_for_flower(flower: Flower, session) -> None:
    pair = await find_images(flower.latin_name)

    info_path, author = await process_info_image(pair.info, flower.latin_name)
    flower.info_image_path = info_path
    flower.info_image_author = author

    main_path, _ = await process_main_image(
        pair.blossom,
        flower.latin_name,
        candidates=pair.blossom_candidates,
        fal_key=settings.fal_key,
    )
    flower.main_image_path = flower.latin_name.replace(" ", "-").lower()

    lock_path = await generate_lock_image(
        main_path,
        flower.latin_name,
        fal_key=settings.fal_key,
    )
    flower.lock_image_path = lock_path
    flower.status = "images_done"
    await session.commit()


async def _load_flowers(
    latin_names: list[str] | None,
    limit: int | None,
) -> list[Flower]:
    async with async_session_factory() as session:
        if latin_names:
            result = await session.execute(
                select(Flower).where(Flower.latin_name.in_(latin_names))
            )
        else:
            q = (
                select(Flower)
                .where(Flower.status.in_(ENRICHED_STATUSES))
                .order_by(Flower.id)
            )
            if limit:
                q = q.limit(limit)
            result = await session.execute(q)
        return result.scalars().all()


def _eta(elapsed: float, done: int, total: int) -> float | None:
    if done <= 0:
        return None
    remaining = total - done
    return round((elapsed / done) * remaining, 1) if remaining > 0 else 0.0


async def main(latin_names: list[str] | None, limit: int | None) -> None:
    configure_logging()
    await create_tables()

    flowers = await _load_flowers(latin_names, limit)
    total = len(flowers)

    if total == 0:
        log.info("image_pipeline.nothing_to_do", hint="Flowers must be enriched before images can run.")
        return

    log.info("image_pipeline.start", n_flowers=total)
    succeeded = failed = 0
    batch_start = time.perf_counter()

    for index, flower in enumerate(flowers, start=1):
        elapsed = time.perf_counter() - batch_start
        log.info(
            "image_pipeline.processing",
            latin_name=flower.latin_name,
            progress=f"{index}/{total}",
            eta_s=_eta(elapsed, succeeded + failed, total),
        )
        flower_start = time.perf_counter()
        async with async_session_factory() as session:
            try:
                f = await session.get(Flower, flower.id)
                if f.status not in ENRICHED_STATUSES:
                    log.warning(
                        "image_pipeline.skip",
                        latin_name=flower.latin_name,
                        reason=f"status={f.status!r} — must be enriched first",
                    )
                    continue
                await _run_images_for_flower(f, session)
                succeeded += 1
                log.info(
                    "image_pipeline.done",
                    latin_name=flower.latin_name,
                    elapsed_s=round(time.perf_counter() - flower_start, 1),
                )
            except Exception as exc:
                failed += 1
                log.error(
                    "image_pipeline.error",
                    latin_name=flower.latin_name,
                    error=str(exc),
                    exc_type=type(exc).__name__,
                )

    total_elapsed = round(time.perf_counter() - batch_start, 1)
    log.info(
        "image_pipeline.complete",
        succeeded=succeeded,
        failed=failed,
        total=total,
        elapsed_s=total_elapsed,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Flora image pipeline")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--name", type=str, help="Single Latin name")
    group.add_argument("--file", type=Path, help="Text file with one Latin name per line")
    group.add_argument("--limit", type=int, help="Process the first N enriched flowers in the DB")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if args.name:
        names: list[str] | None = [args.name]
        limit = None
    elif args.file:
        names = args.file.read_text().splitlines()
        limit = None
    elif args.limit:
        names = None
        limit = args.limit
    else:
        names = FLOWERS
        limit = None

    asyncio.run(main(names, limit))
