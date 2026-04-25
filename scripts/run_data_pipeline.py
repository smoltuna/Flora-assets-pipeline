"""Run the full data pipeline (scrape → embed → RAG → synthesise → translate).

Edit the FLOWERS list below to select which flowers to process, or pass
overrides on the command line.

Usage:
  uv run python scripts/run_data_pipeline.py                  # process FLOWERS list
  uv run python scripts/run_data_pipeline.py --name "Rosa canina"
  uv run python scripts/run_data_pipeline.py --file flowers.txt
  uv run python scripts/run_data_pipeline.py --limit 5        # first N pending flowers in DB
"""
from __future__ import annotations

# ── Flowers to process ───────────────────────────────────────────────────────
FLOWERS = [
    "Rosa canina",
    "Helleborus niger",
    # Add more Latin names here …
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

from database import async_session_factory, create_tables  # noqa: E402
from log_config import configure_logging  # noqa: E402
from models import Flower  # noqa: E402
from tasks.pipeline import run_pipeline  # noqa: E402

log = structlog.get_logger()


async def _ensure_flowers(latin_names: list[str]) -> None:
    """Insert any flowers that are not yet in the database."""
    async with async_session_factory() as session:
        for name in latin_names:
            name = name.strip()
            if not name:
                continue
            existing = await session.execute(select(Flower).where(Flower.latin_name == name))
            if existing.scalar_one_or_none() is None:
                session.add(Flower(latin_name=name, status="pending"))
                log.info("seed.added", latin_name=name)
        await session.commit()


async def _load_flowers(
    latin_names: list[str] | None,
    limit: int | None,
) -> list[Flower]:
    """Return the Flower rows to process."""
    async with async_session_factory() as session:
        if latin_names:
            result = await session.execute(
                select(Flower).where(Flower.latin_name.in_(latin_names))
            )
        else:
            q = select(Flower).where(Flower.status == "pending").order_by(Flower.id)
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

    if latin_names:
        await _ensure_flowers(latin_names)

    flowers = await _load_flowers(latin_names, limit)
    total = len(flowers)

    if total == 0:
        log.info("data_pipeline.nothing_to_do")
        return

    log.info("data_pipeline.start", n_flowers=total)
    succeeded = failed = 0
    batch_start = time.perf_counter()

    for index, flower in enumerate(flowers, start=1):
        elapsed = time.perf_counter() - batch_start
        log.info(
            "data_pipeline.processing",
            latin_name=flower.latin_name,
            progress=f"{index}/{total}",
            eta_s=_eta(elapsed, succeeded + failed, total),
        )
        flower_start = time.perf_counter()
        async with async_session_factory() as session:
            try:
                await run_pipeline(flower.id, session)
                succeeded += 1
                log.info(
                    "data_pipeline.done",
                    latin_name=flower.latin_name,
                    elapsed_s=round(time.perf_counter() - flower_start, 1),
                )
            except Exception as exc:
                failed += 1
                log.error(
                    "data_pipeline.error",
                    latin_name=flower.latin_name,
                    error=str(exc),
                    exc_type=type(exc).__name__,
                )

    total_elapsed = round(time.perf_counter() - batch_start, 1)
    log.info(
        "data_pipeline.complete",
        succeeded=succeeded,
        failed=failed,
        total=total,
        elapsed_s=total_elapsed,
        avg_s_per_flower=round(total_elapsed / total, 1),
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Flora data pipeline")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--name", type=str, help="Single Latin name to process")
    group.add_argument("--file", type=Path, help="Text file with one Latin name per line")
    group.add_argument(
        "--limit",
        type=int,
        help="Process the first N pending flowers already in the database",
    )
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
