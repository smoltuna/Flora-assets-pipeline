"""Process all pending flowers through the full pipeline sequentially.

Usage:
  uv run python scripts/run_batch.py
  uv run python scripts/run_batch.py --limit 10
  uv run python scripts/run_batch.py --latin-name "Rosa canina"
"""
from __future__ import annotations

import asyncio
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

import structlog  # noqa: E402
from sqlalchemy import select  # noqa: E402

from database import async_session_factory, create_tables  # noqa: E402
from log_config import configure_logging  # noqa: E402
from models import Flower  # noqa: E402
from tasks.pipeline import run_pipeline  # noqa: E402

log = structlog.get_logger()


async def run_batch(limit: int | None = None, latin_name: str | None = None) -> None:
    configure_logging()
    await create_tables()

    async with async_session_factory() as session:
        if latin_name:
            result = await session.execute(
                select(Flower).where(Flower.latin_name == latin_name)
            )
            flowers = result.scalars().all()
        else:
            q = select(Flower).where(Flower.status == "pending").order_by(Flower.id)
            if limit:
                q = q.limit(limit)
            result = await session.execute(q)
            flowers = result.scalars().all()

    log.info("batch.start", n_flowers=len(flowers))

    succeeded = 0
    failed = 0
    for flower in flowers:
        log.info("batch.processing", latin_name=flower.latin_name, flower_id=flower.id)
        async with async_session_factory() as session:
            try:
                await run_pipeline(flower.id, session)
                succeeded += 1
                log.info("batch.done", latin_name=flower.latin_name)
            except Exception as e:
                failed += 1
                log.error("batch.error", latin_name=flower.latin_name,
                          error=str(e), exc_type=type(e).__name__)

    log.info("batch.complete", succeeded=succeeded, failed=failed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run pipeline for pending flowers")
    parser.add_argument("--limit", type=int, default=None, help="Max flowers to process")
    parser.add_argument("--latin-name", type=str, default=None, help="Process a specific flower")
    args = parser.parse_args()
    asyncio.run(run_batch(limit=args.limit, latin_name=args.latin_name))


if __name__ == "__main__":
    main()
