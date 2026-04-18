"""Load a list of latin names into the flowers table.

Usage:
  uv run python scripts/seed_flowers.py --names "Rosa canina" "Bellis perennis"
  uv run python scripts/seed_flowers.py --file flowers.txt
"""
from __future__ import annotations

import asyncio
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from database import async_session_factory, create_tables  # noqa: E402
from models import Flower  # noqa: E402
from sqlalchemy import select  # noqa: E402


async def seed(latin_names: list[str]) -> None:
    await create_tables()
    async with async_session_factory() as session:
        added = 0
        skipped = 0
        for name in latin_names:
            name = name.strip()
            if not name:
                continue
            existing = await session.execute(select(Flower).where(Flower.latin_name == name))
            if existing.scalar_one_or_none():
                skipped += 1
                continue
            session.add(Flower(latin_name=name, status="pending"))
            added += 1
        await session.commit()
        print(f"Seeded {added} flowers, skipped {skipped} duplicates.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed flowers into the database")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--names", nargs="+", help="Latin names to add")
    group.add_argument("--file", type=Path, help="Text file with one latin name per line")
    args = parser.parse_args()

    if args.file:
        names = args.file.read_text().splitlines()
    else:
        names = args.names

    asyncio.run(seed(names))


if __name__ == "__main__":
    main()
