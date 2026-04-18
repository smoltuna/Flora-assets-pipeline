"""Export flowers to Flora xcassets-compatible JSON format.

Mirrors the export router logic for CLI use.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from sqlalchemy import select

from database import async_session_factory, create_tables
from models import Flower, Translation
from routers.export import _build_payload


async def export_one(flower_id: int, output_dir: Path) -> None:
    await create_tables()
    async with async_session_factory() as session:
        flower = await session.get(Flower, flower_id)
        if not flower:
            print(f"Flower {flower_id} not found.", file=sys.stderr)
            sys.exit(1)

        trans_result = await session.execute(
            select(Translation).where(Translation.flower_id == flower_id)
        )
        translations = trans_result.scalars().all()

    output_dir.mkdir(parents=True, exist_ok=True)
    payload = _build_payload(flower, translations)
    out = output_dir / f"{flower.latin_name.replace(' ', '_').lower()}.json"
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"Exported: {out}")


async def export_all(output_dir: Path) -> None:
    await create_tables()
    async with async_session_factory() as session:
        result = await session.execute(
            select(Flower).where(Flower.status.in_(["enriched", "images_done", "complete"]))
        )
        flowers = result.scalars().all()

        output_dir.mkdir(parents=True, exist_ok=True)
        for flower in flowers:
            trans_result = await session.execute(
                select(Translation).where(Translation.flower_id == flower.id)
            )
            translations = trans_result.scalars().all()
            payload = _build_payload(flower, translations)
            out = output_dir / f"{flower.latin_name.replace(' ', '_').lower()}.json"
            out.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
            print(f"Exported: {out}")

    print(f"\nTotal: {len(flowers)} flowers exported to {output_dir}")
