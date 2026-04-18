"""Export endpoint — trigger xcassets export for a flower or all complete flowers."""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_db
from models import Flower, Translation

router = APIRouter()


class ExportResult(BaseModel):
    exported: int
    output_path: str


@router.get("/{flower_id}")
async def export_flower(flower_id: int, db: AsyncSession = Depends(get_db)) -> JSONResponse:
    """Return the Flora-compatible JSON payload for a single flower."""
    flower = await db.get(Flower, flower_id)
    if not flower:
        raise HTTPException(status_code=404, detail="Flower not found")

    translations_result = await db.execute(
        select(Translation).where(Translation.flower_id == flower_id)
    )
    translations = translations_result.scalars().all()

    payload = _build_payload(flower, translations)
    return JSONResponse(content=payload)


@router.post("/batch", response_model=ExportResult)
async def export_batch(
    output_dir: str = "/tmp/flora_export",
    db: AsyncSession = Depends(get_db),
) -> ExportResult:
    """Export all enriched flowers to JSON files in the Flora xcassets format."""
    result = await db.execute(
        select(Flower).where(Flower.status.in_(["enriched", "images_done", "complete"]))
    )
    flowers = result.scalars().all()

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    for flower in flowers:
        trans_result = await db.execute(
            select(Translation).where(Translation.flower_id == flower.id)
        )
        translations = trans_result.scalars().all()
        payload = _build_payload(flower, translations)
        (out_path / f"{flower.latin_name.replace(' ', '_').lower()}.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False)
        )

    return ExportResult(exported=len(flowers), output_path=str(out_path))


def _build_payload(flower: Flower, translations: list[Translation]) -> dict:
    """Build Flora-compatible JSON matching flowers.json schema."""
    trans_map: dict[str, Translation] = {t.language: t for t in translations}

    def localized(field: str, lang: str) -> str | None:
        t = trans_map.get(lang)
        return getattr(t, field, None) if t else None

    return {
        "latinName": flower.latin_name,
        "name": flower.common_name or flower.latin_name,
        "description": flower.description or "",
        "funFact": flower.fun_fact or "",
        "wikiDescription": flower.wiki_description or "",
        "habitat": flower.habitat or "",
        "etymology": flower.etymology or "",
        "culturalInfo": flower.cultural_info or "",
        "petalColorHex": flower.petal_color_hex or "#FFFFFF",
        "careInfo": flower.care_info or {},
        "edibilityRating": flower.edibility_rating,
        "medicinalRating": flower.medicinal_rating,
        "otherUsesRating": flower.other_uses_rating,
        "infoImageAuthor": flower.info_image_author or "",
        "year": flower.feature_year,
        "month": flower.feature_month,
        "day": flower.feature_day,
        "confidence": flower.confidence_scores or {},
        "localizations": {
            lang: {
                "name": localized("name", lang),
                "description": localized("description", lang),
                "funFact": localized("fun_fact", lang),
                "habitat": localized("habitat", lang),
            }
            for lang in ("de", "fr", "es", "it", "zh", "ja")
            if lang in trans_map
        },
    }
