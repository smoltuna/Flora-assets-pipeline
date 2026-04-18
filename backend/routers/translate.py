"""Translation endpoints — RAG-grounded translation via MarianMT + Llama."""
from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_db
from models import Flower, Translation

router = APIRouter()

SUPPORTED_LANGUAGES = ("de", "fr", "es", "it", "zh", "ja")


class TranslationOut(BaseModel):
    language: str
    name: str | None
    description: str | None
    fun_fact: str | None
    wiki_description: str | None
    habitat: str | None
    etymology: str | None
    cultural_info: str | None
    source_method: str | None

    model_config = {"from_attributes": True}


@router.post("/{flower_id}", status_code=202)
async def translate_flower(
    flower_id: int,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Trigger translation for all supported languages in background."""
    flower = await db.get(Flower, flower_id)
    if not flower:
        raise HTTPException(status_code=404, detail="Flower not found")
    if not flower.description:
        raise HTTPException(status_code=400, detail="Flower must be enriched before translation")

    background_tasks.add_task(_run_translations_bg, flower_id)
    return {"flower_id": flower_id, "status": "queued", "languages": list(SUPPORTED_LANGUAGES)}


@router.get("/{flower_id}", response_model=list[TranslationOut])
async def get_translations(flower_id: int, db: AsyncSession = Depends(get_db)) -> list[TranslationOut]:
    flower = await db.get(Flower, flower_id)
    if not flower:
        raise HTTPException(status_code=404, detail="Flower not found")
    result = await db.execute(select(Translation).where(Translation.flower_id == flower_id))
    return [TranslationOut.model_validate(t) for t in result.scalars().all()]


async def _run_translations_bg(flower_id: int) -> None:
    from database import async_session_factory
    from services.translation.translator import translate_flower as do_translate
    async with async_session_factory() as session:
        await do_translate(flower_id, session)
