"""CRUD + status endpoints for flowers."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_db
from models import Flower, RawSource, Translation

router = APIRouter()


# --- Pydantic schemas ---

class FlowerCreate(BaseModel):
    latin_name: str
    common_name: str | None = None


class FlowerOut(BaseModel):
    id: int
    latin_name: str
    common_name: str | None
    status: str
    description: str | None
    fun_fact: str | None
    wiki_description: str | None
    habitat: str | None
    etymology: str | None
    cultural_info: str | None
    petal_color_hex: str | None
    care_info: dict | None
    edibility_rating: int | None
    medicinal_rating: int | None
    other_uses_rating: int | None
    weed_potential: str | None
    info_image_path: str | None
    info_image_author: str | None
    main_image_path: str | None
    lock_image_path: str | None
    feature_year: int | None
    feature_month: int | None
    feature_day: int | None
    confidence_scores: dict | None
    wikipedia_url: str | None

    model_config = {"from_attributes": True}


class FlowerStatusOut(BaseModel):
    id: int
    latin_name: str
    status: str
    sources_scraped: list[str]

    model_config = {"from_attributes": True}


# --- Endpoints ---

@router.post("", response_model=FlowerOut, status_code=201)
async def create_flower(body: FlowerCreate, db: AsyncSession = Depends(get_db)) -> FlowerOut:
    existing = await db.execute(select(Flower).where(Flower.latin_name == body.latin_name))
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=409, detail="Flower already exists")
    flower = Flower(latin_name=body.latin_name, common_name=body.common_name)
    db.add(flower)
    await db.commit()
    await db.refresh(flower)
    return FlowerOut.model_validate(flower)


@router.get("", response_model=list[FlowerOut])
async def list_flowers(
    status: str | None = None,
    limit: int = 50,
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
) -> list[FlowerOut]:
    q = select(Flower).order_by(Flower.id).limit(limit).offset(offset)
    if status:
        q = q.where(Flower.status == status)
    result = await db.execute(q)
    return [FlowerOut.model_validate(f) for f in result.scalars().all()]


@router.get("/{flower_id}", response_model=FlowerOut)
async def get_flower(flower_id: int, db: AsyncSession = Depends(get_db)) -> FlowerOut:
    flower = await db.get(Flower, flower_id)
    if not flower:
        raise HTTPException(status_code=404, detail="Flower not found")
    return FlowerOut.model_validate(flower)


@router.get("/{flower_id}/status", response_model=FlowerStatusOut)
async def get_flower_status(flower_id: int, db: AsyncSession = Depends(get_db)) -> FlowerStatusOut:
    flower = await db.get(Flower, flower_id)
    if not flower:
        raise HTTPException(status_code=404, detail="Flower not found")
    sources_result = await db.execute(select(RawSource.source).where(RawSource.flower_id == flower_id))
    sources = [row[0] for row in sources_result.all()]
    return FlowerStatusOut(
        id=flower.id,
        latin_name=flower.latin_name,
        status=flower.status,
        sources_scraped=sources,
    )


@router.delete("/{flower_id}", status_code=204)
async def delete_flower(flower_id: int, db: AsyncSession = Depends(get_db)) -> None:
    flower = await db.get(Flower, flower_id)
    if not flower:
        raise HTTPException(status_code=404, detail="Flower not found")
    await db.delete(flower)
    await db.commit()
