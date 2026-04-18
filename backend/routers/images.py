"""Image pipeline endpoints — Wikimedia Commons search, rembg processing, lock image."""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_db
from models import Flower

router = APIRouter()


class ImageResult(BaseModel):
    flower_id: int
    latin_name: str
    info_image_path: str | None = None
    main_image_path: str | None = None
    lock_image_path: str | None = None
    info_image_author: str | None = None
    status: str


@router.post("/{flower_id}", response_model=ImageResult)
async def process_images(
    flower_id: int,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
) -> ImageResult:
    """Trigger full image pipeline in background (Wikimedia → rembg → lock)."""
    flower = await db.get(Flower, flower_id)
    if not flower:
        raise HTTPException(status_code=404, detail="Flower not found")
    if flower.status not in ("enriched", "images_done", "complete"):
        raise HTTPException(status_code=400, detail=f"Flower must be enriched first (status: {flower.status})")

    background_tasks.add_task(_run_images_bg, flower_id)
    return ImageResult(
        flower_id=flower_id,
        latin_name=flower.latin_name,
        status="queued",
    )


@router.get("/{flower_id}", response_model=ImageResult)
async def get_image_status(flower_id: int, db: AsyncSession = Depends(get_db)) -> ImageResult:
    flower = await db.get(Flower, flower_id)
    if not flower:
        raise HTTPException(status_code=404, detail="Flower not found")
    return ImageResult(
        flower_id=flower_id,
        latin_name=flower.latin_name,
        info_image_path=flower.info_image_path,
        main_image_path=flower.main_image_path,
        lock_image_path=flower.lock_image_path,
        info_image_author=flower.info_image_author,
        status=flower.status,
    )


@router.get("/{flower_id}/serve/{image_type}")
async def serve_image(
    flower_id: int,
    image_type: str,
    db: AsyncSession = Depends(get_db),
) -> FileResponse:
    """Serve a processed image file for display in the frontend UI."""
    flower = await db.get(Flower, flower_id)
    if not flower:
        raise HTTPException(status_code=404, detail="Flower not found")

    path_map = {
        "info": flower.info_image_path,
        "main": flower.main_image_path,
        "lock": flower.lock_image_path,
    }
    if image_type not in path_map:
        raise HTTPException(status_code=400, detail="image_type must be info, main, or lock")

    image_path = path_map[image_type]
    if not image_path:
        raise HTTPException(status_code=404, detail=f"No {image_type} image for this flower")

    p = Path(image_path)
    if not p.exists():
        raise HTTPException(status_code=404, detail="Image file not found on disk")

    media_type = "image/png" if p.suffix.lower() == ".png" else "image/jpeg"
    return FileResponse(p, media_type=media_type)


async def _run_images_bg(flower_id: int) -> None:
    from database import async_session_factory
    from services.images.wikimedia import find_images
    from services.images.processor import process_info_image, process_main_image
    from services.images.lock_gen import generate_lock_image
    import structlog
    log = structlog.get_logger()

    async with async_session_factory() as session:
        flower = await session.get(Flower, flower_id)
        if not flower:
            return

        try:
            pair = await find_images(flower.latin_name)

            info_path, author = await process_info_image(pair.info, flower.latin_name)
            flower.info_image_path = info_path
            flower.info_image_author = author

            main_path = await process_main_image(pair.blossom, flower.latin_name)
            flower.main_image_path = main_path

            lock_path = await generate_lock_image(main_path, flower.latin_name)
            flower.lock_image_path = lock_path

            flower.status = "images_done"
            await session.commit()
            log.info("images.done", flower_id=flower_id)
        except Exception as e:
            log.error("images.error", flower_id=flower_id, error=str(e))
