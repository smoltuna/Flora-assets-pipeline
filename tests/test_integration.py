"""Integration tests — require a real PostgreSQL instance with pgvector.

These tests spin up the FastAPI app against a live database to verify the full
request/response cycle, model persistence, and API contract.

Run with:
    DATABASE_URL=postgresql+asyncpg://flora:flora@localhost:5432/flora \\
        pytest tests/test_integration.py -v -m integration

The CI workflow provisions a pgvector/pgvector:pg17 service container, so these
tests run automatically on push. They are skipped when DATABASE_URL points to
SQLite or is absent.
"""
from __future__ import annotations

import os

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport

# Skip entire module if no PostgreSQL DATABASE_URL is configured
pytestmark = pytest.mark.integration

_DB_URL = os.getenv("DATABASE_URL", "")
_HAS_PG = _DB_URL.startswith("postgresql")

if not _HAS_PG:
    pytest.skip(
        "Integration tests require DATABASE_URL pointing to PostgreSQL",
        allow_module_level=True,
    )


@pytest.fixture(scope="module", autouse=True)
def set_test_db(monkeypatch_module):
    """Point the app at the test database before importing the app."""
    import os
    os.environ.setdefault("DATABASE_URL", _DB_URL)


@pytest.fixture(scope="module")
def monkeypatch_module():
    """Module-scoped monkeypatch (pytest built-in is function-scoped)."""
    from _pytest.monkeypatch import MonkeyPatch
    mp = MonkeyPatch()
    yield mp
    mp.undo()


@pytest_asyncio.fixture(scope="module")
async def client():
    """Async HTTP client wired to the FastAPI app with a real DB."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

    from main import app
    from database import create_tables
    await create_tables()

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


# ── Health check ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_health(client: AsyncClient):
    resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


# ── Flower CRUD ───────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_create_flower(client: AsyncClient):
    resp = await client.post("/flowers", json={"latin_name": "Testus plantus integration"})
    assert resp.status_code == 201
    data = resp.json()
    assert data["latin_name"] == "Testus plantus integration"
    assert data["status"] == "pending"
    assert data["id"] > 0


@pytest.mark.asyncio
async def test_create_flower_duplicate_returns_409(client: AsyncClient):
    await client.post("/flowers", json={"latin_name": "Testus duplicatus"})
    resp = await client.post("/flowers", json={"latin_name": "Testus duplicatus"})
    assert resp.status_code == 409


@pytest.mark.asyncio
async def test_list_flowers(client: AsyncClient):
    await client.post("/flowers", json={"latin_name": "Testus listus"})
    resp = await client.get("/flowers")
    assert resp.status_code == 200
    flowers = resp.json()
    assert isinstance(flowers, list)
    assert any(f["latin_name"] == "Testus listus" for f in flowers)


@pytest.mark.asyncio
async def test_list_flowers_status_filter(client: AsyncClient):
    await client.post("/flowers", json={"latin_name": "Testus pending filter"})
    resp = await client.get("/flowers?status=pending")
    assert resp.status_code == 200
    flowers = resp.json()
    assert all(f["status"] == "pending" for f in flowers)


@pytest.mark.asyncio
async def test_get_flower_by_id(client: AsyncClient):
    create_resp = await client.post("/flowers", json={"latin_name": "Testus getbyid"})
    flower_id = create_resp.json()["id"]

    resp = await client.get(f"/flowers/{flower_id}")
    assert resp.status_code == 200
    assert resp.json()["id"] == flower_id


@pytest.mark.asyncio
async def test_get_flower_not_found(client: AsyncClient):
    resp = await client.get("/flowers/999999")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_delete_flower(client: AsyncClient):
    create_resp = await client.post("/flowers", json={"latin_name": "Testus deleteme"})
    flower_id = create_resp.json()["id"]

    del_resp = await client.delete(f"/flowers/{flower_id}")
    assert del_resp.status_code == 204

    get_resp = await client.get(f"/flowers/{flower_id}")
    assert get_resp.status_code == 404


@pytest.mark.asyncio
async def test_flower_status_endpoint(client: AsyncClient):
    create_resp = await client.post("/flowers", json={"latin_name": "Testus statuscheck"})
    flower_id = create_resp.json()["id"]

    resp = await client.get(f"/flowers/{flower_id}/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "pending"
    assert data["sources_scraped"] == []


# ── Export endpoint ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_export_pending_flower_returns_400(client: AsyncClient):
    """Export requires enriched status; pending flowers should be rejected."""
    create_resp = await client.post("/flowers", json={"latin_name": "Testus exportpending"})
    flower_id = create_resp.json()["id"]

    resp = await client.get(f"/export/{flower_id}")
    assert resp.status_code == 400
