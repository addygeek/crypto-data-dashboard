"""Tests for API endpoints."""
import pytest
from httpx import AsyncClient, ASGITransport
from datetime import datetime

from app.main import app
from app.db.database import init_db, drop_db


@pytest.fixture(scope="module")
async def setup_db():
    """Set up test database."""
    await init_db()
    yield
    await drop_db()


@pytest.fixture
async def client(setup_db):
    """Create async test client."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.mark.asyncio
async def test_root_endpoint(client: AsyncClient):
    """Test root endpoint returns API info."""
    response = await client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "name" in data
    assert "endpoints" in data


@pytest.mark.asyncio
async def test_health_endpoint(client: AsyncClient):
    """Test health check endpoint."""
    response = await client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "timestamp" in data


@pytest.mark.asyncio
async def test_assets_endpoint_empty(client: AsyncClient):
    """Test assets endpoint returns empty list initially."""
    response = await client.get("/api/assets")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)


@pytest.mark.asyncio
async def test_prices_missing_asset(client: AsyncClient):
    """Test prices endpoint with missing asset parameter."""
    response = await client.get("/api/prices")
    assert response.status_code == 422  # Validation error


@pytest.mark.asyncio
async def test_prices_nonexistent_asset(client: AsyncClient):
    """Test prices endpoint with non-existent asset."""
    response = await client.get("/api/prices?asset=INVALID")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_compare_missing_assets(client: AsyncClient):
    """Test compare endpoint with missing assets parameter."""
    response = await client.get("/api/compare")
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_anomalies_nonexistent_asset(client: AsyncClient):
    """Test anomalies endpoint with non-existent asset."""
    response = await client.get("/api/anomalies?asset=INVALID")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_volatility_nonexistent_asset(client: AsyncClient):
    """Test volatility endpoint with non-existent asset."""
    response = await client.get("/api/volatility?asset=INVALID")
    assert response.status_code == 404
