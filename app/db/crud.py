"""CRUD operations for database models."""
from datetime import datetime
from typing import List, Optional
from decimal import Decimal

from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import CryptoAsset, PriceSnapshot


# ============== CryptoAsset Operations ==============

async def get_or_create_asset(
    session: AsyncSession,
    coingecko_id: str,
    symbol: str,
    name: str
) -> CryptoAsset:
    """Get existing asset or create new one."""
    result = await session.execute(
        select(CryptoAsset).where(CryptoAsset.coingecko_id == coingecko_id)
    )
    asset = result.scalar_one_or_none()
    
    if asset is None:
        asset = CryptoAsset(
            coingecko_id=coingecko_id,
            symbol=symbol.upper(),
            name=name
        )
        session.add(asset)
        await session.flush()
    
    return asset


async def get_all_assets(session: AsyncSession) -> List[CryptoAsset]:
    """Get all supported crypto assets."""
    result = await session.execute(
        select(CryptoAsset).order_by(CryptoAsset.symbol)
    )
    return list(result.scalars().all())


async def get_asset_by_symbol(session: AsyncSession, symbol: str) -> Optional[CryptoAsset]:
    """Get asset by symbol."""
    result = await session.execute(
        select(CryptoAsset).where(CryptoAsset.symbol == symbol.upper())
    )
    return result.scalar_one_or_none()


async def get_asset_by_coingecko_id(session: AsyncSession, coingecko_id: str) -> Optional[CryptoAsset]:
    """Get asset by CoinGecko ID."""
    result = await session.execute(
        select(CryptoAsset).where(CryptoAsset.coingecko_id == coingecko_id)
    )
    return result.scalar_one_or_none()


# ============== PriceSnapshot Operations ==============

async def create_snapshot(
    session: AsyncSession,
    asset_id: int,
    price_usd: Decimal,
    volume_24h: Optional[Decimal] = None,
    market_cap: Optional[Decimal] = None,
    price_change_24h: Optional[Decimal] = None,
    timestamp: Optional[datetime] = None
) -> PriceSnapshot:
    """Create a new price snapshot."""
    snapshot = PriceSnapshot(
        asset_id=asset_id,
        price_usd=price_usd,
        volume_24h=volume_24h,
        market_cap=market_cap,
        price_change_24h=price_change_24h,
        timestamp=timestamp or datetime.utcnow()
    )
    session.add(snapshot)
    await session.flush()
    return snapshot


async def get_snapshots_by_asset(
    session: AsyncSession,
    asset_id: int,
    from_date: Optional[datetime] = None,
    to_date: Optional[datetime] = None,
    limit: Optional[int] = None
) -> List[PriceSnapshot]:
    """Get price snapshots for an asset within a time range."""
    query = select(PriceSnapshot).where(PriceSnapshot.asset_id == asset_id)
    
    if from_date:
        query = query.where(PriceSnapshot.timestamp >= from_date)
    if to_date:
        query = query.where(PriceSnapshot.timestamp <= to_date)
    
    query = query.order_by(PriceSnapshot.timestamp.asc())
    
    if limit:
        query = query.limit(limit)
    
    result = await session.execute(query)
    return list(result.scalars().all())


async def get_snapshots_for_comparison(
    session: AsyncSession,
    asset_ids: List[int],
    from_date: Optional[datetime] = None,
    to_date: Optional[datetime] = None
) -> List[PriceSnapshot]:
    """Get snapshots for multiple assets for comparison."""
    query = select(PriceSnapshot).where(PriceSnapshot.asset_id.in_(asset_ids))
    
    if from_date:
        query = query.where(PriceSnapshot.timestamp >= from_date)
    if to_date:
        query = query.where(PriceSnapshot.timestamp <= to_date)
    
    query = query.order_by(PriceSnapshot.timestamp.asc())
    
    result = await session.execute(query)
    return list(result.scalars().all())


async def get_latest_snapshot(session: AsyncSession, asset_id: int) -> Optional[PriceSnapshot]:
    """Get the most recent snapshot for an asset."""
    result = await session.execute(
        select(PriceSnapshot)
        .where(PriceSnapshot.asset_id == asset_id)
        .order_by(PriceSnapshot.timestamp.desc())
        .limit(1)
    )
    return result.scalar_one_or_none()


async def get_snapshot_count(session: AsyncSession, asset_id: Optional[int] = None) -> int:
    """Get total count of snapshots."""
    from sqlalchemy import func
    
    query = select(func.count(PriceSnapshot.id))
    if asset_id:
        query = query.where(PriceSnapshot.asset_id == asset_id)
    
    result = await session.execute(query)
    return result.scalar() or 0
