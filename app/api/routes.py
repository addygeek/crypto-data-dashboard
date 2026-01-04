"""API route definitions."""
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel

from app.db.database import get_db
from app.db import crud
from app.services.anomaly import detect_anomalies, calculate_volatility, Anomaly

router = APIRouter()


# ============== Pydantic Models ==============

class AssetResponse(BaseModel):
    """Response model for crypto asset."""
    id: int
    symbol: str
    name: str
    coingecko_id: str
    
    class Config:
        from_attributes = True


class PriceResponse(BaseModel):
    """Response model for price data."""
    timestamp: str
    price_usd: float
    volume_24h: Optional[float] = None
    market_cap: Optional[float] = None
    price_change_24h: Optional[float] = None


class PriceHistoryResponse(BaseModel):
    """Response for historical price data."""
    asset: AssetResponse
    prices: List[PriceResponse]
    count: int


class ComparisonDataPoint(BaseModel):
    """Single data point for comparison."""
    timestamp: str
    prices: dict  # {symbol: price}


class ComparisonResponse(BaseModel):
    """Response for comparison data."""
    assets: List[str]
    data: List[ComparisonDataPoint]


class AnomalyResponse(BaseModel):
    """Response for anomaly."""
    timestamp: str
    asset_symbol: str
    price: float
    anomaly_type: str
    severity: float
    price_change: float


class VolatilityResponse(BaseModel):
    """Response for volatility data."""
    timestamp: str
    volatility: float


# ============== Endpoints ==============

@router.get("/assets", response_model=List[AssetResponse])
async def list_assets(db: AsyncSession = Depends(get_db)):
    """
    Get list of all supported cryptocurrencies.
    
    Returns all crypto assets currently tracked in the system.
    """
    assets = await crud.get_all_assets(db)
    return [
        AssetResponse(
            id=a.id,
            symbol=a.symbol,
            name=a.name,
            coingecko_id=a.coingecko_id
        )
        for a in assets
    ]


@router.get("/prices", response_model=PriceHistoryResponse)
async def get_prices(
    asset: str = Query(..., description="Asset symbol (e.g., BTC, ETH)"),
    from_date: Optional[str] = Query(None, alias="from", description="Start date (ISO format)"),
    to_date: Optional[str] = Query(None, alias="to", description="End date (ISO format)"),
    limit: Optional[int] = Query(None, description="Maximum number of records"),
    db: AsyncSession = Depends(get_db)
):
    """
    Get historical price data for an asset.
    
    - **asset**: Asset symbol (e.g., BTC, ETH)
    - **from**: Start date in ISO format (optional)
    - **to**: End date in ISO format (optional)
    - **limit**: Maximum number of records (optional)
    """
    # Find asset by symbol
    asset_obj = await crud.get_asset_by_symbol(db, asset)
    if not asset_obj:
        raise HTTPException(status_code=404, detail=f"Asset '{asset}' not found")
    
    # Parse dates
    from_dt = None
    to_dt = None
    if from_date:
        try:
            from_dt = datetime.fromisoformat(from_date.replace("Z", "+00:00"))
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid 'from' date format")
    if to_date:
        try:
            to_dt = datetime.fromisoformat(to_date.replace("Z", "+00:00"))
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid 'to' date format")
    
    # Get snapshots
    snapshots = await crud.get_snapshots_by_asset(
        db, asset_obj.id, from_dt, to_dt, limit
    )
    
    return PriceHistoryResponse(
        asset=AssetResponse(
            id=asset_obj.id,
            symbol=asset_obj.symbol,
            name=asset_obj.name,
            coingecko_id=asset_obj.coingecko_id
        ),
        prices=[
            PriceResponse(
                timestamp=s.timestamp.isoformat(),
                price_usd=float(s.price_usd),
                volume_24h=float(s.volume_24h) if s.volume_24h else None,
                market_cap=float(s.market_cap) if s.market_cap else None,
                price_change_24h=float(s.price_change_24h) if s.price_change_24h else None,
            )
            for s in snapshots
        ],
        count=len(snapshots)
    )


@router.get("/compare", response_model=ComparisonResponse)
async def compare_assets(
    assets: str = Query(..., description="Comma-separated asset symbols (e.g., BTC,ETH)"),
    from_date: Optional[str] = Query(None, alias="from", description="Start date (ISO format)"),
    to_date: Optional[str] = Query(None, alias="to", description="End date (ISO format)"),
    normalize: bool = Query(False, description="Normalize prices to percentage change"),
    db: AsyncSession = Depends(get_db)
):
    """
    Get comparison data for multiple assets.
    
    - **assets**: Comma-separated asset symbols (e.g., BTC,ETH)
    - **from**: Start date in ISO format (optional)
    - **to**: End date in ISO format (optional)
    - **normalize**: If true, return percentage change from first value
    """
    symbols = [s.strip().upper() for s in assets.split(",")]
    
    # Find all assets
    asset_objs = []
    for symbol in symbols:
        asset_obj = await crud.get_asset_by_symbol(db, symbol)
        if asset_obj:
            asset_objs.append(asset_obj)
    
    if not asset_objs:
        raise HTTPException(status_code=404, detail="No valid assets found")
    
    # Parse dates
    from_dt = None
    to_dt = None
    if from_date:
        try:
            from_dt = datetime.fromisoformat(from_date.replace("Z", "+00:00"))
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid 'from' date format")
    if to_date:
        try:
            to_dt = datetime.fromisoformat(to_date.replace("Z", "+00:00"))
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid 'to' date format")
    
    # Get snapshots for all assets
    asset_ids = [a.id for a in asset_objs]
    all_snapshots = await crud.get_snapshots_for_comparison(db, asset_ids, from_dt, to_dt)
    
    # Group by timestamp and organize by asset
    symbol_map = {a.id: a.symbol for a in asset_objs}
    base_prices = {}  # For normalization
    
    # Group snapshots by timestamp
    timestamp_data = {}
    for snapshot in all_snapshots:
        ts_key = snapshot.timestamp.isoformat()
        if ts_key not in timestamp_data:
            timestamp_data[ts_key] = {}
        
        symbol = symbol_map[snapshot.asset_id]
        price = float(snapshot.price_usd)
        
        # Track first price for normalization
        if symbol not in base_prices:
            base_prices[symbol] = price
        
        # Normalize if requested
        if normalize and base_prices[symbol] != 0:
            price = ((price - base_prices[symbol]) / base_prices[symbol]) * 100
        
        timestamp_data[ts_key][symbol] = price
    
    # Convert to response format
    comparison_data = [
        ComparisonDataPoint(timestamp=ts, prices=prices)
        for ts, prices in sorted(timestamp_data.items())
    ]
    
    return ComparisonResponse(
        assets=[a.symbol for a in asset_objs],
        data=comparison_data
    )


@router.get("/anomalies", response_model=List[AnomalyResponse])
async def get_anomalies(
    asset: str = Query(..., description="Asset symbol (e.g., BTC)"),
    threshold: float = Query(2.5, description="Z-score threshold for anomaly detection"),
    from_date: Optional[str] = Query(None, alias="from", description="Start date (ISO format)"),
    to_date: Optional[str] = Query(None, alias="to", description="End date (ISO format)"),
    db: AsyncSession = Depends(get_db)
):
    """
    Get detected price anomalies for an asset.
    
    - **asset**: Asset symbol (e.g., BTC)
    - **threshold**: Z-score threshold (default: 2.5)
    - **from**: Start date (optional)
    - **to**: End date (optional)
    """
    # Find asset
    asset_obj = await crud.get_asset_by_symbol(db, asset)
    if not asset_obj:
        raise HTTPException(status_code=404, detail=f"Asset '{asset}' not found")
    
    # Parse dates
    from_dt = None
    to_dt = None
    if from_date:
        try:
            from_dt = datetime.fromisoformat(from_date.replace("Z", "+00:00"))
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid 'from' date format")
    if to_date:
        try:
            to_dt = datetime.fromisoformat(to_date.replace("Z", "+00:00"))
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid 'to' date format")
    
    # Get snapshots
    snapshots = await crud.get_snapshots_by_asset(db, asset_obj.id, from_dt, to_dt)
    
    # Detect anomalies
    anomalies = detect_anomalies(snapshots, threshold, asset_symbol=asset_obj.symbol)
    
    return [
        AnomalyResponse(
            timestamp=a.timestamp.isoformat(),
            asset_symbol=a.asset_symbol,
            price=a.price,
            anomaly_type=a.anomaly_type,
            severity=a.severity,
            price_change=a.price_change,
        )
        for a in anomalies
    ]


@router.get("/volatility", response_model=List[VolatilityResponse])
async def get_volatility(
    asset: str = Query(..., description="Asset symbol (e.g., BTC)"),
    window: int = Query(20, description="Rolling window size"),
    from_date: Optional[str] = Query(None, alias="from", description="Start date (ISO format)"),
    to_date: Optional[str] = Query(None, alias="to", description="End date (ISO format)"),
    db: AsyncSession = Depends(get_db)
):
    """
    Get rolling volatility data for an asset.
    
    - **asset**: Asset symbol (e.g., BTC)
    - **window**: Rolling window size (default: 20)
    """
    # Find asset
    asset_obj = await crud.get_asset_by_symbol(db, asset)
    if not asset_obj:
        raise HTTPException(status_code=404, detail=f"Asset '{asset}' not found")
    
    # Parse dates
    from_dt = None
    to_dt = None
    if from_date:
        try:
            from_dt = datetime.fromisoformat(from_date.replace("Z", "+00:00"))
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid 'from' date format")
    if to_date:
        try:
            to_dt = datetime.fromisoformat(to_date.replace("Z", "+00:00"))
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid 'to' date format")
    
    # Get snapshots
    snapshots = await crud.get_snapshots_by_asset(db, asset_obj.id, from_dt, to_dt)
    
    # Calculate volatility
    volatility_data = calculate_volatility(snapshots, window)
    
    return [
        VolatilityResponse(timestamp=v["timestamp"], volatility=v["volatility"])
        for v in volatility_data
    ]


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}
