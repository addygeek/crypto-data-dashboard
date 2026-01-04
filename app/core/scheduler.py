"""Background job scheduler for data ingestion."""
import asyncio
import logging
from typing import Optional
from datetime import datetime

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger

from app.config import get_settings
from app.core.coingecko import fetch_crypto_prices, get_coingecko_client
from app.core.normalizer import normalize_simple_price
from app.db.database import get_session
from app.db import crud

logger = logging.getLogger(__name__)
settings = get_settings()


class DataIngestionScheduler:
    """Scheduler for periodic crypto data ingestion."""
    
    def __init__(self):
        self.scheduler: Optional[AsyncIOScheduler] = None
        self._is_running = False
    
    def start(self):
        """Start the scheduler."""
        if self._is_running:
            logger.warning("Scheduler already running")
            return
        
        self.scheduler = AsyncIOScheduler()
        
        # Add ingestion job
        self.scheduler.add_job(
            self._ingest_data,
            trigger=IntervalTrigger(minutes=settings.ingestion_interval_minutes),
            id="crypto_ingestion",
            name="Crypto Data Ingestion",
            replace_existing=True,
            max_instances=1,
        )
        
        self.scheduler.start()
        self._is_running = True
        logger.info(f"Scheduler started. Ingestion interval: {settings.ingestion_interval_minutes} minutes")
    
    def stop(self):
        """Stop the scheduler."""
        if self.scheduler and self._is_running:
            self.scheduler.shutdown(wait=False)
            self._is_running = False
            logger.info("Scheduler stopped")
    
    async def _ingest_data(self):
        """Fetch and store crypto price data."""
        logger.info("Starting data ingestion...")
        start_time = datetime.utcnow()
        
        try:
            # Fetch prices from CoinGecko
            raw_data = await fetch_crypto_prices()
            
            if not raw_data:
                logger.error("No data received from CoinGecko")
                return
            
            # Normalize the data
            normalized_data = normalize_simple_price(raw_data)
            
            if not normalized_data:
                logger.error("No data after normalization")
                return
            
            # Store in database
            async with get_session() as session:
                for price_data in normalized_data:
                    # Get or create the asset
                    asset = await crud.get_or_create_asset(
                        session,
                        coingecko_id=price_data.coingecko_id,
                        symbol=price_data.symbol,
                        name=price_data.name,
                    )
                    
                    # Create snapshot
                    await crud.create_snapshot(
                        session,
                        asset_id=asset.id,
                        price_usd=price_data.price_usd,
                        volume_24h=price_data.volume_24h,
                        market_cap=price_data.market_cap,
                        price_change_24h=price_data.price_change_24h,
                        timestamp=price_data.timestamp,
                    )
                
                await session.commit()
            
            elapsed = (datetime.utcnow() - start_time).total_seconds()
            logger.info(f"Data ingestion complete. Stored {len(normalized_data)} snapshots in {elapsed:.2f}s")
            
        except Exception as e:
            logger.error(f"Error during data ingestion: {e}", exc_info=True)
    
    async def run_once(self):
        """Run ingestion once immediately."""
        await self._ingest_data()


# Global scheduler instance
_scheduler: Optional[DataIngestionScheduler] = None


def get_scheduler() -> DataIngestionScheduler:
    """Get or create global scheduler instance."""
    global _scheduler
    if _scheduler is None:
        _scheduler = DataIngestionScheduler()
    return _scheduler


async def start_ingestion():
    """Start data ingestion scheduler."""
    scheduler = get_scheduler()
    scheduler.start()
    # Run once immediately
    await scheduler.run_once()


async def stop_ingestion():
    """Stop data ingestion scheduler."""
    scheduler = get_scheduler()
    scheduler.stop()
    
    # Also close the CoinGecko client
    client = get_coingecko_client()
    await client.close()
