"""FastAPI main application entry point."""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.db.database import init_db
from app.core.scheduler import start_ingestion, stop_ingestion
from app.api.routes import router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    logger.info("Starting Crypto Dashboard API...")
    
    # Initialize database
    await init_db()
    logger.info("Database initialized")
    
    # Start data ingestion scheduler
    await start_ingestion()
    logger.info("Data ingestion scheduler started")
    
    yield
    
    # Shutdown
    logger.info("Shutting down...")
    await stop_ingestion()
    logger.info("Scheduler stopped")


# Create FastAPI application
app = FastAPI(
    title="Live Crypto Market Monitoring Dashboard",
    description="Real-time crypto price tracking with historical analysis, comparison, and anomaly detection.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router, prefix="/api", tags=["Market Data"])


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Live Crypto Market Monitoring Dashboard",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "assets": "/api/assets",
            "prices": "/api/prices?asset=BTC",
            "compare": "/api/compare?assets=BTC,ETH",
            "anomalies": "/api/anomalies?asset=BTC",
            "volatility": "/api/volatility?asset=BTC",
            "health": "/api/health",
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True
    )
