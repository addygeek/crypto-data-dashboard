"""Application Configuration."""
from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import List


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Database
    database_url: str = "sqlite+aiosqlite:///./crypto_data.db"
    
    # CoinGecko
    coingecko_api_key: str = ""
    coingecko_base_url: str = "https://api.coingecko.com/api/v3"
    
    # Scheduler
    ingestion_interval_minutes: int = 5
    
    # Supported Assets (CoinGecko IDs)
    supported_assets: str = "bitcoin,ethereum,solana,ripple,dogecoin"
    
    # Server
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    
    @property
    def asset_list(self) -> List[str]:
        """Get supported assets as a list."""
        return [a.strip() for a in self.supported_assets.split(",")]
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
