"""CoinGecko API Client with async support."""
import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

import httpx

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class CoinGeckoClient:
    """Async client for CoinGecko API."""
    
    BASE_URL = "https://api.coingecko.com/api/v3"
    
    def __init__(self):
        self.base_url = settings.coingecko_base_url
        self.api_key = settings.coingecko_api_key
        self._client: Optional[httpx.AsyncClient] = None
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None or self._client.is_closed:
            headers = {"Accept": "application/json"}
            if self.api_key:
                headers["x-cg-demo-api-key"] = self.api_key
            
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers=headers,
                timeout=30.0,
            )
        return self._client
    
    async def close(self):
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
    
    async def _make_request(
        self,
        endpoint: str,
        params: Optional[Dict] = None,
        retries: int = 3
    ) -> Optional[Any]:
        """Make API request with retry logic."""
        client = await self._get_client()
        
        for attempt in range(retries):
            try:
                response = await client.get(endpoint, params=params)
                
                # Handle rate limiting
                if response.status_code == 429:
                    wait_time = 2 ** attempt * 10  # Exponential backoff
                    logger.warning(f"Rate limited. Waiting {wait_time}s before retry...")
                    await asyncio.sleep(wait_time)
                    continue
                
                response.raise_for_status()
                return response.json()
                
            except httpx.HTTPStatusError as e:
                logger.error(f"HTTP error on attempt {attempt + 1}: {e}")
                if attempt < retries - 1:
                    await asyncio.sleep(2 ** attempt)
                else:
                    raise
            except httpx.RequestError as e:
                logger.error(f"Request error on attempt {attempt + 1}: {e}")
                if attempt < retries - 1:
                    await asyncio.sleep(2 ** attempt)
                else:
                    raise
        
        return None
    
    async def get_prices(self, coin_ids: List[str]) -> Optional[Dict]:
        """
        Get current prices for multiple coins.
        
        Args:
            coin_ids: List of CoinGecko coin IDs (e.g., ['bitcoin', 'ethereum'])
        
        Returns:
            Dict with coin data or None if request fails
        """
        params = {
            "ids": ",".join(coin_ids),
            "vs_currencies": "usd",
            "include_24hr_vol": "true",
            "include_24hr_change": "true",
            "include_market_cap": "true",
            "include_last_updated_at": "true",
        }
        
        return await self._make_request("/simple/price", params=params)
    
    async def get_coin_list(self) -> Optional[List[Dict]]:
        """Get list of all supported coins."""
        return await self._make_request("/coins/list")
    
    async def get_coin_details(self, coin_id: str) -> Optional[Dict]:
        """Get detailed information for a coin."""
        params = {
            "localization": "false",
            "tickers": "false",
            "market_data": "true",
            "community_data": "false",
            "developer_data": "false",
        }
        return await self._make_request(f"/coins/{coin_id}", params=params)
    
    async def get_market_data(
        self,
        coin_ids: List[str],
        vs_currency: str = "usd"
    ) -> Optional[List[Dict]]:
        """
        Get market data for coins with more details.
        
        Returns list of coin data including:
        - current_price
        - market_cap
        - total_volume
        - price_change_24h
        - price_change_percentage_24h
        """
        params = {
            "ids": ",".join(coin_ids),
            "vs_currency": vs_currency,
            "order": "market_cap_desc",
            "per_page": 250,
            "page": 1,
            "sparkline": "false",
            "price_change_percentage": "24h",
        }
        
        return await self._make_request("/coins/markets", params=params)


# Global client instance
_client: Optional[CoinGeckoClient] = None


def get_coingecko_client() -> CoinGeckoClient:
    """Get or create global CoinGecko client."""
    global _client
    if _client is None:
        _client = CoinGeckoClient()
    return _client


async def fetch_crypto_prices(coin_ids: Optional[List[str]] = None) -> Optional[Dict]:
    """
    Convenience function to fetch crypto prices.
    
    Args:
        coin_ids: List of coin IDs or None to use configured assets
    
    Returns:
        Price data dictionary
    """
    if coin_ids is None:
        coin_ids = settings.asset_list
    
    client = get_coingecko_client()
    return await client.get_prices(coin_ids)


async def fetch_market_data(coin_ids: Optional[List[str]] = None) -> Optional[List[Dict]]:
    """
    Convenience function to fetch market data.
    
    Args:
        coin_ids: List of coin IDs or None to use configured assets
    
    Returns:
        List of market data dictionaries
    """
    if coin_ids is None:
        coin_ids = settings.asset_list
    
    client = get_coingecko_client()
    return await client.get_market_data(coin_ids)
