"""Data normalization layer for API responses."""
from datetime import datetime, timezone
from decimal import Decimal
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class NormalizedPriceData:
    """Normalized price data structure."""
    coingecko_id: str
    symbol: str
    name: str
    price_usd: Decimal
    volume_24h: Optional[Decimal]
    market_cap: Optional[Decimal]
    price_change_24h: Optional[Decimal]
    timestamp: datetime
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "coingecko_id": self.coingecko_id,
            "symbol": self.symbol,
            "name": self.name,
            "price_usd": float(self.price_usd),
            "volume_24h": float(self.volume_24h) if self.volume_24h else None,
            "market_cap": float(self.market_cap) if self.market_cap else None,
            "price_change_24h": float(self.price_change_24h) if self.price_change_24h else None,
            "timestamp": self.timestamp.isoformat(),
        }


def _safe_decimal(value: Any) -> Optional[Decimal]:
    """Safely convert value to Decimal."""
    if value is None:
        return None
    try:
        return Decimal(str(value))
    except (ValueError, TypeError, Exception):
        # Catch all exceptions including decimal.InvalidOperation
        return None


def _parse_timestamp(value: Any) -> datetime:
    """Parse timestamp from various formats."""
    if value is None:
        return datetime.now(timezone.utc)
    
    if isinstance(value, (int, float)):
        # Unix timestamp
        return datetime.fromtimestamp(value, tz=timezone.utc)
    
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            pass
    
    return datetime.now(timezone.utc)


# Mapping of CoinGecko IDs to symbols and names
COIN_MAPPING = {
    "bitcoin": ("BTC", "Bitcoin"),
    "ethereum": ("ETH", "Ethereum"),
    "solana": ("SOL", "Solana"),
    "ripple": ("XRP", "XRP"),
    "dogecoin": ("DOGE", "Dogecoin"),
    "cardano": ("ADA", "Cardano"),
    "polkadot": ("DOT", "Polkadot"),
    "chainlink": ("LINK", "Chainlink"),
    "litecoin": ("LTC", "Litecoin"),
    "binancecoin": ("BNB", "BNB"),
}


def normalize_simple_price(raw_data: Dict) -> List[NormalizedPriceData]:
    """
    Normalize response from /simple/price endpoint.
    
    Expected format:
    {
        "bitcoin": {
            "usd": 42000,
            "usd_24h_vol": 25000000000,
            "usd_24h_change": 2.5,
            "usd_market_cap": 800000000000,
            "last_updated_at": 1704000000
        },
        ...
    }
    """
    normalized = []
    now = datetime.now(timezone.utc)
    
    for coin_id, data in raw_data.items():
        try:
            # Get symbol and name from mapping or use defaults
            symbol, name = COIN_MAPPING.get(coin_id, (coin_id.upper()[:5], coin_id.title()))
            
            # Extract and convert values
            price = _safe_decimal(data.get("usd"))
            if price is None:
                logger.warning(f"Missing price for {coin_id}, skipping")
                continue
            
            timestamp = _parse_timestamp(data.get("last_updated_at"))
            
            normalized_data = NormalizedPriceData(
                coingecko_id=coin_id,
                symbol=symbol,
                name=name,
                price_usd=price,
                volume_24h=_safe_decimal(data.get("usd_24h_vol")),
                market_cap=_safe_decimal(data.get("usd_market_cap")),
                price_change_24h=_safe_decimal(data.get("usd_24h_change")),
                timestamp=timestamp,
            )
            normalized.append(normalized_data)
            
        except Exception as e:
            logger.error(f"Error normalizing data for {coin_id}: {e}")
            continue
    
    return normalized


def normalize_market_data(raw_data: List[Dict]) -> List[NormalizedPriceData]:
    """
    Normalize response from /coins/markets endpoint.
    
    Expected format:
    [
        {
            "id": "bitcoin",
            "symbol": "btc",
            "name": "Bitcoin",
            "current_price": 42000,
            "market_cap": 800000000000,
            "total_volume": 25000000000,
            "price_change_24h": 1000,
            "price_change_percentage_24h": 2.5,
            "last_updated": "2024-01-01T00:00:00.000Z"
        },
        ...
    ]
    """
    normalized = []
    
    for coin_data in raw_data:
        try:
            coin_id = coin_data.get("id")
            if not coin_id:
                continue
            
            price = _safe_decimal(coin_data.get("current_price"))
            if price is None:
                logger.warning(f"Missing price for {coin_id}, skipping")
                continue
            
            normalized_data = NormalizedPriceData(
                coingecko_id=coin_id,
                symbol=coin_data.get("symbol", "").upper(),
                name=coin_data.get("name", coin_id.title()),
                price_usd=price,
                volume_24h=_safe_decimal(coin_data.get("total_volume")),
                market_cap=_safe_decimal(coin_data.get("market_cap")),
                price_change_24h=_safe_decimal(coin_data.get("price_change_percentage_24h")),
                timestamp=_parse_timestamp(coin_data.get("last_updated")),
            )
            normalized.append(normalized_data)
            
        except Exception as e:
            logger.error(f"Error normalizing market data: {e}")
            continue
    
    return normalized
