"""Tests for data ingestion and normalization."""
import pytest
from datetime import datetime, timezone
from decimal import Decimal

from app.core.normalizer import (
    normalize_simple_price,
    normalize_market_data,
    NormalizedPriceData,
    _safe_decimal,
    _parse_timestamp,
)
from app.services.anomaly import (
    calculate_z_scores,
    calculate_price_changes,
    detect_anomalies,
)
from app.db.models import PriceSnapshot


class TestNormalizer:
    """Tests for data normalization."""
    
    def test_normalize_simple_price(self):
        """Test normalization of simple price API response."""
        raw_data = {
            "bitcoin": {
                "usd": 42000.50,
                "usd_24h_vol": 25000000000,
                "usd_24h_change": 2.5,
                "usd_market_cap": 800000000000,
                "last_updated_at": 1704000000
            },
            "ethereum": {
                "usd": 2500.25,
                "usd_24h_vol": 15000000000,
                "usd_24h_change": -1.2,
                "usd_market_cap": 300000000000,
                "last_updated_at": 1704000000
            }
        }
        
        result = normalize_simple_price(raw_data)
        
        assert len(result) == 2
        
        btc = next(d for d in result if d.coingecko_id == "bitcoin")
        assert btc.symbol == "BTC"
        assert btc.name == "Bitcoin"
        assert btc.price_usd == Decimal("42000.50")
        assert btc.volume_24h == Decimal("25000000000")
        assert btc.price_change_24h == Decimal("2.5")
    
    def test_normalize_missing_price(self):
        """Test normalization skips entries without price."""
        raw_data = {
            "bitcoin": {"usd": 42000},
            "invalid": {"volume": 1000}  # No USD price
        }
        
        result = normalize_simple_price(raw_data)
        
        assert len(result) == 1
        assert result[0].coingecko_id == "bitcoin"
    
    def test_safe_decimal_conversion(self):
        """Test safe decimal conversion."""
        assert _safe_decimal(100) == Decimal("100")
        # Float comparison needs rounding due to float precision
        result = _safe_decimal(100.5)
        assert result is not None
        assert float(result) == 100.5
        assert _safe_decimal("200") == Decimal("200")
        assert _safe_decimal(None) is None
        assert _safe_decimal("invalid") is None
    
    def test_parse_timestamp(self):
        """Test timestamp parsing."""
        # Unix timestamp
        ts = _parse_timestamp(1704000000)
        assert ts.year == 2023 or ts.year == 2024
        
        # ISO string
        ts = _parse_timestamp("2024-01-01T00:00:00Z")
        assert ts.year == 2024
        
        # None defaults to now
        ts = _parse_timestamp(None)
        assert ts.date() == datetime.now(timezone.utc).date()


class TestAnomalyDetection:
    """Tests for anomaly detection."""
    
    def test_calculate_z_scores(self):
        """Test Z-score calculation."""
        # Use values with some variance so std != 0
        values = [100, 101, 99, 100, 102, 150]  # Last value is anomaly
        z_scores = calculate_z_scores(values, window=5)
        
        # Last value should have high Z-score (150 is far from ~100 mean)
        assert abs(z_scores[-1]) > 2
    
    def test_calculate_z_scores_insufficient_data(self):
        """Test Z-scores with insufficient data."""
        values = [100, 101, 102]
        z_scores = calculate_z_scores(values, window=5)
        
        # All should be NaN
        import math
        assert all(math.isnan(z) for z in z_scores)
    
    def test_calculate_price_changes(self):
        """Test price change calculation."""
        values = [100, 110, 99, 100]
        changes = calculate_price_changes(values)
        
        assert changes[0] == 0  # First has no change
        assert changes[1] == 10.0  # +10%
        assert abs(changes[2] - (-10.0)) < 0.1  # ~-10%
    
    def test_detect_anomalies_empty(self):
        """Test anomaly detection with insufficient data."""
        # Create mock snapshots
        class MockSnapshot:
            def __init__(self, price, ts):
                self.price_usd = Decimal(str(price))
                self.timestamp = ts
                self.asset_id = 1
        
        snapshots = [MockSnapshot(100, datetime.now()) for _ in range(5)]
        anomalies = detect_anomalies(snapshots, window=20)
        
        # Not enough data
        assert len(anomalies) == 0


class TestNormalizedPriceData:
    """Tests for NormalizedPriceData dataclass."""
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        data = NormalizedPriceData(
            coingecko_id="bitcoin",
            symbol="BTC",
            name="Bitcoin",
            price_usd=Decimal("42000"),
            volume_24h=Decimal("25000000000"),
            market_cap=Decimal("800000000000"),
            price_change_24h=Decimal("2.5"),
            timestamp=datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        )
        
        result = data.to_dict()
        
        assert result["coingecko_id"] == "bitcoin"
        assert result["symbol"] == "BTC"
        assert result["price_usd"] == 42000.0
        assert result["volume_24h"] == 25000000000.0
