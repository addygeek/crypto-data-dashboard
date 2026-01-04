"""Anomaly detection service for crypto prices."""
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from dataclasses import dataclass
from decimal import Decimal

import numpy as np

from app.db.models import PriceSnapshot

logger = logging.getLogger(__name__)


@dataclass
class Anomaly:
    """Represents a detected anomaly."""
    timestamp: datetime
    asset_id: int
    asset_symbol: str
    price: float
    anomaly_type: str  # 'spike' or 'drop'
    severity: float  # Z-score
    price_change: float  # Percentage change
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "asset_id": self.asset_id,
            "asset_symbol": self.asset_symbol,
            "price": self.price,
            "anomaly_type": self.anomaly_type,
            "severity": round(self.severity, 2),
            "price_change": round(self.price_change, 2),
        }


def calculate_z_scores(values: List[float], window: int = 20) -> List[float]:
    """
    Calculate rolling Z-scores for a series of values.
    
    Args:
        values: List of price values
        window: Rolling window size for mean/std calculation
    
    Returns:
        List of Z-scores (NaN for insufficient data)
    """
    if len(values) < window:
        return [float('nan')] * len(values)
    
    arr = np.array(values, dtype=np.float64)
    z_scores = []
    
    for i in range(len(arr)):
        if i < window - 1:
            z_scores.append(float('nan'))
        else:
            window_data = arr[i - window + 1:i + 1]
            mean = np.mean(window_data[:-1])  # Exclude current point
            std = np.std(window_data[:-1])
            
            if std == 0:
                z_scores.append(0.0)
            else:
                z_score = (arr[i] - mean) / std
                z_scores.append(float(z_score))
    
    return z_scores


def calculate_price_changes(values: List[float]) -> List[float]:
    """Calculate percentage changes between consecutive values."""
    if len(values) < 2:
        return [0.0] * len(values)
    
    changes = [0.0]  # First value has no change
    for i in range(1, len(values)):
        if values[i - 1] != 0:
            change = ((values[i] - values[i - 1]) / values[i - 1]) * 100
            changes.append(change)
        else:
            changes.append(0.0)
    
    return changes


def detect_anomalies(
    snapshots: List[PriceSnapshot],
    z_threshold: float = 2.5,
    window: int = 20,
    asset_symbol: str = "UNKNOWN"
) -> List[Anomaly]:
    """
    Detect anomalies in price data using Z-score method.
    
    Args:
        snapshots: List of price snapshots (must be ordered by timestamp)
        z_threshold: Z-score threshold for anomaly detection
        window: Rolling window size
        asset_symbol: Symbol for reference
    
    Returns:
        List of detected anomalies
    """
    if len(snapshots) < window:
        logger.debug(f"Insufficient data for anomaly detection (need {window}, got {len(snapshots)})")
        return []
    
    # Extract prices
    prices = [float(s.price_usd) for s in snapshots]
    timestamps = [s.timestamp for s in snapshots]
    asset_ids = [s.asset_id for s in snapshots]
    
    # Calculate Z-scores
    z_scores = calculate_z_scores(prices, window)
    
    # Calculate price changes
    price_changes = calculate_price_changes(prices)
    
    # Detect anomalies
    anomalies = []
    for i, (z, change) in enumerate(zip(z_scores, price_changes)):
        if np.isnan(z):
            continue
        
        if abs(z) >= z_threshold:
            anomaly_type = "spike" if z > 0 else "drop"
            anomaly = Anomaly(
                timestamp=timestamps[i],
                asset_id=asset_ids[i],
                asset_symbol=asset_symbol,
                price=prices[i],
                anomaly_type=anomaly_type,
                severity=abs(z),
                price_change=change,
            )
            anomalies.append(anomaly)
    
    return anomalies


def calculate_volatility(
    snapshots: List[PriceSnapshot],
    window: int = 20
) -> List[Dict]:
    """
    Calculate rolling volatility (standard deviation).
    
    Returns list of {timestamp, volatility} dictionaries.
    """
    if len(snapshots) < window:
        return []
    
    prices = np.array([float(s.price_usd) for s in snapshots])
    timestamps = [s.timestamp for s in snapshots]
    
    volatility_data = []
    for i in range(window - 1, len(prices)):
        window_prices = prices[i - window + 1:i + 1]
        
        # Calculate returns
        returns = np.diff(window_prices) / window_prices[:-1]
        
        # Volatility as standard deviation of returns
        volatility = float(np.std(returns) * 100)  # As percentage
        
        volatility_data.append({
            "timestamp": timestamps[i].isoformat(),
            "volatility": round(volatility, 4),
        })
    
    return volatility_data
