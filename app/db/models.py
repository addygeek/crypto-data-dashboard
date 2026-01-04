"""Database models for crypto data storage."""
from datetime import datetime
from decimal import Decimal
from typing import Optional

from sqlalchemy import Column, Integer, String, DateTime, Numeric, ForeignKey, Index
from sqlalchemy.orm import relationship, declarative_base

Base = declarative_base()


class CryptoAsset(Base):
    """Model representing a cryptocurrency asset."""
    
    __tablename__ = "crypto_assets"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    coingecko_id = Column(String(100), unique=True, nullable=False, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    name = Column(String(100), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationship to price snapshots
    snapshots = relationship("PriceSnapshot", back_populates="asset", lazy="dynamic")
    
    def __repr__(self):
        return f"<CryptoAsset(symbol='{self.symbol}', name='{self.name}')>"


class PriceSnapshot(Base):
    """Model representing a price snapshot at a point in time."""
    
    __tablename__ = "price_snapshots"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    asset_id = Column(Integer, ForeignKey("crypto_assets.id"), nullable=False)
    price_usd = Column(Numeric(20, 8), nullable=False)
    volume_24h = Column(Numeric(30, 2), nullable=True)
    market_cap = Column(Numeric(30, 2), nullable=True)
    price_change_24h = Column(Numeric(10, 4), nullable=True)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)
    
    # Relationship to asset
    asset = relationship("CryptoAsset", back_populates="snapshots")
    
    # Indexes for efficient querying
    __table_args__ = (
        Index("ix_snapshot_asset_timestamp", "asset_id", "timestamp"),
        Index("ix_snapshot_timestamp", "timestamp"),
    )
    
    def __repr__(self):
        return f"<PriceSnapshot(asset_id={self.asset_id}, price=${self.price_usd}, time={self.timestamp})>"
    
    def to_dict(self):
        """Convert snapshot to dictionary."""
        return {
            "id": self.id,
            "asset_id": self.asset_id,
            "price_usd": float(self.price_usd) if self.price_usd else None,
            "volume_24h": float(self.volume_24h) if self.volume_24h else None,
            "market_cap": float(self.market_cap) if self.market_cap else None,
            "price_change_24h": float(self.price_change_24h) if self.price_change_24h else None,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
        }
