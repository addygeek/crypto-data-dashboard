"""Utilities for running async backend logic in Streamlit."""
import asyncio
import logging
from typing import Any, Coroutine
import threading

import streamlit as st
import nest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.database import init_db, get_session
from app.core.scheduler import start_ingestion, stop_ingestion

# Apply nest_asyncio to allow nested event loops (Streamlit -> asyncio.run)
nest_asyncio.apply()

logger = logging.getLogger(__name__)

def run_async(coro: Coroutine[Any, Any, Any]) -> Any:
    """
    Run an async coroutine synchronously.
    
    This is necessary because Streamlit runs in a separate thread/loop 
    and we need to block for the result of backend operations.
    """
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
    return loop.run_until_complete(coro)

@st.cache_resource
def initialize_backend():
    """
    Initialize the DB and Scheduler once.
    
    This function uses st.cache_resource to ensure it only runs once
    per Streamlit server session, not on every script rerun.
    """
    logger.info("Initializing monolithic backend...")
    
    async def _init():
        await init_db()
        await start_ingestion()
        
    run_async(_init())
    logger.info("Backend initialized.")

def get_db_session() -> AsyncSession:
    """Get a database session generator."""
    # Note: In a real async-to-sync bridge, handling the session context properly is tricky.
    # We'll use a helper that yields the session for a context manager.
    return get_session()
