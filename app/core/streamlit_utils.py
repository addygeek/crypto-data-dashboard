"""Utilities for running async backend logic in Streamlit."""
import asyncio
import logging
from typing import Any, Coroutine
import threading

import streamlit as st
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.database import init_db, get_session
from app.core.scheduler import start_ingestion, stop_ingestion

logger = logging.getLogger(__name__)

def run_async(coro: Coroutine[Any, Any, Any]) -> Any:
    """
    Run an async coroutine synchronously in a separate thread.
    
    This avoids conflicts with Streamlit's internal event loop (especially on Cloud 
    where uvloop might be used) by creating a fresh loop in a dedicated thread.
    """
    result = []
    error = []

    def _target():
        try:
            # Create a new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # Run the coroutine
            res = loop.run_until_complete(coro)
            result.append(res)
            
            # Clean up
            loop.close()
        except Exception as e:
            error.append(e)

    # Start thread
    t = threading.Thread(target=_target)
    t.start()
    t.join()

    if error:
        raise error[0]
        
    return result[0] if result else None

@st.cache_resource
def initialize_backend():
    """
    Initialize the DB and Scheduler once.
    """
    logger.info("Initializing monolithic backend...")
    
    async def _init():
        await init_db()
        await start_ingestion()
        
    try:
        run_async(_init())
        logger.info("Backend initialized.")
    except Exception as e:
        logger.error(f"Failed to initialize backend: {e}")
        st.error(f"Backend initialization failed: {e}")

def get_db_session() -> AsyncSession:
    """Get a database session generator."""
    return get_session()
