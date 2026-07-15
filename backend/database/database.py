"""Database connectivity layer for TrustLens AI backend.

Configures the async SQLAlchemy 2.0 engine and session factory used
throughout the application, exposes the declarative ``Base`` that all
ORM models inherit from, and provides the ``get_db`` FastAPI
dependency for per-request session management. Also exposes helpers
for graceful engine disposal and lightweight connectivity checks used
by the system status endpoint.
"""

import logging
from collections.abc import AsyncGenerator

from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from core.config import settings

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    """Declarative base class from which all ORM models must inherit."""


engine: AsyncEngine = create_async_engine(
    str(settings.DATABASE_URL),
    echo=settings.DATABASE_ECHO,
    pool_size=settings.DATABASE_POOL_SIZE,
    max_overflow=settings.DATABASE_MAX_OVERFLOW,
    pool_pre_ping=True,
)

AsyncSessionLocal: async_sessionmaker[AsyncSession] = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    autoflush=False,
    autocommit=False,
    expire_on_commit=False,
)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Yield a request-scoped async database session.

    Intended for use as a FastAPI dependency (``Depends(get_db)``).
    The session is committed automatically if the request handler
    completes without error, rolled back if an exception propagates,
    and always closed once the request finishes.

    Yields:
        AsyncSession: An active SQLAlchemy async session bound to the
            application's engine.
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            logger.exception("Database session rolled back due to an error.")
            raise
        finally:
            await session.close()


async def check_database_connection() -> bool:
    """Check whether the database is currently reachable.

    Executes a trivial ``SELECT 1`` against the configured database
    using a short-lived connection. Never raises; any failure is
    logged and reported as an unreachable database.

    Returns:
        bool: ``True`` if the database responded successfully,
            ``False`` otherwise.
    """
    try:
        async with engine.connect() as connection:
            await connection.execute(text("SELECT 1"))
        return True
    except Exception:
        logger.exception("Database connectivity check failed.")
        return False


async def dispose_engine() -> None:
    """Dispose of the async engine's connection pool.

    Should be called once during application shutdown to gracefully
    close all pooled database connections and release resources.

    Returns:
        None
    """
    await engine.dispose()
    logger.info("Database engine connection pool disposed.")