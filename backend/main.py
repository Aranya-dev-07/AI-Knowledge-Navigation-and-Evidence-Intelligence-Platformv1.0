"""Application entry point for TrustLens AI backend.

This module is responsible for constructing the FastAPI application
instance, wiring up configuration, logging, middleware, routers, and
application lifecycle (startup/shutdown) events. It is the single
process entry point executed by Uvicorn.

Typical usage:
    Run directly for local development::

        python main.py

    Or via Uvicorn's CLI for production::

        uvicorn main:app --host 0.0.0.0 --port 8000
"""

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.router import api_router
from core.config import settings
from core.logging import setup_logging
from database.database import dispose_engine
from database.init_db import init_db

setup_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application startup and shutdown events.

    Handles initialization of external resources (database schema
    verification/creation) on startup and graceful release of those
    resources (database engine disposal) on shutdown.

    Args:
        app: The FastAPI application instance being managed.

    Yields:
        None: Control is yielded back to FastAPI while the application
            serves requests. Code after the ``yield`` runs on shutdown.
    """
    logger.info("Starting %s v%s", settings.APP_NAME, settings.APP_VERSION)
    try:
        await init_db()
        logger.info("Database initialization completed successfully.")
    except Exception:
        logger.exception("Database initialization failed during startup.")
        raise

    yield

    logger.info("Shutting down %s", settings.APP_NAME)
    try:
        await dispose_engine()
        logger.info("Database engine disposed successfully.")
    except Exception:
        logger.exception("Error occurred while disposing database engine.")


def create_application() -> FastAPI:
    """Construct and configure the FastAPI application instance.

    Assembles the application with metadata, CORS middleware, the
    aggregated API router, and lifecycle event handlers. Isolated in
    a factory function to support testability (e.g. constructing a
    fresh app instance per test session).

    Returns:
        FastAPI: A fully configured FastAPI application instance.
    """
    application = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description=(
            "TrustLens AI \u2014 AI Knowledge Navigation and Intelligence "
            "Platform backend API."
        ),
        debug=settings.DEBUG,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    application.include_router(api_router, prefix=settings.API_V1_PREFIX)

    return application


app = create_application()


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower(),
    )