"""System endpoints for TrustLens AI backend.

Exposes lightweight, unauthenticated endpoints used for service
discovery, liveness/readiness probing, and detailed status reporting
(including database connectivity). These endpoints are typically
consumed by load balancers, container orchestrators, and monitoring
tooling.
"""

import logging
import time

from fastapi import APIRouter, status
from pydantic import BaseModel, Field

from core.config import settings
from database.database import check_database_connection

logger = logging.getLogger(__name__)

router = APIRouter()

# Monotonic reference point captured at module import time, used to
# derive an approximate process uptime for the /status endpoint.
_PROCESS_START_TIME: float = time.monotonic()


class RootResponse(BaseModel):
    """Response schema for the API root endpoint."""

    message: str = Field(..., description="Human-readable welcome message.")
    application: str = Field(..., description="Application name.")
    version: str = Field(..., description="Current application version.")
    docs_url: str = Field(..., description="Path to the Swagger UI documentation.")


class HealthResponse(BaseModel):
    """Response schema for the liveness health check endpoint."""

    status: str = Field(..., description="Overall liveness status of the API.")


class StatusResponse(BaseModel):
    """Response schema for the detailed status endpoint."""

    status: str = Field(..., description="Overall API status.")
    version: str = Field(..., description="Current application version.")
    environment: str = Field(..., description="Deployment environment name.")
    database_connected: bool = Field(
        ..., description="Whether the database is currently reachable."
    )
    uptime_seconds: float = Field(
        ..., description="Approximate process uptime in seconds since import."
    )


@router.get(
    "/",
    response_model=RootResponse,
    status_code=status.HTTP_200_OK,
    summary="API root",
)
async def read_root() -> RootResponse:
    """Return basic API identification information.

    Returns:
        RootResponse: A welcome message along with application name,
            version, and a pointer to the interactive documentation.
    """
    logger.debug("Root endpoint accessed.")
    return RootResponse(
        message=f"Welcome to {settings.APP_NAME}",
        application=settings.APP_NAME,
        version=settings.APP_VERSION,
        docs_url="/docs",
    )


@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Liveness health check",
)
async def health_check() -> HealthResponse:
    """Report a lightweight liveness signal for the API process.

    This endpoint intentionally avoids any external dependency checks
    (e.g. database) so that it responds quickly and reliably for
    liveness probes.

    Returns:
        HealthResponse: A simple status payload indicating the API
            process is alive and serving requests.
    """
    return HealthResponse(status="ok")


@router.get(
    "/status",
    response_model=StatusResponse,
    status_code=status.HTTP_200_OK,
    summary="Detailed system status",
)
async def system_status() -> StatusResponse:
    """Report detailed system status, including database connectivity.

    Performs an active check against the database to determine
    connectivity and computes an approximate process uptime.

    Returns:
        StatusResponse: Aggregated status information including
            application version, environment, database connectivity,
            and uptime in seconds.
    """
    db_connected = await check_database_connection()
    if not db_connected:
        logger.warning("System status check: database is unreachable.")

    uptime_seconds = time.monotonic() - _PROCESS_START_TIME

    return StatusResponse(
        status="operational" if db_connected else "degraded",
        version=settings.APP_VERSION,
        environment=settings.ENVIRONMENT,
        database_connected=db_connected,
        uptime_seconds=round(uptime_seconds, 2),
    )