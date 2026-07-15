"""Centralized application configuration for TrustLens AI backend.

Loads and validates all environment-driven configuration using
Pydantic Settings, exposing a single, importable ``settings``
singleton. Every other module in the backend that needs configuration
values (application metadata, database credentials, JWT parameters,
CORS origins, logging level, etc.) must import it from this module
rather than reading environment variables directly.
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field, PostgresDsn, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application-wide configuration loaded from environment variables.

    Attributes:
        APP_NAME: Human-readable application name.
        APP_VERSION: Semantic version of the running application.
        ENVIRONMENT: Deployment environment identifier.
        DEBUG: Whether debug mode (verbose errors, auto-reload) is enabled.
        HOST: Network interface Uvicorn binds to.
        PORT: TCP port Uvicorn listens on.
        API_V1_PREFIX: URL prefix under which all v1 API routes are mounted.
        ALLOWED_ORIGINS: List of origins permitted by CORS middleware.
        DATABASE_URL: Fully qualified async PostgreSQL connection string.
        DATABASE_ECHO: Whether SQLAlchemy should echo emitted SQL statements.
        DATABASE_POOL_SIZE: Number of persistent connections in the pool.
        DATABASE_MAX_OVERFLOW: Number of additional connections allowed
            beyond ``DATABASE_POOL_SIZE`` under load.
        JWT_SECRET_KEY: Secret key used to sign and verify JWT tokens.
        JWT_ALGORITHM: Signing algorithm used for JWT tokens.
        ACCESS_TOKEN_EXPIRE_MINUTES: Lifetime of issued access tokens.
        REFRESH_TOKEN_EXPIRE_DAYS: Lifetime of issued refresh tokens.
        LOG_LEVEL: Minimum severity level captured by the logging system.
        LOG_FORMAT: Output format for log records (``"rich"`` or ``"json"``).
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # --- Project settings ---
    APP_NAME: str = Field(default="TrustLens AI", description="Application name.")
    APP_VERSION: str = Field(default="0.1.0", description="Application version.")
    ENVIRONMENT: Literal["development", "staging", "production", "testing"] = Field(
        default="development", description="Deployment environment."
    )
    DEBUG: bool = Field(default=False, description="Enable debug mode.")
    HOST: str = Field(default="0.0.0.0", description="Host interface to bind Uvicorn to.")
    PORT: int = Field(default=8000, ge=1, le=65535, description="Port Uvicorn listens on.")
    API_V1_PREFIX: str = Field(default="/api/v1", description="API version prefix.")
    ALLOWED_ORIGINS: list[str] = Field(
        default_factory=lambda: ["http://localhost:3000"],
        description="Origins permitted by CORS middleware.",
    )

    # --- Database settings ---
    DATABASE_URL: PostgresDsn = Field(
        ...,
        description="Async PostgreSQL connection URL, e.g. "
        "postgresql+asyncpg://user:password@host:5432/dbname",
    )
    DATABASE_ECHO: bool = Field(default=False, description="Echo raw SQL statements.")
    DATABASE_POOL_SIZE: int = Field(default=10, ge=1, description="DB connection pool size.")
    DATABASE_MAX_OVERFLOW: int = Field(
        default=5, ge=0, description="Additional overflow connections beyond pool size."
    )

    # --- JWT settings ---
    JWT_SECRET_KEY: str = Field(
        ..., min_length=32, description="Secret key used to sign JWT tokens."
    )
    JWT_ALGORITHM: str = Field(default="HS256", description="JWT signing algorithm.")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(
        default=30, ge=1, description="Access token lifetime, in minutes."
    )
    REFRESH_TOKEN_EXPIRE_DAYS: int = Field(
        default=7, ge=1, description="Refresh token lifetime, in days."
    )

    # --- Logging settings ---
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO", description="Minimum log severity level."
    )
    LOG_FORMAT: Literal["rich", "json"] = Field(
        default="rich", description="Log output format."
    )

    @field_validator("ALLOWED_ORIGINS", mode="before")
    @classmethod
    def _split_allowed_origins(cls, value: str | list[str]) -> list[str]:
        """Parse a comma-separated origins string into a list.

        Allows ``ALLOWED_ORIGINS`` to be supplied in ``.env`` files as a
        plain comma-separated string (e.g. ``"http://a.com,http://b.com"``)
        while still accepting a native list when set programmatically.

        Args:
            value: Raw value from the environment, either a comma-separated
                string or an already-parsed list.

        Returns:
            list[str]: Normalized list of origin strings with whitespace
                stripped and empty entries removed.
        """
        if isinstance(value, str):
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        return value

    @field_validator("JWT_SECRET_KEY")
    @classmethod
    def _validate_secret_strength(cls, value: str) -> str:
        """Reject obviously insecure placeholder JWT secrets.

        Args:
            value: The candidate JWT secret key.

        Raises:
            ValueError: If the secret matches a known insecure placeholder.

        Returns:
            str: The validated secret key.
        """
        insecure_defaults = {"changeme", "secret", "your-secret-key"}
        if value.lower() in insecure_defaults:
            raise ValueError(
                "JWT_SECRET_KEY must not use an insecure placeholder value."
            )
        return value


@lru_cache
def get_settings() -> Settings:
    """Construct and cache the application ``Settings`` singleton.

    Wrapped in ``lru_cache`` so environment parsing/validation occurs
    exactly once per process, and repeated calls return the same
    cached instance.

    Returns:
        Settings: The cached, validated application settings instance.
    """
    return Settings()


settings: Settings = get_settings()