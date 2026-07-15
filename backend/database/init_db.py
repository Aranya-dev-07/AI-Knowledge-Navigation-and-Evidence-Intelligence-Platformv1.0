"""Database initialization for TrustLens AI backend.

Responsible for importing all ORM model modules (so their tables are
registered on ``Base.metadata``) and creating any tables that do not
yet exist. This module is invoked once during application startup
(see ``main.py``'s lifespan handler).

Note on Alembic:
    In production, schema changes should be applied via versioned
    Alembic migrations (``alembic upgrade head``), not this module.
    ``create_all`` here is additive and idempotent \u2014 it only creates
    tables that are missing and never alters or drops existing ones \u2014
    making it safe to run alongside Alembic-managed environments (e.g.
    for local development bootstrapping or first-run convenience). The
    same ``Base.metadata`` imported here is also what Alembic's
    ``env.py`` should target as its ``target_metadata`` for autogenerating
    migrations.
"""

import logging

from database.database import Base, engine

# Import every ORM model module so its table(s) are registered on
# Base.metadata before create_all() is invoked. Alembic's env.py
# should perform the same imports for autogeneration to see all models.
from models.user import User  # noqa: F401

logger = logging.getLogger(__name__)


async def init_db() -> None:
    """Initialize the database schema for the application.

    Creates any tables declared on ``Base.metadata`` that do not
    already exist in the target database. Safe to call on every
    application startup: existing tables are left untouched.

    Raises:
        Exception: Re-raises any exception encountered while creating
            tables, after logging it, so the application can fail
            fast on startup rather than run against an unusable
            database.

    Returns:
        None
    """
    try:
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
        logger.info("Database tables verified/created successfully.")
    except Exception:
        logger.exception("Database initialization failed while creating tables.")
        raise