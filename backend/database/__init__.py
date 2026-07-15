"""Database package for TrustLens AI backend.

Re-exports the core persistence primitives \u2014 the async engine, session
factory, declarative base, and ``get_db`` dependency \u2014 from
``database.database`` so they can be imported directly as
``from database import Base, get_db`` in addition to their
fully-qualified submodule path. Performs no schema initialization;
that responsibility belongs solely to ``database.init_db``.
"""

from database.database import AsyncSessionLocal, Base, engine, get_db

__all__ = ["engine", "AsyncSessionLocal", "Base", "get_db"]