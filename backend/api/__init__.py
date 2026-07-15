"""API package for TrustLens AI backend.

Re-exports the aggregated ``api_router`` so it can be imported as
``from api import api_router`` in addition to
``from api.router import api_router``.
"""

from api.router import api_router

__all__ = ["api_router"]