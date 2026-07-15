"""Root API router for TrustLens AI backend.

Aggregates all versioned route modules into a single ``APIRouter``
instance that is mounted onto the FastAPI application in ``main.py``.
This module performs pure route registration only \u2014 no business
logic, validation, or persistence concerns belong here.

Note:
    The ``/api/v1`` version prefix is applied once, centrally, in
    ``main.py`` via ``settings.API_V1_PREFIX`` when this router is
    included into the FastAPI app (see
    ``application.include_router(api_router, prefix=settings.API_V1_PREFIX)``).
    This module therefore only defines *resource* prefixes
    (``/system``, ``/auth``, ``/user``) so that the final resolved
    paths become ``/api/v1/system``, ``/api/v1/auth``, ``/api/v1/user``,
    etc., without double-prefixing.
"""

from fastapi import APIRouter

from api.routes import auth, system, user

api_router = APIRouter()

api_router.include_router(
    system.router,
    prefix="/system",
    tags=["System"],
)

api_router.include_router(
    auth.router,
    prefix="/auth",
    tags=["Authentication"],
)

api_router.include_router(
    user.router,
    prefix="/user",
    tags=["User"],
)

# NOTE: Future AI-related route modules (e.g. routes/knowledge.py,
# routes/search.py, routes/insights.py) should be imported and
# registered here following the same pattern, keeping this module
# the single source of truth for API composition.