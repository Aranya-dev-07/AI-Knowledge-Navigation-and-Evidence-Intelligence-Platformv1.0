"""Services package for TrustLens AI backend.

Re-exports the primary business-logic service classes so they can be
imported directly as ``from services import AuthService, UserService``
in addition to their fully-qualified submodule paths. Contains no
business logic itself \u2014 purely a composition/export point.
"""

from services.auth_service import AuthService
from services.user_service import UserService

__all__ = ["AuthService", "UserService"]