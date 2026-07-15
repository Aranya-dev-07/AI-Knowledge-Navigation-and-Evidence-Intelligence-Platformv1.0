"""Routes package for TrustLens AI backend.

Re-exports the individual route modules so ``api.router`` can import
them as ``from api.routes import auth, system, user``. Route
registration itself happens exclusively in ``api.router``; this
module performs no endpoint definitions or registration logic.
"""

from api.routes import auth, system, user

__all__ = ["auth", "system", "user"]