"""Core package for TrustLens AI backend.

Exposes the most commonly used core components \u2014 application
``settings``, logging setup, and security/JWT utilities \u2014 so they can
be imported directly from ``core`` (e.g. ``from core import settings``)
in addition to their fully-qualified submodule paths.

Security utilities are exposed lazily (via module-level
``__getattr__``, PEP 562) rather than imported eagerly at package load
time. ``core.security`` depends on ``database.database`` and
``models.user``, which in turn import ``core.config``; eagerly
importing ``core.security`` here would risk a circular import if some
other module imports ``database.database`` (directly or transitively)
before ``core`` has finished initializing. Lazy resolution sidesteps
that ordering hazard entirely while still keeping the ``from core
import get_current_user`` style import ergonomic.
"""

from typing import Any

from core.config import Settings, settings
from core.logging import get_logger, setup_logging

_SECURITY_EXPORTS = frozenset(
    {
        "hash_password",
        "verify_password",
        "create_access_token",
        "create_refresh_token",
        "decode_token",
        "get_current_user",
        "TokenType",
        "TokenError",
    }
)

__all__ = [
    "Settings",
    "settings",
    "setup_logging",
    "get_logger",
    *sorted(_SECURITY_EXPORTS),
]


def __getattr__(name: str) -> Any:
    """Lazily resolve security-related attributes on first access.

    Implements PEP 562 module-level attribute lookup so that
    ``core.security`` (and its dependency chain through
    ``database.database`` and ``models.user``) is only imported the
    first time a security attribute is actually requested, rather
    than at ``core`` package import time.

    Args:
        name: The attribute name being accessed on the ``core`` package.

    Raises:
        AttributeError: If ``name`` is not a recognized core export.

    Returns:
        Any: The resolved attribute from ``core.security``.
    """
    if name in _SECURITY_EXPORTS:
        from core import security

        return getattr(security, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")