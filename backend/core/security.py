"""Security utilities for TrustLens AI backend.

Provides password hashing/verification (via Passlib + bcrypt) and JWT
access/refresh token creation and validation (via python-jose). Also
exposes the ``get_current_user`` FastAPI dependency used by protected
routes to resolve the authenticated user from a bearer token.

All cryptographic secrets and token lifetimes are read exclusively
from ``core.config.settings`` \u2014 no secrets are hardcoded here.
"""

import logging
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Any, Final

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import ExpiredSignatureError, JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.ext.asyncio import AsyncSession

from core.config import settings
from database.database import get_db
from models.user import User

logger = logging.getLogger(__name__)

_pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

_oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl=f"{settings.API_V1_PREFIX}/auth/login",
    auto_error=True,
)

_SUBJECT_CLAIM: Final[str] = "sub"
_TOKEN_TYPE_CLAIM: Final[str] = "type"
_EXPIRY_CLAIM: Final[str] = "exp"
_ISSUED_AT_CLAIM: Final[str] = "iat"


class TokenType(StrEnum):
    """Enumerates the distinct categories of JWT issued by this service."""

    ACCESS = "access"
    REFRESH = "refresh"


class TokenError(Exception):
    """Raised when a JWT is missing, malformed, expired, or of the wrong type.

    Callers in the service layer are expected to catch this exception
    and translate it into a domain-specific error (e.g. an invalid
    refresh token error) as appropriate for their context.
    """


def hash_password(plain_password: str) -> str:
    """Hash a plaintext password using bcrypt.

    Args:
        plain_password: The user-supplied plaintext password.

    Returns:
        str: The salted bcrypt hash of the password, safe for storage.
    """
    return _pwd_context.hash(plain_password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a plaintext password against a stored bcrypt hash.

    Args:
        plain_password: The plaintext password supplied by the caller.
        hashed_password: The previously stored bcrypt password hash.

    Returns:
        bool: ``True`` if the password matches the hash, ``False`` otherwise.
    """
    try:
        return _pwd_context.verify(plain_password, hashed_password)
    except (ValueError, TypeError):
        logger.warning("Password verification failed due to malformed hash.")
        return False


def _create_token(subject: str | int, token_type: TokenType, expires_delta: timedelta) -> str:
    """Build and sign a JWT for the given subject and token type.

    Args:
        subject: The unique identifier (typically user ID) the token
            represents.
        token_type: Whether this is an access or refresh token.
        expires_delta: How far in the future the token should expire.

    Returns:
        str: The encoded, signed JWT string.
    """
    now = datetime.now(UTC)
    payload: dict[str, Any] = {
        _SUBJECT_CLAIM: str(subject),
        _TOKEN_TYPE_CLAIM: token_type.value,
        _ISSUED_AT_CLAIM: now,
        _EXPIRY_CLAIM: now + expires_delta,
    }
    return jwt.encode(payload, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)


def create_access_token(subject: str | int) -> str:
    """Create a short-lived JWT access token for the given subject.

    Args:
        subject: The unique identifier (typically user ID) to encode.

    Returns:
        str: The encoded access token.
    """
    return _create_token(
        subject=subject,
        token_type=TokenType.ACCESS,
        expires_delta=timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES),
    )


def create_refresh_token(subject: str | int) -> str:
    """Create a long-lived JWT refresh token for the given subject.

    Args:
        subject: The unique identifier (typically user ID) to encode.

    Returns:
        str: The encoded refresh token.
    """
    return _create_token(
        subject=subject,
        token_type=TokenType.REFRESH,
        expires_delta=timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS),
    )


def decode_token(token: str, expected_type: TokenType) -> dict[str, Any]:
    """Decode and validate a JWT, enforcing its expected token type.

    Args:
        token: The encoded JWT string to decode.
        expected_type: The ``TokenType`` the token must declare via its
            ``type`` claim (e.g. reject a refresh token presented where
            an access token is required, and vice versa).

    Raises:
        TokenError: If the token is expired, malformed, has an invalid
            signature, or does not match ``expected_type``.

    Returns:
        dict[str, Any]: The decoded JWT payload/claims.
    """
    try:
        payload: dict[str, Any] = jwt.decode(
            token,
            settings.JWT_SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM],
        )
    except ExpiredSignatureError as exc:
        raise TokenError("Token has expired.") from exc
    except JWTError as exc:
        raise TokenError("Token is invalid or malformed.") from exc

    if payload.get(_TOKEN_TYPE_CLAIM) != expected_type.value:
        raise TokenError(
            f"Expected a '{expected_type.value}' token, got "
            f"'{payload.get(_TOKEN_TYPE_CLAIM)}'."
        )

    if _SUBJECT_CLAIM not in payload:
        raise TokenError("Token payload is missing the subject claim.")

    return payload


async def get_current_user(
    token: str = Depends(_oauth2_scheme),
    db: AsyncSession = Depends(get_db),
) -> User:
    """Resolve the authenticated ``User`` from a bearer access token.

    Intended for use as a FastAPI dependency on protected routes. On
    any validation failure, raises an HTTP 401 response so route
    handlers never need to handle token errors themselves.

    Args:
        token: The bearer JWT extracted from the ``Authorization`` header.
        db: Injected async database session used to load the user.

    Raises:
        HTTPException: 401 if the token is invalid, expired, of the
            wrong type, or does not correspond to an existing, active
            user.

    Returns:
        User: The authenticated, active user's ORM instance.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials.",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = decode_token(token, expected_type=TokenType.ACCESS)
        user_id = int(payload[_SUBJECT_CLAIM])
    except (TokenError, ValueError, KeyError) as exc:
        logger.info("Access token validation failed: %s", exc)
        raise credentials_exception from exc

    user = await db.get(User, user_id)
    if user is None:
        logger.info("Access token references non-existent user id=%s.", user_id)
        raise credentials_exception

    if not user.is_active:
        logger.info("Access token references inactive user id=%s.", user_id)
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This user account is inactive.",
        )

    return user