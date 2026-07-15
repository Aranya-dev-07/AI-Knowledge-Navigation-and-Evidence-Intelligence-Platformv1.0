"""Authentication business logic for TrustLens AI backend.

Encapsulates all rules for registering users, verifying credentials,
and issuing/rotating JWT token pairs. Route handlers in
``api.routes.auth`` depend on ``AuthService`` (via ``get_auth_service``)
and never touch the database or JWT machinery directly.
"""

import logging
from dataclasses import dataclass

from fastapi import Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.security import (
    TokenError,
    TokenType,
    create_access_token,
    create_refresh_token,
    decode_token,
    hash_password,
    verify_password,
)
from database.database import get_db
from models.user import User

logger = logging.getLogger(__name__)

_MAX_USERNAME_COLLISION_ATTEMPTS = 100


class EmailAlreadyRegisteredError(Exception):
    """Raised when attempting to register an email that already exists."""


class InvalidCredentialsError(Exception):
    """Raised when login credentials are missing, incorrect, or the account is inactive."""


class InvalidRefreshTokenError(Exception):
    """Raised when a refresh token is invalid, expired, malformed, or stale."""


@dataclass(frozen=True, slots=True)
class TokenPair:
    """An issued pair of JWT access and refresh tokens.

    Attributes:
        access_token: Short-lived JWT used to authenticate API requests.
        refresh_token: Long-lived JWT used to obtain new access tokens.
    """

    access_token: str
    refresh_token: str


class AuthService:
    """Business logic for user registration, login, and token refresh."""

    def __init__(self, db: AsyncSession) -> None:
        """Initialize the service with a request-scoped database session.

        Args:
            db: The async SQLAlchemy session used for all persistence
                operations performed by this service instance.
        """
        self._db = db

    async def _generate_unique_username(self, email: str) -> str:
        """Derive a unique username from the local part of an email address.

        The registration endpoint only collects an email, password,
        and full name, but ``User.username`` is a unique, required
        column. This derives a base username from the email's local
        part (before the ``@``) and appends a numeric suffix on
        collision until a free username is found.

        Args:
            email: The email address to derive a username from.

        Raises:
            RuntimeError: If no unique username could be generated
                after a bounded number of attempts, indicating an
                unexpectedly large number of collisions.

        Returns:
            str: A username guaranteed not to already exist at the
                time of the check.
        """
        base_username = email.split("@", 1)[0].strip().lower() or "user"

        candidate = base_username
        for attempt in range(_MAX_USERNAME_COLLISION_ATTEMPTS):
            result = await self._db.execute(select(User.id).where(User.username == candidate))
            if result.scalar_one_or_none() is None:
                return candidate
            candidate = f"{base_username}{attempt + 1}"

        raise RuntimeError(
            f"Could not generate a unique username for email '{email}' "
            f"after {_MAX_USERNAME_COLLISION_ATTEMPTS} attempts."
        )

    async def register(self, email: str, password: str, full_name: str) -> User:
        """Register a new user account.

        Args:
            email: The new user's unique email address.
            password: The new user's plaintext password, to be hashed
                before storage.
            full_name: The new user's full display name.

        Raises:
            EmailAlreadyRegisteredError: If a user with ``email``
                already exists.

        Returns:
            User: The newly created, persisted user instance.
        """
        existing = await self._db.execute(select(User.id).where(User.email == email))
        if existing.scalar_one_or_none() is not None:
            raise EmailAlreadyRegisteredError(f"Email already registered: {email}")

        username = await self._generate_unique_username(email)

        user = User(
            username=username,
            email=email,
            hashed_password=hash_password(password),
            full_name=full_name,
            is_active=True,
        )
        self._db.add(user)
        await self._db.flush()
        await self._db.refresh(user)

        logger.info("Registered new user id=%s email=%s.", user.id, user.email)
        return user

    async def login(self, email: str, password: str) -> TokenPair:
        """Authenticate a user by email and password and issue tokens.

        Args:
            email: The email address supplied by the caller.
            password: The plaintext password supplied by the caller.

        Raises:
            InvalidCredentialsError: If no active user matches
                ``email``, or the supplied password does not match
                the stored hash.

        Returns:
            TokenPair: A newly issued access and refresh token pair.
        """
        result = await self._db.execute(select(User).where(User.email == email))
        user = result.scalar_one_or_none()

        if user is None or not verify_password(password, user.hashed_password):
            logger.info("Login failed for email=%s: invalid credentials.", email)
            raise InvalidCredentialsError("Invalid email or password.")

        if not user.is_active:
            logger.info("Login failed for email=%s: account inactive.", email)
            raise InvalidCredentialsError("Invalid email or password.")

        logger.info("User id=%s logged in successfully.", user.id)
        return TokenPair(
            access_token=create_access_token(subject=user.id),
            refresh_token=create_refresh_token(subject=user.id),
        )

    async def refresh(self, refresh_token: str) -> TokenPair:
        """Exchange a valid refresh token for a new access/refresh token pair.

        Implements refresh token rotation: a brand new refresh token
        is issued alongside the new access token, and the presented
        refresh token is not reusable beyond this call from the
        caller's perspective (rotation is enforced client-side by
        discarding the old token; server-side revocation/denylisting
        is out of scope for this service).

        Args:
            refresh_token: The previously issued refresh token to
                validate and exchange.

        Raises:
            InvalidRefreshTokenError: If the token is malformed,
                expired, not of type "refresh", or no longer
                corresponds to an existing, active user.

        Returns:
            TokenPair: A newly issued access and refresh token pair.
        """
        try:
            payload = decode_token(refresh_token, expected_type=TokenType.REFRESH)
            user_id = int(payload["sub"])
        except (TokenError, ValueError, KeyError) as exc:
            logger.info("Refresh token rejected: %s", exc)
            raise InvalidRefreshTokenError("Invalid or expired refresh token.") from exc

        user = await self._db.get(User, user_id)
        if user is None or not user.is_active:
            logger.info("Refresh token rejected: user id=%s not found or inactive.", user_id)
            raise InvalidRefreshTokenError("Invalid or expired refresh token.")

        logger.info("Issued rotated token pair for user id=%s.", user.id)
        return TokenPair(
            access_token=create_access_token(subject=user.id),
            refresh_token=create_refresh_token(subject=user.id),
        )


def get_auth_service(db: AsyncSession = Depends(get_db)) -> AuthService:
    """FastAPI dependency that constructs a request-scoped ``AuthService``.

    Args:
        db: Injected async database session for this request.

    Returns:
        AuthService: A service instance bound to the request's session.
    """
    return AuthService(db=db)