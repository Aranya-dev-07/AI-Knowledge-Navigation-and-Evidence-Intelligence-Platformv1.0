"""User business logic for TrustLens AI backend.

Encapsulates all rules for retrieving, updating, and deleting user
accounts. Route handlers in ``api.routes.user`` depend on
``UserService`` (via ``get_user_service``) and never touch the
database directly. Current-user *authentication* (JWT resolution) is
handled separately by ``core.security.get_current_user``; this
service is concerned only with what happens to a user record once
identified.
"""

import logging

from fastapi import Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from database.database import get_db
from models.user import User

logger = logging.getLogger(__name__)


class UserNotFoundError(Exception):
    """Raised when a requested user does not exist."""


class EmailAlreadyInUseError(Exception):
    """Raised when updating a user's email to one already used by another account."""


class UserService:
    """Business logic for retrieving, updating, and deleting user accounts."""

    def __init__(self, db: AsyncSession) -> None:
        """Initialize the service with a request-scoped database session.

        Args:
            db: The async SQLAlchemy session used for all persistence
                operations performed by this service instance.
        """
        self._db = db

    async def get_by_id(self, user_id: int) -> User:
        """Retrieve a user by their unique identifier.

        Also serves as the "get current user" lookup when a route
        already holds an authenticated user's id (e.g. re-fetching a
        fresh copy of the current user's record).

        Args:
            user_id: The unique identifier of the user to retrieve.

        Raises:
            UserNotFoundError: If no user with ``user_id`` exists.

        Returns:
            User: The matching user instance.
        """
        user = await self._db.get(User, user_id)
        if user is None:
            raise UserNotFoundError(f"No user found with id={user_id}.")
        return user

    async def update_profile(
        self,
        user: User,
        full_name: str | None = None,
        email: str | None = None,
    ) -> User:
        """Update mutable fields on an existing user's profile.

        Only fields explicitly provided (non-``None``) are changed;
        omitted fields are left untouched.

        Args:
            user: The user instance to update. Must already be
                attached to this service's session (e.g. as resolved
                by ``core.security.get_current_user``).
            full_name: New display name, or ``None`` to leave unchanged.
            email: New email address, or ``None`` to leave unchanged.

        Raises:
            EmailAlreadyInUseError: If ``email`` is provided and
                already belongs to a different user.

        Returns:
            User: The updated, persisted user instance.
        """
        if email is not None and email != user.email:
            result = await self._db.execute(
                select(User.id).where(User.email == email, User.id != user.id)
            )
            if result.scalar_one_or_none() is not None:
                raise EmailAlreadyInUseError(f"Email already in use: {email}")
            user.email = email

        if full_name is not None:
            user.full_name = full_name

        await self._db.flush()
        await self._db.refresh(user)

        logger.info("Updated profile for user id=%s.", user.id)
        return user

    async def delete_user(self, user: User) -> None:
        """Permanently delete a user account.

        Args:
            user: The user instance to delete. Must already be
                attached to this service's session.

        Returns:
            None
        """
        user_id = user.id
        await self._db.delete(user)
        await self._db.flush()
        logger.info("Deleted user id=%s.", user_id)


def get_user_service(db: AsyncSession = Depends(get_db)) -> UserService:
    """FastAPI dependency that constructs a request-scoped ``UserService``.

    Args:
        db: Injected async database session for this request.

    Returns:
        UserService: A service instance bound to the request's session.
    """
    return UserService(db=db)