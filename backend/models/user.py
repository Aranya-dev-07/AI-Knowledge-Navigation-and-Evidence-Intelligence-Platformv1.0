"""User ORM model for TrustLens AI backend.

Defines the ``User`` SQLAlchemy 2.0 declarative model backing the
``users`` table. Consumed by ``core.security`` (current-user
resolution), ``services.auth_service`` (registration, login, token
issuance), and ``services.user_service`` (profile retrieval, update,
deletion).
"""

from datetime import datetime

from sqlalchemy import Boolean, DateTime, String, func
from sqlalchemy.orm import Mapped, mapped_column

from database.database import Base


class User(Base):
    """Represents an application user account.

    Attributes:
        id: Primary key, auto-incrementing user identifier.
        username: Unique, publicly visible handle for the user.
        email: Unique email address used for login and communication.
        hashed_password: Bcrypt hash of the user's password. Never
            exposed outside the persistence and auth layers.
        full_name: The user's full display name.
        is_active: Whether the account is active and permitted to
            authenticate. Deactivated accounts are rejected at login.
        created_at: Timestamp when the record was created, set by the
            database on insert.
        updated_at: Timestamp when the record was last modified, set
            by the database on insert and refreshed on every update.
    """

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(
        String(50), unique=True, index=True, nullable=False
    )
    email: Mapped[str] = mapped_column(
        String(255), unique=True, index=True, nullable=False
    )
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    full_name: Mapped[str] = mapped_column(String(255), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    def __repr__(self) -> str:
        """Return a debug-friendly representation of the user.

        Returns:
            str: A string identifying the user by id, username, and
                email, omitting sensitive fields such as the password
                hash.
        """
        return (
            f"User(id={self.id!r}, username={self.username!r}, "
            f"email={self.email!r}, is_active={self.is_active!r})"
        )