"""User endpoints for TrustLens AI backend.

Exposes authenticated, JWT-protected endpoints for retrieving and
managing user profile data. All business logic (lookup, update,
deletion) is delegated to ``services.user_service.UserService``.
Authentication and current-user resolution are delegated to
``core.security``. This module is responsible only for routing,
request/response schemas, and exception-to-HTTP translation.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, EmailStr, Field

from core.security import get_current_user
from models.user import User
from services.user_service import (
    UserNotFoundError,
    UserService,
    get_user_service,
)

logger = logging.getLogger(__name__)

router = APIRouter()


class UserProfileResponse(BaseModel):
    """Public-facing representation of a user profile."""

    id: int = Field(..., description="Unique identifier of the user.")
    email: EmailStr = Field(..., description="Email address of the user.")
    full_name: str = Field(..., description="Full display name of the user.")
    is_active: bool = Field(..., description="Whether the user account is active.")


class UserUpdateRequest(BaseModel):
    """Request payload for updating the authenticated user's profile.

    All fields are optional; only provided fields are updated.
    """

    full_name: str | None = Field(
        default=None, min_length=1, max_length=255, description="Updated display name."
    )
    email: EmailStr | None = Field(
        default=None, description="Updated email address."
    )


def _to_profile_response(user: User) -> UserProfileResponse:
    """Convert a ``User`` ORM instance into its public response schema.

    Args:
        user: The SQLAlchemy ``User`` model instance to serialize.

    Returns:
        UserProfileResponse: The serialized, public-safe representation.
    """
    return UserProfileResponse(
        id=user.id,
        email=user.email,
        full_name=user.full_name,
        is_active=user.is_active,
    )


@router.get(
    "/me",
    response_model=UserProfileResponse,
    status_code=status.HTTP_200_OK,
    summary="Get the authenticated user's profile",
)
async def read_current_user(
    current_user: User = Depends(get_current_user),
) -> UserProfileResponse:
    """Return the profile of the currently authenticated user.

    Args:
        current_user: The user resolved from the request's JWT access
            token, injected by ``core.security.get_current_user``.

    Returns:
        UserProfileResponse: The authenticated user's public profile.
    """
    return _to_profile_response(current_user)


@router.get(
    "/users/{id}",
    response_model=UserProfileResponse,
    status_code=status.HTTP_200_OK,
    summary="Get a user's profile by ID",
)
async def read_user_by_id(
    id: int,
    current_user: User = Depends(get_current_user),
    user_service: UserService = Depends(get_user_service),
) -> UserProfileResponse:
    """Return the public profile of a user identified by their ID.

    Requires a valid JWT access token; any authenticated user may look
    up any other user's public profile.

    Args:
        id: The unique identifier of the user to retrieve.
        current_user: The authenticated caller, injected via JWT
            verification. Present to enforce authentication even
            though the caller's identity is not otherwise used.
        user_service: Injected user service handling persistence
            lookups.

    Raises:
        HTTPException: 404 if no user with the given ID exists.
        HTTPException: 500 if lookup fails unexpectedly.

    Returns:
        UserProfileResponse: The requested user's public profile.
    """
    del current_user  # Authentication is enforced; identity unused here.
    try:
        user = await user_service.get_by_id(user_id=id)
    except UserNotFoundError as exc:
        logger.info("User lookup failed: no user with id=%s.", id)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found.",
        ) from exc
    except Exception as exc:
        logger.exception("Unexpected error while retrieving user id=%s.", id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to retrieve user at this time.",
        ) from exc

    return _to_profile_response(user)


@router.put(
    "/me",
    response_model=UserProfileResponse,
    status_code=status.HTTP_200_OK,
    summary="Update the authenticated user's profile",
)
async def update_current_user(
    payload: UserUpdateRequest,
    current_user: User = Depends(get_current_user),
    user_service: UserService = Depends(get_user_service),
) -> UserProfileResponse:
    """Update mutable fields on the authenticated user's profile.

    Args:
        payload: Validated request containing the fields to update.
            Unset fields are left unchanged.
        current_user: The user resolved from the request's JWT access
            token.
        user_service: Injected user service handling persistence
            updates.

    Raises:
        HTTPException: 409 if the requested email is already in use.
        HTTPException: 500 if the update fails unexpectedly.

    Returns:
        UserProfileResponse: The updated user profile.
    """
    try:
        updated_user = await user_service.update_profile(
            user=current_user,
            full_name=payload.full_name,
            email=payload.email,
        )
    except UserNotFoundError as exc:
        logger.warning("Update failed: authenticated user id=%s not found.", current_user.id)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found.",
        ) from exc
    except Exception as exc:
        logger.exception("Unexpected error while updating user id=%s.", current_user.id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to update profile at this time.",
        ) from exc

    return _to_profile_response(updated_user)


@router.delete(
    "/me",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete the authenticated user's account",
)
async def delete_current_user(
    current_user: User = Depends(get_current_user),
    user_service: UserService = Depends(get_user_service),
) -> None:
    """Permanently delete the authenticated user's account.

    Args:
        current_user: The user resolved from the request's JWT access
            token.
        user_service: Injected user service handling persistence
            deletion.

    Raises:
        HTTPException: 500 if deletion fails unexpectedly.
    """
    try:
        await user_service.delete_user(user=current_user)
    except Exception as exc:
        logger.exception("Unexpected error while deleting user id=%s.", current_user.id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to delete account at this time.",
        ) from exc

    logger.info("User id=%s deleted their account.", current_user.id)