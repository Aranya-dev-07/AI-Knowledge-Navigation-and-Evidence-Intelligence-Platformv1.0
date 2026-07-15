"""Authentication endpoints for TrustLens AI backend.

Exposes registration, login, and token-refresh endpoints. All
business logic (credential validation, password hashing, token
issuance) is delegated to ``services.auth_service.AuthService``. This
module is responsible only for request/response schema definitions,
routing, and translating service-layer exceptions into appropriate
HTTP responses.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, EmailStr, Field

from services.auth_service import (
    AuthService,
    EmailAlreadyRegisteredError,
    InvalidCredentialsError,
    InvalidRefreshTokenError,
    get_auth_service,
)

logger = logging.getLogger(__name__)

router = APIRouter()


class UserRegisterRequest(BaseModel):
    """Request payload for new user registration."""

    email: EmailStr = Field(..., description="Unique email address of the user.")
    password: str = Field(
        ..., min_length=8, max_length=128, description="Plaintext password (min 8 characters)."
    )
    full_name: str = Field(
        ..., min_length=1, max_length=255, description="Full display name of the user."
    )


class UserLoginRequest(BaseModel):
    """Request payload for user login."""

    email: EmailStr = Field(..., description="Registered email address of the user.")
    password: str = Field(..., min_length=1, description="Plaintext account password.")


class RefreshTokenRequest(BaseModel):
    """Request payload for exchanging a refresh token."""

    refresh_token: str = Field(..., description="Valid, previously issued refresh token.")


class UserPublicResponse(BaseModel):
    """Public-facing representation of a registered user."""

    id: int = Field(..., description="Unique identifier of the user.")
    email: EmailStr = Field(..., description="Email address of the user.")
    full_name: str = Field(..., description="Full display name of the user.")
    is_active: bool = Field(..., description="Whether the user account is active.")


class TokenResponse(BaseModel):
    """Response payload containing issued JWT tokens."""

    access_token: str = Field(..., description="Short-lived JWT access token.")
    refresh_token: str = Field(..., description="Long-lived JWT refresh token.")
    token_type: str = Field(default="bearer", description="Type of the issued token.")


@router.post(
    "/register",
    response_model=UserPublicResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new user",
)
async def register(
    payload: UserRegisterRequest,
    auth_service: AuthService = Depends(get_auth_service),
) -> UserPublicResponse:
    """Register a new user account.

    Args:
        payload: Validated registration request containing email,
            password, and full name.
        auth_service: Injected authentication service handling
            persistence and password hashing.

    Raises:
        HTTPException: 409 if the email is already registered.
        HTTPException: 500 if registration fails unexpectedly.

    Returns:
        UserPublicResponse: The newly created user's public profile.
    """
    try:
        user = await auth_service.register(
            email=payload.email,
            password=payload.password,
            full_name=payload.full_name,
        )
    except EmailAlreadyRegisteredError as exc:
        logger.info("Registration rejected: email already registered.")
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A user with this email is already registered.",
        ) from exc
    except Exception as exc:
        logger.exception("Unexpected error during user registration.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to complete registration at this time.",
        ) from exc

    return UserPublicResponse(
        id=user.id,
        email=user.email,
        full_name=user.full_name,
        is_active=user.is_active,
    )


@router.post(
    "/login",
    response_model=TokenResponse,
    status_code=status.HTTP_200_OK,
    summary="Authenticate a user and issue JWT tokens",
)
async def login(
    payload: UserLoginRequest,
    auth_service: AuthService = Depends(get_auth_service),
) -> TokenResponse:
    """Authenticate a user and issue an access/refresh token pair.

    Args:
        payload: Validated login request containing email and password.
        auth_service: Injected authentication service handling
            credential verification and token issuance.

    Raises:
        HTTPException: 401 if the credentials are invalid.
        HTTPException: 500 if login fails unexpectedly.

    Returns:
        TokenResponse: Newly issued access and refresh tokens.
    """
    try:
        token_pair = await auth_service.login(
            email=payload.email,
            password=payload.password,
        )
    except InvalidCredentialsError as exc:
        logger.info("Login rejected: invalid credentials.")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password.",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc
    except Exception as exc:
        logger.exception("Unexpected error during login.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to complete login at this time.",
        ) from exc

    return TokenResponse(
        access_token=token_pair.access_token,
        refresh_token=token_pair.refresh_token,
    )


@router.post(
    "/refresh-token",
    response_model=TokenResponse,
    status_code=status.HTTP_200_OK,
    summary="Exchange a refresh token for a new token pair",
)
async def refresh_token(
    payload: RefreshTokenRequest,
    auth_service: AuthService = Depends(get_auth_service),
) -> TokenResponse:
    """Exchange a valid refresh token for a new access/refresh token pair.

    Args:
        payload: Validated request containing the refresh token.
        auth_service: Injected authentication service handling token
            verification and reissuance.

    Raises:
        HTTPException: 401 if the refresh token is invalid or expired.
        HTTPException: 500 if refresh fails unexpectedly.

    Returns:
        TokenResponse: Newly issued access and refresh tokens.
    """
    try:
        token_pair = await auth_service.refresh(refresh_token=payload.refresh_token)
    except InvalidRefreshTokenError as exc:
        logger.info("Token refresh rejected: invalid or expired refresh token.")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token.",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc
    except Exception as exc:
        logger.exception("Unexpected error during token refresh.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to refresh token at this time.",
        ) from exc

    return TokenResponse(
        access_token=token_pair.access_token,
        refresh_token=token_pair.refresh_token,
    )