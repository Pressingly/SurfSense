"""Authentication routes for refresh token management."""

import logging

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import RedirectResponse
from sqlalchemy import select

from app.config import config
from app.db import User, async_session_maker
from app.schemas.auth import (
    LogoutAllResponse,
    LogoutRequest,
    LogoutResponse,
    RefreshTokenRequest,
    RefreshTokenResponse,
)
from app.users import current_active_user, get_jwt_strategy
from app.utils.refresh_tokens import (
    create_refresh_token,
    revoke_all_user_tokens,
    revoke_refresh_token,
    rotate_refresh_token,
    validate_refresh_token,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth/jwt", tags=["auth"])


@router.post("/refresh", response_model=RefreshTokenResponse)
async def refresh_access_token(request: RefreshTokenRequest):
    """
    Exchange a valid refresh token for a new access token and refresh token.
    Implements token rotation for security.
    """
    token_record = await validate_refresh_token(request.refresh_token)

    if not token_record:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token",
        )

    # Get user from token record
    async with async_session_maker() as session:
        result = await session.execute(
            select(User).where(User.id == token_record.user_id)
        )
        user = result.scalars().first()

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
        )

    # Generate new access token
    strategy = get_jwt_strategy()
    access_token = await strategy.write_token(user)

    # Rotate refresh token
    new_refresh_token = await rotate_refresh_token(token_record)

    logger.info(f"Refreshed token for user {user.id}")

    return RefreshTokenResponse(
        access_token=access_token,
        refresh_token=new_refresh_token,
    )


@router.post("/revoke", response_model=LogoutResponse)
async def revoke_token(request: LogoutRequest):
    """
    Logout current device by revoking the provided refresh token.
    Does not require authentication - just the refresh token.
    """
    revoked = await revoke_refresh_token(request.refresh_token)
    if revoked:
        logger.info("User logged out from current device - token revoked")
    else:
        logger.warning("Logout called but no matching token found to revoke")
    return LogoutResponse()


@router.post("/logout-all", response_model=LogoutAllResponse)
async def logout_all_devices(user: User = Depends(current_active_user)):
    """
    Logout from all devices by revoking all refresh tokens for the user.
    Requires valid access token.
    """
    await revoke_all_user_tokens(user.id)
    logger.info(f"User {user.id} logged out from all devices")
    return LogoutAllResponse()


@router.get("/forward-auth/token", tags=["auth"])
async def forward_auth_token(request: Request):
    """
    Exchange a ForwardAuth proxy session for a SurfSense JWT.

    This endpoint is intended to be the redirect target for the `/login` path
    when running behind a ForwardAuth reverse proxy (e.g. Traefik + oauth2-proxy).
    The proxy authenticates the user and injects X-Auth-Request-Email /
    X-Auth-Request-User headers; ForwardAuthMiddleware resolves the user from
    those headers and stores it on request.state.forward_auth_user.

    On success the browser is redirected to the frontend /auth/callback page
    with fresh JWT tokens in the query string — identical to the Google OAuth flow.
    """
    if not config.FORWARD_AUTH_ENABLED:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="ForwardAuth is not enabled",
        )

    user = getattr(request.state, "forward_auth_user", None)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="No ForwardAuth session found",
        )

    strategy = get_jwt_strategy()
    access_token = await strategy.write_token(user)
    refresh_token = await create_refresh_token(user.id)

    logger.info("ForwardAuth: issued JWT for user id=%s email=%s", user.id, user.email)

    redirect_url = (
        f"{config.NEXT_FRONTEND_URL}/auth/callback"
        f"?token={access_token}&refresh_token={refresh_token}"
    )
    return RedirectResponse(redirect_url, status_code=302)
