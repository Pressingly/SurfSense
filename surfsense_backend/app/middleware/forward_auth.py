"""
ForwardAuth middleware.

Trusts ``X-Auth-Request-Email`` / ``X-Auth-Request-User`` headers injected by
a ForwardAuth reverse-proxy (e.g. oauth2-proxy, Authelia, Traefik ForwardAuth).

When those headers are present the named account is looked up by email.  If no
matching record exists a new user is provisioned automatically (random secure
password, pre-verified, active) and the normal post-registration hooks run so
the account gets a default search-space, roles, and prompts.

The resolved ``User`` is stored on ``request.state.forward_auth_user`` so the
``current_active_user`` dependency can return it without requiring a Bearer
token.

Security note
-------------
Only enable this middleware (``FORWARD_AUTH_ENABLED=true``) when the backend
is sitting behind a trusted reverse proxy that enforces authentication.
Exposing the backend directly to the internet with this enabled lets anyone
impersonate any user by sending the headers themselves.
"""

import logging
import secrets

from fastapi_users.db import SQLAlchemyUserDatabase
from fastapi_users.password import PasswordHelper
from sqlalchemy import select
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from app.db import User, shielded_async_session

logger = logging.getLogger("surfsense.forward_auth")

_password_helper = PasswordHelper()


class ForwardAuthMiddleware(BaseHTTPMiddleware):
    """Resolve the requesting user from ForwardAuth proxy headers."""

    async def dispatch(self, request: Request, call_next) -> Response:
        email = request.headers.get("x-auth-request-email")
        display_name = request.headers.get("x-auth-request-user")

        if not email:
            return await call_next(request)

        try:
            async with shielded_async_session() as session:
                result = await session.execute(
                    select(User).where(User.email == email.lower())
                )
                user = result.scalars().first()

                if user is None:
                    user = await _provision_user(session, email.lower(), display_name)
                    logger.info(
                        "ForwardAuth: provisioned new user id=%s email=%s",
                        user.id,
                        email,
                    )
                else:
                    logger.debug(
                        "ForwardAuth: authenticated existing user id=%s email=%s",
                        user.id,
                        email,
                    )

                request.state.forward_auth_user = user
        except Exception:
            logger.exception(
                "ForwardAuth: failed to resolve user for email=%s; "
                "falling through to normal auth",
                email,
            )

        return await call_next(request)


async def _provision_user(
    session,
    email: str,
    display_name: str | None,
) -> User:
    """Create a new User row, commit it, then run post-registration hooks."""
    # Import locally to avoid a circular dependency at module load time.
    from app.users import UserManager  # noqa: PLC0415

    hashed_password = _password_helper.hash(secrets.token_urlsafe(32))

    user = User(
        email=email,
        hashed_password=hashed_password,
        is_active=True,
        is_superuser=False,
        is_verified=True,
        display_name=display_name,
    )
    session.add(user)
    await session.commit()
    await session.refresh(user)

    # Reuse the same session so on_after_register's own session doesn't race.
    user_db = SQLAlchemyUserDatabase(session, User)
    user_manager = UserManager(user_db)
    await user_manager.on_after_register(user)

    return user
