import logging
import secrets
import unicodedata

import jwt
from fastapi_users.db import SQLAlchemyUserDatabase
from fastapi_users.password import PasswordHelper  # singleton below
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from app.config import config
from app.db import User, async_session_maker

logger = logging.getLogger(__name__)
_password_helper = PasswordHelper()

_DEFAULT_BYPASS_PATHS = ["/health"]


def _normalise_email(email: str) -> str:
    # NFKC normalisation collapses Unicode lookalikes before lowercasing,
    # preventing homoglyph spoofing (e.g. fullwidth latin chars).
    return unicodedata.normalize("NFKC", email).strip().lower()


def _coerce_bypass_paths(setting) -> list[str]:
    if not setting:
        return list(_DEFAULT_BYPASS_PATHS)
    if isinstance(setting, str):
        return [p.strip() for p in setting.split(",") if p.strip()]
    return list(setting)


def _check_corporate_id(request) -> bool:
    """Verify that the caller's access token belongs to this deployment's tenant.

    When ``SMB_CORPORATE_ID`` is configured, only corporate tokens whose
    ``custom:corporate_id`` claim matches the expected value are allowed.
    Individual (non-corporate) tokens are rejected.  When the setting is
    empty the check is skipped entirely for backward compatibility.
    """
    expected = getattr(config, "SMB_CORPORATE_ID", "")
    if not expected:
        return True
    access_token = request.headers.get("x-auth-request-access-token")
    if not access_token:
        return False
    try:
        claims = jwt.decode(access_token, options={"verify_signature": False})
    except Exception:
        return False
    if claims.get("custom:is_corporate") != "true":
        return False
    return claims.get("custom:corporate_id") == expected


def _is_bypass_path(path: str, bypass_paths: list[str]) -> bool:
    # Match exact path OR a true subpath (e.g. /health/ready) but NOT a path that
    # merely starts with the same characters (e.g. /healthz must NOT bypass /health).
    return any(path == p or path.startswith(p.rstrip("/") + "/") for p in bypass_paths)


class ProxyAuthMiddleware(BaseHTTPMiddleware):
    """
    Starlette middleware for mPass proxy authentication.

    oauth2-proxy sets X-Auth-Request-Email on every request that has passed
    OIDC validation. This middleware reads that header, finds or creates the
    corresponding SurfSense user, and injects them into request.state.proxy_user
    so the current_active_user dependency sees a fully authenticated user
    without requiring a JWT token.

    Security / trust model
    ----------------------
    This middleware trusts X-Auth-Request-Email unconditionally. That is safe
    because:
      1. Traefik ForwardAuth overwrites X-Auth-Request-* headers on every
         request, so they cannot be spoofed by a browser or external client.
      2. In production the app container does not expose its port externally —
         only Traefik is public-facing, so there is no direct path to the app
         that bypasses header rewriting.

    A shared-secret header (set by oauth2-proxy, forwarded via Traefik
    authResponseHeaders, checked here) would add defense-in-depth against a
    misconfigured ingress but is not required given the network topology above.
    Add it if the threat model ever changes (e.g. the app port becomes reachable
    inside a zero-trust network where internal callers could forge headers).
    """

    def __init__(self, app):
        super().__init__(app)
        self.bypass_paths = _coerce_bypass_paths(
            getattr(config, "MPASS_BYPASS_PATHS", None)
        )

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        # Already injected on this request cycle (idempotent)
        if getattr(request.state, "proxy_user", None) is not None:
            return await call_next(request)

        if _is_bypass_path(request.url.path, self.bypass_paths):
            return await call_next(request)

        raw_email = (request.headers.get("x-auth-request-email") or "").strip()
        if raw_email and "@" not in raw_email:
            # Header holds a bare username (user_id_claim=cognito:username).
            domain = getattr(config, "DEFAULT_EMAIL_DOMAIN", "askii.ai")
            raw_email = f"{raw_email}@{domain}"
        if not raw_email:
            raw_username = (request.headers.get("x-auth-request-user") or "").strip()
            domain = getattr(config, "DEFAULT_EMAIL_DOMAIN", "askii.ai")
            if raw_username:
                raw_email = f"{raw_username}@{domain}"
        if not raw_email:
            logger.debug(
                "ProxyAuth: x-auth-request-email missing on %s", request.url.path
            )
            return await call_next(request)

        if not _check_corporate_id(request):
            return JSONResponse(status_code=403, content={"error": "access_denied"})

        user = await self._resolve_user(_normalise_email(raw_email), request)

        # Respect deactivated accounts — mPass authentication does not
        # override an explicit SurfSense account suspension.
        if user is None or not user.is_active:
            logger.warning("ProxyAuth: user inactive or not found for %r", raw_email)
            return await call_next(request)

        logger.debug("ProxyAuth: injected user id=%s for %r", user.id, user.email)
        request.state.proxy_user = user
        return await call_next(request)

    async def _resolve_user(self, email: str, request: Request) -> User | None:
        try:
            async with async_session_maker() as session:
                result = await session.execute(select(User).where(User.email == email))
                user = result.unique().scalar_one_or_none()
                created = False

                if user is None:
                    hashed_password = _password_helper.hash(secrets.token_urlsafe(32))
                    display_name = email.split("@")[0] or None
                    user = User(
                        email=email,
                        hashed_password=hashed_password,
                        display_name=display_name,
                        is_active=True,
                        is_verified=True,
                        is_superuser=False,
                    )
                    session.add(user)
                    try:
                        await session.commit()
                        await session.refresh(user)
                        created = True
                    except IntegrityError as exc:
                        # Concurrent request raced us to the insert — fall back
                        # to SELECT by email and re-raise if still not found.
                        await session.rollback()
                        result = await session.execute(
                            select(User).where(User.email == email)
                        )
                        user = result.unique().scalar_one_or_none()
                        if user is None:
                            logger.error(
                                "ProxyAuth: IntegrityError but user still not found "
                                "for %s: %s",
                                email,
                                exc,
                            )
                            return None

                # Fire UserManager lifecycle hooks. Middleware is a dumb
                # pipe — actual side-effects live in UserManager so the
                # fastapi-users login flows (JWT, Google OAuth) and this
                # proxy-auth path stay behaviourally identical.
                #
                # Per-hook side-effects (so a future maintainer doesn't
                # have to grep UserManager to map them):
                #   on_after_register  → default SearchSpace, RBAC roles,
                #                        owner membership, system prompts,
                #                        SMB auto-join, first-time LiteLLM
                #                        provisioning
                #   on_after_login     → throttled last_login update,
                #                        idempotent LiteLLM provisioning
                #                        (retries any registration-time
                #                        transient failure)
                #
                # Deferred import: app.users transitively imports from
                # this module's import chain via litellm_provisioning →
                # app.db → ProxyAuthMiddleware references. Module-level
                # import here creates a cycle at startup.
                from app.users import UserManager

                if created:
                    # on_after_register opens its own session internally
                    # (it inserts default SearchSpace, RBAC roles,
                    # prompts). Pass a fresh session and re-fetch the
                    # user inside it to avoid DetachedInstanceError —
                    # the outer `user` may have been rolled back by an
                    # IntegrityError race.
                    try:
                        async with async_session_maker() as reg_session:
                            reg_result = await reg_session.execute(
                                select(User).where(User.id == user.id)
                            )
                            reg_user = reg_result.unique().scalar_one_or_none()
                            if reg_user is None:
                                # User row disappeared between the outer
                                # INSERT+commit and this re-SELECT. Only
                                # plausible cause: an operator hard-deleted
                                # the row between the two statements. Log
                                # and skip on_after_register — there is
                                # nothing useful to do without the row.
                                # No exception raised: the surrounding
                                # ``except Exception`` would catch it
                                # without adding diagnostic value over a
                                # direct log line.
                                logger.error(
                                    "ProxyAuth: user %s vanished before "
                                    "on_after_register could run — default "
                                    "search space will not be created",
                                    user.id,
                                )
                            else:
                                reg_db = SQLAlchemyUserDatabase(reg_session, User)
                                reg_manager = UserManager(reg_db)
                                await reg_manager.on_after_register(
                                    reg_user, request=request
                                )
                    except Exception:
                        logger.exception(
                            "ProxyAuth: on_after_register failed for %s — "
                            "user created but default search space may be missing",
                            email,
                        )

                # on_after_login owns the throttled last_login update AND
                # the first-login LiteLLM provisioning hook (race-tolerant
                # via SELECT ... FOR UPDATE inside the service). Fire on
                # every request — both side-effects guard themselves so
                # the steady-state cost is two attribute reads + return.
                try:
                    login_db = SQLAlchemyUserDatabase(session, User)
                    login_manager = UserManager(login_db)
                    await login_manager.on_after_login(user, request=request)
                except Exception:
                    logger.exception("ProxyAuth: on_after_login failed for %s", email)

                return user

        except Exception:
            logger.exception("ProxyAuth: unexpected error resolving user for %s", email)
            return None
