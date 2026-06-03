import logging
import uuid
from datetime import UTC, datetime

import httpx
from fastapi import Depends, HTTPException, Request, Response, status
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi_users import BaseUserManager, FastAPIUsers, UUIDIDMixin, models
from fastapi_users.authentication import (
    AuthenticationBackend,
    BearerTransport,
    JWTStrategy,
)
from fastapi_users.db import SQLAlchemyUserDatabase
from pydantic import BaseModel
from sqlalchemy import update

from app.config import config
from app.db import (
    Prompt,
    SearchSpace,
    SearchSpaceMembership,
    SearchSpaceRole,
    User,
    async_session_maker,
    get_default_roles_config,
    get_user_db,
)
from app.prompts.system_defaults import SYSTEM_PROMPT_DEFAULTS
from app.services.litellm_provisioning import (
    ensure_org_litellm_keys_for_admin,
    ensure_personal_litellm_keys,
    ensure_personal_litellm_keys_for_user,
    read_mpass_access_token,
)
from app.services.smb_auto_join import auto_join_smb_search_space
from app.utils.refresh_tokens import create_refresh_token

logger = logging.getLogger(__name__)

# `on_after_login` is invoked on every authenticated SSO request by
# ProxyAuthMiddleware (proxy auth has no server-side session to amortize
# the cost across, unlike Django sessions or JWT-only flows). Without a
# throttle, last_login would issue one UPDATE per API call. The window
# is small enough to keep the metric usefully fresh, large enough to
# disappear from the request profile.
_LAST_LOGIN_THROTTLE_SECONDS = 300


def _session_from_user_db(user_db: SQLAlchemyUserDatabase):
    """Pull the underlying AsyncSession from fastapi-users' user_db wrapper.

    ``SQLAlchemyUserDatabase.session`` is not part of fastapi-users'
    publicly documented API surface — it is the attribute name fastapi-
    users currently uses to expose the session it was constructed with.
    Isolated here so a future fastapi-users upgrade only requires editing
    one site. See review item P6 #42.
    """
    return user_db.session


class BearerResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str


SECRET = config.SECRET_KEY


if config.AUTH_TYPE == "GOOGLE":
    from httpx_oauth.clients.google import GoogleOAuth2

    google_oauth_client = GoogleOAuth2(
        config.GOOGLE_OAUTH_CLIENT_ID,
        config.GOOGLE_OAUTH_CLIENT_SECRET,
    )


class UserManager(UUIDIDMixin, BaseUserManager[User, uuid.UUID]):
    """
    Custom user manager extending fastapi-users BaseUserManager.

    Authentication returns a generic error for both non-existent accounts
    and incorrect passwords to comply with OWASP WSTG-IDNT-04 and
    prevent user enumeration attacks.
    """

    reset_password_token_secret = SECRET
    verification_token_secret = SECRET

    async def oauth_callback(
        self,
        oauth_name: str,
        access_token: str,
        account_id: str,
        account_email: str,
        expires_at: int | None = None,
        refresh_token: str | None = None,
        request: Request | None = None,
        *,
        associate_by_email: bool = False,
        is_verified_by_default: bool = False,
    ) -> User:
        """
        Override OAuth callback to capture Google profile data (name, avatar).
        """
        # Call parent implementation to create/get user
        user = await super().oauth_callback(
            oauth_name,
            access_token,
            account_id,
            account_email,
            expires_at,
            refresh_token,
            request,
            associate_by_email=associate_by_email,
            is_verified_by_default=is_verified_by_default,
        )

        # Fetch and store Google profile data if not already set
        if oauth_name == "google" and (not user.display_name or not user.avatar_url):
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get(
                        "https://people.googleapis.com/v1/people/me",
                        params={"personFields": "names,photos"},
                        headers={"Authorization": f"Bearer {access_token}"},
                    )
                    response.raise_for_status()
                    profile = response.json()

                update_dict = {}

                # Extract name from names array
                names = profile.get("names", [])
                if not user.display_name and names:
                    display_name = names[0].get("displayName")
                    if display_name:
                        update_dict["display_name"] = display_name

                # Extract photo URL from photos array
                photos = profile.get("photos", [])
                if not user.avatar_url and photos:
                    photo_url = photos[0].get("url")
                    if photo_url:
                        update_dict["avatar_url"] = photo_url

                if update_dict:
                    user = await self.user_db.update(user, update_dict)

            except Exception as e:
                logger.warning(f"Failed to fetch Google profile: {e}")

        return user

    async def on_after_login(
        self,
        user: User,
        request: Request | None = None,
        response: Response | None = None,
    ) -> None:
        """Update last_login (throttled) + auto-provision personal LiteLLM keys.

        Fired on every successful login event:
        - fastapi-users JWT login (``/auth/jwt/login``) — once per credential
          exchange.
        - Google OAuth callback — once per OAuth round trip.
        - ProxyAuthMiddleware — on every authenticated SSO request.

        Both side-effects are best-effort: a failure here must never
        propagate, otherwise login would break for bookkeeping reasons.

        The hook uses ``self.user_db.session`` (the request-scoped session
        wired into the user db) so callers don't pay per-request session
        construction cost.

        Transaction ownership
        ---------------------
        The provisioning service flushes only (writes wrapped in a
        SAVEPOINT for race-loss isolation, see service module docstring).
        This hook commits the session after the wrapper returns so the
        SAVEPOINT-released writes become durable. Empty commits (when the
        wrapper short-circuits on the marker fast path) are cheap; the
        wrapper guarantees no rollback of caller state on its own. See
        review item P6 #56 — the proper long-term fix is a
        ``UserLifecycleService`` that owns transaction lifecycle
        end-to-end so this hook doesn't need to know about commits.

        Inactive accounts
        -----------------
        ``last_login`` is updated regardless of ``is_active`` — a deactivated
        account that still presents valid credentials is useful forensic
        signal ("when did this suspended account last try to access?").
        LiteLLM provisioning is skipped for inactive accounts: a
        deactivated user must not consume Askii credit or accrue config
        rows. fastapi-users' built-in flows (JWT, OAuth) pre-filter
        inactive users before firing this hook, but ProxyAuthMiddleware
        fires it manually for every authenticated request — without this
        guard, every request from a deactivated SSO user would attempt
        provisioning.
        """
        await self._update_last_login_throttled(user)
        if not user.is_active:
            return
        if request is not None:
            session = _session_from_user_db(self.user_db)
            access_token = read_mpass_access_token(request)
            await ensure_personal_litellm_keys_for_user(
                session=session,
                user=user,
                access_token=access_token,
                cfg=config,
            )
            # Commit any pending writes from the provisioning service
            # (it flushes only). Safe to commit unconditionally: on the
            # marker fast path the session has no pending writes; on the
            # SAVEPOINT-released path the writes are the inserts + marker
            # UPDATE. On race-loss / failure the SAVEPOINT already rolled
            # back, so commit is a no-op for provisioning.
            try:
                await session.commit()
            except Exception:
                logger.exception(
                    "on_after_login: post-provisioning commit failed for user %s",
                    user.id,
                )
                try:
                    await session.rollback()
                except Exception:
                    logger.exception(
                        "on_after_login: rollback after failed commit also failed for user %s",
                        user.id,
                    )

            # Org-space LiteLLM provisioning (best-effort, separate from the
            # personal path above). Fires only for the is_owner admin of the
            # shared SMB/Organization space; one-shot per org space via
            # SearchSpace.litellm_auto_provisioned_at. Committed separately so a
            # personal-path commit failure cannot suppress the org write (and
            # vice versa). Reuses the access token already read above.
            await ensure_org_litellm_keys_for_admin(
                session=session,
                user=user,
                access_token=access_token,
                cfg=config,
            )
            try:
                await session.commit()
            except Exception:
                logger.exception(
                    "on_after_login: post-org-provisioning commit failed for user %s",
                    user.id,
                )
                try:
                    await session.rollback()
                except Exception:
                    logger.exception(
                        "on_after_login: rollback after failed org commit also failed for user %s",
                        user.id,
                    )

    async def _update_last_login_throttled(self, user: User) -> None:
        """Update ``user.last_login`` at most once per throttle window.

        Cheap math first (reads attribute on the in-memory user object) so
        the steady-state cost is one comparison and an early return — no
        SQL, no transaction.
        """
        now = datetime.now(UTC)

        # Snapshot the ORM attributes once, up front. A detached/expired user
        # makes even a plain attribute read raise (e.g. MissingGreenlet); capture
        # them under a guard so the throttle math and the failure logger below
        # never re-touch the ORM object on the unhappy path.
        try:
            user_id = user.id
            last_login = user.last_login
        except Exception as e:
            logger.warning(f"Failed to read user for last_login update: {e}")
            return

        if (
            last_login is not None
            and (now - last_login).total_seconds() <= _LAST_LOGIN_THROTTLE_SECONDS
        ):
            return

        try:
            async with async_session_maker() as session:
                await session.execute(
                    update(User).where(User.id == user_id).values(last_login=now)
                )
                await session.commit()
                user.last_login = now  # mirror onto caller's object
        except Exception as e:
            logger.warning(f"Failed to update last_login for user {user_id}: {e}")

    async def on_after_register(self, user: User, request: Request | None = None):
        """
        Called after a user registers. Creates a default search space for the user
        so they can start chatting immediately without manual setup.
        """
        logger.info(f"User {user.id} has registered. Creating default search space...")

        try:
            async with async_session_maker() as session:
                # Create default search space
                default_search_space = SearchSpace(
                    name="My Search Space",
                    description="Your personal search space",
                    user_id=user.id,
                )
                session.add(default_search_space)
                await session.flush()  # Get the search space ID

                # Create default roles
                default_roles = get_default_roles_config()
                owner_role_id = None

                for role_config in default_roles:
                    db_role = SearchSpaceRole(
                        name=role_config["name"],
                        description=role_config["description"],
                        permissions=role_config["permissions"],
                        is_default=role_config["is_default"],
                        is_system_role=role_config["is_system_role"],
                        search_space_id=default_search_space.id,
                    )
                    session.add(db_role)
                    await session.flush()

                    if role_config["name"] == "Owner":
                        owner_role_id = db_role.id

                # Create owner membership
                owner_membership = SearchSpaceMembership(
                    user_id=user.id,
                    search_space_id=default_search_space.id,
                    role_id=owner_role_id,
                    is_owner=True,
                )
                session.add(owner_membership)

                for default in SYSTEM_PROMPT_DEFAULTS:
                    session.add(
                        Prompt(
                            user_id=user.id,
                            default_prompt_slug=default["slug"],
                            name=default["name"],
                            prompt=default["prompt"],
                            mode=default["mode"],
                            version=default["version"],
                        )
                    )

                await session.commit()
                logger.info(
                    f"Created default search space (ID: {default_search_space.id}) for user {user.id}"
                )

                # Best-effort: auto-provision the personal LiteLLM key + 4
                # config rows (agent / doc-summary / image / vision). Gated
                # inside the service on AUTH_TYPE=SSO +
                # AUTO_PROVISION_LITELLM_KEY=true; one-shot per user via
                # `user.litellm_auto_provisioned_at`. A failure here must
                # NOT abort registration — the service's SAVEPOINT rolls
                # back provisioning writes only; the on_after_login retry
                # path will try again on the user's next request.
                # Defense-in-depth try/except (redundant safety net —
                # `logger.warning` rather than `logger.exception` so an
                # alert here flags a contract regression in the service,
                # not a normal-path failure). `request` is Optional in
                # fastapi-users; the service needs it to read the mPass
                # access-token header.
                #
                # The service flushes only (see its module docstring on
                # transaction ownership) — we own the commit for our own
                # provisioning writes. Registration is already durable
                # from the commit above, matching the upstream contract.
                if request is not None:
                    try:
                        await ensure_personal_litellm_keys(
                            session=session,
                            user=user,
                            search_space=default_search_space,
                            access_token=read_mpass_access_token(request),
                            cfg=config,
                        )
                        await session.commit()
                    except Exception as e:
                        logger.warning(
                            "Auto-provisioning LiteLLM keys raised unexpectedly "
                            "for user %s — service's best-effort contract is "
                            "supposed to prevent this (%s: %s); on_after_login "
                            "retry path will run on next request",
                            user.id,
                            type(e).__name__,
                            e,
                        )
        except Exception as e:
            logger.error(
                f"Failed to create default search space for user {user.id}: {e}"
            )

        # SMB auto-join — runs once at user creation, NOT on every request.
        # Per-request auto-join silently re-grants membership to users an
        # operator has explicitly removed (the DELETE /searchspaces/{id}/
        # members/{membership_id} endpoint hard-deletes the row, leaving no
        # tombstone for the auto-join function to consult). Doing it here
        # keeps removed users removed.
        #
        # Trade-off: users who registered BEFORE the SMB workspace existed
        # are not retroactively auto-joined — operators can backfill those
        # with a one-time SQL INSERT against `search_space_memberships`.
        try:
            await auto_join_smb_search_space(user.id)
        except Exception:
            logger.exception(
                "SMB auto-join failed for newly registered user %s", user.id
            )

    async def on_after_forgot_password(
        self, user: User, token: str, request: Request | None = None
    ):
        print(f"User {user.id} has forgot their password. Reset token: {token}")

    async def on_after_request_verify(
        self, user: User, token: str, request: Request | None = None
    ):
        print(f"Verification requested for user {user.id}. Verification token: {token}")


async def get_user_manager(user_db: SQLAlchemyUserDatabase = Depends(get_user_db)):
    yield UserManager(user_db)


def get_jwt_strategy() -> JWTStrategy[models.UP, models.ID]:
    return JWTStrategy(
        secret=SECRET,
        lifetime_seconds=config.ACCESS_TOKEN_LIFETIME_SECONDS,
    )


# # COOKIE AUTH | Uncomment if you want to use cookie auth.
# from fastapi_users.authentication import (
#     CookieTransport,
# )
# class CustomCookieTransport(CookieTransport):
#     async def get_login_response(self, token: str) -> Response:
#         response = RedirectResponse(config.OAUTH_REDIRECT_URL, status_code=302)
#         return self._set_login_cookie(response, token)

# cookie_transport = CustomCookieTransport(
#     cookie_max_age=3600,
# )

# auth_backend = AuthenticationBackend(
#     name="jwt",
#     transport=cookie_transport,
#     get_strategy=get_jwt_strategy,
# )


# BEARER AUTH CODE.
class CustomBearerTransport(BearerTransport):
    async def get_login_response(self, token: str) -> Response:
        import jwt

        # Decode JWT to get user_id for refresh token creation
        try:
            payload = jwt.decode(
                token, SECRET, algorithms=["HS256"], options={"verify_aud": False}
            )
            user_id = uuid.UUID(payload.get("sub"))
            refresh_token = await create_refresh_token(user_id)
        except Exception as e:
            logger.error(f"Failed to create refresh token: {e}")
            # Fall back to response without refresh token
            refresh_token = ""

        bearer_response = BearerResponse(
            access_token=token,
            refresh_token=refresh_token,
            token_type="bearer",
        )

        if config.AUTH_TYPE == "GOOGLE":
            redirect_url = (
                f"{config.NEXT_FRONTEND_URL}/auth/callback"
                f"?token={bearer_response.access_token}"
                f"&refresh_token={bearer_response.refresh_token}"
            )
            return RedirectResponse(redirect_url, status_code=302)
        else:
            return JSONResponse(bearer_response.model_dump())


bearer_transport = CustomBearerTransport(tokenUrl="auth/jwt/login")


auth_backend = AuthenticationBackend(
    name="jwt",
    transport=bearer_transport,
    get_strategy=get_jwt_strategy,
)

fastapi_users = FastAPIUsers[User, uuid.UUID](get_user_manager, [auth_backend])

_jwt_current_optional_user = fastapi_users.current_user(active=True, optional=True)


async def current_active_user(
    request: Request,
    jwt_user: User | None = Depends(_jwt_current_optional_user),
) -> User:
    """
    Returns the authenticated user.

    Checks request.state.proxy_user first (set by ProxyAuthMiddleware when
    mPass proxy auth is active). Falls back to JWT Bearer token validation
    so existing email/password and Google OAuth flows continue to work when
    proxy auth is disabled.

    SMB shared SearchSpace auto-join runs in `on_after_register` (one-shot
    at user creation), NOT here on every request. Per-request auto-join
    silently re-grants membership to users that operators have explicitly
    removed via DELETE /searchspaces/{id}/members/{membership_id}.
    """
    proxy_user = getattr(request.state, "proxy_user", None)
    if proxy_user is not None:
        return proxy_user
    if jwt_user is not None:
        return jwt_user
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Not authenticated",
    )


async def current_optional_user(
    request: Request,
    jwt_user: User | None = Depends(_jwt_current_optional_user),
) -> User | None:
    proxy_user = getattr(request.state, "proxy_user", None)
    return proxy_user if proxy_user is not None else jwt_user
