"""Regression tests for ``UserManager.on_after_login`` is_active gating.

The hook fires on every authenticated request when ProxyAuthMiddleware
calls it manually (bypassing the fastapi-users built-in flows that
pre-filter inactive users). Without the ``is_active`` guard, deactivated
SSO accounts that still present a valid mPass JWT would:

- consume Askii credit on every request (one provisioning attempt per
  request until the wrapper's marker re-check would short-circuit it,
  which only happens AFTER the first successful provision — so a
  deactivated brand-new user could keep racking up Askii calls)
- accrue ``NewLLMConfig`` / ``ImageGenerationConfig`` / ``VisionLLMConfig``
  rows under a suspended identity

``last_login`` is intentionally NOT gated on ``is_active`` — a deactivated
account that still presents valid credentials is forensic signal worth
recording.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.users import UserManager

pytestmark = pytest.mark.unit


class _FakeUser:
    """Stand-in for the User ORM row.

    A bare ``MagicMock`` would auto-vivify ``last_login`` / ``is_active``
    to truthy MagicMock objects, masking the gates we want to assert on.
    """

    def __init__(self, *, is_active: bool, last_login: datetime | None = None) -> None:
        self.id = uuid.uuid4()
        self.is_active = is_active
        self.last_login = last_login
        self.litellm_auto_provisioned_at = None


def _make_user_db_with_session() -> MagicMock:
    """SQLAlchemyUserDatabase stand-in exposing ``.session`` for the hook."""
    user_db = MagicMock()
    user_db.session = AsyncMock()
    return user_db


def _make_request() -> MagicMock:
    """Starlette Request stand-in.

    The hook only forwards this opaquely to the wrapper; we only need a
    truthy non-None value so the ``if request is not None`` branch runs.
    """
    return MagicMock()


async def test_on_after_login_skips_provisioning_when_user_inactive() -> None:
    """Deactivated user must NOT trigger Askii provisioning even when the
    middleware fires the hook on a valid request."""
    user = _FakeUser(is_active=False)
    manager = UserManager(_make_user_db_with_session())

    with (
        patch(
            "app.users.ensure_personal_litellm_keys_for_user",
            new=AsyncMock(),
        ) as wrapper,
        patch.object(
            manager,
            "_update_last_login_throttled",
            new=AsyncMock(),
        ) as throttle,
    ):
        await manager.on_after_login(user, request=_make_request())

    # last_login bookkeeping fires regardless of is_active — forensic signal
    # that the suspended account attempted access.
    throttle.assert_awaited_once_with(user)
    # Provisioning skipped — deactivated accounts must not consume Askii
    # credit or accrue config rows.
    wrapper.assert_not_called()


async def test_on_after_login_fires_provisioning_when_user_active() -> None:
    """Active user with a request triggers both bookkeeping and provisioning."""
    user = _FakeUser(is_active=True)
    request = _make_request()
    user_db = _make_user_db_with_session()
    manager = UserManager(user_db)

    with (
        patch(
            "app.users.ensure_personal_litellm_keys_for_user",
            new=AsyncMock(),
        ) as wrapper,
        patch.object(
            manager,
            "_update_last_login_throttled",
            new=AsyncMock(),
        ) as throttle,
    ):
        await manager.on_after_login(user, request=request)

    throttle.assert_awaited_once_with(user)
    # Hook now extracts the access token from the request and passes it
    # explicitly (P6 #46 — service no longer takes `Request`). Import cfg
    # from `app.users` (NOT `app.config`) because another test in the
    # suite reloads `app.config` and rebinds the singleton — `app.users`
    # holds a stable reference to the instance the hook is using.
    from app.users import config as app_cfg

    assert wrapper.await_count == 1
    call_kwargs = wrapper.await_args.kwargs
    assert call_kwargs["session"] is user_db.session
    assert call_kwargs["user"] is user
    assert call_kwargs["cfg"] is app_cfg
    # access_token is whatever `read_mpass_access_token(request)` returns
    # for the MagicMock request — we don't pin the exact value, only that
    # the kwarg was passed.
    assert "access_token" in call_kwargs


async def test_on_after_login_commits_session_after_provisioning() -> None:
    """Hook owns the commit boundary (P6 #39 / #56).

    The provisioning service flushes only — writes are pending until the
    caller commits. ``on_after_login`` is the caller for the proxy-auth
    and fastapi-users-builtin paths, so the hook must ``commit()`` after
    the wrapper returns. Otherwise the SAVEPOINT-released writes would
    be rolled back when the session closes.

    Empty commit on the marker fast path is acceptable (cheap no-op).
    """
    user = _FakeUser(is_active=True)
    request = _make_request()
    user_db = _make_user_db_with_session()
    manager = UserManager(user_db)

    with (
        patch(
            "app.users.ensure_personal_litellm_keys_for_user",
            new=AsyncMock(),
        ),
        patch.object(
            manager,
            "_update_last_login_throttled",
            new=AsyncMock(),
        ),
    ):
        await manager.on_after_login(user, request=request)

    user_db.session.commit.assert_awaited_once()


async def test_on_after_login_swallows_commit_failure() -> None:
    """Commit failure after provisioning must NOT propagate — login flows
    must never break on bookkeeping. Logged + swallowed."""
    user = _FakeUser(is_active=True)
    request = _make_request()
    user_db = _make_user_db_with_session()
    user_db.session.commit = AsyncMock(side_effect=RuntimeError("conn closed"))
    manager = UserManager(user_db)

    with (
        patch(
            "app.users.ensure_personal_litellm_keys_for_user",
            new=AsyncMock(),
        ),
        patch.object(
            manager,
            "_update_last_login_throttled",
            new=AsyncMock(),
        ),
    ):
        # Must not raise.
        await manager.on_after_login(user, request=request)

    user_db.session.commit.assert_awaited_once()


async def test_on_after_login_skips_provisioning_when_request_is_none() -> None:
    """fastapi-users flows that omit ``request`` (rare but legal per the
    superclass signature) must not raise — the wrapper needs a request to
    read the mPass access token header."""
    user = _FakeUser(is_active=True)
    manager = UserManager(_make_user_db_with_session())

    with (
        patch(
            "app.users.ensure_personal_litellm_keys_for_user",
            new=AsyncMock(),
        ) as wrapper,
        patch.object(
            manager,
            "_update_last_login_throttled",
            new=AsyncMock(),
        ) as throttle,
    ):
        await manager.on_after_login(user, request=None)

    throttle.assert_awaited_once_with(user)
    wrapper.assert_not_called()


async def test_on_after_login_inactive_user_still_bumps_last_login() -> None:
    """Explicit assertion that the throttle runs before the is_active gate,
    matching the documented forensic-signal contract.

    Future refactors that reorder the gates (e.g., short-circuit on
    ``is_active`` before bookkeeping) would silently change audit
    behaviour; this test pins the ordering.
    """
    user = _FakeUser(
        is_active=False,
        last_login=datetime(2026, 1, 1, tzinfo=UTC),
    )
    manager = UserManager(_make_user_db_with_session())

    with (
        patch(
            "app.users.ensure_personal_litellm_keys_for_user",
            new=AsyncMock(),
        ) as wrapper,
        patch.object(
            manager,
            "_update_last_login_throttled",
            new=AsyncMock(),
        ) as throttle,
    ):
        await manager.on_after_login(user, request=_make_request())

    throttle.assert_awaited_once_with(user)
    wrapper.assert_not_called()
