"""
Regression-guard for SurfSense's architectural immunity to the
stale-session-on-user-switch class of bug.

SurfSense's `current_active_user` (in app.users) resolves to:
  proxy_user (set per-request by ProxyAuthMiddleware from
              X-Auth-Request-Email)  IF present, ELSE
  jwt_user   (decoded from Authorization: Bearer JWT)        IF present, ELSE
  raise 401 Not Authenticated.

This precedence is what makes SurfSense immune to the cross-app
stale-session-on-user-switch bug class that affected Plane #29 /
Outline #19 / Penpot #18 / Twenty #8. The other apps had to add an
explicit "compare upstream identity vs session identity → flush on
mismatch" middleware. SurfSense doesn't need that because the upstream
identity (proxy_user) ALWAYS wins over the persisted session (jwt_user)
when both are present.

The cross-app contract:
  awais786/sso-rules-moneta:openspec/specs/proxy-auth-middleware/spec.md

These tests pin the precedence so a future refactor that flips the
priority (or removes the proxy_user check) cannot silently re-introduce
the bug. Without them, a well-intentioned "simplify auth to just use
JWT" change would pass type-checks and ship the regression.
"""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException, Request
from starlette.requests import Request as StarletteRequest

from app.users import current_active_user, current_optional_user


def _make_request(proxy_user=None) -> Request:
    """Build a minimal Request whose .state.proxy_user is settable."""
    scope = {"type": "http", "headers": [], "method": "GET", "path": "/"}
    request = StarletteRequest(scope)
    if proxy_user is not None:
        request.state.proxy_user = proxy_user
    return request


def _make_user(email: str, *, is_active: bool = True):
    user = MagicMock()
    user.id = uuid.uuid4()
    user.email = email
    user.is_active = is_active
    return user


@pytest.mark.asyncio
async def test_proxy_user_wins_when_both_proxy_and_jwt_present():
    """
    GIVEN  proxy_user resolves alice from upstream header
           AND jwt_user resolves bob from Bearer JWT (stale)
    WHEN   current_active_user runs
    THEN   returns alice (the upstream identity)

    This is the architectural-immunity contract. SurfSense never serves
    the stale JWT identity when an upstream identity is asserted on the
    same request. If this test fails, the cross-app stale-session bug
    class is re-introduced into SurfSense.
    """
    alice = _make_user("alice@example.com")
    bob = _make_user("bob@example.com")
    request = _make_request(proxy_user=alice)

    # Bypass the SMB auto-join side effect; we're testing precedence only.
    import app.users as users_module
    original = users_module.auto_join_smb_search_space
    users_module.auto_join_smb_search_space = AsyncMock(return_value=None)
    try:
        result = await current_active_user(request, jwt_user=bob)
    finally:
        users_module.auto_join_smb_search_space = original

    assert result is alice
    assert result is not bob


@pytest.mark.asyncio
async def test_falls_back_to_jwt_when_proxy_user_absent():
    """
    GIVEN  proxy_user is NOT set on the request (no X-Auth-Request-Email)
           AND jwt_user resolves alice from Bearer JWT
    WHEN   current_active_user runs
    THEN   returns alice (the JWT identity)

    Header absence is NOT a logout signal per the cross-app spec — it
    legitimately occurs for internal calls, bypass routes, OPTIONS
    preflight, and direct backend hits at 127.0.0.1. SurfSense MUST
    serve the JWT identity in this case, not 401.
    """
    alice = _make_user("alice@example.com")
    request = _make_request(proxy_user=None)

    import app.users as users_module
    original = users_module.auto_join_smb_search_space
    users_module.auto_join_smb_search_space = AsyncMock(return_value=None)
    try:
        result = await current_active_user(request, jwt_user=alice)
    finally:
        users_module.auto_join_smb_search_space = original

    assert result is alice


@pytest.mark.asyncio
async def test_raises_401_when_neither_proxy_nor_jwt_present():
    """
    GIVEN  proxy_user is NOT set AND jwt_user is None (no Bearer)
    WHEN   current_active_user runs
    THEN   raises 401 Not Authenticated

    Sanity: an entirely unauthenticated request must be rejected.
    """
    request = _make_request(proxy_user=None)

    with pytest.raises(HTTPException) as exc_info:
        await current_active_user(request, jwt_user=None)

    assert exc_info.value.status_code == 401


@pytest.mark.asyncio
async def test_optional_user_returns_proxy_when_both_present():
    """
    GIVEN  proxy_user resolves alice AND jwt_user resolves bob
    WHEN   current_optional_user runs
    THEN   returns alice

    Same precedence rule, optional variant. Returns None instead of
    raising when neither is present.
    """
    alice = _make_user("alice@example.com")
    bob = _make_user("bob@example.com")
    request = _make_request(proxy_user=alice)

    result = await current_optional_user(request, jwt_user=bob)

    assert result is alice


@pytest.mark.asyncio
async def test_optional_user_returns_none_when_neither_present():
    """
    GIVEN  no proxy_user AND no jwt_user
    WHEN   current_optional_user runs
    THEN   returns None (does NOT raise)
    """
    request = _make_request(proxy_user=None)

    result = await current_optional_user(request, jwt_user=None)

    assert result is None
