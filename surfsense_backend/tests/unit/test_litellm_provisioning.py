"""Unit tests for ``app.services.litellm_provisioning``.

The service does two things we care about:

1. It calls the Askii SDK exactly once per successful provision attempt
   (with `models=union` and `default_model=agent_model`), and not at all
   when the feature flag is off / token is missing / the user has already
   been auto-provisioned (``user.litellm_auto_provisioned_at`` is non-NULL).
2. On a successful Askii response it inserts FOUR config rows —
   ``NewLLMConfig`` for agent + doc-summary, ``ImageGenerationConfig`` for
   image gen, ``VisionLLMConfig`` for vision — updates all four
   ``SearchSpace`` FKs, and stamps ``user.litellm_auto_provisioned_at`` —
   all in one transaction.

DB writes are simulated with ``unittest.mock.AsyncMock`` for the session and
small ``_FakeUser`` / ``_FakeSearchSpace`` types so we can observe FK and
marker mutations directly. The Askii SDK is exercised end-to-end via
``httpx.MockTransport``.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from sqlalchemy.exc import SQLAlchemyError
from starlette.requests import Request as StarletteRequest

from app.db import (
    ImageGenerationConfig,
    ImageGenProvider,
    LiteLLMProvider,
    NewLLMConfig,
    VisionLLMConfig,
    VisionProvider,
)
from app.services.litellm_provisioning import (
    ALL_ROW_NAMES,
    LITELLM_KEY_ALIAS,
    ROW_NAME_AGENT,
    ROW_NAME_DOC_SUMMARY,
    ROW_NAME_IMAGE,
    ROW_NAME_VISION,
    ensure_personal_litellm_keys,
    ensure_personal_litellm_keys_for_user,
    read_mpass_access_token,
    should_auto_provision,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_request(*, access_token: str | None) -> StarletteRequest:
    headers: list[tuple[bytes, bytes]] = []
    if access_token is not None:
        headers.append((b"x-auth-request-access-token", access_token.encode()))
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/",
        "headers": headers,
    }
    return StarletteRequest(scope)


class _FakeUser:
    """Minimal stand-in for a User row.

    A bare ``MagicMock`` would auto-vivify ``litellm_auto_provisioned_at``
    to a truthy MagicMock, which would silently flip every test through
    the idempotent short-circuit.
    """

    def __init__(self, *, provisioned_at: datetime | None = None) -> None:
        self.id = uuid.uuid4()
        self.litellm_auto_provisioned_at = provisioned_at


class _FakeSearchSpace:
    """Minimal stand-in for a SearchSpace row.

    A ``MagicMock`` would silently swallow every attribute assignment, so
    we use a real class to make FK assertions meaningful.
    """

    def __init__(self, *, user_id: uuid.UUID, ss_id: int = 1) -> None:
        self.id = ss_id
        self.user_id = user_id
        self.agent_llm_id = 0
        self.document_summary_llm_id = 0
        self.image_generation_config_id = 0
        self.vision_llm_config_id = 0


def _make_session(*, lock_finds_marker: bool = False) -> AsyncMock:
    """An AsyncMock SQLAlchemy session.

    The service makes up to three ``session.execute()`` calls per happy
    path: ``SELECT ... FOR UPDATE`` (marker re-check under lock), then
    ``UPDATE SearchSpace`` (FK wiring), then ``UPDATE User`` (marker
    stamp). Only the first inspects its return value (``scalar_one_or_none``);
    the UPDATEs ignore the cursor result, so a single ``return_value`` is
    enough.

    ``session.begin_nested()`` returns an async context manager (the
    SAVEPOINT). The default mock enters / exits cleanly; exceptions
    raised inside the ``async with`` block propagate (as in production
    where SAVEPOINT rolls back and re-raises).

    Set ``lock_finds_marker=True`` to simulate losing the race — a
    sibling request stamped the marker during our Askii call.
    """
    session = AsyncMock()
    lock_result = MagicMock()
    lock_result.scalar_one_or_none.return_value = (
        datetime.now(UTC) if lock_finds_marker else None
    )
    session.execute = AsyncMock(return_value=lock_result)

    added: list[Any] = []
    session.add = MagicMock(side_effect=added.append)
    next_id = [99]

    async def _flush() -> None:
        # Simulate DB autoincrement so caller can read `.id` after flush.
        for obj in added:
            if getattr(obj, "id", None) is None:
                obj.id = next_id[0]
                next_id[0] += 1

    session.flush = AsyncMock(side_effect=_flush)
    session.commit = AsyncMock()
    session.rollback = AsyncMock()

    # SAVEPOINT context manager. AsyncMock's __aenter__ / __aexit__ are
    # auto-async; __aexit__ returns False so any exception raised inside
    # `async with session.begin_nested():` propagates upward (matches
    # real SAVEPOINT behaviour: rollback-to-savepoint, then re-raise).
    nested_cm = AsyncMock()
    nested_cm.__aenter__ = AsyncMock(return_value=nested_cm)
    nested_cm.__aexit__ = AsyncMock(return_value=False)
    session.begin_nested = MagicMock(return_value=nested_cm)

    session._added = added  # type: ignore[attr-defined]
    return session


def _mock_transport(handler: Any) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        base_url="https://api.askii.test",
        transport=httpx.MockTransport(handler),
    )


def _set_config(monkeypatch: pytest.MonkeyPatch, **overrides: Any) -> Any:
    """Patch the shared config singleton AND return it for ``cfg=`` passing.

    The service now takes ``cfg`` as an explicit parameter (review item
    P6 #45) so tests can drop ``monkeypatch`` entirely; this helper is
    kept for tests that haven't yet migrated to a pure :func:`_cfg_ns`
    fixture — they call ``cfg = _set_config(monkeypatch, ...)`` and pass
    the returned reference into the service.
    """
    from app.config import config as cfg

    defaults = {
        "AUTH_TYPE": "SSO",
        "AUTO_PROVISION_LITELLM_KEY": True,
        "ASKII_BASE_URL": "https://api.askii.test",
        "ASKII_LITELLM_BASE_URL": "",  # blank ⇒ inherits ASKII_BASE_URL
        "ASKII_AGENT_MODEL": "gpt-5.4-mini",
        "ASKII_DOCUMENT_SUMMARY_MODEL": "",  # blank ⇒ inherits agent by default
        "ASKII_IMAGE_GEN_MODEL": "gpt-image-1.5",
        "ASKII_VISION_MODEL": "gpt-5.4-nano",
        "ASKII_LITELLM_KEY_DURATION_DAYS": 90,
    }
    defaults.update(overrides)
    for name, value in defaults.items():
        monkeypatch.setattr(cfg, name, value, raising=False)
    return cfg


# ---------------------------------------------------------------------------
# should_auto_provision
# ---------------------------------------------------------------------------


def _cfg_ns(**overrides: Any) -> SimpleNamespace:
    base = {
        "AUTH_TYPE": "SSO",
        "AUTO_PROVISION_LITELLM_KEY": True,
        "ASKII_BASE_URL": "https://api.askii.test",
        "ASKII_LITELLM_BASE_URL": "",  # optional
        "ASKII_AGENT_MODEL": "gpt-5.4-mini",
        "ASKII_IMAGE_GEN_MODEL": "gpt-image-1.5",
        "ASKII_VISION_MODEL": "gpt-5.4-nano",
        "ASKII_DOCUMENT_SUMMARY_MODEL": "",
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def test_should_auto_provision_happy_path() -> None:
    assert should_auto_provision(_cfg_ns()) is True


def test_should_auto_provision_doc_summary_blank_is_ok() -> None:
    """Doc-summary is optional — blank value must NOT disable the gate."""
    assert should_auto_provision(_cfg_ns(ASKII_DOCUMENT_SUMMARY_MODEL="")) is True


def test_should_auto_provision_litellm_base_url_blank_is_ok() -> None:
    """LITELLM_BASE_URL is optional — blank inherits ASKII_BASE_URL, gate stays open."""
    assert should_auto_provision(_cfg_ns(ASKII_LITELLM_BASE_URL="")) is True


# ---------------------------------------------------------------------------
# read_mpass_access_token — public helper for HTTP callers (P6 #46)
# ---------------------------------------------------------------------------


def test_read_mpass_access_token_returns_stripped_header_value() -> None:
    """Header present → stripped token returned."""
    req = _make_request(access_token="  cognito-jwt  ")
    assert read_mpass_access_token(req) == "cognito-jwt"


def test_read_mpass_access_token_returns_none_when_header_missing() -> None:
    """Header absent → None (caller should treat as 'not behind mPass')."""
    req = _make_request(access_token=None)
    assert read_mpass_access_token(req) is None


def test_read_mpass_access_token_returns_none_for_whitespace_only_header() -> None:
    """Header present but blank/whitespace → None (same as absent)."""
    req = _make_request(access_token="   ")
    assert read_mpass_access_token(req) is None


def test_should_auto_provision_off_when_auth_type_not_sso() -> None:
    assert should_auto_provision(_cfg_ns(AUTH_TYPE="GOOGLE")) is False


def test_should_auto_provision_off_when_flag_false() -> None:
    assert should_auto_provision(_cfg_ns(AUTO_PROVISION_LITELLM_KEY=False)) is False


def test_should_auto_provision_off_when_base_url_blank() -> None:
    assert should_auto_provision(_cfg_ns(ASKII_BASE_URL="")) is False


def test_should_auto_provision_off_when_agent_model_blank() -> None:
    assert should_auto_provision(_cfg_ns(ASKII_AGENT_MODEL="")) is False


def test_should_auto_provision_off_when_image_model_blank() -> None:
    assert should_auto_provision(_cfg_ns(ASKII_IMAGE_GEN_MODEL="")) is False


def test_should_auto_provision_off_when_vision_model_blank() -> None:
    assert should_auto_provision(_cfg_ns(ASKII_VISION_MODEL="")) is False


def test_all_row_names_is_the_four_constants() -> None:
    assert {
        ROW_NAME_AGENT,
        ROW_NAME_DOC_SUMMARY,
        ROW_NAME_IMAGE,
        ROW_NAME_VISION,
    } == ALL_ROW_NAMES


# ---------------------------------------------------------------------------
# ensure_personal_litellm_keys — gate + early returns
# ---------------------------------------------------------------------------


async def test_returns_false_when_feature_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _set_config(monkeypatch, AUTO_PROVISION_LITELLM_KEY=False)
    session = _make_session()
    user = _FakeUser()
    ss = _FakeSearchSpace(user_id=user.id)

    sdk_called = False

    def handler(req: httpx.Request) -> httpx.Response:
        nonlocal sdk_called
        sdk_called = True
        return httpx.Response(200, json={})

    client = _mock_transport(handler)
    try:
        ok = await ensure_personal_litellm_keys(
            session=session,
            user=user,
            search_space=ss,
            access_token="jwt",
            http_client=client,
            cfg=cfg,
        )
    finally:
        await client.aclose()

    assert ok is False
    assert sdk_called is False
    session.add.assert_not_called()
    assert ss.agent_llm_id == 0
    assert ss.document_summary_llm_id == 0
    assert ss.image_generation_config_id == 0
    assert ss.vision_llm_config_id == 0
    assert user.litellm_auto_provisioned_at is None


async def test_returns_false_when_auth_type_not_sso(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _set_config(monkeypatch, AUTH_TYPE="GOOGLE")
    session = _make_session()
    user = _FakeUser()
    ss = _FakeSearchSpace(user_id=user.id)

    client = _mock_transport(lambda r: httpx.Response(200, json={}))
    try:
        ok = await ensure_personal_litellm_keys(
            session=session,
            user=user,
            search_space=ss,
            access_token="jwt",
            http_client=client,
            cfg=cfg,
        )
    finally:
        await client.aclose()
    assert ok is False
    session.execute.assert_not_called()
    assert user.litellm_auto_provisioned_at is None


async def test_skips_when_access_token_header_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _set_config(monkeypatch)
    session = _make_session()
    user = _FakeUser()
    ss = _FakeSearchSpace(user_id=user.id)

    sdk_called = False

    def handler(req: httpx.Request) -> httpx.Response:
        nonlocal sdk_called
        sdk_called = True
        return httpx.Response(200, json={})

    client = _mock_transport(handler)
    try:
        ok = await ensure_personal_litellm_keys(
            session=session,
            user=user,
            search_space=ss,
            access_token=None,
            http_client=client,
            cfg=cfg,
        )
    finally:
        await client.aclose()

    assert ok is False
    assert sdk_called is False
    session.add.assert_not_called()
    assert user.litellm_auto_provisioned_at is None


async def test_idempotent_when_user_already_provisioned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """User.litellm_auto_provisioned_at is the one-shot marker.

    Once it's set, the service short-circuits to ``True`` without touching
    the session, hitting Askii, or wiring any FKs — irrespective of which
    search space is passed or whether config rows still exist.
    """
    cfg = _set_config(monkeypatch)
    stamped = datetime(2026, 5, 1, 12, 0, 0, tzinfo=UTC)
    session = _make_session()
    user = _FakeUser(provisioned_at=stamped)
    ss = _FakeSearchSpace(user_id=user.id)

    sdk_called = False

    def handler(req: httpx.Request) -> httpx.Response:
        nonlocal sdk_called
        sdk_called = True
        return httpx.Response(200, json={})

    client = _mock_transport(handler)
    try:
        ok = await ensure_personal_litellm_keys(
            session=session,
            user=user,
            search_space=ss,
            access_token="jwt",
            http_client=client,
            cfg=cfg,
        )
    finally:
        await client.aclose()

    assert ok is True
    assert sdk_called is False
    session.add.assert_not_called()
    session.execute.assert_not_called()
    assert ss.agent_llm_id == 0  # not touched on idempotent path
    assert user.litellm_auto_provisioned_at == stamped  # marker unchanged


# ---------------------------------------------------------------------------
# ensure_personal_litellm_keys — full 4-row provisioning flow
# ---------------------------------------------------------------------------


async def test_happy_path_inserts_four_rows_links_fks_and_stamps_marker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _set_config(
        monkeypatch,
        ASKII_BASE_URL="https://api.askii.test",
        ASKII_LITELLM_BASE_URL="https://litellm.askii.test",  # explicit override
        ASKII_AGENT_MODEL="gpt-5.4-mini",
        ASKII_DOCUMENT_SUMMARY_MODEL="gpt-doc-summary",  # explicit, different model
        ASKII_IMAGE_GEN_MODEL="gpt-image-1.5",
        ASKII_VISION_MODEL="gpt-5.4-nano",
        ASKII_LITELLM_KEY_DURATION_DAYS=30,
    )
    session = _make_session()
    user = _FakeUser()
    ss = _FakeSearchSpace(user_id=user.id, ss_id=42)

    before = datetime.now(UTC)

    recorded: list[httpx.Request] = []

    def handler(req: httpx.Request) -> httpx.Response:
        recorded.append(req)
        return httpx.Response(
            200,
            json={
                "api_key": "sk-provisioned-key-1234567890",
                "key_name": "moneta-user-001",
                "user_id": "askii-user-1",
                "expires": "2026-08-01T00:00:00Z",
            },
        )

    client = _mock_transport(handler)
    try:
        ok = await ensure_personal_litellm_keys(
            session=session,
            user=user,
            search_space=ss,
            access_token="cognito-jwt",
            http_client=client,
            cfg=cfg,
        )
    finally:
        await client.aclose()

    assert ok is True

    # exactly one Askii call, on the provision endpoint, with the right
    # token / alias / models / default_model
    assert len(recorded) == 1
    assert recorded[0].url.path == "/platform/provision-key"
    sent = json.loads(recorded[0].content)
    assert sent["mpass_token"] == "cognito-jwt"
    assert sent["key_alias"] == LITELLM_KEY_ALIAS
    assert sent["duration_days"] == 30
    assert sent["default_model"] == "gpt-5.4-mini"
    # All four model env vars are different — expect the union sorted
    assert sorted(sent["models"]) == sorted(
        ["gpt-5.4-mini", "gpt-doc-summary", "gpt-image-1.5", "gpt-5.4-nano"]
    )

    # exactly four rows added — agent + doc-summary as NewLLMConfig,
    # image as ImageGenerationConfig, vision as VisionLLMConfig
    llm_rows = [o for o in session._added if isinstance(o, NewLLMConfig)]
    image_rows = [o for o in session._added if isinstance(o, ImageGenerationConfig)]
    vision_rows = [o for o in session._added if isinstance(o, VisionLLMConfig)]
    assert len(llm_rows) == 2
    assert len(image_rows) == 1
    assert len(vision_rows) == 1

    by_name = {r.name: r for r in llm_rows}
    assert by_name[ROW_NAME_AGENT].provider == LiteLLMProvider.OPENAI
    assert by_name[ROW_NAME_AGENT].model_name == "gpt-5.4-mini"
    assert by_name[ROW_NAME_AGENT].api_key == "sk-provisioned-key-1234567890"
    assert by_name[ROW_NAME_AGENT].api_base == "https://litellm.askii.test"
    assert by_name[ROW_NAME_AGENT].search_space_id == 42
    assert by_name[ROW_NAME_AGENT].user_id == user.id

    assert by_name[ROW_NAME_DOC_SUMMARY].model_name == "gpt-doc-summary"
    assert by_name[ROW_NAME_DOC_SUMMARY].api_key == "sk-provisioned-key-1234567890"

    img = image_rows[0]
    assert img.name == ROW_NAME_IMAGE
    assert img.provider == ImageGenProvider.OPENAI
    assert img.model_name == "gpt-image-1.5"
    assert img.api_key == "sk-provisioned-key-1234567890"
    assert img.search_space_id == 42

    vis = vision_rows[0]
    assert vis.name == ROW_NAME_VISION
    assert vis.provider == VisionProvider.OPENAI
    assert vis.model_name == "gpt-5.4-nano"
    assert vis.api_key == "sk-provisioned-key-1234567890"

    # all four FKs wired up to the new row IDs
    assert ss.agent_llm_id == by_name[ROW_NAME_AGENT].id
    assert ss.document_summary_llm_id == by_name[ROW_NAME_DOC_SUMMARY].id
    assert ss.image_generation_config_id == img.id
    assert ss.vision_llm_config_id == vis.id

    # one-shot marker stamped on the user, in the same transaction
    assert user.litellm_auto_provisioned_at is not None
    assert before <= user.litellm_auto_provisioned_at <= datetime.now(UTC)

    # SAVEPOINT was opened for the lock + write block.
    session.begin_nested.assert_called_once()
    # Service does NOT commit — caller (on_after_register / on_after_login)
    # owns the outer transaction commit. See P6 #39 (A1).
    session.commit.assert_not_called()
    # Service does NOT globally rollback either — SAVEPOINT handles
    # local cleanup. See P6 #40 (A2).
    session.rollback.assert_not_called()


async def test_doc_summary_inherits_agent_when_env_blank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Blank ASKII_DOCUMENT_SUMMARY_MODEL ⇒ doc-summary row uses agent model;
    the `models` list sent to Askii has 3 unique entries (agent appears once)."""
    cfg = _set_config(
        monkeypatch,
        ASKII_AGENT_MODEL="gpt-5.4-mini",
        ASKII_DOCUMENT_SUMMARY_MODEL="",  # blank ⇒ inherit agent
        ASKII_IMAGE_GEN_MODEL="gpt-image-1.5",
        ASKII_VISION_MODEL="gpt-5.4-nano",
    )
    session = _make_session()
    user = _FakeUser()
    ss = _FakeSearchSpace(user_id=user.id, ss_id=7)

    recorded: list[httpx.Request] = []

    def handler(req: httpx.Request) -> httpx.Response:
        recorded.append(req)
        return httpx.Response(
            200,
            json={
                "api_key": "sk-foo",
                "key_name": "k1",
                "user_id": "u1",
                "expires": None,
            },
        )

    client = _mock_transport(handler)
    try:
        ok = await ensure_personal_litellm_keys(
            session=session,
            user=user,
            search_space=ss,
            access_token="jwt",
            http_client=client,
            cfg=cfg,
        )
    finally:
        await client.aclose()

    assert ok is True
    sent = json.loads(recorded[0].content)
    assert sorted(sent["models"]) == sorted(
        ["gpt-5.4-mini", "gpt-image-1.5", "gpt-5.4-nano"]
    )
    assert sent["default_model"] == "gpt-5.4-mini"

    llm_rows = {r.name: r for r in session._added if isinstance(r, NewLLMConfig)}
    assert llm_rows[ROW_NAME_AGENT].model_name == "gpt-5.4-mini"
    assert llm_rows[ROW_NAME_DOC_SUMMARY].model_name == "gpt-5.4-mini"


async def test_litellm_base_url_inherits_askii_base_url_when_blank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Blank ASKII_LITELLM_BASE_URL ⇒ api_base on every row falls back to
    ASKII_BASE_URL. Covers the common single-host deployment shape."""
    cfg = _set_config(
        monkeypatch,
        ASKII_BASE_URL="https://api.askii.test",
        ASKII_LITELLM_BASE_URL="",  # explicit blank → fallback
    )
    session = _make_session()
    user = _FakeUser()
    ss = _FakeSearchSpace(user_id=user.id, ss_id=5)

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "api_key": "sk-x",
                "key_name": "k1",
                "user_id": "u1",
                "expires": None,
            },
        )

    client = _mock_transport(handler)
    try:
        ok = await ensure_personal_litellm_keys(
            session=session,
            user=user,
            search_space=ss,
            access_token="jwt",
            http_client=client,
            cfg=cfg,
        )
    finally:
        await client.aclose()

    assert ok is True
    # All 4 rows' api_base should be the ASKII_BASE_URL fallback.
    for row in session._added:
        assert row.api_base == "https://api.askii.test"


async def test_transient_500_returns_false_no_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _set_config(monkeypatch)
    session = _make_session()
    user = _FakeUser()
    ss = _FakeSearchSpace(user_id=user.id)

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"detail": "boom"})

    client = _mock_transport(handler)
    try:
        ok = await ensure_personal_litellm_keys(
            session=session,
            user=user,
            search_space=ss,
            access_token="jwt",
            http_client=client,
            cfg=cfg,
        )
    finally:
        await client.aclose()

    assert ok is False
    session.add.assert_not_called()
    assert ss.agent_llm_id == 0
    assert ss.image_generation_config_id == 0
    assert user.litellm_auto_provisioned_at is None
    session.commit.assert_not_called()


async def test_auth_401_returns_false_no_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _set_config(monkeypatch)
    session = _make_session()
    user = _FakeUser()
    ss = _FakeSearchSpace(user_id=user.id)

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"detail": "bad token"})

    client = _mock_transport(handler)
    try:
        ok = await ensure_personal_litellm_keys(
            session=session,
            user=user,
            search_space=ss,
            access_token="bad-jwt",
            http_client=client,
            cfg=cfg,
        )
    finally:
        await client.aclose()

    assert ok is False
    session.add.assert_not_called()
    assert user.litellm_auto_provisioned_at is None


async def test_validation_422_returns_false_no_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _set_config(monkeypatch)
    session = _make_session()
    user = _FakeUser()
    ss = _FakeSearchSpace(user_id=user.id)

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            422,
            json={
                "detail": [
                    {
                        "loc": ["body", "duration_days"],
                        "msg": "must be ≤ 365",
                        "type": "value_error",
                    }
                ]
            },
        )

    client = _mock_transport(handler)
    try:
        ok = await ensure_personal_litellm_keys(
            session=session,
            user=user,
            search_space=ss,
            access_token="jwt",
            http_client=client,
            cfg=cfg,
        )
    finally:
        await client.aclose()

    assert ok is False
    session.add.assert_not_called()
    assert user.litellm_auto_provisioned_at is None


async def test_db_write_failure_after_provision_returns_false_savepoint_rolls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the upstream Askii call succeeds but the DB write fails, the
    service must catch the error, return False (not raise), and let the
    SAVEPOINT roll back the partial write. The marker must NOT have been
    stamped (atomic with the row insert)."""
    cfg = _set_config(monkeypatch)
    session = _make_session()
    # Override flush so the four-row insert blows up after the SDK call
    # succeeded.
    session.flush = AsyncMock(side_effect=SQLAlchemyError("flush failed"))
    user = _FakeUser()
    ss = _FakeSearchSpace(user_id=user.id, ss_id=9)

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "api_key": "sk-after-flush-explodes",
                "key_name": "k",
                "user_id": "u",
                "expires": None,
            },
        )

    client = _mock_transport(handler)
    try:
        ok = await ensure_personal_litellm_keys(
            session=session,
            user=user,
            search_space=ss,
            access_token="jwt",
            http_client=client,
            cfg=cfg,
        )
    finally:
        await client.aclose()

    assert ok is False
    # SAVEPOINT was opened; its __aexit__ handles the rollback (real
    # SQLAlchemy: ROLLBACK TO SAVEPOINT). Service must NOT call the
    # session-level rollback (would discard caller's pending writes).
    session.begin_nested.assert_called_once()
    session.rollback.assert_not_called()
    session.commit.assert_not_called()
    # FKs must not be wired up when the write failed.
    assert ss.agent_llm_id == 0
    assert ss.document_summary_llm_id == 0
    assert ss.image_generation_config_id == 0
    assert ss.vision_llm_config_id == 0
    # And the one-shot marker stays NULL so the on_after_login retry
    # path can try again.
    assert user.litellm_auto_provisioned_at is None


async def test_race_loss_inside_lock_returns_true_without_writes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the in-lock SELECT FOR UPDATE finds a non-NULL marker (another
    worker stamped it during our Askii call), return True without writing
    rows — the upstream Askii key we just provisioned is orphaned and
    self-expires."""
    cfg = _set_config(monkeypatch)
    user = _FakeUser()  # marker NULL → cheap-check passes
    ss = _FakeSearchSpace(user_id=user.id, ss_id=13)
    session = _make_session(lock_finds_marker=True)

    sdk_called = False

    def handler(req: httpx.Request) -> httpx.Response:
        nonlocal sdk_called
        sdk_called = True
        return httpx.Response(
            200,
            json={
                "api_key": "sk-we-just-wasted",
                "key_name": "orphan-key",
                "user_id": "u",
                "expires": None,
            },
        )

    client = _mock_transport(handler)
    try:
        ok = await ensure_personal_litellm_keys(
            session=session,
            user=user,
            search_space=ss,
            access_token="jwt",
            http_client=client,
            cfg=cfg,
        )
    finally:
        await client.aclose()

    assert ok is True  # caller's perspective: provisioning succeeded
    assert sdk_called is True  # we did pay the Askii cost
    session.add.assert_not_called()  # but we did not insert duplicate rows
    session.commit.assert_not_called()
    # Race-loss path raises _RaceLoss inside the SAVEPOINT; the SAVEPOINT
    # context manager handles rollback. No global session-level rollback.
    session.begin_nested.assert_called_once()
    session.rollback.assert_not_called()
    assert ss.agent_llm_id == 0
    assert ss.document_summary_llm_id == 0
    assert ss.image_generation_config_id == 0
    assert ss.vision_llm_config_id == 0
    # The local user object's marker stays NULL — only the in-DB row of
    # the sibling worker is set; our caller's view is unchanged.
    assert user.litellm_auto_provisioned_at is None


async def test_unexpected_exception_caught_and_savepoint_handles_rollback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Any exception escaping the inner helpers (e.g. lock acquisition
    timeout, unexpected ORM error) must be caught by the outer best-effort
    ``except Exception`` — function returns False, does not raise, and
    does NOT call the session-level rollback (the SAVEPOINT handles its
    own rollback)."""
    cfg = _set_config(monkeypatch)
    user = _FakeUser()
    ss = _FakeSearchSpace(user_id=user.id)

    session = AsyncMock()
    # The lock SELECT FOR UPDATE raises an unexpected error.
    session.execute = AsyncMock(side_effect=RuntimeError("lock timeout"))
    session.add = MagicMock()
    session.flush = AsyncMock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    nested_cm = AsyncMock()
    nested_cm.__aenter__ = AsyncMock(return_value=nested_cm)
    nested_cm.__aexit__ = AsyncMock(return_value=False)
    session.begin_nested = MagicMock(return_value=nested_cm)

    sdk_called = False

    def handler(req: httpx.Request) -> httpx.Response:
        nonlocal sdk_called
        sdk_called = True
        return httpx.Response(
            200,
            json={
                "api_key": "sk-doomed",
                "key_name": "k",
                "user_id": "u",
                "expires": None,
            },
        )

    client = _mock_transport(handler)
    try:
        ok = await ensure_personal_litellm_keys(
            session=session,
            user=user,
            search_space=ss,
            access_token="jwt",
            http_client=client,
            cfg=cfg,
        )
    finally:
        await client.aclose()

    assert ok is False
    assert sdk_called is True  # Askii was reached; the lock SELECT failed
    # SAVEPOINT opened; its __aexit__ rolls back. No session-level
    # rollback (would discard caller's pending writes).
    session.begin_nested.assert_called_once()
    session.rollback.assert_not_called()
    session.add.assert_not_called()
    session.commit.assert_not_called()
    assert user.litellm_auto_provisioned_at is None


# ---------------------------------------------------------------------------
# ensure_personal_litellm_keys_for_user — wrapper
# ---------------------------------------------------------------------------


async def test_wrapper_returns_false_when_feature_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _set_config(monkeypatch, AUTO_PROVISION_LITELLM_KEY=False)
    session = AsyncMock()
    session.execute = AsyncMock()
    user = _FakeUser()

    ok = await ensure_personal_litellm_keys_for_user(
        session=session,
        user=user,
        access_token="jwt",
        cfg=cfg,
    )

    assert ok is False
    session.execute.assert_not_called()


async def test_wrapper_short_circuits_when_marker_set_on_user(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hot path: marker already set in-memory ⇒ no DB I/O, no Askii."""
    cfg = _set_config(monkeypatch)
    session = AsyncMock()
    session.execute = AsyncMock()
    stamped = datetime(2026, 5, 1, tzinfo=UTC)
    user = _FakeUser(provisioned_at=stamped)

    ok = await ensure_personal_litellm_keys_for_user(
        session=session,
        user=user,
        access_token="jwt",
        cfg=cfg,
    )

    assert ok is True
    session.execute.assert_not_called()


async def test_wrapper_observes_marker_set_in_db_after_race(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cold path with fresh-marker re-check: caller's user is stale
    (marker NULL) but the DB row shows non-NULL because a sibling
    transaction stamped it. The wrapper mirrors the value onto the
    caller's object and returns True without any SearchSpace lookup or
    Askii call."""
    cfg = _set_config(monkeypatch)
    user = _FakeUser()  # marker NULL on Python obj
    stamped_in_db = datetime(2026, 5, 1, tzinfo=UTC)

    marker_result = MagicMock()
    marker_result.scalar_one_or_none.return_value = stamped_in_db
    session = AsyncMock()
    session.execute = AsyncMock(return_value=marker_result)

    ok = await ensure_personal_litellm_keys_for_user(
        session=session,
        user=user,
        access_token="jwt",
        cfg=cfg,
    )

    assert ok is True
    # Wrapper mirrored the DB value onto the caller's user object.
    assert user.litellm_auto_provisioned_at == stamped_in_db
    # Exactly one SELECT — the marker re-check. No SearchSpace lookup.
    assert session.execute.await_count == 1


async def test_wrapper_returns_false_when_user_has_no_search_space(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Race so fast that the user row exists but the default SearchSpace
    has not been committed yet. Wrapper returns False; lazy guard retries."""
    cfg = _set_config(monkeypatch)
    user = _FakeUser()

    marker_result = MagicMock()
    marker_result.scalar_one_or_none.return_value = None  # marker still NULL
    ss_result = MagicMock()
    ss_result.scalar_one_or_none.return_value = None  # no SearchSpace yet

    session = AsyncMock()
    session.execute = AsyncMock(side_effect=[marker_result, ss_result])

    ok = await ensure_personal_litellm_keys_for_user(
        session=session,
        user=user,
        access_token="jwt",
        cfg=cfg,
    )

    assert ok is False
    assert session.execute.await_count == 2  # marker + SearchSpace lookups
    assert user.litellm_auto_provisioned_at is None


async def test_wrapper_swallows_unexpected_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Best-effort contract: any SQL error during marker re-check is
    logged and swallowed; wrapper returns False."""
    cfg = _set_config(monkeypatch)
    user = _FakeUser()

    session = AsyncMock()
    session.execute = AsyncMock(side_effect=RuntimeError("conn closed"))

    ok = await ensure_personal_litellm_keys_for_user(
        session=session,
        user=user,
        access_token="jwt",
        cfg=cfg,
    )

    assert ok is False
    assert user.litellm_auto_provisioned_at is None


# ---------------------------------------------------------------------------
# Regression: MissingGreenlet on expired ORM object (best-effort guarantee)
# ---------------------------------------------------------------------------


class _ExpiredUser:
    """User stand-in that raises on every ORM-attribute read.

    Mimics what SQLAlchemy does when async code touches an expired /
    detached object: the sync attribute load attempt fails because no
    greenlet is available to await the implicit SELECT. The error class
    is ``sqlalchemy.exc.MissingGreenlet`` in production; a bare
    ``RuntimeError`` is enough to exercise the swallow path.
    """

    def __init__(self) -> None:
        self._raise = RuntimeError(
            "MissingGreenlet: greenlet_spawn has not been called; can't call await_only()"
        )

    @property
    def id(self) -> uuid.UUID:
        raise self._raise

    @property
    def litellm_auto_provisioned_at(self) -> datetime | None:
        raise self._raise


class _ExpiredSearchSpace:
    """SearchSpace stand-in that raises on every ORM-attribute read."""

    def __init__(self) -> None:
        self._raise = RuntimeError("MissingGreenlet: search_space attribute")

    @property
    def id(self) -> int:
        raise self._raise


async def test_wrapper_does_not_raise_when_user_attribute_load_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression for the production crash: an expired ``user`` ORM object
    must not propagate ``MissingGreenlet`` out of the wrapper.

    Before the fix, ``user.id`` and ``user.litellm_auto_provisioned_at``
    were accessed OUTSIDE the wrapper's try/except, so a sync attribute
    load raised straight through to ``UserManager.on_after_login`` and
    surfaced as a 500. With the fix, every attribute read sits inside
    the try block; any failure is logged and swallowed.
    """
    cfg = _set_config(monkeypatch)
    session = AsyncMock()
    session.execute = AsyncMock()  # never reached — fails at user.id

    ok = await ensure_personal_litellm_keys_for_user(
        session=session,
        user=_ExpiredUser(),
        access_token="jwt",
        cfg=cfg,
    )

    assert ok is False
    # Session never touched — the first guarded read (user.id) raised
    # before any DB I/O could be issued.
    session.execute.assert_not_called()


async def test_core_does_not_raise_when_user_id_capture_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same regression at the core-function layer: ``user.id`` and
    ``search_space.id`` captures must be inside the outer try/except.

    Hit via the direct entry point ``ensure_personal_litellm_keys``
    (used by ``on_after_register`` — which DOES have a valid user, but
    we pin the defensive contract here so a future change to the
    register flow can't silently break the swallow guarantee).
    """
    cfg = _set_config(monkeypatch)
    session = AsyncMock()
    session.execute = AsyncMock()
    session.rollback = AsyncMock()
    session.begin_nested = MagicMock()  # should NOT be entered

    sdk_called = False

    def handler(req: httpx.Request) -> httpx.Response:
        nonlocal sdk_called
        sdk_called = True
        return httpx.Response(200, json={})

    client = _mock_transport(handler)
    try:
        ok = await ensure_personal_litellm_keys(
            session=session,
            user=_ExpiredUser(),
            search_space=_ExpiredSearchSpace(),
            access_token="jwt",
            http_client=client,
            cfg=cfg,
        )
    finally:
        await client.aclose()

    assert ok is False
    # Failed at the first attribute read — Askii never reached.
    assert sdk_called is False
    # Caller's session is untouched: no SAVEPOINT opened (failure was
    # before the lock+write block), no global rollback (would discard
    # caller's pending writes).
    session.begin_nested.assert_not_called()
    session.rollback.assert_not_called()
