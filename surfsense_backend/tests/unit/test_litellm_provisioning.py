"""Unit tests for ``app.services.litellm_provisioning``.

The service does two things we care about:

1. It calls the Askii SDK exactly once per successful provision attempt
   (with `models=union` and `default_model=agent_model`), and not at all
   when the feature flag is off / token is missing / the agent marker
   row already exists.
2. On a successful Askii response it inserts FOUR config rows —
   `NewLLMConfig` for agent + doc-summary, `ImageGenerationConfig` for
   image gen, `VisionLLMConfig` for vision — and updates all four
   `SearchSpace` FKs in one transaction.

DB writes are simulated with `unittest.mock.AsyncMock` for the session and
a small `_FakeSearchSpace` so we can observe FK mutations directly. The
Askii SDK is exercised end-to-end via `httpx.MockTransport`.
"""

from __future__ import annotations

import json
import uuid
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
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
    LITELLM_KEY_ALIAS,
    ROW_NAME_AGENT,
    ROW_NAME_DOC_SUMMARY,
    ROW_NAME_IMAGE,
    ROW_NAME_VISION,
    ensure_personal_litellm_keys,
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


def _make_user() -> Any:
    user = MagicMock()
    user.id = uuid.uuid4()
    return user


class _FakeSearchSpace:
    """Minimal stand-in for a SearchSpace row.

    A `MagicMock` would silently swallow every attribute access, so we use a
    real class to make FK assertions meaningful.
    """

    def __init__(self, *, user_id: uuid.UUID, ss_id: int = 1) -> None:
        self.id = ss_id
        self.user_id = user_id
        self.agent_llm_id = 0
        self.document_summary_llm_id = 0
        self.image_generation_config_id = 0
        self.vision_llm_config_id = 0


def _make_session(*, existing_row: NewLLMConfig | None) -> AsyncMock:
    """An AsyncMock SQLAlchemy session.

    Returns the same mock result for every `.execute()` — fine because the
    service makes exactly one SELECT before its inserts. The result's
    `scalar_one_or_none()` returns ``existing_row`` so idempotency-on /
    idempotency-off paths can be controlled per test.
    """
    session = AsyncMock()
    select_result = MagicMock()
    select_result.scalar_one_or_none.return_value = existing_row
    session.execute = AsyncMock(return_value=select_result)

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
    session._added = added  # type: ignore[attr-defined]
    return session


def _mock_transport(handler: Any) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        base_url="https://api.askii.test",
        transport=httpx.MockTransport(handler),
    )


def _set_config(monkeypatch: pytest.MonkeyPatch, **overrides: Any) -> None:
    """Patch the shared config singleton with the values needed for each test."""
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


# ---------------------------------------------------------------------------
# ensure_personal_litellm_keys — gate + early returns
# ---------------------------------------------------------------------------


async def test_returns_false_when_feature_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_config(monkeypatch, AUTO_PROVISION_LITELLM_KEY=False)
    session = _make_session(existing_row=None)
    user = _make_user()
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
            request=_make_request(access_token="jwt"),
            http_client=client,
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


async def test_returns_false_when_auth_type_not_sso(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_config(monkeypatch, AUTH_TYPE="GOOGLE")
    session = _make_session(existing_row=None)
    user = _make_user()
    ss = _FakeSearchSpace(user_id=user.id)

    ok = await ensure_personal_litellm_keys(
        session=session,
        user=user,
        search_space=ss,
        request=_make_request(access_token="jwt"),
        http_client=_mock_transport(lambda r: httpx.Response(200, json={})),
    )
    assert ok is False
    session.execute.assert_not_called()


async def test_skips_when_access_token_header_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_config(monkeypatch)
    session = _make_session(existing_row=None)
    user = _make_user()
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
            request=_make_request(access_token=None),
            http_client=client,
        )
    finally:
        await client.aclose()

    assert ok is False
    assert sdk_called is False
    session.add.assert_not_called()


async def test_idempotent_when_marker_row_exists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_config(monkeypatch)
    existing = NewLLMConfig(
        name=ROW_NAME_AGENT,
        provider=LiteLLMProvider.OPENAI,
        model_name="gpt-5.4-mini",
        api_key="sk-existing",
        search_space_id=1,
        user_id=uuid.uuid4(),
    )
    session = _make_session(existing_row=existing)
    user = _make_user()
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
            request=_make_request(access_token="jwt"),
            http_client=client,
        )
    finally:
        await client.aclose()

    assert ok is True
    assert sdk_called is False
    session.add.assert_not_called()
    assert ss.agent_llm_id == 0  # not touched on idempotent path


# ---------------------------------------------------------------------------
# ensure_personal_litellm_keys — full 4-row provisioning flow
# ---------------------------------------------------------------------------


async def test_happy_path_inserts_four_rows_and_links_all_fks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_config(
        monkeypatch,
        ASKII_BASE_URL="https://api.askii.test",
        ASKII_LITELLM_BASE_URL="https://litellm.askii.test",  # explicit override
        ASKII_AGENT_MODEL="gpt-5.4-mini",
        ASKII_DOCUMENT_SUMMARY_MODEL="gpt-doc-summary",  # explicit, different model
        ASKII_IMAGE_GEN_MODEL="gpt-image-1.5",
        ASKII_VISION_MODEL="gpt-5.4-nano",
        ASKII_LITELLM_KEY_DURATION_DAYS=30,
    )
    session = _make_session(existing_row=None)
    user = _make_user()
    ss = _FakeSearchSpace(user_id=user.id, ss_id=42)

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
            request=_make_request(access_token="cognito-jwt"),
            http_client=client,
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

    session.commit.assert_awaited()


async def test_doc_summary_inherits_agent_when_env_blank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Blank ASKII_DOCUMENT_SUMMARY_MODEL ⇒ doc-summary row uses agent model;
    the `models` list sent to Askii has 3 unique entries (agent appears once)."""
    _set_config(
        monkeypatch,
        ASKII_AGENT_MODEL="gpt-5.4-mini",
        ASKII_DOCUMENT_SUMMARY_MODEL="",  # blank ⇒ inherit agent
        ASKII_IMAGE_GEN_MODEL="gpt-image-1.5",
        ASKII_VISION_MODEL="gpt-5.4-nano",
    )
    session = _make_session(existing_row=None)
    user = _make_user()
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
            request=_make_request(access_token="jwt"),
            http_client=client,
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
    _set_config(
        monkeypatch,
        ASKII_BASE_URL="https://api.askii.test",
        ASKII_LITELLM_BASE_URL="",  # explicit blank → fallback
    )
    session = _make_session(existing_row=None)
    user = _make_user()
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
            request=_make_request(access_token="jwt"),
            http_client=client,
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
    _set_config(monkeypatch)
    session = _make_session(existing_row=None)
    user = _make_user()
    ss = _FakeSearchSpace(user_id=user.id)

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"detail": "boom"})

    client = _mock_transport(handler)
    try:
        ok = await ensure_personal_litellm_keys(
            session=session,
            user=user,
            search_space=ss,
            request=_make_request(access_token="jwt"),
            http_client=client,
        )
    finally:
        await client.aclose()

    assert ok is False
    session.add.assert_not_called()
    assert ss.agent_llm_id == 0
    assert ss.image_generation_config_id == 0
    session.commit.assert_not_called()


async def test_auth_401_returns_false_no_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_config(monkeypatch)
    session = _make_session(existing_row=None)
    user = _make_user()
    ss = _FakeSearchSpace(user_id=user.id)

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"detail": "bad token"})

    client = _mock_transport(handler)
    try:
        ok = await ensure_personal_litellm_keys(
            session=session,
            user=user,
            search_space=ss,
            request=_make_request(access_token="bad-jwt"),
            http_client=client,
        )
    finally:
        await client.aclose()

    assert ok is False
    session.add.assert_not_called()


async def test_validation_422_returns_false_no_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_config(monkeypatch)
    session = _make_session(existing_row=None)
    user = _make_user()
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
            request=_make_request(access_token="jwt"),
            http_client=client,
        )
    finally:
        await client.aclose()

    assert ok is False
    session.add.assert_not_called()
