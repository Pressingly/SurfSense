"""Auto-provision personal LiteLLM keys for a SurfSense user via Askii.

Single Askii key (``key_alias="FOSS Server"``) is provisioned with access to
the union of model names a search space needs, then **four** SurfSense
config rows are inserted (one per SearchSpace FK) so chat, document
summarisation, image generation, and vision-aided document analysis all
work out of the box for SSO users.

Fires from two places — single source of truth in
:func:`ensure_personal_litellm_keys`:

1. ``UserManager.on_after_register`` — best-effort on first SSO login.
2. ``GET /searchspaces/{id}`` — lazy guard so AC4 ("one more attempt
   when the user lands on My Space") is satisfied when the login-time
   attempt fails.

Gated on ``AUTH_TYPE=SSO`` AND ``AUTO_PROVISION_LITELLM_KEY=true`` AND the
three required model env vars (``ASKII_AGENT_MODEL`` / ``ASKII_IMAGE_GEN_MODEL``
/ ``ASKII_VISION_MODEL``) being non-empty — those vars default to empty so
a half-configured deploy fails closed at the gate. ``ASKII_BASE_URL`` is
also checked for non-empty by the gate but defaults to ``https://api.askii.ai``
(prod); override to a sandbox or self-hosted endpoint by setting it
explicitly. ``ASKII_DOCUMENT_SUMMARY_MODEL`` is optional — blank inherits
the agent model. ``ASKII_LITELLM_BASE_URL`` is also optional — blank
inherits ``ASKII_BASE_URL`` (typical case: the platform API and the
LiteLLM proxy share one host); set it only when they diverge.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from askii import AskiiConfig, AsyncAskii
from askii._errors import (
    AskiiAuthError,
    AskiiError,
    AskiiNotFoundError,
    AskiiRateLimitError,
    AskiiServerError,
    AskiiTransportError,
    AskiiValidationError,
)
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError

from app.db import (
    ImageGenerationConfig,
    ImageGenProvider,
    LiteLLMProvider,
    NewLLMConfig,
    SearchSpace,
    VisionLLMConfig,
    VisionProvider,
)

if TYPE_CHECKING:
    import httpx
    from askii.models import ProvisionKeyResponse
    from sqlalchemy.ext.asyncio import AsyncSession
    from starlette.requests import Request

    from app.db import User

logger = logging.getLogger(__name__)

# Askii key alias (one upstream key, scoped to all four model slots).
LITELLM_KEY_ALIAS = "FOSS Server"

# Row names — one per SearchSpace FK. The agent row doubles as the dedup
# marker (see ROW_NAME_AGENT below).
ROW_NAME_AGENT = "FOSS Server - Agent"
ROW_NAME_DOC_SUMMARY = "FOSS Server - Document Summary"
ROW_NAME_IMAGE = "FOSS Server - Image"
ROW_NAME_VISION = "FOSS Server - Vision"


def should_auto_provision(cfg: Any) -> bool:
    """Return True iff the feature flag is on AND fully configured.

    All of ``AUTH_TYPE=="SSO"``, ``AUTO_PROVISION_LITELLM_KEY`` truthy, and
    the three required model env vars (agent / image / vision) non-empty
    must hold — those vars default to empty so the feature fails closed
    unless an operator turns it on. ``ASKII_BASE_URL`` is checked for
    non-empty for safety but defaults to prod (``https://api.askii.ai``);
    override explicitly to point at sandbox / self-hosted. Doc-summary
    and ``ASKII_LITELLM_BASE_URL`` are optional — blank inherits agent
    model and ``ASKII_BASE_URL`` respectively at provisioning time.
    """
    return (
        getattr(cfg, "AUTH_TYPE", "") == "SSO"
        and bool(getattr(cfg, "AUTO_PROVISION_LITELLM_KEY", False))
        and bool(getattr(cfg, "ASKII_BASE_URL", ""))
        and bool(getattr(cfg, "ASKII_AGENT_MODEL", ""))
        and bool(getattr(cfg, "ASKII_IMAGE_GEN_MODEL", ""))
        and bool(getattr(cfg, "ASKII_VISION_MODEL", ""))
    )


async def ensure_personal_litellm_keys(
    *,
    session: AsyncSession,
    user: User,
    search_space: SearchSpace,
    request: Request,
    http_client: httpx.AsyncClient | None = None,
) -> bool:
    """Ensure ``user`` has the four 'FOSS Server' config rows for ``search_space``.

    Idempotent: returns ``True`` immediately if the agent marker row already
    exists. On a successful provision, inserts four rows (agent +
    doc-summary as ``NewLLMConfig``, image as ``ImageGenerationConfig``,
    vision as ``VisionLLMConfig``) and points all four
    ``search_space.*_id`` FKs at them. Failures are caught and logged;
    we never raise out of this function.

    ``http_client`` is for tests (inject an ``httpx.MockTransport`` backed
    client); production callers leave it as ``None``.
    """
    from app.config import config

    if not should_auto_provision(config):
        return False

    # Lock the SearchSpace row for the duration of this transaction so two
    # concurrent provisioning attempts (e.g. user opens two tabs during their
    # first My Space load) serialize cleanly. The second waiter then sees
    # the agent marker row committed by the first and short-circuits below,
    # avoiding duplicate Askii keys + duplicate config rows.
    await session.execute(
        select(SearchSpace.id)
        .where(SearchSpace.id == search_space.id)
        .with_for_update()
    )

    existing = await session.execute(
        select(NewLLMConfig).where(
            NewLLMConfig.user_id == user.id,
            NewLLMConfig.search_space_id == search_space.id,
            NewLLMConfig.name == ROW_NAME_AGENT,
        )
    )
    if existing.scalar_one_or_none() is not None:
        return True

    access_token = (request.headers.get("x-auth-request-access-token") or "").strip()
    if not access_token:
        logger.warning(
            "LiteLLM auto-provision skipped: missing X-Auth-Request-Access-Token "
            "for user %s on search_space %s",
            user.id,
            search_space.id,
        )
        return False

    agent_model: str = config.ASKII_AGENT_MODEL
    doc_summary_model: str = config.ASKII_DOCUMENT_SUMMARY_MODEL or agent_model
    image_model: str = config.ASKII_IMAGE_GEN_MODEL
    vision_model: str = config.ASKII_VISION_MODEL
    models_for_askii = sorted(
        {agent_model, doc_summary_model, image_model, vision_model}
    )

    response = await _provision_via_askii(
        access_token=access_token,
        base_url=config.ASKII_BASE_URL,
        duration_days=config.ASKII_LITELLM_KEY_DURATION_DAYS,
        models=models_for_askii,
        default_model=agent_model,
        user_id=user.id,
        http_client=http_client,
    )
    if response is None:
        return False

    api_key = response.api_key.get_secret_value()
    api_base = config.ASKII_LITELLM_BASE_URL or config.ASKII_BASE_URL

    agent_row = NewLLMConfig(
        name=ROW_NAME_AGENT,
        description="Auto-provisioned LLM (Askii)",
        provider=LiteLLMProvider.OPENAI,
        model_name=agent_model,
        api_key=api_key,
        api_base=api_base,
        litellm_params={},
        system_instructions="",
        use_default_system_instructions=True,
        citations_enabled=True,
        search_space_id=search_space.id,
        user_id=user.id,
    )
    doc_summary_row = NewLLMConfig(
        name=ROW_NAME_DOC_SUMMARY,
        description="Auto-provisioned document-summary LLM (Askii)",
        provider=LiteLLMProvider.OPENAI,
        model_name=doc_summary_model,
        api_key=api_key,
        api_base=api_base,
        litellm_params={},
        system_instructions="",
        use_default_system_instructions=True,
        citations_enabled=True,
        search_space_id=search_space.id,
        user_id=user.id,
    )
    image_row = ImageGenerationConfig(
        name=ROW_NAME_IMAGE,
        description="Auto-provisioned image-gen (Askii)",
        provider=ImageGenProvider.OPENAI,
        model_name=image_model,
        api_key=api_key,
        api_base=api_base,
        litellm_params={},
        search_space_id=search_space.id,
        user_id=user.id,
    )
    vision_row = VisionLLMConfig(
        name=ROW_NAME_VISION,
        description="Auto-provisioned vision (Askii)",
        provider=VisionProvider.OPENAI,
        model_name=vision_model,
        api_key=api_key,
        api_base=api_base,
        litellm_params={},
        search_space_id=search_space.id,
        user_id=user.id,
    )

    try:
        for row in (agent_row, doc_summary_row, image_row, vision_row):
            session.add(row)
        await session.flush()  # populate .id on each

        search_space.agent_llm_id = agent_row.id
        search_space.document_summary_llm_id = doc_summary_row.id
        search_space.image_generation_config_id = image_row.id
        search_space.vision_llm_config_id = vision_row.id

        await session.commit()
    except SQLAlchemyError:
        await session.rollback()
        logger.exception(
            "LiteLLM auto-provision DB write failed for user %s on search_space %s "
            "(Askii key already provisioned; lazy guard will retry on next My Space load)",
            user.id,
            search_space.id,
        )
        return False

    logger.info(
        "Auto-provisioned LiteLLM key '%s' (key_name=%s) and 4 config rows "
        "for user %s on search_space %s",
        LITELLM_KEY_ALIAS,
        response.key_name,
        user.id,
        search_space.id,
    )
    return True


async def _provision_via_askii(
    *,
    access_token: str,
    base_url: str,
    duration_days: int,
    models: list[str],
    default_model: str,
    user_id: Any,
    http_client: httpx.AsyncClient | None,
) -> ProvisionKeyResponse | None:
    """Call the SDK once, classify any error, return the response or None.

    Auth / validation / not-found errors are non-retryable (logged at WARN
    or ERROR). Rate-limit / 5xx / transport errors are retryable (logged at
    INFO) — the lazy guard on the next My Space load will try again.

    ``base_url`` is plumbed in explicitly (rather than relying on the SDK's
    ``AskiiConfig.from_env()`` reading ``ASKII_BASE_URL`` independently) so
    the SurfSense config is the single source of truth for the outbound
    Askii endpoint, even if a future refactor moves it off env vars.
    """
    askii_config = AskiiConfig.from_env(base_url=base_url)
    try:
        async with AsyncAskii(
            token=access_token,
            config=askii_config,
            http_client=http_client,
        ) as client:
            return await client.keys.provision(
                key_alias=LITELLM_KEY_ALIAS,
                duration_days=duration_days,
                models=models,
                default_model=default_model,
            )
    except AskiiAuthError as e:
        logger.warning(
            "LiteLLM auto-provision auth-rejected for user %s (status=%s)",
            user_id,
            e.status,
        )
    except AskiiValidationError as e:
        logger.error(
            "LiteLLM auto-provision validation error for user %s (status=%s, field_errors=%s)",
            user_id,
            e.status,
            [(fe.path, fe.type) for fe in e.field_errors],
        )
    except AskiiNotFoundError as e:
        logger.error(
            "LiteLLM auto-provision 404 for user %s (status=%s)",
            user_id,
            e.status,
        )
    except (AskiiRateLimitError, AskiiServerError, AskiiTransportError) as e:
        logger.info(
            "LiteLLM auto-provision transient failure for user %s (%s) — "
            "lazy guard will retry on next My Space load",
            user_id,
            type(e).__name__,
        )
    except AskiiError as e:
        logger.warning(
            "LiteLLM auto-provision unexpected SDK error for user %s: %s",
            user_id,
            type(e).__name__,
        )
    return None


__all__ = [
    "LITELLM_KEY_ALIAS",
    "ROW_NAME_AGENT",
    "ROW_NAME_DOC_SUMMARY",
    "ROW_NAME_IMAGE",
    "ROW_NAME_VISION",
    "ensure_personal_litellm_keys",
    "should_auto_provision",
]
