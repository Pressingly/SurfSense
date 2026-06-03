"""Auto-provision personal LiteLLM keys for a SurfSense user via Askii.

A single upstream Askii key (``key_alias="FOSS Server"``) is provisioned
with access to the union of model names a search space needs, then **four**
SurfSense config rows are inserted (one per ``SearchSpace`` FK) so chat,
document summarisation, image generation, and vision-aided document
analysis all work out of the box for SSO users.

One-shot, per user
------------------

Auto-provisioning runs **exactly once per user, ever**. The marker is the
``user.litellm_auto_provisioned_at`` timestamp column:

- ``NULL`` → never provisioned → eligible.
- non-NULL → provisioned at this time → permanently ineligible, regardless
  of whether the four config rows still exist and regardless of which
  ``SearchSpace`` is being viewed.

Two intentional consequences fall out of this:

1. If the user deletes their auto-provisioned config rows from
   ``/llm-configs``, they are **not** silently recreated. The marker is
   sticky.
2. If an already-provisioned user creates a new ``SearchSpace``, the new
   space's ``agent_llm_id`` / ``document_summary_llm_id`` /
   ``image_generation_config_id`` / ``vision_llm_config_id`` FKs are left
   ``NULL``. The user picks existing rows in the UI.

Triggers
--------

Provisioning fires from two ``UserManager`` lifecycle hooks — both
best-effort, never propagating exceptions to the caller:

1. :meth:`app.users.UserManager.on_after_register` — eager path on first
   SSO user creation (has a ``SearchSpace`` in scope from the default-space
   creation that precedes it).
2. :meth:`app.users.UserManager.on_after_login` via the
   :func:`ensure_personal_litellm_keys_for_user` wrapper — steady-state
   path that tolerates a transient failure at registration (the wrapper
   re-SELECTs the marker from the DB, so a race-loser on registration is
   retried transparently on the user's next login).

Transaction ownership (Unit of Work)
------------------------------------

This service does **not** ``commit()`` or globally ``rollback()`` the
caller's session. It uses ``session.begin_nested()`` (SAVEPOINT) to
isolate the lock-and-write block from the caller's outer transaction:

- Race-loss inside the SAVEPOINT: SAVEPOINT rolls back (releasing the row
  lock), caller's pending writes untouched, function returns ``True``.
- Unexpected error inside the SAVEPOINT: SAVEPOINT rolls back, exception
  is caught by the outer best-effort handler, caller's pending writes
  untouched, function returns ``False``.
- Success: SAVEPOINT releases into the outer transaction. The caller
  (``UserManager.on_after_register`` / ``UserManager.on_after_login``)
  ``commit()``s the outer transaction, making provisioning durable.

This matches the SQLAlchemy 2.0 Unit-of-Work guidance: services flush,
callers commit. See review item P6 #56 for the planned
``UserLifecycleService`` that will own session lifecycle end-to-end.

Persistence pattern
-------------------

Both the ``User.litellm_auto_provisioned_at`` marker and the four
``SearchSpace.*_id`` FK assignments are written via explicit
``UPDATE ... WHERE`` statements (not ORM attribute mutation). Reason:
both ``user`` and ``search_space`` are frequently passed in detached from
this service's session (``request.state.proxy_user`` is loaded by
``ProxyAuthMiddleware``'s own session; ``on_after_register`` adds the
search space to its own session before calling here). Attribute mutation
on a detached object is silently dropped at commit time. UPDATE
statements bypass the attached/detached distinction. The Python objects
are mirrored after the UPDATE so the caller's in-request view stays
consistent.

Gating
------

Provisioning is gated on ``AUTH_TYPE=="SSO"`` AND ``AUTO_PROVISION_LITELLM_KEY``
truthy AND the three required model env vars
(``ASKII_AGENT_MODEL`` / ``ASKII_IMAGE_GEN_MODEL`` / ``ASKII_VISION_MODEL``)
non-empty. Those vars default to empty so a half-configured deploy fails
closed at the gate. ``ASKII_BASE_URL`` is also checked for non-empty but
defaults to ``https://api.askii.ai`` (prod) — override to a sandbox or
self-hosted endpoint by setting it explicitly.
``ASKII_DOCUMENT_SUMMARY_MODEL`` is optional — blank inherits the agent
model. ``ASKII_LITELLM_BASE_URL`` is also optional — blank inherits
``ASKII_BASE_URL`` (typical case: the platform API and the LiteLLM proxy
share one host); set it only when they diverge.
"""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING

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
from sqlalchemy import select, update

from app.db import (
    ImageGenerationConfig,
    ImageGenProvider,
    LiteLLMProvider,
    NewLLMConfig,
    SearchSpace,
    User,
    VisionLLMConfig,
    VisionProvider,
)
from app.services.smb_auto_join import find_smb_search_space
from app.utils.rbac import is_search_space_owner

if TYPE_CHECKING:
    import httpx
    from askii.models import ProvisionKeyResponse
    from sqlalchemy.ext.asyncio import AsyncSession
    from starlette.requests import Request

    from app.config import Config

# ``read_mpass_access_token(request)`` below is the one place in this
# module that touches ``starlette.requests.Request``. Service entry
# points take ``access_token: str | None`` instead so they can be invoked
# from non-HTTP contexts (CLI, Celery worker, test) without faking a
# Request. See review item P6 #46.

logger = logging.getLogger(__name__)

# Askii key alias (one upstream key, scoped to all four model slots).
LITELLM_KEY_ALIAS = "FOSS Server"

# Row names — one per SearchSpace FK.
ROW_NAME_AGENT = "FOSS Server - Agent"
ROW_NAME_DOC_SUMMARY = "FOSS Server - Document Summary"
ROW_NAME_IMAGE = "FOSS Server - Image"
ROW_NAME_VISION = "FOSS Server - Vision"

ALL_ROW_NAMES: frozenset[str] = frozenset(
    {ROW_NAME_AGENT, ROW_NAME_DOC_SUMMARY, ROW_NAME_IMAGE, ROW_NAME_VISION}
)


class _RaceLossError(Exception):
    """Internal sentinel raised inside the SAVEPOINT when a sibling
    transaction stamped the marker during our Askii call.

    Raising rolls back the SAVEPOINT (releasing the row lock acquired by
    ``SELECT ... FOR UPDATE``) without touching the caller's outer
    transaction. Caught by the entry function's outer try/except and
    converted to a ``True`` return (one-shot semantics: someone else
    provisioned, the caller's view is consistent).
    """


# ---------------------------------------------------------------------------
# Gate
# ---------------------------------------------------------------------------


def should_auto_provision(cfg: Config) -> bool:
    """Return True iff the feature flag is on AND fully configured.

    All of ``AUTH_TYPE=="SSO"``, ``AUTO_PROVISION_LITELLM_KEY`` truthy, and
    the three required model env vars (agent / image / vision) non-empty
    must hold. ``ASKII_BASE_URL`` is checked for non-empty but defaults to
    prod; override explicitly to point at sandbox / self-hosted. Doc-summary
    and ``ASKII_LITELLM_BASE_URL`` are optional — blank inherits agent model
    and ``ASKII_BASE_URL`` respectively at provisioning time.
    """
    return (
        cfg.AUTH_TYPE == "SSO"
        and bool(cfg.AUTO_PROVISION_LITELLM_KEY)
        and bool(cfg.ASKII_BASE_URL)
        and bool(cfg.ASKII_AGENT_MODEL)
        and bool(cfg.ASKII_IMAGE_GEN_MODEL)
        and bool(cfg.ASKII_VISION_MODEL)
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


async def ensure_personal_litellm_keys(
    *,
    session: AsyncSession,
    user: User,
    search_space: SearchSpace,
    access_token: str | None,
    cfg: Config,
    http_client: httpx.AsyncClient | None = None,
) -> bool:
    """Ensure ``user`` has been auto-provisioned, wiring rows into ``search_space``.

    One-shot per user: returns ``True`` immediately if
    ``user.litellm_auto_provisioned_at`` is non-NULL — irrespective of
    whether the four config rows still exist or which search space the
    caller passed.

    On a successful provision this inserts four rows
    (agent + doc-summary as :class:`NewLLMConfig`, image as
    :class:`ImageGenerationConfig`, vision as :class:`VisionLLMConfig`),
    points the four ``search_space.*_id`` FKs at them, and stamps
    ``user.litellm_auto_provisioned_at``. All writes go through a
    ``session.begin_nested()`` SAVEPOINT so the caller's outer transaction
    is untouched on race-loss or unexpected error. The caller commits the
    outer transaction; a partial failure leaves the marker NULL so the
    next-login retry path will try again.

    Best-effort contract: never raises. Any unexpected error is caught at
    the outer ``try/except`` and returns ``False``; the SAVEPOINT
    context-manager rolls back automatically (releasing any held row
    lock) before the exception reaches the handler.

    Locking: the outbound Askii call happens *without* a DB lock so a slow
    upstream cannot stall every concurrent tab. Inside the SAVEPOINT,
    ``SELECT user.litellm_auto_provisioned_at FOR UPDATE`` locks the user
    row and re-checks the marker — race-loss raises the internal
    ``_RaceLossError`` sentinel, the SAVEPOINT rolls back (releasing the lock),
    and we return ``True``. The orphaned upstream Askii key auto-expires.

    Transaction ownership: this function does NOT commit and does NOT
    rollback the caller's session. See module docstring "Transaction
    ownership" section.

    ``cfg`` is the SurfSense ``Config`` (caller's responsibility — keeps
    this service callable from non-HTTP contexts and decouples it from
    the ``app.config.config`` singleton). ``access_token`` is the mPass
    Cognito access token, ``None`` when the caller is not behind mPass
    (common in dev hitting the backend directly) — service skips with a
    WARN log in that case.

    ``http_client`` is for tests (inject an :class:`httpx.MockTransport`
    backed client); production callers leave it ``None``.
    """
    if not should_auto_provision(cfg):
        return False

    # Best-effort contract: every attribute read on ``user`` / ``search_space``
    # MUST be inside the outer try/except. ORM attribute access on a detached
    # or expired object raises MissingGreenlet in an async context (sync load
    # attempt under asyncpg). The ID captures below are the first reads on
    # the path that could expire the user/search_space ORM object, so they
    # go inside the try; ``user_id`` / ``search_space_id`` stay None if
    # capture itself fails, and the except handler logs "user=None / ss=None"
    # as the operator-visible signal.
    user_id: uuid.UUID | None = None
    search_space_id: int | None = None
    askii_key_name: str | None = None

    try:
        user_id = user.id
        search_space_id = search_space.id

        if user.litellm_auto_provisioned_at is not None:
            return True

        if access_token is None:
            logger.warning(
                "LiteLLM auto-provision skipped: missing mPass access token "
                "for user %s on search_space %s",
                user_id,
                search_space_id,
            )
            return False

        models = _resolve_models(cfg)
        response = await _provision_via_askii(
            access_token=access_token,
            base_url=cfg.ASKII_BASE_URL,
            duration_days=cfg.ASKII_LITELLM_KEY_DURATION_DAYS,
            models=models,
            user_id=user_id,
            http_client=http_client,
        )
        if response is None:
            return False

        askii_key_name = response.key_name
        api_key = response.api_key.get_secret_value()
        api_base = cfg.ASKII_LITELLM_BASE_URL or cfg.ASKII_BASE_URL
        rows = _build_config_rows(
            user_id=user_id,
            search_space_id=search_space_id,
            models=models,
            api_key=api_key,
            api_base=api_base,
        )

        # SAVEPOINT isolates lock + writes from caller's outer transaction.
        # On clean exit: SAVEPOINT releases (writes pending until caller
        # commits). On exception (including _RaceLossError): SAVEPOINT rolls
        # back, caller's outer transaction state unchanged.
        try:
            async with session.begin_nested():
                await _check_marker_under_lock(session, user_id)
                await _persist_and_wire(
                    session=session,
                    user=user,
                    user_id=user_id,
                    search_space=search_space,
                    search_space_id=search_space_id,
                    rows=rows,
                )
        except _RaceLossError:
            logger.debug(
                "LiteLLM auto-provision lost a race for user %s — upstream "
                "Askii key '%s' is orphaned (auto-expires)",
                user_id,
                askii_key_name,
            )
            return True

        logger.info(
            "Auto-provisioned LiteLLM key '%s' (key_name=%s) and 4 config rows "
            "for user %s on search_space %s",
            LITELLM_KEY_ALIAS,
            askii_key_name,
            user_id,
            search_space_id,
        )
        return True

    except Exception:
        # Best-effort contract: swallow anything that escaped the inner
        # helpers. The SAVEPOINT context-manager has already rolled back
        # any in-progress write block, so the caller's outer transaction
        # is clean. Do NOT call session.rollback() — that would discard
        # the caller's pending writes too.
        logger.exception(
            "LiteLLM auto-provision unexpectedly raised for user %s on search_space %s "
            "— swallowed per best-effort contract; retry on next login",
            user_id,
            search_space_id,
        )
        return False


# ---------------------------------------------------------------------------
# Internals — gating + IO helpers
# ---------------------------------------------------------------------------


def read_mpass_access_token(request: Request) -> str | None:
    """Pull the mPass Cognito access token from the ForwardAuth header.

    Returns the stripped token, or ``None`` when the header is missing or
    blank (which means the caller is not going through mPass — common in
    dev when hitting the backend directly).

    Public utility (kept in this module because the access-token concept
    is only useful in conjunction with the Askii provisioning path). HTTP
    callers extract the token via this helper and pass it as
    ``access_token=`` to :func:`ensure_personal_litellm_keys` /
    :func:`ensure_personal_litellm_keys_for_user`.
    """
    token = (request.headers.get("x-auth-request-access-token") or "").strip()
    return token or None


def _resolve_models(cfg: Config) -> tuple[str, str, str, str]:
    """Resolve the four model env vars, applying the doc-summary fallback.

    Returns ``(agent, doc_summary, image, vision)``. ``doc_summary`` falls
    back to ``agent`` when ``ASKII_DOCUMENT_SUMMARY_MODEL`` is blank, so
    operators do not have to repeat the agent model.

    Side effect of the fallback: when blank, ``_build_config_rows`` will
    produce TWO :class:`NewLLMConfig` rows (agent + doc-summary) pointing
    at the same upstream model. This is intentional — the search space
    still needs distinct FK targets for ``agent_llm_id`` and
    ``document_summary_llm_id``, even when both delegate to the same
    model upstream.
    """
    agent = cfg.ASKII_AGENT_MODEL
    return (
        agent,
        cfg.ASKII_DOCUMENT_SUMMARY_MODEL or agent,
        cfg.ASKII_IMAGE_GEN_MODEL,
        cfg.ASKII_VISION_MODEL,
    )


async def _check_marker_under_lock(
    session: AsyncSession,
    user_id: uuid.UUID,
) -> None:
    """Acquire row lock on ``user`` and re-check the one-shot marker.

    Caller MUST run this inside ``session.begin_nested()`` so the lock
    (and any subsequent rollback) is scoped to the SAVEPOINT.

    Raises :class:`_RaceLossError` when a sibling request stamped the marker
    while our Askii call was in flight. The enclosing SAVEPOINT context
    rolls back on the raise, releasing the row lock. Returns ``None``
    when this caller has the lock and the marker is still NULL — caller
    proceeds with the insert.
    """
    result = await session.execute(
        select(User.litellm_auto_provisioned_at)
        .where(User.id == user_id)
        .with_for_update()
    )
    if result.scalar_one_or_none() is not None:
        raise _RaceLossError


def _build_config_rows(
    *,
    user_id: uuid.UUID,
    search_space_id: int,
    models: tuple[str, str, str, str],
    api_key: str,
    api_base: str,
) -> tuple[NewLLMConfig, NewLLMConfig, ImageGenerationConfig, VisionLLMConfig]:
    """Build the four config rows in memory. Pure — no I/O, no FK wiring.

    Takes IDs (not ORM objects) so the caller can safely capture them
    once at entry and pass them down without further attribute reads on
    potentially-detached ORM objects.
    """
    agent_model, doc_summary_model, image_model, vision_model = models
    common = {
        "api_key": api_key,
        "api_base": api_base,
        "litellm_params": {},
        "search_space_id": search_space_id,
        "user_id": user_id,
    }
    llm_common = {
        **common,
        "system_instructions": "",
        "use_default_system_instructions": True,
        "citations_enabled": True,
    }
    agent_row = NewLLMConfig(
        name=ROW_NAME_AGENT,
        description="Auto-provisioned LLM (Askii)",
        provider=LiteLLMProvider.OPENAI,
        model_name=agent_model,
        **llm_common,
    )
    doc_summary_row = NewLLMConfig(
        name=ROW_NAME_DOC_SUMMARY,
        description="Auto-provisioned document-summary LLM (Askii)",
        provider=LiteLLMProvider.OPENAI,
        model_name=doc_summary_model,
        **llm_common,
    )
    image_row = ImageGenerationConfig(
        name=ROW_NAME_IMAGE,
        description="Auto-provisioned image-gen (Askii)",
        provider=ImageGenProvider.OPENAI,
        model_name=image_model,
        **common,
    )
    vision_row = VisionLLMConfig(
        name=ROW_NAME_VISION,
        description="Auto-provisioned vision (Askii)",
        provider=VisionProvider.OPENAI,
        model_name=vision_model,
        **common,
    )
    return agent_row, doc_summary_row, image_row, vision_row


async def _persist_and_wire(
    *,
    session: AsyncSession,
    user: User,
    user_id: uuid.UUID,
    search_space: SearchSpace,
    search_space_id: int,
    rows: tuple[NewLLMConfig, NewLLMConfig, ImageGenerationConfig, VisionLLMConfig],
) -> None:
    """Add the four rows, wire the SearchSpace FKs, stamp the user marker.

    Flush-only (no commit, no rollback). Caller owns transaction
    boundaries. A failure here propagates to the caller's SAVEPOINT
    handler, which rolls back the SAVEPOINT (leaving the outer
    transaction clean) — the marker stays NULL, so the next-login retry
    path will try again.

    Both the marker and the four FKs are written via explicit
    ``UPDATE ... WHERE`` statements rather than ORM attribute mutation.
    Reason: ``user`` is frequently detached from ``session`` (both
    callers — ``ProxyAuthMiddleware`` and ``UserManager.on_after_register``
    — pass a ``User`` loaded by a sibling session). ``search_space`` is
    less consistently detached but the contract should be symmetric so a
    future caller cannot silently break either write by passing a
    detached object.

    The Python objects are mirrored after the UPDATE so the caller's
    in-request view stays consistent for the remainder of the request.
    """
    agent_row, doc_summary_row, image_row, vision_row = rows
    for row in rows:
        session.add(row)
    await session.flush()  # populate .id on each new row

    now = datetime.now(UTC)
    await session.execute(
        update(SearchSpace)
        .where(SearchSpace.id == search_space_id)
        .values(
            agent_llm_id=agent_row.id,
            document_summary_llm_id=doc_summary_row.id,
            image_generation_config_id=image_row.id,
            vision_llm_config_id=vision_row.id,
        )
    )
    await session.execute(
        update(User).where(User.id == user_id).values(litellm_auto_provisioned_at=now)
    )

    # Mirror onto Python objects. NOT load-bearing for persistence — the
    # UPDATE statements above own that. Only purpose: keep the caller's
    # in-memory view consistent with the DB for the remainder of the
    # request, so code that reads e.g. ``search_space.agent_llm_id``
    # after we return sees the wired value rather than the pre-call NULL.
    # Tests also assert against these attributes (a real DB-roundtrip in
    # tests would be heavier than the mirror).
    search_space.agent_llm_id = agent_row.id
    search_space.document_summary_llm_id = doc_summary_row.id
    search_space.image_generation_config_id = image_row.id
    search_space.vision_llm_config_id = vision_row.id
    user.litellm_auto_provisioned_at = now


async def _check_space_marker_under_lock(
    session: AsyncSession,
    search_space_id: int,
) -> None:
    """Acquire a row lock on the org ``SearchSpace`` and re-check its marker.

    SearchSpace-scoped analogue of :func:`_check_marker_under_lock`. Caller
    MUST run this inside ``session.begin_nested()`` so the lock (and any
    rollback) is scoped to the SAVEPOINT.

    Raises :class:`_RaceLossError` when a sibling request stamped
    ``SearchSpace.litellm_auto_provisioned_at`` while our Askii call was in
    flight. Returns ``None`` when this caller holds the lock and the marker is
    still NULL — caller proceeds with the insert.
    """
    result = await session.execute(
        select(SearchSpace.litellm_auto_provisioned_at)
        .where(SearchSpace.id == search_space_id)
        .with_for_update()
    )
    if result.scalar_one_or_none() is not None:
        raise _RaceLossError


async def _persist_and_wire_space(
    *,
    session: AsyncSession,
    search_space: SearchSpace,
    search_space_id: int,
    rows: tuple[NewLLMConfig, NewLLMConfig, ImageGenerationConfig, VisionLLMConfig],
) -> None:
    """Add the four rows, wire the SearchSpace FKs, stamp the space marker.

    SearchSpace-scoped analogue of :func:`_persist_and_wire`. Flush-only (no
    commit, no rollback) — caller owns transaction boundaries. Because the
    one-shot marker lives on the same ``searchspaces`` row as the four FKs, a
    single ``UPDATE`` wires the FKs and stamps the marker together (the
    personal path needs two UPDATEs since its marker is on the ``user`` table).
    Writes go through an explicit ``UPDATE ... WHERE`` rather than ORM attribute
    mutation because ``search_space`` is frequently detached from this
    service's session; the Python object is mirrored afterwards so the caller's
    in-request view stays consistent.
    """
    agent_row, doc_summary_row, image_row, vision_row = rows
    for row in rows:
        session.add(row)
    await session.flush()  # populate .id on each new row

    now = datetime.now(UTC)
    await session.execute(
        update(SearchSpace)
        .where(SearchSpace.id == search_space_id)
        .values(
            agent_llm_id=agent_row.id,
            document_summary_llm_id=doc_summary_row.id,
            image_generation_config_id=image_row.id,
            vision_llm_config_id=vision_row.id,
            litellm_auto_provisioned_at=now,
        )
    )

    # Mirror onto the Python object (not load-bearing — the UPDATE above owns
    # persistence). Keeps the caller's in-memory view consistent and lets tests
    # assert against attributes without a DB round-trip.
    search_space.agent_llm_id = agent_row.id
    search_space.document_summary_llm_id = doc_summary_row.id
    search_space.image_generation_config_id = image_row.id
    search_space.vision_llm_config_id = vision_row.id
    search_space.litellm_auto_provisioned_at = now


# ---------------------------------------------------------------------------
# Upstream Askii call
# ---------------------------------------------------------------------------


async def _provision_via_askii(
    *,
    access_token: str,
    base_url: str,
    duration_days: int,
    models: tuple[str, str, str, str],
    user_id: uuid.UUID,
    http_client: httpx.AsyncClient | None,
) -> ProvisionKeyResponse | None:
    """Call the Askii SDK once, classify any error, return the response or None.

    Auth / validation / not-found errors are non-retryable (logged at WARN
    or ERROR). Rate-limit / 5xx / transport errors are retryable (logged at
    INFO) — the ``on_after_login`` wrapper retry path will try again on
    the user's next authenticated request.

    ``base_url`` is plumbed in explicitly (rather than relying on the SDK's
    ``AskiiConfig.from_env()`` reading ``ASKII_BASE_URL`` independently) so
    the SurfSense config is the single source of truth for the outbound
    Askii endpoint.
    """
    agent_model, doc_summary_model, image_model, vision_model = models
    # ``set`` deduplicates (doc_summary commonly inherits agent → 3 unique
    # names not 4). ``sorted`` is for deterministic test assertions only —
    # Askii treats the model list as unordered; production correctness
    # does NOT depend on order.
    models_for_askii = sorted(
        {agent_model, doc_summary_model, image_model, vision_model}
    )
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
                models=models_for_askii,
                default_model=agent_model,
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
            "on_after_login wrapper will retry on next authenticated request",
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


async def ensure_personal_litellm_keys_for_user(
    *,
    session: AsyncSession,
    user: User,
    access_token: str | None,
    cfg: Config,
) -> bool:
    """High-level entry point: provision against the user's default SearchSpace.

    Convenience wrapper for callers that have a ``user`` but not a specific
    ``SearchSpace`` in scope (``UserManager.on_after_login``,
    ``ProxyAuthMiddleware``). Auto-discovers the user's first SearchSpace
    by id and delegates to :func:`ensure_personal_litellm_keys`.

    Hot path is one Python attribute read + return: when the in-memory
    ``user`` object already carries a non-NULL marker, this exits before
    touching the session. Only requests that may have raced against a
    sibling provision pay the SELECT + SearchSpace lookup; the service's
    own ``SELECT user FOR UPDATE`` serializes any concurrent callers and
    short-circuits the loser via its race-loss branch.

    Best-effort: swallows every exception so callers can wire this into
    request hot paths without explicit guards. Returns ``False`` for any
    skipped or failed outcome (gate off, no default SearchSpace yet,
    transient upstream error). Returns ``True`` only when the user is
    confirmed provisioned.

    ``cfg`` / ``access_token`` semantics: see
    :func:`ensure_personal_litellm_keys`.
    """
    if not should_auto_provision(cfg):
        return False

    # Best-effort contract: the entire body runs inside try/except. ORM
    # attribute reads on a detached or expired ``user`` raise MissingGreenlet
    # in an async context (sync attribute load attempt under asyncpg), so
    # even the cheap fast-path ``user.id`` / ``user.litellm_auto_provisioned_at``
    # reads MUST be guarded — they would otherwise propagate to the caller
    # (``UserManager.on_after_login`` → ``ProxyAuthMiddleware``) and surface
    # as a 500 instead of the documented "swallow + retry on next login"
    # outcome.
    user_id: uuid.UUID | None = None
    try:
        user_id = user.id

        if user.litellm_auto_provisioned_at is not None:
            return True

        # Re-check via SELECT because ``user`` may be detached from
        # ``session`` or carry a stale Python attribute (the marker was
        # committed in a sibling session before this caller loaded ``user``).
        marker_result = await session.execute(
            select(User.litellm_auto_provisioned_at).where(User.id == user_id)
        )
        fresh_marker = marker_result.scalar_one_or_none()
        if fresh_marker is not None:
            user.litellm_auto_provisioned_at = fresh_marker
            return True

        # Oldest owned SearchSpace wins (``order_by(id)`` ascending) so
        # the choice is stable across rename / reorder. This also matches
        # ``on_after_register``'s default "My Search Space" — the very
        # first SS created for any user — so the eager path
        # (on_after_register → ensure_personal_litellm_keys with the
        # just-created SS) and this lazy retry path target the same row.
        ss_result = await session.execute(
            select(SearchSpace)
            .where(SearchSpace.user_id == user_id)
            .order_by(SearchSpace.id)
            .limit(1)
        )
        search_space = ss_result.scalar_one_or_none()
        if search_space is None:
            # ``on_after_register`` hasn't completed (or failed) — there is
            # no SearchSpace yet to wire FKs onto. Retry on next login.
            return False

        return await ensure_personal_litellm_keys(
            session=session,
            user=user,
            search_space=search_space,
            access_token=access_token,
            cfg=cfg,
        )
    except Exception:
        # ``user_id`` may be None if ``user.id`` itself raised (expired ORM
        # object) — log "user=None" so operators can grep for it.
        logger.exception(
            "ensure_personal_litellm_keys_for_user: hook failed for user %s "
            "— will retry on next login",
            user_id,
        )
        return False


async def ensure_org_litellm_keys(
    *,
    session: AsyncSession,
    admin_user: User,
    org_search_space: SearchSpace,
    access_token: str | None,
    cfg: Config,
    http_client: httpx.AsyncClient | None = None,
) -> bool:
    """Auto-provision a fresh Askii key onto the shared Organization space.

    Org-space analogue of :func:`ensure_personal_litellm_keys`. One-shot per
    org space via ``org_search_space.litellm_auto_provisioned_at`` (the
    SearchSpace marker, NOT the per-user one): the first ``is_owner`` admin to
    log in mints one fresh upstream Askii key under their mPass token and wires
    the four ``SearchSpace.*_id`` FKs onto the org space; subsequent
    admins/logins short-circuit on the marker.

    Same shape and guarantees as the personal path: the outbound Askii call
    happens *without* a DB lock; the marker re-check + writes run inside a
    ``session.begin_nested()`` SAVEPOINT with ``SELECT ... FOR UPDATE`` on the
    org ``SearchSpace`` row (race-loss → ``_RaceLossError`` → SAVEPOINT
    rollback → return ``True``, orphaned upstream key auto-expires).

    Best-effort contract: never raises; returns ``False`` on any skip/failure
    so the next-login retry path tries again. Does NOT commit or rollback the
    caller's session (caller owns the commit).

    ``admin_user`` is the org-space owner whose mPass ``access_token`` mints the
    key and who owns the four config rows. ``cfg`` / ``access_token`` /
    ``http_client`` semantics match :func:`ensure_personal_litellm_keys`.
    """
    if not should_auto_provision(cfg):
        return False

    user_id: uuid.UUID | None = None
    search_space_id: int | None = None
    askii_key_name: str | None = None

    try:
        user_id = admin_user.id
        search_space_id = org_search_space.id

        if org_search_space.litellm_auto_provisioned_at is not None:
            return True

        if access_token is None:
            logger.warning(
                "Org LiteLLM auto-provision skipped: missing mPass access token "
                "for admin %s on org search_space %s",
                user_id,
                search_space_id,
            )
            return False

        models = _resolve_models(cfg)
        response = await _provision_via_askii(
            access_token=access_token,
            base_url=cfg.ASKII_BASE_URL,
            duration_days=cfg.ASKII_LITELLM_KEY_DURATION_DAYS,
            models=models,
            user_id=user_id,
            http_client=http_client,
        )
        if response is None:
            return False

        askii_key_name = response.key_name
        api_key = response.api_key.get_secret_value()
        api_base = cfg.ASKII_LITELLM_BASE_URL or cfg.ASKII_BASE_URL
        rows = _build_config_rows(
            user_id=user_id,
            search_space_id=search_space_id,
            models=models,
            api_key=api_key,
            api_base=api_base,
        )

        # SAVEPOINT isolates lock + writes from the caller's outer transaction
        # (see ensure_personal_litellm_keys / module docstring).
        try:
            async with session.begin_nested():
                await _check_space_marker_under_lock(session, search_space_id)
                await _persist_and_wire_space(
                    session=session,
                    search_space=org_search_space,
                    search_space_id=search_space_id,
                    rows=rows,
                )
        except _RaceLossError:
            logger.debug(
                "Org LiteLLM auto-provision lost a race for org search_space %s — "
                "upstream Askii key '%s' is orphaned (auto-expires)",
                search_space_id,
                askii_key_name,
            )
            return True

        logger.info(
            "Auto-provisioned org LiteLLM key '%s' (key_name=%s) and 4 config rows "
            "on org search_space %s by admin %s",
            LITELLM_KEY_ALIAS,
            askii_key_name,
            search_space_id,
            user_id,
        )
        return True

    except Exception:
        logger.exception(
            "Org LiteLLM auto-provision unexpectedly raised for admin %s on org "
            "search_space %s — swallowed per best-effort contract; retry on next login",
            user_id,
            search_space_id,
        )
        return False


async def ensure_org_litellm_keys_for_admin(
    *,
    session: AsyncSession,
    user: User,
    access_token: str | None,
    cfg: Config,
) -> bool:
    """Provision the shared Organization space if ``user`` is its ``is_owner`` admin.

    High-level entry point for :meth:`app.users.UserManager.on_after_login`.
    Locates the shared SMB/Organization ``SearchSpace`` by name
    (:func:`app.services.smb_auto_join.find_smb_search_space`), short-circuits
    on its one-shot marker, gates on ``is_owner`` membership
    (:func:`app.utils.rbac.is_search_space_owner`), then delegates to
    :func:`ensure_org_litellm_keys`.

    Best-effort: swallows every exception so callers can wire this into the
    login hot path without guards. Returns ``False`` for any skipped/failed
    outcome (gate off, no mPass access token, no SMB space configured/created
    yet, caller is not the org-space admin, transient upstream error) and
    ``True`` only when the org space is confirmed provisioned.

    ``cfg`` / ``access_token`` semantics: see
    :func:`ensure_personal_litellm_keys`.
    """
    if not should_auto_provision(cfg):
        return False

    # Provisioning mints an Askii key from the mPass Cognito access token —
    # skip the SMB-space lookup + ownership query entirely when there is no
    # token (non-mPass logins: fastapi-users JWT, Google OAuth, dev hitting the
    # backend directly). The core ensure_org_litellm_keys re-checks this as
    # defense-in-depth; here it just avoids the two DB round-trips. The marker
    # stays NULL, so the admin's next mPass login retries.
    if access_token is None:
        return False

    user_id: uuid.UUID | None = None
    try:
        user_id = user.id

        org_search_space = await find_smb_search_space(session)
        if org_search_space is None:
            # No shared SMB/Organization space configured or created yet —
            # nothing to provision. Retry on the admin's next login.
            return False

        if org_search_space.litellm_auto_provisioned_at is not None:
            return True

        if not await is_search_space_owner(session, user_id, org_search_space.id):
            # Only the is_owner admin provisions the shared key. Any other
            # member is a no-op (not an error).
            return False

        return await ensure_org_litellm_keys(
            session=session,
            admin_user=user,
            org_search_space=org_search_space,
            access_token=access_token,
            cfg=cfg,
        )
    except Exception:
        logger.exception(
            "ensure_org_litellm_keys_for_admin: hook failed for user %s — "
            "will retry on next login",
            user_id,
        )
        return False


__all__ = [
    "ALL_ROW_NAMES",
    "LITELLM_KEY_ALIAS",
    "ROW_NAME_AGENT",
    "ROW_NAME_DOC_SUMMARY",
    "ROW_NAME_IMAGE",
    "ROW_NAME_VISION",
    "ensure_org_litellm_keys",
    "ensure_org_litellm_keys_for_admin",
    "ensure_personal_litellm_keys",
    "ensure_personal_litellm_keys_for_user",
    "read_mpass_access_token",
    "should_auto_provision",
]
