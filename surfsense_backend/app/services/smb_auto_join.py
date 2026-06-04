"""Ensure SSO users are members of the shared SMB SearchSpace (SMB_DEFAULT_WORKSPACE_NAME)."""

from __future__ import annotations

import logging
import uuid

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from app.config import config
from app.db import (
    SearchSpace,
    SearchSpaceMembership,
    SearchSpaceRole,
    async_session_maker,
)

logger = logging.getLogger(__name__)


def _smb_workspace_slug() -> str:
    """The configured shared SMB/Organization workspace name, or "".

    Priority: ``SMB_DEFAULT_WORKSPACE_NAME`` then ``SMB_NAME``. Stripped;
    empty when neither is set (callers treat that as "no shared space").
    """
    slug = (
        getattr(config, "SMB_DEFAULT_WORKSPACE_NAME", None)
        or getattr(config, "SMB_NAME", "")
        or ""
    )
    return slug.strip() if isinstance(slug, str) else ""


async def find_smb_search_space(session) -> SearchSpace | None:
    """Return the shared SMB/Organization ``SearchSpace`` by name, or None.

    Matches the space whose ``name`` equals :func:`_smb_workspace_slug`,
    skipping ``"[DELETING] "`` soft-delete tombstones and preferring the
    oldest by id. Returns None when no name is configured or no space
    matches. Uses the caller's ``session`` so it can run inside an existing
    request-scoped transaction (e.g. the login hook) as well as the
    standalone session opened by :func:`auto_join_smb_search_space`.
    """
    slug = _smb_workspace_slug()
    if not slug:
        return None

    not_deleting = ~SearchSpace.name.startswith("[DELETING] ")
    result = await session.execute(
        select(SearchSpace)
        .where(SearchSpace.name == slug, not_deleting)
        .order_by(SearchSpace.id.asc())
        .limit(1)
    )
    return result.scalars().first()


async def auto_join_smb_search_space(user_id: uuid.UUID) -> None:
    """
    ``_auto_join_workspace``: match the search space whose
    name equals ``SMB_DEFAULT_WORKSPACE_NAME`` or ``SMB_NAME``. If none exists,
    do nothing (no fallback to another space).

    Uses the default invite role (Editor) when absent. Idempotent.
    Safe when auth uses Bearer JWT only: runs from ``current_active_user`` as well.
    """
    smb_slug = _smb_workspace_slug()
    if not smb_slug:
        return

    async with async_session_maker() as session:
        space = await find_smb_search_space(session)
        if space is None:
            logger.debug(
                "SMB auto-join: no search space named %r — skipping",
                smb_slug,
            )
            return

        existing_member = await session.execute(
            select(SearchSpaceMembership.id).where(
                SearchSpaceMembership.user_id == user_id,
                SearchSpaceMembership.search_space_id == space.id,
            )
        )
        if existing_member.scalars().first() is not None:
            return

        role_result = await session.execute(
            select(SearchSpaceRole).where(
                SearchSpaceRole.search_space_id == space.id,
                SearchSpaceRole.is_default == True,  # noqa: E712
            )
        )
        role = role_result.scalars().first()
        if role is None:
            role_result = await session.execute(
                select(SearchSpaceRole).where(
                    SearchSpaceRole.search_space_id == space.id,
                    SearchSpaceRole.name == "Editor",
                )
            )
            role = role_result.scalars().first()
        if role is None:
            logger.warning(
                "SMB auto-join: no default or Editor role for search space %s — skipping",
                space.id,
            )
            return

        membership = SearchSpaceMembership(
            user_id=user_id,
            search_space_id=space.id,
            role_id=role.id,
            is_owner=False,
        )
        session.add(membership)
        try:
            await session.commit()
            logger.info(
                "SMB auto-join: joined user %s to search space %s (role=%s)",
                user_id,
                space.id,
                role.name,
            )
        except IntegrityError:
            await session.rollback()
            logger.debug(
                "SMB auto-join: race for user %s / space %s — already joined",
                user_id,
                space.id,
            )
