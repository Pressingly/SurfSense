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


async def auto_join_smb_search_space(user_id: uuid.UUID) -> None:
    """
    Same resolution as Plane ``_auto_join_workspace``: match the search space whose
    name equals ``SMB_DEFAULT_WORKSPACE_NAME`` or ``SMB_NAME``. If none exists,
    do nothing (no fallback to another space).

    Uses the default invite role (Editor) when absent. Idempotent.
    Safe when auth uses Bearer JWT only: runs from ``current_active_user`` as well.
    """
    smb_slug = (
        getattr(config, "SMB_DEFAULT_WORKSPACE_NAME", None)
        or getattr(config, "SMB_NAME", "")
        or ""
    )
    if isinstance(smb_slug, str):
        smb_slug = smb_slug.strip()
    if not smb_slug:
        return

    async with async_session_maker() as session:
        not_deleting = ~SearchSpace.name.startswith("[DELETING] ")
        space_result = await session.execute(
            select(SearchSpace)
            .where(SearchSpace.name == smb_slug, not_deleting)
            .order_by(SearchSpace.id.asc())
            .limit(1)
        )
        space = space_result.scalars().first()
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
