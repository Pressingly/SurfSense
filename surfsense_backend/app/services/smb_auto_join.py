"""Ensure SSO users are members of the shared SMB SearchSpace (SMB_NAME)."""

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
    If ``SMB_NAME`` is set and a matching SearchSpace exists, insert membership
    for ``user_id`` using the default invite role (Editor) when absent.

    Idempotent. Safe when auth uses Bearer JWT only (no ForwardAuth headers):
    ``ProxyAuthMiddleware`` skips resolution for those requests, so this runs
    from ``current_active_user`` as well.
    """
    smb = (getattr(config, "SMB_NAME", None) or "").strip()
    if not smb:
        return

    async with async_session_maker() as session:
        not_deleting = ~SearchSpace.name.startswith("[DELETING] ")
        space_result = await session.execute(
            select(SearchSpace)
            .where(SearchSpace.name == smb, not_deleting)
            .order_by(SearchSpace.id.asc())
            .limit(1)
        )
        space = space_result.scalars().first()
        if space is None:
            logger.debug(
                "SMB auto-join: no search space named %r — skipping",
                smb,
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
