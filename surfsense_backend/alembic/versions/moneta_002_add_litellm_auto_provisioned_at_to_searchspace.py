"""moneta_002_add_litellm_auto_provisioned_at_to_searchspace

Revision ID: moneta_002
Revises: moneta_001
Create Date: 2026-06-02

Fork-migration convention
-------------------------
Moneta-fork migration (``moneta_NNN`` namespace) — see
``surfsense_backend/CLAUDE.md`` § "Fork migrations". Chains off ``moneta_001``.
Next fork migration should be ``moneta_003``.

Adds ``litellm_auto_provisioned_at`` (nullable timestamp) to the
``searchspaces`` table. This is the one-shot marker for **org-space** Askii
LiteLLM auto-provisioning (``app.services.litellm_provisioning``): when the
``is_owner`` admin of the shared SMB/Organization space logs in, a fresh Askii
key is minted and the four ``SearchSpace.*_id`` LLM FKs are wired onto that
space — exactly once.

- NULL  → org space has never been auto-provisioned → eligible.
- non-NULL → provisioned once at this time → permanently ineligible,
  regardless of whether the four config rows still exist.

It is the ``SearchSpace``-scoped analogue of the user-scoped
``user.litellm_auto_provisioned_at`` marker added in ``moneta_001``.

No backfill: org-space auto-provisioning did not exist before this migration,
so there are no already-provisioned org spaces to stamp.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "moneta_002"
down_revision: str | None = "moneta_001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    conn = op.get_bind()
    existing_columns = [
        col["name"] for col in sa.inspect(conn).get_columns("searchspaces")
    ]

    if "litellm_auto_provisioned_at" not in existing_columns:
        op.add_column(
            "searchspaces",
            sa.Column(
                "litellm_auto_provisioned_at",
                sa.TIMESTAMP(timezone=True),
                nullable=True,
            ),
        )


def downgrade() -> None:
    conn = op.get_bind()
    existing_columns = [
        col["name"] for col in sa.inspect(conn).get_columns("searchspaces")
    ]

    if "litellm_auto_provisioned_at" in existing_columns:
        op.drop_column("searchspaces", "litellm_auto_provisioned_at")
