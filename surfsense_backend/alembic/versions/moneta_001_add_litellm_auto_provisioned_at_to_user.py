"""moneta_001_add_litellm_auto_provisioned_at_to_user

Revision ID: moneta_001
Revises: 143
Create Date: 2026-05-25

Fork-migration convention
-------------------------
This is the first Moneta-fork migration. To avoid revision-ID collisions
with upstream SurfSense (which uses sequential integers like ``143``,
``144``, ...), Moneta-fork migrations use the ``moneta_NNN`` namespace.
Next fork migration should be ``moneta_002``, then ``moneta_003``, etc.
See ``surfsense_backend/CLAUDE.md`` § "Fork migrations" for the full
convention.

Adds ``litellm_auto_provisioned_at`` (nullable timestamp) to the ``user``
table. This column is the one-shot marker for the Askii LiteLLM auto-
provisioning service (``app.services.litellm_provisioning``):

- NULL  → user has never been auto-provisioned → eligible.
- non-NULL → user was auto-provisioned once at this time → permanently
  ineligible, regardless of whether the four config rows still exist.

Replaces the previous marker, which was the existence of a ``new_llm_configs``
row named ``"FOSS Server - Agent"``. The old marker silently re-ran
provisioning whenever a user deleted their own config rows or created a
new search space — both unintended.

Post-deploy backfill (run once, BEFORE the new code path goes live, if you
already have auto-provisioned users you do not want re-provisioned):

    UPDATE "user"
       SET litellm_auto_provisioned_at = now()
     WHERE id IN (
         SELECT DISTINCT user_id
           FROM new_llm_configs
          WHERE name = 'FOSS Server - Agent'
       );

Without the backfill, existing auto-provisioned users get one extra
provisioning attempt on their next qualifying request (the new gate sees
NULL and proceeds). That attempt is a no-op upstream (Askii happily issues
another key) and inserts a fresh set of four config rows next to the
ones already present — harmless but wasteful.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "moneta_001"
down_revision: str | None = "143"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    conn = op.get_bind()
    existing_columns = [col["name"] for col in sa.inspect(conn).get_columns("user")]

    if "litellm_auto_provisioned_at" not in existing_columns:
        op.add_column(
            "user",
            sa.Column(
                "litellm_auto_provisioned_at",
                sa.TIMESTAMP(timezone=True),
                nullable=True,
            ),
        )


def downgrade() -> None:
    op.drop_column("user", "litellm_auto_provisioned_at")
