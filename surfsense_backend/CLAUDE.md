# SurfSense Backend — Moneta Fork

This file documents Moneta-fork-specific conventions for `surfsense_backend/`.
Repo-wide guidance lives in `/Volumes/data/github/pressingly/CLAUDE.md`;
SurfSense-upstream conventions in `surfsense/.cursor/`.

## Fork migrations

This is a fork of upstream SurfSense. Upstream uses sequential integer Alembic
revisions (`...141`, `142`, `143`, ...). To avoid revision-ID collisions when
pulling upstream merges, **Moneta-fork migrations use the `moneta_NNN` namespace**:

```python
revision: str = "moneta_NNN"
down_revision: str | None = "<previous revision — fork or upstream>"
```

### Adding a new fork migration

1. Find the latest revision in `alembic/versions/` (either upstream integer or
   prior `moneta_*` — whichever is more recent).
2. Pick the next available `moneta_NNN`. Current next: `moneta_002`.
3. Name the file `moneta_NNN_<short_description>.py`.
4. Set `revision = "moneta_NNN"`, `down_revision = "<latest revision found in step 1>"`.

### Rationale

- **Integer collisions impossible** — upstream uses ints; we use prefixed strings.
- **Sortable namespace** — `moneta_001`, `moneta_002`, ... read left-to-right.
- **Self-documenting provenance** — anyone reading `alembic history` sees the fork
  boundary instantly.
- **Zero infrastructure change** — Alembic accepts any string as `revision`.

### When pulling upstream into the fork

If upstream adds new migrations (e.g., upstream goes `143 → 144 → 145`) and a
fork migration `moneta_001` already chains off `143`:

- Upstream `144` chains off `143` (its natural parent).
- Fork `moneta_001` chains off `143` (where the fork branched).

Result: two heads. Resolve with `alembic merge -m "merge upstream Nxx into fork"
moneta_001 <upstream_head>` which generates a `moneta_NNN_merge_*.py` that lists
both as parents. Future fork migrations chain off the merge revision.

## Other backend notes

(Add other Moneta-fork backend conventions here as they emerge.)
