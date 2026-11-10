# Neon PostgreSQL migration (users, credits, orders, history)

Status: **implementation and local/PostgreSQL verification complete; production cutover is blocked pending the Neon connection string and Sanchit's explicit approval.**

No paid infrastructure is used. Neon Free (0.5 GiB storage) is the target; Render stays on the Free plan without a disk.

---

## 1. What changed

| Layer | Before | After |
|---|---|---|
| Storage | One `credits.db` SQLite file per container | `DATABASE_URL` selects PostgreSQL; SQLite remains the local/rollback backend |
| Routes | Raw SQL in `payments.py`, `rate_limiter.py`, `credits_store.py` | Repository functions in `credits_store.py`; routes never open a DB connection |
| Credit deduction | `BEGIN IMMEDIATE` + read/update | One conditional `UPDATE ... WHERE credits_remaining >= ?` (atomic on both engines) |
| Payment claim | SQLite transaction | One PostgreSQL transaction; order row claim + balance update, idempotent under replay |
| Rate limits | SQLite transaction | Transaction + per-device `SELECT ... FOR UPDATE` on PostgreSQL |
| Health | SQLite path only | `storage_backend`, `database_configured`, `database_reachable`, latency, size, connections |
| Backup / rollback | Manual | Verified SQLite backup, merge, PostgreSQL import, SQLite rollback export |

Files:

- `backend/app/storage/database.py` — SQLite/PostgreSQL adapters, pooled connections, schema init, health.
- `backend/app/storage/schema.py` — lockstep schema + column contract.
- `backend/app/utils/credits_store.py` — all user/credit/order/visual repository functions.
- `backend/app/utils/rate_limiter.py` — transactional sliding windows.
- `backend/scripts/db_backup_sqlite.py` — verified backup + portable JSON export.
- `backend/scripts/db_merge_sqlite.py` — root-wins merge, conflict archive, ID remap.
- `backend/scripts/db_migrate_postgres.py` — dry-run/apply/resume/verify importer.
- `backend/scripts/db_rollback_export.py` — repeatable-read PostgreSQL → SQLite snapshot.
- `backend/scripts/neon_usage.py` — storage/connection monitor against the Free limit.
- `.github/workflows/backend-tests.yml` — PostgreSQL 16 integration job, no secrets.

---

## 2. Configuration

Set only in the environment or a gitignored `.env`; never commit it.

```dotenv
# Neon pooled connection string. Keep sslmode=require.
DATABASE_URL=postgresql://USER:PASSWORD@HOST/DB?sslmode=require

DATABASE_POOL_MIN_SIZE=1
DATABASE_POOL_MAX_SIZE=5
DATABASE_CONNECT_TIMEOUT_SECONDS=10
DATABASE_STATEMENT_TIMEOUT_MS=15000
DATABASE_APPLICATION_NAME=snapnote
```

`DATABASE_URL` empty keeps SQLite (`CREDITS_DB_PATH`) for local development and emergency rollback. The URL is never logged; `/health` prints only:

```json
{
  "storage_backend": "postgres",
  "database_configured": true,
  "database_reachable": true,
  "database_latency_ms": 12.4,
  "database_size_bytes": 10858979,
  "database_connections": 2,
  "database_max_connections": 100,
  "database_server_version": "16.2",
  "credits_db_source": "postgres"
}
```

Production Render service (free, no disk) must receive `DATABASE_URL` as a `sync:false` env var. `render.yaml` no longer declares a disk.

---

## 3. Backups (completed before any migration)

Backups are in `backups/pre-migration/` (gitignored; contains PII and password hashes — never commit).

| Source | Rows | SHA-256 |
|---|---|---|
| `credits.db` (root, primary) | 133 devices, 23 users, 37 orders, 388 request logs, 2,671 visuals | `3eb247242204c2bfe68d9ed01a9edaa382fbbafc0800b1ddef8bdaa898c48132` |
| `backend/credits.db` (secondary) | 62 devices, 1 user, 3 orders, 80 request logs, 636 visuals | `517e06f0d57b691cafd11d214271bfa6dfb5568fd561f5eaf97a22735acb7a3e` |

Each backup passed `PRAGMA integrity_check = ok`, a DB-API backup copy, count/sum/fingerprint comparison, and a portable `.json` export. The source files were never modified by the merge/importer.

Merged snapshot (`backups/pre-migration/credits.merged.db`):

| Table | Rows | Fingerprint |
|---|---:|---|
| users | 24 | `0cacd866…d368da` |
| device_credits | 148 | `cf0c86be…f148ce` |
| razorpay_orders | 39 (38 completed, 1 created) | `d214bdf1…e4ee7d` |
| request_log | 439 | `0fef64c6…482893` |
| visual_explanations | 3,307 | `add9ebeb…08219` |

Merge result: 15 backend-only devices, 1 backend-only user, 2 backend-only orders, 51 new request-log rows (29 exact duplicates skipped), 636 backend-only visuals, 1 conflicting order archived. Root won every overlapping value; the conflicting backend order snapshot is in `merge_archive`.

---

## 4. Exact migration procedure

### 4.1 Prepare Neon (free plan only)

1. Create a Neon project in the **same region as the Render service (Virginia / us-east-2)** to keep query latency low.
2. Create the production database (e.g. `snapnote`) and a **separate test branch/database** (e.g. `snapnote_test`). Never run tests against production.
3. Copy the **pooled** connection string. Keep `sslmode=require`.
4. Put only the test URL in `backend/.env` first:

```dotenv
DATABASE_URL=postgresql://.../snapnote_test?sslmode=require
```

### 4.2 Re-run the backup + merge (fresh copies)

```powershell
$env:PYTHONPATH = "backend"
python -m scripts.db_backup_sqlite --out backups\pre-migration
python -m scripts.db_merge_sqlite `
  --primary credits.db `
  --secondary backend\credits.db `
  --out backups\pre-migration\credits.merged.db
```

The merge refuses to overwrite an existing file; use a new `--out` name or move the old one aside. It opens both sources read-only and writes a reconciliation report next to the merged DB.

### 4.3 Prove the test branch

```powershell
$env:PYTHONPATH = "backend"
$env:TEST_DATABASE_URL = "postgresql://.../snapnote_test?sslmode=require"
python -m pytest backend\tests\test_postgres_integration.py -q
python -m scripts.db_migrate_postgres --source backups\pre-migration\credits.merged.db
python -m scripts.db_migrate_postgres --source backups\pre-migration\credits.merged.db --apply
python -m scripts.db_migrate_postgres --source backups\pre-migration\credits.merged.db --verify-only
python -m scripts.neon_usage --json
python -m scripts.db_rollback_export --out backups\rollback\neon-test.db
```

`--apply` imports in one transaction. A non-empty target is refused unless `--mode resume`, which inserts only missing rows and exports the target before writing. `--verify-only` compares every table's row count and SHA-256 fingerprint; it exits `2` on any mismatch. The importer never deletes or truncates.

### 4.4 Production cutover (requires explicit approval)

1. Confirm `/health` before the change: `git_commit`, current uptime, and the test-only data report in section 5.
2. In Render → `snapnote-api` → Environment, add `DATABASE_URL` (pooled production URL, `sync:false`). Keep `JWT_SECRET` as-is.
3. Deploy `master` after the PR merges. The new code initializes the schema and fails startup loudly if PostgreSQL is unreachable.
4. Verify:
   - `/health` → `storage_backend="postgres"`, `database_configured=true`, `database_reachable=true`, `credits_db_source="postgres"`.
   - `python -m scripts.neon_usage` from a trusted machine with the production URL.
   - Create a controlled test device, spend one credit, record the balance.
   - Restart the API (Render dashboard/API), then confirm the same balance.
   - Trigger a redeploy, then confirm the balance again.
   - Confirm an order created before the restart is still present with the same status.
5. Do **not** delete the local `credits.db` files. They remain the recovery source.
6. If verification fails: follow section 6.

### 4.5 Rollback

Rollback is a new, verified SQLite snapshot — not the assumption that an old file is current:

```powershell
$env:PYTHONPATH = "backend"
python -m scripts.db_rollback_export --out backups\rollback\pre-rollback.db
```

Then set:

```dotenv
DATABASE_URL=
CREDITS_DB_PATH=C:\absolute\path\to\pre-rollback.db
```

and redeploy. The export reads PostgreSQL in `REPEATABLE READ, READ ONLY`, writes a new SQLite file, re-runs `integrity_check`, and compares fingerprints. Any writes after the cutover exist only in PostgreSQL; export immediately before switching back.

---

## 5. Production data-risk report (current state)

Verified through the Render API and production logs:

- Service `srv-d9j7n8d8nd3s73avais0` is on the **free plan**, `disk: null`, region Virginia.
- Last deploy ended `2026-11-09T19:10:47Z` (clean rebuild `dep-dbp1knijnfac7395b0og`, commit `50ea02d`). The container filesystem, including `/app/credits.db`, is ephemeral.
- Logs from `2026-11-09T19:10Z` to `2026-11-10T16:05Z` contain 979 entries. Request totals: 3 `/api/auth/device`, 4 `/api/extract/diagram`, 1 `/api/extract/revision`, 2 `/api/extract/visual`, **zero** `/api/auth/signup`, `/api/auth/login`, `/api/auth/google`, `/api/auth/chrome`, and **zero** `/api/payments/*`.
- The only device prefixes visible are `prod-hin` (the Hinglish smoke device) and `bd9d0d8c` (the lazy-Explain-Visually check device). They are test devices created by this work.
- Live `/health` at `2026-11-10` reports commit `50ea02d5586fc5623601203ddc8dda0095bb4a03`, `credits_db_absolute=false`, `credits_db_on_mount=false`, `auth_secret_storage="environment"`, confirming the SQLite file is not durable.

**Conclusion:** the current production SQLite contains no observable genuine student accounts, paid orders, or non-test balances. It appears to contain only our smoke-test device rows. Data from before the last deploy was already lost when the free container was replaced and cannot be recovered from the running service. There is no Render shell/disk on the free plan, and deploying code would restart the container and destroy the current SQLite file before it could be copied. The local root + backend databases (backed up above) are therefore the only recoverable data source and are the data being migrated.

This report is the basis for the approval request; no production deploy or destructive migration has been performed.

---

## 6. Free-plan limits and monitoring

- Neon Free: **0.5 GiB storage**. `scripts/neon_usage.py` warns at 80% and exits non-zero at 90%.
- Connections: pool max 5; `/health` exposes current/max connections. Do not raise the pool above what the plan supports.
- Compute hours: check the Neon console; the script cannot query them.
- Autosuspend: the first request after idle may pay a wake-up latency; the pool connect timeout is 10 s and `/health` reports `database_reachable`.
- Keep `DATABASE_STATEMENT_TIMEOUT_MS=15000` so a stalled query cannot hold a request indefinitely.
- No paid plan, add-on, or billing change is part of this work.

---

## 7. Verification evidence (local PostgreSQL 16 / pgserver)

- SQLite repository contract: 150+ assertions including 10-way concurrent deduction and 6-way order replay.
- PostgreSQL contract: schema init, credit lifecycle, overdraft, 10-way concurrent deduction, 6-way concurrent order completion (exactly one credit), wrong-device rejection, unique email, immutable visuals, rate limiter, health.
- Migration: 24 users, 148 devices, 39 orders, 439 request logs, 3,307 visuals imported; source and target SHA-256 fingerprints identical; resume import inserted 0 rows; rollback export fingerprints identical.
- Neon-specific verification is pending the connection string. The GitHub `postgres-integration` job runs the same tests against real PostgreSQL 16 with no secrets.

---

## 8. Security notes

- `DATABASE_URL` is read from the environment, then `backend/.env`; it is never printed. `safe_dsn()` exposes only host and database.
- Use the Neon pooled URL, `sslmode=require`, and a dedicated database role. Rotate the password if it is ever pasted into a chat, log, or commit.
- `backups/`, `*.db`, `*.db-wal`, `*.db-shm`, and `.env` are gitignored.
- Backups contain password hashes and study notes. Restrict access and delete them only after the new database is verified and recoverable.
