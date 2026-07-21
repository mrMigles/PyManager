# pgdb.py
# -*- coding: utf-8 -*-
"""
PostgreSQL provisioning module for PyManager.

Provides idempotent database/role provisioning per app-id and env-var helpers.
All Postgres logic is isolated here so main.py stays free of DB concerns.

psycopg is imported lazily so that tests and environments without Postgres can
import this module without errors.

Environment variables read at module load time:
  PG_ADMIN_HOST      (default: "postgres")
  PG_ADMIN_PORT      (default: 5432)
  PG_ADMIN_USER      (default: "postgres")
  PG_ADMIN_PASSWORD  (required to enable Postgres; feature disabled when absent)
  PG_ADMIN_DB        (default: "postgres")
  PG_APP_HOST        (default: same as PG_ADMIN_HOST; injected into apps)
  PG_APP_PORT        (default: same as PG_ADMIN_PORT; injected into apps)
"""

import logging
import os
import re
import secrets
from typing import Dict, Iterable, Optional

# psycopg is an optional runtime dependency.  The import is attempted at module
# load time so tests can monkeypatch pgdb._psycopg / pgdb._psycopg_sql without
# fighting Python's module import cache.
try:
    import psycopg as _psycopg        # type: ignore
    import psycopg.sql as _psycopg_sql  # type: ignore
except ImportError:
    _psycopg = None        # type: ignore
    _psycopg_sql = None    # type: ignore

logger = logging.getLogger("script-bot.pgdb")

# ---------------------------------------------------------------------------
# Module-level config (read once from env)
# ---------------------------------------------------------------------------

_ADMIN_HOST: str = os.getenv("PG_ADMIN_HOST", "postgres")
_ADMIN_PORT: int = int(os.getenv("PG_ADMIN_PORT", "5432"))
_ADMIN_USER: str = os.getenv("PG_ADMIN_USER", "postgres")
_ADMIN_PASSWORD: Optional[str] = os.getenv("PG_ADMIN_PASSWORD")
_ADMIN_DB: str = os.getenv("PG_ADMIN_DB", "postgres")

_APP_HOST: str = os.getenv("PG_APP_HOST", _ADMIN_HOST)
_APP_PORT: int = int(os.getenv("PG_APP_PORT", str(_ADMIN_PORT)))

_REGISTRY_SCHEMA = "pymanager"
_REGISTRY_TABLE = f"{_REGISTRY_SCHEMA}.registry"

_CREATE_SCHEMA_SQL = f"CREATE SCHEMA IF NOT EXISTS {_REGISTRY_SCHEMA}"
_CREATE_TABLE_SQL = f"""
CREATE TABLE IF NOT EXISTS {_REGISTRY_TABLE} (
    app_id      text        PRIMARY KEY,
    db_name     text        NOT NULL,
    username    text        NOT NULL,
    password    text        NOT NULL,
    created_at  timestamptz NOT NULL DEFAULT now()
)
"""


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def pg_enabled() -> bool:
    """Return True when Postgres admin credentials are configured."""
    return bool(_ADMIN_PASSWORD)


def _sanitize_identifier(name: str) -> str:
    """Derive a safe lowercase Postgres identifier from an app id.

    Result is at most 32 characters, begins with a letter, and contains only
    [a-z0-9_].  Prefix 'app_' is prepended when the cleaned name starts with
    a digit or is empty.
    """
    clean = re.sub(r"[^a-z0-9_]", "_", name.lower())
    clean = clean.strip("_")
    if not clean or clean[0].isdigit():
        clean = "app_" + clean
    return clean[:32]


def _admin_conn(dbname: Optional[str] = None):
    """Open a synchronous psycopg connection to the admin database."""
    return _psycopg.connect(
        host=_ADMIN_HOST,
        port=_ADMIN_PORT,
        user=_ADMIN_USER,
        password=_ADMIN_PASSWORD,
        dbname=dbname or _ADMIN_DB,
        autocommit=True,
    )


def _ensure_registry() -> None:
    """Create the pymanager schema and registry table if they do not exist."""
    with _admin_conn() as conn:
        conn.execute(_CREATE_SCHEMA_SQL)
        conn.execute(_CREATE_TABLE_SQL)
    logger.debug("Registry table ensured")


def get_app_database(app_id: str) -> Optional[Dict[str, str]]:
    """Return the registry row for *app_id*, or None if not provisioned."""
    if not pg_enabled():
        return None
    try:
        with _admin_conn() as conn:
            row = conn.execute(
                f"SELECT app_id, db_name, username, password"
                f" FROM {_REGISTRY_TABLE} WHERE app_id = %s",
                (app_id,),
            ).fetchone()
        if row is None:
            return None
        return {
            "app_id": row[0],
            "db_name": row[1],
            "username": row[2],
            "password": row[3],
        }
    except Exception:
        logger.exception("Failed to query registry for app %s", app_id)
        return None


def get_database_observability(app_ids: Iterable[str]) -> Dict[str, Dict[str, float]]:
    """Return size and average row read/write rates for provisioned app DBs.

    Rates are calculated from ``pg_stat_database`` counters since their last
    reset, falling back to the PostgreSQL server start time.  Missing registry
    data or an unavailable PostgreSQL server produces an empty result so the
    main process observability view remains usable.
    """
    requested_ids = list(dict.fromkeys(app_ids))
    if not pg_enabled() or not requested_ids:
        return {}

    try:
        with _admin_conn() as conn:
            rows = conn.execute(
                f"SELECT r.app_id, pg_database_size(r.db_name::name),"
                " COALESCE(s.tup_returned, 0),"
                " COALESCE(s.tup_inserted, 0)"
                "   + COALESCE(s.tup_updated, 0)"
                "   + COALESCE(s.tup_deleted, 0),"
                " GREATEST(EXTRACT(EPOCH FROM (clock_timestamp() -"
                "   COALESCE(s.stats_reset, pg_postmaster_start_time()))), 1)"
                f" FROM {_REGISTRY_TABLE} AS r"
                " LEFT JOIN pg_stat_database AS s ON s.datname = r.db_name"
                " WHERE r.app_id = ANY(%s::text[])",
                (requested_ids,),
            ).fetchall()
    except Exception:
        logger.exception("Failed to collect PostgreSQL observability metrics")
        return {}

    metrics: Dict[str, Dict[str, float]] = {}
    for app_id, size_bytes, rows_read, rows_written, elapsed_seconds in rows:
        elapsed = max(float(elapsed_seconds), 1.0)
        metrics[app_id] = {
            "size_bytes": float(size_bytes),
            "read_rows_per_second": float(rows_read) / elapsed,
            "write_rows_per_second": float(rows_written) / elapsed,
        }
    return metrics


def provision_database(app_id: str) -> Dict[str, str]:
    """Idempotently provision a Postgres database + role for *app_id*.

    If a registry row already exists the existing credentials are returned
    immediately without touching Postgres.  Otherwise:
      1. CREATE ROLE <username> LOGIN PASSWORD ...
      2. CREATE DATABASE <db_name> OWNER <username>
      3. Connect to the new database and:
         - GRANT ALL PRIVILEGES ON DATABASE <db_name> TO <username>
         - GRANT ALL ON SCHEMA public TO <username>
         - CREATE EXTENSION IF NOT EXISTS vector
      4. Insert a row into pymanager.registry

    All DDL is executed with autocommit=True because CREATE DATABASE cannot
    run inside a transaction block.

    Returns a dict with keys: app_id, db_name, username, password.
    Raises RuntimeError if pg_enabled() is False.
    """
    if not pg_enabled():
        raise RuntimeError("Postgres is not configured (PG_ADMIN_PASSWORD not set)")

    existing = get_app_database(app_id)
    if existing:
        logger.info("Database already provisioned for app %s", app_id)
        return existing

    sql = _psycopg_sql
    _ensure_registry()

    base = _sanitize_identifier(app_id)
    db_name = base
    username = base
    password = secrets.token_urlsafe(24)

    logger.info("Provisioning database '%s' with role '%s' for app %s", db_name, username, app_id)

    with _admin_conn() as conn:
        # CREATE ROLE (idempotent guard — role may already exist from a
        # previous partial run that crashed before writing the registry row).
        conn.execute(
            sql.SQL("CREATE ROLE {role} LOGIN PASSWORD {pwd}").format(
                role=sql.Identifier(username),
                pwd=sql.Literal(password),
            )
        )
        conn.execute(
            sql.SQL("CREATE DATABASE {db} OWNER {role}").format(
                db=sql.Identifier(db_name),
                role=sql.Identifier(username),
            )
        )
        conn.execute(
            sql.SQL("GRANT ALL PRIVILEGES ON DATABASE {db} TO {role}").format(
                db=sql.Identifier(db_name),
                role=sql.Identifier(username),
            )
        )

    # Connect to the new database to enable the vector extension and grant
    # public schema privileges.
    with _admin_conn(dbname=db_name) as conn:
        conn.execute(
            sql.SQL("GRANT ALL ON SCHEMA public TO {role}").format(
                role=sql.Identifier(username),
            )
        )
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # Write registry row.
    with _admin_conn() as conn:
        conn.execute(
            f"INSERT INTO {_REGISTRY_TABLE} (app_id, db_name, username, password)"
            " VALUES (%s, %s, %s, %s)"
            " ON CONFLICT (app_id) DO NOTHING",
            (app_id, db_name, username, password),
        )

    logger.info("Provisioning complete for app %s", app_id)
    return {"app_id": app_id, "db_name": db_name, "username": username, "password": password}


def app_pg_env(app_id: str) -> Dict[str, str]:
    """Return the PG_* env vars to inject for *app_id*.

    Returns an empty dict when Postgres is not configured or the app has no
    provisioned database.
    """
    row = get_app_database(app_id)
    if not row:
        return {}
    return {
        "PG_USERNAME": row["username"],
        "PG_PASSWORD": row["password"],
        "PG_DATABASE": row["db_name"],
        "PG_HOST": _APP_HOST,
        "PG_PORT": str(_APP_PORT),
    }
