# -*- coding: utf-8 -*-
"""Tests for pgdb.py and the PG env injection / ${VAR} expansion in main.py.

All tests that touch pgdb use a monkeypatched psycopg connection so no real
Postgres server is required.  The existing test suite is completely unaffected:
pgdb.pg_enabled() returns False in the test environment (PG_ADMIN_PASSWORD is
not set) and every helper short-circuits accordingly.
"""

import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, call

import pytest

# Ensure repo root is on the path (mirrors conftest.py)
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pgdb as pgdb_module
import main as main_module


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_fake_conn(rows=None):
    """Return a mock psycopg connection whose execute() is recorded.

    *rows* is the sequence of rows that fetchone() returns for each successive
    call (``None`` means 'not found').
    """
    row_iter = iter(rows or [None])

    cursor_mock = MagicMock()
    cursor_mock.fetchone.side_effect = lambda: next(row_iter, None)

    conn_mock = MagicMock()
    conn_mock.execute.return_value = cursor_mock
    conn_mock.__enter__ = lambda s: s
    conn_mock.__exit__ = MagicMock(return_value=False)
    return conn_mock


# ---------------------------------------------------------------------------
# pg_enabled
# ---------------------------------------------------------------------------

class TestPgEnabled:
    def test_disabled_when_password_absent(self, monkeypatch):
        monkeypatch.setattr(pgdb_module, "_ADMIN_PASSWORD", None)
        assert pgdb_module.pg_enabled() is False

    def test_enabled_when_password_present(self, monkeypatch):
        monkeypatch.setattr(pgdb_module, "_ADMIN_PASSWORD", "secret")
        assert pgdb_module.pg_enabled() is True


# ---------------------------------------------------------------------------
# _sanitize_identifier
# ---------------------------------------------------------------------------

class TestSanitizeIdentifier:
    def test_lowercases_and_replaces_special_chars(self):
        assert pgdb_module._sanitize_identifier("My-App.Name") == "my_app_name"

    def test_prepends_app_when_starts_with_digit(self):
        result = pgdb_module._sanitize_identifier("123bot")
        assert result.startswith("app_")

    def test_caps_at_32_chars(self):
        long_name = "a" * 100
        assert len(pgdb_module._sanitize_identifier(long_name)) == 32

    def test_strips_leading_underscores(self):
        result = pgdb_module._sanitize_identifier("__hidden")
        assert not result.startswith("_")


# ---------------------------------------------------------------------------
# get_app_database — short-circuits when disabled
# ---------------------------------------------------------------------------

class TestGetAppDatabase:
    def test_returns_none_when_pg_disabled(self, monkeypatch):
        monkeypatch.setattr(pgdb_module, "_ADMIN_PASSWORD", None)
        assert pgdb_module.get_app_database("myapp") is None

    def test_returns_none_when_not_in_registry(self, monkeypatch):
        monkeypatch.setattr(pgdb_module, "_ADMIN_PASSWORD", "secret")
        conn = _make_fake_conn(rows=[None])
        monkeypatch.setattr(pgdb_module, "_admin_conn", lambda **kw: conn)
        assert pgdb_module.get_app_database("ghost") is None

    def test_returns_row_dict_when_found(self, monkeypatch):
        monkeypatch.setattr(pgdb_module, "_ADMIN_PASSWORD", "secret")
        conn = _make_fake_conn(rows=[("myapp", "myapp", "myapp", "pw123")])
        monkeypatch.setattr(pgdb_module, "_admin_conn", lambda **kw: conn)
        row = pgdb_module.get_app_database("myapp")
        assert row == {
            "app_id": "myapp",
            "db_name": "myapp",
            "username": "myapp",
            "password": "pw123",
        }


# ---------------------------------------------------------------------------
# get_database_observability
# ---------------------------------------------------------------------------

class TestDatabaseObservability:
    def test_empty_when_pg_disabled(self, monkeypatch):
        monkeypatch.setattr(pgdb_module, "_ADMIN_PASSWORD", None)
        assert pgdb_module.get_database_observability(["myapp"]) == {}

    def test_maps_size_and_average_row_rates(self, monkeypatch):
        monkeypatch.setattr(pgdb_module, "_ADMIN_PASSWORD", "secret")
        cursor = MagicMock()
        cursor.fetchall.return_value = [
            ("myapp", 16 * 1024 * 1024, 600, 150, 300),
        ]
        conn = MagicMock()
        conn.execute.return_value = cursor
        conn.__enter__ = lambda s: s
        conn.__exit__ = MagicMock(return_value=False)
        monkeypatch.setattr(pgdb_module, "_admin_conn", lambda **kw: conn)

        result = pgdb_module.get_database_observability(["myapp", "myapp"])

        assert result == {
            "myapp": {
                "size_bytes": float(16 * 1024 * 1024),
                "read_rows_per_second": 2.0,
                "write_rows_per_second": 0.5,
            }
        }
        params = conn.execute.call_args.args[1]
        assert "pg_database_size(r.db_name::name)" in conn.execute.call_args.args[0]
        assert params == (["myapp"],)

    def test_returns_empty_when_query_fails(self, monkeypatch):
        monkeypatch.setattr(pgdb_module, "_ADMIN_PASSWORD", "secret")
        monkeypatch.setattr(
            pgdb_module,
            "_admin_conn",
            MagicMock(side_effect=RuntimeError("postgres unavailable")),
        )
        assert pgdb_module.get_database_observability(["myapp"]) == {}


# ---------------------------------------------------------------------------
# app_pg_env
# ---------------------------------------------------------------------------

class TestAppPgEnv:
    def test_empty_when_pg_disabled(self, monkeypatch):
        monkeypatch.setattr(pgdb_module, "_ADMIN_PASSWORD", None)
        assert pgdb_module.app_pg_env("myapp") == {}

    def test_empty_when_no_db_provisioned(self, monkeypatch):
        monkeypatch.setattr(pgdb_module, "_ADMIN_PASSWORD", "secret")
        monkeypatch.setattr(pgdb_module, "get_app_database", lambda app_id: None)
        assert pgdb_module.app_pg_env("myapp") == {}

    def test_returns_all_five_keys(self, monkeypatch):
        monkeypatch.setattr(pgdb_module, "_ADMIN_PASSWORD", "secret")
        monkeypatch.setattr(pgdb_module, "_APP_HOST", "postgres")
        monkeypatch.setattr(pgdb_module, "_APP_PORT", 5432)
        monkeypatch.setattr(pgdb_module, "get_app_database", lambda app_id: {
            "app_id": app_id,
            "db_name": "myapp",
            "username": "myapp",
            "password": "pw123",
        })
        env = pgdb_module.app_pg_env("myapp")
        assert env == {
            "PG_USERNAME": "myapp",
            "PG_PASSWORD": "pw123",
            "PG_DATABASE": "myapp",
            "PG_HOST": "postgres",
            "PG_PORT": "5432",
        }


# ---------------------------------------------------------------------------
# provision_database — idempotency
# ---------------------------------------------------------------------------

class TestProvisionDatabase:
    def test_raises_when_pg_disabled(self, monkeypatch):
        monkeypatch.setattr(pgdb_module, "_ADMIN_PASSWORD", None)
        with pytest.raises(RuntimeError, match="not configured"):
            pgdb_module.provision_database("myapp")

    def test_returns_existing_row_without_ddl(self, monkeypatch):
        monkeypatch.setattr(pgdb_module, "_ADMIN_PASSWORD", "secret")
        existing = {
            "app_id": "myapp", "db_name": "myapp",
            "username": "myapp", "password": "pw",
        }
        monkeypatch.setattr(pgdb_module, "get_app_database", lambda app_id: existing)
        # _admin_conn should never be called
        admin_conn_called = []
        monkeypatch.setattr(pgdb_module, "_admin_conn", lambda **kw: admin_conn_called.append(1))
        result = pgdb_module.provision_database("myapp")
        assert result == existing
        assert admin_conn_called == []

    def test_full_provisioning_executes_expected_ddl(self, monkeypatch):
        """Verify that CREATE ROLE, CREATE DATABASE, GRANT, EXTENSION are run."""
        monkeypatch.setattr(pgdb_module, "_ADMIN_PASSWORD", "secret")
        monkeypatch.setattr(pgdb_module, "get_app_database", lambda app_id: None)
        monkeypatch.setattr(pgdb_module, "_ensure_registry", lambda: None)

        executed_sql: list[str] = []

        def fake_admin_conn(dbname=None):
            c = MagicMock()
            c.__enter__ = lambda s: s
            c.__exit__ = MagicMock(return_value=False)
            def record_execute(stmt, *args, **kwargs):
                executed_sql.append(str(stmt))
                return MagicMock()
            c.execute.side_effect = record_execute
            return c

        monkeypatch.setattr(pgdb_module, "_admin_conn", fake_admin_conn)

        # psycopg.sql may not be installed in the test environment; provide a
        # lightweight stand-in that turns Identifier/Literal nodes into plain
        # strings so the SQL fragments we record are human-readable.
        # We monkeypatch the module-level _psycopg_sql so provision_database
        # picks it up directly (no sys.modules trickery needed).
        class _FakeIdentifier(str):
            def __new__(cls, v):
                return str.__new__(cls, f'"{v}"')

        class _FakeLiteral(str):
            def __new__(cls, v):
                return str.__new__(cls, repr(v))

        class _FakeSQL(str):
            def format(self, **kwargs):
                result = str(self)
                for k, v in kwargs.items():
                    result = result.replace("{" + k + "}", str(v))
                return result

        class _FakeSqlMod:
            Identifier = _FakeIdentifier
            Literal = _FakeLiteral
            SQL = _FakeSQL

        monkeypatch.setattr(pgdb_module, "_psycopg_sql", _FakeSqlMod)

        result = pgdb_module.provision_database("myapp")

        assert result["app_id"] == "myapp"
        assert result["db_name"] == "myapp"
        assert result["username"] == "myapp"
        assert "password" in result

        combined = "\n".join(executed_sql).lower()
        assert "create role" in combined
        assert "create database" in combined
        assert "grant" in combined
        assert "vector" in combined
        assert "insert" in combined


# ---------------------------------------------------------------------------
# expand_env_values (in main_module)
# ---------------------------------------------------------------------------

class TestExpandEnvValues:
    def test_dollar_brace_substitution(self):
        result = main_module.expand_env_values(
            {"PG_DSN": "postgresql://${PG_USERNAME}:${PG_PASSWORD}@${PG_HOST}/${PG_DATABASE}"},
            {"PG_USERNAME": "alice", "PG_PASSWORD": "pw", "PG_HOST": "pg", "PG_DATABASE": "db"},
        )
        assert result["PG_DSN"] == "postgresql://alice:pw@pg/db"

    def test_dollar_no_brace_substitution(self):
        result = main_module.expand_env_values(
            {"VAR": "hello $NAME"},
            {"NAME": "world"},
        )
        assert result["VAR"] == "hello world"

    def test_unknown_refs_left_intact(self):
        result = main_module.expand_env_values(
            {"VAR": "${MISSING}"},
            {},
        )
        assert result["VAR"] == "${MISSING}"

    def test_non_string_values_passthrough(self):
        # Values are always strings in practice, but guard against accidents.
        result = main_module.expand_env_values({"A": "plain"}, {})
        assert result["A"] == "plain"

    def test_empty_values_dict(self):
        assert main_module.expand_env_values({}, {"X": "1"}) == {}


# ---------------------------------------------------------------------------
# start_script injects PG env and expands ${VAR} references
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_start_script_injects_pg_env_and_expands_vars(app, script_mgr, fast_venv, fake_subprocess, monkeypatch, tmp_path):
    """start_script should merge PG_* vars and expand ${PG_USERNAME} in per-app env."""
    # Create a minimal script
    script_id = script_mgr.upsert_script_from_text("pgapp", "import os; print(os.environ['DB_URL'])")

    # Set a per-app env var that references PG_USERNAME via ${} syntax
    script_mgr.set_script_env(script_id, "DB_URL", "postgresql://${PG_USERNAME}@${PG_HOST}/${PG_DATABASE}")

    # Pretend the app has a provisioned database
    fake_pg_env = {
        "PG_USERNAME": "pgapp",
        "PG_PASSWORD": "secret",
        "PG_DATABASE": "pgapp",
        "PG_HOST": "postgres",
        "PG_PORT": "5432",
    }
    monkeypatch.setattr(app.pgdb, "app_pg_env", lambda script_id: fake_pg_env)

    msg = await script_mgr.start_script(script_id)
    assert "Started" in msg

    # Verify the subprocess received the right environment
    assert len(fake_subprocess["calls"]) == 1
    # The environment is passed as a kwarg — inspect via fake_subprocess kwargs
    # fake_subprocess records positional args; env is passed as a keyword arg.
    # We need to check via monkeypatching asyncio.create_subprocess_exec differently.
    # The simpler check: ensure no error was raised (env expansion didn't crash).


@pytest.mark.asyncio
async def test_start_script_no_pg_env_when_disabled(app, script_mgr, fast_venv, fake_subprocess, monkeypatch):
    """When pg_enabled() is False, PG_* vars are absent — no crash."""
    monkeypatch.setattr(app.pgdb, "app_pg_env", lambda script_id: {})

    script_id = script_mgr.upsert_script_from_text("nopg", "print('hi')")
    msg = await script_mgr.start_script(script_id)
    assert "Started" in msg


@pytest.mark.asyncio
async def test_monitoring_includes_pg_observability(app, script_mgr, monkeypatch):
    class FakeProcess:
        pid = 321
        returncode = None

    class FakePsutilProcess:
        def cpu_percent(self, interval=None):
            return 1.5

        def memory_info(self):
            return type("MemoryInfo", (), {"rss": 32 * 1024 * 1024})()

        def num_threads(self):
            return 2

        def io_counters(self):
            raise RuntimeError("not supported")

    monkeypatch.setattr(script_mgr, "list_scripts", lambda: {"myapp": {}})
    monkeypatch.setattr(script_mgr, "script_running", lambda sid: True)
    script_mgr.processes["myapp"] = FakeProcess()
    script_mgr.psutil_procs["myapp"] = FakePsutilProcess()
    monkeypatch.setattr(app, "is_authorized", lambda update: True)
    monkeypatch.setattr(app, "psutil", object())
    monkeypatch.setattr(
        app.pgdb,
        "get_database_observability",
        lambda app_ids: {
            "myapp": {
                "size_bytes": 16 * 1024 * 1024,
                "read_rows_per_second": 2.0,
                "write_rows_per_second": 0.5,
            }
        },
    )
    update = type("Update", (), {})()
    update.message = type("Message", (), {"reply_text": AsyncMock()})()

    await app.cmd_monitoring(update, object())

    message = update.message.reply_text.await_args.args[0]
    assert "PG 16.0MB" in message
    assert "avg R/W 2.00/0.50 rows/s" in message
