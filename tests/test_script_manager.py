# -*- coding: utf-8 -*-
"""Unit tests for ScriptManager: legacy scripts (BWC), env, autostart, versions/rollback."""
import json

import pytest


# --- legacy scripts (type="script") - must keep working exactly as before ------

def test_upsert_script_from_text_creates_py_file(script_mgr, app):
  script_id = script_mgr.upsert_script_from_text("hello world", "print('hi')")
  assert script_id == "hello_world"

  script = script_mgr.get_script(script_id)
  assert script["type"] == "script"
  assert script["file"] == str(script_mgr.script_file_path(script_id))
  assert app.read_text_file(script_mgr.script_file_path(script_id)) == "print('hi')"


def test_load_meta_defaults_missing_type_to_script_for_bwc(app):
  # Simulate an old meta.json written before the "type" field existed.
  app.META_FILE.write_text(json.dumps({
      "global_env": {},
      "scripts": {
        "legacy": {"env": {}, "autostart": False, "versions": [], "updated_at": None, "file": "x.py"},
      },
  }))
  mgr = app.ScriptManager()
  assert mgr.get_type("legacy") == "script"
  assert mgr.is_project_type("legacy") is False


def test_upsert_script_overwrite_snapshots_previous_version(script_mgr):
  sid = script_mgr.upsert_script_from_text("bot", "print(1)")
  script_mgr.upsert_script_from_text("bot", "print(2)")

  versions = script_mgr.get_versions(sid)
  assert len(versions) == 1

  current = script_mgr.script_file_path(sid).read_text()
  assert current == "print(2)"


def test_versions_are_capped_at_max(script_mgr, app):
  sid = script_mgr.upsert_script_from_text("bot", "print(0)")
  for i in range(1, app.MAX_VERSIONS_PER_SCRIPT + 5):
    script_mgr.upsert_script_from_text("bot", f"print({i})")

  versions = script_mgr.get_versions(sid)
  assert len(versions) == app.MAX_VERSIONS_PER_SCRIPT
  for v in versions:
    assert app.Path(v["file"]).exists()


@pytest.mark.asyncio
async def test_rollback_restores_previous_script_content(script_mgr):
  sid = script_mgr.upsert_script_from_text("bot", "print('v1')")
  script_mgr.upsert_script_from_text("bot", "print('v2')")

  versions = script_mgr.get_versions(sid)
  ts_v1 = versions[0]["ts"]

  msg = await script_mgr.rollback_to(sid, ts_v1)
  assert "Rolled back" in msg
  assert script_mgr.script_file_path(sid).read_text() == "print('v1')"


@pytest.mark.asyncio
async def test_rollback_is_correct_even_with_colliding_timestamps(script_mgr, monkeypatch, app):
  """Regression test: rolling back used to read the target version file AFTER
  snapshotting the current file, which could silently overwrite it if both
  snapshots landed on the same (1s-resolution) timestamp. The fix reads the
  target bytes before doing anything destructive."""
  sid = script_mgr.upsert_script_from_text("bot", "print('v1')")
  script_mgr.upsert_script_from_text("bot", "print('v2')")

  versions = script_mgr.get_versions(sid)
  ts_v1 = versions[0]["ts"]

  # Force the "snapshot current before rollback" step to reuse the exact same
  # timestamp (and therefore the exact same file name) as the version we're
  # about to roll back to.
  monkeypatch.setattr(app, "now_ts_str", lambda: ts_v1)

  msg = await script_mgr.rollback_to(sid, ts_v1)
  assert "Rolled back" in msg
  assert script_mgr.script_file_path(sid).read_text() == "print('v1')"


@pytest.mark.asyncio
async def test_rollback_unknown_version_reports_error(script_mgr):
  sid = script_mgr.upsert_script_from_text("bot", "print(1)")
  msg = await script_mgr.rollback_to(sid, "no-such-ts")
  assert "not found" in msg.lower()


@pytest.mark.asyncio
async def test_start_script_missing_file_reports_error(script_mgr):
  msg = await script_mgr.start_script("does-not-exist")
  assert "not found" in msg.lower()


# --- env vars -------------------------------------------------------------------

def test_get_env_keys_for_script_detects_from_source(script_mgr):
  sid = script_mgr.upsert_script_from_text("bot", "import os\nos.getenv('TOKEN')\n")
  assert script_mgr.get_env_keys_for_script(sid) == ["TOKEN"]


def test_set_and_del_script_env(script_mgr):
  sid = script_mgr.upsert_script_from_text("bot", "print(1)")
  script_mgr.set_script_env(sid, "TOKEN", "abc")
  assert script_mgr.get_script(sid)["env"]["TOKEN"] == "abc"

  assert script_mgr.del_script_env(sid, "TOKEN") is True
  assert "TOKEN" not in script_mgr.get_script(sid)["env"]
  assert script_mgr.del_script_env(sid, "TOKEN") is False


def test_global_env_set_and_del(script_mgr):
  script_mgr.set_global_env("FOO", "bar")
  assert script_mgr.get_global_env()["FOO"] == "bar"
  assert script_mgr.del_global_env("FOO") is True
  assert "FOO" not in script_mgr.get_global_env()


def test_autostart_toggle(script_mgr):
  sid = script_mgr.upsert_script_from_text("bot", "print(1)")
  assert script_mgr.set_autostart(sid, True) is True
  assert script_mgr.get_script(sid)["autostart"] is True
  assert script_mgr.set_autostart("nope", True) is False


# --- requirements.txt handling ---------------------------------------------------

def test_add_requirements_is_idempotent(script_mgr, app):
  changed1 = script_mgr.add_requirements(["requests", "flask==2.0"])
  changed2 = script_mgr.add_requirements(["requests"])
  assert changed1 is True
  assert changed2 is False
  assert script_mgr._read_requirements_lines() == ["requests", "flask==2.0"]


def test_add_requirements_scoped_to_custom_path(script_mgr, tmp_path):
  custom = tmp_path / "app_reqs.txt"
  script_mgr.add_requirements(["requests"], path=custom)
  assert custom.exists()
  assert script_mgr._read_requirements_lines(custom) == ["requests"]
  # shared requirements.txt must be untouched
  assert script_mgr._read_requirements_lines() == []


# --- pip install scoping (shared vs per-app venv) --------------------------------

@pytest.mark.asyncio
async def test_pip_install_legacy_script_targets_shared_interpreter(script_mgr, app, fake_subprocess):
  fake_subprocess["stdout"] = b"Successfully installed requests\n"
  sid = script_mgr.upsert_script_from_text("bot", "print(1)")

  msg = await script_mgr.pip_install("requests", script_id=sid)

  assert len(fake_subprocess["calls"]) == 1
  called_python = fake_subprocess["calls"][0][0]
  assert called_python == app.sys.executable
  assert "requirements.txt" in msg
  assert script_mgr._read_requirements_lines() == ["requests"]


@pytest.mark.asyncio
async def test_pip_install_project_app_targets_its_own_venv(script_mgr, app, fake_subprocess, fast_venv, tmp_path):
  staging = tmp_path / "staging"
  staging.mkdir()
  (staging / "main.py").write_text("print('hi')\n")
  result = await script_mgr.setup_project_app("demo", staging, "project")
  assert result["status"] == "ready"

  fake_subprocess["stdout"] = b"Successfully installed requests\n"
  msg = await script_mgr.pip_install("requests", script_id="demo")

  called_python = fake_subprocess["calls"][0][0]
  expected_python = str(app.venv_python_path(app.VENVS_DIR / "demo"))
  assert called_python == expected_python
  assert called_python != app.sys.executable

  app_req_file = app.APPS_DIR / "demo" / "requirements.txt"
  assert app_req_file.exists()
  assert "requests" in app_req_file.read_text()
  # the shared/global requirements.txt must stay untouched
  assert script_mgr._read_requirements_lines() == []
  assert "demo" in msg or "requirements.txt" in msg
