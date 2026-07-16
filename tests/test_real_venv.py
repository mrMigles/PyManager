# -*- coding: utf-8 -*-
"""End-to-end tests that spin up a *real* virtualenv (no mocking of create_venv).
Slower than the rest of the suite (a few seconds), but this is what actually
protects the core isolation guarantee: every app gets its own venv, and only
system-site-packages (i.e. python-telegram-bot/psutil) are shared."""
import sys

import pytest


@pytest.mark.asyncio
async def test_create_venv_is_isolated_but_shares_system_site_packages(app, tmp_path):
  venv_dir = tmp_path / "venvs" / "demo"
  ok, out = await app.create_venv(venv_dir)
  assert ok, out

  python_exe = app.venv_python_path(venv_dir)
  assert python_exe.exists()

  # The venv is created with --system-site-packages, so it must be able to see
  # whatever is importable in the outer/system interpreter (here: pytest itself,
  # which stands in for python-telegram-bot in the real container image).
  proc = await app.asyncio.create_subprocess_exec(
      str(python_exe), "-c", "import pytest",
      stdout=app.asyncio.subprocess.PIPE, stderr=app.asyncio.subprocess.STDOUT,
  )
  out, _ = await proc.communicate()
  assert proc.returncode == 0, out.decode()


@pytest.mark.asyncio
async def test_create_venv_is_idempotent(app, tmp_path):
  venv_dir = tmp_path / "venvs" / "demo"
  ok1, _ = await app.create_venv(venv_dir)
  assert ok1
  # Second call should short-circuit (interpreter already exists) rather than
  # re-running `python -m venv`.
  ok2, msg2 = await app.create_venv(venv_dir)
  assert ok2
  assert "already exists" in msg2


@pytest.mark.asyncio
async def test_full_project_app_lifecycle_runs_in_its_own_venv(script_mgr, app, tmp_path):
  staging = tmp_path / "staging"
  staging.mkdir()
  (staging / "main.py").write_text(
      "import sys\n"
      "print('running from:', sys.executable)\n"
  )

  result = await script_mgr.setup_project_app("demo", staging, "project")
  assert result["status"] == "ready"
  assert result["venv_ok"] is True

  expected_python = app.venv_python_path(app.VENVS_DIR / "demo")
  assert expected_python.exists()
  assert expected_python != app.Path(sys.executable)

  start_msg = await script_mgr.start_script("demo")
  assert "Started" in start_msg

  for _ in range(50):
    if not script_mgr.script_running("demo"):
      break
    await __import__("asyncio").sleep(0.1)

  log_tail = script_mgr.read_log_tail("demo")
  assert "running from:" in log_tail
  assert str(expected_python) in log_tail
