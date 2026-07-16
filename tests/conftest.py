# -*- coding: utf-8 -*-
"""
Shared pytest fixtures for the PyManager test suite.

main.py reads BOT_TOKEN/OWNER_ID/DATA_DIR and creates its data directories at
*import time*, so we set safe defaults before the very first `import main`
happens (below). Each test then gets its own fully isolated data directory by
monkeypatching main's path constants (and a fresh ScriptManager) via the
`app` fixture - no module reloads, no shared state between tests, and nothing
is ever written to a real /data.
"""
import os
import sys
import tempfile
from pathlib import Path

import pytest

# --- Set up safe defaults BEFORE importing main -----------------------------
_BOOTSTRAP_DATA_DIR = Path(tempfile.mkdtemp(prefix="pymanager_bootstrap_"))

os.environ.setdefault("BOT_TOKEN", "123456:TEST-TOKEN")
os.environ.setdefault("OWNER_ID", "1")
os.environ.setdefault("DATA_DIR", str(_BOOTSTRAP_DATA_DIR))

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

import main as main_module  # noqa: E402  (must import after env vars are set)


@pytest.fixture
def app(monkeypatch, tmp_path):
  """The `main` module with every data path isolated under tmp_path, and a
  fresh `script_mgr` bound to it. Use this in any test that touches disk state."""
  data_dir = tmp_path
  paths = {
      "DATA_DIR": data_dir,
      "SCRIPTS_DIR": data_dir / "scripts",
      "LOGS_DIR": data_dir / "logs",
      "VERSIONS_DIR": data_dir / "versions",
      "APPS_DIR": data_dir / "apps",
      "APP_VERSIONS_DIR": data_dir / "app_versions",
      "VENVS_DIR": data_dir / "venvs",
      "GITREPOS_DIR": data_dir / "gitrepos",
      "TMP_DIR": data_dir / "tmp",
  }
  for name, path in paths.items():
    path.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(main_module, name, path)

  monkeypatch.setattr(main_module, "META_FILE", data_dir / "meta.json")
  monkeypatch.setattr(main_module, "REQUIREMENTS_FILE", data_dir / "requirements.txt")

  mgr = main_module.ScriptManager()
  monkeypatch.setattr(main_module, "script_mgr", mgr)

  return main_module


@pytest.fixture
def script_mgr(app):
  return app.script_mgr


class FakeProcess:
  """Stand-in for asyncio.subprocess.Process, used to avoid spawning real
  pip/git subprocesses in tests that only care about *what* gets run."""

  def __init__(self, returncode=0, stdout=b""):
    self.returncode = returncode
    self._stdout = stdout
    self.pid = 12345

  async def communicate(self):
    return self._stdout, b""

  async def wait(self):
    return self.returncode

  def terminate(self):
    pass

  def kill(self):
    pass


@pytest.fixture
def fake_subprocess(monkeypatch):
  """Patches asyncio.create_subprocess_exec, recording every invocation.
  `calls` is a list of the positional args each call was made with.
  `result` controls what every call returns; override per-test as needed."""
  import asyncio

  state = {"calls": [], "returncode": 0, "stdout": b""}

  async def _fake_create_subprocess_exec(*args, **kwargs):
    state["calls"].append(args)
    return FakeProcess(returncode=state["returncode"], stdout=state["stdout"])

  monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_create_subprocess_exec)
  return state


@pytest.fixture
def fast_venv(monkeypatch, app):
  """Replaces real `python -m venv` calls with an instant fake: creates the
  expected interpreter file so `.exists()` checks pass, without spawning a
  subprocess. Use this in tests that don't care about the venv's contents,
  to keep the suite fast."""

  async def _fake_create_venv(venv_dir: Path):
    py = main_module.venv_python_path(venv_dir)
    py.parent.mkdir(parents=True, exist_ok=True)
    py.write_text("#!/usr/bin/env python\n")
    py.chmod(0o755)
    return True, "fake venv"

  monkeypatch.setattr(main_module, "create_venv", _fake_create_venv)
  return main_module
