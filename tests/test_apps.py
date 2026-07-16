# -*- coding: utf-8 -*-
"""Tests for archive-based apps: setup, entry-point detection, versioning/rollback,
each with an isolated (mocked, for speed) venv. See test_real_venv.py for a slower
end-to-end test that spins up a *real* virtualenv."""
import pytest


def _write_project(root, files):
  root.mkdir(parents=True, exist_ok=True)
  for rel, content in files.items():
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
  return root


@pytest.mark.asyncio
async def test_setup_project_app_single_file_auto_selects_entry(script_mgr, fast_venv, tmp_path):
  staging = _write_project(tmp_path / "staging", {"bot.py": "print('hi')\n"})

  result = await script_mgr.setup_project_app("demo", staging, "project")

  assert result["status"] == "ready"
  assert result["entry"] == "bot.py"
  assert result["venv_ok"] is True

  script = script_mgr.get_script("demo")
  assert script["type"] == "project"
  assert script["entry"] == "bot.py"
  assert not staging.exists()  # moved, not copied


@pytest.mark.asyncio
async def test_setup_project_app_prefers_main_py(script_mgr, fast_venv, tmp_path):
  staging = _write_project(tmp_path / "staging", {
      "utils.py": "x = 1\n",
      "main.py": "print('entry')\n",
  })
  result = await script_mgr.setup_project_app("demo", staging, "project")
  assert result["status"] == "ready"
  assert result["entry"] == "main.py"


@pytest.mark.asyncio
async def test_setup_project_app_ambiguous_without_main_py(script_mgr, fast_venv, tmp_path):
  staging = _write_project(tmp_path / "staging", {
      "a.py": "x = 1\n",
      "b.py": "x = 2\n",
  })
  result = await script_mgr.setup_project_app("demo", staging, "project")
  assert result["status"] == "ambiguous"
  assert set(result["candidates"]) == {"a.py", "b.py"}
  assert script_mgr.pending_entry_choice["demo"] == result["candidates"]


@pytest.mark.asyncio
async def test_setup_project_app_detects_requirements(script_mgr, fast_venv, tmp_path):
  staging = _write_project(tmp_path / "staging", {
      "main.py": "print('hi')\n",
      "requirements.txt": "requests\nflask==2.0\n",
  })
  result = await script_mgr.setup_project_app("demo", staging, "project")
  assert result["requirements_pkgs"] == ["requests", "flask==2.0"]


@pytest.mark.asyncio
async def test_reupload_snapshots_previous_app_version(script_mgr, fast_venv, tmp_path):
  staging1 = _write_project(tmp_path / "s1", {"main.py": "print('v1')\n"})
  await script_mgr.setup_project_app("demo", staging1, "project")

  staging2 = _write_project(tmp_path / "s2", {"main.py": "print('v2')\n"})
  result = await script_mgr.setup_project_app("demo", staging2, "project")

  assert result["status"] == "ready"
  versions = script_mgr.get_versions("demo")
  assert len(versions) == 1

  current = (script_mgr.app_root_dir("demo") / "main.py").read_text()
  assert current.strip() == "print('v2')"


@pytest.mark.asyncio
async def test_reupload_preserves_env_and_autostart(script_mgr, fast_venv, tmp_path):
  staging1 = _write_project(tmp_path / "s1", {"main.py": "print('v1')\n"})
  await script_mgr.setup_project_app("demo", staging1, "project")
  script_mgr.set_script_env("demo", "TOKEN", "secret")
  script_mgr.set_autostart("demo", True)

  staging2 = _write_project(tmp_path / "s2", {"main.py": "print('v2')\n"})
  await script_mgr.setup_project_app("demo", staging2, "project")

  script = script_mgr.get_script("demo")
  assert script["env"]["TOKEN"] == "secret"
  assert script["autostart"] is True


@pytest.mark.asyncio
async def test_app_rollback_restores_directory_contents(script_mgr, fast_venv, tmp_path):
  staging1 = _write_project(tmp_path / "s1", {"main.py": "print('v1')\n", "helper.py": "x = 1\n"})
  await script_mgr.setup_project_app("demo", staging1, "project")

  staging2 = _write_project(tmp_path / "s2", {"main.py": "print('v2')\n"})
  await script_mgr.setup_project_app("demo", staging2, "project")

  versions = script_mgr.get_versions("demo")
  ts_v1 = versions[0]["ts"]

  msg = await script_mgr.rollback_to("demo", ts_v1)
  assert "Rolled back" in msg

  root = script_mgr.app_root_dir("demo")
  assert (root / "main.py").read_text().strip() == "print('v1')"
  assert (root / "helper.py").exists()  # v2 didn't have this file - full dir was restored


@pytest.mark.asyncio
async def test_app_rollback_correct_despite_colliding_timestamps(script_mgr, fast_venv, tmp_path, monkeypatch, app):
  """Same regression as the legacy-script version, but for the tar.gz-based app
  versioning path."""
  staging1 = _write_project(tmp_path / "s1", {"main.py": "print('v1')\n"})
  await script_mgr.setup_project_app("demo", staging1, "project")

  staging2 = _write_project(tmp_path / "s2", {"main.py": "print('v2')\n"})
  await script_mgr.setup_project_app("demo", staging2, "project")

  versions = script_mgr.get_versions("demo")
  ts_v1 = versions[0]["ts"]
  monkeypatch.setattr(app, "now_ts_str", lambda: ts_v1)

  msg = await script_mgr.rollback_to("demo", ts_v1)
  assert "Rolled back" in msg
  root = script_mgr.app_root_dir("demo")
  assert (root / "main.py").read_text().strip() == "print('v1')"


@pytest.mark.asyncio
async def test_get_env_keys_scans_all_files_in_project(script_mgr, fast_venv, tmp_path):
  staging = _write_project(tmp_path / "staging", {
      "main.py": "import os\nos.getenv('A')\n",
      "sub/worker.py": "import os\nos.environ['B']\n",
  })
  await script_mgr.setup_project_app("demo", staging, "project")
  assert script_mgr.get_env_keys_for_script("demo") == ["A", "B"]


@pytest.mark.asyncio
async def test_start_script_project_type_uses_venv_python(script_mgr, fast_venv, app, tmp_path, fake_subprocess):
  staging = _write_project(tmp_path / "staging", {"main.py": "print('hi')\n"})
  await script_mgr.setup_project_app("demo", staging, "project")

  # start_script would normally spawn the fake venv's python - just check it
  # picks the *right* interpreter and target, without actually running anything.
  msg = await script_mgr.start_script("demo")
  assert "Started" in msg

  assert len(fake_subprocess["calls"]) == 1
  python_exe, run_target = fake_subprocess["calls"][0]
  assert python_exe == str(app.venv_python_path(app.VENVS_DIR / "demo"))
  assert run_target == str(script_mgr.app_root_dir("demo") / "main.py")


@pytest.mark.asyncio
async def test_ensure_app_environments_recreates_missing_venv(script_mgr, fast_venv, app, tmp_path):
  # deliberately no requirements.txt here: we only want to exercise venv
  # (re)creation, not the separate pip-install subprocess call.
  staging = _write_project(tmp_path / "staging", {"main.py": "print('hi')\n"})
  await script_mgr.setup_project_app("demo", staging, "project")

  # simulate the venv having been wiped (e.g. fresh volume)
  import shutil
  shutil.rmtree(app.VENVS_DIR / "demo")
  assert not app.venv_python_path(app.VENVS_DIR / "demo").exists()

  await script_mgr.ensure_app_environments()
  assert app.venv_python_path(app.VENVS_DIR / "demo").exists()
