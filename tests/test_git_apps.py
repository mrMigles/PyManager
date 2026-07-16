# -*- coding: utf-8 -*-
"""Tests for the GitHub-repo app flow, using a local git repo (no network access).
Skipped automatically if `git` isn't available on PATH."""
import shutil
import subprocess

import pytest

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")


def _run_git(*args):
  subprocess.run(["git", *args], check=True, capture_output=True)


@pytest.fixture
def local_repo(tmp_path):
  repo_dir = tmp_path / "local_repo"
  repo_dir.mkdir()
  _run_git("init", "-q", "-b", "main", str(repo_dir))
  _run_git("-C", str(repo_dir), "config", "user.email", "test@example.com")
  _run_git("-C", str(repo_dir), "config", "user.name", "Test")

  (repo_dir / "subapp").mkdir()
  (repo_dir / "subapp" / "main.py").write_text("print('v1')\n")
  (repo_dir / "subapp" / "requirements.txt").write_text("requests\n")
  (repo_dir / "readme.md").write_text("root file\n")
  _run_git("-C", str(repo_dir), "add", "-A")
  _run_git("-C", str(repo_dir), "commit", "-q", "-m", "v1")
  return repo_dir


@pytest.mark.asyncio
async def test_git_clone_or_pull_clones_and_detects_branch(app, local_repo, tmp_path):
  dest = tmp_path / "clone_dest"
  ok, out = await app.git_clone_or_pull(f"file:///{local_repo.as_posix()}", "main", dest)
  assert ok, out

  branch = await app.git_current_branch(dest)
  assert branch == "main"
  assert (dest / "subapp" / "main.py").read_text().strip() == "print('v1')"


@pytest.mark.asyncio
async def test_git_clone_or_pull_pulls_new_commits(app, local_repo, tmp_path):
  dest = tmp_path / "clone_dest"
  ok, _ = await app.git_clone_or_pull(f"file:///{local_repo.as_posix()}", "main", dest)
  assert ok

  (local_repo / "subapp" / "main.py").write_text("print('v2')\n")
  _run_git("-C", str(local_repo), "add", "-A")
  _run_git("-C", str(local_repo), "commit", "-q", "-m", "v2")

  ok2, out2 = await app.git_clone_or_pull(f"file:///{local_repo.as_posix()}", "main", dest)
  assert ok2, out2
  assert (dest / "subapp" / "main.py").read_text().strip() == "print('v2')"


@pytest.mark.asyncio
async def test_setup_git_app_imports_subfolder(script_mgr, app, fast_venv, local_repo, monkeypatch):
  # setup_git_app builds the clone URL from owner/repo; point it at our local repo
  # by making the constructed https URL resolve to a local file:// path instead.
  info = {
      "owner": "owner",
      "repo": "myrepo",
      "branch": "main",
      "path": "subapp",
      "app_id": "myrepo-subapp",
  }

  original_git_clone_or_pull = app.git_clone_or_pull

  async def fake_clone_or_pull(repo_url, branch, dest):
    return await original_git_clone_or_pull(f"file:///{local_repo.as_posix()}", branch, dest)

  monkeypatch.setattr(app, "git_clone_or_pull", fake_clone_or_pull)

  result = await app.setup_git_app(info)
  assert result["status"] == "ready"
  assert result["entry"] == "main.py"
  assert result["requirements_pkgs"] == ["requests"]

  script = script_mgr.get_script("myrepo-subapp")
  assert script["type"] == "git"
  assert script["git"]["path"] == "subapp"
  # only the subfolder should have been imported, not the repo root's readme.md
  root = script_mgr.app_root_dir("myrepo-subapp")
  assert (root / "main.py").exists()
  assert not (root / "readme.md").exists()


@pytest.mark.asyncio
async def test_sync_git_app_detects_dependency_changes(script_mgr, app, fast_venv, local_repo, monkeypatch):
  info = {"owner": "owner", "repo": "myrepo", "branch": "main", "path": "subapp", "app_id": "myrepo-subapp"}

  original_git_clone_or_pull = app.git_clone_or_pull

  async def fake_clone_or_pull(repo_url, branch, dest):
    return await original_git_clone_or_pull(f"file:///{local_repo.as_posix()}", branch, dest)

  monkeypatch.setattr(app, "git_clone_or_pull", fake_clone_or_pull)

  await app.setup_git_app(info)

  # no upstream change yet -> sync should report no dependency changes
  result = await app.sync_git_app("myrepo-subapp")
  assert result["status"] == "ready"
  assert result["deps_changed"] is False

  # add a new dependency upstream and commit
  (local_repo / "subapp" / "requirements.txt").write_text("requests\nflask\n")
  (local_repo / "subapp" / "main.py").write_text("print('v2')\n")
  _run_git("-C", str(local_repo), "add", "-A")
  _run_git("-C", str(local_repo), "commit", "-q", "-m", "v2")

  result2 = await app.sync_git_app("myrepo-subapp")
  assert result2["status"] == "ready"
  assert result2["deps_changed"] is True

  root = script_mgr.app_root_dir("myrepo-subapp")
  assert (root / "main.py").read_text().strip() == "print('v2')"
  assert "flask" in (root / "requirements.txt").read_text()
