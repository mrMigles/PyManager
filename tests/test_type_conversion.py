# -*- coding: utf-8 -*-
"""Tests for cross-type app conversion (script ↔ project ↔ git).

Covers:
- script → project and project → script conversions (artifact cleanup, meta update)
- git ↔ project as sub-cases of app ↔ app (covered lightly via rollback tests)
- Rollback across a type boundary in both directions (full revert: type, files, meta)
- Running process is stopped before conversion / rollback replaces on-disk state
- conflicting_type / _set_pending_conversion / _clear_pending_conversion helpers
- env/autostart survive a conversion
- Version kind tags written correctly and pass through get_versions
- "Change Source" flow: pending_source_change state, force_target_id routing
"""
import shutil
from pathlib import Path
from typing import Dict

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_project(root: Path, files: Dict[str, str]) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for rel, content in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
    return root


# ---------------------------------------------------------------------------
# conflicting_type
# ---------------------------------------------------------------------------

def test_conflicting_type_returns_none_for_new_id(script_mgr):
    assert script_mgr.conflicting_type("nonexistent", "script") is None


def test_conflicting_type_returns_none_for_same_type(script_mgr):
    script_mgr.upsert_script_from_text("mybot", "print(1)")
    assert script_mgr.conflicting_type("mybot", "script") is None


@pytest.mark.asyncio
async def test_conflicting_type_returns_old_type_when_different(script_mgr, fast_venv, tmp_path):
    staging = _write_project(tmp_path / "s", {"main.py": "print(1)\n"})
    await script_mgr.setup_project_app("demo", staging, "project")
    assert script_mgr.conflicting_type("demo", "script") == "project"
    assert script_mgr.conflicting_type("demo", "git") == "project"
    assert script_mgr.conflicting_type("demo", "project") is None


# ---------------------------------------------------------------------------
# _set_pending_conversion / _clear_pending_conversion
# ---------------------------------------------------------------------------

def test_set_pending_conversion_replaces_old_stash_and_cleans_temp(script_mgr, tmp_path):
    staged1 = tmp_path / "stage1"
    staged1.mkdir()
    script_mgr._set_pending_conversion("demo", {"action": "archive", "staged_root": str(staged1)})
    assert "demo" in script_mgr.pending_type_conversion

    staged2 = tmp_path / "stage2"
    staged2.mkdir()
    script_mgr._set_pending_conversion("demo", {"action": "archive", "staged_root": str(staged2)})
    # Old staged dir should have been cleaned up.
    assert not staged1.exists()
    assert "demo" in script_mgr.pending_type_conversion


def test_clear_pending_conversion_removes_staged_dir(script_mgr, tmp_path):
    staged = tmp_path / "staged"
    staged.mkdir()
    script_mgr._set_pending_conversion("demo", {"action": "archive", "staged_root": str(staged)})
    script_mgr._clear_pending_conversion("demo")
    assert not staged.exists()
    assert "demo" not in script_mgr.pending_type_conversion


# ---------------------------------------------------------------------------
# script → project conversion
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_script_to_project_clears_py_file(script_mgr, fast_venv, app, tmp_path):
    script_mgr.upsert_script_from_text("mybot", "print('v1')\n")
    py_file = script_mgr.script_file_path("mybot")
    assert py_file.exists()

    staging = _write_project(tmp_path / "s", {"main.py": "print('app')\n"})
    await script_mgr.setup_project_app("mybot", staging, "project")

    # Old .py file must be gone.
    assert not py_file.exists()
    # New type is "project".
    s = script_mgr.get_script("mybot")
    assert s["type"] == "project"
    assert "root_dir" in s
    assert "file" not in s


@pytest.mark.asyncio
async def test_script_to_project_preserves_env_and_autostart(script_mgr, fast_venv, tmp_path):
    script_mgr.upsert_script_from_text("mybot", "print('v1')\n")
    script_mgr.set_script_env("mybot", "TOKEN", "secret")
    script_mgr.set_autostart("mybot", True)

    staging = _write_project(tmp_path / "s", {"main.py": "print('app')\n"})
    await script_mgr.setup_project_app("mybot", staging, "project")

    s = script_mgr.get_script("mybot")
    assert s["env"]["TOKEN"] == "secret"
    assert s["autostart"] is True


@pytest.mark.asyncio
async def test_script_to_project_creates_version_tagged_script(script_mgr, fast_venv, tmp_path):
    script_mgr.upsert_script_from_text("mybot", "print('v1')\n")
    staging = _write_project(tmp_path / "s", {"main.py": "print('app')\n"})
    await script_mgr.setup_project_app("mybot", staging, "project")

    versions = script_mgr.get_versions("mybot")
    assert len(versions) == 1
    assert versions[0]["kind"] == "script"


# ---------------------------------------------------------------------------
# project → script conversion
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_project_to_script_clears_app_dirs(script_mgr, fast_venv, app, tmp_path):
    staging = _write_project(tmp_path / "s", {"main.py": "print('app')\n"})
    await script_mgr.setup_project_app("mybot", staging, "project")

    root_dir = Path(script_mgr.get_script("mybot")["root_dir"])
    venv_dir = Path(script_mgr.get_script("mybot")["venv_dir"])
    assert root_dir.exists()
    assert venv_dir.exists()

    script_mgr.upsert_script_from_text("mybot", "print('script')\n")

    # App source tree and venv must be gone.
    assert not root_dir.exists()
    assert not venv_dir.exists()
    s = script_mgr.get_script("mybot")
    assert s["type"] == "script"
    assert "root_dir" not in s
    assert "venv_dir" not in s
    assert "entry" not in s
    assert script_mgr.script_file_path("mybot").exists()


@pytest.mark.asyncio
async def test_project_to_script_creates_version_tagged_app(script_mgr, fast_venv, tmp_path):
    staging = _write_project(tmp_path / "s", {"main.py": "print('v1')\n"})
    await script_mgr.setup_project_app("mybot", staging, "project")

    script_mgr.upsert_script_from_text("mybot", "print('script')\n")

    versions = script_mgr.get_versions("mybot")
    assert len(versions) == 1
    v = versions[0]
    assert v["kind"] == "app"
    assert v["app_type"] == "project"
    assert v["file"].endswith(".tar.gz")


@pytest.mark.asyncio
async def test_project_to_script_preserves_env_and_autostart(script_mgr, fast_venv, tmp_path):
    staging = _write_project(tmp_path / "s", {"main.py": "print('v1')\n"})
    await script_mgr.setup_project_app("mybot", staging, "project")
    script_mgr.set_script_env("mybot", "KEY", "val")
    script_mgr.set_autostart("mybot", True)

    script_mgr.upsert_script_from_text("mybot", "print('script')\n")

    s = script_mgr.get_script("mybot")
    assert s["env"]["KEY"] == "val"
    assert s["autostart"] is True


# ---------------------------------------------------------------------------
# git → script conversion
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_git_to_script_clears_app_and_repo_dirs(script_mgr, fast_venv, app, tmp_path):
    git_info = {"owner": "o", "repo": "r", "branch": "main", "path": None, "repo_dir": str(tmp_path / "gitrepo")}
    staging = _write_project(tmp_path / "s", {"main.py": "print('git')\n"})
    await script_mgr.setup_project_app("mybot", staging, "git", git_info)

    root_dir = Path(script_mgr.get_script("mybot")["root_dir"])
    venv_dir = Path(script_mgr.get_script("mybot")["venv_dir"])
    repo_dir = Path(git_info["repo_dir"])
    repo_dir.mkdir(parents=True, exist_ok=True)  # simulate existing clone

    script_mgr.upsert_script_from_text("mybot", "print('script')\n")

    assert not root_dir.exists()
    assert not venv_dir.exists()
    assert not repo_dir.exists()
    s = script_mgr.get_script("mybot")
    assert s["type"] == "script"
    assert "git" not in s


# ---------------------------------------------------------------------------
# Rollback across type boundary: project → script (rolling back to script version)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_rollback_from_project_to_script_version(script_mgr, fast_venv, app, tmp_path):
    # 1. Start as script
    script_mgr.upsert_script_from_text("mybot", "print('script-v1')\n")
    py_file = script_mgr.script_file_path("mybot")

    # 2. Convert to project
    staging = _write_project(tmp_path / "s", {"main.py": "print('app-v1')\n"})
    await script_mgr.setup_project_app("mybot", staging, "project")
    assert not py_file.exists()

    root_dir = Path(script_mgr.get_script("mybot")["root_dir"])

    # 3. Roll back to the script version
    versions = script_mgr.get_versions("mybot")
    script_version = next(v for v in versions if v.get("kind") == "script")
    msg = await script_mgr.rollback_to("mybot", script_version["ts"])
    assert "Rolled back" in msg

    # 4. State should be script again
    s = script_mgr.get_script("mybot")
    assert s["type"] == "script"
    assert py_file.exists()
    assert "script-v1" in py_file.read_text()
    # App directory must be gone
    assert not root_dir.exists()
    # Meta keys must not contain app fields
    assert "root_dir" not in s
    assert "venv_dir" not in s


# ---------------------------------------------------------------------------
# Rollback across type boundary: script → project (rolling back to project version)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_rollback_from_script_to_project_version(script_mgr, fast_venv, app, tmp_path):
    # 1. Start as project
    staging = _write_project(tmp_path / "s", {"main.py": "print('app-v1')\n", "helper.py": "x=1\n"})
    await script_mgr.setup_project_app("mybot", staging, "project")

    # 2. Convert to script
    script_mgr.upsert_script_from_text("mybot", "print('script-v1')\n")
    py_file = script_mgr.script_file_path("mybot")
    assert py_file.exists()

    # 3. Roll back to the project version
    versions = script_mgr.get_versions("mybot")
    app_version = next(v for v in versions if v.get("kind") == "app")
    msg = await script_mgr.rollback_to("mybot", app_version["ts"])
    assert "Rolled back" in msg

    # 4. State should be project again
    s = script_mgr.get_script("mybot")
    assert s["type"] == "project"
    root = Path(s["root_dir"])
    assert (root / "main.py").exists()
    assert (root / "helper.py").exists()
    # .py script file must be gone
    assert not py_file.exists()
    assert "file" not in s


# ---------------------------------------------------------------------------
# Cross-type rollback is itself reversible (rollback the rollback)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_cross_type_rollback_is_reversible(script_mgr, fast_venv, app, tmp_path):
    # script → project → rollback to script → rollback back to project
    script_mgr.upsert_script_from_text("mybot", "print('script-v1')\n")

    staging = _write_project(tmp_path / "s", {"main.py": "print('app-v1')\n"})
    await script_mgr.setup_project_app("mybot", staging, "project")

    # Roll back to script
    versions = script_mgr.get_versions("mybot")
    script_ver = next(v for v in versions if v.get("kind") == "script")
    await script_mgr.rollback_to("mybot", script_ver["ts"])
    assert script_mgr.get_script("mybot")["type"] == "script"

    # Roll back to project (the snapshot taken when we reverted to script)
    versions2 = script_mgr.get_versions("mybot")
    app_ver = next(v for v in versions2 if v.get("kind") == "app")
    await script_mgr.rollback_to("mybot", app_ver["ts"])

    s = script_mgr.get_script("mybot")
    assert s["type"] == "project"
    root = Path(s["root_dir"])
    assert (root / "main.py").exists()


# ---------------------------------------------------------------------------
# Running process is stopped before conversion / rollback replaces disk state
# ---------------------------------------------------------------------------

class _FakeRunningProcess:
    """A fake process that stays 'running' (returncode=None) until terminate() is called."""

    def __init__(self):
        self.returncode = None
        self.pid = 99999

    def terminate(self):
        self.returncode = 0

    def kill(self):
        self.returncode = -9

    async def wait(self):
        return self.returncode


@pytest.mark.asyncio
async def test_setup_project_app_stops_running_process(script_mgr, fast_venv, app, tmp_path):
    script_mgr.upsert_script_from_text("mybot", "print(1)\n")

    # Inject a fake running process directly (FakeProcess from conftest exits instantly,
    # making it impossible to assert script_running() after start).
    fake_proc = _FakeRunningProcess()
    script_mgr.processes["mybot"] = fake_proc
    assert script_mgr.script_running("mybot")

    staging = _write_project(tmp_path / "s", {"main.py": "print('app')\n"})
    await script_mgr.setup_project_app("mybot", staging, "project")

    # The stop guard inside setup_project_app must have terminated the process.
    assert not script_mgr.script_running("mybot")
    assert script_mgr.get_script("mybot")["type"] == "project"


@pytest.mark.asyncio
async def test_rollback_stops_running_process(script_mgr, fast_venv, app, tmp_path, fake_subprocess):
    # fake_subprocess prevents rollback_to from spawning a real Python process on restart.
    script_mgr.upsert_script_from_text("mybot", "print('v1')\n")
    script_mgr.upsert_script_from_text("mybot", "print('v2')\n")

    fake_proc = _FakeRunningProcess()
    script_mgr.processes["mybot"] = fake_proc
    assert script_mgr.script_running("mybot")

    versions = script_mgr.get_versions("mybot")
    ts = versions[0]["ts"]
    await script_mgr.rollback_to("mybot", ts)

    # The original fake process was terminated before the files were replaced.
    assert fake_proc.returncode == 0


# ---------------------------------------------------------------------------
# Version kind tags
# ---------------------------------------------------------------------------

def test_snapshot_current_version_tags_kind_script(script_mgr):
    script_mgr.upsert_script_from_text("mybot", "print('v1')\n")
    script_mgr.snapshot_current_version("mybot")
    versions = script_mgr.get_versions("mybot")
    assert versions[0]["kind"] == "script"


@pytest.mark.asyncio
async def test_snapshot_current_app_version_tags_kind_and_app_type(script_mgr, fast_venv, tmp_path):
    staging = _write_project(tmp_path / "s", {"main.py": "print('v1')\n"})
    await script_mgr.setup_project_app("demo", staging, "project")
    script_mgr.snapshot_current_app_version("demo")
    versions = script_mgr.get_versions("demo")
    assert versions[0]["kind"] == "app"
    assert versions[0]["app_type"] == "project"


@pytest.mark.asyncio
async def test_snapshot_git_app_version_includes_git_info(script_mgr, fast_venv, tmp_path):
    git_info = {"owner": "o", "repo": "r", "branch": "main", "path": None, "repo_dir": "/fake"}
    staging = _write_project(tmp_path / "s", {"main.py": "print('v1')\n"})
    await script_mgr.setup_project_app("demo", staging, "git", git_info)
    script_mgr.snapshot_current_app_version("demo")
    versions = script_mgr.get_versions("demo")
    v = versions[0]
    assert v["kind"] == "app"
    assert v["app_type"] == "git"
    assert v["git"]["repo"] == "r"


# ---------------------------------------------------------------------------
# Legacy versions (no "kind" field) still roll back correctly via extension inference
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_rollback_legacy_script_version_without_kind(script_mgr, app):
    """Simulate a version dict that was created before the 'kind' tag was added."""
    script_mgr.upsert_script_from_text("mybot", "print('v1')\n")
    py_file = script_mgr.script_file_path("mybot")

    # Manually inject a legacy version dict (no "kind") pointing at the current .py file.
    import shutil as _shutil
    from pathlib import Path as P
    vdir = script_mgr.versions_dir_for("mybot")
    ver_file = vdir / "mybot_legacy.py"
    _shutil.copy2(py_file, ver_file)
    s = script_mgr.get_script("mybot")
    s.setdefault("versions", []).insert(0, {"ts": "01.01.2000-00-00-00", "file": str(ver_file)})
    script_mgr.save_meta()

    script_mgr.upsert_script_from_text("mybot", "print('v2')\n")

    msg = await script_mgr.rollback_to("mybot", "01.01.2000-00-00-00")
    assert "Rolled back" in msg
    assert "v1" in py_file.read_text()


# ---------------------------------------------------------------------------
# "Change Source" — pending_source_change state and stash routing
# ---------------------------------------------------------------------------

def test_pending_source_change_is_set_and_cleared(script_mgr):
    """pending_source_change stores the target id and can be popped."""
    script_mgr.upsert_script_from_text("recap", "print('v1')\n")
    user_id = 42
    script_mgr.pending_source_change[user_id] = "recap"
    assert script_mgr.pending_source_change.get(user_id) == "recap"
    result = script_mgr.pending_source_change.pop(user_id, None)
    assert result == "recap"
    assert user_id not in script_mgr.pending_source_change


@pytest.mark.asyncio
async def test_change_source_archive_with_mismatched_name_stashes_with_force_target(
    script_mgr, fast_venv, tmp_path
):
    """An archive uploaded while a source-change is pending should be stashed under
    the force_target_id, even when the archive filename differs from the app id."""
    # Pre-existing script app called "recap"
    script_mgr.upsert_script_from_text("recap", "print('old')\n")
    assert script_mgr.get_script("recap")["type"] == "script"

    # Prepare a project directory (simulating what handle_archive_upload would have
    # extracted — the stash path is what we care about, not the bot's UI flow).
    staging = _write_project(tmp_path / "stage", {"main.py": "print('new')\n"})

    # Simulate what handle_archive_upload does when force_target_id="recap" is set:
    # it overrides app_id, detects no *conflicting* type (same id, same type != project
    # → old_type=None), but force_target_id is truthy so it stashes + asks for confirmation.
    old_type = script_mgr.conflicting_type("recap", "project")  # "script" — a conflict!
    assert old_type == "script"

    script_mgr._set_pending_conversion("recap", {
        "action": "archive",
        "staged_root": str(staging),
        "staging": None,
        "old_type": old_type,
    })

    assert "recap" in script_mgr.pending_type_conversion
    stash = script_mgr.pending_type_conversion["recap"]
    assert stash["action"] == "archive"
    assert stash["old_type"] == "script"


@pytest.mark.asyncio
async def test_change_source_same_type_git_stashes_for_confirmation(
    script_mgr, fast_venv, tmp_path
):
    """Re-pointing a git app at a *different* repo (same type) via Change Source should
    still land in pending_type_conversion so the user must confirm."""
    git_info = {"owner": "acme", "repo": "py-bots-recap", "branch": "main", "path": None, "repo_dir": "/fake"}
    staging = _write_project(tmp_path / "stage", {"main.py": "print('new')\n"})
    await script_mgr.setup_project_app("recap", staging, "git", git_info)
    assert script_mgr.get_script("recap")["type"] == "git"

    # force_target_id is set → no conflicting type (same type), but needs_confirm is True.
    # Simulate the stash the handler would create.
    effective_old_type = (script_mgr.get_script("recap") or {}).get("type", "git")
    assert effective_old_type == "git"

    new_git_info = {"owner": "acme", "repo": "totally-different-repo", "branch": "main", "path": None, "repo_dir": "/fake2"}
    script_mgr._set_pending_conversion("recap", {
        "action": "git",
        "staged_root": str(staging),
        "git_info": new_git_info,
        "old_type": effective_old_type,
    })

    stash = script_mgr.pending_type_conversion.get("recap")
    assert stash is not None
    assert stash["git_info"]["repo"] == "totally-different-repo"


@pytest.mark.asyncio
async def test_change_source_git_to_script_conversion_stashes_correctly(
    script_mgr, fast_venv, tmp_path
):
    """Change Source: git app → .py file — stash should carry action=script_file."""
    git_info = {"owner": "o", "repo": "r", "branch": "main", "path": None, "repo_dir": "/fake"}
    staging = _write_project(tmp_path / "stage", {"main.py": "print('old')\n"})
    await script_mgr.setup_project_app("mybot", staging, "git", git_info)

    file_bytes = b"print('new')\n"
    old_type = script_mgr.conflicting_type("mybot", "script")
    assert old_type == "git"

    script_mgr._set_pending_conversion("mybot", {
        "action": "script_file",
        "file_bytes": file_bytes,
        "original_name": "something_else.py",
        "old_type": old_type,
    })

    stash = script_mgr.pending_type_conversion["mybot"]
    assert stash["action"] == "script_file"
    assert stash["file_bytes"] == file_bytes
    assert stash["old_type"] == "git"


@pytest.mark.asyncio
async def test_change_source_confirmed_converts_app(script_mgr, fast_venv, tmp_path):
    """After the user confirms via convtype:yes, setup_project_app should run and
    the app type should flip from script to project."""
    script_mgr.upsert_script_from_text("recap", "print('old')\n")

    staging = _write_project(tmp_path / "stage", {"main.py": "print('new')\n"})
    script_mgr._set_pending_conversion("recap", {
        "action": "archive",
        "staged_root": str(staging),
        "staging": None,
        "old_type": "script",
    })

    # Simulate convtype:yes — pop the stash and run the conversion.
    stash = script_mgr.pending_type_conversion.pop("recap")
    result = await script_mgr.setup_project_app("recap", Path(stash["staged_root"]), "project")

    assert result.get("status") != "error"
    assert script_mgr.get_script("recap")["type"] == "project"
    # Old .py file should be gone (cleanup ran inside setup_project_app)
    assert not script_mgr.script_file_path("recap").exists()


def test_change_source_cancel_clears_pending_source_change(script_mgr):
    """/cancel should clear pending_source_change without touching anything else."""
    script_mgr.upsert_script_from_text("recap", "print('v1')\n")
    user_id = 7
    script_mgr.pending_source_change[user_id] = "recap"
    script_mgr.pending_new[user_id] = "other"

    # Simulate /cancel logic
    cleared = []
    if script_mgr.pending_source_change.pop(user_id, None) is not None:
        cleared.append("source change")
    if script_mgr.pending_new.pop(user_id, None) is not None:
        cleared.append("new script")

    assert "source change" in cleared
    assert "new script" in cleared
    assert user_id not in script_mgr.pending_source_change
    assert user_id not in script_mgr.pending_new
