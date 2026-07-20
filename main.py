# script_manager_bot.py
# -*- coding: utf-8 -*-
"""
Telegram Script Manager Bot

Features:
- Manage Python scripts: upload/edit/run/stop/logs/env/autostart
- Apps from archives (.zip/.tar.gz/.tgz/.tar/.7z): auto-extract, detect requirements.txt,
  detect entry point (main.py / single file / ask), each app gets its own isolated venv
- Apps from GitHub repos (optionally a subfolder): clone, detect same as above, plus a
  Sync button to re-pull latest code and offer to install any new dependencies
- Script versioning: keep last 10 versions with timestamps + rollback menu (also for apps)
- Better /list and /menu with emojis and extra info
- /monitoring: CPU/MEM (and some optional stats) for running scripts (uses psutil if installed)
- pip installs are stored in requirements.txt and installed on manager startup

Env:
- BOT_TOKEN (required)
- OWNER_ID (required, int)
- DATA_DIR (optional, default: /data)
"""

import os
import sys
import json
import re
import shlex
import string
import asyncio
import logging
import logging.handlers
import shutil
import tarfile
import zipfile
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from collections import deque
from datetime import datetime

import pgdb

from telegram import Update, InlineKeyboardMarkup, InlineKeyboardButton
from telegram.ext import (
  Application,
  CommandHandler,
  MessageHandler,
  CallbackQueryHandler,
  ContextTypes,
  filters,
)

# Optional monitoring dependency
try:
  import psutil  # type: ignore
except Exception:
  psutil = None  # type: ignore


# Paths
DATA_DIR = Path(os.getenv("DATA_DIR", "/data"))
SCRIPTS_DIR = DATA_DIR / "scripts"
LOGS_DIR = DATA_DIR / "logs"
VERSIONS_DIR = DATA_DIR / "versions"
META_FILE = DATA_DIR / "meta.json"
REQUIREMENTS_FILE = DATA_DIR / "requirements.txt"

# New: multi-file apps (from archives or GitHub repos), each with its own isolated venv
APPS_DIR = DATA_DIR / "apps"
APP_VERSIONS_DIR = DATA_DIR / "app_versions"
VENVS_DIR = DATA_DIR / "venvs"
GITREPOS_DIR = DATA_DIR / "gitrepos"
TMP_DIR = DATA_DIR / "tmp"

DATA_DIR.mkdir(parents=True, exist_ok=True)
SCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)
VERSIONS_DIR.mkdir(parents=True, exist_ok=True)
APPS_DIR.mkdir(parents=True, exist_ok=True)
APP_VERSIONS_DIR.mkdir(parents=True, exist_ok=True)
VENVS_DIR.mkdir(parents=True, exist_ok=True)
GITREPOS_DIR.mkdir(parents=True, exist_ok=True)
TMP_DIR.mkdir(parents=True, exist_ok=True)

# Directories to ignore when scanning a project for .py files / requirements.txt
IGNORED_PROJECT_DIRS = {
    ".git", "__pycache__", "venv", ".venv", "env", ".env", "node_modules",
    ".idea", ".vscode", ".mypy_cache", ".pytest_cache", "site-packages",
    "dist", "build", ".github", "__MACOSX",
}

ARCHIVE_EXTENSIONS = (".tar.gz", ".tgz", ".tar", ".zip", ".7z")

GITHUB_TREE_URL_RE = re.compile(
    r'^https?://github\.com/(?P<owner>[^/\s]+)/(?P<repo>[^/\s]+?)(?:\.git)?/tree/(?P<branch>[^/\s]+)/(?P<path>[^\s]+?)/?$'
)
GITHUB_REPO_URL_RE = re.compile(
    r'^https?://github\.com/(?P<owner>[^/\s]+)/(?P<repo>[^/\s]+?)(?:\.git)?/?$'
)

# Logging
MAX_LOG_SIZE_BYTES = 100 * 1024  # cap bot.log and per-script logs at 100 KB each
LOG_TRIM_INTERVAL_SECONDS = 60

bot_log_path = DATA_DIR / "bot.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
      logging.StreamHandler(sys.stdout),
      logging.handlers.RotatingFileHandler(
          str(bot_log_path), maxBytes=MAX_LOG_SIZE_BYTES, backupCount=1, encoding="utf-8"
      ),
    ],
)
logger = logging.getLogger("script-bot")


def trim_file_to_tail(path: Path, max_bytes: int = MAX_LOG_SIZE_BYTES) -> None:
  """Keep only the last `max_bytes` of a file, dropping the oldest content.

  Used for per-script log files, which are written to directly by subprocess
  stdout/stderr (not via the logging module), so they need their own cap.
  """
  try:
    size = path.stat().st_size
  except FileNotFoundError:
    return
  if size <= max_bytes:
    return
  try:
    with path.open("r+b") as f:
      f.seek(-max_bytes, os.SEEK_END)
      f.readline()  # drop partial line so the tail starts cleanly
      tail = f.read()
      f.seek(0)
      f.write(tail)
      f.truncate()
  except Exception:
    logger.exception("Failed to trim log file %s", path)


async def trim_script_logs_periodically() -> None:
  while True:
    try:
      for log_file in LOGS_DIR.glob("*.log"):
        trim_file_to_tail(log_file)
    except Exception:
      logger.exception("Log trim sweep failed")
    await asyncio.sleep(LOG_TRIM_INTERVAL_SECONDS)

BOT_TOKEN = os.getenv("BOT_TOKEN")
OWNER_ID_ENV = os.getenv("OWNER_ID")
if not BOT_TOKEN:
  raise RuntimeError("BOT_TOKEN env is required")
if not OWNER_ID_ENV:
  raise RuntimeError("OWNER_ID env is required")
OWNER_ID = int(OWNER_ID_ENV)

# Telegram callback_data limit is 64 bytes-ish; keep it short
MAX_VERSIONS_PER_SCRIPT = 10
LOG_TAIL_LINES = 10


def is_authorized(update: Update) -> bool:
  user = update.effective_user
  ok = bool(user and user.id == OWNER_ID)
  if not ok:
    logger.warning("Unauthorized access attempt from user_id=%s", getattr(user, "id", None))
  return ok


def now_ts_str() -> str:
  # Example: 01.12.2025-12-25-09
  return datetime.now().strftime("%d.%m.%Y-%H-%M-%S")


def pretty_dt_from_ts(ts: str) -> str:
  # Accept "dd.mm.yyyy-HH-MM-SS"
  try:
    d = datetime.strptime(ts, "%d.%m.%Y-%H-%M-%S")
    return d.strftime("%d.%m.%Y %H:%M:%S")
  except Exception:
    return ts


def safe_trim_text(text: str, limit: int = 3800) -> str:
  # Telegram message hard limit is 4096; keep some headroom.
  if len(text) <= limit:
    return text
  return "...\n" + text[-limit:]


def extract_env_keys(source: str) -> set[str]:
  keys: set[str] = set()
  pattern1 = r"(?:os\.)?(?:getenv|environ\.get)\(\s*['\"]([A-Za-z_][A-Za-z0-9_]*)['\"]"
  pattern2 = r"(?:os\.)?environ\[\s*['\"]([A-Za-z_][A-Za-z0-9_]*)['\"]\s*\]"
  for pat in (pattern1, pattern2):
    for m in re.findall(pat, source):
      keys.add(m)
  return keys


def expand_env_values(values: Dict[str, str], base: Dict[str, str]) -> Dict[str, str]:
  """Expand $VAR / ${VAR} references in *values* using *base* as the lookup.

  Unknown references are left untouched (safe_substitute).  This lets users
  write e.g. ``PG_USER=${PG_USERNAME}`` in their per-app env and have it
  resolve to the injected Postgres username at runtime.
  """
  result: Dict[str, str] = {}
  for k, v in values.items():
    try:
      result[k] = string.Template(v).safe_substitute(base)
    except Exception:
      result[k] = v
  return result


def read_text_file(path: Path) -> str:
  return path.read_text(encoding="utf-8", errors="replace")


def write_text_file_atomic(path: Path, content: str) -> None:
  tmp = path.with_suffix(path.suffix + ".tmp")
  tmp.write_text(content, encoding="utf-8")
  tmp.replace(path)


def sanitize_id(name: str) -> str:
  # Used only for NEW app ids (archives / github repos) - kept separate from the
  # legacy script id derivation to not change behavior of existing scripts (BWC).
  name = (name or "").strip().replace(" ", "_")
  name = re.sub(r"[^A-Za-z0-9_.-]", "_", name)
  name = name.strip("_.- ")
  return name or "app"


def venv_python_path(venv_dir: Path) -> Path:
  if os.name == "nt":
    return venv_dir / "Scripts" / "python.exe"
  return venv_dir / "bin" / "python"


async def create_venv(venv_dir: Path) -> Tuple[bool, str]:
  py = venv_python_path(venv_dir)
  if py.exists():
    return True, "venv already exists"
  venv_dir.parent.mkdir(parents=True, exist_ok=True)
  logger.info("Creating venv at %s", venv_dir)
  proc = await asyncio.create_subprocess_exec(
      sys.executable,
      "-m",
      "venv",
      "--system-site-packages",
      str(venv_dir),
      stdout=asyncio.subprocess.PIPE,
      stderr=asyncio.subprocess.STDOUT,
  )
  out, _ = await proc.communicate()
  text = out.decode("utf-8", errors="replace")
  ok = proc.returncode == 0 and py.exists()
  if not ok:
    logger.error("venv creation failed for %s: %s", venv_dir, text)
  return ok, text


def strip_archive_ext(filename: str) -> str:
  lower = filename.lower()
  for ext in ARCHIVE_EXTENSIONS:
    if lower.endswith(ext):
      return filename[: -len(ext)]
  return Path(filename).stem


def archive_ext_of(filename: str) -> str:
  # Path(filename).suffix would truncate ".tar.gz" down to ".gz" - keep the full
  # multi-part extension so downstream extraction picks the correct code path.
  lower = filename.lower()
  for ext in ARCHIVE_EXTENSIONS:
    if lower.endswith(ext):
      return filename[len(filename) - len(ext):]
  return Path(filename).suffix


def _validate_zip_members(zf: "zipfile.ZipFile", dest_dir: Path) -> None:
  dest_resolved = dest_dir.resolve()
  for name in zf.namelist():
    target = (dest_dir / name).resolve()
    if not str(target).startswith(str(dest_resolved)):
      raise ValueError(f"Unsafe path in archive: {name}")


def _validate_tar_members(tf: "tarfile.TarFile", dest_dir: Path) -> None:
  dest_resolved = dest_dir.resolve()
  for member in tf.getmembers():
    target = (dest_dir / member.name).resolve()
    if not str(target).startswith(str(dest_resolved)):
      raise ValueError(f"Unsafe path in archive: {member.name}")


def extract_archive(archive_path: Path, dest_dir: Path) -> None:
  dest_dir.mkdir(parents=True, exist_ok=True)
  lower = str(archive_path).lower()
  if lower.endswith(".zip"):
    with zipfile.ZipFile(archive_path) as zf:
      _validate_zip_members(zf, dest_dir)
      zf.extractall(dest_dir)
  elif lower.endswith(".tar.gz") or lower.endswith(".tgz"):
    with tarfile.open(archive_path, "r:gz") as tf:
      _validate_tar_members(tf, dest_dir)
      tf.extractall(dest_dir, filter="data")
  elif lower.endswith(".tar"):
    with tarfile.open(archive_path, "r:") as tf:
      _validate_tar_members(tf, dest_dir)
      tf.extractall(dest_dir, filter="data")
  elif lower.endswith(".7z"):
    try:
      import py7zr  # type: ignore
    except ImportError as e:
      raise RuntimeError("py7zr is not installed on the manager, cannot extract .7z archives") from e
    with py7zr.SevenZipFile(archive_path, mode="r") as zf:
      names = zf.getnames()
      dest_resolved = dest_dir.resolve()
      for name in names:
        target = (dest_dir / name).resolve()
        if not str(target).startswith(str(dest_resolved)):
          raise ValueError(f"Unsafe path in archive: {name}")
      zf.extractall(path=dest_dir)
  else:
    raise ValueError(f"Unsupported archive type: {archive_path.suffix}")


def find_extraction_root(staging: Path) -> Path:
  # Many archives (e.g. GitHub zip exports) wrap everything in one top-level folder.
  entries = [p for p in staging.iterdir() if p.name not in IGNORED_PROJECT_DIRS]
  if len(entries) == 1 and entries[0].is_dir():
    return entries[0]
  return staging


def discover_python_files(root: Path) -> List[str]:
  result: List[str] = []
  root = root.resolve()
  for dirpath, dirnames, filenames in os.walk(root):
    dirnames[:] = [d for d in dirnames if d not in IGNORED_PROJECT_DIRS and not d.startswith(".")]
    for fn in filenames:
      if fn.lower().endswith(".py"):
        rel = str(Path(dirpath, fn).relative_to(root)).replace("\\", "/")
        result.append(rel)
  result.sort()
  return result


def find_requirements_file(root: Path) -> Optional[Path]:
  direct = root / "requirements.txt"
  if direct.exists():
    return direct
  for dirpath, dirnames, filenames in os.walk(root):
    dirnames[:] = [d for d in dirnames if d not in IGNORED_PROJECT_DIRS and not d.startswith(".")]
    for fn in filenames:
      if fn.lower() == "requirements.txt":
        return Path(dirpath) / fn
  return None


def find_requirements_pkgs(root: Path) -> List[str]:
  req = find_requirements_file(root)
  if not req:
    return []
  try:
    return [
        line.strip()
        for line in read_text_file(req).splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
  except Exception:
    return []


def pick_entry_from_candidates(
    py_files: List[str],
    name_hints: Optional[List[str]] = None,
) -> Tuple[Optional[str], bool]:
  """Returns (entry_or_None, ambiguous).

  Auto-detection rules, in order:
  1. A file literally named main.py (shallowest one wins).
  2. A file whose name (without .py) matches one of ``name_hints`` — e.g. the
     app id, the repo name, or the (sub)folder name — since projects often
     name their entry script after the project itself (e.g. myapp/myapp.py).
  3. If there's exactly one .py file at all, that one.
  Otherwise it's ambiguous and the caller should ask the user.
  """
  mains = [f for f in py_files if Path(f).name.lower() == "main.py"]
  if mains:
    mains.sort(key=lambda p: p.count("/"))
    return mains[0], False

  hints = {h.strip().lower() for h in (name_hints or []) if h and h.strip()}
  if hints:
    named = [f for f in py_files if Path(f).stem.lower() in hints]
    if named:
      named.sort(key=lambda p: p.count("/"))
      return named[0], False

  if len(py_files) == 1:
    return py_files[0], False
  if len(py_files) == 0:
    return None, False
  return None, True


def parse_github_url(text: str) -> Optional[Dict[str, Any]]:
  url = (text or "").strip()
  m = GITHUB_TREE_URL_RE.match(url)
  if m:
    owner, repo, branch, path = m.group("owner"), m.group("repo"), m.group("branch"), m.group("path")
    folder = path.rstrip("/").split("/")[-1]
    app_id = sanitize_id(f"{repo}-{folder}")
    return {"owner": owner, "repo": repo, "branch": branch, "path": path, "app_id": app_id}
  m = GITHUB_REPO_URL_RE.match(url)
  if m:
    owner, repo = m.group("owner"), m.group("repo")
    app_id = sanitize_id(repo)
    return {"owner": owner, "repo": repo, "branch": None, "path": None, "app_id": app_id}
  return None


async def _run_cmd(*args: str) -> Tuple[bool, str]:
  proc = await asyncio.create_subprocess_exec(
      *args,
      stdout=asyncio.subprocess.PIPE,
      stderr=asyncio.subprocess.STDOUT,
  )
  out, _ = await proc.communicate()
  text = out.decode("utf-8", errors="replace")
  return proc.returncode == 0, text


async def git_clone_or_pull(repo_url: str, branch: Optional[str], dest: Path) -> Tuple[bool, str]:
  if (dest / ".git").exists():
    ok, out1 = await _run_cmd("git", "-C", str(dest), "fetch", "--depth", "1", "origin", branch or "HEAD")
    if not ok:
      return False, out1
    target = f"origin/{branch}" if branch else "FETCH_HEAD"
    ok, out2 = await _run_cmd("git", "-C", str(dest), "reset", "--hard", target)
    return ok, out1 + "\n" + out2
  # Clean up any partial/failed previous clone attempt so `git clone` doesn't
  # refuse to write into a non-empty directory.
  if dest.exists():
    shutil.rmtree(dest, ignore_errors=True)
  dest.parent.mkdir(parents=True, exist_ok=True)
  cmd = ["git", "clone", "--depth", "1"]
  if branch:
    cmd += ["--branch", branch]
  cmd += [repo_url, str(dest)]
  return await _run_cmd(*cmd)


async def git_current_branch(repo_dir: Path) -> str:
  ok, out = await _run_cmd("git", "-C", str(repo_dir), "rev-parse", "--abbrev-ref", "HEAD")
  if ok:
    branch = out.strip()
    if branch and branch != "HEAD":
      return branch
  return "main"


class ScriptManager:
  def __init__(self) -> None:
    self.meta: Dict[str, Any] = {"global_env": {}, "scripts": {}}

    # runtime state
    self.processes: Dict[str, asyncio.subprocess.Process] = {}
    self.log_handles: Dict[str, Any] = {}  # file handles
    self.watch_tasks: Dict[str, asyncio.Task] = {}
    self.start_times: Dict[str, float] = {}  # monotonic-ish not required; store epoch
    self.psutil_procs: Dict[str, Any] = {}  # psutil.Process objects

    # pending user flows
    self.pending_new: Dict[int, str] = {}
    self.pending_edit: Dict[int, str] = {}
    self.pending_env_value: Dict[int, Dict[str, str]] = {}
    self.pending_pip: Dict[int, Optional[str]] = {}
    self.pending_entry_choice: Dict[str, List[str]] = {}  # app_id -> candidate py files
    # keyed by script_id; holds the staged data for a cross-type conversion awaiting
    # the user's ✅ Convert / ❌ Cancel confirmation
    self.pending_type_conversion: Dict[str, Dict[str, Any]] = {}
    # keyed by user_id; value is the target script_id that the user wants to
    # replace with a new source (file / archive / GitHub link)
    self.pending_source_change: Dict[int, str] = {}

    self.load_meta()

  def load_meta(self) -> None:
    if META_FILE.exists():
      try:
        self.meta = json.loads(read_text_file(META_FILE))
      except Exception:
        logger.exception("Failed to load meta.json, starting with empty meta")
        self.meta = {"global_env": {}, "scripts": {}}

    # Ensure structure
    self.meta.setdefault("global_env", {})
    self.meta.setdefault("scripts", {})

    # Ensure per-script keys
    for sid, s in self.meta["scripts"].items():
      if not isinstance(s, dict):
        self.meta["scripts"][sid] = {}
        s = self.meta["scripts"][sid]
      s.setdefault("env", {})
      s.setdefault("autostart", False)
      s.setdefault("versions", [])  # list[{ts, file}]
      s.setdefault("updated_at", None)
      s.setdefault("type", "script")  # "script" (legacy .py) | "project" (archive) | "git"

    self.save_meta()

  def save_meta(self) -> None:
    try:
      write_text_file_atomic(META_FILE, json.dumps(self.meta, indent=2, ensure_ascii=False))
    except Exception:
      logger.exception("Failed to save meta.json")

  def list_scripts(self) -> Dict[str, Any]:
    return self.meta.get("scripts", {})

  def get_global_env(self) -> Dict[str, str]:
    return self.meta.get("global_env", {})

  def set_global_env(self, key: str, value: str) -> None:
    self.meta.setdefault("global_env", {})[key] = value
    self.save_meta()
    logger.info("Set global env %s", key)

  def del_global_env(self, key: str) -> bool:
    env = self.meta.setdefault("global_env", {})
    existed = key in env
    env.pop(key, None)
    self.save_meta()
    if existed:
      logger.info("Deleted global env %s", key)
    return existed

  # requirements.txt handling

  def _read_requirements_lines(self, path: Optional[Path] = None) -> List[str]:
    p = path or REQUIREMENTS_FILE
    if not p.exists():
      return []
    lines = []
    for raw in read_text_file(p).splitlines():
      line = raw.strip()
      if not line or line.startswith("#"):
        continue
      lines.append(line)
    return lines

  def add_requirements(self, pkgs: List[str], path: Optional[Path] = None) -> bool:
    if not pkgs:
      return False
    p = path or REQUIREMENTS_FILE
    existing = self._read_requirements_lines(p)
    existing_set = set(existing)
    changed = False
    for pkg in pkgs:
      if pkg and pkg not in existing_set:
        existing.append(pkg)
        existing_set.add(pkg)
        changed = True
    if changed:
      content = "\n".join(existing) + "\n"
      write_text_file_atomic(p, content)
    return changed

  async def ensure_requirements_installed(self) -> None:
    # Backward compatibility: migrate old meta pip_packages -> requirements.txt (if present)
    old_pkgs = self.meta.get("pip_packages")
    if isinstance(old_pkgs, list) and old_pkgs:
      migrated = [str(x).strip() for x in old_pkgs if str(x).strip()]
      if migrated:
        self.add_requirements(migrated)
      # keep meta key but stop using it
      logger.info("Migrated meta pip_packages to requirements.txt (%d items)", len(migrated))

    if not REQUIREMENTS_FILE.exists():
      logger.info("No requirements.txt, skipping startup pip install")
      return

    req_lines = self._read_requirements_lines()
    if not req_lines:
      logger.info("requirements.txt is empty, skipping startup pip install")
      return

    logger.info("Ensuring requirements installed on startup: %s", str(REQUIREMENTS_FILE))
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "pip",
        "install",
        "-r",
        str(REQUIREMENTS_FILE),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    out, _ = await proc.communicate()
    text = out.decode("utf-8", errors="replace")
    text = safe_trim_text(text, 3500)
    logger.info("Startup pip install exit=%s, output:\n%s", proc.returncode, text)

  async def ensure_app_environments(self) -> None:
    # Make sure every project/git app has a working venv + its deps installed,
    # e.g. after a fresh volume or container rebuild.
    scripts = self.meta.get("scripts", {})
    for sid, s in list(scripts.items()):
      if not isinstance(s, dict) or s.get("type") not in ("project", "git"):
        continue
      root = Path(s.get("root_dir") or "")
      if not root.exists():
        logger.warning("App %s root_dir missing (%s), skipping env setup", sid, root)
        continue
      venv_dir = Path(s.get("venv_dir") or self.app_venv_dir(sid))
      ok, out = await create_venv(venv_dir)
      if not ok:
        logger.error("Failed to prepare venv for app %s: %s", sid, safe_trim_text(out, 1000))
        continue
      req_file = root / "requirements.txt"
      if req_file.exists():
        python_exe = venv_python_path(venv_dir)
        proc = await asyncio.create_subprocess_exec(
            str(python_exe),
            "-m",
            "pip",
            "install",
            "-r",
            str(req_file),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        out2, _ = await proc.communicate()
        logger.info(
            "Startup pip install for app %s exit=%s:\n%s",
            sid,
            proc.returncode,
            safe_trim_text(out2.decode("utf-8", errors="replace"), 2000),
        )

  # scripts

  def _ensure_script_meta(self, script_id: str, file_path: str) -> None:
    scripts = self.meta.setdefault("scripts", {})
    script = scripts.get(script_id, {})
    if not isinstance(script, dict):
      script = {}
    script.setdefault("env", {})
    script.setdefault("autostart", False)
    script.setdefault("versions", [])
    script.setdefault("updated_at", None)
    script["type"] = "script"
    script["file"] = file_path
    scripts[script_id] = script
    self.save_meta()

  def get_script(self, script_id: str) -> Optional[Dict[str, Any]]:
    return self.meta.get("scripts", {}).get(script_id)

  def get_type(self, script_id: str) -> str:
    s = self.get_script(script_id)
    return (s or {}).get("type", "script")

  def is_project_type(self, script_id: str) -> bool:
    return self.get_type(script_id) in ("project", "git")

  def script_file_path(self, script_id: str) -> Path:
    return SCRIPTS_DIR / f"{script_id}.py"

  def versions_dir_for(self, script_id: str) -> Path:
    d = VERSIONS_DIR / script_id
    d.mkdir(parents=True, exist_ok=True)
    return d

  # apps (archives / github repos) - each gets its own root dir + isolated venv

  def app_root_dir(self, script_id: str) -> Path:
    return APPS_DIR / script_id

  def app_venv_dir(self, script_id: str) -> Path:
    return VENVS_DIR / script_id

  def app_venv_python(self, script_id: str) -> Path:
    return venv_python_path(self.app_venv_dir(script_id))

  def app_versions_dir_for(self, script_id: str) -> Path:
    d = APP_VERSIONS_DIR / script_id
    d.mkdir(parents=True, exist_ok=True)
    return d

  def app_requirements_file(self, script_id: str) -> Optional[Path]:
    script = self.get_script(script_id)
    if not script:
      return None
    root = Path(script.get("root_dir") or "")
    if not root.exists():
      return None
    return root / "requirements.txt"

  def _ensure_app_meta(
      self,
      script_id: str,
      app_type: str,
      root_dir: Path,
      entry: Optional[str],
      venv_dir: Path,
      git_info: Optional[Dict[str, Any]] = None,
  ) -> None:
    scripts = self.meta.setdefault("scripts", {})
    script = scripts.get(script_id, {})
    if not isinstance(script, dict):
      script = {}
    script.setdefault("env", {})
    script.setdefault("autostart", False)
    script.setdefault("versions", [])
    script.setdefault("updated_at", None)
    script["type"] = app_type
    script["root_dir"] = str(root_dir)
    script["entry"] = entry
    script["venv_dir"] = str(venv_dir)
    if git_info is not None:
      script["git"] = git_info
    scripts[script_id] = script
    self.save_meta()

  async def setup_project_app(
      self,
      script_id: str,
      extracted_root: Path,
      app_type: str,
      git_info: Optional[Dict[str, Any]] = None,
  ) -> Dict[str, Any]:
    """Moves extracted_root -> APPS_DIR/script_id (snapshotting previous content if any),
    prepares an isolated venv, and detects the entry point + requirements."""
    dest_root = self.app_root_dir(script_id)

    # Stop any running process before replacing files on disk.
    if self.script_running(script_id):
      await self.stop_script(script_id)

    existing = self.get_script(script_id)
    if existing:
      if existing.get("type", "script") == "script":
        self.snapshot_current_version(script_id)
      else:
        self.snapshot_current_app_version(script_id)
      # Remove artifacts that belong exclusively to the old type.
      self._clear_type_artifacts(script_id, existing, "app")

    extracted_root_name = extracted_root.name
    if dest_root.exists():
      shutil.rmtree(dest_root, ignore_errors=True)
    dest_root.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(extracted_root), str(dest_root))

    py_files = discover_python_files(dest_root)
    name_hints = [script_id, extracted_root_name]
    if git_info:
      name_hints.append(git_info.get("repo"))
      if git_info.get("path"):
        name_hints.append(Path(git_info["path"]).name)
    entry, ambiguous = pick_entry_from_candidates(py_files, name_hints)

    # Preserve a previously confirmed entry point (auto-detected or manually
    # chosen from an ambiguous list) across re-syncs / re-uploads. Without
    # this, re-running auto-detection on every sync can flip back to
    # "ambiguous" (e.g. the repo has several top-level .py files and no
    # main.py) and force the user to re-pick even though nothing meaningful
    # about the entry point actually changed.
    prev_entry = (existing or {}).get("entry")
    if prev_entry and prev_entry in py_files:
      entry, ambiguous = prev_entry, False

    venv_dir = self.app_venv_dir(script_id)
    venv_ok, venv_out = await create_venv(venv_dir)

    self._ensure_app_meta(script_id, app_type, dest_root, entry, venv_dir, git_info)
    script = self.get_script(script_id)
    if script:
      script["updated_at"] = now_ts_str()
      self.save_meta()

    logger.info("App %s set up (type=%s, entry=%s, ambiguous=%s)", script_id, app_type, entry, ambiguous)

    result: Dict[str, Any] = {
        "app_id": script_id,
        "venv_ok": venv_ok,
        "venv_out": venv_out,
        "requirements_pkgs": find_requirements_pkgs(dest_root),
        "py_files": py_files,
    }
    if ambiguous:
      self.pending_entry_choice[script_id] = py_files
      result["status"] = "ambiguous"
      result["candidates"] = py_files
    else:
      # A stale pending choice from a previous ambiguous setup (now resolved,
      # e.g. via a preserved entry) would otherwise dangle around forever.
      self.pending_entry_choice.pop(script_id, None)
      result["status"] = "ready"
      result["entry"] = entry
    return result

  async def install_app_requirements(self, script_id: str) -> str:
    script = self.get_script(script_id)
    if not script:
      return "❌ App not found."
    root = Path(script.get("root_dir") or "")
    req_file = root / "requirements.txt"
    if not req_file.exists():
      return "🟡 No requirements.txt found."

    venv_dir = Path(script.get("venv_dir") or self.app_venv_dir(script_id))
    python_exe = venv_python_path(venv_dir)
    if not python_exe.exists():
      ok, out = await create_venv(venv_dir)
      if not ok:
        return "❌ Failed to prepare venv:\n" + safe_trim_text(out, 1500)

    proc = await asyncio.create_subprocess_exec(
        str(python_exe),
        "-m",
        "pip",
        "install",
        "-r",
        str(req_file),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    out, _ = await proc.communicate()
    text = safe_trim_text(out.decode("utf-8", errors="replace"), 3200)
    return f"📥 pip install -r requirements.txt (exit={proc.returncode})\n{text}"

  def _push_version(
      self,
      script_id: str,
      version_file: Path,
      ts: str,
      *,
      kind: str = "script",
      app_type: Optional[str] = None,
      git: Optional[Dict[str, Any]] = None,
      entry_point: Optional[str] = None,
  ) -> None:
    script = self.get_script(script_id)
    if not script:
      return
    versions = script.setdefault("versions", [])
    if not isinstance(versions, list):
      versions = []
      script["versions"] = versions

    entry: Dict[str, Any] = {"ts": ts, "file": str(version_file), "kind": kind}
    if app_type is not None:
      entry["app_type"] = app_type
    if git is not None:
      entry["git"] = git
    if entry_point is not None:
      entry["entry"] = entry_point
    versions.insert(0, entry)

    # Trim versions list and delete old files
    while len(versions) > MAX_VERSIONS_PER_SCRIPT:
      tail = versions.pop()
      try:
        fp = Path(tail.get("file", ""))
        if fp.exists():
          fp.unlink()
      except Exception:
        logger.exception("Failed to delete old version file for %s", script_id)

    self.save_meta()

  @staticmethod
  def _unique_version_path(vdir: Path, script_id: str, ts: str, ext: str) -> Path:
    # now_ts_str() has 1-second resolution, so two snapshots taken in quick
    # succession (rapid edits, or a snapshot-before-rollback right after a
    # snapshot-on-upload) can land on the same name. Never reuse a version
    # file name - that would silently overwrite a still-referenced version.
    candidate = vdir / f"{script_id}_{ts}{ext}"
    n = 1
    while candidate.exists():
      candidate = vdir / f"{script_id}_{ts}_{n}{ext}"
      n += 1
    return candidate

  def snapshot_current_version(self, script_id: str) -> None:
    script = self.get_script(script_id)
    if not script:
      return
    current = script.get("file")
    if not current:
      return
    cur_path = Path(current)
    if not cur_path.exists():
      return

    ts = now_ts_str()
    vdir = self.versions_dir_for(script_id)
    version_file = self._unique_version_path(vdir, script_id, ts, ".py")
    try:
      shutil.copy2(cur_path, version_file)
      self._push_version(script_id, version_file, ts, kind="script")
      logger.info("Snapshot version for %s -> %s", script_id, version_file.name)
    except Exception:
      logger.exception("Failed to snapshot version for %s", script_id)

  def snapshot_current_app_version(self, script_id: str) -> None:
    script = self.get_script(script_id)
    if not script:
      return
    root = Path(script.get("root_dir") or "")
    if not root.exists():
      return

    ts = now_ts_str()
    vdir = self.app_versions_dir_for(script_id)
    version_file = self._unique_version_path(vdir, script_id, ts, ".tar.gz")
    try:
      with tarfile.open(version_file, "w:gz") as tf:
        tf.add(root, arcname=script_id)
      self._push_version(
          script_id,
          version_file,
          ts,
          kind="app",
          app_type=script.get("type"),
          git=script.get("git"),
          entry_point=script.get("entry"),
      )
      logger.info("Snapshot app version for %s -> %s", script_id, version_file.name)
    except Exception:
      logger.exception("Failed to snapshot app version for %s", script_id)

  def upsert_script_from_text(self, name: str, content: str) -> str:
    script_id = name.strip().replace(" ", "_")
    if not script_id:
      raise ValueError("empty script name")

    # Snapshot the current state before overwriting, then clean up old-type artifacts.
    existing = self.get_script(script_id)
    if existing:
      if existing.get("type", "script") != "script":
        self.snapshot_current_app_version(script_id)
        self._clear_type_artifacts(script_id, existing, "script")
      elif self.script_file_path(script_id).exists():
        self.snapshot_current_version(script_id)

    file_path = self.script_file_path(script_id)
    write_text_file_atomic(file_path, content)
    self._ensure_script_meta(script_id, str(file_path))

    # Update updated_at
    script = self.get_script(script_id)
    if script:
      script["updated_at"] = now_ts_str()
      self.save_meta()

    logger.info("Script %s saved from text", script_id)
    return script_id

  def upsert_script_from_file(
      self,
      original_name: str,
      content_path: Path,
      override_id: Optional[str] = None,
  ) -> str:
    if override_id:
      script_id = override_id
    else:
      stem = Path(original_name).stem
      script_id = stem.strip().replace(" ", "_")
      if not script_id:
        script_id = f"script_{len(self.meta.get('scripts', {})) + 1}"

    # Snapshot the current state before overwriting, then clean up old-type artifacts.
    existing = self.get_script(script_id)
    if existing:
      if existing.get("type", "script") != "script":
        self.snapshot_current_app_version(script_id)
        self._clear_type_artifacts(script_id, existing, "script")
      elif self.script_file_path(script_id).exists():
        self.snapshot_current_version(script_id)

    content = read_text_file(content_path)
    file_path = self.script_file_path(script_id)
    write_text_file_atomic(file_path, content)
    self._ensure_script_meta(script_id, str(file_path))

    script = self.get_script(script_id)
    if script:
      script["updated_at"] = now_ts_str()
      self.save_meta()

    logger.info("Script %s saved from file %s", script_id, original_name)
    return script_id

  def set_script_env(self, script_id: str, key: str, value: str) -> None:
    scripts = self.meta.setdefault("scripts", {})
    script = scripts.setdefault(script_id, {"env": {}, "autostart": False, "versions": [], "updated_at": None})
    env = script.setdefault("env", {})
    env[key] = value
    self.save_meta()
    logger.info("Set env %s for script %s", key, script_id)

  def del_script_env(self, script_id: str, key: str) -> bool:
    script = self.get_script(script_id)
    if not script:
      return False
    env = script.setdefault("env", {})
    existed = key in env
    env.pop(key, None)
    self.save_meta()
    if existed:
      logger.info("Deleted env %s for script %s", key, script_id)
    return existed

  def set_autostart(self, script_id: str, value: bool) -> bool:
    script = self.get_script(script_id)
    if not script:
      return False
    script["autostart"] = bool(value)
    self.save_meta()
    logger.info("Set autostart=%s for script %s", value, script_id)
    return True

  def script_running(self, script_id: str) -> bool:
    proc = self.processes.get(script_id)
    return bool(proc and proc.returncode is None)

  def log_path(self, script_id: str) -> Path:
    return LOGS_DIR / f"{script_id}.log"

  async def _watch_process(self, script_id: str, proc: asyncio.subprocess.Process) -> None:
    try:
      await proc.wait()
    except Exception:
      logger.exception("Wait failed for %s", script_id)
    finally:
      # Mark stop in logs
      try:
        with self.log_path(script_id).open("ab") as f:
          f.write(b"\n=== FINISHED ===\n")
      except Exception:
        logger.exception("Failed to write FINISHED marker for %s", script_id)

      # Close handle
      h = self.log_handles.pop(script_id, None)
      try:
        if h:
          h.flush()
          h.close()
      except Exception:
        pass

      # Cleanup runtime
      self.processes.pop(script_id, None)
      self.psutil_procs.pop(script_id, None)
      self.start_times.pop(script_id, None)
      self.watch_tasks.pop(script_id, None)
      logger.info("Process finished for %s (code=%s)", script_id, proc.returncode)

  async def start_script(self, script_id: str) -> str:
    script = self.get_script(script_id)
    if not script:
      return "❌ Script not found."

    proc = self.processes.get(script_id)
    if proc and proc.returncode is None:
      return "🟡 Already running."

    app_type = script.get("type", "script")
    if app_type == "script":
      file_path = script.get("file")
      if not file_path or not Path(file_path).exists():
        return "❌ Script file not found."
      python_exe = sys.executable
      cwd = str(SCRIPTS_DIR)
      run_target = file_path
    else:
      root = Path(script.get("root_dir") or "")
      entry = script.get("entry")
      if not entry:
        return "❌ Entry point not set. Choose a file to run first."
      entry_path = root / entry
      if not entry_path.exists():
        return f"❌ Entry file not found: {entry}"

      venv_dir = Path(script.get("venv_dir") or self.app_venv_dir(script_id))
      python_exe_path = venv_python_path(venv_dir)
      if not python_exe_path.exists():
        ok, out = await create_venv(venv_dir)
        if not ok:
          return "❌ Failed to create venv:\n" + safe_trim_text(out, 1500)
      python_exe = str(python_exe_path)
      cwd = str(root)
      run_target = str(entry_path)

    env = os.environ.copy()
    env.update(self.meta.get("global_env", {}))
    env.update(pgdb.app_pg_env(script_id))
    env.update(expand_env_values(script.get("env", {}), env))

    log_file = self.log_path(script_id)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    trim_file_to_tail(log_file)
    with log_file.open("ab") as f:
      f.write(b"\n=== START ===\n")

    # Keep handle open while process runs (important!)
    h = log_file.open("ab")
    self.log_handles[script_id] = h

    logger.info("Starting script %s via %s", script_id, python_exe)
    proc = await asyncio.create_subprocess_exec(
        python_exe,
        run_target,
        cwd=cwd,
        env=env,
        stdout=h,
        stderr=h,
    )
    self.processes[script_id] = proc
    self.start_times[script_id] = datetime.now().timestamp()

    # Prime psutil cpu% calculation
    if psutil is not None:
      try:
        p = psutil.Process(proc.pid)
        p.cpu_percent(interval=None)
        self.psutil_procs[script_id] = p
      except Exception:
        pass

    # Watcher
    self.watch_tasks[script_id] = asyncio.create_task(self._watch_process(script_id, proc))

    return f"✅ Started *{script_id}* (PID `{proc.pid}`)".replace("*", "")

  async def stop_script(self, script_id: str) -> str:
    proc = self.processes.get(script_id)
    if not proc:
      return "🟡 Not running."
    if proc.returncode is not None:
      self.processes.pop(script_id, None)
      return "🟡 Already finished."

    logger.info("Stopping script %s", script_id)
    try:
      proc.terminate()
      await asyncio.wait_for(proc.wait(), timeout=10)
    except asyncio.TimeoutError:
      proc.kill()
      await proc.wait()
    except Exception:
      logger.exception("Stop failed for %s", script_id)

    # FINISHED marker and cleanup will be done by watcher too, but do a quick marker here
    try:
      with self.log_path(script_id).open("ab") as f:
        f.write(b"\n=== STOP ===\n")
    except Exception:
      pass

    # Let watcher clean everything; but ensure state is consistent if watcher not running
    if script_id in self.watch_tasks:
      # watcher will do full cleanup
      return f"🛑 Stopped {script_id}."
    else:
      self.processes.pop(script_id, None)
      h = self.log_handles.pop(script_id, None)
      try:
        if h:
          h.flush()
          h.close()
      except Exception:
        pass
      self.psutil_procs.pop(script_id, None)
      self.start_times.pop(script_id, None)
      return f"🛑 Stopped {script_id}."

  async def restart_script(self, script_id: str) -> str:
    if self.script_running(script_id):
      s1 = await self.stop_script(script_id)
      s2 = await self.start_script(script_id)
      return f"{s1}\n{s2}"
    return await self.start_script(script_id)

  async def autostart_all(self) -> None:
    scripts = self.meta.get("scripts", {})
    for script_id, script in scripts.items():
      if isinstance(script, dict) and script.get("autostart"):
        try:
          logger.info("Autostarting script %s", script_id)
          await self.start_script(script_id)
        except Exception:
          logger.exception("Autostart failed for %s", script_id)

  def read_log_tail(self, script_id: str, lines: int = LOG_TAIL_LINES) -> str:
    path = self.log_path(script_id)
    if not path.exists():
      return "No logs yet."
    dq: deque[str] = deque(maxlen=lines)
    with path.open("r", encoding="utf-8", errors="replace") as f:
      for line in f:
        dq.append(line.rstrip("\n"))
    if not dq:
      return "No logs yet."
    return "\n".join(dq)

  def get_env_keys_for_script(self, script_id: str) -> List[str]:
    script = self.get_script(script_id)
    if not script:
      return []
    keys: set[str] = set(script.get("env", {}).keys())

    if script.get("type", "script") == "script":
      file_path = script.get("file")
      if file_path and Path(file_path).exists():
        try:
          source = read_text_file(Path(file_path))
          keys |= extract_env_keys(source)
        except Exception:
          logger.exception("Failed to extract env keys for %s", script_id)
    else:
      root = Path(script.get("root_dir") or "")
      if root.exists():
        for rel in discover_python_files(root):
          try:
            source = read_text_file(root / rel)
            keys |= extract_env_keys(source)
          except Exception:
            continue
    return sorted(keys)

  def get_versions(self, script_id: str) -> List[Dict[str, Any]]:
    script = self.get_script(script_id)
    if not script:
      return []
    versions = script.get("versions", [])
    if not isinstance(versions, list):
      return []
    out: List[Dict[str, Any]] = []
    for v in versions:
      if not isinstance(v, dict):
        continue
      ts = str(v.get("ts", "")).strip()
      fp = str(v.get("file", "")).strip()
      if ts and fp:
        entry: Dict[str, Any] = {"ts": ts, "file": fp}
        for extra_key in ("kind", "app_type", "git", "entry"):
          if extra_key in v:
            entry[extra_key] = v[extra_key]
        out.append(entry)
    return out[:MAX_VERSIONS_PER_SCRIPT]

  # --- type-conversion helpers -------------------------------------------------

  def _clear_type_artifacts(self, script_id: str, old_script: Dict[str, Any], new_kind: str) -> None:
    """Remove the on-disk artifacts and stale meta keys that belong exclusively
    to the *old* type of an app being converted to *new_kind* ("script"/"app").
    The caller is responsible for saving meta afterwards."""
    old_type = old_script.get("type", "script")
    old_kind = "script" if old_type == "script" else "app"
    if old_kind == new_kind:
      return  # nothing to do; same storage model

    if old_kind == "script":
      # script → app: remove the .py source file and its meta key
      old_file = old_script.get("file") or str(self.script_file_path(script_id))
      fp = Path(old_file)
      if fp.exists():
        try:
          fp.unlink()
        except Exception:
          logger.exception("_clear_type_artifacts: failed to delete old script file for %s", script_id)
      old_script.pop("file", None)
    else:
      # app → script: remove the app source tree, its venv, and (for git) the repo clone
      root_dir = old_script.get("root_dir")
      if root_dir:
        shutil.rmtree(root_dir, ignore_errors=True)
      venv_dir = old_script.get("venv_dir")
      if venv_dir:
        shutil.rmtree(venv_dir, ignore_errors=True)
      if old_type == "git":
        git_meta = old_script.get("git", {})
        repo_dir = git_meta.get("repo_dir") or str(GITREPOS_DIR / script_id)
        shutil.rmtree(repo_dir, ignore_errors=True)
      for key in ("root_dir", "venv_dir", "entry", "git"):
        old_script.pop(key, None)

  def conflicting_type(self, script_id: str, new_type: str) -> Optional[str]:
    """Return the existing type string if it differs from *new_type*, else None."""
    s = self.get_script(script_id)
    if not s:
      return None
    existing = s.get("type", "script")
    return existing if existing != new_type else None

  def _set_pending_conversion(self, script_id: str, stash: Dict[str, Any]) -> None:
    """Stash a pending type-conversion, cleaning up any previous stash for the same id."""
    self._clear_pending_conversion(script_id)
    self.pending_type_conversion[script_id] = stash

  def _clear_pending_conversion(self, script_id: str) -> None:
    """Discard a pending type-conversion stash and remove any staged temp directories."""
    stash = self.pending_type_conversion.pop(script_id, None)
    if not stash:
      return
    staged = stash.get("staged_root")
    if staged:
      shutil.rmtree(staged, ignore_errors=True)

  # --- rollback ---------------------------------------------------------------

  async def rollback_to(self, script_id: str, ts: str) -> str:
    """Roll back to a previous version.

    The target version's "kind" field (written by snapshot_current_version /
    snapshot_current_app_version) determines the restore strategy, NOT the
    current type.  This means rolling back past a type-conversion (e.g.
    script → project) fully reverts the type as well — including removing the
    new type's artifacts and restoring the old type's files and meta.

    Versions created before this migration have no "kind" tag; we fall back to
    inferring it from the file extension (.py ⇒ script, .tar.gz ⇒ app).
    """
    script = self.get_script(script_id)
    if not script:
      return "❌ Script not found."

    versions = self.get_versions(script_id)
    chosen = next((v for v in versions if v["ts"] == ts), None)
    if not chosen:
      return "❌ Version not found."

    version_path = Path(chosen["file"])
    if not version_path.exists():
      return "❌ Version file missing on disk."

    # Infer target kind from the version tag, falling back to the file extension
    # for versions created before this migration.
    target_kind: str = chosen.get("kind") or ("script" if version_path.suffix == ".py" else "app")
    current_type = script.get("type", "script")
    current_kind = "script" if current_type == "script" else "app"

    # --- Stage the restore data BEFORE snapshotting to avoid same-second
    # timestamp collisions (reading/extracting now guarantees the chosen version
    # is unaffected by the snapshot we're about to create).
    if target_kind == "script":
      version_bytes = version_path.read_bytes()
      tmp_extract = None
    else:
      tmp_extract = self.app_root_dir(script_id).parent / f"__rollback_tmp_{script_id}"
      shutil.rmtree(tmp_extract, ignore_errors=True)
      try:
        with tarfile.open(version_path, "r:gz") as tf:
          tf.extractall(tmp_extract, filter="data")
      except Exception:
        logger.exception("App rollback extract failed for %s", script_id)
        shutil.rmtree(tmp_extract, ignore_errors=True)
        return "❌ Rollback failed (extract error)."

    was_running = self.script_running(script_id)
    if was_running:
      await self.stop_script(script_id)

    # Snapshot current state so this rollback is itself re-revertible.
    if current_kind == "script":
      if self.script_file_path(script_id).exists():
        self.snapshot_current_version(script_id)
    else:
      cur_root = Path(script.get("root_dir") or self.app_root_dir(script_id))
      if cur_root.exists():
        self.snapshot_current_app_version(script_id)

    # When crossing a type boundary, remove the artifacts of the current type.
    if current_kind != target_kind:
      self._clear_type_artifacts(script_id, script, target_kind)

    # --- Materialize the target version ------------------------------------
    if target_kind == "script":
      try:
        dest_py = self.script_file_path(script_id)
        dest_py.parent.mkdir(parents=True, exist_ok=True)
        dest_py.write_bytes(version_bytes)
      except Exception:
        logger.exception("Rollback copy failed for %s", script_id)
        return "❌ Rollback failed (copy error)."

      script["type"] = "script"
      script["file"] = str(self.script_file_path(script_id))
      for k in ("root_dir", "venv_dir", "entry", "git"):
        script.pop(k, None)

    else:
      target_app_type: str = chosen.get("app_type") or (current_type if current_type != "script" else "project")
      target_git: Optional[Dict[str, Any]] = chosen.get("git")
      dest_root = self.app_root_dir(script_id)

      try:
        assert tmp_extract is not None
        inner = tmp_extract / script_id
        if not inner.exists():
          inner = tmp_extract  # older/foreign archives without the wrapper dir

        shutil.rmtree(dest_root, ignore_errors=True)
        dest_root.mkdir(parents=True, exist_ok=True)
        for item in inner.iterdir():
          shutil.move(str(item), str(dest_root / item.name))
      except Exception:
        logger.exception("App rollback failed for %s", script_id)
        return "❌ Rollback failed (move error)."
      finally:
        if tmp_extract is not None:
          shutil.rmtree(tmp_extract, ignore_errors=True)

      py_files = discover_python_files(dest_root)
      name_hints = [script_id]
      if target_git:
        name_hints.append(target_git.get("repo"))
        if target_git.get("path"):
          name_hints.append(Path(target_git["path"]).name)
      entry, ambiguous = pick_entry_from_candidates(py_files, name_hints)
      # If the snapshotted version recorded a confirmed entry point (set when the
      # owner chose from an ambiguous list), honour it instead of re-running
      # auto-detection — which can return None for multi-file apps and break
      # subsequent start_script calls.
      if chosen.get("entry"):
        entry = chosen["entry"]
        ambiguous = False

      venv_dir = self.app_venv_dir(script_id)
      venv_ok, venv_out = await create_venv(venv_dir)
      if not venv_ok:
        logger.error("Venv creation failed during rollback for %s: %s", script_id, safe_trim_text(venv_out, 500))

      script["type"] = target_app_type
      script["root_dir"] = str(dest_root)
      script["venv_dir"] = str(venv_dir)
      script["entry"] = entry
      if target_git:
        script["git"] = target_git
      else:
        script.pop("git", None)
      script.pop("file", None)

    script["updated_at"] = now_ts_str()
    self.save_meta()

    type_label = {"script": "📄 script", "project": "📦 archive app", "git": "🌐 GitHub app"}.get(
        script.get("type", "script"), script.get("type", "script")
    )
    msg = f"⏪ Rolled back {script_id} to version {pretty_dt_from_ts(ts)} ({type_label})"

    if was_running:
      s = await self.start_script(script_id)
      msg += f"\n{s}"

    return msg

  # pip

  async def pip_install(self, args_str: str, script_id: Optional[str] = None) -> str:
    try:
      args = shlex.split(args_str)
    except ValueError as e:
      return f"❌ Bad args: {e}"
    if not args:
      return "❌ No packages specified."

    # For project/git apps, install into their own isolated venv + requirements.txt.
    # Legacy scripts keep the previous behavior: shared interpreter + shared requirements.txt.
    scoped = bool(script_id) and self.is_project_type(script_id)
    if scoped:
      script = self.get_script(script_id)
      venv_dir = Path(script.get("venv_dir") or self.app_venv_dir(script_id))
      python_exe_path = venv_python_path(venv_dir)
      if not python_exe_path.exists():
        ok, out = await create_venv(venv_dir)
        if not ok:
          return "❌ Failed to prepare venv:\n" + safe_trim_text(out, 1500)
      python_exe = str(python_exe_path)
      req_path = Path(script.get("root_dir")) / "requirements.txt"
    else:
      python_exe = sys.executable
      req_path = REQUIREMENTS_FILE

    logger.info("Running pip install via %s: %s", python_exe, " ".join(args))
    proc = await asyncio.create_subprocess_exec(
        python_exe,
        "-m",
        "pip",
        "install",
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    out, _ = await proc.communicate()
    text = out.decode("utf-8", errors="replace")
    exit_code = proc.returncode

    # Collect package specs to store (best effort)
    to_store: List[str] = []
    for a in args:
      if not a or a.startswith("-"):
        continue
      # Keep original spec (with ==, extras, etc.)
      to_store.append(a)

    saved = self.add_requirements(to_store, path=req_path)

    msg = []
    msg.append(f"$ {python_exe} -m pip install {' '.join(args)}")
    msg.append(f"exit code: {exit_code}")
    if saved:
      msg.append(f"📌 saved to {req_path.name} ({len(to_store)} item(s))" + (f" for app {script_id}" if scoped else ""))
    msg.append("")
    msg.append(text.strip() or "<no output>")

    # Optional import checks for common deps
    to_check = []
    for a in to_store:
      pkg = a.split("==")[0].split("[")[0]
      if pkg in ("openai", "python-telegram-bot", "psutil"):
        to_check.append(pkg)

    if to_check:
      msg.append("")
      msg.append("Import check:")
    for pkg in to_check:
      code = f"import {pkg}; print(getattr({pkg}, '__file__', '<no file>'))"
      check_proc = await asyncio.create_subprocess_exec(
          python_exe,
          "-c",
          code,
          stdout=asyncio.subprocess.PIPE,
          stderr=asyncio.subprocess.STDOUT,
      )
      cout, _ = await check_proc.communicate()
      ctext = cout.decode("utf-8", errors="replace").strip()
      if check_proc.returncode == 0:
        msg.append(f"- import {pkg}: OK -> {ctext}")
      else:
        msg.append(f"- import {pkg}: FAILED -> {ctext}")

    return safe_trim_text("\n".join(msg), 3900)

  # Fancy UI strings

  def _status_emoji(self, script_id: str) -> str:
    return "🟢" if self.script_running(script_id) else "🔴"

  def _autostart_emoji(self, script_id: str) -> str:
    s = self.get_script(script_id)
    return "🚀" if s and s.get("autostart") else "⏸️"

  def _type_emoji(self, script_id: str) -> str:
    return {"script": "📄", "project": "📦", "git": "🌐"}.get(self.get_type(script_id), "📄")

  def _versions_count(self, script_id: str) -> int:
    return len(self.get_versions(script_id))

  def _updated_at_pretty(self, script_id: str) -> str:
    s = self.get_script(script_id)
    ts = (s or {}).get("updated_at")
    if not ts:
      return "—"
    return pretty_dt_from_ts(str(ts))

  def _pid(self, script_id: str) -> str:
    p = self.processes.get(script_id)
    if not p or p.returncode is not None:
      return "—"
    return str(p.pid)

  def script_status_line(self, script_id: str) -> str:
    st = self._status_emoji(script_id)
    ty = self._type_emoji(script_id)
    au = self._autostart_emoji(script_id)
    pid = self._pid(script_id)
    vc = self._versions_count(script_id)
    upd = self._updated_at_pretty(script_id)
    return f"{st}{ty} {au} {script_id}  | PID: {pid} | v:{vc} | updated: {upd}"

  def get_status_lines(self) -> str:
    scripts = self.meta.get("scripts", {})
    if not scripts:
      return "😴 No scripts yet."
    lines = [self.script_status_line(sid) for sid in scripts.keys()]
    return "\n".join(lines)


script_mgr = ScriptManager()


# Keyboards

def main_menu_keyboard() -> InlineKeyboardMarkup:
  scripts = script_mgr.list_scripts()
  buttons = []
  for sid in scripts.keys():
    text = f"{script_mgr._status_emoji(sid)}{script_mgr._type_emoji(sid)} {script_mgr._autostart_emoji(sid)} {sid}"
    buttons.append([InlineKeyboardButton(text=text, callback_data=f"menu:{sid}")])
  if not buttons:
    buttons.append([InlineKeyboardButton(text="😴 No scripts", callback_data="noop")])
  return InlineKeyboardMarkup(buttons)


def script_menu_keyboard(script_id: str) -> InlineKeyboardMarkup:
  # Menu for legacy single-file (.py) scripts - kept exactly as before (full BWC).
  script = script_mgr.get_script(script_id)
  autostart = script.get("autostart") if script else False
  running = script_mgr.script_running(script_id)

  run_text = "▶ Run" if not running else "🔁 Restart"
  stop_text = "⏹ Stop"
  auto_text = "🚀 Autostart: ON" if autostart else "⏸️ Autostart: OFF"

  db_label = "🗄 Database Info" if pgdb.get_app_database(script_id) else "🗄 Create Database"
  buttons = [
    [
      InlineKeyboardButton(run_text, callback_data=f"run:{script_id}"),
      InlineKeyboardButton(stop_text, callback_data=f"stop:{script_id}"),
    ],
    [
      InlineKeyboardButton("📜 Logs", callback_data=f"logs:{script_id}"),
      InlineKeyboardButton("✏ Edit", callback_data=f"edit:{script_id}"),
    ],
    [
      InlineKeyboardButton("🧪 Env", callback_data=f"envmenu:{script_id}"),
      InlineKeyboardButton("📦 Pip install", callback_data=f"pipprompt:{script_id}"),
    ],
    [
      InlineKeyboardButton("⏪ Rollback", callback_data=f"rbmenu:{script_id}"),
      InlineKeyboardButton(auto_text, callback_data=f"auto:{script_id}"),
    ],
    [
      InlineKeyboardButton("🔀 Change Source", callback_data=f"changesrc:{script_id}"),
    ],
    [
      InlineKeyboardButton(db_label, callback_data=f"createdb:{script_id}"),
    ],
    [
      InlineKeyboardButton("⬅ Back", callback_data="back:main"),
    ],
  ]
  return InlineKeyboardMarkup(buttons)


def app_menu_keyboard(script_id: str) -> InlineKeyboardMarkup:
  # Menu for apps imported from an archive or a GitHub repo (own isolated venv).
  script = script_mgr.get_script(script_id)
  autostart = script.get("autostart") if script else False
  running = script_mgr.script_running(script_id)
  app_type = (script or {}).get("type", "project")

  run_text = "▶ Run" if not running else "🔁 Restart"
  stop_text = "⏹ Stop"
  auto_text = "🚀 Autostart: ON" if autostart else "⏸️ Autostart: OFF"

  logs_row = [InlineKeyboardButton("📜 Logs", callback_data=f"logs:{script_id}")]
  if app_type == "git":
    logs_row.append(InlineKeyboardButton("🔄 Sync", callback_data=f"sync:{script_id}"))

  db_label = "🗄 Database Info" if pgdb.get_app_database(script_id) else "🗄 Create Database"
  buttons = [
    [
      InlineKeyboardButton(run_text, callback_data=f"run:{script_id}"),
      InlineKeyboardButton(stop_text, callback_data=f"stop:{script_id}"),
    ],
    logs_row,
    [
      InlineKeyboardButton("🧪 Env", callback_data=f"envmenu:{script_id}"),
      InlineKeyboardButton("📦 Pip install", callback_data=f"pipprompt:{script_id}"),
    ],
    [
      InlineKeyboardButton("⏪ Rollback", callback_data=f"rbmenu:{script_id}"),
      InlineKeyboardButton(auto_text, callback_data=f"auto:{script_id}"),
    ],
    [
      InlineKeyboardButton("🔀 Change Source", callback_data=f"changesrc:{script_id}"),
    ],
    [
      InlineKeyboardButton(db_label, callback_data=f"createdb:{script_id}"),
    ],
    [
      InlineKeyboardButton("⬅ Back", callback_data="back:main"),
    ],
  ]
  return InlineKeyboardMarkup(buttons)


def menu_keyboard_for(script_id: str) -> InlineKeyboardMarkup:
  app_type = script_mgr.get_type(script_id)
  return script_menu_keyboard(script_id) if app_type == "script" else app_menu_keyboard(script_id)


_TYPE_LABEL: Dict[str, str] = {
    "script": "📄 script",
    "project": "📦 archive app",
    "git": "🌐 GitHub app",
}


def _source_change_confirm_text(script_id: str, old_type: str, new_type: str) -> str:
  """Confirmation message for any source-change operation initiated via the
  'Change Source' button.  Uses the full conversion warning when the type
  changes, or a shorter replace-only note when the type stays the same."""
  if old_type != new_type:
    return type_conversion_confirm_text(script_id, old_type, new_type)
  label = _TYPE_LABEL.get(old_type, old_type)
  return (
      f"🔀 Replace the source of {script_id} ({label})?\n\n"
      f"• The current version will be saved — reachable via ⏪ Rollback\n"
      f"• All env vars and autostart settings are preserved\n\n"
      f"Continue?"
  )


def type_conversion_confirm_text(script_id: str, old_type: str, new_type: str) -> str:
  old_label = _TYPE_LABEL.get(old_type, old_type)
  new_label = _TYPE_LABEL.get(new_type, new_type)
  if old_type == "script":
    cleanup_note = "set up a new isolated virtual environment"
  else:
    cleanup_note = "remove the existing source tree and virtual environment"
  return (
      f"⚠️ {script_id} is currently a {old_label}.\n\n"
      f"Converting to a {new_label} will:\n"
      f"• Stop the app if it is running\n"
      f"• Replace its files and {cleanup_note}\n"
      f"• Keep all old versions reachable via ⏪ Rollback, including a full "
      f"revert back to {old_label} mode\n\n"
      f"Continue?"
  )


def type_conversion_confirm_keyboard(script_id: str) -> InlineKeyboardMarkup:
  return InlineKeyboardMarkup([[
      InlineKeyboardButton("✅ Convert", callback_data=f"convtype:yes:{script_id}"),
      InlineKeyboardButton("❌ Cancel", callback_data=f"convtype:no:{script_id}"),
  ]])


def env_menu_text(script_id: str) -> str:
  script = script_mgr.get_script(script_id)
  if not script:
    return "❌ Script not found."
  env = script.get("env", {})
  keys = script_mgr.get_env_keys_for_script(script_id)
  if not keys:
    return f"🧪 Env for {script_id}: no vars detected."
  lines = []
  for k in keys:
    v = env.get(k, "")
    if v == "":
      lines.append(f"• {k}=<not set>")
    else:
      lines.append(f"• {k}={v}")
  return f"🧪 Env for {script_id}:\n" + "\n".join(lines)


def env_menu_keyboard(script_id: str) -> InlineKeyboardMarkup:
  keys = script_mgr.get_env_keys_for_script(script_id)
  buttons: List[List[InlineKeyboardButton]] = []
  for k in keys:
    buttons.append([InlineKeyboardButton(f"🔧 {k}", callback_data=f"envset:{script_id}:{k}")])
  buttons.append(
      [
        InlineKeyboardButton("✅ Done", callback_data=f"envdone:{script_id}"),
        InlineKeyboardButton("▶ Run", callback_data=f"envrun:{script_id}"),
      ]
  )
  buttons.append([InlineKeyboardButton("⬅ Back", callback_data=f"menu:{script_id}")])
  return InlineKeyboardMarkup(buttons)


def rollback_menu_text(script_id: str) -> str:
  versions = script_mgr.get_versions(script_id)
  if not versions:
    return f"⏪ Rollback for {script_id}:\n(no versions yet)\n\nTip: versions appear after you update/overwrite a script."
  lines = [f"⏪ Rollback for {script_id}:"]
  for v in versions:
    lines.append(f"• {pretty_dt_from_ts(v['ts'])}")
  return "\n".join(lines)


def rollback_menu_keyboard(script_id: str) -> InlineKeyboardMarkup:
  versions = script_mgr.get_versions(script_id)
  buttons: List[List[InlineKeyboardButton]] = []
  for v in versions[:MAX_VERSIONS_PER_SCRIPT]:
    ts = v["ts"]
    buttons.append([InlineKeyboardButton(f"🕒 {pretty_dt_from_ts(ts)}", callback_data=f"rbdo:{script_id}:{ts}")])
  buttons.append([InlineKeyboardButton("⬅ Back", callback_data=f"menu:{script_id}")])
  return InlineKeyboardMarkup(buttons)


async def send_env_menu(chat_id: int, script_id: str, context: ContextTypes.DEFAULT_TYPE) -> None:
  await context.bot.send_message(chat_id=chat_id, text=env_menu_text(script_id), reply_markup=env_menu_keyboard(script_id))


async def send_app_menu(chat_id: int, script_id: str, context: ContextTypes.DEFAULT_TYPE) -> None:
  await context.bot.send_message(chat_id=chat_id, text=script_mgr.script_status_line(script_id), reply_markup=menu_keyboard_for(script_id))


# Backward-compatible alias (kept in case anything external referenced this name)
send_script_menu = send_app_menu


# New apps: archives + GitHub repos

async def present_setup_result(
    chat_id: int,
    context: ContextTypes.DEFAULT_TYPE,
    script_id: str,
    result: Dict[str, Any],
    header: str,
) -> None:
  if result.get("status") == "error":
    await context.bot.send_message(
        chat_id=chat_id,
        text=f"❌ {header}\n{safe_trim_text(result.get('message', ''), 3500)}",
    )
    return

  if result.get("status") == "ambiguous":
    candidates = result.get("candidates", [])
    buttons = [
        [InlineKeyboardButton(f"▶ {c}", callback_data=f"entrypick:{script_id}:{i}")]
        for i, c in enumerate(candidates[:20])
    ]
    await context.bot.send_message(
        chat_id=chat_id,
        text=f"{header}\n\n🤔 Found {len(candidates)} Python files and no main.py. Which one should I run?",
        reply_markup=InlineKeyboardMarkup(buttons),
    )
    return

  entry = result.get("entry")
  req_pkgs = result.get("requirements_pkgs") or []
  lines = [header, "", f"📄 Entry point: {entry}"]
  if not result.get("venv_ok", True):
    lines.append("⚠️ Failed to create virtual environment:")
    lines.append(safe_trim_text(result.get("venv_out", ""), 800))

  if req_pkgs:
    lines.append(f"📦 requirements.txt found ({len(req_pkgs)} package(s)): {', '.join(req_pkgs[:15])}")
    await context.bot.send_message(
        chat_id=chat_id,
        text=safe_trim_text("\n".join(lines), 3500),
        reply_markup=InlineKeyboardMarkup(
            [
              [InlineKeyboardButton("📥 Install & Run", callback_data=f"reqinstall:{script_id}")],
              [InlineKeyboardButton("▶ Run without installing", callback_data=f"reqskip:{script_id}")],
            ]
        ),
    )
  else:
    run_msg = await script_mgr.start_script(script_id)
    lines.append("")
    lines.append(run_msg)
    await context.bot.send_message(chat_id=chat_id, text=safe_trim_text("\n".join(lines), 3500))

  await send_env_menu(chat_id, script_id, context)
  await send_app_menu(chat_id, script_id, context)


async def handle_archive_upload(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    doc,
    force_target_id: Optional[str] = None,
) -> None:
  filename = doc.file_name
  # When arriving via "Change Source", the user may have sent an archive whose
  # filename doesn't match the target app's id — honour the override.
  app_id = force_target_id or sanitize_id(strip_archive_ext(filename))

  status_msg = await update.message.reply_text(f"📦 Downloading {filename} ...")
  tmp_archive = TMP_DIR / f"upload_{app_id}_{int(datetime.now().timestamp())}{archive_ext_of(filename)}"
  try:
    file = await doc.get_file()
    await file.download_to_drive(str(tmp_archive))

    await status_msg.edit_text(f"📦 Extracting {filename} ...")
    staging = TMP_DIR / f"extract_{app_id}_{int(datetime.now().timestamp())}"
    shutil.rmtree(staging, ignore_errors=True)
    try:
      extract_archive(tmp_archive, staging)
    except Exception as e:
      logger.exception("Extraction failed for %s", filename)
      await status_msg.edit_text(f"❌ Failed to extract archive: {e}")
      shutil.rmtree(staging, ignore_errors=True)
      return

    extraction_root = find_extraction_root(staging)

    old_type = script_mgr.conflicting_type(app_id, "project")
    # Show confirmation if: (a) a different type already exists, or (b) the
    # user explicitly requested a source change via the "Change Source" button.
    needs_confirm = old_type is not None or force_target_id is not None
    if needs_confirm:
      effective_old_type = old_type or (script_mgr.get_script(app_id) or {}).get("type", "project")
      script_mgr._set_pending_conversion(app_id, {
          "action": "archive",
          "staged_root": str(extraction_root),
          "staging": str(staging) if staging != extraction_root else None,
          "old_type": effective_old_type,
      })
      await status_msg.delete()
      confirm_text = (
          _source_change_confirm_text(app_id, effective_old_type, "project")
          if force_target_id is not None
          else type_conversion_confirm_text(app_id, effective_old_type, "project")
      )
      await update.message.reply_text(confirm_text, reply_markup=type_conversion_confirm_keyboard(app_id))
      return

    await status_msg.edit_text(f"⚙️ Setting up app {app_id} ...")
    result = await script_mgr.setup_project_app(app_id, extraction_root, "project")
    if staging.exists() and staging != extraction_root:
      shutil.rmtree(staging, ignore_errors=True)

    await status_msg.delete()
    await present_setup_result(
        update.effective_chat.id, context, app_id, result, f"✅ App {app_id} imported from {filename}."
    )
  finally:
    tmp_archive.unlink(missing_ok=True)


async def _prepare_git_app(info: Dict[str, Any]) -> Dict[str, Any]:
  """Clone / pull the repo and copy the relevant subtree to a staging directory.
  Returns a dict with ``staged_root`` and ``git_info`` on success, or
  ``{"status": "error", "message": ...}`` on failure.  Does NOT mutate the
  ScriptManager; the caller decides whether to proceed immediately or first
  show a confirmation dialog.
  """
  app_id = info["app_id"]
  repo_dir = GITREPOS_DIR / app_id
  repo_url = f"https://github.com/{info['owner']}/{info['repo']}.git"

  # When the existing clone points to a *different* remote (e.g. the user
  # is replacing a GitHub source via "Change Source"), discard it so
  # git_clone_or_pull performs a fresh clone instead of fetching the wrong
  # origin.
  git_config = repo_dir / ".git" / "config"
  if git_config.exists():
    try:
      if repo_url not in git_config.read_text():
        shutil.rmtree(repo_dir, ignore_errors=True)
    except Exception:
      logger.warning("_prepare_git_app: could not read %s — leaving repo_dir as-is", git_config)

  ok, out = await git_clone_or_pull(repo_url, info.get("branch"), repo_dir)
  if not ok:
    return {"status": "error", "message": out}

  branch = info.get("branch") or await git_current_branch(repo_dir)

  source_root = (repo_dir / info["path"]) if info.get("path") else repo_dir
  if not source_root.exists():
    return {"status": "error", "message": f"Path not found in repo: {info.get('path')}"}

  staging = TMP_DIR / f"gitstage_{app_id}_{int(datetime.now().timestamp())}"
  shutil.rmtree(staging, ignore_errors=True)
  shutil.copytree(source_root, staging, ignore=shutil.ignore_patterns(".git"))

  git_info = {
      "owner": info["owner"],
      "repo": info["repo"],
      "branch": branch,
      "path": info.get("path"),
      "repo_dir": str(repo_dir),
  }
  return {"staged_root": staging, "git_info": git_info, "app_id": app_id}


async def sync_git_app(script_id: str) -> Dict[str, Any]:
  script = script_mgr.get_script(script_id)
  if not script or script.get("type") != "git":
    return {"status": "error", "message": "Not a GitHub app."}

  git_info = script.get("git", {})
  repo_dir = Path(git_info.get("repo_dir") or (GITREPOS_DIR / script_id))
  branch = git_info.get("branch")
  repo_url = f"https://github.com/{git_info.get('owner')}/{git_info.get('repo')}.git"

  ok, out = await git_clone_or_pull(repo_url, branch, repo_dir)
  if not ok:
    return {"status": "error", "message": out}

  source_root = (repo_dir / git_info["path"]) if git_info.get("path") else repo_dir
  if not source_root.exists():
    return {"status": "error", "message": f"Path not found in repo: {git_info.get('path')}"}

  old_req_path = Path(script.get("root_dir", "")) / "requirements.txt"
  old_req_text = read_text_file(old_req_path) if old_req_path.exists() else ""

  staging = TMP_DIR / f"gitstage_{script_id}_{int(datetime.now().timestamp())}"
  shutil.rmtree(staging, ignore_errors=True)
  shutil.copytree(source_root, staging, ignore=shutil.ignore_patterns(".git"))

  was_running = script_mgr.script_running(script_id)
  if was_running:
    await script_mgr.stop_script(script_id)

  result = await script_mgr.setup_project_app(script_id, staging, "git", git_info)

  updated_script = script_mgr.get_script(script_id) or {}
  new_req_path = Path(updated_script.get("root_dir", "")) / "requirements.txt"
  new_req_text = read_text_file(new_req_path) if new_req_path.exists() else ""

  result["deps_changed"] = new_req_text.strip() != old_req_text.strip()
  result["was_running"] = was_running
  return result


async def process_github_url(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    url: str,
    force_target_id: Optional[str] = None,
) -> None:
  info = parse_github_url(url)
  if not info:
    await update.message.reply_text(
        "❌ Doesn't look like a GitHub repo URL.\n"
        "Expected: https://github.com/owner/repo\n"
        "Or with a subfolder: https://github.com/owner/repo/tree/branch/path"
    )
    return

  # When the user arrives via "Change Source", override the app_id that would
  # normally be inferred from the repository name.
  if force_target_id:
    info = {**info, "app_id": force_target_id}

  app_id = info["app_id"]
  status_msg = await update.message.reply_text(f"🌐 Cloning {info['owner']}/{info['repo']} ...")
  prep = await _prepare_git_app(info)
  await status_msg.delete()

  if prep.get("status") == "error":
    await update.message.reply_text(f"❌ Failed to import repo:\n{safe_trim_text(prep.get('message', ''), 3500)}")
    return

  staged_root: Path = prep["staged_root"]
  git_info: Dict[str, Any] = prep["git_info"]

  old_type = script_mgr.conflicting_type(app_id, "git")
  # Show confirmation when: (a) a conflicting type exists, or (b) the user
  # explicitly invoked "Change Source" (so we always confirm before replacing).
  needs_confirm = old_type is not None or force_target_id is not None
  if needs_confirm:
    effective_old_type = old_type or (script_mgr.get_script(app_id) or {}).get("type", "git")
    script_mgr._set_pending_conversion(app_id, {
        "action": "git",
        "staged_root": str(staged_root),
        "git_info": git_info,
        "old_type": effective_old_type,
    })
    confirm_text = (
        _source_change_confirm_text(app_id, effective_old_type, "git")
        if force_target_id is not None
        else type_conversion_confirm_text(app_id, effective_old_type, "git")
    )
    await update.message.reply_text(confirm_text, reply_markup=type_conversion_confirm_keyboard(app_id))
    return

  result = await script_mgr.setup_project_app(app_id, staged_root, "git", git_info)
  result["app_id"] = app_id
  await present_setup_result(
      update.effective_chat.id, context, app_id, result, f"✅ App {app_id} cloned from GitHub."
  )


# Commands

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
  if not is_authorized(update):
    return
  await update.message.reply_text(
      "🤖 Script manager ready.\n\n"
      "📄 Single scripts:\n"
      "• Send a .py file, or /new <name> + text — run/stop/edit/env/pip/rollback as before\n\n"
      "📦 Multi-file apps (archives):\n"
      "• Send a .zip / .tar.gz / .tgz / .tar / .7z file\n"
      "• I'll unpack it, find requirements.txt (offer to install) and detect main.py\n"
      "• If there's only one .py file (or a main.py), I run it automatically\n"
      "• If there are several files and no main.py, I'll ask which one to run\n"
      "• The app name is the archive name (without extension)\n"
      "• Each such app gets its own isolated Python venv (only the telegram lib is shared)\n\n"
      "🌐 Apps from GitHub:\n"
      "• Just send a link, e.g. https://github.com/owner/repo\n"
      "• Or a subfolder link: https://github.com/owner/repo/tree/branch/path\n"
      "  (app name becomes repo-folder, e.g. myrepo-myapp)\n"
      "• Or use /repo <url>\n"
      "• These apps get a 🔄 Sync button: re-pulls the repo, offers to install new deps, and runs it\n\n"
      "🔀 Switching an app's type / source:\n"
      "• Upload a .py / archive / GitHub link whose name matches an existing app of a different type\n"
      "• Or open an app's menu and press 🔀 Change Source, then send any source — even with a\n"
      "  different filename or repo name.  The bot always asks for confirmation before replacing.\n"
      "• All old versions are kept — ⏪ Rollback works across the type boundary and fully restores\n"
      "  the previous type (files, venv, meta), so switching is always reversible\n\n"
      "Main:\n"
      "• /menu — apps/scripts menu\n"
      "• /list — list apps/scripts (rich)\n"
      "• /new <name> — create script from next text message\n"
      "• /repo <github_url> — import an app from a GitHub repo\n"
      "• /cancel — abort any pending input flow (Change Source, /new, env edit, etc.)\n"
      "• /run <id>, /stop <id>\n"
      "• /logs <id>\n"
      "• /monitoring — CPU/MEM for running scripts\n\n"
      "Packages:\n"
      "• /pip <args> — pip install\n"
      "  (scripts: shared requirements.txt; apps: their own venv + requirements.txt)\n\n"
      "Env:\n"
      "• /envmenu <id>\n"
      "• /env <id>, /setenv, /delenv\n"
      "• /globalenv, /setglobal, /delglobal\n\n"
      "Legend: 📄 script  📦 archive app  🌐 GitHub app\n"
      "Tip: For /monitoring install psutil via /pip psutil"
  )


async def cmd_cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
  if not is_authorized(update):
    return
  user_id = update.effective_user.id
  cleared: list[str] = []
  if script_mgr.pending_source_change.pop(user_id, None) is not None:
    cleared.append("source change")
  if script_mgr.pending_new.pop(user_id, None) is not None:
    cleared.append("new script")
  if script_mgr.pending_edit.pop(user_id, None) is not None:
    cleared.append("script edit")
  if script_mgr.pending_env_value.pop(user_id, None) is not None:
    cleared.append("env edit")
  if script_mgr.pending_pip.pop(user_id, None) is not None:
    cleared.append("pip install")
  if cleared:
    await update.message.reply_text(f"🚫 Cancelled: {', '.join(cleared)}.")
  else:
    await update.message.reply_text("🟡 Nothing to cancel.")


async def cmd_menu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
  if not is_authorized(update):
    return
  await update.message.reply_text("📋 Select script:", reply_markup=main_menu_keyboard())


async def cmd_list(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
  if not is_authorized(update):
    return
  await update.message.reply_text(script_mgr.get_status_lines())


async def cmd_new(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
  if not is_authorized(update):
    return
  if not context.args:
    await update.message.reply_text("Usage: /new <script_id>")
    return
  script_id = context.args[0].strip().replace(" ", "_")
  if not script_id:
    await update.message.reply_text("Bad script id.")
    return

  user_id = update.effective_user.id
  script_mgr.pending_new[user_id] = script_id
  script_mgr.pending_edit.pop(user_id, None)
  script_mgr.pending_env_value.pop(user_id, None)
  script_mgr.pending_pip.pop(user_id, None)

  await update.message.reply_text(f"✍️ Send script body as text. It will be saved as {script_id}.py")


async def cmd_repo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
  if not is_authorized(update):
    return
  if not context.args:
    await update.message.reply_text(
        "Usage: /repo <github_url>\n"
        "Example: /repo https://github.com/owner/repo\n"
        "With subfolder: /repo https://github.com/owner/repo/tree/main/subdir\n\n"
        "Tip: you can also just paste the link as a plain message."
    )
    return
  await process_github_url(update, context, context.args[0])


async def cmd_envmenu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
  if not is_authorized(update):
    return
  if not context.args:
    await update.message.reply_text("Usage: /envmenu <script_id>")
    return
  script_id = context.args[0]
  if not script_mgr.get_script(script_id):
    await update.message.reply_text("❌ Script not found.")
    return
  await send_env_menu(update.effective_chat.id, script_id, context)


async def cmd_pip(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
  if not is_authorized(update):
    return
  if not context.args:
    await update.message.reply_text("Usage: /pip <pip install args>")
    return
  args_str = " ".join(context.args)
  msg = await script_mgr.pip_install(args_str)
  await update.message.reply_text(msg)


async def cmd_run(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
  if not is_authorized(update):
    return
  if not context.args:
    await update.message.reply_text("Usage: /run <script_id>")
    return
  script_id = context.args[0]
  msg = await script_mgr.start_script(script_id)
  await update.message.reply_text(msg)


async def cmd_stop(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
  if not is_authorized(update):
    return
  if not context.args:
    await update.message.reply_text("Usage: /stop <script_id>")
    return
  script_id = context.args[0]
  msg = await script_mgr.stop_script(script_id)
  await update.message.reply_text(msg)


async def cmd_logs(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
  if not is_authorized(update):
    return
  if not context.args:
    await update.message.reply_text("Usage: /logs <script_id>")
    return
  script_id = context.args[0]
  if not script_mgr.get_script(script_id):
    await update.message.reply_text("❌ Script not found.")
    return
  text = script_mgr.read_log_tail(script_id, lines=LOG_TAIL_LINES)
  ts = datetime.now().strftime("%H:%M:%S")
  await update.message.reply_text(f"📜 [{ts}] Logs for {script_id} (last {LOG_TAIL_LINES} lines):\n{text}")


async def cmd_env(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
  if not is_authorized(update):
    return
  if not context.args:
    await update.message.reply_text("Usage: /env <script_id>")
    return
  script_id = context.args[0]
  script = script_mgr.get_script(script_id)
  if not script:
    await update.message.reply_text("❌ Script not found.")
    return
  env = script.get("env", {})
  if not env:
    await update.message.reply_text("🧪 No env vars for this script.")
    return
  lines = [f"{k}={v}" for k, v in env.items()]
  await update.message.reply_text("🧪 Script env:\n" + "\n".join(lines))


async def cmd_setenv(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
  if not is_authorized(update):
    return
  if len(context.args) < 3:
    await update.message.reply_text("Usage: /setenv <script_id> <KEY> <VALUE>")
    return
  script_id = context.args[0]
  key = context.args[1]
  value = " ".join(context.args[2:])
  if not script_mgr.get_script(script_id):
    await update.message.reply_text("❌ Script not found.")
    return
  script_mgr.set_script_env(script_id, key, value)
  await update.message.reply_text(f"✅ Set {key} for {script_id}.")


async def cmd_delenv(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
  if not is_authorized(update):
    return
  if len(context.args) < 2:
    await update.message.reply_text("Usage: /delenv <script_id> <KEY>")
    return
  script_id = context.args[0]
  key = context.args[1]
  ok = script_mgr.del_script_env(script_id, key)
  await update.message.reply_text("🟡 Nothing to delete." if not ok else f"🗑️ Deleted {key} from {script_id}.")


async def cmd_globalenv(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
  if not is_authorized(update):
    return
  env = script_mgr.get_global_env()
  if not env:
    await update.message.reply_text("🧊 No global env vars.")
    return
  lines = [f"{k}={v}" for k, v in env.items()]
  await update.message.reply_text("🌍 Global env:\n" + "\n".join(lines))


async def cmd_setglobal(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
  if not is_authorized(update):
    return
  if len(context.args) < 2:
    await update.message.reply_text("Usage: /setglobal <KEY> <VALUE>")
    return
  key = context.args[0]
  value = " ".join(context.args[1:])
  script_mgr.set_global_env(key, value)
  await update.message.reply_text(f"✅ Set global {key}.")


async def cmd_delglobal(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
  if not is_authorized(update):
    return
  if len(context.args) < 1:
    await update.message.reply_text("Usage: /delglobal <KEY>")
    return
  key = context.args[0]
  ok = script_mgr.del_global_env(key)
  await update.message.reply_text("🟡 Nothing to delete." if not ok else f"🗑️ Deleted global {key}.")


async def cmd_autostart(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
  if not is_authorized(update):
    return
  if len(context.args) < 2:
    await update.message.reply_text("Usage: /autostart <script_id> on|off")
    return
  script_id = context.args[0]
  value = context.args[1].lower()
  if value not in ("on", "off"):
    await update.message.reply_text("Value must be 'on' or 'off'.")
    return
  ok = script_mgr.set_autostart(script_id, value == "on")
  if not ok:
    await update.message.reply_text("❌ Script not found.")
    return
  await update.message.reply_text(f"✅ Autostart for {script_id} set to {value}.")


async def cmd_monitoring(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
  if not is_authorized(update):
    return

  running = [sid for sid in script_mgr.list_scripts().keys() if script_mgr.script_running(sid)]
  if not running:
    await update.message.reply_text("😴 No running scripts.")
    return

  if psutil is None:
    await update.message.reply_text(
        "❌ Monitoring needs psutil.\n"
        "Install it: /pip psutil\n"
    )
    return

  lines = ["📈 Monitoring (running scripts):", ""]
  for sid in running:
    p = script_mgr.psutil_procs.get(sid)
    proc = script_mgr.processes.get(sid)
    if not proc or proc.returncode is not None:
      continue
    pid = proc.pid
    try:
      if p is None:
        p = psutil.Process(pid)
        p.cpu_percent(interval=None)
        script_mgr.psutil_procs[sid] = p

      cpu = p.cpu_percent(interval=0.0)
      mem = p.memory_info().rss / (1024 * 1024)
      threads = getattr(p, "num_threads", lambda: 0)()
      uptime_s = 0
      st = script_mgr.start_times.get(sid)
      if st:
        uptime_s = int(datetime.now().timestamp() - st)

      extra = []
      try:
        io = p.io_counters()
        extra.append(f"IO r/w: {int(io.read_bytes/1024/1024)}MB/{int(io.write_bytes/1024/1024)}MB")
      except Exception:
        pass

      # Some psutil versions/platforms may support net_io_counters per-process
      try:
        net = getattr(p, "net_io_counters", None)
        if callable(net):
          nio = net()
          extra.append(f"NET s/r: {int(nio.bytes_sent/1024)}KB/{int(nio.bytes_recv/1024)}KB")
      except Exception:
        pass

      line = (
        f"🟢 {sid} | PID {pid} | up {uptime_s}s | CPU {cpu:.1f}% | MEM {mem:.1f}MB | thr {threads}"
      )
      if extra:
        line += " | " + " | ".join(extra)
      lines.append(line)
    except Exception as e:
      lines.append(f"🟡 {sid} | PID {pid} | monitoring error: {e}")

  await update.message.reply_text(safe_trim_text("\n".join(lines), 3900))


# Handlers

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
  if not is_authorized(update):
    return

  user_id = update.effective_user.id
  text = update.message.text

  # pip prompt flow
  if user_id in script_mgr.pending_pip:
    script_id = script_mgr.pending_pip.pop(user_id)
    value = text.strip()
    if value == "-":
      await update.message.reply_text("🟡 Pip install cancelled.")
    else:
      msg = await script_mgr.pip_install(value, script_id=script_id)
      await update.message.reply_text(msg)
    if script_id:
      await send_app_menu(update.effective_chat.id, script_id, context)
    return

  # env value flow
  if user_id in script_mgr.pending_env_value:
    info = script_mgr.pending_env_value.pop(user_id)
    script_id = info["script_id"]
    key = info["key"]
    value = text.strip()
    if value == "-":
      script_mgr.del_script_env(script_id, key)
      await update.message.reply_text(f"🗑️ Deleted {key} for {script_id}.")
    else:
      script_mgr.set_script_env(script_id, key, text)
      await update.message.reply_text(f"✅ Set {key} for {script_id}.")
    await send_env_menu(update.effective_chat.id, script_id, context)
    return

  # new script flow
  if user_id in script_mgr.pending_new:
    script_id = script_mgr.pending_new.pop(user_id)
    old_type = script_mgr.conflicting_type(script_id, "script")
    if old_type is not None:
      # Stash the text and wait for user confirmation.
      script_mgr._set_pending_conversion(script_id, {
          "action": "script_text",
          "text": text,
          "old_type": old_type,
      })
      await update.message.reply_text(
          type_conversion_confirm_text(script_id, old_type, "script"),
          reply_markup=type_conversion_confirm_keyboard(script_id),
      )
      return
    full_id = script_mgr.upsert_script_from_text(script_id, text)
    await update.message.reply_text(f"✅ Script {full_id} saved.")
    await send_script_menu(update.effective_chat.id, full_id, context)
    await send_env_menu(update.effective_chat.id, full_id, context)
    return

  # edit script flow
  if user_id in script_mgr.pending_edit:
    script_id = script_mgr.pending_edit.pop(user_id)
    was_running = script_mgr.script_running(script_id)
    script_mgr.upsert_script_from_text(script_id, text)
    msg = f"✅ Script {script_id} updated.\n📦 Version saved (last {MAX_VERSIONS_PER_SCRIPT})."
    if was_running:
      msg2 = await script_mgr.restart_script(script_id)
      msg += f"\n{msg2}"
    await update.message.reply_text(safe_trim_text(msg, 3900))
    await send_script_menu(update.effective_chat.id, script_id, context)
    await send_env_menu(update.effective_chat.id, script_id, context)
    return

  # "Change Source" flow — the user previously clicked the button and we are
  # now waiting for the new source.  Only GitHub links are accepted as text;
  # files / archives must be sent as documents (handled in handle_document).
  if user_id in script_mgr.pending_source_change:
    target_id = script_mgr.pending_source_change[user_id]
    gh_info = parse_github_url(text.strip())
    if gh_info:
      del script_mgr.pending_source_change[user_id]
      await process_github_url(update, context, text.strip(), force_target_id=target_id)
    else:
      await update.message.reply_text(
          f"❓ Waiting for a new source for *{target_id}*.\n\n"
          "Send:\n"
          "• A `.py` file — convert to 📄 script\n"
          "• An archive (`.zip` / `.tar.gz` / `.tgz` / `.tar` / `.7z`) — convert to 📦 archive app\n"
          "• A GitHub link — convert to 🌐 GitHub app\n\n"
          "Send /cancel to abort.",
          parse_mode="Markdown",
      )
    return

  # GitHub repo link, sent as a plain message (only if no other flow matched above)
  gh_info = parse_github_url(text.strip())
  if gh_info:
    await process_github_url(update, context, text.strip())
    return

  logger.info("Ignored text from user_id=%s (no pending state)", user_id)


async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
  if not is_authorized(update):
    return
  doc = update.message.document
  filename = doc.file_name or "file"
  lower = filename.lower()

  user_id = update.effective_user.id
  # If the user clicked "Change Source", honour the stored target app id
  # regardless of what the uploaded file is named.
  force_target_id: Optional[str] = script_mgr.pending_source_change.pop(user_id, None)

  if lower.endswith(".py"):
    if force_target_id:
      script_id = force_target_id
    else:
      stem = Path(filename).stem.strip().replace(" ", "_")
      script_id = stem or "script"
    old_type = script_mgr.conflicting_type(script_id, "script")

    tmp_path = SCRIPTS_DIR / ("_upload_tmp_" + filename)
    file = await doc.get_file()
    await file.download_to_drive(str(tmp_path))
    file_bytes = tmp_path.read_bytes()
    tmp_path.unlink(missing_ok=True)

    needs_confirm = old_type is not None or force_target_id is not None
    if needs_confirm:
      effective_old_type = old_type or (script_mgr.get_script(script_id) or {}).get("type", "script")
      script_mgr._set_pending_conversion(script_id, {
          "action": "script_file",
          "file_bytes": file_bytes,
          "original_name": filename,
          "old_type": effective_old_type,
      })
      confirm_text = (
          _source_change_confirm_text(script_id, effective_old_type, "script")
          if force_target_id is not None
          else type_conversion_confirm_text(script_id, effective_old_type, "script")
      )
      await update.message.reply_text(confirm_text, reply_markup=type_conversion_confirm_keyboard(script_id))
      return

    # Write bytes to a temp file so upsert_script_from_file can read it.
    tmp_path2 = SCRIPTS_DIR / ("_upload_tmp2_" + filename)
    tmp_path2.write_bytes(file_bytes)
    actual_id = script_mgr.upsert_script_from_file(filename, tmp_path2)
    tmp_path2.unlink(missing_ok=True)

    await update.message.reply_text(f"✅ Script {actual_id} saved from file.\n📦 Version saved (if it overwrote existing script).")
    await send_app_menu(update.effective_chat.id, actual_id, context)
    await send_env_menu(update.effective_chat.id, actual_id, context)
    return

  if lower.endswith(ARCHIVE_EXTENSIONS):
    await handle_archive_upload(update, context, doc, force_target_id=force_target_id)
    return

  if force_target_id:
    # Restore so the next message can still trigger a source change
    script_mgr.pending_source_change[user_id] = force_target_id

  await update.message.reply_text(
      "❌ Unsupported file type.\n"
      "Send a .py file, or an archive: .zip / .tar.gz / .tgz / .tar / .7z\n"
      "Send /cancel to abort."
  )


async def on_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
  if not is_authorized(update):
    return

  query = update.callback_query
  await query.answer()
  data = query.data or ""

  if data == "noop":
    return

  if data == "back:main":
    await query.edit_message_text("📋 Select script:", reply_markup=main_menu_keyboard())
    return

  if data.startswith("menu:"):
    script_id = data.split(":", 1)[1]
    if not script_mgr.get_script(script_id):
      await query.edit_message_text("❌ Script not found.")
      return
    await query.edit_message_text(script_mgr.script_status_line(script_id), reply_markup=menu_keyboard_for(script_id))
    return

  if data.startswith("run:"):
    script_id = data.split(":", 1)[1]
    msg = await script_mgr.restart_script(script_id)
    await query.edit_message_text(safe_trim_text(msg, 3900), reply_markup=menu_keyboard_for(script_id))
    return

  if data.startswith("stop:"):
    script_id = data.split(":", 1)[1]
    msg = await script_mgr.stop_script(script_id)
    await query.edit_message_text(msg, reply_markup=menu_keyboard_for(script_id))
    return

  if data.startswith("logs:"):
    script_id = data.split(":", 1)[1]
    if not script_mgr.get_script(script_id):
      await query.edit_message_text("❌ Script not found.")
      return
    text = script_mgr.read_log_tail(script_id, lines=LOG_TAIL_LINES)
    ts = datetime.now().strftime("%H:%M:%S")
    await context.bot.send_message(
        chat_id=query.message.chat_id,
        text=f"📜 [{ts}] Logs for {script_id} (last {LOG_TAIL_LINES} lines):\n{text}",
    )
    return

  if data.startswith("edit:"):
    script_id = data.split(":", 1)[1]
    script = script_mgr.get_script(script_id)
    if not script:
      await query.edit_message_text("❌ Script not found.")
      return
    if script.get("type", "script") != "script":
      hint = " Use 🔄 Sync to pull the latest code." if script.get("type") == "git" else " Re-upload an archive with the same name to update it."
      await query.edit_message_text(
          "✏️ Editing multi-file apps via chat isn't supported." + hint,
          reply_markup=menu_keyboard_for(script_id),
      )
      return
    user_id = query.from_user.id
    script_mgr.pending_edit[user_id] = script_id
    script_mgr.pending_new.pop(user_id, None)
    script_mgr.pending_env_value.pop(user_id, None)
    script_mgr.pending_pip.pop(user_id, None)
    await query.edit_message_text(
        f"✏️ Send new content for {script_id}.\n"
        f"📦 Current version will be saved (keep last {MAX_VERSIONS_PER_SCRIPT}).\n"
        f"If it is running, it will be restarted after update."
    )
    return

  if data.startswith("auto:"):
    script_id = data.split(":", 1)[1]
    script = script_mgr.get_script(script_id)
    if not script:
      await query.edit_message_text("❌ Script not found.")
      return
    current = bool(script.get("autostart"))
    script_mgr.set_autostart(script_id, not current)
    await query.edit_message_text(
        f"{'🚀' if not current else '⏸️'} Autostart for {script_id}: {'ON' if not current else 'OFF'}",
        reply_markup=menu_keyboard_for(script_id),
    )
    return

  if data.startswith("envmenu:"):
    script_id = data.split(":", 1)[1]
    if not script_mgr.get_script(script_id):
      await query.edit_message_text("❌ Script not found.")
      return
    await query.edit_message_text(env_menu_text(script_id), reply_markup=env_menu_keyboard(script_id))
    return

  if data.startswith("envset:"):
    _, script_id, key = data.split(":", 2)
    if not script_mgr.get_script(script_id):
      await query.edit_message_text("❌ Script not found.")
      return
    user_id = query.from_user.id
    script_mgr.pending_env_value[user_id] = {"script_id": script_id, "key": key}
    script_mgr.pending_new.pop(user_id, None)
    script_mgr.pending_edit.pop(user_id, None)
    script_mgr.pending_pip.pop(user_id, None)
    await query.edit_message_text(
        f"🔧 Send value for {key} (script {script_id}).\n"
        f"Send '-' to delete."
    )
    return

  if data.startswith("envdone:"):
    script_id = data.split(":", 1)[1]
    if not script_mgr.get_script(script_id):
      await query.edit_message_text("❌ Script not found.")
      return
    await query.edit_message_text("✅ Env updated.", reply_markup=menu_keyboard_for(script_id))
    return

  if data.startswith("envrun:"):
    script_id = data.split(":", 1)[1]
    if not script_mgr.get_script(script_id):
      await query.edit_message_text("❌ Script not found.")
      return
    msg = await script_mgr.restart_script(script_id)
    await query.edit_message_text(safe_trim_text(msg, 3900), reply_markup=menu_keyboard_for(script_id))
    return

  if data.startswith("pipprompt:"):
    script_id = data.split(":", 1)[1]
    user_id = query.from_user.id
    script_mgr.pending_pip[user_id] = script_id
    script_mgr.pending_new.pop(user_id, None)
    script_mgr.pending_edit.pop(user_id, None)
    script_mgr.pending_env_value.pop(user_id, None)
    scoped = script_mgr.is_project_type(script_id)
    await query.edit_message_text(
        "📦 Send pip install args (e.g. `python-telegram-bot==21.6 openai psutil`).\n"
        + ("Installed into this app's own venv and saved into its requirements.txt.\n" if scoped
           else "Saved into requirements.txt automatically.\n")
        + "Send '-' to cancel."
    )
    return

  if data.startswith("rbmenu:"):
    script_id = data.split(":", 1)[1]
    if not script_mgr.get_script(script_id):
      await query.edit_message_text("❌ Script not found.")
      return
    await query.edit_message_text(
        rollback_menu_text(script_id),
        reply_markup=rollback_menu_keyboard(script_id),
    )
    return

  if data.startswith("rbdo:"):
    # rbdo:<script_id>:<ts>
    _, script_id, ts = data.split(":", 2)
    if not script_mgr.get_script(script_id):
      await query.edit_message_text("❌ Script not found.")
      return
    msg = await script_mgr.rollback_to(script_id, ts)
    await query.edit_message_text(safe_trim_text(msg, 3900), reply_markup=menu_keyboard_for(script_id))
    return

  if data.startswith("entrypick:"):
    # entrypick:<script_id>:<index>
    _, script_id, idx_str = data.split(":", 2)
    candidates = script_mgr.pending_entry_choice.pop(script_id, None)
    script = script_mgr.get_script(script_id)
    if not script or not candidates:
      await query.edit_message_text("❌ Selection expired or app not found. Please re-upload / re-sync.")
      return
    try:
      idx = int(idx_str)
      entry = candidates[idx]
    except (ValueError, IndexError):
      await query.edit_message_text("❌ Invalid choice.")
      return

    script["entry"] = entry
    script_mgr.save_meta()
    await query.edit_message_text(f"✅ Entry point set to {entry}.")

    root = Path(script.get("root_dir", ""))
    result = {
        "status": "ready",
        "entry": entry,
        "venv_ok": True,
        "requirements_pkgs": find_requirements_pkgs(root) if root.exists() else [],
    }
    await present_setup_result(
        query.message.chat_id, context, script_id, result, f"✅ App {script_id} configured."
    )
    return

  if data.startswith("reqinstall:"):
    script_id = data.split(":", 1)[1]
    if not script_mgr.get_script(script_id):
      await query.edit_message_text("❌ App not found.")
      return
    await query.edit_message_text("📥 Installing requirements ...")
    install_msg = await script_mgr.install_app_requirements(script_id)
    run_msg = await script_mgr.start_script(script_id)
    await context.bot.send_message(
        chat_id=query.message.chat_id,
        text=safe_trim_text(f"{install_msg}\n\n{run_msg}", 3900),
        reply_markup=menu_keyboard_for(script_id),
    )
    return

  if data.startswith("reqskip:"):
    script_id = data.split(":", 1)[1]
    if not script_mgr.get_script(script_id):
      await query.edit_message_text("❌ App not found.")
      return
    run_msg = await script_mgr.start_script(script_id)
    await query.edit_message_text(run_msg, reply_markup=menu_keyboard_for(script_id))
    return

  if data.startswith("changesrc:"):
    # changesrc:<script_id> — user clicked "Change Source" from the app menu
    script_id = data.split(":", 1)[1]
    if not script_mgr.get_script(script_id):
      await query.edit_message_text("❌ App not found.")
      return
    user_id = query.from_user.id
    # Clear any other pending user-scoped states so we don't mix flows.
    script_mgr.pending_new.pop(user_id, None)
    script_mgr.pending_edit.pop(user_id, None)
    script_mgr.pending_env_value.pop(user_id, None)
    script_mgr.pending_pip.pop(user_id, None)
    script_mgr.pending_source_change[user_id] = script_id
    await query.edit_message_text(
        f"🔀 *{script_id}* — waiting for new source.\n\n"
        "Send:\n"
        "• A `.py` file → convert to 📄 script\n"
        "• An archive (`.zip` / `.tar.gz` / `.tgz` / `.tar` / `.7z`) → convert to 📦 archive app\n"
        "• A GitHub link → convert to 🌐 GitHub app\n\n"
        "Send /cancel to abort.",
        parse_mode="Markdown",
    )
    return

  if data.startswith("convtype:"):
    # convtype:<yes|no>:<script_id>
    parts = data.split(":", 2)
    if len(parts) != 3:
      return
    _, action, script_id = parts
    chat_id = query.message.chat_id

    if action == "no":
      script_mgr._clear_pending_conversion(script_id)
      await query.edit_message_text(f"🚫 Conversion of {script_id} cancelled.")
      return

    if action == "yes":
      stash = script_mgr.pending_type_conversion.pop(script_id, None)
      if not stash:
        await query.edit_message_text("❌ Conversion request expired or already executed.")
        return

      # Stop the running process before mutating the on-disk state.
      if script_mgr.script_running(script_id):
        await script_mgr.stop_script(script_id)

      stash_action = stash.get("action")

      if stash_action == "archive":
        staged_root = Path(stash["staged_root"])
        if not staged_root.exists():
          await query.edit_message_text("❌ Staged archive expired. Please re-upload the file.")
          return
        await query.edit_message_text(f"⚙️ Converting {script_id} to archive app ...")
        result = await script_mgr.setup_project_app(script_id, staged_root, "project")
        # Clean up any outer staging wrapper that might still exist.
        outer = stash.get("staging")
        if outer:
          shutil.rmtree(outer, ignore_errors=True)
        await query.message.delete()
        await present_setup_result(chat_id, context, script_id, result, f"✅ {script_id} converted to archive app.")

      elif stash_action == "git":
        staged_root = Path(stash["staged_root"])
        if not staged_root.exists():
          await query.edit_message_text("❌ Staged repo expired. Please re-send the GitHub link.")
          return
        git_info: Dict[str, Any] = stash["git_info"]
        await query.edit_message_text(f"⚙️ Converting {script_id} to GitHub app ...")
        result = await script_mgr.setup_project_app(script_id, staged_root, "git", git_info)
        result["app_id"] = script_id
        await query.message.delete()
        await present_setup_result(chat_id, context, script_id, result, f"✅ {script_id} converted to GitHub app.")

      elif stash_action == "script_text":
        text_body: str = stash["text"]
        await query.edit_message_text(f"⚙️ Converting {script_id} to script ...")
        full_id = script_mgr.upsert_script_from_text(script_id, text_body)
        await query.edit_message_text(f"✅ {full_id} converted to script.")
        await send_script_menu(chat_id, full_id, context)
        await send_env_menu(chat_id, full_id, context)

      elif stash_action == "script_file":
        original_name: str = stash["original_name"]
        file_bytes: bytes = stash["file_bytes"]
        tmp_conv = TMP_DIR / f"_conv_{script_id}_{int(datetime.now().timestamp())}.py"
        tmp_conv.write_bytes(file_bytes)
        try:
          await query.edit_message_text(f"⚙️ Converting {script_id} to script ...")
          # Pass override_id so the target app keeps its existing id regardless
          # of what the uploaded filename was (fixes "Change Source" with a
          # mismatched filename).
          full_id = script_mgr.upsert_script_from_file(original_name, tmp_conv, override_id=script_id)
        finally:
          tmp_conv.unlink(missing_ok=True)
        await query.edit_message_text(f"✅ {full_id} converted to script.")
        await send_script_menu(chat_id, full_id, context)
        await send_env_menu(chat_id, full_id, context)

      else:
        await query.edit_message_text("❌ Unknown conversion action.")
      return

  if data.startswith("sync:"):
    script_id = data.split(":", 1)[1]
    script = script_mgr.get_script(script_id)
    if not script or script.get("type") != "git":
      await query.edit_message_text("❌ App not found or not a GitHub app.")
      return
    await query.edit_message_text("🔄 Syncing ...")
    result = await sync_git_app(script_id)

    if result.get("status") == "error":
      await context.bot.send_message(
          chat_id=query.message.chat_id,
          text=f"❌ Sync failed:\n{safe_trim_text(result.get('message', ''), 3500)}",
          reply_markup=menu_keyboard_for(script_id),
      )
      return

    if result.get("status") == "ambiguous":
      await present_setup_result(query.message.chat_id, context, script_id, result, f"🔄 Synced {script_id}.")
      return

    run_msg = await script_mgr.start_script(script_id)
    lines = [f"🔄 Synced {script_id}.", run_msg]

    if result.get("deps_changed"):
      updated_script = script_mgr.get_script(script_id) or {}
      req_pkgs = find_requirements_pkgs(Path(updated_script.get("root_dir", "")))
      lines.append("")
      lines.append(f"📦 requirements.txt changed! ({len(req_pkgs)} package(s))" if req_pkgs else "📦 requirements.txt changed!")
      await context.bot.send_message(
          chat_id=query.message.chat_id,
          text=safe_trim_text("\n".join(lines), 3500),
          reply_markup=InlineKeyboardMarkup(
              [[InlineKeyboardButton("📥 Install new requirements & Restart", callback_data=f"reqinstall:{script_id}")]]
          ),
      )
    else:
      await context.bot.send_message(chat_id=query.message.chat_id, text=safe_trim_text("\n".join(lines), 3500))

    await send_app_menu(query.message.chat_id, script_id, context)
    return

  if data.startswith("createdb:"):
    script_id = data.split(":", 1)[1]
    if not script_mgr.get_script(script_id):
      await query.edit_message_text("❌ Script not found.")
      return

    if not pgdb.pg_enabled():
      await query.edit_message_text(
          "❌ Postgres is not configured.\n"
          "Set PG_ADMIN_PASSWORD (and optionally PG_ADMIN_USER) in your environment.",
          reply_markup=menu_keyboard_for(script_id),
      )
      return

    await query.edit_message_text("⏳ Provisioning database ...")
    loop = asyncio.get_event_loop()
    try:
      row = await loop.run_in_executor(None, pgdb.provision_database, script_id)
    except Exception as exc:
      logger.exception("Database provisioning failed for app %s", script_id)
      await context.bot.send_message(
          chat_id=query.message.chat_id,
          text=f"❌ Failed to provision database:\n{exc}",
          reply_markup=menu_keyboard_for(script_id),
      )
      return

    info = (
        f"🗄 Database provisioned for *{script_id}*\n\n"
        f"• `PG_DATABASE` = `{row['db_name']}`\n"
        f"• `PG_USERNAME` = `{row['username']}`\n"
        f"• `PG_PASSWORD` = `{row['password']}`\n"
        f"• `PG_HOST` = `{pgdb._APP_HOST}`\n"
        f"• `PG_PORT` = `{pgdb._APP_PORT}`\n\n"
        f"These are injected automatically when the app runs.\n"
        f"You can reference them in your app's Env vars, e.g.:\n"
        f"`DATABASE_URL=postgresql://${{PG_USERNAME}}:${{PG_PASSWORD}}@${{PG_HOST}}:${{PG_PORT}}/${{PG_DATABASE}}`"
    )
    await context.bot.send_message(
        chat_id=query.message.chat_id,
        text=info,
        parse_mode="Markdown",
        reply_markup=menu_keyboard_for(script_id),
    )
    return


# Startup

async def on_startup(app: Application) -> None:
  logger.info("Bot startup: install requirements + prepare app venvs + autostart")
  asyncio.create_task(trim_script_logs_periodically())
  await script_mgr.ensure_requirements_installed()
  await script_mgr.ensure_app_environments()
  await script_mgr.autostart_all()


def main() -> None:
  application = Application.builder().token(BOT_TOKEN).post_init(on_startup).build()

  application.add_handler(CommandHandler("start", cmd_start))
  application.add_handler(CommandHandler("cancel", cmd_cancel))
  application.add_handler(CommandHandler("menu", cmd_menu))
  application.add_handler(CommandHandler("list", cmd_list))
  application.add_handler(CommandHandler("new", cmd_new))
  application.add_handler(CommandHandler("repo", cmd_repo))

  application.add_handler(CommandHandler("run", cmd_run))
  application.add_handler(CommandHandler("stop", cmd_stop))
  application.add_handler(CommandHandler("logs", cmd_logs))

  application.add_handler(CommandHandler("monitoring", cmd_monitoring))

  application.add_handler(CommandHandler("envmenu", cmd_envmenu))
  application.add_handler(CommandHandler("env", cmd_env))
  application.add_handler(CommandHandler("setenv", cmd_setenv))
  application.add_handler(CommandHandler("delenv", cmd_delenv))

  application.add_handler(CommandHandler("globalenv", cmd_globalenv))
  application.add_handler(CommandHandler("setglobal", cmd_setglobal))
  application.add_handler(CommandHandler("delglobal", cmd_delglobal))
  application.add_handler(CommandHandler("autostart", cmd_autostart))

  application.add_handler(CommandHandler("pip", cmd_pip))

  application.add_handler(CallbackQueryHandler(on_callback))
  application.add_handler(MessageHandler(filters.Document.ALL, handle_document))
  application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

  application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
  main()
