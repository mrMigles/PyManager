# script_manager_bot.py
# -*- coding: utf-8 -*-
"""
Telegram Script Manager Bot

Features:
- Manage Python scripts: upload/edit/run/stop/logs/env/autostart
- Script versioning: keep last 10 versions with timestamps + rollback menu
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
import asyncio
import logging
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List
from collections import deque
from datetime import datetime

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

DATA_DIR.mkdir(parents=True, exist_ok=True)
SCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)
VERSIONS_DIR.mkdir(parents=True, exist_ok=True)

# Logging
bot_log_path = DATA_DIR / "bot.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
      logging.StreamHandler(sys.stdout),
      logging.FileHandler(str(bot_log_path), encoding="utf-8"),
    ],
)
logger = logging.getLogger("script-bot")

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


def read_text_file(path: Path) -> str:
  return path.read_text(encoding="utf-8", errors="replace")


def write_text_file_atomic(path: Path, content: str) -> None:
  tmp = path.with_suffix(path.suffix + ".tmp")
  tmp.write_text(content, encoding="utf-8")
  tmp.replace(path)


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

  def _read_requirements_lines(self) -> List[str]:
    if not REQUIREMENTS_FILE.exists():
      return []
    lines = []
    for raw in read_text_file(REQUIREMENTS_FILE).splitlines():
      line = raw.strip()
      if not line or line.startswith("#"):
        continue
      lines.append(line)
    return lines

  def add_requirements(self, pkgs: List[str]) -> bool:
    if not pkgs:
      return False
    existing = self._read_requirements_lines()
    existing_set = set(existing)
    changed = False
    for p in pkgs:
      if p and p not in existing_set:
        existing.append(p)
        existing_set.add(p)
        changed = True
    if changed:
      content = "\n".join(existing) + "\n"
      write_text_file_atomic(REQUIREMENTS_FILE, content)
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
    script["file"] = file_path
    scripts[script_id] = script
    self.save_meta()

  def get_script(self, script_id: str) -> Optional[Dict[str, Any]]:
    return self.meta.get("scripts", {}).get(script_id)

  def script_file_path(self, script_id: str) -> Path:
    return SCRIPTS_DIR / f"{script_id}.py"

  def versions_dir_for(self, script_id: str) -> Path:
    d = VERSIONS_DIR / script_id
    d.mkdir(parents=True, exist_ok=True)
    return d

  def _push_version(self, script_id: str, version_file: Path, ts: str) -> None:
    script = self.get_script(script_id)
    if not script:
      return
    versions = script.setdefault("versions", [])
    if not isinstance(versions, list):
      versions = []
      script["versions"] = versions

    versions.insert(0, {"ts": ts, "file": str(version_file)})

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
    version_file = vdir / f"{script_id}_{ts}.py"
    try:
      shutil.copy2(cur_path, version_file)
      self._push_version(script_id, version_file, ts)
      logger.info("Snapshot version for %s -> %s", script_id, version_file.name)
    except Exception:
      logger.exception("Failed to snapshot version for %s", script_id)

  def upsert_script_from_text(self, name: str, content: str) -> str:
    script_id = name.strip().replace(" ", "_")
    if not script_id:
      raise ValueError("empty script name")

    # If script already exists, snapshot before overwriting
    if self.get_script(script_id) and self.script_file_path(script_id).exists():
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

  def upsert_script_from_file(self, original_name: str, content_path: Path) -> str:
    stem = Path(original_name).stem
    script_id = stem.strip().replace(" ", "_")
    if not script_id:
      script_id = f"script_{len(self.meta.get('scripts', {})) + 1}"

    # Snapshot before overwriting
    if self.get_script(script_id) and self.script_file_path(script_id).exists():
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

    file_path = script.get("file")
    if not file_path or not Path(file_path).exists():
      return "❌ Script file not found."

    env = os.environ.copy()
    env.update(self.meta.get("global_env", {}))
    env.update(script.get("env", {}))

    log_file = self.log_path(script_id)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("ab") as f:
      f.write(b"\n=== START ===\n")

    # Keep handle open while process runs (important!)
    h = log_file.open("ab")
    self.log_handles[script_id] = h

    logger.info("Starting script %s", script_id)
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        file_path,
        cwd=str(SCRIPTS_DIR),
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

    file_path = script.get("file")
    if file_path and Path(file_path).exists():
      try:
        source = read_text_file(Path(file_path))
        detected = extract_env_keys(source)
        keys |= detected
      except Exception:
        logger.exception("Failed to extract env keys for %s", script_id)
    return sorted(keys)

  def get_versions(self, script_id: str) -> List[Dict[str, str]]:
    script = self.get_script(script_id)
    if not script:
      return []
    versions = script.get("versions", [])
    if not isinstance(versions, list):
      return []
    out: List[Dict[str, str]] = []
    for v in versions:
      if not isinstance(v, dict):
        continue
      ts = str(v.get("ts", "")).strip()
      fp = str(v.get("file", "")).strip()
      if ts and fp:
        out.append({"ts": ts, "file": fp})
    return out[:MAX_VERSIONS_PER_SCRIPT]

  async def rollback_to(self, script_id: str, ts: str) -> str:
    script = self.get_script(script_id)
    if not script:
      return "❌ Script not found."

    versions = self.get_versions(script_id)
    chosen = None
    for v in versions:
      if v["ts"] == ts:
        chosen = v
        break
    if not chosen:
      return "❌ Version not found."

    version_path = Path(chosen["file"])
    if not version_path.exists():
      return "❌ Version file missing on disk."

    was_running = self.script_running(script_id)
    if was_running:
      await self.stop_script(script_id)

    # Snapshot current before rollback
    if self.script_file_path(script_id).exists():
      self.snapshot_current_version(script_id)

    # Replace current script with chosen version (copy, keep version file)
    try:
      shutil.copy2(version_path, self.script_file_path(script_id))
    except Exception:
      logger.exception("Rollback copy failed for %s", script_id)
      return "❌ Rollback failed (copy error)."

    # Update updated_at
    script = self.get_script(script_id)
    if script:
      script["updated_at"] = now_ts_str()
      self.save_meta()

    msg = f"⏪ Rolled back *{script_id}* to version {pretty_dt_from_ts(ts)}".replace("*", "")

    if was_running:
      s = await self.start_script(script_id)
      msg += f"\n{s}"

    return msg

  # pip

  async def pip_install(self, args_str: str) -> str:
    try:
      args = shlex.split(args_str)
    except ValueError as e:
      return f"❌ Bad args: {e}"
    if not args:
      return "❌ No packages specified."

    logger.info("Running pip install via %s: %s", sys.executable, " ".join(args))
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
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

    saved = self.add_requirements(to_store)

    msg = []
    msg.append(f"$ {sys.executable} -m pip install {' '.join(args)}")
    msg.append(f"exit code: {exit_code}")
    if saved:
      msg.append(f"📌 saved to requirements.txt ({len(to_store)} item(s))")
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
          sys.executable,
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
    au = self._autostart_emoji(script_id)
    pid = self._pid(script_id)
    vc = self._versions_count(script_id)
    upd = self._updated_at_pretty(script_id)
    return f"{st} {au} {script_id}  | PID: {pid} | v:{vc} | updated: {upd}"

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
    text = f"{script_mgr._status_emoji(sid)} {script_mgr._autostart_emoji(sid)} {sid}"
    buttons.append([InlineKeyboardButton(text=text, callback_data=f"menu:{sid}")])
  if not buttons:
    buttons.append([InlineKeyboardButton(text="😴 No scripts", callback_data="noop")])
  return InlineKeyboardMarkup(buttons)


def script_menu_keyboard(script_id: str) -> InlineKeyboardMarkup:
  script = script_mgr.get_script(script_id)
  autostart = script.get("autostart") if script else False
  running = script_mgr.script_running(script_id)

  run_text = "▶ Run" if not running else "🔁 Restart"
  stop_text = "⏹ Stop"
  auto_text = "🚀 Autostart: ON" if autostart else "⏸️ Autostart: OFF"

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
      InlineKeyboardButton("⬅ Back", callback_data="back:main"),
    ],
  ]
  return InlineKeyboardMarkup(buttons)


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


async def send_script_menu(chat_id: int, script_id: str, context: ContextTypes.DEFAULT_TYPE) -> None:
  await context.bot.send_message(chat_id=chat_id, text=script_mgr.script_status_line(script_id), reply_markup=script_menu_keyboard(script_id))


# Commands

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
  if not is_authorized(update):
    return
  await update.message.reply_text(
      "🤖 Script manager ready.\n\n"
      "Main:\n"
      "• /menu — scripts menu\n"
      "• /list — list scripts (rich)\n"
      "• /new <name> — create script from next text message\n"
      "• /run <id>, /stop <id>\n"
      "• /logs <id>\n"
      "• /monitoring — CPU/MEM for running scripts\n\n"
      "Packages:\n"
      "• /pip <args> — pip install (saved to requirements.txt)\n\n"
      "Env:\n"
      "• /envmenu <id>\n"
      "• /env <id>, /setenv, /delenv\n"
      "• /globalenv, /setglobal, /delglobal\n\n"
      "Tip: For /monitoring install psutil via /pip psutil"
  )


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
      msg = await script_mgr.pip_install(value)
      await update.message.reply_text(msg)
    if script_id:
      await send_script_menu(update.effective_chat.id, script_id, context)
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

  logger.info("Ignored text from user_id=%s (no pending state)", user_id)


async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
  if not is_authorized(update):
    return
  doc = update.message.document
  if not doc.file_name.lower().endswith(".py"):
    await update.message.reply_text("❌ Only .py files are accepted.")
    return

  tmp_path = SCRIPTS_DIR / ("_upload_tmp_" + doc.file_name)
  file = await doc.get_file()
  await file.download_to_drive(str(tmp_path))

  script_id = script_mgr.upsert_script_from_file(doc.file_name, tmp_path)
  tmp_path.unlink(missing_ok=True)

  await update.message.reply_text(f"✅ Script {script_id} saved from file.\n📦 Version saved (if it overwrote existing script).")
  await send_script_menu(update.effective_chat.id, script_id, context)
  await send_env_menu(update.effective_chat.id, script_id, context)


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
    await query.edit_message_text(script_mgr.script_status_line(script_id), reply_markup=script_menu_keyboard(script_id))
    return

  if data.startswith("run:"):
    script_id = data.split(":", 1)[1]
    msg = await script_mgr.restart_script(script_id)
    await query.edit_message_text(safe_trim_text(msg, 3900), reply_markup=script_menu_keyboard(script_id))
    return

  if data.startswith("stop:"):
    script_id = data.split(":", 1)[1]
    msg = await script_mgr.stop_script(script_id)
    await query.edit_message_text(msg, reply_markup=script_menu_keyboard(script_id))
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
    if not script_mgr.get_script(script_id):
      await query.edit_message_text("❌ Script not found.")
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
        reply_markup=script_menu_keyboard(script_id),
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
    await query.edit_message_text("✅ Env updated.", reply_markup=script_menu_keyboard(script_id))
    return

  if data.startswith("envrun:"):
    script_id = data.split(":", 1)[1]
    if not script_mgr.get_script(script_id):
      await query.edit_message_text("❌ Script not found.")
      return
    msg = await script_mgr.restart_script(script_id)
    await query.edit_message_text(safe_trim_text(msg, 3900), reply_markup=script_menu_keyboard(script_id))
    return

  if data.startswith("pipprompt:"):
    script_id = data.split(":", 1)[1]
    user_id = query.from_user.id
    script_mgr.pending_pip[user_id] = script_id
    script_mgr.pending_new.pop(user_id, None)
    script_mgr.pending_edit.pop(user_id, None)
    script_mgr.pending_env_value.pop(user_id, None)
    await query.edit_message_text(
        "📦 Send pip install args (e.g. `python-telegram-bot==21.6 openai psutil`).\n"
        "Saved into requirements.txt automatically.\n"
        "Send '-' to cancel."
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
    await query.edit_message_text(safe_trim_text(msg, 3900), reply_markup=script_menu_keyboard(script_id))
    return


# Startup

async def on_startup(app: Application) -> None:
  logger.info("Bot startup: install requirements + autostart scripts")
  await script_mgr.ensure_requirements_installed()
  await script_mgr.autostart_all()


def main() -> None:
  application = Application.builder().token(BOT_TOKEN).post_init(on_startup).build()

  application.add_handler(CommandHandler("start", cmd_start))
  application.add_handler(CommandHandler("menu", cmd_menu))
  application.add_handler(CommandHandler("list", cmd_list))
  application.add_handler(CommandHandler("new", cmd_new))

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
