import os
import sys
import json
import re
import shlex
import asyncio
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from collections import deque
from datetime import datetime

from telegram import (
    Update,
    InlineKeyboardMarkup,
    InlineKeyboardButton,
)
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    ContextTypes,
    filters,
)

# Paths
DATA_DIR = Path(os.getenv("DATA_DIR", "/data"))
SCRIPTS_DIR = DATA_DIR / "scripts"
LOGS_DIR = DATA_DIR / "logs"
META_FILE = DATA_DIR / "meta.json"

DATA_DIR.mkdir(parents=True, exist_ok=True)
SCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Bot logging
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


def is_authorized(update: Update) -> bool:
    user = update.effective_user
    ok = bool(user and user.id == OWNER_ID)
    if not ok:
        logger.warning("Unauthorized access attempt from user_id=%s", getattr(user, "id", None))
    return ok


def extract_env_keys(source: str) -> set[str]:
    keys: set[str] = set()
    pattern1 = r"(?:os\.)?(?:getenv|environ\.get)\(\s*['\"]([A-Za-z_][A-Za-z0-9_]*)['\"]"
    pattern2 = r"(?:os\.)?environ\[\s*['\"]([A-Za-z_][A-Za-z0-9_]*)['\"]\s*\]"
    for pat in (pattern1, pattern2):
        for m in re.findall(pat, source):
            keys.add(m)
    return keys


class ScriptManager:
    def __init__(self) -> None:
        self.meta: Dict[str, Any] = {"global_env": {}, "scripts": {}, "pip_packages": []}
        self.processes: Dict[str, asyncio.subprocess.Process] = {}
        self.pending_new: Dict[int, str] = {}
        self.pending_edit: Dict[int, str] = {}
        self.pending_env_value: Dict[int, Dict[str, str]] = {}
        self.pending_pip: Dict[int, Optional[str]] = {}
        self.load_meta()

    def load_meta(self) -> None:
        if META_FILE.exists():
            try:
                with META_FILE.open("r", encoding="utf-8") as f:
                    self.meta = json.load(f)
            except Exception:
                logger.exception("Failed to load meta.json, starting with empty meta")
                self.meta = {"global_env": {}, "scripts": {}, "pip_packages": []}
        # ensure keys exist
        self.meta.setdefault("global_env", {})
        self.meta.setdefault("scripts", {})
        self.meta.setdefault("pip_packages", [])
        self.save_meta()

    def save_meta(self) -> None:
        try:
            tmp = META_FILE.with_suffix(".tmp")
            with tmp.open("w", encoding="utf-8") as f:
                json.dump(self.meta, f, indent=2, ensure_ascii=False)
            tmp.replace(META_FILE)
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

    # pip meta

    def get_pip_packages(self) -> list[str]:
        pkgs = self.meta.get("pip_packages", [])
        if not isinstance(pkgs, list):
            pkgs = []
        return list(pkgs)

    def add_pip_packages(self, pkgs: list[str]) -> None:
        if not pkgs:
            return
        meta_pkgs = self.meta.setdefault("pip_packages", [])
        changed = False
        for p in pkgs:
            if p and p not in meta_pkgs:
                meta_pkgs.append(p)
                changed = True
        if changed:
            logger.info("Added pip packages to meta: %s", ", ".join(pkgs))
            self.save_meta()

    async def ensure_pip_packages_installed(self) -> None:
        pkgs = self.get_pip_packages()
        if not pkgs:
            logger.info("No pip packages in meta, skipping startup install")
            return
        logger.info("Ensuring pip packages installed on startup: %s", " ".join(pkgs))
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "pip",
            "install",
            *pkgs,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        out, _ = await proc.communicate()
        text = out.decode("utf-8", errors="replace")
        if len(text) > 4000:
            text = "...\n" + text[-4000:]
        logger.info("Startup pip install exit=%s, output:\n%s", proc.returncode, text)

    # scripts

    def upsert_script_from_text(self, name: str, content: str) -> str:
        script_id = name.strip().replace(" ", "_")
        if not script_id:
            raise ValueError("empty script name")
        file_path = SCRIPTS_DIR / f"{script_id}.py"
        with file_path.open("w", encoding="utf-8") as f:
            f.write(content)
        self._upsert_script_meta(script_id, str(file_path))
        logger.info("Script %s saved from text", script_id)
        return script_id

    def upsert_script_from_file(self, original_name: str, content_path: Path) -> str:
        stem = Path(original_name).stem
        script_id = stem.strip().replace(" ", "_")
        if not script_id:
            script_id = f"script_{len(self.meta.get('scripts', {})) + 1}"
        file_path = SCRIPTS_DIR / f"{script_id}.py"
        content = content_path.read_text(encoding="utf-8")
        with file_path.open("w", encoding="utf-8") as f:
            f.write(content)
        self._upsert_script_meta(script_id, str(file_path))
        logger.info("Script %s saved from file %s", script_id, original_name)
        return script_id

    def _upsert_script_meta(self, script_id: str, file_path: str) -> None:
        scripts = self.meta.setdefault("scripts", {})
        script = scripts.get(script_id, {})
        script.setdefault("env", {})
        script.setdefault("autostart", False)
        script["file"] = file_path
        scripts[script_id] = script
        self.save_meta()

    def get_script(self, script_id: str) -> Optional[Dict[str, Any]]:
        return self.meta.get("scripts", {}).get(script_id)

    def set_script_env(self, script_id: str, key: str, value: str) -> None:
        scripts = self.meta.setdefault("scripts", {})
        script = scripts.setdefault(script_id, {"env": {}, "autostart": False})
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

    def script_status_line(self, script_id: str) -> str:
        script = self.get_script(script_id)
        if not script:
            return f"{script_id} [missing]"
        running = self.script_running(script_id)
        status = "RUNNING" if running else "STOPPED"
        mark = "A" if script.get("autostart") else "-"
        return f"{script_id} [{status}, autostart={mark}]"

    def get_status_lines(self) -> str:
        scripts = self.meta.get("scripts", {})
        if not scripts:
            return "No scripts yet."
        return "\n".join(self.script_status_line(sid) for sid in scripts.keys())

    def log_path(self, script_id: str) -> Path:
        return LOGS_DIR / f"{script_id}.log"

    async def start_script(self, script_id: str) -> str:
        script = self.get_script(script_id)
        if not script:
            return "Script not found."
        proc = self.processes.get(script_id)
        if proc and proc.returncode is None:
            return "Already running."
        file_path = script.get("file")
        if not file_path or not Path(file_path).exists():
            return "Script file not found."
        env = os.environ.copy()
        env.update(self.meta.get("global_env", {}))
        env.update(script.get("env", {}))

        log_file = self.log_path(script_id)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with log_file.open("ab") as f:
            f.write(b"\n=== START ===\n")
        f = log_file.open("ab")

        logger.info("Starting script %s", script_id)
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            file_path,
            cwd=str(SCRIPTS_DIR),
            env=env,
            stdout=f,
            stderr=f,
        )
        self.processes[script_id] = proc
        return f"Started {script_id} with PID {proc.pid}."

    async def stop_script(self, script_id: str) -> str:
        proc = self.processes.get(script_id)
        if not proc:
            return "Not running."
        if proc.returncode is not None:
            self.processes.pop(script_id, None)
            return "Already finished."
        logger.info("Stopping script %s", script_id)
        proc.terminate()
        try:
            await asyncio.wait_for(proc.wait(), timeout=10)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
        self.processes.pop(script_id, None)
        with self.log_path(script_id).open("ab") as f:
            f.write(b"\n=== STOP ===\n")
        return f"Stopped {script_id}."

    async def autostart_all(self) -> None:
        scripts = self.meta.get("scripts", {})
        for script_id, script in scripts.items():
            if script.get("autostart"):
                try:
                    logger.info("Autostarting script %s", script_id)
                    await self.start_script(script_id)
                except Exception:
                    logger.exception("Autostart failed for %s", script_id)

    def script_running(self, script_id: str) -> bool:
        proc = self.processes.get(script_id)
        return bool(proc and proc.returncode is None)

    def read_log_tail(self, script_id: str, lines: int = 10) -> str:
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

    def get_env_keys_for_script(self, script_id: str) -> list[str]:
        script = self.get_script(script_id)
        if not script:
            return []
        keys: set[str] = set(script.get("env", {}).keys())
        file_path = script.get("file")
        if file_path and Path(file_path).exists():
            try:
                source = Path(file_path).read_text(encoding="utf-8", errors="ignore")
                detected = extract_env_keys(source)
                if detected:
                    logger.info("Detected env keys for %s: %s", script_id, ", ".join(sorted(detected)))
                keys |= detected
            except Exception:
                logger.exception("Failed to extract env keys for %s", script_id)
        return sorted(keys)

    async def pip_install(self, args_str: str) -> str:
        try:
            args = shlex.split(args_str)
        except ValueError as e:
            return f"Bad args: {e}"
        if not args:
            return "No packages specified."

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

        if len(text) > 3000:
            text = "...\n" + text[-3000:]

        msg = []
        msg.append(f"$ {sys.executable} -m pip install {' '.join(args)}")
        msg.append(f"exit code: {exit_code}")
        msg.append("")
        msg.append(text.strip() or "<no output>")

        # collect packages (for meta) and import checks
        to_check = []
        to_store: list[str] = []
        for a in args:
            if not a or a.startswith("-"):
                continue
            pkg = a.split("==")[0].split("[")[0]
            to_store.append(a)
            if pkg in ("openai", "python-telegram-bot"):
                to_check.append(pkg)

        # save to meta for future restarts
        self.add_pip_packages(to_store)

        if to_check:
            msg.append("")
            msg.append("Import check:")
        for pkg in to_check:
            code = f"import {pkg}; import sys; print({pkg}.__file__)"
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

        return "\n".join(msg)


script_mgr = ScriptManager()


def main_menu_keyboard() -> InlineKeyboardMarkup:
    scripts = script_mgr.list_scripts()
    buttons = []
    for script_id in scripts.keys():
        buttons.append(
            [InlineKeyboardButton(text=script_id, callback_data=f"menu:{script_id}")]
        )
    if not buttons:
        buttons.append([InlineKeyboardButton(text="No scripts", callback_data="noop")])
    return InlineKeyboardMarkup(buttons)


def script_menu_keyboard(script_id: str) -> InlineKeyboardMarkup:
    script = script_mgr.get_script(script_id)
    autostart = script.get("autostart") if script else False
    running = script_mgr.script_running(script_id)
    run_text = "▶ Run" if not running else "🔁 Restart"
    stop_text = "⏹ Stop"
    auto_text = "Autostart: ON" if autostart else "Autostart: OFF"

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
            InlineKeyboardButton("Env", callback_data=f"envmenu:{script_id}"),
            InlineKeyboardButton("📦 Pip install", callback_data=f"pipprompt:{script_id}"),
        ],
        [
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
        return "Script not found."
    env = script.get("env", {})
    keys = script_mgr.get_env_keys_for_script(script_id)
    if not keys:
        return f"No env vars detected for {script_id}."
    lines = []
    for k in keys:
        v = env.get(k, "")
        if v == "":
            lines.append(f"{k}=<not set>")
        else:
            lines.append(f"{k}={v}")
    return f"Env for {script_id}:\n" + "\n".join(lines)


def env_menu_keyboard(script_id: str) -> InlineKeyboardMarkup:
    keys = script_mgr.get_env_keys_for_script(script_id)
    buttons: list[list[InlineKeyboardButton]] = []
    for k in keys:
        buttons.append(
            [InlineKeyboardButton(k, callback_data=f"envset:{script_id}:{k}")]
        )
    buttons.append(
        [
            InlineKeyboardButton("✅ Done", callback_data=f"envdone:{script_id}"),
            InlineKeyboardButton("▶ Run", callback_data=f"envrun:{script_id}"),
        ]
    )
    buttons.append(
        [InlineKeyboardButton("⬅ Back", callback_data=f"menu:{script_id}")]
    )
    return InlineKeyboardMarkup(buttons)


async def send_env_menu(chat_id: int, script_id: str, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = env_menu_text(script_id)
    await context.bot.send_message(
        chat_id=chat_id,
        text=text,
        reply_markup=env_menu_keyboard(script_id),
    )


async def send_script_menu(chat_id: int, script_id: str, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = script_mgr.script_status_line(script_id)
    await context.bot.send_message(
        chat_id=chat_id,
        text=text,
        reply_markup=script_menu_keyboard(script_id),
    )


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_authorized(update):
        return
    logger.info("/start from user_id=%s", update.effective_user.id)
    await update.message.reply_text(
        "Script bot ready.\n"
        "/menu - scripts menu\n"
        "/list - list scripts\n"
        "/new <name> - create script from next text message\n"
        "/run <id>, /stop <id>\n"
        "/pip <args> - run pip install\n"
        "/envmenu <id> - env menu for script\n"
        "/env <id>, /setenv, /delenv\n"
        "/globalenv, /setglobal, /delglobal\n"
        "Or use inline buttons via /menu."
    )


async def cmd_menu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_authorized(update):
        return
    logger.info("/menu from user_id=%s", update.effective_user.id)
    await update.message.reply_text(
        "Select script:", reply_markup=main_menu_keyboard()
    )


async def cmd_list(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_authorized(update):
        return
    logger.info("/list from user_id=%s", update.effective_user.id)
    text = script_mgr.get_status_lines()
    await update.message.reply_text(text)


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
    logger.info("/new %s from user_id=%s", script_id, user_id)
    script_mgr.pending_new[user_id] = script_id
    script_mgr.pending_edit.pop(user_id, None)
    script_mgr.pending_env_value.pop(user_id, None)
    script_mgr.pending_pip.pop(user_id, None)
    await update.message.reply_text(
        f"Send script body as text. It will be saved as {script_id}.py"
    )


async def cmd_envmenu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_authorized(update):
        return
    if not context.args:
        await update.message.reply_text("Usage: /envmenu <script_id>")
        return
    script_id = context.args[0]
    if not script_mgr.get_script(script_id):
        await update.message.reply_text("Script not found.")
        return
    logger.info("/envmenu %s from user_id=%s", script_id, update.effective_user.id)
    await send_env_menu(update.effective_chat.id, script_id, context)


async def cmd_pip(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_authorized(update):
        return
    if not context.args:
        await update.message.reply_text("Usage: /pip <pip install args>")
        return
    args_str = " ".join(context.args)
    logger.info("/pip %s from user_id=%s", args_str, update.effective_user.id)
    msg = await script_mgr.pip_install(args_str)
    await update.message.reply_text(msg)


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_authorized(update):
        return
    user_id = update.effective_user.id
    text = update.message.text

    if user_id in script_mgr.pending_pip:
        script_id = script_mgr.pending_pip.pop(user_id)
        value = text.strip()
        if value == "-":
            await update.message.reply_text("Pip install cancelled.")
        else:
            msg = await script_mgr.pip_install(value)
            await update.message.reply_text(msg)
        if script_id:
            await send_script_menu(update.effective_chat.id, script_id, context)
        return

    if user_id in script_mgr.pending_env_value:
        info = script_mgr.pending_env_value.pop(user_id)
        script_id = info["script_id"]
        key = info["key"]
        value = text.strip()
        if value == "-":
            script_mgr.del_script_env(script_id, key)
            await update.message.reply_text(
                f"Deleted {key} for {script_id}."
            )
        else:
            script_mgr.set_script_env(script_id, key, text)
            await update.message.reply_text(
                f"Set {key} for {script_id}."
            )
        await send_env_menu(update.effective_chat.id, script_id, context)
        return

    if user_id in script_mgr.pending_new:
        script_id = script_mgr.pending_new.pop(user_id)
        full_id = script_mgr.upsert_script_from_text(script_id, text)
        await update.message.reply_text(f"Script {full_id} saved.")
        await send_script_menu(update.effective_chat.id, full_id, context)
        await send_env_menu(update.effective_chat.id, full_id, context)
        return

    if user_id in script_mgr.pending_edit:
        script_id = script_mgr.pending_edit.pop(user_id)
        was_running = script_mgr.script_running(script_id)
        script_mgr.upsert_script_from_text(script_id, text)
        msg = f"Script {script_id} updated."
        if was_running:
            stop_msg = await script_mgr.stop_script(script_id)
            start_msg = await script_mgr.start_script(script_id)
            msg += f"\n{stop_msg}\n{start_msg}"
        await update.message.reply_text(msg)
        await send_script_menu(update.effective_chat.id, script_id, context)
        await send_env_menu(update.effective_chat.id, script_id, context)
        return

    logger.info("Ignored text from user_id=%s (no pending state)", user_id)


async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_authorized(update):
        return
    doc = update.message.document
    if not doc.file_name.lower().endswith(".py"):
        await update.message.reply_text("Only .py files are accepted.")
        return
    tmp_path = SCRIPTS_DIR / ("_upload_tmp_" + doc.file_name)
    file = await doc.get_file()
    await file.download_to_drive(str(tmp_path))
    script_id = script_mgr.upsert_script_from_file(doc.file_name, tmp_path)
    tmp_path.unlink(missing_ok=True)
    await update.message.reply_text(f"Script {script_id} saved from file.")
    await send_script_menu(update.effective_chat.id, script_id, context)
    await send_env_menu(update.effective_chat.id, script_id, context)


async def cmd_run(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_authorized(update):
        return
    if not context.args:
        await update.message.reply_text("Usage: /run <script_id>")
        return
    script_id = context.args[0]
    logger.info("/run %s from user_id=%s", script_id, update.effective_user.id)
    msg = await script_mgr.start_script(script_id)
    await update.message.reply_text(msg)


async def cmd_stop(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_authorized(update):
        return
    if not context.args:
        await update.message.reply_text("Usage: /stop <script_id>")
        return
    script_id = context.args[0]
    logger.info("/stop %s from user_id=%s", script_id, update.effective_user.id)
    msg = await script_mgr.stop_script(script_id)
    await update.message.reply_text(msg)


async def cmd_env(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_authorized(update):
        return
    if not context.args:
        await update.message.reply_text("Usage: /env <script_id>")
        return
    script_id = context.args[0]
    script = script_mgr.get_script(script_id)
    if not script:
        await update.message.reply_text("Script not found.")
        return
    env = script.get("env", {})
    if not env:
        await update.message.reply_text("No env vars for this script.")
        return
    lines = [f"{k}={v}" for k, v in env.items()]
    await update.message.reply_text("Script env:\n" + "\n".join(lines))


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
        await update.message.reply_text("Script not found.")
        return
    script_mgr.set_script_env(script_id, key, value)
    await update.message.reply_text(f"Set {key} for {script_id}.")


async def cmd_delenv(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_authorized(update):
        return
    if len(context.args) < 2:
        await update.message.reply_text("Usage: /delenv <script_id> <KEY>")
        return
    script_id = context.args[0]
    key = context.args[1]
    ok = script_mgr.del_script_env(script_id, key)
    if not ok:
        await update.message.reply_text("Nothing to delete.")
    else:
        await update.message.reply_text(f"Deleted {key} from {script_id}.")


async def cmd_globalenv(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_authorized(update):
        return
    env = script_mgr.get_global_env()
    if not env:
        await update.message.reply_text("No global env vars.")
        return
    lines = [f"{k}={v}" for k, v in env.items()]
    await update.message.reply_text("Global env:\n" + "\n".join(lines))


async def cmd_setglobal(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_authorized(update):
        return
    if len(context.args) < 2:
        await update.message.reply_text("Usage: /setglobal <KEY> <VALUE>")
        return
    key = context.args[0]
    value = " ".join(context.args[1:])
    script_mgr.set_global_env(key, value)
    await update.message.reply_text(f"Set global {key}.")


async def cmd_delglobal(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_authorized(update):
        return
    if len(context.args) < 1:
        await update.message.reply_text("Usage: /delglobal <KEY>")
        return
    key = context.args[0]
    ok = script_mgr.del_global_env(key)
    if not ok:
        await update.message.reply_text("Nothing to delete.")
    else:
        await update.message.reply_text(f"Deleted global {key}.")


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
        await update.message.reply_text("Script not found.")
        return
    await update.message.reply_text(
        f"Autostart for {script_id} set to {value}."
    )


async def cmd_logs(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_authorized(update):
        return
    if not context.args:
        await update.message.reply_text("Usage: /logs <script_id>")
        return
    script_id = context.args[0]
    if not script_mgr.get_script(script_id):
        await update.message.reply_text("Script not found.")
        return
    text = script_mgr.read_log_tail(script_id, lines=10)
    ts = datetime.now().strftime("%H:%M:%S")
    await update.message.reply_text(f"[{ts}] Logs for {script_id} (last 10 lines):\n{text}")


async def on_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_authorized(update):
        return
    query = update.callback_query
    await query.answer()
    data = query.data or ""
    logger.info("Callback data=%s from user_id=%s", data, query.from_user.id)

    if data == "noop":
        return
    if data == "back:main":
        await query.edit_message_text(
            "Select script:", reply_markup=main_menu_keyboard()
        )
        return
    if data.startswith("menu:"):
        script_id = data.split(":", 1)[1]
        if not script_mgr.get_script(script_id):
            await query.edit_message_text("Script not found.")
            return
        info = script_mgr.script_status_line(script_id)
        await query.edit_message_text(
            info, reply_markup=script_menu_keyboard(script_id)
        )
        return
    if data.startswith("run:"):
        script_id = data.split(":", 1)[1]
        msg = await script_mgr.start_script(script_id)
        await query.edit_message_text(
            msg, reply_markup=script_menu_keyboard(script_id)
        )
        return
    if data.startswith("stop:"):
        script_id = data.split(":", 1)[1]
        msg = await script_mgr.stop_script(script_id)
        await query.edit_message_text(
            msg, reply_markup=script_menu_keyboard(script_id)
        )
        return
    if data.startswith("logs:"):
        script_id = data.split(":", 1)[1]
        if not script_mgr.get_script(script_id):
            await query.edit_message_text("Script not found.")
            return
        text = script_mgr.read_log_tail(script_id, lines=10)
        ts = datetime.now().strftime("%H:%M:%S")
        await context.bot.send_message(
            chat_id=query.message.chat_id,
            text=f"[{ts}] Logs for {script_id} (last 10 lines):\n{text}",
        )
        return
    if data.startswith("edit:"):
        script_id = data.split(":", 1)[1]
        if not script_mgr.get_script(script_id):
            await query.edit_message_text("Script not found.")
            return
        user_id = query.from_user.id
        script_mgr.pending_edit[user_id] = script_id
        script_mgr.pending_new.pop(user_id, None)
        script_mgr.pending_env_value.pop(user_id, None)
        script_mgr.pending_pip.pop(user_id, None)
        await query.edit_message_text(
            f"Send new content for {script_id}. "
            f"If it is running, it will be restarted after update."
        )
        return
    if data.startswith("auto:"):
        script_id = data.split(":", 1)[1]
        script = script_mgr.get_script(script_id)
        if not script:
            await query.edit_message_text("Script not found.")
            return
        current = bool(script.get("autostart"))
        script_mgr.set_autostart(script_id, not current)
        await query.edit_message_text(
            f"Autostart for {script_id}: {'ON' if not current else 'OFF'}",
            reply_markup=script_menu_keyboard(script_id),
        )
        return
    if data.startswith("envmenu:"):
        script_id = data.split(":", 1)[1]
        if not script_mgr.get_script(script_id):
            await query.edit_message_text("Script not found.")
            return
        await query.edit_message_text(
            env_menu_text(script_id),
            reply_markup=env_menu_keyboard(script_id),
        )
        return
    if data.startswith("envset:"):
        _, script_id, key = data.split(":", 2)
        if not script_mgr.get_script(script_id):
            await query.edit_message_text("Script not found.")
            return
        user_id = query.from_user.id
        script_mgr.pending_env_value[user_id] = {
            "script_id": script_id,
            "key": key,
        }
        script_mgr.pending_new.pop(user_id, None)
        script_mgr.pending_edit.pop(user_id, None)
        script_mgr.pending_pip.pop(user_id, None)
        await query.edit_message_text(
            f"Send value for {key} (script {script_id}). "
            f"Send '-' to delete."
        )
        return
    if data.startswith("envdone:"):
        script_id = data.split(":", 1)[1]
        if not script_mgr.get_script(script_id):
            await query.edit_message_text("Script not found.")
            return
        await query.edit_message_text(
            f"Env updated for {script_id}.",
            reply_markup=script_menu_keyboard(script_id),
        )
        return
    if data.startswith("envrun:"):
        script_id = data.split(":", 1)[1]
        if not script_mgr.get_script(script_id):
            await query.edit_message_text("Script not found.")
            return
        msg = await script_mgr.start_script(script_id)
        await query.edit_message_text(
            msg, reply_markup=script_menu_keyboard(script_id)
        )
        return
    if data.startswith("pipprompt:"):
        script_id = data.split(":", 1)[1]
        user_id = query.from_user.id
        script_mgr.pending_pip[user_id] = script_id
        script_mgr.pending_new.pop(user_id, None)
        script_mgr.pending_edit.pop(user_id, None)
        script_mgr.pending_env_value.pop(user_id, None)
        await query.edit_message_text(
            "Send pip install args (e.g. 'python-telegram-bot==21.6 openai'). "
            "Send '-' to cancel."
        )
        return


async def on_startup(app: Application) -> None:
    logger.info("Bot startup: ensure pip packages and autostart scripts")
    await script_mgr.ensure_pip_packages_installed()
    await script_mgr.autostart_all()


def main() -> None:
    application = (
        Application.builder()
        .token(BOT_TOKEN)
        .post_init(on_startup)
        .build()
    )

    application.add_handler(CommandHandler("start", cmd_start))
    application.add_handler(CommandHandler("menu", cmd_menu))
    application.add_handler(CommandHandler("list", cmd_list))
    application.add_handler(CommandHandler("new", cmd_new))
    application.add_handler(CommandHandler("envmenu", cmd_envmenu))
    application.add_handler(CommandHandler("pip", cmd_pip))

    application.add_handler(CommandHandler("run", cmd_run))
    application.add_handler(CommandHandler("stop", cmd_stop))
    application.add_handler(CommandHandler("env", cmd_env))
    application.add_handler(CommandHandler("setenv", cmd_setenv))
    application.add_handler(CommandHandler("delenv", cmd_delenv))
    application.add_handler(CommandHandler("globalenv", cmd_globalenv))
    application.add_handler(CommandHandler("setglobal", cmd_setglobal))
    application.add_handler(CommandHandler("delglobal", cmd_delglobal))
    application.add_handler(CommandHandler("autostart", cmd_autostart))
    application.add_handler(CommandHandler("logs", cmd_logs))

    application.add_handler(CallbackQueryHandler(on_callback))
    application.add_handler(MessageHandler(filters.Document.ALL, handle_document))
    application.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text)
    )

    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
