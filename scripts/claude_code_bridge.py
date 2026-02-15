"""
Claude Code Bridge Script for Codex Skills.

Runs the `claude` (Claude Code) CLI in non-interactive mode and returns a JSON
envelope suitable for multi-model collaboration.
"""

from __future__ import annotations

import argparse
import codecs
import json
import os
import signal
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple


DEFAULT_MODEL = "claude-opus-4-6"
DEFAULT_READONLY_TOOLS = "Read,Glob,Grep,LS,Write"
DEFAULT_FULL_ACCESS_ALLOWED_TOOLS = "*"
DEFAULT_EXTENDED_THINKING_ENABLED = True
CLAUDE_SETTINGS_ALWAYS_THINKING_KEY = "alwaysThinkingEnabled"
DEFAULT_STEP_MODE = "auto"  # on|auto|off
DEFAULT_STEP_MAX_STEPS = 64
DEFAULT_STEP_CONTINUE_PROMPT = "Continue from where you left off. Do not restate previous content."
CLAUDE_STEP_MAX_TURNS = 1
CLAUDE_CONNECTIVITY_ERROR_MESSAGE = "Failed to connect to Claude. Is Claude installed and on PATH?"
CLAUDE_VERSION_CHECK_TIMEOUT_S = 5.0
PIPE_DRAIN_GRACE_S = 2.0
TERMINATE_GRACE_S = 5.0
IO_READ_CHUNK_SIZE = 8192


def _kill_process_tree(process: subprocess.Popen[Any], *, force: bool) -> None:
    """
    Best-effort termination of the subprocess (and any children holding stdio open).

    Why this exists:
    - Claude Code (a Node CLI) may spawn child processes that inherit stdout/stderr.
      If those children keep the pipe open, `communicate()` can block forever even
      after the parent exits.
    - On POSIX we run the CLI in a new session so we can kill the whole process group.
    - On Windows we fall back to `taskkill /T` when available.
    """
    pid = getattr(process, "pid", None)
    if not pid:
        return

    if os.name != "nt":
        sig = signal.SIGKILL if force else signal.SIGTERM
        try:
            os.killpg(pid, sig)
            return
        except ProcessLookupError:
            return
        except Exception:
            pass

        try:
            process.kill() if force else process.terminate()
        except Exception:
            pass
        return

    taskkill = shutil.which("taskkill")
    if taskkill:
        cmd = [taskkill]
        if force:
            cmd.append("/F")
        cmd.extend(["/PID", str(pid), "/T"])
        try:
            subprocess.run(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            return
        except Exception:
            pass

    try:
        process.kill()
    except Exception:
        pass


class _OutputBuffer:
    """Thread-safe output accumulator with a small tail cache for quick heuristics."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._chunks: List[str] = []
        self._tail: str = ""
        self._last_change_ts = time.monotonic()

    def append(self, chunk: str) -> None:
        if not chunk:
            return
        now = time.monotonic()
        with self._lock:
            self._chunks.append(chunk)
            self._tail = (self._tail + chunk)[-4096:]
            self._last_change_ts = now

    def snapshot(self) -> str:
        with self._lock:
            return "".join(self._chunks)

    def tail(self) -> str:
        with self._lock:
            return self._tail

    def last_change_ts(self) -> float:
        with self._lock:
            return float(self._last_change_ts)


def _parse_settings_arg(settings_arg: str) -> Dict[str, Any]:
    """
    Parse a Claude Code `--settings` argument as either:
      - a JSON object string, or
      - a path to a JSON file.

    Returns a dict to be merged into the session settings.
    """
    raw = settings_arg.strip()
    if not raw:
        return {}

    if raw.startswith("{"):
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            raise ValueError("`--claude-settings` JSON must be an object.")
        return parsed

    path = Path(raw).expanduser()
    if path.is_file():
        parsed = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(parsed, dict):
            raise ValueError("`--claude-settings` file must contain a JSON object.")
        return parsed

    # Last resort: try parsing as JSON (useful if a JSON string doesn't start with "{")
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("`--claude-settings` must be a JSON object or a JSON file path.")
    return parsed


def _strip_thinking_blocks(text: str) -> str:
    """
    Best-effort removal of "thinking" blocks from Claude Code output.

    The CLI typically returns only the final answer, but when extended thinking
    is enabled some environments may include explicit thinking delimiters.
    This keeps the bridge output focused on actionable content by default.
    """
    import re

    if not text:
        return text

    # Common formats seen in tooling integrations:
    #   <thinking> ... </thinking>
    #   <analysis> ... </analysis>
    # Remove those blocks if present; keep surrounding content.
    stripped = re.sub(r"(?is)<thinking>.*?</thinking>\s*", "", text)
    stripped = re.sub(r"(?is)<analysis>.*?</analysis>\s*", "", stripped)
    return stripped.strip()


def _is_thinking_schema_400(text: str) -> bool:
    """
    Detect the common 400 error shape when a proxy/router enables Anthropic
    'thinking' but the upstream message history doesn't preserve thinking blocks
    around tool_use/tool_result.

    This typically looks like:
      - Expected `thinking` or `redacted_thinking`, but found `tool_use`
      - Expected `thinking` or `redacted_thinking`, but found `text`
    """
    if not text:
        return False
    if "API Error: 400" not in text:
        return False
    return "Expected `thinking` or `redacted_thinking`" in text


def _extract_session_id(stdout: str, stderr: str) -> Optional[str]:
    """
    Best-effort session_id extraction from Claude Code stdout/stderr.

    Some failure cases (including certain 400 proxy errors) may emit a session ID
    to stderr, or produce non-JSON output even when `--output-format json` is set.
    This lets the bridge continue the same session instead of silently starting a
    new one.
    """
    import re

    combined = "\n".join([stdout or "", stderr or ""]).strip()
    if not combined:
        return None

    uuid = r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
    patterns = [
        rf'"session_id"\s*:\s*"(?P<id>{uuid})"',
        rf"\bSession ID:\s*(?P<id>{uuid})\b",
        rf"\bsession[_ -]?id\s*[:=]\s*(?P<id>{uuid})\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, combined)
        if match:
            return match.group("id")
    return None


def _build_claude_cmd(
    *,
    claude_bin: str,
    prompt: str,
    output_format: str,
    model: str,
    permission_mode: str,
    tools: Optional[str],
    allowed_tools: Optional[str],
    session_id: str,
    continue_session: bool,
    claude_settings: Dict[str, Any],
    max_turns: Optional[int],
    verbose: bool,
) -> List[str]:
    cmd: List[str] = [
        claude_bin,
        "-p",
        prompt,
        "--output-format",
        output_format,
        "--model",
        model,
        "--permission-mode",
        permission_mode,
    ]

    if verbose:
        cmd.append("--verbose")

    if tools is not None:
        cmd.extend(["--tools", tools])

    if allowed_tools is not None:
        cmd.extend(["--allowedTools", allowed_tools])

    if claude_settings:
        cmd.extend(["--settings", json.dumps(claude_settings, ensure_ascii=False)])

    if max_turns is not None:
        cmd.extend(["--max-turns", str(max_turns)])

    if continue_session and session_id:
        raise ValueError("Cannot use both --continue and --resume.")

    if continue_session:
        cmd.append("--continue")
    elif session_id:
        cmd.extend(["--resume", session_id])

    return cmd


def _get_windows_npm_paths() -> List[Path]:
    """Return candidate directories for npm global installs on Windows."""
    if os.name != "nt":
        return []
    env = os.environ
    paths: List[Path] = []
    if prefix := env.get("NPM_CONFIG_PREFIX") or env.get("npm_config_prefix"):
        paths.append(Path(prefix))
    if appdata := env.get("APPDATA"):
        paths.append(Path(appdata) / "npm")
    if localappdata := env.get("LOCALAPPDATA"):
        paths.append(Path(localappdata) / "npm")
    if programfiles := env.get("ProgramFiles"):
        paths.append(Path(programfiles) / "nodejs")
    return paths


def _augment_path_env(env: Dict[str, str]) -> None:
    """Prepend npm global directories to PATH if missing (Windows only)."""
    if os.name != "nt":
        return
    path_key = next((k for k in env if k.upper() == "PATH"), "PATH")
    path_entries = [p for p in env.get(path_key, "").split(os.pathsep) if p]
    lower_set = {p.lower() for p in path_entries}
    for candidate in _get_windows_npm_paths():
        if candidate.is_dir() and str(candidate).lower() not in lower_set:
            path_entries.insert(0, str(candidate))
            lower_set.add(str(candidate).lower())
    env[path_key] = os.pathsep.join(path_entries)


def _resolve_executable(name: str, env: Dict[str, str]) -> str:
    """Resolve executable path, checking npm directories for .cmd/.bat on Windows."""
    if os.path.isabs(name) or os.sep in name or (os.altsep and os.altsep in name):
        return name
    path_key = next((k for k in env if k.upper() == "PATH"), "PATH")
    path_val = env.get(path_key)
    if resolved := shutil.which(name, path=path_val):
        return resolved
    if os.name == "nt":
        for base in _get_windows_npm_paths():
            for ext in (".cmd", ".bat", ".exe", ".com"):
                candidate = base / f"{name}{ext}"
                if candidate.is_file():
                    return str(candidate)
    return name


def _detect_claude_installation(*, claude_bin: str) -> str:
    """
    Resolve Claude Code binary on PATH (or from an explicit path) and verify it runs.

    Intended for `--help` gating: if Claude isn't installed/working, we fail early
    rather than printing a help message for a skill that can't run.
    """
    env = os.environ.copy()
    _augment_path_env(env)
    resolved = _resolve_executable(claude_bin, env)

    if os.path.isabs(resolved) or os.sep in resolved or (os.altsep and os.altsep in resolved):
        resolved_path = Path(resolved).expanduser()
        if not resolved_path.is_file():
            raise FileNotFoundError(str(resolved_path))
        resolved_str = str(resolved_path)
    else:
        # `_resolve_executable` returns the original name when PATH lookup fails.
        raise FileNotFoundError(claude_bin)

    try:
        proc = subprocess.run(
            [resolved_str, "--version"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            timeout=CLAUDE_VERSION_CHECK_TIMEOUT_S,
        )
    except (FileNotFoundError, OSError, subprocess.SubprocessError) as error:
        raise RuntimeError("Failed to execute `claude --version`.") from error

    if proc.returncode != 0:
        raise RuntimeError("`claude --version` returned a non-zero exit code.")

    return resolved_str


class _HelpWithClaudeCheckAction(argparse.Action):
    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: object,
        option_string: Optional[str] = None,
    ) -> None:
        claude_bin = getattr(namespace, "claude_bin", "claude")
        try:
            claude_path = _detect_claude_installation(claude_bin=claude_bin)
        except Exception:
            parser.exit(status=1, message=f"{CLAUDE_CONNECTIVITY_ERROR_MESSAGE}\n")

        print(f"Claude installed: {claude_path}")
        parser.print_help(sys.stdout)
        parser.exit(status=0)

def _windows_escape(prompt: str) -> str:
    """Windows style string escaping for newlines and special chars in prompt text."""
    return prompt.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")


def _configure_windows_stdio() -> None:
    """Configure stdout/stderr to use UTF-8 encoding on Windows."""
    if os.name != "nt":
        return
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8")
            except (ValueError, OSError):
                pass


def _drain_stream_json_lines(
    buffer: str,
    decoded: str,
    *,
    stream_json_event_callback: Optional[Callable[[Dict[str, Any]], None]],
    result_seen: threading.Event,
    final: bool = False,
) -> str:
    """
    Incrementally parse Claude Code `--output-format stream-json` stdout.

    Returns the remaining (incomplete) line buffer after parsing complete lines.
    """
    buffer = (buffer or "") + (decoded or "")

    def handle_parsed(obj: Any) -> None:
        events: List[Dict[str, Any]] = []
        if isinstance(obj, dict):
            events = [obj]
        elif isinstance(obj, list):
            events = [item for item in obj if isinstance(item, dict)]

        for event in events:
            if stream_json_event_callback is not None:
                try:
                    stream_json_event_callback(event)
                except Exception:
                    # Never allow streaming UI to break the bridge.
                    pass
            if event.get("type") == "result":
                result_seen.set()

    while True:
        if "\n" not in buffer:
            break
        line, buffer = buffer.split("\n", 1)
        raw = line.strip()
        if not raw:
            continue
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            continue
        handle_parsed(parsed)

    if final:
        raw = buffer.strip()
        if raw:
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                return ""
            handle_parsed(parsed)
        return ""

    return buffer


def _run(
    cmd: List[str],
    timeout_s: Optional[float],
    cwd: Optional[Path],
    *,
    stream_json_event_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Tuple[int, str, str]:
    env = os.environ.copy()
    _augment_path_env(env)
    cmd = cmd.copy()
    cmd[0] = _resolve_executable(cmd[0], env)

    output_format: Optional[str] = None
    try:
        fmt_index = cmd.index("--output-format")
        if fmt_index + 1 < len(cmd):
            output_format = cmd[fmt_index + 1]
    except ValueError:
        output_format = None

    popen_kwargs: Dict[str, Any] = {}
    # On POSIX, isolate Claude in its own session so we can kill the whole process
    # group if it (or a child) keeps stdout/stderr open and blocks draining.
    if os.name != "nt":
        popen_kwargs["start_new_session"] = True
    elif hasattr(subprocess, "CREATE_NEW_PROCESS_GROUP"):
        popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP

    process = subprocess.Popen(
        cmd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        cwd=str(cwd) if cwd is not None else None,
        **popen_kwargs,
    )

    stdout_buf = _OutputBuffer()
    stderr_buf = _OutputBuffer()
    stream_json_result_seen = threading.Event()

    def drain(stream: Any, sink: _OutputBuffer, *, is_stdout: bool) -> None:
        if stream is None:
            return
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        line_buf = ""
        try:
            while True:
                chunk = stream.read(IO_READ_CHUNK_SIZE)
                if not chunk:
                    break
                if isinstance(chunk, str):
                    decoded = chunk
                else:
                    decoded = decoder.decode(chunk)
                sink.append(decoded)

                if is_stdout and output_format == "stream-json":
                    line_buf = _drain_stream_json_lines(
                        line_buf,
                        decoded,
                        stream_json_event_callback=stream_json_event_callback,
                        result_seen=stream_json_result_seen,
                    )
        except Exception as error:  # noqa: BLE001 - keep bridge resilient
            sink.append(f"\n[bridge] stream read error: {error}\n")
        finally:
            try:
                final_decoded = decoder.decode(b"", final=True)
                if final_decoded:
                    sink.append(final_decoded)
                    if is_stdout and output_format == "stream-json":
                        line_buf = _drain_stream_json_lines(
                            line_buf,
                            final_decoded,
                            stream_json_event_callback=stream_json_event_callback,
                            result_seen=stream_json_result_seen,
                        )
            except Exception:
                pass
            if is_stdout and output_format == "stream-json":
                _drain_stream_json_lines(
                    line_buf,
                    "",
                    stream_json_event_callback=stream_json_event_callback,
                    result_seen=stream_json_result_seen,
                    final=True,
                )
            try:
                stream.close()
            except Exception:
                pass

    t_out = threading.Thread(target=drain, args=(process.stdout, stdout_buf), kwargs={"is_stdout": True}, daemon=True)
    t_err = threading.Thread(target=drain, args=(process.stderr, stderr_buf), kwargs={"is_stdout": False}, daemon=True)
    t_out.start()
    t_err.start()

    deadline = (time.monotonic() + float(timeout_s)) if timeout_s is not None else None
    timed_out = False
    completed_early = False
    completed_stdout: Optional[str] = None
    completed_stderr: Optional[str] = None

    try:
        while True:
            now = time.monotonic()
            if deadline is not None and now >= deadline:
                timed_out = True
                break

            if output_format == "stream-json" and stream_json_result_seen.is_set():
                completed_early = True
                completed_stdout = stdout_buf.snapshot()
                completed_stderr = stderr_buf.snapshot()
                break

            process_exited = process.poll() is not None
            if process_exited and (not t_out.is_alive()) and (not t_err.is_alive()):
                break

            time.sleep(0.2)
    finally:
        if timed_out or completed_early:
            _kill_process_tree(process, force=False)
            try:
                process.wait(timeout=TERMINATE_GRACE_S)
            except subprocess.TimeoutExpired:
                _kill_process_tree(process, force=True)
                try:
                    process.wait(timeout=TERMINATE_GRACE_S)
                except subprocess.TimeoutExpired:
                    pass

        # If the parent exited but a child kept stdio open, drain threads can
        # stay blocked. Give them a short grace period, then kill the tree.
        t_out.join(timeout=PIPE_DRAIN_GRACE_S)
        t_err.join(timeout=PIPE_DRAIN_GRACE_S)
        if t_out.is_alive() or t_err.is_alive():
            _kill_process_tree(process, force=True)
            try:
                if process.stdout is not None:
                    process.stdout.close()
                if process.stderr is not None:
                    process.stderr.close()
            except Exception:
                pass
            t_out.join(timeout=PIPE_DRAIN_GRACE_S)
            t_err.join(timeout=PIPE_DRAIN_GRACE_S)

    stdout = completed_stdout if completed_early and completed_stdout is not None else stdout_buf.snapshot()
    stderr = completed_stderr if completed_early and completed_stderr is not None else stderr_buf.snapshot()

    if timed_out:
        stdout_tail = stdout_buf.tail().strip()
        stderr = (
            stderr
            + "\n[timeout] Claude Code process timed out."
            + (f" stdout_chars={len(stdout)} stderr_chars={len(stderr)}" if stdout or stderr else "")
            + (f" stdout_tail={stdout_tail[-200:]}" if stdout_tail else "")
        ).strip()
        return 124, stdout, stderr

    if completed_early:
        return 0, stdout, stderr

    return int(process.returncode or 0), stdout, stderr


def _parse_stream_json(stdout: str) -> List[Dict[str, Any]]:
    messages: List[Dict[str, Any]] = []
    for raw_line in stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            parsed = json.loads(line)
            if isinstance(parsed, dict):
                messages.append(parsed)
            elif isinstance(parsed, list):
                messages.extend([item for item in parsed if isinstance(item, dict)])
            else:
                messages.append({"type": "non_json_line", "text": raw_line})
        except json.JSONDecodeError:
            messages.append({"type": "non_json_line", "text": raw_line})
    return messages


def _extract_assistant_texts(messages: List[Dict[str, Any]]) -> List[str]:
    texts: List[str] = []
    for event in messages:
        if not isinstance(event, dict) or event.get("type") != "assistant":
            continue
        message = event.get("message")
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        content = message.get("content")
        if isinstance(content, str):
            if content.strip():
                texts.append(content)
            continue
        if not isinstance(content, list):
            continue
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text") or "")
        block = "".join(parts)
        if block.strip():
            texts.append(block)
    return texts


def _extract_result(messages: List[Dict[str, Any]]) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Return (session_id, result_text, error_text) from a stream-json message list.
    Prefer the last {"type":"result"} entry.
    """
    session_id: Optional[str] = None
    result_text: Optional[str] = None
    error_text: Optional[str] = None

    for msg in messages:
        if isinstance(msg, dict) and msg.get("session_id"):
            session_id = msg.get("session_id")

    for msg in reversed(messages):
        if not isinstance(msg, dict):
            continue
        if msg.get("type") == "result":
            session_id = msg.get("session_id") or session_id
            subtype = msg.get("subtype")
            if subtype and subtype != "success":
                error_text = msg.get("result") or msg.get("error") or str(msg)
            else:
                result_text = msg.get("result")
            break

    return session_id, result_text, error_text


def main() -> None:
    _configure_windows_stdio()

    parser = argparse.ArgumentParser(
        add_help=False,
        description="Claude Code Bridge: run Claude Code CLI non-interactively and return JSON.",
        epilog=(
            "Examples:\n"
            '  %(prog)s --cd /repo --PROMPT "Review auth flow; propose fixes."\n'
            '  %(prog)s --no-full-access --cd /repo --PROMPT "List issues (read-only)."\n'
            '  %(prog)s --cd /repo --SESSION_ID abc123 --PROMPT "Continue."\n'
            "\n"
            "Claude Code often takes 1-2+ minutes. Prefer waiting for completion; avoid rapid retries.\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "-h",
        "--help",
        action=_HelpWithClaudeCheckAction,
        nargs=0,
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )

    req = parser.add_argument_group("required")
    req.add_argument("--PROMPT", required=True, help="Instruction to send to Claude Code.")
    req.add_argument("--cd", required=True, help="Working directory (typically the repo root).")

    session = parser.add_argument_group("session")
    session.add_argument("--SESSION_ID", default="", help="Resume session (default: start a new session).")
    session.add_argument("--model", default=DEFAULT_MODEL, help=f"Model (default: `{DEFAULT_MODEL}`).")

    access = parser.add_argument_group("access")
    access_group = access.add_mutually_exclusive_group()
    access_group.add_argument(
        "--full-access",
        dest="full_access",
        action="store_true",
        help="(default) Full access: can edit files and run tools without prompts.",
    )
    access_group.add_argument(
        "--no-full-access",
        dest="full_access",
        action="store_false",
        help="Read-only planning/review mode (no edits/commands by default).",
    )
    parser.set_defaults(full_access=True)

    thinking = parser.add_argument_group("thinking")
    thinking_group = thinking.add_mutually_exclusive_group()
    thinking_group.add_argument(
        "--extended-thinking",
        dest="extended_thinking",
        action="store_true",
        help="(default) Enable Claude Code extended thinking for this run.",
    )
    thinking_group.add_argument(
        "--no-extended-thinking",
        dest="extended_thinking",
        action="store_false",
        help="Disable Claude Code extended thinking for this run.",
    )
    parser.set_defaults(extended_thinking=DEFAULT_EXTENDED_THINKING_ENABLED)

    advanced = parser.add_argument_group("advanced")
    advanced.add_argument(
        "--agent-messages-mode",
        choices=["result", "concat"],
        default="result",
        help="How to build `agent_messages` in the final JSON. Default: result. NO NEED TO CHANGE THIS unless user requests.",
    )
    advanced.add_argument(
        "--quiet",
        dest="quiet",
        action="store_true",
        help="Disable all stderr output (default: stream Claude assistant text to stderr). NO NEED TO CHANGE THIS unless user requests.",
    )
    advanced.add_argument(
        "--step-mode",
        choices=["on", "auto", "off"],
        default=DEFAULT_STEP_MODE,
        help=(
            "Work around some Anthropic-compatible proxies/routers that enforce strict thinking/tool message schemas. "
            "When enabled, the bridge runs claude in small agentic steps (`--max-turns 1`) and resumes until completion. "
            f"Default: {DEFAULT_STEP_MODE}. NO NEED TO CHANGE THIS unless user requests."
        ),
    )
    advanced.add_argument(
        "--step-max-steps",
        type=int,
        default=DEFAULT_STEP_MAX_STEPS,
        help=f"Maximum number of resume iterations in step mode (default: {DEFAULT_STEP_MAX_STEPS}).",
    )
    advanced.add_argument(
        "--step-continue-prompt",
        default=DEFAULT_STEP_CONTINUE_PROMPT,
        help=f'Prompt to send on each resume in step mode (default: "{DEFAULT_STEP_CONTINUE_PROMPT}").',
    )
    advanced.add_argument(
        "--permission-mode",
        default=None,
        help="Claude Code permission mode. Default: `bypassPermissions` when --full-access; `plan` when --no-full-access.",
    )
    advanced.add_argument(
        "--tools",
        default=None,
        help='Built-in tool set to expose. Use "default" for all tools, "" to disable all tools, or a comma-separated list (e.g. "Bash,Edit,Read").',
    )
    advanced.add_argument(
        "--allowedTools",
        default=None,
        help=(
            f"Comma-separated tools allowed without prompting. Default: `{DEFAULT_FULL_ACCESS_ALLOWED_TOOLS}` when --full-access; "
            f"`{DEFAULT_READONLY_TOOLS}` when --no-full-access."
        ),
    )
    advanced.add_argument(
        "--claude-settings",
        default=None,
        help=(
            "Additional Claude Code settings to apply for this run. "
            "Provide either a JSON object string (e.g. '{\"model\":\"opus\"}') or a path to a JSON file. "
            f"Note: this bridge always sets `{CLAUDE_SETTINGS_ALWAYS_THINKING_KEY}` based on --extended-thinking/--no-extended-thinking."
        ),
    )
    advanced.add_argument(
        "--keep-thinking-blocks",
        action="store_true",
        help="Do not strip <thinking>/<analysis> blocks from the returned agent_messages (debugging).",
    )
    advanced.add_argument("--return-all-messages", action="store_true", help="Return the full streamed JSON event list (debugging).")
    advanced.add_argument("--timeout-s", type=float, default=1800.0, help="Timeout in seconds (default: 30 minutes).")
    advanced.add_argument("--claude-bin", default="claude", help="Claude Code executable name/path (default: `claude`).")

    args = parser.parse_args()

    quiet = bool(args.quiet)
    stderr_lock = threading.Lock()
    emitted_session_id: Optional[str] = None

    def emit_session_id(session_id: Optional[str]) -> None:
        nonlocal emitted_session_id
        if quiet or not session_id:
            return
        with stderr_lock:
            if session_id == emitted_session_id:
                return
            print(f"session_id={session_id}", file=sys.stderr, flush=True)
            emitted_session_id = session_id

    def emit_assistant_text(text: str) -> None:
        if quiet:
            return
        normalized = (text or "").rstrip("\n")
        if not normalized.strip():
            return
        with stderr_lock:
            print(normalized, file=sys.stderr, flush=True)
            print("", file=sys.stderr, flush=True)

    cd_path = Path(args.cd).expanduser()
    if not cd_path.is_dir():
        print(
            json.dumps(
                {"success": False, "error": f"`--cd` must be an existing directory. Got: {args.cd}"},
                indent=2,
                ensure_ascii=False,
            )
        )
        return

    prompt = args.PROMPT
    if os.name == "nt":
        prompt = _windows_escape(prompt)

    permission_mode: str = args.permission_mode or ("bypassPermissions" if args.full_access else "plan")
    tools: str = args.tools if args.tools is not None else ("default" if args.full_access else DEFAULT_READONLY_TOOLS)

    allowed_tools_provided = args.allowedTools is not None
    allowed_tools: Optional[str] = args.allowedTools
    if not allowed_tools_provided:
        allowed_tools = DEFAULT_FULL_ACCESS_ALLOWED_TOOLS if args.full_access else DEFAULT_READONLY_TOOLS

    try:
        claude_settings: Dict[str, Any] = _parse_settings_arg(args.claude_settings) if args.claude_settings else {}
    except Exception as error:  # noqa: BLE001 - keep bridge resilient
        print(
            json.dumps(
                {"success": False, "error": f"Invalid --claude-settings: {error}"},
                indent=2,
                ensure_ascii=False,
            )
        )
        return

    # Ensure controllable defaults: default to extended thinking ON, but allow opt-out.
    claude_settings[CLAUDE_SETTINGS_ALWAYS_THINKING_KEY] = bool(args.extended_thinking)

    output_format = "stream-json"
    verbose = bool(args.return_all_messages)

    if args.SESSION_ID:
        emit_session_id(args.SESSION_ID)

    def handle_stream_event(event: Dict[str, Any]) -> None:
        emit_session_id(event.get("session_id"))
        if event.get("type") != "assistant":
            return
        message = event.get("message")
        if not isinstance(message, dict) or message.get("role") != "assistant":
            return
        content = message.get("content")
        if isinstance(content, str):
            emit_assistant_text(content)
            return
        if not isinstance(content, list):
            return
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text") or "")
        emit_assistant_text("".join(parts))

    def run_one(
        *, run_prompt: str, resume_session_id: str, continue_session: bool, max_turns: Optional[int]
    ) -> Tuple[int, str, str]:
        cmd = _build_claude_cmd(
            claude_bin=args.claude_bin,
            prompt=run_prompt,
            output_format=output_format,
            model=args.model,
            permission_mode=permission_mode,
            tools=tools,
            allowed_tools=allowed_tools,
            session_id=resume_session_id,
            continue_session=continue_session,
            claude_settings=claude_settings,
            max_turns=max_turns,
            verbose=verbose,
        )
        return _run(
            cmd,
            timeout_s=args.timeout_s,
            cwd=cd_path,
            stream_json_event_callback=handle_stream_event if not quiet else None,
        )

    def print_exec_error(error: Exception) -> None:
        print(
            json.dumps(
                {"success": False, "error": f"Failed to execute Claude Code CLI. Is `claude` installed and on PATH?\n\n{error}"},
                indent=2,
                ensure_ascii=False,
            )
        )

    # Step-mode runner: some proxies enable thinking by default and reject multi-turn tool loops.
    # This loop forces Claude Code to stop after each agentic step and resumes until completion.
    def run_with_optional_stepping() -> Tuple[int, str, str]:
        step_mode = str(args.step_mode or DEFAULT_STEP_MODE)
        if step_mode not in ("on", "auto", "off"):
            step_mode = DEFAULT_STEP_MODE

        def summarize_run(stdout_text: str, stderr_text: str) -> Dict[str, Any]:
            """
            Return a small summary dict with keys:
              - session_id
              - subtype
              - is_error
              - result_text
              - parse_error (optional)
            Works for `--output-format stream-json`.
            """
            messages = _parse_stream_json(stdout_text)
            # Find the last result event.
            last_result: Optional[Dict[str, Any]] = None
            for msg in reversed(messages):
                if isinstance(msg, dict) and msg.get("type") == "result":
                    last_result = msg
                    break

            session_id, result_text, error_text = _extract_result(messages)
            return {
                "session_id": session_id,
                "subtype": (last_result or {}).get("subtype"),
                "is_error": bool((last_result or {}).get("is_error")) or bool(error_text),
                "result_text": (result_text or "") if not error_text else (error_text or ""),
                "messages": messages,
            }

        # We treat --step-mode=on as always stepping; auto will attempt one normal run first.
        session_id = args.SESSION_ID
        continue_session = False
        current_prompt = prompt

        def attempt(*, use_step: bool, resume_id: str, use_continue: bool, run_prompt: str) -> Tuple[int, str, str]:
            return run_one(
                run_prompt=run_prompt,
                resume_session_id=resume_id,
                continue_session=use_continue,
                max_turns=CLAUDE_STEP_MAX_TURNS if use_step else None,
            )

        if step_mode in ("off", "auto"):
            try:
                rc0, stdout0, stderr0 = attempt(
                    use_step=False, resume_id=session_id, use_continue=continue_session, run_prompt=current_prompt
                )
            except FileNotFoundError as error:
                print_exec_error(error)
                raise SystemExit(0)

            if step_mode == "off":
                return rc0, stdout0, stderr0

            # Auto: If it did not hit the specific thinking schema 400, just return it.
            # Otherwise, fall through to stepping mode.
            combined0 = "\n".join([stdout0.strip(), stderr0.strip()]).strip()
            if not _is_thinking_schema_400(combined0):
                return rc0, stdout0, stderr0

            # If we can parse a session_id from the failed run, reuse it to avoid redoing tool work.
            try:
                summary0 = summarize_run(stdout0, stderr0)
                session_id = (summary0.get("session_id") or session_id) if isinstance(summary0, dict) else session_id
            except Exception:
                # If parsing fails, keep any user-provided session_id and try a loose extraction.
                extracted = _extract_session_id(stdout0, stderr0)
                if extracted and not session_id:
                    session_id = extracted
            emit_session_id(session_id)

            # If we still don't have a session id (and the user didn't provide one), prefer continuing the
            # most recent conversation in this directory instead of starting a brand new session.
            if not session_id and not args.SESSION_ID:
                continue_session = True

            current_prompt = args.step_continue_prompt

        # Step mode ON (or AUTO fallback):
        # Keep resuming until Claude produces a final successful result.
        for step_index in range(int(args.step_max_steps)):
            try:
                rc, stdout, stderr = attempt(
                    use_step=True, resume_id=session_id, use_continue=continue_session, run_prompt=current_prompt
                )
            except FileNotFoundError as error:
                print_exec_error(error)
                raise SystemExit(0)

            # Update resume id if Claude Code returns one.
            try:
                summary = summarize_run(stdout, stderr)
                session_id = summary.get("session_id") or session_id
                if session_id:
                    continue_session = False
                emit_session_id(session_id)
                subtype = summary.get("subtype")
                is_error = bool(summary.get("is_error"))
                result_text = summary.get("result_text") or ""
            except Exception:
                # If we can't parse, bail and surface raw output.
                return rc, stdout, stderr

            # Completed normally.
            if subtype == "success" and not is_error and result_text.strip():
                return rc, stdout, stderr

            # Claude Code indicates it hit max turns and needs another resume.
            if subtype == "error_max_turns" and session_id:
                current_prompt = args.step_continue_prompt
                continue

            # If we hit the thinking schema error even in step mode, surface it (likely an upstream incompatibility).
            combined = "\n".join([result_text.strip(), stderr.strip(), stdout.strip()]).strip()
            if _is_thinking_schema_400(combined):
                return rc, stdout, stderr

            # Any other terminal condition: return the raw result.
            return rc, stdout, stderr

        # Exceeded max steps.
        return 1, "", f"Step mode exceeded --step-max-steps={args.step_max_steps} without reaching a final result."

    rc, stdout, stderr = run_with_optional_stepping()

    try:
        messages = _parse_stream_json(stdout)
        session_id, result_text, error_text = _extract_result(messages)
        emit_session_id(session_id)

        assistant_texts = _extract_assistant_texts(messages)
        if args.agent_messages_mode == "concat":
            agent_messages = "\n\n".join(assistant_texts).strip()
            if not agent_messages and result_text:
                agent_messages = result_text
        else:
            agent_messages = (result_text or "").strip()

        last_result: Optional[Dict[str, Any]] = next(
            (msg for msg in reversed(messages) if isinstance(msg, dict) and msg.get("type") == "result"),
            None,
        )
        subtype = (last_result or {}).get("subtype")
        is_error = bool((last_result or {}).get("is_error"))
        success = bool(
            rc == 0
            and subtype == "success"
            and not is_error
            and session_id
            and not error_text
            and bool(agent_messages)
        )

        if success:
            if not args.keep_thinking_blocks:
                agent_messages = _strip_thinking_blocks(agent_messages)
            result = {"success": True, "SESSION_ID": session_id, "agent_messages": agent_messages}
        else:
            error_bits = []
            if subtype and subtype != "success":
                error_bits.append(f"[claude subtype] {subtype}")
            if is_error:
                error_bits.append("[claude is_error] true")
            if error_text:
                error_bits.append(f"[claude result] {error_text}")
            if stderr.strip():
                error_bits.append(f"[stderr] {stderr.strip()}")
            if stdout.strip():
                error_bits.append(f"[stdout] {stdout.strip()}")
            if rc != 0:
                error_bits.append(f"[exit_code] {rc}")
            result = {"success": False, "error": "\n".join(error_bits).strip()}
            if session_id:
                result["SESSION_ID"] = session_id

        if args.return_all_messages:
            result["all_messages"] = messages

    except Exception as error:  # noqa: BLE001 - keep bridge resilient
        extracted_session_id = _extract_session_id(stdout, stderr)
        emit_session_id(extracted_session_id)
        result = {
            "success": False,
            "error": f"Bridge failed to parse Claude Code output: {error}\n\n[stderr]\n{stderr.strip()}\n\n[stdout]\n{stdout.strip()}".strip(),
        }
        if extracted_session_id:
            result["SESSION_ID"] = extracted_session_id

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
