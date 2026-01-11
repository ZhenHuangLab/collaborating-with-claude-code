"""
Claude Code Bridge Script for Codex Skills.

Runs the `claude` (Claude Code) CLI in non-interactive mode and returns a JSON
envelope suitable for multi-model collaboration.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


DEFAULT_MODEL = "claude-opus-4-5-20251101"
DEFAULT_READONLY_TOOLS = "Read,Glob,Grep,LS"
DEFAULT_FULL_ACCESS_ALLOWED_TOOLS = "*"


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


def _run(cmd: List[str], timeout_s: Optional[float], cwd: Optional[Path]) -> Tuple[int, str, str]:
    env = os.environ.copy()
    _augment_path_env(env)
    cmd = cmd.copy()
    cmd[0] = _resolve_executable(cmd[0], env)

    process = subprocess.Popen(
        cmd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        cwd=str(cwd) if cwd is not None else None,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        process.kill()
        stdout, stderr = process.communicate()
        return 124, stdout, (stderr + "\n[timeout] Claude Code process timed out.").strip()
    return process.returncode, stdout, stderr


def _parse_single_json(stdout: str) -> Dict[str, Any]:
    """
    Claude Code should emit a single JSON object in `--output-format json`.
    Be tolerant of extra non-JSON lines by extracting the first {...} block.
    """
    stdout = stdout.strip()
    if not stdout:
        raise ValueError("Empty stdout")
    try:
        return json.loads(stdout)
    except json.JSONDecodeError:
        start = stdout.find("{")
        end = stdout.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(stdout[start : end + 1])
        raise


def _parse_stream_json(stdout: str) -> List[Dict[str, Any]]:
    messages: List[Dict[str, Any]] = []
    for raw_line in stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            messages.append(json.loads(line))
        except json.JSONDecodeError:
            messages.append({"type": "non_json_line", "text": raw_line})
    return messages


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

    parser = argparse.ArgumentParser(description="Claude Code Bridge")
    parser.add_argument("--PROMPT", required=True, help="Instruction for the task to send to Claude Code.")
    parser.add_argument("--cd", required=True, help="Working directory to run Claude Code in (typically the repo root).")
    parser.add_argument("--SESSION_ID", default="", help="Resume the specified Claude Code session. Defaults to start a new session.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Model to use. Defaults to `{DEFAULT_MODEL}`.")
    access_group = parser.add_mutually_exclusive_group()
    access_group.add_argument(
        "--full-access",
        dest="full_access",
        action="store_true",
        help="Enable non-interactive full access so Claude Code can edit files and run tools without permission prompts (default).",
    )
    access_group.add_argument(
        "--no-full-access",
        dest="full_access",
        action="store_false",
        help="Disable full access (read-only planning/review). Uses read-only tools and avoids edits/commands by default.",
    )
    parser.set_defaults(full_access=True)

    parser.add_argument(
        "--permission-mode",
        default=None,
        help="Claude Code permission mode. Default: `bypassPermissions` when --full-access; `plan` when --no-full-access.",
    )
    parser.add_argument(
        "--tools",
        default=None,
        help='Built-in tool set to expose. Use "default" for all tools, "" to disable all tools, or a comma-separated list (e.g. "Bash,Edit,Read").',
    )
    parser.add_argument(
        "--allowedTools",
        default=None,
        help=(
            f"Comma-separated tools allowed without prompting. Default: `{DEFAULT_FULL_ACCESS_ALLOWED_TOOLS}` when --full-access; "
            f"`{DEFAULT_READONLY_TOOLS}` when --no-full-access."
        ),
    )
    parser.add_argument("--return-all-messages", action="store_true", help="Return the full streamed JSON event list from Claude Code.")
    parser.add_argument("--timeout-s", type=float, default=1200.0, help="Process timeout in seconds. Defaults to 1200 (20 minutes).")
    parser.add_argument("--claude-bin", default="claude", help="Claude Code executable name/path. Defaults to `claude`.")

    args = parser.parse_args()

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

    output_format = "stream-json" if args.return_all_messages else "json"

    cmd: List[str] = [
        args.claude_bin,
        "-p",
        prompt,
        "--output-format",
        output_format,
        "--model",
        args.model,
        "--permission-mode",
        permission_mode,
    ]

    if tools is not None:
        cmd.extend(["--tools", tools])

    if allowed_tools is not None:
        cmd.extend(["--allowedTools", allowed_tools])

    if args.SESSION_ID:
        cmd.extend(["--resume", args.SESSION_ID])

    try:
        rc, stdout, stderr = _run(cmd, timeout_s=args.timeout_s, cwd=cd_path)
    except FileNotFoundError as error:
        print(
            json.dumps(
                {
                    "success": False,
                    "error": f"Failed to execute Claude Code CLI. Is `claude` installed and on PATH?\n\n{error}",
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return

    try:
        if output_format == "json":
            payload = _parse_single_json(stdout)
            session_id = payload.get("session_id")
            result_text = payload.get("result")
            subtype = payload.get("subtype")
            success = bool(payload.get("type") == "result" and subtype == "success" and session_id and result_text)

            if success:
                result: Dict[str, Any] = {
                    "success": True,
                    "SESSION_ID": session_id,
                    "agent_messages": result_text,
                }
            else:
                error_bits = []
                if subtype and subtype != "success":
                    error_bits.append(f"[claude subtype] {subtype}")
                if stderr.strip():
                    error_bits.append(f"[stderr] {stderr.strip()}")
                error_bits.append(f"[stdout] {stdout.strip()}")
                result = {"success": False, "error": "\n".join(error_bits).strip()}

            if args.return_all_messages:
                result["all_messages"] = [payload]

        else:
            messages = _parse_stream_json(stdout)
            session_id, result_text, error_text = _extract_result(messages)
            success = bool(rc == 0 and session_id and result_text and not error_text)

            if success:
                result = {"success": True, "SESSION_ID": session_id, "agent_messages": result_text}
            else:
                error_bits = []
                if error_text:
                    error_bits.append(f"[claude result] {error_text}")
                if stderr.strip():
                    error_bits.append(f"[stderr] {stderr.strip()}")
                if stdout.strip():
                    error_bits.append(f"[stdout] {stdout.strip()}")
                if rc != 0:
                    error_bits.append(f"[exit_code] {rc}")
                result = {"success": False, "error": "\n".join(error_bits).strip()}

            result["all_messages"] = messages

    except Exception as error:  # noqa: BLE001 - keep bridge resilient
        result = {
            "success": False,
            "error": f"Bridge failed to parse Claude Code output: {error}\n\n[stderr]\n{stderr.strip()}\n\n[stdout]\n{stdout.strip()}".strip(),
        }

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
