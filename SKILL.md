---
name: collaborating-with-claude-code
description: "Delegates code review, debugging, and alternative implementations to Anthropic Claude Code (default model: claude-opus-4-5-20251101) via a JSON bridge script. Supports multi-turn sessions via SESSION_ID."
metadata:
  short-description: Use Claude Code as a collaborator
---

# Collaborating with Claude Code

Use this skill when you want a second model to sanity-check a solution, spot edge cases, propose tests, or suggest an alternative implementation approach.

This skill provides a small JSON bridge script that runs `claude` (Claude Code) in non-interactive “print” mode and returns structured output.

## Requirements

- Claude Code CLI installed (`claude --version`).
- Claude Code authenticated (for example via `ANTHROPIC_API_KEY` in the environment, or whatever auth flow your Claude Code installation uses).
- This skill defaults to **full access** for Claude Code. Only use it in repos/directories you trust.

## Quick start

```bash
python scripts/claude_code_bridge.py --cd "/path/to/repo" --PROMPT "Review the auth flow for bypasses; propose fixes as a unified diff."
```

For **read-only** review (avoid edits/commands), add `--no-full-access`:

```bash
python scripts/claude_code_bridge.py --no-full-access --cd "/path/to/repo" --PROMPT "Review the auth flow and list issues (no code changes)."
```

## Multi-turn sessions

Always capture the returned `SESSION_ID` and pass it back on follow-ups:

```bash
# Start a new session
python scripts/claude_code_bridge.py --cd "/repo" --PROMPT "Summarize the module boundaries and risky areas."

# Continue the same session
python scripts/claude_code_bridge.py --cd "/repo" --SESSION_ID "abc123" --PROMPT "Now propose 5 targeted tests for those risks."
```

## Parameters (bridge script)

- `--PROMPT` (required): Instruction to send to Claude Code.
- `--cd` (required): Working directory to run Claude Code in (typically the repo root).
- `--SESSION_ID` (optional): Resume an existing Claude Code session.
- `--model` (optional): Defaults to `claude-opus-4-5-20251101`.
- `--full-access` / `--no-full-access` (optional): Defaults to `--full-access` (non-interactive full access).
- `--permission-mode` (optional): Defaults to `bypassPermissions` when full access; defaults to `plan` when no full access.
- `--tools` (optional): Defaults to `default` (all tools) when full access; defaults to `Read,Glob,Grep,LS` when no full access.
- `--allowedTools` (optional): Defaults to `*` (allow all tools without prompts) when full access; defaults to `Read,Glob,Grep,LS` when no full access.
- `--return-all-messages` (optional): Return the full streamed JSON event list from Claude Code (useful for debugging).
- `--timeout-s` (optional): Defaults to 1200 seconds (20 minutes).

## Output format

The bridge prints JSON:

```json
{
  "success": true,
  "SESSION_ID": "abc123",
  "agent_messages": "…Claude output…",
  "all_messages": []
}
```

## Recommended delegation patterns

- **Second opinion**: “Propose an alternative approach and tradeoffs.”
- **Code review**: “Find bugs, race conditions, security issues; propose fixes.”
- **Test design**: “Write a test plan + edge cases; include example test code.”
- **Diff review**: “Review this patch; point out regressions and missing cases.”
