---
name: collaborating-with-claude-code
description: "Delegate code implementation/review/debugging/alternatives to Claude Code via a JSON bridge script. Multi-turn via SESSION_ID."
metadata:
  short-description: Use Claude Code as a collaborator
---

# Collaborating with Claude Code

Use this skill for second opinions, code review, test design, or code implementations/alternatives. The bridge runs `claude` (Claude Code) non-interactively ("print" mode) and returns JSON.

The script is located at `~/.codex/skills/collaborating-with-claude-code/scripts/claude_code_bridge.py`.

## Timing

Claude Code often needs **1–2+ minutes** per task.
- Prefer running the bridge directly (no `&`); increase `--timeout-s` as needed (default: 1800s).
- Do **NOT** redirect stdout to a file (e.g. `> /tmp/out.json`).
- By default, the bridge streams Claude's assistant text to `stderr`, and prints only the final JSON envelope to `stdout`.

## Context

- Do **NOT** read the script unless you are modifying it; 
- Before running the script, ALWAYS use `python <script_loc> --help` to get the usage instructions.

## Usage

- please always require claude code to fully understand the codebase before responding or making any changes.
- Put collaborating-with-claude-code terminal commands in the background terminal.
- Always review claude code's responses (or changes it makes) and make sure they are correct, constructive and complete.
- When claude code asks clarifying questions in a multi-turn session, always respond to its questions in that session based on current situation.

## Default

- **full access** (`--full-access`): use only in trusted repos/directories.
- **extended thinking ON** (can disable via `--no-extended-thinking`).
- **step mode AUTO** (can disable via `--step-mode off`).

## Output format

The bridge prints JSON to `stdout`:

```json
{"success": true, "SESSION_ID": "abc123", "agent_messages": "…Claude output…"}
```

## Recommended delegation patterns

- **Guided coding**: "Implement the code for [feature] following these specific steps/constraints."
- **Second opinion**: "Propose an alternative approach and tradeoffs."
- **Code review**: "Find bugs, race conditions, security issues; propose fixes."
- **Test design**: "Write a test plan + edge cases; include example test code."
- **Diff review**: "Review this patch; point out regressions and missing cases."
