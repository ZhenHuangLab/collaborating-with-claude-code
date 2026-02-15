# collaborating-with-claude-code

[中文](README.md) | English

A skill for **Codex CLI**: via a JSON bridge script, it delegates tasks such as **code review / debugging / alternative implementation comparisons** to **Anthropic Claude Code CLI** (default model: `claude-opus-4-6`), and returns results as structured JSON for multi-model collaboration.

The main entry points of this repository are `SKILL.md` (the Codex skill definition) and `scripts/claude_code_bridge.py` (the bridge script).

## Features

Compared to other similar skills / collaboration workflows, this skill has the following key advantages:

1. It uses state-of-the-art context engineering and follows the principle of progressive disclosure, so Codex can learn how to use the skill script with a single tool call, and then invoke it correctly on the second tool call—without needing to search for or read the script again.
2. It is compatible with a wide range of Anthropic-compatible proxies that enforce strict message-structure validation when extended thinking is enabled.
    - For some Anthropic-compatible proxy APIs, when thinking is enabled, an `assistant` message that contains a tool-call chain must follow rules like: the assistant message must start with `thinking/redacted_thinking`, then `tool_use`, followed by the corresponding `tool_result`, etc. However, when using Claude Code in print mode, if such a tool-call chain is produced, the `thinking` content may be filtered out on the Claude Code side, causing the assistant message to start directly with `tool_use`. This can trigger a 400 error from the router, even though the same issue usually does not occur in the interactive Claude Code UI.
    - To address this, the bridge script in this skill adopts a strategy of splitting one long agentic loop into many short loops:
        - Each iteration allows Claude Code to perform only a single agentic turn (at most one tool call, then stop).
        - Then the bridge script automatically sends a short “continue” instruction using the same `session_id` to let it proceed to the next step.
    - This approach maximizes compatibility when running Claude Code via various Anthropic-compatible proxy APIs.
3. It streams Claude Code's assistant text to `stderr` in real-time, making it easier for users and Codex to track progress.

## Install to `~/.codex/skills/`

1) Choose an installation directory (create it if it doesn't exist):

```bash
mkdir -p ~/.codex/skills
```

2) Clone this repository into the skills directory (the folder name is the skill name):

```bash
cd ~/.codex/skills
git clone https://github.com/ZhenHuangLab/collaborating-with-claude-code.git collaborating-with-claude-code
```

3) Verify the file structure (it should include at least `SKILL.md` and `scripts/`):

```bash
ls -la ~/.codex/skills/collaborating-with-claude-code
```

4) Confirm the `claude_code_bridge.py` script path:

By default, it is `~/.codex/skills/collaborating-with-claude-code/scripts/claude_code_bridge.py`. If it changes, update the path in `SKILL.md`.

Tests show that explicitly declaring the correct script path in `SKILL.md` makes Codex execute the bridge script more efficiently.

After that, Codex CLI can discover it when loading local skills; mention `collaborating-with-claude-code` (or `$collaborating-with-claude-code`, or a similar request in natural language) in a conversation to trigger it.

## Dependencies

- Python 3 (to run the bridge script).
- Claude Code CLI installed and available (make sure `claude --version` works).
- Claude Code authenticated (e.g. via the environment variable `ANTHROPIC_API_KEY`, or any other authentication method required by your local Claude Code setup).

> Note: this skill runs Claude Code in **full access** mode by default (non-interactive, bypassing confirmations). Only use it in directories / repositories you trust.

## Run manually (without Codex CLI)

```bash
python <script_loc> --cd "/path/to/repo" --PROMPT "Review the auth flow for bypasses; propose fixes as a unified diff."
```

Read-only review (avoid editing files / running commands):

```bash
python <script_loc> --no-full-access --cd "/path/to/repo" --PROMPT "Review the auth flow and list issues (no code changes)."
```

For a more complete parameter reference and multi-turn session usage, see `SKILL.md`.

## Runtime status output (`stderr`)

- While the bridge is running, it streams Claude assistant text to `stderr` by default (so you can see progress), and prints `session_id=...` once.
- The final structured result stays on `stdout` only (JSON is not mixed with `stderr` output).
- Disable all `stderr` output with `--quiet`.

## Compatibility

Tested on codex v0.87, v0.98, v0.101.0, claude code v2.1.11, v2.1.12, v2.1.25.

## License

MIT License. See `LICENSE`.
