# collaborating-with-claude-code

[中文](README.md) | English

A skill for **Codex CLI**: via a JSON bridge script, it delegates tasks such as **code review / debugging / alternative implementation comparisons** to **Anthropic Claude Code CLI** (default model: `claude-opus-4-5-20251101`), and returns results as structured JSON for multi-model collaboration.

The main entry points of this repository are `SKILL.md` (the Codex skill definition) and `scripts/claude_code_bridge.py` (the bridge script).

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

After that, Codex CLI can discover it when loading local skills; mention `collaborating-with-claude-code` (or `$collaborating-with-claude-code`) in a conversation to trigger it.

## Dependencies

- Python 3 (to run the bridge script).
- Claude Code CLI installed and available (make sure `claude --version` works).
- Claude Code authenticated (e.g. via the environment variable `ANTHROPIC_API_KEY`, or any other authentication method required by your local Claude Code setup).

> Note: this skill runs Claude Code in **full access** mode by default (non-interactive, bypassing confirmations). Only use it in directories / repositories you trust.

## Run manually (without Codex CLI)

```bash
python scripts/claude_code_bridge.py --cd "/path/to/repo" --PROMPT "Review the auth flow for bypasses; propose fixes as a unified diff."
```

Read-only review (avoid editing files / running commands):

```bash
python scripts/claude_code_bridge.py --no-full-access --cd "/path/to/repo" --PROMPT "Review the auth flow and list issues (no code changes)."
```

For a more complete parameter reference and multi-turn session usage, see `SKILL.md`.

## Recommended `AGENTS.md` configuration

It is recommended to add the following prompt to `AGENTS.md` to improve the efficiency and quality of collaboration between Codex and Claude Code:

```
When collaborating with Claude Code:
- please always require claude code to fully understand the codebase before responding or making any changes.
- Put collaborating-with-claude-code terminal commands in the background terminal.
- Always review claude code's responses (or changes it makes) and make sure they are correct, constructive and complete.
- When claude code asks clarifying questions in a multi-turn session, always respond to its questions in that session based on current situation.
```

## License

MIT License. See `LICENSE`.

