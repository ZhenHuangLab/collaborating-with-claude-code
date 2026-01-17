# collaborating-with-claude-code

中文 | [English](README_EN.md)

**Codex CLI** 的一个 skill ：通过一个 JSON bridge 脚本，把“代码审查 / 调试 / 方案对比”等任务委托给 **Anthropic Claude Code CLI**（默认模型：`claude-opus-4-5-20251101`），并以结构化 JSON 结果返回，便于在多模型协作中使用。

本仓库的核心入口是 `SKILL.md`（Codex skill 定义）以及 `scripts/claude_code_bridge.py`（桥接脚本）。

## 安装到 `~/.codex/skills/`

1) 选择一个安装目录（如果不存在就创建）：

```bash
mkdir -p ~/.codex/skills
```

2) 克隆本仓库到 skills 目录下（目录名就是 skill 名）：

```bash
cd ~/.codex/skills
git clone https://github.com/ZhenHuangLab/collaborating-with-claude-code.git collaborating-with-claude-code
```

3) 验证文件结构（至少应包含 `SKILL.md` 和 `scripts/`）：

```bash
ls -la ~/.codex/skills/collaborating-with-claude-code
```

完成后，Codex CLI 在加载本地 skills 时就能发现它；在对话中提到 `collaborating-with-claude-code`（或 `$collaborating-with-claude-code`）即可触发使用。

## 依赖

- Python 3（用于运行 bridge 脚本）。
- 已安装并可用的 Claude Code CLI（确保 `claude --version` 可运行）。
- Claude Code 已完成认证（例如通过环境变量 `ANTHROPIC_API_KEY`，或你本机 Claude Code 所需的其它认证方式）。

> 注意：该 skill 默认以 **full access** 方式运行 Claude Code（非交互、绕过确认），只建议在你信任的目录/仓库中使用。

## 手动运行（不通过 Codex CLI）

```bash
python scripts/claude_code_bridge.py --cd "/path/to/repo" --PROMPT "Review the auth flow for bypasses; propose fixes as a unified diff."
```

只读审查（避免改文件/跑命令）：

```bash
python scripts/claude_code_bridge.py --no-full-access --cd "/path/to/repo" --PROMPT "Review the auth flow and list issues (no code changes)."
```

更完整的参数说明与多轮会话用法见 `SKILL.md`。

## 推荐`AGENTS.md`配置

推荐在`AGENTS.md`中配置以下prompt, 提升codex与claude code的交互效率与效果：

```
When collaborating with Claude Code:
- please always require claude code to fully understand the codebase before responding or making any changes.
- Put collaborating-with-claude-code terminal commands in the background terminal.
- Always review claude code's responses (or changes it makes) and make sure they are correct, constructive and complete.
- When claude code asks clarifying questions in a multi-turn session, always respond to its questions in that session based on current situation.
```

## License

MIT License，详见 `LICENSE`。
