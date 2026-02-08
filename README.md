# collaborating-with-claude-code

中文 | [English](README_EN.md)

**Codex CLI** 的一个 skill ：通过一个 JSON bridge 脚本，把“代码审查 / 调试 / 方案对比”等任务委托给 **Anthropic Claude Code CLI**（默认模型：`claude-opus-4-5-20251101`），并以结构化 JSON 结果返回，便于在多模型协作中使用。

本仓库的核心入口是 `SKILL.md`（Codex skill 定义）以及 `scripts/claude_code_bridge.py`（桥接脚本）。

## Features

本skill相对于其他类似skill/协作模式，具有如下重要优势：

1. 使用SOTA的context engineering技术，遵循渐进式披露 (progressive disclosure) 的原则，让codex只需要一次tool call就可以掌握该skill script的使用方法，第二次tool call即可正确调用，无需再搜索/读取该脚本；
2. 兼容各类“在extended thinking开启后会对message结构进行严格校验的” Anthropic-compatible proxy. 
    - 对于某些Anthropic-compatible proxy API, 在thinking启用后，包含工具调用链路的`assistant`消息，需要满足“assistant message 必须以 `thinking/redacted_thinking` 开头，然后才是 `tool_use`，再配套 `tool_result` ……”这类规则。而在print模式下使用claude code时，如果产生上述tool call 链路，`thinking`部分信息会在claude code侧被filtered掉，assistant message会以`tool_use`开头，从而导致router返回400 Error. 而这一问题在claude code交互界面一般不会出现。
    - 针对这一问题，skill中的 bridge script 采取了“把一次长的agentic loop拆成很多次短的loop”的策略：
        - 每次仅允许 claude code 做一次 agentic turn （最多仅一次tool call就调用停止）
        - 然后bridge script 会用相同的session_id自动发送很短的继续指令，让它进行下一步
    - 通过这种方法，可以最大限度地兼容通过各类Anthropic-compatible proxy API运行的claude code.

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

4) 确认`claude_code_bridge.py`脚本路径：

默认为`~/.codex/skills/collaborating-with-claude-code/scripts/claude_code_bridge.py`. 如果有变动，请在 `SKILL.md`中修改.

测试表明，在 `SKILL.md` 中直接显式声明script的正确路径，会让codex执行bridge script变得更加高效。

完成后，Codex CLI 在加载本地 skills 时就能发现它；在对话中提到 `collaborating-with-claude-code`（或 `$collaborating-with-claude-code`，或自然语言的类似要求）即可触发使用。

## 依赖

- Python 3（用于运行 bridge 脚本）。
- 已安装并可用的 Claude Code CLI（确保 `claude --version` 可运行）。
- Claude Code 已完成认证（例如通过环境变量 `ANTHROPIC_API_KEY`，或你本机 Claude Code 所需的其它认证方式）。

> 注意：该 skill 默认以 **full access** 方式运行 Claude Code（非交互、绕过确认），只建议在你信任的目录/仓库中使用。

## 手动运行（不通过 Codex CLI）

```bash
python <script_loc> --cd "/path/to/repo" --PROMPT "Review the auth flow for bypasses; propose fixes as a unified diff."
```

只读审查（避免改文件/跑命令）：

```bash
python <script_loc> --no-full-access --cd "/path/to/repo" --PROMPT "Review the auth flow and list issues (no code changes)."
```

更完整的参数说明与多轮会话用法见 `SKILL.md`。

## 运行状态输出（stderr）

- 脚本运行过程中，默认会向 `stderr` 输出极简状态：`working...`、`session_id=...`、`done`。
- 最终结构化结果仍只输出到 `stdout`（JSON 不会被状态文本污染）。
- 如需关闭状态输出：`--quiet-status`
- 如需调整心跳频率：`--status-interval-s 30`

## 兼容性

已经在codex v0.87, claude code v2.1.11 and v2.1.12 测试通过.

## License

MIT License，详见 `LICENSE`。
