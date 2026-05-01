# Hermes Agent Simple

A simplified copy of Hermes Agent built from `simple_hermes.md`.
Original project is:[hermes-agent](https://github.com/nousresearch/hermes-agent).
This tree keeps the core agent runtime and trims broad product/platform compatibility surfaces.

## Quick Start

Requirements:

- Python 3.11, 3.12, or 3.13
- `uv` recommended for dependency management
- Node.js, only needed for the TypeScript TUI development workflow

From this checkout:

```bash
cd ~/hermes-agent-simple

# Install Python dependencies and create .venv.
uv sync --all-extras

# Activate the environment for this shell.
source .venv/bin/activate

# Verify the installed entry point.
hermes version
```

If `hermes` is not available outside the virtualenv, create a user-level symlink:

```bash
mkdir -p ~/.local/bin
ln -sf ~/hermes-agent-simple/.venv/bin/hermes ~/.local/bin/hermes
```

Make sure `~/.local/bin` is on `PATH` if your shell does not already include it.

## Configuration

Hermes stores user configuration outside the repo:

```text
~/.hermes/config.yaml   # non-secret settings: model, provider, tools, terminal
~/.hermes/.env          # secrets only: API keys and tokens
~/.hermes/logs/         # agent.log, errors.log, gateway.log
~/.hermes/sessions/     # saved session data
```

Run the setup wizard if you want interactive configuration:

```bash
hermes setup
```

For a custom OpenAI-compatible endpoint, configure the model in `config.yaml` and
the key in `.env`:

```bash
hermes config set model.provider custom
hermes config set model.base_url https://api.example.com/v1
hermes config set model.default gpt-5.5
```

Then edit `~/.hermes/.env`:

```env
OPENAI_API_KEY=your-api-key
```

For official OpenAI, use:

```bash
hermes config set model.provider custom
hermes config set model.base_url https://api.openai.com/v1
hermes config set model.default gpt-4.1
```

For OpenRouter, use:

```bash
hermes config set model.provider openrouter
hermes config set model.default anthropic/claude-sonnet-4.5
```

and add this to `~/.hermes/.env`:

```env
OPENROUTER_API_KEY=your-openrouter-key
```

The `.env` file must contain only `KEY=value` lines. Do not put shell commands
such as `unset ...` in this file.

## Running

Classic interactive CLI:

```bash
hermes
```

Single prompt, useful for scripts:

```bash
hermes -z "Summarize this project"
```

Modern TUI:

```bash
hermes --tui
```

Resume the latest session:

```bash
hermes -c
```

Resume a named or specific session:

```bash
hermes -c "my project"
hermes --resume <session_id>
```

## Diagnostics

Check configuration, entry points, optional dependencies, and tool availability:

```bash
hermes doctor
```

Automatically fix simple local issues, such as a missing `~/.local/bin/hermes`
symlink:

```bash
hermes doctor --fix
```

Inspect the active configuration:

```bash
hermes config show
hermes config path
hermes config env-path
```

View logs:

```bash
hermes logs
hermes logs errors
hermes logs -f
```

`hermes doctor` may report missing API keys for optional tools. These are not
required for basic chat. Add them only when you need the corresponding feature,
for example web search, OpenRouter Mixture-of-Agents, GitHub-backed skills, or
external browser providers.

## Gateway

The simplified gateway is intentionally limited to Feishu and WeCom:

```bash
hermes gateway setup
hermes gateway start
hermes gateway status
hermes gateway stop
```

Supported platform adapters in this tree:

```text
feishu
wecom
wecom_callback
```

## MCP Client

This simplified tree keeps MCP client support:

```bash
hermes mcp add
hermes mcp list
hermes mcp test
hermes mcp configure
hermes mcp login
```

MCP server mode was removed from this tree.

## Development

Run tests through the project wrapper:

```bash
scripts/run_tests.sh
scripts/run_tests.sh tests/path/to/test_file.py::test_name
```

Do not call `pytest` directly for normal verification; the wrapper matches the
project's CI-like environment.

For TUI development:

```bash
cd ui-tui
npm install
npm run dev
npm run build
npm test
```

## Scope

Kept:

```text
AIAgent core loop
tool registry/toolsets/tool dispatch
MCP client integration
memory/session/checkpoint/compression systems
classic CLI
TypeScript TUI + Python tui_gateway
gateway daemon
cron scheduler
Feishu and WeCom platform adapters
P0 skills plus selected P1 skills
plugin architecture for memory/context/image generation
```

Removed:

```text
ACP editor protocol adapter
MCP Server mode
optional-skills default tree
most messaging platform adapters
RL/datagen/website/dashboard source surfaces
large release/test/documentation history
```

## Main Entrypoints

```bash
hermes
hermes --tui
hermes gateway start
hermes mcp add/list/test/configure/login
```

## Platform Scope

The gateway runtime is intentionally narrowed to:

```text
feishu
wecom
wecom_callback
```

`send_message` is also limited to Feishu/WeCom targets.

## Notes

This copy maximizes reuse of upstream classes and functions. Some compatibility
branches remain inside large shared files such as `run_agent.py` and
`gateway/config.py` so the stable runtime can be reused without a risky rewrite.
The public/default product surface is narrowed by removed directories, command
entrypoints, platform adapter factory scope, package metadata, and skills.

See `SIMPLE_HERMES_README.md` and `simple_hermes.md` for the detailed cut line.
