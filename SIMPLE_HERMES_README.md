# Hermes Agent Simple

This tree is a simplified copy of `hermes-agent` built from the rules in
`simple_hermes.md`.

The goal is to keep the stable agent runtime and remove broad product/platform
compatibility surfaces.

## Kept

```text
run_agent.py
  AIAgent, core loop, memory/state, compression, interrupt/steer, delegation

model_tools.py / tools/ / toolsets.py
  tool registry, tool schemas, dispatch, toolsets, MCP client integration

agent/
  memory, compression, provider/runtime helpers used by AIAgent

plugins/
  plugin surfaces, especially memory/context-engine style extension points

cron/
  scheduled agent jobs, narrowed delivery target set

gateway/
  long-running gateway/session service
  retained platform adapters: Feishu, WeCom, WeCom callback

cli.py / hermes_cli/
  classic CLI and command infrastructure

ui-tui/ / tui_gateway/
  TypeScript terminal UI plus Python JSON-RPC backend

skills/
  P0 skills and selected P1 skills only
```

## Removed

```text
acp_adapter/
acp_registry/
mcp_serve.py
optional-skills/
RL/environments/datagen surfaces
website/web/dashboard source
most messaging platform adapters
MCP Server mode
ACP editor protocol mode
```

## Platform Scope

The simplified gateway only instantiates:

```text
feishu
wecom
wecom_callback
```

`send_message` is intentionally limited to Feishu/WeCom delivery. Other platform
enum names remain in a few shared data structures only to keep reused upstream
session/config code stable.

## MCP Scope

Hermes Simple keeps MCP client capability:

```text
hermes mcp add
hermes mcp list
hermes mcp test
hermes mcp configure
hermes mcp login
```

It removes MCP Server capability:

```text
hermes mcp serve
mcp_serve.py
conversation/channel bridge tools for external MCP clients
```

## Skills Scope

Default bundled skills are reduced to general agent work:

```text
planning
debugging
TDD
code review
subagent workflow
Codex delegation
GitHub workflow
MCP client
documents/OCR
Obsidian notes
dogfood QA
selected research/productivity helpers
```

Heavy creative, ML, platform-specific, entertainment, and reference skills were
removed from this tree. They can be reintroduced later as installable optional
skills.
