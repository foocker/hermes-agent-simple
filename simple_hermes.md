# Simple Hermes Notes

## API Client Initialization

The full `run_agent.py` supports many providers and protocols, so the real
initialization path has a lot of branching:

- OpenAI-compatible APIs
- Anthropic Messages API
- AWS Bedrock / Bedrock Converse
- OpenRouter, GitHub Copilot, Kimi, Qwen Portal, RouterMint, ChatGPT/Codex
- API keys, OAuth tokens, AWS SDK credentials, provider-specific headers
- fallback providers and automatic provider routing

For a simplified Hermes, this can be reduced to one fixed provider.

### Minimal Responsibility

The agent only needs one model client that can send chat messages and receive
assistant responses/tool calls.

Minimal flow:

```text
read api_key and base_url
build OpenAI-compatible client
store client kwargs for later rebuild
use this client for every model call
```

### Simplified Code Shape

```python
import os
from openai import OpenAI


class SimpleAgent:
    def __init__(self, model: str, api_key: str | None = None, base_url: str | None = None):
        self.model = model
        self.api_key = api_key or os.environ["OPENAI_API_KEY"]
        self.base_url = base_url or "https://api.openai.com/v1"

        self._client_kwargs = {
            "api_key": self.api_key,
            "base_url": self.base_url,
        }
        self.client = OpenAI(**self._client_kwargs)
```

### What Gets Removed

If only one OpenAI-compatible provider is supported, the simplified version does
not need:

- `api_mode` branching
- Anthropic native client setup
- Bedrock / boto3 credential handling
- provider router resolution
- provider-specific default headers
- OAuth token detection
- fallback provider chain
- special handling for Azure query params unless Azure is the chosen provider
- checks that prevent one provider's token from being sent to another provider

### Why Hermes Is More Complex

Hermes carries product-level complexity because it wants to work for many users
with many API sources. The agent loop itself does not require that complexity.

The core idea can stay simple:

```text
user message
  -> model call
  -> optional tool calls
  -> execute tools
  -> append tool results
  -> next model call
  -> final response
```

Provider adaptation is an outer layer around that loop, not the essence of the
agent architecture.

## Simplification Boundary

The simplified version should not remove necessary engineering controls. The
goal is to simplify broad compatibility layers, not to weaken the agent runtime.

Good simplification targets:

- multiple LLM providers
- multiple LLM API protocols
- provider-specific authentication
- provider-specific headers and request quirks
- many platform entry points
- OS/profile/platform compatibility branches that are not needed for the first product

Things to keep:

- tool registry
- toolset composition
- tool availability checks
- permission checks
- dynamic schemas
- schema sanitization
- tool-definition caching
- plugin/MCP-style dynamic tools if external tool expansion is part of the product
- stable tool dispatch and error wrapping

The right simplified architecture is:

```text
one model backend
  + one agent loop
  + serious tool governance
```

not:

```text
one model backend
  + hardcoded tools with no management layer
```

### Tool Definition Flow

The full Hermes `get_tool_definitions()` is worth preserving in spirit. Its job
is to decide which tools the model is allowed to see for the current turn.

Simplified flow:

```text
selected toolsets
  -> resolve tool names
  -> read matching entries from registry
  -> run availability / permission checks
  -> build dynamic schemas where needed
  -> sanitize schemas for backend compatibility
  -> cache the result
  -> send schemas to the model
```

Minimal code shape:

```python
def get_tool_definitions(enabled_toolsets: list[str]) -> list[dict]:
    cache_key = tool_cache_key(enabled_toolsets, registry.generation, config_fingerprint())
    if cache_key in tool_schema_cache:
        return list(tool_schema_cache[cache_key])

    tool_names = resolve_toolsets(enabled_toolsets)
    definitions = []

    for name in sorted(tool_names):
        entry = registry.get(name)
        if entry is None:
            continue
        if entry.check_fn and not entry.check_fn():
            continue

        schema = entry.schema
        if entry.dynamic_schema_fn:
            schema = entry.dynamic_schema_fn(available_tools=tool_names)

        definitions.append({
            "type": "function",
            "function": {**schema, "name": entry.name},
        })

    definitions = sanitize_tool_schemas(definitions)
    tool_schema_cache[cache_key] = definitions
    return list(definitions)
```

The exact implementation can be smaller than Hermes, but the responsibilities
should remain. Tool governance is part of product stability, not provider
compatibility noise.

## Memory And State Should Stay Structured

The memory/state layer should also not be simplified away. It uses different
storage formats because the data has different access patterns.

Keep these responsibilities separate:

```text
session_*.json
  full raw session trajectory for debugging and recovery

state.db
  SQLite session index, message history, resume/list/search, FTS5 search

MEMORY.md / USER.md
  small curated long-term memory, human-readable and prompt-ready

MemoryManager providers
  optional external memory systems: semantic recall, remote memory, per-user
  or per-session memory, provider-specific tools

CheckpointManager
  transparent filesystem snapshots before file mutation
```

The important design point is that these are not redundant.

JSON is good for complete structured snapshots. SQLite is good for many
sessions, queries, pagination, search, and concurrent access. Markdown is good
for small curated facts that humans can inspect and that the agent can inject
directly into the system prompt.

### Built-In Memory Versus External Providers

The built-in memory store and external memory providers are parallel sources.
One does not automatically replace the other.

```text
AIAgent
  -> MemoryStore
       -> MEMORY.md
       -> USER.md

  -> MemoryManager
       -> active external provider
       -> provider system prompt block
       -> provider prefetch
       -> provider sync_turn
       -> provider tool schemas
```

`MemoryStore` is for small, trusted, manually reviewable long-term memory. It is
loaded at session start and injected as a frozen system-prompt snapshot. Writes
during a session persist immediately to disk, but they do not mutate the current
system prompt. This preserves prompt caching.

`MemoryManager` is an orchestration layer for external memory providers. It lets
the product experiment with different memory strategies without coupling the
agent core to one storage backend or one memory philosophy.

For the simplified Hermes design, keep the memory architecture:

- keep session JSON logs
- keep SQLite `SessionDB`
- keep curated `MEMORY.md` / `USER.md`
- keep `MemoryManager` provider abstraction
- keep checkpoint snapshots

The simplification should target broad platform/provider compatibility, not
the memory and state model.

## Fallback Runtime Snapshot Can Be Removed

Hermes stores a `_primary_runtime` snapshot so it can temporarily switch to a
fallback model/provider during a turn and then restore the preferred runtime on
the next turn.

That snapshot includes:

```text
model
provider
base_url
api_mode
api_key
client kwargs
prompt caching mode
context compressor model
context compressor provider/base_url/api_key
context length
compression threshold
```

This is necessary when fallback can change the model backend, provider,
credentials, API protocol, prompt-caching behavior, or context window.

For a simplified Hermes with one fixed GPT/OpenAI backend and no fallback
provider chain, this runtime snapshot is not needed. The agent can keep one
stable runtime for the whole session:

```python
self.model = model
self.client = OpenAI(api_key=api_key, base_url=base_url)
self.context_compressor = ContextCompressor(
    model=model,
    context_length=context_length,
    threshold_percent=0.5,
)
```

What should remain is the compression feasibility check:

```text
current model context_length
compression threshold
compression summary model
summary model context_length
```

Even with one model provider, compression still needs to know whether the model
used for summarization can fit the content it is asked to summarize. Remove the
fallback restoration machinery, not the compression safety checks.

## Simple AIAgent Surface

The full `AIAgent` is long because it is the central runtime for many concerns:

```text
model client setup
system prompt construction
agent loop
tool execution
session persistence
memory lifecycle
context compression
interrupt / steer
streaming
fallback providers
provider-specific message compatibility
platform callbacks
```

For a simplified Hermes, keep the core runtime responsibilities and remove the
multi-provider / multi-platform compatibility machinery.

### Methods To Keep

Public surface:

```python
__init__()
chat()
run_conversation()
interrupt()
clear_interrupt()
steer()
close()
```

Core internals:

```python
_build_system_prompt()
_check_compression_model_feasibility()
_compress_context()

_persist_session()
_save_session_log()
_flush_messages_to_session_db()

_build_api_kwargs()
_call_model()
_build_assistant_message()

_execute_tool_calls()
_execute_tool_calls_sequential()
_execute_tool_calls_concurrent()
_invoke_tool()

_drain_pending_steer()
_apply_pending_steer_to_tool_results()
```

Memory lifecycle hooks should remain if the memory provider abstraction remains:

```python
_sync_external_memory_for_turn()
commit_memory_session()
shutdown_memory_provider()
```

### Methods Or Areas To Remove First

These mainly exist for broad compatibility and can be removed in a one-provider
design:

```text
switch_model()
fallback activation / primary runtime restore
credential pool recovery
provider-specific credential refresh
Anthropic / Bedrock / Gemini / Copilot adapters
Qwen / Kimi / DeepSeek special message preparation
OpenRouter-specific behavior
vision fallback logic if vision is not in scope
provider-specific prompt caching policy
LM Studio / Ollama context probing
large platform-specific output/display branches
background review machinery
```

### Simplified Loop Shape

The simplified `AIAgent` should still be a real agent runtime, not just a
single model-call wrapper.

```python
class SimpleAgent:
    def chat(self, message: str) -> str:
        return self.run_conversation(message)["final_response"]

    def run_conversation(self, user_message: str, conversation_history=None):
        messages = self._prepare_turn(user_message, conversation_history)
        messages = self._maybe_compress(messages)

        while self._has_iteration_budget():
            api_messages = self._build_api_messages(messages)
            response = self._call_model(api_messages)
            assistant_msg = self._build_assistant_message(response)
            messages.append(assistant_msg)

            if assistant_msg.get("tool_calls"):
                self._execute_tool_calls(assistant_msg, messages)
                continue

            return self._finish_turn(messages, assistant_msg)
```

The loop should preserve these runtime properties:

```text
stable system prompt
tool-calling loop
tool governance
session persistence
memory integration
context compression
interrupt / steer
```

The simplification should remove backend/platform adaptation, not the core
agent runtime discipline.

## Remove Nous Product Coupling

The simplified product should not depend on Nous-specific branding, subscription
copy, provider defaults, portal URLs, or paid Tool Gateway messaging.

This does not mean removing the underlying tool capabilities. Keep useful tools
such as:

```text
web_search
web_extract
browser_navigate
browser_snapshot
browser_click
browser_type
browser_scroll
browser_console
browser_press
browser_get_images
browser_vision
image_generate
text_to_speech
terminal
process
execute_code
```

What should change is the product binding around those tools.

Examples of coupling to replace:

```python
nous_subscription_prompt = build_nous_subscription_prompt(self.valid_tool_names)
```

and related areas such as:

```text
Nous Portal provider defaults
Nous auth status
Nous subscription feature prompts
Nous Tool Gateway upgrade nudges
Nous-specific docs links
Nous model catalog URLs
Nous-managed tool defaults
```

For a simplified Hermes, replace these with either:

```text
neutral local/direct configuration
```

or:

```text
your own product name, docs URL, billing model, and managed-tool backend
```

The functional layer should stay separate from the commercial layer:

```text
Tool capability:
  browser automation, web search, image generation, TTS, code execution

Commercial/product layer:
  who hosts the managed backend, what is paid, what docs/upgrade links to show
```

The simplified version should prefer direct user-owned credentials and local
tool configuration by default. Managed/paid gateways can be added later as an
optional provider, not as a core dependency of the agent runtime.

## Gateway Should Be Narrow, Not Removed

The gateway layer is necessary when the agent must run as a long-lived service
behind messaging platforms. It should remain responsible for:

```text
platform message receive/send
session key -> session id mapping
per-user/per-chat context isolation
slash/control commands
running-agent interrupt
approval/deny flows
message queueing or steer behavior
background process completion notifications
agent cache and idle cleanup
```

What can be simplified is the number of platform adapters.

The full Hermes gateway supports many platforms:

```text
Telegram
Discord
Slack
WhatsApp
Signal
Matrix
Mattermost
Email
SMS
DingTalk
WeCom
Weixin
Feishu
QQ bot
Webhook/API server
...
```

For a simplified product, keep the gateway architecture but implement only the
target platforms. For example:

```text
gateway/
  run.py                 # shared runner and session orchestration
  session.py             # session store
  platforms/
    base.py              # common adapter contract
    wecom.py             # Enterprise WeChat
    feishu.py            # Feishu/Lark
```

The shared runner should not know platform-specific API details. Platform
adapters should normalize incoming events into one internal shape, such as:

```python
MessageEvent(
    platform="wecom",
    user_id="...",
    chat_id="...",
    thread_id=None,
    text="...",
    attachments=[],
)
```

Then the runner handles the common flow:

```text
receive MessageEvent
  -> resolve session
  -> handle control command if needed
  -> create/reuse AIAgent
  -> run_conversation()
  -> send response through the same adapter
```

The simplification target is platform breadth, not gateway responsibilities.
Keep the service/session/control architecture; only reduce the adapter list.

## Simplify `_build_api_kwargs`

The full Hermes `_build_api_kwargs()` handles many backend protocols and
provider-specific request quirks:

```text
Anthropic Messages
AWS Bedrock Converse
OpenRouter provider preferences
Nous-specific options
Qwen portal session metadata
Kimi / Moonshot quirks
NVIDIA NIM quirks
GitHub Models reasoning extras
LM Studio reasoning options
Ollama num_ctx
custom max_tokens parameter names
vision fallback for non-vision models
provider-specific temperature handling
```

For a simplified Hermes that only supports official GPT/OpenAI or
OpenAI-compatible mirror endpoints, this should be reduced to one request
shape.

Codex can remain in scope because it is part of the OpenAI/GPT family. Treat it
as an official OpenAI request mode, not as a third-party provider adapter. The
simplified code can support either:

```text
Chat Completions only
```

or:

```text
Chat Completions + OpenAI Responses/Codex
```

but it should not carry third-party provider quirks in this function.

Required inputs:

```text
model
messages
tools
base_url
api_key
max_tokens
temperature
timeout
```

Minimal code shape:

```python
def _build_api_kwargs(self, api_messages: list[dict]) -> dict:
    kwargs = {
        "model": self.model,
        "messages": api_messages,
    }

    if self.tools:
        kwargs["tools"] = self.tools
        kwargs["tool_choice"] = "auto"

    if self.max_tokens is not None:
        kwargs["max_tokens"] = self.max_tokens

    if self.temperature is not None:
        kwargs["temperature"] = self.temperature

    if self.request_timeout is not None:
        kwargs["timeout"] = self.request_timeout

    return kwargs
```

The OpenAI client itself already holds:

```python
OpenAI(api_key=api_key, base_url=base_url)
```

So `_build_api_kwargs()` does not need to know about provider routing,
credential lookup, portal headers, fallback providers, or protocol adapters.

If mirror endpoints are intended to be OpenAI-compatible, treat compatibility
as a requirement of the mirror. Do not add mirror-specific branches unless a
real production endpoint proves they are necessary.

Keep only small backend-neutral sanitation before this step:

```text
remove invalid tool-call pairs
ensure message roles are valid
ensure tool schemas are sanitized
optionally strip unsupported image parts if vision is not in scope
```

Everything else should stay outside the simplified core.

## Simplify `_interruptible_api_call`

The full Hermes `_interruptible_api_call()` handles several API modes:

```text
codex_responses
anthropic_messages
bedrock_converse
chat_completions
```

It also creates per-request clients, closes stale sockets, kills hung calls,
propagates gateway activity heartbeats, supports interrupt abort, and lets the
outer retry/fallback machinery recover.

For a simplified GPT/OpenAI-compatible backend, remove the protocol branches:

```text
no Anthropic branch
no Bedrock branch
no per-provider rebuild path
no fallback-provider recovery
```

Codex/Responses may remain as a second OpenAI-official path if the product needs
Codex models. In that case, keep one small branch:

```python
if self.api_mode == "responses":
    return self.client.responses.create(**api_kwargs)
return self.client.chat.completions.create(**api_kwargs)
```

The simplification is to remove third-party protocol adapters, not OpenAI's own
GPT/Codex modes.

Still keep the useful runtime behavior:

```text
call runs in a worker thread
main thread can notice interrupt
main thread can notice stale timeout
activity heartbeat can be updated for gateway
timeout raises a clear error
interrupt raises InterruptedError
```

Minimal code shape:

```python
def _call_model(self, api_kwargs: dict):
    result = {"response": None, "error": None}

    def worker():
        try:
            result["response"] = self.client.chat.completions.create(**api_kwargs)
        except Exception as exc:
            result["error"] = exc

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    started = time.time()
    while thread.is_alive():
        thread.join(timeout=0.3)

        if self._interrupt_requested:
            raise InterruptedError("Agent interrupted during API call")

        elapsed = time.time() - started
        if elapsed > self.api_timeout:
            raise TimeoutError(
                f"Model API call timed out after {int(elapsed)}s"
            )

        if int(elapsed) % 30 == 0:
            self._touch_activity(
                f"waiting for model response ({int(elapsed)}s elapsed)"
            )

    if result["error"] is not None:
        raise result["error"]

    return result["response"]
```

If streaming is not required initially, do not implement the separate streaming
API path yet. If streaming is required, keep it as a separate `_stream_model()`
path rather than mixing streaming and non-streaming behavior into one large
function.

The simplified version should preserve interruptibility and timeout safety, but
not carry multi-backend recovery logic.

## Simplify `_build_assistant_message`

The full Hermes `_build_assistant_message()` normalizes assistant responses from
many providers. It preserves or repairs fields for:

```text
inline <think> blocks
reasoning_content
reasoning_details
DeepSeek thinking replay
Kimi / Moonshot thinking replay
Gemini thought signatures
OpenRouter opaque reasoning details
Codex encrypted reasoning items
Codex structured message items
tool_call id / call_id / response_item_id repair
surrogate cleanup
```

For a simplified GPT/OpenAI backend, keep only the OpenAI shapes:

```text
assistant content
finish_reason
tool_calls
optional OpenAI reasoning fields if present
optional Codex Responses continuity fields
```

Minimal Chat Completions shape:

```python
def _build_assistant_message(self, choice) -> dict:
    assistant = choice.message
    msg = {
        "role": "assistant",
        "content": assistant.content or "",
        "finish_reason": choice.finish_reason,
    }

    if getattr(assistant, "tool_calls", None):
        msg["tool_calls"] = [
            {
                "id": tool_call.id,
                "type": tool_call.type,
                "function": {
                    "name": tool_call.function.name,
                    "arguments": tool_call.function.arguments,
                },
            }
            for tool_call in assistant.tool_calls
        ]

    if getattr(assistant, "reasoning", None):
        msg["reasoning"] = assistant.reasoning

    return msg
```

If Codex/Responses remains in scope, preserve only the OpenAI/Codex continuity
fields needed for replay and cache stability:

```python
if getattr(assistant, "codex_reasoning_items", None):
    msg["codex_reasoning_items"] = assistant.codex_reasoning_items

if getattr(assistant, "codex_message_items", None):
    msg["codex_message_items"] = assistant.codex_message_items
```

Remove provider-specific compatibility branches such as:

```text
DeepSeek reasoning_content padding
Kimi / Moonshot thinking replay rules
Gemini extra_content thought signatures
OpenRouter reasoning_details preservation unless using OpenRouter
inline <think> extraction for non-OpenAI providers
provider-specific tool id derivation beyond OpenAI/Codex needs
```

Keep basic sanitation that is backend-neutral:

```text
ensure content is a string
sanitize invalid unicode/surrogates if needed
preserve tool call ids exactly
return messages in OpenAI-compatible assistant-message format
```

## Keep Tool Execution And Session Persistence

`_execute_tool_calls()` and `_persist_session()` should not be heavily
simplified. They are core agent-runtime responsibilities, not multi-provider
compatibility layers.

### `_execute_tool_calls`

Keep the execution model:

```text
read assistant tool_calls
parse function arguments
route to the correct tool handler
execute sequentially or concurrently
append role="tool" messages with matching tool_call_id
preserve tool-call ordering where required
surface tool errors as tool results
support interrupt
support checkpoints before file mutation
support memory/context-engine tool routing
emit progress callbacks where the gateway/CLI needs them
```

Possible simplifications:

```text
remove decorative CLI output
reduce provider-specific tool-call repair
reduce platform-specific progress formatting
omit concurrent execution initially if not needed
```

But do not collapse tool execution into an ad hoc direct function call. Tool
message correctness is essential for the next model turn.

### `_persist_session`

Keep the persistence flow:

```python
def _persist_session(self, messages, conversation_history=None):
    self._session_messages = messages
    self._save_session_log(messages)
    self._flush_messages_to_session_db(messages, conversation_history)
```

The responsibilities should remain:

```text
write full JSON session trajectory
append messages to SQLite SessionDB
avoid duplicate DB writes
preserve searchable history
handle compression session rollover
keep latest in-memory session messages
```

Possible simplifications:

```text
remove batch/trajectory export if not needed
remove provider-specific reasoning fields from persistence
reduce verbose debug metadata
```

The key rule: simplify provider/platform noise, not durability. A production
agent must not lose the conversation state just because a turn exits early,
compresses, errors, or is interrupted.

## Keep `ToolEntry`, But Use A Dataclass

The tool registry should keep an internal structured type for registered tools.
It separates runtime metadata from the OpenAI-facing tool schema.

```text
schema
  what the model sees: name, description, parameters

handler
  the Python function that actually executes the tool

check_fn
  availability / permission check

toolset
  grouping and selection

metadata
  env vars, async flag, emoji, description, result-size limits
```

The full Hermes implementation uses a hand-written class with `__slots__`.
For the simplified implementation, prefer a dataclass:

```python
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass(slots=True)
class ToolEntry:
    name: str
    toolset: str
    schema: dict[str, Any]
    handler: Callable[..., str]
    check_fn: Callable[[], bool] | None = None
    requires_env: list[str] = field(default_factory=list)
    is_async: bool = False
    description: str = ""
    emoji: str = ""
    max_result_size_chars: int | None = None
```

This keeps the benefits of a fixed internal model while making the code easier
to read and maintain.

Do not collapse registry entries into loose dicts. The registry needs a stable
internal type because one registered tool serves several purposes:

```text
schema exposure to the model
toolset filtering
availability checks
runtime dispatch
UI/config display
result budgeting
```

## Keep `model_tools.py` As The Tool Runtime Facade

`model_tools.py` should remain the facade between `AIAgent` and the tool
registry.

It should be responsible for:

```text
discovering built-in tools
discovering plugin tools if plugins are enabled
building model-facing tool definitions
filtering by enabled/disabled toolsets
running availability / permission checks
applying dynamic schemas
sanitizing schemas
coercing model-emitted arguments to schema types
dispatching tool calls
running pre/post/transform tool hooks if plugins are enabled
providing query helpers for CLI/status/debug screens
```

It should not implement the actual tool behavior. Concrete tools should stay in
`tools/*.py` and register themselves with the registry.

### What To Keep

Keep these responsibilities:

```text
get_tool_definitions()
handle_function_call()
coerce_tool_args()
schema caching
dynamic schema updates
schema sanitization
registry dispatch
tool availability checks
toolset query helpers
```

Keep `_run_async()` if any tools are async or if the gateway runs inside an
async event loop.

`_run_async()` is not only for a specific platform such as Feishu or Enterprise
WeChat. It exists because Hermes has a synchronous agent loop but may need to
execute async tool handlers from different runtime contexts:

```text
CLI main thread
gateway asyncio event loop
Feishu / WeCom / other messaging adapters running in async server code
ThreadPoolExecutor workers for concurrent tool calls
plugin tools using async HTTP clients
MCP or browser tools with async internals
```

The problem it solves:

```text
sync agent loop
  -> needs to call async tool handler
  -> must not call asyncio.run() inside an already-running event loop
  -> must not create/close event loops repeatedly when async clients cache loop state
  -> must isolate worker-thread event loops during concurrent tool execution
```

So the simplified version can drop `_run_async()` only if all tools are strictly
synchronous and the product does not run tool dispatch from an async gateway.
For a gateway product, keep it.

### What To Simplify

Remove or postpone:

```text
legacy *_tools toolset names if backward compatibility is not needed
Discord-specific dynamic schemas if Discord is not supported
platform-specific schema patches for unsupported platforms
process-global _last_resolved_tool_names if explicit enabled_tools can be passed
MCP discovery if MCP is not in the first product version
plugin hooks if plugins are not in the first product version
```

If plugins and MCP remain in scope, keep the corresponding discovery and hook
points. The simplification target is old compatibility and unsupported platform
branches, not the tool runtime facade itself.

## Simplify `cli.py`

`cli.py` is the classic terminal entry point for Hermes Agent. It is not the
agent loop itself. Its job is to turn terminal interaction into calls to
`AIAgent.run_conversation()`.

Keep this separation:

```text
cli.py / HermesCLI
  terminal UI, input, slash commands, session commands, display

run_agent.py / AIAgent
  model loop, tools, memory, compression, persistence
```

### CLI Responsibilities To Keep

Keep the basic terminal product:

```text
interactive input loop
single-query mode
agent initialization
conversation_history maintenance
basic slash command dispatch
session new / resume / status
manual compress command
interrupt handling
tool progress display
final response display
basic config loading
```

Core methods to keep in spirit:

```text
HermesCLI.__init__()
HermesCLI.run()
HermesCLI.chat()
HermesCLI._init_agent()
HermesCLI.process_command()
main()
```

### CLI Areas To Simplify

Remove or postpone:

```text
heavy skin/theme system
large banner/spinner/emoji customization
Nous login/subscription/status surfaces
multi-provider model setup UI
provider-specific auth commands
worktree isolation mode if not needed initially
large setup wizard flows
legacy command aliases
platform-specific display quirks
rare slash commands
batch/research/debug-only UI branches
```

The simplified CLI should expose a small stable command set:

```text
/help
/new
/resume
/status
/compress [focus]
/tools
/skills
/quit
```

Single-query mode should remain:

```bash
python cli.py -q "your question"
```

Interactive mode should remain:

```bash
python cli.py
```

The CLI simplification should not change `AIAgent` behavior. It should only
reduce terminal UI/product complexity around the agent.

## Treat `batch_runner.py` As Optional Offline Infrastructure

`batch_runner.py` is not part of the online chat runtime. It is an offline batch
executor for datasets.

Its job:

```text
load JSONL dataset
split prompts into batches
run AIAgent for each prompt
parallelize across workers
save trajectories
collect tool/reasoning statistics
write checkpoint.json
resume interrupted runs
```

This is useful for:

```text
evaluation
benchmarking
trajectory collection
training data generation
tool-use statistics
batch experiments
```

It is not required for:

```text
CLI chat
gateway chat
tool execution
memory
compression
session persistence
```

For a simplified product, `batch_runner.py` can be postponed unless offline
evaluation/data generation is a first-order requirement. If kept, simplify it to
the fixed GPT/OpenAI runtime and remove provider distribution options that are
only needed for OpenRouter/multi-provider experiments.

Keep:

```text
dataset loading
batching
parallel worker execution
checkpoint/resume
trajectory output
summary statistics
```

Remove or postpone:

```text
provider preference routing
OpenRouter-specific distribution controls
large toolset distribution experiments if not needed
container image override support unless batch sandboxes are required
batch-specific reasoning/provider flags outside the GPT/OpenAI scope
```

## Keep Subagents, But Simplify Delegation

Hermes implements subagents through the `delegate_task` tool in
`tools/delegate_tool.py`. The parent model calls `delegate_task`; the agent loop
intercepts it and constructs one or more child `AIAgent` instances.

Core flow:

```text
model emits tool_call: delegate_task
  -> AIAgent._invoke_tool()
  -> AIAgent._dispatch_delegate_task()
  -> delegate_task(parent_agent=self)
  -> _build_child_agent()
  -> child.run_conversation(goal)
  -> return child summary/results to parent
```

This is a useful core capability because it lets the agent split work while
keeping the parent's context small.

Keep:

```text
delegate_task tool schema
single-task delegation
batch delegation with max_concurrent_children
child AIAgent construction
child isolated conversation/task_id
child toolset restriction/inheritance
parent_session_id linkage
active_children tracking
interrupt propagation from parent to children
structured result aggregation
max_spawn_depth guard
```

Simplify or postpone:

```text
child provider/model override
ACP child-agent transport
credential pool leasing
TUI-specific subagent registry/details
complex cost rollup
large timeout diagnostic dumps
nested orchestrator mode in the first version
file-state sibling write reminders if not needed initially
```

The simplified version should still treat subagents as real child agents, not
as plain function calls:

```text
parent agent
  -> child AIAgent
       -> own messages
       -> own tool calls
       -> own task_id / terminal state
       -> final summary returned to parent
```

Do not mix the child agent's full transcript into the parent context. The value
of delegation is that intermediate exploration stays isolated and only the
summary/results return to the parent.

## Keep Cron Scheduling, Narrow Delivery Platforms

The `cron/` directory implements Hermes's internal scheduled-task system. It is
not the system `crontab`; it is an application-level scheduler normally ticked
by the long-running gateway daemon.

Keep this capability if the product needs scheduled reports, monitoring,
reminders, recurring analysis, or proactive agent tasks.

Core responsibilities:

```text
store jobs in ~/.hermes/cron/jobs.json
parse schedules
find due jobs
run due jobs
create short-lived AIAgent for each run
save output to ~/.hermes/cron/output/
advance next_run for recurring jobs
record cron sessions in SessionDB
apply timeout / inactivity protection
optionally deliver output to a messaging platform
```

Keep:

```text
cron/jobs.py
  schedule parsing
  create/list/update/pause/resume/remove jobs
  get_due_jobs()
  advance_next_run()
  save_job_output()

cron/scheduler.py
  tick()
  file lock to prevent overlapping ticks
  run_job()
  cron-specific AIAgent setup
  skill injection for cron jobs
  output wrapping
  timeout / interrupt handling
```

Simplify:

```text
delivery platform list
home-channel env var matrix
legacy delivery aliases
platform-specific delivery quirks
```

For a Feishu/Enterprise WeChat focused product, keep only:

```text
feishu
wecom
local output
```

The simplified flow should be:

```text
gateway daemon
  -> every N seconds call cron.scheduler.tick()
  -> tick locks ~/.hermes/cron/.tick.lock
  -> due jobs run AIAgent with platform="cron"
  -> output saved locally
  -> output delivered to Feishu/WeCom if configured
```

Cron jobs should remain isolated from normal chat memory. Keep `skip_memory=True`
for cron agents unless there is a deliberate product decision to let scheduled
tasks write long-term memory.

The simplification target is delivery breadth, not the scheduler itself.

## Remove ACP Adapter Unless IDE Integration Is Required

The `acp_adapter/` directory exposes Hermes through ACP, the Agent Client
Protocol. Its purpose is editor/IDE integration: Zed, VS Code, JetBrains, or
other ACP clients can start `hermes-acp`, create sessions, send prompts, receive
tool progress, request approvals, and resume previous conversations.

It is an external protocol adapter, not core agent logic.

Current shape:

```text
IDE / ACP client
  -> hermes-acp / hermes acp
  -> acp_adapter.entry.main()
  -> acp.run_agent(HermesACPAgent)
  -> acp_adapter.server.HermesACPAgent
  -> acp_adapter.session.SessionManager
  -> AIAgent.run_conversation()
```

File responsibilities:

```text
acp_adapter/entry.py
  command entry
  load ~/.hermes/.env
  keep stdout reserved for ACP JSON-RPC
  write logs to stderr
  start HermesACPAgent

acp_adapter/server.py
  ACP protocol methods
  initialize/new_session/load_session/resume_session/prompt/cancel
  model selector
  slash command handling
  session MCP registration

acp_adapter/session.py
  map ACP session ids to AIAgent instances
  persist session state through SessionDB
  restore/list/fork sessions
  bind editor cwd to tool task_id

acp_adapter/events.py
  translate Hermes callbacks into ACP events

acp_adapter/permissions.py
  translate Hermes approval requests into ACP permission requests

acp_adapter/tools.py
  translate Hermes tool calls/results into ACP structured tool display
```

For the simplified Hermes, this layer can be removed if the product does not
need IDE/editor integration. The core runtime still works through:

```text
CLI
gateway
cron
batch runner if needed later
```

If IDE integration is required later, reintroduce it as a thin adapter around
the already-stable core:

```text
initialize
new_session
load_session / resume_session
prompt
cancel
basic tool progress events
approval callback
session persistence
cwd binding
```

Postpone or remove in the first simplified version:

```text
multi-provider model selector
rich editor-specific slash command set
fork session support
ACP child-agent transport
per-session MCP dynamic registration
complex structured rendering for every tool kind
client-specific compatibility patches
```

The simplification target is the IDE protocol surface. Do not move ACP-specific
logic into `AIAgent`, `model_tools.py`, or the core tool registry.

## Keep MCP Client, Remove MCP Server

Hermes has two different MCP directions. They should not be treated as the same
capability.

Keep:

```text
tools/mcp_tool.py
  Hermes as MCP client
  connect to external MCP servers
  discover external tools
  register those tools into the Hermes tool registry
  let AIAgent call them like built-in tools
```

Remove from the simplified version:

```text
mcp_serve.py
  Hermes as MCP server
  expose Hermes conversations/messages/channels to other MCP clients
  let other agents read sessions, poll events, send messages, and handle approvals
```

This split matters for Feishu and Enterprise WeChat.

Feishu/WeCom support does not require `mcp_serve.py`:

```text
current agent -> send message to Feishu/WeCom
  gateway/send_message_tool/platform adapter

Feishu/WeCom user -> current agent
  gateway platform adapter receives events

cron job -> push result to Feishu/WeCom
  cron scheduler + gateway/send_message_tool/platform adapter
```

MCP Server is only needed for agent-to-agent or external-client control:

```text
other agent / Codex / Claude / Cursor
  -> MCP client
  -> mcp_serve.py
  -> Hermes sessions/messages/channels
  -> Feishu/WeCom send/read bridge
```

That is not part of the simplified product goal. Do not make Feishu or WeCom
depend on MCP Server.

Remove or postpone:

```text
hermes mcp serve
mcp_serve.py
conversations_list
conversation_get
messages_read
attachments_fetch
events_poll
events_wait
messages_send as MCP tool
channels_list as MCP tool
permissions_list_open
permissions_respond
EventBridge polling SessionDB for MCP clients
OpenClaw-compatible MCP channel bridge surface
```

Keep the MCP client side because it expands the agent's tool ecosystem:

```text
hermes mcp add/remove/list/test/login/configure
mcp_servers config in config.yaml
MCP stdio and HTTP transports
MCP tool discovery
MCP schema normalization
MCP toolset registration
MCP auth/OAuth if needed by selected servers
MCP timeout/reconnect/circuit-breaker behavior
```

The simplification target is MCP Server exposure, not MCP tool integration.

## Keep The TypeScript TUI Architecture

The `ui-tui/` and `tui_gateway/` directories are one terminal product surface,
not two separate UIs.

Architecture:

```text
hermes --tui
  -> hermes_cli/main.py launches Node
  -> ui-tui/src/entry.tsx
  -> GatewayClient spawns python -m tui_gateway.entry
  -> stdio JSON-RPC
  -> tui_gateway calls AIAgent/tools/sessions
```

Responsibilities:

```text
ui-tui/
  TypeScript terminal UI
  Ink/React-style components
  transcript rendering
  composer/input handling
  keyboard and mouse interactions
  status/progress panels
  session picker
  local UI state

tui_gateway/
  Python backend for the TUI
  load env/config
  create/resume sessions
  construct AIAgent
  call run_conversation()
  stream message/tool/session events back to the TUI
  handle approvals, interrupts, steer, compress, slash commands
```

Communication boundary:

```text
Node -> Python stdin
  JSON-RPC requests such as prompt.submit, session.resume, approval.respond

Python stdout -> Node
  JSON-RPC responses and events

Python stderr -> Node
  logs/activity lines
```

This split should be kept. The complexity is mostly product interaction
complexity, not unnecessary provider/platform compatibility. The TypeScript TUI
is a distinctive user-facing capability, while `tui_gateway` keeps Python-owned
agent runtime logic out of the UI process.

Do not simplify in the first pass:

```text
ui-tui / tui_gateway two-process architecture
stdio JSON-RPC transport
transcript
composer
tool progress display
session create/list/resume/history
prompt.submit
session.interrupt
approval.respond
slash command fallback
Python AIAgent backend
```

Keep `cli.py` as the stable fallback terminal entry. The TUI depends on Node,
the Ink runtime, terminal compatibility, and the JSON-RPC bridge; the classic
Python CLI is still useful for debugging and degraded environments.

Possible later trimming, only if product scope demands it:

```text
voice mode
dashboard PTY sidecar/websocket integration
browser.manage special panel
rollback UI
spawn_tree visualization
perf pane / heapdump monitoring
excessive debug-only events
large management panels that duplicate CLI subcommands
```

The simplification target is not the TUI architecture. If trimming is needed
later, trim peripheral features while preserving the UI/runtime boundary.

## Simplify Bundled Skills By Default Scope

Hermes ships many skills under two directories:

```text
skills/
  built-in skills available by default in the repository

optional-skills/
  heavier or more niche skills used as installable/reference skills
```

For the simplified Hermes, keep the skill system itself, but reduce the default
skill set. The target is not to remove the skill mechanism; the target is to
avoid shipping a broad marketplace as the first product surface.

Core rule:

```text
Keep skills that improve general agent work.
Move vertical, entertainment, platform-specific, and heavy ML/creative skills
to optional/reference status.
```

### P0: Keep By Default

These support general coding, planning, debugging, GitHub work, MCP client
integration, documents, memory-adjacent notes, and agent QA.

```text
native-mcp
  MCP client tools

plan
  planning mode

writing-plans
  implementation plans

systematic-debugging
  root-cause debugging

test-driven-development
  TDD workflow

requesting-code-review
  pre-commit/code review

subagent-driven-development
  delegate_task/subagent workflow

spike
  throwaway experiments

hermes-agent-skill-authoring
  author SKILL.md files

codex
  delegate coding to Codex

codebase-inspection
  codebase statistics/inspection

github-auth
  GitHub authentication

github-pr-workflow
  PR lifecycle

github-code-review
  PR review

github-issues
  issue management

github-repo-management
  repository management

webhook-subscriptions
  event-driven agent runs

ocr-and-documents
  OCR and document extraction

obsidian
  local note search/write

dogfood
  product QA and bug reports
```

### P1: Useful, But Optional In The First Cut

These are broadly useful, but not required for the core simplified agent.

```text
arxiv
  paper search

research-paper-writing
  academic paper drafting

blogwatcher
  RSS/blog monitoring

llm-wiki
  local markdown knowledge base

youtube-content
  YouTube transcript summaries

google-workspace
  Gmail/Calendar/Drive/Docs/Sheets

notion
  Notion API

linear
  Linear issues/projects

airtable
  Airtable records

powerpoint
  PPT creation/editing

nano-pdf
  PDF editing

maps
  geocoding/routes/POI

himalaya
  IMAP/SMTP email

humanizer
  humanize text

architecture-diagram
  architecture diagrams

excalidraw
  hand-drawn diagrams

sketch
  quick HTML mockups

popular-web-designs
  web design references

jupyter-live-kernel
  live Jupyter execution
```

### P2: Scenario Skills, Do Not Enable By Default

These are useful only for specific users/platforms.

```text
claude-code
opencode
hermes-agent
apple-notes
apple-reminders
findmy
imessage
spotify
gif-search
songsee
heartmula
xurl
openhue
yuanbao
polymarket
minecraft-modpack-server
pokemon-player
```

### P3: Move To Optional/Reference

These are heavy, creative, ML-specific, or risky/niche enough that they should
not be in the default simplified bundle.

```text
claude-design
comfyui
p5js
manim-video
pixel-art
ascii-art
ascii-video
baoyu-comic
baoyu-infographic
creative-ideation
design-md
pretext
songwriting-and-ai-music
touchdesigner-mcp
huggingface-hub
llama-cpp
vllm
outlines
dspy
axolotl
trl-fine-tuning
unsloth
lm-evaluation-harness
weights-and-biases
audiocraft-audio-generation
segment-anything-model
obliteratus
godmode
```

### optional-skills Policy

Keep `optional-skills/` as a reference/installable library, not as the default
runtime surface.

Potential future picks:

```text
fastmcp
  build MCP servers

docker-management
  Docker operations

1password
  secret management

duckduckgo-search
  free web search

qmd
  local hybrid retrieval

scrapling
  web scraping

chroma / faiss / qdrant
  vector databases

instructor / guidance
  structured LLM output

whisper
  speech-to-text

modal / lambda-labs
  GPU cloud

peft / accelerate / pytorch-fsdp
  model training

telephony
  phone/SMS

siyuan
  SiYuan notes

page-agent
  in-page web agent
```

First simplified target:

```text
default skills = P0 only
optional install = P1
reference/marketplace = P2 + P3 + optional-skills
```

This keeps the skill loading/injection mechanism intact while reducing the
default cognitive load from dozens of unrelated capabilities to a focused agent
workbench.
