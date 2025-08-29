# Ollama Turbo CLI

A production-ready CLI application for interacting with gpt-oss:120b model through Ollama Turbo cloud service, featuring advanced tool calling capabilities and streaming responses.

## Features

- 🚀 **Ollama Turbo Integration**: Connect to gpt-oss:120b on datacenter-grade hardware
- 🔧 **Advanced Tool Calling**: Weather, calculator, file operations, system info
- 📡 **Streaming Responses**: Real-time response streaming with tool execution
- 🧠 **Multi-Round Tool Chaining (Streaming)**: Iteratively call multiple tools in one turn; enabled by default
- 💬 **Interactive Mode**: Continuous conversation with history management
- 🛡️ **Error Handling**: Robust retry logic and graceful error recovery
- ⚙️ **Configurable**: Environment variables and command-line options

## Prompting Strategy (Centralized)

Prompts are centralized in `src/prompt_manager.py` to ensure consistent guidance across the app:

- Initial system directive steers tool selection, chaining, and synthesis.
- Post-tool reprompt instructs the model to synthesize a final textual answer.
- Mem0 context is injected as a single, hygienic system block each turn.
  - The tool selection guide is auto-generated from discovered plugins (from `src/plugins/` and `tools/`), so newly added tools like `web_research` appear automatically.

Environment flags:

- `PROMPT_VERSION` (default: `v1`)
- `PROMPT_INCLUDE_TOOL_GUIDE` (default: `1`) — include a concise tool selection guide in the system prompt.
- `PROMPT_FEWSHOTS` (default: `0`) — append few-shot examples to nudge tool use.

## Plugin Architecture (Dynamic Tools)

Tools are now loaded dynamically via a plugin system.

- Built-in plugins live in `src/plugins/` and wrap the legacy implementations in `src/tools.py` for backward compatibility.
- Third-party plugins can be dropped into the top-level `tools/` directory without modifying core code.
- Plugins are validated using JSON Schema and auto-registered at runtime.
- Optional env: `OLLAMA_TOOLS_DIR` can point to additional plugin directories (supports multiple paths separated by your OS path separator).

### Plugin Contract

Each plugin is a Python module that must define:

- `TOOL_SCHEMA`: OpenAI-style function tool schema with fields `type: "function"`, `function.name`, `function.description`, and `function.parameters` (an object schema or `true` for no-arg tools).
- An implementation callable, provided as one of:
  - `TOOL_IMPLEMENTATION` (callable), or
  - a function named the same as `function.name`, or
  - a function named `execute`.

Minimal example placed at `tools/hello.py`:

```python
TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "hello",
        "description": "Return a friendly greeting message.",
        "parameters": {
            "type": "object",
            "properties": {"name": {"type": "string", "default": "world"}},
            "required": []
        }
    }
}

def execute(name: str = "world") -> str:
    return f"Hello, {name}! 👋"
```

### Backward Compatibility

- Existing tools in `src/tools.py` remain as-is and are exposed via built-in plugins in `src/plugins/`.
- The client now imports dynamic aggregates from `src/plugin_loader.py` to execute tools discovered at runtime.

### Validation and Safety

- Schemas and call arguments are validated using `jsonschema`. Ensure `jsonschema` is installed (see `requirements.txt`).
- Duplicate tool names are skipped with a warning; the first loaded plugin wins.

## Quick Start

### 1. Installation

```bash
git clone <repository-url>
cd ollama-turbo-cli
pip install -r requirements.txt
```

### 2. Get API Key

1. Visit [ollama.com/turbo](https://ollama.com/turbo)
2. Sign up for Ollama Turbo ($20/month)
3. Get your API key from [ollama.com/settings/keys](https://ollama.com/settings/keys)

### 3. Setup Environment

```bash
cp .env.example .env
# Edit .env and add your API key
```

### 4. Run

```bash
# Interactive mode
python -m src.cli --api-key YOUR_API_KEY

# Single message
python -m src.cli --api-key YOUR_API_KEY --message "What's 15 * 8?"

# With streaming
python -m src.cli --message "Weather in London" --stream
```

## Available Tools

### 🌤️ Weather Service

Get current weather for major cities worldwide.

```text
"What's the weather in Tokyo?"
```

### 🧮 Mathematical Calculator

Perform complex mathematical calculations with support for functions.

```text
"Calculate sin(pi/2) + sqrt(16)"
```

### 📁 File Operations

List files and directories with filtering options.

```text
"List all Python files in the current directory"
```

### 💻 System Information

Get comprehensive system information including CPU, memory, and disk usage.

```text
"Show me system information"
```

### 🔎 DuckDuckGo Search

Keyless web search for sources or quick facts (title, URL, snippet).

When to use: need external info discovery or recent data.
Tips: use focused queries (keywords or quoted phrase), keep `max_results` small (1–5).

### 📚 Wikipedia Search

Keyless search of canonical topics via MediaWiki API (title, URL, snippet).

When to use: factual background and canonical entity pages.
Tips: focused term/entity; use `web_fetch` on a returned URL for details.

### 🌐 Web Fetch

Fetch specific HTTPS URLs through an allowlist proxy with SSRF protections.

When to use: read a specific page/API without credentials.
Tips: prefer HTTPS, small timeouts, minimal bytes. Only compact summaries are injected and may be truncated.

### ✅ Reliable Chat (Reliability API)

Grounded and validated answers via the internal Reliability API. Supports non-streaming and SSE streaming with a trailing summary event.

When to use: you want citations, validator checks, or consensus heuristics.
Flags: `ground`, `k`, `cite`, `check` (`off|warn|enforce`), `consensus` (int|bool; sent as integer), `engine`, `eval_corpus`.

Notes:

- If `consensus` is provided as `true` and `k` is unset, the client sends `consensus=3` by default.
- `k` controls retrieval breadth; `consensus` controls the number of model votes. They are coupled only for defaults.
- Fail-closed: if `ground=true` and `cite=true` but no sources resolve, the client returns a clear refusal and sets `summary.status = "no_docs"` with `grounded=false` and empty `citations`.
- Streaming `summary` contract: `status` ∈ {`ok`,`no_docs`,`timeout`,`http_error`}; `provenance` is `retrieval` when grounded else `none`.

Performance tips:

- When responses are slow: omit `consensus`, set `check="off"`, and leave `k` unset to minimize load.
- For complex queries, consider running `web_research` first to fetch sources, then synthesize with citations using your preferred workflow.

Reliable Chat plugin is currently disabled and unavailable.

Reliable Chat streaming aggregation is disabled.

#### Windows-safe curl (SSE) — quick smoke

```powershell
$env:BASE = "http://127.0.0.1:8000"

# Happy path (expect tokens + trailing summary with grounded:true when sources found)
curl.exe --no-buffer $env:BASE/v1/chat/stream -H "content-type: application/json" `
  -d '{"message":"When did Pearl Street Mall open?","ground":true,"cite":true,"k":4}'

# Fail-closed path (expect refusal text + summary.status=no_docs, grounded:false, citations:[])
curl.exe --no-buffer $env:BASE/v1/chat/stream -H "content-type: application/json" `
  -d '{"message":"who won the 2047 denver mayor race","ground":true,"cite":true}'
```

Notes:

- By default, auto-heuristics infer flags from the message (citations → ground+cite, verify → check=warn, consensus → consensus+k).
- Configure base URL via `RELIABILITY_API_BASE` (defaults to `http://127.0.0.1:8000`).
- Requires `API_KEY` if the FastAPI server enforces it; upstream `OLLAMA_API_KEY` is forwarded when present.

### 🔬 Web Research

Multi-hop web research with citations. Orchestrates Plan→Search→Fetch→Extract→Chunk→Rerank→Cite→Cache.

When to use: you need up-to-date facts from multiple sources and deterministic citations. Prefer over `web_fetch` when synthesizing across pages is needed.
Parameters: `top_k` (breadth, default 5), `site_include`/`site_exclude` (scope), `freshness_days` (recency), `force_refresh` (bypass caches).
Returns: compact JSON with results, quotes, page-mapped citations (for PDFs), and archive URLs.

## Usage Examples

### Interactive Mode

```bash
python -m src.cli --api-key sk-your-key-here

👤 You: What's the weather in London and calculate 25 * 4?

🔧 Processing tool calls...
   1. Executing get_current_weather(city=London, unit=celsius)
      ✅ Result: Weather in London: Partly cloudy, 15°C, Humidity: 65%, Wind: 12 mph
   2. Executing calculate_math(expression=25 * 4)
      ✅ Result: Result: 25 * 4 = 100

🤖 Final response: Based on the results, London currently has partly cloudy weather at 15°C, and 25 multiplied by 4 equals 100.
```

### Single Message Mode

```bash
python -m src.cli --api-key sk-your-key --message "List files in /home/user/documents"
```

### Streaming Mode

```bash
python -m src.cli --message "Explain quantum computing" --stream
```

In streaming mode, the assistant can perform multiple tool rounds in a single turn and then synthesize a final textual answer. By default, up to 6 tool-call rounds are allowed before finalization.

## HTTP API (v1)

The project includes a versioned FastAPI server for programmatic access.

### Start the server

```bash
uvicorn src.api.app:app --reload
```

Feature flag: set `API_ENABLED=0` to disable. If `API_KEY` is set, clients must send `X-API-Key`.

### Endpoints

- `GET /health` — root health (FastAPI app)
- `GET /v1/health` — basic health
- `GET /v1/live` — liveness probe
- `GET /v1/ready` — readiness probe
- `POST /v1/chat` — non-streaming chat
- `POST /v1/chat/stream` — SSE streaming chat

### Headers

- `X-API-Key: <key>` — required if `API_KEY` is configured
- `Idempotency-Key: <uuid>` — optional; reserved for future de-duplication

### Request model (POST /v1/chat and /v1/chat/stream)

```json
{
  "message": "Hello",
  "options": {
    "tool_results_format": "string|object"
  }
}
```

`tool_results_format` controls the shape of tool results in responses. Default is `string` for backward compatibility; set `TOOL_RESULTS_FORMAT=object` or pass in `options` to get structured objects.

Reliability fields (all optional):

- `ground` (bool): Enable retrieval + grounding context injection
- `k` (int): General breadth parameter (retrieval top-k and/or consensus runs)
- `cite` (bool): Ask the model to include inline citations when grounded
- `check` ("off"|"warn"|"enforce"): Validator mode
- `consensus` (bool): Enable multi-run consensus (streaming returns trace-only summary)
- `engine` (string): Backend engine alias or URL
- `eval_corpus` (string): Eval corpus identifier for micro-eval

### Responses

Non-streaming (`/v1/chat`):

```json
{
  "content": "Hello there!",
  "tool_results": [
    "calc: 42"
  ]
}
```

With `tool_results_format=object`, `tool_results` becomes a list of objects:

```json
{
  "content": "Hello there!",
  "tool_results": [
    {"tool":"calc","status":"ok","content":42,"metadata":{"args":{"x":40,"y":2}},"error":null}
  ]
}
```

Streaming (`/v1/chat/stream`): Server-Sent Events (SSE). Events are sent as `data: {json}` lines.

- Token chunks: `{ "type": "token", "content": "..." }`
- Final event: `{ "type": "final", "content": "...", "tool_results": [...] }`

Tool-calls or stream errors are silently finalized via a non-streaming fallback, and only a final event is emitted (no tokens), matching CLI behavior.

### Web Research Example

```bash
python -m src.cli --message "Research: What are the latest results on Llama 3.2 vision benchmarks? Include citations."

🔧 Processing tool calls...
   1. Executing web_research(query="latest Llama 3.2 vision benchmarks", top_k=5, freshness_days=365)
      ✅ Result: { "results": [ { "title": "...", "url": "https://...", "quote": "...", "citation": { "page": 3, "line": 120 } } ], "archives": ["https://archive.is/..." ] }

🤖 Final response: Summary with quotes and numbered citations.
```

## Configuration

### Environment Variables

- `OLLAMA_API_KEY`: Your Ollama API key
- `OLLAMA_MODEL`: Model name (default: gpt-oss:120b)
- `OLLAMA_HOST`: API endpoint (default: <https://ollama.com>)
- `RELIABILITY_API_BASE`: Reliability router base URL (default: <http://127.0.0.1:8000>)
- `MAX_CONVERSATION_HISTORY`: Maximum messages to keep (default: 10)
- `STREAM_BY_DEFAULT`: Enable streaming by default (default: false)
- `LOG_LEVEL`: Logging level (default: INFO)
- `REASONING`: Default reasoning effort (low|medium|high, default: `high`)
- `PROMPT_VERSION`: Prompt pack version (default: `v1`)
- `PROMPT_INCLUDE_TOOL_GUIDE`: Include tool selection guide in system prompt (default: `1`)
- `PROMPT_FEWSHOTS`: Append few-shot guidance to the system prompt (default: `0`)
- `MULTI_ROUND_TOOLS`: Enable multi-round tool chaining (default: `true`). In non-streaming mode, the client always finalizes with a textual answer after the first tool round for compatibility.
- `TOOL_MAX_ROUNDS`: Maximum number of tool-call rounds in streaming mode before finalizing (default: `6`).

#### Mem0 (optional)

- `MEM0_API_KEY`: Mem0 API key (enables long-term memory)
- `MEM0_USER_ID`: Memory namespace user id (default: "Braden")
- `MEM0_AGENT_ID`: Optional agent id for tagging
- `MEM0_APP_ID`: Optional app id for tagging
- `MEM0_ORG_ID`: Optional organization id
- `MEM0_PROJECT_ID`: Optional project id
- `MEM0_VERSION`: Preferred Mem0 API version (default: "v2"). The client prefers v2 and gracefully falls back if the SDK doesn't accept the `version` parameter.

Mem0 runtime knobs (advanced):

- `MEM0_ENABLED`: Enable/disable Mem0 (default: `1`)
- `MEM0_DEBUG`: Enable Mem0 debug logs (default: `0`)
- `MEM0_MAX_HITS`: Max memories injected per turn (default: `3`)
- `MEM0_SEARCH_TIMEOUT_MS`: Time-boxed search timeout in ms (default: `200`)
- `MEM0_TIMEOUT_CONNECT_MS`: HTTP connect timeout in ms (default: `1000`)
- `MEM0_TIMEOUT_READ_MS`: HTTP read timeout in ms (default: `2000`)
- `MEM0_ADD_QUEUE_MAX`: Background add queue size (default: `256`)
- `MEM0_BREAKER_THRESHOLD`: Consecutive failures to trip circuit breaker (default: `3`)
- `MEM0_BREAKER_COOLDOWN_MS`: Breaker cooldown in ms (default: `60000`)
- `MEM0_SHUTDOWN_FLUSH_MS`: Shutdown flush timeout in ms (default: `3000`)
- `MEM0_SEARCH_WORKERS`: Tiny thread pool size for Mem0 searches (default: `2`)

### Command Line Options

```bash
--api-key        Ollama API key
--model          Model name (default: gpt-oss:120b)
--message        Single message mode
--stream         Enable streaming
--reasoning      Reasoning effort: low|medium|high (default: high)
--no-tools       Disable tool calling
--log-level      Set logging level
--version        Show version
--help           Show help
```

## Interactive Commands

While in interactive mode:

- `quit` or `exit`: Exit the application
- `clear`: Clear conversation history
- `history`: Show conversation history
- `/mem ...`: Manage memories (list|search|add|get|update|delete|clear|link|export|import)
- `Ctrl+C`: Exit gracefully

## Mem0 Memory System

Mem0 is integrated as the long-term memory store. It is optional and enabled when `MEM0_API_KEY` is present in your `.env`.

- **Initialization**: Loaded silently from environment variables. No runtime prompts.
- **Status banner**: Interactive mode shows `Mem0: enabled (user: <id>)` or `Mem0: disabled (no key)`.
- **Injection hygiene**: At most one "Relevant information:" system block is injected per turn before your message; previous injection blocks are removed each turn. Local conversation history is capped at 10 turns.
- **Natural language intents**: You can type phrases like:
  - "remember ...", "forget ...", "update X to Y", "list/show memories", "link `<id1>` `<id2>`", "search memories for ...", or "what do you know about me".
- **/mem commands**: `list`, `search <q>`, `add <text>`, `get <id>`, `update <id> <text>`, `delete <id>`, `clear`, `link <id1> <id2>`, `export [path.json]`, `import <path.json>`.
- **Export/Import**: Exports to `mem0_export_*.json` including `id`, `memory`, `metadata`, `created_at`, `updated_at`. Git ignores these files by default.
- **Resilience**: All Mem0 operations are wrapped in try/except; failures never block chat. A one-time notice is shown if Mem0 is unavailable; details are logged at DEBUG level.
- **API versioning**: Prefers Mem0 v2 endpoints (passes `version="v2"`), and falls back automatically if the SDK does not support the `version` kwarg. Configure preferred version via `MEM0_VERSION`.
- **Filters**: Minimal filters are used for recall (`user_id` only) for list/search/export until recall is proven.

## Secure Execution & Web Access

- **Shell tool is disabled by default**: Commands run only inside a locked-down sandbox when enabled. Set `SHELL_TOOL_ALLOW=1` and keep an explicit `SHELL_TOOL_ALLOWLIST` (prefix match) for safe commands like `git status`, `ls`, `cat`.

- **Confirmation prompts**: In TTY, `execute_shell` shows a one-line preview and requires confirmation unless `SHELL_TOOL_CONFIRM=0`.

- **Sandbox guarantees**: CPU/mem/pids/disk/time limits, read-only project mount at `/project`, tmpfs workspace, no host env (only `env_vars` pass-through). Windows requires Docker Desktop/WSL2. If Docker is unavailable, execution fails closed with a clear message.

- **Web access is allowlist-only**: The `web_fetch` tool goes through a controlled client with host allowlist and SSRF protections. HTTPS-only by default (`SANDBOX_ALLOW_HTTP=0`). Add domains to `SANDBOX_NET_ALLOW` (supports wildcards like `*.wikipedia.org`).

- **Summarization before injection**: Tool outputs are summarized and capped (`TOOL_CONTEXT_MAX_CHARS`, default 4000). Full logs and payloads are kept under `.sandbox/sessions/` and `.sandbox/cache/` and are not shared with the model.

- **Mem0 safety**: Outputs from `execute_shell` and sensitive `web_fetch` responses are flagged and not persisted to Mem0.

- **Caching & rate limits**: On-host HTTP cache (TTL 10m, LRU ~200MB) accelerates repeat fetches. Per-host rate-limits prevent abuse.

### Enable a one-off command safely

1. Export envs:
   - `SHELL_TOOL_ALLOW=1`
   - Optional: extend `SHELL_TOOL_ALLOWLIST` with a safe prefix (e.g., `git show`)

2. Run your prompt. When asked, confirm the preview.

### Add a domain to the allowlist

- Set `SANDBOX_NET_ALLOW` to include your domain(s), comma-separated, e.g.:

```bash
SANDBOX_NET_ALLOW=api.github.com,*.wikipedia.org,example.com
```

- Keep HTTPS-only unless absolutely necessary (`SANDBOX_ALLOW_HTTP=0`).

### Environment knobs

See `.env.example` for all variables, including:

- `SHELL_TOOL_ALLOW`, `SHELL_TOOL_CONFIRM`, `SHELL_TOOL_ALLOWLIST`, `SHELL_TOOL_MAX_OUTPUT`, `SHELL_TOOL_ROOT`
- `SANDBOX_NET`, `SANDBOX_NET_ALLOW`, `SANDBOX_ALLOW_HTTP`, `SANDBOX_BLOCK_PRIVATE_IPS`, `SANDBOX_MAX_DOWNLOAD_MB`, `SANDBOX_RATE_PER_HOST`, `SANDBOX_HTTP_CACHE_TTL_S`, `SANDBOX_HTTP_CACHE_MB`
- `TOOL_CONTEXT_MAX_CHARS`

## Error Handling

The application includes robust error handling:

- **Network Issues**: Automatic retry with exponential backoff
- **API Errors**: Clear error messages and recovery
- **Tool Failures**: Graceful degradation with error reporting
- **Invalid Input**: Input validation and helpful suggestions
- **Streaming fallback**: Streaming errors are not shown in CLI; the client silently falls back to non-streaming and logs details at DEBUG level only.

## Requirements

- Python 3.8+
- Ollama Turbo subscription ($20/month)
- Internet connection for API calls
- 4GB+ RAM recommended

## License

MIT License - see LICENSE file for details.

## Support

For issues and questions:

1. Check the troubleshooting section
2. Review error logs with `--log-level DEBUG`
3. Verify API key and subscription status
4. Open an issue on GitHub

## Troubleshooting

### Common Issues

#### API Key Errors

- Ensure your API key is correctly set in `.env` or via `--api-key`
- Verify the key is active at [ollama.com/settings/keys](https://ollama.com/settings/keys)

#### Connection Issues

- Check your internet connection
- Verify Ollama Turbo service status
- Try with `--log-level DEBUG` for detailed error messages

#### Tool Execution Failures

- Some tools require specific permissions (e.g., file operations)
- System info tool requires `psutil` package
- Weather data is available for major cities only

#### Performance

- Streaming mode provides better perceived performance
- Tool calls may add latency for complex operations
- Consider adjusting `MAX_CONVERSATION_HISTORY` for memory usage

## Development

### Project Structure

```text
ollama-turbo-cli/
├── src/
│   ├── __init__.py         # Package initialization
│   ├── cli.py              # Main CLI application
│   ├── client.py           # Ollama Turbo client wrapper
│   ├── prompt_manager.py   # Centralized prompt construction and configuration
│   ├── tools.py            # Legacy tool implementations (wrapped by plugins)
│   ├── plugin_loader.py    # Dynamic plugin discovery and validation
│   └── plugins/            # Built-in tool plugins (wrapping legacy tools)
│   └── utils.py            # Utility functions
├── tools/                  # Third-party plugins (auto-discovered)
├── requirements.txt        # Python dependencies
├── setup.py               # Package setup configuration
├── README.md              # This file
├── .env.example           # Environment variables template
└── LICENSE                # MIT license
```

### Web Research Pipeline (Internals)

- **Entry point tool**: `src/plugins/web_research.py` calls `src/web/pipeline.py:run_research()`.
- **Modules under** `src/web/` are internal (not tools). They implement the pipeline stages and safety:
  - Plan → Search → Fetch → Extract → Chunk → Rerank → Cite → Cache
  - Fetch enforces robots.txt crawl-delay and per-host rate/concurrency guards.
  - Extract handles HTML and PDF; citations include exact quotes and PDF page mapping when available.
  - Caching and content-hash dedupe keep repeated runs fast; `force_refresh=true` bypasses caches.
- **When to use**: multi-source questions needing fresh facts and deterministic citations. Prefer over `web_fetch` when synthesizing across multiple pages.
- **Parameters** (tool): `top_k`, `site_include`, `site_exclude`, `freshness_days`, `force_refresh`.
- **Output**: compact JSON with results, citations, and archive URLs suitable for downstream summarization.

### Running Tests

```bash
# Test weather tool
python -m src.cli --message "What's the weather in Paris?"

# Test calculator
python -m src.cli --message "Calculate sqrt(144) + sin(pi/2)"

# Test file operations
python -m src.cli --message "List Python files in the current directory"

# Test system info
python -m src.cli --message "Show system information"

# Test multiple tools
python -m src.cli --message "Weather in London and calculate 15 * 8"

# Run unit tests (includes plugin system tests)
pytest -q
```

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Version History

### v1.1.0 (Current)

- Reliability mode: grounding, citations, validator (off|warn|enforce), consensus (k)
- Streaming reliability with trailing summary (grounded, citations, validator, consensus)
- Heuristic auto-flags based on message intent; CLI/API flags plumbed end-to-end
- Hardened plugin validation and None-stripping for tool args
- Version bump and README/documentation updates

### v1.0.0

- Initial release with gpt-oss:120b support
- Four built-in tools: weather, calculator, files, system
- Streaming and non-streaming modes
- Interactive and single-message modes
- Comprehensive error handling and retry logic
