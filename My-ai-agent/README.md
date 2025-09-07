Ollama Turbo CLI

A production-ready CLI for talking to gpt-oss:120b and DeepSeek V3.1:671B via the Ollama Turbo cloud. It supports real tool calling, streaming, and a tidy developer experience.

Why you’ll like it

🚀 Turbo cloud: Point at Ollama Turbo and go (datacenter hardware, no local GPU needed).

🔧 Real tools: Weather, calculator, files, system info—plus a plugin system for your own.

📡 True streaming: Token-by-token output, with seamless tool loops mid-stream.

🧠 Multi-round tools: Chain tools in one turn (streaming), finalize with text.

🗂️ Interactive mode: History, /mem management, quick commands.

🛡️ Hardened client: Retries, idempotency keys, keep-alive pools, quiet streaming fallbacks.

⚙️ Config everywhere: Flags + env vars; sensible defaults.

How prompting works

All prompts live in src/prompt_manager.py so behavior stays consistent:

An initial system directive guides tool selection/chaining/synthesis.

After tools, a compact reprompt asks the model to synthesize a final answer.

Mem0 (optional) injects one hygienic system block per turn.

A short tool guide is auto-generated from discovered plugins.

Env toggles:

PROMPT_VERSION (default: v1)

PROMPT_INCLUDE_TOOL_GUIDE (default: 1)

PROMPT_FEWSHOTS (default: 0)

Plugins (dynamic tools)

Built-ins live in src/plugins/ (wrapping legacy src/tools.py for compatibility).

Drop third-party tools into top-level tools/—no core changes required.

Automatic discovery + JSON Schema validation.

OLLAMA_TOOLS_DIR may point to extra directories (os-path-separator separated).

Plugin contract (minimal)

Each plugin is a Python module with:

TOOL_SCHEMA: OpenAI-style function schema (type, function.name, function.description, function.parameters).

An implementation:

TOOL_IMPLEMENTATION or

a function named after function.name or

execute(...).

# tools/hello.py
TOOL_SCHEMA = {
  "type": "function",
  "function": {
    "name": "hello",
    "description": "Return a friendly greeting.",
    "parameters": {
      "type": "object",
      "properties": {"name": {"type": "string", "default": "world"}},
      "required": []
    }
  }
}

def execute(name: str = "world") -> str:
    return f"Hello, {name}! 👋"


Duplicates are skipped with a warning (first one wins).

Quick start
1) Install
git clone <repository-url>
cd ollama-turbo-cli
pip install -r requirements.txt

2) Get a key

Go to https://ollama.com/turbo

Subscribe (from $20/month)

Create an API key: https://ollama.com/settings/keys

3) Configure
cp .env.example .env
# Edit .env to include OLLAMA_API_KEY=...

4) Run
# Interactive
python -m src.cli --api-key YOUR_API_KEY

# One-off
python -m src.cli --api-key YOUR_API_KEY --message "What's 15 * 8?"

# Streaming
python -m src.cli --message "Weather in London" --stream

Built-in tools (high level)

🌤️ Weather — “What’s the weather in Tokyo?”

🧮 Calculator — “Calculate sin(pi/2) + sqrt(16)”

📁 Files — “List all Python files in the current directory”

💻 System info — “Show me system information”

🔎 DuckDuckGo search — quick web facts (no key)

📚 Wikipedia — canonical topic lookup

🌐 Web fetch — allowlisted HTTPS fetches (SSRF-hardened)

🔬 Web research — multi-hop search→fetch→extract→chunk→rerank→cite with JSON output & citations

Tip: prefer web_research for multi-source, up-to-date questions with citations.

Usage examples

Interactive

python -m src.cli --api-key sk-your-key-here


Single message

python -m src.cli --api-key sk-your-key --message "List files in /home/user/documents"


Streaming

python -m src.cli --message "Explain quantum computing" --stream


Streaming can perform multiple tool rounds (default cap: 6) then finalize.

HTTP API (v1)

Run a FastAPI server:

uvicorn src.api.app:app --reload


Feature flag: API_ENABLED=0 disables the server.

If API_KEY is set, clients must send X-API-Key.

Endpoints

GET /health, GET /v1/health, /v1/live, /v1/ready

POST /v1/chat — non-streaming

POST /v1/chat/stream — SSE streaming

POST /v1/session / DELETE /v1/session/{id}

GET /v1/models, GET /v1/tools — reflection

Headers

X-API-Key: <key> — if configured

Idempotency-Key: <uuid> — optional de-dup

Responses always include X-Request-Id.

Request model (chat/chat/stream)
{
  "message": "Hello",
  "options": {
    "tool_results_format": "string|object"
  }
}


With tool_results_format=object, tool results are structured objects; default is string.

Streaming shape (SSE)

Token chunk: { "type":"token", "content":"..." }

Final: { "type":"final", "content":"...", "tool_results":[...] }

Optional summary: { "type":"summary", "citations":[...], "consensus":{...}, "validator":{...} }

If a tool is detected mid-stream or a stream error occurs, the client silently finalizes via a non-streaming call and emits a final event only.

Sessions

In-memory history keyed by session_id (TTL via SESSION_TTL_SECONDS, default 3600).

Create → use in requests → delete to clear memory.

Windows PowerShell example included in the original README (works unchanged).

Configuration
Core env vars

OLLAMA_API_KEY, OLLAMA_MODEL (default: gpt-oss:120b), OLLAMA_HOST (default: https://ollama.com)

MAX_CONVERSATION_HISTORY (default: 10)

STREAM_BY_DEFAULT (default: false)

LOG_LEVEL (default: INFO)

REASONING (low|medium|high, default: high)

Prompt flags: PROMPT_VERSION, PROMPT_INCLUDE_TOOL_GUIDE, PROMPT_FEWSHOTS

Tools: MULTI_ROUND_TOOLS (default: true), TOOL_MAX_ROUNDS (default: 6)

Tool results: TOOL_RESULTS_FORMAT (string|object, default: string)

Mem0 (optional)

Keys/ids: MEM0_API_KEY, MEM0_USER_ID, MEM0_AGENT_ID, MEM0_APP_ID, MEM0_ORG_ID, MEM0_PROJECT_ID

Versioning: MEM0_VERSION (prefers "v2"; falls back if SDK ignores version)

Toggles/limits (sensible defaults):

MEM0_ENABLED, MEM0_DEBUG

MEM0_MAX_HITS

MEM0_SEARCH_TIMEOUT_MS, MEM0_TIMEOUT_CONNECT_MS, MEM0_TIMEOUT_READ_MS

MEM0_ADD_QUEUE_MAX, MEM0_SEARCH_WORKERS

MEM0_BREAKER_THRESHOLD, MEM0_BREAKER_COOLDOWN_MS, MEM0_SHUTDOWN_FLUSH_MS

Behavior: one Mem0 “Relevant information:” system block max per turn; non-blocking on failure; export/import supported.

Secure execution & web access

Shell tool is off by default. Enable with SHELL_TOOL_ALLOW=1 and maintain an allowlist (SHELL_TOOL_ALLOWLIST).

Confirmations (TTY): preview prompt unless SHELL_TOOL_CONFIRM=0.

Sandbox: CPU/mem/pids/disk/time caps; read-only project mount; tmpfs workspace; no host env. Windows requires Docker Desktop/WSL2.

Web fetch: HTTPS-only (by default), allowlist via SANDBOX_NET_ALLOW, SSRF-safe client, caching + per-host rate limits.

Summarize before injection: outputs are summarized and capped by TOOL_CONTEXT_MAX_CHARS (default 4000). Full logs live under .sandbox/.

Error handling

Network: retries with backoff.

API: normalized error envelopes.

Tools: graceful degradation, clear messages.

Streaming: failures don’t leak to stdout; the client silently finalizes non-streaming and logs at DEBUG.

API error envelope (normalized):

{
  "error": "Invalid or missing API key",
  "code": "401",
  "request_id": "UUID4",
  "idempotency_key": null
}

Requirements

Python 3.8+

Ollama Turbo subscription

Internet access

4 GB+ RAM recommended

Troubleshooting

API key: set in .env or --api-key; verify at Ollama settings.

Connectivity: check network & service status; try --log-level DEBUG.

Tools: some need extra deps/permissions (e.g., psutil for system info).

Performance: streaming improves perceived latency; multi-round tools add real latency.

Development
Project layout
ollama-turbo-cli/
├─ src/
│  ├─ cli.py                # CLI entry
│  ├─ client.py             # Thin orchestrator around services
│  ├─ prompt_manager.py     # Prompt packs
│  ├─ tools.py              # Legacy tool impls (wrapped by plugins)
│  ├─ plugin_loader.py      # Dynamic discovery/validation
│  ├─ orchestration/        # ChatTurnOrchestrator (std/stream mechanics)
│  ├─ streaming/            # Runner + standard delegates
│  ├─ reliability_integration/  # Facade for retrieval/validator/consensus wiring
│  ├─ plugins/              # Built-in plugins
│  └─ utils.py
├─ tools/                   # Third-party plugins (auto-discovered)
├─ requirements.txt
├─ setup.py
├─ .env.example
└─ LICENSE

Tracing (dev)

Enable show_trace=True to capture breadcrumbs (printed to stderr and available via client.trace):

request:standard:round=0, request:stream:start, request:stream:round=0

mem0:present:... (when Mem0 is on)

reprompt:after-tools, reprompt:after-tools:fallback

tools:used=N / tools:none

Traces reset at the start of each chat() turn and are not auto-cleared after printing.

Web research internals

Entry tool: src/plugins/web_research.py → src/web/pipeline.py:run_research()

Pipeline: Plan → Search → Fetch → Extract → Chunk → Rerank → Cite → Cache

PDF/HTML support, quote-level citations, robots.txt + per-host limits, caching, content-hash dedupe.

Params: top_k, site_include, site_exclude, freshness_days, force_refresh.

Output: compact JSON (results + citations + archives).

Tests
pytest -q


Includes plugin system tests and parity tests across standard/streaming, tool loops, fallback breadcrumbs, etc.

CLI options
--api-key        Ollama API key
--model          Model name (default: gpt-oss:120b)
--message        Single message mode
--stream         Enable streaming
--reasoning      low|medium|high (default: high)
--no-tools       Disable tools
--log-level      Set logging level
--version        Show version
--help           Show help


Interactive commands: quit/exit, clear, history, /mem ... (list/search/add/get/update/delete/clear/link/export/import). Ctrl+C exits gracefully.

Contributing

Please include tests for new functionality.

Version history
v1.1.0 (current)

Heuristic auto-flags; flags plumbed end-to-end

Hardened plugin validation & arg cleanup

Docs/readme refresh

v1.0.0

Initial release (gpt-oss:120b)

Weather, calculator, files, system tools

Streaming + non-streaming, interactive + single message

Robust retries and error handling