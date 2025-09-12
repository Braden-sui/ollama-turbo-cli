## Architecture Overview

This project ships a policy-aware web research pipeline with optional Evidence-First (EF) analysis. The entrypoint is `src/web/pipeline.py::run_research()`, which orchestrates search → fetch → extract → rerank → archive → normalization → sorting, with optional EF add-ons (claims/validators/evidence, counter-claim, reputation, corroboration) and telemetry surfaces.

For a full system diagram and deep-dive, see `docs/architecture.md`.

### Diagram (Mermaid)

The Mermaid source is in `docs/architecture.md`. Most renderers will display it inline; GitHub requires a plugin/preview or conversion.

### EF (Evidence-First) stack (flag-gated)

- Claim extraction, domain validators, evidence scoring
- Optional counter-claim signals, reputation prior, corroboration mapping
- All added as `citation['ef']` with a `confidence_breakdown` (evidence/validators/corroboration/prior)
- No ranking or acceptance changes unless future flags enable it

### Key flags

- EF: `EVIDENCE_FIRST`, `EVIDENCE_FIRST_KILL_SWITCH`
- Corroboration: `WEB_CORROBORATE_ENABLE`
- Counter-claim: `WEB_COUNTER_CLAIM_ENABLE`
- Reputation: `WEB_REPUTATION_ENABLE`
- Ledger: `WEB_VERACITY_LEDGER_ENABLE`
- Wire/Syndication preview: `WEB_WIRE_DEDUP_ENABLE`
- Rescue sweep preview: `WEB_RESCUE_SWEEP`
- Tier sweep controls: `WEB_TIER_SWEEP`, `WEB_TIER_SWEEP_MAX_SITES`, `WEB_TIER_SWEEP_STRICT`
- Debug/Cutover: `WEB_DEBUG_METRICS`, `WEB_CUTOVER_PREP`

### Debug schema (v1)

`answer.debug.schema_version = 1` with sections: `search`, `fetch`, `tier`, `wire`, `rescue`, `extract`, `discard`, `source_type`, `metrics`, `deprecation` (when `WEB_CUTOVER_PREP=1`), and `summary_line`.

## Current State

- PR1–PR3 landed earlier (flags, EF scaffolding with validators/evidence, provisional rendering hooks).
- PR4 completed: rescue preview, wire/syndication dedup preview, corroboration (debug only), all behind flags.
- PR5 completed: counter-claim, reputation prior, veracity ledger scaffolds (debug-only), behind flags.
- PR6 completed: unified debug schema v1, cutover-prep deprecation counters, and metrics scaffolds.

All EF-era features are flag-gated and default to no behavior change (ranking/acceptance unchanged). A kill switch restores legacy behavior instantly.

For migration guidance and validation, see `CUTOVER.md`.

# Ollama Turbo CLI

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

Go to <https://ollama.com/turbo>

Subscribe (from $20/month)

Create an API key: <https://ollama.com/settings/keys>

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

OLLAMA_API_KEY, OLLAMA_MODEL (default: gpt-oss:120b), OLLAMA_HOST (default: <https://ollama.com>)

MAX_CONVERSATION_HISTORY (default: 10)

STREAM_BY_DEFAULT (default: false)

LOG_LEVEL (default: INFO)

REASONING (low|medium|high, default: high)

Prompt flags: PROMPT_VERSION, PROMPT_INCLUDE_TOOL_GUIDE, PROMPT_FEWSHOTS

Tools: MULTI_ROUND_TOOLS (default: true), TOOL_MAX_ROUNDS (default: 6)

Tool results: TOOL_RESULTS_FORMAT (string|object, default: string)

Web research env vars

- BRAVE_API_KEY — Brave Search (recommended)
- TAVILY_API_KEY — Tavily Search
- EXA_API_KEY — Exa Search
- GOOGLE_PSE_KEY and GOOGLE_PSE_CX — both required for Google Programmable Search Engine
- WEB_RESPECT_ROBOTS — respect robots.txt (default: 1)
- WEB_ALLOW_BROWSER — allow headless browser fallback when needed (default: 1)
- SANDBOX_ALLOW_PROXIES — allow outbound via proxies if HTTP(S)_PROXY is set (default: 0)
- HTTP_PROXY / HTTPS_PROXY / ALL_PROXY / NO_PROXY — proxy URLs; strings like "None", "null", "false", or "0" are treated as unset
- WEB_TIMEOUT_CONNECT / WEB_TIMEOUT_READ / WEB_TIMEOUT_WRITE — httpx timeouts (seconds)
- WEB_MAX_CONNECTIONS / WEB_PER_HOST_CONCURRENCY — httpx connection limits
- WEB_EMERGENCY_BOOTSTRAP — enable emergency provider bootstrap inside pipeline when search finds zero (default: 1)
- WEB_DEBUG_METRICS — include search/fetch/dedupe counters in result (default: 0)
- WEB_CLEAN_WIKI_EDIT_ANCHORS — remove "[edit]" artifacts from Wikipedia extracts (default: 1)
- WEB_SITEMAP_ENABLED / WEB_SITEMAP_MAX_URLS / WEB_SITEMAP_INCLUDE_SUBS — optional sitemap augmentation for site-restricted queries

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

Proxies & egress

- If your network requires an egress proxy, set SANDBOX_ALLOW_PROXIES=1 and configure HTTPS_PROXY (and optionally HTTP_PROXY/ALL_PROXY)
- Example: HTTPS_PROXY=http://proxy-host:8080 and NO_PROXY=localhost,127.0.0.1
- The client ignores proxy envs that are set to literal strings like "None"/"null"/"false"/"0"

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

Web research troubleshooting

- Verify keys are loaded (BRAVE/TAVILY/EXA; PSE requires both KEY and CX)
- Enable debug metrics for one run: WEB_DEBUG_METRICS=1 (result.debug.search/fetch counters)
- If outbound is blocked, set SANDBOX_ALLOW_PROXIES=1 and HTTPS_PROXY/HTTP_PROXY
- If zero results for long queries, the pipeline will simplify/variant/boostrap automatically; you can disable bootstrap via WEB_EMERGENCY_BOOTSTRAP=0

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

Provider rotation & fallbacks

- Primary engines (keys optional but recommended): Brave → Tavily → Exa → Google PSE
- If none return results, use keyless DuckDuckGo fallback
- If still empty, simplify the query (drop stopwords, keep ≤6 salient tokens) and retry rotation
- If still empty, try a variant like "&lt;ProperNoun&gt; political makeup 2024"
- If still empty and WEB_EMERGENCY_BOOTSTRAP=1, the pipeline directly calls providers via httpx to bootstrap candidates

HTTP client & proxies

- Uses httpx with HTTP/1.1 (http2 disabled) for compatibility
- Proxies are honored only when SANDBOX_ALLOW_PROXIES=1 and proxy envs are set
- Strings like "None", "null", "false", or "0" in proxy envs are treated as unset

Tests
pytest -q


Includes plugin system tests and parity tests across standard/streaming, tool loops, fallback breadcrumbs, etc.

## Retrieval (RAG) controls and determinism

The CLI supports a lightweight, dependency-free retrieval layer for local corpora and web research. Key flags:

- `--docs-glob` — Glob for local JSON/JSONL corpora. Each row/object should include `id`, `title`, `url`, `timestamp`, `text`.
- `--rag-min-score` — Minimum BM25 score threshold; below this is treated as “no reliable support.”
- `--ground-fallback` — `off | web`. If retrieval returns no hits or the best score is below the threshold, `web` triggers a web research fallback and injects cited context.
- `--k` — Top‑k retrieval. Defaults to 5 (can also be overridden via `RAG_TOPK`).

Determinism

- Chunking: defaults to 1000 tokens per chunk with 200‑token overlap (configurable).
- Tokenizer/stopwords are fixed in `src/reliability/retrieval/pipeline.py`.
- The index includes a fingerprint derived from the full corpus content plus chunking params. Changing files or params changes the fingerprint; touching mtime alone does not.
- `get_index_meta()` exposes `fingerprint`, `num_chunks`, `avgdl`, and `load_warnings` for observability.

Ephemeral web research ingest

- Web citations are normalized into in‑memory doc objects and optionally routed through the retrieval pipeline for BM25 ranking, dedupe, and thresholding. No files are written unless explicitly requested.
- A hard cap (100 docs per query) prevents runaway memory growth. After each run with `ephemeral=True`, the retrieval index is cleared.

## Trace schema and metrics snapshots

For quick, pinned observability, the runtime emits single‑line trace keys:

- `retrieval.topk=N`
- `retrieval.avg_score=0.####`
- `retrieval.hit_rate=0.####`
- `retrieval.fallback_used=0|1`
- `retrieval.latency_ms=###`
- `web.latency_ms=###`
- `citations.count=N`
- `citations.coverage_pct=0.####`

Generate a snapshot artifact with these steps:

PowerShell (Windows)

```powershell
Set-Location ollama-turbo-cli
$env:GENERATE_METRICS_SNAPSHOT = '1'
pytest -q tests/test_metrics_snapshot.py
Get-Content tests/artifacts/metrics_snapshot.txt
```

Make (macOS/Linux)

```bash
cd ollama-turbo-cli
make metrics-snapshot
cat tests/artifacts/metrics_snapshot.txt
```

The snapshot contains the subset of trace lines for both standard and streaming runs so you can diff across model/provider updates.

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
v1.1.1 (current)

Web research fallbacks (simplify/variant) and emergency bootstrap path

Debug metrics (WEB_DEBUG_METRICS) for search/fetch/dedupe counters

Extractor cleanup to remove Wikipedia "[edit]" artifacts (WEB_CLEAN_WIKI_EDIT_ANCHORS)

HTTP client compatibility: HTTP/1.1 and robust proxy handling

v1.1.0

Heuristic auto-flags; flags plumbed end-to-end

Hardened plugin validation & arg cleanup

Docs/readme refresh

v1.0.0

Initial release (gpt-oss:120b)

Weather, calculator, files, system tools

Streaming + non-streaming, interactive + single message

Robust retries and error handling