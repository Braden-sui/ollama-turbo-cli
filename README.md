# Ollama Turbo CLI

A production-ready CLI application for interacting with OpenAI's gpt-oss:120b model through Ollama Turbo cloud service, featuring advanced tool calling capabilities and streaming responses.

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

```
"What's the weather in Tokyo?"
```

### 🧮 Mathematical Calculator
Perform complex mathematical calculations with support for functions.

```
"Calculate sin(pi/2) + sqrt(16)"
```

### 📁 File Operations
List files and directories with filtering options.

```
"List all Python files in the current directory"
```

### 💻 System Information
Get comprehensive system information including CPU, memory, and disk usage.

```
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

### 🖥️ Execute Shell (Sandboxed)

Run short, read-only commands in a locked-down sandbox. Disabled by default.

When to use: diagnostics like `git status`, `ls`, etc.
Tips: requires `SHELL_TOOL_ALLOW=1` and allowlisted prefixes; path-confined to `SHELL_TOOL_ROOT`; only compact summaries are injected.

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

## Configuration

### Environment Variables

- `OLLAMA_API_KEY`: Your Ollama API key
- `OLLAMA_MODEL`: Model name (default: gpt-oss:120b)
- `OLLAMA_HOST`: API endpoint (default: https://ollama.com)
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

- __Initialization__: Loaded silently from environment variables. No runtime prompts.
- __Status banner__: Interactive mode shows `Mem0: enabled (user: <id>)` or `Mem0: disabled (no key)`.
- __Injection hygiene__: At most one "Relevant information:" system block is injected per turn before your message; previous injection blocks are removed each turn. Local conversation history is capped at 10 turns.
- __Natural language intents__: You can type phrases like:
  - "remember ...", "forget ...", "update X to Y", "list/show memories", "link <id1> <id2>", "search memories for ...", or "what do you know about me".
- __/mem commands__: `list`, `search <q>`, `add <text>`, `get <id>`, `update <id> <text>`, `delete <id>`, `clear`, `link <id1> <id2>`, `export [path.json]`, `import <path.json>`.
- __Export/Import__: Exports to `mem0_export_*.json` including `id`, `memory`, `metadata`, `created_at`, `updated_at`. Git ignores these files by default.
- __Resilience__: All Mem0 operations are wrapped in try/except; failures never block chat. A one-time notice is shown if Mem0 is unavailable; details are logged at DEBUG level.
- __Filters__: Minimal filters are used for recall (`user_id` only) for list/search/export until recall is proven.

## Secure Execution & Web Access

- __Shell tool is disabled by default__: Commands run only inside a locked-down sandbox when enabled. Set `SHELL_TOOL_ALLOW=1` and keep an explicit `SHELL_TOOL_ALLOWLIST` (prefix match) for safe commands like `git status`, `ls`, `cat`.

- __Confirmation prompts__: In TTY, `execute_shell` shows a one-line preview and requires confirmation unless `SHELL_TOOL_CONFIRM=0`.

- __Sandbox guarantees__: CPU/mem/pids/disk/time limits, read-only project mount at `/project`, tmpfs workspace, no host env (only `env_vars` pass-through). Windows requires Docker Desktop/WSL2. If Docker is unavailable, execution fails closed with a clear message.

- __Web access is allowlist-only__: The `web_fetch` tool goes through a controlled client with host allowlist and SSRF protections. HTTPS-only by default (`SANDBOX_ALLOW_HTTP=0`). Add domains to `SANDBOX_NET_ALLOW` (supports wildcards like `*.wikipedia.org`).

- __Summarization before injection__: Tool outputs are summarized and capped (`TOOL_CONTEXT_MAX_CHARS`, default 4000). Full logs and payloads are kept under `.sandbox/sessions/` and `.sandbox/cache/` and are not shared with the model.

- __Mem0 safety__: Outputs from `execute_shell` and sensitive `web_fetch` responses are flagged and not persisted to Mem0.

- __Caching & rate limits__: On-host HTTP cache (TTL 10m, LRU ~200MB) accelerates repeat fetches. Per-host rate-limits prevent abuse.

### Enable a one-off command safely

1. Export envs:
   - `SHELL_TOOL_ALLOW=1`
   - Optional: extend `SHELL_TOOL_ALLOWLIST` with a safe prefix (e.g., `git show`)

2. Run your prompt. When asked, confirm the preview.

### Add a domain to the allowlist

- Set `SANDBOX_NET_ALLOW` to include your domain(s), comma-separated, e.g.:
  ```
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

```
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

### v1.0.0 (Current)
- Initial release with gpt-oss:120b support
- Four built-in tools: weather, calculator, files, system
- Streaming and non-streaming modes
- Interactive and single-message modes
- Comprehensive error handling and retry logic
