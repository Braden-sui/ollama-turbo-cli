#!/usr/bin/env python3
"""
Main CLI application for Ollama Turbo.
"""

import argparse
import os
import sys
import logging
from dotenv import load_dotenv, find_dotenv

# Load environment variables from the closest .env.local then .env (searching upward)
try:
    path_local = find_dotenv('.env.local', usecwd=True)
    if path_local:
        load_dotenv(path_local, override=True)
    path_default = find_dotenv('.env', usecwd=True)
    if path_default:
        # Do not override values already loaded from .env.local or process env
        load_dotenv(path_default, override=False)
except Exception:
    # Fail-closed if dotenv is unavailable or search fails
    pass

# Warning silencers are centralized in core.config

from .client import OllamaTurboClient
from .config import from_env as build_config, merge_cli_overrides
from .utils import setup_logging, validate_api_key


def main():
    """Main entry point for Ollama Turbo CLI."""
    parser = argparse.ArgumentParser(
        description="Ollama Turbo CLI with gpt-oss:120b and advanced tool calling",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --api-key sk-xxx                           # Interactive mode
  %(prog)s --message "Calculate 15 * 8"               # Single message
  %(prog)s --message "Weather in London" --stream     # Streaming mode
  %(prog)s --api-key sk-xxx --model gpt-oss:20b      # Different model
        """
    )
    
    parser.add_argument('--api-key', 
                       default=os.getenv('OLLAMA_API_KEY'),
                       help='Ollama API key (or set OLLAMA_API_KEY env var)')
    parser.add_argument('--model', 
                       default=None,
                       help='Model name (default from env OLLAMA_MODEL or config)')
    parser.add_argument('--message', 
                       help='Single message mode (non-interactive)')
    parser.add_argument('--stream', 
                       action='store_true',
                       default=False,
                       help='Enable streaming responses')
    parser.add_argument('--no-tools', 
                       action='store_true',
                       help='Disable tool calling')
    parser.add_argument('--show-trace',
                       action='store_true',
                       help='Show a separated reasoning trace (tools and steps) after the output')
    parser.add_argument('--show-snippets',
                       action='store_true',
                       default=False,
                       help='Show a Sources (raw snippets) section after the final answer when tools were used (Phase 0). Default: on unless --legacy is set')
    parser.add_argument('--legacy',
                       action='store_true',
                       default=False,
                       help='Use legacy CLI layout: print [Tool Results] before the answer instead of the new Sources-after-answer UX')
    parser.add_argument('--quiet',
                       action='store_true',
                       help='Reduce CLI output (suppress helper prints)')
    parser.add_argument('--reasoning', 
                       default=None,
                       choices=['low', 'medium', 'high'],
                       help='Set reasoning effort (low, medium, high). Default: high')
    parser.add_argument('--reasoning-mode',
                       default=None,
                       choices=['system', 'request:top', 'request:options'],
                       help='How to send reasoning effort to the provider: system (system message directive), request:top (top-level payload), request:options (under options). Default: system')
    # Protocol selection
    parser.add_argument('--protocol',
                       default=None,
                       choices=['auto', 'harmony', 'deepseek'],
                       help='Model protocol to use: auto (detect), harmony, deepseek. Default: auto')
    # Generation and display controls (defaults resolved by central config)
    env_tool_print = os.getenv('TOOL_PRINT_LIMIT', '')
    parser.add_argument('--max-output-tokens',
                       type=int,
                       default=None,
                       help='Max output tokens to generate (mapped to Ollama options.num_predict). Default: API default')
    parser.add_argument('--ctx-size',
                       type=int,
                       default=None,
                       help='Context window size (mapped to Ollama options.num_ctx). Default: API default')
    parser.add_argument('--tool-print-limit',
                       type=int,
                       default=None,
                       help='Character limit when printing tool results inline in CLI (defaults from config).')
    # Reliability mode flags (defaults now favor grounded, cited, strict answers)
    parser.add_argument('--ground',
                       action=argparse.BooleanOptionalAction,
                       default=None,
                       help='Enable retrieval-grounded reliability mode (adds external context). Default: on')
    parser.add_argument('--k',
                       type=int,
                       help='Top-k retrieval and/or consensus runs (context-dependent).')
    parser.add_argument('--cite',
                       action=argparse.BooleanOptionalAction,
                       default=None,
                       help='Request inline citations in the final answer when reliability mode is active. Default: on')
    parser.add_argument('--check',
                       choices=['off', 'warn', 'enforce'],
                       default=None,
                       help='Validator/guard mode for reliability: off, warn, or enforce. Default: enforce')
    parser.add_argument('--consensus',
                       action='store_true',
                       help='Enable k-run consensus voting (majority/agree-rate).')
    # Retrieval controls
    parser.add_argument('--docs-glob',
                       default=None,
                       help='Glob pattern for local corpus files (JSON/JSONL). Supports multiple via comma-separated globs.')
    parser.add_argument('--rag-min-score',
                       type=float,
                       default=None,
                       help='Minimum BM25 score threshold; below this is considered no reliable support.')
    parser.add_argument('--ground-fallback',
                       choices=['off', 'web'],
                       default=None,
                       help='If no local hits or below threshold, fallback path: off | web (invoke web_research). Default: off')
    parser.add_argument('--engine',
                       help='Target engine: cloud | local | full URL (http[s]://...). Default: cloud / OLLAMA_HOST.')
    parser.add_argument('--eval',
                       help='Path to a JSONL corpus for micro-evaluation (optional).')
    # Sampling parameters
    # Generic env defaults (apply to any protocol)
    parser.add_argument('--temperature',
                       type=float,
                       default=None,
                       help='Sampling temperature (0..2). Overrides protocol defaults if provided.')
    parser.add_argument('--top-p',
                       dest='top_p',
                       type=float,
                       default=None,
                       help='Top-p nucleus sampling (0..1). Overrides protocol defaults if provided.')
    parser.add_argument('--presence-penalty',
                       type=float,
                       default=None,
                       help='Presence penalty to reduce repetition. Overrides protocol defaults if provided.')
    parser.add_argument('--frequency-penalty',
                       type=float,
                       default=None,
                       help='Frequency penalty to reduce token repetition. Overrides protocol defaults if provided.')
    parser.add_argument('--log-level', 
                       default=os.getenv('LOG_LEVEL', 'INFO'),
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Set logging level')
    # Mem0 configuration
    mem0_group = parser.add_argument_group('Mem0 Configuration', 'Control memory and context settings')
    mem0_group.add_argument('--mem0', 
                          action=argparse.BooleanOptionalAction,
                          default=None,
                          help='Enable/disable Mem0 memory system (default from env/config)')
    mem0_group.add_argument('--mem0-local', 
                          action=argparse.BooleanOptionalAction,
                          default=None,
                          help='Use local Mem0 (OSS) mode instead of cloud')
    mem0_group.add_argument('--mem0-vector', 
                          choices=['qdrant', 'chroma'],
                          default=None,
                          help='Vector store provider for local Mem0')
    mem0_group.add_argument('--mem0-vector-host', 
                          default=None,
                          help='Vector store host or path')
    mem0_group.add_argument('--mem0-vector-port', 
                          type=int,
                          default=None,
                          help='Vector store port (if applicable)')
    mem0_group.add_argument('--mem0-ollama-url', 
                          default=None,
                          help='Ollama base URL for local embeddings/LLM')
    mem0_group.add_argument('--mem0-model', 
                          default=None,
                          help='Model to use for Mem0 (defaults to main model if not set)')
    mem0_group.add_argument('--mem0-embedder', 
                          default=None,
                          help='Embedding model for Mem0')
    mem0_group.add_argument('--mem0-user', 
                          default=None,
                          help='User ID for memory isolation')

    parser.add_argument('--version', 
                       action='version', 
                       version='%(prog)s 1.2.0')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    # Optional startup notice when jsonschema is missing (tool schema validation partially disabled)
    try:
        import jsonschema  # type: ignore
    except Exception:
        if not args.quiet:
            print("ℹ️ Tool schema validation is partially disabled (jsonschema not installed). Run: pip install \"ollama-turbo-cli[tools]\"")
    
    # Auto-select model based on --protocol when --model is not explicitly provided.
    # We treat an explicit --model flag as authoritative and only warn on mismatches.
    try:
        has_model_flag = any(tok == "--model" for tok in sys.argv)
    except Exception:
        has_model_flag = False
    
    try:
        if args.protocol in ("deepseek", "harmony"):
            if not has_model_flag:
                if args.protocol == "deepseek":
                    # Allow override via env, else default to DeepSeek v3.1
                    args.model = os.getenv("DEEPSEEK_DEFAULT_MODEL", "deepseek-v3.1:671b")
                else:
                    # Harmony default
                    args.model = os.getenv("HARMONY_DEFAULT_MODEL", "gpt-oss:120b")
                logger.info(f"Protocol '{args.protocol}' selected; using model: {args.model} (auto)")
            else:
                # User specified a model explicitly; emit gentle warnings on mismatch
                m = (args.model or "").lower()
                if args.protocol == "deepseek" and ("deepseek" not in m):
                    logger.warning(
                        "Protocol 'deepseek' selected but model '%s' does not look like a DeepSeek model. "
                        "Consider --model deepseek-v3.1:671b or set DEEPSEEK_DEFAULT_MODEL.",
                        args.model,
                    )
                if args.protocol == "harmony" and ("deepseek" in m):
                    logger.warning(
                        "Protocol 'harmony' selected with a DeepSeek-looking model '%s'. "
                        "If this is intentional, ignore; otherwise consider --protocol deepseek.",
                        args.model,
                    )
    except Exception:
        # Never fail CLI due to optional convenience logic
        pass
    
    # Validate API key
    if not args.api_key:
        print("❌ Error: API key is required. Set OLLAMA_API_KEY environment variable or use --api-key")
        print("   Get your API key from: https://ollama.com/settings/keys")
        sys.exit(1)
    
    if not validate_api_key(args.api_key):
        print("❌ Error: Invalid API key format")
        sys.exit(1)
    
    try:
        # Centralized config: from env then overlay CLI flags
        cfg = build_config(
            model=args.model,
            protocol=args.protocol,
            quiet=args.quiet,
            show_trace=args.show_trace,
            engine=args.engine,
        )
        cfg = merge_cli_overrides(cfg, args)

        # Resolve effective show_snippets default (new UX default):
        #   - If --legacy is present: force False (legacy behavior)
        #   - Else if --show-snippets is explicitly present: True
        #   - Else: True (default-on for CLI)
        try:
            has_legacy_flag = any(tok == "--legacy" for tok in sys.argv)
        except Exception:
            has_legacy_flag = bool(args.legacy)
        try:
            has_show_snippets_flag = any(tok == "--show-snippets" for tok in sys.argv)
        except Exception:
            has_show_snippets_flag = bool(args.show_snippets)
        if has_legacy_flag:
            effective_show_snippets = False
        elif has_show_snippets_flag:
            effective_show_snippets = True
        else:
            effective_show_snippets = True

        # Initialize client (inject cfg)
        client = OllamaTurboClient(
            api_key=args.api_key,
            model=args.model,
            enable_tools=not args.no_tools,
            show_trace=args.show_trace,
            show_snippets=bool(effective_show_snippets),
            reasoning=args.reasoning,
            reasoning_mode=args.reasoning_mode,
            protocol=args.protocol,
            quiet=args.quiet,
            # Mem0 configuration (kept for backwards compat; cfg takes precedence)
            mem0_enabled=args.mem0,
            mem0_local=args.mem0_local,
            mem0_vector_provider=args.mem0_vector,
            mem0_vector_host=args.mem0_vector_host,
            mem0_vector_port=(args.mem0_vector_port if (args.mem0_vector_port is not None and args.mem0_vector_port > 0) else None),
            mem0_ollama_url=args.mem0_ollama_url,
            mem0_llm_model=args.mem0_model,
            mem0_embedder_model=args.mem0_embedder,
            mem0_user_id=args.mem0_user,
            max_output_tokens=args.max_output_tokens,
            ctx_size=args.ctx_size,
            tool_print_limit=args.tool_print_limit,
            temperature=args.temperature,
            top_p=args.top_p,
            presence_penalty=args.presence_penalty,
            frequency_penalty=args.frequency_penalty,
            # Reliability mode plumbing (no-op until implemented)
            ground=args.ground,
            k=args.k,
            cite=args.cite,
            check=args.check,
            consensus=args.consensus,
            docs_glob=args.docs_glob,
            rag_min_score=args.rag_min_score,
            ground_fallback=args.ground_fallback,
            engine=args.engine,
            eval_corpus=args.eval,
            cfg=cfg,
        )
        
        logger.info(f"Initialized Ollama Turbo client with model: {getattr(client, 'model', args.model)}")
        
        if args.message:
            # Single message mode
            # Force streaming by default for --message to avoid upstream 502s some providers return
            # on non-streaming endpoints. Users can still toggle behavior later if needed.
            if not args.stream:
                args.stream = True
            if not args.quiet:
                print(f"🔄 Sending message to {getattr(client, 'model', args.model)}...")
            response = client.chat(args.message, stream=args.stream)
            if not args.stream:
                # In quiet mode, print only the response body
                if args.quiet:
                    print(response)
                else:
                    print(f"🤖 Response: {response}")
        else:
            # Interactive mode
            client.interactive_mode()
            
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Application error: {e}")
        print(f"❌ Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
