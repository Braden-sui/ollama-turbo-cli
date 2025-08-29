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

# Import after loading env vars to ensure proper configuration
from .client import OllamaTurboClient
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
                       default=os.getenv('OLLAMA_MODEL', 'gpt-oss:120b'),
                       help='Model name (default: gpt-oss:120b)')
    parser.add_argument('--message', 
                       help='Single message mode (non-interactive)')
    parser.add_argument('--stream', 
                       action='store_true',
                       default=os.getenv('STREAM_BY_DEFAULT', 'false').lower() == 'true',
                       help='Enable streaming responses')
    parser.add_argument('--no-tools', 
                       action='store_true',
                       help='Disable tool calling')
    parser.add_argument('--show-trace',
                       action='store_true',
                       help='Show a separated reasoning trace (tools and steps) after the output')
    parser.add_argument('--quiet',
                       action='store_true',
                       help='Reduce CLI output (suppress helper prints)')
    parser.add_argument('--reasoning', 
                       default=os.getenv('REASONING', 'high'),
                       choices=['low', 'medium', 'high'],
                       help='Set reasoning effort (low, medium, high). Default: high')
    parser.add_argument('--reasoning-mode',
                       default=(os.getenv('REASONING_MODE') or 'system'),
                       choices=['system', 'request:top', 'request:options'],
                       help='How to send reasoning effort to the provider: system (system message directive), request:top (top-level payload), request:options (under options). Default: system')
    # Protocol selection
    parser.add_argument('--protocol',
                       default=(os.getenv('OLLAMA_PROTOCOL') or 'auto'),
                       choices=['auto', 'harmony', 'deepseek'],
                       help='Model protocol to use: auto (detect), harmony, deepseek. Default: auto')
    # Generation and display controls
    env_max_out = os.getenv('MAX_OUTPUT_TOKENS')
    env_ctx = os.getenv('CTX_SIZE')
    env_tool_print = os.getenv('TOOL_PRINT_LIMIT', '200')
    parser.add_argument('--max-output-tokens',
                       type=int,
                       default=int(env_max_out) if env_max_out and env_max_out.isdigit() else None,
                       help='Max output tokens to generate (mapped to Ollama options.num_predict). Default: API default')
    parser.add_argument('--ctx-size',
                       type=int,
                       default=int(env_ctx) if env_ctx and env_ctx.isdigit() else None,
                       help='Context window size (mapped to Ollama options.num_ctx). Default: API default')
    parser.add_argument('--tool-print-limit',
                       type=int,
                       default=int(env_tool_print) if env_tool_print.isdigit() else 200,
                       help='Character limit when printing tool results inline in CLI (does not affect tool messages sent to the model).')
    # Reliability mode flags (no-op until pipeline is wired)
    parser.add_argument('--ground',
                       action='store_true',
                       help='Enable retrieval-grounded reliability mode (adds external context).')
    parser.add_argument('--k',
                       type=int,
                       help='Top-k retrieval and/or consensus runs (context-dependent).')
    parser.add_argument('--cite',
                       action='store_true',
                       help='Request inline citations in the final answer when reliability mode is active.')
    parser.add_argument('--check',
                       choices=['off', 'warn', 'enforce'],
                       default='off',
                       help='Validator/guard mode for reliability: off, warn, or enforce (default: off).')
    parser.add_argument('--consensus',
                       action='store_true',
                       help='Enable k-run consensus voting (majority/agree-rate).')
    parser.add_argument('--engine',
                       help='Target engine: cloud | local | full URL (http[s]://...). Default: cloud / OLLAMA_HOST.')
    parser.add_argument('--eval',
                       help='Path to a JSONL corpus for micro-evaluation (optional).')
    # Sampling parameters
    # Generic env defaults (apply to any protocol)
    env_temp = os.getenv('TEMPERATURE')
    env_topp = os.getenv('TOP_P')
    env_pp = os.getenv('PRESENCE_PENALTY')
    env_fp = os.getenv('FREQUENCY_PENALTY')
    parser.add_argument('--temperature',
                       type=float,
                       default=(float(env_temp) if env_temp not in (None, '') else None),
                       help='Sampling temperature (0..2). Overrides protocol defaults if provided.')
    parser.add_argument('--top-p',
                       dest='top_p',
                       type=float,
                       default=(float(env_topp) if env_topp not in (None, '') else None),
                       help='Top-p nucleus sampling (0..1). Overrides protocol defaults if provided.')
    parser.add_argument('--presence-penalty',
                       type=float,
                       default=(float(env_pp) if env_pp not in (None, '') else None),
                       help='Presence penalty to reduce repetition. Overrides protocol defaults if provided.')
    parser.add_argument('--frequency-penalty',
                       type=float,
                       default=(float(env_fp) if env_fp not in (None, '') else None),
                       help='Frequency penalty to reduce token repetition. Overrides protocol defaults if provided.')
    parser.add_argument('--log-level', 
                       default=os.getenv('LOG_LEVEL', 'INFO'),
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Set logging level')
    parser.add_argument('--version', 
                       action='version', 
                       version='%(prog)s 1.1.0')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
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
        # Initialize client
        client = OllamaTurboClient(
            api_key=args.api_key,
            model=args.model,
            enable_tools=not args.no_tools,
            show_trace=args.show_trace,
            reasoning=args.reasoning,
            reasoning_mode=args.reasoning_mode,
            protocol=args.protocol,
            quiet=args.quiet,
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
            engine=args.engine,
            eval_corpus=args.eval
        )
        
        logger.info(f"Initialized Ollama Turbo client with model: {args.model}")
        
        if args.message:
            # Single message mode
            if not args.quiet:
                print(f"🔄 Sending message to {args.model}...")
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
