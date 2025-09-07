from __future__ import annotations

"""
Central configuration surface for Ollama Turbo CLI.

This wraps the dataclasses in src/core/config.py and provides helpers to
construct a runtime config from environment and/or CLI flags. Keep this as the
single source of truth and inject into Client/runner paths.
"""
from typing import Optional
from dataclasses import asdict

from .core.config import ClientRuntimeConfig


def from_env(*, model: Optional[str] = None,
             protocol: Optional[str] = None,
             quiet: Optional[bool] = None,
             show_trace: Optional[bool] = None,
             engine: Optional[str] = None) -> ClientRuntimeConfig:
    """Build a Config using environment variables (no CLI required)."""
    return ClientRuntimeConfig.from_env(
        model=model,
        protocol=protocol,
        quiet=quiet,
        show_trace=show_trace,
        engine=engine,
    )


def merge_cli_overrides(cfg: ClientRuntimeConfig, args) -> ClientRuntimeConfig:
    """Overlay selected CLI flag values into an existing config.

    Args is expected to be an argparse.Namespace with attributes defined in cli.py.
    Missing attributes are ignored.
    """
    # Model/protocol/verbosity
    if getattr(args, 'model', None):
        cfg.model = args.model
    if getattr(args, 'protocol', None):
        cfg.protocol = args.protocol
    if getattr(args, 'quiet', None) is not None:
        cfg.quiet = bool(args.quiet)
    if getattr(args, 'show_trace', None) is not None:
        cfg.show_trace = bool(args.show_trace)

    # Transport-level engine override
    if getattr(args, 'engine', None):
        cfg.transport.engine = args.engine

    # Sampling
    if getattr(args, 'max_output_tokens', None) is not None:
        cfg.sampling.max_output_tokens = args.max_output_tokens
    if getattr(args, 'ctx_size', None) is not None:
        cfg.sampling.ctx_size = args.ctx_size
    if getattr(args, 'temperature', None) is not None:
        cfg.sampling.temperature = args.temperature
    if getattr(args, 'top_p', None) is not None:
        cfg.sampling.top_p = args.top_p
    if getattr(args, 'presence_penalty', None) is not None:
        cfg.sampling.presence_penalty = args.presence_penalty
    if getattr(args, 'frequency_penalty', None) is not None:
        cfg.sampling.frequency_penalty = args.frequency_penalty

    # Tooling
    if getattr(args, 'tool_print_limit', None) is not None:
        cfg.tooling.print_limit = args.tool_print_limit

    # Reliability
    if getattr(args, 'ground', None) is not None:
        cfg.reliability.ground = bool(args.ground)
    if getattr(args, 'k', None) is not None:
        cfg.reliability.k = args.k
    if getattr(args, 'cite', None) is not None:
        cfg.reliability.cite = bool(args.cite)
    if getattr(args, 'check', None):
        cfg.reliability.check = args.check
    if getattr(args, 'consensus', None) is not None:
        cfg.reliability.consensus = bool(args.consensus)
    if getattr(args, 'eval', None):
        cfg.reliability.eval_corpus = args.eval

    # History
    # Keep env-derived cap for now; can add CLI flag later

    return cfg


def to_dict(cfg: ClientRuntimeConfig) -> dict:
    """Serialize Config to a plain dictionary (for debugging/trace)."""
    return asdict(cfg)
