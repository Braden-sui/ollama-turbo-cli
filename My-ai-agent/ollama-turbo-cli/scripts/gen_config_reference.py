from __future__ import annotations

"""
Generate a Configuration Reference markdown from src/core/config.py dataclasses.

Usage:
  python -m scripts.gen_config_reference > CONFIG_REFERENCE.md

Notes:
- This inspects dataclasses and prints field names with default values.
- Units are inferred heuristically: *_ms -> milliseconds, *_s or *_secs -> seconds.
- Environment variable precedence and CLI overlays are handled by code; this output
  focuses on the configuration surface and defaults.
"""
import sys
import os
import importlib.util
from dataclasses import fields, is_dataclass, MISSING
from typing import Any, get_origin, get_args


def _load_config_module():
    """Load src/core/config.py without importing the whole src package."""
    here = os.path.dirname(os.path.abspath(__file__))
    cfg_path = os.path.abspath(os.path.join(here, "..", "src", "core", "config.py"))
    spec = importlib.util.spec_from_file_location("_cfg_model", cfg_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load config module from {cfg_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_cfg_model"] = mod
    spec.loader.exec_module(mod)
    return mod


def _fmt_default(v: Any) -> str:
    try:
        if isinstance(v, str):
            return f"'{v}'"
        return str(v)
    except Exception:
        return "<unrepr>"


def _typename(tp: Any) -> str:
    try:
        origin = get_origin(tp)
        if origin is None:
            return getattr(tp, "__name__", str(tp))
        args = ", ".join(_typename(a) for a in get_args(tp))
        return f"{getattr(origin, '__name__', str(origin))}[{args}]"
    except Exception:
        return str(tp)


def _safe_factory_repr(factory: Any) -> str:
    if factory in (list, dict, set, tuple):
        try:
            return _fmt_default(factory())
        except Exception:
            return "<factory>"
    return "<factory>"


def _infer_units(name: str) -> str:
    n = name.lower()
    if n.endswith("_ms"):
        return " (ms)"
    if n.endswith("_s") or n.endswith("_secs") or n.endswith("_seconds"):
        return " (s)"
    return ""


def _dump_dataclass(title: str, dc: Any) -> str:
    out = [f"### {title}", ""]
    for f in fields(dc):
        units = _infer_units(f.name)
        typ = _typename(getattr(f, 'type', Any))
        # Detect if this field is (or contains) a nested dataclass type
        nested = False
        try:
            t = getattr(f, 'type', None)
            origin = get_origin(t)
            if origin is None and isinstance(t, type) and is_dataclass(t):
                nested = True
            else:
                for a in get_args(t) or ():
                    if isinstance(a, type) and is_dataclass(a):
                        nested = True
                        break
        except Exception:
            pass
        # default handling for dataclass fields
        if f.default is not MISSING:
            default_repr = _fmt_default(f.default)
        elif f.default_factory is not MISSING:
            default_repr = _safe_factory_repr(f.default_factory)
        else:
            default_repr = "<none>"
        suffix = " · nested" if nested else ""
        out.append(f"- {f.name}{units}: default={default_repr}  ·  type={typ}{suffix}")
        # Optional help text via metadata={'help': '...'}
        try:
            help_txt = (f.metadata or {}).get('help') if hasattr(f, 'metadata') else None
            if help_txt:
                out.append(f"  • {help_txt}")
        except Exception:
            pass
    out.append("")
    return "\n".join(out)


def main() -> int:
    print("## Configuration Reference (auto-generated)\n")
    print("Run: `python -m scripts.gen_config_reference > CONFIG_REFERENCE.md`\n")
    # Load config dataclasses directly from file to avoid importing src/__init__.py
    mod = _load_config_module()
    # Top-level client
    print(_dump_dataclass("ClientRuntimeConfig", getattr(mod, "ClientRuntimeConfig")))
    sections = [
        ("RetryConfig", getattr(mod, "RetryConfig")),
        ("TransportConfig", getattr(mod, "TransportConfig")),
        ("StreamingConfig", getattr(mod, "StreamingConfig")),
        ("SamplingConfig", getattr(mod, "SamplingConfig")),
        ("ToolingConfig", getattr(mod, "ToolingConfig")),
        ("Mem0Config", getattr(mod, "Mem0Config")),
        ("ReliabilityConfig", getattr(mod, "ReliabilityConfig")),
        ("HistoryConfig", getattr(mod, "HistoryConfig")),
        ("WebConfig", getattr(mod, "WebConfig")),
    ]
    for name, dc in sections:
        print(_dump_dataclass(name, dc))

    # Operational guidance / notes (static prose)
    print("""
### Environment precedence and loading

- The CLI (`src/cli.py`) and API (`src/api/app.py`) load `.env.local` first (override=True), then `.env` (override=False).
- The web pipeline (`src/web/pipeline.py`) also loads `.env.local` then `.env` on import, so tests and direct imports see the same defaults.
- Programmatic configs set via `src.web.pipeline.set_default_config(cfg.web)` take precedence when callers supply a central `ClientRuntimeConfig`.

### Permissive profile (example .env.local)

```
WEB_RESPECT_ROBOTS=0
WEB_HEAD_GATING=0
SANDBOX_NET_ALLOW=*
WEB_UA=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36
WEB_DEBUG_METRICS=1
```

### Citation policy: exclude certain domains

- Set `WEB_EXCLUDE_CITATION_DOMAINS` to a comma-separated list (e.g., `wikipedia.org,reddit.com`).
- These domains will be used for discovery (search), but they will not be quoted as citations in `run_research()`.

### Wikipedia-guided expansion

- When Wikipedia results appear in search, the pipeline fetches the page, extracts external links from the content, and adds those links as candidates.
- Wikipedia hosts are excluded from citations when `WEB_EXCLUDE_CITATION_DOMAINS` includes `wikipedia.org`.
- Debug counters include `excluded` (filtered candidates) and `wiki_refs_added` (external references added from Wikipedia pages).
""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
