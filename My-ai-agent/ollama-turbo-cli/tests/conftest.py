"""Pytest session bootstrap for this repository.

Responsibilities:
- Ensure `src/` is importable
- Apply early warning filters for third-party deprecations (e.g., Pydantic)
- Set safe, fast defaults for web pipeline behavior under tests
- Enable per-worker cache sharding to minimize FS contention

All environment defaults here are set with `setdefault` so individual tests
can override them with `monkeypatch.setenv` when needed.
"""

# Ensure the project root (containing the 'src' package) is importable during tests
import os
import sys
import warnings

HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Early warning silences for third-party deprecations seen in CI logs
try:
    # Pydantic v2 deprecation about class-based Config (third-party origins)
    warnings.filterwarnings(
        "ignore",
        category=DeprecationWarning,
        module=r"pydantic\._internal\._config",
        message=r"Support for class-based `config` is deprecated, use ConfigDict instead.*",
    )
    # Ollama client's Pydantic field deprecation (may surface indirectly)
    warnings.filterwarnings(
        "ignore",
        category=DeprecationWarning,
        module=r"ollama\._types",
        message=r".*model_fields.*deprecated.*",
    )
except Exception:
    pass

# Fast, safe defaults for web research pipeline during tests (overrideable)
try:
    os.environ.setdefault("WEB_CACHE_PER_WORKER", "1")
    os.environ.setdefault("WEB_TIER_SWEEP", "0")
    os.environ.setdefault("WEB_EMERGENCY_BOOTSTRAP", "0")
    os.environ.setdefault("WEB_RERANK_ENABLED", "0")
    os.environ.setdefault("WEB_ALLOW_BROWSER", "0")
    os.environ.setdefault("WEB_RESPECT_ROBOTS", "0")
    # Keep debug metrics off globally; tests that assert debug counters will override locally
    os.environ.setdefault("WEB_DEBUG_METRICS", "0")
    # Use a shorter read timeout to fail fast if something accidentally reaches the network
    os.environ.setdefault("WEB_TIMEOUT_READ", "5.0")
except Exception:
    pass

# Import centralized logging/warning silencers for tests (side effects only)
try:
    from src.core import config as _core_config  # noqa: F401
except Exception:
    pass
