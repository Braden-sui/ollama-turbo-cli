from __future__ import annotations
import os
from typing import Any

# Centralized feature gates. These mirror existing behavior without changing defaults.
# Keep environment-first logic where the legacy code used envs directly so callers
# can continue to steer behavior without rebuilding configs.

def _env_bool(name: str, default: bool) -> bool:
    try:
        v = os.getenv(name)
        if v is None:
            return default
        return str(v).strip().lower() not in {"0", "false", "no", "off"}
    except Exception:
        return default


def exp_rescue(cfg: Any) -> bool:
    """Rescue sweep preview/enable.
    Backward compatible with WEB_RESCUE_SWEEP and legacy WEB_RESCUE_PREVIEW.
    Defaults to False if neither is set.
    """
    try:
        v = os.getenv("WEB_RESCUE_SWEEP")
        if v is None:
            v = os.getenv("WEB_RESCUE_PREVIEW")
        return False if v is None else (str(v).strip().lower() not in {"0", "false", "no", "off"})
    except Exception:
        return False


def exp_ef(cfg: Any) -> bool:
    """Evidence-first experiment active only if enabled and kill switch is OFF."""
    try:
        return bool(getattr(cfg, 'evidence_first', False)) and not bool(getattr(cfg, 'evidence_first_kill_switch', True))
    except Exception:
        return False


def exp_ledger(cfg: Any) -> bool:
    """Ledger writing toggle (env-driven for safety; default off)."""
    return _env_bool("WEB_VERACITY_LEDGER_ENABLE", False)
