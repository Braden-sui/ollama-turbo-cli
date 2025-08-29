"""
Factory for creating protocol adapters for the Ollama Turbo client.

This module detects the appropriate adapter based on explicit protocol override
or model name heuristics. It performs lazy imports to avoid import errors before
concrete adapters are implemented.
"""
from __future__ import annotations

from typing import Optional

from ..types import ProtocolName
from .base import ProtocolAdapter


def detect_protocol(model: str, protocol: ProtocolName = "auto") -> ProtocolName:
    """Detect protocol from explicit override or model name.

    Heuristic (can be extended as new providers are added):
    - If override provided and not 'auto', honor it.
    - If model contains 'deepseek' -> 'deepseek'.
    - Otherwise fallback to 'harmony'.
    """
    if protocol and protocol != "auto":
        return protocol

    m = (model or "").lower()
    if "deepseek" in m or "ds-" in m:
        return "deepseek"  # tentative; refine as we learn exact SKUs
    return "harmony"


def get_adapter(model: str, protocol: ProtocolName = "auto") -> ProtocolAdapter:
    """Instantiate a concrete adapter for the given model/protocol.

    This uses lazy imports so that Phase 1 scaffolding does not require
    concrete adapter files to exist yet. Until adapters are implemented,
    this will raise a NotImplementedError with guidance.
    """
    resolved = detect_protocol(model, protocol)

    if resolved == "harmony":
        try:
            from .harmony import HarmonyAdapter  # type: ignore
        except Exception as e:  # noqa: BLE001 - surface clear guidance for Phase 2
            raise NotImplementedError(
                "HarmonyAdapter not implemented yet. Implement src/protocols/harmony.py in Phase 2."
            ) from e
        return HarmonyAdapter(model=model, protocol=resolved)

    if resolved == "deepseek":
        try:
            from .deepseek import DeepSeekAdapter  # type: ignore
        except Exception as e:  # noqa: BLE001
            raise NotImplementedError(
                "DeepSeekAdapter not implemented yet. Implement src/protocols/deepseek.py in Phase 4."
            ) from e
        return DeepSeekAdapter(model=model, protocol=resolved)

    # Should never happen due to typing of ProtocolName
    raise ValueError(f"Unknown protocol: {resolved}")
