from __future__ import annotations

"""
Args normalizer for tool execution.

normalize_args(schema, raw) -> dict
- Accepts a JSON Schema (or True/None) and a raw arguments payload that may be a
  dict or a JSON string.
- Returns a validated dict (best-effort if jsonschema not installed).
- Raises ValueError with a helpful message when validation fails.
"""
from typing import Any, Dict, Optional
import json

try:  # optional dependency, align with plugin_loader
    from jsonschema import validate as jsonschema_validate
except Exception:  # pragma: no cover - optional
    jsonschema_validate = None


def normalize_args(schema: Optional[Dict[str, Any]], raw: Any) -> Dict[str, Any]:
    """Return a dict of tool args, parsing from JSON if needed and validating.

    Parameters:
    - schema: JSON Schema for the tool parameters. May be None/True/{}.
    - raw: Raw args (dict, JSON string, or None).

    Returns:
    - Dict[str, Any]: validated arguments

    Raises:
    - ValueError: when the payload cannot be parsed or validated.
    """
    # 1) Parse to dict
    if raw is None:
        obj: Any = {}
    elif isinstance(raw, dict):
        obj = raw
    elif isinstance(raw, str):
        s = raw.strip()
        if not s:
            obj = {}
        else:
            try:
                obj = json.loads(s)
            except Exception as e:  # json parse error
                raise ValueError(f"Tool arguments must be a JSON object string; parse failed: {e}")
    else:
        # Unexpected runtime type from provider
        raise ValueError(f"Tool arguments must be an object; got {type(raw).__name__}")

    if not isinstance(obj, dict):
        raise ValueError("Tool arguments must be a JSON object")

    # 2) Normalize optional fields: drop keys with value None so optional schema
    #    properties don't fail validation when models emit explicit nulls.
    try:
        obj = {k: v for k, v in obj.items() if v is not None}
    except Exception:
        pass

    # 3) Validate against schema if provided and jsonschema is available
    if schema and jsonschema_validate is not None:
        try:
            jsonschema_validate(instance=obj, schema=schema)
        except Exception as e:
            # Normalize common jsonschema error texts for better UX
            msg = str(e)
            # Trim noisy prefixes
            if msg.startswith("\n"):  # some validators add leading newlines
                msg = msg.strip()
            raise ValueError(f"Tool arguments failed validation: {msg}")

    return obj
