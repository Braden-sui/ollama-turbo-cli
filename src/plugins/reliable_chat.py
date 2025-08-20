from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterator, Optional, Tuple

import requests

from ..reliability.planner import auto_reliability_flags


TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "reliable_chat",
        "description": "Call the Reliability API to produce a grounded and validated answer. Supports streaming aggregation with a trailing summary event. Returns a stable object with 'content' and 'summary'.",
        "parameters": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "message": {"type": "string", "description": "User message to answer."},
                "stream": {"type": "boolean", "default": False, "description": "Use streaming SSE endpoint and aggregate tokens + trailing summary."},
                "timeout_s": {"type": "integer", "default": 15, "minimum": 1, "maximum": 120, "description": "Read timeout seconds."},
                "connect_timeout_s": {"type": "integer", "default": 5, "minimum": 1, "maximum": 60, "description": "Connect timeout seconds."},
                "auto": {"type": "boolean", "default": True, "description": "Infer reliability flags from message using heuristics unless explicitly overridden."},
                "ground": {"type": "boolean", "description": "Enable retrieval + grounding context."},
                "k": {"type": "integer", "minimum": 1, "maximum": 16, "description": "Breadth (top-k) for retrieval or consensus."},
                "cite": {"type": "boolean", "description": "Ask model to include inline citations when grounded."},
                "check": {"type": "string", "enum": ["off", "warn", "enforce"], "default": "off", "description": "Validator mode."},
                "consensus": {"type": "boolean", "description": "Enable consensus (trace-only unless streaming summary)."},
                "engine": {"type": "string", "description": "Optional reliability engine override."},
                "eval_corpus": {"type": "string", "description": "Optional evaluation corpus id."},
            },
            "required": ["message"],
        },
    },
}


def _resolve_base_url() -> str:
    # Prefer explicit reliability base; fall back to local dev server
    return (
        os.getenv("RELIABILITY_API_BASE")
        or os.getenv("OLLAMA_RELIABILITY_BASE")
        or "http://127.0.0.1:8000"
    ).rstrip("/")


def _headers() -> Dict[str, str]:
    headers: Dict[str, str] = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
    }
    # App API key (FastAPI server gate)
    api_key = os.getenv("API_KEY")
    if api_key:
        headers["X-API-Key"] = api_key
    # Upstream service key (if router proxies upstream)
    bearer = os.getenv("OLLAMA_API_KEY")
    if bearer:
        headers["Authorization"] = f"Bearer {bearer}"
    return headers


def _safe_json(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False)
    except Exception:
        return json.dumps({"ok": False, "error": {"code": "serialization", "message": "Unserializable object"}})


def _sse_iter_lines(resp: requests.Response) -> Iterator[str]:
    for raw in resp.iter_lines(decode_unicode=True):
        if not raw:
            continue
        # Normalize: response may include leading/trailing spaces
        yield str(raw).strip()


def _parse_stream(resp: requests.Response) -> Tuple[str, Dict[str, Any]]:
    content = ""
    summary: Dict[str, Any] = {}
    last_event: Optional[str] = None
    for line in _sse_iter_lines(resp):
        if line.startswith("event:"):
            last_event = line.split(":", 1)[1].strip()
            continue
        if not line.startswith("data:"):
            continue
        data_str = line.split(":", 1)[1].strip()
        try:
            payload = json.loads(data_str)
        except Exception:
            continue
        if last_event == "summary":
            if isinstance(payload, dict):
                summary = payload
            # reset event after consuming summary
            last_event = None
            continue
        # default channel
        if isinstance(payload, dict):
            typ = payload.get("type")
            if typ == "token":
                content += payload.get("content", "")
            elif typ == "final":
                content = payload.get("content", content or "")
    return content, summary


def execute(
    message: str,
    *,
    stream: bool = False,
    timeout_s: int = 15,
    connect_timeout_s: int = 5,
    auto: bool = True,
    ground: Optional[bool] = None,
    k: Optional[int] = None,
    cite: Optional[bool] = None,
    check: str = "off",
    consensus: Optional[bool] = None,
    engine: Optional[str] = None,
    eval_corpus: Optional[str] = None,
) -> str:
    base = _resolve_base_url()
    headers = _headers()
    timeouts = (int(connect_timeout_s), int(timeout_s))

    # Heuristic flags
    flags: Dict[str, Any] = {}
    if auto:
        flags.update(auto_reliability_flags(message, overrides={
            # only apply overrides if explicitly provided (not None)
            **({"ground": ground} if ground is not None else {}),
            **({"k": k} if k is not None else {}),
            **({"cite": cite} if cite is not None else {}),
            **({"check": check} if check else {}),
            **({"consensus": consensus} if consensus is not None else {}),
            **({"engine": engine} if engine is not None else {}),
            **({"eval_corpus": eval_corpus} if eval_corpus is not None else {}),
        }))
    else:
        flags = {
            "ground": bool(ground or False),
            "k": k,
            "cite": bool(cite or False),
            "check": str(check or "off"),
            "consensus": bool(consensus or False),
            "engine": engine,
            "eval_corpus": eval_corpus,
        }

    payload = {
        "message": message,
        # only include non-default truthy/explicit values to keep payload compact
        **({"ground": True} if flags.get("ground") else {}),
        **({"k": int(flags["k"]) } if isinstance(flags.get("k"), int) else {}),
        **({"cite": True} if flags.get("cite") else {}),
        **({"check": flags.get("check")} if flags.get("check") and flags.get("check") != "off" else {}),
        **({"consensus": True} if flags.get("consensus") else {}),
        **({"engine": flags.get("engine")} if flags.get("engine") else {}),
        **({"eval_corpus": flags.get("eval_corpus")} if flags.get("eval_corpus") else {}),
    }

    try:
        if not stream:
            url = f"{base}/v1/chat"
            r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=timeouts)
            r.raise_for_status()
            data = r.json()
            content = data.get("content", "")
            # Non-streaming endpoint does not return a summary; provide a minimal stable skeleton
            summary = {
                "grounded": bool(flags.get("ground")),
                "citations": [],
                "validator": None if (flags.get("check") in (None, "off")) else {"mode": flags.get("check")},
                "consensus": None,
            }
            return _safe_json({
                "tool": "reliable_chat",
                "ok": True,
                "content": content,
                "summary": summary,
                "inject": content,
                "sensitive": False,
                "meta": {"endpoint": url, "stream": False},
            })
        else:
            url = f"{base}/v1/chat/stream"
            # For SSE, POST with stream=True and aggregate tokens + trailing summary event
            with requests.post(url, headers=headers, data=json.dumps(payload), timeout=timeouts, stream=True) as r:
                r.raise_for_status()
                content, summary = _parse_stream(r)
                if not isinstance(summary, dict):
                    summary = {}
                return _safe_json({
                    "tool": "reliable_chat",
                    "ok": True,
                    "content": content,
                    "summary": summary,
                    "inject": content,
                    "sensitive": False,
                    "meta": {"endpoint": url, "stream": True},
                })
    except requests.exceptions.Timeout as te:
        msg = f"timeout after {timeout_s}s"
        return _safe_json({
            "tool": "reliable_chat",
            "ok": False,
            "content": "",
            "summary": {},
            "error": {"code": "timeout", "message": msg},
            "inject": f"Reliability failed: {msg}",
            "sensitive": False,
        })
    except requests.exceptions.RequestException as rexc:
        return _safe_json({
            "tool": "reliable_chat",
            "ok": False,
            "content": "",
            "summary": {},
            "error": {"code": "http_error", "message": str(rexc)},
            "inject": "Reliability failed: HTTP error",
            "sensitive": False,
        })
    except Exception as e:
        return _safe_json({
            "tool": "reliable_chat",
            "ok": False,
            "content": "",
            "summary": {},
            "error": {"code": "runtime_error", "message": str(e)},
            "inject": "Reliability failed: internal error",
            "sensitive": False,
        })


TOOL_IMPLEMENTATION = execute
TOOL_AUTHOR = "core"
TOOL_VERSION = "1.0.0"
