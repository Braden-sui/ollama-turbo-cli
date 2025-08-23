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
                "timeout_s": {"type": "integer", "default": 60, "minimum": 1, "maximum": 120, "description": "Read timeout seconds."},
                "connect_timeout_s": {"type": "integer", "default": 5, "minimum": 1, "maximum": 60, "description": "Connect timeout seconds."},
                "auto": {"type": "boolean", "default": True, "description": "Infer reliability flags from message using heuristics unless explicitly overridden."},
                "ground": {"type": "boolean", "description": "Enable retrieval + grounding context."},
                "k": {"type": "integer", "minimum": 1, "maximum": 16, "description": "Breadth (top-k) for retrieval or consensus."},
                "cite": {"type": "boolean", "description": "Ask model to include inline citations when grounded."},
                "check": {"type": "string", "enum": ["off", "warn", "enforce"], "default": "off", "description": "Validator mode."},
                # Accept both boolean (legacy) and integer (preferred size). The executor will normalize.
                "consensus": {"type": "integer", "minimum": 1, "maximum": 16, "description": "Consensus size. Boolean true maps to default size (3)."},
                "engine": {"type": "string", "description": "Optional reliability engine override."},
                "eval_corpus": {"type": "string", "description": "Optional evaluation corpus id."},
                "fresh": {"type": "boolean", "default": False, "description": "Bypass memory; require new retrieval."},
                "provenance": {"type": "boolean", "default": True, "description": "Include provenance information."},
                "status": {"type": "boolean", "default": True, "description": "Include status information."},
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

def _parse_stream(resp: requests.Response) -> Tuple[str, Dict[str, Any]]:
    """Parse SSE stream supporting:
    - default channel payloads {type: token|final}
    - event: message with {text: "..."} or {content: "..."}
    - event: summary with a JSON dict
    - multi-line data: segments per SSE spec
    - string payloads
    """
    content = ""
    summary: Dict[str, Any] = {}
    last_event: Optional[str] = None
    buf: list[str] = []

    for raw in resp.iter_lines(decode_unicode=True):
        line = (raw or "").strip()
        if not line:
            # flush buffered event on blank line
            if buf:
                data_str = "\n".join(buf)
                buf.clear()
                try:
                    payload = json.loads(data_str)
                except Exception:
                    payload = data_str  # plain string
                # Ignore keepalive events entirely
                if last_event in ("ping", "keepalive"):
                    last_event = None
                    continue
                if last_event == "summary" and isinstance(payload, dict):
                    summary = payload
                else:
                    if isinstance(payload, dict):
                        typ = payload.get("type")
                        if typ == "token":
                            content += payload.get("content", "")
                        elif typ == "final":
                            content = payload.get("content", content or "")
                        elif last_event == "message":
                            text = payload.get("text") or payload.get("content") or ""
                            content += text
                    elif isinstance(payload, str):
                        up = payload.strip().upper()
                        if up in ("[DONE]", "DONE"):
                            last_event = None
                            continue
                        content += payload
            last_event = None
            continue

        if line.startswith("event:"):
            # flush any buffered data before switching event types
            if buf:
                try:
                    payload = json.loads("\n".join(buf))
                except Exception:
                    payload = "\n".join(buf)
                buf.clear()
                # Ignore keepalive events entirely
                if last_event in ("ping", "keepalive"):
                    last_event = None
                elif last_event == "summary" and isinstance(payload, dict):
                    summary = payload
                else:
                    if isinstance(payload, dict):
                        typ = payload.get("type")
                        if typ == "token":
                            content += payload.get("content", "")
                        elif typ == "final":
                            content = payload.get("content", content or "")
                        elif last_event == "message":
                            text = payload.get("text") or payload.get("content") or ""
                            content += text
                    elif isinstance(payload, str):
                        up = payload.strip().upper()
                        if up in ("[DONE]", "DONE"):
                            last_event = None
                        else:
                            content += payload
            last_event = line.split(":", 1)[1].strip()
            continue
        if line.startswith("data:"):
            data_piece = line.split(":", 1)[1].lstrip()
            if last_event is None:
                # default channel: treat each data line as a complete event
                try:
                    payload = json.loads(data_piece)
                except Exception:
                    payload = data_piece
                if isinstance(payload, dict):
                    typ = payload.get("type")
                    if typ == "token":
                        content += payload.get("content", "")
                    elif typ == "final":
                        content = payload.get("content", content or "")
                elif isinstance(payload, str):
                    up = payload.strip().upper()
                    if up in ("[DONE]", "DONE"):
                        last_event = None
                        continue
                    content += payload
            else:
                # named event (e.g., summary/message): allow multi-line buffering
                if last_event in ("ping", "keepalive"):
                    # drop data for keepalive events
                    continue
                buf.append(data_piece)
            continue
        # ignore id:, retry:, etc.

    # flush at EOF
    if buf:
        try:
            payload = json.loads("\n".join(buf))
        except Exception:
            payload = "\n".join(buf)
        # Ignore keepalive events entirely
        if last_event in ("ping", "keepalive"):
            last_event = None
        elif last_event == "summary" and isinstance(payload, dict):
            summary = payload
        else:
            if isinstance(payload, dict):
                if payload.get("type") == "final":
                    content = payload.get("content", content or "")
                elif last_event == "message":
                    text = payload.get("text") or payload.get("content") or ""
                    content += text
            elif isinstance(payload, str):
                up = payload.strip().upper()
                if up not in ("[DONE]", "DONE"):
                    content += payload
    return content, summary


def execute(
    message: str,
    *,
    stream: bool = False,
    timeout_s: int = 60,
    connect_timeout_s: int = 5,
    auto: bool = True,
    ground: Optional[bool] = None,
    k: Optional[int] = None,
    cite: Optional[bool] = None,
    check: str = "off",
    consensus: Optional[int | bool] = None,
    engine: Optional[str] = None,
    eval_corpus: Optional[str] = None,
    fresh: Optional[bool] = None,
    provenance: Optional[bool] = None,
    status: Optional[bool] = None,
) -> str:
    base = _resolve_base_url()
    headers = _headers()
    # Allow environment overrides for timeouts
    _env_timeout = os.getenv("RELIABILITY_TIMEOUT_S")
    if _env_timeout:
        try:
            timeout_s = int(_env_timeout)
        except Exception:
            pass
    _env_connect_timeout = os.getenv("RELIABILITY_CONNECT_TIMEOUT_S")
    if _env_connect_timeout:
        try:
            connect_timeout_s = int(_env_connect_timeout)
        except Exception:
            pass
    timeouts = (int(connect_timeout_s), int(timeout_s))

    # Heuristic flags
    flags: Dict[str, Any] = {}
    if auto:
        flags.update(auto_reliability_flags(message, overrides={
            # only apply overrides if explicitly provided (not None) and non-default
            **({"ground": ground} if ground is not None else {}),
            **({"k": k} if k is not None else {}),
            **({"cite": cite} if cite is not None else {}),
            **({"check": check} if (check is not None and check != "off") else {}),
            **({"consensus": consensus} if consensus is not None else {}),
            **({"engine": engine} if engine is not None else {}),
            **({"eval_corpus": eval_corpus} if eval_corpus is not None else {}),
            **({"fresh": fresh} if fresh is not None else {}),
        }))
        # Output toggles (not used in payload). Only set when explicitly provided.
        if provenance is not None:
            flags["provenance"] = bool(provenance)
        if status is not None:
            flags["status"] = bool(status)
    else:
        flags = {
            "ground": bool(ground or False),
            "k": k,
            "cite": bool(cite or False),
            "check": str(check or "off"),
            # Preserve raw consensus input for normalization below
            "consensus": consensus,
            "engine": engine,
            "eval_corpus": eval_corpus,
            "fresh": bool(fresh or False) if fresh is not None else None,
            # Output toggles (not used in payload). Only set when explicitly provided.
            "provenance": (bool(provenance) if provenance is not None else None),
            "status": (bool(status) if status is not None else None),
        }

    # Normalize consensus+k coupling
    consensus_k: Optional[int] = None
    raw_consensus = flags.get("consensus")
    raw_k = flags.get("k")
    # Clamp k to schema range if provided
    if isinstance(raw_k, int):
        if raw_k < 1:
            raw_k = 1
        elif raw_k > 16:
            raw_k = 16
    if isinstance(raw_consensus, bool):
        consensus_k = (raw_k if isinstance(raw_k, int) else 3) if raw_consensus else None
    elif isinstance(raw_consensus, int):
        consensus_k = raw_consensus
    elif isinstance(raw_k, int):
        # treat user-provided k as consensus size only if consensus truthy elsewhere
        consensus_k = raw_k if (raw_consensus is True) else None
    # Clamp consensus like k
    if isinstance(consensus_k, int):
        if consensus_k < 1:
            consensus_k = 1
        elif consensus_k > 16:
            consensus_k = 16

    # Build consensus field differently for auto vs manual mode
    # Note: In auto mode we intentionally preserve boolean True to match tests/UX
    # (payload carries intent), while manual mode maps to an integer size (1..16).
    consensus_field: Dict[str, Any] = {}
    if auto:
        # Preserve boolean True from heuristics (tests expect boolean in auto mode)
        if flags.get("consensus") is True:
            consensus_field = {"consensus": True}
    else:
        raw_cons = flags.get("consensus")
        if isinstance(raw_cons, bool):
            if raw_cons is True:
                v = int(raw_k) if isinstance(raw_k, int) else 3
                if v < 1:
                    v = 1
                elif v > 16:
                    v = 16
                consensus_field = {"consensus": v}
        elif isinstance(raw_cons, int):
            v = int(raw_cons)
            if v < 1:
                v = 1
            elif v > 16:
                v = 16
            consensus_field = {"consensus": v}

    payload = {
        "message": message,
        # only include non-default truthy/explicit values to keep payload compact
        **({"ground": True} if flags.get("ground") else {}),
        **({"k": int(raw_k)} if isinstance(raw_k, int) else {}),
        **({"cite": True} if flags.get("cite") else {}),
        **({"check": flags.get("check")} if flags.get("check") and flags.get("check") != "off" else {}),
        **consensus_field,
        **({"engine": flags.get("engine")} if flags.get("engine") else {}),
        **({"eval_corpus": flags.get("eval_corpus")} if flags.get("eval_corpus") else {}),
        **({"fresh": True} if flags.get("fresh") is True else {}),
    }

    def _fail_closed_if_required(_content: str, _summary: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """If cite+ground were requested but we have no grounding/citations, block with a soft refusal."""
        want_cites = bool(flags.get("cite")) and bool(flags.get("ground"))
        grounded = isinstance(_summary, dict) and (_summary.get("grounded") is True)
        has_cites = isinstance(_summary, dict) and bool(_summary.get("citations"))
        if want_cites and (not grounded or not has_cites):
            refusal = (
                "I can’t provide a cited answer right now (no sources retrieved in time). Try again or rephrase?"
            )
            new_summary = {**(_summary or {}), "status": _summary.get("status", "no_docs"), "grounded": False, "citations": []}
            return refusal, new_summary
        return _content, _summary

    try:
        if not stream:
            url = f"{base}/v1/chat"
            h = {**_headers(), "Accept": "application/json"}
            # Observability
            h["User-Agent"] = "ollama-turbo-reliability/1.1.0"
            tid = os.getenv("TRACE_ID")
            if tid:
                h["X-Trace-Id"] = tid
            # tests expect payload via data= (JSON string), not json=
            r = requests.post(url, headers=h, data=json.dumps(payload), timeout=timeouts)
            r.raise_for_status()
            data = r.json()
            content = data.get("content", "")
            # Prefer server-provided summary exactly as-is; otherwise synthesize
            if isinstance(data.get("summary"), dict):
                summary = data.get("summary")
            else:
                summary = {
                    "grounded": bool(flags.get("ground")),
                    "citations": [],
                    "validator": None if (flags.get("check") in (None, "off")) else {"mode": flags.get("check")},
                    "consensus": None,
                }
            content, summary = _fail_closed_if_required(content, summary)
            # Optional output toggles
            if flags.get("provenance") is False and isinstance(summary, dict):
                summary.pop("provenance", None)
            if flags.get("status") is False and isinstance(summary, dict):
                summary.pop("status", None)
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
            h = {**_headers(), "Accept": "text/event-stream"}
            h["User-Agent"] = "ollama-turbo-reliability/1.1.0"
            tid = os.getenv("TRACE_ID")
            if tid:
                h["X-Trace-Id"] = tid
            # tests expect payload via data= (JSON string), not json=
            with requests.post(url, headers=h, data=json.dumps(payload), timeout=timeouts, stream=True) as r:
                r.raise_for_status()
                content, summary = _parse_stream(r)
                if not isinstance(summary, dict):
                    summary = {}
                content, summary = _fail_closed_if_required(content, summary)
                # Optional output toggles
                if flags.get("provenance") is False and isinstance(summary, dict):
                    summary.pop("provenance", None)
                if flags.get("status") is False and isinstance(summary, dict):
                    summary.pop("status", None)
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
        _code = getattr(getattr(rexc, "response", None), "status_code", None)
        _code_str = f" {_code}" if _code is not None else ""
        return _safe_json({
            "tool": "reliable_chat",
            "ok": False,
            "content": "",
            "summary": {},
            "error": {"code": "http_error", "message": str(rexc)},
            "inject": f"Reliability failed: HTTP error{_code_str}",
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
TOOL_VERSION = "1.1.0"
