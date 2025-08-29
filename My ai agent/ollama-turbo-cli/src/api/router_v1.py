from __future__ import annotations

import os
import json
import uuid
from typing import Optional
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from .models import ChatRequest, ChatResponse
from .deps import get_api_key, get_idempotency_key
from src.client import OllamaTurboClient

router = APIRouter(prefix="/v1", tags=["v1"]) 


def _resolve_tool_results_format(req_opt: Optional[dict]) -> str:
    # Request overrides env; default is 'string' for backward compatibility
    if req_opt and req_opt.get("tool_results_format") in {"string", "object"}:
        return req_opt["tool_results_format"]
    env = (os.getenv("TOOL_RESULTS_FORMAT") or "string").strip().lower()
    return "object" if env == "object" else "string"


@router.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@router.get("/live")
async def liveness() -> dict:
    # Minimal liveness probe (process up)
    return {"status": "live"}


@router.get("/ready")
async def readiness() -> dict:
    # Minimal readiness probe; in future, check downstreams if any
    return {"status": "ready"}


@router.post("/chat", response_model=ChatResponse)
async def chat(
    payload: ChatRequest,
    _api_key = Depends(get_api_key),
    idem_key: Optional[str] = Depends(get_idempotency_key),
) -> ChatResponse:
    # Instantiate client per request for now; pool/reuse can be added later
    upstream_key = os.getenv("OLLAMA_API_KEY", "") or _api_key
    client = OllamaTurboClient(
        api_key=upstream_key,
        enable_tools=True,
        quiet=True,
        ground=bool(payload.ground or False),
        k=payload.k,
        cite=bool(payload.cite or False),
        check=(payload.check or 'off'),
        consensus=bool(payload.consensus or False),
        engine=payload.engine,
        eval_corpus=payload.eval_corpus,
    )
    fmt = _resolve_tool_results_format((payload.options or {}).model_dump() if payload.options else None)
    # Respect API-level idempotency key in future by exposing a setter in client;
    # currently client generates a key internally per turn.
    content = client.chat(payload.message, stream=False)
    tool_results = None
    if fmt == "object":
        tool_results = getattr(client, "_last_tool_results_structured", None)
    else:
        tool_results = getattr(client, "_last_tool_results_strings", None)

    return ChatResponse(content=content, tool_results=tool_results or None)


@router.post("/chat/stream")
async def chat_stream(
    payload: ChatRequest,
    _api_key = Depends(get_api_key),
    idem_key: Optional[str] = Depends(get_idempotency_key),
):
    """Server-Sent Events streaming endpoint.

    Behavior:
    - Streams tokens as SSE events: {type: 'token', content: '...'}
    - If tool calls are detected or errors occur, silently falls back to non-streaming finalization
      and emits a final event: {type: 'final', content: '...'}
    """
    upstream_key = os.getenv("OLLAMA_API_KEY", "") or _api_key
    client = OllamaTurboClient(api_key=upstream_key, enable_tools=True, quiet=True)
    fmt = _resolve_tool_results_format((payload.options or {}).model_dump() if payload.options else None)

    def sse_gen():
        # Prepare conversation similar to client.chat()
        try:
            try:
                client._inject_mem0_context(payload.message)
            except Exception:
                pass
            client.conversation_history.append({'role': 'user', 'content': payload.message})
            # Reliability: clear per-request state and optionally prepare grounding context
            try:
                client._last_context_blocks = []
                client._last_citations_map = {}
            except Exception:
                pass
            if bool(payload.ground or False):
                try:
                    client._prepare_reliability_context(payload.message)
                except Exception:
                    pass
            try:
                client._trim_history()
            except Exception:
                pass
            # Set an idempotency key for this turn
            try:
                key = str(uuid.uuid4())
                client._current_idempotency_key = key
                client._set_idempotency_key(key)
            except Exception:
                pass

            # Create initial stream
            try:
                stream = client._create_streaming_response()
            except Exception:
                # Silent fallback: non-streaming
                final = client._handle_standard_chat()
                tr = getattr(client, "_last_tool_results_structured", None) if fmt == "object" else getattr(client, "_last_tool_results_strings", None)
                payload_final = {'type': 'final', 'content': final}
                if tr:
                    payload_final['tool_results'] = tr
                yield f"data: {json.dumps(payload_final)}\n\n"
                # Emit trailing summary event
                grounded = bool(getattr(client, '_last_context_blocks', []) or [])
                citations = list((getattr(client, '_last_citations_map', {}) or {}).keys()) or []
                summary = {
                    'grounded': grounded,
                    'citations': citations,
                    'validator': None,
                    'consensus': {'k': int(payload.k or 1), 'agree_rate': 1.0},
                    'status': ('no_docs' if bool(payload.ground or False) and bool(payload.cite or False) and not citations else 'ok'),
                    'provenance': ('retrieval' if grounded else 'none'),
                }
                try:
                    if (payload.check or 'off') != 'off':
                        from src.reliability.guards.validator import Validator  # local import to avoid router top-level deps
                        summary['validator'] = Validator(mode=str(payload.check or 'off')).validate(final, getattr(client, '_last_context_blocks', []))
                except Exception:
                    pass
                yield f"event: summary\n"
                yield f"data: {json.dumps(summary)}\n\n"
                return

            round_content = ""
            tool_calls_detected = False
            try:
                for chunk in stream:
                    message = chunk.get('message', {})
                    if message.get('content'):
                        piece = message['content']
                        safe_piece = client._strip_harmony_markup(piece)
                        round_content += piece
                        if safe_piece:
                            yield f"data: {json.dumps({'type': 'token', 'content': safe_piece})}\n\n"
                    # If tool calls are present, stop streaming and finalize silently
                    if message.get('tool_calls'):
                        tool_calls_detected = True
                        final = client._handle_standard_chat()
                        tr = getattr(client, "_last_tool_results_structured", None) if fmt == "object" else getattr(client, "_last_tool_results_strings", None)
                        payload_final = {'type': 'final', 'content': final}
                        if tr:
                            payload_final['tool_results'] = tr
                        yield f"data: {json.dumps(payload_final)}\n\n"
                        # Emit trailing summary event (consensus skipped due to tools)
                        grounded = bool(getattr(client, '_last_context_blocks', []) or [])
                        citations = list((getattr(client, '_last_citations_map', {}) or {}).keys()) or []
                        summary = {
                            'grounded': grounded,
                            'citations': citations,
                            'validator': None,
                            'consensus': {'k': 1, 'agree_rate': 1.0},
                            'status': ('no_docs' if bool(payload.ground or False) and bool(payload.cite or False) and not citations else 'ok'),
                            'provenance': ('retrieval' if grounded else 'none'),
                        }
                        try:
                            if (payload.check or 'off') != 'off':
                                from src.reliability.guards.validator import Validator
                                summary['validator'] = Validator(mode=str(payload.check or 'off')).validate(final, getattr(client, '_last_context_blocks', []))
                        except Exception:
                            pass
                        yield f"event: summary\n"
                        yield f"data: {json.dumps(summary)}\n\n"
                        return
            except Exception:
                # Silent fallback on read error
                final = client._handle_standard_chat()
                tr = getattr(client, "_last_tool_results_structured", None) if fmt == "object" else getattr(client, "_last_tool_results_strings", None)
                payload_final = {'type': 'final', 'content': final}
                if tr:
                    payload_final['tool_results'] = tr
                yield f"data: {json.dumps(payload_final)}\n\n"
                # Emit trailing summary event after fallback
                grounded = bool(getattr(client, '_last_context_blocks', []) or [])
                citations = list((getattr(client, '_last_citations_map', {}) or {}).keys()) or []
                summary = {
                    'grounded': grounded,
                    'citations': citations,
                    'validator': None,
                    'consensus': {'k': int(payload.k or 1), 'agree_rate': 1.0},
                    'status': 'http_error' if (bool(payload.ground or False) and bool(payload.cite or False) and not citations) else 'http_error',
                    'provenance': ('retrieval' if grounded else 'none'),
                }
                try:
                    if (payload.check or 'off') != 'off':
                        from src.reliability.guards.validator import Validator
                        summary['validator'] = Validator(mode=str(payload.check or 'off')).validate(final, getattr(client, '_last_context_blocks', []))
                except Exception:
                    pass
                yield f"event: summary\n"
                yield f"data: {json.dumps(summary)}\n\n"
                return

            # No tools; compute final text (respect Harmony final channel)
            final_out = round_content
            try:
                if round_content:
                    cleaned, _, final_seg = client._parse_harmony_tokens(round_content)
                    final_out = final_seg or cleaned or client._strip_harmony_markup(round_content)
            except Exception:
                final_out = client._strip_harmony_markup(round_content)

            # Append to history and persist memory
            client.conversation_history.append({'role': 'assistant', 'content': final_out})
            try:
                client._mem0_add_after_response(getattr(client, '_last_user_message', payload.message), final_out)
            except Exception:
                pass

            # No tool_results for pure text path
            yield f"data: {json.dumps({'type': 'final', 'content': final_out})}\n\n"
            # Emit trailing summary event (trace-only consensus)
            grounded = bool(getattr(client, '_last_context_blocks', []) or [])
            citations = list((getattr(client, '_last_citations_map', {}) or {}).keys()) or []
            summary = {
                'grounded': grounded,
                'citations': citations,
                'validator': None,
                'consensus': None,
                'status': ('no_docs' if bool(payload.ground or False) and bool(payload.cite or False) and not citations else 'ok'),
                'provenance': ('retrieval' if grounded else 'none'),
            }
            # Validator
            try:
                if (payload.check or 'off') != 'off':
                    from src.reliability.guards.validator import Validator
                    summary['validator'] = Validator(mode=str(payload.check or 'off')).validate(final_out, getattr(client, '_last_context_blocks', []))
            except Exception:
                pass
            # Consensus (trace-only): deterministic settings within client
            try:
                if bool(payload.consensus or False) and isinstance(payload.k, int) and (payload.k or 0) > 1:
                    from src.reliability.guards.consensus import run_consensus
                    def _gen_once_stream():
                        kwargs2 = {
                            'model': client.model,
                            'messages': client.conversation_history,
                        }
                        options2: dict = {}
                        if client.max_output_tokens is not None:
                            options2['num_predict'] = client.max_output_tokens
                        if client.ctx_size is not None:
                            options2['num_ctx'] = client.ctx_size
                        options2['temperature'] = 0
                        options2['top_p'] = 0
                        if options2:
                            kwargs2['options'] = options2
                        keep_val2 = client._resolve_keep_alive()
                        if keep_val2 is not None:
                            kwargs2['keep_alive'] = keep_val2
                        resp2 = client.client.chat(**kwargs2)
                        msg2 = resp2.get('message', {})
                        cont2 = msg2.get('content', '') or ''
                        try:
                            cleaned2, _, final2 = client._parse_harmony_tokens(cont2)
                            return (final2 or cleaned2 or client._strip_harmony_markup(cont2)) or ""
                        except Exception:
                            return client._strip_harmony_markup(cont2)
                    cns = run_consensus(_gen_once_stream, k=int(payload.k or 1))
                    summary['consensus'] = {'k': int(payload.k or 1), 'agree_rate': cns.get('agree_rate')}
                else:
                    summary['consensus'] = {'k': 1, 'agree_rate': 1.0}
            except Exception:
                summary['consensus'] = {'k': int(payload.k or 1), 'agree_rate': None}
            yield f"event: summary\n"
            yield f"data: {json.dumps(summary)}\n\n"
        finally:
            try:
                client._clear_idempotency_key()
            except Exception:
                pass

    return StreamingResponse(sse_gen(), media_type="text/event-stream")
