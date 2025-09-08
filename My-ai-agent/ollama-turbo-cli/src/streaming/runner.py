"""
Streaming runner. Phase C extracted `_create_streaming_response` and the streaming loop here
without behavior changes. Functions accept a `ctx` (client) instance.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, cast
import json

from ..utils import RetryableError, truncate_text
from ..reliability.guards.consensus import run_consensus
from ..reliability.guards.validator import Validator
from ..orchestration.orchestrator import ChatTurnOrchestrator
from ..orchestration.context import OrchestrationContext


def create_streaming_response(ctx: OrchestrationContext):
    """Create a streaming response from the API (extracted from client)."""
    try:
        kwargs: Dict[str, Any] = {
            'model': ctx.model,
            'stream': True,
        }
        # Prefer adapter options mapping; pass ctx_size via extra.num_ctx
        adapter_opts: Dict[str, Any] = {}
        if ctx.max_output_tokens is not None:
            adapter_opts['max_tokens'] = ctx.max_output_tokens
        if ctx.ctx_size is not None:
            adapter_opts['extra'] = {'num_ctx': ctx.ctx_size}
        # Sampling parameters
        if ctx.temperature is not None:
            adapter_opts['temperature'] = ctx.temperature
        if ctx.top_p is not None:
            adapter_opts['top_p'] = ctx.top_p
        if ctx.presence_penalty is not None:
            adapter_opts['presence_penalty'] = ctx.presence_penalty
        if ctx.frequency_penalty is not None:
            adapter_opts['frequency_penalty'] = ctx.frequency_penalty

        # Use adapter to format initial messages and merge Mem0 if needed
        norm_msgs, overrides = ctx._prepare_initial_messages_for_adapter(
            include_tools=(ctx.enable_tools and bool(ctx.tools)),
            adapter_opts=adapter_opts,
        )
        kwargs['messages'] = norm_msgs
        # Save for accurate r0 tracing in streaming handler
        try:
            ctx._last_sent_messages = norm_msgs
        except Exception:
            pass
        if overrides:
            kwargs.update(overrides)

        keep_val = ctx._resolve_keep_alive()
        if keep_val is not None:
            kwargs['keep_alive'] = keep_val
        ctx._trace("request:stream:start")
        # Trace Mem0 presence prior to initial streaming dispatch
        ctx._trace_mem0_presence(kwargs.get('messages'), "stream:init")
        # Optional request-level reasoning injection
        ctx._maybe_inject_reasoning(kwargs)
        return ctx.client.chat(**kwargs)
    except Exception as e:
        raise RetryableError(f"Failed to create streaming response: {e}")


def handle_streaming_response(ctx: OrchestrationContext, response_stream, tools_enabled: bool = True) -> str:
    """Complete streaming response handler with tool call support (multi-round optional)."""
    orch = ChatTurnOrchestrator()
    rounds = 0
    aggregated_results: List[str] = []
    preface_content: str = ""
    # Keep a buffer for fallback paths
    full_content: str = ""
    try:
        while True:
            include_tools = tools_enabled and bool(ctx.tools) and (
                rounds == 0 or (ctx.multi_round_tools and rounds < ctx.tool_max_rounds)
            )

            # Use provided stream on the first round; create new ones subsequently
            if rounds == 0:
                stream = response_stream
                # Trace Mem0 presence for the initial streaming round (r0) using actual sent messages if available
                try:
                    sent_msgs = getattr(ctx, '_last_sent_messages', None) or ctx.conversation_history
                except Exception:
                    sent_msgs = ctx.conversation_history
                ctx._trace_mem0_presence(sent_msgs, "stream:r0")
            else:
                # Build subsequent round kwargs via orchestrator (parity with standard path)
                kwargs = orch.build_streaming_round_kwargs(ctx, round_index=rounds, include_tools=include_tools)
                # Trace Mem0 presence prior to subsequent streaming round dispatch
                ctx._trace_mem0_presence(kwargs.get('messages'), f"stream:r{rounds}")
                # Set streaming flag (only difference for streaming path)
                kwargs['stream'] = True
                stream = ctx.client.chat(**kwargs)

            ctx._trace(f"request:stream:round={rounds}{' tools' if include_tools else ''}")

            # Streaming print control: print live on no-tools rounds, and on round 0 until a tool_call is detected
            # Maintain both cleaned (for printing) and raw (for Harmony tool-call parsing) buffers
            round_content = ""
            round_content_clean = ""
            round_raw = ""
            # Track already emitted content length to avoid duplicates across reconnects
            printed_len = 0
            tool_calls: List[Dict[str, Any]] = []
            tool_calls_detected = False
            printed_prefix = False
            end_round_for_tools = False

            # Trace verbosity controls (per-round budgets and de-duplication)
            trace_events_budget = 6
            trace_keys_budget = 4
            trace_raw_logged = False
            _last_types_sig: Optional[str] = None
            _last_keys_sig: Optional[str] = None

            def _iter_stream_chunks(s):
                # Simple wrapper to let us hook per-chunk accounting
                for ch in s:
                    yield ch

            try:
                for chunk in _iter_stream_chunks(stream):
                    # Normalize chunk: some providers emit JSON strings/bytes or SSE lines
                    ck = chunk
                    # Reset events for each chunk to avoid leaking from previous chunks/rounds
                    events: List[Any] = []
                    raw_str = None
                    try:
                        if isinstance(chunk, (bytes, bytearray)):
                            raw_str = chunk.decode('utf-8', errors='ignore')
                        elif isinstance(chunk, str):
                            raw_str = chunk
                        # Parse JSON string bodies
                        if raw_str is not None:
                            s_trim = raw_str.lstrip()
                            if s_trim.startswith('{'):
                                ck = json.loads(s_trim)
                            else:
                                # Try SSE format: lines beginning with 'data:'
                                if 'data:' in raw_str:
                                    events_sse = []
                                    for ln in raw_str.splitlines():
                                        if not ln:
                                            continue
                                        if ln.startswith('data:'):
                                            payload = ln[len('data:'):].strip()
                                            if not payload or payload == '[DONE]':
                                                continue
                                            try:
                                                obj = json.loads(payload)
                                                # Recurse through adapter for each parsed object
                                                for ev in (list(ctx.adapter.parse_stream_events(obj)) or []):
                                                    events_sse.append(ev)
                                                # Also support simple {type,content}
                                                if not events_sse and isinstance(obj, dict) and obj.get('content') is not None:
                                                    events_sse.append({'type': 'token', 'content': str(obj.get('content') or '')})
                                            except Exception:
                                                # As last resort, treat as a token line
                                                events_sse.append({'type': 'token', 'content': payload})
                                    if events_sse:
                                        # Fast-path deliver parsed SSE events
                                        events = events_sse
                                        # Process events block below
                                        pass
                    except Exception:
                        ck = chunk

                    if not events:
                        try:
                            events = list(ctx.adapter.parse_stream_events(ck)) or []
                        except Exception:
                            events = []
                    # If still no events, coerce non-dict chunk objects and parse `.data` payloads
                    if not events and (not isinstance(ck, dict)):
                        try:
                            coerced = None
                            # Pydantic v2
                            if hasattr(ck, 'model_dump') and callable(getattr(ck, 'model_dump', None)):
                                coerced = ck.model_dump()
                            # Pydantic v1 or similar
                            elif hasattr(ck, 'dict') and callable(getattr(ck, 'dict', None)):
                                coerced = ck.dict()
                            # Simple object with __dict__
                            elif hasattr(ck, '__dict__'):
                                coerced = {k: v for k, v in vars(ck).items() if not str(k).startswith('_')}
                            if isinstance(coerced, dict):
                                ck = coerced
                            else:
                                # Some SSE libraries expose a `.data` attribute containing the JSON payload
                                data_payload = getattr(ck, 'data', None)
                                if data_payload is not None:
                                    try:
                                        if isinstance(data_payload, (bytes, bytearray)):
                                            ds = data_payload.decode('utf-8', errors='ignore')
                                        else:
                                            ds = str(data_payload)
                                        ds_trim = ds.lstrip()
                                        if ds_trim.startswith('{'):
                                            try:
                                                ck = json.loads(ds_trim)
                                            except Exception:
                                                pass
                                        elif 'data:' in ds:
                                            # SSE-style payload with data: lines
                                            events_sse2 = []
                                            for ln in ds.splitlines():
                                                if not ln:
                                                    continue
                                                if ln.startswith('data:'):
                                                    payload = ln[len('data:'):].strip()
                                                    if not payload or payload == '[DONE]':
                                                        continue
                                                    try:
                                                        obj = json.loads(payload)
                                                        for ev in (list(ctx.adapter.parse_stream_events(obj)) or []):
                                                            events_sse2.append(ev)
                                                        # Also support simple {type,content}
                                                        if (not events_sse2) and isinstance(obj, dict) and obj.get('content') is not None:
                                                            events_sse2.append({'type': 'token', 'content': str(obj.get('content') or '')})
                                                    except Exception:
                                                        events_sse2.append({'type': 'token', 'content': payload})
                                            if events_sse2:
                                                events = events_sse2
                                    except Exception:
                                        pass
                        except Exception:
                            pass
                    # If we coerced to a dict or still have none, give adapter another chance
                    if not events and isinstance(ck, dict):
                        try:
                            events = list(ctx.adapter.parse_stream_events(ck)) or []
                        except Exception:
                            events = []
                    # Fallbacks if adapter yields nothing
                    if not events and isinstance(ck, dict):
                        # Optional lightweight tracing of chunk keys for diagnostics
                        try:
                            if ctx.show_trace:
                                # Only emit occasionally to avoid noise; suppress pure-metrics chunks
                                kset = set(ck.keys())
                                METRIC_KEYS = {
                                    "model","created_at","done","done_reason",
                                    "total_duration","load_duration",
                                    # Common Ollama metrics
                                    "prompt_eval_count","prompt_eval_duration",
                                    "eval_count","eval_duration"
                                }
                                metrics_only = kset.issubset(METRIC_KEYS)
                                if not metrics_only:
                                    keys = ','.join(list(ck.keys())[:6])
                                    ctx._trace(f"stream:chunk:keys={keys}")
                        except Exception:
                            pass
                        # 1) OpenAI-style consolidated message
                        msg = ck.get('message') or {}
                        if msg.get('content'):
                            events.append({'type': 'token', 'content': ctx._strip_harmony_markup(str(msg.get('content')))})
                        for tc in (msg.get('tool_calls') or []) or []:
                            fn = (tc or {}).get('function') or {}
                            events.append({
                                'type': 'tool_call',
                                'id': str(tc.get('id') or ''),
                                'name': str(fn.get('name') or ''),
                                'arguments': fn.get('arguments')
                            })
                        # 1.5) Ollama-style streaming: top-level 'response' token pieces
                        if not events and (ck.get('response') is not None):
                            events.append({'type': 'token', 'content': str(ck.get('response') or '')})
                        # Optional: top-level tool_calls if present
                        for tc in (ck.get('tool_calls') or []) or []:
                            fn = (tc or {}).get('function') or {}
                            events.append({
                                'type': 'tool_call',
                                'id': str(tc.get('id') or ''),
                                'name': str(fn.get('name') or ''),
                                'arguments': fn.get('arguments')
                            })
                        # 2) SSE-style {type:'token'|'final', content}
                        t = ck.get('type')
                        if not events and t in {'token', 'final'} and (ck.get('content') is not None):
                            events.append({'type': 'token', 'content': str(ck.get('content') or '')})
                        # 3) Top-level content field
                        if not events and (ck.get('content') is not None):
                            events.append({'type': 'token', 'content': str(ck.get('content') or '')})
                        # 4) choices[0].message.content (non-delta streaming or non-stream-like chunk)
                        if not events and (ck.get('choices') is not None):
                            try:
                                choices = ck.get('choices') or []
                                if choices:
                                    message0 = (choices[0] or {}).get('message') or {}
                                    cont0 = message0.get('content')
                                    if cont0:
                                        events.append({'type': 'token', 'content': str(cont0)})
                                    for tc in (message0.get('tool_calls') or []) or []:
                                        fn = (tc or {}).get('function') or {}
                                        events.append({
                                            'type': 'tool_call',
                                            'id': str(tc.get('id') or ''),
                                            'name': str(fn.get('name') or ''),
                                            'arguments': fn.get('arguments')
                                        })
                            except Exception:
                                pass
                    # Trace parsed events for diagnostics (types and count) after normalization
                    try:
                        if ctx.show_trace:
                            # events types trace — only on signature changes and within budget
                            types = ','.join([
                                str(ev.get('type') if isinstance(ev, dict) else type(ev).__name__)
                                for ev in (events or [])
                            ][:6])
                            types_sig = f"{len(events)}|{types}"
                            if trace_events_budget > 0 and types_sig != _last_types_sig:
                                ctx._trace(f"stream:events n={len(events)} types={types}")
                                _last_types_sig = types_sig
                                trace_events_budget -= 1
                            # chunk keys/type trace — only when non-metrics and signature changes within budget
                            try:
                                if isinstance(ck, dict):
                                    kset = set(ck.keys())
                                    METRIC_KEYS = {
                                        "model","created_at","done","done_reason",
                                        "total_duration","load_duration",
                                        # Common Ollama metrics
                                        "prompt_eval_count","prompt_eval_duration",
                                        "eval_count","eval_duration"
                                    }
                                    metrics_only = kset.issubset(METRIC_KEYS)
                                    if not metrics_only and trace_keys_budget > 0:
                                        keys = ','.join(list(ck.keys())[:6])
                                        if keys != _last_keys_sig:
                                            ctx._trace(f"stream:chunk:keys={keys}")
                                            _last_keys_sig = keys
                                            trace_keys_budget -= 1
                                else:
                                    # Only emit type changes within budget
                                    ctype = type(ck).__name__
                                    if trace_keys_budget > 0 and ctype != _last_keys_sig:
                                        ctx._trace(f"stream:chunk:type={ctype}")
                                        _last_keys_sig = ctype
                                        trace_keys_budget -= 1
                            except Exception:
                                pass
                    except Exception:
                        pass
                    # If still nothing and we have a raw string, trace prefix for diagnostics (once per round)
                    if (not events) and (raw_str is not None) and ctx.show_trace and (not trace_raw_logged):
                        try:
                            preview = raw_str[:80].replace('\n', ' ')
                            ctx._trace(f"stream:chunk:raw={preview}")
                            trace_raw_logged = True
                        except Exception:
                            pass

                    for ev in events:
                        et = ev.get('type') if isinstance(ev, dict) else 'token'
                        # Treat both token and final events as printable/content-bearing
                        if et in {'token', 'final'}:
                            piece = str((ev.get('content') if isinstance(ev, dict) else ev) or '')
                            if not piece:
                                continue
                            round_content += piece
                            # Capture raw content for tool-call parsing when tools are enabled
                            try:
                                # Use the normalized/coerced chunk (ck) rather than the original chunk
                                # so we preserve provider-native payloads (incl. Harmony markup) when present.
                                if include_tools and isinstance(ck, dict):
                                    raw_msg = (ck.get('message') or {}).get('content')
                                    if raw_msg:
                                        round_raw += str(raw_msg)
                            except Exception:
                                pass
                            # Maintain a cleaned buffer specifically for printing to avoid leaking markup tokens
                            try:
                                clean_piece = ctx._strip_harmony_markup(piece)
                            except Exception:
                                clean_piece = piece
                            round_content_clean += clean_piece

                            # Compute the new printable segment before any potential tool-call detection
                            new_segment_pre = ""
                            if not ctx.quiet:
                                try:
                                    new_segment_pre = round_content_clean[printed_len:]
                                except Exception:
                                    new_segment_pre = ""

                            # Early incremental Harmony tool-call detection: parse as the buffer grows
                            if include_tools and (round_raw or round_content):
                                try:
                                    source_for_parse = round_raw or round_content
                                    cleaned_inc, parsed_calls_inc, _ = ctx._parse_harmony_tokens(source_for_parse)
                                    # Trace analysis if available
                                    try:
                                        if getattr(ctx.harmony, 'last_analysis', None):
                                            ctx._trace(f"analysis:{truncate_text(ctx.harmony.last_analysis, 180)}")
                                    except Exception:
                                        pass
                                    if parsed_calls_inc:
                                        tool_calls = parsed_calls_inc
                                        tool_calls_detected = True
                                        # Replace round_content with cleaned to remove commentary segments
                                        round_content = cleaned_inc or round_content
                                        try:
                                            round_content_clean = ctx._strip_harmony_markup(round_content)
                                        except Exception:
                                            pass
                                        # Print any accumulated sanitized content up to detection before aborting
                                        if not ctx.quiet:
                                            try:
                                                new_segment = round_content_clean[printed_len:]
                                            except Exception:
                                                new_segment = ""
                                            if new_segment:
                                                if not printed_prefix:
                                                    print("🤖 Assistant: ", end="", flush=True)
                                                    printed_prefix = True
                                                print(new_segment, end="", flush=True)
                                                printed_len = len(round_content_clean)
                                                try:
                                                    if ctx.show_trace:
                                                        ctx._trace(f"stream:print bytes={len(new_segment)} total={printed_len}")
                                                except Exception:
                                                    pass
                                        # Abort further streaming for this round; execute tools immediately
                                        end_round_for_tools = True
                                        break
                                except Exception:
                                    pass
                            if not ctx.quiet and not tool_calls_detected:
                                # Print only the new suffix not yet emitted (handles reconnect duplicates)
                                # Print live until a tool call is detected in the current round
                                if new_segment_pre:
                                    if not printed_prefix:
                                        print("🤖 Assistant: ", end="", flush=True)
                                        printed_prefix = True
                                    # Always print sanitized tokens (no Harmony markup leakage)
                                    print(new_segment_pre, end="", flush=True)
                                    printed_len = len(round_content_clean)
                                    # Trace printing metrics
                                    try:
                                        if ctx.show_trace:
                                            ctx._trace(f"stream:print bytes={len(new_segment_pre)} total={printed_len}")
                                    except Exception:
                                        pass
                        elif et == 'tool_call' and include_tools:
                            tool_calls_detected = True
                            if isinstance(ev, dict):
                                # Normalize to OpenAI-style for tool executor/history
                                oc = {
                                    'type': 'function',
                                    'id': ev.get('id') or '',
                                    'function': {
                                        'name': ev.get('name') or '',
                                        'arguments': ev.get('arguments')
                                    }
                                }
                                # Merge updates for same id
                                existing = False
                                for tc in tool_calls:
                                    if tc.get('id') == oc.get('id'):
                                        tc.update(oc)
                                        existing = True
                                        break
                                if not existing:
                                    tool_calls.append(oc)
                                # Trace tool-call delta detection
                                try:
                                    if ctx.show_trace:
                                        func_raw = oc.get('function')
                                        fname = None
                                        if isinstance(func_raw, dict):
                                            fname = func_raw.get('name')
                                        ctx._trace(f"tools:delta:detected id={oc.get('id')} name={fname}")
                                except Exception:
                                    pass
                            # End this streaming round immediately; execute tools next
                            end_round_for_tools = True
                            break
                    # If an early tool-call was detected, end chunk consumption for this round
                    # by breaking out of the for-chunk loop now.
                    if end_round_for_tools:
                        break
                    # Respect provider 'done' flag strictly when True (do not stop merely on presence)
                    try:
                        if isinstance(ck, dict) and (ck.get('done') is True):
                            break
                    except Exception:
                        pass
            except Exception as se:
                # Stream interrupted (timeout/connection), attempt reconnects handled by outer retry decorator for init only.
                # Here we fall back to non-streaming finalization to preserve UX, per user preference.
                ctx.logger.debug(f"Streaming read error; falling back to non-streaming: {se}")
                ctx._trace("stream:read:error -> fallback")
                try:
                    final = ctx._handle_standard_chat(_suppress_errors=True)
                    # Suppress printing raw error text returned by non-streaming fallback
                    if final and not str(final).startswith("Error during chat:") and not ctx.quiet:
                        print(final)
                    ctx._trace("fallback:success")
                    if final and not str(final).startswith("Error during chat:"):
                        return (round_content + "\n" + final) if round_content else final
                    # If fallback failed or returned error text, suppress to avoid leaking in stream
                    return round_content or ""
                except Exception as e2:
                    ctx.logger.debug(f"Non-streaming fallback also failed: {e2}")
                    ctx._trace("fallback:error")
                    return round_content or ""

            # Update outer buffer for fallback usage
            full_content = round_content

            # Capture first round content as a preface (not printed yet)
            if rounds == 0 and round_content:
                preface_content = ctx._strip_harmony_markup(round_content)

            # If tools were requested and yielded calls, execute them and loop
            # If no canonical tool_calls but Harmony markup is present, parse it
            if not tool_calls and include_tools and (round_raw or round_content):
                try:
                    # Parse Harmony tokens from raw to preserve tool-call markers
                    source_for_parse = round_raw or round_content
                    cleaned, parsed_calls, _ = ctx._parse_harmony_tokens(source_for_parse)
                    # Trace analysis if available
                    try:
                        if getattr(ctx.harmony, 'last_analysis', None):
                            ctx._trace(f"analysis:{truncate_text(ctx.harmony.last_analysis, 180)}")
                    except Exception:
                        pass
                    if parsed_calls:
                        tool_calls = parsed_calls
                        round_content = cleaned
                except Exception:
                    pass

            if tool_calls:
                if not ctx.quiet:
                    print("\n🔧 Processing tool calls...")
                names = [tc.get('function', {}).get('name') for tc in tool_calls]
                ctx._trace(f"tools:detected {len(tool_calls)} -> {', '.join(n for n in names if n)}")

                # Add assistant message with tool calls
                ctx.conversation_history.append({
                    'role': 'assistant',
                    'content': ctx._strip_harmony_markup(round_content),
                    'tool_calls': tool_calls
                })
                # Execute tools
                tool_results = ctx._execute_tool_calls(tool_calls)
                payload, prebuilt_msgs = ctx._payload_for_tools(tool_results, tool_calls)
                tool_strings = getattr(ctx, '_last_tool_results_strings', [])
                ctx._trace(f"tools:executed {len(tool_strings)}")
                aggregated_results.extend(tool_strings)
                # Print tool results immediately during streaming
                if not ctx.quiet and tool_strings:
                    print("[Tool Results]")
                    for s in tool_strings:
                        try:
                            limit = int(ctx.tool_print_limit or 0)
                        except Exception:
                            limit = 0
                        out = s
                        if limit and len(out) > limit:
                            out = out[:limit].rstrip() + "…"
                        print(out)
                    # spacer before reprompted assistant output
                    print()
                # Adapter-driven tool message formatting, then reprompt
                # Note: streaming intentionally passes options=None here to preserve existing behavior,
                # while the non-streaming path passes adapter_opts (see standard.py) for adapter parity.
                # Adapter-driven tool message formatting (streaming preserves options=None)
                orch.format_reprompt_after_tools(ctx, payload, prebuilt_msgs, streaming=True)
                # Reprompt model to synthesize an answer using tool details
                ctx._trace("reprompt:after-tools")
                ctx.conversation_history.append({
                    'role': 'user',
                    'content': ctx.prompt.reprompt_after_tools()
                })
                rounds += 1
                # Next loop iteration may include tools again if multi-round enabled
                continue

            # Empty-stream fallback: if no tool calls were detected and the provider yielded
            # no content tokens for this round, attempt a non-streaming fallback to produce
            # a final message so the agent doesn't appear silent.
            if (not tool_calls) and (not (round_content or '').strip()):
                try:
                    ctx._trace("stream:empty -> fallback:standard")
                except Exception:
                    pass
                try:
                    final = ctx._handle_standard_chat(_suppress_errors=True)
                    # Suppress obvious error strings to avoid leaking internal errors to users
                    if final and not str(final).startswith("Error during chat:"):
                        # Print once if nothing has been printed yet
                        if (not printed_prefix) and (not ctx.quiet):
                            print("🤖 Assistant: ", end="", flush=True)
                            print(ctx._strip_harmony_markup(final), flush=True)
                            printed_prefix = True
                        # Append to history and persist memory
                        ctx.conversation_history.append({'role': 'assistant', 'content': final})
                        ctx._mem0_add_after_response(ctx._last_user_message, final)
                        # Return aggregated tool results (if any) with the final
                        if aggregated_results:
                            prefix = (preface_content + "\n\n") if preface_content else ""
                            return f"{prefix}[Tool Results]\n" + '\n'.join(aggregated_results) + f"\n\n{final}"
                        return final
                except Exception:
                    # Fall through to normal finalization path below (may return empty string)
                    pass

            # No tool calls -> final textual answer for this turn
            if printed_prefix and not ctx.quiet:
                print()  # newline after final streamed content
            # Extract final-channel text if present; otherwise strip markup
            final_out = round_content
            try:
                if round_content:
                    cleaned, _, final_seg = ctx._parse_harmony_tokens(round_content)
                    # Trace analysis if available
                    try:
                        if getattr(ctx.harmony, 'last_analysis', None):
                            ctx._trace(f"analysis:{truncate_text(ctx.harmony.last_analysis, 180)}")
                    except Exception:
                        pass
                    final_out = final_seg or cleaned
                    if not final_out:
                        final_out = ctx._strip_harmony_markup(round_content)
            except Exception:
                final_out = ctx._strip_harmony_markup(round_content)

            # If nothing was printed live, print final once for good UX
            if (not printed_prefix) and (not ctx.quiet) and final_out:
                print("🤖 Assistant: ", end="", flush=True)
                print(ctx._strip_harmony_markup(final_out), flush=True)

            # Reliability integrations (streaming): consensus+validator without altering streamed text
            tools_used_stream = bool(aggregated_results)
            final_out = orch.finalize_reliability_streaming(ctx, final_out, tools_used=tools_used_stream)

            ctx.conversation_history.append({
                'role': 'assistant',
                'content': final_out
            })
            if aggregated_results:
                ctx._trace(f"tools:used={len(aggregated_results)}")
            else:
                ctx._trace("tools:none")
            # Persist memory to Mem0
            ctx._mem0_add_after_response(ctx._last_user_message, final_out)

            if aggregated_results:
                prefix = (preface_content + "\n\n") if preface_content else ""
                return f"{prefix}[Tool Results]\n" + '\n'.join(aggregated_results) + f"\n\n{final_out}"
            return final_out
    except Exception as e:
        # Do not leak streaming errors to CLI; rely on standard handler fallback path
        try:
            ctx._trace(f"stream:error {type(e).__name__}")
        except Exception:
            pass
        return full_content or ""
