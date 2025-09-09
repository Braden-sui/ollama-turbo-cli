from __future__ import annotations

"""
Chat turn orchestrator.

Phase G: Extract the non-streaming (standard) flow into a reusable class without
changing behavior. Streaming runner can be phased in next with the same API.
"""

from typing import Any, Dict, List, cast, Optional
from .context import OrchestrationContext

from ..utils import truncate_text
from ..reliability.guards.consensus import run_consensus
from ..reliability.guards.validator import Validator
from ..types import ChatMessage, AdapterOptions
from ..contracts import to_contract
from ..routing.mode_classifier import classify as classify_mode
from ..logging.audit import write_audit_line


class ChatTurnOrchestrator:
    def _adapter_opts_from_ctx(self, ctx: OrchestrationContext) -> AdapterOptions:
        adapter_opts: AdapterOptions = {}
        if ctx.max_output_tokens is not None:
            adapter_opts['max_tokens'] = ctx.max_output_tokens
        if ctx.ctx_size is not None:
            adapter_opts['extra'] = {'num_ctx': ctx.ctx_size}
        if ctx.temperature is not None:
            adapter_opts['temperature'] = ctx.temperature
        if ctx.top_p is not None:
            adapter_opts['top_p'] = ctx.top_p
        if ctx.presence_penalty is not None:
            adapter_opts['presence_penalty'] = ctx.presence_penalty
        if ctx.frequency_penalty is not None:
            adapter_opts['frequency_penalty'] = ctx.frequency_penalty
        return adapter_opts

    def build_streaming_round_kwargs(self, ctx: OrchestrationContext, round_index: int, include_tools: bool) -> Dict[str, Any]:
        """Build kwargs for subsequent streaming rounds (round > 0). Does not set 'stream'."""
        kwargs: Dict[str, Any] = {
            'model': ctx.model,
            'messages': ctx.conversation_history,
        }
        adapter_opts = self._adapter_opts_from_ctx(ctx)
        try:
            mapped = ctx.adapter.map_options(adapter_opts) if adapter_opts else {}
            if mapped:
                kwargs['options'] = mapped
        except Exception:
            # Fallback to direct fields if adapter mapping fails
            options = {}
            if ctx.max_output_tokens is not None:
                options['num_predict'] = ctx.max_output_tokens
            if ctx.ctx_size is not None:
                options['num_ctx'] = ctx.ctx_size
            if options:
                kwargs['options'] = options
        if include_tools:
            kwargs['tools'] = ctx.tools
        keep_val = ctx._resolve_keep_alive()
        if keep_val is not None:
            kwargs['keep_alive'] = keep_val
        return kwargs

    def format_reprompt_after_tools(
        self,
        ctx: OrchestrationContext,
        payload: Dict[str, Any],
        prebuilt_msgs: Optional[List[Dict[str, Any]]],
        *,
        streaming: bool,
    ) -> None:
        """Adapter-driven tool message formatting with streaming/non-streaming parity."""
        # Best-effort: if web_research ran, extract citations/highlights into ctx for grounded reprompt
        try:
            structured: Optional[List[Dict[str, Any]]] = getattr(ctx, '_last_tool_results_structured', None)  # type: ignore
            if isinstance(structured, list):
                for tr in structured:
                    if not isinstance(tr, dict):
                        continue
                    if str(tr.get('tool') or '') != 'web_research':
                        continue
                    if str(tr.get('status') or 'ok') != 'ok':
                        continue
                    content = tr.get('content')
                    obj: Optional[Dict[str, Any]] = None
                    if isinstance(content, dict):
                        obj = content
                    elif isinstance(content, str):
                        try:
                            import json as _json
                            obj = _json.loads(content)
                        except Exception:
                            obj = None
                    if not isinstance(obj, dict):
                        continue
                    cits = obj.get('citations') or []
                    if not isinstance(cits, list):
                        continue
                    context_blocks: List[Dict[str, Any]] = []
                    citations_map: Dict[str, Any] = {}
                    for i, c in enumerate(cits, 1):
                        if not isinstance(c, dict):
                            continue
                        url = str(c.get('canonical_url') or c.get('url') or '')
                        title = str(c.get('title') or url or f"source {i}")
                        lines = c.get('lines') or []
                        snippets: List[str] = []
                        hl_structs: List[Dict[str, Any]] = []
                        if isinstance(lines, list):
                            for hl in lines[:2]:
                                try:
                                    q = (hl or {}).get('quote') or ''
                                    ln = (hl or {}).get('line')
                                    pg = (hl or {}).get('page')
                                    if q:
                                        suffix = ''
                                        if isinstance(ln, int):
                                            suffix = f" (line {ln})"
                                        if isinstance(pg, int):
                                            suffix = f" (p.{pg})"
                                        snippets.append(f"{q}{suffix}")
                                        loc = (hl or {}).get('loc')
                                        hl_structs.append({'quote': q, 'loc': loc})
                                except Exception:
                                    continue
                        block = {
                            'id': str(i),
                            'title': title,
                            'url': url,
                            'source': 'web',
                            'snippets': snippets,
                            'kind': c.get('kind'),
                            'date': c.get('date'),
                        }
                        context_blocks.append(block)
                        citations_map[str(i)] = {'url': url, 'title': title, 'highlights': hl_structs}
                    try:
                        ctx._last_context_blocks = context_blocks  # type: ignore
                        ctx._last_citations_map = citations_map    # type: ignore
                        ctx._trace(f"reliability:web:citations={len(citations_map)}")
                    except Exception:
                        pass
                    # Only process the first web_research result per turn
                    break
        except Exception:
            # Non-fatal: continue with normal adapter formatting
            pass
        try:
            adapter_opts: Optional[AdapterOptions] = None if streaming else self._adapter_opts_from_ctx(ctx)
            new_msgs, _ovr = ctx.adapter.format_reprompt_after_tools(
                cast(List[ChatMessage], ctx.conversation_history),
                payload,
                options=adapter_opts,
            )
            ctx.conversation_history = new_msgs
        except Exception as e:
            # Fallback: if we have prebuilt messages, use them; otherwise a single aggregated string
            try:
                ctx.logger.debug(f"adapter format_reprompt_after_tools failed: {e}")
            except Exception:
                pass
            try:
                ctx._trace("reprompt:after-tools:fallback")
            except Exception:
                pass
            if prebuilt_msgs:
                # prebuilt messages are dict-shaped; safe to extend
                ctx.conversation_history.extend(prebuilt_msgs)
            else:
                tool_strings = getattr(ctx, '_last_tool_results_strings', [])
                ctx.conversation_history.append({
                    'role': 'tool',
                    'content': '\n'.join(tool_strings)
                })

    def finalize_reliability_streaming(self, ctx: OrchestrationContext, final_out: str, *, tools_used: bool) -> str:
        """Apply streaming-mode reliability consensus+validator behavior and return final text."""
        try:
            if (not tools_used) and ctx.reliability.get('consensus') and isinstance(ctx.reliability.get('consensus_k'), int) and (ctx.reliability.get('consensus_k') or 0) > 1:
                def _gen_once_stream() -> str:
                    kwargs2: Dict[str, Any] = {
                        'model': ctx.model,
                        'messages': ctx.conversation_history,
                    }
                    options2: Dict[str, Any] = {}
                    if ctx.max_output_tokens is not None:
                        options2['num_predict'] = ctx.max_output_tokens
                    if ctx.ctx_size is not None:
                        options2['num_ctx'] = ctx.ctx_size
                    # Deterministic settings for consensus runs
                    options2['temperature'] = 0
                    options2['top_p'] = 0
                    if options2:
                        kwargs2['options'] = options2
                    keep_val2 = ctx._resolve_keep_alive()
                    if keep_val2 is not None:
                        kwargs2['keep_alive'] = keep_val2
                    try:
                        ctx._maybe_inject_reasoning(kwargs2)
                    except Exception:
                        pass
                    resp2 = ctx.client.chat(**kwargs2)
                    msg2 = resp2.get('message', {})
                    cont2 = msg2.get('content', '') or ''
                    try:
                        cleaned2, _, final2 = ctx._parse_harmony_tokens(cont2)
                        return (final2 or cleaned2 or ctx._strip_harmony_markup(cont2)) or ""
                    except Exception:
                        return ctx._strip_harmony_markup(cont2)
                cns = run_consensus(_gen_once_stream, k=int(ctx.reliability.get('consensus_k') or 1))
                if cns.get('final'):
                    final_out = cns['final']
                ctx._trace(f"consensus:agree_rate={cns.get('agree_rate')}")
        except Exception as ce:
            ctx.logger.debug(f"consensus skipped: {ce}")

        try:
            if (ctx.reliability.get('check') or 'off') != 'off':
                report = Validator(mode=str(ctx.reliability.get('check'))).validate(final_out, getattr(ctx, '_last_context_blocks', []))
                ctx._trace(f"validate:mode={report.get('mode')} citations={report.get('citations_present')}")
        except Exception as ve:
            ctx.logger.debug(f"validator skipped: {ve}")
        return final_out
    def handle_standard_chat(self, ctx: OrchestrationContext, *, _suppress_errors: bool = False) -> str:
        """Handle non-streaming chat interaction (delegated from streaming.standard).

        Args:
            ctx: OllamaTurboClient instance (or compatible) providing required attributes and methods.
            _suppress_errors: internal flag used by streaming fallback to reduce noisy logs.
        """
        try:
            # Generation options (reused across rounds)
            options: Dict[str, Any] = {}
            # Prefer adapter options mapping; pass ctx_size via extra.num_ctx
            adapter_opts: Dict[str, Any] = self._adapter_opts_from_ctx(ctx)
            try:
                mapped = ctx.adapter.map_options(adapter_opts) if adapter_opts else {}
                if mapped:
                    options.update(mapped)
            except Exception:
                # Fallback to direct fields if adapter mapping fails
                if ctx.max_output_tokens is not None:
                    options['num_predict'] = ctx.max_output_tokens
                if ctx.ctx_size is not None:
                    options['num_ctx'] = ctx.ctx_size

            rounds = 0
            all_tool_results: List[str] = []
            first_content: str = ""

            while True:
                # Build request (adapter-driven on initial round)
                kwargs: Dict[str, Any] = {
                    'model': ctx.model,
                }
                include_tools = ctx.enable_tools and bool(ctx.tools) and (rounds == 0)

                if rounds == 0:
                    # Decide mode for turn (router v2)
                    try:
                        self._decide_mode_for_turn(ctx)
                    except Exception:
                        pass
                    # Prepare initial messages via adapter (may merge Mem0 into first system)
                    norm_msgs, overrides = ctx._prepare_initial_messages_for_adapter(
                        include_tools=include_tools,
                        adapter_opts=adapter_opts,
                    )
                    kwargs['messages'] = norm_msgs
                    # Save for accurate r0 tracing
                    try:
                        ctx._last_sent_messages = norm_msgs
                    except Exception:
                        pass
                    # Warm models on server by requesting keep_alive if enabled
                    keep_val = ctx._resolve_keep_alive()
                    if keep_val is not None:
                        kwargs['keep_alive'] = keep_val
                    if overrides:
                        kwargs.update(overrides)
                else:
                    # Subsequent rounds: pass-through history and mapped options
                    kwargs['messages'] = ctx.conversation_history
                    keep_val = ctx._resolve_keep_alive()
                    if keep_val is not None:
                        kwargs['keep_alive'] = keep_val
                    if options:
                        kwargs['options'] = options

                # Optional request-level reasoning injection
                ctx._maybe_inject_reasoning(kwargs)

                ctx._trace(f"request:standard:round={rounds}{' tools' if include_tools else ''}")
                # Trace Mem0 presence prior to dispatch on the actual sent messages
                ctx._trace_mem0_presence(kwargs.get('messages'), f"standard:r{rounds}")
                response = ctx.client.chat(**kwargs)

                # Normalize via protocol adapter (parses tool calls and strips markup/final)
                nsr = ctx.adapter.parse_non_stream_response(response)
                message = response.get('message', {})
                content = nsr.get('content') or message.get('content', '') or ''
                tool_calls_raw = message.get('tool_calls') or []
                tool_calls = tool_calls_raw
                if not tool_calls:
                    # Convert adapter-normalized tool calls into OpenAI-style for execution/history
                    nsr_calls = nsr.get('tool_calls', []) or []
                    if nsr_calls:
                        tool_calls = []
                        for i, tc in enumerate(nsr_calls, 1):
                            name_val = tc.get('name') or ((tc.get('function') or {}).get('name')) or ''
                            args_val = tc.get('arguments') or ((tc.get('function') or {}).get('arguments'))
                            tool_calls.append({
                                'type': 'function',
                                'id': (tc.get('id') or f"call_{i}"),
                                'function': {
                                    'name': name_val,
                                    'arguments': args_val,
                                }
                            })
                # Enforce tool round limits to avoid infinite tool loops
                try:
                    max_tool_rounds = int(getattr(ctx, 'tool_max_rounds', 1) or 1)
                    if (not getattr(ctx, 'multi_round_tools', False)) and rounds >= max_tool_rounds:
                        tool_calls = []
                except Exception:
                    pass
                # Trace analysis if the adapter surfaced any (Harmony only)
                try:
                    hp = getattr(ctx.adapter, '_hp', None)
                    if hp is not None and getattr(hp, 'last_analysis', None):
                        ctx._trace(f"analysis:{truncate_text(hp.last_analysis, 180)}")
                except Exception:
                    pass

                # On first round, capture any preface content
                if rounds == 0 and content:
                    first_content = ctx._strip_harmony_markup(content)

                if tool_calls and ctx.enable_tools:
                    # Add assistant message with tool calls
                    ctx.conversation_history.append({
                        'role': 'assistant',
                        'content': ctx._strip_harmony_markup(content),
                        'tool_calls': tool_calls
                    })
                    names = [
                        (tc.get('function', {}) or {}).get('name') or tc.get('name')
                        for tc in tool_calls
                    ]
                    ctx._trace(f"tools:detected {len(tool_calls)} -> {', '.join(n for n in names if n)}")

                    # Tool-aware escalation: force researcher path for web_research/retrieval (turn-local)
                    if any(((n or '') in {'web_research', 'retrieval'}) for n in names):
                        try:
                            ctx.reliability.update({'ground': True, 'cite': True, 'check': 'enforce'})
                            meta = getattr(ctx, '_turn_mode_meta', {}) or {}
                            meta['mode_forced_by_tools'] = True
                            setattr(ctx, '_turn_mode_meta', meta)
                        except Exception:
                            pass

                    # Execute tools
                    tool_results = ctx._execute_tool_calls(tool_calls)
                    payload, prebuilt_msgs = ctx._payload_for_tools(tool_results, tool_calls)
                    tool_strings = getattr(ctx, '_last_tool_results_strings', [])
                    ctx._trace(f"tools:executed {len(tool_strings)}")
                    all_tool_results.extend(tool_strings)

                    # Adapter-driven tool message formatting, then reprompt
                    try:
                        new_msgs, _ovr = ctx.adapter.format_reprompt_after_tools(
                            ctx.conversation_history,
                            payload,
                            options=adapter_opts if isinstance(adapter_opts, dict) else None,
                        )
                        ctx.conversation_history = new_msgs
                    except Exception:
                        # Fallback: if we have prebuilt messages, use them; otherwise a single aggregated string
                        if prebuilt_msgs:
                            ctx.conversation_history.extend(prebuilt_msgs)
                        else:
                            ctx.conversation_history.append({
                                'role': 'tool',
                                'content': '\n'.join(tool_strings)
                            })

                    # Reprompt model using details (gate strict cited synthesis to contexts with citations or special tools)
                    ctx._trace("reprompt:after-tools")
                    gate_strict = False
                    try:
                        gate_strict = bool(getattr(ctx, '_last_citations_map', {}) or []) or any(n in {'web_research','retrieval'} for n in names)
                    except Exception:
                        gate_strict = False
                    reprompt_text = (
                        ctx.prompt.reprompt_after_tools() if gate_strict else (
                            'Based on the tool results, produce <|channel|>final with a clear answer. Summarize; avoid copying raw tool output.'
                        )
                    )
                    ctx.conversation_history.append({'role': 'user', 'content': reprompt_text})

                    rounds += 1
                    # Continue loop for potential additional tool rounds
                    continue
                else:
                    # Final textual answer (already normalized by adapter)
                    final_out = content or ctx._strip_harmony_markup(message.get('content', '') or '')

                    # Reliability integrations (non-streaming): consensus and validator
                    try:
                        # Skip consensus if tools were involved to avoid voting on a different path
                        tools_used_this_turn = bool(all_tool_results)
                        if (not tools_used_this_turn) and ctx.reliability.get('consensus') and isinstance(ctx.reliability.get('consensus_k'), int) and (ctx.reliability.get('consensus_k') or 0) > 1:
                            def _gen_once() -> str:
                                kwargs2: Dict[str, Any] = {
                                    'model': ctx.model,
                                    'messages': ctx.conversation_history,
                                }
                                options2: Dict[str, Any] = {}
                                if ctx.max_output_tokens is not None:
                                    options2['num_predict'] = ctx.max_output_tokens
                                if ctx.ctx_size is not None:
                                    options2['num_ctx'] = ctx.ctx_size
                                # Deterministic settings for consensus runs
                                options2['temperature'] = 0
                                options2['top_p'] = 0
                                if options2:
                                    kwargs2['options'] = options2
                                keep_val2 = ctx._resolve_keep_alive()
                                if keep_val2 is not None:
                                    kwargs2['keep_alive'] = keep_val2
                                # Optional request-level reasoning injection
                                ctx._maybe_inject_reasoning(kwargs2)
                                # Do not include tools for consensus finalization
                                resp2 = ctx.client.chat(**kwargs2)
                                msg2 = resp2.get('message', {})
                                cont2 = msg2.get('content', '') or ''
                                try:
                                    cleaned2, _, final2 = ctx._parse_harmony_tokens(cont2)
                                    return (final2 or cleaned2 or ctx._strip_harmony_markup(cont2)) or ""
                                except Exception:
                                    return ctx._strip_harmony_markup(cont2)
                            cns = run_consensus(_gen_once, k=int(ctx.reliability.get('consensus_k') or 1))
                            if cns.get('final'):
                                final_out = cns['final']
                            ctx._trace(f"consensus:agree_rate={cns.get('agree_rate')}")
                    except Exception as ce:
                        ctx.logger.debug(f"consensus skipped: {ce}")

                    report = None
                    try:
                        if (ctx.reliability.get('check') or 'off') != 'off':
                            report = Validator(mode=str(ctx.reliability.get('check'))).validate(
                                final_out,
                                getattr(ctx, '_last_context_blocks', []),
                                getattr(ctx, '_last_citations_map', {}),
                            )
                            ctx._trace(f"validate:mode={report.get('mode')} citations={report.get('citations_present')}")
                    except Exception as ve:
                        ctx.logger.debug(f"validator skipped: {ve}")
                    # Auto-repair (enforce mode only): one pass
                    try:
                        if isinstance(report, dict) and (ctx.reliability.get('check') == 'enforce') and (not report.get('ok')):
                            bad = [d for d in (report.get('details') or []) if not d.get('passed')]
                            if bad:
                                repair_instructions = (
                                    "One or more cited sentences lacked sufficient overlap with sources. "
                                    "Please either (a) quote directly from one cited source (with [n]), (b) soften with hedging, or (c) remove the claim. "
                                    "Keep inline [n] and ensure numeric claims match exactly after normalization."
                                )
                                ctx.conversation_history.append({'role': 'user', 'content': repair_instructions})
                                resp_fix = ctx.client.chat(model=ctx.model, messages=ctx.conversation_history)
                                msg_fix = resp_fix.get('message', {})
                                fixed_text = msg_fix.get('content', '') or ''
                                # Revalidate once
                                report2 = Validator(mode='enforce').validate(
                                    fixed_text,
                                    getattr(ctx, '_last_context_blocks', []),
                                    getattr(ctx, '_last_citations_map', {}),
                                )
                                # Prefer repaired if ok, else keep original
                                if isinstance(report2, dict) and report2.get('ok'):
                                    final_out = fixed_text
                                    report = report2
                                    ctx._trace("validator:repair:applied")
                                else:
                                    ctx._trace("validator:repair:failed")
                    except Exception:
                        pass

                    ctx.conversation_history.append({
                        'role': 'assistant',
                        'content': final_out
                    })
                    if all_tool_results:
                        ctx._trace(f"tools:used={len(all_tool_results)}")
                    else:
                        ctx._trace("tools:none")
                    # Persist memory to Mem0
                    ctx._mem0_add_after_response(ctx._last_user_message, final_out)

                    if all_tool_results:
                        prefix = (first_content + "\n\n") if first_content else ""
                        out = f"{prefix}[Tool Results]\n" + '\n'.join(all_tool_results) + f"\n\n{final_out}"
                    else:
                        out = final_out
                    # Audit log: store minimal fields
                    try:
                        mode_meta = getattr(ctx, '_turn_mode_meta', {}) or {}
                        citations_map = getattr(ctx, '_last_citations_map', {}) or {}
                        cits = []
                        for idx, ref in (citations_map.items() if isinstance(citations_map, dict) else []):
                            cits.append({'n': idx, 'title': (ref or {}).get('title'), 'url': (ref or {}).get('url'), 'highlights': []})
                        # Aggregated overlap telemetry
                        overlap = (report.get('details') if isinstance(report, dict) else None)
                        hist = {}
                        gated = 0
                        repaired = 0
                        val_fails = 0
                        try:
                            for d in (overlap or []):
                                s = float(d.get('overlap_score') or 0.0)
                                b = f"{int(s*5)/5:.1f}-{int((s*5)+1)/5:.1f}"
                                hist[b] = hist.get(b, 0) + 1
                                if not d.get('passed'):
                                    gated += 1
                                if d.get('repair_action'):
                                    repaired += 1
                                if d.get('value_tokens_present') and (not d.get('numeric_matched')):
                                    val_fails += 1
                        except Exception:
                            pass
                        write_audit_line(
                            mode=str(mode_meta.get('mode') or ('researcher' if ctx.reliability.get('ground') else 'standard')),
                            query=(getattr(ctx, '_last_user_message', '') or ''),
                            answer=out,
                            citations=cits,
                            metrics={'sources': len(cits), 'overlap': overlap, 'overlap_histogram': hist, 'num_sentences_gated': gated, 'num_sentences_repaired': repaired, 'value_mismatch_fails': val_fails},
                            router={'score': mode_meta.get('score'), 'details': mode_meta.get('details')},
                        )
                    except Exception:
                        pass
                    return out

        except Exception as e:
            # In streaming fallback contexts, avoid noisy error logs
            if _suppress_errors:
                ctx.logger.debug(f"Standard chat error (suppressed): {e}")
            else:
                ctx.logger.error(f"Standard chat error: {e}")
            ctx._trace(f"standard:error {type(e).__name__}")
            raise

    def _decide_mode_for_turn(self, ctx: OrchestrationContext) -> None:
        # Resolve forced/default contract via env/ctx
        import os
        forced = os.getenv('FORCED_CONTRACT')
        default = os.getenv('DEFAULT_CONTRACT') or 'researcher'
        # Extract last user message
        user_msg = ''
        for m in reversed(ctx.conversation_history or []):
            if (m or {}).get('role') == 'user':
                user_msg = str((m or {}).get('content') or '')
                break
        planned_tools: List[str] = []
        # Classify
        if forced:
            mode = forced.strip().lower()
            score = 1.0 if mode == 'researcher' else 0.0
            reason = {'forced': True}
        else:
            mode, score, reason = classify_mode(user_msg, session_id='default', planned_tools=planned_tools)
        # Apply contract for this turn (override ctx.reliability transiently)
        contract = to_contract(mode or default)
        ctx.reliability.update({'ground': contract.ground, 'cite': contract.cite, 'check': contract.check})
        # Governance profile tweaks
        try:
            profile = (os.getenv('GOVERNANCE_PROFILE') or '').strip().lower()
            if profile == 'strict':
                ctx.reliability.update({'check': 'enforce', 'consensus': True, 'consensus_k': 3})
            elif profile == 'creative':
                ctx.reliability.update({'check': 'off'})
        except Exception:
            pass
        # Stash for audit
        try:
            setattr(ctx, '_turn_mode_meta', {'mode': mode, 'score': score, 'details': reason})
        except Exception:
            pass
