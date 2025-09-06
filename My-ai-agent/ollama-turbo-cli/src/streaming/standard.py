"""
Standard (non-streaming) chat path scaffolding.

Phase C extracted `_handle_standard_chat` here without changing behavior.
"""

from typing import Any, Dict, List

from ..utils import truncate_text
from ..reliability.guards.consensus import run_consensus
from ..reliability.guards.validator import Validator


def handle_standard_chat(ctx, *, _suppress_errors: bool = False) -> str:
    """Handle non-streaming chat interaction (extracted from client).

    Args:
        ctx: OllamaTurboClient instance (or compatible) providing required attributes and methods.
        _suppress_errors: internal flag used by streaming fallback to reduce noisy logs.
    """
    try:
        # Generation options (reused across rounds)
        options: Dict[str, Any] = {}
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
                    tool_calls = [
                        {
                            'type': 'function',
                            'id': (tc.get('id') or f"call_{i}"),
                            'function': {
                                'name': tc.get('name') or '',
                                'arguments': tc.get('arguments')
                            }
                        }
                        for i, tc in enumerate(nsr_calls, 1)
                    ]
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
                names = [tc.get('function', {}).get('name') for tc in tool_calls]
                ctx._trace(f"tools:detected {len(tool_calls)} -> {', '.join(n for n in names if n)}")

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

                # Reprompt model using details
                ctx._trace("reprompt:after-tools")
                ctx.conversation_history.append({
                    'role': 'user',
                    'content': ctx.prompt.reprompt_after_tools()
                })

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
                        def _gen_once():
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

                try:
                    if (ctx.reliability.get('check') or 'off') != 'off':
                        report = Validator(mode=str(ctx.reliability.get('check'))).validate(final_out, getattr(ctx, '_last_context_blocks', []))
                        ctx._trace(f"validate:mode={report.get('mode')} citations={report.get('citations_present')}")
                except Exception as ve:
                    ctx.logger.debug(f"validator skipped: {ve}")

                ctx.conversation_history.append({
                    'role': 'assistant',
                    'content': final_out
                })
                ctx._trace("tools:none")
                # Persist memory to Mem0
                ctx._mem0_add_after_response(ctx._last_user_message, final_out)

                if all_tool_results:
                    prefix = (first_content + "\n\n") if first_content else ""
                    return f"{prefix}[Tool Results]\n" + '\n'.join(all_tool_results) + f"\n\n{final_out}"
                return final_out
            
    except Exception as e:
        # In streaming fallback contexts, avoid noisy error logs
        if _suppress_errors:
            ctx.logger.debug(f"Standard chat error (suppressed): {e}")
        else:
            ctx.logger.error(f"Standard chat error: {e}")
        ctx._trace(f"standard:error {type(e).__name__}")
        raise
