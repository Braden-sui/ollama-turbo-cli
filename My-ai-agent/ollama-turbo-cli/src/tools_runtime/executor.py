from __future__ import annotations

"""
Tool execution runtime.

This module centralizes tool execution and serialization so the client can delegate without
changing behavior.
"""

from typing import Any, Dict, List
import os
import sys
import json
import time

from ..utils import truncate_text


class ToolRuntimeExecutor:
    @staticmethod
    def execute(ctx, tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Execute tool calls and return structured results.

        Args:
            ctx: Client-like context providing tool_functions, tracing, caps, and flags
            tool_calls: List of OpenAI-style tool call dicts
        Returns:
            List of results with shape:
            { tool: str, status: 'ok'|'error', content: Any|None, metadata: dict, error: Optional[...]}.
        """
        tool_results: List[Dict[str, Any]] = []
        try:
            for i, tc in enumerate(tool_calls, 1):
                function = tc.get('function', {})
                function_name = function.get('name')
                function_args = function.get('arguments') or {}
                # Accept JSON strings per OpenAI tool-calls; fallback to {}
                if isinstance(function_args, str):
                    try:
                        function_args = json.loads(function_args) if function_args.strip() else {}
                    except json.JSONDecodeError:
                        function_args = {}
                elif not isinstance(function_args, dict):
                    function_args = {}
                if not function_name:
                    tool_results.append({
                        'tool': 'unknown',
                        'status': 'error',
                        'content': None,
                        'metadata': {'index': i},
                        'error': {'code': 'invalid_call', 'message': 'Missing function name'}
                    })
                    continue
                # Safe args formatting (recursive redaction)
                SENSITIVE_KEYS = ('token','secret','password','api_key','apikey','auth','credential','cookie','session','key','bearer')
                def _mask_value(v):
                    if isinstance(v, str) and len(v) >= 24 and any(c.isalpha() for c in v) and any(c.isdigit() for c in v):
                        return v[:4] + '…' + v[-4:]
                    return v
                def _mask_obj(obj):
                    if isinstance(obj, dict):
                        return {k: ('***' if any(s in str(k).lower() for s in SENSITIVE_KEYS) else _mask_obj(v))
                                for k, v in obj.items()}
                    if isinstance(obj, (list, tuple, set)):
                        t = type(obj)
                        return t(_mask_obj(x) for x in obj)
                    return _mask_value(obj)
                def _fmt_tool_args(args: Dict[str, Any]) -> str:
                    try:
                        safe = _mask_obj(args or {})
                        return ', '.join(f'{k}={safe[k]}' for k in safe)
                    except Exception:
                        return ''
                if not getattr(ctx, 'quiet', False):
                    print(f"   {i}. Executing {function_name}({_fmt_tool_args(function_args)})")
                ctx._trace(f"tool:exec {function_name}")

                if function_name in getattr(ctx, 'tool_functions', {}):
                    try:
                        # Confirm execute_shell in TTY if required; block in non-interactive unless explicitly allowed
                        if function_name == 'execute_shell':
                            preview = function_args.get('command') or ''
                            require_confirm = os.getenv('CONFIRM_EXECUTE_SHELL', '1').strip().lower() not in {'0', 'false', 'no', 'off'}
                            if require_confirm and not sys.stdin.isatty():
                                msg = "execute_shell blocked in non-interactive mode; set CONFIRM_EXECUTE_SHELL=0 to allow."
                                tool_results.append({
                                    'tool': function_name,
                                    'status': 'error',
                                    'content': None,
                                    'metadata': {'preview': truncate_text(preview, 120), 'non_tty': True},
                                    'error': {'code': 'blocked_non_tty', 'message': msg}
                                })
                                ctx._skip_mem0_after_turn = True
                                continue
                            if require_confirm and sys.stdin.isatty():
                                print(f"   ⚠️ execute_shell preview: {preview}")
                                ans = input("   Proceed? [y/N]: ").strip().lower()
                                if ans not in {'y', 'yes'}:
                                    aborted_msg = f"Execution aborted by user for execute_shell({truncate_text(preview, 120)})"
                                    tool_results.append({
                                        'tool': function_name,
                                        'status': 'error',
                                        'content': None,
                                        'metadata': {'aborted': True, 'preview': truncate_text(preview, 120)},
                                        'error': {'code': 'aborted', 'message': aborted_msg}
                                    })
                                    ctx._skip_mem0_after_turn = True
                                    continue

                        start = time.perf_counter()
                        result = ctx.tool_functions[function_name](**function_args)
                        duration_ms = int((time.perf_counter() - start) * 1000)
                        # Try parse JSON contracts from secure tools
                        injected = None
                        sensitive = False
                        log_path = None
                        try:
                            if isinstance(result, str):
                                parsed = json.loads(result)
                            else:
                                parsed = result  # may already be dict
                            if isinstance(parsed, dict):
                                injected = parsed.get('inject')
                                sensitive = bool(parsed.get('sensitive'))
                                log_path = parsed.get('log_path')
                        except Exception:
                            pass

                        # Determine injection chunk
                        display = result if injected is None else injected
                        if not isinstance(display, str):
                            try:
                                display = json.dumps(display, ensure_ascii=False)
                            except Exception:
                                display = str(display)
                        if len(display) > getattr(ctx, 'tool_context_cap', 4000):
                            display = truncate_text(display, getattr(ctx, 'tool_context_cap', 4000))
                            if log_path:
                                display += f"\n[truncated; full logs stored at {log_path} (not shared with the model)]"

                        if sensitive or function_name == 'execute_shell' or len(display) > getattr(ctx, 'tool_context_cap', 4000):
                            ctx._skip_mem0_after_turn = True
                        if not getattr(ctx, 'show_trace', False) and not getattr(ctx, 'quiet', False):
                            print(f"      ✅ Result: {truncate_text(display, getattr(ctx, 'tool_print_limit', 200))}")
                        ctx._trace(f"tool:ok {function_name}")
                        structured: Dict[str, Any] = {
                            'tool': function_name,
                            'status': 'ok',
                            'content': display,
                            'metadata': {
                                'args': function_args,
                                'index': i,
                                'duration_ms': duration_ms,
                                **({'log_path': log_path} if log_path else {})
                            },
                            'error': None
                        }
                        tool_results.append(structured)
                    except Exception as e:
                        error_result = f"Error executing {function_name}: {str(e)}"
                        tool_results.append({
                            'tool': function_name,
                            'status': 'error',
                            'content': None,
                            'metadata': {'args': function_args},
                            'error': {'code': 'execution_error', 'message': error_result}
                        })
                        ctx._trace(f"tool:error {function_name}")
                else:
                    tool_results.append({
                        'tool': function_name or 'unknown',
                        'status': 'error',
                        'content': None,
                        'metadata': {},
                        'error': {'code': 'unknown_tool', 'message': f"Unknown tool: {function_name}"}
                    })

            return tool_results
        except Exception as e:
            ctx._trace(f"tools:failed {type(e).__name__}")
            return [{
                'tool': 'tools_batch',
                'status': 'error',
                'content': None,
                'metadata': {},
                'error': {'code': 'batch_failure', 'message': f"Tools execution failed: {str(e)}"}
            }]

    @staticmethod
    def serialize_to_string(ctx, tr: Dict[str, Any]) -> str:
        """Serialize a structured tool result to a safe string for model context/CLI output."""
        try:
            tool = tr.get('tool', 'tool')
            status = tr.get('status', 'ok')
            content = tr.get('content')
            if status == 'error' and tr.get('error'):
                err = tr.get('error') or {}
                msg = err.get('message') or 'error'
                return f"{tool}: ERROR - {msg}"
            if isinstance(content, (dict, list)):
                try:
                    content_str = json.dumps(content, ensure_ascii=False)
                except Exception:
                    content_str = str(content)
            else:
                content_str = str(content) if content is not None else ''
            try:
                content_str = truncate_text(content_str, getattr(ctx, 'tool_print_limit', 200))
            except Exception:
                pass
            return f"{tool}: {content_str}"
        except Exception:
            return "tool: (unserializable result)"
