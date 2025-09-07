from __future__ import annotations

"""
Mem0 service implementation.

This service encapsulates Mem0 initialization, context injection, and persistence while
operating on the provided client context (ctx) to preserve behavior and state.
"""

from typing import Any, Dict, List, Optional
import os
import json
import time
import threading
import queue
import atexit
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

from ..utils import truncate_text



class Mem0Service:
    def __init__(self, config: Any) -> None:
        self.config = config

    # ------------------ Initialization ------------------
    def initialize(self, ctx) -> None:
        """Initialize Mem0 client and runtime settings from configuration (attached to ctx)."""
        ctx.mem0_client = None
        ctx.mem0_enabled = False
        # Reranker defaults (always set)
        try:
            self._reranker_use_main = True
            self._reranker_client = None
            self._reranker_host = str(getattr(ctx, 'host', '') or '')
        except Exception:
            pass
        if not ctx.mem0_config.get('enabled', True):
            ctx.logger.debug("Mem0 is disabled via configuration")
            return
        try:
            # Lazy import to avoid hard crash if package isn't installed yet
            try:
                if ctx.mem0_config.get('local', False):
                    from mem0 import Memory as _Mem0Impl  # type: ignore
                else:
                    from mem0 import MemoryClient as _Mem0Impl  # type: ignore
            except Exception as ie:
                ctx.logger.debug(f"Mem0 import skipped: {ie}")
                try:
                    if not getattr(ctx, '_mem0_notice_shown', False) and not ctx.quiet:
                        mode = "local" if ctx.mem0_config.get('local', False) else "cloud"
                        if mode == "local":
                            print("Mem0 disabled: mem0 package not installed for local mode. Install with: pip install mem0 chromadb or qdrant-client; set MEM0_USE_LOCAL=1")
                        else:
                            print("Mem0 disabled: mem0 package not installed. For cloud, set MEM0_API_KEY or install mem0.")
                        ctx._mem0_notice_shown = True
                except Exception:
                    pass
                # Ensure breaker state is reset on disabled init
                try:
                    ctx._mem0_fail_count = 0
                    ctx._mem0_down_until_ms = 0
                    ctx._mem0_breaker_tripped_logged = False
                    ctx._mem0_breaker_recovered_logged = False
                except Exception:
                    pass
                return

            # Runtime knobs (prefer centralized config values if present)
            mc = getattr(ctx, 'mem0_config', {}) or {}
            ctx.mem0_debug = bool(mc.get('debug', False))
            ctx.mem0_max_hits = int(mc.get('max_hits', 3))
            ctx.mem0_timeout_connect_ms = int(mc.get('timeout_connect_ms', 1000))
            ctx.mem0_timeout_read_ms = int(mc.get('timeout_read_ms', 2000))
            ctx.mem0_add_queue_max = int(mc.get('add_queue_max', 256))
            ctx._mem0_breaker_threshold = int(mc.get('breaker_threshold', 3))
            ctx._mem0_breaker_cooldown_ms = int(mc.get('breaker_cooldown_ms', 60000))
            ctx._mem0_shutdown_flush_ms = 3000
            # Clamp for safety
            try:
                ctx.mem0_max_hits = max(1, min(int(ctx.mem0_max_hits), 10))
            except Exception:
                ctx.mem0_max_hits = 3
            try:
                ctx.mem0_proxy_timeout_ms = max(100, int(ctx.mem0_proxy_timeout_ms))
            except Exception:
                ctx.mem0_proxy_timeout_ms = 1200
            try:
                ctx.mem0_add_queue_max = max(1, int(ctx.mem0_add_queue_max))
            except Exception:
                ctx.mem0_add_queue_max = 256
            try:
                ctx._mem0_search_workers = max(1, int(getattr(ctx, '_mem0_search_workers', 2)))
            except Exception:
                ctx._mem0_search_workers = 2

            # API flags
            ctx._mem0_api_version_pref = 'v2'
            ctx._mem0_supports_version_kw = True

            # Proxy reranker (prefer centralized config)
            try:
                ctx.mem0_proxy_model = mc.get('proxy_model') or None
                try:
                    ctx.mem0_proxy_timeout_ms = int(mc.get('proxy_timeout_ms', 1200))
                except Exception:
                    ctx.mem0_proxy_timeout_ms = 1200
                ctx._mem0_proxy_enabled = bool(ctx.mem0_proxy_model)
            except Exception:
                ctx.mem0_proxy_model = None
                ctx.mem0_proxy_timeout_ms = 1200
                ctx._mem0_proxy_enabled = False

            if ctx.mem0_config.get('local', False):
                # Derive separate bases for Mem0 local mode:
                # - llm_base defaults to main chat host (Turbo Cloud) unless overridden
                # - embedder_base defaults to Mem0 embedder base or local when unset
                llm_base = ctx.mem0_config.get('llm_base_url') or ctx.mem0_config.get('ollama_url') or ctx.host
                # Intentionally do not use ctx.mem0_config['ollama_url'] here; default to local unless explicitly set
                embedder_base = ctx.mem0_config.get('embedder_base_url') or 'http://localhost:11434'
                # Stash for reranker client selection later
                self._mem0_llm_base = llm_base
                cfg = {
                    "vector_store": {
                        "provider": ctx.mem0_config['vector_provider'],
                        "config": {
                            "host": ctx.mem0_config['vector_host'],
                            "port": ctx.mem0_config['vector_port'],
                        },
                    },
                    "llm": {
                        "provider": "ollama",
                        "config": {
                            "model": ctx.mem0_config['llm_model'],
                            "ollama_base_url": llm_base
                        },
                    },
                    "embedder": {
                        "provider": "ollama",
                        "config": {
                            "model": ctx.mem0_config['embedder_model'],
                            "ollama_base_url": embedder_base
                        },
                    },
                    "version": ctx._mem0_api_version_pref,
                }
                if not ctx.mem0_config['vector_port']:
                    del cfg["vector_store"]["config"]["port"]
                if ctx.mem0_config['vector_host'] == ':memory:':
                    cfg["vector_store"]["config"]["in_memory"] = True
                if 'user_id' in ctx.mem0_config:
                    cfg["user_id"] = ctx.mem0_config['user_id']
                try:
                    if hasattr(_Mem0Impl, 'from_config'):
                        ctx.mem0_client = _Mem0Impl.from_config(cfg)  # type: ignore[arg-type]
                    else:
                        ctx.mem0_client = _Mem0Impl()  # type: ignore[call-arg]
                    ctx.mem0_mode = 'local'
                    ctx.mem0_enabled = True
                    ctx.logger.info(f"Initialized local Mem0 with config: {json.dumps(cfg, indent=2, default=str)}")
                except Exception as e:
                    ctx.logger.error(f"Mem0 local initialization failed: {e}")
                    try:
                        if not getattr(ctx, '_mem0_notice_shown', False) and not ctx.quiet:
                            print(f"Mem0 local initialization failed: {e}")
                            ctx._mem0_notice_shown = True
                    except Exception:
                        pass
                    return
            else:
                try:
                    api_key = mc.get('api_key')
                    if not api_key:
                        ctx.logger.warning("MEM0_API_KEY not set, disabling Mem0")
                        try:
                            if not getattr(ctx, '_mem0_notice_shown', False) and not ctx.quiet:
                                print("Mem0 disabled: MEM0_API_KEY not set. Export MEM0_API_KEY or use --mem0-local for OSS mode.")
                                ctx._mem0_notice_shown = True
                        except Exception:
                            pass
                        return
                    org_id = mc.get('org_id')
                    project_id = mc.get('project_id')
                    if org_id or project_id:
                        ctx.mem0_client = _Mem0Impl(api_key=api_key, org_id=org_id, project_id=project_id)  # type: ignore[call-arg]
                    else:
                        ctx.mem0_client = _Mem0Impl(api_key=api_key)  # type: ignore[call-arg]
                    ctx.mem0_mode = 'cloud'
                    ctx.mem0_enabled = True
                    ctx.logger.info("Initialized Mem0 cloud client")
                except Exception as e:
                    ctx.logger.error(f"Mem0 cloud initialization failed: {e}")
                    try:
                        if not getattr(ctx, '_mem0_notice_shown', False) and not ctx.quiet:
                            print(f"Mem0 cloud initialization failed: {e}")
                            ctx._mem0_notice_shown = True
                    except Exception:
                        pass
                    return

            # Background worker & search pool
            ctx._mem0_add_queue = queue.Queue(maxsize=max(1, ctx.mem0_add_queue_max))
            ctx._mem0_worker_stop = threading.Event()
            ctx._mem0_worker = threading.Thread(target=lambda: self.worker_loop(ctx), name="mem0-worker", daemon=True)
            ctx._mem0_worker.start()
            atexit.register(lambda: self.shutdown(ctx))
            ctx._mem0_search_pool = ThreadPoolExecutor(max_workers=max(1, ctx._mem0_search_workers), thread_name_prefix="mem0-search")
            atexit.register(lambda: ctx._mem0_search_pool.shutdown(wait=False) if ctx._mem0_search_pool else None)

            # Common fields
            try:
                ctx.mem0_user_id = str(mc.get('user_id', 'cli-user'))
            except Exception:
                ctx.mem0_user_id = 'cli-user'
            ctx.mem0_agent_id = mc.get('agent_id')
            ctx.mem0_app_id = mc.get('app_id')
            mode_str = getattr(ctx, 'mem0_mode', 'unknown')
            ctx.logger.info(f"Mem0 initialized and enabled in {mode_str} mode")

            # Prepare reranker client based on Mem0 LLM base (defaults to main host).
            try:
                llm_base = mc.get('llm_base_url') or getattr(self, '_mem0_llm_base', None) or getattr(ctx, 'host', None)
                main_host = str(getattr(ctx, 'host', '') or '').strip()
                if llm_base and str(llm_base).strip() == main_host:
                    # Same as main chat; mark to reuse primary client dynamically
                    self._reranker_client = None
                    self._reranker_use_main = True
                    self._reranker_host = main_host
                elif llm_base:
                    try:
                        from ollama import Client as _OllamaClient  # type: ignore
                        self._reranker_client = _OllamaClient(host=llm_base, headers={'Authorization': getattr(ctx, 'api_key', '')})
                        self._reranker_host = str(llm_base)
                        self._reranker_use_main = False
                        # Best-effort: align HTTP timeouts with main client
                        try:
                            base_http = getattr(getattr(ctx, 'client', None), '_client', None)
                            rr_http = getattr(self._reranker_client, '_client', None)
                            if base_http is not None and rr_http is not None and hasattr(base_http, 'timeout'):
                                try:
                                    rr_http.timeout = base_http.timeout
                                    ctx.logger.debug("Mem0Service: copied HTTP timeouts to reranker client")
                                except Exception:
                                    pass
                        except Exception:
                            pass
                        ctx.logger.debug(f"Mem0Service: using reranker client host={self._reranker_host}")
                    except Exception as _oi:
                        self._reranker_client = None
                        self._reranker_host = None
                        ctx.logger.debug(f"Mem0Service: reranker client unavailable: {_oi}")
                else:
                    self._reranker_client = None
                    self._reranker_host = None
            except Exception:
                self._reranker_client = None
                self._reranker_use_main = True
                self._reranker_host = str(getattr(ctx, 'host', '') or '')

        except Exception as e:
            ctx.logger.error(f"Mem0 initialization failed: {e}")
            ctx.mem0_client = None
            ctx.mem0_enabled = False
            try:
                if not getattr(ctx, '_mem0_notice_shown', False) and not ctx.quiet:
                    print(f"Mem0 initialization failed: {e}")
                    ctx._mem0_notice_shown = True
            except Exception:
                pass

    # ------------------ Context Injection ------------------
    def inject_context(self, ctx, user_message: str) -> None:
        if not getattr(ctx, 'mem0_enabled', False) or not ctx.mem0_client:
            return
        now_ms = int(time.time() * 1000)
        if getattr(ctx, '_mem0_down_until_ms', 0) and now_ms < ctx._mem0_down_until_ms:
            return
        start = time.time()
        try:
            # Remove previous Mem0 injection blocks
            for idx in range(len(ctx.conversation_history) - 1, -1, -1):
                msg = ctx.conversation_history[idx]
                if msg.get('role') == 'system':
                    c = msg.get('content') or ''
                    if (
                        c.startswith("Previous context from user history (use if relevant):")
                        or c.startswith("Relevant information:")
                        or c.startswith("Relevant user memories")
                    ):
                        ctx.conversation_history.pop(idx)
            # Strip previously merged first-system Mem0 block
            try:
                if ctx.conversation_history and (ctx.conversation_history[0] or {}).get('role') == 'system':
                    first_c = str((ctx.conversation_history[0] or {}).get('content') or '')
                    prefixes = []
                    try:
                        prefixes = ctx.prompt.mem0_prefixes()
                    except Exception:
                        prefixes = ["Previous context from user history (use if relevant):", "Relevant information:", "Relevant user memories"]
                    cut = -1
                    for p in prefixes:
                        if not p:
                            continue
                        pos = first_c.find(p)
                        if pos != -1:
                            cut = pos if cut == -1 else min(cut, pos)
                    if cut != -1:
                        ctx.conversation_history[0]['content'] = first_c[:cut].rstrip()
            except Exception:
                pass

            filters = {"user_id": getattr(ctx, 'mem0_user_id', str(getattr(ctx, 'mem0_config', {}).get('user_id', 'cli-user')))}
            def _do_search():
                search_limit = ctx.mem0_max_hits
                try:
                    if getattr(ctx, '_mem0_proxy_enabled', False) and ctx.mem0_proxy_model:
                        limit_cfg = int(getattr(ctx, 'mem0_config', {}).get('rerank_search_limit', 10))
                        search_limit = max(ctx.mem0_max_hits, limit_cfg)
                except Exception:
                    pass
                return self.search_api(ctx, user_message, filters=filters, limit=search_limit)

            try:
                if ctx._mem0_search_pool:
                    fut = ctx._mem0_search_pool.submit(_do_search)
                    related = fut.result(timeout=max(0.05, getattr(ctx, 'mem0_search_timeout_ms', 500) / 1000.0))
                else:
                    related = self.search_api(ctx, user_message, filters=filters, limit=getattr(ctx, 'mem0_max_hits', 3))
            except FuturesTimeout:
                related = []
            except Exception:
                related = []

            try:
                ctx._trace(f"mem0:search:hits={len(related or [])}")
            except Exception:
                pass

            def _memtxt(m: Dict[str, Any]) -> str:
                return (
                    (m.get('memory') if isinstance(m, dict) else None)
                    or (m.get('text') if isinstance(m, dict) else None)
                    or (m.get('content') if isinstance(m, dict) else None)
                    or str(m)
                )
            texts: List[str] = []
            aug_texts: List[str] = []
            for m in related or []:
                try:
                    txt = _memtxt(m)
                    if txt:
                        s_txt = str(txt)
                        texts.append(s_txt)
                        try:
                            md = (m.get('metadata') if isinstance(m, dict) else None) or {}
                            ts = str(md.get('timestamp') or md.get('ts') or '')
                            src = str(md.get('source') or md.get('app_id') or md.get('agent_id') or '')
                            cat = str(md.get('category') or '')
                            conf = str(md.get('confidence') or '')
                            meta_suffix = f"\n[meta: ts={ts} src={src} cat={cat} conf={conf}]".rstrip()
                        except Exception:
                            meta_suffix = ''
                        aug_texts.append((s_txt + meta_suffix) if meta_suffix else s_txt)
                except Exception:
                    continue

            top_texts = texts[: max(1, ctx.mem0_max_hits)]
            try:
                if getattr(ctx, '_mem0_proxy_enabled', False) and ctx.mem0_proxy_model and texts:
                    try:
                        ctx._trace(f"mem0:rerank:K={ctx.mem0_max_hits}")
                    except Exception:
                        pass
                    order = self.rerank_with_proxy(ctx, user_message, aug_texts or texts, model=ctx.mem0_proxy_model, k=ctx.mem0_max_hits)
                    if order:
                        top_texts = [texts[i] for i in order if 0 <= int(i) < len(texts)]
                        top_texts = top_texts[: max(1, ctx.mem0_max_hits)]
            except Exception:
                pass

            try:
                budget = max(200, int(getattr(ctx, 'mem0_config', {}).get('context_budget_chars', 800)))
            except Exception:
                budget = 800
            acc = []
            used = 0
            for ttxt in top_texts:
                remaining = max(0, budget - used)
                if remaining <= 0:
                    break
                slice_len = min(len(ttxt), remaining)
                snip = ttxt[:slice_len]
                if len(snip) < len(ttxt):
                    snip = snip.rstrip() + "…"
                acc.append(f"- {snip}")
                used += len(snip) + 2
            if not acc:
                dt_ms = int((time.time() - start) * 1000)
                if ctx.mem0_debug and not ctx.quiet:
                    print(f"[mem0] search dt={dt_ms}ms hits=0")
                return
            context = ctx.prompt.mem0_context_block(acc)
            try:
                in_first = bool(getattr(ctx, 'mem0_config', {}).get('in_first_system', False))
            except Exception:
                in_first = False
            if in_first and ctx.conversation_history and (ctx.conversation_history[0] or {}).get('role') == 'system':
                try:
                    base = str((ctx.conversation_history[0] or {}).get('content') or '').rstrip()
                    new_content = (base + "\n\n" + context).strip()
                    ctx.conversation_history[0]['content'] = new_content
                    ctx._trace(f"mem0:inject:first {len(acc)}")
                except Exception:
                    ctx.conversation_history.append({'role': 'system', 'content': context})
                    ctx._trace(f"mem0:inject {len(acc)}")
            else:
                ctx.conversation_history.append({'role': 'system', 'content': context})
                ctx._trace(f"mem0:inject {len(acc)}")
            if getattr(ctx, '_mem0_fail_count', 0) >= ctx._mem0_breaker_threshold and not getattr(ctx, '_mem0_breaker_recovered_logged', False):
                ctx.logger.info("Mem0 recovered; resuming calls")
                ctx._mem0_breaker_recovered_logged = True
                ctx._mem0_breaker_tripped_logged = False
            ctx._mem0_fail_count = 0
            ctx._mem0_down_until_ms = 0
            dt_ms = int((time.time() - start) * 1000)
            if ctx.mem0_debug and not ctx.quiet:
                print(f"[mem0] search dt={dt_ms}ms hits={len(acc)}")
        except Exception as e:
            ctx.logger.debug(f"Mem0 search failed: {e}")
            ctx._trace("mem0:inject:fail")
            ctx._mem0_fail_count = getattr(ctx, '_mem0_fail_count', 0) + 1
            if ctx._mem0_fail_count >= ctx._mem0_breaker_threshold:
                ctx._mem0_down_until_ms = int(time.time() * 1000) + ctx._mem0_breaker_cooldown_ms
                if not getattr(ctx, '_mem0_breaker_tripped_logged', False):
                    ctx.logger.warning("Mem0 circuit breaker tripped; skipping calls temporarily")
                    ctx._mem0_breaker_tripped_logged = True
                    ctx._mem0_breaker_recovered_logged = False
            if not getattr(ctx, '_mem0_notice_shown', False) and not ctx.quiet:
                print("⚠️ Mem0 unavailable; continuing without memory for this session.")
                ctx._mem0_notice_shown = True

    # ------------------ Persistence ------------------
    def persist_turn(self, ctx, user_message: Optional[str], assistant_message: Optional[str]) -> None:
        if not getattr(ctx, 'mem0_enabled', False) or not ctx.mem0_client:
            return
        if getattr(ctx, '_skip_mem0_after_turn', False):
            return
        if not user_message and not assistant_message:
            return
        messages: List[Dict[str, str]] = []
        if user_message:
            messages.append({"role": "user", "content": user_message})
        if assistant_message:
            messages.append({"role": "assistant", "content": assistant_message})
        metadata: Dict[str, Any] = {
            "source": "chat",
            "category": "inferred",
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }
        if getattr(ctx, 'mem0_app_id', None):
            metadata["app_id"] = ctx.mem0_app_id
        if getattr(ctx, 'mem0_agent_id', None):
            metadata["agent_id"] = ctx.mem0_agent_id
        # Preserve historical behavior: route through client's _mem0_enqueue_add
        # so tests and external hooks that monkeypatch the client method still work.
        try:
            enqueue = getattr(ctx, '_mem0_enqueue_add', None)
            if callable(enqueue):
                enqueue(messages, metadata)
            else:
                self.enqueue_add(ctx, messages, metadata)
        except Exception:
            self.enqueue_add(ctx, messages, metadata)
        ctx._trace("mem0:add:queued")

    def enqueue_add(self, ctx, messages: List[Dict[str, str]], metadata: Dict[str, Any]) -> None:
        try:
            if not ctx._mem0_add_queue:
                return
            job = {"messages": messages, "metadata": metadata}
            try:
                ctx._mem0_add_queue.put_nowait(job)
            except queue.Full:
                try:
                    _ = ctx._mem0_add_queue.get_nowait()
                except Exception:
                    pass
                try:
                    ctx._mem0_add_queue.put_nowait(job)
                except Exception:
                    pass
                now = time.time()
                if now - getattr(ctx, '_mem0_last_sat_log', 0) > 60:
                    ctx.logger.warning("mem0 add queue saturated; dropping oldest job")
                    ctx._mem0_last_sat_log = now
        except Exception as e:
            ctx.logger.debug(f"Mem0 enqueue failed: {e}")

    def worker_loop(self, ctx) -> None:
        while not ctx._mem0_worker_stop.is_set():
            try:
                job = None
                try:
                    job = ctx._mem0_add_queue.get(timeout=0.25) if ctx._mem0_add_queue else None
                except queue.Empty:
                    continue
                if not job:
                    continue
                start = time.time()
                try:
                    ids = self.execute_add(ctx, job.get("messages") or [], job.get("metadata") or {})
                    self.enforce_metadata(ctx, ids, job.get("metadata") or {})
                    if ctx.mem0_debug:
                        dt_ms = int((time.time() - start) * 1000)
                        ctx.logger.debug(f"[mem0] add dt={dt_ms}ms ids={ids}")
                    ctx._mem0_fail_count = 0
                    ctx._mem0_down_until_ms = 0
                except Exception as we:
                    ctx.logger.debug(f"Mem0 add worker error: {we}")
                    ctx._mem0_fail_count += 1
                    if ctx._mem0_fail_count >= ctx._mem0_breaker_threshold:
                        ctx._mem0_down_until_ms = int(time.time() * 1000) + ctx._mem0_breaker_cooldown_ms
                        if not getattr(ctx, '_mem0_breaker_tripped_logged', False):
                            ctx.logger.warning("Mem0 circuit breaker tripped; skipping calls temporarily")
                            ctx._mem0_breaker_tripped_logged = True
                            ctx._mem0_breaker_recovered_logged = False
            except Exception:
                continue

    def shutdown(self, ctx) -> None:
        """Gracefully stop the Mem0 worker and flush a bounded number of queued jobs.

        Safe to call multiple times and during interpreter shutdown.
        """
        try:
            try:
                if getattr(ctx, '_mem0_worker_stop', None):
                    ctx._mem0_worker_stop.set()
            except Exception:
                pass
            # Bounded flush window
            deadline = time.time() + max(0.0, getattr(ctx, '_mem0_shutdown_flush_ms', 3000) / 1000.0)
            while time.time() < deadline:
                try:
                    job = ctx._mem0_add_queue.get_nowait() if getattr(ctx, '_mem0_add_queue', None) else None
                except Exception:
                    break
                if not job:
                    break
                try:
                    ids = self.execute_add(ctx, job.get('messages') or [], job.get('metadata') or {})
                    self.enforce_metadata(ctx, ids, job.get('metadata') or {})
                except Exception:
                    continue
            # Best effort: stop search pool
            try:
                if getattr(ctx, '_mem0_search_pool', None):
                    ctx._mem0_search_pool.shutdown(wait=False)
            except Exception:
                pass
        except Exception:
            pass

    # ------------------ Low-level SDK helpers ------------------
    def execute_add(self, ctx, messages: List[Dict[str, str]], metadata: Dict[str, Any]) -> List[str]:
        if not ctx.mem0_client:
            return []
        user_id = getattr(ctx, 'mem0_user_id', str(ctx.mem0_config.get('user_id', 'cli-user')))
        kwargs = {"messages": messages, "user_id": user_id, "version": getattr(ctx, '_mem0_api_version_pref', 'v2'), "metadata": metadata}
        if getattr(ctx, 'mem0_agent_id', None):
            kwargs["agent_id"] = ctx.mem0_agent_id
        try:
            res = ctx.mem0_client.add(**kwargs)
        except TypeError:
            try:
                kwargs_no_ver = dict(kwargs)
                kwargs_no_ver.pop("version", None)
                res = ctx.mem0_client.add(**kwargs_no_ver)
            except TypeError:
                try:
                    kwargs_no_agent = dict(kwargs)
                    kwargs_no_agent.pop("agent_id", None)
                    res = ctx.mem0_client.add(**kwargs_no_agent)
                except TypeError:
                    kwargs_min = {"messages": messages, "user_id": user_id}
                    res = ctx.mem0_client.add(**kwargs_min)
        ids: List[str] = []
        try:
            if isinstance(res, dict):
                if "id" in res:
                    ids.append(str(res["id"]))
                elif "data" in res and isinstance(res["data"], dict) and "id" in res["data"]:
                    ids.append(str(res["data"]["id"]))
                elif "items" in res and isinstance(res["items"], list):
                    for it in res["items"]:
                        if isinstance(it, dict) and "id" in it:
                            ids.append(str(it["id"]))
            elif isinstance(res, list):
                for it in res:
                    if isinstance(it, dict) and "id" in it:
                        ids.append(str(it["id"]))
        except Exception:
            pass
        return ids

    def enforce_metadata(self, ctx, ids: List[str], metadata: Dict[str, Any]) -> None:
        if not ids or not ctx.mem0_client:
            return
        backoffs = [0.5, 1.0, 2.0, 2.0, 2.0]
        for mid in ids:
            ok = False
            for delay in backoffs:
                start_meta = time.time()
                try:
                    try:
                        ctx.mem0_client.update(memory_id=mid, metadata=metadata)
                    except TypeError:
                        ctx.mem0_client.update(memory_id=mid, **{"metadata": metadata})
                    if getattr(ctx, 'mem0_agent_id', None):
                        try:
                            ctx.mem0_client.update(memory_id=mid, agent_id=ctx.mem0_agent_id)
                        except TypeError:
                            pass
                    if getattr(ctx, 'mem0_app_id', None):
                        try:
                            ctx.mem0_client.update(memory_id=mid, app_id=ctx.mem0_app_id)
                        except TypeError:
                            pass
                    ok = True
                    if ctx.mem0_debug:
                        dt_ms = int((time.time() - start_meta) * 1000)
                        ctx.logger.debug(f"[mem0] update(meta+agent) dt={dt_ms}ms id={mid}")
                    break
                except Exception:
                    time.sleep(delay)
            if not ok:
                ctx.logger.warning(f"Mem0 metadata enforcement failed for id={mid}")

    def search_api(self, ctx, query: str, filters: Optional[Dict[str, Any]] = None, limit: Optional[int] = None):
        if not ctx.mem0_client:
            return []
        user_id = None
        try:
            user_id = (filters or {}).get('user_id') if isinstance(filters, dict) else None
        except Exception:
            user_id = None
        attempts = []
        if getattr(ctx, '_mem0_supports_version_kw', True):
            attempts.append(lambda: ctx.mem0_client.search(query, version=ctx._mem0_api_version_pref, filters=filters, limit=limit))
        attempts.append(lambda: ctx.mem0_client.search(query, filters=filters, limit=limit))
        if user_id is not None:
            if getattr(ctx, '_mem0_supports_version_kw', True):
                attempts.append(lambda: ctx.mem0_client.search(query=query, user_id=user_id, version=ctx._mem0_api_version_pref))
            attempts.append(lambda: ctx.mem0_client.search(query=query, user_id=user_id))
            if limit is not None:
                attempts.append(lambda: ctx.mem0_client.search(query=query, user_id=user_id, limit=limit))
        attempts.append(lambda: ctx.mem0_client.search(query))
        last_type_error: Optional[Exception] = None
        for call in attempts:
            try:
                return call()
            except TypeError as te:
                last_type_error = te
                ctx._mem0_supports_version_kw = False
                continue
        if last_type_error is not None:
            raise last_type_error
        raise RuntimeError("Mem0 search failed with all known signatures")

    def get_all_api(self, ctx, filters: Optional[Dict[str, Any]] = None):
        if not ctx.mem0_client:
            return []
        user_id = None
        try:
            user_id = (filters or {}).get('user_id') if isinstance(filters, dict) else None
        except Exception:
            user_id = None
        attempts = []
        if getattr(ctx, '_mem0_supports_version_kw', True):
            attempts.append(lambda: ctx.mem0_client.get_all(filters=filters, version=ctx._mem0_api_version_pref))
        attempts.append(lambda: ctx.mem0_client.get_all(filters=filters))
        if user_id is not None:
            if getattr(ctx, '_mem0_supports_version_kw', True):
                attempts.append(lambda: ctx.mem0_client.get_all(user_id=user_id, version=ctx._mem0_api_version_pref))
            attempts.append(lambda: ctx.mem0_client.get_all(user_id=user_id))
        for call in attempts:
            try:
                return call()
            except TypeError:
                ctx._mem0_supports_version_kw = False
                continue
        raise RuntimeError("Mem0 get_all failed with all known signatures")

    # ------------------ Proxy reranker helpers ------------------
    def llm_generate(self, ctx, *, model: str, system: str, user: str) -> str:
        try:
            try:
                base_sys = ctx.prompt.initial_system_prompt()
            except Exception:
                base_sys = (
                    "You are GPT-OSS running with Harmony channels.\n\n"
                    "— Harmony I/O Protocol —\n"
                    "• Always end with: <|channel|>final then <|message|>...<|end|>\n"
                )
            task_label = str(system or 'task').strip().lower()
            rerank_sys = (
                f"\nTask: {task_label}\n"
                "Return only a JSON array of 0-based indices for the most relevant items, inside the Harmony final channel exactly as:\n"
                "<|channel|>final\n"
                "<|message|>[1,0]\n"
                "<|end|>\n"
                "No other channels or text."
            )
            msgs = [
                {'role': 'system', 'content': (base_sys + rerank_sys)},
                {'role': 'user', 'content': user},
            ]
            kwargs: Dict[str, Any] = {'model': model, 'messages': msgs}
            options: Dict[str, Any] = {'temperature': 0, 'top_p': 0}
            if options:
                kwargs['options'] = options
            try:
                keep_val = ctx._resolve_keep_alive()
                if keep_val is not None:
                    kwargs['keep_alive'] = keep_val
            except Exception:
                pass
            ctx._trace('mem0:proxy:call')
            use_private = (not getattr(self, '_reranker_use_main', False)) and (getattr(self, '_reranker_client', None) is not None)
            if use_private:
                try:
                    ctx._trace(f"mem0:proxy:host {getattr(self, '_reranker_host', '')}")
                except Exception:
                    pass
            oc = getattr(ctx, 'client', None) if getattr(self, '_reranker_use_main', False) else (getattr(self, '_reranker_client', None) or getattr(ctx, 'client', None))
            if oc is None:
                return ''
            resp = oc.chat(**kwargs)
            msg = resp.get('message', {}) if isinstance(resp, dict) else {}
            content = msg.get('content') or ''
            # Return raw content (may include Harmony tokens or bare JSON). The caller
            # will extract the JSON array [..] safely. Avoid dependency on ctx token
            # parsers so tests with minimal ctx pass.
            return content
        except Exception as e:
            ctx.logger.debug(f"mem0 proxy generate failed: {e}")
            return ''

    def rerank_with_proxy(self, ctx, query: str, candidates: List[str], *, model: str, k: Optional[int] = None) -> List[int]:
        try:
            N = len(candidates)
            if N == 0:
                return []
            try:
                base_k = int(k) if k is not None else int(ctx.mem0_max_hits)
            except Exception:
                base_k = 1
            K = max(1, min(base_k, N))
            items = "\n".join(f"[{i}] {c}" for i, c in enumerate(candidates))
            usr_prompt = (
                "Rerank the candidates for the query. Criteria (in order):\n"
                "1) Semantic relevance to the query intent.\n"
                "2) Specificity and user-identifying detail over generic content.\n"
                "3) Recency if a timestamp is present in [meta: ts=...].\n"
                "4) De-duplicate near-identical content; pick the best representative.\n"
                "5) If candidates contradict, prefer the more specific or recent one.\n\n"
                f"Query: {query}\n\n"
                f"Candidates:\n{items}\n\n"
                f"Return a JSON array of unique 0-based indices, sorted by relevance, length <= {K}."
            )
            raw: str = ''
            try:
                with ThreadPoolExecutor(max_workers=1) as _ex:
                    fut = _ex.submit(self.llm_generate, ctx, model=model, system="rerank", user=usr_prompt)
                    raw = fut.result(timeout=max(0.1, (getattr(ctx, 'mem0_proxy_timeout_ms', 1200) or 1200) / 1000.0))
            except FuturesTimeout:
                ctx.logger.debug("mem0 rerank proxy timeout")
                try:
                    ctx._trace("mem0:rerank:timeout")
                except Exception:
                    pass
                return []
            start = raw.find('[')
            end = raw.rfind(']')
            sraw = raw.strip()
            if sraw.startswith('[') and sraw.endswith(']'):
                start = 0
                end = len(sraw) - 1
                raw_to_parse = sraw
            else:
                if start == -1 or end == -1 or end <= start:
                    return list(range(min(K, N)))
                raw_to_parse = raw
            try:
                arr = json.loads(raw_to_parse[start:end+1])
            except Exception:
                return list(range(min(K, N)))
            picked = [int(i) for i in arr if isinstance(i, (int, float)) and 0 <= int(i) < N]
            if not picked:
                return list(range(min(K, N)))
            seen: set = set()
            order: List[int] = []
            for i in picked:
                if i not in seen:
                    seen.add(i)
                    order.append(i)
                if len(order) >= K:
                    break
            return order
        except Exception as e:
            ctx.logger.debug(f"mem0 proxy rerank failed: {e}")
            return []

    # ------------------ CLI helpers ------------------
    def handle_command(self, ctx, cmdline: str) -> None:
        if not cmdline.startswith('/mem'):
            return
        parts = cmdline.split()
        if len(parts) == 1:
            print("ℹ️ Usage: /mem [list|search|add|get|update|delete|clear|link|export|import] ...")
            return
        if not getattr(ctx, 'mem0_enabled', False) or not ctx.mem0_client:
            print("⚠️ Mem0 is not configured. Enable with MEM0_USE_LOCAL=1 (local OSS) or set MEM0_API_KEY (remote platform).")
            return
        sub = parts[1].lower()
        try:
            if sub == 'list':
                query = ' '.join(parts[2:]).strip() if len(parts) > 2 else ''
                filters = {"user_id": ctx.mem0_user_id}
                items = self.get_all_api(ctx, filters=filters)
                if not items:
                    print("📭 No memories found.")
                    return
                print("🧠 Memories:")
                shown = 0
                for i, it in enumerate(items, 1):
                    mem_text = it.get('memory') or (it.get('data') or {}).get('memory')
                    print(f"  {i}. {it.get('id')}: {truncate_text(mem_text or '', 200)}")
                    shown += 1
                if shown == 0:
                    print("  (no items match your query)")
            elif sub == 'search':
                query = ' '.join(parts[2:]).strip()
                if not query:
                    print("Usage: /mem search <query>")
                    return
                filters = {"user_id": ctx.mem0_user_id}
                results = self.search_api(ctx, query, filters=filters)
                if not results:
                    print("🔍 No matching memories.")
                    return
                print("🔍 Top matches:")
                for i, it in enumerate(results[:10], 1):
                    mem_text = it.get('memory') or (it.get('data') or {}).get('memory')
                    print(f"  {i}. {it.get('id')}: {truncate_text(mem_text or '', 200)}")
            elif sub == 'add':
                text = ' '.join(parts[2:]).strip()
                if not text:
                    print("Usage: /mem add <text>")
                    return
                self.execute_add(ctx, [{"role": "user", "content": text}], {"source": "cli", "category": "manual"})
                print("✅ Memory added.")
            elif sub == 'get':
                mem_id = (parts[2] if len(parts) > 2 else '').strip()
                if not mem_id:
                    print("Usage: /mem get <memory_id>")
                    return
                item = ctx.mem0_client.get(memory_id=mem_id)
                print(json.dumps(item, indent=2))
            elif sub == 'update':
                if len(parts) < 4:
                    print("Usage: /mem update <memory_id> <new text>")
                    return
                mem_id = parts[2]
                new_text = ' '.join(parts[3:])
                try:
                    ctx.mem0_client.update(memory_id=mem_id, text=new_text)
                except TypeError:
                    ctx.mem0_client.update(memory_id=mem_id, data=new_text)
                print("✅ Memory updated.")
            elif sub == 'delete':
                mem_id = (parts[2] if len(parts) > 2 else '').strip()
                if not mem_id:
                    print("Usage: /mem delete <memory_id>")
                    return
                ctx.mem0_client.delete(memory_id=mem_id)
                print("🗑️ Memory deleted.")
            elif sub == 'link':
                if len(parts) < 4:
                    print("Usage: /mem link <id1> <id2>")
                    return
                id1, id2 = parts[2], parts[3]
                try:
                    ctx.mem0_client.link(memory1_id=id1, memory2_id=id2, user_id=ctx.mem0_user_id)
                    print("🔗 Memories linked.")
                except Exception:
                    print("ℹ️ Linking not available in this plan/SDK.")
            elif sub == 'export':
                out_path = (parts[2] if len(parts) > 2 else '').strip()
                if not out_path:
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    out_path = f"mem0_export_{ts}.json"
                filters = {"user_id": ctx.mem0_user_id}
                items = self.get_all_api(ctx, filters=filters)
                payload = []
                for it in items or []:
                    payload.append({
                        "id": it.get("id"),
                        "memory": it.get("memory") or (it.get('data') or {}).get('memory'),
                        "metadata": it.get("metadata") or {},
                        "created_at": it.get("created_at"),
                        "updated_at": it.get("updated_at"),
                    })
                with open(out_path, 'w', encoding='utf-8') as f:
                    json.dump(payload, f, ensure_ascii=False, indent=2)
                print(f"📦 Exported {len(payload)} memories to {out_path}")
            elif sub == 'import':
                if len(parts) < 3:
                    print("Usage: /mem import <path.json>")
                    return
                in_path = parts[2]
                if not os.path.exists(in_path):
                    print("❌ File not found.")
                    return
                with open(in_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                count = 0
                for item in data if isinstance(data, list) else []:
                    text = item.get('memory')
                    if not text:
                        continue
                    meta = item.get('metadata') or {}
                    if getattr(ctx, 'mem0_app_id', None):
                        meta.setdefault('app_id', ctx.mem0_app_id)
                    if getattr(ctx, 'mem0_agent_id', None):
                        meta.setdefault('agent_id', ctx.mem0_agent_id)
                    try:
                        self.execute_add(ctx, [{"role": "user", "content": text}], meta)
                        count += 1
                    except Exception:
                        continue
                print(f"✅ Imported {count} memories.")
            elif sub in ('clear', 'delete-all'):
                ctx.mem0_client.delete_all(user_id=ctx.mem0_user_id)
                print("🧹 All memories for user cleared.")
            else:
                print("Unknown /mem subcommand. Use list|search|add|get|update|delete|clear|link|export|import")
        except Exception as e:
            print(f"❌ Mem0 command error: {e}")

    def handle_nlu(self, ctx, text: str) -> bool:
        if not getattr(ctx, 'mem0_enabled', False) or not ctx.mem0_client:
            return False
        try:
            import re
            lower = text.lower()
            norm = re.sub(r"\s+", " ", lower).strip().rstrip(".!?")
            if norm == "list memories":
                filters = {"user_id": ctx.mem0_user_id}
                items = self.get_all_api(ctx, filters=filters)
                if not items:
                    print("📭 No memories found.")
                    return True
                print("🧠 Memories:")
                for i, it in enumerate(items[:10], 1):
                    mem_text = it.get('memory') or (it.get('data') or {}).get('memory')
                    print(f"  {i}. {mem_text}")
                return True
            return False
        except Exception as e:
            ctx.logger.debug(f"Mem0 NLU list error: {e}")
            return False

    # ------------------ Shutdown ------------------
    def shutdown(self, ctx) -> None:
        try:
            if not ctx._mem0_add_queue:
                return
            ctx._mem0_worker_stop.set()
            deadline = time.time() + max(0.0, ctx._mem0_shutdown_flush_ms / 1000.0) if hasattr(ctx, '_mem0_shutdown_flush_ms') else time.time() + 3.0
            while time.time() < deadline:
                try:
                    job = ctx._mem0_add_queue.get_nowait()
                except queue.Empty:
                    break
                try:
                    ids = self.execute_add(ctx, job.get("messages") or [], job.get("metadata") or {})
                    self.enforce_metadata(ctx, ids, job.get("metadata") or {})
                except Exception:
                    pass
        except Exception:
            pass
