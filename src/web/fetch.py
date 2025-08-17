from __future__ import annotations
import os
import re
import io
import time
import json
import math
import hashlib
import random
import atexit
from collections import OrderedDict
from email.utils import parsedate_to_datetime
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
import threading
from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode

from .config import WebConfig
from .robots import RobotsPolicy


@dataclass
class FetchResult:
    ok: bool
    status: int
    url: str
    final_url: str
    headers: Dict[str, str]
    content_type: str
    bytes: int
    body_path: Optional[str]
    meta_path: Optional[str]
    cached: bool
    browser_used: bool
    reason: Optional[str] = None


def _canonicalize_url(url: str) -> str:
    try:
        p = urlparse(url)
        # Drop fragment, normalize scheme/host
        scheme = p.scheme.lower() if p.scheme else 'https'
        netloc = p.netloc.lower()
        # Remove default ports
        if (scheme == 'http' and netloc.endswith(':80')) or (scheme == 'https' and netloc.endswith(':443')):
            netloc = netloc.rsplit(':', 1)[0]
        # Normalize path
        path = p.path or '/'
        # Filter tracking query params and sort
        try:
            params = []
            for k, v in parse_qsl(p.query, keep_blank_values=True):
                kl = k.lower()
                if kl.startswith('utm_') or kl in {'gclid', 'fbclid', 'ref', 'mc_cid', 'mc_eid', 'igshid'}:
                    continue
                params.append((k, v))
            query = urlencode(sorted(params)) if params else ''
        except Exception:
            query = p.query
        return urlunparse((scheme, netloc, path, p.params, query, ''))
    except Exception:
        return url


def _host_allowed(cfg: WebConfig, host: str) -> bool:
    allow = os.getenv('SANDBOX_NET_ALLOW', cfg.sandbox_allow or '')
    if not allow:
        return True  # if no allowlist defined, default allow
    patterns = [x.strip() for x in allow.split(',') if x.strip()]
    for pat in patterns:
        if re.fullmatch(pat.replace('*', '.*'), host):
            return True
    return False


def _check_ip_blocks(host: str) -> None:
    import socket, ipaddress
    infos = socket.getaddrinfo(host, None)
    for info in infos:
        ip = info[4][0]
        ip_obj = ipaddress.ip_address(ip)
        if ip_obj.is_private or ip_obj.is_loopback or ip_obj.is_link_local or ip_obj.is_multicast or ip_obj.is_reserved:
            raise ValueError(f"Blocked private/loopback IP: {ip}")
        if ip.startswith('169.254.169.254'):
            raise ValueError(f"Blocked metadata IP: {ip}")


def _cache_paths(cfg: WebConfig, url: str) -> Tuple[str, str]:
    os.makedirs(cfg.cache_root, exist_ok=True)
    key = hashlib.sha256(_canonicalize_url(url).encode()).hexdigest()
    return os.path.join(cfg.cache_root, f"{key}.bin"), os.path.join(cfg.cache_root, f"{key}.json")


def _httpx_client(cfg: WebConfig):
    import httpx  # type: ignore
    timeout = httpx.Timeout(connect=cfg.timeout_connect, read=cfg.timeout_read, write=cfg.timeout_write)
    limits = httpx.Limits(max_connections=cfg.max_connections, max_keepalive_connections=cfg.max_keepalive)
    return httpx.Client(http2=True, timeout=timeout, limits=limits, headers={"User-Agent": cfg.user_agent}, follow_redirects=cfg.follow_redirects)


# Small LRU pool of httpx.Clients keyed by origin
_CLIENT_POOL: "OrderedDict[str, Any]" = OrderedDict()
_CLIENT_POOL_LOCK = threading.Lock()


def _origin_of(url: str) -> str:
    p = urlparse(url)
    return f"{p.scheme.lower()}://{p.netloc.lower()}"


def _get_client_for_origin(cfg: WebConfig, origin: str):
    try:
        with _CLIENT_POOL_LOCK:
            if origin in _CLIENT_POOL:
                client = _CLIENT_POOL.pop(origin)
                _CLIENT_POOL[origin] = client
                return client
        # create new
        client = _httpx_client(cfg)
        with _CLIENT_POOL_LOCK:
            _CLIENT_POOL[origin] = client
            # Evict LRU if over capacity
            while len(_CLIENT_POOL) > max(1, cfg.client_pool_size):
                _, old = _CLIENT_POOL.popitem(last=False)
                try:
                    old.close()
                except Exception:
                    pass
        return client
    except Exception:
        # Fallback: create a one-off client
        return _httpx_client(cfg)


def _close_all_clients():
    try:
        with _CLIENT_POOL_LOCK:
            for client in list(_CLIENT_POOL.values()):
                try:
                    client.close()
                except Exception:
                    pass
            _CLIENT_POOL.clear()
    except Exception:
        pass


atexit.register(_close_all_clients)


def _should_escalate_to_browser(status: int, ctype: str, body: bytes) -> bool:
    if status in (403, 401):
        return True
    if ('text/html' in ctype or 'application/xhtml+xml' in ctype) and len(body) < 1024:
        # Likely JS app shell
        return True
    # Heuristic: if HTML but contains minimal text and lots of JS
    txt = body.decode(errors='ignore')[:4096]
    if '<script' in txt.lower() and len(re.sub(r"<[^>]+>", ' ', txt).strip()) < 200:
        return True
    return False


_last_fetch: Dict[str, float] = {}
_lf_lock = threading.Lock()


class _TokenBucket:
    def __init__(self, capacity: int, rate: float) -> None:
        self.capacity = max(1, capacity)
        self.rate = max(0.01, rate)
        self.tokens = float(self.capacity)
        self.last = time.time()
        self.lock = threading.Lock()

    def acquire_wait(self, need: float = 1.0) -> float:
        with self.lock:
            now = time.time()
            # Refill
            elapsed = now - self.last
            if elapsed > 0:
                self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
                self.last = now
            if self.tokens >= need:
                self.tokens -= need
                return 0.0
            # compute wait time for needed tokens
            deficit = need - self.tokens
            wait = deficit / self.rate
            # Don't modify tokens yet; just report wait
            self.last = now
            return max(0.0, wait)


_TB_MAP: Dict[str, _TokenBucket] = {}
_TB_LOCK = threading.Lock()


def _tb_for_host(host: str, cfg: WebConfig) -> _TokenBucket:
    with _TB_LOCK:
        tb = _TB_MAP.get(host)
        if tb is None:
            tb = _TokenBucket(cfg.rate_tokens_per_host, cfg.rate_refill_per_sec)
            _TB_MAP[host] = tb
        return tb


def _parse_retry_after(val: str) -> float:
    try:
        # seconds
        secs = int(val.strip())
        return float(max(0, secs))
    except Exception:
        pass
    try:
        dt = parsedate_to_datetime(val)
        if dt is not None:
            return max(0.0, (dt.timestamp() - time.time()))
    except Exception:
        pass
    return 0.0

def fetch_url(url: str, *, cfg: Optional[WebConfig] = None, robots: Optional[RobotsPolicy] = None, force_refresh: bool = False, use_browser_if_needed: bool = True) -> FetchResult:
    cfg = cfg or WebConfig()
    robots = robots or RobotsPolicy(cfg)
    url = _canonicalize_url(url)

    parsed = urlparse(url)
    if parsed.scheme == 'http' and not cfg.sandbox_allow_http:
        return FetchResult(False, 0, url, url, {}, '', 0, None, None, False, False, reason='HTTP blocked by policy')
    if not _host_allowed(cfg, parsed.hostname or ''):
        return FetchResult(False, 0, url, url, {}, '', 0, None, None, False, False, reason=f'Host not in allowlist: {parsed.hostname}')
    try:
        _check_ip_blocks(parsed.hostname or '')
    except Exception as e:
        return FetchResult(False, 0, url, url, {}, '', 0, None, None, False, False, reason=str(e))

    if cfg.respect_robots:
        rec = robots.check(url)
        if not rec.allow:
            delay = rec.crawl_delay or 0
            if delay:
                time.sleep(min(delay, 5))
            return FetchResult(False, 0, url, url, {}, '', 0, None, None, False, False, reason='Blocked by robots.txt')
        # Enforce crawl-delay for allowed hosts as well
        host = parsed.hostname or ''
        if rec.crawl_delay:
            with _lf_lock:
                last = _last_fetch.get(host, 0.0)
                now = time.time()
                wait = rec.crawl_delay - (now - last)
            if wait and wait > 0:
                time.sleep(min(wait, 5))
            with _lf_lock:
                _last_fetch[host] = time.time()

    # Per-host token bucket
    host = parsed.hostname or ''
    try:
        tb_wait = _tb_for_host(host, cfg).acquire_wait(1.0)
        if tb_wait > 0:
            time.sleep(min(tb_wait, 5.0))
    except Exception:
        pass

    body_path, meta_path = _cache_paths(cfg, url)
    now = time.time()
    cached = False
    if not force_refresh and os.path.isfile(meta_path) and os.path.isfile(body_path):
        try:
            meta = json.loads(open(meta_path, 'r', encoding='utf-8').read())
            if now - float(meta.get('ts', 0)) <= cfg.cache_ttl_seconds:
                raw = open(body_path, 'rb').read()
                return FetchResult(True, int(meta.get('status', 200)), url, meta.get('final_url', url), dict(meta.get('headers', {})), meta.get('content_type', ''), len(raw), body_path, meta_path, True, bool(meta.get('browser_used', False)))
        except Exception:
            pass

    # Prepare conditional headers if any (for revalidation)
    prev_meta: Optional[Dict[str, Any]] = None
    prev_etag = None
    prev_lm = None
    try:
        if os.path.isfile(meta_path):
            prev_meta = json.loads(open(meta_path, 'r', encoding='utf-8').read())
            hdrs = dict(prev_meta.get('headers', {}))
            prev_etag = hdrs.get('etag')
            prev_lm = hdrs.get('last-modified')
    except Exception:
        prev_meta = None
        prev_etag = None
        prev_lm = None

    # Build common headers
    accept = cfg.accept_header_override or "text/html,application/xhtml+xml,application/pdf;q=0.9,text/plain;q=0.8,*/*;q=0.1"
    common_headers = {"Accept": accept}

    # HTTP path
    try:
        import httpx  # type: ignore
        client = _get_client_for_origin(cfg, _origin_of(url))
        # Optional HEAD gating
        if cfg.head_gating_enabled:
            try:
                hresp = client.head(url, headers=common_headers)
                h_ct = hresp.headers.get('content-type', '')
                h_cl = hresp.headers.get('content-length')
                if hresp.status_code in (429, 503) and cfg.respect_retry_after:
                    ra = hresp.headers.get('retry-after')
                    if ra:
                        delay = _parse_retry_after(ra)
                        if delay > 0:
                            time.sleep(min(delay, cfg.retry_backoff_max))
                if h_cl is not None:
                    try:
                        if int(h_cl) > cfg.max_download_bytes:
                            return FetchResult(False, int(hresp.status_code or 0), url, str(hresp.request.url), dict(hresp.headers), h_ct, 0, None, None, False, False, reason='content too large')
                    except Exception:
                        pass
                if h_ct:
                    allowed = any(x in h_ct for x in [
                        'text/html', 'application/xhtml+xml', 'application/pdf', 'text/plain'
                    ])
                    if not allowed:
                        return FetchResult(False, int(hresp.status_code or 0), url, str(hresp.request.url), dict(hresp.headers), h_ct, 0, None, None, False, False, reason='unsupported content-type')
            except Exception:
                pass

        # Retry with backoff and jitter
        attempts = max(1, cfg.retry_attempts)
        last_exc: Optional[Exception] = None
        data = b''
        headers: Dict[str, str] = {}
        status = 0
        content_type = ''
        resp_final_url = url
        for i in range(attempts):
            try:
                req_headers = dict(common_headers)
                if prev_etag:
                    req_headers['If-None-Match'] = str(prev_etag)
                if prev_lm:
                    req_headers['If-Modified-Since'] = str(prev_lm)
                with client.stream("GET", url, headers=req_headers) as resp:
                    status = resp.status_code
                    # Handle 304 revalidation
                    if status == 304 and prev_meta and os.path.isfile(body_path):
                        try:
                            raw = open(body_path, 'rb').read()
                        except Exception:
                            raw = b''
                        pm_headers = dict(prev_meta.get('headers', {}))
                        return FetchResult(True, int(prev_meta.get('status', 200)), url, prev_meta.get('final_url', url), pm_headers, prev_meta.get('content_type', ''), len(raw), body_path, meta_path, True, bool(prev_meta.get('browser_used', False)))
                    if status >= 500 or status in (429, 503):
                        ra = resp.headers.get('retry-after')
                        if cfg.respect_retry_after and ra:
                            delay = _parse_retry_after(ra)
                            if delay > 0:
                                time.sleep(min(delay, cfg.retry_backoff_max))
                        raise RuntimeError(f"server error {status}")
                    # Read with cap
                    buf = bytearray()
                    for chunk in resp.iter_bytes():
                        if chunk:
                            buf += chunk
                            if len(buf) >= cfg.max_download_bytes:
                                break
                    data = bytes(buf)
                    headers = {k: v for k, v in resp.headers.items()}
                    content_type = headers.get('content-type', '')
                    resp_final_url = str(resp.request.url)
                break
            except Exception as e:
                last_exc = e
                backoff = min(cfg.retry_backoff_max, cfg.retry_backoff_base * (2 ** i))
                time.sleep(backoff + random.random() * 0.2)
        else:
            raise last_exc or RuntimeError("request failed")

        final_url = resp_final_url
        raw = data
        browser_needed = use_browser_if_needed and _should_escalate_to_browser(status, content_type, raw)
    except Exception as e:
        status = 0
        final_url = url
        headers = {}
        content_type = ''
        raw = b''
        browser_needed = use_browser_if_needed  # escalate on network errors

    browser_used = False
    if browser_needed and cfg.allow_browser:
        bres = fetch_with_browser(url, cfg=cfg)
        if bres and bres.ok:
            status = bres.status
            final_url = bres.final_url
            headers = bres.headers
            content_type = bres.content_type
            raw = open(bres.body_path, 'rb').read() if bres.body_path and os.path.isfile(bres.body_path) else raw
            browser_used = True
        else:
            # keep HTTP result if any
            pass

    # Save to cache
    try:
        meta = {
            'ts': now,
            'status': status,
            'final_url': final_url,
            'headers': headers,
            'content_type': content_type,
            'browser_used': browser_used,
        }
        open(body_path, 'wb').write(raw)
        open(meta_path, 'w', encoding='utf-8').write(json.dumps(meta))
        cached = False
    except Exception:
        body_path = None
        meta_path = None
        cached = False

    return FetchResult(True if status and status < 400 else False, int(status or 0), url, final_url, headers, content_type, len(raw), body_path, meta_path, cached, browser_used)


def fetch_with_browser(url: str, *, cfg: Optional[WebConfig] = None) -> Optional[FetchResult]:
    cfg = cfg or WebConfig()
    try:
        from playwright.sync_api import sync_playwright  # type: ignore
    except Exception:
        return None

    parsed = urlparse(url)
    if parsed.scheme == 'http' and not cfg.sandbox_allow_http:
        return FetchResult(False, 0, url, url, {}, '', 0, None, None, False, True, reason='HTTP blocked by policy')

    body_path, meta_path = _cache_paths(cfg, url + "#browser")
    html = ""
    headers: Dict[str, str] = {}
    final_url = url
    status = 200

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(user_agent=cfg.user_agent)
        # Light stealth: tweak navigator properties where safe
        if cfg.browser_stealth_light:
            try:
                context.add_init_script("""
                    Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
                """)
            except Exception:
                pass
        page = context.new_page()
        network_logs: list[dict] = []
        # Build block list for resource types
        _block_types = set()
        try:
            for t in (cfg.browser_block_resources or '').split(','):
                t = t.strip().lower()
                if t:
                    _block_types.add(t)
        except Exception:
            _block_types = set()
        def _route_guard(route):
            try:
                req = route.request
                rtype = (getattr(req, 'resource_type', None) or '').lower()
                url = req.url or ''
                network_logs.append({'url': url, 'method': req.method})
                # Block heavy/analytics resources
                if rtype in _block_types:
                    return route.abort()
                lower = url.lower()
                if any(k in lower for k in ['analytics', 'doubleclick', 'googletagmanager', 'adservice', 'pixel', 'mixpanel', 'segment.io', 'facebook.net']):
                    return route.abort()
            except Exception:
                pass
            try:
                route.continue_()
            except Exception:
                try:
                    route.fallback()
                except Exception:
                    pass
        try:
            page.route("**/*", _route_guard)
        except Exception:
            pass

        try:
            page.goto(url, wait_until="load", timeout=int(cfg.timeout_read * 1000))
            # Auto-detect infinite scroll (bounded)
            for _ in range(max(1, cfg.browser_max_pages)):
                page.wait_for_timeout(cfg.browser_wait_ms)
                before = page.content()
                page.evaluate("window.scrollBy(0, document.body.scrollHeight)")
                page.wait_for_timeout(cfg.browser_wait_ms)
                after = page.content()
                if before == after:
                    break
            html = page.content()
            final_url = page.url
            try:
                headers['content-type'] = 'text/html; charset=utf-8'
            except Exception:
                pass
            # Screenshot for debugging
            try:
                shot_path = os.path.join(cfg.cache_root, 'last_screenshot.png')
                page.screenshot(path=shot_path)
            except Exception:
                pass
        finally:
            try:
                context.close(); browser.close()
            except Exception:
                pass

    raw = html.encode('utf-8', errors='ignore')
    # Save
    try:
        meta = {
            'ts': time.time(),
            'status': status,
            'final_url': final_url,
            'headers': headers,
            'content_type': headers.get('content-type', ''),
            'browser_used': True,
        }
        open(body_path, 'wb').write(raw)
        open(meta_path, 'w', encoding='utf-8').write(json.dumps(meta))
    except Exception:
        pass
    return FetchResult(True, status, url, final_url, headers, headers.get('content-type', ''), len(raw), body_path, meta_path, False, True)
