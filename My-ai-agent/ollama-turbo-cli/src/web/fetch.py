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
import ipaddress
import fnmatch
from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode

from .config import WebConfig
from .robots import RobotsPolicy


# Note: Remote DNS semantics
# - When a proxy is enabled and selected for a domain host, we deliberately skip local DNS
#   to avoid leaking queries (remote DNS via proxy). IP-literals are always resolved locally
#   and checked against private/link-local/multicast/reserved ranges to enforce SSRF policy.
# - With HTTP(S) proxies, any peer IP obtained post-connect is the proxy's IP, not the origin.
#   Do not treat it as origin verification.
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
    # Single source of truth: use cfg.sandbox_allow only (no env fallback here)
    allow = cfg.sandbox_allow or ''
    if not allow:
        return True  # if no allowlist defined, default allow
    for pat in (p.strip() for p in allow.split(',') if p.strip()):
        if fnmatch.fnmatch(host or '', pat):
            return True
    return False


def _check_ip_blocks(host: str) -> None:
    import socket, ipaddress
    # Strip brackets for IPv6 literals like "[::1]" to keep getaddrinfo happy
    raw = (host or '').strip('[]')
    infos = socket.getaddrinfo(raw, None)
    for info in infos:
        ip_any = info[4][0]
        ip_str = str(ip_any)
        ip_obj = ipaddress.ip_address(ip_str)
        if ip_obj.is_private or ip_obj.is_loopback or ip_obj.is_link_local or ip_obj.is_multicast or ip_obj.is_reserved:
            raise ValueError(f"Blocked private/loopback IP: {ip_str}")
        if ip_str.startswith('169.254.169.254'):
            raise ValueError(f"Blocked metadata IP: {ip_str}")


def _cache_paths(cfg: WebConfig, url: str) -> Tuple[str, str]:
    # Optional per-worker sharding to reduce parallel contention
    try:
        root = cfg.cache_root
        if (os.getenv('WEB_CACHE_PER_WORKER', '0').strip().lower() not in {'0','false','no','off'}):
            worker = os.getenv('PYTEST_XDIST_WORKER') or os.getenv('WORKER_ID')
            if worker:
                root = os.path.join(root, worker)
        os.makedirs(root, exist_ok=True)
    except Exception:
        root = cfg.cache_root
        try:
            os.makedirs(root, exist_ok=True)
        except Exception:
            pass
    key = hashlib.sha256(_canonicalize_url(url).encode()).hexdigest()
    return os.path.join(root, f"{key}.bin"), os.path.join(root, f"{key}.json")


def _idna_ascii(host: str) -> str:
    try:
        return host.encode('idna').decode('ascii') if host else host
    except Exception:
        return host


def _is_ip_literal(host: str) -> bool:
    try:
        ip = (host or '').strip('[]')
        ipaddress.ip_address(ip)
        return True
    except Exception:
        return False


def _bypass_proxy(host: str, no_proxy: str) -> bool:
    try:
        h = host or ''
        patterns = [p.strip() for p in (no_proxy or '').split(',') if p.strip()]
        for pat in patterns:
            if pat.startswith('.'):
                if h.endswith(pat) or h == pat.lstrip('.'):
                    return True
            else:
                if fnmatch.fnmatch(h, pat):
                    return True
    except Exception:
        pass
    return False


def _cfg_proxy_for_scheme(cfg: WebConfig, scheme: str) -> Optional[str]:
    try:
        if scheme.lower() == 'https':
            return cfg.https_proxy or cfg.all_proxy or None
        if scheme.lower() == 'http':
            return cfg.http_proxy or cfg.all_proxy or None
        return cfg.all_proxy or None
    except Exception:
        return None


def _httpx_client(cfg: WebConfig):
    import httpx
    # httpx requires either a default timeout or all four: connect, read, write, pool
    timeout = httpx.Timeout(
        connect=cfg.timeout_connect,
        read=cfg.timeout_read,
        write=cfg.timeout_write,
        pool=cfg.timeout_connect,
    )
    limits = httpx.Limits(max_connections=cfg.max_connections, max_keepalive_connections=cfg.max_keepalive)
    # Build proxies from centralized cfg (do not read env here)
    proxies = None
    def _norm_proxy(u: Optional[str]) -> Optional[str]:
        try:
            if not u:
                return None
            s = str(u).strip()
            if s.lower() in {"none", "null", "false", "0"}:
                return None
            # Basic URL shape check; accept http(s) and socks schemes
            if not (s.startswith("http://") or s.startswith("https://") or s.startswith("socks")):
                return None
            return s
        except Exception:
            return None
    try:
        if cfg.sandbox_allow_proxies:
            http_p = _norm_proxy(cfg.http_proxy) or _norm_proxy(cfg.all_proxy)
            https_p = _norm_proxy(cfg.https_proxy) or _norm_proxy(cfg.all_proxy)
            if http_p or https_p:
                proxies = {}
                if http_p:
                    proxies["http"] = http_p
                if https_p:
                    proxies["https"] = https_p
    except Exception:
        proxies = None
    # Always disable trust_env so only cfg drives behavior
    client_kwargs = {
        "timeout": timeout,
        "limits": limits,
        "headers": {"User-Agent": cfg.user_agent},
        "follow_redirects": cfg.follow_redirects,
        "trust_env": False,
    }
    # Prefer HTTP/1.1 for wider compatibility; set http2 only if supported
    try:
        client_kwargs["http2"] = False
    except Exception:
        pass
    # Include proxies only when allowed and supported
    if proxies:
        client_kwargs["proxies"] = proxies
    try:
        return httpx.Client(**client_kwargs)
    except TypeError:
        # Fallback: remove proxies if this httpx version doesn't accept the kwarg
        client_kwargs.pop("proxies", None)
        try:
            return httpx.Client(**client_kwargs)
        except TypeError:
            # Final fallback: remove http2 flag if unsupported
            client_kwargs.pop("http2", None)
            return httpx.Client(**client_kwargs)


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
    # IDNA-normalize host for allowlist matching
    if not _host_allowed(cfg, _idna_ascii(parsed.hostname or '')):
        return FetchResult(False, 0, url, url, {}, '', 0, None, None, False, False, reason=f'Host not in allowlist: {parsed.hostname}')
    # Apply remote DNS semantics: if proxies are enabled and would be used for this host (and the host is not an IP literal),
    # skip local DNS resolution to avoid leaks. Always resolve and check IP literals, or when not using a proxy.
    try:
        host = parsed.hostname or ''
        a_host = _idna_ascii(host)
        use_proxy = False
        try:
            if cfg.sandbox_allow_proxies:
                proxy_url = _cfg_proxy_for_scheme(cfg, parsed.scheme) or _cfg_proxy_for_scheme(cfg, 'http')
                if proxy_url:
                    use_proxy = not _bypass_proxy(a_host, cfg.no_proxy or '')
        except Exception:
            use_proxy = False
        if _is_ip_literal(a_host):
            _check_ip_blocks(a_host)
        else:
            if not use_proxy:
                _check_ip_blocks(a_host)
    except Exception as e:
        return FetchResult(False, 0, url, url, {}, '', 0, None, None, False, False, reason=str(e))

    if cfg.respect_robots:
        rec = robots.check(url)
        if not rec.allow:
            delay = rec.crawl_delay or 0
            if delay:
                time.sleep(min(delay, cfg.max_crawl_delay_s))
            return FetchResult(False, 0, url, url, {}, '', 0, None, None, False, False, reason='Blocked by robots.txt')
        # Enforce crawl-delay for allowed hosts as well
        host = parsed.hostname or ''
        if rec.crawl_delay:
            with _lf_lock:
                last = _last_fetch.get(host, 0.0)
                now = time.time()
                wait = rec.crawl_delay - (now - last)
            if wait and wait > 0:
                time.sleep(min(wait, cfg.max_crawl_delay_s))
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
            with open(meta_path, 'r', encoding='utf-8') as _mf:
                meta = json.load(_mf)
            if now - float(meta.get('ts', 0)) <= cfg.cache_ttl_seconds:
                with open(body_path, 'rb') as _bf:
                    raw = _bf.read()
                return FetchResult(True, int(meta.get('status', 200)), url, meta.get('final_url', url), dict(meta.get('headers', {})), meta.get('content_type', ''), len(raw), body_path, meta_path, True, bool(meta.get('browser_used', False)))
        except Exception:
            pass

    # Prepare conditional headers if any (for revalidation)
    prev_meta: Optional[Dict[str, Any]] = None
    prev_etag = None
    prev_lm = None
    try:
        if os.path.isfile(meta_path):
            with open(meta_path, 'r', encoding='utf-8') as _pmf:
                prev_meta = json.load(_pmf)
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
                t_start = time.time()
                t_first: Optional[float] = None
                with client.stream("GET", url, headers=req_headers) as resp:
                    status = resp.status_code
                    # Handle 304 revalidation
                    if status == 304 and prev_meta and os.path.isfile(body_path):
                        try:
                            with open(body_path, 'rb') as _bf2:
                                raw = _bf2.read()
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
                            if t_first is None:
                                t_first = time.time()
                            if len(buf) >= cfg.max_download_bytes:
                                break
                    data = bytes(buf)
                    headers = {k: v for k, v in resp.headers.items()}
                    # Attach timings as debug headers
                    try:
                        ttfb_ms = int(((t_first or time.time()) - t_start) * 1000)
                        ttc_ms = int((time.time() - t_start) * 1000)
                        headers['x-debug-ttfb-ms'] = str(max(0, ttfb_ms))
                        headers['x-debug-ttc-ms'] = str(max(0, ttc_ms))
                    except Exception:
                        pass
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
        # Revalidate final URL host/scheme and SSRF/IP policy similar to sandbox path
        try:
            f_parsed = urlparse(final_url)
            if f_parsed.scheme == 'http' and not cfg.sandbox_allow_http:
                return FetchResult(False, int(status or 0), url, final_url, headers, content_type, len(data or b''), None, None, False, False, reason='HTTP blocked by policy')
            # Use IDNA-normalized host for allowlist check
            if not _host_allowed(cfg, _idna_ascii(f_parsed.hostname or '')):
                return FetchResult(False, int(status or 0), url, final_url, headers, content_type, len(data or b''), None, None, False, False, reason=f'Final host not in allowlist: {f_parsed.hostname}')
            # Apply remote DNS semantics for final host
            f_host = _idna_ascii(f_parsed.hostname or '')
            use_proxy_f = False
            try:
                if cfg.sandbox_allow_proxies:
                    purl = _cfg_proxy_for_scheme(cfg, f_parsed.scheme) or _cfg_proxy_for_scheme(cfg, 'http')
                    if purl:
                        use_proxy_f = not _bypass_proxy(f_host, cfg.no_proxy or '')
            except Exception:
                use_proxy_f = False
            if _is_ip_literal(f_host):
                _check_ip_blocks(f_host)
            else:
                if not use_proxy_f:
                    _check_ip_blocks(f_host)
        except Exception as reval_err:
            return FetchResult(False, int(status or 0), url, final_url, headers, content_type, len(data or b''), None, None, False, False, reason=str(reval_err))
        raw = data
        # Alt UA fallback for HTML shells / small bodies before escalating to browser
        try:
            need_alt = False
            if ('text/html' in (content_type or '')):
                if (status in (403, 401)) or (len(raw or b'') < 1024):
                    need_alt = True
            if need_alt:
                alt_headers = dict(common_headers)
                alt_headers['User-Agent'] = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36'
                with client.stream("GET", url, headers=alt_headers) as resp2:
                    st2 = resp2.status_code
                    buf2 = bytearray()
                    for chunk in resp2.iter_bytes():
                        if chunk:
                            buf2 += chunk
                            if len(buf2) >= cfg.max_download_bytes:
                                break
                    raw2 = bytes(buf2)
                    ct2 = resp2.headers.get('content-type', '')
                    if st2 == 200 and ('text/html' in (ct2 or '')) and len(raw2) >= len(raw or b''):
                        status = st2
                        content_type = ct2
                        raw = raw2
                        final_url = str(resp2.request.url)
                        headers = {k: v for k, v in resp2.headers.items()}
        except Exception:
            pass
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
            try:
                # Attach approximate timings for browser path
                headers['x-debug-ttfb-ms'] = headers.get('x-debug-ttfb-ms', '0')
                headers['x-debug-ttc-ms'] = headers.get('x-debug-ttc-ms', '0')
            except Exception:
                pass
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
        # Atomic writes to avoid partial reads by parallel workers
        tmp_body = body_path + ".tmp"
        tmp_meta = meta_path + ".tmp"
        try:
            with open(tmp_body, 'wb') as _bfo:
                _bfo.write(raw)
            os.replace(tmp_body, body_path)
        finally:
            try:
                if os.path.exists(tmp_body):
                    os.remove(tmp_body)
            except Exception:
                pass
        try:
            with open(tmp_meta, 'w', encoding='utf-8') as _mfo:
                json.dump(meta, _mfo)
            os.replace(tmp_meta, meta_path)
        finally:
            try:
                if os.path.exists(tmp_meta):
                    os.remove(tmp_meta)
            except Exception:
                pass
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
        # Use conservative launch args for CI/parallel stability
        browser = p.chromium.launch(headless=True, args=["--disable-dev-shm-usage", "--no-sandbox"])  # args ignored if not supported
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
            # Try to reach network idle to reduce teardown races
            try:
                page.wait_for_load_state("networkidle", timeout=int(max(500, cfg.browser_wait_ms)))
            except Exception:
                pass
            html = page.content()
            final_url = page.url
            try:
                headers['content-type'] = 'text/html; charset=utf-8'
            except Exception:
                pass
            # Screenshot for debugging
            try:
                import time as _t, threading as _th, os as _os
                ts = int(_t.time() * 1000)
                pid = _os.getpid()
                tid = getattr(_th.current_thread(), 'ident', 0)
                shot_name = f"last_screenshot_{pid}_{tid}_{ts}.png"
                shot_path = os.path.join(cfg.cache_root, shot_name)
                page.screenshot(path=shot_path)
            except Exception:
                pass
        finally:
            try:
                try:
                    page.close()
                except Exception:
                    pass
                context.close()
                browser.close()
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
        tmp_body = body_path + ".tmp"
        tmp_meta = meta_path + ".tmp"
        try:
            with open(tmp_body, 'wb') as _bfo:
                _bfo.write(raw)
            os.replace(tmp_body, body_path)
        finally:
            try:
                if os.path.exists(tmp_body):
                    os.remove(tmp_body)
            except Exception:
                pass
        try:
            with open(tmp_meta, 'w', encoding='utf-8') as _mfo:
                json.dump(meta, _mfo)
            os.replace(tmp_meta, meta_path)
        finally:
            try:
                if os.path.exists(tmp_meta):
                    os.remove(tmp_meta)
            except Exception:
                pass
    except Exception:
        pass
    return FetchResult(True, status, url, final_url, headers, headers.get('content-type', ''), len(raw), body_path, meta_path, False, True)
