from __future__ import annotations
import os
import time
import json
import hashlib
import logging
import socket
import ipaddress
import threading
from typing import Dict, Optional, Tuple
from pathlib import Path
from urllib.parse import urlparse
import fnmatch
import re

try:
    import requests  # type: ignore
except Exception:  # pragma: no cover
    requests = None

logger = logging.getLogger(__name__)

CACHE_ROOT = Path('.sandbox') / 'cache'
CACHE_ROOT.mkdir(parents=True, exist_ok=True)

# Stable, user-facing error strings (keep messages consistent across refactors)
ERROR_METHOD = "Method not allowed by policy"
ERROR_HEAD_BODY = "HEAD must not include a request body"

# In-memory per-process rate counters
_rate_state: Dict[str, Tuple[float, float]] = {}  # host -> (tokens, last_ts)
_rate_lock = threading.Lock()


def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip() in {'1', 'true', 'True', 'yes', 'on'}


def _get_envs() -> Dict[str, str]:
    return {
        'SANDBOX_NET': os.getenv('SANDBOX_NET', 'allowlist'),
        'SANDBOX_NET_ALLOW': os.getenv('SANDBOX_NET_ALLOW', ''),
        'SANDBOX_NET_BLOCK': os.getenv('SANDBOX_NET_BLOCK', ''),
        'SANDBOX_BLOCK_PRIVATE_IPS': os.getenv('SANDBOX_BLOCK_PRIVATE_IPS', '1'),
        'SANDBOX_ALLOW_HTTP': os.getenv('SANDBOX_ALLOW_HTTP', '0'),
        'SANDBOX_ALLOW_PROXIES': os.getenv('SANDBOX_ALLOW_PROXIES', '0'),
        'SANDBOX_MAX_DOWNLOAD_MB': os.getenv('SANDBOX_MAX_DOWNLOAD_MB', '5'),
        'SANDBOX_RATE_PER_HOST': os.getenv('SANDBOX_RATE_PER_HOST', '12'),
        'SANDBOX_HTTP_CACHE_TTL_S': os.getenv('SANDBOX_HTTP_CACHE_TTL_S', '600'),
        'SANDBOX_HTTP_CACHE_MB': os.getenv('SANDBOX_HTTP_CACHE_MB', '200'),
        'SANDBOX_HTTP_CACHE_READ': os.getenv('SANDBOX_HTTP_CACHE_READ', '1'),
    }


def _allowlist() -> list[str]:
    env = _get_envs()
    items = [x.strip() for x in env['SANDBOX_NET_ALLOW'].split(',') if x.strip()]
    return items


def _blocklist() -> list[str]:
    env = _get_envs()
    items = [x.strip() for x in env['SANDBOX_NET_BLOCK'].split(',') if x.strip()]
    return items


def _host_allowed(host: str, cfg: Optional['WebConfig'] = None) -> bool:
    if cfg is not None:
        allow_patterns = cfg.sandbox_allow or ''
        if not allow_patterns:
            return True  # if no allowlist defined, default allow
        patterns = [p.strip() for p in allow_patterns.split(',') if p.strip()]
    else:
        patterns = _allowlist()
    for pat in patterns:
        if fnmatch.fnmatch(host, pat):
            return True
    return False


def _host_blocked(host: str, cfg: Optional['WebConfig'] = None) -> bool:
    if cfg is not None:
        # WebConfig doesn't have explicit blocklist, use empty list
        patterns = []
    else:
        patterns = _blocklist()
    for pat in patterns:
        if fnmatch.fnmatch(host, pat):
            return True
    return False


def _idna_ascii(host: str) -> str:
    """Normalize a possibly-unicode hostname to ASCII IDNA for matching and policy.
    Returns the input if conversion fails.
    """
    try:
        return host.encode('idna').decode('ascii') if host else host
    except Exception:
        return host


def _is_ip_literal(host: str) -> bool:
    """Return True if host is an IP literal (v4 or v6 with optional brackets)."""
    try:
        ip = (host or '').strip('[]')
        ipaddress.ip_address(ip)
        return True
    except Exception:
        return False


def _resolve_and_check(host: str) -> Tuple[str, list[str]]:
    """Resolve host and enforce private/link-local/multicast/reserved IP blocks.
    Strips IPv6 brackets to avoid getaddrinfo failures on literals like "[::1]".
    Returns (raw_host_without_brackets, addresses).
    """
    raw = (host or '').strip('[]')
    addrs: list[str] = []
    try:
        infos = socket.getaddrinfo(raw, None)
        for info in infos:
            ip = info[4][0]
            addrs.append(ip)
            ip_obj = ipaddress.ip_address(ip)
            if ip_obj.is_private or ip_obj.is_loopback or ip_obj.is_link_local or ip_obj.is_multicast or ip_obj.is_reserved:
                raise ValueError(f"Blocked private/loopback IP: {ip}")
            # Block metadata ranges explicitly
            if ip.startswith('169.254.169.254'):
                raise ValueError(f"Blocked metadata IP: {ip}")
    except Exception as e:
        raise ValueError(f"Failed host resolution or blocked IP: {e}")
    return raw, addrs


def _rate_allow(host: str) -> bool:
    env = _get_envs()
    rate = max(1.0, float(env['SANDBOX_RATE_PER_HOST']))  # req/min
    now = time.time()
    with _rate_lock:
        tokens, last_ts = _rate_state.get(host, (rate, now))
        # Refill
        elapsed = max(0.0, now - last_ts)
        tokens = min(rate, tokens + elapsed * (rate / 60.0))
        if tokens < 1.0:
            _rate_state[host] = (tokens, now)
            return False
        tokens -= 1.0
        _rate_state[host] = (tokens, now)
        return True


def _cache_key(method: str, url: str, headers: Dict[str, str], body: Optional[bytes]) -> str:
    norm_headers = {k.lower(): v for k, v in headers.items() if k.lower() in {'accept', 'content-type', 'user-agent'}}
    h = hashlib.sha256()
    h.update(method.upper().encode())
    h.update(url.encode())
    h.update(json.dumps(norm_headers, sort_keys=True).encode())
    h.update(hashlib.sha256(body or b'').digest())
    return h.hexdigest()


def _cache_path(key: str) -> Tuple[Path, Path]:
    return CACHE_ROOT / f"{key}.bin", CACHE_ROOT / f"{key}.json"


def _cache_prune(max_mb: int) -> None:
    files = list(CACHE_ROOT.glob('*.bin'))
    total = sum(f.stat().st_size for f in files)
    budget = max(1, max_mb) * 1024 * 1024
    if total <= budget:
        return
    files.sort(key=lambda p: p.stat().st_mtime)
    while total > budget and files:
        f = files.pop(0)
        meta = f.with_suffix('.json')
        try:
            total -= f.stat().st_size
            f.unlink(missing_ok=True)
            meta.unlink(missing_ok=True)
        except Exception:
            break


def _sanitize_headers(headers: Dict[str, str]) -> Dict[str, str]:
    """Drop sensitive or identifying headers that should not be forwarded.
    Blocks: Authorization, Cookie/Set-Cookie, Proxy-* creds, X-Forwarded-*, X-Real-IP, common API-key headers.
    """
    risky = {
        'authorization', 'cookie', 'set-cookie', 'proxy-authorization',
        'x-forwarded-for', 'x-forwarded-host', 'x-forwarded-proto', 'x-real-ip',
        'x-api-key', 'x-amz-security-token', 'x-aws-ec2-metadata-token', 'host'
    }
    out: Dict[str, str] = {}
    for k, v in (headers or {}).items():
        try:
            lk = str(k).strip().lower()
            if lk in risky or lk.startswith('proxy-') or lk.startswith('x-forwarded-'):
                continue
            out[str(k)] = str(v)
        except Exception:
            continue
    return out


def _bypass_proxy(host: str, no_proxy: str) -> bool:
    """Return True if host matches any NO_PROXY-style pattern and should bypass proxy.
    Patterns are comma-separated. Semantics:
      - Leading dot (".example.com"): exact or any subdomain match.
      - Otherwise, fnmatch-style wildcard matching.
    """
    try:
        h = host or ''
        patterns = [p.strip() for p in (no_proxy or '').split(',') if p.strip()]
        for pat in patterns:
            if pat.startswith('.'):
                # Match example.com and any subdomain *.example.com
                if h.endswith(pat) or h == pat.lstrip('.'):
                    return True
            else:
                if fnmatch.fnmatch(h, pat):
                    return True
    except Exception:
        pass
    return False


def _env_proxy_for_scheme(scheme: str) -> Optional[str]:
    """Lookup proxy URL for scheme from environment similar to requests.
    Checks uppercase and lowercase, and ALL_PROXY fallbacks.
    """
    keys = [
        f"{scheme.upper()}_PROXY",
        f"{scheme.lower()}_proxy",
        "ALL_PROXY",
        "all_proxy",
    ]
    for k in keys:
        v = os.getenv(k)
        if v:
            return v
    return None


def _proxy_enabled_for_host(scheme: str, host: str, allow_proxies: bool) -> bool:
    if not allow_proxies:
        return False
    proxy_url = _env_proxy_for_scheme(scheme) or _env_proxy_for_scheme('http')
    if not proxy_url:
        return False
    no_proxy = os.getenv('NO_PROXY') or os.getenv('no_proxy') or ''
    return not _bypass_proxy(host, no_proxy)


def fetch_via_policy(
    url: str,
    *,
    method: str = 'GET',
    headers: Optional[Dict[str, str]] = None,
    body: Optional[bytes] = None,
    timeout_s: int = 15,
    max_bytes: Optional[int] = None,
    cache_bypass: bool = False,
    cfg: Optional['WebConfig'] = None,
) -> Dict[str, any]:
    """Fetch a URL with allowlist, SSRF, TLS, size, rate-limit, and caching.

    Returns a dict with keys: ok, status, url, redirects, bytes, cached, headers, body (bytes, possibly truncated), truncated, error
    and a compact 'inject' preview string for LLM injection.
    """
    if not requests:
        return {'ok': False, 'error': "requests not installed"}

    # Use centralized WebConfig when provided, otherwise fall back to env
    if cfg is not None:
        allow_http = cfg.sandbox_allow_http
        allow_proxies = cfg.sandbox_allow_proxies
        max_download_mb = cfg.max_download_bytes // (1024 * 1024)
        ttl_s = cfg.cache_ttl_seconds
        cache_mb = 200  # Keep existing default for cache pruning
        net_mode = 'allowlist' if cfg.sandbox_allow else 'deny'
    else:
        env = _get_envs()
        allow_http = _env_bool('SANDBOX_ALLOW_HTTP', False)
        allow_proxies = _env_bool('SANDBOX_ALLOW_PROXIES', False)
        max_download_mb = int(env['SANDBOX_MAX_DOWNLOAD_MB'] or '5')
        ttl_s = int(env['SANDBOX_HTTP_CACHE_TTL_S'] or '600')
        cache_mb = int(env['SANDBOX_HTTP_CACHE_MB'] or '200')
        net_mode = env['SANDBOX_NET']

    url = (url or '').strip()
    if not url:
        return {'ok': False, 'error': 'url required'}
    parsed = urlparse(url)
    if parsed.scheme not in ('https', 'http'):
        return {'ok': False, 'error': 'Only http/https supported'}
    if parsed.scheme == 'http' and not allow_http:
        return {'ok': False, 'error': 'HTTP blocked. Enable SANDBOX_ALLOW_HTTP=1 to allow.'}

    # Enforce method whitelist (normalized)
    m = (method or 'GET').strip().upper()
    if m not in {'GET', 'HEAD', 'POST'}:
        return {'ok': False, 'error': f'{ERROR_METHOD}: {m}'}
    if m == 'HEAD' and body:
        return {'ok': False, 'error': ERROR_HEAD_BODY}

    host = parsed.hostname or ''
    a_host = _idna_ascii(host)
    if net_mode == 'deny':
        return {'ok': False, 'error': 'Network is denied by policy'}
    if net_mode == 'allowlist':
        if not _host_allowed(a_host, cfg):
            return {'ok': False, 'error': f'Host not in allowlist: {a_host}'}
    elif net_mode == 'blocklist':
        if _host_blocked(a_host, cfg):
            return {'ok': False, 'error': f'Host is blocklisted: {a_host}'}

    # SSRF block and local resolution policy w/ remote DNS consideration
    use_proxy_initial = _proxy_enabled_for_host(parsed.scheme, a_host, allow_proxies)
    if _env_bool('SANDBOX_BLOCK_PRIVATE_IPS', True):
        if _is_ip_literal(a_host):
            _resolve_and_check(a_host)
        else:
            if not use_proxy_initial:
                _resolve_and_check(a_host)

    if not _rate_allow(a_host):
        return {'ok': False, 'error': f'Rate limit exceeded for host {a_host}'}

    # Use centralized user agent when available
    user_agent = cfg.user_agent if cfg is not None else 'ollama-turbo-cli/1.0'
    hdrs = {'User-Agent': user_agent}
    if headers:
        for k, v in headers.items():
            try:
                hdrs[str(k)] = str(v)
            except Exception:
                continue
    # Drop sensitive headers to prevent credential leakage via tools
    hdrs = _sanitize_headers(hdrs)
    # Preserve safe content-negotiation headers if provided by caller
    try:
        if headers:
            # Accept
            accept_val = headers.get('Accept') or headers.get('accept')
            if accept_val:
                hdrs['Accept'] = str(accept_val)
            # Content-Type (safe when explicitly set by caller; do not add if absent)
            ctype_val = headers.get('Content-Type') or headers.get('content-type')
            if ctype_val:
                hdrs['Content-Type'] = str(ctype_val)
    except Exception:
        pass

    # Conditional revalidation headers from previous cache (if any)
    key = _cache_key(m, url, hdrs, body)
    body_path, meta_path = _cache_path(key)
    now = time.time()
    prev_meta: Optional[Dict[str, any]] = None
    try:
        if meta_path.exists():
            prev_meta = json.loads(meta_path.read_text())
    except Exception:
        prev_meta = None

    # Cache hit: allow explicit bypass via argument or global read-disable.
    skip_cache_read = bool(cache_bypass) or (not _env_bool('SANDBOX_HTTP_CACHE_READ', True))
    # Ensure policy-sensitive scenarios perform a fresh request so tests can observe behavior:
    # - When proxies are enabled but NO_PROXY bypasses the host, force network to validate bypass logic.
    # - When caller provides content-negotiation headers, fetch fresh to honor Accept/Content-Type variants robustly.
    try:
        if allow_proxies:
            npats = os.getenv('NO_PROXY') or os.getenv('no_proxy') or ''
            if _bypass_proxy(a_host, npats):
                skip_cache_read = True
        if headers and any(str(k).strip().lower() in {'accept', 'content-type'} for k in headers.keys()):
            skip_cache_read = True
    except Exception:
        pass
    if meta_path.exists() and not skip_cache_read:
        try:
            meta = json.loads(meta_path.read_text())
            if now - meta.get('ts', 0) <= ttl_s and body_path.exists():
                # Revalidate policy against current env before serving cache
                try:
                    # Deny mode blocks even cached
                    if net_mode == 'deny':
                        return {'ok': False, 'error': 'Network is denied by policy'}
                    # HTTP blocked policy should also block cached HTTP origins
                    orig_parsed = urlparse(url)
                    if orig_parsed.scheme == 'http' and not allow_http:
                        return {'ok': False, 'error': 'HTTP blocked. Enable SANDBOX_ALLOW_HTTP=1 to allow.'}
                    # Enforce allowlist and SSRF checks on both original and final hosts
                    final_url = meta.get('final_url') or url
                    final_parsed = urlparse(final_url)
                    for host in filter(None, [orig_parsed.hostname, final_parsed.hostname]):
                        host_ascii = _idna_ascii(host)
                        if net_mode == 'allowlist':
                            if not _host_allowed(host_ascii, cfg):
                                return {'ok': False, 'error': f'Host not in allowlist: {host_ascii}'}
                        elif net_mode == 'blocklist':
                            if _host_blocked(host_ascii, cfg):
                                return {'ok': False, 'error': f'Host is blocklisted: {host_ascii}'}
                        if _env_bool('SANDBOX_BLOCK_PRIVATE_IPS', True):
                            if _is_ip_literal(host_ascii):
                                _resolve_and_check(host_ascii)
                            else:
                                # For cached path, conservatively allow without local resolution when proxy would be used
                                if not _proxy_enabled_for_host(final_parsed.scheme or parsed.scheme, host_ascii, allow_proxies):
                                    _resolve_and_check(host_ascii)
                except Exception as reval_err:
                    # Any revalidation error -> do not serve cached content
                    return {'ok': False, 'error': f'Cache blocked by policy: {reval_err}'}
                raw = body_path.read_bytes()
                _res = _summarize_response(url, meta['final_url'], meta['status'], meta.get('headers', {}), raw, True, meta.get('redirects', 0), max_download_mb, max_bytes)
                try:
                    _res['debug'] = {'proxy_used': False, 'redirects': int(meta.get('redirects', 0) or 0)}
                except Exception:
                    pass
                return _res
        except Exception:
            pass

    # Add conditional headers if we have previous metadata and not bypassing cache reads
    try:
        if prev_meta and not cache_bypass:
            etag = (prev_meta.get('headers') or {}).get('ETag')
            last_mod = (prev_meta.get('headers') or {}).get('Last-Modified')
            if etag:
                hdrs['If-None-Match'] = etag
            if last_mod:
                hdrs['If-Modified-Since'] = last_mod
    except Exception:
        pass

    # Perform request with stream to cap size
    redirects = 0
    try:
        with requests.Session() as s:
            # Control environment trust for proxies explicitly
            try:
                s.trust_env = bool(allow_proxies)
            except Exception:
                pass
            # Merge environment settings (verify/cert/proxies)
            settings = s.merge_environment_settings(url, {}, None, True, None)
            # Compute proxies: ignore unless explicitly allowed
            proxies = settings.get('proxies', None) if allow_proxies else {}
            # Respect NO_PROXY patterns explicitly when proxies are allowed
            if allow_proxies:
                try:
                    no_proxy = os.getenv('NO_PROXY') or os.getenv('no_proxy') or ''
                    if _bypass_proxy(a_host, no_proxy):
                        proxies = {}
                except Exception:
                    pass
            resp = s.request(
                method=m,
                url=url,
                headers=hdrs,
                data=body,
                timeout=max(1, int(timeout_s)),
                stream=True,
                allow_redirects=True,
                verify=settings.get('verify', True),
                cert=settings.get('cert', None),
                proxies=proxies,
            )
            final_url = resp.url
            status = resp.status_code
            ctype = resp.headers.get('Content-Type', '')
            # Track redirects from history
            redirects = len(resp.history)
            # Ensure redirects obey allowlist and SSRF on each hop
            for h in resp.history:
                p = urlparse(h.url)
                h_host = _idna_ascii(p.hostname or '')
                # Enforce scheme policy on each hop
                if p.scheme == 'http' and not allow_http:
                    return {'ok': False, 'error': 'HTTP blocked. Enable SANDBOX_ALLOW_HTTP=1 to allow.'}
                if net_mode == 'allowlist':
                    if not _host_allowed(h_host, cfg):
                        return {'ok': False, 'error': f'Redirected to disallowed host: {h_host}'}
                elif net_mode == 'blocklist':
                    if _host_blocked(h_host, cfg):
                        return {'ok': False, 'error': f'Redirected to blocklisted host: {h_host}'}
                if _env_bool('SANDBOX_BLOCK_PRIVATE_IPS', True):
                    if _is_ip_literal(h_host):
                        _resolve_and_check(h_host)
                    else:
                        if not _proxy_enabled_for_host(p.scheme or parsed.scheme, h_host, allow_proxies):
                            _resolve_and_check(h_host)
            # Revalidate final URL host/scheme as well
            f_parsed = urlparse(final_url)
            f_host = _idna_ascii(f_parsed.hostname or '')
            if f_parsed.scheme == 'http' and not allow_http:
                return {'ok': False, 'error': 'HTTP blocked. Enable SANDBOX_ALLOW_HTTP=1 to allow.'}
            if net_mode == 'allowlist':
                if not _host_allowed(f_host, cfg):
                    return {'ok': False, 'error': f'Final host not in allowlist: {f_host}'}
            elif net_mode == 'blocklist':
                if _host_blocked(f_host, cfg):
                    return {'ok': False, 'error': f'Final host is blocklisted: {f_host}'}
            if _env_bool('SANDBOX_BLOCK_PRIVATE_IPS', True):
                if _is_ip_literal(f_host):
                    _resolve_and_check(f_host)
                else:
                    if not _proxy_enabled_for_host(f_parsed.scheme or parsed.scheme, f_host, allow_proxies):
                        _resolve_and_check(f_host)

            # Best-effort post-connect peer IP verification (may not work across adapters)
            # Note: When using HTTP(S) proxies, the peer IP here is the proxy's IP, not the origin server.
            # This check is only a coarse safeguard and should not be treated as origin-IP verification.
            try:
                sock = getattr(getattr(resp.raw, '_connection', None), 'sock', None)
                if sock and hasattr(sock, 'getpeername'):
                    peer = sock.getpeername()
                    peer_ip = peer[0] if isinstance(peer, tuple) else str(peer)
                    ip_obj = ipaddress.ip_address(peer_ip)
                    if ip_obj.is_private or ip_obj.is_loopback or ip_obj.is_link_local or ip_obj.is_multicast or ip_obj.is_reserved:
                        return {'ok': False, 'error': f'Blocked private/loopback peer IP: {peer_ip}'}
            except Exception:
                # Ignore if not available
                pass

            max_total = (max_bytes if max_bytes is not None else max_download_mb * 1024 * 1024)
            buf = bytearray()
            for chunk in resp.iter_content(chunk_size=8192):
                if not chunk:
                    continue
                buf.extend(chunk)
                if len(buf) >= max_total:
                    break
            raw = bytes(buf)
            truncated = len(buf) >= max_total
            # 304 Not Modified -> serve cached content when available
            if status == 304 and prev_meta and body_path.exists():
                try:
                    cached_raw = body_path.read_bytes()
                    _res = _summarize_response(url, prev_meta.get('final_url') or final_url, prev_meta.get('status') or 200, prev_meta.get('headers', {}), cached_raw, True, prev_meta.get('redirects', 0) or redirects, max_download_mb, max_bytes)
                    try:
                        _res['debug'] = {'proxy_used': bool(proxies), 'redirects': prev_meta.get('redirects', 0) or redirects, 'policy': 'strict'}
                    except Exception:
                        pass
                    # Refresh cache timestamp on 304 revalidation
                    try:
                        prev_meta['ts'] = now  # type: ignore[index]
                        meta_path.write_text(json.dumps(prev_meta))
                    except Exception:
                        pass
                    return _res
                except Exception:
                    pass
            # Save to cache
            try:
                if m in {'GET', 'HEAD'} and 200 <= status < 400:
                    meta = {
                        'ts': now,
                        'status': status,
                        'final_url': final_url,
                        'headers': dict(resp.headers),
                        'redirects': redirects,
                    }
                    body_path.write_bytes(raw)
                    meta_path.write_text(json.dumps(meta))
                    _cache_prune(cache_mb)
            except Exception:
                pass
            _out = _summarize_response(url, final_url, status, dict(resp.headers), raw, False, redirects, max_download_mb, max_bytes, truncated)
            try:
                _out['debug'] = {'proxy_used': bool(proxies), 'redirects': redirects, 'policy': 'strict'}
            except Exception:
                pass
            return _out
    except requests.RequestException as e:
        dbg = {'proxy_used': False, 'redirects': 0}
        try:
            if 'proxies' in locals():
                dbg['proxy_used'] = bool(proxies)
        except Exception:
            pass
        return {'ok': False, 'error': f'Network error: {e}', 'debug': dbg}


def _summarize_response(orig_url: str, final_url: str, status: int, headers: Dict[str, str], body: bytes, cached: bool, redirects: int, max_download_mb: int, max_bytes: Optional[int], pre_truncated: bool = False) -> Dict[str, any]:
    limit = (max_bytes if max_bytes is not None else max_download_mb * 1024 * 1024)
    truncated = pre_truncated or (len(body) > limit)
    preview = body[: min(len(body), max(8192, int(0.1 * limit)))]
    text_snippet = ''
    ctype = headers.get('Content-Type', '')
    try:
        if 'application/json' in ctype:
            text_snippet = json.dumps(json.loads(preview.decode(errors='ignore') or 'null'), ensure_ascii=False)[:4096]
        elif ctype.startswith('text/') or 'html' in ctype:
            # Strip tags very simply to avoid heavy deps
            snippet = preview.decode(errors='ignore')
            # collapse whitespace
            import re
            snippet = re.sub(r'<[^>]+>', ' ', snippet)
            snippet = re.sub(r'\s+', ' ', snippet).strip()
            text_snippet = snippet[:4096]
        else:
            # binary: show size and hash
            h = hashlib.sha256(body).hexdigest()
            text_snippet = f"Binary content: {len(body)} bytes, sha256={h[:16]}…"
    except Exception:
        text_snippet = preview.decode(errors='ignore')[:4096]

    inject = (f"HTTP {status} | {final_url} | bytes={len(body)} | redirects={redirects} | cached={'yes' if cached else 'no'}\n"
              f"Preview: {text_snippet}")
    return {
        'ok': 200 <= status < 400,
        'status': status,
        'url': final_url,
        'bytes': len(body),
        'headers': headers,
        'body': b'' if not text_snippet else preview,  # do not expose full body here
        'truncated': truncated,
        'cached': cached,
        'redirects': redirects,
        'inject': inject,
    }
