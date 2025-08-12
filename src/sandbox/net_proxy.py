from __future__ import annotations
import os
import time
import json
import hashlib
import logging
import socket
import ipaddress
from typing import Dict, Optional, Tuple
from pathlib import Path
from urllib.parse import urlparse
import fnmatch

try:
    import requests  # type: ignore
except Exception:  # pragma: no cover
    requests = None

logger = logging.getLogger(__name__)

CACHE_ROOT = Path('.sandbox') / 'cache'
CACHE_ROOT.mkdir(parents=True, exist_ok=True)

# In-memory per-process rate counters
_rate_state: Dict[str, Tuple[float, float]] = {}  # host -> (tokens, last_ts)


def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip() in {'1', 'true', 'True', 'yes', 'on'}


def _get_envs() -> Dict[str, str]:
    return {
        'SANDBOX_NET': os.getenv('SANDBOX_NET', 'allowlist'),
        'SANDBOX_NET_ALLOW': os.getenv('SANDBOX_NET_ALLOW', ''),
        'SANDBOX_BLOCK_PRIVATE_IPS': os.getenv('SANDBOX_BLOCK_PRIVATE_IPS', '1'),
        'SANDBOX_ALLOW_HTTP': os.getenv('SANDBOX_ALLOW_HTTP', '0'),
        'SANDBOX_MAX_DOWNLOAD_MB': os.getenv('SANDBOX_MAX_DOWNLOAD_MB', '5'),
        'SANDBOX_RATE_PER_HOST': os.getenv('SANDBOX_RATE_PER_HOST', '12'),
        'SANDBOX_HTTP_CACHE_TTL_S': os.getenv('SANDBOX_HTTP_CACHE_TTL_S', '600'),
        'SANDBOX_HTTP_CACHE_MB': os.getenv('SANDBOX_HTTP_CACHE_MB', '200'),
    }


def _allowlist() -> list[str]:
    env = _get_envs()
    items = [x.strip() for x in env['SANDBOX_NET_ALLOW'].split(',') if x.strip()]
    return items


def _host_allowed(host: str) -> bool:
    patterns = _allowlist()
    for pat in patterns:
        if fnmatch.fnmatch(host, pat):
            return True
    return False


def _resolve_and_check(host: str) -> Tuple[str, list[str]]:
    addrs: list[str] = []
    try:
        infos = socket.getaddrinfo(host, None)
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
    return host, addrs


def _rate_allow(host: str) -> bool:
    env = _get_envs()
    rate = max(1.0, float(env['SANDBOX_RATE_PER_HOST']))  # req/min
    now = time.time()
    tokens, last_ts = _rate_state.get(host, (rate, now))
    # Refill
    elapsed = max(0.0, now - last_ts)
    tokens = min(rate, tokens + elapsed * (rate / 60.0))
    if tokens < 1.0:
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


def fetch_via_policy(
    url: str,
    *,
    method: str = 'GET',
    headers: Optional[Dict[str, str]] = None,
    body: Optional[bytes] = None,
    timeout_s: int = 15,
    max_bytes: Optional[int] = None,
) -> Dict[str, any]:
    """Fetch a URL with allowlist, SSRF, TLS, size, rate-limit, and caching.

    Returns a dict with keys: ok, status, url, redirects, bytes, cached, headers, body (bytes, possibly truncated), truncated, error
    and a compact 'inject' preview string for LLM injection.
    """
    if not requests:
        return {'ok': False, 'error': "requests not installed"}

    env = _get_envs()
    allow_http = _env_bool('SANDBOX_ALLOW_HTTP', False)
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

    host = parsed.hostname or ''
    if net_mode == 'deny':
        return {'ok': False, 'error': 'Network is denied by policy'}
    if not _host_allowed(host):
        return {'ok': False, 'error': f'Host not in allowlist: {host}'}

    # SSRF block
    if _env_bool('SANDBOX_BLOCK_PRIVATE_IPS', True):
        _resolve_and_check(host)

    if not _rate_allow(host):
        return {'ok': False, 'error': f'Rate limit exceeded for host {host}'}

    hdrs = {'User-Agent': 'ollama-turbo-cli/1.0'}
    if headers:
        for k, v in headers.items():
            try:
                hdrs[str(k)] = str(v)
            except Exception:
                continue

    key = _cache_key(method, url, hdrs, body)
    body_path, meta_path = _cache_path(key)
    now = time.time()

    # Cache hit
    if meta_path.exists():
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
                        if not _host_allowed(host):
                            return {'ok': False, 'error': f'Host not in allowlist: {host}'}
                        if _env_bool('SANDBOX_BLOCK_PRIVATE_IPS', True):
                            _resolve_and_check(host)
                except Exception as reval_err:
                    # Any revalidation error -> do not serve cached content
                    return {'ok': False, 'error': f'Cache blocked by policy: {reval_err}'}
                raw = body_path.read_bytes()
                return _summarize_response(url, meta['final_url'], meta['status'], meta.get('headers', {}), raw, True, meta.get('redirects', 0), max_download_mb, max_bytes)
        except Exception:
            pass

    # Perform request with stream to cap size
    redirects = 0
    try:
        with requests.Session() as s:
            # Merge environment settings (verify/cert/proxies)
            settings = s.merge_environment_settings(url, {}, None, True, None)
            resp = s.request(
                method=method.upper(),
                url=url,
                headers=hdrs,
                data=body,
                timeout=max(1, int(timeout_s)),
                stream=True,
                allow_redirects=True,
                verify=settings.get('verify', True),
                cert=settings.get('cert', None),
                proxies=settings.get('proxies', None),
            )
            final_url = resp.url
            status = resp.status_code
            ctype = resp.headers.get('Content-Type', '')
            # Track redirects from history
            redirects = len(resp.history)
            # Ensure redirects obey allowlist and SSRF on each hop
            for h in resp.history:
                p = urlparse(h.url)
                h_host = p.hostname or ''
                if not _host_allowed(h_host):
                    return {'ok': False, 'error': f'Redirected to disallowed host: {h_host}'}
                if _env_bool('SANDBOX_BLOCK_PRIVATE_IPS', True):
                    _resolve_and_check(h_host)
            # Revalidate final URL host/scheme as well
            f_parsed = urlparse(final_url)
            f_host = f_parsed.hostname or ''
            if f_parsed.scheme == 'http' and not allow_http:
                return {'ok': False, 'error': 'HTTP blocked. Enable SANDBOX_ALLOW_HTTP=1 to allow.'}
            if not _host_allowed(f_host):
                return {'ok': False, 'error': f'Final host not in allowlist: {f_host}'}
            if _env_bool('SANDBOX_BLOCK_PRIVATE_IPS', True):
                _resolve_and_check(f_host)

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
            # Save to cache
            try:
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
            return _summarize_response(url, final_url, status, dict(resp.headers), raw, False, redirects, max_download_mb, max_bytes, truncated)
    except requests.RequestException as e:
        return {'ok': False, 'error': f'Network error: {e}'}


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
