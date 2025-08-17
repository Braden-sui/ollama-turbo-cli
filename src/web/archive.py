from __future__ import annotations
import os
import json
import time
from typing import Optional, Dict
from urllib.parse import quote

from .config import WebConfig


def get_memento(url: str, *, cfg: Optional[WebConfig] = None) -> Dict[str, str]:
    """Query Wayback availability API for an existing snapshot.

    Returns a dict with keys: {'archive_url': str, 'timestamp': str}
    If none found, values are empty strings.
    """
    cfg = cfg or WebConfig()
    api = f"https://archive.org/wayback/available?url={quote(url, safe='')}"
    out = {'archive_url': '', 'timestamp': ''}
    try:
        import httpx  # type: ignore
        timeout = cfg.timeout_read
        headers = {'User-Agent': cfg.user_agent, 'Accept': 'application/json'}
        with httpx.Client(timeout=timeout, headers=headers) as c:
            r = c.get(api)
            if r.status_code == 200:
                data = r.json()
                closest = (data.get('archived_snapshots') or {}).get('closest') or {}
                if closest.get('available') and closest.get('url'):
                    out['archive_url'] = str(closest.get('url') or '')
                    out['timestamp'] = str(closest.get('timestamp') or '')
    except Exception:
        return {'archive_url': '', 'timestamp': ''}
    return out


def save_page_now(url: str, *, cfg: Optional[WebConfig] = None) -> Dict[str, str]:
    cfg = cfg or WebConfig()
    api_url = f"https://web.archive.org/save/{quote(url, safe='')}"
    # Retry with backoff, respect 429 Retry-After if configured
    attempts = max(1, cfg.retry_attempts)
    backoff = max(0.0, cfg.retry_backoff_base)
    out = {'archive_url': '', 'timestamp': ''}
    try:
        import httpx  # type: ignore
        for i in range(attempts):
            try:
                with httpx.Client(headers={'User-Agent': cfg.user_agent}, timeout=cfg.timeout_read, follow_redirects=True) as client:
                    resp = client.post(api_url)
                # Parse headers
                arch = resp.headers.get('content-location', '') or resp.headers.get('Content-Location', '')
                if arch and arch.startswith('/'):
                    arch = f"https://web.archive.org{arch}"
                mdt = resp.headers.get('memento-datetime', '') or resp.headers.get('Memento-Datetime', '')
                # Some responses include a memento link header
                if not arch:
                    link = resp.headers.get('link') or resp.headers.get('Link')
                    if isinstance(link, str) and 'rel="memento"' in link:
                        # crude parse: <URL>; rel="memento"
                        try:
                            start = link.find('<')
                            end = link.find('>')
                            if start != -1 and end != -1 and end > start:
                                arch = link[start+1:end]
                        except Exception:
                            arch = ''
                if arch:
                    out = {'archive_url': arch, 'timestamp': mdt}
                    break
                # Handle 429 if configured
                if resp.status_code == 429 and cfg.archive_retry_on_429 and i < attempts - 1:
                    ra = resp.headers.get('retry-after') or resp.headers.get('Retry-After')
                    try:
                        delay = float(ra) if ra and str(ra).strip().isdigit() else backoff * (2 ** i)
                    except Exception:
                        delay = backoff * (2 ** i)
                    delay = min(delay, cfg.retry_backoff_max)
                    time.sleep(max(0.0, delay))
                    continue
                # Non-429 without archive location: backoff and retry
                if i < attempts - 1:
                    time.sleep(min(cfg.retry_backoff_max, backoff * (2 ** i)))
            except Exception:
                if i < attempts - 1:
                    time.sleep(min(cfg.retry_backoff_max, backoff * (2 ** i)))
                continue
    except Exception:
        return {'archive_url': '', 'timestamp': ''}
    return out
