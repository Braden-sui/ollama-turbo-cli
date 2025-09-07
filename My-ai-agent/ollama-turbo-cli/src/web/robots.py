from __future__ import annotations
import time
import re
import os
import json
from dataclasses import dataclass
from typing import Dict, Optional
from urllib.parse import urlparse
from urllib import robotparser
from .config import WebConfig
from .fetch import _httpx_client


@dataclass
class RobotsRecord:
    ts: float
    allow: bool
    crawl_delay: Optional[float]
    sitemaps: list[str]
    raw: str


class RobotsPolicy:
    def __init__(self, cfg: Optional[WebConfig] = None) -> None:
        self.cfg = cfg or WebConfig()
        self._mem: Dict[str, RobotsRecord] = {}
        os.makedirs(self.cfg.cache_root, exist_ok=True)

    def _cache_path(self, host: str) -> str:
        safe = re.sub(r"[^a-zA-Z0-9._-]", "_", host)
        return os.path.join(self.cfg.cache_root, f"robots_{safe}.json")

    def _load_cache(self, host: str) -> Optional[RobotsRecord]:
        path = self._cache_path(host)
        try:
            if os.path.isfile(path):
                data = json.loads(open(path, 'r', encoding='utf-8').read())
                cd = data.get('crawl_delay')
                try:
                    cd_f = float(cd) if cd is not None else None
                except Exception:
                    cd_f = None
                return RobotsRecord(
                    ts=float(data.get('ts', 0)),
                    allow=bool(data.get('allow', True)),
                    crawl_delay=cd_f,
                    sitemaps=list(data.get('sitemaps', []) or []),
                    raw=str(data.get('raw', '')),
                )
        except Exception:
            return None
        return None

    def _save_cache(self, host: str, rec: RobotsRecord) -> None:
        path = self._cache_path(host)
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(rec.__dict__, f)
        except Exception:
            pass

    def _fetch_robots(self, base: str) -> RobotsRecord:
        """Fetch and parse robots.txt using urllib.robotparser and cache it."""
        host = urlparse(base).hostname or ''
        rp = robotparser.RobotFileParser()
        robots_url = f"{urlparse(base).scheme}://{host}/robots.txt"
        try:
            headers = {"User-Agent": self.cfg.user_agent, "Accept": "text/plain,*/*;q=0.1"}
            with _httpx_client(self.cfg) as client:
                resp = client.get(robots_url, headers=headers, timeout=self.cfg.timeout_read)
                raw = resp.text if resp.status_code == 200 else ''
        except Exception:
            raw = ''
        rp.parse(raw.splitlines())
        allow = True
        try:
            allow = rp.can_fetch(self.cfg.user_agent, base)
        except Exception:
            allow = True
        crawl_delay = None
        try:
            crawl_delay = rp.crawl_delay(self.cfg.user_agent)
        except Exception:
            crawl_delay = None
        sitemaps: list[str] = []
        try:
            sitemaps = list(getattr(rp, 'site_maps', []) or [])
        except Exception:
            sitemaps = []
        rec = RobotsRecord(ts=time.time(), allow=bool(allow), crawl_delay=crawl_delay, sitemaps=sitemaps, raw=raw)
        return rec

    def check(self, url: str) -> RobotsRecord:
        parsed = urlparse(url)
        host = parsed.hostname or ''
        now = time.time()
        # Memory
        rec = self._mem.get(host)
        if rec and (now - rec.ts) < self.cfg.robots_ttl_seconds:
            return rec
        # Disk
        drec = self._load_cache(host)
        if drec and (now - drec.ts) < self.cfg.robots_ttl_seconds:
            self._mem[host] = drec
            return drec
        # Fetch
        base = f"{parsed.scheme}://{host}/"
        rec = self._fetch_robots(base)
        self._mem[host] = rec
        self._save_cache(host, rec)
        return rec
