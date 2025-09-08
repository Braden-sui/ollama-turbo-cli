import sys
from pathlib import Path
import json

# Ensure project root on sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Import pipeline first to trigger dotenv loading inside pipeline module
from src.web.pipeline import WebConfig, _httpx_client  # type: ignore


def main():
    cfg = WebConfig()
    print(json.dumps({
        'ua': cfg.user_agent,
        'allow': cfg.sandbox_allow,
        'respect_robots': cfg.respect_robots,
        'allow_browser': cfg.allow_browser,
        'sandbox_allow_proxies': cfg.sandbox_allow_proxies,
        'http_proxy': cfg.http_proxy,
        'https_proxy': cfg.https_proxy,
        'all_proxy': cfg.all_proxy,
        'no_proxy': cfg.no_proxy,
    }, ensure_ascii=False))

    endpoints = [
        ("duckduckgo_api", "https://api.duckduckgo.com/", {"q": "test", "format": "json", "no_html": "1", "no_redirect": "1", "t": "net_check"}),
        ("duckduckgo_lite", "https://duckduckgo.com/lite/", {"q": "test"}),
        ("duckduckgo_html", "https://html.duckduckgo.com/html/", {"q": "test"}),
    ]

    with _httpx_client(cfg) as c:
        for name, url, params in endpoints:
            try:
                r = c.get(url, params=params, timeout=cfg.timeout_read)
                # Limit body preview
                body = (r.text or "")[:200].replace("\n", " ")
                print(json.dumps({
                    'name': name,
                    'url': str(r.request.url),
                    'status': r.status_code,
                    'content_type': r.headers.get('content-type', ''),
                    'preview': body,
                }, ensure_ascii=False))
            except Exception as e:
                print(json.dumps({
                    'name': name,
                    'url': url,
                    'error': str(e),
                }, ensure_ascii=False))


if __name__ == '__main__':
    main()
