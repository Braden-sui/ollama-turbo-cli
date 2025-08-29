import types

from src.web.search import search


class _DummyReq:
    def __init__(self, url: str):
        self.url = url


class _DummyResp:
    def __init__(self, status_code=200, text="", content=b"", url=""):
        self.status_code = status_code
        self.text = text
        self.content = content or text.encode()
        self.request = _DummyReq(url)
        self.headers = {}


class _FakeHTTPXClient:
    def __init__(self, timeout=None, headers=None, follow_redirects=False):
        self.timeout = timeout
        self.headers = headers or {}
        self.follow_redirects = follow_redirects

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def get(self, url, params=None):
        if url.endswith("/robots.txt"):
            txt = "Sitemap: https://example.com/sitemap.xml\n"
            return _DummyResp(200, text=txt, url=url)
        if url.endswith("/sitemap.xml"):
            xml = (
                "<?xml version='1.0' encoding='UTF-8'?>"
                "<urlset xmlns='http://www.sitemaps.org/schemas/sitemap/0.9'>"
                "  <url><loc>https://example.com/a</loc></url>"
                "  <url><loc>https://example.com/b</loc></url>"
                "</urlset>"
            )
            return _DummyResp(200, content=xml.encode("utf-8"), url=url)
        # default not found
        return _DummyResp(404, text="", url=url)

    def post(self, url, json=None):
        return _DummyResp(404, text="", url=url)


def test_sitemap_ingestion_returns_results(monkeypatch):
    # Enable sitemap ingestion via env
    monkeypatch.setenv("WEB_SITEMAP_ENABLED", "1")
    monkeypatch.setenv("WEB_SITEMAP_MAX_URLS", "5")
    monkeypatch.setenv("WEB_SITEMAP_TIMEOUT_S", "2")
    monkeypatch.setenv("WEB_SITEMAP_INCLUDE_SUBS", "1")

    # Ensure provider keys unset so provider calls are skipped
    monkeypatch.delenv("BRAVE_API_KEY", raising=False)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.delenv("EXA_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_PSE_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_PSE_CX", raising=False)

    # Patch httpx.Client used in sitemap discovery/parsing
    import src.web.search as search_mod
    # Replace just the Client class within httpx module used by search_mod
    fake_httpx = types.SimpleNamespace(Client=_FakeHTTPXClient)
    monkeypatch.setattr(search_mod, "httpx", fake_httpx, raising=True)

    results = search("any query", site="example.com")
    # We expect entries from the sitemap
    urls = sorted([r.url for r in results if r.source == "sitemap"])[:2]
    assert urls == ["https://example.com/a", "https://example.com/b"]
