import json
import pytest

from src.plugin_loader import reload_plugins, TOOL_FUNCTIONS


def call_tool(name: str, **kwargs) -> dict:
    fn = TOOL_FUNCTIONS[name]
    out = fn(**kwargs)
    try:
        return json.loads(out)
    except Exception:
        return {"raw": out}


@pytest.fixture(autouse=True)
def _env_defaults(monkeypatch):
    # Default: allowlist with example.com
    monkeypatch.setenv('SANDBOX_NET', 'allowlist')
    monkeypatch.setenv('SANDBOX_NET_ALLOW', 'example.com')
    monkeypatch.setenv('SANDBOX_ALLOW_HTTP', '0')
    monkeypatch.setenv('SANDBOX_BLOCK_PRIVATE_IPS', '1')
    monkeypatch.setenv('SANDBOX_HTTP_CACHE_TTL_S', '600')
    monkeypatch.setenv('SANDBOX_HTTP_CACHE_MB', '50')
    monkeypatch.setenv('SANDBOX_ALLOW_PROXIES', '0')
    reload_plugins()
    yield


def test_cache_bypass_true_skips_cache(tmp_path, monkeypatch):
    from src.sandbox import net_proxy as np

    # Isolate cache
    cache_dir = tmp_path / 'cache'
    cache_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(np, 'CACHE_ROOT', cache_dir)

    # Avoid real DNS
    monkeypatch.setattr(np, '_resolve_and_check', lambda host: (host, ['93.184.216.34']))

    call_count = {'n': 0}

    class FakeResp:
        def __init__(self, url: str):
            self.url = url
            self.status_code = 200
            self.headers = {'Content-Type': 'text/plain'}
            self.history = []

        def iter_content(self, chunk_size=8192):
            yield b'ok'

    class FakeSession:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def merge_environment_settings(self, url, proxies, stream, verify, cert):
            return {'verify': True, 'cert': None, 'proxies': None}

        def request(self, method, url, headers, data, timeout, stream, allow_redirects, verify, cert, proxies):
            call_count['n'] += 1
            return FakeResp(url)

    class FakeRequests:
        def Session(self):
            return FakeSession()

    monkeypatch.setattr(np, 'requests', FakeRequests())

    # First call populates cache
    r1 = call_tool('web_fetch', url='https://example.com')
    assert r1.get('ok') is True

    # Second call with explicit cache_bypass should force a new network request
    r2 = call_tool('web_fetch', url='https://example.com', cache_bypass=True)
    assert r2.get('ok') is True
    assert r2.get('cached') is False
    assert call_count['n'] == 2


def test_no_proxy_bypasses_proxies_when_opt_in(monkeypatch):
    from src.sandbox import net_proxy as np

    # Allow proxies and set NO_PROXY to bypass for example.com
    monkeypatch.setenv('SANDBOX_ALLOW_PROXIES', '1')
    monkeypatch.setenv('NO_PROXY', 'example.com')

    # Avoid real DNS
    monkeypatch.setattr(np, '_resolve_and_check', lambda host: (host, ['93.184.216.34']))

    captured = {}

    class FakeResp:
        def __init__(self, url: str):
            self.url = url
            self.status_code = 200
            self.headers = {'Content-Type': 'text/plain'}
            self.history = []

        def iter_content(self, chunk_size=8192):
            yield b'ok'

    class FakeSession:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def merge_environment_settings(self, url, proxies, stream, verify, cert):
            # Simulate environment proxies that would normally be used
            return {'verify': True, 'cert': None, 'proxies': {'http': 'http://proxy:8080', 'https': 'http://proxy:8080'}}

        def request(self, method, url, headers, data, timeout, stream, allow_redirects, verify, cert, proxies):
            captured['proxies'] = proxies
            return FakeResp(url)

    class FakeRequests:
        def Session(self):
            return FakeSession()

    monkeypatch.setattr(np, 'requests', FakeRequests())

    res = call_tool('web_fetch', url='https://example.com')
    assert res.get('ok') is True
    # NO_PROXY should force bypass even when proxies are allowed
    assert captured.get('proxies') == {}
