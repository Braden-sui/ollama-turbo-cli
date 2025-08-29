import os
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
def _env_defaults(tmp_path, monkeypatch):
    # Sandbox web defaults
    monkeypatch.setenv('SANDBOX_NET', 'allowlist')
    monkeypatch.setenv('SANDBOX_NET_ALLOW', 'example.com')
    monkeypatch.setenv('SANDBOX_ALLOW_HTTP', '0')
    monkeypatch.setenv('SANDBOX_BLOCK_PRIVATE_IPS', '1')
    monkeypatch.setenv('SANDBOX_HTTP_CACHE_TTL_S', '600')
    monkeypatch.setenv('SANDBOX_HTTP_CACHE_MB', '50')
    # Proxies off by default
    monkeypatch.setenv('SANDBOX_ALLOW_PROXIES', '0')

    reload_plugins()
    yield


def test_method_whitelist_disallows_put(monkeypatch):
    # No network call should be attempted; method check happens early
    res = call_tool('web_fetch', url='https://example.com', method='PUT')
    assert res.get('ok') is False
    assert 'Method not allowed' in (res.get('error') or '')


def test_proxies_disabled_by_default(monkeypatch):
    from src.sandbox import net_proxy as np

    # Avoid real DNS resolution
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
            # Simulate environment proxies that should be ignored by policy
            return {'verify': True, 'cert': None, 'proxies': {'http': 'http://proxy:8080', 'https': 'http://proxy:8080'}}

        def request(self, method, url, headers, data, timeout, stream, allow_redirects, verify, cert, proxies):
            captured['proxies'] = proxies
            return FakeResp(url)

    class FakeRequests:
        def Session(self):
            return FakeSession()

    monkeypatch.setattr(np, 'requests', FakeRequests())

    res = call_tool('web_fetch', url='https://example.com', cache_bypass=True)
    assert res.get('ok') is True
    # Proxies should be disabled by default -> proxies param is an empty dict
    assert captured.get('proxies') == {}


def test_proxies_opt_in_enabled(monkeypatch):
    from src.sandbox import net_proxy as np

    # Allow proxies via env
    monkeypatch.setenv('SANDBOX_ALLOW_PROXIES', '1')

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
            return {'verify': True, 'cert': None, 'proxies': {'http': 'http://proxy:8080', 'https': 'http://proxy:8080'}}

        def request(self, method, url, headers, data, timeout, stream, allow_redirects, verify, cert, proxies):
            captured['proxies'] = proxies
            return FakeResp(url)

    class FakeRequests:
        def Session(self):
            return FakeSession()

    monkeypatch.setattr(np, 'requests', FakeRequests())

    # Explicitly bypass cache to exercise proxy path
    res = call_tool('web_fetch', url='https://example.com', cache_bypass=True)
    assert res.get('ok') is True
    # When opt-in is enabled, proxies from environment should flow through
    assert captured.get('proxies') == {'http': 'http://proxy:8080', 'https': 'http://proxy:8080'}


def test_cache_bypass_skips_cache_read(tmp_path, monkeypatch):
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
    # Second call with explicit cache_bypass should skip cache read and perform a network request
    r2 = call_tool('web_fetch', url='https://example.com', cache_bypass=True)
    assert r2.get('ok') is True
    assert r2.get('cached') is False
    assert call_count['n'] == 2


def test_peer_ip_post_connect_blocking(monkeypatch):
    from src.sandbox import net_proxy as np

    # Avoid real DNS
    monkeypatch.setattr(np, '_resolve_and_check', lambda host: (host, ['93.184.216.34']))

    class FakeSock:
        def getpeername(self):
            return ('127.0.0.1', 443)

    class FakeConn:
        sock = FakeSock()

    class FakeRaw:
        _connection = FakeConn()

    class FakeResp:
        def __init__(self, url: str):
            self.url = url
            self.status_code = 200
            self.headers = {'Content-Type': 'text/plain'}
            self.history = []
            self.raw = FakeRaw()

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
            return FakeResp(url)

    class FakeRequests:
        def Session(self):
            return FakeSession()

    monkeypatch.setattr(np, 'requests', FakeRequests())

    # Explicitly bypass cache to force post-connect peer IP check
    res = call_tool('web_fetch', url='https://example.com', cache_bypass=True)
    assert res.get('ok') is False
    assert 'peer IP' in (res.get('error') or '')
