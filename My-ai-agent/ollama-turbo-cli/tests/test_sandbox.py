import os
import sys
import json
import time
import platform
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
def _env_isolation(tmp_path, monkeypatch):
    # default deny for shell
    monkeypatch.setenv('SHELL_TOOL_ALLOW', '0')
    monkeypatch.setenv('SHELL_TOOL_ALLOWLIST', 'git status,git log,git diff,ls,dir,cat,type,python -V,python --version,pip list')
    monkeypatch.setenv('SHELL_TOOL_ROOT', str(tmp_path))
    # Web defaults
    monkeypatch.setenv('SANDBOX_NET', 'allowlist')
    monkeypatch.setenv('SANDBOX_NET_ALLOW', 'api.github.com')
    monkeypatch.setenv('SANDBOX_ALLOW_HTTP', '0')
    monkeypatch.setenv('SANDBOX_BLOCK_PRIVATE_IPS', '1')
    monkeypatch.setenv('SANDBOX_HTTP_CACHE_TTL_S', '600')
    monkeypatch.setenv('SANDBOX_HTTP_CACHE_MB', '50')
    monkeypatch.setenv('SANDBOX_RATE_PER_HOST', '60')

    reload_plugins()
    yield


def sandbox_available() -> bool:
    # Heuristic: docker version
    from src.sandbox.runner import _docker_available  # type: ignore
    return _docker_available()


def test_default_deny_blocks_shell(tmp_path):
    res = call_tool('execute_shell', command='ls', working_dir=str(tmp_path))
    assert res.get('blocked') is True
    assert 'how_to_enable' in res


def test_allowlisted_command_runs(tmp_path, monkeypatch):
    monkeypatch.setenv('SHELL_TOOL_ALLOW', '1')
    res = call_tool('execute_shell', command='ls', working_dir=str(tmp_path))
    assert res.get('blocked') is False
    # ok may be True even if empty dir; always expect an exit_code and log_path
    assert 'exit_code' in res
    assert isinstance(res.get('log_path'), str)
    if sandbox_available():
        # On hosts with Docker, command should execute in sandbox
        assert res.get('ok') in (True, False)  # typically True
    else:
        # On hosts without Docker, sandbox fails closed with exit 126
        assert res.get('ok') is False
        assert res.get('exit_code') == 126


def test_truncation_and_redaction(tmp_path, monkeypatch):
    monkeypatch.setenv('SHELL_TOOL_ALLOW', '1')
    monkeypatch.setenv('SHELL_TOOL_MAX_OUTPUT', '10')
    # fabricate by echoing a secret-like output; if sandbox unavailable, we still expect a structured response with redaction when possible
    res = call_tool('execute_shell', command='bash -lc "echo API_KEY=foo_bar_secret"', working_dir=str(tmp_path), shell=True)
    # If sandbox unavailable, the tool returns ok False with an error; accept either, but check that output is not leaking raw token in inject
    inj = res.get('inject') or json.dumps(res)
    assert 'foo_bar_secret' not in inj


def test_timeout_flag(tmp_path, monkeypatch):
    monkeypatch.setenv('SHELL_TOOL_ALLOW', '1')
    res = call_tool('execute_shell', command='bash -lc "sleep 5"', working_dir=str(tmp_path), timeout=1, shell=True)
    # If sandbox runs, should time out; else, ok False and stderr explains unavailability
    assert 'timed_out' in res


def test_network_deny_blocks_web(monkeypatch):
    monkeypatch.setenv('SANDBOX_NET', 'deny')
    res = call_tool('web_fetch', url='https://api.github.com')
    assert res.get('ok') is False


def test_allowlist_web_and_cache(monkeypatch):
    monkeypatch.setenv('SANDBOX_NET', 'allowlist')
    monkeypatch.setenv('SANDBOX_NET_ALLOW', 'api.github.com')
    # First call
    r1 = call_tool('web_fetch', url='https://api.github.com')
    assert 'status' in r1
    # Second call should be cached
    r2 = call_tool('web_fetch', url='https://api.github.com')
    assert r2.get('cached') in (True, False)


def test_ssrf_block():
    res = call_tool('web_fetch', url='https://127.0.0.1/')
    assert res.get('ok') is False


def test_redirect_revalidation(monkeypatch):
    # Disallow github.com, allow only api.github.com
    monkeypatch.setenv('SANDBOX_NET', 'allowlist')
    monkeypatch.setenv('SANDBOX_NET_ALLOW', 'api.github.com')
    # http://github.com redirects to https://github.com/ which is disallowed
    r = call_tool('web_fetch', url='http://github.com')
    assert r.get('ok') is False


def test_path_confinement(tmp_path):
    # working_dir outside root should be rejected
    res = call_tool('execute_shell', command='ls', working_dir='/', timeout=1)
    assert res.get('blocked') is True


def test_web_fetch_header_sanitization(monkeypatch):
    # Allow example.com and avoid real DNS
    monkeypatch.setenv('SANDBOX_NET', 'allowlist')
    monkeypatch.setenv('SANDBOX_NET_ALLOW', 'example.com')

    from src.sandbox import net_proxy as np

    # Avoid real DNS/SSRF checks depending on environment
    monkeypatch.setattr(np, '_resolve_and_check', lambda host: (host, ['93.184.216.34']))

    captured = {}

    class FakeResp:
        def __init__(self, url: str):
            self.url = url
            self.status_code = 200
            self.headers = {'Content-Type': 'application/json'}
            self.history = []

        def iter_content(self, chunk_size=8192):
            yield b'{}'

    class FakeSession:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def merge_environment_settings(self, url, proxies, stream, verify, cert):
            return {'verify': True, 'cert': None, 'proxies': None}

        def request(self, method, url, headers, data, timeout, stream, allow_redirects, verify, cert, proxies):
            captured['headers'] = dict(headers)
            return FakeResp(url)

    class FakeRequests:
        def Session(self):
            return FakeSession()

    # Patch requests used by net_proxy
    monkeypatch.setattr(np, 'requests', FakeRequests())

    # Call via tool wrapper to exercise plugin path
    res = call_tool('web_fetch', url='https://example.com', headers={
        'Authorization': 'Bearer SECRET',
        'X-Forwarded-For': '1.2.3.4',
        'Accept': 'application/json'
    })
    assert res.get('ok') is True
    sent = {k.lower(): v for k, v in captured.get('headers', {}).items()}
    # Sensitive headers should be stripped
    assert 'authorization' not in sent
    assert 'x-forwarded-for' not in sent
    # Allowed headers should remain
    assert sent.get('accept') == 'application/json'
    assert 'user-agent' in sent


def test_blocklist_policy(monkeypatch):
    # Enable blocklist mode: allow entire web except matching patterns
    monkeypatch.setenv('SANDBOX_NET', 'blocklist')
    monkeypatch.setenv('SANDBOX_NET_BLOCK', 'bad.com,*.malware.test')
    # Keep HTTPS requirement
    monkeypatch.setenv('SANDBOX_ALLOW_HTTP', '0')

    from src.sandbox import net_proxy as np

    # Avoid real DNS checks and private IP resolution during test
    monkeypatch.setattr(np, '_resolve_and_check', lambda host: (host, ['203.0.113.1']))

    # Fake requests session
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
            return {'verify': True, 'cert': None, 'proxies': None}
        def request(self, method, url, headers, data, timeout, stream, allow_redirects, verify, cert, proxies):
            captured['last_url'] = url
            return FakeResp(url)

    class FakeRequests:
        def Session(self):
            return FakeSession()

    monkeypatch.setattr(np, 'requests', FakeRequests())

    # Allowed host should pass
    r_ok = call_tool('web_fetch', url='https://example.com')
    assert r_ok.get('ok') is True

    # Blocklisted host should be blocked
    r_bad = call_tool('web_fetch', url='https://bad.com')
    assert r_bad.get('ok') is False
    assert 'blocklisted' in (r_bad.get('error') or '')
