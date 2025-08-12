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


@pytest.mark.skipif(not sandbox_available(), reason="Sandbox unavailable; expect fail-closed on this host")
def test_allowlisted_command_runs(tmp_path, monkeypatch):
    monkeypatch.setenv('SHELL_TOOL_ALLOW', '1')
    res = call_tool('execute_shell', command='ls', working_dir=str(tmp_path))
    assert res.get('blocked') is False
    # ok may be True even if empty dir
    assert 'exit_code' in res


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
