import os
from types import SimpleNamespace

import pytest

import src.web.pipeline as pipeline_mod
from src.web.pipeline import run_research
from src.web.config import WebConfig


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch, tmp_path):
    # Avoid any surprise network or cache reuse in this focused test
    monkeypatch.setenv("WEB_RESPECT_ROBOTS", "0")
    monkeypatch.setenv("WEB_ALLOW_BROWSER", "0")
    monkeypatch.setenv("WEB_EMERGENCY_BOOTSTRAP", "0")
    monkeypatch.setenv("WEB_CACHE_ROOT", str(tmp_path / ".webcache"))
    yield


def _stub_search(*a, **k):
    return []


def test_flags_reflect_env_overrides(monkeypatch):
    # Ensure env controls override cfg defaults
    monkeypatch.setenv("EVIDENCE_FIRST", "1")
    monkeypatch.setenv("EVIDENCE_FIRST_KILL_SWITCH", "0")

    # Prevent any fetch/extract path
    monkeypatch.setattr(pipeline_mod, "search", _stub_search, raising=True)

    out = run_research("quick check", top_k=1, force_refresh=True)
    pol = out.get("policy", {})
    assert isinstance(pol, dict)
    assert pol.get("evidence_first") is True
    assert pol.get("evidence_first_kill_switch") is False


def test_flags_reflect_cfg_when_env_absent(monkeypatch):
    # Remove env so cfg values are used
    monkeypatch.delenv("EVIDENCE_FIRST", raising=False)
    monkeypatch.delenv("EVIDENCE_FIRST_KILL_SWITCH", raising=False)

    cfg = WebConfig(evidence_first=True, evidence_first_kill_switch=False)

    # Prevent any fetch/extract path
    monkeypatch.setattr(pipeline_mod, "search", _stub_search, raising=True)

    out = run_research("another check", top_k=1, force_refresh=True, cfg=cfg)
    pol = out.get("policy", {})
    assert isinstance(pol, dict)
    assert pol.get("evidence_first") is True
    assert pol.get("evidence_first_kill_switch") is False
