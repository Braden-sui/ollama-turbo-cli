import sys
import types
from typing import Any, Dict, List

import pytest

from src.client import OllamaTurboClient


class DummyOllamaClient:
    def __init__(self):
        self.calls: List[Dict[str, Any]] = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        return {"message": {"content": "ok"}}


def _install_fake_local_mem0(monkeypatch):
    mod = types.ModuleType("mem0")

    class _FakeLocalMemClient:
        last_cfg: Dict[str, Any] = {}

        def search(self, query, version='v2', filters=None, limit=None):
            return []

        def add(self, *args, **kwargs):
            return {"id": "mem-local-1"}

        def update(self, *args, **kwargs):
            return True

        def get_all(self, *args, **kwargs):
            return []

        def delete_all(self, *args, **kwargs):
            return None

        def link(self, *args, **kwargs):
            return None

        def get(self, *args, **kwargs):
            return {}

    class Memory:  # type: ignore
        @classmethod
        def from_config(cls, cfg: Dict[str, Any]):
            _FakeLocalMemClient.last_cfg = cfg
            return _FakeLocalMemClient()

    mod.Memory = Memory
    monkeypatch.setitem(sys.modules, 'mem0', mod)
    return _FakeLocalMemClient


def test_mem0_local_init_and_injection(monkeypatch):
    # Install a fake local mem0 SDK exposing Memory.from_config
    FakeClient = _install_fake_local_mem0(monkeypatch)

    # Force local mode via constructor flags; use in-memory vector store
    client = OllamaTurboClient(
        api_key='test',
        enable_tools=False,
        quiet=True,
        mem0_local=True,
        mem0_vector_provider='chroma',
        mem0_vector_host=':memory:',
        mem0_vector_port=0,
        mem0_ollama_url='http://localhost:11434',
        mem0_llm_model='llama3',
        mem0_embedder_model='nomic-embed-text',
        mem0_user_id='u-test',
    )

    # Verify local initialization
    assert getattr(client, 'mem0_enabled', False) is True
    assert getattr(client, 'mem0_mode', '') == 'local'
    assert isinstance(client.mem0_client, FakeClient)

    # Validate config passed to Memory.from_config
    cfg = FakeClient.last_cfg
    assert isinstance(cfg, dict)
    assert cfg.get('version') == 'v2'
    # Vector store: chroma, in-memory, no port key when port=0
    vs = cfg.get('vector_store') or {}
    assert vs.get('provider') == 'chroma'
    vcfg = vs.get('config') or {}
    assert vcfg.get('host') == ':memory:'
    assert vcfg.get('in_memory') is True
    assert 'port' not in vcfg
    # LLM and embedder settings
    assert (cfg.get('llm') or {}).get('config', {}).get('model') == 'llama3'
    assert (cfg.get('llm') or {}).get('config', {}).get('ollama_base_url') == 'http://localhost:11434'
    assert (cfg.get('embedder') or {}).get('config', {}).get('model') == 'nomic-embed-text'
    # User routing
    assert cfg.get('user_id') == 'u-test'

    # Now make sure Mem0 system context is injected before user in a request
    def fake_search(query, version='v2', filters=None, limit=None):
        return [{"memory": "Prefers local mode."}]
    monkeypatch.setattr(client.mem0_client, 'search', fake_search, raising=False)

    dummy = DummyOllamaClient()
    client.client = dummy

    out = client.chat('hi there', stream=False)
    assert isinstance(out, str) and out

    # Inspect first call messages for Mem0 system block before user
    assert len(dummy.calls) >= 1
    msgs = dummy.calls[0].get('messages') or []
    assert isinstance(msgs, list) and len(msgs) >= 2

    prefix = client.prompt.mem0_prefix()
    mem_idxs = [i for i, m in enumerate(msgs) if m.get('role') == 'system' and prefix in str(m.get('content',''))]
    user_idx = next(i for i, m in enumerate(msgs) if m.get('role') == 'user' and m.get('content') == 'hi there')
    assert mem_idxs and min(mem_idxs) < user_idx

    # Mem0 content contains our memory
    mem_block = msgs[mem_idxs[0]]
    assert 'Prefers local mode.' in str(mem_block.get('content',''))
