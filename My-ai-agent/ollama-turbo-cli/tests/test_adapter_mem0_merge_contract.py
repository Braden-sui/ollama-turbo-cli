import pytest

from src.protocols.harmony import HarmonyAdapter
from src.prompt_manager import PromptManager


def test_harmony_adapter_no_merge_when_mem0_block_none():
    adapter = HarmonyAdapter(model="gpt-oss:120b")
    # Start with one system message (no Mem0 content)
    msgs = [{"role": "system", "content": "Base system."}, {"role": "user", "content": "Hi"}]

    out, overrides = adapter.format_initial_messages(messages=msgs, tools=None, options=None, mem0_block=None)

    assert isinstance(out, list) and len(out) >= 2
    first = out[0]
    assert first.get("role") == "system"
    # Ensure Mem0 prefix is not present in first system when mem0_block is None
    prefix = PromptManager.mem0_prefix()
    assert prefix not in str(first.get("content", ""))


def test_harmony_adapter_merges_mem0_block_into_first_system():
    adapter = HarmonyAdapter(model="gpt-oss:120b")
    # A simple first system and a user
    msgs = [{"role": "system", "content": "Base system."}, {"role": "user", "content": "Hi"}]

    # Provide a Mem0 block for merge (client removes the separate block and passes this)
    prefix = PromptManager.mem0_prefix()
    mem0_block = f"{prefix}\n- A fact\n"

    out, overrides = adapter.format_initial_messages(messages=msgs, tools=None, options=None, mem0_block=mem0_block)

    assert isinstance(out, list) and len(out) >= 2
    first = out[0]
    assert first.get("role") == "system"
    content = str(first.get("content", ""))
    # Assert mem0 block merged exactly once into first system
    assert prefix in content
    assert content.count(prefix) == 1
    # No separate Mem0 system message injected by adapter
    extra_mem_blocks = [m for m in out[1:] if m.get("role") == "system" and str(m.get("content", "")).startswith(prefix)]
    assert len(extra_mem_blocks) == 0
