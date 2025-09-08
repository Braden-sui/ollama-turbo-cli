import types
import json
import sys
import os
from pathlib import Path

# Ensure project root (parent of scripts/) is on sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Import the plugin module
import src.plugins.duckduckgo as d

# Fake response object
class FakeResp:
    def __init__(self, status_code=200, text='', json_data=None):
        self.status_code = status_code
        self.text = text
        self._json = json_data
    def json(self):
        if self._json is None:
            raise ValueError("no json")
        return self._json

calls = []

def fake_get(url, params=None, headers=None, timeout=None):
    calls.append((url, params))
    # Simulate API returning 202 (common for gating)
    if url.startswith("https://api.duckduckgo.com/"):
        return FakeResp(status_code=202, json_data=None)
    # First HTML fallback URL returns a DDG redirect link pattern
    if url.startswith("https://duckduckgo.com/lite/"):
        html = '<a href="/l/?uddg=https%3A%2F%2Fexample.com%2Ffoo">Example result</a>'
        return FakeResp(status_code=200, text=html)
    # Second HTML endpoint (not needed if first succeeded)
    if url.startswith("https://html.duckduckgo.com/html/"):
        html = '<a href="https://example.com/bar">Another result</a>'
        return FakeResp(status_code=200, text=html)
    return FakeResp(status_code=404, text="")

# Monkeypatch requests used by the plugin to avoid any real network calls
orig_requests = d.requests
try:
    d.requests = types.SimpleNamespace(get=fake_get)
    out = d.duckduckgo_search("test query", max_results=2)
    print(json.dumps(out, ensure_ascii=False))
finally:
    d.requests = orig_requests
