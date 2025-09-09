import os
import json

from src.logging.audit import write_audit_line


def test_audit_log_writes_jsonl(tmp_path, monkeypatch):
    # Redirect runs dir to temp by chdir
    monkeypatch.chdir(tmp_path)
    write_audit_line(
        mode='researcher',
        query='who is head of the FDA in 2025?',
        answer='The head is X [1].',
        citations=[{'n': '1', 'title': 'T', 'url': 'U', 'highlights': [{'quote': 'q', 'loc': 'p.1'}]}],
        metrics={'sources': 1, 'forced_refresh_used': False, 'freshness_days': 45},
        router={'score': 0.8, 'details': {'signals': {'risky': True}, 'locked': False}},
    )
    path = tmp_path / 'runs' / 'audit.jsonl'
    assert path.is_file()
    lines = path.read_text(encoding='utf-8').strip().splitlines()
    assert len(lines) >= 1
    obj = json.loads(lines[-1])
    assert obj.get('mode') == 'researcher'
    assert 'router' in obj and 'details' in obj['router']

