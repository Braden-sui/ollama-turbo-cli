import os
from src.reliability.guards.validator import Validator


def _make_map(q1: str, q2: str):
    return {
        '1': {'title': 'A', 'url': 'U1', 'highlights': [{'quote': q1}]},
        '2': {'title': 'B', 'url': 'U2', 'highlights': [{'quote': q2}]},
    }


def test_multi_cite_any_pass(monkeypatch):
    monkeypatch.setenv('OVERLAP_MULTI_CITE_POLICY', 'any')
    v = Validator(mode='enforce')
    claim = 'Revenue grew 10% [1][2].'
    cmap = _make_map('grew 10%', 'grew 2%')
    rep = v.validate(claim, [], cmap)
    assert rep.get('ok') is True


def test_multi_cite_all_fail(monkeypatch):
    monkeypatch.setenv('OVERLAP_MULTI_CITE_POLICY', 'all')
    v = Validator(mode='enforce')
    claim = 'Revenue grew 10% [1][2].'
    cmap = _make_map('grew 10%', 'grew 2%')
    rep = v.validate(claim, [], cmap)
    assert rep.get('ok') is False
