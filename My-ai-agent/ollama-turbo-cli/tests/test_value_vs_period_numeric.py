import os
from src.reliability.guards.validator import Validator


def test_value_vs_period_numeric_requires_value_match(monkeypatch):
    # Require value match: claim has 3.1% but quote has 2.9% only; period Q1-2025 matches but should not count
    monkeypatch.setenv('OVERLAP_REQUIRE_VALUE_MATCH', '1')
    claim = "Revenue grew 3.1% in Q1 2025 [1]."
    quotes = [{'quote': 'In Q1 2025, revenue increased by 2.9% year over year.'}]
    citations_map = {'1': {'title': 'T', 'url': 'U', 'highlights': quotes}}
    v = Validator(mode='enforce')
    rep = v.validate(claim, [], citations_map)
    # Expect failure
    assert rep.get('ok') is False
    det = rep.get('details') or []
    assert det and det[0]['passed'] is False

