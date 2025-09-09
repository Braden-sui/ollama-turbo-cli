from src.reliability.validation.overlap import _tokens as tok


def test_suffix_multiplier_enabled(monkeypatch):
    monkeypatch.setenv('OVERLAP_ENABLE_SUFFIX_NORMALIZATION', '1')
    t = tok('raised $3.2M')
    assert any(x == '3200000' for x in t)


def test_thinspace_thousands(monkeypatch):
    # 300 -> 3600
    s = '3\u00A0600'
    t = tok(s)
    assert any(x == '3600' for x in t)

