import os

from src.reliability.guards.validator import Validator
from src.reliability.validation.overlap import _tokens as tok


def test_cite_parsing_variants():
    v = Validator()
    s1 = "Claim A [1] [2]."
    s2 = "Claim B [1, 2]."
    s3 = "Claim C [1-3]."
    out1 = v._sentences_with_cites(s1)
    out2 = v._sentences_with_cites(s2)
    out3 = v._sentences_with_cites(s3)
    assert out1[0][1] == ['1','2']
    assert out2[0][1] == ['1','2']
    assert out3[0][1] == ['1','2','3']


def test_comma_decimal_normalization(monkeypatch):
    monkeypatch.setenv('OVERLAP_ALLOW_COMMA_DECIMALS', '1')
    t = tok("rose 3,1%")
    assert '3.1%' in t

