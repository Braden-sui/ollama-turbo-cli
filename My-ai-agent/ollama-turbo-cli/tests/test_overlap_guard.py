from src.reliability.validation.overlap import sentence_quote_overlap


def test_overlap_guard_flags_low_overlap():
    claim = "The company shipped millions [1]."
    highlights = [{"quote": "The company shipped 400,000 units globally"}]
    score = sentence_quote_overlap(claim, highlights)
    assert score < 0.18

