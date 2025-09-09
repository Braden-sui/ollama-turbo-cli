import re
from src.web.search import _year_guard
from src.web.config import WebConfig

def test_year_guard_strips_trailing_year():
    cfg = WebConfig()
    q = "AI research 2024"
    sanitized, counters = _year_guard(q, cfg)
    assert sanitized == "AI research"
    assert counters["stripped_year_tokens"] >= 1


def test_year_guard_strips_trailing_month_year():
    cfg = WebConfig()
    q = "Elon Musk current activities Sep 2025"
    sanitized, counters = _year_guard(q, cfg)
    assert sanitized == "Elon Musk current activities"
    assert counters["stripped_year_tokens"] >= 1


def test_year_guard_preserves_quoted_year():
    cfg = WebConfig()
    q = "Elon Musk '2024 roadmap' current activities"
    sanitized, counters = _year_guard(q, cfg)
    # Should preserve quoted year and not strip the internal '2024'
    assert "2024" in sanitized
    # Should not add a month-year either
    assert not re.search(r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\s+20\d{2}\b", sanitized, flags=re.I)
