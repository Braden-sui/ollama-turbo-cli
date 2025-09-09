from src.reliability.validation.citefence import validate


def test_citations_required_for_facty_paragraph():
    ans = "Revenue grew 25% year over year in 2025.\n\nThis is a summary."
    report = validate(ans, {"1": {"url": "u", "title": "t"}})
    assert report["ok"] is False
    assert any("facty-without-citation" in it for it in report["issues"])

