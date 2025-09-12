from __future__ import annotations

from src.web.wire_dedup import collapse_citations


def _cit(url: str, title: str, text: str):
    return {
        "canonical_url": url,
        "title": title,
        "lines": [{"text": text}],
    }


def test_collapse_groups_by_text_and_apex():
    cits = [
        _cit("https://www.reuters.com/article/abc", "A", "X"),
        _cit("https://reuters.com/world/abc", "A", "X"),  # same apex + text
        _cit("https://apnews.com/article/abc", "A", "X"),  # different apex
        _cit("https://example.com/news/1", "A", "X"),      # non-wire same text
        _cit("https://example.com/news/2", "B", "Y"),
    ]

    collapsed, meta = collapse_citations(cits)
    assert isinstance(collapsed, list) and isinstance(meta, dict)
    # Kept should be <= total
    assert len(collapsed) <= len(cits)
    # Meta groups reflect grouping
    assert meta.get("total") == len(cits)
    assert meta.get("kept") == len(collapsed)
    assert isinstance(meta.get("groups"), list)
    assert meta.get("collapsed_count") >= 1
