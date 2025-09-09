from src.web.normalize import canonicalize, dedupe_citations


def test_canonicalization_drops_tracking_and_trailing():
    u1 = "http://Example.com/path/?utm_source=foo&ref=bar&id=1"
    u2 = "https://example.com/path?id=1"
    assert canonicalize(u1) == u2


def test_dedupe_collapses_similar_urls_titles():
    cits = [
        {"canonical_url": "http://example.com/x?utm_medium=a", "title": "Title"},
        {"canonical_url": "https://example.com/x", "title": " title  "},
    ]
    out = dedupe_citations(cits)
    assert len(out) == 1

