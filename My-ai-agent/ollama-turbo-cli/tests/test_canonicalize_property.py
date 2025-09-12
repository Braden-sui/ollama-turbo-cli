from __future__ import annotations

import random
import urllib.parse as up

from src.web.normalize import canonicalize


def test_canonicalize_idempotent_and_strips_tracking():
    random.seed(123)
    bases = [
        "http://Example.COM/path/",
        "https://example.com/path/sub/",
        "https://news.example.net/",
    ]
    tracking_keys = [
        "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
        "gclid", "fbclid", "ref", "mc_cid", "mc_eid", "igshid"
    ]
    queries = []
    for b in bases:
        params = {k: f"v{random.randint(1,9)}" for k in random.sample(tracking_keys, 5)}
        params.update({"q": "test", f"x{random.randint(1,9)}": "y"})
        q = up.urlencode(params)
        queries.append(f"{b}?{q}")
    for u in queries:
        c1 = canonicalize(u)
        c2 = canonicalize(c1)
        assert c1 == c2
        parsed = up.urlparse(c1)
        qs = dict(up.parse_qsl(parsed.query))
        # none of the tracking keys should remain
        for k in tracking_keys:
            assert k not in qs
