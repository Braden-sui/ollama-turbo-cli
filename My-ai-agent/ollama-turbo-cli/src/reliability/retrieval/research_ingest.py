from __future__ import annotations

from typing import List, Dict, Any
import time

# Normalize citations (from web_research) into ephemeral doc objects compatible with RetrievalPipeline
# Doc schema:
# {"id":"url#chunk1","title":"Page Title","url":"https://example.com","timestamp":"2025-09-10","text":"...snippet..."}

def citations_to_docs(citations: List[Dict[str, Any]], *, max_docs: int = 100) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not isinstance(citations, list):
        return out
    now_iso = time.strftime('%Y-%m-%d')
    # Merge by URL to dedupe across queries/variants
    bucket: Dict[str, Dict[str, Any]] = {}
    for c in citations:
        try:
            url = str(c.get('canonical_url') or c.get('url') or '')
            if not url:
                continue
            title = str(c.get('title') or url)
            ts = str(c.get('date') or now_iso)
            lines = c.get('lines') or []
            quotes: List[str] = []
            if isinstance(lines, list):
                for hl in lines:
                    q = ''
                    try:
                        q = (hl or {}).get('quote') or ''
                    except Exception:
                        q = ''
                    q = str(q).strip()
                    if q:
                        quotes.append(q)
            ent = bucket.get(url)
            if not ent:
                ent = {'id': url, 'title': title, 'url': url, 'timestamp': ts, 'text': '', 'source': 'web'}
                bucket[url] = ent
            # Merge unique quotes
            existing = set((ent.get('text') or '').split('\n')) if ent.get('text') else set()
            for q in quotes:
                if q not in existing:
                    existing.add(q)
            # Re-serialize merged quotes (keep title if still empty)
            merged = [q for q in existing if q]
            ent['text'] = "\n".join(merged) if merged else (title or url)
        except Exception:
            continue
    # Emit up to max_docs entries in insertion order
    for i, (u, ent) in enumerate(bucket.items()):
        out.append(ent)
        if len(out) >= max_docs:
            break
    return out
