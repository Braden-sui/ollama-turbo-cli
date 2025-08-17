import json
from types import SimpleNamespace

from src.web.pipeline import run_research


def test_pipeline_pdf_page_mapping(monkeypatch, tmp_path):
    # Fake search returns a single PDF result
    fake_sr = SimpleNamespace(title="PDF Doc", url="https://example.com/file.pdf", snippet="", source="fake", published=None)
    monkeypatch.setattr('src.web.pipeline.search', lambda *args, **kwargs: [fake_sr])

    # Fake fetch returns minimal fields (we don't read body_path since we monkeypatch extract)
    def fake_fetch(url, **kwargs):
        return SimpleNamespace(ok=True, url=url, final_url=url, status=200, content_type='application/pdf', body_path=None, headers={}, browser_used=False, reason=None)
    monkeypatch.setattr('src.web.pipeline.fetch_url', fake_fetch)

    # Fake extract returns PDF with page line mapping
    ex = SimpleNamespace(
        ok=True,
        kind='pdf',
        markdown='Page1\nLine2\nLine3\n' * 10,
        title='PDF Doc',
        date=None,
        meta={'page_start_lines': [1, 11, 21]},
        used={},
        risk='LOW',
        risk_reasons=[],
    )
    monkeypatch.setattr('src.web.pipeline.extract_content', lambda *args, **kwargs: ex)

    # Fake rerank points to lines across pages 2 and 3
    ranked = [
        {
            'id': '1-30',
            'score': 0.9,
            'start_line': 1,
            'end_line': 30,
            'highlights': [
                {'line': 12, 'text': 'match on page 2'},
                {'line': 25, 'text': 'match on page 3'},
            ],
            'preview': []
        }
    ]
    monkeypatch.setattr('src.web.pipeline.rerank_chunks', lambda *args, **kwargs: ranked)
    # Avoid external archive call
    monkeypatch.setattr('src.web.pipeline.save_page_now', lambda *args, **kwargs: {'archive_url': '', 'timestamp': ''})

    out = run_research('query', top_k=1, force_refresh=True)
    assert 'citations' in out and len(out['citations']) == 1
    cit = out['citations'][0]
    assert cit['kind'] == 'pdf'
    lines = cit['lines']
    # Two lines mapped
    assert len(lines) == 2
    # Check pages computed from page_start_lines
    assert lines[0]['line'] == 12 and lines[0]['page'] == 2 and lines[0]['quote'] == 'match on page 2'
    assert lines[1]['line'] == 25 and lines[1]['page'] == 3 and lines[1]['quote'] == 'match on page 3'
