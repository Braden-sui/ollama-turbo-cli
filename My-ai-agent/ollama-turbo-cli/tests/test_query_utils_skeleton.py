def test_preserve_quoted_and_years_skeleton():
    from src.web.query_utils import preserve_quoted_and_numerals
    q = 'compare "2020" vs 2018'
    preserved = preserve_quoted_and_numerals(q)
    assert "2020" in preserved
    assert "2018" in preserved


def test_generate_variants_soft_preserves_skeleton():
    from src.web.query_utils import generate_variants
    q = 'What changed between 2018 and 2020 for "parliamentary elections" and turnout?'
    variants = generate_variants(q, mode='soft', max_tokens=12, stopword_profile='minimal')
    assert variants and variants[0].startswith('What')
    assert any(('parliamentary' in v) or ('2020' in v) for v in variants)

