from src.routing.query_profile import defaults_for_query


def test_newsy_defaults():
    k, d = defaults_for_query("latest CPI this week")
    assert k == 5 and 30 <= d <= 60


def test_older_defaults():
    k, d = defaults_for_query("OpenGL changes since 2018")
    assert k == 7 and d >= 1000

