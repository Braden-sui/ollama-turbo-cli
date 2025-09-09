from src.routing.mode_classifier import classify


def test_bedtime_story_standard():
    mode, score, reason = classify("tell me a bedtime story about the moon")
    assert mode == 'standard'


def test_fda_head_researcher():
    mode, score, reason = classify("who is head of the FDA in 2025?")
    assert mode == 'researcher'

