from src.routing.mode_classifier import classify
from src.state.session import get_session


def test_recency_digits_ask_sources_researcher():
    s = get_session('t1'); s.last_mode='standard'; s.last_score=0; s.lock_remaining=0
    mode, score, reason = classify("Please cite sources: revenue in 2025 latest", session_id='t1')
    assert mode == 'researcher'


def test_creative_phrasing_standard():
    s = get_session('t2'); s.last_mode='researcher'; s.last_score=0; s.lock_remaining=0
    mode, score, reason = classify("brainstorm a story idea about dragons", session_id='t2')
    assert mode == 'standard'


def test_hysteresis_lock_sticky():
    s = get_session('t3'); s.last_mode='standard'; s.last_score=0; s.lock_remaining=0
    # Decisive researcher turn
    mode, score, reason = classify("Who is head of FDA in 2025?", session_id='t3')
    assert mode == 'researcher'
    # Weakly creative should remain researcher while lock active
    mode2, score2, reason2 = classify("write a poem about ducks", session_id='t3')
    assert mode2 == 'researcher'


def test_subtask_max_pick_researcher():
    mode, score, reason = classify("Write a friendly intro. Also, how many patients were in the trial?", session_id='t4')
    assert mode == 'researcher'


def test_referee_band_records_vote():
    # Ambiguous prompt; ensure referee triggers
    s = get_session('t5'); s.last_mode='standard'; s.last_score=0.5; s.lock_remaining=0
    mode, score, reason = classify("Summarize findings", session_id='t5')
    assert 'referee_used' in reason and isinstance(reason.get('referee_vote'), (str, type(None)))

