from __future__ import annotations

from typing import Any, Dict, List, Tuple
import os
import re
import json

from ..state.session import get_session


def _load_cfg() -> Dict[str, Any]:
    path = os.getenv("MODE_CFG") or os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "mode.yaml")
    try:
        import yaml  # type: ignore
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        # YAML not available or file missing: fallback to embedded defaults
        return {
            "thresholds": {"promote": 0.62, "demote": 0.42, "referee_band": 0.05},
            "hysteresis": {"lock_turns": 2, "decay": 0.75},
            "signals": {
                "risky_terms": [
                    "health", "medical", "finance", "trading", "policy", "regulation", "fda", "sec", "lawsuit", "safety", "recall", "compliance", "gdpr",
                    # geopolitics (bias to researcher + strict cited synthesis)
                    "israel", "gaza", "palestine", "west bank", "hamas", "idf", "hezbollah", "lebanon", "iran"
                ],
                "facty_tokens": ["according to", "study", "report", "filed", "launched", "revenue", "patients", "evidence", "cite", "source", "versus", "vs"],
                "recency_tokens": ["today", "this week", "latest", "breaking", "in 2024", "in 2025"],
                "style_tokens_standard": ["brainstorm", "imagine", "story", "poem", "outline", "riff", "ideas", "metaphor"],
                "ask_for_sources": ["cite", "sources", "references", "footnotes"],
            },
            "weights": {
                "numerals": 0.20, "risky": 0.25, "facty": 0.20, "recency": 0.15, "qtype": 0.10,
                "compare": 0.08, "ask_src": 0.15, "link": 0.08, "style_standard": -0.18, "prior": 0.25,
            },
            "tool_bias": {"web_research": 0.20, "retrieval": 0.15},
            "referee": {"bump_researcher": 0.08, "bump_standard": -0.08},
        }


_CFG = _load_cfg()


def _tokenize_sentences(text: str) -> List[str]:
    parts = re.split(r"[\.!?\n]+", text or "")
    return [p.strip() for p in parts if p.strip()]


def _score_sentence(s: str, planned_tools: List[str]) -> Tuple[float, Dict[str, Any]]:
    s_l = s.lower()
    cfg = _CFG
    sig = cfg.get("signals", {})
    w = cfg.get("weights", {})

    # Signals
    numerals = bool(re.search(r"(\d|%|\$|€|£|¥|Q[1-4]\s*20\d{2}|\b\d{4}\b)", s))
    risky = any(t in s_l for t in sig.get("risky_terms", []))
    facty = any(t in s_l for t in sig.get("facty_tokens", []))
    recency = any(t in s_l for t in sig.get("recency_tokens", [])) or bool(re.search(r"\b(20[2-3]\d)\b", s_l))
    qtype = bool(re.search(r"\b(who|what|when|where|why|how|how many|how much)\b", s_l))
    compare = (" vs " in s_l) or ("versus" in s_l) or ("compare" in s_l)
    ask_src = any(t in s_l for t in sig.get("ask_for_sources", []))
    link = ("http://" in s_l) or ("https://" in s_l)
    style_std = any(t in s_l for t in sig.get("style_tokens_standard", []))

    # Aggregate
    score = 0.0
    score += w.get("numerals", 0.0) if numerals else 0.0
    score += w.get("risky", 0.0) if risky else 0.0
    score += w.get("facty", 0.0) if facty else 0.0
    score += w.get("recency", 0.0) if recency else 0.0
    score += w.get("qtype", 0.0) if qtype else 0.0
    score += w.get("compare", 0.0) if compare else 0.0
    score += w.get("ask_src", 0.0) if ask_src else 0.0
    score += w.get("link", 0.0) if link else 0.0
    score += w.get("style_standard", 0.0) if style_std else 0.0
    # Synergy: facty + question (e.g., "how many patients") implies researcher intent
    if facty and qtype:
        try:
            bonus = float(w.get("facty_qtype_bonus", 0.35))
        except Exception:
            bonus = 0.35
        score += bonus

    # Tool bias
    tb = cfg.get("tool_bias", {})
    for t in (planned_tools or []):
        score += float(tb.get(str(t), 0.0) or 0.0)

    details = {
        "signals": {
            "numerals": numerals,
            "risky": risky,
            "facty": facty,
            "recency": recency,
            "qtype": qtype,
            "compare": compare,
            "ask_src": ask_src,
            "link": link,
            "style_standard": style_std,
        }
    }
    return score, details


def classify(message: str, *, session_id: str = "default", planned_tools: List[str] | None = None) -> Tuple[str, float, Dict[str, Any]]:
    cfg = _CFG
    thr = cfg.get("thresholds", {})
    promote = float(thr.get("promote", 0.62))
    demote = float(thr.get("demote", 0.42))
    band = float(thr.get("referee_band", 0.05))
    hyst = cfg.get("hysteresis", {})
    lock_turns = int(hyst.get("lock_turns", 2))
    decay = float(hyst.get("decay", 0.75))

    # Sentence-level scoring; pick max
    sentences = _tokenize_sentences(message or "") or [message or ""]
    best_score = -1.0
    best_sent = sentences[0]
    best_details: Dict[str, Any] = {}
    for s in sentences:
        sc, det = _score_sentence(s, planned_tools or [])
        if sc > best_score:
            best_score, best_sent, best_details = sc, s, det

    # Prior/hysteresis
    sess = get_session(session_id)
    prior_contrib = 0.0
    locked = False
    if sess.lock_remaining > 0:
        locked = True
    combined = best_score
    if sess.last_score:
        prior_contrib = float(cfg.get("weights", {}).get("prior", 0.25)) * (sess.last_score * decay)
        combined += prior_contrib

    # LLM referee (simulated) around midline
    referee_used = False
    referee_vote = None
    mid = (promote + demote) / 2.0
    if abs(combined - mid) <= band:
        referee_used = True
        # Simulated referee decision: if risky or ask_src signals were present, nudge toward researcher
        if best_details.get("signals", {}).get("risky") or best_details.get("signals", {}).get("ask_src"):
            bump = float(cfg.get("referee", {}).get("bump_researcher", 0.08))
            combined += bump
            referee_vote = "researcher"
        else:
            bump = float(cfg.get("referee", {}).get("bump_standard", -0.08))
            combined += bump
            referee_vote = "standard"

    # Decision
    if locked:
        mode = sess.last_mode
    else:
        mode = "researcher" if combined >= promote else ("standard" if combined <= demote else (sess.last_mode or "standard"))

    # Update session (lock on decisive decision)
    if mode != sess.last_mode:
        if mode == "researcher" and combined >= promote:
            sess.lock_remaining = lock_turns
        elif mode == "standard" and combined <= demote:
            sess.lock_remaining = lock_turns
        else:
            sess.lock_remaining = max(0, sess.lock_remaining - 1)
    else:
        sess.lock_remaining = max(0, sess.lock_remaining - 1)
    sess.last_mode = mode
    sess.last_score = combined

    reason = {
        "winning_sentence": best_sent,
        "signals": best_details.get("signals", {}),
        "prior": prior_contrib,
        "locked": locked,
        "referee_used": referee_used,
        "referee_vote": referee_vote,
    }
    return mode, combined, reason
