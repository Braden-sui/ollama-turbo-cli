from __future__ import annotations

from typing import List, Dict, Any, Tuple
import re
import os
from ..validation.overlap import _tokens as _ovl_tokens


class Validator:
    """CiteFence-like validator: checks that cited sentences are supported by quotes.

    Modes: off | warn | enforce (behavior left to caller; this class only reports).
    """

    def __init__(self, mode: str = "off") -> None:
        self.mode = mode if mode in {"off", "warn", "enforce"} else "off"

    def _cfg(self) -> Dict[str, Any]:
        def _b(name: str, d: bool) -> bool:
            v = os.getenv(name)
            if v is None:
                return d
            return str(v).strip().lower() in {"1","true","yes","on"}
        def _f(name: str, d: float) -> float:
            try:
                v = os.getenv(name)
                return float(v) if v not in (None,"") else d
            except Exception:
                return d
        def _i(name: str, d: int) -> int:
            try:
                v = os.getenv(name)
                return int(v) if v not in (None,"") else d
            except Exception:
                return d
        tw_raw = os.getenv("OVERLAP_TOKEN_WEIGHTS","announced:0.3,launched:0.3,reported:0.3,shipped:0.3")
        weights: Dict[str,float] = {}
        try:
            for part in (tw_raw or '').split(','):
                if ':' in part:
                    k,v = part.split(':',1)
                    k = k.strip().lower(); v=v.strip()
                    if k:
                        weights[k] = float(v)
        except Exception:
            weights = {}
        return {
            'threshold': _f('OVERLAP_THRESHOLD', 0.18),
            'discount_word_only_numeric': _f('OVERLAP_DISCOUNT_WORD_ONLY_NUMERIC', 0.25),
            'require_numeric_match': _b('OVERLAP_REQUIRE_NUMERIC_MATCH', False),
            'min_claim_tokens': _i('OVERLAP_MIN_CLAIM_TOKENS', 3),
            'allow_multi_cite': _b('OVERLAP_ALLOW_MULTI_CITE', True),
            'token_weights': weights,
        }

    def _sentences_with_cites(self, text: str) -> List[Tuple[str, List[str]]]:
        out: List[Tuple[str,List[str]]] = []
        for m in re.finditer(r"([^\.!?\n]+?)\s*((?:\[\d+\])+)(?:[\.!?]|$)", text or "", re.M):
            sent = m.group(1).strip()
            cites = re.findall(r"\[(\d+)\]", m.group(2) or "")
            if sent and cites:
                out.append((sent, cites))
        return out

    def _weighted_recall(self, claim_tokens: List[str], quote_tokens: List[str], *, weights: Dict[str, float]) -> Tuple[float, bool, bool]:
        cset = set(claim_tokens)
        qset = set(quote_tokens)
        cnum = any(any(ch.isdigit() for ch in t) for t in cset)
        qnum_match = any((t in qset) and any(ch.isdigit() for ch in t) for t in cset)
        def w(tok: str) -> float:
            return float(weights.get(tok, 1.0))
        denom = sum(w(t) for t in cset)
        inter = sum(w(t) for t in (cset & qset))
        score = inter / max(1.0, denom)
        return score, cnum, qnum_match

    def validate(self, answer: str, context_blocks: List[Dict[str, Any]], citations_map: Dict[str, Any] | None = None) -> Dict[str, Any]:
        cfg = self._cfg()
        citations_present = "[" in (answer or "") and "]" in (answer or "")
        details: List[Dict[str, Any]] = []
        ok = True
        token_weights: Dict[str, float] = cfg.get('token_weights') or {}

        qtok_cache: Dict[str, List[str]] = {}
        def _qtoks(q: str) -> List[str]:
            if q in qtok_cache:
                return qtok_cache[q]
            ts = _ovl_tokens(q)
            qtok_cache[q] = ts
            return ts

        for sent, cite_ids in self._sentences_with_cites(answer or ""):
            highlights: List[Dict[str, Any]] = []
            missing = False
            for cid in cite_ids:
                meta = (citations_map or {}).get(str(cid)) if isinstance(citations_map, dict) else None
                hls = (meta or {}).get('highlights') if isinstance(meta, dict) else None
                if hls and isinstance(hls, list):
                    highlights.extend(hls)
                else:
                    missing = True
            ctoks = _ovl_tokens(sent)
            if len(ctoks) < int(cfg.get('min_claim_tokens') or 3):
                continue
            qtoks_union: List[str] = []
            for h in highlights:
                q = str((h or {}).get('quote') or '')
                if q:
                    qtoks_union.extend(_qtoks(q))
            score, has_num, num_match = self._weighted_recall(ctoks, qtoks_union, weights=token_weights)
            discount_applied = False
            if has_num and (not num_match):
                if bool(cfg.get('require_numeric_match', False)):
                    score = 0.0
                else:
                    score *= float(cfg.get('discount_word_only_numeric', 0.25) or 0.25)
                    discount_applied = True
            passed = (score >= float(cfg.get('threshold', 0.18))) and (not missing)
            if not passed:
                ok = False
            details.append({
                'sentence': sent,
                'cites': cite_ids,
                'overlap_score': round(score, 4),
                'numeric_matched': bool(num_match),
                'threshold': float(cfg.get('threshold', 0.18)),
                'discount_applied': discount_applied,
                'missing_highlights': bool(missing),
                'passed': passed,
            })

        return {
            'mode': self.mode,
            'status': 'ok' if ok else 'fail',
            'ok': ok,
            'citations_present': citations_present,
            'details': details,
        }
