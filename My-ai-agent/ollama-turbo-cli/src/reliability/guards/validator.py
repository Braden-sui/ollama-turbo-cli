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
        """Extract sentences ending with citation clusters.

        Accept formats:
        - [1][2]
        - [1] [2]
        - [1, 2]
        - [1-3] / [1–3]
        Allow closing ) ] " ” ’ before final punctuation.
        """
        out: List[Tuple[str,List[str]]] = []
        pattern = re.compile(r"(.+?)\s*(\[(?:[0-9,\s\-–]+)\](?:\s*\[(?:[0-9,\s\-–]+)\])*)\s*[)\]""”’]*\s*(?:[\.!?]|$)", re.M)
        for m in pattern.finditer(text or ""):
            sent = (m.group(1) or '').strip()
            cluster = m.group(2) or ''
            # extract ids from each [...] group
            ids: List[int] = []
            for g in re.findall(r"\[([^\]]+)\]", cluster):
                parts = [p.strip() for p in re.split(r",", g) if p.strip()]
                for p in parts:
                    if re.match(r"^\d+$", p):
                        ids.append(int(p))
                    else:
                        mm = re.match(r"^(\d+)\s*[\-–]\s*(\d+)$", p)
                        if mm:
                            a, b = int(mm.group(1)), int(mm.group(2))
                            lo, hi = (a, b) if a <= b else (b, a)
                            ids.extend(list(range(lo, hi + 1)))
            cites = [str(i) for i in ids]
            if sent and cites:
                out.append((sent, cites))
        return out

    def _weighted_recall(self, claim_tokens: List[str], quote_tokens: List[str], *, weights: Dict[str, float]) -> Tuple[float, bool, bool, bool]:
        cset = set(claim_tokens)
        qset = set(quote_tokens)
        # Split numeric tokens into value vs period
        def _is_year(tok: str) -> bool:
            return len(tok) == 4 and tok.isdigit() and tok.startswith('20')
        def _is_quarter(tok: str) -> bool:
            return tok.startswith('q') and '-' in tok and tok[1].isdigit()
        claim_values = {t for t in cset if any(ch.isdigit() for ch in t) and not (_is_year(t) or _is_quarter(t))}
        claim_periods = {t for t in cset if _is_year(t) or _is_quarter(t)}
        value_match = any(t in qset for t in claim_values)
        period_match = any(t in qset for t in claim_periods)
        has_value = bool(claim_values)
        def w(tok: str) -> float:
            return float(weights.get(tok, 1.0))
        denom = sum(w(t) for t in cset)
        inter = sum(w(t) for t in (cset & qset))
        score = inter / max(1.0, denom)
        return score, has_value, value_match, period_match

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
            policy = os.getenv('OVERLAP_MULTI_CITE_POLICY', 'union').strip().lower() or 'union'
            threshold = float(cfg.get('threshold', 0.18))
            discount_applied = False
            # Build token sets per cite
            per_cite_tokens: List[List[str]] = []
            # approximate number of sources with highlights
            num_sources_covered = 0
            if policy == 'union' or policy not in {'union','any','all'}:
                qtoks_union: List[str] = []
                for h in highlights:
                    q = str((h or {}).get('quote') or '')
                    if q:
                        qtoks_union.extend(_qtoks(q))
                score, has_value, value_match, period_match = self._weighted_recall(ctoks, qtoks_union, weights=token_weights)
                if highlights:
                    num_sources_covered = 1  # union
                if has_value and (not value_match):
                    if bool(cfg.get('require_numeric_match', False)) or bool(os.getenv('OVERLAP_REQUIRE_VALUE_MATCH','0').lower() in {'1','true','yes','on'}):
                        score = 0.0
                    else:
                        score *= float(cfg.get('discount_word_only_numeric', 0.25) or 0.25)
                        discount_applied = True
                passed = (score >= threshold) and (not missing)
            else:
                # Build token sets per cited source id
                src_map: Dict[str, List[str]] = {}
                for cid in cite_ids:
                    src_map.setdefault(cid, [])
                for h in highlights:
                    q = str((h or {}).get('quote') or '')
                    if not q:
                        continue
                    for cid in cite_ids:
                        # naive: attach all highlights to every id (if unioned upstream); if backend groups, adapt here
                        src_map[cid].extend(_qtoks(q))
                scores = []
                all_ok = True
                any_ok = False
                value_presence = False
                value_match_any = False
                period_match_any = False
                for cid, toks in src_map.items():
                    if toks:
                        num_sources_covered += 1
                    s, has_value, value_m, period_m = self._weighted_recall(ctoks, toks, weights=token_weights)
                    # numeric policy
                    if has_value and (not value_m):
                        if bool(cfg.get('require_numeric_match', False)) or bool(os.getenv('OVERLAP_REQUIRE_VALUE_MATCH','0').lower() in {'1','true','yes','on'}):
                            s = 0.0
                        else:
                            s *= float(cfg.get('discount_word_only_numeric', 0.25) or 0.25)
                    scores.append(s)
                    value_presence = value_presence or has_value
                    value_match_any = value_match_any or value_m
                    period_match_any = period_match_any or period_m
                    any_ok = any_ok or (s >= threshold)
                    all_ok = all_ok and (s >= threshold)
                passed = any_ok if policy == 'any' else all_ok
            # Enforce value-match hard when configured
            try:
                require_val = bool(cfg.get('require_numeric_match', False)) or bool(os.getenv('OVERLAP_REQUIRE_VALUE_MATCH','0').lower() in {'1','true','yes','on'})
            except Exception:
                require_val = False
            if require_val and ('has_value' in locals()) and ('value_match' in locals()) and has_value and (not value_match):
                passed = False
            if not passed:
                ok = False
            details.append({
                'sentence': sent,
                'cites': cite_ids,
                'overlap_score': round(score if 'score' in locals() else 0.0, 4),
                'numeric_matched': bool('value_match' in locals() and value_match),
                'value_tokens_present': bool('has_value' in locals() and has_value),
                'period_match': bool('period_match' in locals() and period_match),
                'policy': policy,
                'num_sources_covered': num_sources_covered,
                'threshold': threshold,
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
