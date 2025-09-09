from __future__ import annotations

from typing import List, Dict, Any, Tuple
import re
import os
import logging
from ..validation.overlap import _tokens as _ovl_tokens


class Validator:
    """CiteFence-like validator: checks that cited sentences are supported by quotes.

    Modes: off | warn | enforce (behavior left to caller; this class only reports).
    """

    def __init__(self, mode: str = "off") -> None:
        self.mode = mode if mode in {"off", "warn", "enforce"} else "off"

    def _cfg(self) -> Dict[str, Any]:
        # Load from YAML with precedence: env > local.yaml > profile > reliability.yaml > defaults
        ycfg: Dict[str, Any] = {}
        try:
            import yaml  # type: ignore
            base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'config')
            base_path = os.getenv('RELIABILITY_CFG') or os.path.join(base_dir, 'reliability.yaml')
            if os.path.isfile(base_path):
                with open(base_path, 'r', encoding='utf-8') as f:
                    ycfg = yaml.safe_load(f) or {}
            # Overlay profile (if any)
            profile = (os.getenv('GOVERNANCE_PROFILE') or os.getenv('OVERLAP_PROFILE') or '').strip().lower()
            if profile:
                prof_path = os.path.join(base_dir, 'profiles', f'{profile}.yaml')
                if os.path.isfile(prof_path):
                    with open(prof_path, 'r', encoding='utf-8') as f:
                        prof = yaml.safe_load(f) or {}
                        # shallow merge
                        for k, v in (prof or {}).items():
                            if isinstance(v, dict) and isinstance(ycfg.get(k), dict):
                                ycfg[k].update(v)
                            else:
                                ycfg[k] = v
            # Overlay local.yaml
            local_path = os.path.join(base_dir, 'local.yaml')
            if os.path.isfile(local_path):
                with open(local_path, 'r', encoding='utf-8') as f:
                    loc = yaml.safe_load(f) or {}
                    for k, v in (loc or {}).items():
                        if isinstance(v, dict) and isinstance(ycfg.get(k), dict):
                            ycfg[k].update(v)
                        else:
                            ycfg[k] = v
        except Exception:
            ycfg = {}
        yover = (((ycfg or {}).get('reliability') or {}).get('overlap') or {})
        # Log once
        global _CFG_LOGGED  # type: ignore
        try:
            _ = _CFG_LOGGED  # type: ignore[name-defined]
        except NameError:  # pragma: no cover - minimal guard
            _CFG_LOGGED = False  # type: ignore
        if not _CFG_LOGGED:
            prof = (os.getenv('GOVERNANCE_PROFILE') or os.getenv('OVERLAP_PROFILE') or 'balanced')
            th = yover.get('threshold', 0.18)
            pol = yover.get('multi_cite_policy', 'union')
            rvm = yover.get('require_value_match', False)
            # Only log this line if explicitly enabled to avoid stdout noise
            try:
                if (os.getenv('OVERLAP_LOG') or '').strip().lower() in {'1','true','yes','on'}:
                    logging.getLogger(__name__).info(
                        f"[reliability/overlap] profile={prof} threshold={th} policy={pol} require_value_match={rvm}"
                    )
            except Exception:
                pass
            _CFG_LOGGED = True  # type: ignore
        def _b(name: str, d: bool) -> bool:
            v = os.getenv(name)
            if v is None:
                v2 = yover.get(name.lower().replace('overlap_','').replace('_','-')) if isinstance(yover, dict) else None
                if isinstance(v2, bool):
                    return v2
                return d
            return str(v).strip().lower() in {"1","true","yes","on"}
        def _f(name: str, d: float) -> float:
            try:
                v = os.getenv(name)
                if v not in (None,""):
                    return float(v)
                v2 = yover.get(name.lower().replace('overlap_','').replace('_','-')) if isinstance(yover, dict) else None
                return float(v2) if v2 is not None else d
            except Exception:
                return d
        def _i(name: str, d: int) -> int:
            try:
                v = os.getenv(name)
                if v not in (None,""):
                    return int(v)
                v2 = yover.get(name.lower().replace('overlap_','').replace('_','-')) if isinstance(yover, dict) else None
                return int(v2) if v2 is not None else d
            except Exception:
                return d
        tw_raw = os.getenv("OVERLAP_TOKEN_WEIGHTS")
        if not tw_raw and isinstance(yover, dict) and isinstance(yover.get('token_weights'), dict):
            tw_raw = ','.join(f"{k}:{v}" for k,v in yover['token_weights'].items())
        if not tw_raw:
            tw_raw = "announced:0.3,launched:0.3,reported:0.3,shipped:0.3"
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
        # Profile-aware defaults
        try:
            prof_name = (os.getenv('GOVERNANCE_PROFILE') or os.getenv('OVERLAP_PROFILE') or 'balanced').strip().lower()
        except Exception:
            prof_name = 'balanced'
        default_threshold = 0.22 if prof_name == 'strict' else 0.18
        default_policy = 'any' if prof_name == 'strict' else 'union'
        default_rvm = True if prof_name == 'strict' else False
        return {
            'threshold': _f('OVERLAP_THRESHOLD', default_threshold),
            'discount_word_only_numeric': _f('OVERLAP_DISCOUNT_WORD_ONLY_NUMERIC', 0.25),
            'require_numeric_match': _b('OVERLAP_REQUIRE_NUMERIC_MATCH', False),
            'require_value_match': _b('OVERLAP_REQUIRE_VALUE_MATCH', default_rvm),
            'min_claim_tokens': _i('OVERLAP_MIN_CLAIM_TOKENS', 3),
            'allow_multi_cite': _b('OVERLAP_ALLOW_MULTI_CITE', True),
            'policy': os.getenv('OVERLAP_MULTI_CITE_POLICY') or yover.get('multi_cite_policy') or default_policy,
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
        pattern = re.compile(
            r'(.+?)\s*(\[(?:[0-9,\s\-–]+)\](?:\s*\[(?:[0-9,\s\-–]+)\])*)\s*[)\]"”’]*\s*(?:[.!?]|$)',
            re.M,
        )
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
            require_val = bool(cfg.get('require_numeric_match', False)) or bool(os.getenv('OVERLAP_REQUIRE_VALUE_MATCH','0').lower() in {'1','true','yes','on'})
            if policy == 'union' or policy not in {'union','any','all'}:
                qtoks_union: List[str] = []
                for h in highlights:
                    q = str((h or {}).get('quote') or '')
                    if q:
                        qtoks_union.extend(_qtoks(q))
                score, has_value, value_match, period_match = self._weighted_recall(ctoks, qtoks_union, weights=token_weights)
                # Recompute value-match robustly using simple sets to avoid edge cases
                def _is_year(tok: str) -> bool:
                    return len(tok) == 4 and tok.isdigit() and tok.startswith('20')
                def _is_quarter(tok: str) -> bool:
                    return tok.startswith('q') and '-' in tok and tok[1].isdigit()
                cvals = {t for t in ctoks if any(ch.isdigit() for ch in t) and not (_is_year(t) or _is_quarter(t))}
                qvals = {t for t in qtoks_union if any(ch.isdigit() for ch in t) and not (_is_year(t) or _is_quarter(t))}
                val_match = bool(cvals & qvals)
                has_value = bool(cvals)
                value_match = val_match
                # strict percent check if required
                if require_val:
                    claim_pct = {t for t in cvals if t.endswith('%')}
                    quote_pct = {t for t in qvals if t.endswith('%')}
                    if claim_pct and not (claim_pct & quote_pct):
                        value_match = False
                if highlights:
                    num_sources_covered = 1  # union
                if has_value and (not value_match):
                    if require_val:
                        score = 0.0
                    else:
                        score *= float(cfg.get('discount_word_only_numeric', 0.25) or 0.25)
                        discount_applied = True
                passed = (score >= threshold) and (not missing)
                if require_val and has_value and (not value_match):
                    passed = False
            else:
                # Build token sets per cited source id from citations_map
                src_map: Dict[str, List[str]] = {}
                for cid in cite_ids:
                    src_map.setdefault(cid, [])
                    meta = (citations_map or {}).get(str(cid)) if isinstance(citations_map, dict) else None
                    hls = (meta or {}).get('highlights') if isinstance(meta, dict) else None
                    if hls and isinstance(hls, list):
                        for h in hls:
                            q = str((h or {}).get('quote') or '')
                            if q:
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
                    # Reinforce value_m via set intersection
                    def _is_year(tok: str) -> bool:
                        return len(tok) == 4 and tok.isdigit() and tok.startswith('20')
                    def _is_quarter(tok: str) -> bool:
                        return tok.startswith('q') and '-' in tok and tok[1].isdigit()
                    cvals = {t for t in ctoks if any(ch.isdigit() for ch in t) and not (_is_year(t) or _is_quarter(t))}
                    qvals = {t for t in toks if any(ch.isdigit() for ch in t) and not (_is_year(t) or _is_quarter(t))}
                    value_m = bool(cvals & qvals)
                    if require_val:
                        claim_pct = {t for t in cvals if t.endswith('%')}
                        quote_pct = {t for t in qvals if t.endswith('%')}
                        if claim_pct and not (claim_pct & quote_pct):
                            value_m = False
                    # numeric policy
                    if has_value and (not value_m):
                        if require_val:
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

        # If strict value match required, fail when any cited sentence lacks value match
        if bool(os.getenv('OVERLAP_REQUIRE_VALUE_MATCH','0').lower() in {'1','true','yes','on'}):
            for d in details:
                if d.get('value_tokens_present') and not d.get('numeric_matched'):
                    ok = False

        return {
            'mode': self.mode,
            'status': 'ok' if ok else 'fail',
            'ok': ok,
            'citations_present': citations_present,
            'details': details,
        }
