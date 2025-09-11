from __future__ import annotations

from typing import List, Dict, Any, Optional, Tuple
import os
import json
import math
import re
import glob
import hashlib


DEFAULT_CHUNK_TOKENS = 1000
DEFAULT_OVERLAP_TOKENS = 200


class RetrievalPipeline:
    """Retrieval: JSON/JSONL loader → token chunking → BM25 ranking → dedupe.

    Dependency-light BM25 with simple tokenization and stopword removal.
    Caches an in-memory index per (docs_glob, eval_corpus) on file mtimes.
    """

    def __init__(self) -> None:
        self._cache_key: Optional[str] = None
        self._cache_mtimes: Dict[str, float] = {}
        self._chunks: List[Dict[str, Any]] = []
        self._df: Dict[str, int] = {}
        self._avgdl: float = 1.0
        self._index_meta: Dict[str, Any] = {}

    # ---------------------------- Public API ----------------------------
    def run(
        self,
        query: str,
        k: int = 8,
        *,
        docs_glob: Optional[str] = None,
        eval_corpus: Optional[str] = None,
        docs_in_memory: Optional[List[Dict[str, Any]]] = None,
        min_score: Optional[float] = None,
        max_chunks_per_doc: int = 200,
        chunk_tokens: int = DEFAULT_CHUNK_TOKENS,
        overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
        ephemeral: bool = False,
    ) -> List[Dict[str, Any]]:
        """Return up to k ranked chunks with metadata.

        Each result dict includes: {doc_id, chunk_id, title, url, text, score}.
        """
        if docs_in_memory is not None:
            self._ensure_index_from_docs(docs_in_memory, max_chunks_per_doc, chunk_tokens, overlap_tokens)
        else:
            sources = self._resolve_sources(docs_glob, eval_corpus)
            if not sources:
                return []
            self._ensure_index(sources, max_chunks_per_doc, chunk_tokens, overlap_tokens)
        q_terms = self._tokenize_query(query)
        if not q_terms:
            if ephemeral:
                self.reset()
            return []
        scored = self._rank_bm25(q_terms)
        # Title and coverage boosts
        scored2 = self._apply_boosts(scored, q_terms)
        # Dedupe by (doc_id, para_key) – prefer highest score
        deduped = self._dedupe(scored2)
        # Thresholding
        if min_score is not None:
            deduped = [r for r in deduped if float(r.get('score') or 0.0) >= float(min_score)]
        out = deduped[: max(1, int(k))]
        if ephemeral:
            self.reset()
        return out

    # ---------------------------- Indexing -----------------------------
    def _resolve_sources(self, docs_glob: Optional[str], eval_corpus: Optional[str]) -> List[str]:
        paths: List[str] = []
        # Compute a stable project root relative to this file (src/…/pipeline.py -> project root two dirs up)
        try:
            _here = os.path.abspath(os.path.dirname(__file__))
            _src_root = os.path.abspath(os.path.join(_here, "..", ".."))  # .../ollama-turbo-cli
            _proj_root = os.path.abspath(os.path.join(_src_root, ".."))     # .../My-ai-agent
        except Exception:
            _src_root = os.getcwd()
            _proj_root = os.getcwd()
        if docs_glob:
            try:
                # First: as provided (relative to current working directory)
                matches = sorted(glob.glob(docs_glob))
                if not matches and not os.path.isabs(docs_glob):
                    # Second: relative to package root (ollama-turbo-cli)
                    alt1 = os.path.join(_src_root, docs_glob)
                    matches = sorted(glob.glob(alt1))
                if not matches and not os.path.isabs(docs_glob):
                    # Third: relative to repository root
                    alt2 = os.path.join(_proj_root, docs_glob)
                    matches = sorted(glob.glob(alt2))
                paths.extend(matches)
            except Exception:
                pass
        path_env = os.getenv('RAG_LOCAL_DOCS') or os.getenv('EVAL_CORPUS') or eval_corpus
        if path_env:
            p = path_env
            if not os.path.isabs(p):
                cand = os.path.join(_proj_root, p)
                if os.path.isfile(cand):
                    p = cand
            paths.append(p)
        # Keep only existing files
        uniq: List[str] = []
        seen = set()
        for p in paths:
            if not p or p in seen:
                continue
            if os.path.exists(p) and os.path.isfile(p):
                uniq.append(p)
                seen.add(p)
        return uniq

    def _ensure_index(self, files: List[str], max_chunks_per_doc: int, chunk_tokens: int, overlap_tokens: int) -> None:
        key = f"{'|'.join(files)}|ct={int(chunk_tokens)}|ov={int(overlap_tokens)}|mc={int(max_chunks_per_doc)}"
        mtimes = {p: os.path.getmtime(p) for p in files}
        if (self._cache_key == key) and all(self._cache_mtimes.get(p) == mtimes.get(p) for p in files):
            return
        # (Re)build index
        chunks: List[Dict[str, Any]] = []
        # Compute content fingerprint for determinism
        fp = hashlib.sha256()
        fp.update(f"ct={int(chunk_tokens)};ov={int(overlap_tokens)};mc={int(max_chunks_per_doc)}".encode())
        warn_count = 0
        for p in files:
            try:
                ext = p.lower()
                if ext.endswith('.jsonl'):
                    with open(p, 'r', encoding='utf-8') as f:
                        lines = [line for line in f]
                        for line in lines:
                            fp.update(line.encode('utf-8', errors='ignore'))
                        rows = []
                        for line in lines:
                            if not line.strip():
                                continue
                            try:
                                rows.append(json.loads(line))
                            except Exception:
                                warn_count += 1
                else:
                    with open(p, 'r', encoding='utf-8') as f:
                        text = f.read()
                        fp.update(text.encode('utf-8', errors='ignore'))
                        try:
                            data = json.loads(text)
                        except Exception:
                            warn_count += 1
                            data = []
                    rows = data if isinstance(data, list) else []
            except Exception:
                rows = []
            for obj in rows:
                if not isinstance(obj, dict):
                    continue
                text = str(obj.get('text') or obj.get('content') or '')
                if not text:
                    continue
                doc_id = str(obj.get('id') or f"{os.path.basename(p)}:{len(chunks)}")
                title = str(obj.get('title') or '')
                url = str(obj.get('url') or '')
                ts = obj.get('timestamp')
                # Chunk by tokens
                chs = self._chunk_text(text, chunk_tokens, overlap_tokens)
                # Cap chunks per doc with stratified sampling across the text
                if len(chs) > max_chunks_per_doc:
                    stride = max(1, len(chs) // max_chunks_per_doc)
                    chs = [chs[i] for i in range(0, len(chs), stride)][:max_chunks_per_doc]
                for ci, (tokens, original_text) in enumerate(chs):
                    chunk_id = f"{doc_id}#{ci*max(1, chunk_tokens-overlap_tokens)}"
                    chunks.append({
                        'doc_id': doc_id,
                        'chunk_id': chunk_id,
                        'title': title,
                        'url': url,
                        'ts': ts,
                        'text': original_text,
                        'tokens': tokens,
                        'dl': len(tokens),
                    })
        # Build DF and avgdl
        df: Dict[str, int] = {}
        total_dl = 0
        for ch in chunks:
            total_dl += int(ch['dl'])
            seen_terms = set(ch['tokens'])
            for t in seen_terms:
                df[t] = df.get(t, 0) + 1
        avgdl = (total_dl / max(1, len(chunks))) if chunks else 1.0
        self._chunks = chunks
        self._df = df
        self._avgdl = float(avgdl)
        self._cache_key = key
        self._cache_mtimes = mtimes
        self._index_meta = {
            'fingerprint': fp.hexdigest(),
            'files': list(files),
            'chunk_tokens': int(chunk_tokens),
            'overlap_tokens': int(overlap_tokens),
            'max_chunks_per_doc': int(max_chunks_per_doc),
            'num_chunks': len(self._chunks),
            'avgdl': float(self._avgdl),
            'load_warnings': int(warn_count),
        }

    def _ensure_index_from_docs(self, rows: List[Dict[str, Any]], max_chunks_per_doc: int, chunk_tokens: int, overlap_tokens: int) -> None:
        """Build index from in-memory docs; skips persistent file caching. Fingerprint reflects content."""
        chunks: List[Dict[str, Any]] = []
        fp = hashlib.sha256()
        fp.update(f"ct={int(chunk_tokens)};ov={int(overlap_tokens)};mc={int(max_chunks_per_doc)}".encode())
        warn_count = 0
        for obj in (rows or []):
            if not isinstance(obj, dict):
                continue
            text = str(obj.get('text') or obj.get('content') or '')
            if not text:
                continue
            for k in ('id','title','url','timestamp'):
                try:
                    v = obj.get(k)
                    if v is not None:
                        fp.update(str(v).encode('utf-8', errors='ignore'))
                except Exception:
                    pass
            doc_id = str(obj.get('id') or f"mem:{len(chunks)}")
            title = str(obj.get('title') or '')
            url = str(obj.get('url') or '')
            ts = obj.get('timestamp')
            chs = self._chunk_text(text, chunk_tokens, overlap_tokens)
            if len(chs) > max_chunks_per_doc:
                stride = max(1, len(chs) // max_chunks_per_doc)
                chs = [chs[i] for i in range(0, len(chs), stride)][:max_chunks_per_doc]
            for ci, (tokens, original_text) in enumerate(chs):
                chunk_id = f"{doc_id}#{ci*max(1, chunk_tokens-overlap_tokens)}"
                chunks.append({
                    'doc_id': doc_id,
                    'chunk_id': chunk_id,
                    'title': title,
                    'url': url,
                    'ts': ts,
                    'text': original_text,
                    'tokens': tokens,
                    'dl': len(tokens),
                })
        # Build DF and avgdl
        df: Dict[str, int] = {}
        total_dl = 0
        for ch in chunks:
            total_dl += int(ch['dl'])
            seen_terms = set(ch['tokens'])
            for t in seen_terms:
                df[t] = df.get(t, 0) + 1
        avgdl = (total_dl / max(1, len(chunks))) if chunks else 1.0
        self._chunks = chunks
        self._df = df
        self._avgdl = float(avgdl)
        # No persistent cache key; index_meta for introspection only
        self._cache_key = None
        self._cache_mtimes = {}
        self._index_meta = {
            'fingerprint': fp.hexdigest(),
            'files': [],
            'chunk_tokens': int(chunk_tokens),
            'overlap_tokens': int(overlap_tokens),
            'max_chunks_per_doc': int(max_chunks_per_doc),
            'num_chunks': len(self._chunks),
            'avgdl': float(self._avgdl),
            'load_warnings': int(warn_count),
        }

    # ------------------------------ Search -----------------------------
    _STOP = {
        'the','a','an','of','and','or','to','is','are','was','were','be','on','in','for','with','as','by','at','from','that','this','it','its','into','we','you','they','i','me','my','our','your','their'
    }

    def _tokenize(self, text: str) -> List[str]:
        s = (text or '').lower()
        # preserve numerals and words; drop punctuation
        toks = re.findall(r"[a-z0-9]+", s)
        return [t for t in toks if t and (t not in self._STOP)]

    def _tokenize_query(self, q: str) -> List[str]:
        toks = self._tokenize(q)
        # de-dup but preserve order
        out: List[str] = []
        seen = set()
        for t in toks:
            if t not in seen:
                seen.add(t)
                out.append(t)
        return out

    def _chunk_text(self, text: str, chunk_tokens: int, overlap_tokens: int) -> List[Tuple[List[str], str]]:
        """Chunk text into token windows with overlap. Returns list of (tokens, snippet_text)."""
        toks = self._tokenize(text)
        if not toks:
            return []
        step = max(1, int(chunk_tokens) - int(overlap_tokens))
        wins: List[Tuple[List[str], str]] = []
        for i in range(0, len(toks), step):
            window = toks[i:i+int(chunk_tokens)]
            if not window:
                break
            snippet = " ".join(window)
            wins.append((window, snippet))
            if len(window) < chunk_tokens:
                break
        return wins

    def _rank_bm25(self, q_terms: List[str]) -> List[Dict[str, Any]]:
        N = max(1, len(self._chunks))
        k1 = 1.5
        b = 0.75
        # precompute idf
        idf: Dict[str, float] = {}
        for t in q_terms:
            df_t = self._df.get(t, 0)
            # BM25 idf with +0.5 smoothing
            idf[t] = math.log((N - df_t + 0.5) / (df_t + 0.5) + 1e-9)
        results: List[Dict[str, Any]] = []
        for ch in self._chunks:
            dl = float(ch['dl'])
            score = 0.0
            # term frequencies for this chunk
            tf: Dict[str, int] = {}
            for t in ch['tokens']:
                if t in q_terms:
                    tf[t] = tf.get(t, 0) + 1
            if not tf:
                continue
            for t, f in tf.items():
                w = idf.get(t, 0.0)
                if w == 0.0:
                    continue
                denom = f + k1 * (1 - b + b * (dl / max(1.0, self._avgdl)))
                score += w * ((f * (k1 + 1)) / max(1e-9, denom))
            if score <= 0:
                continue
            r = {k: ch[k] for k in ('doc_id','chunk_id','title','url','ts','text')}
            # carry tokens for simple highlighting downstream
            r['_tokens'] = ch.get('tokens')
            r['score'] = float(score)
            results.append(r)
        # sort high to low
        results.sort(key=lambda x: x.get('score', 0.0), reverse=True)
        return results

    def _apply_boosts(self, rows: List[Dict[str, Any]], q_terms: List[str]) -> List[Dict[str, Any]]:
        if not rows:
            return rows
        qs = set(q_terms)
        out: List[Dict[str, Any]] = []
        for r in rows:
            score = float(r.get('score') or 0.0)
            title = (r.get('title') or '').lower()
            # Title boost: +5% if any query term in title, +10% if >=2 terms
            m = sum(1 for t in qs if t and t in title)
            if m >= 2:
                score *= 1.10
            elif m == 1:
                score *= 1.05
            # First-chunk boost: if chunk start offset is near 0, +5%
            try:
                ch = str(r.get('chunk_id') or '')
                off = int(ch.split('#', 1)[1]) if '#' in ch else 0
                if off <= 50:
                    score *= 1.05
            except Exception:
                pass
            # Coverage bonus: up to +10% based on fraction of unique terms matched in chunk text
            txt = (r.get('text') or '').lower()
            uniq_matches = sum(1 for t in qs if t and t in txt)
            if qs:
                score *= (1.0 + min(0.10, 0.10 * (uniq_matches / max(1, len(qs)))))
            r2 = dict(r)
            r2['score'] = float(score)
            # Compute simple highlights (token index and term) for up to 3 query terms
            try:
                tokens = list(r.get('_tokens') or [])
                hi = []
                if tokens:
                    seen_terms = set()
                    for idx, tok in enumerate(tokens):
                        if tok in qs and tok not in seen_terms:
                            hi.append({'term': tok, 'token_index': idx})
                            seen_terms.add(tok)
                            if len(hi) >= 3:
                                break
                if hi:
                    r2['highlights'] = hi
            except Exception:
                pass
            # Drop internal tokens field in final row
            r2.pop('_tokens', None)
            out.append(r2)
        out.sort(key=lambda x: x.get('score', 0.0), reverse=True)
        return out

    def _dedupe(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        seen: set[Tuple[str, int]] = set()
        for r in rows:
            doc_id = str(r.get('doc_id') or '')
            ch = str(r.get('chunk_id') or '')
            # chunk_id format: "<doc_id>#<start_offset>"; bucket paragraphs by 400-token windows
            try:
                off_str = ch.split('#', 1)[1]
                start = int(off_str) if off_str.isdigit() else 0
            except Exception:
                start = 0
            bucket = start // 400
            key = (doc_id, bucket)
            if key in seen:
                continue
            seen.add(key)
            out.append(r)
        return out

    # --------------------------- Introspection --------------------------
    def get_index_meta(self) -> Dict[str, Any]:
        return dict(self._index_meta)

    def reset(self) -> None:
        """Clear in-memory index to free memory and ensure fresh state for next run."""
        self._cache_key = None
        self._cache_mtimes = {}
        self._chunks = []
        self._df = {}
        self._avgdl = 1.0
        self._index_meta = {}
