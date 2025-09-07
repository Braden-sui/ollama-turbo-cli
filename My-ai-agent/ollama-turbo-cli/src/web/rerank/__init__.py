from __future__ import annotations
import re
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any

from ..config import WebConfig
from .router import rerank as _adapter_rerank


@dataclass
class Chunk:
    id: str
    text: str
    start_line: int
    end_line: int


def _estimate_tokens(text: str) -> int:
    # Simple approximation: 1 token ~ 4 chars avg
    return max(1, int(len(text) / 4))


def chunk_text(markdown: str, *, target_tokens: int = 768) -> List[Chunk]:
    lines = markdown.splitlines()
    chunks: List[Chunk] = []
    buf: list[str] = []
    start_line = 1
    cur_tokens = 0

    def flush(end_line: int):
        nonlocal buf, start_line, cur_tokens
        if not buf:
            return
        txt = "\n".join(buf)
        cid = f"{start_line}-{end_line}"
        chunks.append(Chunk(id=cid, text=txt, start_line=start_line, end_line=end_line))
        buf = []
        cur_tokens = 0
        start_line = end_line + 1

    for i, line in enumerate(lines, start=1):
        # Prefer breaks at headings or blank lines
        if line.strip().startswith('#') and buf:
            flush(i - 1)
        buf.append(line)
        cur_tokens += _estimate_tokens(line + '\n')
        if cur_tokens >= target_tokens and (line.strip() == '' or line.strip().startswith('#')):
            flush(i)
    flush(len(lines))
    return chunks


def _simple_rerank(query: str, chunks: List[Chunk]) -> List[Tuple[Chunk, float]]:
    q = re.findall(r"\w+", query.lower())
    out: List[Tuple[Chunk, float]] = []
    for ch in chunks:
        t = ch.text.lower()
        score = sum(t.count(w) for w in q)
        out.append((ch, float(score)))
    out.sort(key=lambda x: (-x[1], x[0].start_line))
    return out


def rerank_chunks(query: str, chunks: List[Chunk], *, cfg: Optional[WebConfig] = None, top_k: int = 5) -> List[Dict[str, Any]]:
    cfg = cfg or WebConfig()
    if not chunks:
        return []

    # Build doc list and ask adapter for best indices
    docs = [ch.text for ch in chunks]
    indices: List[int]
    try:
        indices = _adapter_rerank(cfg, query, docs, top_n=min(top_k, len(docs)))
    except Exception:
        indices = []

    results: List[Tuple[Chunk, float]] = []
    used = "adapter"
    if indices:
        # Assign descending scores by rank if provider didn't return explicit scores
        for rank, idx in enumerate(indices):
            if 0 <= idx < len(chunks):
                results.append((chunks[idx], float(len(indices) - rank)))
    else:
        # fallback
        results = _simple_rerank(query, chunks)[:top_k]
        used = "simple"

    out: List[Dict[str, Any]] = []
    for ch, score in results[:top_k]:
        # Highlight first matching snippet lines with absolute line numbers
        qwords = re.findall(r"\w+", query.lower())
        lines = ch.text.splitlines()
        highlights: list[Dict[str, Any]] = []
        for idx, ln in enumerate(lines, start=0):
            if any(w in ln.lower() for w in qwords):
                abs_line = ch.start_line + idx
                highlights.append({'line': abs_line, 'text': ln.strip()})
            if len(highlights) >= 3:
                break
        out.append({
            'id': ch.id,
            'score': float(score),
            'start_line': ch.start_line,
            'end_line': ch.end_line,
            'highlights': highlights,
            'preview': lines[:3],
        })
    return out
