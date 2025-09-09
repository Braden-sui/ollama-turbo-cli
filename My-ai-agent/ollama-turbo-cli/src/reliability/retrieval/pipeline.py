from __future__ import annotations

from typing import List, Dict, Any, Optional
import os
import json


class RetrievalPipeline:
    """Retrieval pipeline stub: query → dedupe → chunk → compress → id assign.

    Returns empty list by default to preserve behavior until wired.
    """

    def run(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Return up to k docs from a local corpus for private RAG.

        Minimal wiring: if a JSONL/JSON corpus is provided via env (RAG_LOCAL_DOCS or EVAL_CORPUS),
        load and return entries with fields: text, title, url, source.
        """
        _ = query  # reserved for future reranking against query
        path: Optional[str] = os.getenv('RAG_LOCAL_DOCS') or os.getenv('EVAL_CORPUS')
        if not path or not os.path.exists(path):
            return []
        docs: List[Dict[str, Any]] = []
        try:
            if path.lower().endswith('.jsonl'):
                with open(path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                        except Exception:
                            continue
                        if not isinstance(obj, dict):
                            continue
                        text = obj.get('text') or obj.get('content') or ''
                        if not text:
                            continue
                        docs.append({
                            'id': str(obj.get('id') or len(docs) + 1),
                            'title': str(obj.get('title') or '')[:200],
                            'url': str(obj.get('url') or ''),
                            'source': str(obj.get('source') or 'private'),
                            'text': str(text),
                        })
                        if len(docs) >= k:
                            break
            else:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if isinstance(data, list):
                    for obj in data:
                        if not isinstance(obj, dict):
                            continue
                        text = obj.get('text') or obj.get('content') or ''
                        if not text:
                            continue
                        docs.append({
                            'id': str(obj.get('id') or len(docs) + 1),
                            'title': str(obj.get('title') or '')[:200],
                            'url': str(obj.get('url') or ''),
                            'source': str(obj.get('source') or 'private'),
                            'text': str(text),
                        })
                        if len(docs) >= k:
                            break
        except Exception:
            return []
        return docs
