from __future__ import annotations
import os
import re
import io
import json
from dataclasses import dataclass
from typing import Optional, Dict, Any

from .config import WebConfig
from .fetch import _httpx_client

# Logging noise control is centralized in core.config


@dataclass
class ExtractResult:
    ok: bool
    kind: str  # html|pdf|text|binary
    markdown: str
    title: str
    date: Optional[str]
    meta: Dict[str, Any]
    used: Dict[str, bool]  # which extractors were used
    risk: str
    risk_reasons: list[str]


def _assess_risk(text: str) -> tuple[str, list[str]]:
    reasons: list[str] = []
    t = text.lower()
    flags = [
        ("ignore previous", 5),
        ("system prompt", 4),
        ("developer message", 3),
        ("hidden", 2),
        ("data:text", 3),
        ("base64", 2),
        ("#instructions", 2),
        ("prompt injection", 5),
        ("instructions: ", 2),
    ]
    score = 0
    for phrase, w in flags:
        if phrase in t:
            reasons.append(f"found '{phrase}'")
            score += w
    # Long data URIs
    if re.search(r"data:[^;]+;base64,[a-z0-9/+]{200,}", t):
        reasons.append("long data URI")
        score += 5
    risk = 'LOW'
    if score >= 8:
        risk = 'HIGH'
    elif score >= 4:
        risk = 'MED'
    return risk, reasons


def _html_to_markdown(html: str) -> tuple[str, Dict[str, Any], Dict[str, bool]]:
    # Logging noise control is centralized in core.config
    used = {"trafilatura": False, "readability": False, "jina": False}
    meta: Dict[str, Any] = {"title": "", "date": None, "lang": ""}
    markdown = ""
    if not isinstance(html, str) or not html.strip():
        # Do not call extractors with empty/None html
        return "", meta, used
    # Best-effort language detection from html tag
    try:
        mlang = re.search(r"<html[^>]*lang=\"([^\"]+)\"", html, flags=re.I)
        if mlang:
            meta["lang"] = (mlang.group(1) or '').lower()
    except Exception:
        pass
    # Try trafilatura
    try:
        import trafilatura  # type: ignore
        art = trafilatura.extract(html, include_formatting=True, include_links=True, output_format='markdown')
        if art:
            markdown = art
            used["trafilatura"] = True
            meta["title"] = trafilatura.bare_extraction(html).get('title', '') if hasattr(trafilatura, 'bare_extraction') else ''
            # Try to parse date from common meta tags and JSON-LD
            try:
                m = re.search(r"<meta[^>]+property=\"article:published_time\"[^>]+content=\"([^\"]+)\"", html, flags=re.I)
                if m:
                    meta["date"] = m.group(1)
                if not meta.get('date'):
                    m2 = re.search(r"<time[^>]+datetime=\"([^\"]+)\"", html, flags=re.I)
                    if m2:
                        meta["date"] = m2.group(1)
                # OpenGraph published/updated time
                if not meta.get('date'):
                    m3 = re.search(r"<meta[^>]+property=\"og:published_time\"[^>]+content=\"([^\"]+)\"", html, flags=re.I)
                    if m3:
                        meta["date"] = m3.group(1)
                if not meta.get('date'):
                    m4 = re.search(r"<meta[^>]+property=\"og:updated_time\"[^>]+content=\"([^\"]+)\"", html, flags=re.I)
                    if m4:
                        meta["date"] = m4.group(1)
                # itemprop and common name fallbacks
                if not meta.get('date'):
                    m5 = re.search(r"<meta[^>]+itemprop=\"datePublished\"[^>]+content=\"([^\"]+)\"", html, flags=re.I)
                    if m5:
                        meta["date"] = m5.group(1)
                if not meta.get('date'):
                    m6 = re.search(r"<meta[^>]+name=\"(date|pubdate|publishdate)\"[^>]+content=\"([^\"]+)\"", html, flags=re.I)
                    if m6:
                        meta["date"] = m6.group(2)
                if not meta.get('date'):
                    for js in re.findall(r"<script[^>]+type=\"application/ld\+json\"[^>]*>(.*?)</script>", html, flags=re.I|re.S):
                        try:
                            data = json.loads(js.strip())
                        except Exception:
                            continue
                        def _find_date(obj):
                            if isinstance(obj, dict):
                                if (obj.get('@type') in {'NewsArticle','Article'}) and obj.get('datePublished'):
                                    return obj.get('datePublished')
                                for v in obj.values():
                                    r = _find_date(v)
                                    if r:
                                        return r
                            if isinstance(obj, list):
                                for it in obj:
                                    r = _find_date(it)
                                    if r:
                                        return r
                            return None
                        d = _find_date(data)
                        if d:
                            meta['date'] = d
                            break
            except Exception:
                pass
    except Exception:
        pass
    # Fallback readability-lxml
    if not markdown:
        try:
            from readability import Document  # type: ignore
            doc = Document(html)
            title = doc.short_title() or ''
            content_html = doc.summary()
            # naive convert: strip tags
            text = re.sub(r"<[^>]+>", "\n", content_html)
            text = re.sub(r"\n{2,}", "\n\n", text)
            markdown = f"# {title}\n\n{text.strip()}\n"
            used["readability"] = True
            meta["title"] = title
            # Best-effort date parse with readability path too
            try:
                m = re.search(r"<meta[^>]+property=\"article:published_time\"[^>]+content=\"([^\"]+)\"", html, flags=re.I)
                if m:
                    meta["date"] = m.group(1)
            except Exception:
                pass
        except Exception:
            pass
    return markdown or "", meta, used


def _pdf_to_text(bin_data: bytes) -> tuple[str, Dict[str, Any], Dict[str, bool]]:
    used = {"pdfminer": False, "pymupdf": False, "ocrmypdf": False}
    text = ""
    meta: Dict[str, Any] = {}
    pages: list[str] = []
    # Prefer PyMuPDF for speed/robustness
    try:
        import fitz  # PyMuPDF type: ignore
        doc = fitz.open(stream=bin_data, filetype="pdf")
        parts: list[str] = []
        for page in doc:
            parts.append(page.get_text())
        text = "\n\n".join(parts)
        pages = parts[:]
        used["pymupdf"] = True
    except Exception:
        pass
    # Fallback to pdfminer if PyMuPDF failed or yielded empty
    if not text:
        try:
            from pdfminer.high_level import extract_text  # type: ignore
            with io.BytesIO(bin_data) as fp:
                text = extract_text(fp) or ""
            used["pdfminer"] = True
        except Exception:
            pass
    # OCR fallback if text density is very low
    if len((text or '').strip()) < 40:
        # Attempt OCR if ocrmypdf exists in PATH
        try:
            import shutil, subprocess, tempfile
            if shutil.which('ocrmypdf'):
                used["ocrmypdf"] = True
                with tempfile.TemporaryDirectory() as td:
                    in_path = os.path.join(td, 'in.pdf')
                    out_path = os.path.join(td, 'out.pdf')
                    with open(in_path, 'wb') as f:
                        f.write(bin_data)
                    subprocess.run(['ocrmypdf', '--skip-text', in_path, out_path], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    # Re-extract
                    try:
                        from pdfminer.high_level import extract_text as ex2  # type: ignore
                        with open(out_path, 'rb') as f:
                            text = ex2(f) or text
                    except Exception:
                        pass
        except Exception:
            pass
    # Derive pages and page_start_lines from text if not obtained via PyMuPDF
    if not pages:
        # pdfminer typically uses form-feed '\f' as page separator
        if '\f' in text:
            pages = text.split('\f')
        else:
            pages = [text] if text else []
    page_start_lines: list[int] = []
    cur = 1
    for ptxt in pages:
        page_start_lines.append(cur)
        cur += len(ptxt.splitlines())
    meta["page_start_lines"] = page_start_lines
    meta["pages"] = len(pages)
    return text, meta, used


def extract_content(fetch_meta: Dict[str, Any], *, cfg: Optional[WebConfig] = None) -> ExtractResult:
    cfg = cfg or WebConfig()
    ctype = (fetch_meta.get('content_type') or '').lower()
    body_path = fetch_meta.get('body_path')
    raw = b''
    if body_path and os.path.isfile(body_path):
        raw = open(body_path, 'rb').read()
    title = ''
    date = None
    markdown = ''
    meta: Dict[str, Any] = {}
    used: Dict[str, bool] = {}

    if 'pdf' in ctype or (body_path and body_path.lower().endswith('.pdf')):
        text, meta_pdf, used_pdf = _pdf_to_text(raw)
        markdown = text
        meta.update(meta_pdf)
        used.update(used_pdf)
        kind = 'pdf'
    elif 'html' in ctype or ('<html' in raw[:200].decode(errors='ignore').lower() if raw else False):
        md, meta_html, used_html = _html_to_markdown(raw.decode(errors='ignore'))
        markdown = md
        meta.update(meta_html)
        used.update(used_html)
        kind = 'html'
        # If empty, try Jina Reader fallback
        if not markdown:
            try:
                reader_url = f"https://r.jina.ai/{fetch_meta.get('final_url') or fetch_meta.get('url')}"
                with _httpx_client(cfg) as client:
                    jr = client.get(reader_url, timeout=cfg.timeout_read)
                    if jr.status_code == 200:
                        markdown = jr.text
                        used["jina"] = True
            except Exception:
                pass
    else:
        # plain text or binary
        if raw:
            try:
                markdown = raw.decode('utf-8')
                kind = 'text'
            except Exception:
                markdown = ''
                kind = 'binary'
        else:
            markdown = ''
            kind = 'binary'

    # Optional cleanup for common wiki artifacts such as "[edit]" anchors
    try:
        if cfg.clean_wiki_edit_anchors and markdown:
            # Remove markdown links whose visible text is exactly 'edit' (case-insensitive)
            markdown = re.sub(r"\[\s*edit\s*\]\([^)]+\)", "", markdown, flags=re.I)
            # Occasionally appears as nested brackets from HTML → markdown conversions
            markdown = re.sub(r"\[\s*\[?\s*edit\s*\]?\s*\]\([^)]+\)", "", markdown, flags=re.I)
            # Remove stray '[edit]' tokens
            markdown = re.sub(r"\[\s*edit\s*\]", "", markdown, flags=re.I)
            # Trim leftover excess whitespace before newlines
            markdown = re.sub(r"[ \t]+\n", "\n", markdown)
    except Exception:
        pass

    risk, reasons = _assess_risk((markdown or '')[:10000])
    return ExtractResult(
        ok=bool(markdown),
        kind=kind,
        markdown=markdown,
        title=meta.get('title', ''),
        date=meta.get('date'),
        meta=meta,
        used=used,
        risk=risk,
        risk_reasons=reasons,
    )
