#!/usr/bin/env python3
from __future__ import annotations

import re
import sys
import subprocess
from pathlib import Path


PATTERNS = [
    (r"AKIA[0-9A-Z]{16}", "AWS Access Key"),
    (r"(?i)(api[_-]?key|token|secret|password|authorization|bearer)\s*[:=]\s*['\"]?[A-Za-z0-9_\-]{16,}", "Generic API/Secret"),
    (r"sk-[A-Za-z0-9]{20,}", "OpenAI-like Secret Key"),
    (r"AIza[0-9A-Za-z\-_]{35}", "Google API Key"),
    (r"(?i)stripe_(live|test)_[A-Za-z0-9]{24,}", "Stripe Key"),
    (r"xox[baprs]-[A-Za-z0-9\-]{10,}", "Slack Token"),
    (r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----", "Private Key Block"),
    (r"eyJ[\w-]{10,}\.[\w-]{10,}\.[\w-]{10,}", "JWT-like Token"),
]

KEY_VALUE_RE = re.compile(r"^\s*(password|secret|token|api[_-]?key|authorization)\s*:\s*(.+)$", re.I)


def git_ls_yaml() -> list[Path]:
    try:
        out = subprocess.check_output(["git", "ls-files", "-co", "--exclude-standard"], text=True)
        files = [Path(p) for p in out.splitlines() if p.strip()]
    except Exception:
        # Fallback: glob
        files = list(Path('.').rglob('*'))
    out = []
    for p in files:
        s = str(p)
        if p.suffix.lower() in {'.yml', '.yaml'}:
            if p.name.endswith('.example.yaml') or p.name.endswith('.example.yml'):
                continue
            out.append(p)
    return out


def load_allowlist() -> list[str]:
    p = Path('scripts/verify_yaml_allowlist.txt')
    if not p.exists():
        return []
    lines = []
    for line in p.read_text(encoding='utf-8').splitlines():
        s = line.strip()
        if not s or s.startswith('#'):
            continue
        lines.append(s)
    return lines


def is_allowed(path: Path, allow: list[str]) -> bool:
    sp = str(path).replace('\\', '/')
    return any(a in sp for a in allow)


def scan_file(path: Path) -> list[tuple[int, str, str]]:
    findings: list[tuple[int, str, str]] = []
    try:
        text = path.read_text(encoding='utf-8', errors='ignore')
    except Exception as e:
        return [(0, 'error', f'cannot read: {e}')]
    for i, line in enumerate(text.splitlines(), 1):
        for rx, desc in PATTERNS:
            if re.search(rx, line):
                findings.append((i, desc, line.strip()))
        m = KEY_VALUE_RE.match(line)
        if m:
            val = m.group(2).strip()
            if val and val.lower() not in {'', 'null', 'none', 'placeholder'} and not val.startswith(('#', '""', "''")):
                findings.append((i, f"Key '{m.group(1)}' has non-empty value", line.strip()))
    return findings


def main() -> int:
    allow = load_allowlist()
    targets = git_ls_yaml()
    # Respect .gitignore implicitly via git ls-files; additionally skip explicitly ignored
    # by checking git check-ignore if available
    checked: list[Path] = []
    for p in targets:
        if is_allowed(p, allow):
            continue
        checked.append(p)

    total = 0
    for p in checked:
        f = scan_file(p)
        if f:
            for (ln, desc, line) in f:
                print(f"{p}:{ln}: {desc}: {line}")
            total += len(f)
    if total == 0:
        print("YAML SECRET SCAN: CLEAN")
        return 0
    else:
        print(f"YAML SECRET SCAN: FOUND {total} ISSUE(S)")
        return 1


if __name__ == '__main__':
    sys.exit(main())

