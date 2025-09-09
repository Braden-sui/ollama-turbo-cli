import sys
import os
from pathlib import Path
from datetime import datetime

# List of files to include (relative to project root: this script's parent directory)
FILES = [
    # Core pipeline
    "src/web/pipeline.py",
    "src/web/search.py",
    "src/web/fetch.py",
    "src/web/extract.py",
    "src/web/rerank.py",
    "src/web/archive.py",
    "src/web/robots.py",
    "src/web/progress.py",
    # Configuration
    "src/core/config.py",
    "src/web/config.py",
    "CONFIG_REFERENCE.md",
    "scripts/gen_config_reference.py",
    # Agent/tool integration
    "src/plugins/web_research.py",
    "src/client.py",
    # Environment and policy (exclude .env to avoid secrets)
    ".env.local",
    ".env.example",
    # Diagnostics and tests
    "scripts/net_check.py",
    "tests/test_web_pipeline_exhaustive.py",
    # Project readme
    "README.md",
]

BANNER = "=" * 80


def read_text_safe(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        return f"<ERROR reading {path}: {e}>\n"


def main() -> int:
    here = Path(__file__).resolve().parent
    root = here  # script lives under project_root/scripts
    if root.name == "scripts":
        root = root.parent

    out_path = None
    if len(sys.argv) > 1:
        out_path = Path(sys.argv[1]).resolve()
    else:
        out_path = root / "MONOLITH_research_pipeline.txt"

    lines = []
    lines.append(BANNER)
    lines.append("Research Pipeline Monolith\n")
    lines.append(f"Generated: {datetime.now().isoformat()}\n")
    lines.append(f"Project root: {root}\n")
    lines.append(BANNER + "\n\n")

    included = 0
    missing = []
    for rel in FILES:
        p = (root / rel).resolve()
        header = f"FILE: {rel}"
        lines.append(BANNER)
        lines.append(header)
        lines.append(BANNER + "\n")
        if p.exists() and p.is_file():
            txt = read_text_safe(p)
            # Ensure ending newline
            if not txt.endswith("\n"):
                txt += "\n"
            lines.append(txt)
            included += 1
        else:
            lines.append(f"<MISSING: {rel}>\n")
            missing.append(rel)
        lines.append("\n")

    out_path.write_text("\n".join(lines), encoding="utf-8")

    print({
        "written": str(out_path),
        "files_included": included,
        "files_missing": missing,
        "bytes": out_path.stat().st_size,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
