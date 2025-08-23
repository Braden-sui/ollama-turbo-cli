---
trigger: glob
globs: **/*.py
---

<python_style>

Target the project’s declared Python version; infer from pyproject.toml, runtime.txt, or CI.

Use type hints everywhere; add or tighten mypy config if present.

Default tools: pytest, ruff, black if present. If absent, propose minimal setup with commands.
</python_style>

<python_testing>

Arrange‑Act‑Assert; keep tests fast and deterministic.

Mock network and time; avoid sleeping in tests.

Name tests descriptively; cover edge cases before micro‑benchmarks.
</python_testing>

<python_quality>

Separate pure logic from I/O. Isolate side effects.

Provide simple profiling plan before nontrivial perf work.
</python_quality>

