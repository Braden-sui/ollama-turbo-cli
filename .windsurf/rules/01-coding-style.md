---
trigger: model_decision
description: Suggested activation: "apply when repo language or formatter is ambiguous; mirror local conventions")
---

<coding_style>

Mirror local conventions. Detect formatter and linter configs (e.g., pyproject.toml, ruff.toml, eslint.config.js, prettier.config.*).

Prefer small, reversible diffs. Keep edits atomic per concern.

Use early returns, clear names, and narrow function scope.

Avoid deep nesting; extract helpers instead.

Keep public APIs stable; mark breaking changes and provide migration notes.
</coding_style>