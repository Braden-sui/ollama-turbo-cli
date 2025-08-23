---
trigger: glob
globs: **/*.{ts,tsx}
---

<ts_style>

Use strict typing; no any without a one‑line justification.

Prefer discriminated unions and readonly where helpful.

Keep public types stable; export types adjacent to modules.
</ts_style>

<react_ui>

Prefer functional components and hooks; no unnecessary context.

Co-locate tests and stories when present; keep components small and focused.

Accessibility first: label controls, keyboard nav, color‑contrast notes.
</react_ui>

<ts_tooling>

Respect existing eslint and prettier configs. Don’t override; conform.

If changing build config (Vite/Next), show exact diff and rationale.
</ts_tooling>