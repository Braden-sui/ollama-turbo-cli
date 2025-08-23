---
trigger: manual
---

<plan_template>

Objective: one sentence describing the refactor’s goal.

Non-goals: things explicitly out of scope.

Scope: files/modules to touch; caps on blast radius.

Invariants: behavioral contracts that must not change; list affected public APIs.

Risks: call out concurrency, I/O, and schema coupling risks.

Rollback: how to revert in one commit or command; what files to restore.
</plan_template>

<execution_steps>

Baseline: run tests/linters; record failing ones if any.

Create seams: extract pure helpers; inject dependencies; isolate side effects.

Extract: move logic into smaller functions/classes; reduce nesting; prefer composition.

Replace: update call sites incrementally; keep old path behind a toggle until parity is proven.

Delete: remove dead code once coverage shows no references.

Docs: update inline docs/comments to reflect new structure.

Tests: tighten/adjust tests to match structure while preserving behavior.
</execution_steps>

<diff_and_commits>

Show file-by-file diffs and a short rationale per file.

Commit subjects in imperative mood; bodies explain why, risks, and verification.
</diff_and_commits>

