---
trigger: model_decision
description: Whenever refactors of code is being undertaken.
---

# 06-refactor-surgeon

<goal>
Perform behavior-preserving refactors to improve clarity, cohesion, and testability.
Default to no functional change unless explicitly authorized.
Keep diffs small, reviewable, and reversible.
</goal>

<triggers>
Use when: renaming, extracting functions/modules, reducing duplication, or reorganizing code without changing public contracts or side effects.
</triggers>

<preconditions>
- Collect a quick inventory of the target files and current tests covering them.
- List clear <invariants> that must not change (outputs, side effects, error types/messages, performance bounds if relevant).
- If tests are weak or missing, write minimal smoke/characterization tests first (see 07-test-builder).
</preconditions>

<invariants>
- No public API shape change (types, names, response formats).
- No new I/O, network, or DB access.
- Preserve logging levels and semantics.
- Preserve exceptions: kinds, messages, and when they are thrown.
- Preserve numeric tolerance/precision where applicable.
</invariants>

<plan>
1) Explain the refactor in 3–6 bullets.
2) Propose file-by-file edits and expected diffs.
3) Call out risks (e.g., implicit coupling, reflection/dynamic imports).
4) Show a rollback plan (git commands or patch strategy).
</plan>

<edits>
- Extract pure functions before moving stateful logic.
- Co-locate helpers with callers or create a dedicated module with a stable import path.
- Replace boolean flags with strategy or polymorphic functions where simple.
- Remove duplication by lifting shared logic; keep names stable to reduce churn.
</edits>

<tests>
- Run existing tests first; include the command you would run.
- If coverage is thin, add focused characterization tests around the changed seams.
- Collect before/after outputs for critical paths to prove invariants.
</tests>

<diff_policy>
- Prefer multiple small commits over one giant blast.
- Keep unrelated formatting changes out of scope (respect project formatter).
</diff_policy>

<rollback>
- Provide exact commands to revert (e.g., `git restore -SW .` or `git revert <sha>`).
- If a multi-commit refactor, give the sequence to unwind safely.
</rollback>

<deliverables>
- A short plan with invariants and risks.
- Diffs with brief commit messages.
- Test results (summary), and any new tests.
- Rollback snippet.
</deliverables>

<anti_goals>
- Do not reformat the entire codebase.
- Do not rename public exports or endpoints.
- Do not change configuration, build, or deployment files.
</anti_goals>
