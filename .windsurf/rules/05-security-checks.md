---
trigger: manual
---

@sec-check
<threat_model>

Identify entry points, authn/authz boundaries, and data flows touched by this change.

List trust assumptions; validate inputs at boundaries.
</threat_model>

<secrets_and_pii>

Never commit or print secrets; scrub examples.

Avoid logging PII; if unavoidable, redact at source.
</secrets_and_pii>