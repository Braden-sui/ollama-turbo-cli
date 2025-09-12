# Cutover Preparation Guide (Evidence-First Pipeline)

This document summarizes flags, telemetry, and the validation checklist to safely cut over from legacy allowlist-first behavior to the evidence-first, claim-centric path.

## Flags

- EVIDENCE_FIRST: enable EF orchestrator.
- EVIDENCE_FIRST_KILL_SWITCH: global kill switch (true disables EF regardless of EVIDENCE_FIRST).
- WEB_CORROBORATE_ENABLE: attach corroborators and corroboration slice in EF (debug-only).
- WEB_COUNTER_CLAIM_ENABLE: attach counter-claim debug block per citation.
- WEB_REPUTATION_ENABLE: attach reputation inputs and prior slice in EF.
- WEB_WIRE_DEDUP_ENABLE: compute wire/syndication dedup meta (debug-only).
- WEB_RESCUE_SWEEP: run adaptive allowlist-seeded rescue and attach debug metadata.
- WEB_PREFILTER_DISCOURAGED: drop discouraged hosts pre-worker (optional optimization).
- WEB_TIER_FIRST_PASS: small tier-0 warm start pass to seed discovery.
- WEB_VERACITY_LEDGER_ENABLE: append JSONL entries under `<cache_root>/veracity_ledger.jsonl`.
- WEB_CUTOVER_PREP: enable deprecation telemetry for legacy hard gates; no behavior change.
- WEB_DEBUG_METRICS: surface `answer.debug` sections.

## Debug schema

`answer.debug.schema_version = 1`

Sections (when enabled):
- search: counts and timings.
- fetch: attempted/ok/fail, dedupe/excluded, fail reasons.
- tier: counts by tier, categories, prefilter stats, tier warm start additions.
- wire: dedup meta preview (total/kept/collapsed_count/groups).
- rescue: rescue sweep meta or stub when enabled.
- extract: extraction failure count.
- discard: discard counters (e.g., missing_dateline).
- source_type: counts by source type.
- metrics:
  - time_to_first_trustworthy_cite_ms
  - corroborated_recent_share
  - calibration_hist.by_tier (placeholder)
- deprecation (WEB_CUTOVER_PREP=1): counters for discouraged_domain, blocked_source, blocked_liveblog/map, excluded_domain.
- summary_line: quick text summary of sources and counts.

## Rollback

If any regression occurs:
- Set `EVIDENCE_FIRST_KILL_SWITCH=1` to immediately restore legacy behavior.
- Set all experimental flags above back to `0`.

## Validation checklist

- EF off + kill switch on: behavior identical to baseline.
- EF on + kill switch off, all other flags off: same ranking/selection; only debug fields added.
- Corroboration and counter-claim enabled: ef.reasons shows corroborators/counter_claim; final ranking unchanged.
- Wire dedup enabled: `debug.wire` present; no ranking change.
- Rescue sweep enabled: `debug.rescue` present; no ranking change.
- Deprecation telemetry: `debug.deprecation` counters increase when legacy gates fire.
- Metrics present and reasonable on non-trivial queries:
  - time_to_first_trustworthy_cite_ms is non-null when a Tier 0/1 or trusted domain appears.
  - corroborated_recent_share in [0,1].

## Notes

- All changes are additive, flag-gated, and maintain backward compatibility.
- No external services are introduced; all logic uses local modules.
- Avoid printing secrets. Ledger entries contain only query, citations URLs, and flags.
