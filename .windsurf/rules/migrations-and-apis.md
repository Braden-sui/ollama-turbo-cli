---
trigger: model_decision
description: - Any change to DB schema, wire protocol, REST/GraphQL/GRPC contracts, CLI/SDK surface, or serialization formats. - Any deprecation of fields, endpoints, events, or message types.
---

# 08-migrations-and-apis

<triggers>
- Any change to DB schema, wire protocol, REST/GraphQL/GRPC contracts, CLI/SDK surface, or serialization formats.
- Any deprecation of fields, endpoints, events, or message types.
</triggers>

<compat_strategy>
Adopt "expand → migrate → contract":
1) Expand: add new fields/endpoints/versions while old continue to work.
2) Migrate: dual-write or backfill; support reads from both shapes.
3) Contract: remove old shape only after adoption metrics, comms, and a grace period.
</compat_strategy>

<api_changes>
Update validators/types and downstream callers in the same change when possible.
Provide deprecation notes or migration steps if breaking.
Version public endpoints or add content-negotiation when behavior changes.
Return explicit error codes/messages for removed or unsupported features.
</api_changes>

<data_migrations>
Provide forward/backward migration steps and backfill notes.
Include verification plan and safe rollback.
Use idempotent scripts; wrap DDL in transactions where supported.
For large tables: online/partitioned migrations, batched backfills, and throttling.
</data_migrations>

<state_migrations>
- For config/state stores (KV, caches, topics), include reindex steps and TTL impacts.
- Maintain consumers for both old and new topics/keys during migration.
</state_migrations>

<locking_and_downtime>
- Avoid table locks in peak windows; schedule off-peak or use online tools.
- Provide a zero-downtime path or clearly document expected impact and length.
</locking_and_downtime>

<observability>
- Add metrics/logs to detect mixed-shape traffic and migration completion.
- Include alarms for error spikes and lag.
</observability>

<tests>
- Add contract tests for both shapes during expand/migrate.
- Add migration tests: apply forward, verify, roll back, verify.
- Snapshot representative payloads before and after.
</tests>

<docs>
- Update API docs, schema diagrams, and examples in the same PR.
- Add deprecation timeline, feature flag names, and cutover steps.
</docs>

<release_plan>
- Phased rollout plan, canary first.
- Communication notes (internal/external) and deprecation dates.
</release_plan>

<rollback>
- Provide commands/scripts to reverse schema changes or switch read paths.
- Keep a backup/snapshot reference and TTL.
</rollback>

<deliverables>
- Migration plan (expand/migrate/contract), scripts, and verification checks.
- Updated API/spec docs and examples.
- Tests and how to run them.
- Rollout schedule and rollback steps.
</deliverables>

<anti_goals>
- No breaking changes without deprecation and migration steps.
- No destructive DDL in a single step on hot paths.
</anti_goals>
