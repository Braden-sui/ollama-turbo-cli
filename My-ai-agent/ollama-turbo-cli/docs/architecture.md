# Evidence-First Research Pipeline Architecture

This document describes the overall architecture of the web research pipeline, its core components, and the optional Evidence-First (EF) analysis stack. It also outlines search and fetch tool integration and all relevant flags/telemetry surfaces.

## Diagram (Mermaid)

```mermaid
flowchart TD
  A[run_research(query, cfg)\nsrc/web/pipeline.py] --> B[Search\nsrc/web/search.py]
  B -->|SearchResult[]| C[_build_citation(sr)\npolicy+fetch+extract+rerank]

  subgraph Citation Build
    C --> D[Policy Checks\nallowlist/blocklist/exclude\nsource_type: liveblog/map]
    D -->|ok| E[Fetch\nfetch_url\nsrc/web/fetch.py → sandbox]
    D -->|drop| Z1[items += reason\n(debug.fetch + deprecation)]
    E --> F[Extract\nextract_content\nsrc/web/extract.py]
    F --> G[Recency Discipline\nparse dates; accept/soft-accept/reject\n(debug.discard)]
    G --> H[Dedupe\nurl + body-hash per-host\n(normalize.py)]
    H --> I[Rerank\nchunk_text + rerank_chunks]
    I --> J[Archive (optional)\nget_memento/save_page_now]
    J --> K[Attach Citation Fields\ntrust, tier, excerpts]
    K --> L{EF enabled?\nEVIDENCE_FIRST=1\nkill_switch=0}
    L -- yes --> M[EF Analysis\nclaims + validators + evidence\n+ counter-claim + reputation]
    L -- no --> N[Skip EF]
  end

  C --> O[Canonicalize + Dedupe\nsrc/web/normalize.py]
  O --> P[Sort + Provisional]

  P --> Q{Tier Sweep?\nno trusted}
  Q -- yes --> R[Site-restricted Searches\n(seeds by category/tier/allowlist)] --> S[Merge, Dedupe, Re-sort]
  Q -- no --> S

  S --> T{Rescue Sweep?\nWEB_RESCUE_SWEEP}
  T -- yes --> U[adaptive_rescue\ndebug.rescue]
  T -- no --> V[skip]

  S --> W{Wire Dedup Preview?\nWEB_WIRE_DEDUP_ENABLE}
  W -- yes --> X[collapse_citations\ndebug.wire]
  W -- no --> Y[skip]

  S --> AA[Corroboration (debug)\nWEB_CORROBORATE_ENABLE]
  AA --> AB[compute_corroboration\nattach ef.reasons.corroborators\nset small corroboration slice]

  AB --> AC[Assemble Answer\npolicy + debug schema v1]
  V --> AC
  Y --> AC
  U --> AC

  AC --> AD{Ledger?\nWEB_VERACITY_LEDGER_ENABLE}
  AD -- yes --> AE[Append JSONL\n<cache_root>/veracity_ledger.jsonl]
  AD -- no --> AF[done]
```

## Core Modules and Responsibilities

- `src/web/pipeline.py`
  - Entrypoint `run_research()` orchestrates:
    - YearGuard, search, citation build (policy/fetch/extract/rerank/archive), normalization, sorting, tier sweep.
    - Optional EF analysis (claims/validators/evidence, counter-claim, reputation, corroboration) — all flag-gated, debug-only.
    - Optional rescue sweep and wire/syndication dedup preview — debug-only.
    - Rich `answer.debug` schema v1 telemetry (search/fetch/tier/rescue/wire/extract/discard/source_type/metrics/deprecation/summary_line).
  - Provides centralized config plumbing via `_DEFAULT_CFG` and `set_default_config()`.

- `src/web/search.py`
  - `search()` returns `SearchResult` records (title, url, snippet, source, published) used by the pipeline.
  - Providers (e.g., DuckDuckGo) consume centralized `WebConfig` for UA, proxies, cache hints.

- `src/web/fetch.py` → `sandbox/net_proxy.py`, `plugins/web_fetch.py`
  - `fetch_url()` delegates to policy-aware fetch (sandbox): http/proxy allowances, max bytes, cache TTL, centralized UA.
  - Returns structured result: `ok, status, final_url, headers, content_type, body_path, browser_used, cached, reason`.

- `src/web/extract.py`
  - `extract_content(meta)` returns normalized content: `{ok, kind, markdown, title, date, meta, used, risk, risk_reasons}`.

- `src/web/rerank.py`
  - `chunk_text()` and `rerank_chunks()` score relevant excerpts; pipeline converts to `lines` with `src/web/loc.py::format_loc()`.

- `src/web/normalize.py`
  - `canonicalize(url)`: scheme/host/path/query normalization (UTM/tracking removal).
  - `dedupe_citations(citations)`: stable dedupe by `(canonical_url, normalized_title)`.

- `src/web/allowlist_tiered.py`
  - Tier lookup (0/1/2), categories, discouraged hosts; used for trust and policy decisions.

## Evidence-First (EF) Stack (Flag-Gated)

- Claims: `src/web/claims.py::extract_claims(snapshot)`
- Validators: `src/validators/claim_validation.py::validate_claim()`
- Evidence: `src/web/evidence.py::score_evidence()`
- Corroboration: `src/web/corroborate.py::claim_key(), compute_corroboration()`
- Counter-claim: `src/web/counter_claim.py::evaluate_counter_claim()`
- Reputation: `src/web/reputation.py::compute_prior()`
- Ledger: `src/web/ledger.py::log_veracity()`

EF attaches to each citation as `citation['ef']`, including `reasons` and a `confidence_breakdown` (evidence/validators/corroboration/prior; final_score is debug-only under current flags).

## Key Flags

- EF & safety: `EVIDENCE_FIRST`, `EVIDENCE_FIRST_KILL_SWITCH`.
- Corroboration: `WEB_CORROBORATE_ENABLE`.
- Counter-claim: `WEB_COUNTER_CLAIM_ENABLE`.
- Reputation: `WEB_REPUTATION_ENABLE`.
- Ledger: `WEB_VERACITY_LEDGER_ENABLE`.
- Wire/Syndication dedup preview: `WEB_WIRE_DEDUP_ENABLE`.
- Rescue sweep preview: `WEB_RESCUE_SWEEP`.
- Tier sweep: `WEB_TIER_SWEEP`, `WEB_TIER_SWEEP_MAX_SITES`, `WEB_TIER_SWEEP_STRICT`.
- Debug and cutover prep: `WEB_DEBUG_METRICS`, `WEB_CUTOVER_PREP`.

## Telemetry (Debug Schema v1)

- `search`: counts, elapsed_ms, compression mode.
- `fetch`: attempted/ok/failed, dedupe_skips, excluded, fail_reasons.
- `tier`: tier_counts, categories_seen, prefiltered_discouraged, tier_first_added.
- `rescue`: rescue meta or stub.
- `wire`: dedup grouping meta.
- `extract`: extraction failures.
- `discard`: missing_dateline, out_of_window, etc.
- `source_type`: article/liveblog/map counts.
- `metrics`: `time_to_first_trustworthy_cite_ms`, `corroborated_recent_share`, `calibration_hist.by_tier`.
- `deprecation`: legacy gate counters (when `WEB_CUTOVER_PREP=1`).
- `summary_line`: compact run summary.

## DuckDuckGo and Fetch Tools

- DuckDuckGo: `src/plugins/duckduckgo.py`
  - Uses centralized `WebConfig` (`_DEFAULT_CFG` from pipeline) for UA/proxy/cache.
  - Returns results consumed by `src/web/search.py`.

- Fetch: `src/plugins/web_fetch.py` → `src/sandbox/net_proxy.py::fetch_via_policy()`
  - Enforces sandbox policy: http/proxy allowances, max bytes, cache TTL, UA.
  - Returns structured fetch results used by `pipeline._build_citation()`.

## Cutover and Safety Controls

- Kill switch restores legacy behavior immediately.
- All EF-era logic is flag-gated and debug-only by default.
- `CUTOVER.md` documents flags, rollback, and validation.
