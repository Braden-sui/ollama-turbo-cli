Web Query Configuration (Commit 1–3)

The following knobs are available for controlling query variants and recency behavior. Defaults preserve current behavior.

- `WEB_QUERY_COMPRESSION_MODE` (default: `aggressive`): `off | soft | aggressive`
  - off: only raw query is used
  - soft: preserve quoted phrases and numerals; gentle pruning with phrase extraction
  - aggressive: legacy fallback behavior (stopwords+years pruning, short token budget)
- `WEB_QUERY_MAX_TOKENS_FALLBACK` (default: `6`): token budget for fallback variants.
- `WEB_STOPWORD_PROFILE` (default: `standard`): `minimal | standard`
- `WEB_VARIANT_PARALLEL` (default: `false`): if `true`, allows issuing fallback variants in parallel (not enabled by default).
- `WEB_VARIANT_MAX` (default: `2`): max number of fallback variants to attempt when raw yields nothing.
- `WEB_RECENCY_SOFT_ACCEPT_WHEN_EMPTY` (default: `false`): when strict recency yields no citations, allow up to 1–2 undated, allowlisted/high-trust citations (marked `undated: true`).

API Preset via ChatOptions

The HTTP API accepts an optional `web_profile` in `ChatOptions` (`quick | balanced | rigorous`) to set reasonable presets:

- quick: soft compression, 12-token fallback, recency soft-accept enabled
- balanced: soft compression, 12-token fallback
- rigorous: defaults unchanged

