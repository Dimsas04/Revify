# Architecture index

## Implementation Guide

This index is not the full contract. Do not implement from this file alone; follow the artifact paths below.

### Global artifacts

- `unit_graph.yaml` — entry points, signatures, dynamic entry points, and shared references.
- `migration_boundary.yaml` — implementation scope; use `must_rewrite`, not every source anchor.
- `wire_contracts.yaml` — HTTP and review-output contracts.
- `shared_modules.yaml` — shared CSV/cache state and its wrap strategy.
- `cross_unit_state.yaml` — cache/file flows that must remain job-safe.
- `project-structure.md` — runtime topology and active versus experimental scraper paths.
- `tech-stack.md` — current frameworks and runtime dependencies.

### Unit: extract-features

- External trigger: `POST /api/extract-features`
- Read `units/extract-features/behavior.yaml` for early feature availability and background side effects.
- Read `units/extract-features/bindings.yaml` for route wiring.
- Read `units/extract-features/unit_decomposition.yaml` for advisory split candidates only.
- Filter global state rows where writer or reader is `extract-features`.
- Completion evidence: preserved early feature publication, review readiness, and explicit error status; verify with concurrent-job tests.

### Unit: analyze-product

- External trigger: `POST /api/analyze`
- Read `units/analyze-product/behavior.yaml` for cache reuse and analysis failure behavior.
- Read `units/analyze-product/bindings.yaml` for route and 409 behavior.
- Read `units/analyze-product/unit_decomposition.yaml` for advisory split candidates only.
- Filter global contracts and state rows where unit is `analyze-product`.
- Completion evidence: selected-feature behavior, product-scoped review reuse, and no stale/partial CSV reads.

### Unit: scraper-tool

- External trigger: CrewAI review-scraper agent tool invocation.
- Read `units/scraper-tool/behavior.yaml` for login/CAPTCHA, pagination, error, and output behavior.
- Read `units/scraper-tool/bindings.yaml` for CrewAI wiring and required environment configuration.
- Read `units/scraper-tool/unit_decomposition.yaml` for advisory split candidates only.
- Filter `wire_contracts.yaml` for `review-csv`; filter shared/state rows where unit is `scraper-tool`.
- Completion evidence: measured scrape latency, record count/field parity, browser cleanup, retry/timeout behavior, and job-scoped publication.
