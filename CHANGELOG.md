# Changelog

Notable changes to Fugal. Format loosely follows [Keep a Changelog](https://keepachangelog.com/);
versions are [semantic](https://semver.org/), where the public surface is the CLI, the HTTP
endpoints, the `Fugal` class, and the `.npz` head contract in `docs/HEAD_FORMAT.md`.

**A note on what a version number does not cover.** The router head is data, not code.
Replacing `data/router_head.npz` changes every routing decision without changing any
version here. Head changes are called out explicitly under their release.

## [1.0.0] — 2026-09-08

Initial public release.

- **The router.** A 17-model logistic head over a mean-pooled, L2-normalised Qwen3-0.6B
  hidden state, and the price-discounted decision rule `utility = p - λ·mean_cost`
  (`docs/HEAD_FORMAT.md`). The shipped head is v1 (`mean_cost` baked in at fit time, λ=1.0); the
  v2 format (per-model token statistics, routing cost computed from the current price sheet
  at load) and a `context` declaration (`standalone` / `multiturn`) are supported. Subsetting
  with `--models` / `FUGAL_MODELS` is exact. The router reads the latest *user* turn — never
  a tool result, and history only when the head declares it was fit on it.
- **The server.** OpenAI Chat Completions and Anthropic Messages wire shapes from one
  process, with streaming and tool calling in both, exactly one worker call per turn, and a
  free `POST /v1/route`. Bearer auth (`FUGAL_SERVE_TOKEN`), a cumulative `--spend-cap`, an
  optional tokenless public tier bounded by `--daily-cap` and a per-IP rate limit, streamed
  spend settled even on client disconnect, and browser defences on by default: no CORS
  headers unless `--cors-origin` allows an origin, and a Host check on loopback binds
  against DNS rebinding. `SECURITY.md` has the threat model. The caller's `temperature` and
  `max_tokens` are passed through (the latter clamped to the model's real output limit);
  neither is invented when absent.
- **The CLI.** `--route` (free inspector, one-shot or interactive), `--query`, `--serve`,
  `--models`, `--router-lambda` / `FUGAL_LAMBDA`.
- **Verification.** Unit tests for the wire-shape adapters, the decision rule on a synthetic
  head, and the whole server through `build_app()` with a fake router (no backbone, <1s);
  `verify/verify_head.py` (head properties, pure numpy); `verify/verify_routing.py` (one call
  per turn against the real backbone, mocked worker); `verify/verify_calibration.py` (AUC,
  Brier, ECE and routed accuracy-vs-cost against baselines — needs a graded fixture,
  `docs/EVALUATION.md`).
- **The price sheet.** `data/models_2026-06.json` is billing input (and routing input for a
  v2 head). `scripts/refresh_prices.py` re-syncs it with OpenRouter, `--check` reports drift
  without writing, and a weekly workflow opens a PR when prices move.
