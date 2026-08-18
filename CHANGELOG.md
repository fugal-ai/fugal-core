# Changelog

Notable changes to Fugal. Format loosely follows [Keep a Changelog](https://keepachangelog.com/);
versions are [semantic](https://semver.org/), where the public surface is the CLI, the HTTP
endpoints, the `Fugal` class, and the `.npz` head contract in `docs/HEAD_FORMAT.md`.

**A note on what a version number does not cover.** The router head is data, not code.
Replacing `data/router_head.npz` changes every routing decision without changing any
version here. Head changes are called out explicitly under their release.

## [Unreleased]

### Added

- `tests/test_router_math.py` — unit tests covering sigmoid, L2 normalization,
  independent rows, lambda monotonicity, price calculation, and `load_head()` (both head
  formats, unpriced refusal, lambda resolution order, subsetting).
- `tests/test_adapters.py` — 45 stdlib `unittest` cases over the wire-shape adapters
  (`anthropic_to_body`, tool-call conversion in both directions, `compose_system`,
  `clamp_max_tokens`, `clean_history`). No network, no backbone, runs in milliseconds, and
  now gated in the fast CI job. These are the functions whose bugs do not raise — a dropped
  `tool_call_id` just hangs a tool loop.
- `verify/verify_head.py` — head properties in pure numpy, no backbone, no network, <1s:
  the five-array contract, price coverage, non-degeneracy, `p_solve ∈ (0,1)` over random
  hidden states, subsetting exactness and rank stability, and λ monotonicity in cost.
- `verify/verify_calibration.py` + `docs/EVALUATION.md` — per-model AUC/Brier, pooled ECE
  with a reliability table, and routed accuracy-and-cost against five baselines including a
  hindsight-chosen best fixed model and a per-question oracle. Needs a graded fixture,
  which this repo does not ship; `docs/EVALUATION.md` specifies the format.
- `scripts/refresh_prices.py`, with `--check` for reporting drift without writing.
- `.github/workflows/prices.yml` — weekly scheduled check of the price sheet against
  OpenRouter; opens a PR with the refreshed sheet when drift exceeds tolerance.
- `pyproject.toml`: `pip install -e .` and a `fugal` console script.
- `SECURITY.md`, `CODE_OF_CONDUCT.md`, issue and PR templates, Dependabot for Actions.

### Changed

- **Head format v2: routing costs come from the current price sheet.** The `.npz` head may
  now carry `mean_in_tokens` / `mean_out_tokens` (frozen fit-time measurements) instead of a
  baked-in `mean_cost`; `load_head()` computes `mean_cost` from the current
  `data/models_2026-06.json` at load, so `scripts/refresh_prices.py` keeps the routing
  trade-off itself honest, not just billing. v1 heads (including the shipped one) still
  work, with a startup note that their routing costs are frozen at fit time. The head may
  also declare `context` (`"standalone"`, the default, or `"multiturn"`). See
  `docs/HEAD_FORMAT.md`.
- **The router reads the latest user turn — never a tool result, and history only when the
  head declares it can use it.** Previously the router was fed conversation history (which
  the standalone-question head never saw at fit time, an undocumented distribution shift)
  and, mid tool loop, the last message — a tool *result* the head cannot score. Routing now
  targets the latest user text turn, which also makes a whole tool loop deterministic:
  same turn, same hidden state, same worker. Heads with `context="multiturn"` get history
  automatically.
- **`temperature` is omitted when the caller sent none**, so each worker keeps its own
  provider default instead of inheriting `0.0` from the proxy. A caller-supplied value is
  passed through unchanged.
- **`FUGAL_LAMBDA`** environment variable overrides the head's cost sensitivity, for parity
  with `--router-lambda` (precedence: argument, then env, then the head's trained default).
- **OpenAI content parts accepted.** List-shaped `content` (typed text parts) is flattened
  in the query, the history, and system-role messages instead of being rejected or dropped.
- **Async server rewrite.** `ThreadingHTTPServer` replaced with Starlette + uvicorn. Real
  SSE streaming from OpenRouter via httpx — clients see tokens as they arrive instead of
  receiving a buffered response. The torch forward pass runs in `asyncio.to_thread()` so it
  never blocks the event loop. CORS and Host-rebinding checks moved to a clean ASGI
  middleware layer. All endpoints, auth, spend caps, and rate limiting unchanged.
- **Single HTTP client.** `requests` replaced by `httpx` in `or_request()`, which was
  the only call site. The project now uses `httpx` for both sync worker calls and async
  streaming — one fewer dependency. New dependencies: `starlette`, `uvicorn`, `httpx`.
  Removed: `requests`.
- `Fugal.__init__` factored into pure `load_prices()` / `load_head()` functions, unit-tested
  without torch.
- `NOTICE` now reproduces OpenFugu's copyright attribution (Apache-2.0 §4(d)) and
  enumerates Fugal's changes to the derived material (§4(b)); `fugal/router.py` carries an
  in-file provenance block naming exactly which parts derive from `openfugu/mini.py`, and
  the inherited `[EXEC]` provenance tags are now explained rather than left bare.
- `docs/HEAD_FORMAT.md` documents the ablation-validated choices: mean-pooling over all input
  tokens (replacing `HIDDEN_POS = -2`) and a clean routing prompt (replacing the inherited
  TRINITY dispatcher prompt with `num_agents=7`). Both were selected via systematic ablation
  (16 combinations, 40 CV folds each) and the head was refit under the winning setup.
- `requirements.txt` no longer caps `transformers` below 5; `router.py` already handled the
  `dtype=`/`torch_dtype=` split in both directions.
- `--spend-cap` help text now states that it is checked before each call (so concurrent
  requests can overshoot), resets on restart, and counts the price sheet's numbers.
- CI Actions bumped: `actions/checkout@v7`, `actions/setup-python@v7`, `actions/cache@v6`.

### Removed

- `fugal/_console.py` — `use_utf8()` consolidated into `fugal/__init__.py`.
- `data/eval_fixture.example.jsonl` — format is documented in `docs/EVALUATION.md`.
- `deploy/` — 233 lines describing one specific production host, and the only part of the
  repo nobody had executed. `setup_arm.sh` in particular cloned a URL and apt-installed as
  root. What was reusable (the unit's `ExecStart`, the spend cap, `MemoryMax`) is now a
  short appendix in `docs/INTEGRATION.md`. `python -m fugal --serve` is unchanged.

### Fixed

- **Unpriced models are refused at startup** instead of billed at an invented fallback rate
  (`1e-6/3e-6` per token), which silently mis-counted `meta.cost`, `X-Fugal-Cost-USD`, and
  every spend cap for that model. Exclude a model with `--models` or re-sync the sheet.
- **Streamed spend is settled even when the client disconnects mid-stream.** The accounting
  ran only after a completed stream, so a dropped connection spent OpenRouter credit that
  never counted against `--spend-cap` / `--daily-cap`. Both stream generators now settle
  exactly once, in a `finally`. (The usage totals arrive in the upstream's final chunk, so a
  very early disconnect can still under-count; the OpenRouter key's hard limit remains the
  outermost brake.)
- **`data/models_2026-06.json` prices were stale, which made spend caps wrong.** Six of
  seventeen had moved since the snapshot — `deepseek/deepseek-v4-pro` by +167%, so its real
  cost was under-counted almost 3x. `_price()` feeds `meta["cost"]`, the `X-Fugal-Cost-USD`
  header, `--spend-cap` and `--daily-cap`, so a stale sheet silently miscounts all four.
  Sheet refreshed, and `scripts/refresh_prices.py` now regenerates it in place.
  `mean_cost` in the head is unchanged and stays frozen: it is routing input, fit alongside
  `W`, not billing input.
- **Client temperature honoured.** The `temperature` field from client requests is now
  passed through to the worker model. Previously every call used `temperature=0.0`
  regardless of what the client asked for. (When absent it is omitted entirely — see the
  Changed entry above.)
- `verify_head.py`'s lambda check was a tautology (`|λ·0.01 − λ·0.01| < 1e-12`, unfailable);
  it now asserts the decision itself — the winner flips exactly where the p-gap crosses
  `λ × Δcost`.
- File handle leak in `Fugal.__init__` (`json.load(open(...))` → `with` statement).
- Non-transient OpenRouter errors no longer waste retries (raised as `ValueError` with
  immediate re-raise instead of `RuntimeError` caught by the retry loop).
- Docs claimed model subsetting was "bit-identical". It is identical to within one ulp
  (~1e-16): BLAS blocks a `(17, 1024)` matrix-vector product differently than a
  `(3, 1024)` one. The mathematical claim — independent rows, nothing normalised across
  models — is exact, and the resulting ranking is unchanged, which is now the property
  asserted. Found by `verify/verify_head.py` on its first run.
- Doc drift: README / `docs/INTEGRATION.md` / `answer_iter` claimed the router "reads only
  the latest message" while the code passed history, and INTEGRATION.md still said
  temperature was fixed at 0.0. Docs and behaviour now agree.

### Security

- **`allow_pickle=True` removed from all `.npz` loads.** The head's `models` array is
  `<U29` (fixed-width Unicode), which does not need pickle — verified on the shipped head.
  `allow_pickle=True` on untrusted `.npz` files is a remote code execution vector; in the
  Bittensor subnet context (validators loading miner-submitted heads), this was the one
  thing standing between the head-loading pipeline and safe operation.
- **Browsers can no longer reach the server by default.** CORS headers used to be sent for
  whatever `Origin` arrived, and the documented way to run Fugal is on localhost with no
  token and a live OpenRouter key — so any page a user visited could spend their credit
  from the browser and read the answer back. No CORS headers are now sent unless an origin
  is explicitly allowed with `--cors-origin` / `FUGAL_CORS_ORIGINS`.
- **DNS-rebinding defence.** On a loopback bind the `Host` header must name loopback;
  anything else gets a 403. Add others with `--allow-host`. Not enforced on a non-loopback
  bind, where a reverse proxy legitimately forwards a public hostname.
- Bearer-token comparison now uses `hmac.compare_digest` instead of `==`.

## [1.0.0]

Initial public release: the 17-model utility head, the price-discounted decision rule, the
OpenAI and Anthropic serving shapes, and the CPU-only deployment recipe.
