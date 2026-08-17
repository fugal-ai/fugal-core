# Changelog

Notable changes to Fugal. Format loosely follows [Keep a Changelog](https://keepachangelog.com/);
versions are [semantic](https://semver.org/), where the public surface is the CLI, the HTTP
endpoints, the `Fugal` class, and the `.npz` head contract in `docs/HEAD_FORMAT.md`.

**A note on what a version number does not cover.** The router head is data, not code.
Replacing `data/router_head.npz` changes every routing decision without changing any
version here. Head changes are called out explicitly under their release.

## [Unreleased]

### Changed

- **Async server rewrite.** `ThreadingHTTPServer` replaced with Starlette + uvicorn. Real
  SSE streaming from OpenRouter via httpx — clients see tokens as they arrive instead of
  receiving a buffered response. The torch forward pass runs in `asyncio.to_thread()` so it
  never blocks the event loop. CORS and Host-rebinding checks moved to a clean ASGI
  middleware layer. All endpoints, auth, spend caps, and rate limiting unchanged.
- New dependencies: `starlette`, `uvicorn`, `httpx`.
- CI Actions bumped: `actions/checkout@v7`, `actions/setup-python@v7`, `actions/cache@v6`.

### Fixed

- File handle leak in `Fugal.__init__` (`json.load(open(...))` → `with` statement).
- Non-transient OpenRouter errors no longer waste retries (raised as `ValueError` with
  immediate re-raise instead of `RuntimeError` caught by the retry loop).
- Constructor now warns when a head model has no price entry.

### Added

- `tests/test_router_math.py` — 7 unit tests covering sigmoid, L2 normalization,
  independent rows, lambda monotonicity, and price calculation.

### Removed

- `fugal/_console.py` — `use_utf8()` consolidated into `fugal/__init__.py`.
- `data/eval_fixture.example.jsonl` — format is documented in `docs/EVALUATION.md`.
- `deploy/` — 233 lines describing one specific production host, and the only part of the
  repo nobody had executed. `setup_arm.sh` in particular cloned a URL and apt-installed as
  root. What was reusable (the unit's `ExecStart`, the spend cap, `MemoryMax`) is now a
  short appendix in `docs/INTEGRATION.md`. `python -m fugal --serve` is unchanged.

### Security

- **Browsers can no longer reach the server by default.** CORS headers used to be sent for
  whatever `Origin` arrived, and the documented way to run Fugal is on localhost with no
  token and a live OpenRouter key — so any page a user visited could spend their credit
  from the browser and read the answer back. No CORS headers are now sent unless an origin
  is explicitly allowed with `--cors-origin` / `FUGAL_CORS_ORIGINS`.
- **DNS-rebinding defence.** On a loopback bind the `Host` header must name loopback;
  anything else gets a 403. Add others with `--allow-host`. Not enforced on a non-loopback
  bind, where a reverse proxy legitimately forwards a public hostname.
- Bearer-token comparison now uses `hmac.compare_digest` instead of `==`.

### Fixed

- **`data/models_2026-06.json` prices were stale, which made spend caps wrong.** Six of
  seventeen had moved since the snapshot — `deepseek/deepseek-v4-pro` by +167%, so its real
  cost was under-counted almost 3x. `_price()` feeds `meta["cost"]`, the `X-Fugal-Cost-USD`
  header, `--spend-cap` and `--daily-cap`, so a stale sheet silently miscounts all four.
  Sheet refreshed, and `scripts/refresh_prices.py` now regenerates it in place.
  `mean_cost` in the head is unchanged and stays frozen: it is routing input, fit alongside
  `W`, not billing input.
- Docs claimed model subsetting was "bit-identical". It is identical to within one ulp
  (~1e-16): BLAS blocks a `(17, 1024)` matrix-vector product differently than a
  `(3, 1024)` one. The mathematical claim — independent rows, nothing normalised across
  models — is exact, and the resulting ranking is unchanged, which is now the property
  asserted. Found by `verify/verify_head.py` on its first run.

### Added

- `verify/verify_head.py` — head properties in pure numpy, no backbone, no network, <1s:
  the five-array contract, price coverage, non-degeneracy, `p_solve ∈ (0,1)` over random
  hidden states, subsetting exactness and rank stability, and λ monotonicity in cost.
- `verify/verify_calibration.py` + `docs/EVALUATION.md` — per-model AUC/Brier, pooled ECE
  with a reliability table, and routed accuracy-and-cost against five baselines including a
  hindsight-chosen best fixed model and a per-question oracle. Needs a graded fixture,
  which this repo does not ship; `docs/EVALUATION.md` specifies the format.
- `tests/test_adapters.py` — 45 stdlib `unittest` cases over the wire-shape adapters
  (`anthropic_to_body`, tool-call conversion in both directions, `compose_system`,
  `clamp_max_tokens`, `clean_history`). No network, no backbone, runs in milliseconds, and
  now gated in the fast CI job. These are the functions whose bugs do not raise — a dropped
  `tool_call_id` just hangs a tool loop.
- `scripts/refresh_prices.py`, with `--check` for reporting drift without writing.
- `pyproject.toml`: `pip install -e .` and a `fugal` console script.
- `SECURITY.md`, `CODE_OF_CONDUCT.md`, issue and PR templates, Dependabot for Actions.

### Changed

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

## [1.0.0]

Initial public release: the 17-model utility head, the price-discounted decision rule, the
OpenAI and Anthropic serving shapes, and the CPU-only deployment recipe.
