# Fugal

[![CI](https://github.com/jtdoherty/fugal-core/actions/workflows/ci.yml/badge.svg)](https://github.com/jtdoherty/fugal-core/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)

**One forward pass picks the model. Then it calls that one model, once.**

A 0.6B model reads your question and predicts, for each of 17 frontier models, the probability
that model answers it correctly. Discount those odds by price, take the best one, call it. The
router generates nothing and never asks a second model to check the first — a routing decision
costs one forward pass on a CPU core and zero API credit.

```python
h = hidden_state(question)          # one forward pass of Qwen3-0.6B, ~1-2s on a CPU core
p = sigmoid(W @ h + b)              # P(each of 17 models solves this)
utility = p - lam * mean_cost       # odds, discounted by price
worker = argmax(utility)            # call this one, once
```

That is the whole system. `W` is `(17, 1024)` and lives in `data/router_head.npz` — **73 KB**.
Each row is an independent logistic head for one model. It is the only novel artifact here.

## See it route, for free

No API key, no account, nothing to sign up for. The router runs locally.

```bash
pip install -e .                            # adds a `fugal` command; python -m fugal works too
python scripts/fetch_backbone.py            # one-time: Qwen3-0.6B (~1.5 GB) -> artifacts/
python -m fugal --route "what is 15% of 240?"
```

(The router never needs a GPU. On Linux, `pip install torch --index-url
https://download.pytorch.org/whl/cpu` *before* `pip install -e .` keeps you on the CPU wheel.)

```
  model                             p_solve   $/query  utility
  deepseek/deepseek-v4-flash          0.985   0.00026    0.985  ################### <-- WORKER
  deepseek/deepseek-v4-pro            0.985   0.00154    0.983  ###################
  meta-llama/llama-4-maverick         0.983   0.00030    0.983  ###################
  moonshotai/kimi-k2.6                0.991   0.00956    0.981  ###################
  openai/gpt-5.4-nano                 0.981   0.00055    0.981  ###################
  ...
  google/gemini-3.5-flash             0.981   0.01215    0.969  ###################
  google/gemini-3.1-pro-preview       0.994   0.02537    0.968  ###################
```

Read the first and last rows: Gemini Pro is likelier to be right (0.994 vs 0.985) and costs
a hundred times more per query, so the cheaper model wins on utility. That trade is the
product. A harder question lowers every model's odds and widens the gaps between them:

```bash
$ python -m fugal --route "prove that the halting problem is undecidable"
  model                             p_solve   $/query  utility
  deepseek/deepseek-v4-flash          0.979   0.00026    0.979  ################### <-- WORKER
  deepseek/deepseek-v4-pro            0.978   0.00154    0.976  ###################
  moonshotai/kimi-k2.6                0.986   0.00956    0.976  ###################
```

With the shipped head that move is small — see "What this does not claim" below.

Bare `python -m fugal --route` opens an interactive inspector: type questions, see the table,
`/quit` to leave. Still $0.

## Route among *your* models

Most people hold keys for three or four providers, not seventeen. Restrict the pool:

```bash
$ python -m fugal --models "openai/gpt-5.4-nano,deepseek/deepseek-v4-flash,meta-llama/llama-4-maverick" \
    --route "what is 15% of 240?"
  model                             p_solve   $/query  utility
  deepseek/deepseek-v4-flash          0.985   0.00026    0.985  ################### <-- WORKER
  meta-llama/llama-4-maverick         0.983   0.00030    0.983  ###################
  openai/gpt-5.4-nano                 0.981   0.00055    0.981  ###################
```

`FUGAL_MODELS="a,b,c"` does the same thing as an environment variable, including for the
server. **Subsetting is exact, not an approximation**: each row of `W` is an independent
logistic head, so dropping rows cannot disturb the ones that remain — the scores above are
the same ones those models get in the full table, and the ranking among them is unchanged.
(`verify/verify_head.py` asserts both over random subsets. Scores can differ in the last
bit — BLAS blocks a 17-row matrix-vector product differently from a 3-row one — which is
floating-point arithmetic, not the models interacting.)

The reverse does not hold. You cannot add a model the head was never fit on; a `p_solve` row
has to be measured, not invented. `docs/HEAD_FORMAT.md` gives the `.npz` contract and says
exactly what it would take.

## Actually answer the question

This is the part that spends money. Set an OpenRouter key first:

```bash
export FUGAL_API_KEY=sk-or-...
python -m fugal --query "what is 15% of 240?"
```

It prints the routing decision, calls that one model, and prints the answer with what it cost.

## Serve it to your existing tools

```bash
python -m fugal --serve --port 8090 --spend-cap 5.00
```

One process, two wire shapes: `POST /v1/chat/completions` (OpenAI) and `POST /v1/messages`
(Anthropic), both with streaming and tool calling, plus `POST /v1/route` which is always $0 and
needs no key. Claude Code and OpenClaw both work against it today —
[`docs/INTEGRATION.md`](docs/INTEGRATION.md) has the exact configuration, and an honest list of
what does not work.

**Always pass `--spend-cap`.** It is a cumulative USD ceiling; past it the server returns
402 instead of spending. It is checked before each call, so concurrent requests can overshoot
slightly, and it resets on restart — a brake, not an accounting record. Set a hard limit on
the OpenRouter key too; that is the only one that survives a bug in this code.

Browsers are locked out by default: no CORS headers unless you name an origin with
`--cors-origin`, and on a loopback bind the `Host` header must name loopback. Otherwise any
page you happened to be visiting could spend your credit from your own machine. See
[`SECURITY.md`](SECURITY.md).

## Layout

```
fugal/router.py     the model: backbone -> hidden state -> head -> one worker call
fugal/serve.py      the wire-shape adapters, the HTTP server (build_app) and the CLI
data/router_head.npz        the trained head: W, b, models, costs, lam  (73 KB)
data/models_2026-06.json    price sheet, USD per million tokens, plus max output tokens
docs/HEAD_FORMAT.md         the .npz contract, and the limits of subsetting
docs/INTEGRATION.md         wiring Fugal into Claude Code / OpenClaw / SDKs
docs/EVALUATION.md          how to check the head is any good, and what that needs
verify/verify_routing.py    the core promise: exactly one model call per turn
verify/verify_head.py       head properties, pure numpy, no backbone, <1s
verify/verify_calibration.py  is p_solve true? (needs a graded fixture — see EVALUATION.md)
tests/test_adapters.py      the OpenAI/Anthropic wire-shape conversions
tests/test_router_math.py   the decision rule and head loading, on a synthetic head
tests/test_server.py        every endpoint, tier, stream shape and browser defence
scripts/fetch_backbone.py   one-time download of Qwen3-0.6B
scripts/refresh_prices.py   re-sync the price sheet with OpenRouter
```

Everything above except the head and the price sheet is checkable on a fresh clone:

```bash
python -m unittest discover -s tests    # <1s, no backbone, no network
python verify/verify_head.py            # <1s, no backbone, no network
python verify/verify_routing.py         # needs the backbone; mocked worker, $0
```

Three things ever need updating, and everything else is code: the **head** (retrain as the
model landscape changes; forks drop in their own via `data/router_head.npz` or `FUGAL_HEAD`),
**λ** (`--router-lambda` / `FUGAL_LAMBDA` overrides the head's default of 1.0; higher
trades down to cheaper models on easy questions), and the **price sheet**
(`scripts/refresh_prices.py` re-syncs it from OpenRouter). `FUGAL_MODEL` points at any
Qwen3-0.6B checkout if you already have one and want to skip the download.

## What this does not claim

- The head was fit on **standalone questions**, and it declares that (`context` in the
  artifact — see `docs/HEAD_FORMAT.md`), so the router reads only the latest *user* turn:
  never a tool result, and no history, because hidden states from a distribution the head
  never saw make it route worse, silently. On a follow-up turn ("now in Rust") the routing
  signal is therefore weaker than on a fresh question. A head retrained on multi-turn
  transcripts flips this by declaring `context="multiturn"` — no code change.
- The 17 model ids are a **2026-06 snapshot**, and the shipped head is the **v1 format**: its
  `mean_cost` bakes in fit-time prices, so routing trades against those until the head is
  regenerated. The v2 format fixes this by storing measured token counts instead and
  computing routing cost from the current sheet at load (`docs/HEAD_FORMAT.md`).
  `data/models_2026-06.json` is *billing* input either way: it is what `meta.cost`,
  `X-Fugal-Cost-USD` and the spend caps are computed from, so it should be current. Run
  `python scripts/refresh_prices.py` (or `--check`) to keep it that way; a stale sheet means
  your cap is counting the wrong dollars.
- **On the shipped head, `p_solve` is high and nearly flat.** Across twelve deliberately
  varied test questions (a greeting, arithmetic, a translation, a Byzantine-consensus
  design, nonsense) every model scored between 0.91 and 0.997, and each model's score moved
  by about 0.01 from question to question. At `λ=1.0` the price term is usually larger than
  those gaps, so the cheapest capable models win most queries and the pricier ones win only
  when the head sees a real difference. Whether that reflects the models or a weak head is
  exactly what `verify/verify_calibration.py` measures — and it needs the graded fixture
  described next.
- **`ROUTER_SYSTEM_PROMPT` and mean-pooling are part of the trained artifact.** The head was
  fit on hidden states produced under that exact prompt, mean-pooled across all input tokens.
  Changing either silently invalidates the head. Both were chosen via a systematic ablation
  (16 combinations, 40 CV folds each) — see `docs/HEAD_FORMAT.md`.
- The evidence behind the head — the model×question matrix it was fit on, the end-to-end
  benchmarks, the dated reports — lives in a separate research repository that is **not
  public**. This repo ships the artifact and the code that runs it, not the study. So
  nothing here proves `p_solve` is *calibrated*, only that it is well-formed
  (`verify/verify_head.py`) and that exactly one model is called
  (`verify/verify_routing.py`). [`docs/EVALUATION.md`](docs/EVALUATION.md) specifies a
  held-out fixture format; hand one to `verify/verify_calibration.py` and it will report
  AUC, calibration error, and routed accuracy-and-cost against a hindsight-chosen best
  fixed model, a cheapest-always baseline, and a per-question oracle.

## License

Apache-2.0 (see `LICENSE`). The hidden-state extraction path derives from Apache-2.0 work
credited in `NOTICE`. The backbone (Qwen3-0.6B, Apache-2.0) is fetched at setup time and
never redistributed here. The router head is original work and ships with the repo under
the same licence.
