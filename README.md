# Fugal

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
pip install -r requirements.txt
python scripts/fetch_backbone.py            # one-time: Qwen3-0.6B (~1.5 GB) -> artifacts/
python -m fugal --route "what is 15% of 240?"
```

```
  model                             p_solve   $/query  utility
  qwen/qwen3.7-plus                   0.972   0.00439    0.964  ################### <-- WORKER
  anthropic/claude-sonnet-5           0.974   0.00784    0.958  ###################
  google/gemini-3.5-flash             0.958   0.01215    0.934  ###################
  anthropic/claude-opus-4.8           0.950   0.00929    0.931  ##################
  openai/gpt-5.5                      0.966   0.01965    0.927  ###################
  mistralai/mistral-large-2512        0.922   0.00140    0.919  ##################
  ...
```

Read the top two rows: Sonnet is marginally likelier to be right (0.974 vs 0.972) and costs
79% more, so the cheaper model wins on utility. That trade is the product.

Ask a harder question and the whole board moves down:

```bash
$ python -m fugal --route "prove that the halting problem is undecidable"
  model                             p_solve   $/query  utility
  minimax/minimax-m3                  0.646   0.00193    0.642  ############ <-- WORKER
  z-ai/glm-5.2                        0.624   0.00660    0.611  ############
  anthropic/claude-sonnet-5           0.626   0.00784    0.610  ############
```

Bare `python -m fugal --route` opens an interactive inspector: type questions, see the table,
`/quit` to leave. Still $0.

## Route among *your* models

Most people hold keys for three or four providers, not seventeen. Restrict the pool:

```bash
$ python -m fugal --models "openai/gpt-5.4-nano,deepseek/deepseek-v4-flash,meta-llama/llama-4-maverick" \
    --route "what is 15% of 240?"
  model                             p_solve   $/query  utility
  deepseek/deepseek-v4-flash          0.897   0.00026    0.896  ################# <-- WORKER
  meta-llama/llama-4-maverick         0.868   0.00030    0.867  #################
  openai/gpt-5.4-nano                 0.864   0.00055    0.863  #################
```

`FUGAL_MODELS="a,b,c"` does the same thing as an environment variable, including for the
server. **Subsetting is exact, not an approximation**: each row of `W` is an independent
logistic head, so dropping rows cannot disturb the ones that remain — the scores above are
identical to those models' scores in the full table.

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

**Always pass `--spend-cap`.** It is a hard cumulative USD ceiling; past it the server returns
402 instead of spending.

## Layout

```
fugal/router.py     the model: backbone -> hidden state -> head -> one worker call
fugal/serve.py      the CLI and the HTTP server
data/router_head.npz        the trained head: W, b, models, mean_cost, lam  (73 KB)
data/models_2026-06.json    price sheet, USD per million tokens
docs/HEAD_FORMAT.md         the .npz contract, and the limits of subsetting
docs/INTEGRATION.md         wiring Fugal into Claude Code / OpenClaw / SDKs
verify/verify_routing.py    the core promise: exactly one model call per turn
```

Two knobs worth knowing. `--router-lambda` overrides the head's cost sensitivity (shipped:
`λ=2.0`; higher trades down to cheaper models on easy questions). `FUGAL_MODEL` points at any
Qwen3-0.6B checkout if you already have one and want to skip the download.

## What this does not claim

- The head was fit on **standalone questions**. On a follow-up turn ("now in Rust") the routing
  signal is weaker than on a fresh question, and the router reads only the latest message.
- The 17 model ids and the price sheet are a **2026-06 snapshot**. Prices move; `mean_cost` is
  a fixed measurement, not a live feed.
- The evidence behind the head — the model×question matrix it was fit on, the end-to-end
  benchmarks, the dated reports — lives in a separate research repository that is **not
  public**. This repo ships the artifact and the code that runs it, not the study.
- `ROUTER_SYSTEM_PROMPT` and `num_agents=7` in `router.py` are **part of the trained artifact**,
  not style. The head was fit on hidden states produced under that exact string; `7` is not the
  number of models (there are 17). Changing either silently invalidates the head.

## Lineage

- Derived from the open reverse-engineering of Sakana AI's Fugu orchestrator started at
  [trotsky1997/OpenFugu](https://github.com/trotsky1997/OpenFugu), whose TRINITY reconstruction
  is where the router's design comes from. See `NOTICE`. Fugal is independent of, and not
  affiliated with, Sakana AI.
- What is added here: the utility head over 17 current models, the price-discounted decision
  rule, and the OpenAI/Anthropic serving shapes.
- Fugal no longer depends on any TRINITY artifact at runtime — it runs on an unmodified
  Qwen3-0.6B backbone with its own head.

## License

Apache-2.0 (see `LICENSE`). The backbone (Qwen3-0.6B, Apache-2.0) is fetched at setup time and
never redistributed here — see `NOTICE`. The router head is original work and ships with the
repo under the same licence.
