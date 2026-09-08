# The router head: `data/router_head.npz`

73 KB. This is the only trained artifact in the repo; everything else is code. The design
principle: **everything the routing decision conditions on is declared inside this file**
— the code reads the declarations and behaves accordingly, so upgrading routing behaviour
means shipping a new head, never editing code.

```
W                (17, 1024)  float32   one independent logistic row per model
b                (17,)       float32   per-model bias
models           (17,)       <U29      OpenRouter model ids, index-aligned to every array
lam              scalar      float64   cost sensitivity; the shipped head is 1.0

# v2 heads (current format)
mean_in_tokens   (17,)       float64   mean input tokens per query, measured at fit time
mean_out_tokens  (17,)       float64   mean output tokens per query, measured at fit time
context          scalar      str       "standalone" (default when absent) or "multiturn"

# optional provenance (any head)
backbone_revision scalar     str       HuggingFace commit of the Qwen3-0.6B it was fit on
provenance       scalar      str       free text: fit date, training run, code commit

# v1 heads (the currently shipped head; supported, the server banner says so)
mean_cost        (17,)       float64   mean USD per query, measured at fit-time PRICES
```

Index `i` means the same model in every array. That alignment is the whole contract.

## How it is used

```python
h = hidden_state(question)          # (1024,), mean-pooled hidden state, L2-normalised
p = 1 / (1 + exp(-(W @ h + b)))     # (17,) — P(model i solves this question)
utility = p - lam * mean_cost       # (17,)
worker = models[argmax(utility)]
```

For a **v2 head**, `mean_cost` is computed at load time from the token measurements and the
CURRENT price sheet: `mean_cost[i] = mean_in_tokens[i]·price_in[i] + mean_out_tokens[i]·price_out[i]`.
That split is deliberate: how many tokens a query averages is a *measurement*, frozen with
the head, but what those tokens cost is a *market fact* that moves — so
`scripts/refresh_prices.py` keeps the routing trade-off itself honest. A v1 head bakes the
whole `mean_cost` in, which freezes fit-time prices into every routing decision; it still
works; the server banner says which format it loaded.

`context` declares what transcript distribution the head was fit on. The router feeds
conversation history into the forward pass **only** for a `"multiturn"` head; a
`"standalone"` head (the shipped one) routes on the latest user turn alone, because hidden
states from a distribution the head never saw make it route worse, silently.

`lam` converts dollars into probability points: at `lam=1.0`, a model must be 1 percentage
point likelier to be right to justify one extra cent per query. Override it at runtime with
`--router-lambda`, `FUGAL_LAMBDA`, or `Fugal(router_lambda=...)` — that changes the decision
rule, never the stored head.

Inspect the shipped one:

```bash
python - <<'PY'
import numpy as np
z = np.load("data/router_head.npz")
for k in z.files:
    print(f"{k:10} {str(z[k].shape):12} {z[k].dtype}")
print(float(z["lam"]), [str(m) for m in z["models"]][:3])
PY
```

## Subsetting is free and exact

Every row of `W` is a separate logistic regression. Nothing is normalised across models — no
softmax, no shared denominator — so removing rows cannot change the scores of the rows that
remain:

```bash
python -m fugal --models "openai/gpt-5.5,deepseek/deepseek-v4-pro" --route "..."
FUGAL_MODELS="openai/gpt-5.5,deepseek/deepseek-v4-pro" python -m fugal --serve
```

The `p_solve` printed for a model in a two-model pool is the same one it gets in the full
seventeen, and the ranking among the survivors is unchanged. Use this when you hold keys for
some providers and not others. Fugal validates the list against `models` and refuses unknown
ids rather than silently ignoring them.

One caveat, so nobody is surprised by it: the two scores can differ in the final bit
(~1e-16). BLAS picks a different blocking for a `(17, 1024)` matrix-vector product than for a
`(2, 1024)` one, so the same dot product accumulates in a different order. That is arithmetic,
not the models influencing each other — it is thirteen orders of magnitude below the smallest
utility gap in the shipped head, and `verify/verify_head.py` asserts both the bound and the
rank stability over 200 random subsets.

## Adding a model is a different thing entirely

You cannot append a row. A row of `W` is what was learned from that model's measured
performance across a large set of questions; there is no way to derive it from the model's
name, its price, or the other rows. Nothing in this repo can produce one, and inventing one
would produce confident routing that means nothing.

What it actually takes:

1. Run the candidate model over the same question set the existing head was fit on, and grade
   each answer. That grading is the expensive part and the part that has to be trustworthy.
2. Fit a logistic head on `(hidden_state(question) -> solved?)` for that model, using hidden
   states from the **same backbone under the same `ROUTER_SYSTEM_PROMPT`** (see the warning
   below).
3. Append the row to `W`, its bias to `b`, its id to `models`, and its measured mean token
   counts to `mean_in_tokens` / `mean_out_tokens` (v2; for a v1 head, its measured mean cost
   to `mean_cost`). Save with the same keys, and make sure the model has a row in the price
   sheet — an unpriced model is refused at startup.

Point `FUGAL_HEAD` at your `.npz` to serve it. That work lives in the research repository, not
here.

`backbone_revision` is what `scripts/fetch_backbone.py` downloads when the head declares
it, so a head can bring its own backbone pin. `provenance` is printed by the server banner
so a routing-quality report can say which head it was. Both are read with pickle disabled,
like every other array: a head from a source you do not control cannot run code on load.

## Shipping a new head

Heads are trained outside this repo (the subnet), and this is the procedure for landing one
here. Every step is mechanical except the last, which is the only one that tells you whether
the head is any good.

1. **Save it in the v2 format** with `context`, `backbone_revision` and `provenance` set. A
   v1 head works, but freezes fit-time prices into every routing decision until replaced.
2. **Drop it in** as `data/router_head.npz` (or point `FUGAL_HEAD` at it to try first).
3. **Re-sync the price sheet**: `python scripts/refresh_prices.py`. The sheet must cover
   every model the head scores or the server refuses to start; the model list is taken
   from the head, so a head that adds or drops a model rewrites the sheet accordingly.
4. **Check the artifact**: `python verify/verify_head.py` (CI runs this on every push, and
   fails on a head the sheet does not cover).
5. **Check the backbone pin**: if `backbone_revision` changed, `python
   scripts/fetch_backbone.py` again — the CI cache key includes it.
6. **Measure it**: `python verify/verify_calibration.py --fixture <held-out graded rows>`
   and put the end-to-end table in the PR. The subnet grades model×question outcomes, so a
   held-out fixture (`docs/EVALUATION.md`) is a by-product of training, not extra work.
   Every documented example table (README, `docs/INTEGRATION.md`) must be regenerated
   from real runs of the new head; `CONTRIBUTING.md` says why.

For heads published more often than this repo releases, publish the `.npz` as a versioned
release asset and point `FUGAL_HEAD` at it. Rolling back is pointing at the previous one.

## Two ways to silently break it

**The head and the backbone travel together.** These weights were fit on hidden states from an
unmodified `Qwen/Qwen3-0.6B` at one HuggingFace revision, which `scripts/fetch_backbone.py`
pins. Serve them on a different revision or a fine-tuned backbone and nothing crashes — the
router just routes worse, which is the worst failure mode available.

**The prompt is part of the artifact.** The hidden state is taken with
`ROUTER_SYSTEM_PROMPT` as the system message and the question as the user message, formatted
as raw `role: content` lines rather than a chat template, and mean-pooled across all input
tokens. All of that was fixed when the head was fit. Editing the prompt or pooling strategy
in `router.py` invalidates the head without any visible error.

## How these constants were chosen

The router prompt, mean-pooling strategy, and transcript format were selected via a systematic
ablation (4 positions x 4 prompts = 16 combinations, 10 seeds x 4-fold CV each). Mean-pooling
outperformed all single-token positions (including the penultimate-token position used in prior
work), and the clean routing prompt was statistically indistinguishable from alternatives
(p > 0.20). The ablation data is in the research repository.

If you fit your own head, use the same prompt and pooling strategy, or rerun the ablation
under your setup.

## Future direction

Per-model mean token counts are still a per-model *constant* — a one-line question and a
"write me 3,000 lines" request pay the same routing cost penalty. The theoretically right
quantity is `E[cost | query, model]`; the input half is already known at routing time (the
query's own token count), and the output half could become a per-query prediction. Both are
head-format extensions: when they land they will be new declared arrays here, not code
changes.

## Explicit success contract

New `fugal-success-v1` heads have a distinct, strictly validated artifact and
embedding profile. See [SUCCESS_CONTRACT.md](SUCCESS_CONTRACT.md). Existing v1/v2
heads keep the behavior documented above; they are never reinterpreted.
