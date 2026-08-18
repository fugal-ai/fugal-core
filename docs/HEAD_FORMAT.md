# The router head: `data/router_head.npz`

73 KB, five arrays. This is the only trained artifact in the repo; everything else is code.

```
W          (17, 1024)  float32   one independent logistic row per model
b          (17,)       float32   per-model bias
models     (17,)       <U29      OpenRouter model ids, index-aligned to W / b / mean_cost
mean_cost  (17,)       float64   mean USD per query for that model, measured, not quoted
lam        scalar      float64   cost sensitivity; the shipped head is 2.0
```

Index `i` means the same model in all four arrays. That alignment is the whole contract.

## How it is used

```python
h = hidden_state(question)          # (1024,), mean-pooled hidden state, L2-normalised
p = 1 / (1 + exp(-(W @ h + b)))     # (17,) — P(model i solves this question)
utility = p - lam * mean_cost       # (17,)
worker = models[argmax(utility)]
```

`lam` converts dollars into probability points: at `lam=2.0`, a model must be 2 percentage
points likelier to be right to justify one extra cent per query. Override it at runtime with
`--router-lambda` or `Fugal(router_lambda=...)` — that changes the decision rule, never the
stored head.

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
3. Append the row to `W`, its bias to `b`, its id to `models`, and its measured mean cost to
   `mean_cost`. Save with the same five keys.

Point `FUGAL_HEAD` at your `.npz` to serve it. That work lives in the research repository, not
here.

## Two ways to silently break it

**The head and the backbone travel together.** These weights were fit on hidden states from an
unmodified `Qwen/Qwen3-0.6B`. Serve them on a different or fine-tuned backbone and nothing
crashes — the router just routes worse, which is the worst failure mode available.

**The prompt is part of the artifact.** The hidden state is taken with
`ROUTER_SYSTEM_PROMPT` as the system message and the question as the user message, formatted
as raw `role: content` lines rather than a chat template, and mean-pooled across all input
tokens. All of that was fixed when the head was fit. Editing the prompt or pooling strategy
in `router.py` invalidates the head without any visible error.

## How these constants were chosen

The router prompt, mean-pooling strategy, and transcript format were selected via a systematic
ablation (4 positions x 4 prompts = 16 combinations, 10 seeds x 4-fold CV each). Mean-pooling
outperformed all single-token positions (including TRINITY's original penultimate-token
position), and the clean routing prompt was statistically indistinguishable from the inherited
TRINITY dispatcher prompt (p > 0.20). The ablation data is in the research repository.

If you fit your own head, use the same prompt and pooling strategy, or rerun the ablation
under your setup.
