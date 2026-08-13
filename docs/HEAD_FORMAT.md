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
h = hidden_state(question)          # (1024,), penultimate-token hidden state, L2-normalised
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
z = np.load("data/router_head.npz", allow_pickle=True)
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
`ROUTER_SYSTEM_PROMPT.format(num_agents=7)` as the system message and the question as the user
message, formatted as raw `role: content` lines rather than a chat template. All of that was
fixed when the head was fit. `num_agents=7` does not mean "7 models" — there are 17. Editing
either constant in `router.py` invalidates the head without any visible error.

## Where the constants came from, and what that does not prove

`HIDDEN_POS = -2`, the router prompt string, `num_agents=7` and the raw `role: content`
transcript format all come from OpenFugu's reconstruction of the TRINITY coordinator (see
`NOTICE`), where they were established against the released TRINITY checkpoint — which
adapts the backbone with SVF and reads a bias-free 10-row agent/role logit head.

**Fugal does neither.** It runs an unmodified Qwen3-0.6B and a 17-row logistic head. So the
reason to keep these constants is not that they were shown optimal for this setup — it is
the narrower and sufficient one that *this head was fit under them*, so changing them puts
the head and its conditioning out of step. Nothing in this repo demonstrates that `-2` beats
`-1` or mean-pooling, or that this prompt beats a different one, for a head fit without SVF.
Establishing that would mean refitting under each variant and comparing on held-out graded
outcomes — see `docs/EVALUATION.md`.

Stated plainly because the distinction matters if you fit your own head: these are fixed
constants, not tuned ones.
