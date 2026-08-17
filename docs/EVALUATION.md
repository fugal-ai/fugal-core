# Checking whether the head is any good

Three things in this repo can be verified, and they are not the same thing.

| What | How | Needs |
|---|---|---|
| The artifact is well-formed and the decision rule behaves as documented | `python verify/verify_head.py` | nothing — pure numpy, <1s |
| Exactly one model is called per turn, multi-turn included | `python verify/verify_routing.py` | the backbone; mocked worker, $0 |
| **`p_solve` means what it says, and routing beats picking one model** | `python verify/verify_calibration.py --fixture ...` | **a graded fixture this repo does not ship** |

The first two pass on a fresh clone and are wired into CI. The third is the one that
matters most and the one this repo cannot answer by itself, because it needs evidence
rather than code.

## Why the evidence is not here

A row of `W` was learned from one model's measured performance across a large set of
questions. The measuring — running every candidate model over every question and grading
the answers — is the expensive part, the part that has to be trustworthy, and it lives in a
separate research repository that is not public (see the README's "What this does not
claim"). This repo ships the artifact and the code that runs it.

That is an honest split, but it leaves a gap: nothing a stranger can run tells them whether
`0.972` is a real probability or a confident-looking number. `verify_calibration.py` closes
the gap the moment you hand it a fixture — and exporting a held-out fixture reveals far less
than publishing the study.

## The fixture format

JSONL, one question per line. `//` lines and blank lines are skipped.

```json
{"id": "q-0001", "source": "gsm8k/test", "question": "What is 15% of 240?",
 "solved": {"openai/gpt-5.5": 1, "anthropic/claude-sonnet-5": 1, "qwen/qwen3.7-plus": 0}}
```

- `question` — the user turn, exactly as the router would receive it. **Standalone
  questions only.** The head was fit on those; grading follow-up turns measures something
  the router was never built to do.
- `solved` — `1` if that model's answer to that question was graded correct, `0` if not.
  A model is evaluated only if it appears on **every** row; models graded on some rows and
  not others are reported and excluded, because a per-model metric computed over a
  different subset per model is not comparable to the others.
- Any other key is ignored. `id` and `source` are recommended so a surprising number can be
  chased back to a question.

`data/eval_fixture.example.jsonl` shows the shape. **Its outcomes are placeholders, not
measurements** — the script refuses to print metrics for a fixture that small unless you
pass `--allow-tiny`, which exists for checking plumbing, never for quoting.

## Two requirements that decide whether the numbers mean anything

**Held out.** The fixture must contain questions the head was *not* fit on. Evaluating a
logistic head on its own training set reports how well it memorised, in the flattering
direction, with no warning that it is doing so.

**Same conditioning.** Hidden states must come from an unmodified `Qwen/Qwen3-0.6B` under
the same `ROUTER_SYSTEM_PROMPT`, transcript formatted as raw `role: content`, mean-pooled
across all input tokens. `verify_calibration.py` uses the shipped constants automatically,
so this only bites if you are grading against a head you fit under different conditioning —
see docs/HEAD_FORMAT.md.

## Running it

```bash
python verify/verify_calibration.py --fixture data/eval_fixture.jsonl --cache /tmp/h.npz
```

One local forward pass per question, roughly 1–2 s each on a CPU core, no API calls, `$0`.
`--cache` stores the hidden states so re-running at a different `--router-lambda` is
instant — which is the cheapest way to see how much the cost term actually costs you.

## Reading the output

**Per model — AUC and Brier.** AUC asks whether `p_solve` *ranks* correctly: given one
question this model got right and one it got wrong, how often is the right one scored
higher? 0.5 is chance. Brier asks whether the number is the right *size*. A head can rank
well and still be badly scaled, and only the second one breaks the price trade-off, because
`utility = p - λ·mean_cost` compares a probability against dollars.

**Pooled — ECE and the reliability table.** Bins every model×question pair by predicted
`p_solve` and prints the actual solve rate in each bin. This is where "does 0.9 mean 90%"
gets answered. A positive gap means the head is underconfident, negative means over.

**End to end — the table that is the actual claim.** Routed accuracy and cost against:

- **best fixed model (hindsight)** — the single model with the highest accuracy *on this
  fixture*, chosen with the answers in hand. It flatters itself and that is the point: it
  is the honest bar. A router that cannot beat one model picked in hindsight is not earning
  its forward pass.
- **cheapest model always** — the floor on cost.
- **uniformly random model** — the floor on decision quality.
- **per-question oracle** — always picking a model that got it right, if any did. The
  ceiling for *any* router on this fixture. The gap between routed and oracle is the
  headroom a better head could recover.

The script states the result plainly, including when routing loses. If routing gives up
accuracy for cost, it says so and reports the exchange rate rather than declaring victory —
whether that trade is worth taking is a judgement about your workload, not a result.
