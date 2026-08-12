# Contributing

Small repo, few rules. The ones below exist because breaking them is silent.

## Run this before you open a PR

```bash
python -m compileall -q fugal scripts verify
python verify/verify_routing.py                      # needs the backbone; no key, no spend
python -m fugal --route "what is 15% of 240?"        # $0
```

CI runs the same things plus a string-hygiene gate.

## Things that look like cleanups and are not

**`ROUTER_SYSTEM_PROMPT` and `num_agents=7` in `fugal/router.py` are part of the trained
artifact.** The head was fit on hidden states produced under that exact string, with the
transcript formatted as raw `role: content` lines rather than a chat template. `7` is not the
number of models — there are 17. Change any of it and routing quietly degrades with no error.
Same for `HIDDEN_POS = -2` and the L2 normalisation of `h`.

**The head and the backbone travel together.** `data/router_head.npz` was fit on unmodified
`Qwen/Qwen3-0.6B` hidden states. A different or fine-tuned backbone does not crash; it just
routes worse.

**Do not add a model to the head by hand.** A row of `W` is measured, not derived. See
`docs/HEAD_FORMAT.md` for what adding one actually requires.

## Scope

This repo is the router and the server around it. Things that belong elsewhere: billing,
accounts, API-key issuance, durable usage records, hosted-service configuration. If a change
needs any of those, it belongs in whatever service wraps Fugal, not here — and CI will fail
the build if those strings appear.

Evaluation harnesses and head training also live outside this repo. What ships here is the
artifact and the code that runs it.

## Docs

Every command in the README and in `docs/` must be one that was actually executed, with its
real output. Two of the three quickstart commands in an earlier version of this README could
not work on a fresh clone; that is the failure mode this rule exists to prevent.

## Licence

Apache-2.0. By contributing you agree your contribution is licensed under it.
