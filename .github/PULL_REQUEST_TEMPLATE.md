<!-- Keep this short. Delete anything that doesn't apply. -->

## What and why

<!-- What changes, and what problem it solves. -->

## Checks run

```
python -m unittest discover -s tests    # milliseconds, no backbone
python verify/verify_head.py            # <1s, no backbone
python verify/verify_routing.py         # needs the backbone; no key, no spend
```

- [ ] The above pass
- [ ] Every command I added or changed in the README / `docs/` is one I actually ran, with
      its real output pasted (see CONTRIBUTING.md)

## Things that look like cleanups and are not

- [ ] I did **not** change `ROUTER_SYSTEM_PROMPT`, the mean-pooling in `hidden()`, the raw
      `role: content` transcript format, or the L2 normalisation of `h`. These are part of
      the trained artifact — changing any of them degrades routing silently, with no error.
      (If the change is deliberate, say so and explain how the head was refit.)
- [ ] I did **not** hand-edit `data/router_head.npz`. A row of `W` is measured, not derived
      (`docs/HEAD_FORMAT.md`).
- [ ] Nothing here belongs to the private half of the codebase — billing, accounts, API-key
      issuance, durable usage records. CI fails the build on those strings.

## Cost and safety

- [ ] Still exactly one worker call per turn (`verify/verify_routing.py` covers this)
- [ ] No change to spend-cap, rate-limit, auth, CORS or Host-check behaviour — or if there
      is, it is described above and `SECURITY.md` is updated
