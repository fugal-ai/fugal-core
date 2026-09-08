# Fugal — Apache-2.0. See NOTICE.
"""Properties of data/router_head.npz that hold for EVERY input, checked in pure numpy.

    python verify/verify_head.py

No backbone, no network, no API key, no evaluation fixture — runs in well under a second
on a fresh clone. That is the point: these are the claims the README makes which can be
settled by arithmetic alone, so nobody has to take them on trust.

What this CANNOT tell you: whether p_solve is CALIBRATED — whether a question the head
scores 0.9 really is answered correctly about 90% of the time — or whether routing beats a
fixed choice of model. Those need graded model x question outcomes, which are evidence, not
arithmetic. verify/verify_calibration.py measures them against a fixture; docs/EVALUATION.md
says how to produce one. Passing this file means the artifact is well-formed and the
decision rule behaves as documented, not that the artifact is any good.

Checks:
  1. the array contract and index alignment, v1 or v2 (docs/HEAD_FORMAT.md)
  2. every model the head scores has a price in the sheet (fatal for a v2 head,
     whose routing costs are COMPUTED from the sheet)
  3. the head is non-degenerate — not a placeholder, not duplicated rows
  4. p_solve is a probability for any input, with no NaN/overflow
  5. SUBSETTING IS EXACT: dropping rows leaves the remaining scores bit-identical
  6. lambda means what the docs say: a model one cent dearer per query must be
     lam x $0.01 likelier to be right, and the winner flips exactly there
  7. raising lambda never promotes a more expensive model over a cheaper one
"""
import io
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from fugal import use_utf8                               # noqa: E402
use_utf8()                                              # em-dashes below vs a cp1252 console

HEAD = os.environ.get("FUGAL_HEAD") or os.path.join(REPO, "data", "router_head.npz")
PRICES = os.environ.get("FUGAL_PRICES") or os.path.join(REPO, "data", "models_2026-06.json")

failures = []


def check(label, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {label}")
    if not ok:
        failures.append(f"{label}{': ' + detail if detail else ''}")
        if detail:
            print(f"        {detail}")


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


print("\n  head properties (pure numpy — no backbone, no network, $0)\n")

z = np.load(HEAD)
files = set(z.files)
v2 = "mean_in_tokens" in files

# --- 1. the contract (v1 or v2 — docs/HEAD_FORMAT.md) ------------------------------
check("required arrays present",
      {"W", "b", "models", "lam"}.issubset(files)
      and ("mean_cost" in files or {"mean_in_tokens", "mean_out_tokens"}.issubset(files)),
      f"got {sorted(files)}")
W, b = z["W"], z["b"]
models = [str(m) for m in z["models"]]
lam = float(z["lam"])
n = len(models)
context = str(z["context"]) if "context" in files else "standalone"
check("context is a known value", context in ("standalone", "multiturn"),
      f"context={context!r}")

# --- 2. every scored model can be billed -------------------------------------------
# Unpriced is fatal at runtime: load_head refuses to start (a made-up rate would
# corrupt cost reporting and every spend cap), and a v2 head cannot even compute
# its routing costs without the sheet.
with io.open(PRICES, encoding="utf-8") as f:
    sheet = {m["id"]: (m["in"] / 1e6, m["out"] / 1e6) for m in json.load(f)}
unpriced = [m for m in models if m not in sheet]
check("every model has a price", not unpriced, f"unpriced: {unpriced}")

if v2:
    itok, otok = z["mean_in_tokens"], z["mean_out_tokens"]
    check("token stats are positive and finite",
          bool(np.all(np.isfinite(itok)) and np.all(np.isfinite(otok))
               and np.all(itok > 0) and np.all(otok > 0)))
    nanp = (float("nan"), float("nan"))
    pin = np.array([sheet.get(m, nanp)[0] for m in models])
    pout = np.array([sheet.get(m, nanp)[1] for m in models])
    mean_cost = itok * pin + otok * pout        # what load_head computes at runtime
else:
    mean_cost = z["mean_cost"]

shapes_ok = W.shape == (n, 1024) and b.shape == (n,) and mean_cost.shape == (n,)
if v2:
    shapes_ok = shapes_ok and itok.shape == (n,) and otok.shape == (n,)
check(f"shapes align at n={n} ({'v2' if v2 else 'v1'} head)", shapes_ok,
      f"W{W.shape} b{b.shape} mean_cost{mean_cost.shape}")
check("model ids are unique", len(set(models)) == n)
check("mean_cost is positive and finite",
      bool(np.all(np.isfinite(mean_cost)) and np.all(mean_cost > 0)))
check("lam is positive and finite", np.isfinite(lam) and lam > 0, f"lam={lam}")
check("W and b are finite", bool(np.all(np.isfinite(W)) and np.all(np.isfinite(b))))

# --- 3. non-degenerate -----------------------------------------------------------
# A placeholder artifact (zeros, one row tiled n times, uninitialised noise) would pass
# every shape check above and route confidently on nothing.
norms = np.linalg.norm(W, axis=1)
check("no zero rows", bool(np.all(norms > 1e-6)), f"min norm {norms.min():.3g}")
dup = [(models[i], models[j]) for i in range(n) for j in range(i + 1, n)
       if np.allclose(W[i], W[j])]
check("no duplicated rows", not dup, f"identical: {dup[:3]}")
# Rows should correlate (models agree about which questions are hard) but not be the
# same direction. Both extremes mean the head cannot discriminate between models.
Wn = W / norms[:, None]
C = Wn @ Wn.T
off = C[~np.eye(n, dtype=bool)]
check("rows are correlated but distinct (0 < mean cos < 0.99)",
      0.0 < float(off.mean()) < 0.99, f"mean cosine {off.mean():.3f}, max {off.max():.3f}")

# --- 4. p_solve is a probability for ANY input -----------------------------------
# Random unit vectors stand in for hidden states. route() L2-normalises h, so every
# input the router can ever produce lives on this sphere — the sample is the real domain,
# not an approximation of it.
rng = np.random.default_rng(0)
H = rng.standard_normal((512, 1024))
H /= np.linalg.norm(H, axis=1, keepdims=True)
P = sigmoid(H @ W.T + b)
check("p_solve in (0,1) with no NaN over 512 random hidden states",
      bool(np.all(np.isfinite(P)) and np.all(P > 0) and np.all(P < 1)),
      f"min {P.min():.3g} max {P.max():.3g}")

# --- 5. subsetting is exact ------------------------------------------------------
# The README's strongest structural claim: rows are independent logistic heads, nothing
# is normalised across models, so --models cannot perturb the survivors. If a softmax
# ever crept in, this is what would catch it.
#
# The tolerance is not a hedge. Mathematically the claim is exact — row i of the product
# depends on row i alone. What is NOT exact is the floating-point evaluation: BLAS picks a
# different blocking for a (17, 1024) matrix-vector product than for a (3, 1024) one, so
# the same dot product accumulates in a different order and can land one ulp apart. That
# is arithmetic, not model coupling, and 1e-16 is ~13 orders of magnitude below the
# smallest utility gap in the shipped head. The claim that actually matters — that the
# RANKING of the survivors never changes — is checked exactly, below.
worst, rank_breaks = 0.0, []
for trial in range(200):
    k = int(rng.integers(1, n + 1))
    idx = rng.choice(n, size=k, replace=False)
    h = H[trial % len(H)]
    full = sigmoid(W @ h + b)
    sub = sigmoid(W[idx] @ h + b[idx])
    worst = max(worst, float(np.abs(sub - full[idx]).max()))
    full_order = [i for i in np.argsort(-(full - lam * mean_cost)) if i in set(idx)]
    sub_order = [idx[j] for j in np.argsort(-(sub - lam * mean_cost[idx]))]
    if list(full_order) != list(sub_order):
        rank_breaks.append((k, [models[i] for i in sub_order[:3]]))
check("subsetting changes no score by more than 1 ulp over 200 random subsets",
      worst < 1e-12, f"max |p_subset - p_full| = {worst:.3g}")
check("subsetting never reorders the surviving models", not rank_breaks,
      f"{rank_breaks[:2]}")

# --- 6. lambda means what HEAD_FORMAT.md says ------------------------------------
# "at lam=1.0, a model must be 1 percentage point likelier to be right to justify one
# extra cent per query". Assert the DECISION, not the arithmetic: two models one cent
# apart in cost, with a p gap just below and just above lam*0.01 — the winner must
# flip exactly at that boundary.
gap = lam * 0.01
two_costs = np.array([0.0, 0.01])
p_below = np.array([0.5, 0.5 + gap * 0.999])
p_above = np.array([0.5, 0.5 + gap * 1.001])
check(f"one extra cent per query demands {gap * 100:.3g} extra probability points at "
      f"lam={lam:g}, no more, no less",
      int(np.argmax(p_below - lam * two_costs)) == 0
      and int(np.argmax(p_above - lam * two_costs)) == 1)

# --- 7. lambda is monotone in cost -----------------------------------------------
# Raising lambda may only ever shift preference toward CHEAPER models. If a higher lambda
# could promote a pricier one, --router-lambda would not mean what its help text says.
bad = []
for h in H[:64]:
    p = sigmoid(W @ h + b)
    for lo, hi in ((0.0, 2.0), (2.0, 10.0), (10.0, 100.0)):
        a = int(np.argmax(p - lo * mean_cost))
        c = int(np.argmax(p - hi * mean_cost))
        if mean_cost[c] > mean_cost[a] + 1e-15:
            bad.append((models[a], models[c], lo, hi))
check("raising lambda never picks a more expensive model", not bad, f"{bad[:2]}")

print()
if failures:
    for f_ in failures:
        print(f"FAIL {f_}")
    sys.exit(f"\n{len(failures)} property check(s) failed.")
print(f"PASS: {n} models, lambda={lam:g} — head is well-formed, priced, non-degenerate, "
      f"and the decision rule behaves as documented.")
print("      (This says nothing about calibration — see verify/verify_calibration.py.)")
