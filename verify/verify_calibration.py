# Fugal — Apache-2.0. See NOTICE.
"""Does the head's p_solve mean anything? Measure it against graded outcomes.

    python verify/verify_calibration.py --fixture data/eval_fixture.jsonl

This is the file that answers the only question a stranger should really care about:
`p_solve = 0.972` is printed with three decimal places, but is it TRUE? Nothing else in
this repo can tell you. verify/verify_head.py proves the artifact is well-formed;
verify/verify_routing.py proves exactly one model is called. Neither is evidence that the
router picks well, and a well-formed head full of noise would pass both.

It needs a fixture this repo does not ship — graded model x question outcomes — because
that is measurement, not code. docs/EVALUATION.md specifies the format and how to produce
one. If you are evaluating a head you did not fit, the fixture must be HELD OUT from
whatever it was fit on, or every number below is meaningless in the flattering direction.

Reports:
  per model   AUC, Brier score, base rate — is p_solve ranked correctly and sized correctly
  pooled      expected calibration error + a reliability table — does 0.9 mean 90%
  end-to-end  routed accuracy and cost against five baselines, which is the actual claim

Cost: one local forward pass per question (~1-2 s on a CPU core), no API calls, $0.
Pass --cache to save hidden states so re-runs with a different --router-lambda are instant.
"""
from __future__ import annotations
import argparse, io, json, os, sys, time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ.pop("FUGAL_API_KEY", None)          # nothing here may make a paid call
os.environ.pop("OPENROUTER_API_KEY", None)

from fugal import use_utf8                      # noqa: E402

MIN_ROWS = 50          # below this, sampling noise swamps every number printed


def auc(scores, labels):
    """ROC AUC via the Mann-Whitney U identity, with ties at half credit.

    Hand-rolled so this file runs on the repo's stated dependencies — pulling in
    scikit-learn to compute one rank statistic would be a worse trade for a check
    people are meant to run on a fresh clone."""
    pos, neg = labels == 1, labels == 0
    npos, nneg = int(pos.sum()), int(neg.sum())
    if npos == 0 or nneg == 0:
        return float("nan")            # undefined: the model solved all or none of them
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), float)
    ranks[order] = np.arange(1, len(scores) + 1)
    s = np.sort(scores)                # average ranks within tie groups
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2
        i = j + 1
    return (ranks[pos].sum() - npos * (npos + 1) / 2) / (npos * nneg)


def ece(p, y, bins=10):
    """Expected calibration error + the reliability table it is computed from."""
    edges = np.linspace(0, 1, bins + 1)
    rows, total, err = [], len(p), 0.0
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        m = (p >= lo) & (p < hi if hi < 1.0 else p <= hi)
        if not m.any():
            continue
        conf, acc, k = float(p[m].mean()), float(y[m].mean()), int(m.sum())
        err += (k / total) * abs(conf - acc)
        rows.append((lo, hi, k, conf, acc))
    return err, rows


def load_fixture(path):
    rows = []
    with io.open(path, encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            try:
                r = json.loads(line)
                q, solved = r["question"], r["solved"]
                assert isinstance(q, str) and q.strip() and isinstance(solved, dict)
            except Exception as e:
                sys.exit(f"{path}:{ln}: bad fixture row ({e}). See docs/EVALUATION.md.")
            rows.append((q, {k: int(bool(v)) for k, v in solved.items()}))
    return rows


def main():
    ap = argparse.ArgumentParser(prog="python verify/verify_calibration.py")
    ap.add_argument("--fixture", default=os.path.join(REPO, "data", "eval_fixture.jsonl"),
                    help="JSONL of graded outcomes; see docs/EVALUATION.md")
    ap.add_argument("--router-lambda", type=float, default=None,
                    help="evaluate a different cost sensitivity than the head's")
    ap.add_argument("--cache", metavar="NPZ", default=None,
                    help="save/reuse hidden states so re-runs skip the forward passes")
    ap.add_argument("--bins", type=int, default=10)
    ap.add_argument("--allow-tiny", action="store_true",
                    help=f"run on fewer than {MIN_ROWS} questions anyway (numbers will be "
                         f"noise; never quote them)")
    args = ap.parse_args()
    use_utf8()

    if not os.path.exists(args.fixture):
        sys.exit(f"no fixture at {args.fixture}\n"
                 f"  This repo ships the router, not the study behind it. "
                 f"docs/EVALUATION.md\n"
                 f"  specifies the fixture format and how to produce one.")

    rows = load_fixture(args.fixture)
    if not rows:
        sys.exit("fixture is empty")
    if len(rows) < MIN_ROWS and not args.allow_tiny:
        sys.exit(f"fixture has {len(rows)} rows; at least {MIN_ROWS} are needed for any of "
                 f"these numbers to mean anything.\n  Pass --allow-tiny to run anyway (for "
                 f"checking the plumbing, never for quoting).")

    from fugal.router import Fugal, ROUTER_SYSTEM_PROMPT
    from fugal import success_contract as success
    f2 = Fugal(router_lambda=args.router_lambda)
    is_success = f2.head_format == success.CONTRACT
    # Same conditioning the live router uses. Reproduced here rather than calling route()
    # because we need the hidden state itself, not the ranking it produces.
    router_sys = ROUTER_SYSTEM_PROMPT
    models, mean_cost, lam = f2.models, f2.mean_cost, f2.lam

    graded = [m for m in models if all(m in s for _, s in rows)]
    skipped = [m for m in models if m not in graded]
    if not graded:
        sys.exit("no model in the head is graded on every fixture row; "
                 "see docs/EVALUATION.md")

    # --- hidden states -> p_solve -------------------------------------------------
    cache_ok = args.cache and os.path.exists(args.cache)
    if cache_ok and is_success:
        H = success.load_cache(args.cache, [q for q, _ in rows])
    elif cache_ok:
        cz = np.load(args.cache, allow_pickle=False)
        H = cz["H"]
        if len(H) != len(rows) or list(cz["questions"]) != [q for q, _ in rows]:
            sys.exit(f"{args.cache} does not match this fixture; delete it and re-run.")
        print(f"  reusing {len(H)} cached hidden states from {args.cache}")
    else:
        H, t0 = [], time.time()
        for i, (q, _) in enumerate(rows, 1):
            msgs = [{"role": "system", "content": router_sys},
                    {"role": "user", "content": q}]
            with f2._rlock:
                h = f2.router.hidden(msgs).float().cpu().numpy()
            H.append(h if is_success else h / max(np.linalg.norm(h), 1e-8))
            if i % 25 == 0 or i == len(rows):
                el = time.time() - t0
                print(f"  hidden states {i}/{len(rows)}  ({el:.0f}s, "
                      f"{el / i:.2f}s/question)", flush=True)
        H = np.asarray(H)
        if args.cache:
            if is_success:
                success.save_cache(args.cache, [q for q, _ in rows], H)
            else:
                np.savez_compressed(args.cache, H=H, questions=[q for q, _ in rows])
            print(f"  cached to {args.cache}")

    idx = [models.index(m) for m in graded]
    P = success.predictions(f2.W[idx], f2.b[idx], H) if is_success else 1 / (1 + np.exp(-(H @ f2.W[idx].T + f2.b[idx])))          # (n_questions, n_graded)
    Y = np.array([[s[m] for m in graded] for _, s in rows], float)
    cost = np.asarray(mean_cost)[idx]

    print(f"\n  {len(rows)} questions x {len(graded)} graded models, lambda={lam:g}")
    if skipped:
        print(f"  not graded in this fixture, excluded: {', '.join(skipped)}")

    # --- per model ----------------------------------------------------------------
    print(f"\n  {'model':34}{'n':>6}{'base':>8}{'mean p':>8}{'AUC':>8}{'Brier':>8}")
    for j, m in enumerate(graded):
        a = auc(P[:, j], Y[:, j])
        brier = float(((P[:, j] - Y[:, j]) ** 2).mean())
        print(f"  {m:34}{len(rows):6d}{Y[:, j].mean():8.3f}{P[:, j].mean():8.3f}"
              f"{a:8.3f}{brier:8.3f}")
    aucs = np.array([auc(P[:, j], Y[:, j]) for j in range(len(graded))])
    finite = aucs[np.isfinite(aucs)]
    print(f"  {'mean AUC':34}{'':6}{'':8}{'':8}{finite.mean():8.3f}"
          if len(finite) else "  mean AUC: undefined")

    # --- calibration --------------------------------------------------------------
    e, table = ece(P.ravel(), Y.ravel(), bins=args.bins)
    print(f"\n  calibration (all model x question pairs pooled)   ECE = {e:.4f}")
    print(f"  {'p_solve bin':>16}{'n':>8}{'mean p':>9}{'actual':>9}{'gap':>8}")
    for lo, hi, k, conf, acc in table:
        print(f"  {f'[{lo:.1f}, {hi:.1f})':>16}{k:8d}{conf:9.3f}{acc:9.3f}{acc - conf:+8.3f}")

    # --- end to end ---------------------------------------------------------------
    # The comparison that matters. "Best fixed model" is chosen with hindsight on this very
    # fixture, so it flatters itself and is the honest bar to clear: a router that cannot
    # beat one model picked in hindsight is not earning its forward pass.
    util = P - lam * cost
    picked = success.rank(P, cost, lam)[:, 0] if is_success else np.argmax(util, axis=1)
    routed_acc = float(Y[np.arange(len(rows)), picked].mean())
    routed_cost = float(cost[picked].mean())

    per_model_acc = Y.mean(axis=0)
    best_j = int(np.argmax(per_model_acc))
    cheap_j = int(np.argmin(cost))
    rng = np.random.default_rng(0)
    rand_acc = float(np.mean([Y[i, rng.integers(len(graded))] for i in range(len(rows))]))
    oracle_acc = float(Y.max(axis=1).mean())

    print(f"\n  {'strategy':34}{'accuracy':>10}{'$/query':>10}")
    print(f"  {'Fugal (routed)':34}{routed_acc:10.3f}{routed_cost:10.5f}  <-- this repo")
    print(f"  {'best fixed model (hindsight)':34}{per_model_acc[best_j]:10.3f}"
          f"{cost[best_j]:10.5f}  {graded[best_j]}")
    print(f"  {'cheapest model always':34}{per_model_acc[cheap_j]:10.3f}"
          f"{cost[cheap_j]:10.5f}  {graded[cheap_j]}")
    print(f"  {'uniformly random model':34}{rand_acc:10.3f}{cost.mean():10.5f}")
    print(f"  {'per-question oracle (ceiling)':34}{oracle_acc:10.3f}{'':>10}")

    print()
    beats_best = routed_acc >= per_model_acc[best_j]
    cheaper = routed_cost <= cost[best_j]
    if beats_best and cheaper:
        print("  Routing dominates the best fixed model on BOTH accuracy and cost.")
    elif cheaper:
        d = (per_model_acc[best_j] - routed_acc) * 100
        s = cost[best_j] / routed_cost if routed_cost else float("inf")
        print(f"  Routing gives up {d:.1f} accuracy points for {s:.1f}x cheaper queries. "
              f"Whether that is a good trade is a judgement, not a result.")
    else:
        print("  Routing does NOT beat the best fixed model here, on either axis. "
              "Report that.")
    print("  Remember: 'best fixed model' was chosen using this fixture's own answers.")


if __name__ == "__main__":
    main()
