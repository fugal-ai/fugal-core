#!/usr/bin/env python3
# Fugal — Apache-2.0. See NOTICE.
"""
refresh_prices.py — bring data/models_2026-06.json back in line with OpenRouter.

    python scripts/refresh_prices.py            # show the drift, then rewrite in place
    python scripts/refresh_prices.py --check    # show the drift, change nothing, exit 1 if any
    python scripts/refresh_prices.py --tolerance 10   # --check only cares about >10% moves

WHY THIS MATTERS MORE THAN IT LOOKS. Two different numbers in this repo are made of prices,
and only one of them should ever move:

  mean_cost (in data/router_head.npz)  is ROUTING input. It is a measurement taken when the
      head was fit, and it is frozen on purpose — `utility = p - lam * mean_cost` has to be
      evaluated against the same costs the head was fit under. This script never touches it.

  data/models_2026-06.json             is BILLING input. Fugal._price() multiplies these
      numbers by real token counts to produce meta["cost"], the X-Fugal-Cost-USD header, and
      the running total that --spend-cap and --daily-cap compare against. When a provider
      moves its price and this file does not, every one of those is quietly wrong, and a
      "cap" that miscounts is worse than no cap.

So: the routing trade-off is a fixed historical measurement, the money counter is supposed to
be current. Run this whenever you care about the second one being true. `git log
data/models_2026-06.json` is the record of when it last ran.

The file name keeps the date of the original snapshot; its CONTENTS are refreshed in place.

The set of models is defined by the head, not by this script — a row of W cannot be invented
(docs/HEAD_FORMAT.md), so a model that vanishes from OpenRouter is reported loudly and its
last known price is kept rather than dropped.
"""
from __future__ import annotations
import argparse, io, json, os, sys, urllib.request

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)                                # run from anywhere

from fugal import use_utf8                               # noqa: E402
HEAD = os.path.join(REPO, "data", "router_head.npz")
PRICES = os.path.join(REPO, "data", "models_2026-06.json")
OR_MODELS = "https://openrouter.ai/api/v1/models"       # public, needs no key


def fetch_live():
    """OpenRouter's public catalogue -> {id: (usd_per_1M_in, usd_per_1M_out)}."""
    req = urllib.request.Request(OR_MODELS, headers={"User-Agent": "fugal-refresh-prices"})
    with urllib.request.urlopen(req, timeout=60) as r:
        data = json.loads(r.read().decode("utf-8"))["data"]
    out = {}
    for m in data:
        p = m.get("pricing") or {}
        try:
            # OpenRouter quotes USD per token; the sheet is USD per million.
            out[m["id"]] = (float(p["prompt"]) * 1e6, float(p["completion"]) * 1e6)
        except (KeyError, TypeError, ValueError):
            continue                                    # unpriced/variant entries
    return out


def load_sheet():
    with io.open(PRICES, encoding="utf-8") as f:
        return {m["id"]: (float(m["in"]), float(m["out"])) for m in json.load(f)}


def head_models():
    """The head defines which models exist. Import numpy late so --help works without it."""
    import numpy as np
    z = np.load(HEAD)
    return [str(m) for m in z["models"]]


def money(x):
    """Up to 4 decimals, trailing zeros trimmed, never scientific notation.

    NOT round-to-2: sub-cent models are the ones the cost term actually discriminates
    between, and llama-4-maverick at 0.696 written as 0.70 is a 0.6% billing error baked
    into every future invoice for the sake of a tidier column."""
    s = f"{x:.4f}".rstrip("0")
    return s + "0" if s.endswith(".") else s


def write_sheet(models, prices):
    """Rewrite the sheet in head order, one aligned row per model.

    Hand-formatted rather than json.dump'd: this file is read by people comparing it against
    a provider's pricing page, and a column that lines up is the whole reason it is legible.
    """
    w = max(len(json.dumps(m)) for m in models)
    iw = max(len(money(p[0])) for p in prices.values())
    rows = []
    for m in models:
        pin, pout = prices[m]
        rows.append(f'  {{"id": {json.dumps(m) + ",":<{w + 1}} '
                    f'"in": {money(pin) + ",":<{iw + 1}} "out": {money(pout)}}}')
    text = "[\n" + ",\n".join(rows) + "\n]\n"
    json.loads(text)                                    # never write a file we cannot read
    with io.open(PRICES, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def main():
    use_utf8()                                          # the drift report uses >= and a warning sign
    ap = argparse.ArgumentParser(
        prog="python scripts/refresh_prices.py",
        description="Refresh data/models_2026-06.json from OpenRouter's live catalogue.")
    ap.add_argument("--check", action="store_true",
                    help="report drift and exit 1 if any exceeds --tolerance; write nothing")
    ap.add_argument("--tolerance", type=float, default=5.0, metavar="PCT",
                    help="--check only: percent move that counts as drift (default 5)")
    args = ap.parse_args()

    models = head_models()
    sheet = load_sheet()
    live = fetch_live()

    missing = [m for m in models if m not in sheet]
    if missing:
        sys.exit(f"price sheet is missing head models: {', '.join(missing)}")

    gone, drifted, new_prices = [], [], {}
    print(f"  {'model':34}{'in $/M':>10}{'out $/M':>10}{'drift':>10}")
    for m in models:
        old = sheet[m]
        if m not in live:
            gone.append(m)
            new_prices[m] = old
            print(f"  {m:34}{old[0]:10.3f}{old[1]:10.3f}{'GONE':>10}")
            continue
        new = live[m]
        new_prices[m] = new
        # Blended midpoint is only for deciding whether to SAY something; both numbers are
        # always written. It keeps a 0.001 rounding change from reading like news.
        o, n = (old[0] + old[1]) / 2, (new[0] + new[1]) / 2
        pct = ((n / o) - 1) * 100 if o else 0.0
        if abs(pct) >= args.tolerance:
            drifted.append((m, pct))
            print(f"  {m:34}{new[0]:10.3f}{new[1]:10.3f}{pct:9.0f}%"
                  f"   was {old[0]:g}/{old[1]:g}")
        else:
            print(f"  {m:34}{new[0]:10.3f}{new[1]:10.3f}{'':>10}")

    print()
    if gone:
        print(f"  ⚠ no longer on OpenRouter: {', '.join(gone)}")
        print("    Last known price kept. The head still scores these models, so routing can "
              "still pick one — see docs/HEAD_FORMAT.md before removing a row.")
    if not drifted and not gone:
        print(f"  sheet is current (no move ≥ {args.tolerance:g}%).")

    if args.check:
        if drifted or gone:
            sys.exit(f"\n{len(drifted)} price(s) drifted, {len(gone)} model(s) gone. "
                     f"Run without --check to rewrite the sheet.")
        return

    if new_prices == sheet:
        print("  nothing to write.")
        return
    write_sheet(models, new_prices)
    print(f"  wrote {os.path.relpath(PRICES, REPO)} ({len(drifted)} price(s) updated).")
    print("  mean_cost in the head is UNCHANGED and stays that way — it is routing input, "
          "not billing input.")


if __name__ == "__main__":
    main()
