# Fugal — Apache-2.0. See NOTICE.
"""Regression: routing is EXACTLY ONE model call per turn, on every turn.

That is the product's core promise, and it is the one property that cannot be checked by
reading the code. The bug this locks down was real: the answer path used to fall through to

    if thr is not None and not history and meta["p_solve"] >= thr:

so a follow-up turn — where `not history` is False — silently ran a second and sometimes a
third model call the user was never told about. A SINGLE-TURN TEST PASSES EVEN IF THIS IS
WRONG, which is why case (B) below is the important one.

Asserts, with or_call mocked (no network, no key, zero spend):
  A) no history   -> exactly one model call; stages are route, worker, final
  B) WITH history -> exactly one model call; the history reaches the worker intact
  C) the model that was routed to is the model that answered (final_model == the call)
  D) a follow-up turn does not re-route to a second model or re-ask the first

Needs the backbone (one local forward pass per case, ~2s each):
    FUGAL_MODEL=/path/to/Qwen3-0.6B python verify/verify_routing.py
"""
import os, sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)                        # run from anywhere
os.environ.pop("FUGAL_API_KEY", None)           # belt and braces: no live call can succeed
os.environ.pop("OPENROUTER_API_KEY", None)

from fugal import router                        # noqa: E402

calls = []


def fake_or_call(model, prompt, max_tokens=4096, temperature=0.0, timeout=180,
                 retries=3, history=None, system=None, **kw):
    # **kw so a future or_call parameter cannot silently turn this file into a TypeError
    # that never reaches an assertion — which is exactly how an earlier guardrail test rotted.
    calls.append({"model": model, "prompt": prompt, "history": list(history or []),
                  "system": system})
    return "mocked worker reply", 10, 5


def fake_or_request(*a, **kw):
    raise AssertionError("or_request called: the plain-chat path must go through or_call")


# Fugal.answer_iter resolves these in router's namespace, so patch them there.
router.or_call = fake_or_call
router.or_request = fake_or_request

HIST = [{"role": "user", "content": "Write a python function to reverse a string."},
        {"role": "assistant", "content": "def rev(s): return s[::-1]"}]

EXPECTED_STAGES = ["route", "worker", "final"]
failures = []


def check(label, history):
    calls.clear()
    events = list(F2.answer_iter("now show it in Rust", history=history))
    stages = [e["stage"] for e in events]
    meta = next((e["meta"] for e in events if e["stage"] == "final"), {})
    workers = [c["model"] for c in calls]

    problems = []
    if stages != EXPECTED_STAGES:
        problems.append(f"stages {stages} != {EXPECTED_STAGES}")
    if len(workers) != 1:
        problems.append(f"expected exactly 1 model call, got {len(workers)}: {workers}")
    if workers and meta.get("final_model") != workers[0]:
        problems.append(f"final_model {meta.get('final_model')} != the model called "
                        f"{workers[0]}")
    if len(meta.get("steps") or []) != 1:
        problems.append(f"meta.steps records {len(meta.get('steps') or [])} steps, want 1")
    if calls and len(calls[0]["history"]) != len(history or []):
        problems.append(f"worker saw {len(calls[0]['history'])} history turns, "
                        f"sent {len(history or [])}")

    print(f"  {label:28} stages={stages}")
    print(f"  {'':28} calls={len(workers)}  worker={meta.get('final_model')}  "
          f"p_solve={meta.get('p_solve', 0):.3f}")
    if problems:
        failures.append((label, problems))
        for p in problems:
            print(f"  {'':28} !! {p}")


print("\n  routing: one model call per turn, history or not (mocked, $0)\n")
F2 = router.Fugal()
check("A) no history", None)
check("B) WITH history", HIST)
check("C) longer history", HIST * 3)

print()
if failures:
    for label, problems in failures:
        for p in problems:
            print(f"FAIL {label}: {p}")
    sys.exit(f"\n{len(failures)} case(s) failed — routing is NOT one call per turn.")
print("PASS: one model call per turn, on every turn; the routed model is the one that answers.")
