#!/usr/bin/env python3
# Fugal — Apache-2.0. See NOTICE.
"""
router.py — the whole product, in one file.

A tiny router (one forward pass of Qwen3-0.6B, NO generation) predicts, per query,
which model has the best odds-vs-price trade. Then it calls that ONE model and
returns its answer. That is the entire system:

    h = hidden_state(question)             # one forward pass, ~1s on a CPU core
    p = sigmoid(W @ h + b)                 # P(each model solves this), 17 models
    utility = p - lambda * mean_cost       # odds, discounted by price
    worker  = argmax(utility)              # call this one, once

`W` is (17, 1024) and lives in data/router_head.npz — 73 KB. Each row is an
independent logistic head for one model, which is why you can route among a
SUBSET of the models for free (see `models=` / FUGAL_MODELS below): dropping a
row cannot disturb the others. Adding a NEW model cannot be done here — that
needs evidence for that model. See docs/HEAD_FORMAT.md.

Serving (OpenAI + Anthropic wire shapes), the CLI, and the spend controls live in
serve.py. This file is the model.

PROVENANCE (Apache-2.0 s4(b) — this file contains modified third-party material).
The hidden-state extraction path below — ROUTER_SYSTEM_PROMPT, HIDDEN_POS,
FugalRouter.format_transcript and FugalRouter.hidden — is DERIVED FROM
`openfugu/mini.py` in github.com/trotsky1997/OpenFugu, Copyright 2026 The OpenFugu
Contributors, Apache-2.0. Fugal's changes to it: the SVF backbone adaptation was
removed (this runs an unmodified Qwen3-0.6B, no TRINITY checkpoint), the bias-free
(10, 1024) agent/role logit head was replaced by the (17, 1024) per-model logistic
head with bias and the price-discounted decision rule, the hidden state is now
L2-normalised, and the multi-turn Coordinator loop was not carried over. See NOTICE.

The `[EXEC]` tags on two constants below are OpenFugu's provenance notation, kept so
the constants stay traceable to where they were established: [EXEC] means the value
was reproduced by running real weights rather than read off a paper. Fugal inherited
these values; see the caveat in docs/HEAD_FORMAT.md about what that does and does not
justify.
"""
from __future__ import annotations
import json, os, random, threading, time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The backbone. scripts/fetch_backbone.py puts it in artifacts/; point FUGAL_MODEL at any
# other Qwen3-0.6B checkout (e.g. the HuggingFace cache) to skip the copy.
os.environ.setdefault("FUGAL_MODEL", os.path.join(REPO, "artifacts", "Qwen3-0.6B"))

# The trained head. Override with FUGAL_HEAD to serve one you fit yourself
# (docs/HEAD_FORMAT.md gives the .npz contract).
HEAD = os.environ.get("FUGAL_HEAD") or os.path.join(REPO, "data", "router_head.npz")
PRICES = os.environ.get("FUGAL_PRICES") or os.path.join(REPO, "data", "models_2026-06.json")

HIDDEN_POS = -2            # penultimate-token hidden state [EXEC, from OpenFugu]
OR_URL = "https://openrouter.ai/api/v1/chat/completions"

# The router conditions on this exact system prompt plus the question. It is byte-identical
# to ROUTER_SYSTEM_PROMPT in OpenFugu's openfugu/mini.py, which reconstructed it from the
# TRINITY coordinator (see NOTICE), and Fugal's head was fit on hidden states produced under
# it — so the string is part of the trained artifact, not a stylistic choice. `num_agents=7` is likewise fixed by what the head was fit under;
# it does NOT mean "we route among 7 models" (we route among 17). Changing either invalidates
# the head.
ROUTER_SYSTEM_PROMPT = (
    "You are a message dispatcher whose job is to coordinate {num_agents} agents "
    "to solve a problem. You check the problem and the discussion history and then "
    "decide which agent should respond next. Your first generated token's hidden "
    "state will be used as signal for decision making.")
ROUTER_NUM_AGENTS = 7

# House system prompt prepended to EVERY worker call. Because the product routes each message
# to a different underlying model, without a shared instruction each model answers in its own
# voice and claims its own identity ("I'm ChatGPT" / "I'm Claude") — jarring when the model
# silently changes between turns. This pins ONE identity and ONE output format across all of
# them, which is the whole point of making a router feel like a single assistant. A
# client-supplied system prompt (e.g. an agent harness like Claude Code) is APPENDED after
# this, never dropped — see compose_system().
WORKER_SYSTEM_PROMPT = (
    "You are Fugal, a helpful AI assistant. Each message may be handled by a different "
    "underlying model, so keep one consistent voice and identity: never claim to be, or "
    "speculate about, any specific model or the company that built you. Lead with the answer, "
    "then supporting detail; be concise and direct. Use Markdown for structure and fenced code "
    "blocks with a language tag for code. If you are unsure or lack the information to answer, "
    "say so plainly instead of guessing."
)

# Output-length budget for a worker call. The caller's max_tokens is honoured (Anthropic
# requires the field; OpenAI clients often set it) because a coding harness asking for a long
# file and silently getting 4096 tokens looks like the model truncating, not like a proxy
# ignoring the request. Clamped so one request cannot order an unbounded generation.
DEFAULT_MAX_TOKENS = 4096
MAX_MAX_TOKENS = 32000


def clamp_max_tokens(requested):
    """Caller's max_tokens -> a sane worker budget. Junk/absent falls back to the default."""
    try:
        n = int(requested)
    except (TypeError, ValueError):
        return DEFAULT_MAX_TOKENS
    return max(1, min(n, MAX_MAX_TOKENS)) if n > 0 else DEFAULT_MAX_TOKENS


def compose_system(client_system):
    """House identity + the caller's own system prompt (if any). The house prompt leads so
    identity and format stay consistent no matter which model answers; the client's
    instructions follow and are never discarded."""
    cs = (client_system or "").strip()
    return WORKER_SYSTEM_PROMPT + ("\n\n" + cs if cs else "")


# ---- the backbone: question -> hidden state ---------------------------------
class FugalRouter:
    """Qwen3-0.6B, used only for its penultimate-token hidden state.

    The backbone's own text output is never used and no tokens are ever generated,
    which is what makes a routing decision one forward pass (~1s on a CPU core)
    instead of an LLM call.
    """

    def __init__(self, model_dir: str, dtype: str = "float32", device: str | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        # transformers >=5 uses dtype=, <5 uses torch_dtype= — support both
        td = getattr(torch, dtype)
        try:
            self.model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=td).eval()
        except TypeError:
            self.model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=td).eval()
        if device:
            self.model.to(device)
        self.device = next(self.model.parameters()).device

    @staticmethod
    def format_transcript(messages: list[dict]) -> str:
        # raw 'role: content', NOT a chat template. [EXEC, from OpenFugu] — the 95%-vs-11%
        # figure is OpenFugu's measurement against the TRINITY checkpoint, not Fugal's.
        # Fugal keeps the format because its own head was fit under it.
        return "\n".join(f'{m["role"]}: {m["content"]}' for m in messages)

    def hidden(self, messages: list[dict]):
        torch = self.torch
        ids = self.tok(self.format_transcript(messages), return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.model(**ids)          # backbone only; LM head unused
        return out.last_hidden_state[0, HIDDEN_POS, :]


# ---- the OpenRouter worker call ---------------------------------------------
_RETRYABLE = {408, 409, 429, 500, 502, 503, 504}


def or_request(model, messages, max_tokens=4096, temperature=0.0, timeout=180, retries=3,
               tools=None, tool_choice=None):
    """One OpenRouter chat call, returning the RAW assistant message plus token usage.

    This is the full-fidelity path: `messages` is passed through verbatim (including
    role="tool" results), and `tools` is forwarded so the model can emit tool_calls.
    OpenRouter relays both to the underlying model - there is nothing to implement here,
    only fields to stop discarding.

    Returns (message_dict, prompt_tokens, completion_tokens). The message may carry
    `content`, `tool_calls`, or both.
    """
    import requests
    key = os.environ.get("FUGAL_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
    payload = {"model": model, "temperature": temperature, "max_tokens": max_tokens,
               "messages": messages}
    if tools:
        payload["tools"] = tools
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice
    last = None
    for attempt in range(retries):
        try:
            r = requests.post(OR_URL, timeout=timeout,
                headers={"Authorization": f"Bearer {key}"}, json=payload)
            if r.status_code in _RETRYABLE:
                last = RuntimeError(f"http {r.status_code}")
                raise last
            r.raise_for_status()
            j = r.json()
            if "choices" not in j:
                raise RuntimeError(str(j.get("error", j))[:200])
            msg = j["choices"][0]["message"] or {}
            u = j.get("usage", {}) or {}
            return msg, int(u.get("prompt_tokens", 0)), int(u.get("completion_tokens", 0))
        except (requests.Timeout, requests.ConnectionError) as e:
            last = e
        except RuntimeError as e:
            last = e
        if attempt < retries - 1:
            time.sleep((2 ** attempt) + random.random())
    raise last if last else RuntimeError("or_request failed")


def or_call(model, prompt, max_tokens=4096, temperature=0.0, timeout=180, retries=3,
            history=None, system=None):
    """Text-only wrapper over or_request. Kept at its original 3-tuple RETURN contract
    because several call sites and the mock in verify/ depend on that shape - the tool path
    uses or_request directly rather than widening this. `system`, when given, is prepended as
    a system message (the house identity/format prompt for worker calls)."""
    msgs = ([{"role": "system", "content": system}] if system else []) \
        + list(history or []) + [{"role": "user", "content": prompt}]
    msg, itok, otok = or_request(model, msgs, max_tokens=max_tokens, temperature=temperature,
                                 timeout=timeout, retries=retries)
    text = (msg.get("content") or "").strip() or (msg.get("reasoning") or "")
    return text, itok, otok


# ---- the router: hidden state -> which model answers -------------------------
class Fugal:
    """Route once, call one model, return its answer.

    models: restrict routing to a subset of the head's model list (or set FUGAL_MODELS
        to a comma-separated list). Use this when you only hold keys for some providers.
        Each row of W is an independent logistic head, so a subset is exact, not an
        approximation — the remaining models score exactly as they would have.
    router_lambda: override the head's cost sensitivity. Higher => trade down to cheaper
        workers on easy queries. None keeps the trained default.
    """

    def __init__(self, models=None, router_lambda=None):
        if not os.path.exists(HEAD):
            raise SystemExit(
                f"router head missing: {HEAD}\n"
                f"  Expected data/router_head.npz, or set FUGAL_HEAD to your own "
                f"(format: docs/HEAD_FORMAT.md).")
        z = np.load(HEAD, allow_pickle=True)
        all_models = [str(m) for m in z["models"]]

        sel = models if models is not None else os.environ.get("FUGAL_MODELS")
        if isinstance(sel, str):
            sel = [s.strip() for s in sel.split(",") if s.strip()]
        if sel:
            unknown = [m for m in sel if m not in all_models]
            if unknown:
                raise SystemExit(
                    f"unknown model(s) for this head: {', '.join(unknown)}\n"
                    f"  The head scores exactly these {len(all_models)}:\n    "
                    + "\n    ".join(all_models)
                    + "\n  Routing to a model the head was not fit on is not possible; "
                      "see docs/HEAD_FORMAT.md.")
            idx = [all_models.index(m) for m in sel]
        else:
            idx = list(range(len(all_models)))

        self.models = [all_models[i] for i in idx]
        self.W, self.b, self.mean_cost = z["W"][idx], z["b"][idx], z["mean_cost"][idx]
        self.lam = float(z["lam"]) if router_lambda is None else float(router_lambda)
        self.prices = {m["id"]: (m["in"] / 1e6, m["out"] / 1e6)
                       for m in json.load(open(PRICES))}
        mdir = os.environ["FUGAL_MODEL"]
        if not os.path.isdir(mdir):
            raise SystemExit(
                f"backbone not found: {mdir}\n"
                f"  Fetch it once:  python scripts/fetch_backbone.py\n"
                f"  Or point FUGAL_MODEL at an existing Qwen3-0.6B directory.")
        self.router = FugalRouter(mdir)
        self._rlock = threading.Lock()      # torch forward is not thread-safe

    def _price(self, model, itok, otok):
        pin, pout = self.prices.get(model, (1e-6, 3e-6))
        return itok * pin + otok * pout

    def route(self, query):
        """query -> (models ranked best-first, their p_solve in the same order)."""
        msgs = [{"role": "system",
                 "content": ROUTER_SYSTEM_PROMPT.format(num_agents=ROUTER_NUM_AGENTS)},
                {"role": "user", "content": query}]
        with self._rlock:
            h = self.router.hidden(msgs).float().cpu().numpy()
        h = h / max(np.linalg.norm(h), 1e-8)
        p = 1 / (1 + np.exp(-(self.W @ h + self.b)))
        util = p - self.lam * self.mean_cost
        order = np.argsort(-util)
        return [self.models[i] for i in order], p[order]

    def answer_iter(self, query, history=None, tools=None, messages=None, system=None,
                    max_tokens=None):
        """Route once and answer, yielding one event per stage as it happens.
        The last event is {"stage":"final","content":...,"meta":...}. This is the
        single source of truth: answer() and the streaming server both consume it.
        `history` = prior {role, content} turns; the router routes on the latest
        query alone (what the head was fit on), the worker sees history.
        `system` = the caller's own system prompt; it is merged AFTER the house
        prompt (compose_system) so every model shares one identity and format."""
        # in/out token totals are accumulated alongside cost so the Anthropic-shaped
        # endpoint can report a real usage block instead of an estimate.
        sys_prompt = compose_system(system)
        mtok = clamp_max_tokens(max_tokens)
        meta = {"cost": 0.0, "steps": [], "input_tokens": 0, "output_tokens": 0}
        ranked, probs = self.route(query)
        first = ranked[0]
        meta["p_solve"] = float(probs[0])          # router confidence in the chosen worker
        meta["ranked"] = list(ranked[:3])
        meta["final_model"] = first
        yield {"stage": "route", "model": first, "p_solve": float(probs[0]),
               "alternatives": ranked[1:3]}

        # --- tool-calling turn --------------------------------------------------------
        # `messages` is passed through verbatim so tool results (role="tool", with their
        # tool_call_id) reach the model intact - the pipeline's usual flattening to a
        # query string destroys exactly the structure a tool loop runs on.
        if tools:
            base = messages if messages is not None else (
                list(history or []) + [{"role": "user", "content": query}])
            # House identity leads; strip any stray system turns from `base` first so the
            # merged prompt (which already contains the client's system via compose_system)
            # is the single system message rather than one of several the model may ignore.
            msgs = [{"role": "system", "content": sys_prompt}] \
                + [m for m in base if m.get("role") != "system"]
            msg, itok, otok = or_request(first, msgs, tools=tools, max_tokens=mtok)
            c = self._price(first, itok, otok); meta["cost"] += c
            meta["input_tokens"] += itok; meta["output_tokens"] += otok
            meta["steps"].append({"role": "worker", "model": first, "cost": c})
            meta["tool_calls"] = msg.get("tool_calls") or []
            yield {"stage": "worker", "model": first, "cost": c, "cost_total": meta["cost"]}
            yield {"stage": "final", "content": (msg.get("content") or ""), "meta": meta}
            return

        reply, itok, otok = or_call(first, query, history=history, system=sys_prompt,
                                    max_tokens=mtok)
        c = self._price(first, itok, otok); meta["cost"] += c
        meta["input_tokens"] += itok; meta["output_tokens"] += otok
        meta["steps"].append({"role": "worker", "model": first, "cost": c})
        yield {"stage": "worker", "model": first, "cost": c, "cost_total": meta["cost"]}
        yield {"stage": "final", "content": reply, "meta": meta}

    def answer(self, query, verbose=False, history=None, tools=None, messages=None,
               system=None, max_tokens=None):
        reply, meta = None, None
        for ev in self.answer_iter(query, history=history, tools=tools, messages=messages,
                                   system=system, max_tokens=max_tokens):
            if verbose and ev["stage"] == "route":
                alts = ", ".join(ev["alternatives"])
                print(f"  router: {ev['model']} (p_solve={ev['p_solve']:.2f}; next: {alts})")
            if ev["stage"] == "final":
                reply, meta = ev["content"], ev["meta"]
        return reply, meta
