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

Everything the routing decision conditions on is declared INSIDE the head artifact,
never hardcoded here: a v2 head carries per-model token statistics (so mean_cost is
computed from the CURRENT price sheet at load) and a `context` flag saying what
transcript distribution it was fit on (standalone questions vs multi-turn). The code
reads those declarations; upgrading routing behaviour means shipping a new head.

Serving (OpenAI + Anthropic wire shapes), the CLI, and the spend controls live in
serve.py. This file is the model.

PROVENANCE (Apache-2.0 §4(b)). The hidden-state extraction path (ROUTER_SYSTEM_PROMPT,
FugalRouter.format_transcript, FugalRouter.hidden) is derived from `openfugu/mini.py`,
Copyright 2026 The OpenFugu Contributors, Apache-2.0. The derived material has been
substantially modified. See NOTICE.
"""
from __future__ import annotations
import json, os, random, threading, time
from typing import NamedTuple

import numpy as np

from fugal import success_contract as success

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The backbone. scripts/fetch_backbone.py puts it in artifacts/; point FUGAL_MODEL at any
# other Qwen3-0.6B checkout (e.g. the HuggingFace cache) to skip the copy.
os.environ.setdefault("FUGAL_MODEL", os.path.join(REPO, "artifacts", "Qwen3-0.6B"))

# The trained head. Override with FUGAL_HEAD to serve one you fit yourself
# (docs/HEAD_FORMAT.md gives the .npz contract).
HEAD = os.environ.get("FUGAL_HEAD") or os.path.join(REPO, "data", "router_head.npz")
PRICES = os.environ.get("FUGAL_PRICES") or os.path.join(REPO, "data", "models_2026-06.json")

OR_URL = "https://openrouter.ai/api/v1/chat/completions"

# The router conditions on this system prompt plus the question. The head was fit on hidden
# states produced under it, so the string is part of the trained artifact — changing it
# invalidates the head unless you retrain (see docs/HEAD_FORMAT.md).
ROUTER_SYSTEM_PROMPT = (
    "You are a routing model. Given a question, your hidden state will be used to "
    "predict which language model can best answer it. Read the question carefully.")

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


def clamp_max_tokens(requested, model_max=None):
    """Caller's max_tokens -> a sane worker budget. Junk/absent falls back to the default.

    When *model_max* is given (from the price sheet's ``max_out``), it replaces
    the global ``MAX_MAX_TOKENS`` so the ceiling is the backend model's actual
    limit rather than an arbitrary constant.
    """
    ceiling = model_max if model_max is not None else MAX_MAX_TOKENS
    try:
        n = int(requested)
    except (TypeError, ValueError):
        return min(DEFAULT_MAX_TOKENS, ceiling)
    return max(1, min(n, ceiling)) if n > 0 else min(DEFAULT_MAX_TOKENS, ceiling)


def compose_system(client_system):
    """House identity + the caller's own system prompt (if any). The house prompt leads so
    identity and format stay consistent no matter which model answers; the client's
    instructions follow and are never discarded."""
    cs = (client_system or "").strip()
    return WORKER_SYSTEM_PROMPT + ("\n\n" + cs if cs else "")


# ---- the backbone: question -> hidden state ---------------------------------
class FugalRouter:
    """Qwen3-0.6B, used only for its mean-pooled hidden state.

    The backbone's own text output is never used and no tokens are ever generated,
    which is what makes a routing decision one forward pass (~1s on a CPU core)
    instead of an LLM call.
    """

    def __init__(self, model_dir: str, dtype: str = "float32", device: str | None = None,
                 success_profile: bool = False):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.success_profile = success_profile
        if success_profile:
            if dtype != "float32" or device not in (None, "cpu"):
                raise ValueError("success embeddings require CPU float32")
            success.check_backbone(model_dir)
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
        # raw 'role: content', NOT a chat template. The head was fit under this format.
        return "\n".join(f'{m["role"]}: {m["content"]}' for m in messages)

    def hidden(self, messages: list[dict]):
        torch = self.torch
        if self.success_profile:
            return torch.from_numpy(self.embed_questions([messages[-1]["content"]])[0])
        ids = self.tok(self.format_transcript(messages), return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.model(**ids)          # backbone only; LM head unused
        return out.last_hidden_state[0].mean(dim=0)


    def embed_questions(self, questions, batch_size=8):
        if not self.success_profile:
            raise ValueError("batch embedding requires a success profile")
        return success.embed(self.tok, self.model.model, questions, batch_size)


# ---- the OpenRouter worker call ---------------------------------------------
_RETRYABLE = {408, 409, 429, 500, 502, 503, 504}


def or_request(model, messages, max_tokens=4096, temperature=None, timeout=180, retries=3,
               tools=None, tool_choice=None):
    """One OpenRouter chat call, returning the RAW assistant message plus token usage.

    This is the full-fidelity path: `messages` is passed through verbatim (including
    role="tool" results), and `tools` is forwarded so the model can emit tool_calls.
    OpenRouter relays both to the underlying model - there is nothing to implement here,
    only fields to stop discarding.

    `temperature=None` means the field is OMITTED, so each worker keeps its provider's
    own default. A value is only ever sent when the caller actually asked for one —
    routing to a different model per message is no reason to override every model's
    sampling defaults with an opinion of ours.

    Returns (message_dict, usage_dict). The message may carry `content`, `tool_calls`,
    or both. usage is OpenRouter's block verbatim — prompt_tokens, completion_tokens and,
    with usage accounting requested, `cost`: what this call was actually charged.
    """
    import httpx
    key = os.environ.get("FUGAL_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
    payload = {"model": model, "max_tokens": max_tokens, "messages": messages,
               "usage": {"include": True}}
    if temperature is not None:
        payload["temperature"] = temperature
    if tools:
        payload["tools"] = tools
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice
    last = None
    for attempt in range(retries):
        try:
            r = httpx.post(OR_URL, timeout=timeout,
                headers={"Authorization": f"Bearer {key}"}, json=payload)
            if r.status_code in _RETRYABLE:
                last = RuntimeError(f"http {r.status_code}")
                raise last
            r.raise_for_status()
            j = r.json()
            if "choices" not in j:
                raise ValueError(str(j.get("error", j))[:200])
            return j["choices"][0]["message"] or {}, j.get("usage") or {}
        except (httpx.TimeoutException, httpx.NetworkError) as e:
            last = e
        except RuntimeError as e:
            last = e
        except ValueError:
            raise
        if attempt < retries - 1:
            time.sleep((2 ** attempt) + random.random())
    raise last if last else RuntimeError("or_request failed")


def or_call(model, prompt, max_tokens=4096, temperature=None, timeout=180, retries=3,
            history=None, system=None):
    """Text-only wrapper over or_request -> (text, usage). The tool path uses or_request
    directly. `system`, when given, is prepended as a system message (the house
    identity/format prompt for worker calls)."""
    msgs = ([{"role": "system", "content": system}] if system else []) \
        + list(history or []) + [{"role": "user", "content": prompt}]
    msg, usage = or_request(model, msgs, max_tokens=max_tokens, temperature=temperature,
                            timeout=timeout, retries=retries)
    return (msg.get("content") or "").strip() or (msg.get("reasoning") or ""), usage


def call_cost(prices, model, usage):
    """USD for one worker call -> (cost, source).

    OpenRouter reports what it actually charged in `usage.cost` (credits, priced one to
    one in USD); that is the number the spend caps should count, so it wins whenever it is
    present. The price sheet is the fallback — an estimate from token counts — and the
    source is reported so a cost figure always says which it was. No invented rate:
    load_head refuses to start with an unpriced model, so a sheet miss here is a bug."""
    cost = usage.get("cost")
    if cost is not None:
        return float(cost), "openrouter"
    pin, pout = prices[model]
    return (int(usage.get("prompt_tokens") or 0) * pin
            + int(usage.get("completion_tokens") or 0) * pout), "sheet"


# ---- loading the trained artifact ---------------------------------------------
def load_prices(path):
    """data/models_*.json -> {model_id: (usd_per_input_token, usd_per_output_token)}.

    Also returns a second dict {model_id: max_output_tokens} when the sheet
    carries ``max_out`` (added by refresh_prices.py from OpenRouter metadata).
    Callers that need only prices can ignore the second return value.
    """
    with open(path) as f:
        rows = json.load(f)
    prices = {m["id"]: (m["in"] / 1e6, m["out"] / 1e6) for m in rows}
    max_out = {m["id"]: int(m["max_out"]) for m in rows if "max_out" in m}
    return prices, max_out


class Head(NamedTuple):
    """What load_head returns. Everything routing conditions on, plus where it came from."""
    models: list
    W: "np.ndarray"
    b: "np.ndarray"
    mean_cost: "np.ndarray"
    lam: float
    context: str                # "standalone" | "multiturn"
    fmt: str                    # "v1" | "v2"
    backbone_revision: str      # HF commit of the Qwen3-0.6B it was fit on, or ""
    provenance: str             # free text: fit date, run id, training-code commit, or ""


def load_head(head_path, prices, models=None, router_lambda=None) -> Head:
    """The .npz head -> Head(models, W, b, mean_cost, lam, context, fmt, ...). Pure numpy.

    Two head formats (docs/HEAD_FORMAT.md), reported as fmt="v1" / "v2":
      v2 carries mean_in_tokens / mean_out_tokens — frozen MEASUREMENTS of how many
         tokens a query for each model averaged at fit time — and mean_cost is computed
         here from the CURRENT price sheet, so refresh_prices.py keeps routing honest.
      v1 carries a baked-in mean_cost, which freezes fit-time prices into every routing
         decision. Supported until a v2 head replaces it; the server banner says which
         format is loaded.

    `context` declares what transcript distribution the head was fit on ("standalone"
    questions, the default, or "multiturn"); the caller feeds the router history only
    when the head says it can use it. `backbone_revision` and `provenance` are optional
    strings that say which backbone commit the head was fit on and where it came from;
    scripts/fetch_backbone.py pins the download to the declared revision.

    lam resolution order: the router_lambda argument, then FUGAL_LAMBDA, then the
    head's trained default. A model with no entry in the price sheet is a hard error:
    inventing a rate would silently corrupt every cost report and spend cap.
    """
    if not os.path.exists(head_path):
        raise SystemExit(
            f"router head missing: {head_path}\n"
            f"  Expected data/router_head.npz, or set FUGAL_HEAD to your own "
            f"(format: docs/HEAD_FORMAT.md).")
    with open(head_path, "rb") as head_file:
        z = success.read_archive(head_file.read(success.MAX_BYTES + 1))
    is_success = "contract" in z
    if not is_success and ({"profile_id", "cost_profile_id"} & set(z)):
        raise ValueError("success metadata requires an explicit contract version")
    if is_success:
        success.validate_head(z)
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
        idx = ([i for i, m in enumerate(all_models) if m in sel] if is_success
               else [all_models.index(m) for m in sel])
    else:
        idx = list(range(len(all_models)))
    out_models = [all_models[i] for i in idx]

    unpriced = [m for m in out_models if m not in prices]
    if unpriced:
        raise SystemExit(
            f"no price for: {', '.join(unpriced)}\n"
            f"  Every routed model must be billable, or cost reports and spend caps lie.\n"
            f"  Run `python scripts/refresh_prices.py` to re-sync the sheet, or exclude "
            f"the model(s) with --models / FUGAL_MODELS.")

    fmt = success.CONTRACT if is_success else ("v2" if "mean_in_tokens" in z else "v1")
    if is_success:
        # Price sheets are explicit inputs; refreshed prices can change selection.
        subset = dict(z, models=z["models"][idx], mean_in_tokens=z["mean_in_tokens"][idx],
                      mean_out_tokens=z["mean_out_tokens"][idx])
        mean_cost = success.estimated_cost(subset, prices)
    elif fmt == "v2":
        itok, otok = z["mean_in_tokens"][idx], z["mean_out_tokens"][idx]
        pin = np.array([prices[m][0] for m in out_models])
        pout = np.array([prices[m][1] for m in out_models])
        mean_cost = itok * pin + otok * pout
    else:
        mean_cost = z["mean_cost"][idx]

    if router_lambda is not None:
        lam = float(router_lambda)
    elif os.environ.get("FUGAL_LAMBDA"):
        lam = float(os.environ["FUGAL_LAMBDA"])
    else:
        lam = float(z["lam"])
    if is_success and (not np.isfinite(lam) or lam < 0):
        raise ValueError("lambda must be finite and nonnegative")
    def text(key):
        return str(z[key]) if key in z else ""
    return Head(out_models, z["W"][idx], z["b"][idx], mean_cost, lam,
                text("context") or "standalone", fmt,
                text("backbone_revision"), text("provenance"))


# ---- the router: hidden state -> which model answers -------------------------
class Fugal:
    """Route once, call one model, return its answer.

    models: restrict routing to a subset of the head's model list (or set FUGAL_MODELS
        to a comma-separated list). Use this when you only hold keys for some providers.
        Each row of W is an independent logistic head, so a subset is exact, not an
        approximation — the remaining models score exactly as they would have.
    router_lambda: override the head's cost sensitivity (also FUGAL_LAMBDA). Higher =>
        trade down to cheaper workers on easy queries. None keeps the trained default.
    """

    def __init__(self, models=None, router_lambda=None):
        self.prices, self.max_out_tokens = load_prices(PRICES)
        self.head = load_head(HEAD, self.prices, models=models, router_lambda=router_lambda)
        self.models, self.W, self.b, self.mean_cost, self.lam = self.head[:5]
        self.head_context, self.head_format = self.head.context, self.head.fmt
        mdir = os.environ["FUGAL_MODEL"]
        if not os.path.isdir(mdir):
            raise SystemExit(
                f"backbone not found: {mdir}\n"
                f"  Fetch it once:  python scripts/fetch_backbone.py\n"
                f"  Or point FUGAL_MODEL at an existing Qwen3-0.6B directory.")
        self.router = (FugalRouter(mdir, success_profile=True)
                       if self.head_format == success.CONTRACT else FugalRouter(mdir))
        self._rlock = threading.Lock()      # torch forward is not thread-safe

    def _price(self, model, usage):
        return call_cost(self.prices, model, usage)

    def _charge(self, meta, model, usage):
        """Record one worker call on meta: cost, its source, token totals, the step."""
        c, src = self._price(model, usage)
        meta["cost"] += c
        meta["cost_source"] = src
        meta["input_tokens"] += int(usage.get("prompt_tokens") or 0)
        meta["output_tokens"] += int(usage.get("completion_tokens") or 0)
        meta["steps"].append({"role": "worker", "model": model, "cost": c})
        return c

    def route(self, query, history=None):
        """query -> (models ranked best-first, their p_solve in the same order).
        Pass history ONLY for a head fit on multi-turn transcripts (head_context ==
        "multiturn"); the shipped head was fit on standalone questions, and feeding
        it hidden states from a distribution it never saw routes worse, silently.
        answer_iter and the server gate this on the head's own declaration."""
        if self.head_format == success.CONTRACT:
            history = None
        msgs = [{"role": "system", "content": ROUTER_SYSTEM_PROMPT},
                *(history or []),
                {"role": "user", "content": query}]
        with self._rlock:
            h = self.router.hidden(msgs).float().cpu().numpy()
        if self.head_format == success.CONTRACT:
            p = success.predictions(self.W, self.b, h)
            order = success.rank(p, self.mean_cost, self.lam)
            return [self.models[i] for i in order], p[order]
        h = h / max(np.linalg.norm(h), 1e-8)
        p = 1 / (1 + np.exp(-(self.W @ h + self.b)))
        util = p - self.lam * self.mean_cost
        order = np.argsort(-util)
        return [self.models[i] for i in order], p[order]

    def answer_iter(self, query, history=None, tools=None, messages=None, system=None,
                    max_tokens=None, temperature=None, route_on=None):
        """Route once and answer, yielding one event per stage as it happens.
        The last event is {"stage":"final","content":...,"meta":...}. This is the
        single source of truth: answer() and the streaming server both consume it.
        `history` = prior {role, content} turns; the WORKER always sees it, the
        ROUTER sees it only if the head declares it was fit on multi-turn
        transcripts (head_context) — otherwise it routes on `route_on` alone.
        `route_on` = what the router reads, when it differs from the worker query:
        on a tool-loop continuation the last message is a tool RESULT, which the
        head cannot score, so the server passes the originating user turn instead.
        `system` = the caller's own system prompt; it is merged AFTER the house
        prompt (compose_system) so every model shares one identity and format."""
        # in/out token totals are accumulated alongside cost so the Anthropic-shaped
        # endpoint can report a real usage block instead of an estimate.
        sys_prompt = compose_system(system)
        meta = {"cost": 0.0, "cost_source": None, "steps": [],
                "input_tokens": 0, "output_tokens": 0}
        ranked, probs = self.route(
            route_on or query,
            history=history if self.head_context == "multiturn" else None)
        first = ranked[0]
        mtok = clamp_max_tokens(max_tokens, self.max_out_tokens.get(first))
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
            msg, usage = or_request(first, msgs, tools=tools, max_tokens=mtok,
                                    temperature=temperature)
            c = self._charge(meta, first, usage)
            meta["tool_calls"] = msg.get("tool_calls") or []
            yield {"stage": "worker", "model": first, "cost": c, "cost_total": meta["cost"]}
            yield {"stage": "final", "content": (msg.get("content") or ""), "meta": meta}
            return

        reply, usage = or_call(first, query, history=history, system=sys_prompt,
                               max_tokens=mtok, temperature=temperature)
        c = self._charge(meta, first, usage)
        yield {"stage": "worker", "model": first, "cost": c, "cost_total": meta["cost"]}
        yield {"stage": "final", "content": reply, "meta": meta}

    def answer(self, query, verbose=False, history=None, tools=None, messages=None,
               system=None, max_tokens=None, temperature=None, route_on=None):
        reply, meta = None, None
        for ev in self.answer_iter(query, history=history, tools=tools, messages=messages,
                                   system=system, max_tokens=max_tokens,
                                   temperature=temperature, route_on=route_on):
            if verbose and ev["stage"] == "route":
                alts = ", ".join(ev["alternatives"])
                print(f"  router: {ev['model']} (p_solve={ev['p_solve']:.2f}; next: {alts})")
            if ev["stage"] == "final":
                reply, meta = ev["content"], ev["meta"]
        return reply, meta
