#!/usr/bin/env python3
# Fugal — Apache-2.0. See NOTICE.
"""
serve.py — the CLI and the HTTP server. The router itself is router.py.

Usage:
  python -m fugal --route                             # interactive routing inspector ($0)
  python -m fugal --route "what is 15% of 240?"       # one-shot, same table ($0)
  python -m fugal --query "..."                       # route once, answer once (spends credit)
  python -m fugal --serve --port 8090 --spend-cap 5.00
      # OpenAI + Anthropic compatible endpoint with real streaming. Optional bearer auth
      # via FUGAL_SERVE_TOKEN; --spend-cap is a hard cumulative USD ceiling.
      # GET /health, GET /v1/models, POST /v1/route ($0, no key), POST /v1/chat/completions,
      # POST /v1/messages.
      # Browsers are locked out by default: no CORS headers unless --cors-origin says so,
      # and on a loopback bind the Host header must name loopback (DNS rebinding).

Add --models "a,b,c" (or set FUGAL_MODELS) to route among a subset of the head's 17 models —
useful when you hold keys for some providers and not others. See docs/HEAD_FORMAT.md.

Env: FUGAL_API_KEY (OpenRouter, live calls only), FUGAL_MODEL (backbone directory),
FUGAL_HEAD (your own head), FUGAL_LAMBDA (cost sensitivity), FUGAL_MODELS (subset).

Layout: the wire-shape adapters are pure functions (tests/test_adapters.py); build_app()
turns a Fugal into an ASGI app and takes every policy knob as an argument, which is what
lets tests/test_server.py drive every endpoint with a fake router and no backbone; serve()
is build_app() plus uvicorn.
"""
from __future__ import annotations
import argparse, asyncio, hmac, json, os, sys, time, traceback, uuid
from collections import deque
from datetime import datetime, timezone

import httpx
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from . import use_utf8
from .router import Fugal, call_cost, clamp_max_tokens, compose_system, OR_URL

MAX_BODY = 256 * 1024
LOOPBACK = {"127.0.0.1", "localhost", "::1"}


def client_system_from_body(body):
    """The caller's own system prompt, pulled from wherever their client shape put it: a
    top-level `system` field (Anthropic, set by anthropic_to_body) or system-role messages
    (OpenAI). List-shaped content is flattened — OpenAI content parts use the same
    {"type":"text"} block shape Anthropic does. Joined into one string so it can be
    appended to the house prompt rather than silently dropped."""
    if body.get("system"):
        return body["system"]
    parts = []
    for m in (body.get("messages") or []):
        if isinstance(m, dict) and m.get("role") == "system":
            c = anthropic_text(m.get("content"))
            if c and c.strip():
                parts.append(c)
    return "\n\n".join(parts)


# ---- Anthropic <-> OpenAI shape adapters -------------------------------------
def anthropic_text(content):
    """Anthropic message content -> plain text.

    Content is a bare string OR a list of typed blocks. Non-text blocks are dropped
    rather than stringified: the router reads the question, and an image placeholder in
    the prompt is worse than its absence."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for b in content:
            if isinstance(b, str):
                out.append(b)
            elif isinstance(b, dict) and b.get("type") == "text":
                out.append(b.get("text") or "")
        return chr(10).join(p for p in out if p)
    return ""


def anthropic_tools_to_openai(tools):
    """Anthropic tool definitions -> OpenAI function definitions.

    Anthropic: {name, description, input_schema}
    OpenAI:    {type:"function", function:{name, description, parameters}}
    OpenRouter speaks the OpenAI shape, so this runs on the way out."""
    out = []
    for t in (tools or []):
        if not isinstance(t, dict) or not t.get("name"):
            continue
        out.append({"type": "function", "function": {
            "name": t["name"],
            "description": t.get("description") or "",
            "parameters": t.get("input_schema") or {"type": "object", "properties": {}},
        }})
    return out


def openai_calls_to_anthropic(tool_calls):
    """OpenAI tool_calls -> Anthropic tool_use content blocks.

    Arguments arrive as a JSON *string* from OpenAI and must be an object in Anthropic's
    `input`. Models emitting malformed argument JSON is common enough to handle
    explicitly: the block is still returned, with empty input, rather than dropped - the
    client sees the attempt and can reply with an error instead of hanging forever on a
    tool call that never arrives."""
    blocks = []
    for c in (tool_calls or []):
        fn = (c or {}).get("function") or {}
        try:
            args = json.loads(fn.get("arguments") or "{}")
            if not isinstance(args, dict):
                args = {}
        except Exception:
            args = {}
        blocks.append({"type": "tool_use",
                       "id": c.get("id") or ("toolu_" + uuid.uuid4().hex[:20]),
                       "name": fn.get("name") or "", "input": args})
    return blocks


def anthropic_to_body(req):
    """Anthropic Messages request -> the OpenAI-ish body the rest of serve() understands,
    so routing and history are shared rather than forked.

    Reconciled here: content blocks flattened to strings (clean_history requires str and
    would otherwise drop every turn); the top-level `system` carried through as a real
    system prompt (returned under body["system"], merged with the house prompt at the worker
    call — it used to be faked as a first user turn, which the worker read as conversation);
    and tool_use / tool_result blocks converted to OpenAI's tool_calls / role="tool"
    shape. That last one carries tool_call_id, without which a model cannot match a
    result to the call it made and a tool loop cannot proceed."""
    msgs = []
    sys_prompt = anthropic_text(req.get("system") or "").strip()
    for m in (req.get("messages") or []):
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        if role not in ("user", "assistant"):
            continue
        content = m.get("content")
        blocks = content if isinstance(content, list) else []
        calls = [b for b in blocks if isinstance(b, dict) and b.get("type") == "tool_use"]
        results = [b for b in blocks if isinstance(b, dict) and b.get("type") == "tool_result"]
        text = anthropic_text(content)

        if role == "assistant" and calls:
            msgs.append({"role": "assistant", "content": text or None,
                         "tool_calls": [{"id": c.get("id"), "type": "function",
                                         "function": {"name": c.get("name") or "",
                                                      "arguments": json.dumps(c.get("input") or {})}}
                                        for c in calls]})
        elif text.strip():
            msgs.append({"role": role, "content": text})

        for r in results:
            msgs.append({"role": "tool", "tool_call_id": r.get("tool_use_id") or "",
                         "content": anthropic_text(r.get("content")) or ""})
    out = {"messages": msgs}
    if sys_prompt:
        out["system"] = sys_prompt
    return out


def anthropic_response(req_id, text, meta):
    """The non-streaming Anthropic Messages response envelope.

    `model` reports the model that ACTUALLY answered, not the fugal/auto that was
    requested - the product is that those differ, and a client logging the model should
    log the truth. stop_reason is tool_use when the turn ended in tool calls, which is
    the signal every agent harness branches on to run the tool and continue."""
    calls = openai_calls_to_anthropic(meta.get("tool_calls"))
    content = []
    if text:
        content.append({"type": "text", "text": text})
    content.extend(calls)
    if not content:
        content.append({"type": "text", "text": ""})
    return {"id": "msg_" + req_id.replace("req_", ""),
            "type": "message",
            "role": "assistant",
            "model": "fugal/" + str(meta.get("final_model") or "auto"),
            "content": content,
            "stop_reason": "tool_use" if calls else "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": int(meta.get("input_tokens") or 0),
                      "output_tokens": int(meta.get("output_tokens") or 0)},
            "fugal": meta}


def clean_history(body, max_msgs=12, max_chars=24000):
    """Sanitize a chat request's messages[:-1] into a worker-safe history list:
    only user/assistant text turns (content parts flattened), each capped, oldest
    dropped past the budget."""
    hist = []
    for m in (body.get("messages") or [])[:-1]:
        if not isinstance(m, dict):
            continue
        role, content = m.get("role"), m.get("content")
        if isinstance(content, list):
            content = anthropic_text(content)      # OpenAI parts share the text-block shape
        if role in ("user", "assistant") and isinstance(content, str) and content.strip():
            hist.append({"role": role, "content": content[:8000]})
    hist = hist[-max_msgs:]
    while hist and sum(len(m["content"]) for m in hist) > max_chars:
        hist.pop(0)
    return hist


def routing_query(messages, fallback=""):
    """What the ROUTER reads: the latest user turn with non-empty text.

    Never a tool result and never assistant text — the head was fit on user questions,
    and on a tool-loop continuation the last message is a tool RESULT (a directory
    listing, file contents), which the head cannot score. Routing on the originating
    user turn instead also makes the decision deterministic across a whole tool loop:
    same turn, same hidden state, same worker — sticky routing with no state."""
    for m in reversed(messages or []):
        if isinstance(m, dict) and m.get("role") == "user":
            t = anthropic_text(m.get("content")) if not isinstance(m.get("content"), str) \
                else m["content"]
            if t and t.strip():
                return t
    return fallback


# ---- the routing inspector: what the router would do, for $0 -----------------
def print_ranking(f2, query):
    """The ranked table. Local forward pass only — no API call, no key, no spend."""
    ranked, probs = f2.route(query)
    mc = dict(zip(f2.models, f2.mean_cost, strict=True))
    print(f"  {'model':<32} {'p_solve':>8} {'$/query':>9} {'utility':>8}")
    for i, (m, p) in enumerate(zip(ranked, probs, strict=True)):
        bar = "#" * int(p * 20)
        mark = " <-- WORKER" if i == 0 else ""
        print(f"  {m:<32} {p:>8.3f} {mc[m]:>9.5f} {p - f2.lam * mc[m]:>8.3f}  {bar}{mark}")


def route_once(query, models=None, router_lambda=None):
    print_ranking(Fugal(models=models, router_lambda=router_lambda), query)


def route_repl(models=None, router_lambda=None):
    f2 = Fugal(models=models, router_lambda=router_lambda)
    print("Fugal route inspector (free, no API calls). Type a query, or /quit.")
    while True:
        try:
            q = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q or q == "/quit":
            break
        print_ranking(f2, q)
    print("bye")


# ---- streaming from the worker -----------------------------------------------
class StreamAccount:
    """What one streamed worker call accumulates as chunks arrive: the usage block
    (OpenRouter sends it in its LAST chunk) and tool-call fragments, which stream as
    index-keyed deltas that must be concatenated before a client can act on them."""

    def __init__(self):
        self.usage = {}
        self.settled = False
        self._tcalls = {}

    def add_tool_delta(self, tc):
        idx = tc.get("index", 0)
        cur = self._tcalls.setdefault(idx, {"id": "", "type": "function",
                                            "function": {"name": "", "arguments": ""}})
        if tc.get("id"):
            cur["id"] = tc["id"]
        fn = tc.get("function") or {}
        if fn.get("name"):
            cur["function"]["name"] = fn["name"]
        if fn.get("arguments"):
            cur["function"]["arguments"] += fn["arguments"]

    def tool_calls(self):
        return [self._tcalls[i] for i in sorted(self._tcalls)]


async def worker_stream(payload, acct):
    """POST `payload` to OpenRouter with stream=True and yield text deltas as they arrive.
    Tool-call fragments and the usage block accumulate on `acct`. An upstream or transport
    failure is yielded as bracketed text, so the client sees what happened inline instead of
    a stream that just stops. Module-level so tests can substitute a fake upstream."""
    key = os.environ.get("FUGAL_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
    try:
        async with httpx.AsyncClient() as client, client.stream(
                "POST", OR_URL, json=payload, timeout=180.0,
                headers={"Authorization": f"Bearer {key}"}) as response:
            if response.status_code != 200:
                body = (await response.aread()).decode(errors="replace")[:200]
                yield f"[upstream error {response.status_code}: {body}]"
                return
            async for line in response.aiter_lines():
                line = line.strip()
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if chunk.get("usage"):
                    acct.usage = chunk["usage"]
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                if delta.get("content"):
                    yield delta["content"]
                for tc in delta.get("tool_calls") or []:
                    acct.add_tool_delta(tc)
    except Exception as exc:
        yield f"\n\n[stream error: {str(exc)[:100]}]"


# ---- the app -----------------------------------------------------------------
def build_app(f2, *, host="127.0.0.1", token=None, spend_cap=None, daily_cap=None,
              rate_limit=6, cors_origins=None, allow_hosts=None):
    """OpenAI- and Anthropic-compatible ASGI app over one router.

    Every request is routed to ONE model and that model's answer is returned, once —
    streaming, tool calling and multi-turn included. Routing (the torch forward pass)
    runs in a thread so it never blocks the event loop; worker streams are relayed from
    OpenRouter as they arrive.

    Tiers:
      - POST /v1/route — $0 local forward pass only, no token needed, rate-limited.
      - POST /v1/chat/completions | /v1/messages without a token — allowed ONLY when
        daily_cap is set; counts against a per-UTC-day budget + per-IP rate limit.
      - With the `token` bearer — bypasses the daily cap and rate limit (still bounded
        by the cumulative spend_cap safety net).

    Browser defences, both ON by default:
      - No CORS headers unless an origin is explicitly allowed.
      - When `host` is loopback, the Host header must name loopback."""
    cors = set(cors_origins or [])
    loopback_bind = host in LOOPBACK
    hosts_ok = LOOPBACK | {h.lower() for h in (allow_hosts or [])}
    # The meter is touched ONLY from the event loop, by functions that never await, so
    # every read-modify-write is atomic with respect to other coroutines — no lock. This
    # is load-bearing: add_spend must also be callable from a generator's cleanup path
    # (client disconnected mid-stream), where awaiting is not an option.
    meter = {"spent": 0.0, "n": 0, "day": None, "day_spent": 0.0}
    hits: dict[str, deque] = {}
    hits_lock = asyncio.Lock()

    def _day_spent():
        d = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if meter["day"] != d:
            meter["day"], meter["day_spent"] = d, 0.0
        return meter["day_spent"]

    def add_spend(cost, public):
        meter["spent"] += cost
        meter["n"] += 1
        if public:
            _day_spent()
            meter["day_spent"] += cost

    def capped():
        return spend_cap is not None and meter["spent"] >= spend_cap

    async def rate_limited(ip, limit=rate_limit, window=60.0):
        now = time.time()
        async with hits_lock:
            if len(hits) > 10000:
                for k in [k for k, dq in hits.items() if not dq or now - dq[-1] > window]:
                    del hits[k]
            dq = hits.setdefault(ip, deque())
            while dq and now - dq[0] > window:
                dq.popleft()
            if len(dq) >= limit:
                return True
            dq.append(now)
        return False

    def cors_headers(request):
        origin = request.headers.get("origin")
        if not origin or not cors:
            return {}
        if "*" in cors:
            return {"access-control-allow-origin": "*"}
        if origin in cors:
            return {"access-control-allow-origin": origin, "vary": "Origin"}
        return {}

    def _json(code, obj, extra_headers=None):
        return JSONResponse(obj, status_code=code,
                            headers={"connection": "close", **(extra_headers or {})})

    def _err(code, msg, typ="invalid_request_error"):
        return _json(code, {"error": {"message": msg, "type": typ}})

    def authed(request):
        if not token:
            return True
        h = request.headers.get("authorization", "")
        given = h[7:].strip() if h.startswith("Bearer ") else \
            (request.headers.get("x-api-key") or "").strip()
        return hmac.compare_digest(given, token)

    def client_ip(request):
        peer = request.client.host if request.client else "127.0.0.1"
        if peer in ("127.0.0.1", "::1"):
            xff = request.headers.get("x-forwarded-for")
            if xff:
                return xff.split(",")[0].strip()
        return peer

    async def read_json(request):
        """-> (parsed body, None) or (None, error response). Bounded by Content-Length
        before a byte of body is read; a chunked request with no length is refused too."""
        n = int(request.headers.get("content-length") or 0)
        if n <= 0 or n > MAX_BODY:
            return None, _err(400, "empty or oversized request body")
        try:
            return await request.json(), None
        except Exception:
            return None, _err(400, "body must be valid JSON")

    def read_body(raw, anthropic=False):
        """-> (worker query, routing query, body). The worker query is the last message
        (what the model must respond to); the routing query is the latest USER text turn
        (what the head can score) — they differ on tool-loop continuations."""
        if anthropic:
            body = anthropic_to_body(raw)
            body["stream"] = bool(raw.get("stream"))
            body["max_tokens"] = raw.get("max_tokens")
            body["temperature"] = raw.get("temperature")
            if raw.get("tools"):
                body["tools"] = anthropic_tools_to_openai(raw.get("tools"))
        else:
            body = raw
        if body.get("messages"):
            last = body["messages"][-1]
            q = last.get("content") if isinstance(last, dict) else None
            if isinstance(q, list):
                q = anthropic_text(q)          # OpenAI content parts, same block shape
        else:
            q = body.get("query")
        if not isinstance(q, str) or not q.strip():
            raise ValueError("query must be a non-empty string")
        return q, routing_query(body.get("messages"), fallback=q), body

    def worker_messages(q, hist, tools, msgs, sys_prompt):
        """The message list the worker sees. With tools, the client's list goes through
        verbatim (tool_call_id linkage must survive); the house prompt leads either way and
        any stray system turns are dropped so there is exactly one system message."""
        if tools:
            base = msgs if msgs is not None else list(hist or []) + [{"role": "user", "content": q}]
            return [{"role": "system", "content": sys_prompt}] \
                + [m for m in base if m.get("role") != "system"]
        return [{"role": "system", "content": sys_prompt}, *(hist or []),
                {"role": "user", "content": q}]

    def log(kind, meta, public, hist):
        print(f"[serve] {kind}  worker={meta['final_model']}  ${meta['cost']:.5f}  "
              f"p_solve={meta.get('p_solve', 0):.2f}  hist={len(hist or [])}  "
              f"tier={'public' if public else 'full'}", flush=True)

    # ---- route handlers ----------------------------------------------------------

    async def health(request):
        ds = _day_spent()
        return _json(200, {"status": "ok", "queries": meter["n"],
                           "spend_usd": round(meter["spent"], 5),
                           "spend_cap_usd": spend_cap,
                           "daily_cap_usd": daily_cap,
                           "day_spent_usd": round(ds, 5)})

    async def models_list(request):
        return _json(200, {"object": "list", "data": [
            {"id": "fugal/auto", "object": "model", "owned_by": "fugal",
             "description": "Routes each request to the model with the best "
                            "odds-vs-price trade for that question."}]})

    async def route_only(request):
        if await rate_limited("r|" + client_ip(request), limit=max(rate_limit * 2, 12)):
            return _err(429, "rate limit exceeded; try again in a minute", "rate_limit_error")
        raw, err = await read_json(request)
        if err:
            return err
        try:
            _, rq, _ = read_body(raw)
        except Exception:
            return _err(400, "body must be JSON with messages[].content or a query field")
        if len(rq) > 8000:
            return _err(400, "query too long for the router (8000 chars max)")
        ranked, probs = await asyncio.to_thread(f2.route, rq)
        mc = {m: float(c) for m, c in zip(f2.models, f2.mean_cost, strict=True)}
        return _json(200, {
            "object": "fugal.route",
            "worker": ranked[0], "p_solve": round(float(probs[0]), 4),
            "lambda": f2.lam,
            "ranked": [{"model": m, "p_solve": round(float(p), 4),
                        "est_cost_usd": round(mc[m], 6),
                        "utility": round(float(p) - f2.lam * mc[m], 4)}
                       for m, p in zip(ranked, probs, strict=True)]})

    async def completions(request, anthropic):
        """Shared handler for the OpenAI and Anthropic wire shapes."""
        public = not authed(request)
        req_id = "req_" + uuid.uuid4().hex[:20]
        if public:
            if daily_cap is None:
                return _err(401, "missing or invalid bearer token", "authentication_error")
            if await rate_limited("c|" + client_ip(request)):
                return _err(429, "rate limit exceeded; try again in a minute",
                            "rate_limit_error")
            if _day_spent() >= daily_cap:
                return _err(402, "the daily budget is used up — resets at midnight UTC; "
                                 "route-only mode (POST /v1/route) stays free",
                            "spend_cap_error")
        if not (os.environ.get("FUGAL_API_KEY") or os.environ.get("OPENROUTER_API_KEY")):
            return _err(503, "no OpenRouter key configured (set FUGAL_API_KEY); "
                             "POST /v1/route works without one", "server_error")
        raw, err = await read_json(request)
        if err:
            return err
        try:
            q, rq, body = read_body(raw, anthropic=anthropic)
        except Exception:
            return _err(400, "body must be JSON with messages[].content or a query field")
        if public and len(q) > 4000:
            return _err(400, "query too long for the public tier (4000 chars max)")
        if capped():
            return _err(402, f"spend cap ${spend_cap} reached; restart to reset",
                        "spend_cap_error")

        hist = clean_history(body, max_msgs=8, max_chars=8000) if public \
            else clean_history(body)
        tools = body.get("tools") or None
        msgs = body.get("messages") if tools else None
        csys = client_system_from_body(body)
        mtok = body.get("max_tokens")
        # None = the caller sent no temperature, and none is forwarded — each worker
        # keeps its provider default rather than inheriting an opinion of ours.
        temp = body.get("temperature")
        if temp is not None:
            try:
                temp = float(temp)
            except (TypeError, ValueError):
                temp = None

        if body.get("stream"):
            gen = (stream_anthropic if anthropic else stream_openai)(
                q, rq, public, hist, tools, msgs, csys, mtok, temp)
            return StreamingResponse(gen, media_type="text/event-stream",
                                     headers={"cache-control": "no-cache",
                                              "connection": "close"})

        reply, meta = await asyncio.to_thread(
            f2.answer, q, history=hist, tools=tools, messages=msgs,
            system=csys, max_tokens=mtok, temperature=temp, route_on=rq)
        add_spend(meta["cost"], public)
        log("ok", meta, public, hist)
        extra = {"x-fugal-cost-usd": f"{meta['cost']:.6f}"}
        if anthropic:
            return _json(200, anthropic_response(req_id, reply, meta), extra)
        tcalls = meta.get("tool_calls") or []
        message = {"role": "assistant", "content": reply or None}
        if tcalls:
            message["tool_calls"] = tcalls
        return _json(200, {"id": req_id, "object": "chat.completion",
                           "model": f"fugal/{meta['final_model']}",
                           "choices": [{"index": 0,
                                        "finish_reason": "tool_calls" if tcalls else "stop",
                                        "message": message}],
                           "fugal": meta}, extra)

    async def guarded(request, anthropic):
        try:
            return await completions(request, anthropic)
        except Exception as e:
            traceback.print_exc()
            return _err(500, f"internal error: {str(e)[:120]}", "server_error")

    async def chat_completions(request):
        return await guarded(request, anthropic=False)

    async def messages(request):
        return await guarded(request, anthropic=True)

    # ---- streaming ---------------------------------------------------------------
    # Both wire shapes share one upstream: route, open the worker stream, relay text as it
    # arrives, then settle. Only the SSE framing differs.

    async def open_stream(q, rq, hist, tools, msgs, csys, mtok, temperature):
        """Route once and build the worker payload. -> (model, meta, payload, acct)."""
        ranked, probs = await asyncio.to_thread(
            f2.route, rq, history=hist if f2.head_context == "multiturn" else None)
        first = ranked[0]
        meta = {"p_solve": float(probs[0]), "ranked": list(ranked[:3]),
                "final_model": first, "cost": 0.0, "cost_source": None, "steps": [],
                "input_tokens": 0, "output_tokens": 0}
        payload = {"model": first,
                   "max_tokens": clamp_max_tokens(mtok, f2.max_out_tokens.get(first)),
                   "messages": worker_messages(q, hist, tools, msgs, compose_system(csys)),
                   "stream": True, "stream_options": {"include_usage": True},
                   "usage": {"include": True}}
        if temperature is not None:
            payload["temperature"] = temperature
        if tools:
            payload["tools"] = tools
        return first, meta, payload, StreamAccount()

    def settle(acct, meta, public, hist, kind):
        """Charge the meter for a stream, exactly once: on the normal path, AND from
        `finally` when the client disconnects mid-stream (Starlette closes the generator),
        so streamed spend still counts against the caps. usage (with OpenRouter's own
        charge in `cost`) is the upstream's LAST chunk, so a very early disconnect can
        still under-count — the hard limit on the OpenRouter key itself remains the
        outermost brake (SECURITY.md)."""
        if acct.settled:
            return
        acct.settled = True
        cost, src = call_cost(f2.prices, meta["final_model"], acct.usage)
        meta.update(cost=cost, cost_source=src,
                    input_tokens=int(acct.usage.get("prompt_tokens") or 0),
                    output_tokens=int(acct.usage.get("completion_tokens") or 0))
        meta["steps"].append({"role": "worker", "model": meta["final_model"], "cost": cost})
        tcalls = acct.tool_calls()
        if tcalls:
            meta["tool_calls"] = tcalls
        add_spend(cost, public)
        log(kind, meta, public, hist)

    async def stream_openai(q, rq, public, hist, tools, msgs, csys, mtok, temperature):
        first, meta, payload, acct = await open_stream(q, rq, hist, tools, msgs, csys, mtok,
                                                       temperature)

        def sse(delta, finish=None, **extra):
            return (f"data: " + json.dumps({
                "id": cid, "object": "chat.completion.chunk", "model": f"fugal/{first}",
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
                **extra}) + "\n\n").encode()

        cid = "chatcmpl-" + uuid.uuid4().hex[:24]
        try:
            yield sse({"role": "assistant", "content": ""},
                      fugal={"stage": "route", "model": first, "p_solve": meta["p_solve"]})
            async for text in worker_stream(payload, acct):
                yield sse({"content": text})
            settle(acct, meta, public, hist, "stream ok")
            tcalls = meta.get("tool_calls")
            if tcalls:
                yield sse({"tool_calls": tcalls})
            yield sse({}, finish="tool_calls" if tcalls else "stop", fugal={"meta": meta})
            yield b"data: [DONE]\n\n"
        finally:
            settle(acct, meta, public, hist, "stream ok")

    async def stream_anthropic(q, rq, public, hist, tools, msgs, csys, mtok, temperature):
        first, meta, payload, acct = await open_stream(q, rq, hist, tools, msgs, csys, mtok,
                                                       temperature)

        def sse(kind, obj):
            return f"event: {kind}\ndata: {json.dumps(obj)}\n\n".encode()

        def block(index, kind, blk):
            return sse(kind, {"type": kind, "index": index, **blk})

        mid = "msg_" + uuid.uuid4().hex[:24]
        try:
            yield sse("message_start", {"type": "message_start", "message": {
                "id": mid, "type": "message", "role": "assistant",
                "model": f"fugal/{first}", "content": [],
                "stop_reason": None, "stop_sequence": None,
                "usage": {"input_tokens": 0, "output_tokens": 0}}})
            yield block(0, "content_block_start", {"content_block": {"type": "text", "text": ""}})
            yield sse("ping", {"type": "ping", "fugal": {
                "stage": "route", "model": first, "p_solve": meta["p_solve"]}})
            async for text in worker_stream(payload, acct):
                yield block(0, "content_block_delta",
                            {"delta": {"type": "text_delta", "text": text}})
            yield block(0, "content_block_stop", {})
            settle(acct, meta, public, hist, "anthropic stream ok")
            tcalls = openai_calls_to_anthropic(meta.get("tool_calls"))
            for i, blk in enumerate(tcalls, start=1):
                yield block(i, "content_block_start", {"content_block": {
                    "type": "tool_use", "id": blk["id"], "name": blk["name"], "input": {}}})
                yield block(i, "content_block_delta", {"delta": {
                    "type": "input_json_delta", "partial_json": json.dumps(blk["input"])}})
                yield block(i, "content_block_stop", {})
            yield sse("message_delta", {
                "type": "message_delta",
                "delta": {"stop_reason": "tool_use" if tcalls else "end_turn",
                          "stop_sequence": None},
                "usage": {"output_tokens": meta["output_tokens"]},
                "fugal": meta})
            yield sse("message_stop", {"type": "message_stop"})
        finally:
            settle(acct, meta, public, hist, "anthropic stream ok")

    # ---- CORS / Host middleware (raw ASGI) ---------------------------------------

    def host_name(request):
        h = (request.headers.get("host") or "").strip().lower()
        if h.startswith("["):
            return h[1:h.find("]")] if "]" in h else h
        return h.rsplit(":", 1)[0] if h.count(":") == 1 else h

    async def middleware(scope, receive, send):
        if scope["type"] != "http":
            await inner(scope, receive, send)
            return
        request = Request(scope)
        if loopback_bind and host_name(request) not in hosts_ok:
            await _err(403, "Host header not allowed; reach this server as localhost, "
                            "or pass --allow-host", "permission_error")(scope, receive, send)
            return
        extra = cors_headers(request)
        if request.method == "OPTIONS":
            if request.headers.get("origin") and not extra:
                await _err(403, "origin not allowed; start the server with "
                                "--cors-origin https://your.app",
                           "permission_error")(scope, receive, send)
                return
            await Response(status_code=204, headers={
                **extra,
                "access-control-allow-methods": "GET, POST, OPTIONS",
                "access-control-allow-headers":
                    "Authorization, Content-Type, x-api-key, anthropic-version",
                "connection": "close"})(scope, receive, send)
            return
        if not extra:
            await inner(scope, receive, send)
            return

        async def send_with_cors(message):
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                headers += [(k.encode(), v.encode()) for k, v in extra.items()]
                message = {**message, "headers": headers}
            await send(message)

        await inner(scope, receive, send_with_cors)

    inner = Starlette(routes=[
        Route("/health", health, methods=["GET"]),
        Route("/v1/models", models_list, methods=["GET"]),
        Route("/v1/route", route_only, methods=["POST"]),
        Route("/route", route_only, methods=["POST"]),
        Route("/v1/chat/completions", chat_completions, methods=["POST"]),
        Route("/chat/completions", chat_completions, methods=["POST"]),
        Route("/v1/messages", messages, methods=["POST"]),
        Route("/messages", messages, methods=["POST"]),
    ])
    return middleware


def serve(port, host="127.0.0.1", spend_cap=None, daily_cap=None, rate_limit=6,
          router_lambda=None, models=None, cors_origins=None, allow_hosts=None):
    """Load the router, build the app, print what it is doing, and serve it with uvicorn."""
    f2 = Fugal(models=models, router_lambda=router_lambda)
    token = os.environ.get("FUGAL_SERVE_TOKEN")
    app = build_app(f2, host=host, token=token, spend_cap=spend_cap, daily_cap=daily_cap,
                    rate_limit=rate_limit, cors_origins=cors_origins, allow_hosts=allow_hosts)

    auth = "bearer-auth ON" if token else "no auth (bind localhost only)"
    cap = f"spend cap ${spend_cap}" if spend_cap is not None else "no spend cap"
    pub = (f"public tier ON: ${daily_cap}/day, {rate_limit}/min/IP"
           if daily_cap is not None else "public tier OFF (token required for answers)")
    fmt = f"{f2.head_format} head" + (" (routing costs frozen at fit time — "
                                      "docs/HEAD_FORMAT.md)" if f2.head_format == "v1" else "")
    cors = sorted(set(cors_origins or []))
    hosts = sorted(LOOPBACK | {h.lower() for h in (allow_hosts or [])})
    print(f"Fugal serving on http://{host}:{port}/v1/chat/completions  [{auth}; {cap}; {pub}]")
    print(f"  routing among {len(f2.models)} models, lambda={f2.lam:g}, "
          f"head context={f2.head_context}, {fmt}")
    if f2.head.provenance or f2.head.backbone_revision:
        print(f"  head: {f2.head.provenance or '(no provenance)'}; backbone "
              f"{f2.head.backbone_revision[:12] or 'revision not declared'}")
    print(f"  browser: CORS {'allowed for ' + ', '.join(cors) if cors else 'OFF'}; "
          f"Host check {'ON (' + ', '.join(hosts) + ')' if host in LOOPBACK else 'OFF (non-loopback bind)'}")
    if not (os.environ.get("FUGAL_API_KEY") or os.environ.get("OPENROUTER_API_KEY")):
        print("  ⚠ no FUGAL_API_KEY set — POST /v1/route works, answers will fail")
    print("  GET /health · GET /v1/models")
    print("  POST /v1/chat/completions (OpenAI shape) · POST /v1/messages (Anthropic shape)")
    print("  POST /v1/route ($0, no key)")

    import uvicorn
    try:
        uvicorn.run(app, host=host, port=port, log_level="warning")
    except KeyboardInterrupt:
        print("\nshutting down")


def main():
    use_utf8()
    ap = argparse.ArgumentParser(
        prog="python -m fugal",
        description="Fugal — one forward pass picks the model, then calls it once.")
    ap.add_argument("--route", nargs="?", const="", metavar="QUERY",
                    help="show the ranked routing table ($0, no API call). With a QUERY it is "
                         "one-shot; bare, it opens an interactive inspector.")
    ap.add_argument("--query", help="route once, then answer with the chosen model "
                                    "(spends OpenRouter credit)")
    ap.add_argument("--serve", action="store_true",
                    help="run the OpenAI + Anthropic compatible endpoint")
    ap.add_argument("--models", metavar="A,B,C",
                    help="route among a SUBSET of the head's models (also FUGAL_MODELS). "
                         "Exact, not approximate — see docs/HEAD_FORMAT.md.")
    ap.add_argument("--router-lambda", type=float, default=None,
                    help="override the head's cost sensitivity (also FUGAL_LAMBDA; higher "
                         "= prefer cheaper workers on easy queries; unset = the trained "
                         "default)")
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--spend-cap", type=float, default=None,
                    help="serve only: cumulative USD ceiling; past it the server returns 402. "
                         "A brake, not an accounting record: it is checked before each call, "
                         "so concurrent requests can overshoot it slightly, it resets on "
                         "restart, and it counts what OpenRouter reports it charged for "
                         "each call (the price sheet only when that is missing). Set a hard "
                         "limit on the OpenRouter key as well.")
    ap.add_argument("--cors-origin", action="append", default=None, metavar="ORIGIN",
                    help="serve only: allow browser requests from this origin (repeatable; "
                         "also FUGAL_CORS_ORIGINS, comma-separated). Default is NO CORS at "
                         "all, which is what stops an unrelated web page from spending your "
                         "credit. '*' allows everything.")
    ap.add_argument("--allow-host", action="append", default=None, metavar="HOST",
                    help="serve only: extra value accepted in the Host header. Only enforced "
                         "on a loopback bind, where it is the DNS-rebinding defence; "
                         "localhost and 127.0.0.1 are always allowed.")
    ap.add_argument("--daily-cap", type=float, default=None,
                    help="serve only: enable the tokenless PUBLIC tier with this shared "
                         "USD budget per UTC day (unset = answers require the bearer token)")
    ap.add_argument("--rate-limit", type=int, default=6,
                    help="serve only: public-tier requests per minute per IP (default 6)")
    args = ap.parse_args()

    if args.route is not None:
        if args.route.strip():
            route_once(args.route, args.models, args.router_lambda)
        else:
            route_repl(args.models, args.router_lambda)
    elif args.query:
        if not (os.environ.get("FUGAL_API_KEY") or os.environ.get("OPENROUTER_API_KEY")):
            sys.exit("--query calls a model, which needs an OpenRouter key:\n"
                     "  export FUGAL_API_KEY=sk-or-...\n"
                     "Routing itself is free — try --route \"your question\" instead.")
        f2 = Fugal(models=args.models, router_lambda=args.router_lambda)
        reply, meta = f2.answer(args.query, verbose=True)
        print(f"\nanswer ({meta['final_model']}, ${meta['cost']:.5f}):\n{reply}")
    elif args.serve:
        env_cors = [o.strip() for o in
                    (os.environ.get("FUGAL_CORS_ORIGINS") or "").split(",") if o.strip()]
        serve(args.port, host=args.host, spend_cap=args.spend_cap,
              daily_cap=args.daily_cap, rate_limit=args.rate_limit,
              router_lambda=args.router_lambda, models=args.models,
              cors_origins=(args.cors_origin or []) + env_cors,
              allow_hosts=args.allow_host)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
