#!/usr/bin/env python3
# Fugal — Apache-2.0. See NOTICE.
"""
serve.py — the CLI and the HTTP server. The router itself is router.py.

Usage:
  python -m fugal --route                             # interactive routing inspector ($0)
  python -m fugal --route "what is 15% of 240?"       # one-shot, same table ($0)
  python -m fugal --query "..."                       # route once, answer once (spends credit)
  python -m fugal --serve --port 8090 --spend-cap 5.00
      # OpenAI + Anthropic compatible endpoint. Threaded; optional bearer auth via
      # FUGAL_SERVE_TOKEN; --spend-cap is a hard cumulative USD ceiling.
      # GET /health, GET /v1/models, POST /v1/route ($0, no key), POST /v1/chat/completions,
      # POST /v1/messages.
      # Browsers are locked out by default: no CORS headers unless --cors-origin says so,
      # and on a loopback bind the Host header must name loopback (DNS rebinding).

Add --models "a,b,c" (or set FUGAL_MODELS) to route among a subset of the head's 17 models —
useful when you hold keys for some providers and not others. See docs/HEAD_FORMAT.md.

Env: FUGAL_API_KEY (OpenRouter, live calls only), FUGAL_MODEL (backbone directory).
"""
from __future__ import annotations
import argparse, hmac, json, os, sys, threading, time, uuid

from ._console import use_utf8
from .router import Fugal, clamp_max_tokens, compose_system, or_request, or_call  # noqa: F401


def client_system_from_body(body):
    """The caller's own system prompt, pulled from wherever their client shape put it: a
    top-level `system` field (Anthropic, set by anthropic_to_body) or system-role messages
    (OpenAI). Joined into one string so it can be appended to the house prompt rather than
    silently dropped, which is what clean_history() used to do to it."""
    if body.get("system"):
        return body["system"]
    parts = [m.get("content") for m in (body.get("messages") or [])
             if isinstance(m, dict) and m.get("role") == "system"
             and isinstance(m.get("content"), str)]
    return "\n\n".join(p for p in parts if p and p.strip())


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
    only user/assistant string turns, each capped, oldest dropped past the budget."""
    hist = []
    for m in (body.get("messages") or [])[:-1]:
        if not isinstance(m, dict):
            continue
        role, content = m.get("role"), m.get("content")
        if role in ("user", "assistant") and isinstance(content, str) and content.strip():
            hist.append({"role": role, "content": content[:8000]})
    hist = hist[-max_msgs:]
    while hist and sum(len(m["content"]) for m in hist) > max_chars:
        hist.pop(0)
    return hist


# ---- the routing inspector: what the router would do, for $0 -----------------
def print_ranking(f2, query):
    """The ranked table. Local forward pass only — no API call, no key, no spend."""
    ranked, probs = f2.route(query)
    mc = {m: c for m, c in zip(f2.models, f2.mean_cost)}
    print(f"  {'model':<32} {'p_solve':>8} {'$/query':>9} {'utility':>8}")
    for i, (m, p) in enumerate(zip(ranked, probs)):
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


# ---- the server --------------------------------------------------------------
def serve(port, host="127.0.0.1", spend_cap=None, daily_cap=None, rate_limit=6,
          router_lambda=None, models=None, cors_origins=None, allow_hosts=None):
    """OpenAI- and Anthropic-compatible endpoint. Threaded, optional bearer auth
    (FUGAL_SERVE_TOKEN), a hard cumulative spend cap, and /health + /v1/models.

    Every request is routed to ONE model and that model's answer is returned, once —
    streaming, tool calling and multi-turn included.

    Tiers:
      - POST /v1/route — $0 local forward pass only, no token needed, rate-limited.
      - POST /v1/chat/completions | /v1/messages without a token — allowed ONLY when
        --daily-cap is set; counts against a per-UTC-day budget + per-IP rate limit.
      - With the FUGAL_SERVE_TOKEN bearer — bypasses the daily cap and rate limit
        (still bounded by the cumulative --spend-cap safety net).

    Browser defences, both ON by default because the documented way to run this is
    `--serve` on localhost with NO token and a live OpenRouter key:
      - No CORS headers are sent unless an origin is explicitly allowed
        (`cors_origins` / --cors-origin / FUGAL_CORS_ORIGINS). Reflecting whatever
        Origin arrived would let any page the user happens to visit spend their
        credit from the browser and read the answer back.
      - When bound to loopback, the Host header must name loopback (or something
        passed to `allow_hosts`). That is what stops DNS rebinding: an attacker
        resolving evil.example to 127.0.0.1 arrives with Host: evil.example."""
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from collections import deque
    from datetime import datetime, timezone
    f2 = Fugal(models=models, router_lambda=router_lambda)
    token = os.environ.get("FUGAL_SERVE_TOKEN")          # None => no auth (localhost)
    cors = set(cors_origins or [])
    # Host checking applies only to a loopback bind. Bound to 0.0.0.0 the server is
    # behind a reverse proxy (docs/INTEGRATION.md), which forwards the public hostname — there
    # is no rebinding to defend against and a fixed allowlist would just reject traffic.
    loopback_bind = host in ("127.0.0.1", "localhost", "::1")
    hosts_ok = {"localhost", "127.0.0.1", "::1"} | {h.lower() for h in (allow_hosts or [])}
    meter = {"spent": 0.0, "n": 0, "day": None, "day_spent": 0.0}
    mlock = threading.Lock()
    hits = {}                                            # ip -> deque[timestamps]
    hlock = threading.Lock()
    MAX_BODY = 256 * 1024
    ROUTES = ("/v1/chat/completions", "/chat/completions")
    # Anthropic Messages shape. Same router and same caps — only the request and response
    # envelopes differ. Claude Code speaks this and NOT the OpenAI shape, which is why one
    # endpoint was never enough.
    ANTHROPIC_ROUTES = ("/v1/messages", "/messages")

    def utc_day():
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def day_spent():                                    # call under mlock
        d = utc_day()
        if meter["day"] != d:
            meter["day"], meter["day_spent"] = d, 0.0
        return meter["day_spent"]

    def rate_limited(ip, limit=rate_limit, window=60.0):
        """Sliding-window per-IP limiter. Returns True when over the limit."""
        now = time.time()
        with hlock:
            if len(hits) > 10000:                       # bound memory under IP churn
                hits.clear()
            dq = hits.setdefault(ip, deque())
            while dq and now - dq[0] > window:
                dq.popleft()
            if len(dq) >= limit:
                return True
            dq.append(now)
        return False

    def add_spend(cost, public):
        with mlock:
            meter["spent"] += cost
            meter["n"] += 1
            if public:
                day_spent()                             # rolls the day if needed
                meter["day_spent"] += cost

    class H(BaseHTTPRequestHandler):
        def _json(self, code, obj, extra_headers=()):
            data = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Connection", "close")
            for k, v in self._cors():
                self.send_header(k, v)
            for k, v in extra_headers:
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(data)

        def _cors(self):
            """CORS headers — ONLY for an origin that was explicitly allowed.

            Returning nothing is the safe default: without Access-Control-Allow-Origin
            the browser refuses to hand the response back to the page, so a drive-by
            fetch from an unrelated site cannot read answers (or preflight its way to
            sending an Authorization header). Pass --cors-origin to opt a real web app
            in; '*' is accepted but means any page on the internet."""
            origin = self.headers.get("Origin")
            if not origin or not cors:
                return []
            if "*" in cors:
                return [("Access-Control-Allow-Origin", "*")]
            if origin in cors:
                return [("Access-Control-Allow-Origin", origin), ("Vary", "Origin")]
            return []

        def _host_ok(self):
            """Host header names this machine (loopback binds only) — anti-DNS-rebinding.

            A page on evil.example whose DNS answer is 127.0.0.1 reaches this server with
            the browser's same-origin policy satisfied; CORS never enters into it. The
            Host header is the one field that still carries the attacker's name."""
            if not loopback_bind:
                return True
            h = (self.headers.get("Host") or "").strip().lower()
            if not h:
                return False
            if h.startswith("["):                       # [::1]:8090
                name = h[1:h.find("]")] if "]" in h else h
            else:                                       # host:port, or a bare IPv6 literal
                name = h.rsplit(":", 1)[0] if h.count(":") == 1 else h
            return name in hosts_ok

        def _guard(self):
            """True if the request may proceed; otherwise the 403 has already been sent."""
            if self._host_ok():
                return True
            self._err(403, "Host header not allowed; reach this server as localhost, "
                           "or pass --allow-host", "permission_error")
            return False

        def _err(self, code, msg, typ="invalid_request_error"):
            self._json(code, {"error": {"message": msg, "type": typ}})

        def _auth_value(self):
            """The caller's credential, from whichever header their client uses.
            Anthropic SDKs and Claude Code send `x-api-key`; OpenAI-shaped clients send
            `Authorization: Bearer`. Accepting both is what lets one token work everywhere."""
            h = self.headers.get("Authorization", "")
            if h.startswith("Bearer "):
                return h[7:].strip()
            return (self.headers.get("x-api-key") or "").strip()

        def _authed(self):
            # compare_digest, not ==, so a network attacker cannot walk the token out
            # one byte at a time off response timing — this token is the only thing between
            # a reachable port and a live OpenRouter key.
            if not token:
                return True
            return hmac.compare_digest(self._auth_value(), token)

        def _client_ip(self):
            # trust X-Forwarded-For only when the peer is the local reverse proxy
            peer = self.client_address[0]
            if peer in ("127.0.0.1", "::1"):
                xff = self.headers.get("X-Forwarded-For")
                if xff:
                    return xff.split(",")[0].strip()
            return peer

        def _capped(self):
            with mlock:
                return spend_cap is not None and meter["spent"] >= spend_cap

        def do_OPTIONS(self):
            if not self._guard():
                return
            hdrs = self._cors()
            if self.headers.get("Origin") and not hdrs:
                # Fail the preflight explicitly rather than 204-ing without the header:
                # the browser blocks the real request either way, but a 403 tells whoever
                # is configuring a legitimate web app that --cors-origin is what they want.
                return self._err(403, "origin not allowed; start the server with "
                                      "--cors-origin https://your.app", "permission_error")
            self.send_response(204)
            for k, v in hdrs:
                self.send_header(k, v)
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers",
                             "Authorization, Content-Type, x-api-key, anthropic-version")
            self.send_header("Connection", "close")
            self.end_headers()

        def do_GET(self):
            if not self._guard():
                return
            path = self.path.rstrip("/") or "/"
            if path == "/health":
                with mlock:
                    ds = day_spent()
                    m = dict(meter)
                return self._json(200, {"status": "ok", "queries": m["n"],
                                        "spend_usd": round(m["spent"], 5),
                                        "spend_cap_usd": spend_cap,
                                        "daily_cap_usd": daily_cap,
                                        "day_spent_usd": round(ds, 5)})
            if path == "/v1/models":
                # "fugal/auto" IS the product — one id meaning "you pick nothing, we route".
                return self._json(200, {"object": "list", "data": [
                    {"id": "fugal/auto", "object": "model", "owned_by": "fugal",
                     "description": "Routes each request to the model with the best "
                                    "odds-vs-price trade for that question."}]})
            return self._err(404, "not found", "not_found_error")

        def _read_query(self, anthropic=False):
            """Parse the request body; returns (query, body) or None (error already sent).

            With anthropic=True the Anthropic Messages envelope is normalised first, so
            everything downstream — history, routing, metering — is shape-agnostic."""
            n = int(self.headers.get("Content-Length") or 0)
            if n <= 0 or n > MAX_BODY:
                self._err(400, "empty or oversized request body")
                return None
            try:
                raw = json.loads(self.rfile.read(n))
                if anthropic:
                    body = anthropic_to_body(raw)
                    # `stream` is the one top-level field the shared path still reads; it is
                    # carried across explicitly rather than by merging the whole raw body,
                    # so an Anthropic-only field can never reach OpenAI-shaped handling.
                    body["stream"] = bool(raw.get("stream"))
                    # max_tokens is REQUIRED on an Anthropic request; carry it across
                    # explicitly like `stream` so the worker honours the caller's budget.
                    body["max_tokens"] = raw.get("max_tokens")
                    # Tools are translated on the way in so only ONE shape exists past
                    # this point; OpenRouter speaks OpenAI's.
                    if raw.get("tools"):
                        body["tools"] = anthropic_tools_to_openai(raw.get("tools"))
                else:
                    body = raw
                if "messages" in body:
                    q = body["messages"][-1]["content"]
                else:
                    q = body["query"]
                assert isinstance(q, str) and q.strip()
            except Exception:
                self._err(400, "body must be JSON with messages[].content or a query field")
                return None
            return q, body

        def _route_only(self):
            """$0 tier: local router forward pass only, never any API call. No token needed."""
            if rate_limited("r|" + self._client_ip(), limit=max(rate_limit * 2, 12)):
                return self._err(429, "rate limit exceeded; try again in a minute",
                                 "rate_limit_error")
            parsed = self._read_query()
            if parsed is None:
                return
            q, _ = parsed
            if len(q) > 8000:
                return self._err(400, "query too long for the router (8000 chars max)")
            ranked, probs = f2.route(q)
            mc = {m: float(c) for m, c in zip(f2.models, f2.mean_cost)}
            return self._json(200, {
                "object": "fugal.route",
                "worker": ranked[0], "p_solve": round(float(probs[0]), 4),
                "lambda": f2.lam,
                "ranked": [{"model": m, "p_solve": round(float(p), 4),
                            "est_cost_usd": round(mc[m], 6),
                            "utility": round(float(p) - f2.lam * mc[m], 4)}
                           for m, p in zip(ranked, probs)]})

        def do_POST(self):
            try:
                if not self._guard():
                    return
                path = self.path.rstrip("/") or "/"
                if path in ("/v1/route", "/route"):
                    return self._route_only()
                anthropic = path in ANTHROPIC_ROUTES
                if path not in ROUTES and not anthropic:
                    return self._err(404, "unknown route; use POST /v1/chat/completions, "
                                          "POST /v1/messages, or POST /v1/route",
                                     "not_found_error")
                # tiers: valid token (or no token configured) = full access;
                # tokenless requests are the PUBLIC tier — only allowed when --daily-cap
                # is set, and subject to per-IP rate limit + the shared daily budget.
                public = not self._authed()
                req_id = "req_" + uuid.uuid4().hex[:20]
                if public:
                    if daily_cap is None:
                        return self._err(401, "missing or invalid bearer token",
                                         "authentication_error")
                    if rate_limited("c|" + self._client_ip()):
                        return self._err(429, "rate limit exceeded; try again in a minute",
                                         "rate_limit_error")
                    with mlock:
                        exhausted = day_spent() >= daily_cap
                    if exhausted:
                        return self._err(402, "the daily budget is used up — resets at "
                                              "midnight UTC; route-only mode (POST /v1/route) "
                                              "stays free", "spend_cap_error")
                parsed = self._read_query(anthropic=anthropic)
                if parsed is None:
                    return
                q, body = parsed
                if public and len(q) > 4000:
                    return self._err(400, "query too long for the public tier (4000 chars max)")
                if self._capped():
                    return self._err(402, f"spend cap ${spend_cap} reached; restart to reset",
                                     "spend_cap_error")
                # public tier carries less context — history tokens are billed too
                hist = (clean_history(body, max_msgs=8, max_chars=8000) if public
                        else clean_history(body))
                # With tools present the FULL message list goes through verbatim - the
                # usual flatten-to-a-query-string destroys tool_call_id linkage, which is
                # the only thing tying a result back to the call that produced it.
                tools = body.get("tools") or None
                msgs = body.get("messages") if tools else None
                # The caller's system prompt, from whichever field their shape uses. It is
                # merged after the house prompt at the worker call rather than dropped
                # (OpenAI shape) or faked as a user turn (Anthropic shape), which is what an
                # agent harness like Claude Code depends on to behave at all.
                csys = client_system_from_body(body)
                mtok = body.get("max_tokens")

                if body.get("stream"):
                    if anthropic:
                        return self._stream_anthropic(q, public, hist, tools, msgs, csys, mtok)
                    return self._stream(q, public, hist, tools, msgs, csys, mtok)

                reply, meta = f2.answer(q, history=hist, tools=tools, messages=msgs,
                                        system=csys, max_tokens=mtok)
                add_spend(meta["cost"], public)
                if anthropic:
                    return self._json(200, anthropic_response(req_id, reply, meta),
                                      extra_headers=[("X-Fugal-Cost-USD",
                                                      f"{meta['cost']:.6f}")])
                tcalls = meta.get("tool_calls") or []
                message = {"role": "assistant", "content": reply or None}
                if tcalls:
                    message["tool_calls"] = tcalls
                resp = {"id": req_id, "object": "chat.completion",
                        "model": f"fugal/{meta['final_model']}",
                        "choices": [{"index": 0,
                                     "finish_reason": "tool_calls" if tcalls else "stop",
                                     "message": message}],
                        "fugal": meta}
                self._json(200, resp, extra_headers=[("X-Fugal-Cost-USD", f"{meta['cost']:.6f}")])
                print(f"[serve] ok  worker={meta['final_model']}  ${meta['cost']:.5f}  "
                      f"p_solve={meta.get('p_solve', 0):.2f}  "
                      f"hist={len(hist)}  tier={'public' if public else 'full'}", flush=True)
            except BrokenPipeError:
                pass
            except Exception as e:
                import traceback; traceback.print_exc()   # journald: 500s must be debuggable
                try:
                    self._err(500, f"internal error: {str(e)[:120]}", "server_error")
                except Exception:
                    pass

        def _stream(self, q, public=False, history=None, tools=None, messages=None,
                    system=None, max_tokens=None):
            cid = "chatcmpl-" + uuid.uuid4().hex[:24]
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            for k, v in self._cors():
                self.send_header(k, v)
            self.end_headers()

            def sse(obj):
                self.wfile.write(f"data: {json.dumps(obj)}\n\n".encode())
                self.wfile.flush()

            def chunk(delta, finish=None, extra=None):
                o = {"id": cid, "object": "chat.completion.chunk", "model": "fugal/auto",
                     "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
                if extra:
                    o["fugal"] = extra
                sse(o)

            try:
                chunk({"role": "assistant", "content": ""})
                final = None
                for ev in f2.answer_iter(q, history=history, tools=tools, messages=messages,
                                         system=system, max_tokens=max_tokens):
                    if ev["stage"] == "final":
                        final = ev
                        break
                    chunk({}, extra={"stage": ev})     # live waterfall event (OpenAI extension)
                text, meta = final["content"], final["meta"]
                words = text.split(" ")
                for i in range(0, len(words), 5):       # stream content in small groups
                    chunk({"content": (" " if i else "") + " ".join(words[i:i + 5])})
                tcalls = meta.get("tool_calls") or []
                if tcalls:
                    chunk({"tool_calls": tcalls})
                chunk({}, finish="tool_calls" if tcalls else "stop", extra={"meta": meta})
                self.wfile.write(b"data: [DONE]\n\n"); self.wfile.flush()
                add_spend(meta["cost"], public)
                print(f"[serve] stream ok  worker={meta['final_model']}  ${meta['cost']:.5f}  "
                      f"p_solve={meta.get('p_solve', 0):.2f}  hist={len(history or [])}  "
                      f"tier={'public' if public else 'full'}", flush=True)
            except BrokenPipeError:
                pass

        def _stream_anthropic(self, q, public=False, history=None, tools=None, messages=None,
                              system=None, max_tokens=None):
            """SSE in the Anthropic Messages event vocabulary.

            Deliberately NOT a reskin of _stream: the two protocols differ in more than
            field names. Anthropic sends a named `event:` line alongside each `data:`,
            wraps text in indexed content blocks that must be opened and closed, carries
            stop_reason on a message_delta, and has NO [DONE] sentinel - a client that
            waits for one hangs forever. Getting any of those wrong looks like a working
            stream right up until the client never terminates."""
            mid = "msg_" + uuid.uuid4().hex[:24]
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            for k, v in self._cors():
                self.send_header(k, v)
            self.end_headers()

            def sse(kind, obj):
                # SSE framing: "event: <name>", "data: <json>", blank line. chr(10) rather
                # than an escape so the framing survives any codegen that rewrites this file.
                lf = chr(10)
                self.wfile.write(("event: " + kind + lf
                                  + "data: " + json.dumps(obj) + lf + lf).encode())
                self.wfile.flush()

            try:
                sse("message_start", {"type": "message_start", "message": {
                    "id": mid, "type": "message", "role": "assistant",
                    "model": "fugal/auto", "content": [],
                    "stop_reason": None, "stop_sequence": None,
                    "usage": {"input_tokens": 0, "output_tokens": 0}}})
                sse("content_block_start", {"type": "content_block_start", "index": 0,
                                            "content_block": {"type": "text", "text": ""}})
                # Routing happens before any text exists. Anthropic has no event for "still
                # working", so the waterfall is emitted as ping events, which every client
                # is required to tolerate and ignore.
                final = None
                for ev in f2.answer_iter(q, history=history, tools=tools, messages=messages,
                                         system=system, max_tokens=max_tokens):
                    if ev["stage"] == "final":
                        final = ev
                        break
                    sse("ping", {"type": "ping", "fugal": ev})
                text, meta = final["content"], final["meta"]
                words = text.split(" ")
                for i in range(0, len(words), 5):
                    sse("content_block_delta", {
                        "type": "content_block_delta", "index": 0,
                        "delta": {"type": "text_delta",
                                  "text": (" " if i else "") + " ".join(words[i:i + 5])}})
                sse("content_block_stop", {"type": "content_block_stop", "index": 0})

                # Tool calls ride as their own indexed content blocks after the text one.
                # The whole argument object goes in a SINGLE input_json_delta rather than
                # being chunked: we already have the complete JSON, and emitting partial
                # fragments we never actually received would be inventing a stream that
                # did not happen. Clients accumulate either way.
                tcalls = openai_calls_to_anthropic(meta.get("tool_calls"))
                for i, blk in enumerate(tcalls, start=1):
                    sse("content_block_start", {
                        "type": "content_block_start", "index": i,
                        "content_block": {"type": "tool_use", "id": blk["id"],
                                          "name": blk["name"], "input": {}}})
                    sse("content_block_delta", {
                        "type": "content_block_delta", "index": i,
                        "delta": {"type": "input_json_delta",
                                  "partial_json": json.dumps(blk["input"])}})
                    sse("content_block_stop", {"type": "content_block_stop", "index": i})

                sse("message_delta", {
                    "type": "message_delta",
                    "delta": {"stop_reason": "tool_use" if tcalls else "end_turn",
                              "stop_sequence": None},
                    "usage": {"output_tokens": int(meta.get("output_tokens") or 0)},
                    "fugal": meta})
                sse("message_stop", {"type": "message_stop"})
                add_spend(meta["cost"], public)
                print(f"[serve] anthropic stream ok  worker={meta['final_model']}  "
                      f"${meta['cost']:.5f}  hist={len(history or [])}  "
                      f"tier={'public' if public else 'full'}", flush=True)
            except BrokenPipeError:
                pass

        def log_message(self, *a):
            pass

    srv = ThreadingHTTPServer((host, port), H)
    auth = "bearer-auth ON" if token else "no auth (bind localhost only)"
    cap = f"spend cap ${spend_cap}" if spend_cap is not None else "no spend cap"
    pub = (f"public tier ON: ${daily_cap}/day, {rate_limit}/min/IP"
           if daily_cap is not None else "public tier OFF (token required for answers)")
    print(f"Fugal serving on http://{host}:{port}/v1/chat/completions  [{auth}; {cap}; {pub}]")
    print(f"  routing among {len(f2.models)} models, lambda={f2.lam:g}")
    print(f"  browser: CORS {'allowed for ' + ', '.join(sorted(cors)) if cors else 'OFF'}"
          f"; Host check {'ON (' + ', '.join(sorted(hosts_ok)) + ')' if loopback_bind else 'OFF (non-loopback bind)'}")
    if not (os.environ.get("FUGAL_API_KEY") or os.environ.get("OPENROUTER_API_KEY")):
        # Say it at startup rather than letting every answer come back as a 401 from
        # OpenRouter that reads like our bug.
        print("  ⚠ no FUGAL_API_KEY set — POST /v1/route works, answers will fail")
    print("  GET /health · GET /v1/models")
    print("  POST /v1/chat/completions (OpenAI shape) · POST /v1/messages (Anthropic shape)")
    print("  POST /v1/route ($0, no key)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down")
        srv.shutdown()


def main():
    # The ranked table, the help text and the error messages use em-dashes; a Windows
    # console defaults to cp1252 and raises on them. Only when run as a program: importing
    # fugal.serve must not reach in and rebind a library user's streams.
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
                    help="override the head's cost sensitivity (higher = prefer cheaper "
                         "workers on easy queries; unset = the trained default)")
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--spend-cap", type=float, default=None,
                    help="serve only: cumulative USD ceiling; past it the server returns 402. "
                         "A brake, not an accounting record: it is checked before each call, "
                         "so concurrent requests can overshoot it slightly, it resets on "
                         "restart, and it counts the price sheet's numbers — run "
                         "scripts/refresh_prices.py to keep those true. Set a hard limit on "
                         "the OpenRouter key as well.")
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
