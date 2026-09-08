# Fugal — Apache-2.0. See NOTICE.
"""The HTTP server, driven end to end with a fake router and a fake upstream.

    python -m unittest discover -s tests -v

No backbone, no network, no key, no spend. build_app() takes the router as an argument,
so a two-line stand-in with the same surface (route / answer / prices) is enough to
exercise every endpoint, every tier, both streaming shapes, and the browser defences.
These are the properties SECURITY.md promises; a test here is what stops a refactor from
quietly un-promising one.
"""
import json
import os
import unittest
from unittest import mock

import numpy as np
from starlette.testclient import TestClient

from fugal import serve as S

CHEAP, DEAR = "cheap/model", "dear/model"


class FakeFugal:
    """Just enough of Fugal for the server: a fixed ranking and a fixed-cost answer."""
    models = [CHEAP, DEAR]
    mean_cost = np.array([0.001, 0.010])
    lam = 2.0
    head_context = "standalone"
    max_out_tokens = {CHEAP: 8192}
    prices = {CHEAP: (1e-6, 2e-6), DEAR: (5e-6, 1e-5)}

    def __init__(self, answer_cost=0.0123, tool_calls=None):
        self.answer_cost, self.tool_calls, self.calls = answer_cost, tool_calls, []

    def _price(self, model, itok, otok):
        pin, pout = self.prices[model]
        return itok * pin + otok * pout

    def route(self, query, history=None):
        self.calls.append(("route", query, history))
        return [CHEAP, DEAR], np.array([0.9, 0.8])

    def answer(self, query, **kw):
        self.calls.append(("answer", query, kw))
        meta = {"cost": self.answer_cost, "final_model": CHEAP, "p_solve": 0.9,
                "ranked": [CHEAP, DEAR], "steps": [{"role": "worker", "model": CHEAP,
                                                    "cost": self.answer_cost}],
                "input_tokens": 10, "output_tokens": 5}
        if self.tool_calls:
            meta["tool_calls"] = self.tool_calls
        return "the answer", meta


def fake_upstream(texts=("Hel", "lo"), tool_deltas=(), usage=None, error=None):
    """A stand-in for worker_stream: yields `texts`, feeds `tool_deltas` and `usage` into
    the account the way the real reader does when OpenRouter sends them."""
    async def stream(payload, acct):
        stream.payloads.append(payload)
        if error:
            yield error
            return
        for t in texts:
            yield t
        for tc in tool_deltas:
            acct.add_tool_delta(tc)
        acct.usage = usage if usage is not None else {"prompt_tokens": 100,
                                                      "completion_tokens": 50}
    stream.payloads = []
    return stream


def client(f2=None, **opts):
    """A TestClient that arrives as localhost — both the Host header the loopback check
    requires and the loopback peer address that makes X-Forwarded-For trusted."""
    app = S.build_app(f2 or FakeFugal(), **opts)
    return TestClient(app, base_url="http://localhost", client=("127.0.0.1", 50000))


def sse_events(text):
    """Parse an SSE body into a list of (event or None, data-string)."""
    out = []
    for frame in text.strip().split("\n\n"):
        ev, data = None, None
        for line in frame.split("\n"):
            if line.startswith("event: "):
                ev = line[7:]
            elif line.startswith("data: "):
                data = line[6:]
        out.append((ev, data))
    return out


CHAT = {"messages": [{"role": "user", "content": "what is 15% of 240?"}]}

# Answers need an OpenRouter key; the fake router never calls one, but the server checks.
setUpModule = lambda: os.environ.setdefault("FUGAL_API_KEY", "sk-or-test")   # noqa: E731


class TestFreeEndpoints(unittest.TestCase):
    def test_health_starts_at_zero(self):
        h = client().get("/health").json()
        self.assertEqual(h["status"], "ok")
        self.assertEqual((h["queries"], h["spend_usd"]), (0, 0.0))

    def test_models_lists_the_router(self):
        ids = [m["id"] for m in client().get("/v1/models").json()["data"]]
        self.assertEqual(ids, ["fugal/auto"])

    def test_route_is_free_and_needs_no_token(self):
        c = client(token="secret")
        r = c.post("/v1/route", json={"query": "reverse a linked list"})
        self.assertEqual(r.status_code, 200, r.text)
        j = r.json()
        self.assertEqual(j["worker"], CHEAP)
        self.assertEqual([x["model"] for x in j["ranked"]], [CHEAP, DEAR])
        self.assertAlmostEqual(j["ranked"][0]["utility"], 0.9 - 2.0 * 0.001, places=4)
        self.assertEqual(c.get("/health").json()["spend_usd"], 0.0)

    def test_route_reads_the_latest_user_turn_from_messages(self):
        f2 = FakeFugal()
        c = client(f2)
        c.post("/v1/route", json={"messages": [
            {"role": "user", "content": [{"type": "text", "text": "from parts"}]},
            {"role": "assistant", "content": "a"},
            {"role": "tool", "tool_call_id": "c", "content": "tool output"}]})
        self.assertEqual(f2.calls[-1][1], "from parts")

    def test_route_rejects_bad_bodies(self):
        c = client()
        self.assertEqual(c.post("/v1/route", content=b"{not json",
                                headers={"content-type": "application/json"}).status_code, 400)
        self.assertEqual(c.post("/v1/route", json={}).status_code, 400)
        self.assertEqual(c.post("/v1/route", json={"query": "x" * 8001}).status_code, 400)
        big = {"query": "x" * (S.MAX_BODY + 1)}
        self.assertEqual(c.post("/v1/route", json=big).status_code, 400)

    def test_unprefixed_paths_are_mounted_too(self):
        c = client()
        self.assertEqual(c.post("/route", json={"query": "q"}).status_code, 200)


class TestCompletions(unittest.TestCase):
    def test_openai_shape_reports_the_model_that_answered_and_the_cost(self):
        c = client()
        r = c.post("/v1/chat/completions", json=CHAT)
        self.assertEqual(r.status_code, 200, r.text)
        j = r.json()
        self.assertEqual(j["model"], f"fugal/{CHEAP}")
        self.assertEqual(j["choices"][0]["message"]["content"], "the answer")
        self.assertEqual(j["choices"][0]["finish_reason"], "stop")
        self.assertEqual(r.headers["x-fugal-cost-usd"], "0.012300")
        self.assertEqual(c.get("/health").json()["spend_usd"], 0.0123)

    def test_openai_tool_calls_set_finish_reason(self):
        tc = [{"id": "call_1", "type": "function",
               "function": {"name": "ls", "arguments": "{}"}}]
        j = client(FakeFugal(tool_calls=tc)).post("/v1/chat/completions", json=CHAT).json()
        self.assertEqual(j["choices"][0]["finish_reason"], "tool_calls")
        self.assertEqual(j["choices"][0]["message"]["tool_calls"], tc)

    def test_anthropic_shape(self):
        r = client().post("/v1/messages", json={
            "system": "be terse", "max_tokens": 100,
            "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]})
        self.assertEqual(r.status_code, 200, r.text)
        j = r.json()
        self.assertEqual(j["type"], "message")
        self.assertEqual(j["model"], f"fugal/{CHEAP}")
        self.assertEqual(j["content"], [{"type": "text", "text": "the answer"}])
        self.assertEqual(j["stop_reason"], "end_turn")
        self.assertEqual(j["usage"], {"input_tokens": 10, "output_tokens": 5})

    def test_what_reaches_the_router(self):
        # The worker gets history, the client system prompt, max_tokens, a float
        # temperature, and routes on the latest USER turn — not the tool result.
        f2 = FakeFugal()
        client(f2).post("/v1/chat/completions", json={
            "temperature": "0.3", "max_tokens": 77,
            "messages": [{"role": "system", "content": "be terse"},
                         {"role": "user", "content": "first"},
                         {"role": "assistant", "content": "ok"},
                         {"role": "user", "content": "second"}]})
        kind, q, kw = f2.calls[-1]
        self.assertEqual((kind, q), ("answer", "second"))
        self.assertEqual(kw["history"], [{"role": "user", "content": "first"},
                                         {"role": "assistant", "content": "ok"}])
        self.assertEqual(kw["system"], "be terse")
        self.assertEqual((kw["max_tokens"], kw["temperature"]), (77, 0.3))
        self.assertEqual(kw["route_on"], "second")

    def test_junk_temperature_is_dropped_not_forwarded(self):
        f2 = FakeFugal()
        client(f2).post("/v1/chat/completions", json={**CHAT, "temperature": "hot"})
        self.assertIsNone(f2.calls[-1][2]["temperature"])

    def test_bad_bodies(self):
        c = client()
        self.assertEqual(c.post("/v1/chat/completions", json={"messages": []}).status_code, 400)
        self.assertEqual(c.post("/v1/chat/completions", json={"messages": [
            {"role": "user", "content": "   "}]}).status_code, 400)
        self.assertEqual(c.post("/v1/chat/completions", content=b"",
                                headers={"content-type": "application/json"}).status_code, 400)

    def test_no_key_is_a_clear_503_not_an_internal_error(self):
        with mock.patch.dict(os.environ, {"FUGAL_API_KEY": "", "OPENROUTER_API_KEY": ""}):
            c = client()
            r = c.post("/v1/chat/completions", json=CHAT)
            self.assertEqual(r.status_code, 503)
            self.assertIn("FUGAL_API_KEY", r.json()["error"]["message"])
            self.assertEqual(c.post("/v1/route", json={"query": "still free"}).status_code, 200)

    def test_internal_errors_are_500_not_a_dropped_connection(self):
        f2 = FakeFugal()
        f2.answer = mock.Mock(side_effect=RuntimeError("upstream exploded"))
        with mock.patch.object(S.traceback, "print_exc"):       # keep the test log clean
            r = client(f2).post("/v1/chat/completions", json=CHAT)
        self.assertEqual(r.status_code, 500)
        self.assertIn("upstream exploded", r.json()["error"]["message"])


class TestAuthAndTiers(unittest.TestCase):
    def test_no_token_configured_means_open(self):
        self.assertEqual(client().post("/v1/chat/completions", json=CHAT).status_code, 200)

    def test_token_required_when_configured(self):
        c = client(token="secret")
        self.assertEqual(c.post("/v1/chat/completions", json=CHAT).status_code, 401)
        self.assertEqual(c.post("/v1/chat/completions", json=CHAT,
                                headers={"authorization": "Bearer wrong"}).status_code, 401)
        self.assertEqual(c.post("/v1/chat/completions", json=CHAT,
                                headers={"authorization": "Bearer secret"}).status_code, 200)
        # The Anthropic SDK sends x-api-key; one credential must work for both shapes.
        self.assertEqual(c.post("/v1/messages", json={**CHAT, "max_tokens": 10},
                                headers={"x-api-key": "secret"}).status_code, 200)

    def test_daily_cap_opens_the_public_tier(self):
        c = client(FakeFugal(answer_cost=0.6), token="secret", daily_cap=1.0)
        self.assertEqual(c.post("/v1/chat/completions", json=CHAT).status_code, 200)
        self.assertEqual(c.post("/v1/chat/completions", json=CHAT).status_code, 200)
        r = c.post("/v1/chat/completions", json=CHAT)      # 1.2 spent >= 1.0
        self.assertEqual(r.status_code, 402)
        self.assertEqual(r.json()["error"]["type"], "spend_cap_error")
        # ...but the token holder is not bound by the daily cap.
        self.assertEqual(c.post("/v1/chat/completions", json=CHAT,
                                headers={"authorization": "Bearer secret"}).status_code, 200)
        self.assertEqual(c.get("/health").json()["day_spent_usd"], 1.2)

    def test_public_tier_limits_query_length(self):
        c = client(token="secret", daily_cap=1.0)
        long = {"messages": [{"role": "user", "content": "x" * 4001}]}
        self.assertEqual(c.post("/v1/chat/completions", json=long).status_code, 400)
        self.assertEqual(c.post("/v1/chat/completions", json=long,
                                headers={"authorization": "Bearer secret"}).status_code, 200)

    def test_public_tier_is_rate_limited_per_ip(self):
        c = client(token="secret", daily_cap=100.0, rate_limit=2)
        codes = [c.post("/v1/chat/completions", json=CHAT).status_code for _ in range(3)]
        self.assertEqual(codes, [200, 200, 429])
        # A different client IP (via X-Forwarded-For from the loopback peer) has its own bucket.
        self.assertEqual(c.post("/v1/chat/completions", json=CHAT,
                                headers={"x-forwarded-for": "203.0.113.9"}).status_code, 200)

    def test_spend_cap_is_a_hard_brake(self):
        c = client(FakeFugal(answer_cost=0.02), spend_cap=0.03)
        self.assertEqual(c.post("/v1/chat/completions", json=CHAT).status_code, 200)
        self.assertEqual(c.post("/v1/chat/completions", json=CHAT).status_code, 200)
        r = c.post("/v1/chat/completions", json=CHAT)      # 0.04 >= 0.03
        self.assertEqual(r.status_code, 402)
        self.assertEqual(c.post("/v1/route", json={"query": "still free"}).status_code, 200)


class TestBrowserDefences(unittest.TestCase):
    def test_no_cors_header_unless_an_origin_is_allowed(self):
        r = client().get("/health", headers={"origin": "https://evil.example"})
        self.assertEqual(r.status_code, 200)
        self.assertNotIn("access-control-allow-origin", r.headers)

    def test_preflight_from_an_unlisted_origin_is_refused(self):
        r = client().options("/v1/chat/completions", headers={"origin": "https://evil.example"})
        self.assertEqual(r.status_code, 403)

    def test_allowed_origin_is_honoured(self):
        c = client(cors_origins=["https://good.example"])
        r = c.get("/health", headers={"origin": "https://good.example"})
        self.assertEqual(r.headers["access-control-allow-origin"], "https://good.example")
        self.assertEqual(r.headers["vary"], "Origin")
        r = c.options("/v1/messages", headers={"origin": "https://good.example"})
        self.assertEqual(r.status_code, 204)
        self.assertIn("x-api-key", r.headers["access-control-allow-headers"])
        r = c.get("/health", headers={"origin": "https://other.example"})
        self.assertNotIn("access-control-allow-origin", r.headers)

    def test_wildcard(self):
        r = client(cors_origins=["*"]).get("/health", headers={"origin": "https://any.example"})
        self.assertEqual(r.headers["access-control-allow-origin"], "*")

    def test_rebound_host_is_refused_on_a_loopback_bind(self):
        c = client()
        self.assertEqual(c.get("/health", headers={"host": "evil.example"}).status_code, 403)
        for ok in ("localhost", "127.0.0.1:8090", "[::1]:8090", "LOCALHOST:8090"):
            self.assertEqual(c.get("/health", headers={"host": ok}).status_code, 200, ok)

    def test_allow_host_extends_the_list(self):
        c = client(allow_hosts=["Fugal.Local"])
        self.assertEqual(c.get("/health", headers={"host": "fugal.local:8090"}).status_code, 200)

    def test_host_check_is_off_for_a_public_bind(self):
        c = client(host="0.0.0.0")
        self.assertEqual(c.get("/health", headers={"host": "api.example.com"}).status_code, 200)


class TestStreaming(unittest.TestCase):
    def _openai(self, upstream, f2=None, body=None):
        with mock.patch.object(S, "worker_stream", upstream):
            c = client(f2 or FakeFugal())
            r = c.post("/v1/chat/completions", json={**(body or CHAT), "stream": True})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertTrue(r.headers["content-type"].startswith("text/event-stream"))
        return c, sse_events(r.text)

    def _anthropic(self, upstream, f2=None, body=None):
        with mock.patch.object(S, "worker_stream", upstream):
            c = client(f2 or FakeFugal())
            r = c.post("/v1/messages", json={**(body or CHAT), "max_tokens": 50, "stream": True})
        self.assertEqual(r.status_code, 200, r.text)
        return c, sse_events(r.text)

    def test_openai_stream_shape_and_settlement(self):
        c, ev = self._openai(fake_upstream(texts=("Hel", "lo")))
        chunks = [json.loads(d) for _, d in ev if d != "[DONE]"]
        self.assertEqual(ev[-1], (None, "[DONE]"))
        self.assertEqual(chunks[0]["choices"][0]["delta"]["role"], "assistant")
        self.assertEqual(chunks[0]["fugal"]["stage"], "route")
        text = "".join(ch["choices"][0]["delta"].get("content") or "" for ch in chunks)
        self.assertEqual(text, "Hello")
        self.assertEqual(chunks[-1]["choices"][0]["finish_reason"], "stop")
        meta = chunks[-1]["fugal"]["meta"]
        self.assertEqual((meta["input_tokens"], meta["output_tokens"]), (100, 50))
        # 100 * 1e-6 + 50 * 2e-6, charged to the meter exactly once
        self.assertAlmostEqual(meta["cost"], 0.0002)
        self.assertEqual(c.get("/health").json(), {**c.get("/health").json(),
                                                   "queries": 1, "spend_usd": 0.0002})

    def test_openai_stream_assembles_tool_call_fragments(self):
        deltas = [{"index": 0, "id": "call_1", "function": {"name": "grep", "arguments": '{"pa'}},
                  {"index": 0, "function": {"arguments": 'ttern": "x"}'}}]
        _, ev = self._openai(fake_upstream(texts=(), tool_deltas=deltas))
        chunks = [json.loads(d) for _, d in ev if d != "[DONE]"]
        tc = [ch["choices"][0]["delta"]["tool_calls"] for ch in chunks
              if "tool_calls" in ch["choices"][0]["delta"]]
        self.assertEqual(tc, [[{"id": "call_1", "type": "function",
                                "function": {"name": "grep", "arguments": '{"pattern": "x"}'}}]])
        self.assertEqual(chunks[-1]["choices"][0]["finish_reason"], "tool_calls")

    def test_anthropic_stream_event_vocabulary(self):
        deltas = [{"index": 0, "id": "toolu_1", "function": {"name": "ls", "arguments": '{"p": "/"}'}}]
        c, ev = self._anthropic(fake_upstream(texts=("a", "b"), tool_deltas=deltas))
        kinds = [k for k, _ in ev]
        self.assertEqual(kinds, ["message_start", "content_block_start", "ping",
                                 "content_block_delta", "content_block_delta",
                                 "content_block_stop",
                                 "content_block_start", "content_block_delta",
                                 "content_block_stop", "message_delta", "message_stop"])
        self.assertNotIn("[DONE]", [d for _, d in ev])
        data = [json.loads(d) for _, d in ev]
        self.assertEqual(data[0]["message"]["model"], f"fugal/{CHEAP}")
        self.assertEqual([d["delta"]["text"] for d in data[3:5]], ["a", "b"])
        self.assertEqual(data[6]["content_block"], {"type": "tool_use", "id": "toolu_1",
                                                    "name": "ls", "input": {}})
        self.assertEqual(json.loads(data[7]["delta"]["partial_json"]), {"p": "/"})
        self.assertEqual(data[9]["delta"]["stop_reason"], "tool_use")
        self.assertEqual(data[9]["usage"]["output_tokens"], 50)
        self.assertEqual(c.get("/health").json()["queries"], 1)

    def test_upstream_error_is_reported_inline_and_the_stream_still_closes(self):
        _, ev = self._openai(fake_upstream(error="[upstream error 402: no credit]"))
        chunks = [json.loads(d) for _, d in ev if d != "[DONE]"]
        self.assertIn("upstream error 402", chunks[1]["choices"][0]["delta"]["content"])
        self.assertEqual(chunks[-1]["choices"][0]["finish_reason"], "stop")
        self.assertEqual(ev[-1], (None, "[DONE]"))
        self.assertEqual(chunks[-1]["fugal"]["meta"]["cost"], 0.0)

    def test_worker_payload(self):
        # The upstream sees: one system message (house prompt + client's), the history,
        # the query, the clamped max_tokens, stream + usage flags, and no temperature
        # unless the caller sent one.
        up = fake_upstream()
        self._openai(up, body={"messages": [{"role": "system", "content": "be terse"},
                                            {"role": "user", "content": "q1"},
                                            {"role": "assistant", "content": "a1"},
                                            {"role": "user", "content": "q2"}],
                               "max_tokens": 10 ** 9})
        p = up.payloads[0]
        self.assertEqual(p["model"], CHEAP)
        self.assertEqual(p["max_tokens"], 8192)          # FakeFugal.max_out_tokens[CHEAP]
        self.assertEqual([m["role"] for m in p["messages"]],
                         ["system", "user", "assistant", "user"])
        self.assertTrue(p["messages"][0]["content"].endswith("be terse"))
        self.assertEqual(p["messages"][-1]["content"], "q2")
        self.assertEqual(p["stream_options"], {"include_usage": True})
        self.assertNotIn("temperature", p)

    def test_tools_pass_the_client_messages_through_verbatim(self):
        up = fake_upstream()
        tools = [{"type": "function", "function": {"name": "ls", "parameters": {}}}]
        msgs = [{"role": "user", "content": "list files"},
                {"role": "assistant", "content": None,
                 "tool_calls": [{"id": "c1", "type": "function",
                                 "function": {"name": "ls", "arguments": "{}"}}]},
                {"role": "tool", "tool_call_id": "c1", "content": "a.py b.py"}]
        f2 = FakeFugal()
        self._openai(up, f2=f2, body={"messages": msgs, "tools": tools})
        p = up.payloads[0]
        self.assertEqual(p["tools"], tools)
        self.assertEqual(p["messages"][1:], msgs)         # tool_call_id linkage intact
        self.assertEqual(f2.calls[-1][1], "list files")   # routed on the user turn


class TestStreamAccount(unittest.TestCase):
    def test_fragments_concatenate_in_index_order(self):
        a = S.StreamAccount()
        a.add_tool_delta({"index": 1, "id": "b", "function": {"name": "g", "arguments": "{"}})
        a.add_tool_delta({"index": 0, "id": "a", "function": {"name": "f", "arguments": "{}"}})
        a.add_tool_delta({"index": 1, "function": {"arguments": "}"}})
        a.add_tool_delta({"index": 1, "function": {"arguments": None}})   # tolerated
        self.assertEqual([c["id"] for c in a.tool_calls()], ["a", "b"])
        self.assertEqual(a.tool_calls()[1]["function"]["arguments"], "{}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
