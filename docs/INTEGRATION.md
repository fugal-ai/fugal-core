# Using Fugal as your day-to-day LLM

Fugal serves two wire shapes from one process, so most clients can point at it without a
shim:

| Endpoint | Shape | Who speaks it |
|---|---|---|
| `POST /v1/chat/completions` | OpenAI **Chat Completions** | OpenAI SDK, OpenClaw, most gateways |
| `POST /v1/messages` | **Anthropic Messages** | Claude Code, Anthropic SDKs |
| `POST /v1/route` | Fugal-only, **$0** | routing inspector — never calls a paid model |

Both paths are also mounted without the `/v1` prefix (`/chat/completions`, `/messages`), so a
client that appends its own prefix still lands correctly.

One request → one routed model → one answer. Streaming, tool calling, and multi-turn all cost
exactly one model call per turn; `verify/verify_routing.py` is the regression that keeps it
that way.

---

## 1. Start the server

```bash
python -m fugal --serve --port 8090 --spend-cap 5.00
```

`FUGAL_API_KEY` (an OpenRouter key) must be set for answers. Routing does not need it.

**Always pass `--spend-cap`.** It is a cumulative USD ceiling for the process; past it the
server returns 402 instead of spending. Restarting resets it.

### Auth

By default there is **no auth** — bind to localhost only. To require a token:

```bash
export FUGAL_SERVE_TOKEN=some-long-random-string
```

Clients may send it as either `Authorization: Bearer <token>` or `x-api-key: <token>`;
accepting both is what lets one credential work across OpenAI- and Anthropic-shaped clients.

With a token set, tokenless requests are refused with 401 — unless you also pass `--daily-cap`,
which opens an anonymous public tier bounded by that shared per-UTC-day budget and the per-IP
`--rate-limit`. `/v1/route` stays free and open either way.

Check it came up the way you expect:

```bash
curl -s localhost:8090/health
curl -s -X POST localhost:8090/v1/route -H 'Content-Type: application/json' \
  -d '{"query":"reverse a linked list in python"}'
```

`/v1/route` costs nothing and is the fastest way to sanity-check routing.

---

## 2. Claude Code — works today

Claude Code speaks Anthropic Messages, which is exactly what `/v1/messages` implements.

```bash
export ANTHROPIC_BASE_URL=http://127.0.0.1:8090
export ANTHROPIC_AUTH_TOKEN=$FUGAL_SERVE_TOKEN   # omit if no token configured
claude
```

Verified working: non-streaming, streaming (correct `message_start` → `content_block_*` →
`message_delta` → `message_stop` event vocabulary, no `[DONE]` sentinel), tool calls in both
directions, and the client's `system` prompt and `max_tokens`.

---

## 3. OpenClaw — works today

OpenClaw takes custom providers in `openclaw.json` under `models.providers`, and supports both
of our shapes via its `api` field.

```json5
{
  agents: {
    defaults: {
      model: { primary: "fugal/auto" }
    }
  },
  models: {
    providers: {
      fugal: {
        baseUrl: "http://127.0.0.1:8090/v1",
        apiKey: "${FUGAL_SERVE_TOKEN}",
        api: "openai-completions",     // or "anthropic-messages"
        timeoutSeconds: 300,
        models: [
          {
            id: "auto",
            name: "Fugal (routed)",
            contextWindow: 200000,
            maxTokens: 8192,
            input: ["text"],
            cost: { input: 0, output: 0 }   // real cost is metered server-side
          }
        ]
      }
    }
  }
}
```

Use `api: "anthropic-messages"` with `baseUrl: "http://127.0.0.1:8090"` if you prefer the
Anthropic path — both are implemented.

---

## 4. Codex CLI — blocked, needs a `/v1/responses` endpoint

**This one does not work today**, and it is not a config problem.

Codex CLI custom providers must declare `wire_api`, and as of Codex 0.122 (Feb 2026) the
`"chat"` value was **removed** — custom providers must use `wire_api = "responses"`, i.e. the
OpenAI **Responses API**. Fugal implements Chat Completions and Anthropic Messages, but **not**
`/v1/responses`, so Codex errors rather than falling back.

Options, cheapest first:

1. **Use Claude Code or OpenClaw instead** — both work now.
2. **Put a Responses→Chat-Completions translating proxy in front of Fugal** (several gateways
   do this).
3. **Implement `POST /v1/responses` in `serve()`** — the same shape-adapter pattern already
   used for `/v1/messages` (`anthropic_to_body` / `anthropic_response`). This is the real fix.

Once implemented, the config would be:

```toml
model = "auto"
model_provider = "fugal"

[model_providers.fugal]
name = "Fugal"
base_url = "http://127.0.0.1:8090/v1"
env_key = "FUGAL_SERVE_TOKEN"
wire_api = "responses"
```

---

## 5. Plain SDK / curl

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8090/v1", api_key="unused-or-your-token")
r = client.chat.completions.create(model="fugal/auto",
                                   messages=[{"role": "user", "content": "hello"}])
print(r.model)                    # fugal/<the model that actually answered>
```

The response carries a `fugal` block with `final_model`, `cost`, `p_solve`, and `steps`, plus
an `X-Fugal-Cost-USD` header — so you can always see who answered and what it cost.

---

## 6. Keeping it running

Nothing here needs a deployment recipe — `--serve` on localhost is the whole thing. If you want
it up across reboots, a systemd unit is enough:

```ini
[Unit]
Description=Fugal router
After=network-online.target

[Service]
WorkingDirectory=/path/to/fugal-core
EnvironmentFile=/etc/fugal.env        # mode 600: FUGAL_API_KEY, FUGAL_SERVE_TOKEN
Environment=FUGAL_MODEL=/path/to/fugal-core/artifacts/Qwen3-0.6B
ExecStart=/path/to/.venv/bin/python -m fugal --serve --port 8090 --spend-cap 25.00
Restart=on-failure
MemoryMax=6G                          # fp32 Qwen3-0.6B is ~2.4 GB; loading peaks above that

[Install]
WantedBy=multi-user.target
```

If you expose it beyond localhost: set `FUGAL_SERVE_TOKEN`, put TLS in front of it, and
remember that the outermost spend brake is the hard limit on the OpenRouter key itself, not
`--spend-cap`. Serving it to anonymous strangers is a different problem than this repo solves.

---

## Known limits

- **Plain-chat history is capped** at 12 messages / 24k chars (`clean_history`), 8/8k on the
  public tier. Requests that carry `tools` bypass this — the full message list is forwarded
  verbatim so tool_call_id linkage survives. Agent harnesses therefore keep full context;
  long *non-tool* chats get truncated from the oldest end.
- **Routing reads the last message only.** The head was fit on standalone questions, so on
  a follow-up (or a tool-result turn) the routing signal is weaker than on a fresh question.
- **`temperature` is fixed at 0.0** and the caller's value is ignored.
- **Cost barely influences routing.** The shipped head is **λ=2.0**, and the accuracy spread
  across the 17 models is wide enough that price acts as a tiebreaker rather than a driver:
  `"hi"` routes to `anthropic/claude-sonnet-5`, and still does at `--router-lambda 10`. Where
  the trade does bite, it bites usefully — on "what is 15% of 240?" a model 0.2 points less
  likely to be right but 44% cheaper wins. Use `--models` to bound cost by construction if
  that matters more to you than accuracy.
