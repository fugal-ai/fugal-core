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

### Browsers

Two defaults exist because "no auth on localhost with a live key" is the documented setup, and
that is exactly what a hostile web page can reach:

- **No CORS headers are sent** unless you allow an origin: `--cors-origin https://your.app`
  (repeatable, or `FUGAL_CORS_ORIGINS=a,b`). Without this the browser refuses to hand the
  response to the page, so a drive-by `fetch()` cannot read your answers — or spend your
  credit getting them. A preflight from an unlisted origin gets a 403 rather than a silent
  failure, so a legitimate web app finds out what to configure.
- **The `Host` header must name loopback** when bound to loopback. This is the DNS-rebinding
  defence: a page on `evil.example` whose DNS answer is `127.0.0.1` satisfies same-origin
  policy and CORS never enters into it, but it still arrives with `Host: evil.example`. Add
  legitimate names with `--allow-host`. The check is off for non-loopback binds, where a
  reverse proxy is forwarding a real public hostname.

Neither applies to a CLI, an SDK, or a terminal agent — `curl`, the OpenAI SDK, Claude Code
and OpenClaw all work unchanged.

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
- **Routing reads the latest *user* turn.** Never a tool result and never history — the
  shipped head declares it was fit on standalone questions (`context` in the artifact, see
  `docs/HEAD_FORMAT.md`), and feeding it anything else makes it route worse, silently. Two
  consequences worth knowing: a whole tool loop routes on the user turn that started it, so
  one model handles the loop end to end; and on a follow-up turn ("now in Rust") the routing
  signal is weaker than on a fresh question. A head retrained on multi-turn transcripts
  declares `context="multiturn"` and gets history automatically.
- **`temperature` is passed through when the caller sends one**, and omitted otherwise so
  each worker keeps its own provider default.
- **Streamed spend is settled even if the client disconnects mid-stream**, but the token
  usage arrives in the upstream's final chunk, so a very early disconnect can under-count.
  The hard limit on the OpenRouter key itself is the outermost brake.
- **Reported cost is only as good as the price sheet.** `X-Fugal-Cost-USD`, the `fugal.cost`
  field and the spend caps are all computed from `data/models_2026-06.json`. Providers move
  their prices; `python scripts/refresh_prices.py` re-syncs it, and `--check` reports drift
  without writing. The `mean_cost` used for *routing* is frozen with the head on purpose and
  is not touched.
- **Cost decides most routes.** With the shipped head, `p_solve` is high and nearly flat
  across the 17 models (the README's "What this does not claim" has the numbers), so at
  **λ=2.0** the cheapest capable models win most queries: `"hi"`, `"what is 15% of 240?"`
  and `"reverse a linked list in python"` all route to `deepseek/deepseek-v4-flash`, and
  still do at `--router-lambda 10`. A pricier model wins only where the head sees a real
  gap. Use `--models` to bound the pool if you want a floor on capability rather than a
  ceiling on cost.
