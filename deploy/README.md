# Running Fugal as a service

Target: a CPU-only Linux box (developed on Ubuntu aarch64 / OCI A1 flex). The router is one
forward pass of Qwen3-0.6B in float32 on the CPU; the models it routes to are OpenRouter API
calls, so the box needs RAM and a network, not a GPU.

```
internet ──► Caddy :443 (auto-TLS)
                 │  /v1/*  /health   (reverse proxy)
                 ▼
             python -m fugal --serve  :8090   (systemd: fugal.service)
                 │
                 ▼
             OpenRouter API  (FUGAL_API_KEY, spend-capped)
```

Request tiers, all enforced in `fugal/serve.py`:

- `POST /v1/route` — $0 local forward pass, no token, rate-limited. Always free.
- `POST /v1/chat/completions` (OpenAI) and `POST /v1/messages` (Anthropic) — answers.
  Without the bearer token these are allowed only when `--daily-cap` is set, and are bounded
  by that shared per-UTC-day budget plus the per-IP `--rate-limit`. **A tokenless public tier
  is anonymous** — one visitor can drain the whole daily cap, so size it as money you are
  willing to lose, or drop `--daily-cap` and require the token.
- Bearer `FUGAL_SERVE_TOKEN` — full tier, bypasses the daily cap and rate limit, still bounded
  by the cumulative `--spend-cap`.

## 0. Inventory first (if the box runs other services)

```bash
ss -tlnp                 # 8090 must be free; 80/443 free or shared deliberately
free -h; df -h; nproc    # ~3 GB free RAM (fp32 0.6B ≈ 2.4 GB) + ~3 GB disk
```

## 1. One-time setup

```bash
sudo bash deploy/setup_arm.sh      # idempotent: apt deps, venv, CPU torch, backbone, smoke test
```

The script ends with a router smoke test and prints the forward-pass latency (expect 0.5–2 s on
a modern ARM core). `scripts/fetch_backbone.py` pulls Qwen3-0.6B; the router head ships in the
repo.

## 2. Secrets — `/etc/fugal.env` (never committed, never echoed)

```bash
sudo install -m 600 /dev/null /etc/fugal.env
sudoedit /etc/fugal.env
```

```ini
FUGAL_API_KEY=sk-or-...              # OpenRouter key — also set a hard spend limit on the key
FUGAL_SERVE_TOKEN=<openssl rand -hex 24>
```

## 3. Service

```bash
sudo cp deploy/fugal.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now fugal
curl -s localhost:8090/health        # {"status":"ok", "spend_cap_usd":25.0, ...}
```

Keep the caps (`--daily-cap`, `--rate-limit`, `--spend-cap`) in the `ExecStart`, and keep
`MemoryMax` — check the real high-water mark with `systemctl status fugal` and tighten it.

## 4. Reverse proxy

Put TLS + a hostname in front of `:8090` with `deploy/Caddyfile` (append it to whatever Caddy
owns :80/:443, or run it standalone). It forwards `/v1/*` + `/health` only. See the notes in
that file for the host-vs-container upstream address, and keep `:8090` **closed to the public
internet** — Fugal binds `0.0.0.0` only so a containerised proxy can reach it over the bridge;
a firewall is the boundary. Verify from off-box: `curl --max-time 5 http://PUBLIC_IP:8090/health`
must time out.

## 5. Smoke tests ($0)

```bash
curl -s https://HOST/health
curl -s https://HOST/v1/route -X POST -H 'Content-Type: application/json' \
     -d '{"query":"What is 17*23?"}'                                 # ranked table, no spend

# rate limit: the tail of a burst should turn 429
for i in $(seq 1 8); do curl -s -o /dev/null -w '%{http_code} ' https://HOST/v1/route \
     -X POST -H 'Content-Type: application/json' -d '{"query":"hi"}'; done; echo

# with a token configured and no --daily-cap, an unauthenticated answer must 401
# BEFORE any paid call
curl -s -o /dev/null -w '%{http_code}\n' https://HOST/v1/chat/completions \
     -X POST -H 'Content-Type: application/json' \
     -d '{"model":"fugal/auto","messages":[{"role":"user","content":"hi"}]}'   # expect 401
```

## 6. Update

```bash
cd /opt/fugal-core && sudo -u fugal git pull && sudo systemctl restart fugal
```

## Pre-launch checklist (public, with a live key)

- [ ] Unauthenticated `/v1/chat/completions` returns 401, or `--daily-cap` is a sum you are
      willing to lose to strangers every day.
- [ ] `--daily-cap`, `--rate-limit`, `--spend-cap` are in the unit, and the OpenRouter key has
      its own hard spend limit set on OpenRouter's side (defense in depth).
- [ ] `curl --max-time 5 http://PUBLIC_IP:8090/health` times out (backend not exposed).
- [ ] Router forward-pass latency from §1 recorded.
