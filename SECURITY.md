# Security

## Reporting a vulnerability

Use GitHub's [private vulnerability reporting](https://github.com/fugal-ai/fugal-core/security/advisories/new)
on this repository. Please do not open a public issue for anything that could be used
against a running deployment before it is fixed.

Include what you did, what happened, and what you expected. A `curl` that reproduces it is
worth more than a description. Expect an acknowledgement within a week.

## What this software is, in security terms

Fugal is **an HTTP server that holds an API key which spends real money**. That is the
threat model in one sentence, and almost every finding worth reporting is a way to make it
spend, or to make it leak, without the operator's intent.

Two things bound the damage, and both are the operator's job to set:

- `--spend-cap` — a cumulative USD brake for the process. It is checked *before* each call,
  so concurrent requests can overshoot it slightly, and it resets on restart. It is a
  brake, not an accounting record.
- **A hard spend limit on the OpenRouter key itself**, set in OpenRouter's dashboard. This
  is the only limit that survives a bug in this code, and it is the one that matters.

Set both. `--spend-cap` counts what OpenRouter reports it charged for each call
(`usage.cost`), falling back to `data/models_2026-06.json` only when a response carries no
charge; every cost figure says which it used (`cost_source`).

## Defaults you should know about

- **No authentication by default.** With `FUGAL_SERVE_TOKEN` unset, anyone who can reach
  the port can spend. The default bind is `127.0.0.1`. Do not move it off loopback without
  setting a token.
- **No CORS by default.** Browsers are locked out unless you pass `--cors-origin`. This is
  deliberate: Fugal's documented setup is a keyed server on localhost, and reflecting an
  arbitrary `Origin` would let any page you happen to visit spend your credit and read the
  answer back.
- **Host header checked on loopback binds.** A request arriving at `127.0.0.1:8090` with
  `Host: evil.example` is rejected — that is DNS rebinding, and CORS does not stop it. Add
  legitimate names with `--allow-host`. The check is off for non-loopback binds, where a
  reverse proxy forwards a real public hostname.
- **The tokenless public tier is anonymous.** `--daily-cap` opens it; one visitor can drain
  the whole daily budget. Size it as money you are willing to lose, or omit it.
- `POST /v1/route` is always free, needs no token, and never makes an API call. It is rate
  limited per IP.

## Handling of credentials

`FUGAL_API_KEY` / `OPENROUTER_API_KEY` and `FUGAL_SERVE_TOKEN` are read from the
environment, are never written to disk by this code, and are never logged. The systemd
snippet in `docs/INTEGRATION.md` keeps them in `/etc/fugal.env` at mode 600, which is the
recommended shape if you run it as a service. Request bodies are not persisted; the server prints
one summary line per request (worker, cost, history length, tier) and no content.

## Out of scope

- The behaviour of the models Fugal routes to. Their content, their refusals, and their
  own security properties are theirs.
- Prompt injection reaching a worker model through a request body. Fugal is a router: it
  forwards what you send. If you put untrusted text in front of a tool-calling agent, that
  is your trust boundary to defend, not this proxy's.
- Cost overruns from routing decisions you disagree with. Use `--models` to bound the pool
  by construction.
- Anything requiring an attacker who already has your OpenRouter key or shell on the box.

## Supported versions

The latest release on `main`. This is a small project; there are no backported fixes.
