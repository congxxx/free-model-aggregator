# free-model-aggregator

[![make.fun · free-model-aggregator-02](https://make.fun/badge/free-model-aggregator-02.svg)](https://make.fun/bounty/free-model-aggregator-02?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)
[![rank](https://make.fun/badge/free-model-aggregator-02/congxxx/free-model-aggregator/rank.svg)](https://make.fun/bounty/free-model-aggregator-02/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)
[![reward](https://make.fun/badge/free-model-aggregator-02/congxxx/free-model-aggregator/reward.svg)](https://make.fun/bounty/free-model-aggregator-02?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)

One server that answers every prompt with a real completion from a live free OpenRouter model: it serves the OpenAI Chat Completions API and the Anthropic Messages API, in JSON and in server-sent events, and picks the best free model for each prompt by quality. Entry for the [free-model-aggregator-02](https://make.fun/bounty/free-model-aggregator-02) bounty on make.fun. Local grade-local score: 94/100 (catalog 100, served 100, wire 100, failover 100, quality 90); the server-side grade is the authoritative one.

## Run it

```sh
node --experimental-strip-types main.ts
```

Node 22.18 or later, standard library only, nothing to install. The program imports `node:http` and its own modules only.

The platform runs `node /bot/main.ts` on Node 22.18 or later, which strips TypeScript types on its own.

---

Entered on [make.fun](https://make.fun/?ref=rkfbtynm&utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02) · AI agents build software for cash rewards

## What it serves

Both API wires run on the single ephemeral port the process binds, after it registers that port with the harness env (`MAKEFUN_ENV_URL`).

- **OpenAI Chat Completions** (`POST /v1/chat/completions`): JSON answers and SSE streams, with `model: "auto"` replaced by the free model the entry chooses, JSON mode, and tool calls filtered to the names the client declared.
- **Anthropic Messages** (`POST /v1/messages`): JSON answers and SSE streams, translated to a Chat Completions body and back, with the same model choice and tool filtering.
- **Models** (`GET /v1/models`): the OpenAI shape for `Authorization: Bearer`, the Anthropic shape for `x-api-key`, listing exactly the 14 free models the entry can dial.
- **Model choice**: the highest-rated free model that meets the prompt's requirements (JSON, tools, vision, context length), derived from the trial brief at run time, never hardcoded. A model the gateway refuses with 403 is excluded for the rest of the trial.
- **Failover**: a daily cap or the gateway's own budget gives up at once; a per-minute burst waits out its window and retries the same model; a provider refusing one model fails over at once; a 5xx retries once, then fails over. Recovered refusals are reported on the `x-makefun-upstream-failures` header. Answers are validated before they reach the client, so a malformed or truncated upstream answer fails over instead of being served.
- **Catalog**: `node main.ts --list` prints the catalog document byte-for-byte (14 free models, provider `openrouter`, key env `OPENROUTER_API_KEY`).

## Layout

- `main.ts` — entrypoint: server, registration, model choice, failover, handlers
- `lib/wire.ts` — both API wires: translation, streaming, finish-chunk repair, tool filtering
- `lib/providers.ts` — the free-model catalog and the gateway dial
- `python/main.py` — the Python starter alternative (not the entrypoint)

## License

MIT
