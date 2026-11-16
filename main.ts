// free-model-aggregator-02's starter kit: one server, two client wires, and a real free model behind
// every answer.
//
// This is a working entry to start from, not a competitive answer. It is the house baseline for the
// bounty, deliberately weak, and worth a fraction of the score: use it to see the shape of a trial, then
// replace its routing, its failover and its catalog with your own. The gaps are marked TODO below.
//
// WHAT A TRIAL GIVES YOU
//   A trial runs this server as `node /bot/main.ts` from your repository's root, with Node 22's standard
//   library alone (nothing is installed; TypeScript runs directly, and a relative import needs its .ts
//   extension). The env is at $MAKEFUN_ENV_URL; your model calls go through the grader's gateway at
//   $OPENAI_BASE_URL, an OpenAI-compatible endpoint that reaches OpenRouter on the key you pushed to the
//   competition. The brief at $MAKEFUN_PROMPT_FILE carries two documents:
//     - the free model list: each free OpenRouter model's gateway id
//       (`openrouter/<vendor>/<model>:free`), its published rating in milli-points, what it can do
//       (tools, vision, JSON mode, context length) and its free flag;
//     - the six prompts: each one's id, its api (openai or anthropic), whether it streams, its weight
//       (1 or 2) and the requirements a model must meet to serve it. Each prompt carries a unique token
//       the answer must contain.
//   The server:
//     1. listens on 0.0.0.0 at a free port (never 8080: the env itself is there on one layout);
//     2. POSTs $MAKEFUN_ENV_URL/register {"protocol_version": 1, "port": <port>, "catalog": <catalog>};
//     3. serves GET /v1/models (answering in the shape of the auth header it got: x-api-key means
//        Anthropic, Authorization means OpenAI), POST /v1/chat/completions and POST /v1/messages, each
//        with a JSON body or with server-sent events;
//     4. answers each prompt by dialling ONE free model through the gateway. OPENAI_API_KEY in the
//        sandbox is a dummy the gateway swaps for the OpenRouter key you pushed; the real key never
//        enters the sandbox;
//     5. polls GET $MAKEFUN_ENV_URL/work every 100 ms and exits 0 when the env says {"done": true}. A
//        process still running when the wall ends scores zero, so never wait for anything else.
//   `node main.ts --list` prints the same catalog JSON and exits 0 before any server starts; the
//   register body's catalog is that same document, byte for byte.
//
// THE MODEL RULE
//   Whatever this server reports in a response's `model` field must be the model it actually dialled.
//   The house records every gateway call and scores the model that really answered. This starter sends
//   the brief's model id in the gateway request and puts that same id back on every answer and every
//   streamed chunk (the provider's own answer spells it without the gateway's `openrouter/` prefix), so
//   its claim is the call it made, never another. A prompt whose reported model has no matching
//   successful call in the house's record earns nothing.
//
// THE BASELINE'S WEAKNESSES (each is a scoring component you can win)
//   - quality (60%): it ignores the published ratings. It takes the first free model in the brief, in
//     the brief's order, that can meet the request's requirements. TODO(quality): rank the free
//     candidates by rating_milli at the prompt's requirements and prefer the best.
//   - failover (10%): a gateway call answered with a 4xx or a 5xx becomes a 502 to the client, and that
//     prompt is lost. TODO(failover): retry the prompt on the next free model that meets it, and count
//     every call it costs, because the tie-break is the number of gateway calls.
//   - rate limits: after a 429 it keeps picking the same first model, so one rate-limited model takes
//     every later prompt down with it. TODO(cooldowns): track each model's refusals and back off per
//     model, and never spend a call on a model that is cooling down.
//   - the catalog (5%): it lists every free model the brief names, which is exactly the fourteen ids
//     this column scores. TODO(catalog): keep it exact — register that same set and answer GET /v1/models
//     with it on both auth headers; the column checks the register body and both answers against exactly
//     those fourteen ids, so listing fewer loses it as surely as listing more.
//   The trial's rules are in the kit's RULES.txt; read them for the score's columns and the tie-break.
//
// FILES
//   main.ts            this server: the brief, the catalog, the routes, the register and the poll.
//   lib/providers.ts   the brief's types and parser, the free model list, the catalog.
//   lib/wire.ts        Anthropic <-> Chat Completions translation, the model field, the SSE writers.
//   LICENSE            MIT: keep it or replace it with another OSI-approved license.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  buildCatalog,
  capabilitiesOf,
  freeModels,
  isRecord,
  meetsRequirements,
  readBrief,
  type CatalogReport,
  type Json,
  type ModelRow,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicEventsFromOpenAI,
  anthropicModelList,
  anthropicToOpenAI,
  openAIError,
  openAIJsonWithModel,
  openAIModelList,
  openAIStreamWithModel,
  openAIToAnthropic,
  textStream,
  validateAndRepairOpenAIStream,
  type DeclaredTools,
  type ServedModel,
} from './lib/wire.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** The most bytes of one client request body this baseline reads. */
const MAX_BODY_BYTES = 8 * 1024 * 1024;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';
const GATEWAY_URL = (process.env.OPENAI_BASE_URL ?? '').replace(/\/+$/, '');
/** The trial's dummy key for the entry's gateway lane; the gateway swaps it for the entrant's own key. */
const GATEWAY_KEY = process.env.OPENAI_API_KEY ?? '';

const brief = readBrief(PROMPT_FILE);
const catalog: CatalogReport = buildCatalog(brief);
/** Which prompt row the request now arriving is: the brief's prompts are in the client's arrival order. */
let manifestIndex = 0;

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/**
 * The gateway's Chat Completions route. OPENAI_BASE_URL is the OpenAI-compatible root (`…/v1` on the
 * trial network); a base that does not already end in `/v1` gets one, so both spellings reach the route.
 */
function chatCompletionsUrl(): string {
  if (GATEWAY_URL === '') return '';
  return GATEWAY_URL.endsWith('/v1') ? `${GATEWAY_URL}/chat/completions` : `${GATEWAY_URL}/v1/chat/completions`;
}

/** The free models the brief lists, as GET /v1/models names them. */
function servedModels(): ServedModel[] {
  return freeModels(brief).map((model) => ({ id: model.id, provider: model.provider, display_name: model.display_name }));
}

/** The free models meeting the requirements, best first, before cooldowns and exclusions. */
function rankedFor(requires: Requirement): ModelRow[] {
  return freeModels(brief)
    .filter((model) => meetsRequirements(requires, capabilitiesOf(model)))
    .sort((a, b) => b.rating_milli - a.rating_milli);
}

/**
 * The free models that can serve a request, best first: every free model meeting the requirements,
 * ranked by the brief's rating_milli (descending), ties keeping the brief's order (Array.sort is
 * stable). A model that refused twice in this trial is cooling down and is skipped, and a model the
 * gateway refused with 403 — a policy refusal, not a transient one; the live trial showed
 * inkling-2-small is "only available on agentic harnesses" — is excluded for the rest of the trial,
 * so one dead model cannot take later prompts down with it. Derived from the brief at runtime; no
 * model id is hardcoded, so the same rule picks inkling-2-small (1475) for plain/tools prompts and
 * nemotron-3.5-super-vl (1303, the best JSON-capable model) for JSON-mode prompts on the practice
 * table, and drops to the next best the moment the top model proves undiallable.
 */
function routeFor(requires: Requirement): ModelRow[] {
  return rankedFor(requires).filter((model) => !isCoolingDown(model.id) && !excludedModels.has(model.id));
}

/** Gateway calls spent on one prompt at most: initial + 3 (one same-model retry, two failovers). */
const MAX_ATTEMPTS_PER_PROMPT = 4;
/** Rows of the failover trail at most; the rules cap the header at 8 rows and 8 KiB. */
const MAX_FAILURE_HEADER_ROWS = 8;
const MAX_FAILURE_HEADER_BYTES = 8 * 1024;
/** Pause between attempts, in milliseconds: short enough never to threaten the 30 s client timeout. */
const RETRY_BACKOFF_MS = 100;
/** One gateway call's whole budget, in milliseconds: the env aborts a prompt at 30 s, so a call
 * that has not answered (or finished streaming) by 25 s is a lost prompt either way — fail over
 * instead of lingering. The signal also bounds the body drain, so a stalled stream cannot hang. */
const DIAL_TIMEOUT_MS = 25000;
/** The longest a per-minute rate-limit wait may run, in milliseconds: the retry must still fit
 * the client's 30 s budget after the wait. */
const MAX_RATE_LIMIT_WAIT_MS = 20000;
/** The wait when a per-minute refusal carries no usable reset window: a fixed, header-free guess. */
const FALLBACK_RATE_LIMIT_WAIT_MS = 15000;

/** What kind of refusal a gateway answer is: the failover policy branches on it. */
type RefusalKind = 'daily_cap' | 'per_minute' | 'provider_rate_limit' | 'gateway_budget' | 'policy' | 'transient' | 'other';

/** One refused gateway call, as the failover trail reports it. */
interface UpstreamFailure {
  status: number;
  code: string;
  model: string | null;
  /** Internal: the failover policy's branch. Stripped before the trail header is built. */
  kind: RefusalKind;
  /**
   * Internal: for a per-minute 429, the milliseconds to wait before retrying the same model —
   * computed purely from the response's own reset and date headers, never from the wall clock,
   * so a replay waits exactly what the live pass waited. `null` means the window is more than
   * MAX_RATE_LIMIT_WAIT_MS away: give up instead of waiting.
   */
  waitMs?: number | null;
}

/** Refusals per model id in this trial; a model that refused twice is cooling down. */
const refusalCounts = new Map<string, number>();

/** Models the gateway refused with 403 this trial: a policy refusal, excluded for the trial. */
const excludedModels = new Set<string>();

function isCoolingDown(id: string): boolean {
  return (refusalCounts.get(id) ?? 0) >= 2;
}

function recordRefusal(id: string): void {
  refusalCounts.set(id, (refusalCounts.get(id) ?? 0) + 1);
}

/**
 * What a 429 means depends on why it was refused — the free tier's caps are account-wide, not
 * per-model, so the policy must branch instead of retrying blindly:
 * - a daily cap (`free-models-per-day`, or the daily limit source): nothing on this account can
 *   serve until 00:00 UTC — give up at once;
 * - the gateway's own budget (`request_cap`, `budget_exhausted`): the trial's budget is spent —
 *   give up at once;
 * - a provider refusing one model (`Provider returned error` + "temporarily rate-limited
 *   upstream"): another model is immediately servable — fail over at once;
 * - anything else 429: the account-wide per-minute burst — wait out the window, then retry the
 *   same model once (another model would hit the same cap).
 * A 403 is a policy refusal (the model is excluded for the trial); a 5xx is transient; any
 * other status is neither.
 */
function classifyRefusal(status: number, code: string, message: string, limitSource: string, raw: string): RefusalKind {
  if (status === 403) return 'policy';
  if (status === 429) {
    if (code === 'request_cap' || code === 'budget_exhausted') return 'gateway_budget';
    const hay = `${message} ${limitSource} ${raw}`.toLowerCase();
    if (hay.includes('per-day') || hay.includes('per_day') || limitSource.toLowerCase().includes('daily')) return 'daily_cap';
    if (message.startsWith('Provider returned error') || hay.includes('temporarily rate-limited upstream')) return 'provider_rate_limit';
    return 'per_minute';
  }
  if (status >= 500) return 'transient';
  return 'other';
}

/** A reset instant as milliseconds since the epoch: a seconds epoch, a milliseconds epoch, or an HTTP date. */
function parseResetTime(raw: string): number | null {
  const n = Number(raw);
  if (Number.isFinite(n) && n > 0) return n < 1e11 ? n * 1000 : n;
  const d = Date.parse(raw);
  return Number.isNaN(d) ? null : d;
}

/**
 * How long to wait out a per-minute refusal, from the response's own headers: the reset instant
 * minus the response's date, both recorded in the cassette, so a replay waits exactly what the
 * live pass waited. `null` means give up: the window is more than 20 s away, or there is no
 * reset header to compute a window from at all. A fixed 15 s only when the reset is known but
 * the date is not (the window's end is known, "now" is not).
 */
function perMinuteWaitMs(headers: Headers): number | null {
  const resetRaw = headers.get('x-ratelimit-reset');
  if (resetRaw === null || resetRaw === '') return null;
  const reset = parseResetTime(resetRaw);
  if (reset === null) return null;
  const dateRaw = headers.get('date');
  if (dateRaw === null) return FALLBACK_RATE_LIMIT_WAIT_MS;
  const dateMs = Date.parse(dateRaw);
  if (Number.isNaN(dateMs)) return FALLBACK_RATE_LIMIT_WAIT_MS;
  const remaining = reset - dateMs;
  if (!Number.isFinite(remaining) || remaining <= 0) return 0;
  if (remaining > MAX_RATE_LIMIT_WAIT_MS) return null;
  return remaining;
}

/**
 * A trail code that can ride an HTTP header: Node rejects header values outside Latin-1, and the
 * rules read a trail over 8 KiB as no trail at all, so anything outside Latin-1 or a control
 * character becomes '?', and the code is capped at 64 characters.
 */
function sanitizeTrailCode(code: string): string {
  const latin1 = [...code].map((ch) => ((ch.codePointAt(0) ?? 0) <= 255 ? ch : '?')).join('');
  const cleaned = latin1.replace(/[\u0000-\u001f\u007f]/g, '?');
  return cleaned.length > 64 ? cleaned.slice(0, 64) : cleaned;
}

/** A pause between attempts; it never enters a request, so replays stay byte-identical. */
function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/** One refused gateway response as a trail row: the code from its error body when parseable. */
async function refusalOf(upstream: Response, model: string): Promise<UpstreamFailure> {
  let code = upstream.statusText !== '' ? upstream.statusText : 'upstream_error';
  let message = '';
  let limitSource = '';
  let raw = '';
  let text = '';
  try {
    text = await upstream.text();
  } catch {
    text = '';
  }
  if (text !== '') {
    try {
      const parsed: unknown = JSON.parse(text);
      if (isRecord(parsed) && isRecord(parsed.error)) {
        const errCode = parsed.error.code;
        if (typeof errCode === 'string' && errCode !== '') code = errCode;
        else if (typeof errCode === 'number' && Number.isFinite(errCode)) code = String(errCode);
        else if (typeof parsed.error.type === 'string' && parsed.error.type !== '') code = parsed.error.type;
        if (typeof parsed.error.message === 'string') message = parsed.error.message;
        if (isRecord(parsed.error.metadata)) {
          if (typeof parsed.error.metadata.limit_source === 'string') limitSource = parsed.error.metadata.limit_source;
          if (typeof parsed.error.metadata.raw === 'string') raw = parsed.error.metadata.raw;
        }
      }
    } catch {
      // Not JSON: the HTTP status text stands.
    }
  }
  const kind = classifyRefusal(upstream.status, code, message, limitSource, raw);
  const failure: UpstreamFailure = { status: upstream.status, code: sanitizeTrailCode(code), model, kind };
  if (kind === 'per_minute') failure.waitMs = perMinuteWaitMs(upstream.headers);
  return failure;
}

/** The failover trail for a served prompt: a JSON array of at most 8 rows, at most 8 KiB. */
function failuresHeader(failures: readonly UpstreamFailure[]): string {
  if (failures.length === 0) return '';
  const rows: UpstreamFailure[] = failures
    .slice(0, MAX_FAILURE_HEADER_ROWS)
    .map((f) => ({ status: f.status, code: f.code, model: f.model }));
  let text = JSON.stringify(rows);
  while (rows.length > 1 && Buffer.byteLength(text) > MAX_FAILURE_HEADER_BYTES) {
    rows.pop();
    text = JSON.stringify(rows);
  }
  return text;
}

/** The requirements of the prompt now arriving: the brief's row by arrival order, else the body's own. */
function requirementsFor(body: Json): Requirement {
  const entry = brief.prompts[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  // TODO(workload): the brief's prompt list is the declared source of each request's requirements; this
  // fallback is a guess for a brief that holds none.
  return requirementsFromBody(body);
}

/** Requirements read off a request body, when the brief has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown, extraHeaders: Record<string, string> = {}): void {
  // A client that went away while the answer was being fetched leaves a destroyed response;
  // writing to it would throw, so there is nothing to send.
  if (res.destroyed) return;
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text), ...extraHeaders });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/** POST one request to the gateway's Chat Completions route, naming the model this entry dialled. */
async function dial(model: string, body: Json): Promise<Response> {
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    authorization: `Bearer ${GATEWAY_KEY}`,
  };
  if (body.stream === true) headers.accept = 'text/event-stream';
  return fetch(chatCompletionsUrl(), {
    method: 'POST',
    headers,
    body: JSON.stringify({ ...body, model }),
    signal: AbortSignal.timeout(DIAL_TIMEOUT_MS),
  });
}

/**
 * The models that can serve a request, best first, in tiers. The strict requirements come
 * first; when they have no untried model — every eligible model is cooling down, or none meets
 * them at all — the list relaxes rather than giving up: the same requirements ignoring
 * cooldowns, then the requirements without the context floor, then the best free model
 * overall. A prompt with no eligible model is still served by the best free model rather
 * than answered 503 with zero gateway calls. In the normal case the first tier is exactly
 * routeFor, so the hot path is unchanged.
 */
function candidatesFor(requires: Requirement, tried: ReadonlySet<string>): ModelRow[] {
  const tiers: ModelRow[][] = [routeFor(requires), rankedFor(requires).filter((m) => !excludedModels.has(m.id))];
  if (requires.min_context !== undefined) {
    const { min_context: _dropped, ...rest } = requires;
    tiers.push(routeFor(rest), rankedFor(rest).filter((m) => !excludedModels.has(m.id)));
  }
  tiers.push(routeFor({}), rankedFor({}).filter((m) => !excludedModels.has(m.id)));
  for (const tier of tiers) {
    const list = tier.filter((m) => !tried.has(m.id));
    if (list.length > 0) return list;
  }
  return [];
}

type DialOutcome =
  | { ok: true; model: ModelRow; failures: UpstreamFailure[]; stream: false; json: Json }
  | { ok: true; model: ModelRow; failures: UpstreamFailure[]; stream: true; sseText: string }
  | { ok: false; failures: UpstreamFailure[] };

/** A 200-with-error body as a refusal: the status from its numeric code, else 502. */
function errorBodyFailure(parsed: Json, model: string): UpstreamFailure {
  const err = parsed.error as Json;
  const codeNum = typeof err.code === 'number' ? err.code : null;
  const status = codeNum !== null && codeNum >= 400 && codeNum < 600 ? codeNum : 502;
  const code =
    typeof err.code === 'string' && err.code !== ''
      ? err.code
      : typeof err.code === 'number'
        ? String(err.code)
        : typeof err.type === 'string' && err.type !== ''
          ? err.type
          : 'upstream_error';
  const message = typeof err.message === 'string' ? err.message : '';
  const metadata = isRecord(err.metadata) ? err.metadata : {};
  const limitSource = typeof metadata.limit_source === 'string' ? metadata.limit_source : '';
  const raw = typeof metadata.raw === 'string' ? metadata.raw : '';
  const kind = classifyRefusal(status, code, message, limitSource, raw);
  const failure: UpstreamFailure = { status, code: sanitizeTrailCode(code), model, kind };
  return failure;
}

/**
 * A gateway JSON answer, validated before it is returned as served: an empty or malformed body,
 * a top-level error object, or a body with no usable choice is a refusal the entry fails over
 * from — not a 500 to the client and not a 200 carrying an error. A 200 with an error body
 * keeps the error's own status and kind, so a daily cap inside one still gives up at once.
 */
function validateJsonAnswer(text: string, model: string, headers?: Headers): { ok: true; body: Json } | { ok: false; failure: UpstreamFailure } {
  const malformed: UpstreamFailure = { status: 502, code: 'malformed_response', model, kind: 'transient' };
  if (text.trim() === '') return { ok: false, failure: malformed };
  let parsed: unknown;
  try {
    parsed = JSON.parse(text);
  } catch {
    return { ok: false, failure: malformed };
  }
  if (!isRecord(parsed)) return { ok: false, failure: malformed };
  if (isRecord(parsed.error)) {
    const failure = errorBodyFailure(parsed, model);
    if (failure.kind === 'per_minute' && headers !== undefined) failure.waitMs = perMinuteWaitMs(headers);
    return { ok: false, failure };
  }
  if (!Array.isArray(parsed.choices) || parsed.choices.length === 0) return { ok: false, failure: malformed };
  if (!isRecord(parsed.choices[0])) return { ok: false, failure: malformed };
  return { ok: true, body: parsed };
}

/**
 * A gateway answer, drained and validated before it is returned as served. A stream is read
 * fully first (buffer-then-relay): nothing reaches the client until the whole upstream body is
 * known to be a complete, well-formed stream, so a mid-stream death or a mid-stream error event
 * is a refusal the entry fails over from instead of a half answer the client cannot use. The
 * drain is bounded by the fetch's own timeout signal. The parsed JSON body (or the repaired
 * SSE text) rides the outcome, so the handlers never re-parse it.
 */
async function validateUpstreamBody(
  upstream: Response,
  model: ModelRow,
  stream: boolean,
  declaredTools: DeclaredTools,
): Promise<{ ok: true; json: Json; stream: false } | { ok: true; sseText: string; stream: true } | { ok: false; failure: UpstreamFailure }> {
  let text: string;
  try {
    text = await upstream.text();
  } catch {
    return { ok: false, failure: { status: 502, code: 'upstream_error', model: model.id, kind: 'transient' } };
  }
  if (!stream) {
    const validated = validateJsonAnswer(text, model.id, upstream.headers);
    if (!validated.ok) return { ok: false, failure: validated.failure };
    return { ok: true, json: validated.body, stream: false };
  }
  const repaired = validateAndRepairOpenAIStream(text, declaredTools);
  if (repaired === null) {
    return { ok: false, failure: { status: 502, code: 'upstream_stream_error', model: model.id, kind: 'transient' } };
  }
  return { ok: true, sseText: repaired, stream: true };
}

/**
 * Serve one prompt through the gateway, spending at most MAX_ATTEMPTS_PER_PROMPT calls. The best
 * candidate goes first. What a refusal does next depends on its kind, not just its status:
 * - a daily cap or the gateway's own budget: give up at once — no model on this account can
 *   serve until the cap resets, and every further call is a certain refusal;
 * - a per-minute burst: wait out the window (from the response's own headers, capped at 20 s),
 *   then retry the SAME model once — another model would hit the same account-wide cap; a
 *   window over 20 s away, or a second per-minute refusal after the wait, gives up at once;
 * - a provider refusing one model: fail over at once, no same-model retry — another model is
 *   immediately servable, and the model is not excluded (the refusal is transient);
 * - a 5xx or a network failure: one same-model retry, then fail over;
 * - a 403: the model is excluded for the rest of the trial and the attempt fails over at once,
 *   without burning a retry on a model the gateway will refuse again. Any other 4xx likewise
 *   fails over at once.
 * Every refusal is recorded, for the trail and for cooldowns; a model that refuses twice cools
 * down for later prompts. A malformed or truncated answer is a refusal like any other: the
 * attempt fails over instead of handing the client a broken answer.
 */
async function dialWithFailover(requires: Requirement, buildBody: (model: string) => Json, declaredTools: DeclaredTools): Promise<DialOutcome> {
  const failures: UpstreamFailure[] = [];
  const tried = new Set<string>();
  let retriedOnce = false;
  let waitedOnce = false;
  let last: { model: ModelRow; kind: RefusalKind; status: number; waitMs: number | null } | null = null;
  for (let attempt = 0; attempt < MAX_ATTEMPTS_PER_PROMPT; attempt++) {
    let model: ModelRow | undefined;
    let waitMs = 0;
    if (last !== null) {
      if (last.kind === 'daily_cap' || last.kind === 'gateway_budget') break;
      if (last.kind === 'per_minute') {
        if (waitedOnce || last.waitMs === null) break;
        waitMs = last.waitMs;
        waitedOnce = true;
        model = freeModels(brief).find((c) => c.id === last.model.id);
      } else if (last.kind === 'transient' && !retriedOnce) {
        model = freeModels(brief).find((c) => c.id === last.model.id);
        if (model !== undefined) retriedOnce = true;
      }
      // provider_rate_limit, policy, other: fail over at once (model stays undefined).
    }
    if (model === undefined) model = candidatesFor(requires, tried)[0];
    if (model === undefined) break;
    tried.add(model.id);
    if (waitMs > 0) await sleep(waitMs);
    else if (attempt > 0) await sleep(RETRY_BACKOFF_MS);
    const reqBody = buildBody(model.id);
    const stream = reqBody.stream === true;
    let upstream: Response;
    try {
      upstream = await dial(model.id, reqBody);
    } catch {
      const failure: UpstreamFailure = { status: 502, code: 'upstream_error', model: model.id, kind: 'transient' };
      failures.push(failure);
      recordRefusal(model.id);
      last = { model, kind: 'transient', status: 502, waitMs: null };
      continue;
    }
    if (!upstream.ok) {
      const failure = await refusalOf(upstream, model.id);
      failures.push(failure);
      recordRefusal(model.id);
      if (failure.status === 403) excludedModels.add(model.id);
      last = { model, kind: failure.kind, status: failure.status, waitMs: failure.waitMs ?? null };
      continue;
    }
    if (upstream.body === null) {
      const failure: UpstreamFailure = { status: 502, code: 'empty_response', model: model.id, kind: 'transient' };
      failures.push(failure);
      recordRefusal(model.id);
      last = { model, kind: 'transient', status: 502, waitMs: null };
      continue;
    }
    const validated = await validateUpstreamBody(upstream, model, stream, declaredTools);
    if (!validated.ok) {
      failures.push(validated.failure);
      recordRefusal(model.id);
      if (validated.failure.status === 403) excludedModels.add(model.id);
      last = { model, kind: validated.failure.kind, status: validated.failure.status, waitMs: validated.failure.waitMs ?? null };
      continue;
    }
    if (validated.stream) return { ok: true, model, failures, stream: true, sseText: validated.sseText };
    return { ok: true, model, failures, stream: false, json: validated.json };
  }
  return { ok: false, failures };
}

/** A refused or broken gateway call as a 502 in the client's wire: the prompt is lost. */
function refused(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, text: string): void {
  if (wire === 'anthropic') {
    sendError(res, wire, 502, 'api_error', `the gateway answered HTTP ${status}${text === '' ? '' : `: ${text.slice(0, 500)}`}`);
  } else {
    sendError(res, wire, 502, 'upstream_error', `the gateway answered HTTP ${status}${text === '' ? '' : `: ${text.slice(0, 500)}`}`);
  }
}

/** The tool names a client declared, as a set: an answer may only name these. */
function declaredToolsOf(body: Json): DeclaredTools {
  if (!Array.isArray(body.tools)) return new Set<string>();
  const names = new Set<string>();
  for (const raw of body.tools) {
    if (!isRecord(raw)) continue;
    if (typeof raw.name === 'string' && raw.name !== '') names.add(raw.name);
    else if (isRecord(raw.function) && typeof raw.function.name === 'string' && raw.function.name !== '') names.add(raw.function.name);
  }
  return names;
}

/**
 * An OpenAI-shaped body's output cap, normalised the way anthropicToOpenAI already does: the
 * gateway refuses a call with no cap (or one over 8192), so a missing cap defaults to 1024 and
 * an oversized one is pulled back to 8192. `max_completion_tokens` is the same cap under the
 * other name: it is read when `max_tokens` is absent, then dropped so the gateway never sees
 * two caps.
 */
function normalizeOpenAICap(body: Json): Json {
  const out: Json = { ...body };
  const cap = typeof body.max_tokens === 'number' ? body.max_tokens : typeof body.max_completion_tokens === 'number' ? body.max_completion_tokens : undefined;
  out.max_tokens = typeof cap === 'number' && Number.isFinite(cap) && cap > 0 ? Math.min(cap, 8192) : 1024;
  delete out.max_completion_tokens;
  return out;
}

/** POST /v1/chat/completions: an OpenAI-shaped client, answered by the best free model that meets it. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const normalized = normalizeOpenAICap(body);
  const declared = declaredToolsOf(body);
  const outcome = await dialWithFailover(requirementsFor(body), (model) => ({ ...normalized, model }), declared);
  if (!outcome.ok) {
    const last = outcome.failures[outcome.failures.length - 1];
    if (last === undefined) return sendError(res, 'openai', 503, 'no_free_model', 'no free model in the brief meets this request');
    return refused(res, 'openai', last.status, last.code);
  }
  const { model, failures } = outcome;
  const trail = failuresHeader(failures);
  const trailHeaders = trail === '' ? {} : { 'x-makefun-upstream-failures': trail };
  if (outcome.stream) {
    try {
      res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive', ...trailHeaders });
      for await (const frame of openAIStreamWithModel(textStream(outcome.sseText), model.id)) {
        if (res.destroyed) break;
        res.write(frame);
      }
    } catch {
      // The client went away mid-stream: nothing left to send.
    } finally {
      if (!res.writableEnded) res.end();
    }
    return;
  }
  sendJson(res, 200, openAIJsonWithModel(outcome.json, model.id, declared), trailHeaders);
}

/** POST /v1/messages: an Anthropic-shaped client, rewritten to Chat Completions and answered back. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const declared = declaredToolsOf(body);
  const outcome = await dialWithFailover(requirementsFor(body), (model) => anthropicToOpenAI(body, model), declared);
  if (!outcome.ok) {
    const last = outcome.failures[outcome.failures.length - 1];
    if (last === undefined) return sendError(res, 'anthropic', 503, 'overloaded_error', 'no free model in the brief meets this request');
    return refused(res, 'anthropic', last.status, last.code);
  }
  const { model, failures } = outcome;
  const trail = failuresHeader(failures);
  const trailHeaders = trail === '' ? {} : { 'x-makefun-upstream-failures': trail };
  if (outcome.stream) {
    try {
      res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive', ...trailHeaders });
      for await (const event of anthropicEventsFromOpenAI(textStream(outcome.sseText), model.id)) {
        if (res.destroyed) break;
        res.write(event);
      }
    } catch {
      // The client went away mid-stream: nothing left to send.
    } finally {
      if (!res.writableEnded) res.end();
    }
    return;
  }
  sendJson(res, 200, openAIToAnthropic(outcome.json, model.id, declared), trailHeaders);
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messages(req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/**
 * POST /register with the port and the one catalog; the env answers with the trial's prompt
 * count. One 5xx, timeout or refused connection used to end the trial before a single prompt
 * (exit 1, score 0); registration is retried instead — up to 4 attempts, 500 ms apart — and
 * only the last failure throws. Registration happens before any prompt is served, so the
 * retries cost nothing but a few seconds of the wall.
 */
async function register(port: number, report: CatalogReport): Promise<void> {
  let lastError: unknown = null;
  for (let attempt = 0; attempt < 4; attempt++) {
    if (attempt > 0) await sleep(500);
    try {
      const res = await fetch(`${ENV_URL}/register`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
        signal: AbortSignal.timeout(5000),
      });
      if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
      const reply: unknown = await res.json();
      if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the trial holds ${reply.requests} prompts`);
      return;
    } catch (err) {
      lastError = err;
      log(`register attempt ${attempt + 1} failed: ${(err as Error).message ?? String(err)}`);
    }
  }
  throw lastError;
}

/**
 * Listen on 0.0.0.0 at a free port that is not 8080 (the env's on one layout). The OS's
 * ephemeral range never holds 8080, so the first listen almost always stands; if it ever
 * lands on 8080 the server is closed and another port is tried instead of failing the trial.
 */
async function listenOnFreePort(handler: (req: IncomingMessage, res: ServerResponse) => void): Promise<{ server: Server; port: number }> {
  for (let attempt = 0; attempt < 5; attempt++) {
    const server = createServer(handler);
    try {
      await listen(server);
    } catch {
      continue;
    }
    const address = server.address();
    if (address !== null && typeof address !== 'string' && address.port !== 8080) return { server, port: address.port };
    server.close();
  }
  throw new Error('could not listen on a port other than 8080');
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  if (chatCompletionsUrl() === '') throw new Error('OPENAI_BASE_URL is not set');
  const { server, port } = await listenOnFreePort((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  log(`listening on 0.0.0.0:${port} with ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
