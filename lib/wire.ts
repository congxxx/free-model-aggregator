// The client wires, the one upstream wire, and the model field.
//
// The trial's client calls the entry as an OpenAI service (POST /v1/chat/completions) or an Anthropic
// one (POST /v1/messages), with a JSON body or with server-sent events. The entry dials the grader's
// gateway, an OpenAI-compatible endpoint, so every prompt goes upstream as a Chat Completions request
// and the Anthropic path is translated both ways:
//
//   - the OpenAI path passes the client's body (with the dialled model's id) to the gateway and relays
//     the gateway's JSON or SSE answer, with every `model` field set to the id this entry dialled;
//   - the Anthropic path rewrites the request to Chat Completions (system prompt, image blocks, tool
//     definitions, tool results and tool choice included) and rewrites the gateway's answer into a
//     Messages object, or into the message_start / content_block_start / content_block_delta /
//     content_block_stop / message_delta / message_stop event order, with the stop reason and the usage
//     translated. Tool calls become tool_use blocks and input_json_delta events.
//
// THE MODEL FIELD
//   The house records every gateway call and scores the model that really answered, so a response's
//   `model` must be the model this entry dialled. OpenRouter's own answer names a free model without
//   the gateway's `openrouter/` prefix; these functions put the dialled id back on every answer and
//   every streamed chunk. They never make a model up: the id came from the brief and went to the
//   gateway in the request's `model`.
//
// Everything here is a pure function of its arguments, except the stream relays: no clock and no
// randomness, so the same body and the same gateway answer give the same bytes on a replay. The two
// model listings use fixed timestamps for the same reason: a `created` that moved between runs would be
// a nondeterministic document.

import { isRecord, type Json } from './providers.ts';

/** One model the entry serves, as GET /v1/models lists it. */
export interface ServedModel {
  id: string;
  provider: string;
  display_name: string;
}

/** A fixed epoch second for every listed model's `created`: nothing in the reply reads the clock. */
const FIXED_OPENAI_CREATED = 1767225600;
/** A fixed instant for every listed model's `created_at`: nothing in the reply reads the clock. */
const FIXED_ANTHROPIC_CREATED_AT = '2026-01-01T00:00:00Z';

/** GET /v1/models on an OpenAI-shaped request: the models this entry serves, in OpenAI's list shape. */
export function openAIModelList(models: readonly ServedModel[]): Json {
  return {
    object: 'list',
    data: models.map((m) => ({ id: m.id, object: 'model', created: FIXED_OPENAI_CREATED, owned_by: m.provider })),
  };
}

/** GET /v1/models on an Anthropic-shaped request (an x-api-key header): Anthropic's model list shape. */
export function anthropicModelList(models: readonly ServedModel[]): Json {
  return {
    data: models.map((m) => ({ type: 'model', id: m.id, display_name: m.display_name, created_at: FIXED_ANTHROPIC_CREATED_AT })),
    has_more: false,
  };
}

/** An error in Anthropic's shape (the Messages API's error envelope). */
export function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

/** An error in OpenAI's shape (Chat Completions). */
export function openAIError(message: string, type: string, code: string | null): Json {
  return { error: { message, type, code } };
}

/** The text of a string or of a block array's text blocks; '' when there is none. */
function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('\n');
}

/** One Anthropic content block as an OpenAI content part: text or an inline image; null for the rest. */
function partOf(block: Json): Json | null {
  if (block.type === 'text' && typeof block.text === 'string') return { type: 'text', text: block.text };
  if (block.type === 'image' && isRecord(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      const media = typeof source.media_type === 'string' ? source.media_type : 'image/png';
      return { type: 'image_url', image_url: { url: `data:${media};base64,${source.data}` } };
    }
    if (source.type === 'url' && typeof source.url === 'string') return { type: 'image_url', image_url: { url: source.url } };
  }
  return null;
}

/** A tool_result block's content as text: its text blocks, or its JSON when it holds something else. */
function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = textOf(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

/** A tool call's arguments as an object: the gateway's JSON string, or {} when it is not one. */
function argumentsOf(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch {
      // Not JSON: an empty input.
    }
  }
  return {};
}

/** OpenAI's finish_reason as Anthropic's stop_reason. */
export function stopReasonOf(finish: unknown): string {
  if (finish === 'length') return 'max_tokens';
  if (finish === 'tool_calls' || finish === 'function_call') return 'tool_use';
  if (finish === 'content_filter') return 'refusal';
  return 'end_turn';
}

/**
 * An Anthropic Messages request as an OpenAI Chat Completions request that names `model`: the system
 * prompt becomes a system message, image blocks become image_url data URLs, tool_use blocks become
 * tool_calls, tool_result blocks become tool messages, tools and tool_choice are rewritten, and the
 * output cap, temperature, top_p, stop sequences and streaming pass through.
 */
export function anthropicToOpenAI(body: Json, model: string): Json {
  const messages: Json[] = [];
  const system = textOf(body.system);
  if (system !== '') messages.push({ role: 'system', content: system });
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const assistant = raw.role === 'assistant';
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: assistant ? 'assistant' : 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (assistant) {
      const toolCalls: Json[] = [];
      for (const block of content) {
        if (!isRecord(block) || block.type !== 'tool_use') continue;
        toolCalls.push({
          id: typeof block.id === 'string' ? block.id : '',
          type: 'function',
          function: {
            name: typeof block.name === 'string' ? block.name : '',
            arguments: JSON.stringify(isRecord(block.input) ? block.input : {}),
          },
        });
      }
      const text = textOf(content);
      messages.push({
        role: 'assistant',
        content: text !== '' ? text : null,
        ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
      });
      continue;
    }
    const parts: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'tool_result') {
        messages.push({
          role: 'tool',
          tool_call_id: typeof block.tool_use_id === 'string' ? block.tool_use_id : '',
          content: toolResultText(block.content),
        });
        continue;
      }
      const part = partOf(block);
      if (part !== null) parts.push(part);
    }
    if (parts.length > 0) messages.push({ role: 'user', content: parts });
  }

  const out: Json = { model, messages };
  // The gateway refuses a call with no output cap (or one over 8192): the env always sends one, but
  // a body without it must still get a cap rather than a certain refusal. A cap over the gateway's
  // ceiling is pulled back to it; an early stop is a legal stop reason, a refusal is a lost prompt.
  const cap = body.max_tokens;
  out.max_tokens = typeof cap === 'number' && Number.isFinite(cap) && cap > 0 ? Math.min(cap, 8192) : 1024;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) out.stop = body.stop_sequences;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || typeof raw.name !== 'string') return [];
      return [
        {
          type: 'function',
          function: {
            name: raw.name,
            description: typeof raw.description === 'string' ? raw.description : '',
            parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
          },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (isRecord(choice)) {
    if (choice.type === 'auto') out.tool_choice = 'auto';
    else if (choice.type === 'any') out.tool_choice = 'required';
    else if (choice.type === 'none') out.tool_choice = 'none';
    else if (choice.type === 'tool' && typeof choice.name === 'string') out.tool_choice = { type: 'function', function: { name: choice.name } };
  }
  if (body.stream === true) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

function numberOrZero(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

/** A Chat Completions answer as an Anthropic Messages object. `model` is the id the entry dialled. */
export function openAIToAnthropic(body: Json, model: string, declaredTools: DeclaredTools = null): Json {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  const content: Json[] = [];
  const text = typeof message.content === 'string' ? message.content : textOf(message.content);
  if (text !== '') content.push({ type: 'text', text });
  const calls = declaredToolCalls(message.tool_calls, declaredTools);
  for (const call of calls) {
    if (!isRecord(call) || !isRecord(call.function)) continue;
    content.push({
      type: 'tool_use',
      id: typeof call.id === 'string' ? call.id : 'toolu_unknown',
      name: typeof call.function.name === 'string' ? call.function.name : '',
      input: argumentsOf(call.function.arguments),
    });
  }
  if (content.length === 0) content.push({ type: 'text', text: '' });
  const usage = isRecord(body.usage) ? body.usage : {};
  // The wire check needs an output count of at least 1. A completion that produced text or a tool
  // call really spent at least one output token, so a missing or zero count from upstream is a
  // reporting gap, not a zero: report 1. An empty completion keeps 0 and fails on content anyway.
  const produced = text !== '' || content.some((b) => b.type === 'tool_use');
  const outputTokens = numberOrZero(usage.completion_tokens);
  return {
    id: `msg_${typeof body.id === 'string' ? body.id : 'aggregate'}`,
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: stopReasonOf(choice.finish_reason),
    stop_sequence: null,
    usage: { input_tokens: numberOrZero(usage.prompt_tokens), output_tokens: outputTokens > 0 ? outputTokens : produced ? 1 : 0 },
  };
}

/** The finish reasons the client wires accept. Anything else (notably "error") is a refusal, not an answer. */
const LEGAL_FINISH_REASONS = new Set(['stop', 'length', 'tool_calls', 'function_call', 'content_filter']);

/**
 * The tool names a client declared, as a set: a tool call whose name is not in it is not the
 * client's, so it must not reach the client. `null` means "do not filter" (callers that do not
 * know the declared names); an empty set drops every tool call.
 */
export type DeclaredTools = ReadonlySet<string> | null;

/** A Chat Completions answer's tool calls, keeping only the ones the client declared. */
function declaredToolCalls(calls: unknown, declared: DeclaredTools): Json[] {
  if (!Array.isArray(calls)) return [];
  return calls.flatMap((call) => {
    if (!isRecord(call) || !isRecord(call.function)) return [];
    if (declared !== null && typeof call.function.name === 'string' && !declared.has(call.function.name)) return [];
    return [call];
  });
}

/** A Chat Completions answer with `model` set to the id the entry dialled: the one truthful claim. */
export function openAIJsonWithModel(body: Json, model: string, declaredTools: DeclaredTools = null): Json {
  const out: Json = { ...body, model };
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  const keptCalls = declaredToolCalls(message.tool_calls, declaredTools);
  if (declaredTools !== null && Array.isArray(message.tool_calls)) {
    out.choices = choices.map((c, i) => (i === 0 && isRecord(c) ? { ...c, message: { ...message, tool_calls: keptCalls } } : c));
  }
  // The wire check needs an output count of at least 1. A completion that produced text or a tool
  // call really spent at least one output token, so a missing or zero count from upstream is a
  // reporting gap, not a zero: report 1. (The Anthropic path already floors the same way.)
  const text = typeof message.content === 'string' ? message.content : textOf(message.content);
  const produced = text !== '' || keptCalls.length > 0;
  if (produced) {
    const usage = isRecord(body.usage) ? { ...body.usage } : {};
    if (numberOrZero(usage.completion_tokens) <= 0) usage.completion_tokens = 1;
    out.usage = usage;
  }
  return out;
}

/** One server-sent event: `event: <name>` when named, then the JSON data line, then the blank line. */
export function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/**
 * One SSE frame with every data payload's `model` set to `model`, and its blank line kept: comments,
 * `event:` lines and payloads that are not JSON objects pass through untouched. Chat Completions names
 * the model on its chunks; the client sees the id the entry dialled, never the provider's spelling.
 */
function frameWithModel(frame: string, model: string): string {
  const lines = frame.split('\n').map((line) => {
    const trimmed = line.trim();
    if (!trimmed.startsWith('data:')) return line;
    const payload = trimmed.slice(5).trim();
    if (payload === '' || payload === '[DONE]') return line;
    try {
      const chunk: unknown = JSON.parse(payload);
      if (isRecord(chunk)) return `data: ${JSON.stringify({ ...chunk, model })}`;
    } catch {
      // Not JSON: relay the line as it came.
    }
    return line;
  });
  return `${lines.join('\n')}\n\n`;
}

/**
 * Validate a drained Chat Completions SSE body and repair what the client wire would reject.
 * Returns the repaired SSE text, or null when the stream is unusable and the entry should fail
 * over instead of relaying it: a mid-stream death (no finish chunk), a finish_reason the wires
 * do not accept (notably "error", the shape a mid-stream rate limit takes), a data line that is
 * not JSON, or a stream that produced no content and no tool call at all (an empty answer cannot
 * carry the prompt's token, so another model deserves the attempt). Repairs: later `[DONE]`
 * frames are dropped, a missing `[DONE]` is appended after the finish chunk, a missing usage
 * chunk is synthesised (output floored to 1 when content was produced), and tool-call deltas
 * naming a tool the client never declared are removed (by call index, so a name chunk and its
 * argument chunks go together). Double finish chunks are left alone: openAIStreamWithModel
 * already merges them into exactly one. This is a pure function of the drained text, so a
 * replay sees the same bytes.
 */
export function validateAndRepairOpenAIStream(text: string, declaredTools: DeclaredTools = null): string | null {
  const folded = text.replace(/\r\n/g, '\n');
  const rawFrames = folded.split('\n\n').filter((f) => f.trim() !== '');
  const kept: string[] = [];
  const droppedIndexes = new Set<number>();
  let lastToolIndex = 0;
  let sawContent = false;
  let sawFinish = false;
  let sawUsage = false;
  let sawDone = false;
  let keptToolCall = false;
  let firstChunk: Json | null = null;

  for (const frame of rawFrames) {
    const lines = frame.split('\n');
    const dataLines = lines.filter((line) => line.trim().startsWith('data:'));
    const otherLines = lines.filter((line) => !line.trim().startsWith('data:'));
    if (dataLines.length === 0) {
      kept.push(frame);
      continue;
    }
    let isDone = false;
    const rebuiltData: string[] = [];
    for (const line of dataLines) {
      const payload = line.trim().slice(5).trim();
      if (payload === '') continue;
      if (payload === '[DONE]') {
        isDone = true;
        continue;
      }
      let chunk: unknown;
      try {
        chunk = JSON.parse(payload);
      } catch {
        return null; // A data line that is not JSON: the stream is malformed.
      }
      if (!isRecord(chunk)) return null;
      if (isRecord(chunk.error)) return null; // A top-level error mid-stream is a refusal.
      if (firstChunk === null) firstChunk = chunk;
      if (isRecord(chunk.usage)) sawUsage = true;
      const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
      const first = isRecord(choices[0]) ? choices[0] : null;
      if (first !== null) {
        const delta = isRecord(first.delta) ? first.delta : null;
        if (delta !== null) {
          if (typeof delta.content === 'string' && delta.content !== '') sawContent = true;
          if (Array.isArray(delta.tool_calls)) {
            const keptCalls: Json[] = [];
            for (const call of delta.tool_calls) {
              if (!isRecord(call)) continue;
              const idx = typeof call.index === 'number' ? call.index : lastToolIndex;
              lastToolIndex = idx;
              const fn = isRecord(call.function) ? call.function : null;
              const name = fn !== null && typeof fn.name === 'string' ? fn.name : '';
              if (name !== '' && declaredTools !== null && !declaredTools.has(name)) droppedIndexes.add(idx);
              if (droppedIndexes.has(idx)) continue;
              keptCalls.push(call);
            }
            if (keptCalls.length > 0) {
              sawContent = true;
              keptToolCall = true;
            }
            if (keptCalls.length === 0) delete (delta as Record<string, unknown>).tool_calls;
            else (delta as Record<string, unknown>).tool_calls = keptCalls;
          }
        }
        const finish = first.finish_reason;
        if (finish !== null && finish !== undefined) {
          if (typeof finish !== 'string' || !LEGAL_FINISH_REASONS.has(finish)) return null;
          // A finish of tool_calls with no tool call left (all were dropped as undeclared)
          // would claim a tool call that is not there: the answer is really a text answer.
          if (finish === 'tool_calls' && !keptToolCall) first.finish_reason = 'stop';
          sawFinish = true;
        }
      }
      rebuiltData.push(`data: ${JSON.stringify(chunk)}`);
    }
    if (isDone) {
      if (sawDone) continue; // Drop a second [DONE]: the client counts them.
      sawDone = true;
      kept.push('data: [DONE]');
      continue;
    }
    if (rebuiltData.length === 0) continue; // A frame that lost every line (all tool calls dropped).
    kept.push([...otherLines, ...rebuiltData].join('\n'));
  }

  if (!sawFinish) return null; // Truncated: no finish chunk, so no complete answer.
  if (!sawContent) return null; // Empty: no content and no (declared) tool call to carry the token.
  if (!sawUsage) {
    const id = firstChunk !== null && typeof firstChunk.id === 'string' ? firstChunk.id : 'chatcmpl-repair';
    const created = firstChunk !== null && typeof firstChunk.created === 'number' ? firstChunk.created : 0;
    const usageFrame = `data: ${JSON.stringify({ id, object: 'chat.completion.chunk', created, choices: [], usage: { prompt_tokens: 0, completion_tokens: 1, total_tokens: 1 } })}`;
    // The usage chunk goes before [DONE]: the client reads usage from the frames before it.
    const doneIdx = kept.findIndex((f) => f.trim() === 'data: [DONE]');
    if (doneIdx !== -1) kept.splice(doneIdx, 0, usageFrame);
    else kept.push(usageFrame);
  }
  if (!sawDone) kept.push('data: [DONE]');
  return `${kept.join('\n\n')}\n\n`;
}

/** One drained SSE body as a one-shot stream, for the translators that read from a stream. */
export function textStream(text: string): ReadableStream<Uint8Array> {
  const bytes = new TextEncoder().encode(text);
  return new ReadableStream({
    start(c) {
      c.enqueue(bytes);
      c.close();
    },
  });
}

/**
 * A Chat Completions SSE stream as it came, frame by frame, every data payload's `model` the id the
 * entry dialled. Frames are split on the blank line that ends them, as the SSE grammar does; a frame cut
 * across reads is held until its end arrives, and a body that ends without one is relayed as it is.
 *
 * A provider asked for usage repeats the finish chunk: a bare finish chunk, then the same chunk
 * again carrying `usage`. The client counts finish chunks, so exactly one may reach it: the first
 * finish chunk is held, a later one's usage is merged into it, and the single merged chunk is emitted
 * when `[DONE]` arrives (or the body ends). Content in a later finish chunk, if any, is relayed as
 * a delta before it, so nothing is lost.
 */
export async function* openAIStreamWithModel(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  const reader = upstream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let pendingFinish: Record<string, unknown> | null = null;

  const finishOf = (frame: string): Record<string, unknown> | null => {
    for (const line of frame.split('\n')) {
      const trimmed = line.trim();
      if (!trimmed.startsWith('data:')) continue;
      const payload = trimmed.slice(5).trim();
      if (payload === '' || payload === '[DONE]') continue;
      try {
        const chunk: unknown = JSON.parse(payload);
        if (!isRecord(chunk) || !Array.isArray(chunk.choices)) continue;
        const first = chunk.choices[0];
        if (isRecord(first) && first.finish_reason !== null && first.finish_reason !== undefined) return chunk;
      } catch {
        // Not JSON: not a finish chunk.
      }
    }
    return null;
  };
  const isDoneFrame = (frame: string): boolean =>
    frame.split('\n').some((line) => line.trim() === 'data: [DONE]');
  const contentOf = (chunk: Record<string, unknown>): string => {
    if (!Array.isArray(chunk.choices)) return '';
    const first = chunk.choices[0];
    if (!isRecord(first) || !isRecord(first.delta)) return '';
    return typeof first.delta.content === 'string' ? first.delta.content : '';
  };
  const flushFinish = function* (): Generator<string> {
    if (pendingFinish !== null) {
      yield `data: ${JSON.stringify({ ...pendingFinish, model })}\n\n`;
      pendingFinish = null;
    }
  };
  function* handleFrame(frame: string): Generator<string> {
    if (isDoneFrame(frame)) {
      yield* flushFinish();
      yield frameWithModel(frame, model);
      return;
    }
    const finish = finishOf(frame);
    if (finish === null) {
      yield frameWithModel(frame, model);
      return;
    }
    if (pendingFinish === null) {
      pendingFinish = finish;
      return;
    }
    // A second finish chunk: any content it carries is a delta that precedes the finish.
    const content = contentOf(finish);
    if (content !== '') {
      const firstChoice: Record<string, unknown> =
        Array.isArray(finish.choices) && isRecord(finish.choices[0]) ? finish.choices[0] : {};
      const delta: Record<string, unknown> = { ...finish, choices: [{ ...firstChoice, finish_reason: null }] };
      delete delta.usage;
      yield frameWithModel(`data: ${JSON.stringify(delta)}\n\n`, model);
    }
    if (finish.usage !== undefined) pendingFinish = { ...pendingFinish, usage: finish.usage };
  }

  // CRLF is folded on the accumulated buffer, not on each read: a \r\n pair split across two reads
  // would otherwise survive, merge its frames and leak stray \r bytes into the relayed stream.
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer = `${buffer}${decoder.decode(value, { stream: true })}`.replace(/\r\n/g, '\n');
      let at = buffer.indexOf('\n\n');
      while (at !== -1) {
        const frame = buffer.slice(0, at);
        buffer = buffer.slice(at + 2);
        yield* handleFrame(frame);
        at = buffer.indexOf('\n\n');
      }
    }
    buffer = `${buffer}${decoder.decode()}`.replace(/\r\n/g, '\n');
    if (buffer.trim() !== '') yield* handleFrame(buffer);
    yield* flushFinish();
  } finally {
    reader.releaseLock();
  }
}

/**
 * A Chat Completions SSE stream as the Anthropic event sequence: message_start on the first chunk,
 * content_block_start/delta/stop per text block or tool call, then message_delta with the stop reason and
 * usage, then message_stop. `model` is the id the entry dialled, named back to the client.
 */
export async function* anthropicEventsFromOpenAI(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  const reader = upstream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let started = false;
  let finished = false;
  /** The stop reason a finish_reason named, held until the stream's usage chunk has arrived. */
  let pendingStop: string | null = null;
  let textIndex = -1;
  let nextIndex = 0;
  const toolIndexes = new Map<number, number>();
  let inputTokens = 0;
  let outputTokens = 0;
  /** Whether any text or tool-call content was emitted: a stream that produced content spent output. */
  let sawContent = false;

  const startEvents = (): string[] => {
    if (started) return [];
    started = true;
    return [
      sseEvent('message_start', {
        type: 'message_start',
        message: {
          id: 'msg_stream',
          type: 'message',
          role: 'assistant',
          model,
          content: [],
          stop_reason: null,
          stop_sequence: null,
          usage: { input_tokens: inputTokens, output_tokens: 0 },
        },
      }),
    ];
  };
  const closeEvents = (): string[] => {
    const out: string[] = [];
    if (textIndex !== -1) {
      out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: textIndex }));
      textIndex = -1;
    }
    for (const index of [...toolIndexes.values()].sort((a, b) => a - b)) out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index }));
    toolIndexes.clear();
    return out;
  };
  const finishEvents = (stopReason: string): string[] => {
    if (finished) return [];
    finished = true;
    // The wire check needs an output count of at least 1. A stream that produced content really
    // spent at least one output token, so a missing or zero count from upstream is a reporting gap,
    // not a zero: report 1. An empty stream keeps 0 and fails on content anyway.
    const output = outputTokens > 0 ? outputTokens : sawContent ? 1 : 0;
    return [
      ...closeEvents(),
      sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: stopReason, stop_sequence: null }, usage: { output_tokens: output } }),
      sseEvent('message_stop', { type: 'message_stop' }),
    ];
  };

  /** One SSE frame's lines as Anthropic events, mutating the stream's state. */
  const eventsOfFrame = (frame: string): string[] => {
    const out: string[] = [];
    for (const line of frame.split('\n')) {
      const trimmed = line.trim();
      if (!trimmed.startsWith('data:')) continue;
      const payload = trimmed.slice(5).trim();
      if (payload === '[DONE]') {
        out.push(...startEvents(), ...finishEvents(pendingStop ?? 'end_turn'));
        continue;
      }
      let chunk: unknown;
      try {
        chunk = JSON.parse(payload);
      } catch {
        continue;
      }
      if (!isRecord(chunk)) continue;
      if (isRecord(chunk.usage)) {
        inputTokens = numberOrZero(chunk.usage.prompt_tokens);
        outputTokens = numberOrZero(chunk.usage.completion_tokens);
      }
      const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
      const choice = isRecord(choices[0]) ? choices[0] : null;
      if (choice === null) continue;
      const delta = isRecord(choice.delta) ? choice.delta : {};
      out.push(...startEvents());
      if (typeof delta.content === 'string' && delta.content !== '') {
        sawContent = true;
        if (textIndex === -1) {
          textIndex = nextIndex;
          nextIndex += 1;
          out.push(sseEvent('content_block_start', { type: 'content_block_start', index: textIndex, content_block: { type: 'text', text: '' } }));
        }
        out.push(sseEvent('content_block_delta', { type: 'content_block_delta', index: textIndex, delta: { type: 'text_delta', text: delta.content } }));
      }
      const calls = Array.isArray(delta.tool_calls) ? delta.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call)) continue;
        const upstreamIndex = typeof call.index === 'number' ? call.index : toolIndexes.size;
        let index = toolIndexes.get(upstreamIndex);
        if (index === undefined) {
          if (textIndex !== -1) {
            out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: textIndex }));
            textIndex = -1;
          }
          index = nextIndex;
          nextIndex += 1;
          toolIndexes.set(upstreamIndex, index);
          const fn = isRecord(call.function) ? call.function : {};
          out.push(
            sseEvent('content_block_start', {
              type: 'content_block_start',
              index,
              content_block: {
                type: 'tool_use',
                id: typeof call.id === 'string' && call.id !== '' ? call.id : `toolu_${index}`,
                name: typeof fn.name === 'string' ? fn.name : '',
                input: {},
              },
            }),
          );
        }
        const fn = isRecord(call.function) ? call.function : {};
        if (typeof fn.arguments === 'string' && fn.arguments !== '') {
          sawContent = true;
          out.push(sseEvent('content_block_delta', { type: 'content_block_delta', index, delta: { type: 'input_json_delta', partial_json: fn.arguments } }));
        }
      }
      // Hold the stop reason until the stream's last chunk (the usage one) and [DONE] have arrived: OpenAI
      // sends the usage after the finish_reason, and Anthropic's message_delta carries the output tokens.
      if (choice.finish_reason !== null && choice.finish_reason !== undefined) pendingStop = stopReasonOf(choice.finish_reason);
    }
    return out;
  };

  // CRLF is folded on the accumulated buffer, not on each read: a \r\n pair split across two reads
  // would otherwise survive and merge its frames (a frame cut across reads is held until its end).
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer = `${buffer}${decoder.decode(value, { stream: true })}`.replace(/\r\n/g, '\n');
      let at = buffer.indexOf('\n\n');
      while (at !== -1) {
        const frame = buffer.slice(0, at);
        buffer = buffer.slice(at + 2);
        for (const event of eventsOfFrame(frame)) yield event;
        at = buffer.indexOf('\n\n');
      }
    }
    buffer = `${buffer}${decoder.decode()}`.replace(/\r\n/g, '\n');
    if (buffer.trim() !== '') for (const event of eventsOfFrame(buffer)) yield event;
  } finally {
    reader.releaseLock();
  }
  if (!finished) {
    for (const event of startEvents()) yield event;
    for (const event of finishEvents(pendingStop ?? 'end_turn')) yield event;
  }
}
