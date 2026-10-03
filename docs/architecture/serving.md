# Serving from the C engine

`pocketllm serve` was the one entry point in this tree with no implementation at all: `cli.py`'s
`_cmd_serve` parsed its flags and exited, while the HTTP server it would start sat one layer away,
finished and tested. This page is the design record for closing that gap — what the missing piece
was, why it was smaller than it looked in one direction and larger in another, and what the result
promises. It is written in the present tense of the design and annotated where the implementation
changed the answer; the adapter itself is `python/pocketllm/server/native_backend.py`.

## What is already there

The gap is not symmetric, and the asymmetry is the whole design.

**The HTTP surface is done.** `python/pocketllm/server/openai.py` is a real server on the standard
library's `ThreadingHTTPServer` — `/v1/chat/completions`, `/v1/completions`, `/v1/models`,
`/metrics`, `/health`, `/ready`, `/alive`, and cancellation through `DELETE /v1/requests/<id>`, with
SSE streaming on both generation endpoints. It is device-neutral: it never touches an accelerator,
it drives whatever object satisfies the `EngineBackend` protocol, and it does the request-shape
audit, the n-choice fan-out, the usage folding and the Prometheus metrics itself. `metrics.py` is a
dependency-free Prometheus text exporter with real cumulative histograms.

None of that is aspirational. `tests/serving/test_server.py` drives it over a real socket against a
scripted fake backend, and the suite passes:

```bash
python -m pytest tests/serving/ -q     # 79 passed
```

**The generation path is done.** `python/pocketllm/native.py` wraps `libpocketllm.so`, and
`pocketllm run` drives it end to end — open, tokenize, one batched prefill, then a loop of
argmax-or-sample, decode, single-token forward — verified token for token against llama.cpp. The
sampling ops landed with this work: `--temperature`, `--top-k`, `--top-p` and `--min-p` reach the C
kernel, and the uniform draw is the host's, so the engine holds no RNG.

**The request pipeline is half wired.** `protocol/contract.py` (the per-runtime field audit) and
`choices.py` (n-choice expansion, seed derivation, usage folding) are both called by the handler and
both tested. `protocol/templating.py`, `protocol/prompt.py` and `protocol/logprobs.py` are written
but have no caller yet — they are the parts of the pipeline that need a tokenizer and a producer, and
neither exists on this path.

## What is missing

**One object.** Nothing in the tree implements `EngineBackend`. A
`grep -rn EngineBackend python/` returns five hits: the `Protocol` definition in `api/backend.py`,
its export from `api/__init__.py`, the type hint on `choices.py`'s docstring, and the two in
`server/openai.py`. No class anywhere carries the methods. The six Python backend directories are
declarations; `reference/` is a kernel backend, not a serving one; and `native.Engine` is a ctypes
handle, not an engine.

The protocol is nine members (`api/backend.py`):

| Member | What the adapter owes |
|---|---|
| `capabilities` | a `BackendCapabilities` — which of batch, streaming, cancellation this path actually has |
| `generate(requests)` | one run per request, results in the order given |
| `stream(request)` | one run, yielded as `TokenEvent`s |
| `audit_request(body, endpoint)` | the first field this path cannot honour, or `None` |
| `health()` | liveness, for `/health` and `/ready` |
| `prepare()` | eager initialization before the first request |
| `metrics()` | engine-held values, folded into `/metrics` |
| `cancel(request_id)` | stop a generation in flight |
| `close()` | release the session |

## The two hard parts

Neither is the protocol translation, which is mechanical. They are the two things the C engine
provides no answer for.

## Concurrency: one session, one sequence, no lock

This is the constraint that decides the shape of the whole feature.

The server is a `ThreadingHTTPServer`: **one thread per request, genuinely concurrent.** The C
session is the opposite. `src/runtime/session.h` documents `position_` as "the next position the
model will write … the KV cache's length", one per `Session`, and the class is explicitly not
thread-safe. `src/` and `native.py` contain no lock of any kind.

Two concurrent requests on one `Engine` therefore race on `position_` and on the KV arena, and the
failure is not a crash to be caught — it is one request's tokens interleaved into another's context
and wrong text returned with a 200. That is the worst failure a server can have, and it is the
default outcome of wiring the handler to the engine without addressing it.

**The decision: serialize, and say so.**

- One session, guarded by a lock, one generation at a time. Requests queue.
- `capabilities.supports_batch = False`, which is a truthful declaration and not a placeholder.
- `supports_cancellation` is `False` for now too — see below.

This is not a compromise against the project's rules; it is what they say. *One process owns one
device* is the invariant this repository was rebuilt around, and a single-card engine that serves
requests one at a time, with the client queueing, is that rule expressed at the serving layer.
Concurrent batching needs per-request KV slots in the C model — a real feature, with a real design
cost, and one this page does not attempt. It is filed below as the follow-on.

What serialization must not do is lie. A `BackendCapabilities` that claimed `supports_batch = True`
would let the handler's fan-out dispatch several choices expecting them to overlap, and the queue
would grow behind a promise the engine cannot keep.

## Cancellation: a blocking call with no poll

`Session::forward` is a C call that runs to completion. Nothing in the ABI observes a flag mid-call,
so a request cannot be abandoned once its forward has started.

An adapter can still honour `cancel()` for the part that is interruptible: the decode loop between
forwards is host code, so a cancel flag checked there stops the next token. What it cannot do is
abort a forward already in flight — at a long context or a slow device that is the *majority* of the
wall clock.

**The decision: `supports_cancellation = False`, and the cancellation endpoint answers honestly.**
The handler already maps `RequestCancelledError` to 499 and consults
`capabilities.supports_cancellation`; reporting `False` means a client that asks to cancel a queued
request gets a truthful refusal rather than a promise that silently does nothing. Between-request
cancellation (drop a request before its turn) is worth having and is not the same feature.

## The fields to refuse

The C ABI applies temperature, top-k, top-p and min-p. It applies nothing else, and a field sent
with a value that would have changed the answer must be refused *by name* rather than ignored — this
is exactly what `ServedFields` exists for. The adapter's `audit_request` returns the audit built
against:

| Field | Verdict | Why |
|---|---|---|
| `temperature`, `top_k`, `top_p`, `min_p` | **honoured** | the sampling ops, native |
| `seed` | **honoured** | the draw is the host's — `random.Random(seed)` in the adapter |
| `max_tokens` | **honoured** | the host's decode loop owns the budget |
| `stop` | **honoured (host-side)** | matched against decoded text between tokens |
| `n` | **honoured (host-side)** | `choices.py` expands it into n runs; see the cost below |
| `logprobs`, `top_logprobs` | **refused** | no producer reports per-position rankings |
| `frequency_penalty`, `presence_penalty` | **refused** | no C op |
| `repetition_penalty` | **refused** | no C op |
| `logit_bias` | **refused** | no C op |
| `response_format` / structured outputs | **refused** | no constrained decoding |
| `best_of`, `echo`, `suffix` | **refused** | not implemented on this path |
| `parallel_tool_calls` | **refused** | no tool-call limiting |

`n` and `stop` are worth flagging as *honoured* rather than refused, because they are served by the
host and not the engine — and that has a cost under serialization. `n=4` is four full generations
through one session, holding the lock, so a request for four choices occupies the engine four times
as long. The alternative — refusing `n` — costs a field every OpenAI client assumes. The host-side
answer is kept, and the lock is the reason a client will notice.

## The prompt: tokenization and the chat template

A chat request needs text rendered into the model's prompt format before it can be tokenized.

**The template is in the checkpoint.** `tokenizer.chat_template` is a GGUF metadata key, and the
Qwen3-0.6B checkpoint carries a 4168-character Jinja template (chatml-style `<|im_start|>`, tool-call
format included). Nothing needs to be vendored. What is needed is a way to read it out — either a C
metadata accessor behind a new ABI call, or a host-side GGUF metadata parse, which is the cheaper
first step and does not touch the ABI.

**Rendering** needs a Jinja engine. `jinja2` is a new runtime dependency, which this package has
avoided (`pyproject.toml` declares numpy and nothing else). The alternatives are a small renderer for
the subset these templates use, or refusing templated chat until one exists. This is an open question
below rather than a decision — it is the one choice here that trades a dependency against a
capability.

**Tokenization** is not a problem: `native.Engine.encode`/`decode` work over the checkpoint's own BPE
with BOS/EOS and special-token handling. The adapter tokenizes through `native.Engine` and does not
touch `pocketllm.tokenizer`, whose `GgufTokenizer` raises today.

One gap to name: the C `encode` takes `add_special` and `parse_special`, and the template's rendered
string already contains the special markers. Whether the markers arrive as literal text to be parsed
or as ids to splice is a detail the adapter has to get right, and it is the kind of detail that
produces fluent-but-wrong output, so it wants a test against a known prompt.

It is settled, and the way it is settled is worth recording because both flags are counter-intuitive
together. The adapter calls `encode(text, add_special=False, parse_special=True)`: `parse_special`
because the rendered template *is* the text containing `<|im_start|>`, which must become one token id
rather than a run of literal characters, and `add_special=False` because the template has already
emitted whatever BOS its author wanted — asking for one as well would put a second at the front. On
the real checkpoint the round trip is exact: `<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n`
tokenizes to 15 ids and decodes back to itself, character for character.

## The shape of the change

| File | Change |
|---|---|
| `python/pocketllm/server/native_backend.py` | **new** — the `EngineBackend` adapter: the lock, the decode loop, the audit, the capabilities, and the ctypes calls |
| `python/pocketllm/cli.py` | `_cmd_serve` builds the adapter from `EngineArgs` and calls `serve()` instead of raising |
| `python/pocketllm/server/openai.py` | unchanged |
| `python/pocketllm/api/*` | unchanged |
| `tests/test_package_boundaries.py` | the new module's import rule, scoped to it rather than widening `pocketllm.server` |
| `tests/serving/test_native_backend.py` | **new** — the adapter against the HTTP surface, over a socket, with a real checkpoint where one is present and skipped where it is not |

The adapter is host-side and pure Python: it needs no C change, and that is deliberate. The C engine
is finished for this purpose; the missing piece is on the Python side.

### Where it went, and why not where this page first proposed

The design above put the adapter in `backends/native_serve.py`. It is in
`server/native_backend.py`, and the import table in `tests/test_package_boundaries.py` is what decides
it rather than taste: `pocketllm.backends` may import `kernels`, `quant` and itself, while the adapter
needs `protocol` (the chat template and the field audit) and `native` (the engine). A device backend
implements the kernel ABI; this implements the *serving* contract. The rule is scoped to the one
module — `"pocketllm.server.native_backend"` — rather than widening `pocketllm.server`, so a future
module in that package cannot reach a device without someone writing it down here. `native` is a
`dlopen` and the loader is numpy-only, so the HTTP layer stays testable without a card.

### What implementing it found, which this page did not anticipate

Recorded because each was a real defect rather than a refinement, and the first two are the kind that
ship silently.

**Decoding one token at a time mangles text.** A BPE split lands mid-character, and `decode` renders
an incomplete UTF-8 sequence as U+FFFD that the next token cannot repair. On the real checkpoint,
per-token decoding turns `한국어 텍스트` into `한국어 ���스트` and `Ünïcödé ñ` into `Ünïcödé ��` — with
nothing raised. The loop decodes the whole generated list each step and strips a *trailing*
replacement character, which is sound because a partial sequence can only ever sit at the end.

**The end-of-text token decodes to its literal spelling.** `<|im_end|>` was being streamed as content,
because the first draft yielded the text before checking for EOS — while the collected path broke
first, so the two paths disagreed about the same generation. The check now precedes the yield.

**`enable_thinking` had to be passed explicitly.** Qwen3's template tests
`enable_thinking is defined and enable_thinking is false`, so an *undefined* variable is neither and
the prompt stops at `<|im_start|>assistant\n` — leaving the model to open its own ` thinking` block, in
which the entire answer arrives as reasoning. The serving default now matches the protocol layer's
(`thinking_mode == "chat"`).

**And the serialization claim is now measured, not argued.** Six concurrent requests on this host:
6/6 correct answers behind the lock, 0/6 without it, and **no error either way** — the unlocked run
returns wrong text with a 200. That is what makes the lock a correctness constraint rather than a
performance note, and it is why the tests assert on the text and on the observed overlap rather than
on whether anything raised.

## What it will and will not promise

**Will:** an OpenAI-compatible endpoint on this host, backed by the real Qwen3 in `src/`, greedy by
default and sampling when asked, streaming, with metrics and a health check, serving one request at a
time and queueing the rest.

**Will not:** concurrent batch inference; cancellation of a forward already running; log probabilities;
penalties; structured outputs. Each of those is refused by name at the door rather than ignored.

## Open questions

All four were carried into the implementation and resolved there, which is the reason to keep them
written down rather than deleted — three were settled by the code and one by a rule that already
existed.

1. **Jinja rendering — resolved: optional, not a dependency.** The adapter imports `jinja2` lazily
   and falls back to the protocol layer's plain rendering when it is absent. `pip install pocketllm`
   is unchanged and a phone build does not carry a template engine; the cost is that a chat request
   on a host without Jinja gets an untemplated prompt, which the model was not trained on. This is
   the one resolution here that is a genuine trade rather than a win, and it is why the fallback is
   documented at the point a reader would hit it.
2. **The metadata read — resolved: host-side GGUF parse.** `pocketllm.loader.gguf.bundle` reads
   `tokenizer.chat_template`, `tokenizer.ggml.eos_token_id` and `qwen3.context_length` from the same
   file the engine opens. The ABI is untouched, which matters more than where the knowledge
   "belongs": the C accessor would be a second way to ask a question the loader already answers.
   Revisit if a non-GGUF checkpoint format arrives for this path.
3. **`n` under the lock — resolved: honoured.** `ServedFields.choices = True` and the host fan-out
   runs the choices serially, so a request for four choices holds the engine four times as long. A
   field every OpenAI client assumes was worth the serialized cost; a client that asked for it can
   see the latency and cannot see a refusal.
4. **Where the adapter lives — resolved: `server/native_backend.py`.** Decided by
   `tests/test_package_boundaries.py` rather than by preference; see *Where it went* above.

## The follow-on

Per-request KV slots in `Session` — a slot table, a slot per in-flight request, and a scheduler that
batches the prefills — is what turns this into a concurrent server. It is a C-side feature touching
`src/runtime/session.*`, `src/model/qwen3.*` and the ABI, and it is the natural next step once the
single-request path is real and measurable.