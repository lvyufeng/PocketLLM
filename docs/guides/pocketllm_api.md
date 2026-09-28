# PocketLLM API and Backend Guide

PocketLLM presents one user-facing API over two independent execution planes:

- **Torch** uses the existing PyTorch/Triton runtimes under `src/`.
- **C++** uses the native `cpp_engine` runtime and the selected CUDA or Ascend backend.

The common API does not imply shared kernels, KV-cache layouts, or schedulers. Those remain backend- and hardware-specific so that Turing CUDA and Ascend optimizations are not weakened by a lowest-common-denominator abstraction.

## Offline API

```python
from pocketllm import EngineArgs, LLM, SamplingParams

llm = LLM(EngineArgs(
    model="/path/to/checkpoint",
    backend="auto",  # or "torch" / "cpp"
    tensor_parallel_size=4,
    max_model_len=65536,
))

outputs = llm.generate(
    ["Explain speculative decoding.", "Explain continuous batching."],
    SamplingParams(max_tokens=128, temperature=0.0),
)
for output in outputs:
    print(output.text, output.usage.as_dict())
```

Pre-tokenized input is also accepted:

```python
outputs = llm.generate([[1, 42, 17]], SamplingParams(max_tokens=16))
```

For chat-shaped inputs, use the library-first chat surface. It accepts the same normalized message
and optional fields as `/v1/chat/completions`, and returns the same list-shaped result as
`generate()` (one result for the supplied conversation):

```python
messages = [
    {"role": "system", "content": "Answer concisely."},
    {"role": "user", "content": "What is 2+2?"},
]
outputs = llm.chat(
    messages,
    SamplingParams(max_tokens=32, temperature=0.0),
    reasoning_effort="low",
)
print(outputs[0].text)
```

`chat()` also accepts `reasoning`, `tools`, `tool_choice`, `response_format`, and an optional
`request_id`. The request body is normalized through the same backend-neutral builder used by the
HTTP endpoint; checkpoint-owned chat templates remain the authority for model-specific prompt
encoding. Caller-owned message and tool structures are not mutated.

Use `generate_stream()` or `chat_stream()` for token events and `cancel(request_id)` to request
cancellation at a safe generation boundary. The initial C++ compatibility adapter is serialized and
exposes native greedy generation; unsupported sampling or request features report
`UnsupportedFeatureError` rather than being silently ignored. Native streaming decodes the cumulative
token sequence before emitting each delta, so BPE and UTF-8 token boundaries are handled by the
tokenizer.

## Async API

`AsyncLLM` mirrors every offline entry point: `generate`, `generate_stream`, `chat`, and
`chat_stream`.

```python
from pocketllm import AsyncLLM, EngineArgs, SamplingParams

async with AsyncLLM(EngineArgs(model="/path/to/checkpoint")) as llm:
    result = (await llm.generate("Hello", SamplingParams(max_tokens=32)))[0]
    async for event in llm.generate_stream("Stream this"):
        print(event.text, end="", flush=True)

    chat_result = (await llm.chat(
        [{"role": "user", "content": "Explain KV caching."}],
        SamplingParams(max_tokens=32),
    ))[0]
    print(chat_result.text)
    async for event in llm.chat_stream(
        [{"role": "user", "content": "Stream a short answer."}],
    ):
        print(event.text, end="", flush=True)
```

`AsyncLLM` currently provides non-blocking application integration around the backend contract. Its
own concurrency is the executor's `max_workers`, which defaults to 1, so it adds no batching of its
own; the batch path belongs to the backend under it and is a separate decision. The async chat
methods reuse the same executor-backed lifecycle and `TokenEvent` contract as the sync facade.

## CLI and server

```bash
# Installed console script
pocketllm serve \
  --model /path/to/checkpoint \
  --backend auto \
  --tensor-parallel-size 4 \
  --max-model-len 65536 \
  --port 8000

# Source-tree equivalent
python -m pocketllm serve \
  --model /path/to/checkpoint \
  --backend auto \
  --tensor-parallel-size 4 \
  --max-model-len 65536 \
  --port 8000

# DeepSeek-V4.1-Flash, TP4, one request at a time
python -m pocketllm serve \
  --model /mnt/data3/DeepSeek-V4.1-Flash \
  --backend v41 \
  --tensor-parallel-size 4 \
  --max-model-len 2048 \
  --port 8000 \
  --expert-pool-rows 288 \
  --prefill-chunk-tokens 4096 \
  --decode-graphs \
  --threads 22
``` 

For `tensor_parallel_size > 1`, the CLI supervises local tensor-parallel ranks by default. It creates a
private per-run rendezvous directory and NCCL-ID path, assigns `RANK`/`LOCAL_RANK`/`WORLD_SIZE` and
`TP_RANK`/`TP_WORLD`, starts every rank without a shell, and waits for all ranks to finish loading
before rank 0 is considered ready. Only rank 0 binds the HTTP listener. A rank failure, startup
timeout, or received `SIGINT`/`SIGTERM` causes the supervisor to stop and reap the whole group.
Use `--tensor-parallel-startup-timeout SECONDS` and `--tensor-parallel-shutdown-timeout SECONDS`
to tune lifecycle bounds; `--tensor-parallel-master-addr`, `--tensor-parallel-master-port`, and
`--tensor-parallel-rendezvous-dir` are available for deployments that need explicit rendezvous
placement. A caller-provided rendezvous directory is treated as a parent for a fresh private run
directory and is never removed by PocketLLM.

The built-in supervisor works with every adapter that serves a checkpoint, and the native `cpp` one
included: `CppBackend.run_worker` enters the engine's own worker loop, `warmup_tp` brings the NCCL
communicator up inside construction, the NCCL-ID path arrives through the environment the supervisor
publishes, and each rank's card is its own index — which works precisely because the supervisor hands
every rank the same visible device list rather than narrowing it per rank. A V4.1 backend is the
case that shaped the rest: rank 0 loads inside the rendezvous window, because a backend handed back
unloaded would find the group gone.
A rank it starts runs **one program**, `pocketllm/backends/worker.py`, whichever runtime it is
serving: what tells it which one is `POCKETLLM_WORKER_BACKEND`, set from the `WORKERS` registry, and
the two things that still differ per runtime are the adapter to import and whether the checkpoint is
already loaded when the adapter is constructed. Adding a runtime therefore means adding a
`WorkerSpec` entry rather than writing a second worker script.
`--no-tensor-parallel-supervisor` remains supported for `torchrun` and hand-written rank launchers,
and is the opt-out for a rank layout this supervisor does not produce. A per-rank
`CUDA_VISIBLE_DEVICES` is a layout it does produce — and is still honoured — but it is no longer the
way to name cards: `--device-ids 2,3` names every rank's card once, and the offset the `cpp` adapter
used to apply on top of a narrowed list is not applied on top of an explicit list. This process supervisor is not a scheduler: it starts ranks and reaps them, and
what runs inside those ranks is the backend's own business. Whether a served backend batches is
decided separately and per backend — see [Batching on the `cpp` backend](#batching-on-the-cpp-backend)
— so `--tensor-parallel-size 4` says nothing about the width a request sees.

### Batching on the `cpp` backend

This backend owns a continuous-batching scheduler, and it is the default path. The two flags are the
same decision seen twice, so the contradiction between them is refused rather than resolved:

| Command | Effect |
| --- | --- |
| `--backend cpp` | The batch scheduler, 8 slots. |
| `--backend cpp --max-batch-size 4` | The batch scheduler, 4 slots. A width above 1 asks for the scheduler on its own. |
| `--backend cpp --no-enable-batching` | The serialized session: one request at a time, no scheduler. |
| `--backend cpp --no-enable-batching --max-batch-size 4` | `ConfigurationError`. A width is a request for a scheduler and the opt-out refuses it; picking one of the two here would leave the other flag accepted and ignored. |
| `--backend-option enable_batching=false` | The serialized session. The backend option is the spelling the `scripts/` benchmarks use and it wins over the flag. |

**The default width is 8, not 1.** A width of 1 is not a batch, so a default of 1 would make the
default path serial — which is exactly what it used to be. The number is not cosmetic: the engine
sizes its KV cache from it at construction, so a width that does not reach the engine is a width the
server refuses at the first concurrent request. Which of the two paths the backend resolved, and at
what width, is what `capabilities.details` reports through the Python API — `details["scheduler"]`
and `details["max_batch_size"]` — and the scheduler's own view of it is
[`engine_caps()`](#scheduler-backed-async-requests). Neither is on the HTTP surface today: `/health`
reports the backend and its model, not its capabilities.

**A width above 1 costs a lone request some latency.** The scheduler runs the width's rows whether or
not that many requests are present, so a single request pays for the width it was given: with the
prompt cache held fixed, ~17% more wall than the serialized session at width 2 and ~18% at width 8,
of which about ten points is decode. More than that, the scheduler's prefill path does not consult
the prefix cache, so a prompt the serialized session would have resumed for nothing is re-forwarded
in full — which is the larger of the two costs for a client that repeats its prompt, and the reason
`--max-batch-size 2` (or `--no-enable-batching`) is the setting for a deployment that serves one
caller at a time. What the width buys is concurrency: on the real CLI, two concurrent requests
reached 30.6 aggregate tok/s through the default path against 25.6 through the serialized one. Both
figures and the method behind them are in
[the concurrency acceptance page](../performance/cpp_openai_concurrency_validation.md#what-the-width-costs-a-lone-request).

Once the batch path is selected, the sampling options the OpenAI surface accepts are honoured subject
to the engine's own limits: see [Request fields](#request-fields) for which values the engine under
this adapter can apply per request, and what happens to the ones it cannot.

### Backend options

Every option a runtime accepts is *declared* by that runtime — its type, its default, its bounds and
its accepted values — next to the code that reads it. `pocketllm/backends/options.py` is the single
reader and each adapter's `OPTIONS` is the single list, so the things a launch can get wrong are
answered by one statement rather than by three that can disagree.

**Each declaration is also a flag.** `python -m pocketllm serve --help` lists them, one flag per
option, under the section the declaration names — `--expert-pool-rows` beside `--chunk-rows` under
*Expert arena*, `--prefix-cache-bytes` and `--prefix-cache-head-tokens` under *Prefix cache*. The
help line says who reads the flag (`v41 only`, `v41 and mimo`) and what each of them answers when
nobody names it, because one parser holds every runtime's flags and `--backend` may still be `auto`
when it is parsed. The `--backend-option KEY=VALUE` spelling stays and is the more specific of the
two — it wins over the flag — which is what makes it the escape hatch for a key with no flag and the
long form of one that has.

| What a launch did | What happens |
| --- | --- |
| Named a flag the selected runtime does not read | `ConfigurationError` naming the flag and who does read it: the flags are on one command line, and a tuning option that silently does nothing is how a run ends up measured on the wrong lever. |
| Named a key the runtime does not declare | `ConfigurationError` naming the key and listing the ones it does. |
| Gave a value the declared type cannot read | `ConfigurationError` naming the key: `prefill_chunk` is a whole number, `prefix_cache_bytes` takes a `k`/`m`/`g` suffix, `pin` is a flag and reads `true`/`yes`/`on` and their negatives as well as a JSON boolean. |
| Named one option twice, once by an older spelling | `ConfigurationError`. `chunk_rows`/`expert_rows` and `expert_deal`/`deal` are each one option under two names, and which of the two was meant is not knowable from the values. |

A concept more than one runtime reads is declared once
(`pocketllm/backends/shared_options.py`): `prefill_chunk`, `prefix_cache_bytes`,
`prefix_cache_head_tokens` and `expert_deal`. Each runtime references that declaration and states
only what it answers for itself — its own default, its own way of resolving an unset value — so the
flag means one thing wherever it is read, and `--help` prints the answers side by side
(`when unset: 4g on v41 and mimo; 2g on xing4`). One declaration is spelled by a host flag instead
of a generated one: `prefill_chunk` is `--prefill-chunk-tokens`, which the native engine reads by
that name.

`device` used to be declared here too, and it is not an option any more. One name did two jobs — the
platform on the top level, a card on each of the three runtimes, read three different ways — so it
split: **`--device auto|cuda|ascend|cpu`** is the platform and **`--device-ids 2,3`** is the cards,
both on the host beside `--tensor-parallel-size`, because a card list that means the same thing on
every runtime is a fact about the launch. Rank *r* takes the r-th entry, which is the pair
`CUDA_VISIBLE_DEVICES=$rank` + `--device 0` written once for the whole world; unset keeps that pair
working. The old spelling — `--device cuda:2`, `--device 0`, `--backend-option device=...` — is
refused by name, with `--device-ids` in the message. See
[the migration note](../migration/device-splits-into-platform-and-cards.md).

The keys the CLI fills in on every launch — `engine_kind`, `routed_experts_device`, `pd_mode`, and
`nccl_id_path` for a sharded one — are accepted by every runtime and read by none of the model
option parsers, so one launch command line works for every backend.

The unified server provides:

- `GET /health`
- `GET /alive`
- `GET /ready`
- `GET /metrics`
- `GET /v1/models`
- `POST /v1/chat/completions`
- `POST /v1/completions`
- `DELETE /v1/requests/<request_id>`

`/ready` returns HTTP 503 while model loading is incomplete. `/metrics` uses dependency-free Prometheus text exposition and can later be wrapped by a richer exporter.

### The scheduler gauges

On the `cpp` backend with batching on, `/metrics` additionally carries the live `BatchScheduler`'s
own admission state:

| Series | Meaning |
| --- | --- |
| `pocketllm_requests_running` | Requests the scheduler is currently holding a slot for. |
| `pocketllm_requests_waiting` | Requests admitted but not yet running. |
| `pocketllm_slots_free` | Slots the running set is not using. |
| `pocketllm_kv_blocks{state=...}` | `total` / `free` / `reserved` / `cache_pinned`, on an engine that pages only. |

These are the same numbers, from the same `BatchScheduler::Stats`, that the native
`pocketllm_engine` server publishes — its `pocket_requests_running` is this server's
`pocketllm_requests_running`, with the same suffix and the same meaning, so the two hosts are
compared by substituting the prefix rather than by a translation table.

A Python runtime driven by the same scheduler publishes the same series, which is the point of
driving it from there: `xing4`, `v41` and `mimo` with `--enable-batching` report
`pocketllm_requests_running` out of the one `BatchScheduler` library, under the same names, without a
second implementation of it. What differs is the width the runtime declares — each declares one slot,
so that gauge reads 1 rather than the running set a wide engine has, and `pocketllm_slots_free` is 0
or 1. A runtime is charged for the width it declares, and a declaration of one is not a claim of
concurrency.

The gauges are published whether or not the model is loaded. They are the *scheduler's* numbers and
not the model's, and a scrape during a long load is exactly when an operator wants to know whether
the process has a scheduler in it at all — which is the question the series answers.

`requests_running` is the one to watch. Two concurrent clients reaching a server that serializes
them under a lock and two reaching a scheduler produce identical tokens and identical responses; the
only place they differ is this gauge, which is why it is the reading the concurrency acceptance
measurements take. `scripts/bench_cpp_scheduler_metrics.py` samples it while a group of requests is
in flight and reports the peak.

On the serialized path there is no scheduler, so none of these series is exported — not as zero.
An absent series cannot be mistaken for a measurement; a zero would read as "the scheduler is idle".

That list is the whole HTTP surface. **`/v1/embeddings` is deliberately unsupported** — PocketLLM
serves the checkpoint's text-generation path, and nothing in either plane computes a pooled
embedding, so there is no head to return, no `/v1/moderations`, `/v1/audio`, or `/v1/images` either.
An unregistered path answers 404 rather than accepting a request it would have to reinterpret.
Callers that need embeddings should run an embedding model; adding a pooling head to this engine is
a separate project from serving generation.

## Request fields

A request field is accepted only when the server acts on it. Every documented OpenAI request field
therefore falls into one of three groups, and a field in the second group has to be removed rather
than trusted.

**Which runtime acts on what.** Whether a field's *value* has a shape this server can read is one
policy for every runtime and is checked before dispatch. Whether a runtime's answer applies the field
at all depends on the runtime, and each declares its own answer through `BackendBase.audit_request`
(the `cpp` backend is the adapter that declares one today; a runtime that has declared nothing
refuses nothing and is subject to the shape checks alone). `--backend cpp` serves `stop`, `n`,
`logprobs`, `thinking_mode` and `add_generation_prompt`.

Two of those are served by code that is not in any adapter, and saying so is the point of the
division:

- **`n` is the host's dispatch.** A request for `n` choices is `n` requests to the runtime, built by
  `pocketllm/choices.py` and run by whichever entry point received the request — the HTTP server or
  `LLM.chat`. One implementation for every runtime, which is why it is not in an adapter. The one
  `n` still refused is a stochastic request against an engine that samples at engine-wide values: the
  choices differ only in the seed they are handed, and that engine reads no seed it was given, so all
  `n` would be one text presented as independent samples.
- **`logprobs` is the scheduler's.** The ranking comes off the scheduler's result, so a build whose
  scheduler was not created (`batching=false` selects the serialized compatibility session) refuses
  the field by name rather than answering with an empty array.

### Implemented

| Field | Endpoints | Behaviour |
| --- | --- | --- |
| `messages` | chat | The conversation, rendered by the checkpoint's own chat template (see [Request normalization](#request-normalization)). |
| `prompt` | completions | Tokenized and prefilled unchanged. |
| `max_tokens`, `max_completion_tokens` | both | The generation budget. `max_completion_tokens` wins when a request carries both, which is OpenAI's rule for the deprecated/current pair. A body carrying neither — or carrying `null` for either, which clients do send — asks for no cap: the answer runs until EOS or the context limit, resolved against the engine's own context the way vLLM (`max_model_len - input_length`) and SGLang resolve an absent cap. A backend that sizes its own caches — v41 — refuses a request whose prompt and budget together do not fit them, as a 400 before any work starts. |
| `temperature`, `top_p`, `top_k`, `seed` | both | Applied when the engine declares per-request sampling and top-k; otherwise a value that differs from the engine's effective one is a 400 from the sampling check rather than a silent substitution. |
| `stream` | both | Selects SSE deltas terminated by `[DONE]`. |
| `n` | both | The number of choices. Served by running the request `n` times, so the response holds one entry per choice with `index` running 0..n-1; see [Choices](#choices). |
| `response_format` | chat | Applied when the engine declares structured outputs; `text`, `json_object` and `json_schema` are supported there, and the request is refused when it is not. |
| `tools` | chat | Tool definitions reach the chat template, and a call the model writes back is reported in the assistant message's `tool_calls` rather than left in the text; see [Tool calls](#tool-calls). |
| `tool_choice` | chat | `"none"` drops the definitions, `"required"` and a named function become an instruction in the prompt; see [Tool calls](#tool-calls). |
| `stop` | both | Matched against the decoded text as it is produced, so the completion ends at the first occurrence of any sequence and the sequence itself is not part of the answer. The field is a string or a list of strings; a value of another shape is a 400. |
| `logprobs` | both | The sampled token's own log probability, and — on chat, up to `top_logprobs` of — the alternatives ranked at the same position; see [Log probabilities](#log-probabilities). A boolean on chat, a count on completions. |
| `top_logprobs` | chat | How many alternatives to rank per position alongside the sampled token. `0` reports the sampled token's probability and no alternatives. |
| `thinking_mode`, `reasoning_effort`, `add_generation_prompt`, `drop_thinking`, `request_id` | chat | PocketLLM extensions, not OpenAI fields. `thinking_mode` is `"chat"` or `"thinking"` and decides whether an answer is split at `</think>` into `content` and `reasoning_content`; `add_generation_prompt` (default `true`) decides whether the rendered prompt ends with the assistant header the model answers into, and `false` encodes the conversation as it stands — how a caller continues an assistant turn or inspects what the template does. |

#### Stop sequences

`stop` is matched against the **decoded text**, not against token ids. A stop string is not one
token — `"USER:"` is three in most vocabularies — and a sequence can begin inside one token and end
inside the next, so the only place it exists as a unit is the text the caller reads anyway. Matching
is applied to the cumulative text as it is produced, which gives the field the same meaning on a
non-streaming response and on a stream. The earliest occurrence of any sequence in the list ends the
completion, the sequence itself is not part of the answer, and `finish_reason` is reported as
`"stop"`.

Three details are worth knowing before relying on the field:

- **A partial sequence is withheld while streaming.** If the answer so far ends in a run of
  characters that is the beginning of a stop sequence, those bytes are held rather than sent,
  because the next token may complete the sequence and text already written to the socket cannot be
  taken back. Once generation ends the same bytes can no longer complete anything, so they are
  flushed as part of the answer. Nothing is withheld when the trailing characters cannot begin a
  sequence, which is the usual case — the hold is bounded by the longest sequence, not by the length
  of the text.
- **On chat, `stop` applies to the answer and not to `reasoning_content`.** The split between the
  two is read out of the text at `</think>`, and the search for a sequence starts after that marker.
  The block is a separate field whose text is not the completion, and the difference shows on the
  values clients actually send: `"\n\n"` is a common stop sequence and a reasoning block is full of
  blank lines, so matching the whole decode would end the answer before the model had written any
  of it — which a client reads as an empty answer rather than a truncated one.
- **`usage.completion_tokens` counts the tokens the engine generated**, which can exceed the number
  of tokens in the returned text when a sequence truncated it. The engine is not stopped early: the
  scheduler ends a request on token ids, and a client sequence is not one, so the request runs to its
  budget and only the text handed back is cut.

#### Choices

`n` is the number of completions one request asks for, and the server serves it by running the
request `n` times: each choice is its own runtime request, with its own seed derived from the
request's `seed` and — when `response_format` asks for one — its own grammar. The response carries
one entry per choice with `index` running 0..`n`-1. `usage` is counted the way OpenAI counts it:
`prompt_tokens` once for the request, `completion_tokens` the sum over the choices.

On a streamed response the choices arrive **one after another**, not interleaved, and each chunk
names the choice it belongs to in `index`. Interleaving would need the runtime to be driving several
requests at once, and a stream here is serialized against the engine's one mutable KV session — so
the choices are streamed in index order and the client reassembles them by index, which is what an
OpenAI stream is for. Each choice opens with its own `role: "assistant"` delta, the same way a
single-choice stream does. The practical consequence is latency rather than correctness: choice 2 of
3 does not begin until choice 1 has finished, so a client that wants all of them early is better off
sending `n` separate requests.

Four consequences are worth knowing:

- **Under greedy decoding every choice is the same text.** With `temperature` at 0 the seed is not
  read, so `n=3` returns the greedy answer three times. That is what a greedy request for three
  choices asks for; a caller who wants three different answers has to sample. The corresponding
  refusal is on the other side: an engine that fixes its sampling distribution engine-wide while the
  request asks for stochastic sampling cannot vary a choice at all, so `n>1` there is a 400 — three
  identical texts would otherwise be handed back as three independent samples.
- **`n` is refused above 128**, and refused for a fraction, a non-number, or a count below 1. The
  ceiling is this server's, not OpenAI's: one choice is one scheduler request, so the field is what
  bounds how much of the queue a single client can occupy.
- **The timeout is the request's, not the choice's.** A group of choices gets the one budget a
  single-choice request would have had, so a request wide enough that some of its choices wait
  behind the batch comes back with fewer entries than `n`. That is a 200 with a short `choices`
  array — the choices that did arrive are real answers — and not a failure. A response with no
  entries at all is a 500, or a 504 when the deadline was the reason. Cancelling the request cancels
  every choice.

#### Log probabilities

`logprobs` reports the probability the model assigned to each token it generated, and — when a count
of alternatives is given — the probabilities it assigned to the tokens it did *not* generate. The
two endpoints spell the same request differently, and this server follows each spelling rather than
picking one: on chat `logprobs` is a **boolean** and the number of alternatives lives in
`top_logprobs`, while on `/v1/completions` `logprobs` is the **count** itself. They are not
interchangeable, and the difference is not cosmetic — on chat `logprobs=false` means "not asked for",
while on completions `logprobs=0` is a real request for the sampled token's own probability with no
alternatives. A count sent to chat, or a boolean sent to completions, is a 400.

The answer is an array of one object per generated token, in order, under the choice's `logprobs`
key:

```json
{"logprobs":{"content":[
  {"token":"1","logprob":-0.0001234,"bytes":[49],"top_logprobs":[
    {"token":"1","logprob":-0.0001234,"bytes":[49]},
    {"token":"2","logprob":-9.21,"bytes":[50]}
  ]}
]}}
```

- **`token` is the surface text of one token**, not a word: `bytes` holds its UTF-8 encoding, which
  is how a caller reassembles text that a multi-byte character was split across. A token holding one
  piece of a multi-byte character is not valid UTF-8 on its own, so a client that wants the exact
  bytes should read `bytes` rather than re-encoding `token` — concatenating the `bytes` arrays in
  order reproduces the answer.
- **`logprob` is a natural log**, so it is always ≤ 0 and `exp(logprob)` is the probability.
- **`top_logprobs` ranks the model's own distribution, not the sampler's candidate set.** It is
  computed from the same raw logits the sampler draws from but over the whole vocabulary and before
  `temperature`, `top_k` or `top_p` touch it, so the numbers are comparable across positions and
  across requests. Ranking only the sampler's top-k candidates would inflate every probability by
  whatever mass the truncation dropped. Under `temperature` 0 the generated token is the argmax and
  is therefore the first entry, with the same `logprob` reported twice; when the request samples, the
  generated token is somewhere inside the requested alternatives rather than necessarily first.

Four things are worth knowing before relying on the field:

- **The array covers the text, not the token budget.** A stop token or a client `stop` sequence cuts
  the answer, and the array is cut with it — a position the caller never received is not reported.
  `usage.completion_tokens` still counts the tokens the engine generated, so it can exceed the number
  of entries in `content`.
- **On chat it covers the answer and not the reasoning block.** A thinking model decodes its
  reasoning first, and `message.content` is what follows it, so the array starts where the content
  starts: a client indexing into `content` and a client indexing into `logprobs.content` are looking
  at the same position. The reasoning block's own probabilities are therefore not reported, even
  though they were generated.
- **Streaming is not supported**, because a chunk carries the text of its token with no ranking
  beside it. `{"stream":true,"logprobs":...}` is a 400 rather than a stream that looks the same as
  one whose request asked for no ranking at all.
- **The engine has to declare it, and the runtime has to be able to ask.** `logprobs` is refused
  when the capability is off, which is the case for speculative decoding (its verify step ranks no
  tokens, so a row that emitted several has no ranking for the rest) and for the Ascend backend. On
  `--backend cpp` the ranking comes off the batch scheduler, so a build running the serialized
  compatibility session (`batching=false`) refuses the field by name rather than answering with an
  empty array. The limit on alternatives is this server's — 20 per position, above OpenAI's
  documented range — and a request past it is a 400 naming the ceiling.

#### Tool calls

`tools` is forwarded to the checkpoint's own chat template, and a call the model writes back is
reported in the assistant message's `tool_calls` array instead of being left in the text. The array
uses the OpenAI shape — an `id`, `type` set to `"function"`, and `function.name` with
`function.arguments` as a JSON string — and `finish_reason` is `"tool_calls"`, which is what a client
branching on the field expects to see before it runs the call and sends the result back in a
`tool`-role message.

That round trip is the one the acceptance test drives: the assistant message the server returned is
replayed verbatim alongside a `role: "tool"` result keyed on its `id`, and the answer comes back as an
ordinary `stop`. Both turns were checked through the `openai` and `langchain-openai` clients as well as
over bare HTTP — the [tool-calling acceptance record](../performance/cpp_openai_tool_acceptance.md)
has the request and response shapes.

Five things are worth knowing before relying on the field:

- **The schema is what types the arguments.** Qwen's template writes a call as XML —
  `<tool_call><function=NAME><parameter=ARG>VALUE</parameter></function></tool_call>` — and that shape
  records no type of its own: `<parameter=days>3</parameter>` is one character more than
  `<parameter=days>two</parameter>`. Each value is therefore read back against the type the request's
  own `parameters.properties` declares, so an argument declared `integer` arrives as the number `3`
  and one whose declared type is missing, unknown, or not parseable from the text arrives as the
  characters that were written. A request with no `tools` at all leaves every argument a string,
  because nothing in the text can settle whether `01234` meant a number.
- **Parsing is all-or-nothing.** A completion whose call was truncated at the token budget, is
  malformed, or has prose between two calls yields **no** `tool_calls`, and the text stays in
  `content` exactly as generated. A half-read call whose arguments look complete is worse than one
  whose syntax the caller can see.
- **Only a call syntax this server has read is parsed.** That is Qwen's template (`qwen3_5`,
  including the `qwen3_5_text` spelling) and DeepSeek-V4's own encoder, which already parsed its
  DSML calls. Any other architecture keeps the older behaviour and leaves the call in `content`;
  inventing a parse for a syntax nobody has read would drop or corrupt calls silently. The
  selection is one implementation (`pocketllm/protocol/templating.py`), so the checkpoint's
  architecture decides it the same way whichever backend served the request — that module used to be
  the C++ front end's sidecar, which is why a `cpp` request through `pocketllm serve` answered with
  the call as prose while the same checkpoint through the native binary answered with `tool_calls`.
- **Streaming is not supported.** A streamed response carries the call syntax as content, exactly as
  it did before, and reports the engine's own `finish_reason`. Ask for a non-streaming response when
  you want `tool_calls`. The *reasoning* split is a different matter and does happen on a stream: a
  thinking-mode answer sends everything before `</think>` as `reasoning_content` deltas, so a client
  watches the reasoning instead of waiting for the answer.
- **The selection policy is applied.** `tool_choice: "auto"` is the default; `"none"` drops the
  definitions before the template is rendered, and `"required"` or a named function becomes an
  instruction in the prompt naming what must be called. The policy reaches the model rather than
  constraining the sampler, which is what SGLang does with the same field, so a client that asks for
  a call still reads the answer's `tool_calls` to find out whether one was made. A `tool_choice`
  naming a function that is not among `tools` is a 400. `parallel_tool_calls: false` is a 400: the
  model decides how many calls it makes and nothing here limits the count.

### Refused with HTTP 400

Each of these is refused only at a value that would change the output. The same field at the value
naming what the server already does — `logprobs=false` on chat, penalties of zero, an empty `stop`
list, an empty `logit_bias`, `echo=false` — is accepted, so a client that sends the documented
defaults explicitly is not punished for it. The two entries for a field this server *does* implement
are shape checks on the endpoint that defines the value, not refusals of the feature.

| Field | Endpoints | Refused when | What this server does instead |
| --- | --- | --- | --- |
| `stop` | both | the value is not a string and not a list of strings | Nothing is matched, so a well-formed `stop` is refused on shape alone rather than half-applied. Empty strings match nothing and are accepted, which is what makes an empty `stop` list — or the empty entries some clients pad it with — harmless. |
| `logprobs` | chat | not a boolean | A count is the other endpoint's spelling of the field; see [Log probabilities](#log-probabilities). |
| `logprobs` | completions | not a whole number in 0..20 | It is the number of alternatives to rank per position, above this server's ceiling of 20. |
| `top_logprobs` | completions | any value but `null` | The completions endpoint names the count in `logprobs` itself. |
| `top_logprobs` | chat | not a whole number in 0..20, or positive while `logprobs` is absent or `false` | There is no ranking to take alternatives from unless the request asked for log probabilities. |
| `logprobs` | both | asked for on a streaming request | A streamed chunk carries the text of its token with no ranking beside it. |
| `frequency_penalty`, `presence_penalty` | both | non-zero | The sampler has no repetition or presence term, so the request is generated as if the penalty were 0. |
| `logit_bias` | both | the object is not empty | No per-token bias is applied, so biased tokens are sampled at their unmodified probability. |
| `best_of` | completions | not 1 | One candidate is generated per request; there is no second candidate to compare it against. |
| `suffix` | completions | non-empty | The completion is returned on its own, with no suffix appended. |
| `echo` | completions | `true` | `text` holds only the generated continuation, never the prompt. |
| `tool_choice` | chat | a function choice naming a tool that is not in `tools` | A request error rather than a named-field refusal: the policy is applied to the prompt, so a name the prompt cannot be given is a request the server cannot render. |
| `parallel_tool_calls` | chat | `false` | The number of tool calls the model emits is not limited. |
| `stream_options.include_usage` | both | `true` on a streaming request | A stream is delta chunks followed by `[DONE]`, and none of them carries `usage`. A non-streaming response already reports usage, so the option is satisfied there and accepted. |

The refusal uses the OpenAI error shape with `type` set to `invalid_request_error`, `param` set to
the offending field, and `code` to `unsupported_feature`, so a client can act on it without parsing
the prose:

```json
{"error":{"message":"\"stop\" = 5 is not supported by this server: a stop sequence is a string, or a list of strings, and this value is neither. Send \"stop\" as a string or an array of strings.","type":"invalid_request_error","param":"stop","code":"unsupported_feature"}}
```

The check runs before dispatch, so a request this server will not serve is refused whole rather than
streamed halfway and abandoned. Which fields are refused at which values is
`pocketllm/protocol/contract.py`, and the adapter's own answer is
`BackendBase.audit_request`: the *shape* of a field is the same on every runtime and is checked
host-side, while whether a runtime's answer applies a field at all depends on the runtime and is
declared by it. A runtime that has not declared anything refuses nothing, which is why the sections
above describe the socket rather than a policy — the `cpp` backend is the adapter that declares one
today.

### Accepted and inert

These cannot change the generated text, so they are accepted and ignored rather than refused:
`user`, `store`, `metadata`, `service_tier`, and `model`. The server serves exactly one model and
echoes its configured name back, so a `model` naming something else is not a routing request it can
honour — but rejecting it would break clients over nothing.

`parallel_tool_calls` is the exception that shows the rule is applied per value rather than per
field: `true` is inert and accepted, while `false` asks for a limit that is not enforced and is
refused with the rest of the table above.

## Configuration precedence

Prefer typed `EngineArgs` and explicit CLI options. `EngineArgs.from_env()` exists as a compatibility bridge for legacy deployments. Runtime tuning variables are named `POCKETLLM_*` (renamed from `DSV4_*`, a breaking change — see [the migration note](../migration/dsv4-to-pocket-rename.md)); `QWEN_*` and related names are unchanged. Backend-specific tuning belongs in `backend_options` and must not be assumed portable between CUDA and Ascend.

## Native C++ Python module

The native bridge is optional and does not affect CPU-only imports. You can build it as part of
`pip install` (recommended) or manually via CMake.

### Via pip install

```bash
POCKETLLM_BUILD_CPP=1 pip install --no-build-isolation .
```

The `--no-build-isolation` flag ensures the active environment's Torch is the one that drives the
Torch extension build. Without `POCKETLLM_BUILD_CPP=1`, the install skips the native module and
produces only the Torch runtime.

The native module installs top-level (`import pocketllm_cpp`), so no manual copy is needed.

### Manual CMake build

```bash
cmake -S cpp_engine -B cpp_engine/build-python \
  -DPOCKET_BACKEND=cuda \
  -DPOCKET_BUILD_PYTHON=ON \
  -Dpybind11_DIR="$(python -c 'import pybind11; print(pybind11.get_cmake_dir())')"
cmake --build cpp_engine/build-python --target pocketllm_cpp -j
```

Add `cpp_engine/build-python/python` to `PYTHONPATH` for a build-tree smoke test:

```bash
PYTHONPATH=cpp_engine/build-python/python python -c \
  'import pocketllm_cpp; print(pocketllm_cpp.backend)'
```

The module exposes token-oriented `QwenEngine` and low-level `PersistentEngine` value types. Device-touching calls (prefill, decode, generate, verify, warmup, reset) release the Python GIL; cheap accessors and construction do not. It intentionally does not expose CUDA/ACL handles or Torch tensors.

### Scheduler-backed async requests

`QwenBatchScheduler` wraps an engine in the same continuous-batching scheduler the native
OpenAI server uses, so Python can submit concurrent requests and stream tokens without going
through HTTP.

```python
import threading
import pocketllm_cpp

engine = pocketllm_cpp.QwenEngine(checkpoint, pocketllm_cpp.QwenEngineOptions())
scheduler = pocketllm_cpp.QwenBatchScheduler(engine, max_batch_size=4)

sampling = pocketllm_cpp.QwenBatchSamplingParams()
sampling.max_new_tokens = 64

done = threading.Event()

def on_token(request_id, token):
    print(token, flush=True)

def on_complete(result):
    print(result.finish_reason)
    done.set()

request_id = scheduler.submit_request(
    prompt_tokens, sampling, callback=on_complete, on_token=on_token)
done.wait()
scheduler.stop()
```

Omit both callbacks to poll instead; `poll_result` returns `None` on timeout:

```python
request_id = scheduler.submit_request(prompt_tokens, sampling)
result = scheduler.poll_result(request_id, timeout_ms=30000)
```

Both callbacks run on the scheduler's background thread, so they **must not block** — time
spent there delays every other running request. Push the token onto a queue and return. An
exception raised inside a callback is reported on stderr and swallowed rather than being allowed
to cross the thread boundary and terminate unrelated requests.

`engine_caps()` reports what the engine actually supports, which is what a caller should branch on
rather than assuming:

```python
caps = scheduler.engine_caps()
caps.max_slots, caps.continuous_batching, caps.chunked_prefill, caps.paged_kv
caps.per_request_sampling, caps.per_request_top_k
```

`max_batch_size()` returns the effective batch size, which may be lower than requested because it
is clamped to `caps.max_slots`. `set_prefill_token_budget(tokens)` controls how much prefill runs
per schedule iteration: smaller values let decode interleave sooner at some cost to prefill
throughput, and `0` disables chunking so each prompt runs to completion in one call. It has no
effect on engines that do not declare `chunked_prefill`.

The engine must outlive the scheduler; the scheduler holds a non-owning pointer, matching the C++
ownership model. Call `stop()` for a deterministic shutdown rather than relying on collection order.

## Backend selection

`backend="auto"` picks the C++ adapter only when the native module is importable and the checkpoint
is a Qwen3.5 safetensors model; anything else, including GGUF, stays on Torch. An explicit
`backend="cpp"` for an unsupported checkpoint raises `UnsupportedFeatureError` before any CUDA
initialization instead of failing deep inside the native loader.

`backend="v41"` is the third adapter and the first thing `auto` tests for: a checkpoint whose config
says `deepseek_v41` — at the root, or `deepseek_v41_text` under `text_config`, which is where the
released V4.1 file keeps it — goes to it before the native adapter is even considered, because the
native engine has no factory for that architecture. It runs the `src/models/deepseek_v4_1` PyTorch
runtime over the checkpoint's safetensors shards, one process a rank under `--tensor-parallel-size`,
and it reports `supports_batch=False`: one mutable KV state, serialized at the backend boundary.
`--backend v41` on a GGUF checkpoint, or on a config that is not V4.1, raises
`UnsupportedFeatureError` before anything loads.

Capabilities reported by the C++ adapter follow the linked device backend. An Ascend build advertises
only the speculative methods it implements, since the external DSpark and DFlash2 drafters are
CUDA-only. The V4.1 adapter advertises no logprobs — a request asking for them is refused rather than
served without them — while prefix caching follows `prefix_cache_bytes`, which defaults to 4 GiB a
rank and can be set to zero to turn the reuse off. Its `cancellation` detail names the mechanism
rather than promising a latency: a cancellation is a per-step collective between the ranks and cannot
interrupt a prompt's forward.

`backend="mimo"` is the adapter for MiMo-V2.6-Flash. `--backend mimo` names it, and `auto` reaches it
too — the checkpoint's `model_type` is `mimo_v2`, which the factory recognizes the way it recognizes
`deepseek_v41`, so a MiMo release gets this adapter from either. What it runs is
[MiMo-V2.6-Flash](../models/mimo-v2.6-flash.md)'s runtime: `src/models/mimo_v2` over the release,
the routed experts in host memory, one process a rank under `--tensor-parallel-size`, and it reports
`supports_batch=False`. Two things about it are not the other adapters':

- **Rank 0 cannot start a request without telling the workers.** Every routed layer closes with an
  `all_reduce`, so a rank that is not running the request its peers are running is at a *different*
  collective and NCCL answers that by hanging. The adapter broadcasts each request — prompt ids,
  budget, sampler, seed — before it runs it, and a cancel is a per-step `broadcast` of one flag for
  the same reason. A rank that decided to stop on its own would leave three peers inside a layer.
- **`max_model_len` is the KV cache.** A MiMo deployment sizes one cache at startup (32768
  positions by default) and every request is clamped to what is left of it; a prompt that fills it
  is refused before any work starts, with the number and the flag to raise. Its options are
  `--prefill-chunk-tokens` (tokens a prefill call, 2048 by default), `--chunk-rows` (experts a
  grouped expert call, which trades arena bytes for call count), `--slots`, `--expert-deal` (`deal`
  is the older spelling of the same option as a `--backend-option` key) and `--pin`, and an unknown
  one is a `ConfigurationError` rather than a silent default.

## Request normalization

`pocketllm.protocol` holds the request normalization the server runs: OpenAI content-block
flattening, tool attachment and `tool_choice` instructions, `reasoning`/`reasoning_effort` handling,
tool-call shaping, and stop-string truncation. There is one implementation, and it imports neither
Torch nor the native module. It used to be shared with a second, model-owned server
(`src.server.openai`, since retired with the rest of the duplicate front ends — see
[#447](https://github.com/lvyufeng/PocketLLM/issues/447)); the module that server's runtime half
became is `src/models/deepseek_v4/serving.py`.

`/v1/chat/completions` puts the normalized messages, thinking mode, reasoning effort, and tool
metadata in `GenerationRequest.metadata`. The shared prompt boundary first asks the selected
checkpoint tokenizer to apply its own `chat_template` with an assistant generation prompt. This is
the same model-owned-template contract used by vLLM/SGLang and preserves model-specific special
tokens, reasoning controls, and tool formatting. For DeepSeek checkpoints whose tokenizer has no
chat template, the validated legacy `src.encoding.deepseek_v4.encode_messages` format is used instead.
`GenerationRequest.prompt` still carries a deterministic `role: content` rendering only as a last-resort
fallback for generic tokenizers that provide neither format. `/v1/completions` passes `prompt` through
unchanged and validates that a list prompt contains only strings.

The template receives normalized tool definitions and a private compatibility copy of prior tool-call
arguments; public request metadata is never mutated. Template-specific reasoning names are mapped to
the vocabulary accepted by the checkpoint (for example, `high`/`max` map to Qwen's `xhigh`).
Unsupported model-specific template features remain the responsibility of the selected backend.

**DeepSeek-V4.1-Flash is the case that contract does not cover, because its tokenizer has no
template at all**: the format is a Python module the release ships inside the checkpoint, at
`<checkpoint>/encoding/encoding.py`. `backend="v41"` therefore renders a chat request by importing
that module by path and calling its own `encode_messages`, rather than by falling back to a generic
renderer the model was never trained on. What this repository wraps around it is the reasoning-effort
vocabulary (`1`–`100`, or `minimal`/`medium`/`low`/`high`/`max`) and a tolerant `</think>` split for
a reply that stops mid-reasoning; tool definitions ride along on the first system message, where the
control plane already put them. A checkpoint without that module, or an effort the encoder will not
render, is refused with `ConfigurationError` and a pointer to `/v1/completions`, so a caller can send
a prompt it rendered itself. `/v1/completions` on this backend passes the prompt through the
tokenizer untouched, which is also what the launcher does with `--prompt`.

A backend that separates reasoning from content can set `reasoning_content` and `tool_calls` in its
result or event metadata; those are forwarded to the response and to streamed deltas. A backend that
does not simply omits them.

## Termination semantics

The C++ adapter decides when generation stops. The first EOS token ends the request, is excluded from
the returned token ids and text, and yields `finish_reason="stop"`. `finish_reason="length"` means the
token budget ended first. Usage counts the EOS step the engine executed, so streaming and offline
usage agree.

`QwenEngine.generate` takes no EOS argument and keeps mutating its session for the whole token budget,
so when an EOS id is known both the offline and streamed paths drive `prefill`/`decode_step`
themselves and stop at EOS. Running `generate()` and truncating afterwards would leave the recurrent
state and prefix cache positioned past text the caller never saw, corrupting reuse for the next
request. Native `generate()` is still used when no EOS id is available, where the token budget is the
only stopping rule.

EOS ids are resolved in order: `backend_options["eos_token_id"]`, the native engine's `eos_id`, the
native config, the checkpoint's `generation_config.json`, the checkpoint's `config.json`, then the
tokenizer. `generation_config.json` is preferred over the tokenizer because chat checkpoints commonly
stop on a turn-end token that differs from the tokenizer's EOS. A non-integer override is rejected
rather than guessed. When no EOS is available, `capabilities.details["eos_source"]` reports `none` and
only the token budget can end generation. Streaming never issues another `decode_step` after EOS, and
it never calls `reset()` per request, since `QwenEngine::reset()` would clear the prefix cache that
`prefill()` relies on.

## Cancellation semantics

`cancel(request_id)` returns `True` only for a request that is currently active, and cancellation is
observed at safe boundaries between generation steps. It never interrupts a running device kernel and
never rolls back a partially executed native step. `DELETE /v1/requests/<request_id>` returns HTTP 404
for an unknown or already-finished request.

The existing `pocketllm_engine` executable and its CLI remain supported. The shared Python server is a migration path, not a replacement that invalidates existing production commands.
