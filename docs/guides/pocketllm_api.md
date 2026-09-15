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

`AsyncLLM` currently provides non-blocking application integration around the backend contract. It does not claim device-level continuous batching. Backend schedulers will add that capability independently. The async chat methods reuse the same executor-backed lifecycle and `TokenEvent` contract as the sync facade; they do not add a scheduler.

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

The built-in supervisor currently works with the Torch backend by reusing its existing NCCL/Gloo
worker loop. The Python C++ Qwen adapter does not yet expose a native worker entry point, so
`backend="cpp"` must use the legacy `pocketllm_engine` launcher or opt out with
`--no-tensor-parallel-supervisor`. Existing `torchrun` and manual rank launchers remain compatible
through that opt-out. This process supervisor is not a scheduler and does not provide continuous
batching or request-local native state.

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

### Implemented

| Field | Endpoints | Behaviour |
| --- | --- | --- |
| `messages` | chat | The conversation, rendered by the checkpoint's own chat template (see [Request normalization](#request-normalization)). |
| `prompt` | completions | Tokenized and prefilled unchanged. |
| `max_tokens`, `max_completion_tokens` | both | The generation budget. `max_completion_tokens` wins when a request carries both, which is OpenAI's rule for the deprecated/current pair. |
| `temperature`, `top_p`, `top_k`, `seed` | both | Applied when the engine declares per-request sampling and top-k; otherwise a value that differs from the engine's effective one is a 400 from the sampling check rather than a silent substitution. |
| `stream` | both | Selects SSE deltas terminated by `[DONE]`. |
| `response_format` | chat | Applied when the engine declares structured outputs; `text`, `json_object` and `json_schema` are supported there, and the request is refused when it is not. |
| `tools` | chat | Tool definitions reach the chat template. The model still chooses; see `tool_choice` below. |
| `stop` | both | Matched against the decoded text as it is produced, so the completion ends at the first occurrence of any sequence and the sequence itself is not part of the answer. The field is a string or a list of strings; a value of another shape is a 400. |
| `thinking_mode`, `reasoning_effort`, `add_generation_prompt`, `drop_thinking`, `request_id` | chat | PocketLLM extensions, not OpenAI fields. |

#### Stop sequences

`stop` is matched against the **decoded text**, not against token ids. A stop string is not one
token — `"USER:"` is three in most vocabularies — and a sequence can begin inside one token and end
inside the next, so the only place it exists as a unit is the text the caller reads anyway. Matching
is applied to the cumulative text as it is produced, which gives the field the same meaning on a
non-streaming response and on a stream. The earliest occurrence of any sequence in the list ends the
completion, the sequence itself is not part of the answer, and `finish_reason` is reported as
`"stop"`.

Three details are worth knowing before relying on the field:

- **A partial sequence is withheld while streaming.** If the text so far ends in a run of characters
  that is the beginning of a stop sequence, those bytes are held rather than sent, because the next
  token may complete the sequence and text already written to the socket cannot be taken back. Once
  generation ends the same bytes can no longer complete anything, so they are flushed as part of the
  answer. Nothing is withheld when the trailing characters cannot begin a sequence, which is the
  usual case — the hold is bounded by the longest sequence, not by the length of the text.
- **On chat, `stop` applies to the answer and not to `reasoning_content`.** The reasoning block is a
  separate field that ends on a token id, and a sequence that appeared inside it would otherwise
  truncate the answer that follows.
- **`usage.completion_tokens` counts the tokens the engine generated**, which can exceed the number
  of tokens in the returned text when a sequence truncated it. The engine is not stopped early: the
  scheduler ends a request on token ids, and a client sequence is not one, so the request runs to its
  budget and only the text handed back is cut.

### Refused with HTTP 400

Each of these is refused only at a value that would change the output. The same field at the value
naming what the server already does — `n=1`, `logprobs=false`, penalties of zero, an empty `stop`
list, an empty `logit_bias`, `echo=false` — is accepted, so a client that sends the documented
defaults explicitly is not punished for it.

| Field | Endpoints | Refused when | What this server does instead |
| --- | --- | --- | --- |
| `n` | both | not 1 | `choices` always holds exactly one entry, with index 0. |
| `stop` | both | the value is not a string and not a list of strings | Nothing is matched, so a well-formed `stop` is refused on shape alone rather than half-applied. Empty strings match nothing and are accepted, which is what makes an empty `stop` list — or the empty entries some clients pad it with — harmless. |
| `logprobs` | chat | anything but `false` | No `logprobs` object is returned on any choice. |
| `logprobs` | completions | any value | It is a count there, where even `0` asks for the sampled token's logprob, so no value is inert. |
| `top_logprobs` | both | not 0 | There are no per-token logprobs to rank alternatives within. |
| `frequency_penalty`, `presence_penalty` | both | non-zero | The sampler has no repetition or presence term, so the request is generated as if the penalty were 0. |
| `logit_bias` | both | the object is not empty | No per-token bias is applied, so biased tokens are sampled at their unmodified probability. |
| `best_of` | completions | not 1 | One candidate is generated per request; there is no second candidate to compare it against. |
| `suffix` | completions | non-empty | The completion is returned on its own, with no suffix appended. |
| `echo` | completions | `true` | `text` holds only the generated continuation, never the prompt. |
| `tool_choice` | chat | anything but `"auto"` | Tool definitions reach the chat template, but the model is not constrained to call a tool, skip them, or call one function, so the policy has no effect. |
| `parallel_tool_calls` | chat | `false` | The number of tool calls the model emits is not limited. |
| `stream_options.include_usage` | both | `true` on a streaming request | A stream is delta chunks followed by `[DONE]`, and none of them carries `usage`. A non-streaming response already reports usage, so the option is satisfied there and accepted. |

The refusal uses the OpenAI error shape with `type` set to `invalid_request_error` and `param` set
to the offending field, so a client can act on it without parsing the prose:

```json
{"error":{"message":"\"n\" = 3 is not supported by this server: \"choices\" always holds exactly one entry and its index is always 0. Remove \"n\", or set it to 1 and read the single choice.","type":"invalid_request_error","param":"n","code":null}}
```

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

Capabilities reported by the C++ adapter follow the linked device backend. An Ascend build advertises
only the speculative methods it implements, since the external DSpark and DFlash2 drafters are
CUDA-only.

## Request normalization

`pocketllm.protocol` holds the request normalization shared by the unified server and the legacy
`src.server.openai` server: OpenAI content-block flattening, tool attachment and `tool_choice`
instructions, `reasoning`/`reasoning_effort` handling, tool-call shaping, and stop-string truncation.
There is one implementation, and it imports neither Torch nor the native module.

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
