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

Use `generate_stream()` for token events and `cancel(request_id)` to request cancellation at a safe generation boundary. The initial C++ compatibility adapter is serialized and exposes native greedy generation; unsupported sampling or request features report `UnsupportedFeatureError` rather than being silently ignored. Native streaming decodes the cumulative token sequence before emitting each delta, so BPE and UTF-8 token boundaries are handled by the tokenizer.

## Async API

```python
from pocketllm import AsyncLLM, EngineArgs, SamplingParams

async with AsyncLLM(EngineArgs(model="/path/to/checkpoint")) as llm:
    result = (await llm.generate("Hello", SamplingParams(max_tokens=32)))[0]
    async for event in llm.generate_stream("Stream this"):
        print(event.text, end="", flush=True)
```

`AsyncLLM` currently provides non-blocking application integration around the backend contract. It does not claim device-level continuous batching. Backend schedulers will add that capability independently.

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

The CLI does not automatically launch tensor-parallel worker processes; start the configured
workers using the deployment's existing launcher when `tensor_parallel_size > 1`.

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

## Configuration precedence

Prefer typed `EngineArgs` and explicit CLI options. `EngineArgs.from_env()` exists as a compatibility bridge for legacy deployments. Existing `DSV4_*`, `QWEN_*`, and related environment variables remain available to the underlying runtimes. Backend-specific tuning belongs in `backend_options` and must not be assumed portable between CUDA and Ascend.

## Native C++ Python module

The native bridge is optional and does not affect CPU-only imports:

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
metadata in `GenerationRequest.metadata`, so a backend applies its own chat template. The Torch
adapter encodes them with the DeepSeek template that the legacy server uses, so both servers build
the same prompt. `GenerationRequest.prompt` still carries a deterministic `role: content` rendering as
a fallback for backends that have no template of their own. `/v1/completions` passes `prompt` through
unchanged and validates that a list prompt contains only strings.

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

The existing `dsv4_cpp_engine` executable and its CLI remain supported. The shared Python server is a migration path, not a replacement that invalidates existing production commands.
