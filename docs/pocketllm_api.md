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

## Cancellation semantics

`cancel(request_id)` returns `True` only for a request that is currently active, and cancellation is
observed at safe boundaries between generation steps. It never interrupts a running device kernel and
never rolls back a partially executed native step. `DELETE /v1/requests/<request_id>` returns HTTP 404
for an unknown or already-finished request.

The existing `dsv4_cpp_engine` executable and its CLI remain supported. The shared Python server is a migration path, not a replacement that invalidates existing production commands.
