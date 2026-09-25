# Ternary-Bonsai-2-27B

A 27B hybrid-attention text model — 48 Gated DeltaNet layers and 16 full-attention layers over a
**dense** 17,408-wide MLP — released as a GGUF whose weights are **1.75 bits each**. PocketLLM runs
the whole checkpoint on **one** 2080 Ti, behind the same OpenAI-compatible server as its other
models.

- **Backend**: `pocketllm serve`, `cpp` adapter (the default for this checkpoint — no flag needed)
- **Parallelism**: 1 GPU
- **Context**: the checkpoint declares 262,144; **245,760 tokens** is what fits on a 22 GiB card at
  an FP16 KV cache, and the full 262,144 fits with `--kv-cache-dtype fp8`
- **Validated on**: 1×RTX 2080 Ti 22 GiB, PCIe Gen3

## Overview

The checkpoint is `prism-ml/Ternary-Bonsai-2-27B-gguf`, and its header says what it is: 64 blocks
with `full_attention_interval = 4`, hidden 5120, 24 query heads over 4 KV heads at head length 256,
a dense MLP of 17408, and `general.architecture = qwen35`. That last field is why this model needs
no new runtime: PocketLLM's Qwen3.8-27B path already dispatches on it, because the two are the same
architecture. The attention, the recurrent layers, the loader, the server, the batching and the
prefix cache are the ones already validated for that model.

What is new is the container, and there are two ideas in it.

**The weights are ternary, and they are consumed as ternary.** Every large matrix in the file is
GGML type **143** (`PTQ1_0`): 128 weights in 28 bytes, one scale per block, 1.75 bits a weight. That
is 5.53 GiB for a 27B-parameter model, and there are no high-precision escape hatches behind the
label — `token_embd` and `output.weight` are ternary too. The kernels expand each block's trits to
signed bytes and take an ordinary integer dot product, so nothing falls back to fp16 ten times the
size.

**The file's weights are pre-rotated, and the activations are rotated to match.** Every folded
matrix in the file stores `W · R⁻¹` for a normalized Sylvester–Walsh–Hadamard transform `R`, and
`token_embd` is the one tensor that stores `R` applied rather than inverted. The runtime applies
`R` to the activation before each of those matmuls; the product is what the model computed before
rotation, and the rotation is what makes a 1.75-bit weight able to represent that function at all.
The transform is declared in the file (`prism.hadamard.*`), not inferred from a name, and the engine
refuses a container whose declaration it cannot honour rather than guessing.

The practical consequence of the packing is the memory: 5.53 GiB of weights leaves a 22 GiB card
roughly 15 GiB for KV, which is what a 245,760-token context costs here. The FP8 sibling of this
model needs four cards.

### One token's path

A decode step reads the whole 5.53 GiB once through a `__dp4a` GEMV, then runs the 16 full-attention
layers over a KV cache that grows with context and the 48 Gated DeltaNet layers over a recurrent
state that does not. That split is why decode is nearly flat in context — see the table below — and
why the model fits a long context on one card at all: only a quarter of the layers pay for depth.

## Run it

### Serve it

```bash
python -m pocketllm serve \
  --model /path/to/Ternary-Bonsai-2-27B-PTQ1_0.gguf \
  --served-model-name bonsai \
  --max-model-len 8192 \
  --port 8000
```

`--backend` is not needed: the adapter registry reads the file's own `general.architecture`
(`qwen35`), canonicalizes it to `qwen3_5`, and selects the native engine that claims that name. The
checkpoint is one `.gguf` file and the tokenizer, the special-token ids and the chat template all
come out of its header, so nothing else has to be passed and no companion directory is required.

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "bonsai", "messages": [{"role": "user", "content": "Hello!"}], "stream": true}'
```

The endpoint serves `/health`, `/ready`, `/v1/models`, `/v1/chat/completions`, `/v1/completions`,
SSE streaming, `DELETE /v1/requests/{id}` and `/metrics`.

The model must be named as the **file**. The released directory holds a second artifact beside it
(`Ternary-Bonsai-2-27B-PQ2_0.gguf`), and a directory naming two models is refused rather than
resolved by listing order:

```text
UnsupportedFeatureError: the native C++ adapter serves the Qwen3.5 GGUF export, as a single .gguf
file declaring general.architecture=qwen35; other GGUF checkpoints must use backend='torch'
```

### Serving options

| Option | Default | What it does |
| --- | ---: | --- |
| `--max-model-len` | 8192 | Positions the caches hold. **The KV arena is sized from this at start**, so it is a memory decision as much as a context one: 64 KiB a token. |
| `--kv-cache-dtype` | `auto` → `fp16` | `fp8` halves the KV and is what the checkpoint's own 262,144 fits with. It changes the arithmetic, so it is not free. |
| `--enable-prefix-caching` | on | Resumes a prompt that repeats the one just served instead of forwarding it again. |
| `--prefill-chunk-tokens` | 8192 | Tokens one prefill call takes at once. |
| `--backend-option enable_batching=true` | off | Turns on the shared batch scheduler; `--backend-option max_batch_size=4` sizes it. **Off by default: it buys aggregate throughput and costs per-request latency — see Known limitations.** |
| `--backend-option kv_paged=true` | off | Paged KV blocks instead of one contiguous arena. Memory-neutral on its own, and measured to give up the prefix resume — see Known limitations. |
| `--tensor-parallel-size` | 1 | The engine supports TP4, which is how the FP8 sibling is served, but **TP > 1 was not measured for this artifact**. |

### Without a server

```bash
# the engine, all 64 layers, greedy, on the released file: prints the generated ids and text
cpp_engine/build-python/tests/test_qwen_ternary_engine

# the weight map against the FP8 sibling, and the transform against the fork's own bytes
cpp_engine/build-python/tests/test_qwen_gguf_weights
cpp_engine/build-python/tests/test_qwen_hadamard_ops

# the container alone: 851 tensors, read from the file's header
python -m pytest tests/test_ptq1_0_layout.py -q
```

## What is supported

| Capability | State |
| --- | --- |
| OpenAI-compatible serving (chat, completions, streaming, cancel, metrics) | Supported, validated on one card |
| The 1.75-bit `PTQ1_0` container, both phases | Supported; a ternary tensor is never upcast to fp16 |
| The checkpoint's Hadamard transform, read from the file's declaration | Supported |
| 245,760-token context (FP16 KV) / 262,144 (fp8 KV) | Supported and measured |
| Cross-request prefix resume | Supported, on by default; one prompt deep |
| Concurrent requests | Serialized by default; the batch scheduler is opt-in and buys 1.14–1.78× aggregate at the cost of per-request latency |
| Sampling (`temperature`, `top_k`, `top_p`) | Supported by the engine; the adapter exposes greedy generation only |
| MTP, DSpark, DFlash2 speculative decoding | **Not validated on this checkpoint** |
| `Ternary-Bonsai-2-27B-PQ2_0.gguf` (type 142, 2.13 bits) | Declared in the loader, **no kernel** — it is refused, not upcast |
| TP > 1 | Unmeasured for this artifact |
| Vision | The chat template renders images; nothing reads them |

## Performance

One RTX 2080 Ti, `CUDA_VISIBLE_DEVICES=0`, the released `PTQ1_0` file, one request at a time,
`--max-model-len 32768`, greedy. The prompts are real repository documents cut to an exact token
count, sent through the checkpoint's own chat template, so `prompt tokens` below is what the engine
counted rather than an estimate.

| | Result |
| --- | ---: |
| Prefill, 4,096-token prompt | 6.44 s — **636.0 tok/s** |
| Prefill, 8,192-token prompt | 12.82 s — **639.1 tok/s** |
| Prefill, 4,097-token prompt | 18.11 s — 226.3 tok/s |
| Decode, at a 4,096-token context | **25.9 tok/s** — 38.6 ms a token |
| Decode, at an 8,192-token context | 25.4 tok/s |
| Decode, at a 22,389-token context | 24.1 tok/s |
| A prompt that repeats the one just served | TTFT **0.023 s**, nothing re-forwarded |

**Read the first and third rows together.** They are the same content, one token apart, and the only
difference is that 4,096 is a whole number of 64-token tiles while 4,097 is not. Prefill runs at
**636–647 tok/s** — flat from 2,048 tokens to 22,378 — except that a prompt whose length is not a
multiple of 64 pays a penalty of up to 12 seconds once, in its last partial tile, which is what makes
the unaligned row look like a slow rate. Give the engine a multiple of 64 and it is at its real
speed; there is no alignment knob, and the penalty is measured and reproducible but not yet
diagnosed ([#406](https://github.com/lvyufeng/PocketLLM/issues/406)). See
[the design document](../architecture/bonsai_2_27b_design.md#the-ragged-tail-what-actually-governs-prefill).

Decode falls 7% from a 4,096- to a 22,389-token context — the 48 recurrent layers do not grow with
depth and the 16 attention layers do.

`scripts/bench_pocketllm_serve_phases.py` produced these rows from the server's own `/metrics`
deltas rather than from chunk arrival times; the command is in
[the design document](../architecture/bonsai_2_27b_design.md#evidence).

### Against the upstream reference, same card

The number that says where this runtime stands is the fork's own `llama-bench` on the same artifact
and the same card, from [the reference gate](../architecture/ternary_bonsai_2_reference_gate.md):

| | PocketLLM, 1 card | llama.cpp `prism`, 1 card |
| --- | ---: | ---: |
| Prefill | **636.0 tok/s at 4,096** (639.1 at 8,192) | 642.5 tok/s at 4,096 (615.2 at 8,192) |
| Decode | 25.9 tok/s | 30.7 tok/s |

Prefill is **level with the reference** — 99% of it at 4,096 tokens and 104% at 8,192, compared
aligned to aligned, because `llama-bench` prefills each prompt in one piece and never pays the
ragged-tail penalty above. Decode is 84% of it. Decode is the side with the real gap and the side the
1.75-bit packing was supposed to buy, and the earlier stages measured it doing exactly that
([the dense GEMM](../architecture/ternary_bonsai_2_dense_gemm.md): 2.4× the FP8-width arm at a tenth
of the bytes).

## Hardware and memory

| | |
| --- | --- |
| Cards | 1. Nothing here needs NVLink or a second card. |
| Weights | **5.53 GiB** — 402 ternary tensors, one file, 5,946,648,928 bytes |
| Weights + runtime, no request | **6,566 MiB** on the card; the 895 MiB above the file is the CUDA context, the engine's arena and the materialized F32/BF16 tensors |
| KV | **64 KiB a token** — 16 attention layers × 4 KV heads × 256 × 2 (K and V) × 2 bytes — allocated for the whole `--max-model-len` at start |
| Idle at `--max-model-len 8192` | 7,078 MiB |
| After one cold 4K-token prompt | **8,952 MiB**, and flat from there — the prefill scratch, cached per thread at its high-water mark |
| The most that fits | **245,760 tokens** — 15,360 MiB of KV + 6,566 MiB of everything else, of 22,528 MiB |
| 262,144 tokens | Needs `--kv-cache-dtype fp8`; 15,014 MiB on the card |

Both halves of that budget are measured, and the KV half is the same number derived two ways: the
architecture gives 64 KiB a token, and four fresh processes at `--max-model-len` 2,048, 8,192,
16,384 and 32,768 measured the card at 6,694, 7,078, 7,590 and 8,614 MiB — exactly 64 KiB a token
above a **6,566 MiB** intercept.

**The fixed footprint is not the whole story, and the rest of it is bounded.** A prefill needs device
scratch for the chunk it is working on, and that buffer is cached per thread at its high-water mark
rather than reallocated per chunk — so the card does not return to 7,078 MiB after the first long
prompt. It settles at **8,952 MiB** and stays there: eight cold 4,096-token prompts in one server
produced 7,078 → 8,952 MiB, flat after the first. That is the workspace, it is one per thread, and it
does not grow with how many prompts the server has answered.

## Known limitations

- **A prompt whose token count is not a multiple of 64 pays a one-off penalty of up to 12 seconds.**
  The engine's last partial tile is processed by something slow: a 4,096-token prompt prefills in
  6.44 s and a 4,097-token one in 18.11 s, same content. The cost tracks the *unused* slots in that
  tile — about 0.4 s for each — so it is 12.2 s for a 1-token tail and 1.9 s for a 32-token one, and
  nothing at all when the tile is full. Prefill beyond it is flat at 1.55 ms a token. It is measured
  and reproducible on every prompt tried and the mechanism is not yet identified; it is tracked as
  [#406](https://github.com/lvyufeng/PocketLLM/issues/406). Note that the chat template adds tokens,
  so the count that matters is the one the server reports, not the length of the text you sent.
- **Concurrency is off by default.** With `--backend-option enable_batching=true --backend-option
  max_batch_size=4` the aggregate decode rate rises 1.78× at a short prompt and 1.14× at a 2,044-token
  one, while per-request latency grows with the batch — at 2,044 tokens the batched group's median is
  41.1 s where the serial queue's median is 35.7 s, so four callers wait longer in total than they
  would have queued. Turn it on for aggregate throughput, not for latency.
- **An earlier build of this branch reported that a greedy answer could depend on the batch.** Four
  identical requests produced two distinct texts, and a prompt choosing between `"5:00:00"` and
  `"0:15:00"` gave one row each. That measurement came from a build whose greedy runs did not stop at
  `</s>` on those prompts — all 64 tokens were `"!"` — and on the current build the same probe
  answers identically across all four rows, with the prefix cache on and off and over repeated runs.
  The caveat is therefore not claimed here; the earlier observation is unexplained rather than
  withdrawn, and it is recorded in [the design document](../architecture/bonsai_2_27b_design.md).
- **`--backend-option kv_paged=true` gives up the prefix resume.** Measured: with paging on, a
  repeated prompt was forwarded again in full every time, where the default arena answers the repeat
  from the stored state. Decode is also about 3% slower (25.0 against 25.8 tok/s). Paging exists for
  the case where one arena will not do; it is not the better default here.
- **Prefix reuse is one prompt deep.** The engine keeps the state of the request it just served, so
  a repeat of *that* prompt resumes and a repeat of an earlier one does not. Sending prompt A, then
  B, then A again forwards A in full the second time. It is a resume, not a store.
- **Decode is 84% of the reference on the same card** (25.9 against 30.7 tok/s). This is inherited
  from the [dense-GEMM stage](../architecture/ternary_bonsai_2_dense_gemm.md), not introduced by the
  server, and it is where the format's remaining work is: the packing already buys 2.4× against the
  FP8-width arm at a tenth of the bytes.
- **Speculative decoding is not validated here.** MTP, DSpark and DFlash2 exist in this engine for
  the FP8 checkpoint; none was run against the ternary artifact, and this record does not claim
  that the flags are wired to anything for it.
- **The adapter is greedy only.** `temperature`, `top_k` and `top_p` are accepted by the native
  engine but the C++ adapter refuses a non-greedy request rather than silently ignoring it.
- **`PQ2_0` cannot be served.** The loader decodes its geometry and refuses it by name; there is no
  kernel for type 142.

## Where the detail is

- [Design and measurements](../architecture/bonsai_2_27b_design.md) — the container, the transform,
  the kernels, the probes, and every number above with the command that produced it.
- [The reference gate](../architecture/ternary_bonsai_2_reference_gate.md) — what the upstream
  runtime measures on this card, and the four facts the adaptation rests on.
- [The sm_75 dense GEMM](../architecture/ternary_bonsai_2_dense_gemm.md) — the two ternary kernels
  and the two gaps they leave.
- [Benchmarking and reporting rules](../guides/benchmarking.md) — the convention every number here
  follows.
- The support matrix in [models/README.md](README.md).
