# DeepSeek-V4.1-Flash

A 552B-parameter multimodal mixture-of-experts model with a causal encoder-decoder backbone, a
shared-compressed-KV attention scheme (CSA2) and a 196B-parameter Engram conditional-memory table.
PocketLLM runs the released 475.24 GiB checkpoint over four consumer cards — the dense tree on the
GPUs, the routed experts in a pinned host bank — and serves it through the OpenAI-compatible API.

- **Backend**: `--backend v41` (OpenAI-compatible server, one request at a time)
- **Parallelism**: 4 ranks, one process a card
- **Context**: up to 262,144 tokens
- **Validated on**: 4×RTX 2080 Ti 22 GiB, PCIe Gen3, no NVLink

## Overview

The released checkpoint is 475.24 GiB across 48 shards and 96,085 tensors. 58% of it is routed experts
and 39.8% is two Engram tables, and the four cards hold 88 GiB between them — so the checkpoint does
not fit, and where its parts live is the whole design:

- **The dense tree and the packed FP4 experts execute on the cards.** All four ranks run the 40
  attention blocks, the shared experts, the norms, the embedding and the head, out of a memory-mapped
  checkpoint, in the checkpoint's own FP4 and FP8 formats.
- **The routed experts stay on the host.** A 457.8 GiB bank of them is pinned in host memory and each
  rank stages the rows a layer's routing asks for over PCIe. This is why the runtime needs no
  all-to-all: what crosses back is 20 KiB a card a layer.
- **The Engram tables are read where they lie.** 189.13 GiB of n-gram tables are gathered from the
  shards per lookup, or copied into RAM with `--backend-option resident_engram`.

Requests are served one at a time, but a request that shares a prefix with one already served
forwards only the tail: cross-request prefix caching is on by default.

The model is text-only here. The checkpoint's vision tower is audited and never loaded.

## Run it

### Serve it

```bash
DEEPSEEK_V41_RESIDENT_EXPERTS=1 python -m pocketllm serve \
  --model /path/to/DeepSeek-V4.1-Flash \
  --backend v41 \
  --tensor-parallel-size 4 \
  --max-model-len 32768 \
  --port 8000 \
  --backend-option expert_pool_rows=148 \
  --backend-option prefill_chunk=4096 \
  --backend-option decode_graphs=true \
  --backend-option threads=22
```

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "v41", "messages": [{"role": "user", "content": "Hello!"}], "stream": true}'
```

The CLI's own supervisor starts one process a rank — rank 0 binds the listener, ranks 1–3 are NCCL
workers — and all four must report ready before the server answers. Startup is not quick: with
`DEEPSEEK_V41_RESIDENT_EXPERTS=1` each rank pins its share of the 457.8 GiB bank, about 100 s a rank,
and the 48 shards load in another 130 s. `--max-model-len 262144` is the longest context the runtime
accepts.

The endpoint serves `/health`, `/ready`, `/v1/models`, `/v1/chat/completions`, `/v1/completions`, SSE
streaming and `/metrics`. A response reports how much of its prompt was reused, the way OpenAI does,
as `usage.prompt_tokens_details.cached_tokens`.

### Serving options

| Option | Default | What it does |
| --- | ---: | --- |
| `expert_pool_rows` | 288 | Expert rows a chunk's arena holds, per rank. An arena the pass re-draws; worth 8.44× on a 512-token prefill against its own off state, and 1.305× on a decode. **148 is not optional at 262144**, which is what the long-context numbers above are taken at; 0 is the control column, and it turns the batched prefill off with it. |
| `prefill_chunk` | — | Tokens a prefill call takes at once. The width is the prefill knob, and 4096 is the width the long-context numbers need. |
| `decode_graphs` | `false` | Capture a decode step as a CUDA graph. |
| `threads` | — | Host threads a rank's CPU work uses. 22 is one NUMA node on this box. |
| `prefix_cache_bytes` | `4g` | Prefix-cache budget, a rank. `0` is what the CLI's `--enable-prefix-caching` off spells; a `k`/`m`/`g` suffix is accepted. |
| `prefix_cache_head_tokens` | 1024 | The head anchor a snapshot is taken at. |
| `expert_deal` | `id` | How the four cards share a row's experts. `id` balances the bytes a chunk's expert H2D moves — 1.49× / 1.42× on prefill at 32768 / 262144 against `sorted`, which is the opt-in alternative. |
| `expert_buffers` | 2 | Expert arenas the pipeline keeps in flight. |
| `resident_engram` | `false` | Copy the two Engram tables into RAM instead of gathering from the shards. 189.13 GiB and roughly 750 s of reading, once. |
| `expert_device` / `expert_world` | resolved per run | Where the routed experts execute, in the loader's convention: at TP>1 this resolves to the cards (`cuda:0` plus the rank) and `expert_world` to the world size; a single-process run keeps the experts where the dense tree is unless told otherwise. |

`DEEPSEEK_V41_RESIDENT_EXPERTS=1` is the environment-variable form of the pinned host bank and is what
the command above uses. `DEEPSEEK_V41_INDEXER_ROW_SPLIT=1` is the one knob that reaches the attention's
indexer rather than the experts: it shards that module by query rows instead of by index head, which
removes the indexer's per-key-tile score collective — behind a flag, default off, priced at 0.914× on
a 256K prefill.

### Without a server

```bash
# the whole backbone on the host CPU -- a correctness check, not a benchmark
python -m src.models.deepseek_v4_1.generate \
  --checkpoint /mnt/data3/DeepSeek-V4.1-Flash \
  --prompt "The capital of France is" --max-new-tokens 8

# the dense tree cut across the four cards, one process a rank
torchrun --nproc_per_node=4 -m src.cli.generate_v41 \
  --checkpoint /mnt/data3/DeepSeek-V4.1-Flash \
  --prompt "The capital of France is" --max-new-tokens 8

# the metadata audit, which reads headers only and needs no checkpoint download
python scripts/audit_dsv41_headers.py --checkpoint-dir /path/to/DeepSeek-V4.1-Flash --require-complete
```

## What is supported

| Capability | State |
| --- | --- |
| OpenAI-compatible serving (chat, completions, streaming) | Supported |
| Four-rank TP4, one process a card | Supported |
| Dense tree and packed FP4 experts on the cards | Supported |
| 457.8 GiB routed-expert bank pinned in host memory | Supported |
| 262,144-token context | Supported |
| Cross-request prefix caching | Supported, on by default |
| Engram lookup (189.13 GiB of tables) | Read on demand, from the shards or from RAM — there is no GPU consumer of the rows |
| Vision tower and aligner (263 tensors) | **Not implemented** — audited and never loaded; the text path carries no image mask |
| MTP / DSpark (3 draft layers, 7.39 GiB) | Present in the checkpoint, not executed |
| Batching, continuous batching, chunked prefill | **Not implemented** — one request at a time |
| Numeric parity with the released reference | **Unclaimed** — the reference stack needs `torch>=2.10.0` and `tilelang==0.1.8`, neither available here |

## Performance

Four ranks, one process a card, a real checkpoint, one request at a time. Every figure is a served
number over `pocketllm serve --backend v41`, 64 greedy tokens a leg.

| | Prefill | Decode | Step |
| --- | ---: | ---: | ---: |
| 260,244-token prompt | **150.3–152.0 tok/s** | **3.48–3.54 tok/s** | 253–262 ms |
| 1,364-token prompt | **137.5–140.8 tok/s** | **4.45–4.53 tok/s** | — |
| 1,364-token prompt, first request | 108.0 tok/s | — | — |

The long prompt is the *faster* prefill of the two, because a short one is dominated by fixed
per-process cost, and the first request on a fresh server is the slowest because it pays the capture
pass. Decode is the half that does not travel with prompt length at the same rate — the same service
reads 4.45–4.53 tok/s at 1,364 prompt tokens.

Two contrast rows, both from the same four cards and neither a served path. On the host CPU alone a
generated token costs **15 to 42 s**, and 99.7% of it is turning the fp4 expert codes into bf16
numbers rather than reading them. With the routed experts on the cards and the dense tree still on the
host, a step is **1.06–1.14 s**; with the dense tree cut across the cards as well, **722–747 ms**,
which is what `torchrun -m src.cli.generate_v41` does.

## Hardware and memory

| | |
| --- | --- |
| Cards | 4, one process each, TP4. Nothing here needs NVLink. |
| Card memory | The dense tree, the packed FP4 expert working set and the KV cache; the routed experts are not resident |
| Host memory | **457.8 GiB** pinned for the expert bank with `DEEPSEEK_V41_RESIDENT_EXPERTS=1`, plus 189.13 GiB more if `resident_engram` is set |
| Storage | The checkpoint's 48 safetensors shards, 475.24 GiB, memory-mapped and read on demand |
| Startup | ~100 s a rank to pin the bank, ~130 s to load the shards |

Memory, not kernels, is the structural problem: 58.0% of the checkpoint is routed experts and 39.8%
is two Engram tables, and a 4×22 GiB deployment can hold neither. Any V4.1 plan has to place experts
and Engram in host memory or on disk before kernel work matters.

## Known limitations

- **One request at a time, and there is no batched path.** `--backend v41` takes a single request lock
  and reports `supports_batch=False`: no continuous batching, no chunked prefill, no paged KV pool.
  Prefix caching changes what a request costs, not how many run at once.
- **No MTP and no speculative decoding.** The three DSpark draft layers are 7.39 GiB the loader
  deliberately leaves in the shards.
- **No numeric oracle.** The released reference needs `torch>=2.10.0` and `tilelang==0.1.8` and this
  host has neither, and its sm_75 cards have no FP4 tensor core for the reference's own fallbacks. The
  acceptance evidence is generated text — greedy decode returns `' Paris'` and then `'.'` for `"The
  capital of France is"` on every path — not a logit comparison.
- **Engram is addressable and gathered, but not measured against the reference.** The hash front end
  reproduces both declared row counts exactly to the row and the rows are read on demand; what is open
  is the gate and value projections' parity with the reference, and the fact that gathering from the
  shards is not a viable steady state — a cold 512-token prefill's gathers measure 253.4 s against
  0.009 s once the table is resident.
- **Vision is unvalidated in both directions.** 263 tensors are accounted for and shape-checked; no
  image has been processed.
- **The audit validates metadata consistency, not correctness.** A checkpoint could satisfy all 39
  checks and still be unusable, and a value that is *consistently* wrong in both the config and the
  shapes would pass.
- **The two config layouts are not interchangeable field for field.** A consumer handed only
  `inference/config.json` has no token ids, no `topk_method` and no `param_dtype`, and its top-level
  `dtype` is the *quantization* dtype rather than the storage one.

## Where the detail is

- [Design and measurements](../architecture/deepseek_v4_1_flash_design.md) — the tensor inventory and
  audit, the CSA2 layers, the loader, the config schema, the Engram front end, and every measurement
  behind a design choice.
- [Cross-request prefix caching on V4.1](../architecture/v41_prefix_cache.md) — what a snapshot is,
  the two anchors, the equivalence argument, and the metrics.
- [What one request costs, through the launcher](../performance/deepseek_v4_1_flash_single_request_capability.md)
  and [the device experts](../performance/deepseek_v4_1_flash_device_experts.md) — the per-leg flags
  and the split of a graphed step.
- [Benchmarking rules](../guides/benchmarking.md) — required before any V4.1 number is quoted.
- The support matrix in [models/README.md](README.md).
