# MiMo-V2.6-Flash

A 48-layer hybrid-attention mixture-of-experts text model — 9 global-attention layers and 39
sliding-window layers with a per-head sink, 256 routed experts activated top-8 per token. PocketLLM
runs the released safetensors checkpoint on four consumer cards, with the routed experts left in a
host-resident bank and dealt out over the ranks.

- **Backend**: `--backend mimo` (OpenAI-compatible server, one request at a time, cross-request
  prefix caching on by default)
- **Parallelism**: 4 ranks, expert parallelism plus a four-way attention split
- **Context**: up to 262,144 tokens
- **Validated on**: 4×RTX 2080 Ti 22 GiB, PCIe Gen3, no NVLink

## Overview

The checkpoint is 149.81 GiB of MXFP4 experts plus an FP8 dense stack. Neither fits a 22 GiB card, so
PocketLLM splits the job in two:

- **The backbone stays on the cards.** The 48 transformer layers, the attention and the dense
  projections execute on the GPUs, and the attention's four-way split follows the checkpoint's own
  `qkv_proj` partition.
- **The routed experts stay in host memory.** All 256 experts of all 47 routed layers live in a
  single `/dev/shm` segment that every rank attaches to. A decode step draws two experts a rank a
  layer and copies them over PCIe; a prefill chunk takes its share of the experts.

That arrangement is why the runtime exists: the checkpoint does not fit on the hardware, and the
expert copies — not the arithmetic — are what a token costs. Everything measured about it is in
[the design document](../architecture/mimo_v2_6_flash_design.md).

The model is text-only. The checkpoint's vision tower and audio encoders are not read.

## Run it

### Serve it

```bash
python -m pocketllm serve \
  --backend mimo \
  --model /path/to/MiMo-V2.6-Flash \
  --tensor-parallel-size 4 \
  --max-model-len 262144 \
  --port 8000 \
  --backend-option prefill_chunk=2048 \
  --backend-option chunk_rows=16
```

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "mimo", "messages": [{"role": "user", "content": "Hello!"}], "stream": true}'
```

The first start fills the 149.81 GiB expert bank into `/dev/shm` from the checkpoint — about twelve
minutes at 213 MiB/s. A later run attaches to the existing segment in 0.07 s, and all four ranks
attach to the *same* segment rather than filling four. `rm -rf /dev/shm/pocketllm_mimo_experts_*`
gives the memory back; the next start refills it.

The endpoint serves `/health`, `/ready`, `/v1/models`, `/v1/chat/completions`, `/v1/completions`, SSE
streaming, `DELETE /v1/requests/{id}` and `/metrics`. One request runs at a time — see
**Known limitations** — but a request that shares a prefix with one already served forwards only the
tail: cross-request prefix caching is on by default.

### Serving options

| Option | Default | What it does |
| --- | ---: | --- |
| `prefill_chunk` | 2048 | Tokens a prefill call takes at once. The width is the prefill knob. |
| `chunk_rows` | 16 | Expert rows a chunk's arena holds, per rank. Costs card memory; does not change the decode step. |
| `resident_rows` | 0 | Of each routed layer's hottest experts, how many to keep on the card. **0.585 GiB a row** over the 47 routed layers, so 16 rows is 9.36 GiB — worth 1.33× at a short context, and more than a 262144-token cache leaves room for. A deployment's decision, not a constant. |
| `slots` | — | Expert arena slots, i.e. how many calls the pipeline keeps in flight. |
| `deal` | `sorted` | Which deal divides the experts. `sorted` balances a decode step's drawing; `id` gives each rank a fixed 64 experts a layer and is what a prefill chunk needs. A served run keeps both arenas and dispatches on the row count of the call. |
| `prefix_cache_bytes` | `4g` | Host memory a rank's prefix store may hold, as an integer or a `k`/`m`/`g` suffix. `0` is what `--no-enable-prefix-caching` folds it to. |
| `prefix_cache_head_tokens` | 1024 | The fixed-length anchor a cold prefill also stores, which is what serves a *different* conversation that shares the same rendered header. `0` stores the prompt's end alone. |

### Without a server

```bash
# the whole backbone on one card, decoded on the CPU -- a correctness check, not a benchmark
python scripts/verify_mimo_v2_real_checkpoint.py --tokens 8

# what a whole token costs on one card, and how much of it is the expert copy
python tests/bench_mimo_v2_model.py

# the four-rank step, and the token-at-a-time floor against a chunked prefill
torchrun --nproc_per_node=4 tests/bench_mimo_v2_ep.py --steps 8
torchrun --nproc_per_node=4 tests/bench_mimo_v2_prefill.py \
    --tokens 4096 --chunk 256,512,1024,2048,4096 --floor 16 --band 0
```

## What is supported

| Capability | State |
| --- | --- |
| OpenAI-compatible serving (chat, completions, streaming, cancel, metrics) | Supported |
| Four-rank expert parallelism | Supported |
| Four-way attention split along the checkpoint's own `qkv_proj` partition | Supported |
| Chunked prefill with a grouped multi-token expert kernel | Supported |
| 262,144-token context | Supported and measured |
| Resident expert set (`resident_rows`) | Supported, off by default |
| Sampling (`temperature`, `top_k`, `top_p`, `seed`) | Supported; greedy unless a temperature is given, which is the checkpoint's own default |
| Repetition penalty, logit bias, grammar | **Not implemented** |
| Cross-request prefix caching | Supported, on by default; `usage.prompt_tokens_details.cached_tokens` reports the reuse |
| Batching, a scheduler | **Not implemented** — one request at a time |
| MTP (3 layers) and the DFlash drafter | Present in the checkpoint, not executed |
| Vision tower, audio encoders | Out of scope |

## Performance

Four ranks, one process a card, a real checkpoint, single requests. The slowest rank is the number,
because every routed layer closes with a barrier.

| | Result |
| --- | ---: |
| Prefill, 262,144-token prompt, 2048-token chunks | **104.04 tok/s** |
| Prefill, 4096-token prompt, 2048-token chunks | **174 tok/s** |
| Decode, short context | **156.3 ms — 6.40 tok/s** |
| Decode, short context, 16 resident rows a layer | **117.3 ms — 8.53 tok/s** |
| Decode, 262,144-token context | **180.0 ms — 5.56 tok/s** |
| Decode, 262,144-token context, 8 resident rows a layer | **174.3 ms — 5.74 tok/s** |
| One card, for contrast | 610 ms — 1.64 tok/s |

A decode step at a long context is bounded below by the expert copy: 1198.5 MiB a rank over PCIe, at
a PCIe 3.0 link's own rate. Nothing on this path hides that, which is why `resident_rows` — the one
lever that removes bytes instead of overlapping them — matters more here than any kernel does.

### A chat turn that reuses a prefix

One process, four ranks, one prompt of 4096 real tokens, the three arms interleaved inside it
(`tests/probe_mimo_v2_prefix_cache.py --tokens 4096 --prefix 3072 --chunk 2048`). These are that
process's arms read against each other, not rows of the table above:

| | Result |
| --- | ---: |
| All 4096 tokens from zero | 20.07 s — 204.0 tok/s |
| The same prompt cut at 3072 with no store | 23.90 s — 171.4 tok/s |
| A stored 3072-token prefix restored, 1024 forwarded | **7.02 s — 145.9 tok/s** |
| The state back: a restore, and a snapshot on the cold path | 15.1 ms / 20.0 ms |
| A prompt that *is* a stored entry: restore and sample | 14.2 ms |

The third row is a turn; the first is what a turn cost before. The state behind it costs 8039 bytes a
token a rank — 23.6 MiB for 3072 tokens, of which 6.1 MiB is the fixed window rings — against a 4 GiB
budget. The rate is lower than the cold arm's because a 1024-token tail is a narrower chunk, which is
also why the middle row is slower than the first. The three arms' last rows differ by 4.85e-02 of
their 31.6 peak — all of it the cut, none of it the store, which moves the row by 0.00e+00 — and all
three draw the same next token.

## Hardware and memory

| | |
| --- | --- |
| Cards | 4, one process each. Nothing here needs NVLink. |
| Card memory | **10.21 GiB a card** at a 262,144-token context, of 22 GiB available |
| Host memory | **149.81 GiB** of `/dev/shm` for the expert bank, shared by all four ranks; up to **4 GiB a rank** of pageable memory for the prefix store |
| Storage | The checkpoint's safetensors shards, read once when the bank is filled |

`resident_rows=16` adds 9.36 GiB a card, which is why it and a 262144-token KV cache do not fit
together. At that depth eight rows fit and buy 3%; at a short context sixteen fit and buy 33%.

## Known limitations

- **One request at a time, and that is structural.** Every routed layer closes with an all-reduce at
  the same point in every rank's program, so a rank that is not running the request its peers are
  running is not idle but at a *different* collective — and NCCL answers a mismatch by hanging.
  Rank 0 broadcasts each request to the workers before running it, and a cancel or a stop string has
  to be agreed across the ranks rather than acted on by one. A second request waits on a lock.
- **The attention and the dense linears are torch above 16384 keys.** The decode step's rotation,
  softmax and norms are CUDA kernels; the rest of the attention, and every dense projection, are
  PyTorch at every depth.
- **Decode is copy-bound, not compute-bound.** At a 262,144-token context a step is 180 ms, of which
  the expert H2D the kernel waited for is 99.9 ms at a PCIe 3.0 x16 link's ceiling. Prefetching it
  from the previous token's draw was measured and closed — a rank's rows hold the expert they held a
  step earlier 9–13.5% of the time.
- **The expert bank costs ~150 GiB of `/dev/shm`,** and the first start spends about twelve minutes
  filling it. The prefix store is ordinary pageable host memory and deliberately not `/dev/shm`: 4 GiB
  a rank, 16 GiB over four.
- **A resumed prefill is not bit-identical to a cold one, and the difference is the cut, not the
  store.** The attention's gemms change shape with a chunk, so a prompt cut at the stored length gets
  the continuation's arithmetic for the tokens past it. The probe puts a number on both halves: the
  state the store hands back is the state a cold prefill had, bit for bit (an un-stored resumption of
  the same tokens moves the row by **0.00e+00** against the stored one), and where the prompt is cut
  moves the prompt's last row by **4.85e-02 of its 31.6 peak**. A greedy chain can therefore move a
  token where the cold answer sat near a rounding boundary, while the state behind it is the same
  either way; on the probe's prompt the three arms drew the same token. This is the same boundary
  effect the chunk width already has on this runtime, and it is why a resumed prefill is not held to
  the byte-for-byte gate the V4.1 path is.
- **The head anchor costs the cold path one chunk boundary.** A `1024`-token first forward instead of
  a `prefill_chunk`-wide one, paid only on a miss.
- **MTP and DFlash are not executed,** and vision and audio are out of scope.
- **The prefix store is not extended during decode.** A second turn resends the history and gets it
  back; the tokens the *first* turn generated are prefilled again.

## Where the detail is

- [Design and measurements](../architecture/mimo_v2_6_flash_design.md) — the runtime's design, every
  measurement behind a decision, and the probes each number comes from.
- [Serving latency metrics](../guides/latency_metrics.md) and
  [benchmarking rules](../guides/benchmarking.md) — how the numbers above were taken, and what may be
  compared with what.
- The support matrix in [models/README.md](README.md).
