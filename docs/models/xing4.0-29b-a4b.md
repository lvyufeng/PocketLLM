# Xing4.0-29B-A4B

A 40-block mixture-of-experts text model — MLA attention, 64 routed experts activated top-4 plus one
shared, and **four residual streams per block** instead of one — released as an official IQ4_NL GGUF
at 18.72 GiB. PocketLLM runs the whole checkpoint on **one** 2080 Ti, behind the same
OpenAI-compatible server as its other models.

- **Backend**: `--backend xing4` (also selected automatically from the GGUF's own header)
- **Parallelism**: 1 GPU. There is no sharding, because there is nothing to shard.
- **Context**: 32,768 out of the box; the card's own ceiling is measured at about 40,960 tokens
- **Validated on**: 1×RTX 2080 Ti 22 GiB, PCIe Gen3

## Overview

The checkpoint is `XingChen-AGI/Xing4.0-29B-A4B` — 29B parameters of which about 4B are active per
token — and its shape is DeepSeek's: an MLA trunk with a compressed latent cache, and a sigmoid-routed
MoE with a shared expert. What is new is the block around the sublayers.

| Field | Value |
| --- | ---: |
| Trunk blocks | 40, plus one NextN/MTP block that this runtime does not execute |
| Hidden size | 3,584 |
| Attention | MLA: `q_lora_rank` 768, `kv_lora_rank` 512, `qk_nope` 128, `qk_rope` 64, `v_head` 128 |
| Query heads | 32, over **one** KV latent (`head_count_kv = 1`) |
| Routed experts | 64, activated top-4, `expert_ffn` 1,024, plus 1 shared |
| Dense blocks | Blocks 0–1 have a 9,216-wide ordinary FFN |
| Residual streams | **4** per block, mixed by a matrix hyper-connection |
| RoPE | YaRN, `original_context_length` 4,096, factor 64 |
| Vocabulary | 131,072 |
| Quantization | IQ4_NL, 4.5 bits a weight — GGML type 20, 32 weights in 18 bytes |

Three things decide how this runtime is built.

**The checkpoint is smaller than the card, which is the whole point.** 18.72 GiB of file against
22 GiB of card, so all 64 experts of all 38 MoE blocks stay **resident**: 14.7 GiB of the 17.94 GiB
resident footprint is expert weights, read once at load and indexed by expert thereafter. Every other
MoE in this repository streams its active experts from a host bank or a disk because its checkpoint is
far larger than a card; this one would pay a PCIe round trip for weights that are already where they
are needed. It is also why there is no tensor-parallel path: `ep_size = 1`, nothing spills, and a
second card buys context rather than rate — see **Known limitations**.

**Four residual streams, and the sublayer never sees them.** The state is
`[tokens, 4, 3584]`; the block projects all four flattened together into 24 coefficients
(`pre`, `post`, and a 4×4 `comb`), collapses them to one 3,584-wide stream for the sublayer, and
rebuilds the four from the sublayer's output:

```
collapsed  = Σ_s pre[s] · hidden[s]
hidden[d]  = post[d] · sublayer_out + Σ_s comb[d, s] · hidden[s]
```

`comb` is put through **20 Sinkhorn iterations** to make it doubly stochastic, which is what keeps the
four streams from compounding rather than behaving like one. None of that is a residual you can
accumulate — `pre`, `post` and `comb` are recomputed from the current state at every sublayer — and
none of it is cheap in the naive form either: 20 Sinkhorn rounds × 2 gates × 40 blocks is 1,600
dependent small-matrix steps a token. PocketLLM runs the whole thing in one kernel, one block a row,
which is where decode's 2.45× came from
([the design record](../architecture/xing4_0_29b_a4b_design.md#3-the-hyper-connection-in-one-kernel)).

**The attention is the largest single read of a decode token, and it is not quantized.** The
quantizer left every attention projection at BF16 and quantized 93% of the checkpoint's parameters,
so a decode token reads 2.12 GiB of attention weights against the routed experts' 0.88 GiB:

| Per decode token | GiB | Share |
| --- | ---: | ---: |
| BF16 attention, all 40 blocks | 2.117 | 56.3% |
| Routed experts, 4 of 64 selected | 0.877 | 23.3% |
| Q6_K `output.weight` | 0.359 | 9.5% |
| Shared expert | 0.219 | 5.8% |
| Dense FFN, blocks 0–1 | 0.104 | 2.8% |
| `hc_*`, router, norms, embedding | 0.086 | 2.3% |
| **Total** | **3.761** | |

On a 2080 Ti's 616 GB/s that is a 6.56 ms floor, or 152 tok/s. **The measured decode rate is 5.7
tok/s, and the reason is not the bytes.** A step submits 11,536 kernel launches, each costing about
15 µs of host time, so 178 ms of host submission against 46.5 ms of device work: the GPU is idle
three-quarters of a decode step and the card's bandwidth is barely touched. That is the honest state
of this runtime and it is where its remaining work is — see **Known limitations**.

### One token's path

A prefill chunk forward-passes the block stack over the chunk, writing 46 KiB a token a layer into
the absorbed latent cache; a decode step forward-passes the same stack over one token. Both run the
same 40 blocks; the difference is the shape of the score matrix, which is `chunk × tokens` at prefill
and `1 × tokens` at decode, and which is why prefill's cost is quadratic in the context and decode's
is not.

## Run it

### Serve it

```bash
python -m pocketllm serve \
  --backend xing4 \
  --model /path/to/xing4_0-29b-IQ4_NL.gguf \
  --tokenizer-path /path/to/Xing4.0-29B-A4B \
  --served-model-name xing4 \
  --max-model-len 32768 \
  --port 8000
```

The checkpoint is **two things**, and this is the one launch in the repository that needs both: the
`.gguf` carries the weights and the vocabulary, and a directory carries the tokenizer's merges, the
`config.json` (which is where `mhc_h_res_clamp_min`/`_max` live — the ±30 the Sinkhorn logits are
clamped to, and no GGUF key carries them) and the chat template. `--tokenizer-path` is how the
directory is named; a launch that gives the file and nothing else is refused with the paths it looked
at rather than served with the wrong vocabulary.

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "xing4", "messages": [{"role": "user", "content": "Hello!"}], "stream": true}'
```

The endpoint serves `/health`, `/ready`, `/v1/models`, `/v1/chat/completions`, `/v1/completions`,
SSE streaming, `DELETE /v1/requests/{id}` and `/metrics`. One request runs at a time — see **Known
limitations** — but a request whose prompt repeats one already served forwards only the tail: the
prefix store is on by default.

### Serving options

| Option | Default | What it does |
| --- | ---: | --- |
| `--max-model-len` | 32768 | Positions the cache holds, **and the number the prefill chunk is derived from**: 46,080 bytes a token across the 40 layers, so this is 1.41 GiB at the default. A context whose narrowest chunk cannot fit the card is refused at start with the room and the requirement. |
| `--device` | `cuda` | The card. One, and it has to have 18 GiB free. |
| `--enable-prefix-caching` | on | `--no-enable-prefix-caching` turns the store off. |
| `--backend-option gguf=PATH` | — | The weights, when `--model` names the checkpoint directory instead of the file. |
| `--backend-option prefill_chunk=N` | derived | Tokens a prefill forward takes. Derived from the card's free memory and the context; naming one is for reproducing a measurement, not for tuning. |
| `--backend-option prefix_cache_bytes=N` | 2 GiB | The store's host-side budget. |
| `--backend-option tokenizer=DIR` | — | The tokenizer directory, same as `--tokenizer-path`. |
| `--backend-option use_kernel=false` | `true` | Runs the hyper-connection in PyTorch instead of the fused kernel. **2.45× slower at decode**; it exists so the two can be compared. |
| `--tensor-parallel-size` | 1 | Not implemented for this checkpoint; a value above 1 is refused rather than silently ignored. |

### Without a server

```bash
# the fused hyper-connection kernel against the PyTorch reference, and its cost
python -m pytest tests/test_xing4_0_hyper_connection.py -q

# the grouped MoE kernel against a dense evaluation of the same route table
python -m pytest tests/test_xing4_0_moe.py -q

# the prefill rate and the decode rate on real prose, one card
python scripts/bench_xing4_0_e2e.py --device cuda:2 --lengths 512,4096,32768

# the kernel's own cost, priced by in-process A-B-A-B interleaving
python scripts/bench_xing4_0_hyper_connection.py --device cuda:2 --steps 8
```

## What is supported

| Capability | State |
| --- | --- |
| OpenAI-compatible serving (chat, completions, streaming, cancel, metrics) | Supported, validated on one card from the released checkpoint |
| The complete 40-block trunk, all 64 experts resident on the card | Supported |
| MLA attention, absorbed form, with the YaRN re-base | Supported |
| The 4-stream matrix hyper-connection, in one CUDA kernel | Supported, on by default |
| IQ4_NL (GGML type 20) routed experts through the grouped MoE kernel | Supported |
| Chunked prefill | Supported; the chunk is derived from the card and the context |
| Cross-request prefix reuse | Supported, on by default |
| Sampling (`temperature`, `top_k`, `top_p`) | Supported; greedy by default |
| Memory-fitted context, up to the card's ceiling | Supported; measured at about 40,960 tokens |
| Concurrent requests | **Not implemented** — requests serialize, one at a time |
| Tensor parallelism | **Not implemented** — the checkpoint fits one card whole |
| The NextN/MTP block (`blk.40`, 0.93 GiB) | **Not executed** — it is not loaded and there is no speculative path for it |
| LoRA, adapters, logprobs, tool calling | **Not implemented** |
| A context of 262,144 tokens, the checkpoint's own limit | **Not reachable on one card** |

## Performance

One RTX 2080 Ti 22 GiB, `--device cuda:2`, the released `xing4_0-29b-IQ4_NL.gguf`, one request at a
time, `--max-model-len 32768`, greedy, **32 generated tokens**. The prompts are a real repository
document cut to an exact token count — not synthetic ids, because the router's draws are a function
of the activations and a prompt made of noise routes to experts a served request never picks.

| Prompt | Prefill | Decode, all steps | Decode, steady | First step |
| ---: | ---: | ---: | ---: | ---: |
| 512 | 84.39 tok/s (6.07 s) | 6.50 tok/s | 6.51 tok/s | 158 ms |
| 4,096 | 78.69 tok/s (52.1 s) | 6.52 tok/s | 6.50 tok/s | 143 ms |
| 16,384 | 68.54 tok/s (239 s) | 6.45 tok/s | 6.43 tok/s | 141 ms |
| 32,768 | 63.84 tok/s (513 s) | 6.07 tok/s | 6.06 tok/s | 152 ms |

Every row is at a 128-token prefill chunk, which is what the card's room at a 32,768-token context
affords — the chunk is a memory decision and not a tuning one, see **Hardware and memory**.

Three things to read with the table:

- **Decode is flat in context** — 6.50 → 6.07 tok/s from a 512- to a 32,768-token context, a 7% fall
  across 64× the depth. The absorbed cache means a decode step reads 576 values a layer a token
  regardless of how much context is behind it, so growing the context barely changes what a step
  touches.
- **Prefill falls 24% from 512 to 32,768 tokens**, which is the quadratic attention term appearing
  against a linear MoE term: the MoE kernels are 75% of prefill at short prompts and the attention
  takes a growing share of a long one.
- **The first decode step is reported separately because a cold one is not a steady one.** In a warm
  process it is 141–158 ms, indistinguishable from the rest. In a *fresh* process, where the allocator
  has never held a request-sized working set, it is 0.6 s after a 4,096-token prefill and 3.0 s after
  a prefill with a 1024-wide chunk — a one-off allocation, paid once, and large enough to pull a
  32-token answer's average measurably below its steady rate. Both figures are real; a report that
  gave only one would be wrong about either the first request or the engine.

A prompt that repeats one already served resumes instead of forwarding: the store reports
`pocketllm_prefix_cache_reused_tokens_total 19` and the second request answers in **1.15 s** where the
first took several — see [the record](../architecture/xing4_0_29b_a4b_design.md#7-serving) for the
transcript.

### On two cards

There is no tensor-parallel path for this checkpoint, so the two-card row is two processes rather than
one sharded model — and that is the shape the checkpoint asks for, because it fits one card whole.
Same 512-token prompt, `--max-model-len 8192`, chunk 128, 32 decode steps:

| | Prefill | Decode | Steady |
| --- | ---: | ---: | ---: |
| One process, one card | 86.28 tok/s | 6.48 tok/s | 6.50 tok/s |
| Two processes, cards 0 and 1 — the first | 84.79 tok/s | 6.52 tok/s | 6.52 tok/s |
| Two processes, cards 0 and 1 — the second | 85.89 tok/s | 6.53 tok/s | 6.53 tok/s |

Both concurrent runs hold the single-run rate to within 2%, so the aggregate is 2× at no cost to
either: neither the card nor the host submission path is shared. A 4×2080 Ti box therefore serves this
checkpoint as four independent servers, and the number of interest is the one the header asks about —
**17.94 GiB of 22 GiB resident, 3.4 GiB left over, so about 85% of each card is the model and the rest
is context.**

### Against the Qwen3.8-27B paths at comparable size

This is the only checkpoint here that is smaller than the card it runs on, so the comparison is
two-sided: what one card buys, and what this runtime gives up against the engines it shares a
repository with.

| | Xing4.0-29B-A4B, 1 card, IQ4_NL | Ternary-Bonsai-2-27B, 1 card, 1.75 bit | Qwen3.8-27B-FP8, 4 cards, TP4 |
| --- | ---: | ---: | ---: |
| Weights | 17.94 GiB resident | 5.53 GiB | 6.86 GiB a rank |
| Prefill, 4,096 tokens | **78.7 tok/s** | 636.0 tok/s | 1,729 tok/s |
| Decode, 4,096-token context | **6.52 tok/s** | 25.9 tok/s | 43.8 tok/s |
| Architecture | MLA + 64-expert MoE, 4 streams | hybrid GQA, dense FFN | full GQA, dense FFN |
| Runtime | PyTorch eager + raw-block kernels | C++ engine | C++ engine |

**The gap is the runtime, not the checkpoint.** Bonsai runs a smaller model faster because its native
C++ engine submits a step's work as a handful of launches where this one submits 11,536; the two
checkpoints' byte counts per token differ by less than the two runtimes' launch counts do. Moving this
path into the C++ engine is the obvious next step and it is not in this stage's scope — what the stage
establishes is the checkpoint's shape, its parity, and its numbers in the runtime it has.

## Hardware and memory

| | |
| --- | --- |
| Cards | 1. Nothing here needs NVLink or a second card. |
| File | **18.72 GiB** — 20,104,012,544 bytes, 977 tensors, GGUF type 25 (mostly IQ4_NL) |
| Resident | **17.94 GiB** of weights with every expert on the card; the NextN block (0.93 GiB) is not loaded |
| Card, idle with a 8,192-position cache | **18,976 MiB** of 22,528 MiB — 3,552 MiB free |
| Free at load, before any cache | **3.40 GiB** |
| KV cache | **46,080 bytes a token** across the 40 layers — 512 latent + 64 rope key, fp16 |
| Cache at the default 32,768 positions | 1.41 GiB |
| Prefill workspace | The score path is **8 bytes an element** (`8 · 32 · chunk · context`), plus 126 KiB a chunk token — 1,051 MiB at a 128-token chunk and a 32,768-token context |
| The most that fits | **Between 40,960 and 49,152 tokens.** 40,960 prefills with 2 MiB to spare at the last chunk; 49,152 OOMs |
| 262,144 tokens, the checkpoint's own limit | Needs 11.8 GiB of cache against the 3.4 GiB that are free, before the prefill workspace |

The ceiling is a measurement rather than an arithmetic identity, and the arithmetic that predicts it
is the one the adapter uses: the KV cache and the prefill's score path want the same few GiB, so past
some context there is no chunk the loop can run at all. With the 768 MiB reserve this adapter holds
back, the largest context it will accept is about 34,816 tokens; the hardware's own edge is the
40,960 measured above. Both directions are stated because they are different numbers for different
questions — what the runtime will promise, and what the card can just barely do.

## Known limitations

- **Decode runs at 6.5 tok/s against a 152 tok/s byte floor, and the gap is host submission.** A
  decode step is 11,536 kernel launches; at about 15 µs each that is 178 ms of host time against
  46.5 ms of device work, so the GPU is idle three-quarters of the step and the card's 616 GB/s is
  barely used. Nothing about the checkpoint explains it — the bytes a token reads are 3.76 GiB, which
  the card could deliver in 6.6 ms. The levers are CUDA graphs and fewer, larger kernels, and the
  fused hyper-connection kernel already showed what the second one is worth (2.45×, and it removed
  5,800 of those launches at once). This number varies ±15% run to run because it is a host-side
  measurement; the mechanism does not.
- **Requests serialize.** `capabilities.supports_batch` is False and the adapter refuses a second
  concurrent request rather than interleaving it. The trunk's forward flattens its input to a single
  token axis, so two sequences handed to it together would attend to each other. A served deployment
  that needs concurrency wants one process a card, not batching — which is also what the memory
  budget suggests, since the checkpoint uses 85% of a card and its batching headroom is 3.4 GiB.
- **There is no tensor-parallel path, and two cards were measured as two processes rather than as
  TP2.** `--tensor-parallel-size 2` is refused: `ep_size = 1`, every expert is already on the card,
  and a per-layer collective would add to the host cost that decode is bound by instead of subtracting
  from it. What a second card is for here is **aggregate throughput and context**, and both were
  measured. Two server processes, one a card, prefill the same 512-token prompt concurrently at
  **84.79 and 85.89 tok/s against 86.28 alone** and decode at **6.52 and 6.53 tok/s against 6.48
  alone** — 2× the throughput at no cost to either, because neither the card nor the host path is
  shared. Running one process a card is therefore the supported multi-card shape for this checkpoint,
  and the TP2 question — whether sharding the MoE and the MLA path would raise a *single* stream's
  rate — is left open rather than claimed.
- **The multi-turn chat template gives up most of the prefix reuse.** The generation prompt renders as
  `<_bot><think>\n`, and a turn's assistant message is prefixed with `</think>`, so turn N's prompt
  shares tokens with turn N−1's only up to the `<_bot>` marker — 10 of 24 in the released template's
  own two-turn render. Cross-*turn* prefix reuse therefore recovers the system and user prefix and not
  the conversation; the repeat-the-same-prompt case, which is what the store's counters are usually
  read for, hits fully.
- **The NextN/MTP block is loaded nowhere.** `blk.40` is 24 tensors and 0.93 GiB, it has no `hc_*`
  tensors, and this repository's record on speculative decoding is that it is acceptance-dependent —
  a length-2 MTP path on a GGUF cost more than it returned for DeepSeek-V4. It is excluded from the
  trunk and there is no flag that runs it, so no speculative speedup is claimed here.
- **The context ceiling is about 40,960 tokens, not the checkpoint's 262,144.** The declared context
  is unreachable on a 22 GiB card: 11.8 GiB of cache against 3.4 GiB free. A quantized cache would
  extend it and is not implemented.
- **A degenerate prompt shape produces degenerate text, and always has.** The model only behaves on
  the `<_bot><think>\n` continuation form its template produces. A raw completion prompt
  (`The capital of France is`) and a prompt ending in a bare `<_end>` both put the next-token
  distribution on whitespace and quotes rather than an answer, with or without the fused kernel and
  independently of the cache. This is a property of the checkpoint's tuning, not of the runtime, and
  it is worth knowing before reading a prompt through the completions endpoint.
- **Prompt length is measured, not estimated.** The chat template adds the system block and the
  `<_bot><think>` marker, so the token count the server reports is several tokens above the text you
  sent, and the memory and context numbers above are about the former.

## Where the detail is

- [Design and measurements](../architecture/xing4_0_29b_a4b_design.md) — the fused hyper-connection
  kernel and what it costs, the MoE's deterministic route reduction, the residual-width fix, the
  serving path, and every number above with the command that produced it.
- [The checkpoint audit](../architecture/xing4_0_29b_a4b_audit.md) — the block read out of the
  released remote code, the artifact inventory, and the four places the issue tree's reading did not
  hold.
- [Benchmarking and reporting rules](../guides/benchmarking.md) — the convention every number here
  follows.
- The support matrix in [models/README.md](README.md).
