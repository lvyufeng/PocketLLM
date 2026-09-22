# Cross-request prefix caching on the V4.1 serving path

**Served behavior.** `pocketllm serve --backend v41` forward-passed every request's whole prompt,
because both decode loops open with `front.reset_state(1)`: the one mutable KV state is wiped at the
top of a request, so nothing a request left could be read by the next one. A chat loop that resends
its history therefore paid for turns 1..N-1 again on turn N, which for a local single-request
service is the dominant cost — 256K context is ~2554 s of prefill for a full prompt, and a warm
repeat of the same prompt measured the same as a cold one.

This is what vLLM's prefix caching and SGLang's RadixAttention exist to remove, and the design here
is borrowed from both: a key that encodes the whole prefix (vLLM's block-hash chain), matched at the
longest length available (SGLang's longest-prefix rule). What is *not* borrowed is vLLM's paged KV.
The V4.1 caches are pre-allocated contiguous buffers — a 128-slot window ring keyed by
`position % 128`, and compressed tables with one row per `compress_ratio` positions — so a
block-granular store would be a rewrite of the cache layout rather than a patch. The store is
therefore a small set of **whole-prompt anchors**.

The mechanism lives in `src/models/deepseek_v4_1/prefix_cache.py`;
this page is the serving-level design — what is stored, how a request is resumed, what it costs, what
it deliberately does not do.

## What a stored prefix is

A snapshot is one row per cached position plus the fixed state that position left behind. On one
rank, at anchor length `p`:

| Buffer | Layers | Rows kept | Bytes at `p` |
| --- | --- | --- | --- |
| `window_kv_cache` | all 40 | the whole 128-slot ring | 5.0 MiB, fixed |
| `compress_kv_cache` | 2, 8, 14 (ratio 2) | `p // 2` | `1536 * p` |
| `compress_kv_cache` | 20 (ratio 1) | `p` | `1024 * p` |
| `k_cache` | 2, 8, 14 (ratio 2) | `p // 2` | `384 * p` |
| `k_cache` | 20, 24, 28, 32, 36 (ratio 1) | `p` | `1280 * p` |
| `kv_state`, `score_state` | 2, 8, 14, 20 | `p % ratio` | ~28 KB, fixed |
| `EngramHashIds.cache` | `LoadedBackbone.hash_ids` | `cache[:, :p]` | `8 * p` |

**≈ 4232 bytes a token + 5.0 MiB.** At 4K that is 21.9 MiB, at 32K 137 MiB, at 262144 1.036 GiB. The
ratio of each table is derived from the buffer's own shape (`max_seq_len // shape[1]`), so it cannot
drift from the tree, and the two grouped tables are the only ones cut at all.

Two things are deliberately not stored. `freqs_cis` is sliced by absolute position, so a resume needs
no RoPE re-base. The ring and the compressed tables are position-agnostic — the ring's slot for a
position is `position % 128`, a pure function of the absolute position — so they are restored as they
were. `EngramHashIds.cache` is not a registered buffer, so `graphs.snapshot` does not see it and the
loader carries it explicitly; a fresh `reset()` fills it with `DEAD`, and a continuation's first
token reads the previous three positions.

Every rank stores its own copy. The two large caches are the replicated MLA latent, the registered
shapes carry no `world` division, and a rank cannot read another card's memory. The ranks agree on
what a request reuses without a collective, which is what keeps the per-step collectives from
desynchronizing: the store is a pure function of the request sequence and the budget is identical on
all four.

## Anchors

A cold prefill of `L` tokens stores two entries; a resumed one stores one.

| Anchor | Length | What it serves |
| --- | --- | --- |
| head | `1024` | a *different* conversation with the same rendered header — the same system message, tools JSON and reasoning-effort prefix |
| end | `L` exactly | the same conversation's next turn, whose prompt is this one plus the answer plus the new user turn |

The head anchor needs the first forward to *end* at 1024, because a position's ring slot can only be
observed at a forward boundary — a later chunk has overwritten it. A cold prefill therefore runs as
`front(ids[:1024], 0, chunk=1024)` then `front(ids[1024:], 1024, chunk=prefill_chunk)`, with the head
snapshot taken between the two calls. That extra boundary is paid **on a miss only**; a resumed
prefill skips it. `prefix_cache_head_tokens=0` removes both the anchor and the boundary.

The end anchor is stored at the exact length, not rounded down to a 64-token block: state at position
`p` exists only at a forward boundary, so an "aligned" anchor is not free, and keying by exact length
is what makes the lookup exact.

## How a request is resumed

1. The prompt's hash chain is walked once, `O(len(ids) / 64)` blake2b steps.
2. Candidate lengths are the lengths in the store's index, walked descending; a candidate that is not
   a multiple of 64 costs one extra hash of its remainder. The first candidate whose chain matches
   wins, and a mismatch falls through to the next-shallower one rather than failing.
3. `reset_state(1)`, then the saved slices are written back with `copy_` and never a rebind — recorded
   decode graphs hold raw pointers into these buffers.
4. The remainder is forwarded as a *continuation* at `start_pos = cached_len`, which is an ordinary
   chunk boundary.

A resume introduces no new class of numerical divergence. `prefill_chunk` already creates these
boundaries, and `tests/test_models_deepseek_v4_1_chunked_prefill.py` pins each of them bit-equal to a
one-shot prefill over `CHUNKS = (1,2,3,4,5,7,8,9,11,12)`; a cached prefix is that same split point
moved to `cached_len`. The one honest caveat is at 262144 tokens, where `index_topk = 512` is
narrower than the reachable prefix and a different chunking can group the selected vectors
differently — the same divergence the chunk size already carries. That is why the acceptance gate is
a byte comparison of two completions rather than a tolerance.

A request whose prompt *is* a stored entry — a retry, a turn whose history is unchanged — needs only
the row its first new token comes from, and the anchor already computed it. `Entry.logits` carries
that row, so the repeat is a restore and a sample and forwards nothing at all. Replaying the prompt's
last token instead, which looks cheaper, is wrong: the compressor is a recurrence over a group, and a
replay re-emits the group's row from a state that has already counted the position.
`prefix_cache.py`'s own docstring records the test that pins the divergence, so the shortcut cannot
be reintroduced as an optimization.

## Serving surface

| Knob | Default | Meaning |
| --- | --- | --- |
| `--enable-prefix-caching` / `--no-enable-prefix-caching` | on | the switch. It was plumbed to the v41 path and consumed nothing before this; off folds the budget to zero |
| `--backend-option prefix_cache_bytes=4g` | `4g` a rank | the store's byte budget. Accepts a plain integer or a `k`/`m`/`g` suffix; `0` is the same off as the CLI flag |
| `--backend-option prefix_cache_head_tokens=1024` | `1024` | the head anchor's length, or `0` for the end anchor alone |

The budget is ordinary pageable host memory and deliberately **not** `/dev/shm`, which the 457.8 GiB
resident expert bank already holds at 91%. 4 GiB a rank is roughly a million tokens and 16 GiB over
four ranks; a 262144-token entry is 1.036 GiB, about a quarter of it, so a few long conversations fit
and the least-recently-used is evicted past that.

A response reports what it reused the way OpenAI does, as a subset of `prompt_tokens` rather than a
discount on it, and only when it is nonzero — a cold response is byte-identical to the one this
server returned before the field existed:

```json
{"usage": {"prompt_tokens": 1372, "completion_tokens": 64, "total_tokens": 1436,
           "prompt_tokens_details": {"cached_tokens": 1024}}}
```

`/metrics` gains six series, published by the engine and set — not added — at scrape time, because
the store is the one keeping the totals:

| Series | Type | Meaning |
| --- | --- | --- |
| `pocketllm_prefix_cache_hits_total` | counter | lookups answered from the store |
| `pocketllm_prefix_cache_misses_total` | counter | lookups that had to forward |
| `pocketllm_prefix_cache_reused_tokens_total` | counter | prompt tokens answered by a restore |
| `pocketllm_prefix_cache_entries` | gauge | entries held |
| `pocketllm_prefix_cache_bytes` | gauge | bytes held, readable against the budget below |
| `pocketllm_prefix_cache_budget_bytes` | gauge | the configured budget |

The counters are read from a snapshot the backend rebinds at the end of each request under the
request lock, not walked at scrape time: the store's `stats()` walks its index, and a walk concurrent
with a store is a `dictionary changed size`. The price is that a scrape during a request reports the
state the previous request left. Only rank 0 publishes; a worker's store has no exporter behind it.

## Non-goals

- **Paged or block-granular KV.** The ring and the compressed tables are contiguous with a partial
  group that is not block-aligned; a paged store is a layout rewrite, not a patch to this one.
- **Snapshots during decode**, which would also reuse the previous turn's generated tokens. The C++
  engine's `state_snapshot_interval_tokens` is the model for it. Prompts dominate in the local chat
  loop, so it is a follow-on.
- **Multi-request batching.** `_generate_one` still holds one request lock; this changes what a
  single request costs, not how many run at once.

## Where it is tested

| File | What it pins |
| --- | --- |
| `tests/test_models_deepseek_v4_1_prefix_cache.py` | the store itself, and — over the chunked-prefill file's toy geometry — that snapshot, reset, restore and a tail forward leave the caches `torch.equal` to a one-shot prefill |
| `tests/test_models_deepseek_v4_1_generate.py` | the loop's side: the anchors a cold prefill takes, the head chunk on a miss only, and the reuse a resumed request reports |
| `tests/test_v41_backend.py` | the adapter: one store a rank across requests, the CLI switch as the one off switch, `usage.prompt_tokens_details`, and the counters being a snapshot rather than a scrape-time walk |
| `tests/test_backend_contract.py` | the channel: an engine's own values reaching the exposition, a counter set rather than added, and a reused prompt's `cached_tokens` |

## Evidence and related notes

- [DeepSeek-V4.1-Flash](../models/deepseek-v4.1-flash.md) — the checkpoint, its runtime, and where the serving path sits in it
- [The served gate run](../performance/deepseek_v4_1_flash_served_gate.md) — the prefill and decode numbers this path was gated on
- [Chunked prefill on V4.1](../performance/deepseek_v4_1_flash_chunked_prefill.md) — why a forward boundary is not a divergence, and the one row where context length changes the answer
- [vLLM and SGLang architecture analysis](vllm_sglang_architecture_analysis.md) — the comparison this borrows its prefix-reuse model from
- [Serving latency metrics](../guides/latency_metrics.md) — the latency families, and how `/metrics` is read
