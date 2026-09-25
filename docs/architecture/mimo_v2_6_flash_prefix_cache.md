# Cross-request prefix caching on the MiMo-V2.6-Flash serving path

**Served behavior.** `src/models/mimo_v2/generate.py`'s loop opened with `cache.reset()`, so every
request began at position zero and a chat loop that resends its history forward-passed the history
again on every turn. On this runtime that is expensive in a way it is not on a card-resident model:
a prefill chunk reaches nearly every routed expert, so the tokens already in the prompt are paid for
in expert copies over PCIe as well as in arithmetic — a 2048-token prompt is ~12 s of four-rank
prefill at the served 2048-token chunk, while the new text in a chat turn is usually tens of tokens.

The store is the same one V4.1 serves from, and
[that page](v41_prefix_cache.md) is where the key, the longest-prefix match, the byte-budget LRU and
the `Entry.logits` row are described — none of it is restated here. What is different is **what a
snapshot is**: V4.1 stores a window ring plus compressed tables, and this model's cache is two
buffers a layer where a windowed layer is a ring and a global layer is the context. That difference
is the whole of `src/models/mimo_v2/prefix_cache.py`.

## What a stored prefix is

`MimoV2KVCache` holds `[kv_heads, slots, head_dim]` keys and `[kv_heads, slots, v_head_dim]` values a
layer, and the two families fill them differently. On one rank of the served four-way split:

| Buffer | Layers | Rows kept | Bytes |
| --- | --- | --- | --- |
| `layers.{l}.key` / `.value`, windowed | the 39 with a window | **the whole 128-slot ring** | 6.1 MiB over all of them, fixed |
| `layers.{l}.key` / `.value`, global | 0, 5, 11, 17, 23, 29, 35, 41, 47 | `min(slots, written)` = `p` | `5760 * p` |
| `written` | all 48 | one `int64` a layer | 384 B |

**≈ 5760 bytes a token + 6.1 MiB.** At 1024 that is 11.7 MiB of state, at 4096 28.6 MiB, at 262144
**1.41 GiB**. The served shape's figures are a quarter of the unsplit attention's — 23040 bytes a
token and 24.4 MiB of rings — because the split divides the key heads with the query heads: a global
layer keeps 1 of its 4 key heads a rank and a windowed layer 2 of its 8.

The two families are stored differently for a reason that is arithmetic and not preference:

- **A ring is stored whole.** A windowed layer is written modulo its length, so after the prompt
  passes 128 positions the live ones are scattered across every slot — there is no leading run of
  rows to cut. There is also nothing to win: all thirty-nine together are 6.1 MiB, which is 0.4% of
  what the nine global layers cost at 262144, and they are a fixed cost whatever the prompt's length.
- **A global layer is cut to its written prefix.** Its buffer is the context appended from zero, so a
  prefix of `p` positions lives in rows `[0, p)` and the rest of the buffer has never been written.
  This is the cut that makes the store worth having, and the header it is taken against is the
  layer's own write head rather than a guess.

`Entry.nbytes` also counts the logits row, which is the point of an exact repeat: on this checkpoint
that row is 152576 float32 = 610 KiB, so a 1024-token entry prices at 12.3 MiB of a 4 GiB budget —
half of it the rings, and 5% of it the row.

## Why the rows a snapshot does not carry are not a correctness problem

A restore copies into the leading rows and leaves a global layer's tail holding whatever the
*previous* request wrote there. That is sound because every read of the cache is bounded by the write
head: `prefix(layer, upto)` refuses an `upto` above `written`, and the positions it returns are
`[upto - slots, upto)`, all below the head. The continuation's own appends then land in that tail
before anything reads it — `append` places position `q` at slot `q % slots`, which for a global layer
is row `q` — so the stale rows are overwritten rather than read. The head is what says which rows
mean anything, not the buffer.

That is why the snapshot carries `written` as a tensor beside the buffers: it is not a buffer of the
cache, no walk over the cache would find it, and a snapshot that dropped it would answer the
continuation's first token out of whatever span the previous request had left written.

`tests/test_mimo_prefix_cache.py` pins this the way a claim about unreachable memory has to be pinned:
the rows past the snapshot are filled with poison before the restore, and the resumed completion is
required to be **bit-identical** to the cold one. A tolerance would have accepted a resume that read
them.

## How a request is resumed

`generate.Generation` gained `cached_tokens`, and `_resume` is the three-way dispatch:

1. **A hit shorter than the prompt** — restore the rows, then forward `ids[cached_len:]` at
   `start_pos = cached_len`. The position is the whole of the argument: the rotation, the attention
   bounds and the ring's slot arithmetic all read absolute positions.
2. **A hit as long as the prompt** — forward nothing and sample `Entry.logits`. Replaying the prompt's
   last token instead, which looks cheaper, is not merely inexact here: the window a query reads is a
   function of the query's own position, so a query at `p-1` wants `[p-1-window, p-1)` while the entry
   holds the state *after* `p`, whose ring has already evicted the key below the window and whose slot
   for it now holds position `p-1`. The row the first new token comes from was computed by the forward
   that produced the anchor, so it rides with the entry and there is nothing to re-derive. The test
   counts the calls and requires none.
3. **A miss** — the cold prompt, with the head anchor taken in the middle of it.

| Anchor | Length | What it serves |
| --- | --- | --- |
| head | `prefix_cache_head_tokens`, 1024 by default | a *different* conversation with the same rendered header — the same system message and tools JSON, diverging after it |
| end | the prompt's own length, exactly | the same conversation's next turn, and a retry of this one |

Both anchors need the forward to *end* at the anchor's length, because state exists only at a
forward boundary; the cold path therefore runs as `prefill(ids[:1024])` then
`prefill(ids[1024:], start_pos=1024)`. The extra boundary is paid on a miss only. The end anchor is
keyed at the exact length and not rounded to the store's 64-token block, for the same reason.

**A resumed prefill is not bit-identical to a cold one, and this model cannot claim it is.** The
attention's gemms change shape with a chunk, and
`tests/test_models_mimo_v2_device_attention.py::test_a_chunked_prefill_is_a_one_shot_prefill` holds
the boundary at `atol=1e-4` on one layer's output where V4.1's chunked-prefill file pins its own
boundaries bit-equal — which is the reason that page can make the stronger claim and this one cannot.
A resume is a chunk boundary at the stored length, generally not a multiple of `prefill_chunk`, and
through 48 layers and the head the same boundary moves the prompt's last row by **4.85e-02 of its
peak**; the section below is where that number and its null arm come from. What the store is *not*
allowed to do is change the state: the rows it hands back are the rows a cold prefill wrote, and the
tests hold a resume to that at the tolerance the chunked-prefill file uses and no looser.

**The reuse is agreed on without a collective.** Every rank builds its own store, and a rank that
resumed where its peers prefilled would enter a layer's all-reduce alone and hang. Nothing is
exchanged to make them agree, because the decision is a function of the request and the launcher's
options and of nothing local: the key is the prompt's own tokens, the budget and the anchor lengths
come from the command line, and the eviction that follows from them is the same on every rank because
the attention split is even — a snapshot's byte count is one number, and
`test_every_rank_of_a_split_attends_over_the_same_number_of_heads` measures it rather than assuming
it. The four-rank probe prints the same entry on all four — `23.6 MiB for 3072 tokens`.

## Serving surface

| Knob | Default | Meaning |
| --- | --- | --- |
| `--enable-prefix-caching` / `--no-enable-prefix-caching` | on | the switch; off folds the budget to zero |
| `--backend-option prefix_cache_bytes=4g` | `4g` a rank | the store's byte budget, as an integer or a `k`/`m`/`g` suffix |
| `--backend-option prefix_cache_head_tokens=1024` | `1024` | the head anchor's length, or `0` for the end anchor alone |

4 GiB a rank is 16 GiB over four, and it is ordinary pageable host memory and deliberately **not**
`/dev/shm`, which the 149.81 GiB expert bank already holds. At 5760 bytes a token it is about 745,000
tokens of global-layer state, so eleven 64k conversations fit and the least-recently-used is evicted
past that; a single 262144-token entry is 1.41 GiB and about a third of the budget.

A response reports what it reused the way OpenAI does — a subset of `prompt_tokens`, not a discount
on it — and only when it is nonzero:

```json
{"usage": {"prompt_tokens": 2103, "completion_tokens": 64, "total_tokens": 2167,
           "prompt_tokens_details": {"cached_tokens": 1024}}}
```

`capabilities.supports_prefix_caching` is `True` while a store exists, and `/metrics` carries the same
six `pocketllm_prefix_cache_*` series the V4.1 path publishes, from the same names, set rather than
added at scrape time from a snapshot the backend rebinds under the request lock.

## What it costs and what it saves

`tests/probe_mimo_v2_prefix_cache.py`, four ranks on the released checkpoint, a real document through
the checkpoint's own tokenizer,
`--tokens 4096 --prefix 3072 --chunk 2048 --chunk-rows 16 --rounds 2 --budget 8`, the slowest rank:

| | Result |
| --- | ---: |
| Cold, all 4096 tokens from zero | **20.07 s — 204.0 tok/s** |
| Warm, the stored 3072 restored and 1024 forwarded | **7.02 s — 145.9 tok/s** |
| The same prompt again, cut at 3072 with no store (the null arm) | 23.90 s — 171.4 tok/s |
| Snapshot and admission, paid on a cold prefill | 20.0 ms |
| Restore of the 3072-token entry | 15.1 ms |
| An exact repeat: a restore and a sample, nothing forwarded | 14.2 ms |
| The entry | 23.6 MiB for 3072 tokens — 8039 bytes a token |

The two arms forward different token counts, so their rates are rates and not a speedup: a chat turn
saves the tokens it does not forward, at whatever a prefill token costs that session. The null arm is
the honest control for the rate, and it is not the cold rate either — 23.90 s against 20.07 for the
same 4096 tokens, because cutting at 3072 leaves a 1024-token chunk and the design record's sweep has
a 1024-token chunk at 134 tok/s against 174 at 2048. A resumed prefill is a chunk boundary at the
stored length, generally not a multiple of `prefill_chunk`, and it is priced like one.

**The store itself is exact, and the probe is built to show that and not to assume it.** Three arms
produce the same prompt's last row:

| The row, against a peak of 31.6 | |
| --- | ---: |
| cold against rechunk — where the prompt is cut | 4.85e-02 |
| warm against rechunk — what the store adds | **0.00e+00** |
| The token the row would draw | 15521, on all three, both rounds |

`rechunk` forwards the same tokens the warm arm does, from the same state, one of the two having taken
that state out to host memory and back. The difference is zero: the snapshot, the copy and the restore
reproduce the state at 3072 tokens of released checkpoint exactly, which is the release-scale version
of what the poison test pins at toy scale. The whole 4.85e-02 is the chunk boundary — the cold arm
cuts the prompt at 2048 and a resume cuts it at 3072, and a different cut is a different arithmetic on
this runtime, which the design record already documents for chunk width. Two honest consequences:

- **A resumed prefill is not a cold one, to ~5% of the row's peak at this depth**, and a greedy chain
  can therefore move a token where the cold answer sat near a rounding boundary. V4.1 can claim the
  stronger property because its chunk boundaries are not a source of difference; this model's are, and
  no amount of store correctness removes that. On this prompt it did not move the token — the three
  arms draw the same id, which is the reading a single prompt supports and not a guarantee.
- **It is not a reason to turn the store off.** The alternative to a resume is the null arm: the same
  prompt, 23.90 s against 7.02, on the state the store hands back bit for bit. What a turn buys is the
 16.9 s between them, and what it costs is the cut.

## Non-goals

- **Batching.** The store changes what a single request costs, not how many run at once; requests
  still serialize on one lock, and `supports_batch` is still `False`. Every routed layer closes with
  an all-reduce at a fixed point in every rank's program, so concurrency here is a scheduler over the
  collective rather than a batch dimension — a separate piece of work with a separate design.
- **Snapshots during decode**, which would also reuse the tokens the previous turn *generated*. This
  is the one that would make the store pay for itself on a long single conversation rather than on a
  resent history.
- **Paged or block-granular KV.** The ring's live rows are scattered and the global layers are
  contiguous; a paged store is a layout rewrite of `MimoV2KVCache`, not a patch to this one.
- **A store that survives the process.** It is rebuilt per rank, per launch. A 262144-token entry
  takes a full prefill to produce and would be worth persisting, but nothing about the current layout
  (a pageable host dict keyed by blake2b) promises that a later build's snapshot is the one this
  build wrote — `geometry_tag` is what stands in for that promise today.

## Where it is tested

| File | What it pins |
| --- | --- |
| `tests/test_mimo_prefix_cache.py` | the model's half: the rows a snapshot carries per family, the copy in and out, a poisoned tail that must not be read, resumes cut at 1, 3, 4, 7 and 11 tokens, a resume past the window and one that wraps twice, the geometry tag, and every rank attending over the same number of heads. The loop's side: an exact repeat forwarding nothing, a longer prompt forwarding only the remainder, the head anchor, and a miss being the prefill it always was. And the adapter: one store across requests, the switch, the byte budget, and a cache that describes no state reporting no store rather than a broken one |
| `tests/probe_mimo_v2_prefix_cache.py` | the numbers: cold, warm and the un-stored null on one prompt, the store and restore costs, an exact repeat, the relative movement of the last row with the store's own contribution separated from the cut's, and the token each row would draw |
| `tests/test_mimo_serving.py` | the served path the store sits in, unchanged: a cold request is one prefill and one step a token, and a request with no store at all still runs |

## Evidence and related notes

- [Cross-request prefix caching on V4.1](v41_prefix_cache.md) — the shared store, and the same feature
  on the other model this repository serves
- [MiMo-V2.6-Flash: design and measurements](mimo_v2_6_flash_design.md) — the runtime this path sits in
- [The attention split over the ranks](mimo_v2_6_flash_design.md#the-attention-split-over-the-ranks) —
  why a snapshot is a quarter of the heads, and why the four ranks agree
- [MiMo-V2.6-Flash](../models/mimo-v2.6-flash.md) — the model guide
- [Benchmarking and reporting](../guides/benchmarking.md) — how the numbers above were taken
