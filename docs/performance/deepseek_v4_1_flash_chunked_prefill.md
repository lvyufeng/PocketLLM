# DeepSeek-V4.1-Flash: chunked prefill for a prompt too long for one forward

The [routed-experts sitting](deepseek_v4_1_flash_device_experts.md) ends with a 512-token prefill at
14.73 tok/s, and the [bottleneck pass](deepseek_v4_1_flash_remaining_bottlenecks.md) ends with the
batched expert call that earned −27.0% of it. Neither asked what this page is about: **how long a
prompt this host can prefill at all, and what the answer costs.** The limit was never the rate. Two of
the activations one forward holds are linear in the sequence length and neither is needed past the row
it is computed on, so a 256K prompt does not fit in a 22 GiB card however fast the card is — the
forward dies of allocation, at 640 MiB asked for with 572 MiB free, inside `hc_post`.

`Backbone.forward(..., chunk=n)` splits the prompt into `n`-token forwards that compose into the same
forward, because every layer already keeps its own state in its caches. That is the whole of the 256K
support: **262144 tokens in 3711.0 s, 70.6 tok/s on all four ranks, 57.98 s a 4096-token chunk, peak
17.34 GiB of the 22000 the card reports.** It is only *reasonable* if the chunks are exact rather
than approximate, so the first half of this page is what a chunk has to preserve and the tests that
hold it to that; the second half is what a chunk costs, and where those seconds go.

## Run record

| | |
| --- | --- |
| Model | DeepSeek-V4.1-Flash, released checkpoint, fp8 dense + packed-fp4 experts |
| Checkpoint | `/mnt/data3/DeepSeek-V4.1-Flash`, 48 shards, 475.24 GiB, resident bank attached from the 457.78 GiB `/dev/shm` segment |
| Commit | `5e7ff05` on `feature/v41-256k-context`, stacked on `89d0e88` on `perf/v41-hc-token-tile` and comments only on that branch since (`83ed600`), both against `master` `a533a0a` |
| Configuration | TP4, one process a card, `torchrun --nproc_per_node=4`, `--threads 22`, `DEEPSEEK_V41_RESIDENT_EXPERTS=1`, `--pool-rows 148` |
| GPUs | 4 x RTX 2080 Ti, 22528 MiB each, GPU0-GPU1 PHB and GPU2-GPU3 NV2, cross-pairs SYS |
| CPU / RAM | 2 x Xeon E5-2696 v4, 88 hardware threads, 1007 GiB RAM |
| Software | Python 3.11.14, torch 2.9.1+cu128, `deepseek` conda env |
| Probes | `/tmp/probe_v41_chunk_scaling.py` (the sweep), `/tmp/probe_v41_chunk_profile.py` (one chunk, phase by phase), `/tmp/probe_v41_seed.py` (ten taps inside layer 0 and layer 1), `/tmp/diag_retention.sh` (the per-process retention diagnostic) |

## Why a chunk and not a bigger card

The two activations are `hc_mixes`'s and `main_hiddens`. `Block.hc_mixes` flattens the
Hyper-Connections copies into one `[s, hc_mult * dim]` tensor and casts it to fp32 — 262144 x 20480 x 4
bytes, **20.0 GiB at 256K**, from a call whose output is a handful of numbers a token. `main_hiddens`
is three MTP taps of `h.mean(dim=2)` at `[s, dim]` bf16, another **7.5 GiB**. Neither is needed for
more than the row being computed, and `hc_mixes`'s is the row's own: it is produced by one sub-block
and consumed by the next, which is why a chunk's first sub-block starts from
`make_identity_pre_mix(h, self.hc_mult)`.

Splitting the prompt is therefore the only way past 22 GiB, and it costs one thing up front: a chunked
forward returns the **last** chunk's `main_hidden`, because the earlier chunks' rows are exactly the
linear-in-`s` tensor the chunk exists to not hold. The MTP head predicts from the tail of the
sequence, so that is the row it wanted; a caller that wants the full-sequence form does not get one,
which is stated in `Backbone.forward`'s docstring rather than discovered.

A third activation is linear in `s` and has no such escape: the block's own residual stream, which
Hyper-Connections make `hc_mult` = 4 copies of a 5120-wide hidden state wide — 40 KiB a token in bf16,
so **40 MiB a 1024 tokens** and 1.25 GiB at 32768. Two of it are live across a sub-block, the
`residual` the block reads and the value `hc_post` returns, and no tile removes either fact, so it —
and not the two above — is what stops the *chunk* getting wider. That is measured in
[the cost section](#what-a-chunked-prefill-costs); the two above are why the chunk has to exist at
all.

## What a chunk has to preserve

A chunk past the first is not a decode step and not a prefill. Five caches carry the state across the
boundary — the sliding-window ring, the compressed-KV cache, the index-key cache, and the compressor's
two group buffers — and `_is_continuation(pos, seqlen) = not pos.first() and seqlen > 1` is what tells
the two apart. **It turns on the count of queries and not on the position**, which is the property
`test_a_chunk_of_one_token_is_the_decode_path_and_not_the_continuation_one` pins: a one-token forward
at a position past zero is the decode body, which is what the CUDA graph captures, and a rule keyed on
the position would keep every existing test passing while quietly moving the capture path's
arithmetic.

Three things have to hold, and each is a test in
`tests/test_models_deepseek_v4_1_chunked_prefill.py`:

* **The window is a window, not a chunk.** A query in a later chunk sees `window_size` positions that
  reach back past its own chunk into rows the earlier chunk wrote. The ring is written and read by
  absolute position (`slot = position % window_size`) on both paths, and the continuation body hands
  `sparse_attn` the ring and the chunk concatenated so a position is named by which half it fell in.
  The row that comes out is the same *positions in the same order* as the one-shot row: oldest first,
  which is the order the prefill branch emits and the order `sparse_attn`'s denominator sums in. The
  ring half is read **before** the ring is advanced, because a chunk's rows go into exactly the slots
  the `window_size` positions just in front of the chunk live in.
* **A group is `compress_ratio` positions, not `compress_ratio` tokens of a chunk.** A chunk boundary
  lands inside a group in general, so a chunk's first tokens close a group the previous chunk opened
  and those tokens pool in the carried state; a trailing partial group waits there for the next chunk
  to close it.
* **The indexer reaches what the query can reach.** `compress_lens` is counted from absolute position
  and the reachable prefix is `pos.group(ratio, seqlen)` groups wide, so a chunk's first query can see
  the whole history in front of it. A count that started at 0 would mask all of it away, which is why
  `Pos.upto` takes the query count as an argument.

The comparison is `torch.equal` and not a tolerance, at a chunk width chosen so nothing is truncated,
and it is taken on every cache the module tree holds rather than on a hand-written list — the failure
this guards against is a cache a chunk forgot to carry, and a list is how one goes unnoticed. Ten
widths from 1 to 12 against one one-shot prefill; the four V4.1 files that need no device —
`test_models_deepseek_v4_1_chunked_prefill.py`, `_attention.py`, `_modules.py` and `_config.py` —
run **54 passed** with `CUDA_VISIBLE_DEVICES=""`, so the property is checkable on a host with no card
in it. The files that do open a device (`_loader.py`, `_tp.py`) are run on the cards separately.

### What exactness does and does not mean here

Equal caches are what makes a chunk *compose*; they are not a claim that the logits come out
bit-identical, and this page does not make that claim. A chunk's bodies are tiled differently from a
one-shot forward's — the continuation path walks the chunk where the prefill path runs one vectorized
expression — so the fp32 reductions associate differently and the two orderings separate from the
tail of the first chunk on. That is the same class of difference `generate.py` already documents
between a one-shot prefill and stepping the same tokens one at a time, bisected in
[the host-run page](deepseek_v4_1_flash_host_run.md); what the tests above rule out is the *other*
kind, a chunk naming the wrong positions or leaving a cache behind.

Two bounds come with that, both stated on the flag rather than discovered:

* A chunk has to be at least `index_topk * compress_ratio` tokens — 1024 for the ratio-2 layers —
  for its selection to have the candidates a one-shot forward's has. Below that the indexer chooses
  among fewer compressed positions and the answer genuinely differs, which is `--prefill-chunk-tokens`
  own caveat and not a rounding matter.
* `/tmp/probe_v41_seed.py` measures that first kind at real scale and names where it starts. One
  4093-token prompt at a 2048-token chunk, both arms on the four cards, ten taps read inside layer 0
  and layer 1 with the compared span cut to the last chunk's 2045 rows. Three of those taps come out
  **bit-identical**: layer 0's attention output, its MoE input, and its *dense* shared expert. The
  first of the three is the control, being the last tensor the two arms were already known to agree
  on; the second is what every cache a boundary carries is ultimately consumed by, and the third is
  the one MoE path whose call shape did not change. The gate is tapped as the *set* of experts a token
  picks rather than elementwise, because a top-k over scores that can tie is not pinned in order, and
  the sets agree exactly: **0 of 2045 rows pick a different expert set**, and none of them reorder.

  The first differences are `1e-6`, in the gate's weights on 89% of their entries and in the **routed**
  experts on 89% of theirs — the two modules whose GEMM row count went from 4093 to 2045. That is the
  shape of a re-tiling and not of a boundary carrying a wrong value: a slot written in the wrong place,
  a group counted from the wrong position or an indexer reaching too short would all have shown up in
  the attention output or the MoE input first, and those are the two taps that agreed. So the boundary
  state composes, and what is left is the number of rows the GEMM is handed. `moe_out` then differs by
  at most `2^-8` — one bf16 ULP of a value in `[1, 2)`, a rounding step and not an error — on 0.03% of
  its entries. From there it travels and grows: layer 1's Engram by `2^-9` on 0.004% of its entries,
  layer 1's attention input by `3.7e-4` on 0.007%, and the last row's logits end **0.2159** apart,
  with the argmax unchanged.

  Running the one-shot arm a second time is the control for all of it, and it is **bit-identical at
  every one of the ten taps**, last-row logits included. So none of the above is run-to-run drift: the
  engine is deterministic, and a chunked prefill reproduces exactly at a given chunk width. What it
  cannot do is reproduce the one-shot logits, which is the point of this section.

## What a chunked prefill costs

One process a leg. The sweep ran five legs in one process on the theory that the caches are sized by
the longest leg and every leg then reads the same arena; the second leg died allocating 640 MiB with
572 MiB free. The retention diagnostic says why, and it is not a leak: the first forward of a process
allocates about **3.08 GiB of per-layer buffers that persist**, 12023 MiB of live tensors becoming
15085/15085/16156/16236 across the four ranks, and a second leg settles at exactly those numbers to
the MiB. So a leg's free memory is 3782-4102 MiB rather than the 8832 the caches alone leave, and the
leg that answers the question has to be the first leg of its process.

The three long legs share 262208-wide caches, which is what makes them readable against each other —
12023 MiB allocated and 8832 MiB free before the first chunk, whatever the prompt is. The three
32768-token legs use 32832-wide caches instead, because a chunk width is a question about the chunk
and not about the cache budget, and paying for a 256K cache to answer it would put the widest chunk
out of reach for no reason.

| Prompt | Chunk | s | tok/s | s a chunk | staged rows a token | peak GiB |
| --- | --- | --- | --- | --- | --- | --- |
| 32768 | 4096 | 456.3–458.9 | 71.4–71.8 | 57.04–57.36 | 1.27–2.36 | 15.14–16.42 |
| 131072 | 4096 | 1824.1–1825.1 | 71.8–71.9 | 57.00–57.03 | 1.23–2.29 | 15.48–16.85 |
| 262144 | 4096 | 3711.0–3712.5 | 70.6 | 57.98–58.01 | 1.23–2.34 | 15.93–17.34 |
| 262144 | 8192 | OOM after one chunk | — | 117.5–118.3 | — | 17.78–19.77 |
| 32768 | 16384 | OOM in the first chunk | — | — | — | 20.22–20.54 |
| 32768 | 32768 | OOM in the first chunk | — | — | — | 19.66–19.68 |

**262144 tokens fit, and 8192-token chunks do not.** Three legs above 4096 say where the ceiling is
and what it is made of. The 8192 leg — at 262144, the hardest caches in the sweep — ran its first
chunk at 117.5–118.3 s and then died on rank 2 asking for **320.00 MiB with 216.31 MiB free**, 20.34
GiB of the card's 21.48 GiB already in use; the 16384 and 32768 legs, at 32768 tokens, died on their
first chunk and produced no time at all. Two different allocations stopped them and they are worth
telling apart.

At 8192 the traceback ends inside `_hc_post_pass`, at `modules.py:582`, on the fp32 `comb * residual`
broadcast: at the 1024-token tile `DEEPSEEK_V41_HC_TOKEN_TILE` defaults to, that temporary is
`[tile, hc_mult, hc_mult, dim]` fp32 — 320 MiB — and the card had 216 MiB left to give. At 16384 the
traceback ends one frame up, at `hc_post`'s own `torch.cat` (`modules.py:594`), asking for **640 MiB**;
at 32768 the same line asks for **1.25 GiB**. Those two are exactly `chunk * 40 KiB`, which is the
concatenation of the tiles and therefore `hc_post`'s return value: bf16 `[1, chunk, hc_mult, dim]`, the
sub-block's residual stream. That one is not a temporary and no tile size removes it — the block reads
it, writes it back through `hc_post`, and the next sub-block reads it again — so it is **40 MiB a
1024 tokens**: 320 MiB at 8192, 640 at 16384, 1.25 GiB at 32768, the three requests verbatim. A block
holds two of them at once, the `residual` it reads and the value it returns, and `torch.cat` needs the
second while the first is still live. (The 320 MiB at 8192 invites the wrong reading, because it is
also what the whole chunk's residual comes to: 1024 x hc_mult x hc_mult x dim in fp32 and
8192 x hc_mult x dim in bf16 are the same number of bytes. The frame, not the size, is what tells the
two failures apart.) That is the ceiling: between 8192 and 16384 tokens, set by Hyper-Connections'
`hc_mult` copies rather than by this implementation, and narrower than the prompt by a factor of 16 to
32.

**The 8192 leg prices nothing about width, and that is the trap in it.** It ran one chunk, and that
chunk is the first — which carries the warm-up every leg pays: at 4096 the first chunk costs 64.35 s
against a steady 56.1–57.9. The tokens it covers are the 4096 leg's chunks 1 and 2, **64.35 + ~56.2 =
120.6 s**, so the wider chunk is **118.25 s against 120.6 s on the same 8192 tokens**: cheaper, not
dearer, and by 2%, which is inside one chunk's noise. The three legs above 4096 are in the table
because a ceiling is a result and not one of them can price a width. Whether 4096 is the fastest
width at all is the question the last section answers from below.

**A 262144-token prompt prefills in 3711.0 s at 70.6 tok/s on every one of the four ranks**, at 57.98
s a chunk, peaking at 15.93 GiB on rank 0 and 17.34 GiB on rank 3 of the 22000 the card reports. That
is the 256K support the split was for, and the two numbers that say it is *support* and not a lucky
allocation are that the peak does not move and the rate does not fall. **The deal is the one lever
left on the table here**, and on this page every figure is the `sorted` deal, which is no longer the
default: the id deal is the same leg at **3121.0 s and 83.99 tok/s, 1.19x, 0.53 staged rows a token
against 2.31**, for a peak of 17.72 GiB a card — [the expert page prices both deals, with the parity
and the arena](deepseek_v4_1_flash_device_experts.md#the-deal-is-a-choice-and-dealing-ids-instead-of-positions-balances-the-staged-set).
Those are *this* tree's numbers: **on the tree that ships, which is this one plus the three prefill
kernels that landed after it, the same pair is 2532.75 s to 1789.92 s and 103.50 to 146.46 tok/s,
1.41x**, because taking a third out of a chunk's compute leaves the staged rows a larger share of what
is left ([the re-take](deepseek_v4_1_flash_device_experts.md#the-same-pair-re-taken-on-the-tree-that-ships-149x-and-142x)).
The three 4096-token legs put
that second claim three ways: steady state, which is every chunk past the first — the first carries
the warm-up and costs 64.4 s in all three legs, whatever the prompt — is **56.07 s at a 32768-token
prompt, 56.76 at 131072 and 57.88 at 262144**. Eight times the context costs 1.81 s a chunk, 3.2%.
The per-chunk seconds across the whole 64-chunk leg are

| Chunk | 1 | 13 | 25 | 37 | 49 | 61 | 64 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| context | 4096 | 53248 | 102400 | 151552 | 200704 | 249856 | 262144 |
| s | 64.35 | 56.14 | 56.74 | 57.33 | 58.48 | 60.34 | 60.69 |

— flat to 2% from 20K to 150K of context, then rising to 60.69 s by the last chunk, for a 4.55 s
context term against a 56 s floor. The indexer is the only quadratic thing in the model, and at a
4096-token chunk its share of the wall is small enough that the second half of the prompt looks like
a slow drift rather than a curve — which is the first thing the attribution below has to be read
against, because it says the lever is not there.

The staged-row counts differ by rank on purpose and it is the deal, not a defect: rank 0 and rank 1
own two of a token's six routed experts each and rank 2 and rank 3 own one, so the first pair stages
2.31–2.34 rows a token against the second pair's 1.23, and all four print the same top-8. (That
spread is the other deal's whole subject, and the `id` deal leaves all four ranks within 2% of each
other — 138490, 136607, 139853, 138794 rows over the same leg.)

### How wide the expert pool, at 256K

`--pool-rows` is the other width knob, and it trades memory for chunks: it is how many expert rows sit
resident on the card, and a row that is not resident is a row the forward stages. At 32768 tokens at
chunk 4096, the same probe and the same `--max-seq-len 32832` on both arms:

| pool rows | s a chunk | tok/s | peak GiB allocated | peak GiB reserved | staged rows a token |
| --- | --- | --- | --- | --- | --- |
| 148 | 56.88 | 72.0 | 15.14 | 15.72 | 2.36 |
| 288 | 52.81 | 77.6 | 19.31 | 19.65 | 1.71 |

The top-8 is `[18014, 1, 19533, 21, 19, 20583, 2012, 14972]` on all four ranks in both arms. The
wider pool is **−4.07 s a chunk, −7.2%**, bought with **+4.34 GiB** of settled allocation — and that
price is where it stops being affordable.

Most of the 4.34 GiB is arithmetic rather than measurement. `_shapes` is read out of the checkpoint,
and an expert row there is six tensors: `w1` and `w3` at `[2304, 2560]` int8 with a `[2304, 160]` E8M0
scale each, `w2` at `[5120, 1152]` with a `[5120, 72]` scale — **18,800,640 bytes, 17.93 MiB a row**.
`arena_rows = rows_per_card + hot_rows + pool_rows` is one arena a card that all forty layers share,
so 288 − 148 = 140 rows is 2.451 GiB, and the post-load allocation moves by exactly **2512 MiB** — the
same 2512 MiB at a 32832-wide and at a 262208-wide cache, because the arena is sized by the pool and
not by the context. The remaining **~1.83 GiB** appears only inside a forward, and it is the batched
expert call's own intermediates: a wider pool means more draws hit and fewer rows conflict, so
`_chunk_bounds` cuts the same 4096-token forward into **fewer and bigger** chunks. That is also the
whole of why it is faster, which is why the memory and the speedup are one finding and not two.

At 262144 the wider pool does not finish. The p288 leg runs its first chunk at 57.62–58.81 s and peaks
at 20442 MiB — against p148's 16573 at the same chunk — and then dies in the **second** chunk on all
four ranks:

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 320.00 MiB. GPU 1 has a total capacity
of 21.48 GiB of which 170.31 MiB is free. ... 19.59 GiB is allocated by PyTorch
```

The 320.00 MiB is the allocation that stopped the 8192-token chunk above, `_hc_post_pass`'s fp32
`comb * residual` at the 1024-token Hyper-Connections tile, and 170.31 MiB free is what a 4096-token
chunk has left once the arena is 2.45 GiB bigger. **The 256K configuration therefore ships at 148
pool rows**, and every `s a chunk` number on this page was measured at it.

## Where a chunk's seconds go

The width curve prices a chunk and its slope. It does not say what the other four fifths are, and two
of the three candidates are host work that a device-side clock would not see, so the answer has to be
a tap: `/tmp/probe_v41_chunk_profile.py` prefills to 32768 tokens exactly as the sweep does and then
times **one** 4096-token chunk with a host `synchronize()` around every phase of every layer — 22
taps, 346,042 calls on rank 0 — and reports all four ranks, because the layer split means they are not
the same measurement.

A barrier is not a neutral instrument, so the run measures its own price: the chunk after the
instrumented one is the same width at the same cache size with the taps off.

| one chunk, pool 148, 32768 tokens of context | r0 | r1 | r2 | r3 |
| --- | ---: | ---: | ---: | ---: |
| instrumented | 65.54 s | 65.54 | 65.54 | 65.54 |
| taps off | 56.96 s | 56.96 | 56.96 | 56.96 |
| what the taps cost | **8.58 s** | 8.58 | 8.58 | 8.58 |

**56.96 s is the sweep's own number at these arguments** (56.83–57.00 s at `--chunks 4096`), which is
what says the chunk under the instrument is the chunk the width curve priced. The quiet chunk is one
chunk further into the prompt than the instrumented one, so it carries about 0.1 s of context the
other does not; over 4096 tokens that is inside the 8.58 s being measured and not a second finding.

The taps nest, so the table is a tree: a row indented under another is that parent separated out, not
a second cost. A row's `calls` count is a per-rank maximum, which matters only for the two ranks that
own fewer MoE layers.

| phase | r0 s | r1 s | r2 s | r3 s | calls | of the wall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `moe` | 50.41 | 50.42 | 50.41 | 50.42 | 40 | 76.9% |
| — `moe.routed` | 47.36 | 47.42 | 31.93 | 32.49 | 40 | |
| —— `_issue_chunk`, the grouped fp4 GEMM | 18.13 | 18.03 | 10.66 | 10.53 | 158 | |
| —— `_stage_misses` | 18.56 | 18.60 | 10.66 | 10.92 | 163840 | |
| ——— `_upload`, the expert H2D | 16.67 | 16.72 | 8.99 | 9.04 | 8658 | |
| ——— `_take_buffer` | 0.13 | 0.13 | 0.07 | 0.08 | 8658 | |
| —— `_resolve_row` | 5.22 | 5.23 | 5.34 | 5.26 | 163840 | |
| —— `_drain_chunk` | 0.36 | 0.36 | 0.35 | 0.35 | 158 | |
| —— `_route_ids` | 0.00 | 0.00 | 0.00 | 0.01 | 40 | |
| `attn` | 12.85 | 12.85 | 12.82 | 12.85 | 40 | 19.6% |
| — `attn.sparse` | 8.05 | 8.01 | 8.26 | 7.97 | 40 | |
| — `attn.compress_kv` | 2.12 | 2.14 | 2.12 | 2.14 | 38 | |
| — `attn.indexer` | 2.09 | 2.12 | 2.09 | 2.12 | 8 | |
| — `attn.window` | 0.42 | 0.40 | 0.43 | 0.37 | 40 | |
| `hc_post` | 0.96 | 0.95 | 0.96 | 0.96 | 80 | 1.5% |
| `hc_mixes` | 0.40 | 0.40 | 0.40 | 0.40 | 80 | 0.6% |
| `hc_pre` | 0.29 | 0.29 | 0.29 | 0.29 | 81 | 0.4% |
| `engram` | 0.47 | 0.47 | 0.49 | 0.46 | 2 | |
| `norm` | 0.16 | 0.16 | 0.16 | 0.16 | 169 | |

Rank 0 staged 9,509 expert rows and rank 1 9,516, ranks 2 and 3 5,029 and 5,020, at 15.05–16.33 GiB
peak, and the four ranks' top-8 tokens are the same eight in the same order,
`[455, 1, 223, 8077, 1004, 539, 764, 330]`. `moe.routed` reads 47.4 s on ranks 0–1 against 32.5 s on
2–3 for a `moe` column that is identical on all four: the layer split gives ranks 0–1 about half
again as much of the routed path, the experts are distributed so the ranks are lockstepped, and the
two that own less wait out the difference — inside `moe`'s own body rather than inside a tap on ranks
2 and 3, which is why the column that is not opened is the one that agrees.

**The five phases of a block cover 99.0% of the wall on all four ranks** — 50.41 + 12.85 + 0.96 + 0.40
+ 0.29 = 64.91 of 65.54 s — which is the check this table exists to pass: a block's phases tile a
chunk, so a coverage that is not ~100% would say a tap is missing, not that there is a sixth phase.
The same line over the warm-up's eight chunks reads 99.0–99.2%. The print carries one more line and
it is a warning about the first: the rows inside `routed.*` sum to **6.9×** the 8.58 s the whole
instrument costs on ranks 0–1 and 4.2× on ranks 2–3, because `_upload` and `_take_buffer` are counted
inside `_stage_misses` as well as themselves. No nested row is added to another anywhere below, and
that multiplication is printed so it cannot be.

**Reading the instrument.** The 346,042 wrapped calls pay the 8.58 s and all but ~600 of them are
inside the routed expert call — the five block phases and the attention taps are 582 calls between
them — so the instrument is charged to the MoE and the rest of the table is clean:

- **The MoE is 41.9 s of the quiet 56.96 s chunk (73.5%)**, attention 12.85 s (22.6%), and the
  Hyper-Connections arithmetic, the norms, the Engram, the embedding, the head and the residual adds
  are 2.2 s between them (3.9%).
- **`_issue_chunk` 18.13 s and `_drain_chunk` 0.36 s over 158 calls each** — about four a layer — and
  158 calls is a few hundredths of a second of barrier, so the grouped fp4 GEMM is a measurement and
  not an upper bound: **18.5 s, 32% of a chunk.**
- **`_upload` 16.67 s for the 9,509 rows rank 0 stages** — 1.75 ms a row, 166.5 GiB at 10.0 GiB/s. The
  uninstrumented width curve prices the same rows at 1.465 ms and 12.8 GB/s, and the instrumented
  chunk's wall is 13% above the quiet one, so the 19% between those two row rates is the instrument
  rather than a second mechanism. Both readings are the same finding: **a row's 17.93 MiB crosses at
  two thirds to four fifths of what a PCIe 3.0 x16 link is rated at, and it is not hidden behind
  anything.**
- **The rest of the routed call is not 6.7–9.5 s of host bookkeeping, and the row this bullet was
  read from is the instrument.** Every tap here pays two barriers and `_resolve_row` is 163,840 of
  them, so the run was repeated with the barrier split out of every number
  (`/tmp/probe_v41_chunk_profile_host.py`: the same 22 taps, each recording its preamble sync, its
  call body and its postamble sync, then the same width again with the taps off). **`_resolve_row`'s
  body is 1.52 s over its 163,840 calls — 9.3 µs a call** — and the same run reads it at 1.52 s in
  this chunk on the tree its three kernel changes are merged into and at 1.52–1.55 s a chunk across
  the warm-up's eight before either: four independent 163,840-call groups agreeing to 2%, which is
  the cleanest instance-level measurement in the run. The 5.22 s
  this table reads for the row is therefore the barriers around it: a tap whose body is empty costs
  **11.98 µs a call** on this host, 11.12 of it the two `synchronize()`s
  (`/tmp/probe_tap_price.py`), which is 1.96 s over these calls, and what is left of the row is the
  preamble barrier waiting on copies and a grouped GEMM the previous call had already issued. What
  the routed path spends on the host is its *bodies*, and those are **5.11 s**: `_upload` 2.34 s over
  8,634 calls (271 µs a call), the per-row loop 1.52 s, `DeviceRoutedExperts.forward`'s own glue
  1.97 s, `_issue_chunk` 0.57 s, `_drain_chunk` 0.34 s and `_stage_misses` itself 0.13 s. A
  Python-level per-token loop at 20–29 µs a call over 327,680 of them is not one of the rows of this
  chunk; 2.7% of it is, and the paragraph below prices the part of that a rewrite could take.

  What those microseconds are *not* is the loop's own list and sort work, which is worth knowing
  before anyone saves them twice. A shim carrying the real `_split`, the real per-card dictionary
  probes and a `_pool_row` that answers out of a dict runs the same call at the same shapes — 4096
  rows, topk 8, world 4, 384 experts — at **4.6 µs a call**: 1.1 µs for `route[row].tolist()` and
  its `int()`s, 2.1 µs for `_split`, 1.5 µs for the probe loop (`/tmp/probe_v41_resolve_cost.py`;
  between them the shim's own arithmetic closes, 1.1 + 2.1 + 1.5 = 4.7 against 4.6 measured, and one
  `route.tolist()` for the whole chunk is 0.8 ms against 4.6 ms for the per-row form). So of the
  9.3 µs a `_resolve_row` costs in situ, about 3 µs is the loop and the balance is `_pool_row` and
  the state it walks — the pool's own row arithmetic, its eviction bookkeeping, and the class
  members the shim does not have. That floor is what makes the rewrite small: the shim's 4.6 µs a
  call is 0.75 s over these 163,840 calls, so of the row's 1.52 s at most 0.8 s sits above what a
  call that did nothing but this must spend — 1.4% of a chunk, against the copies' 24–29% — while a
  vectorized `_split` or a `tolist()` hoisted out of the loop, the two thirds the shim cannot avoid,
  is well under half a percent. The pool's half of those calls is where what is left is, and the row
  below is the one to price before spending it.

`_take_buffer` is worth naming separately, because it is where this path used to lose its seconds:
0.13 s over 8,658 calls, against **6.89 s of a 30.35 s class wall** before the rotation was made to
advance only over rows that stage. 0.13 s is less than the barrier costs those same 8,658 calls, so
the slot wait is now nothing rather than reduced.

**Attention is 22.6% of a chunk and the score pass is most of it.** `attn.sparse` is 8.05 s over 40
calls — 201 ms a layer, 14.1% of the quiet chunk — and 40 calls makes it a measurement. The rest of
the attention is the compressed path, `attn.compress_kv` 2.12 s over 38 calls and `attn.indexer`
2.09 s over 8, and those two are the only rows of the table with a context term: the warm-up's eight
chunks, whose caches run from 0 to 28672 tokens, average 1.59 s and 1.57 s where the chunk sitting at
32768 pays 2.12 and 2.09. Every other row of the two tables agrees to a few percent — `attn.sparse`
reads 7.95 against 8.05, because its index row is the same width wherever in the prompt the chunk is.
**Over a 262144-token prefill it is the compressed path that gets more expensive per chunk and the
score pass that stays flat**, and the score pass is the one the sparse-attention work targets.

**And the ceiling the width runs into costs almost nothing.** The Hyper-Connections arithmetic that
stops a chunk at 8192 tokens — `hc_post` 0.96 s, `hc_mixes` 0.40, `hc_pre` 0.29 over 80, 80 and 81
calls a chunk — is **1.65 s, 2.9% of a chunk**, and at 240 calls of the 346,042 it reads clean. The
chunk is capped by what the residual stream *holds*, at 40 MiB a 1024 tokens, and not by what the
arithmetic on it costs.

**Which resolves the intercept of the width curve.** The fit says a chunk costs **10.4 ms a token plus
1.465 ms a staged row**, and the 1.465 is the copies above. The other constant is what this table
splits, per token: **4.51 ms of grouped fp4 GEMM** (`_issue_chunk` + `_drain_chunk`, 18.49 s),
**3.14 ms of attention** (12.85 s), **0.56 ms of everything else in a block** (2.28 s), and
**1.25 ms of the routed path's host bodies** (5.11 s, the row above as the barrier split measures it
rather than as the table reads it). That is 9.5 ms against the fit's 10.4, and the 0.9 between them
is the host side of what a 22-name tap set does not open — the embedding, the head, the residual
adds and the bodies of the block's small ops. So a chunk's two constants are now six measured terms:
**two thirds of a chunk is device arithmetic and device copies, and the host's share of it is one and
a third milliseconds a token in the middle.**

**What is left, in the order the rows are large.** The grouped fp4 GEMM, 18.5 s and 32% of a chunk,
over four calls a layer with no host work inside them. The expert H2D, 13.9–16.7 s and 24–29%, where
both ways to buy bytes back are unavailable at 262144 (a wider chunk does not fit above 4096, a wider
pool dies in the second chunk at 288 rows) and the copies already run at two thirds to four fifths of
the link. The score pass, 8.05 s and 14%. And the routed path's host bodies, 5.11 s and 9%, of which
the per-row loop is 1.52 s — the row is not the 12–17% this page first read off the table and it is
not the one to attack before the copies, but it is the row that says how much of this chunk is one
process's Python, though not its loops: `_split` and the per-row `tolist()` are 3 µs of the 9.3 and
the rest is the pool.

## Reproducing

The sweep is one leg a process, ordered by what is at stake rather than by length:

```bash
# DEEPSEEK_V41_RESIDENT_EXPERTS=1, four ranks, one leg a line. The caches are pinned with
# --max-seq-len so a 32768-token leg can be read against a 262144-token one.
torchrun --nproc_per_node=4 /tmp/probe_v41_chunk_scaling.py \
    --lengths 262144 --chunks 4096 --max-seq-len 262208 --pool-rows 148 --threads 22 \
    --out /tmp/leg_k256_c4096.pt
```

`/tmp/legs_256k.sh` is the six legs in the order they were run, one `torchrun` each, with the reason
recorded at the top of the file: the first forward of a process allocates about 3.08 GiB of per-layer
buffers that persist, so a leg's free memory is 8832 MiB minus that and the leg that answers the
question has to be the first leg of its process.

The width sweep is the same command with the **caches held at 32832** and only `--chunks` moving,
which is what makes its three legs readable against each other rather than against the long ones:

```bash
for chunk in 1024 2048 4096; do
    torchrun --nproc_per_node=4 /tmp/probe_v41_chunk_scaling.py \
        --lengths 32768 --chunks "$chunk" --max-seq-len 32832 --pool-rows 148 --threads 22 \
        --out "/tmp/leg_k32_c${chunk}.pt"
done
```

The phase table is one chunk of the same configuration, instrumented, and the chunk after it is the
same width with the taps off, so the run prices its own instrument. `--max-seq-len` has to hold both
the instrumented chunk and the quiet one, i.e. `--at` plus twice `--chunk`:

```bash
DEEPSEEK_V41_RESIDENT_EXPERTS=1 torchrun --nproc_per_node=4 /tmp/probe_v41_chunk_profile.py \
    --at 32768 --chunk 4096 --max-seq-len 41024 --pool-rows 148 --threads 22 \
    --out /tmp/chunk_profile_p148.pt
```

The driver is the code, not the probe — this is the call the 256K number above is a forward of:

```python
# 64 forwards of 4096 tokens over 262208-wide caches, which is 262144 tokens of prompt.
for c0 in range(0, total, chunk):
    h, logits, _ = model(ids[:, c0:min(c0 + chunk, total)], start_pos + c0)
```

`chunk` is `Backbone.forward`'s parameter and nothing else on the path changes: a caller that passes
no chunk gets `total`, so the one-shot forward is the same code with a wider loop body rather than a
second implementation. `generate(..., prefill_chunk=4096)`, which is
`--prefill-chunk-tokens 4096` on `src/cli/generate_v41.py`, is the flag that reaches it; it defaults
to off, and its help carries the floor — `index_topk * compress_ratio`, 1024 tokens — as a stated
bound rather than a discovered one.

## How wide a chunk

The ceiling is a memory ceiling, and memory is not the only thing that moves with the chunk, so the
width was swept from below as well — where everything fits and the width is the only variable. Three
legs, one process each, sharing the rest of their configuration: 32768 tokens, `--max-seq-len 32832`
so the caches are the same 32832-wide ones, `--pool-rows 148`.

| chunk | tokens in | tok/s | s a chunk | ms a token | staged rows a token | peak GiB allocated |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | 607.9–609.4 | 53.8–53.9 | 19.00–19.04 | 18.57 | 5.57 | 12.62–12.75 |
| 2048 | 507.3–510.6 | 64.2–64.6 | 31.71–31.92 | 15.52 | 3.49 | 13.54–14.12 |
| 4096 | 454.6–456.0 | 71.9–72.1 | 56.83–57.00 | 13.89 | 2.37 | 15.14–16.42 |

**Wider is cheaper a token at every width measured, and the reason is the last column rather than the
arithmetic.** A chunk resolves the pool and fills it once, so the narrower the chunk the fewer tokens
share the rows it brought in and the more of the next chunk's draws miss: halving 4096 to 2048 costs
**47% more staged rows a token**, and halving again to 1024 costs 60% more again. Those rows are the
H2D of the expert arena — 17.93 MiB each, read straight out of the resident bank — and they are the
dominant per-token term:

| chunk | ms a token | staged rows a token | `10.4 + 1.465 × rows` |
| ---: | ---: | ---: | ---: |
| 1024 | 18.57 | 5.57 | 18.58 |
| 2048 | 15.52 | 3.49 | 15.53 |
| 4096 | 13.89 | 2.37 | 13.87 |

Two constants, fitted to the 1024 and the 4096 legs, put the 2048 leg within **0.01 ms** of what it
measures — which is the whole of the width curve: **~10.4 ms of compute a token, plus ~1.47 ms a
staged row a token.** That slope is a rate: 17.93 MiB in 1.465 ms is **12.8 GB/s**, against the
15.75 GB/s a PCIe 3.0 x16 link is rated at, so a row's bytes cross at around four fifths of what the
link can carry. And the slope is not a fit to these three legs alone. The pool A/B above is a fourth
configuration, on a change that has nothing to do with the chunk — 2.37 → 1.71 rows a token for
**4.07 s a chunk** — and it lands on **1.51 ms a row**, 3% from the width sweep's 1.465. Whatever the
mechanism is, it is proportional to staged rows a token and to nothing about how the forward is cut.
[The profile below](#where-a-chunks-seconds-go) puts a tap on the copies themselves.

**Which reorders the two ceilings.** A chunk is not priced by its width; width is only how a chunk
reaches its rows, and the intercept of that fit is what a chunk costs when its rows are free: **10.4
ms a token, 96 tok/s**. The 256K configuration runs at 13.89 ms a token of it, so **a quarter of a
256K prefill is expert bytes** — and both of the levers that buy bytes back are the ones the card
refuses at 262144: a wider chunk does not fit above 4096, and a wider pool dies in the second chunk
at 288 rows. The width curve says what a narrower chunk would cost in the other direction (18.57 ms a
token at 1024), which is the room the 4096 sits in: between the memory it cannot have and the rate it
would pay for less.
