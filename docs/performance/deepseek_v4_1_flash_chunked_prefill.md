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
support: **262144 tokens on all four ranks, 1789.92 s at 146.46 tok/s, peak 17.72 GiB of the 22000 the
card reports.** The split itself landed at 3711.0 s and 70.6 tok/s; the `id` expert deal took that to
3121.0 and 83.99, and the three stacked prefill kernels under it took it to the 1789.92 — [both legs of
that last step are re-taken on the expert
page](deepseek_v4_1_flash_device_experts.md#the-same-pair-re-taken-on-the-tree-that-ships-149x-and-142x),
which is where the deal's own 1.19x is. What this page adds is what a chunk has to *preserve* and what
a chunk *costs*: the split is only reasonable if the chunks are exact rather than approximate, so the
first half is the five caches a boundary carries and the tests that hold it to that, and the second half
is the seconds inside a chunk — including [the one row that grows with
context](#the-one-row-that-grows-with-context).

## Run record

| | |
| --- | --- |
| Model | DeepSeek-V4.1-Flash, released checkpoint, fp8 dense + packed-fp4 experts |
| Checkpoint | `/mnt/data3/DeepSeek-V4.1-Flash`, 48 shards, 475.24 GiB, resident bank attached from the 457.78 GiB `/dev/shm` segment |
| Commit | `5e7ff05` on `feature/v41-256k-context`, stacked on `89d0e88` on `perf/v41-hc-token-tile` and comments only on that branch since (`83ed600`), both against `master` `a533a0a` |
| The tree the headline is on | `master` `126ac19`, one deal later than the sweep above: the `id` deal is the default at `38edf9b`, and the three prefill kernels of `7102c19` (`attn.sparse`), `aa83816` and `b394ddb` (the MoE's weights and its reduce) are under it. The two 262144 legs are taken on the same code as a rebase, `/tmp/deal3` with `DEEPSEEK_V41_EXPERT_DEAL=id`; both phase tables in [the row that grows with context](#the-one-row-that-grows-with-context) are taken on `126ac19` itself, and their `attn.sparse` of 2.05 s against the sweep's 8.05 s is what says so |
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

**That lever was read A-B-A-B afterwards, and it is real, larger than the single reading, and located
somewhere else.** Four processes — pool 148 twice, then 288 twice — at `--at 32768 --chunk 4096
--max-seq-len 41024` with `DEEPSEEK_V41_EXPERT_DEAL=id`, one 4096-token chunk a process, rank 0
(`/tmp/probe_v41_chunk_profile_host.py`, `/tmp/chunk_ctl_{a1,a2,b1,b2}.pt.r0`):

| column (s) | 148, p1 | 148, p2 | 288, p1 | 288, p2 | floor | lever |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `quiet` — the same width at 36864, taps off | 27.923 | 27.559 | 25.436 | 25.537 | 0.364 | **−2.254 (−8.1%)** |
| the tapped chunk at 32768 | 35.522 | 35.748 | 34.758 | 34.498 | 0.260 | −1.007 (−2.9%) |
| `moe` | 26.445 | 26.695 | 25.593 | 25.446 | 0.250 | −1.051 |
| — `moe.routed` | 21.384 | 21.163 | 21.792 | 21.315 | 0.477 | **+0.280 (+1.3%)** |
| —— `routed.resolve` | 6.877 | 6.764 | 7.190 | 6.990 | 0.200 | +0.269 |
| —— `routed.stage` | 6.839 | 6.756 | 6.900 | 6.890 | 0.083 | +0.097 |
| —— `routed.upload` | 3.843 | 3.851 | 3.832 | 3.825 | 0.008 | −0.018 |
| `attn` | 6.776 | 6.776 | 6.908 | 6.783 | 0.125 | +0.069 |

Two processes of the *same* setting differ by 0.364 s on `quiet` — **1.3% of its own value** — so the
−8.1% is **6.2× the floor** and the lever stands. It also sizes the working rule that a single-run
`quiet` or `moe.routed` move under ~10% is not an effect: on this evidence that threshold is
conservative by most of an order of magnitude, since `moe.routed`'s own floor is 2.2% and `quiet`'s is
1.3%. Two repeats bound a spread from below, so read the floor column as a floor.

**And the faster chunk is not a staging win.** `staged` is 2178 rows in both 148 arms against 2174 in
both 288 arms — 0.2% — and `routed.upload` moves 3.843 → 3.832 s, which is the same bytes: a wider
pool did **not** make more draws hit. The routed sub-phases net **+0.11 s against the wider pool**, the
wrong sign, while the ~1.0 s that does appear sits in the `moe` block's own body, the part its three
children do not cover. So the mechanism is a per-layer issue/wait effect at the MoE boundary rather
than the row-hit accounting proposed above, and the accounting is still the right way to size the
memory — it is the memory *explanation* that the phases do not support. This is a second reading of the
same lever on a later tree, not a re-measurement of the table above; its own memory column reproduces
the arithmetic anyway, 16118 MiB allocated at 148 rows against 18631 at 288 — +2513 MiB, against the
2512 the row size predicts.

The second proposed mechanism does not survive either, and here the source is what says so rather than
a further run. `_issue_chunk` is called **40 times, once a layer, in every arm** — and
`_forward_chunked` calls it once per bound `_chunk_bounds` returns, so 40 calls over 40 layers means
the whole 4096-row batch is **one bound** at either width. One is the floor, so at this chunk there is
nothing left for a wider pool to consolidate: the "fewer and bigger chunks" above cannot happen at
32768 rather than merely not having happened. It did split on the tree the phase table above was read
on, where the same row counted 158 calls — about four bounds a layer — which is a second reason to
read that column as a property of the tree and not of the pool.

The 0.2% is the more interesting number, because the pool *is* the thing it should move. `pool_lru` is
**one LRU arena a card shared by all forty layers** (the same `arena_rows` the memory arithmetic
sizes), so 148 rows is about **3.7 rows a layer** and 288 about 7.2, against the ~54 rows a layer this
chunk stages in both. Both widths are therefore far inside the region where a cyclic sweep of the
layer's working set thrashes the cache, where least-recently-used is the worst replacement policy there
is and capacity buys almost nothing until it spans the whole working set: 140 more rows bought **4 of
2178 misses**. That is what makes the earlier paragraph's causal chain a *memory* sized one and not an
effect one — the ~1.83 GiB is real and `_chunk_bounds`' rule is real, but the pool is not the lever
they are attached to, and buying more rows is not a way to buy fewer staged rows. On a 22528 MiB card
that matters: the +2513 MiB bought −8.1% of a chunk and no misses, which is the trade the 256K
configuration declines for a reason other than the OOM below.

The token column separates by setting and by nothing else. All four ranks print one top-8 a process,
both 148 processes print `[455, 1, 223, 8077, 764, 330, 343, 334]` and both 288 processes print
`[455, 1, 223, 8077, 330, 334, 764, 343]`, so the two same-setting pairs are exact repeats of one
another and the wider pool permutes the **5th–8th** ids without changing the set. `topk` on the logits,
so it is the logits that moved, and if anything the floor here is *stronger* than the indexer probe's:
there two arms with no knob between them disagreed, and here they do not.

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
| —— `attn.indexer` | 2.09 | 2.12 | 2.09 | 2.12 | 8 | |
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
the attention is the compressed path, `attn.compress_kv` 2.12 s over 38 calls, and `attn.indexer`
2.09 s over 8 is **inside** it rather than beside it: `_compress_kv` calls `_compress_topk_idxs` calls
`Indexer.forward`, so those eight calls are a subset of the 38 and the 30 ms between the two rows is
the pooling, the rotary, the fp4 quantize and the cache write. The nesting is visible in the numbers
as a constant offset, and it is the contained row that carries the context term: the warm-up's eight
chunks, whose caches run from 0 to 28672 tokens, average 1.59 s and 1.57 s where the chunk sitting at
32768 pays 2.12 and 2.09 — the same 0.02–0.03 s part of it in both. Every other row of the two tables
agrees to a few percent — `attn.sparse` reads 7.95 against 8.05, because its index row is the same
width wherever in the prompt the chunk is. **Over a 262144-token prefill it is the compressed path that gets more
expensive per chunk and the score pass that stays flat**, and the score pass is the one the
sparse-attention work targets.

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

**That ordering is the 32768 one and it does not survive to 256K.** `attn.compress_kv` is the only row
of this table with a context term — `attn.indexer` is *inside* it and not a second cost beside it, see
below — and across the leg it goes from 7.7% of a chunk to 16.4%, which leaves it and `moe.routed`'s
21.8 s as the two largest things in a 256K chunk — no other row reaches 10 s.
[Below](#the-one-row-that-grows-with-context) is what is inside that one row and which of its levers
are still open.

### The one row that grows with context

`attn.compress_kv` is **2.10 s at 32768 and 5.14 s at 262144** (2.45x) over 38 calls at both lengths,
and the row nested under it — `attn.indexer`, 2.08 and 5.11 s over 8 of those 38 — is **inside** it
rather than beside it: `Attention._compress_kv` calls `_compress_topk_idxs`, and `_compress_topk_idxs`
is what calls `Indexer.forward`. Only 8 of the 38 calls reach the indexer, because a compressor emits a
new row only every `compress_ratio` positions, and on those 8 the child is 99% of the parent: the
0.02–0.03 s between the two rows is the pooling, the rotary, the fp4 quantize and the cache write, the
same gap in both chunks and in every warm-up chunk. **So the context term is one row and its magnitude
is `compress_kv`'s**, and the quiet chunk is **27.39 s at 32768 and 31.34 s at 262144**, with the row
**2.10 s of the first (7.7%) against 5.14 s of the second (16.4%)**. Both chunks are this tree's — the
three prefill kernels and the `id` deal are both in — and both fit the ship: 27.39 s at 32768 is the
sweep's 57.04 s through the three kernels and the `id` deal, and 31.34 s at 262144 sits 5% above that
leg's own last chunk of 29.82 s. Those are the two denominators used below; the table above's 56.96 s is
the same chunk on the branch without the three kernels, and its `attn.sparse` row of 8.05 s is that
tree's score pass rather than this one's 2.05 s.

**The pair this section first gave — 3.19 s against 10.24 s, 11.6% to 32.7% — added a parent row to the
child nested inside it, so it counted the indexer's 2.46x twice.** Neither half is a row of either
table. The 262144 one is a sum of two columns and not exactly the sum of the `total` columns either —
5.14 + 5.11 reads 10.25 against the 10.24 it was written with, so one of the two came off a per-rank
column rather than the maximum — and the 32768 one does not reproduce from the artifact this section
cites at all (`/tmp/chunk_deal_id32.log`, whose own two rows read 2.10 and 2.08 s, a sum of 4.18). The
corrected shares above are what that log and `/tmp/chunk_deal_id256.log` support. (A phase table's
`total` column is a per-rank maximum, so a row read off it is the straggler's; over the four ranks the
two rows mean 2.07 and 2.10 s at 32768 and 5.10 and 5.12 s at 262144. The `2.069` and `5.097` s the
older text called the indexer's row are rank 0's `sync` column — 2.0691 and 5.0972 — which is one of the
two columns the rule below says to read apart, and the arms are quoted against that same 2.069 s.)
Across this leg the one row is most of the growth and every other row is flat, read off the `total`
column of both tables: `attn` goes 6.75 → 9.83 s, of which `compress_kv` is 3.04 of the 3.80 s the
instrumented chunk gains, while `moe.routed` reads 21.48 → 21.77 s and `attn.sparse` 2.11 → 2.16 s, and
`hc_post`, `hc_mixes`, `hc_pre`, `engram` and `norm` are unmoved — while the chunk goes 27.39 → 31.34 s.

**Read the tap's two time columns apart or it will mislead you**, the same rule the phase table above
needs. A wrapper that drains the GPU before each call records the *enqueue* in `body` and the GPU
backlog standing at the boundary in `sync`, so a tap over a collective or a D2H read shows its host
blocking time rather than its kernel's duration. The indexer's own tap is opened below.

Four arms on the same tree priced what is per-tile inside the row:

| arm | tiles moved | `attn.indexer` at 32768 | ratio |
| --- | --- | ---: | ---: |
| `INDEXER_QUERY_TILE` 2048 → 4096 | none — `key_tile` halves as `q_tile` doubles, so the tile's size and its count both stay put | 2.069 → 2.111 | 1.02x |
| `INDEXER_SCORE_BUDGET` 2^26 → 2^28 | `key_tile` 4096 → 16384, i.e. 30 of 1072 tile events (2.8%) | 2.069 → 2.096 | 1.01x |
| both | 30 | 2.069 → 2.066 | 1.00x |
| `INDEXER_CAND_TILE` 64 → 256 | `span` 512 → 2048, so 1072 → 304 tile events (**3.5x**) | **2.069 → 1.871** | **0.90x** |

The counts come from the checkpoint's own layout. `index_source_layer_ids`
`[2, 8, 14, 20, 24, 28, 32, 36]` are the eight indexers and `kv_source_layer_ids` `[2, 8, 14, 20]` the
four that publish an `index_k`; each indexer reads the `index_k` of the nearest source in front of it,
so its width is `end_pos // ratio`, with `compress_ratios` 2 for layers 2–19 and 1 for 20–39.
`candidate_source_layer_id` 20 splits the eight: layers 2/8/14 and 20 run the prefix path, and
24/28/32/36 run the candidate path. At `--at 32768 --chunk 4096` the widths are 18432 and 36864, so

| path | layers | tiling | calls |
| --- | --- | --- | ---: |
| prefix | 2/8/14 | `q_tile` 2048 x `key_tile` 4096 = 2 x 5 | 10 each |
| prefix | 20 | 2 x 9 | 18 |
| candidate | 24/28/32/36 | `q_tile` 512 x (`span` 512 over 16384 keys = 32) | 256 each |

**1072 einsum calls, 96% of them the candidate path** — which is what makes the two prefix knobs the
nulls they are. `INDEXER_SCORE_BUDGET` sizes `key_tile` and so reaches 30 of those 1072 events, a 2.8%
cut, and `INDEXER_QUERY_TILE` doubles `q_tile` and halves `key_tile`, leaving the tile's size and its
count exactly where they were. The FLOPs disagree with the counts: a prefix tile is 17.2 GFLOP and a
candidate one 0.54, so the chunk's 1374 GFLOP splits 825 / 550 between the levels against the counts'
4% / 96%. **Read the two nulls as "at 32768 the prefix path is not the row" — it is 0.525 s of the
2.628 s one, which the level split below measures directly — never as "nothing per-tile is", and read
the row itself as 90% something that is neither the tile count nor the arithmetic** — the one arm that
reaches the candidate tiles cuts them 3.5x and buys 0.198 s of a 2.07 s row.

At 262144 the two widths are 133120 and 266240, so the 48 prefix tiles become **328** and the total
1072 → **1352**; the candidate path's do not move, because its `keys` is `candidate_topk_blocks` 2048
blocks of `candidate_block_size` 8 — 16384 gathered positions a query whatever the context is.
`/tmp/probe_v41_indexer_steps.py` takes that row apart in place, with a host `synchronize()` around
every call so a tap records a body and a sync the way the 22-tap table does:

| one 4096-token chunk at 262144 | body s | calls | µs/call | sync s | worst rank |
| --- | ---: | ---: | ---: | ---: | ---: |
| `indexer` | 5.677 | 8 | 709569.3 | 0.015 | 1 |
| — `push` | 0.309 | 1482 | 208.3 | 0.359 | 2 |
| — `einsum` | 0.113 | 1352 | 83.6 | 1.815 | 2 |
| —— `reduce` | 0.201 | 1352 | 148.5 | 2.381 | 2 |
| — `stream_prefix` | 3.525 | 4 | 881326.6 | 0.001 | 1 |
| — `stream_candidates` | 2.084 | 4 | 520967.0 | 0.000 | 1 |

The 1352 is the probe checking its own geometry against the counts above, and the instrumented 5.677 s
against the uninstrumented 5.097 is the same ~11% the 22-tap instrument costs. **Read the two time
columns apart or this table will mislead you.** The wrapper drains the GPU before every call, so `body`
is the *enqueue* — `reduce`'s 0.201 s over 1352 calls is 148.5 µs of `.float()`, `all_reduce` and
`.to(bf16)` per call and says nothing at all about the collective's duration — and `sync` is the GPU
backlog standing at the boundary plus that call's own kernel. What the bookkeeping does support is

```
indexer body 5.677 = sum(inner bodies) 0.66 + sum(inner syncs) 4.60 + unnamed CPU 0.42
```

so **the row is GPU-bound at 256K: ~4.6 s of GPU against ~1.1 s of host**, and the elementwise that
sits between the taps — a relu, a weights multiply and a head sum over a `[2048, 8, 4096]` bf16 score,
134 MB a tile — lands in its neighbours' sync columns rather than in a row of its own. Read the parts
as bounds and never as a partition.

The level split is the one thing the two lengths disagree about, and the 32768 half of the probe is
what shows it. `stream_prefix` is **0.525 s at 32768 against 3.525 s at 262144** — 48 prefix tiles
against 328 — while `stream_candidates` is **2.035 against 2.084 s** over 1024 tiles both times. So
the candidate path is **2.035 s of the 2.628 s instrumented row at 32768, 77% of it**, and 37% of the
one at 262144, and **nothing in it scales with context**: flat to 2.4%, 1.99 ms a c-iteration against
2.04. An earlier reading here — that 2.084 s "would be more than the entire row at 32768", so
something inside the candidate path must grow with the width — compared an instrumented 262144 number
against an *uninstrumented* 32768 row and landed on the answer it was looking for: the candidate
path's own cost at 32768 is 2.035 s. The L2 hypothesis that reading bought was **unsupported and, at
that point, untested**: the gather out of an `index_k` that is 4 MiB at 32768 (inside this card's 5.5
MiB of L2) against 68 MiB at 262144 (outside it) predicts a per-iteration cost materially lower when
the index fits, and the 2.4% between 2.035 and 2.084 s — 0.049 s over 1024 of them — is the whole of
what that difference is worth.

**The capacity question is now answered by a direct sweep, and the answer is no.** A synthetic
c-iteration — the same geometry, an index built from block ids the way the loop builds it, no
collective and no ranks — prices the pieces at both widths (`/tmp/bench_indexer_cand_tile.py`,
30 iterations, one RTX 2080 Ti):

| µs a c-iteration | width 36864 (`index_k` 9.0 MiB) | width 266240 (65.0 MiB) |
| --- | ---: | ---: |
| gather, scattered | 666.9 | 684.6 |
| einsum over the gathered tile | 690.3 | 703.0 |
| mask | 65.8 | 72.6 |
| merge (cat + topk + gather) | 135.8 | 136.9 |
| amax/amin boolean read | 71.9 | 76.8 |
| **whole** | **1612.4** | **1577.1** |

**A 7.2x change in the width — from under twice this card's 5.5 MiB of L2 to twelve times it — moves
the whole tile by 2.2%**, so the gather is neither capacity- nor residency-bound: 667 against 685 µs
is the access pattern's own price, and the same 64 MiB of gathered rows either way. The einsum costs
the same from a contiguous `[1, q, m, d]` copy as from the gathered tile (690.2 against 689.8 µs at
36864, 616.9 against 618.0 at 266240), so the halves are independent and neither is a layout artifact
of the other. That leaves the 1612 µs against the in-situ **1988 µs** a c-iteration (2.035 s over 1024)
as the collective's 202 µs plus ~175 µs of enqueue the synthetic loop does not pay.

**The einsum's 690 µs is the shape, not the bytes.** `out[q, h, m] = sum_d Q[q, h, d] K[q, m, d]` has
`h` as one operand's only free dimension and `m` as the other's, so it is 512 batches of
`[8, 128] @ [128, 512]` — `M` = the model's 8 index heads — and the same 0.537 GFLOP as a single
`[4096, 128] @ [128, 512]` GEMM costs less than a sixth as much: the batched-to-wide ratio is 6.51,
6.27, 6.53 and 6.15 on four readings of `/tmp/bench_indexer_cand_score.py`, and the ratio is the part
that survives this box's clock bins, so that is the claim — the absolutes in those readings move
together by 25% between an allocation-warm and a warm card (`--preheat` tags a bin; without it a
table here is a reading of the card's state rather than of the shape). `M` cannot be raised — each
query gathers its own rows, so there
is no operand shared across the batch — and transposing the pairing to put the 8 on `N` buys nothing
(672.8 µs against 652.7 at batch 512's shape). Its 64 MiB of input in 690 µs is 97 GB/s, *below* the
gather's own 201 GB/s,
which is the same statement from the bandwidth side: this half is not waiting on memory.

**What that prices.** The shipped pair is 1357 µs a c-iteration and moves ~192 MiB (gather read plus
write, einsum read); a fused gather-and-dot would read the 64 MiB scattered and write 4 MiB, and at the
gather's own measured 201 GB/s that is **~350 µs** — so the fusion's ceiling is a **~1.0 s** cut of the
2.035 s row, and the row is 1024 tiles at both lengths, so the same second is on the 32768 chunk as on
the 262144 one. Nothing is implemented: the number is a bound built from the two measured rates, and
the arithmetic would have to be kept in bf16-input, fp32-accumulate to land on the same `k` values.
Note *values* rather than bytes: this level cannot be bit-identical by construction, and the
candidate stream's own section below measures why. The in-tree precedent for one pass over
gather-score-select is `src/kernels/ops.py`'s `_decode_sparse_attn_kernel`, which already fuses
exactly that for decode.

**Two levers, and what gates each — and each one owns a different regime.** The collective:
`make_all_reduce` upcasts to fp32 around the `all_reduce` and the closure casts the answer back to
bf16 anyway, so a prefix-tile message travels on the wire at **33.6 MB against the 16.8 MB of the
tensor it carries** — ~11 GB a chunk at 262144 against the candidate path's ~1 GB — for a rounding on
a value that is bf16 the moment it leaves the closure. Whether that volume is what the collective
costs is a property of the fabric rather than of the tap, so it is measured directly
(`/tmp/bench_nccl_indexer_shape.py` sends both real shapes in both dtypes on the real PHB/NV2/SYS
topology), and the fabric turns out to be the constant: a `[2048, 4096]` level-one tile is **5469.8 µs
at float32 against 2877.5 µs at bfloat16** — 32 MiB of wire at 6.1 GB/s against 16 MiB at 5.8 GB/s —
and a `[512, 512]` level-two tile is 201.7 against 126.2 µs.

Over the tile counts above that is **0.469 s at 32768 and 2.001 s at 262144** of fp32 wire — 0.263 of
it level one and 0.207 level two at 32768, 1.794 and 0.207 at 262144 — against 0.267 and 1.073 in half
the bytes: **1.794 s of the 3.525 s prefix path at 262144 — 51% of it and 35% of the whole 5.10 s row
— and 0.263 s of the 0.525 s one at 32768.** The prediction is checkable in situ and it checks out on
the one tap column that can see a collective, `reduce`'s `sync`, which drains the GPU backlog standing
at the call boundary: **0.469 s predicted against 0.530 measured at 32768 and 2.001 against 2.381 at
262144, 0.89 and 0.84**, over two lengths whose level-one tile counts differ 6.8×. That ratio is what
makes the extrapolation a measurement rather than arithmetic — one agreement would be luck. (Do not
*add* the sync columns: `einsum`'s 1.104 s at 32768 and its 1.815 s at 262144 are the same backlog
seen from a different boundary.) Parity of the picked ids is the gate on shipping it, because NCCL
sums a ring in the wire dtype and the score's O(600) values carry 8 mantissa bits there — and if the
wire dtype moves at all it should move to fp16 before bf16, which is the same 2 bytes with 10 mantissa
bits. **That gate is closed on the evidence so far:** `INDEXER_REDUCE_BITS=16` moves the selection on
all eight indexer layers, against a baseline whose own disagreement — two arms with no knob moved —
reproduces to the digit across runs, and layer 2 goes from 3083 differing rows of 4096 to all 4096 and
from 621210 differing elements to 1640362. fp16 on the wire is a different function, so the volume of
that collective is a lever the numerics has shut for now.

**The closed gate still paid for the price model, and that is why it was run.** The same probe was run
at 262144 with `--reduce-dtype fp16` (`/tmp/chunk_indexer_steps_262144_fp16.log`) — not as a candidate
but as the one measurement that could falsify the extrapolation, and it lands on it. `stream_prefix`
**3.525 → 2.674 s, −0.851**, against the 0.928 s the microbench predicted for halving level one;
`reduce`'s `sync` 2.381 → 1.551 (−0.830); the whole `indexer` row 5.677 → 4.758 (−0.919); the chunk
**29.74 → 29.14 s**. `stream_candidates`, which the price model says has almost no level-one wire in
it, moves 2.084 → 2.016, −0.068. So the model that puts 1.79 s a chunk on the fp32 collective is right
to within one percent, and the overlap below is a lever on a cost that is now measured from both ends.
Note the row gains 0.919 s where the chunk gains 0.60 s: about a third of the row's collective is
already hidden behind other work at the chunk level, and *that* is the number the overlap's ceiling has
to be read against rather than the whole 4.758 s row.

The overlap is the other one, and it is the larger where the row is. The einsum of tile i+1 and the
reduce of tile i are independent — only `_TopKStream.push`'s D2H read of the score needs the reduce
finished — so issuing tile k's `all_reduce` on a second stream and joining it `depth` tiles later is a
scheduling change with no numerics in it, and the joins are FIFO, which is what keeps the pushed
sequence — and so the selection — *identical* rather than merely equivalent.
`/tmp/bench_indexer_reduce_overlap.py` runs the shipped per-tile arithmetic at these shapes on this
fabric, four ranks, and compares the arms elementwise — `identical True` at every depth:

| ms a tile | width 16384 | width 133120 |
|---|---:|---:|
| floor — the arithmetic with no collective at all | 4.85 | **4.40** |
| serial — the shipped order | 10.31 | **9.83** |
| lookahead, depth 1 | 8.88 | 8.44 |
| lookahead, depth 2 | **7.95** | **6.91** |
| lookahead, depth 4 | 8.07 | 6.98 |
| lookahead, depth 8 | 8.05 | 7.11 |

The serial arm is the floor plus the collective to the tenth of a millisecond (4.40 + 5.43 = 9.83), and
the in-situ prefix tile is 10.75 — the bench is the real loop. **A depth of two hides 2.92 ms of the
5.43 ms collective: 54% of it and 30% of the tile; deeper buys nothing.** That is the ceiling, and it is
half of the whole collective rather than the whole of it — with two streams in flight the pipeline
settles at 6.91 where `max(4.40, 5.43)` = 5.43 would be the floor, so the collective costs about 1.5 ms
a tile more when it runs beside the arithmetic than when it runs alone, because NCCL's kernels want the
same SMs the einsum does on a four-card Turing box.

In situ the same 54% does not survive intact, and the instrument that shows it is
`/tmp/probe_depth_inproc.py` — the arms in *one* process on one set of ranks with the knob rebound and
the state reset between them, so this box's per-load spread cancels by construction. The four-process
A-B-A-B (`/tmp/run_depth_ab.sh`, `0/2/2/0`) cannot resolve it: its first arm came back 29.80 s against
its own setting's 27.48 s, with 0.57 s of the excess in `stream_candidates`, a row no depth can touch.
Six serial arms across two sittings at 32768 read `stream_prefix` at **0.472–0.500 s** — 28 ms of spread
over a whole sitting — against the pipelined arms' **0.355–0.392 s four times, −0.08 to −0.15 s under
every serial arm (17–29%), and 0.490 and 0.501 twice.** `stream_candidates`, the row no depth can touch,
ranges 1.537–1.613 over the same twelve arms and does not separate by depth at all. A
`_ReducePipeline.push` count of 48 against the
serial arms' 0 says the pipeline ran in all six, and the two exceptions are, in both sittings, the one
pipelined arm that immediately follows another pipelined arm — so that difference is a second *state*
and not a guard that failed to fire, and nothing here explains what sets it. That caveat is why the
change is behind a flag rather than in the default. At 16384 the same probe moves the row 0.263 → 0.219 s
over both of its pipelined arms.

Over 328 prefix tiles the in-situ rate is ~2.2 ms a tile rather than the bench's 2.92, so ~0.7 s of the
3.525 s prefix path at 262144; the row-vs-chunk transfer the fp16 arm measured above — 0.919 s of row to
0.60 s of chunk — leaves **~0.46 s of the 29.74 s chunk, 1.6%, not the 1.79 s that hiding the whole
collective would be, and nothing at all in the second state.** At 32768 it is 0.10 s of a 27.2 s chunk:
visible on the row and invisible on the wall. So it is implemented behind
`DEEPSEEK_V41_INDEXER_REDUCE_DEPTH` (`attention.py`'s `_ReducePipeline`, **default 0 — the shipped
order**), which returns the serial order inside a capture and whenever `tp is None`; moving it into the
default needs a chunk-level demonstration this box has not given. The retile is the third and it owns 32768: the 0.198 s
`INDEXER_CAND_TILE` 64 → 256 buys is 9.6% of the row 2.07 s *because* the candidate path is 77% of it
there, and the same lever at 262144 is 0.198 s of a 5.10 s row, **3.9%**, because that path does not
grow with the width.

**The candidate stream's own early-out is the fourth lever, and it is the one that ships off.**
`_TopKStream.push` returns without merging when `amax(tile) < amin(held)` and the buffer is already at
width `k`, which is the test the prefix level wants: there a 4096-key tile is narrowed into a 512-wide
buffer and most tiles of most query tiles are below the running k-th. This level builds its stream with
`k = min(index_topk, width)` — 512, which is exactly its own `span` — so the buffer holds the top-k's
own width from its *first* push and the running k-th value sits inside the incoming tiles rather than
above them. The guard is live, but the test is a device→host read on every push and a hit is worth one
merge; that is why the default now answers `False` (`INDEXER_CAND_SKIP_TEST`), and it is also what a
capture already builds — `_TopKStream` refuses that read inside one — so the eager path is being made
to agree with the recorded one rather than to differ from it.

`/tmp/bench_indexer_cand_overlap.py` prices the read on the real fabric, four ranks, both cache widths,
A-B-A-B with the read as one factor and the prefix path's depth-1 lookahead as the other (µs a
c-iteration, 255 pushes a stream):

| µs a c-iteration | width 16384 | width 131072 |
| --- | ---: | ---: |
| serial, read on (what shipped) | 1523.7 | 1543.2 |
| serial, read off | **1350.1** | **1381.3** |
| lookahead depth 1, read on | 1583.2 | 1609.8 |
| lookahead depth 1, read off | **1200.2** | **1233.5** |

Every arm is elementwise identical to every other arm and every arm reports `reads 255 skips 0
merges 255`. Two things are in that table. The read is 173.6 and 161.9 µs of the shipped tile — 11.4%
and 10.5% — and with it gone the depth-1 lookahead, which *costs* 3.9% against the same read-on serial
arm, becomes a **21% saving**. They are the same effect: the read drains the compute stream at every
push, and that is precisely what stops `_ReducePipeline` from deferring this path's collective, so
dropping the read is the precondition for the overlap here as well as its own 11%.

In situ the same knob is worth less than the fabric says, and the instrument is
`/tmp/probe_cand_skip_inproc.py --at 8192 --chunk 4096 --arms 0 1 1 0`: one process, the state reset
between arms, `INDEXER_REDUCE_DEPTH` held at 0 so this stays one lever, and `_TopKStream.push` wrapped
to count pushes and early-outs per path. The A-B-A-B order is what makes the column readable — the two
`skip 0` arms are the determinism floor, and a `1`-versus-`0` difference means nothing until they
agree.

| arm | `INDEXER_CAND_SKIP_TEST` | chunk s | `stream_candidates` | cand push/skip | logit max\|δ\| vs arm 0 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0 | 25.08 | 1.102 | 896/0 | 0 |
| 1 | 1 | — | 1.284 | 896/1 | 1.695 |
| 2 | 1 | — | 1.236 | 896/1 | 2.206 |
| 3 | 0 | — | 1.137 | 896/0 | 1.695 |

**The timing agrees in sign with the fabric and the parity column does not survive its own floor.**
`stream_candidates` is 1.102 and 1.137 s with the read gone against 1.284 and 1.236 s with it, a
~0.14 s move of a ~1.12 s row on a 25.08 s chunk — 0.56% of the wall — and the guard fires **once in
896 pushes**, so the test is not dead here, it is merely worth one merge. The logits, which were meant
to be the exactness evidence, cannot carry it: arms 0 and 3 are the same setting with no knob between
them and they disagree by 1.695, exactly what arm 1 came back with, while arm 2 came back 2.206 — so
every pairwise comparison among the four differs and the column attributes nothing to either setting.
The leading explanation is the prefill MoE epilogue's plain `atomicAdd`
(`moe_fp4_grouped_w2_wmma_scatter_kernel`, `src/csrc/cuda_kernel_impl.cu`) — the deterministic-reduce
default covers the single-token and multi-slot paths only, so the grouped prefill path accumulates
its routed output in whatever order the blocks reach the accumulator — which is why
`/tmp/probe_v41_prefill_moe_order.py` exists to price it. The exactness rests instead on the argument,
on `tests/test_models_deepseek_v4_1_attention.py`'s two streams, and on
`/tmp/check_cand_guard_equiv.py`'s six trials at this level's geometry.

**What the argument is, and how far it reaches.** Every value held is *strictly* above every value in
the tile, so the k largest of the union are the k the buffer already has: the merge would return the
same multiset, and the skip is exact rather than a tie-break. It is exact in values and no further —
*which* member of an equal-valued group gets named is `torch.topk`'s choice, so a change that removes
merges can move a named position among entries the level scored identically, and only at the pushes
where the skip would have fired. The same freedom is already in the shipped code across any change of
tiling: at this level's real arithmetic — `relu(q·k)` times a weight, summed over the 8 heads, left in
bf16 — the k-th value is shared by 23 entries of a 12288-candidate union in the sample checked, so the
boundary is routinely an equal group. Nothing here can be bit-identical by construction, and the
selection is value-exact.

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
