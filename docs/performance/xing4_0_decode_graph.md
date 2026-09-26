# Xing4.0-29B-A4B: the decode step, captured a bucket at a time

[The launch-count probe](xing4_0_decode_launch_gap.md) measured the ceiling and left the two things a
decode *loop* needs. A whole-step graph at a **frozen position** ran in 37.9 ms against an eager 178.6,
bit-identical — but a capture bakes in every Python value it reads, and a decode step reads a position
in four places: the row the cache writes, the rotary table's row, the causal mask's offset, and the
attention's read width.

**This page is the loop, and three things it establishes.** The position reaches the card as an index
tensor in all four places; the read width is frozen by rounding it up to a power-of-two **bucket** and
masking the rows past the position; and the result is **a replayed step at 38.6 ms against an eager
148.1 ms — 3.84×** at a 4,096-token context, with the *bucket itself* costing nothing (147.5 against
148.1 ms, 1.00×) and the capture's own parity exact to the bit.

**A fourth is a caveat rather than a result, and it is the reason this stage does not serve anything
yet.** Freezing `N` at a bucket means an eager step reads a *different width* than a bucketed one, and a
wider `N` re-associates the two reductions the attention does over it: `max |dlogits|` is up to 0.4375
against the unbucketed path, and 0 of 224 sampled positions changed the token. The capture is exact
*against its own width*; against today's eager step it is a behaviour change, small and measured, and it
is why wiring this into `pocketllm/backends/xing4_backend.py` is its own change with its own acceptance.

| | |
|---|---|
| Hardware | 1 x RTX 2080 Ti (sm_75), `--device cuda:2` |
| Checkpoint | `/mnt/data2/Xing4.0-29B-A4B-GGUF/xing4_0-29b-IQ4_NL.gguf` (18.72 GiB, 17.84 GiB resident) + the release directory's tokenizer and `config.json` |
| Context | **4,096 tokens** for the step table; 512 / 4,096 / 16,384 / 32,768 for the end-to-end table. Real prose, greedy, one request, 32 generated tokens |
| Commit | `perf/xing4-0-decode-buckets`, stacked on `fix/xing4-0-prefill-decode-clock` (PR #430) so the rates are read on the corrected side of the prefill/decode seam |
| Question | [#427][issue], stage 2 of [the approved plan](../architecture/xing4_0_29b_a4b_design.md#6-where-a-decode-steps-time-goes): does the ceiling hold for a loop, and what does the bucket cost |
| Run | 2026-09-26 |

## The position, in the two forms it is read in

`src/models/xing4_0/decode_pos.py` is one number in two spellings, and the split is forced rather than
stylistic. A slice's bounds are Python values at *record* time, so `latent[:, start_pos:end]`,
`arange(start_pos, start_pos + seq)` and `k_pos > start_pos` each freeze the position they were
recorded at. What a capture can hold is an index tensor.

| Read | Host path | Captured path | Why the second |
| --- | --- | --- | --- |
| the row the cache writes | `latent[:, p] = rows` | `index_copy_(1, [p], rows)` | `at::indexing` reads a 0-dim integer tensor back to the host to form the copy, and a capture forbids the read — `cudaErrorStreamCaptureUnsupported`, reported a step later as `StreamCaptureInvalidated`. A 1-element 1-d tensor is an *index tensor* and lowers to a kernel. |
| the rotary table's row | `arange(p, p + 1)` | `arange(1) + p` | the bounds of the first are Python values. |
| the causal mask's offset | `k_pos > p` | `k_pos > p` (0-dim tensor) | the comparison broadcasts either way; only the tensor survives a capture. |
| the read width | the cache's length | the bucket | **this is the hard one** — see below. |

The first three are mechanical, and `tests/test_xing4_0_decode_pos.py` pins them on the values *and
through a real CUDA capture*, including the refusal above as a negative control: the naive slice
spelling is recorded and asserted to fail, so `write_row`'s reshape cannot later be simplified away as a
style preference.

## The bucket: `N` rounded up, and the rows past the position masked

The fourth read is not mechanical, because the width is the attention's `N`: the second dimension of the
score GEMM and the width of the softmax. **No index tensor can make it vary** — a shape is as frozen in a
recording as a slice bound is.

So the width is rounded up to the next rung of a **ladder** — the powers of two from 64, then the cache's
capacity itself — and the rows past the position are masked to `-inf`. A step at position 4,097 reads
8,192 rows and hides 4,095 of them. Which rung runs is decided on the host, before the launch, exactly
where V4.1 chooses between its compressor's two bodies: a recording cannot decide that, and a reading
that could would not be a recording.

| Context | Rung | Rows read | Waste |
| ---: | ---: | ---: | ---: |
| 512 | 1,024 | 1,024 | 2.00× |
| 4,096 (capacity 8,192) | 8,192 | 8,192 | 2.00× |
| 16,384 | 32,768 | 32,768 | 2.00× |
| 32,768 (capacity 32,808) | 32,808 | 32,808 | **1.00×** |

**The tail rung is the capacity and not the next power of two above it**, which matters more than it
looks: a slice cannot read past the buffer, and at this checkpoint's own limit it is what makes the waste
vanish. A 32,768-token context sized to the 32,800 positions the run needs has its ladder end at 32,808,
so its last steps read 1.00× the rows they need — the worst case is 2×, at every context below the top of
the ladder.

`deepseek_v4_1`'s `Pos.upto` solves the same problem the other way: it reads the *whole* cache on every
step, so its `N` is one constant for the life of the run. The trade is symmetric — a constant `N` costs a
32K context the whole 32K rows a token, a ladder costs a recording a rung — and this model takes the
ladder because the eager path is what a server runs and it should not read rows a step does not need.
The two designs coincide at the top of the ladder.

## The three arms, in one process

The end-to-end bench puts each arm in its own process, which is right for a rate and wrong for a small
lever: this host's own spread across processes is 138 to 178 ms on the same step, so a comparison of two
arms that differ by 1% has to be taken with the arms alternating in one process. `eager` is the forward as
it always ran, `bucket` the same forward reading a rung, `graph` the captured step replayed. Each arm gets
its **own cache** — all three write the row their position names, and two of them do not write the same
bytes, so a shared cache would let whichever arm ran last decide what the next step of the others reads.

| Arm | median | min | max | tok/s | vs eager |
| --- | ---: | ---: | ---: | ---: | ---: |
| eager | 148.1 ms | 138.3 | 156.4 | 6.75 | 1.00× |
| bucket | 147.5 ms | 138.8 | 158.8 | 6.78 | **1.00×** |
| **graph** | **38.6 ms** | 38.4 | 39.9 | **25.91** | **3.84×** |

48 steps an arm, interleaved and rotated so a monotone drift in the host lands on all three, at a
4,096-token context with an 8,192-position cache, so the bucket reads 2.00× the rows it needs — the worst
case the ladder admits.

**The bucket is free and the recording is the whole of the win.** 147.5 against 148.1 ms is inside the
spread; the same script at the same settings on the run before this one gives 134.4 against 134.9. That is
worth stating plainly because "read twice the rows" sounds like it should cost something, and on this path
it does not: the attention is a small share of a step that is otherwise 40 blocks of MoE and
hyper-connection, and the wider read is a longer `N` in two GEMMs rather than a second pass over the
cache.

**And the graph's row is the one the frozen probe promised.** 38.6 ms here against 38.0 for the captured
step at a *frozen* position, and 37.9 on the stage-1 page — three different harnesses, one number. What
stage 2 added to it is a loop: the recorded step is replayed at 48 different positions and the logits
follow the loop.

## The two parities, and the one that is not exact

Capturing is faithful, and the bucket is not bit-exact — the two are different claims and both are
measured.

**The replay is exact.** `holder.step_eager` submits the same step at the same rung one launch at a time,
which isolates the *capture* from the *bucket*: two arms that read the same width and differ only in how
their ops reach the card. `max |dlogits| = 0.000e+00`, element for element across all 131,072 of them.
`step_eager` exists for exactly this comparison, and it is the oracle a served run's disagreement should
be settled with.

**The bucket moves the logits, and the size is stated rather than bounded away.** The rows past the
position are masked to exactly zero probability, so what is left is float **re-association**: the
softmax's sum and the value contraction both reduce over `N`, and a wider `N` groups the same terms
differently. Measured on the released checkpoint at a 4,096-token context and a 2.00× rung, over 16
positions with a fresh cache a position: `max |dlogits| = 0.4375`, 16 of 16 positions differ, and **0 of
16 change the argmax**. The control — the bucket spelling at a width equal to the position, which is the
unbucketed width — is `0.000e+00`, so the harness is comparing the width and not something else.

Two things about that 0.4375 that a reader should not have to rediscover:

- **It does not depend on how much wider the read is.** A depth sweep at a 200-token context gives the
  same `max |dlogits|` at a 1.27× rung and a 2.55× one, at every depth: 0.0625 at 2 blocks, 0.0625 at 8,
  0.414 at 16, 0.789 at 40. The extra terms are exact zeros, so their *count* cannot matter; what changes
  is which kernel the wider `N` selects and therefore how the sum is grouped.
- **It grows with depth, and it never changed a token.** 0 argmax flips across all 224 sampled positions
  in that sweep, 0 in the 16 above, and the four end-to-end runs below produce the same 33-token greedy
  answer in both arms at every context. That is the evidence for saying the bucket is safe to serve on
  this checkpoint; it is not a proof, and a path that switched a served deployment from the unbucketed
  step to a bucketed one would be a behaviour change and not a bit-exact substitution.

## End to end

The same bench the model page's table comes from, one arm a process, at the four contexts. The answer
column is a sha256 of the greedy token ids, first 12 hex — the two arms are supposed to produce the same
tokens and a rate table is where that would otherwise go unnoticed.

**One arm a process is forced here and not chosen.** Three caches would be 4.2 GiB at a 32,768-token
context against the 3.4 GiB this card has free, so the arms cannot share a process at the lengths that
matter — which is why the *paired* comparison is the step table above and this one is the rate a client
would see.

| Prompt | Arm | Prefill | Decode, steady | First step | Rung | Rows read |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 512 | eager | 81.58 tok/s (6.28 s) | 6.61 tok/s (151.3 ms) | 169 ms | — | 513 |
| 512 | bucket | 77.26 tok/s (6.63 s) | 6.86 tok/s (145.8 ms) | 157 ms | 1,024 | 2.00× |
| 512 | graph | 80.81 tok/s (6.34 s) | **27.09 tok/s (36.9 ms)** | 618 ms | 1,024 | 2.00× |
| 4,096 | eager | 79.38 tok/s (51.6 s) | 6.59 tok/s (151.8 ms) | 157 ms | — | 4,097 |
| 4,096 | bucket | 74.93 tok/s (54.7 s) | 6.45 tok/s (155.1 ms) | 155 ms | 8,192 | 2.00× |
| 4,096 | graph | 78.51 tok/s (52.2 s) | **24.83 tok/s (40.3 ms)** | 583 ms | 8,192 | 2.00× |
| 16,384 | eager | 69.53 tok/s (235.6 s) | 6.57 tok/s (152.1 ms) | 152 ms | — | 16,385 |
| 16,384 | graph | 69.38 tok/s (236.1 s) | **20.61 tok/s (48.5 ms)** | 573 ms | 32,768 | 2.00× |
| 32,768 | eager | 64.36 tok/s (509.1 s) | 6.90 tok/s (144.9 ms) | 146 ms | — | 32,769 |
| 32,768 | graph | 64.45 tok/s (508.4 s) | **20.73 tok/s (48.2 ms)** | 567 ms | 32,808 | 1.00× |

**Every digest matches its own length across the three arms**, in the order the table is in:
`83c9e4c38f51`, `08ae3c7b4c6e`, `0a78dafb8b76`, `f7b24ebb927e`.

| Prompt | Decode was | Decode is | Speedup |
| ---: | ---: | ---: | ---: |
| 512 | 6.61 tok/s | **27.09** | 4.10× |
| 4,096 | 6.59 tok/s | **24.83** | 3.77× |
| 16,384 | 6.57 tok/s | **20.61** | 3.14× |
| 32,768 | 6.90 tok/s | **20.73** | 3.00× |

**The graph turns the flattest curve in this repository into one that falls with context, and the fall is
the card's.** Unbucketed, decode is flat to 0.6% from 512 to 32,768 tokens — the absorbed cache reads 576
values a layer a token however much is behind it. Replayed, it drops 27.09 → 20.73 tok/s across the same
range, because the step is now the *device* work and the device work includes the attention's read: 36.9
ms at a 1,024-row rung, 40.3 at 8,192, 48.2 at 32,808. The bucket's read width is what the middle of that
range is made of; the endpoints differ by 32× in rows and 31% in time, which is the price of a graph
having to name its `N` at all.

**The host is now 4% of the step, and that is the whole point.** One `replay()` submission costs
**1.7 ms** — flat at 1.7, 1.7, 1.7 and 1.8 ms across the four contexts — against the ~148 ms the same
step cost the host when it was launched one op at a time. The card is what the step is, so this is the
number the *next* stage has to move, and no amount of submission-side work can move it.

**The first step is where the captures go, and they are paid once.** 567–618 ms against the eager arm's
146–169 ms, and 0.6 s of that is the recording: a rung is four real forwards at its width — one to
allocate, `CAPTURE_WARMUP` on a side stream, one recorded — and they land in the first decode step of the
first request that needs them, which is what `reserve` is for. A run at 32,768 tokens reaches four rungs
(1,024, 8,192, 32,768, 32,808) for 2.3 s of capture and **26.0 MiB of pool in total**, against a 24.0 MiB
pool for one rung of the forty-block step on the stage-1 page. Rungs are recorded once and live as long as
the holder, so a served process pays this on its first request and no request after it.

## What the graph does not fix

- **The 46 ms of device work a step costs is now ~all of it.** The step is 100% device-busy where it was
  26%, and what is left is the same 40 blocks of arithmetic. That is the number
  [stage 3 of the plan](../architecture/xing4_0_29b_a4b_design.md#6-where-a-decode-steps-time-goes) — the
  casts, the four norms a block and the router's per-step `float()` — and §4's decode attention have to
  be priced against.
- **The replay still submits ~5,300 kernels to the device.** A graph removes the *host* submission, not
  the launches: `cudaGraphLaunch` hands the same 5,577 kernel instances to the card, and at ~1.3 µs of
  device-side launch each that is a floor no kernel-level work can go below. It is why the hyper-connection
  fusion removed 5,800 host launches and was still worth 2.17× on that block.
- **Nothing is served yet.** `pocketllm/backends/xing4_backend.py` still calls `generate` without a
  `decode_step`, so a served request runs the eager path and gets the eager rate. Wiring the holder in is
  a small change — one `DecodeGraphs` against the process's cache at load, one argument in `_loop` — and
  it is deliberately not in this change: the bucket's logits are not bit-identical to the unbucketed
  path, the adapter has a prefix cache and a cancel path that the bench does not exercise, and those want
  their own acceptance rather than a footnote here.

## Run record

| Claim | Command |
| --- | --- |
| The three arms, the rung, the pool, the two parities | `scripts/step_arms_xing4.py --device cuda:2 --context 4096 --steps 24 --rounds 2 --capacity 8192` |
| The end-to-end table and the answer digests | `scripts/bench_xing4_0_e2e.py --device cuda:2 --decode {eager,bucket,graph} --lengths 512,4096,16384,32768 --decode-steps 32 --max-model-len 32768` |
| The bucket's delta against depth, and the 0.4375 | a scratch sweep over `Xing4_0GGUFModel(..., block_count=n)` for n in 2/4/8/16/40, transcribed above: each depth run at a 1.27× and a 2.55× rung over 16 positions, one fresh cache a position |
| The position's two spellings, and the capture's refusal | `python -m pytest tests/test_xing4_0_decode_pos.py -q` |
| The parities, the ladder and the lifetime | `python -m pytest tests/test_xing4_0_decode_graph.py -q` |

Three things about the numbers a reader should not have to reconstruct. **The eager column is a host
measurement and it moves; the graph column is the card's and does not.** Two runs of the step script give
graph 38.4 and 38.6 ms while eager moved from 134.9 to 148.1 — the in-process table above is the run the
command in this table reproduces, and its thirds are the figures to quote. The end-to-end ratios carry the
host's *cross-process* spread on top of that, which is why the 4.10× at 512 is not the 3.84× the paired
run gives. **The prefill column carries that spread too**, and
its 512 row is the worst of it: 84.39 and 79.15 were the same command twice on the seam record, and the
77.26–81.58 here is a third and fourth reading rather than a change. **And the 32,768-token eager row is
the fast end of the spread** (6.90 against 6.57 at 16,384 on a step that should not get faster with
context), which makes its 3.00× the conservative end of the four ratios; the seam record documents the
same 6.76-to-7.06 swing for that row.

[issue]: https://github.com/lvyufeng/PocketLLM/issues/427
