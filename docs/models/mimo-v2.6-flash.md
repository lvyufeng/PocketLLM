# MiMo-V2.6-Flash

## Runtime status

**Heterogeneous, four ranks, one sequence.** PocketLLM opens the released
checkpoint, derives the expert layout from the shards' own headers, runs the full
48-layer text backbone on the host as a reference, and **runs that same backbone on
one RTX 2080 Ti**: the dense stack and the attention out of the released FP8
weights, the routed experts out of a host-resident bank, one token a step through a
KV cache, to logits. A token costs **610 ms** — 1.64 tokens a second — of which
74% is the expert copy, and the argmax agrees with the float32 host reference at
full depth.

**And the same model on four cards**, with the experts dealt out: a rank owns a
quarter of them, the router stays replicated, and a routed layer's partial is summed
with one 16 KiB all-reduce. A token is then **275 ms — 3.63 tokens a second**, four
ranks produce byte-identical logits, and the decode is the same nine tokens the
one-rank run gives. The attention was still replicated at that point, on all four ranks; it
is divided now, along the checkpoint's own four-way partition, and the section on it is
below. That 275 is this stage's number and not
the page's: two device-to-host round trips a layer that nothing needed were 58 ms of it, and the
step is **177.6 ms — 5.63 tokens a second** below.

**A prompt is a chunk now, and not a token loop.** `forward_chunk` takes a chunk of
tokens through `moe_multi_token_fp4_forward` — the kernel the V4.1 path already called
and this one did not — whose arena holds a rank's *share* of a layer's experts and
whose pairs are the chunk's drawings grouped by expert, and `prefill` runs a prompt in
chunks of the caller's width. On four ranks a 4096-token prompt goes through at **134
tokens a second at a 1024-token chunk and 174 at 2048**, against **2.93** for the same
prompt fed one token at a time: 46 and 59 times the rate on the same weights, with the
four ranks byte-identical on the prompt's last row. The width is the knob, and what
capped it was the attention's block loop rather than the link — a 4096-token chunk ran
out of card, and the next paragraph is what removed that.

**The width that ran out of card now fits, and 256k runs.** The block loop's tile was
its score block, `[kv_heads, groups, rows, block]` float32, whose width is the *chunk*
and which is 1 GiB at 4096 rows; the chunk's rows are now split into steps sized to a
fixed tile budget, which is a rounding-order difference and not an approximation. With
that, a **262144-token prompt runs end to end on four ranks at a 2048-token chunk** —
104.04 tokens a second, 10.21 GiB on the card, the four ranks' last row byte-identical —
and a 64k prompt runs at **104.4 tokens a second at a 4096-token chunk**, against 95.5 at
2048 and 3.0 fed one token at a time. The bound is also what makes the 256k row below a
real run rather than a cache that was sized for one. The 256k rate is the attention split's
and not this stage's: the same prompt with the attention replicated on every rank is
**48.37 tokens a second** and 18.71 GiB on the card, and the two arms are a table apart in
the 256k section below.

**Decode is 5.6 tokens a second at a short context** on four ranks, which is the same
step that measured 3.63 before — the difference is two device-to-host round trips a
layer that were pure validation. `_check_bounds` reads `upper.max()` and
`(upper < lower).any()` back to the host to check the caller's bounds, and the model
builds those bounds itself two functions above it, in Python, where both facts are
already known. 48 layers × 2 round trips were **96 of a token's 145
`cudaStreamSynchronize` calls**, each one stalling the host on the kernel it had just
queued; the block loop's two `searchsorted`s per key block were another 128 a layer at
64k. Removing the first and batching the second is 3.63 to **5.63 tokens a second** at
eight tokens of context, and the decode's own `attn` column is **91.9 to 62.2 ms a token**
at 64k.

**And it is a service.** `pocketllm serve --backend mimo` is an OpenAI-compatible
endpoint on the same four ranks: `/health`, `/ready`, `/v1/models`, chat completions,
`/v1/completions`, SSE streaming, a cancel that reaches a running loop through a
collective the ranks agree on, and `/metrics` carrying the arena and the deal. Rank 0
hands every request to the workers over a broadcast before it runs it, because every
routed layer closes with an all-reduce and a rank that was not told about a request is
not idle — it is at a different collective. A four-rank served run is in this page's
own section.

What that is not: a kernel, batching, or a sampler worth shipping. The attention and
the dense linears are still torch; the served path is one request, one sequence; and a
256k decode step is 323.8 ms, most of it the attention reading a quarter of a million keys
nine times over — once a rank, since this stage divides that work along the checkpoint's
own four-way partition, which takes the same step to **197.2 ms and 5.07 tokens a second**,
past the five a second this page was written against. What *is* gone from the earlier
list is 256k, serving, the replicated attention and the attention's own copy of its prefix;
the numbers are below.

What exists:

| Capability | State |
| --- | --- |
| Checkpoint headers, expert layout, dense-key inventory | Implemented and tested against the release |
| Per-layer CPU parity against the checkpoint's own remote code | Implemented, 21 tests |
| MXFP4 / FP8-block / BF16 dequantizers | Implemented as torch references |
| Full 48-layer backbone on the release, on the host | Implemented and verified: 2.09 nats/token on an English passage against a uniform floor of 11.94, with grammatical greedy continuations |
| Host-resident expert bank (149.81 GiB, one shared segment) | Implemented; fills from the release in 12.0 min at 213 MiB/s |
| Device (CUDA) routed experts, one token, out of the bank | Implemented and verified against the host reference |
| Device attention, both families, with a KV cache | Implemented in torch and verified against the host reference; **not a kernel** |
| Device dense stack and the model loop | Implemented: 48 layers, a KV cache, greedy decode, on one card |
| End-to-end device decode on the release | Verified against the host reference at full depth and measured: 1.64 tok/s, 610 ms a token |
| Router reached from C++ instead of from Python | Implemented and held to `torch.equal` against `layers.gate_and_route`, over both groupings: **190.2 us a call against 498.5** and 8.9 ms of a token's host time against 23.4 |
| Expert parallelism over four ranks | Implemented and verified: 5.63 tok/s, 177.6 ms a token at a short context, four ranks byte-identical and the same tokens as one rank |
| Attention parallelism over the same ranks | Implemented and verified: the checkpoint's own four-way `qkv_proj` partition, joined by an all-gather held to `0.00e+00` against the whole path — **2.15x end to end on a 262144-token prompt**, 4.65x on a decode step at that depth, the KV cache divided the same way |
| Chunked prefill, grouped multi-token expert kernel | Implemented and verified: 134 tok/s at a 1024-token chunk and 174 at 2048 on four ranks, 46-59x the token-at-a-time loop, four ranks byte-identical |
| 256k context | Verified and measured: a 262144-token prompt through four ranks at **104.04 tok/s**, four-identical last row, 10.21 GiB on the card against 22 — the same prompt with the attention replicated is 48.37, so 2.15x of it is the split. A decode step at that depth is **197.2 ms — 5.07 tokens a second**, 38.8 of it the attention and 99.9 the expert copy |
| OpenAI-compatible serving | Implemented and exercised on four ranks: chat, completions, streaming, cancel, metrics |
| Batching, a scheduler, a sampler | Not implemented — one request at a time, `argmax` unless a temperature is given |
| MTP (3 layers) and the DFlash drafter | Located and described; not executed |
| Vision tower, audio encoders | Out of scope |

## Model specification

| Field | Value |
| --- | ---: |
| Layers | 48 (9 global-attention, 39 sliding-window) |
| Global-attention layers | 0, 5, 11, 17, 23, 29, 35, 41, 47 |
| Hidden size | 4096 |
| Vocabulary | 152,576 |
| Query heads | 64 |
| KV heads | 4 global / 8 sliding-window |
| Head dimension | 192 |
| Value head dimension | 128 |
| RoPE dimension | 64 (`partial_rotary_factor` 0.334) |
| RoPE base | 1e7 global / 1e4 sliding-window |
| Sliding window | 128 |
| Attention value scale | 0.707 |
| Attention sink bias | Sliding-window family only, per query head |
| Fused `qkv_proj` width | 13,568 global / 14,848 sliding-window |
| Fused `qkv_proj` row order | Four tensor-parallel shards of `[q \| k \| v]` |
| `o_proj` width | 8192 → 4096 |
| Dense layer | Layer 0 only, FFN intermediate 16,384 |
| Routed layers | 47, 256 experts, top-8, `moe_intermediate_size` 2048 |
| Router | sigmoid scoring, `noaux_tc`, `n_group` 1, `topk_group` 1, weights renormalised, no shared experts |

The two attention families are not a tuning difference: the fused projection's
output width, the KV head count and the RoPE base all change with the pattern, and
a reader that derives one width and reuses it gets a plausible tensor of the wrong
shape.

The fused projection's *row order* is not in the config and is the same kind of
trap. Its output is one tensor holding query, key and value concatenated, and
every candidate order has the right shape, so a wrong one reads a plausible
tensor from the right place and only shows up in the logits — the release's own
`modeling_mimo_v2.py` splits it as one run of `[q | k | v]`, which is *not* how
the released weights are stored. They are four tensor-parallel shards of
`[q | k | v]`, which is the order the serving stack requires (its loader refuses
to run at any attention tensor-parallel size but 4) and the order the weights
measure: the row-magnitude profile turns over once every 3392 rows on a global
layer and every 3712 on a windowed one, exactly one quarter of the tensor each,
and the low stretch is the layer's value block.

The FP8 scale of that projection follows the same sharding. On a global-attention
layer `self_attn.qkv_proj.weight_scale_inv` has **108 rows** for a weight with 106
row-blocks of 128, while every sliding-window layer matches exactly (116 for
116). The 108 is 4 × 27: the projection was quantised one shard at a time, so its
tiles restart at every shard boundary rather than running across the whole
weight. A global layer's shard is 3392 rows — 26.5 tiles, which is why the two
readings disagree — and a sliding-window layer's is 3712, exactly 29, which is
why only the global layers move. Reading the scale as one run of 106 tiles pulls
each shard's first 128 rows, the head of its query block, onto the previous
shard's last tile, which covers its small value block. `loader.py` passes
`QKV_SHARDS` for this one weight; nothing else in the checkpoint is affected.

### Weight formats

| Where | Format |
| --- | --- |
| Routed experts | MXFP4: `[N, K/2]` uint8 holding two E2M1 codes per byte, one E8M0 byte per 32 input columns |
| Dense linears except `o_proj` | FP8 E4M3 under 128×128 tile scales, tile-normalised (`w = w_fp8 * scale`) |
| `o_proj`, norms, router, embedding, head | BF16 |

The expert layout is contiguous and verified rather than assumed: `ep{N}` holds
experts `4N..4N+3` for all 47 routed layers, every non-expert tensor is in `ep0`,
and one expert is a single contiguous 12.75 MiB run in the order
`down_proj.weight, down_proj.weight_scale, gate_proj.*, up_proj.*`.

Both fused-projection traps above were found the same way, by disbelieving a
plausible number. With the row order read contiguously — the release's own
reading — the backbone assigns an English passage **13.51** nats/token, worse
than the 11.94 a uniform distribution over the vocabulary costs. Reading the rows
as four shards but leaving the scale as one run of tiles gives **10.10**: better
than the floor and still not a language model. Both together give **2.09**, and
`The capital of France is Paris, and the capital of Japan is` puts `' Tokyo'` at
rank 0 with a logit of 19.63. Neither fix is visible in a tensor's shape, and a
test that only compares shapes passes on all three readings.

## Implemented execution path

`src/models/mimo_v2/` is the whole text model: a host reference that runs on the
release, and the device path built next to it.

| Module | What it is |
| --- | --- |
| `config.py` | The schema and the derived geometry: `qkv_out`, per-family KV heads, RoPE dimension and base, window, sink, value scale |
| `layers.py` | The host implementation of one decoder layer and of the backbone |
| `quant.py` | The three storage layouts as torch references (E2M1 codebook, E8M0, MXFP4 unpack, FP8 block dequant) |
| `loader.py` | Header-level access to the release: the expert map, byte ranges, dense tensors, MXFP4 views |
| `weights.py` | The bridge from the shards into `layers.py`, including experts that stay packed until selected |
| `bank.py` | The routed experts resident in host memory, one shared segment, filled once per boot |
| `device_experts.py` | One layer's routed experts computed on a card, staged from the bank as they are drawn: one token's draw, or a chunk's pairs grouped by expert |
| `device_attention.py` | One layer's attention on a card, both families, against a per-layer KV cache |
| `device_model.py` | The backbone on a card: embedding, forty-eight layers, final norm, head, a token a step and a prompt a chunk |

The routed experts are never expanded. A layer's 256 experts are 3.2 GiB dense
and the model's would be 4.7 TB; `MimoV2Mxfp4Experts` holds the checkpoint's own
uint8 views and dequantizes the eight experts a token actually selects.

## Heterogeneous execution

The experts do not fit on the cards and never will: 47 routed layers hold
47 × 256 experts of 12.75 MiB, which is **149.81 GiB**, and four RTX 2080 Ti hold
88 GiB between them. The path taken here is the one DeepSeek-V4.1-Flash already
takes — keep the experts in host memory, where all of them fit, and move only the
ones a token draws to the card that computes them.

`bank.py` is one POSIX shared-memory segment holding every routed layer's experts,
filled once per boot of the host and read in place by the DMA engines. The layout
a bank needs is already verified from the shards' own headers: shard `ep{N}` owns
experts `4N..4N+3` of every routed layer and those four are one contiguous run, so
a layer is 64 reads of 51 MiB and the checkpoint is 3008 reads. Measured on the
release: **149.81 GiB in 12.0 minutes at 213 MiB/s**, and `cudaHostRegister` over
the whole segment returns 0 in **9.0 s** — after which a `non_blocking` H2D reads
the bank's own pages instead of staging through PyTorch's pinned ring.

One trap is worth naming because it is silent: a shard stores its four experts in
*name* order and not numeric order, so `ep2` holds `10, 11, 8, 9`. A bank that put
expert `e` at `e * expert_bytes` would hand the device path four experts that are
each the wrong one — real experts, right shapes, wrong numbers, no error anywhere.

`device_experts.py` runs the drawn experts on the card. The kernel is
`moe_single_token_fp4_forward` from `src/csrc`, which takes the released storage
format directly — `[N, K/2]` E2M1 codes beside `[N, K/32]` E8M0 scales — so the
expert arithmetic is shared with the V4.1 path rather than reimplemented, and no
new CUDA was needed. What the kernel does not take directly is the checkpoint's
arrangement: an expert is one 12.75 MiB record in `down, gate, up` order and the
kernel wants three tensors in `w1, w2, w3`, so a draw is six copies an expert into
an arena laid out the kernel's way, on a copy stream ordered against the compute
stream in both directions.

One layer's decode draw, on the release:

| | Value |
| --- | ---: |
| Experts drawn, staged | 8, 102 MiB |
| Staging | 14.18 ms (7.02 GiB/s) |
| Kernel | 1.28 ms |
| Draw, staged and computed | 15.58 ms |

The 7.02 GiB/s above is the *pageable* reading, and it is what the pin exists to
delete: with the bank registered in place the same draw stages at 10.4 GiB/s,
which is 10 to 11 ms, and the link is then within 20% of what a PCIe 3.0 x16 slot
does. That is what sets the shape of the rest: forty-seven draws of 102 MiB is
**4.68 GiB for one token**, no arrangement of the same bytes gets under the link's
rate, and the only lever that moves it is not copying them — which is what the
expert-parallel stage below is for, and what its own section measures.

The kernel's arithmetic is not the float32 reference's, and the whole difference is
the activation quantisation: it takes int8 activations, one scale a row, and
accumulates in float32. Against a host emulation of that same quantisation the device
output agrees to **4e-7**, which is float32 rounding — so the kernel is exact for its
own arithmetic and every difference from the float32 reference is that quantisation
rather than a bug. On a released layer with an ordinary draw the two agree to a few
parts in a thousand of the output's peak, and worse where the summed output is small
and the hidden is not, which is cancellation and not something a tolerance fixes. The
end-to-end cost of it is a property of the whole backbone and belongs to a run of it.

## Attention on the card

`device_attention.py` runs one layer's attention on a device out of the released
weights, for both families, against a per-layer cache. It is the same arithmetic as
`layers.py` and it is diffed against it: on the release, in float32, an entire
attention block agrees with the host reference to **2e-6 of the output's peak**, which
is the reassociation of the same float32 sums and nothing else.

Four things about it are the model and not the code.

**The projection is cut, not reordered.** The fused `qkv_proj` is stored as four
tensor-parallel shards of `[q | k | v]`, so the device runs the linear in the stored
row order and cuts the *output* with `split_fused_qkv`. Reordering the weight is the
same answer for 2x the bytes and one more place to get the layout wrong —
`fused_qkv_row_order` exists for a kernel that wants three contiguous matrices, and a
test holds the two readings to the same permutation.

**The value scale is applied before the cache.** A cached value has already been
scaled by 0.707; scaling it again after the read is a plausible-looking constant
error in every windowed layer.

**The sink is a column, not a bias.** It takes part in the row maximum and in the
denominator, so a sink that wins takes the row and a sink that loses is invisible —
which is *not* what an additive logit bias does, since a bias cancels in a softmax
and would change nothing at any value. The tests falsify the additive reading
explicitly.

**The window is 128 keys, and only a window.** A global layer at 256k reads 256k
keys a token and a windowed layer reads 128, which is why the cache gives a windowed
layer a 128-slot ring and a global layer the whole context. That difference is
asserted rather than described: the same 100 keys are rewritten in both layers' caches
and only the global layer's decode step changes, by a margin, while the windowed
layer's is bit-identical.

The attention itself is two implementations of one softmax, chosen by size, because
the two costs are not the same cost. A single pass materialises the scores — and pays
a *launch* a block when it is written as a loop, sixty-five blocks of a few
microseconds each being a millisecond of nothing — so a decode step, which is one
query, takes it. A prefill chunk takes the online block loop, which holds no
`[queries, keys]` tensor and costs 256k the memory 128 costs. The block loop's bounds
are what make a window affordable: a key block is answered by a slice of the query
rows found with two binary searches, so a cacheless 4096-token chunk with a
window of 128 evaluates **about a quarter** of the pairs a dense pass would — a
block plus a window a block, against the whole chunk — and the loop reports both
counts. Those two searches are taken for **every block at once** and read back once,
because one per block per bound is a device-to-host round trip and a round trip here
stalls the pipeline on whatever was queued behind it: a 64k chunk against a 1k block is
128 of them a layer. What that is worth depends on what the host has to do while it
waits, and on a prefill it is **half a percent** — 690.3 s to 686.1 s for a 64k prompt
at a 2048-token chunk — because the host is behind the device there anyway. On a decode
step, where 96 of a token's 145 round trips came from the same kind of check, it is
29.7 ms a token at 64k; see the decode section below.

**The loop's tile is the chunk's width, and it is the thing at 256k that runs out of
card.** `blocked_attention` materialises `[kv_heads, groups, rows, block]` float32 of
scores, `rows` is the query slice of a key block, and a global layer's late key blocks
are visible to the whole chunk — so the tile is 1 GiB at 4096 rows on a card holding
14.3 GiB of weights, 5.65 GiB of 256k cache and a 408 MiB arena. The bound is now
`row_step`: the *rows* of a slice are split into steps sized to a fixed budget of
**2²⁶ scores**, which is 256 MiB of float32 with about as much again alive as their
exponentials, and which at the release's `[4 kv, 16 groups, 1024 block]` is **1024 rows a
step**. Splitting is not free — a slice rebuilds a mask and launches two smaller gemms,
and the split cost **9%** of a 64k prefill when the budget was 32 MiB — but it is what
makes a 4096-token chunk fit where a 1 GiB tile did not, and the budget is wide enough
that a 64k prompt's attention runs four steps against a chunk of 4096.

**The budget is a constant and not the card's free memory, and that was measured.**
Reading `mem_get_info` at every block-loop attention is the obvious design — the same
chunk needs a 32 MiB tile against a 128-key window and a 256 MiB one against a 256k
cache — and it is the wrong one, because the reading is **per rank**. Four ranks of a
lockstep layer then pick four different steps, do four different amounts of work, and
every routed layer's `all_reduce` charges all four of them the slowest one's time. A
256k prompt under a free-memory budget measured **8246.7 s** with the four ranks'
attention columns spread 19.47 to 25.47 ms a token; the same prompt under a constant
step took **5424.9 s** with the columns spread 15.03 to 15.53. Fifty-two percent for a
"memory" bound that was really a schedule — and the constant fixes the spread as well
as the mean, because a step every rank agrees on is a step no rank is charged for
missing. A constant is also the only kind of step a CUDA graph could capture.

The narrower constant is worse in the same direction, which is why the comment on
`DEFAULT_TILE_SCORES` records 2²³ as an intermediate: the same 256k prompt under a
2²³ step — 128 rows a step — took **6446.8 s** at 19.36 ms of attention a token, so the
budget buys what it spends and a step of a thousand rows is the one this hardware
wants.

The split is arithmetic and not an approximation: a row sees its own key blocks in the
same order with the same running maximum, and what changes is the row count of a gemm,
which is a summation order inside a dot product. Measured against the whole-slice loop
at `[8 heads, 2 kv, 37 rows, 211 keys]`, the largest difference any step below the
whole made was **4.5e-08** on outputs of order 0.05, and the two tests that hold it are
`test_a_narrower_row_step_is_the_same_answer_to_a_rounding` and
`test_the_call_bounds_its_tile_and_not_only_its_total`.

The single pass has a second measurement in it, and it is the kind that only shows up
in a profile. Folding the 64 query heads into `[kv_heads, groups, ...]` batches and
broadcasting one key head is what the block loop does and is right there; in a single
piece, where the *keys* are long, it is a batch dimension of one expanded over
sixteen, which cuBLAS materialises — 162 ms for a 65k-key decode where four per-head
gemms are 5.8 ms. The fold is the cheaper of the two below about a thousand keys,
which is a window, and the per-head walk above it, which is a global layer.

One bug in here is worth naming because nothing else in the file could have caught it.
The online softmax's running maximum was initialised from the sink by expanding a
`[heads, 1]` tensor over `queries` and calling `contiguous()` — which, for a single
query, is *already contiguous* and returns the sink itself. The running maximum then
wrote the row maximum back into the layer's own parameter, so the call that did it was
correct and the next call was wrong, by an amount that decays with how much the sink
still matters. It is why the last test in the attention file calls twice and compares
the parameter with itself.

## The whole model on one card

`device_model.py` is the assembly: the embedding, forty-eight layers of two adds
each, the final norm and the head, with a KV cache and one token per step. Layer 0
is the dense layer — global attention and a 16,384-wide FFN — and the other
forty-seven draw their experts. Three things in it are the reference's and are easy
to get wrong in a way that still runs: the embedding is **not** scaled by
`sqrt(hidden)`, the head is **not** tied to the embedding, and a layer's residual
is added in the hidden dtype on both sides, with the routed sum rounded once where
the kernel leaves it in float32.

The head is the one tensor the model keeps in float32 while everything else runs
bf16. Its weights are bf16 in the checkpoint either way, so what the cast buys is
*resolution*: the head's output is what a sampler reads, and a bfloat16 logit at a
magnitude of twenty is quantised to 0.125 — which is exactly the size of the margin
the third greedy step below turns on. Two and a half gibibytes of head and about two
milliseconds a token is a cheap price for logits that are not on a grid.

The one that is this stage's own is the arena. Forty-seven layers at 102 MiB each
would be 4.8 GiB of a 22 GiB card for state that is read once a layer a token, so
the arena is **one object shared by every routed layer** and the layer a draw
belongs to is a property of the call rather than of the module. A model that
forgot to pass it would stage layer 1's experts while computing layer 2 — the
right shape, the right id range, the wrong numbers, and no error anywhere. The
test that catches it makes the source's codes depend on the layer, so the
comparison against the host is what fails.

The bank is asked to pin itself on construction (`MimoV2ExpertBank.pin_if_enabled`,
which honours `POCKETLLM_MIMO_PIN_RESIDENT_EXPERTS=0`). That is what makes the
staging copies legal sources: without the registration a `non_blocking` copy from
the segment goes through PyTorch's own pinned ring and **pays for the bytes
twice**. Registering 149.81 GiB takes 21 to 27 s and is idempotent, so the
forty-seven layers pay it once.

A token on the release, bf16, eight-token context, warm cache, the first token
discarded — `tests/bench_mimo_v2_model.py`:

| | Decode, 8-token context | The same loop over a prompt |
| --- | ---: | ---: |
| Wall | 610.0 ms | 609.0 ms |
| Attention, all 48 layers | 63.6 ms | 62.8 ms |
| FFN and expert staging | 532.9 ms | 532.6 ms |
| Experts drawn, staged | 376, 4794 MiB | 376, 4794 MiB |
| The copy, at the measured 10.4 GiB/s | 450.2 ms | 450.2 ms |
| **The copy, as a share of the token** | **73.8%** | **73.9%** |

376 experts is exactly 47 × 8, which is what the model says it draws, and the
4794 MiB that carries is 4.68 GiB of host-to-device traffic for **one token**. The
attention is 64 ms of the 610 and the rest is the copy: the split columns are
CUDA events around every layer's attention and FFN, so the wait for a slot's
experts lands in the FFN column, which is where it belongs.

The card holds 12.9 GiB for the whole model — 11.4 GiB of weights, 204 MiB of
arena, 5.1 MiB of cache at a short context — and builds in 33 s from the release's
mmap. Eight gibibytes of a 22 GiB card are free, which is the room the next stage's
wider arena has to live in.

This is the first place where the port can be asked what it actually predicts, and
the answer is the host's. Feeding `The capital of France is Paris, and the capital
of Japan is` to the forty-eight layers on the card puts `' Tokyo'` at rank 0 with a
logit of **19.625**, against **19.63** recorded for the float32 host reference
earlier in this page — the same token, and a two-thousandth of a logit apart.

On a chat-templated prompt, three greedy steps of the same weights on two very
different machines — float32 on 44 CPU cores, bfloat16 on one card, 163 s a step
against 0.65 s:

| Step | Host, first three | Card, first three | Host's top-8 set | Card's top-8 set |
| ---: | --- | --- | --- | --- |
| 0 | `<think>` 46.223 | `<think>` 45.414 | same eight | same eight |
| 1 | `The` 26.247 | `The` 25.981 | same eight | same eight |
| 2 | ` user` 24.230 | ` user` 23.750 | same eight | same eight |

Identical tokens at every step and, at every step, the *same eight candidates* —
with the ordering inside the set free from rank four down, where the host's
`<|im_end|>`, `</think>` and `The` are 19.263, 19.119 and 19.117 and the card's are
`The`, `</think>` and `<|im_end|>` at 20.029, 18.953 and 18.665. The top logit
carries 0.8 of a logit of disagreement on 46, which is the bfloat16 storage and the
kernel's int8 activations accumulating over 48 layers; the two-layer test below
measures the same quantity at the start of the stack, 2.8e-3 to 7.9e-3 on a stream
whose peak is 0.55.

Greedily, the card then answers the question:

```
<think>The user is asking about the capital of France.</think>The capital of France
is **Paris**. 🇫🇷<|im_end|>
```

which is a reasoning block, a turn, a bolded answer and an emoji.

Two honest notes about what this is not. The prompt columns above are the same
single-token loop run over a prompt, so on this stage's page they are a **floor** and not a
prefill — the grouped kernel that makes a chunk a chunk arrives with the prefill stage, and
that is what the section on the prompt as chunks measures. And the
agreement above is an agreement about the *top* of a distribution: a token whose
candidates are a tenth of a logit apart is a token the two machines are free to
disagree on, and the third step's margin was 0.125 on the card against 0.463 on the
host. That is what 0.8 of a logit of disagreement does to a near-tie, and it is why
a serving path wants a sampler rather than an `argmax`.

One bug in here is worth naming, because it is the shape of failure this stage
creates. `MimoV2DeviceModel.step` returned a squeezed `[vocab]` row, and the
generation loop indexed it with `[-1]` — which took the row's *last logit* instead
of the row. `argmax` of a scalar is 0, so from the second token on the loop fed
itself token zero: the model ran at full speed, drew eight experts a layer per step,
and printed `'!'` twenty-three times. Nothing about that is slow and nothing about
it errors, and the fix is one character. What caught it was comparing a continuation
against the host's; what would have caught it sooner is the test that now exists,
which asserts the row's shape and then that `greedy` reproduces a loop written out
by hand.

## The experts dealt out over the cards

One rank's 610 ms token is 74% expert copy, and the copy is a per-token quantity: eight
experts a layer, 12.75 MiB each, 4.68 GiB a token. Four ranks that each own a quarter of
the experts move a quarter of the bytes, which is what `ep.py` is — and it is the one
change in this page that turned out not to be worth what the arithmetic predicted, in a way
worth recording.

**The router is replicated and there is no dispatch.** Every rank runs the same gate over
the same hidden state and draws the same eight experts, so nothing about *which* experts
are needed crosses the fabric. That is affordable because the gate is a `[1, 4096] x
[4096, 256]` matmul and because `gate_and_route` is one function every rank runs in the
same dtype on the same bytes: the draw is a deterministic function of the row, so every
rank agrees on it without being told. A path that had to guess would need the ids on the
wire and the shape of the design would change.

**One collective a routed layer, and it is the layer's output.** The kernel sums a weighted
set of drawn rows, so a rank holding a subset of the draw holds a *partial* sum and the
layer's answer is the sum of the partials: one `all_reduce` of `[1, 4096]` fp32, sixteen
kilobytes, forty-seven times a token. Back to back on this fabric that message is **128 µs**,
which over a token is **6 ms** — a fifth of a percent of the token, and the reason a
per-layer collective is affordable at all here where V4.1's 80 MiB activation tiles are not.

**The deal is where the measurement corrected the arithmetic.** `sorted` gives sorted
position `p` to rank `p % world`, so a top-8 draw over four ranks is exactly 2, 2, 2, 2;
`id` gives `expert % world`, which partitions the *experts* over the ranks and is what a
chunked prefill wants (a chunk draws nearly every expert, and only `id` stops a rank from
staging all of them). On decode, `sorted` — now the default — is the faster one, and it is worth
more than the 27% this table shows once the attention is not in the way: on the split arm the same
comparison measures **179.0 against 253.2 ms at eight tokens of context and 177.0 against 270.8 at
32768** — 41% and 53% — which is what a step built on the chunk's deal costs:

| Deal | Decode step | Staged a step, by rank | Arena |
| --- | ---: | --- | ---: |
| `sorted` (default) | **275.2 ms** — 3.63 tok/s | 1198.5, 1198.5, 1198.5, 1198.5 MiB | 51 MiB |
| `id` | 348.6 ms — 2.87 tok/s | 1285.6, 1253.8, 1175.1, 1079.5 MiB | 204 MiB |

The whole of that 73 ms is the imbalance, not the bytes: the two deals stage the same
bytes *in total* over a token — 47 x 8 experts either way — but a per-layer collective
charges **every** rank the *widest* draw's work, and the widest of four ranks under `id` is
3.4 experts on average where `sorted` is exactly 2. 47 x (3.4 - 2) x 12.75 MiB is 0.84 GiB,
which at 10.4 GiB/s is 81 ms — the measured 73, to the round of the estimate. A deal that
is worse per rank can be better in lockstep, and the numbers are the only way to know which.

The 2, 2, 2, 2 split also halves what the arena has to hold: under `sorted` a rank can only
ever be dealt `ceil(top_k / world)` rows, so the arena is 51 MiB a slot against 204. And
under `id` a rank of four owns nothing in **10% of top-8 draws** — `(3/4)^8` — which is not
a corner case: the rank has to arrive at the collective with a zero, which is why the empty
share is a value and not a skipped call.

**What the four ranks agree on.** The multi-rank run gathers every rank's last logits after
the first decode step and compares them bit for bit — all four are byte-identical, which is
the collective's own check: a rank that staged a row it did not own, dropped one it did or
summed a partial twice disagrees here. Then the four greedy continuations, which are the
same nine ids, and the same nine the one-rank run produces on the same prompt:

```
14925 227 60096 72653 86162 85033 145420 54575 145959
```

Adding a rank changes the order the eight partial sums are added in and so the last bits of
every logit; nine tokens of agreement and identical argmax at every step is what that costs
here, and the one-rank comparison in the tests is the check that it stays that way.

**And the token is no longer copy-bound.** A per-layer profile on eight routed layers of
the release, four ranks, CUDA events on the stream each span belongs to, under the `id`
deal:

| Span | ms a routed layer | What it is |
| --- | ---: | --- |
| Attention | 1.5 | the replicated torch attention, and what is left of it |
| Router | 0.5 → **0.2** | the gate and the top-k, whose result the host needs. The 0.5 is the reference's Python dispatches; reached from C++ it is 0.19 ms a layer, measured |
| Expert copy | 2.4 | 25.5 MiB, at the link's own 10.4 GiB/s |
| Expert kernel | 0.4 | two experts instead of eight |
| The collective | **1.6 in situ** | against **0.128 back to back** |
| The rest | 0.8 | the norms, the adds, the head, the host's own issue |

Read the last two rows together: the same message costs **1.6 ms inside a layer and 0.128 ms
in a loop**, a factor of twelve, and the twelve is the *lockstep* rather than the message.
Re-measured on its own, the message is 0.126 ms and is latency rather than bandwidth —
sixteen times the bytes cost the same 80 µs — so there is nothing in the transport to fix;
see the probe in the evidence below. A collective is a barrier, four ranks' per-layer host
work is not equal, and every rank pays the slowest one's time; the `id` deal doubles that by
making the copy itself unequal. It is why the imbalance above is worth 21%, and it is the
measurement that redraws the next stage's target: the step is now attention plus copy plus a
lockstep, and 48 x 5.7 ms is the 275 the `sorted` arm measures. The attention row was the one
that did not have to be replicated, and it no longer is: at this depth the split below takes
**49 ms off the whole step, 1.02 ms a routed layer**, which is most of that row.

The whole-model phase totals behind that table, at 8 prompt tokens and 8 decode steps:

| | `sorted` (default) | `id` |
| --- | ---: | ---: |
| Decode step | **275.2 ms** | 348.6 ms |
| Prefill-shaped pass, 8 tokens | 3.80 tok/s | 2.96 tok/s |
| Attention, all 48 layers | 65-88 ms | 72-81 ms |
| FFN and staging | 171-198 ms | 254-264 ms |
| Experts staged a step | 94.0 | 84.7-100.8 |

## The prompt as chunks

A prefill is not a decode step repeated, which is why `device_experts.py` has two entry
points and why they are not two spellings of one call. `forward` takes one row and one draw
of `top_k`; `forward_chunk` takes a chunk and every row's own drawing. The difference is not
the batch — the kernel has always been batched, and a single-token call is a batch of one —
but the **layout**. A draw of eight names eight experts out of a quarter of them, so the
arena holds the draw and the kernel reads one row a drawing. A chunk of four thousand tokens
draws nearly every expert there is, so what a rank computes is the subset of the chunk's
*pairs* whose expert it owns, and the kernel wants those grouped by expert with an arena row
a group. That kernel is `moe_multi_token_fp4_forward`.

**The layout is built on the card, from the routing alone.** A table maps a global expert id
to the row that holds it — or to a sentinel, for the ones this rank does not hold — and one
stable sort by that row puts every pair the rank does not own behind every pair it does.
`searchsorted` of the sorted rows against `0..n` is then the counts' exclusive prefix sum in
one op: deliberately not `bincount`, whose CUDA implementation bounds-checks with two
blocking device-to-host reads, and V4.1 replaced the same call for the same reason. What is
left is one host read a layer, for the band boundaries, and that is the decode path's own
price — it pulls the whole draw across with `indices.tolist()` every layer.

**The deal has to change with the shape.** `id` partitions the *experts*, so a rank of four
stages a quarter of the layer's 256 and computes a quarter of the pairs. `sorted` partitions
a *drawing*: over one token that is exactly two experts a rank, and over a chunk a rank's
positions reach every expert there is, so every rank stages all 256 and computes its own
quarter of the pairs anyway — the same arithmetic for four times the copy. `forward_chunk`
refuses `sorted` at a world over one rather than serving it slowly, which is why the prefill
build names `deal="id"` while decode keeps the `sorted` default that is 27% faster there.
The deal is per *module* and not per call, so a serving run that has to do both keeps two —
and a run that keeps one pays the difference on whichever half it built for: 41 to 53% on a
step, measured, which is why `bench_mimo_v2_prefill.py` builds the served pair by default and
`--deal id` is the explicitly chunk-only build.

**A band is how the arena stays bounded.** One kernel call stages its whole working set, so
the arena has to be as wide as the experts one call holds. A rank that owns more than that is
computed in bands of that width, one call each, summed in float32 — and the bands cost no
layout work, because they are slices of the one sorted pair list: band `b` owns slots
`[b·w, (b+1)·w)` and its pairs are the contiguous run between two prefix sums. A band whose
experts the chunk never drew is skipped, which is what stops a narrow band from being a way
to do *more* work rather than a way to spend less memory. The width trades arena bytes for
call count and the calls of adjacent bands overlap the way layers do, on the slots.

A 4096-token prompt through four ranks on the release, experts out of the bank,
`tests/bench_mimo_v2_prefill.py --band 0`:

| Chunk | tok/s | Calls | MiB/token | The copy | Copy, share of a token | Attention | FFN and staging |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| one token | 2.93 | 4096 | 1207 | 111 ms | 33% | — | — |
| 256 | 56.3 | 16 | 149.8 | 13.8 ms | 78% | 1.63 ms | 16.01 ms |
| 512 | 93.2 | 8 | 74.9 | 6.9 ms | 65% | 1.67 ms | 8.94 ms |
| 1024 | **134.7** | 4 | 37.5 | 3.7 ms | 49% | 1.90 ms | 5.40 ms |
| 2048 | **174.6** | 2 | 18.7 | 1.7 ms | 30% | 1.95 ms | 3.64 ms |
| 4096 | — | 1 | — | — | — | — | did not fit |

The shape of that table is one fact. A chunk's bytes are a cost per **call** and not per
token: a rank's share of a layer is 816 MiB whatever the chunk width, so a wider chunk pays
the same bytes for more tokens and `MiB/token` falls as `1/chunk` until the whole layer stack
is 47 × 816 MiB = 37.5 GiB a call. The attention's per-token cost does not fall — 1.63 to
1.95 ms a token across the whole range, and *rising*, because a wider chunk attends to more
keys — and neither does the host's share of the layer. So the rate is the copy amortised and
everything else a constant, and the crossing point where the copy stops being the majority is
around a 1024-token chunk.

The band trades the same two quantities in the other direction. The same run at `--band 16` —
a quarter of a rank's share a call, four calls a layer, and an arena of 408 MiB instead of
1632 — measures **111.2 tokens a second at a 1024-token chunk and 137.3 at 2048**, 17% and
21% below the one-band numbers, for 1.2 GiB less card. What the narrower band costs is not
bytes but calls: four bands of 16 experts move exactly what one band of 64 moves, and pay four
kernel launches a layer instead of one, four slot waits, and three extra `[rows, dim]` float32
accumulations of the partials — 201 MiB a layer, 9.4 GiB a chunk at this width, which is 0.9 s
of the measured 6.4 s the wider band saves, so the rest of it is the calls themselves.

The floor row is the same prompt fed one token at a time, which is what the single-token
kernel can do: 2.93 tokens a second, 1207 MiB a token on rank 0 — 1108, 1287, 1208 and 1192
across the four, since a rank's draws are its own — and the same 47 draws a layer that a decode
step makes. That is the path a prefill used to be, and the grouped kernel is **46 to 60 times**
it on the same weights.

**There was a ceiling and it was not the link.** A 4096-token chunk used to not fit: the
attention's block loop materialises `[kv_heads, groups, rows, block]` float32 of scores, and a
*global* layer's late key blocks are visible to the whole chunk, so its tile was 4 × 16 × 4096 ×
1024 × 4 bytes — 1 GiB, on a card that is already holding 14.3 GiB of weights and 1.6 GiB of
arena. A windowed layer never did this: its query slice is a block plus a window, so its tile is
fixed. The nine global layers were what capped the chunk width, and the fix went into that loop
rather than into the expert path: the tile's rows are now a step sized to a fixed budget, and the
chunk is no longer the unit that has to fit. The ceiling moved with it and did not disappear — the
*whole layer* has to fit, and at 256k the numbers in the next section are what that costs.

**And the width is not monotone.** At a 64k prompt a 4096-token chunk beats a 2048-token one,
104.4 against 95.5 tokens a second, which is the copy amortised and nothing else. At **256k it
reverses**: the same prompt at a 4096-token chunk did not finish in four hours where 2048 finished
in under two. The tile is part of it and not all of it — the constant budget gives a 4096-row chunk
twice the steps of a 2048-row one for the same total rows, so the wider chunk pays more masks and
more gemms — but the attention's cost at 256k depth is 15.1 ms of every prefill token where at 32k
it is 3.6, and a wider chunk attends to more keys per token in exactly the way that number
describes. What is recorded is the measurement and not a model of it: the run below is a
2048-token chunk, the width was swept at 64k and at a 32k prompt against a 256k cache, and it was
not swept at 256k depth because a sweep there is two hours an arm.

**What the four ranks agree on.** The prefill's last row, gathered from all four after the
widest chunk, is byte-identical — `0.0` between rank 0 and every other — and each rank staged
18.7 MiB a token, the same number, because under `id` the shares differ in *which* experts
they are and not in how many. The one-rank comparison in the tests is the check that it stays
that way: a chunk is the same answer as the same rows one at a time to **3.0e-7** of the
logits' own peak on two released layers, and takes the same token at every position.

## 256k, and what a token at that depth costs

The cache is per layer and the arithmetic is per family: a global layer's buffer is the whole
context, so the nine of them at 262144 positions with four key heads of 192 and 128 are 5.65 GiB,
and a windowed layer's is a 128-slot ring whose thirty-nine are 0.6 GiB between them. On the
release, four ranks, a 2048-token chunk, a 408 MiB arena at a band of 16 experts and 64 experts a
rank, a 262144-token prompt through it end to end — and then the same prompt again, back to back on
the same cards, with the attention divided four ways instead of replicated on every rank:

| | Replicated | **Split four ways** |
| --- | ---: | ---: |
| Prompt | 262,144 tokens, 128 calls of 2048 | the same prompt, the same 128 calls |
| Wall | 5419.3 s | **2519.6 s** |
| Rate | 48.37 tokens a second | **104.04 tokens a second** |
| Attention, all 48 layers | 15.45 ms a token | **4.38 ms a token** |
| FFN and staging | 5.19 ms a token | 5.20 ms a token |
| Expert copy | 1.7 ms a token — 8.4% of the token, 18.6 MiB | 1.8 — 18.2% |
| On the card | 18.71 GiB allocated, 2.04 GiB free | **10.21 GiB allocated** |
| The four ranks' last row | byte-identical, `0.0` between rank 0 and every other | byte-identical, `0.0` |
| The first tokens at that depth, by rank | the same three on all four | the same three |

**The attention was the whole of the difference, and a rank is now the whole of the fix.** At 64k a
token costs 5.1 ms of attention and 5.3 of everything else; at 256k the attention is 15.45 against
the same 5.19, and the copy's share falls from 17% to 8% not because the copy got cheaper but
because the token got three times more expensive. That is the shape of the requirement: a global
layer's decode reads 256k keys and a global layer's prefill attends to everything below it, so the
cost of a token at depth `p` is linear in `p` and no arrangement of the experts changes it. What a
longer context costs here is attention, and this paragraph used to conclude "what would move it is a
kernel — a flash-style pass that does not materialise the score tile, or tensor cores in a dtype the
parity tests still accept — and not another rank." **A rank moved it**: the attention column is
15.45 → 4.38, the wall is 2.15× shorter and the rate is past the hundred a second this was written
against, because the four ranks were each computing all sixty-four query heads over the same quarter
of a million keys and now each computes a quarter of them from the checkpoint's own partition. The
second half of that sentence still stands on its own: the tile budget moved the same number in the
*good* direction without changing the work — the same prompt under the narrower 2²³ step cost 19.4 ms
of attention a token and 6446.8 s of wall, and the 2²⁶ default is 15.45 and 5419.3 s for the same
arithmetic. And the split leaves the kernel where it was: 4.38 ms of the 9.6 a token is still a
score tile materialised in torch, and a `swa` layer's is a 128-key ring that costs the same at every
depth, which is a kernel's job and not a rank's.

**And a decode token at 256k.** Two numbers, and they differ by a deal. A *served* run's step at that
depth is the probe's, `sorted` with both arenas: **323.8 ms a token replicated — 3.09 tokens a second
— against 205.6 and 4.86 split**. A prefill bench's `--decode` column is not that: `id` is the only
deal a chunk can use, so a bench that builds one module measures its steps through the prefill's own,
and the pair it reported at this depth is **399.2 ms replicated against 281.2 split** — two steps and
no warm-up, from before there was a flag. `bench_mimo_v2_prefill.py --deal` now names the *step's*
deal and defaults to `sorted`, which is what `pocketllm serve` builds, and the flag's own A/B at 8192
tokens of context reads **199.2 ms a token, 5.02 a second, against 259.7 and 3.85 under `id`**: 23%,
where the interleaved step probe measures 41 to 53% at the depths it covers, because two arms in two
processes are not an A/B on this box and the probe's own docstring says so. Of the 281.2 the attention
is **161.8 to 163.3 ms replicated and 48.2 split** and the FFN and staging 228.2 to 230.3 against
226.0, with 18.48 GiB on the card against **10.21**. The FFN column is the number to look at now: it
does not move under the split, and it is 226 ms of the 281. For contrast, the same step at eight
tokens of context is 177.6 ms with 139.6 of it outside the routed path. **The expert path is what
grows next**, and that is the less obvious half: a token draws the same 47 × 8 experts whatever the
context, so the bytes are the same 1198 MiB a rank — but the path is a lockstep rather than a copy,
four ranks whose bytes are equal and whose times are not, so what a deeper context adds to it is the
waiting rather than the traffic. A client at 256k should expect **five tokens a second** after a
first token that costs the prompt, and the two paragraphs below are what that five is made of.

**And that step is the copy plus everything else, in series.** `probe_mimo_v2_decode_phases.py` at
that depth, four ranks: **copy stall 99.9 ms** of a 204.7 ms token, attention 45.3, expert kernel
12.5 to 13.0 in 47 calls, collective 3.8 to 13.9, router 4.7 to 12.5, and 27.6 ms of dense linears,
the head, the sampler and the host — the router's own event pair is recorded after a call that blocks
the host on the layer's earlier work, so its number is the largest that is double-counted rather than
a cost of its own. The stall is the part of the H2D the kernel *waited* for, and it is 1198.5 MiB at
**12.0 GiB/s**: a PCIe 3.0 x16 link at essentially its rate, which makes the bytes the floor and the
*schedule* the only thing that decides whether they are paid or hidden. Two cuts were measured
against it.

**The attention's own copy of the prefix is gone.** `MimoV2KVCache.append_and_span` hands the attention
the span it is about to read — `[0, start_pos + n)`, the prefix with this call's own keys on the end of
it — as a view of the cache buffer whenever nothing has wrapped, so the `torch.cat` that used to build
it, **160 MiB of keys and values a global layer a token** (read and written, 2.8 GiB a token over the
nine of them) with the attention split over four ranks, is never made.
`probe_mimo_v2_decode_prefix.py`, the two arms interleaved at 262144 and four steps each:
**197.2 ms a token against 202.3** — 5.07 against 4.94 a second — with the attention column **38.8
against 44.3**, and the two rounds differing by 3.9 and 6.3 ms in the same direction. (The split's own
A/B above measured 205.6 for the same configuration before this cut existed, which is the 3 ms of
drift between two processes on this box rather than a disagreement.) It is exact by construction: the
buffer *is* the span in slot order while nothing has wrapped, so the view is the concatenation's own
bytes in the concatenation's own order, and the tests compare the two spellings of it on released
layers with `torch.equal`. A ring is not a span — its prefix is the last `slots`, which is a
rearrangement — so a windowed layer past its window reads as it always did, and that is also the arm
the probe compares against: a cache whose method answers `None` is the code that ran before this.

**And the copy cannot be predicted, so it cannot be prefetched.** The one thing a rank knows early
about layer `L` is what it staged for layer `L` of the *previous* token — the same rows of the same
arena — so the question is whether that prediction is worth acting on. `probe_mimo_v2_expert_reuse.py`,
32 greedy steps of a real 8192-token prompt on four ranks: a row holds the expert it held a step
earlier **nine to thirteen and a half times in a hundred** — 0.18 to 0.27 of a rank's two, with 76 to
84% of the 1457 (layer, step) pairs keeping neither row — and the rank's whole *set* repeats in **1.9
to 2.4%** of the draws. So a prefetch issued from that prediction would be right about one row in
eight and wrong on the rest, spending a link that is already the floor on bytes the next step does not
read; against the ~3% a re-draw of the same expert would score if the 256 were balanced, it is three
to four times better than nothing and no schedule at all. The draws of this router are that
scattered, and it is the same fact that makes a wider batch no better — a chunk's or a draft's tokens
draw nearly disjoint experts, so staging them together is one copy either way.

## The round trips a decode token pays

A four-rank decode step at eight tokens of context is **177.6 ms — 5.63 tokens a second**, up
from 235.4 ms and 4.25 before this stage, and none of that 58 ms is arithmetic: it is host round
trips that were checked, twice a layer, for facts the caller already knew.

The profiler's own tables say where they were. Per step, on the release at four ranks:
**5,263 `cudaLaunchKernel` calls, 145 `cudaStreamSynchronize` calls at 777 µs each — 112.7 ms of
the host's 325.2 — and 1,521 `aten::copy_`**. The syncs are the ones that matter, because a sync
is not a cost that overlaps: the host stops until the device has drained everything queued,
including the expert copy it just launched.

| Source | Round trips a step | What it was for |
| --- | ---: | --- |
| `_check_bounds`' value checks, two a layer | 96 | `int(upper.max()) >= keys` and `bool((upper < lower).any())` |
| `indices.tolist()`, one a routed layer | 47 | the draw, which the host stages from |
| `sample_token`'s row, one a step | 1 | the logits the sampler reads |
| **Total** | **145** | |

The first row is the one that went. `_check_bounds` is the low-level API's own guard, and it is
worth its round trips for a caller that hands in bounds computed somewhere else — which is every
test and every direct caller. The *model* is not such a caller: it derives `upper` from
`prefix_len + torch.arange(sequence)` and knows the key tensor is `prefix_len + sequence` long,
and it derives `lower` from `upper` or from zero. Both facts are Python integers two functions
above the check. So `attention` and both paths under it take `validate`, the model passes
`False`, and the two round trips a layer become two comparisons in an `if`. The third row stays,
because the sampler genuinely needs the row, and the second stays, because the host stages the
draw and cannot stage a number it has not read.

Where the 58 ms landed is measurable and it is not evenly spread: at eight tokens of context the
step is 177.6 ms with the routed path at 38.0 and the rest of the layer at 139.6, against 77.4
and 158.0 before. The routed path lost 39.4 ms to 47 removed round trips — 0.84 ms each, which is
the price of the copy that was queued behind each one — and the attention lost the rest.

## The host is the step

The round trips were the first half of it, and after them the step is not waiting on anything: it
is *working*. A step is a function call that returns, and then a wait; timed apart at 32768
positions on four ranks — `tests/probe_mimo_v2_decode_host.py --depth 32768` — the call is **183.3
ms** and the wait behind the queue is **6.1**, of 189.4 ms a token and 5.28 tokens a second, with
all four ranks within a tenth of a millisecond of each other. The card is idle for most of a token,
and everything below is the host catching up.

What it is catching up on is launches and the dispatcher work around them. The profiler's own
counting, over three steps of the same configuration and with the two cuts below *not* applied:
**118,442 dispatcher ops — 39,481 a token, 822 a layer** — of which **16,860 are `cudaLaunchKernel`
(5,620 a token, 117 a layer)** and 2,454 are `cudaMemcpyAsync` (818 a token). Below those sit
`aten::copy_` 1,518 a token, `aten::to` 1,637, `aten::mul` 653, `aten::add` 509, `aten::cat` 480 and
15,174 `aten::as_strided`, which are the views. Five thousand six hundred launches for one token's
forty-eight layers is 117 a layer, and a layer's own arithmetic is a dozen kernels. The per-op
*times* that come with that table are not quoted here: they are instrumented, and the same op
weighted 32.9 ms a token in one process and 96.3 in the next.

**Two of the dispatches were per-call work that does not have to be per-call**, and both are in the
attention:

* **The RoPE table.** `build_rope_cos_sin` is an outer product, two transcendental kernels and a
  concatenation to produce one row — and for a decode step it produces the row for position *p*,
  which is the row it produced for *p−1* the token before. The model now builds one
  `[capacity, rope_dim]` table a *family* (the nine global layers share a theta and the thirty-nine
  windowed ones another, so two tables cover the stack; forty-eight of them would be 3.2 GiB at
  128k positions against the 268 MiB the two cost) and the layer indexes it.
* **The fused qkv cut.** `split_fused_qkv` builds `[q | k | v]` out of nineteen slices and six
  concatenations; `qkv.index_select(-1, order)` is the same three tensors from one kernel, with
  `order` from `fused_qkv_row_order`, which already existed for callers that wanted the *weight*
  reordered. Both are exact permutations, and both are checked as permutations rather than as
  logits.

**What they are worth, measured against themselves.** Four configurations, interleaved, six steps
an arm, two rounds, one process, best and mean of the two rounds in ms a token — the same run as
the 183.3 ms above, later and on a busier box:

| RoPE table | qkv gather | Best | Mean |
| :---: | :---: | ---: | ---: |
| off | off | 270.8 | 272.6 |
| **on** | off | 263.2 | 263.8 |
| off | **on** | 264.0 | 264.5 |
| **on** | **on** | **253.5** | **254.5** |

**6.4% of a decode token, and the arms are the shipped paths** — the probe turns the two features
off by writing the attributes the layer reads, so what is measured is production code with a
feature disabled and not a monkeypatched copy of it. The same measurement against monkeypatched
equivalents, in an earlier and faster process, put the pair at 251.0 → 231.7. Nothing about the
answer moves either way: `order` is checked to be a permutation of `range(qkv_out)` and the gather
to be `torch.equal` to the split, the table to be `torch.equal` to the cos/sin the call would have
built, and the released checkpoint draws the same nine tokens on the same prompt before and after —
`[14925, 227, 60096, 72653, 86162, 85033, 145420, 54575, 145959]`, identical on all four ranks in
both revisions.

**Read that table as a comparison and not as a rate.** The two processes that produced this stage's
numbers had the same configuration at 194 ms and at 252 ms an afternoon apart — the box carries
other work, and a step this host-bound inherits all of it — and the run above has its own first arm
at 189.4 against the 253.5 to 270.8 the arms after it took. That is why the arms are interleaved
and why the four rows come from one process: a single A-B across two processes said the same patch
was worth 28 ms in one pair and 0.8 ms in the next.

The remaining 117 launches a layer are not free and are not addressed. What they are is the next
stage, and what it takes is fewer, larger kernels rather than fewer round trips.


## The attention split over the ranks

The experts were the first thing the four ranks divided; the attention was not divided at all. Every
rank computed all sixty-four query heads over the same quarter of a million keys, four times over, and
at 256k that replication is **15.45 ms of every prefill token and most of a decode step**. This stage
divides it, along the checkpoint's own partition: the prompt at that depth goes from **48.37 to
104.04 tokens a second**, and a decode step from **323.8 to 205.6 ms** — 197.2 after the prefix cut
described further down this page, which is why every split number here is the split stage's own.

**The partition is the checkpoint's and not this repository's.** `quant.QKV_SHARDS` is four, and it is a
fact about the released file: the fused `qkv_proj` is stored as four groups of `[q | k | v]` with the
FP8 block scale blocked *inside each share*, and `fused_qkv_row_order` refuses any other reading. Group
`g` holds query heads `[16g, 16g + 16)` together with the key heads those query heads attend to — one
for a global layer, which is 64 query heads over 4, and two for a windowed one, which is 64 over 8. So a
rank of four takes one contiguous quarter of the projection's rows, and the query, key and value heads
that go with it follow from the geometry rather than from a choice: `num_key_value_groups` is unchanged
at both families, which is the whole reason a share is the same arithmetic on fewer heads and not a
different attention. Nothing is remapped and nothing has to be agreed — the weights were already on the
card in this shape before this stage existed.

**The join is a gather and not a reduce, and that is an exactness argument.** `o_proj` reads all of
`o_in` at once, so the four shares' `pre_o` have to be back in one tensor before it runs: the answer is
the *concatenation* of the quarters, each rank's own heads in their own columns. The other way was
measured on a released layer at 32768 keys — each rank running its own quarter of `o_proj` and the four
partials added in fp32, which is how the routed experts' split is joined — and it lands **2.9e-3 to
7.5e-3 of the attention output's own peak** away from the whole path, because each partial is rounded to
bfloat16 before it is summed and the answer therefore rounds four times where the whole path rounds
once. The concatenation is `0.00e+00`. It costs nothing to prefer, either: a `[1, 8192]` bfloat16
gather and a `[1, 4096]` float32 reduction are the same sixteen kilobytes a token.

**`all_gather_into_tensor` joins along the first axis; the attention wants the last.** That is the trap
this stage turned on. The primitive takes a buffer of `world · rows` rows and hands rank `r` the rows
`[r · rows, (r + 1) · rows)`, so a caller that allocates the `[rows, world · width]` it wants — the
obvious thing to write — gets a **silent scramble** for any `rows` over one: every piece present, each
in the wrong place, the same size, made of the same numbers, no error and no shape mismatch. It is
*exact* at `rows == 1`, where the flat layout happens to be the concatenation, and `rows == 1` is every
decode step, which is the worst possible failure mode for this bug. The first version of this stage
passed its decode comparison and its token comparison and was wrong: layer 0 on a 512-row chunk
differed by 1.9e-01, and the search for it ruled out the share's weight read, the q/k/v cut, the
probabilities and the per-key-head matmuls — each exact to 1e-07 — before the collective's own layout
was the answer. `make_all_gather` transposes, and `test_the_join_writes_each_rank_down_its_own_columns`
fakes the collective to ask which columns each rank's piece came out in: a question answerable without a
fabric or a second card, and the one the real primitive cannot be asked.

**Two geometries, and a caller has to be told which one it is asking about.** A share's `pre_o` is
its own heads' output and the join turns it into the *layer's* width, so a module that keeps one
`shape` has to pick which lie to tell: a caller that asks what the layer produces once the pieces are
joined wants 4096 columns and a caller that asks what this rank holds wants 1024. `full_shape` is the
layer's, `shape` is the share's, and `MimoV2KVCache.shape(layer)` answers with whichever one the cache
was built to hold — which is not bookkeeping. `append` checks the key heads it is handed against its
own, so a filler that read the layer's geometry while filling a rank's share is refused at the first
token, which is the good outcome and is how the cache's own accessor came to exist.

**The budgets are divided too, and that is a choice with a price.** A share's single-pass test is
`heads · queries · keys` and its tile is `kv_heads · groups · rows · block`; a share's heads are a
quarter of the layer's and its `num_key_value_groups` is not, so both quantities are a quarter of the
whole's at a share of four. Dividing `budget` and `DEFAULT_TILE_SCORES` by `shards` therefore makes a
share step through *the same number of steps* the whole layer does, at a quarter of the width each —
which is what the four-rank chunk path's `0.00e+00` rests on, because the block loop's step size is the
order its float32 accumulates in. It is not the fastest arrangement: an undivided budget would let a
share take rows four times further a step and so pay a quarter of the launches for the same work. What
it would not do is agree with the whole path bit for bit, and a split is worth less without that than
with it.

**The one arithmetic that is not bit-exact is a windowed layer's decode step.** Below `FOLD_KEYS` keys
the attention takes its folded path, which batches the head loop, and cuBLAS tiles that gemm by the
batch size it is handed: a share has two key heads where the layer has eight, so the same arithmetic
comes out one float32 ULP apart — **1.19e-07 of a 5.10e-01 peak** in `pre_o`, the last bit of a bfloat16
after the projection. A global layer past `FOLD_KEYS` keys, and any chunk of queries in either family,
are exact. That is why the probe's tolerance is half a bfloat16 ULP and no looser, and why the token
probe exists beside it: a token stream is the coarsest observable there is, so a stream that agrees says
little, and `probe_mimo_v2_split_tokens.py` prints the top-2 logit margin beside it so that the
agreement has a size. On the release at 8192 tokens of context, with the split and without, on a real
prompt and on a drawn one: the same prefill logits to the last digit (`|sum|` 4.740900e+05, peak
2.903074e+01, margin 1.025631e+01, argmax 8374) and the same greedy stream.

**What it is worth, layer by layer.** `tests/probe_mimo_v2_attention_split.py --layers 0 1`, one card,
the whole attention against one share of it, one cache and one hidden state a row, six timed rounds
after an untimed one:

| Layer | Family | Rows | Keys | Whole ms | Share ms | Gain |
| ---: | :---: | ---: | ---: | ---: | ---: | ---: |
| 0 | global | 1 | 1024 | 2.437 | 1.400 | 1.74× |
| 0 | global | 1 | 32768 | 2.478 | 1.416 | 1.75× |
| 0 | global | 1 | 262144 | 15.660 | 3.367 | **4.65×** |
| 0 | global | 512 | 262144 | 1595.934 | 394.924 | **4.04×** |
| 1 | windowed | 1 | 32768 | 1.582 | 1.551 | 1.02× |
| 1 | windowed | 512 | 262144 | 14.539 | 3.946 | **3.68×** |

A share's arithmetic is a quarter of the layer's and its wall is not: 1.74× and not 4× at a short
prefix, because a call has a floor that does not divide. The deep rows are where the split pays, and the
reason is the same fact as everywhere else in this document — a global layer's cost is linear in the
keys it reads, so at 262144 a share is doing a quarter of a great deal and the floor is lost in it. The
windowed row is the other half of the story. **A windowed layer's decode step gains nothing at all** —
1.02×, and that is not a defect: its ring cache is 128 slots however deep the context is, so there is
nothing there to divide. Its *chunk* path gains the full 3.68×, and a chunk path is what a prefill
spends.

**And the step, which is what decode is.** `tests/probe_mimo_v2_decode_host.py`, four ranks, four
processes an arm in an A-B-A-B order — whole, split, whole, split — so that a drift across the quarter
hour shows up as the two arms of one configuration disagreeing rather than as a gap. Every figure is
one process's own reading, six timed steps after two warm-up steps:

| Depth | Split | Arm 1 | Arm 2 | tok/s | Cache |
| ---: | :---: | ---: | ---: | ---: | ---: |
| 32768 | off | 227.9 ms | 224.5 ms | 4.39 / 4.45 | 0.74 GiB |
| 32768 | **on** | **177.3 ms** | **176.6 ms** | **5.64 / 5.66** | **0.18 GiB** |
| 262144 | off | 306.0 ms | 341.5 ms | 3.27 / 2.93 | 5.66 GiB |
| 262144 | **on** | **202.0 ms** | **209.2 ms** | **4.95 / 4.78** | **1.41 GiB** |

**The arms do not overlap at either depth.** At 32768 the two whole arms are 227.9 and 224.5 and the
two split arms 177.3 and 176.6: a **49 ms gap against a 3.4 ms spread**, 226.2 against 177.0 and 1.28×
the tokens. At 262144 the whole pair disagrees with itself by 35.5 ms — this box carries other work —
and the split pair by 7.2, and arm against arm the gap is 104 and 132 ms. That gap is what the layer
table above predicts: nine global layers × (15.660 − 3.367) = **110.6 ms**, so the in-process layer
measurement and the two-process step measurement agree, which is the reason both were taken. A step at
262144 is therefore **205.6 ms against a mean 323.8, 1.58× the tokens**, and the earlier two-process
pair — one whole arm at 305.6 against one split arm at 232.0 — reads the same direction with a smaller
gap, which is what a single A-B across two processes is worth on this box.

**And the prompt, which was the half that was short.** The same question asked the way the requirement
asks it: `tests/bench_mimo_v2_prefill.py --tokens 262144 --chunk 2048 --band 16`, four ranks, the
attention replicated in one arm and divided in the other, run back to back on the same cards.

| 262,144 tokens through a 2048-token chunk | Replicated | **Split four ways** |
| --- | ---: | ---: |
| Wall | 5419.3 s | **2519.6 s** |
| Rate | 48.37 tok/s | **104.04 tok/s** |
| Attention, all 48 layers | 15.45 ms a token | **4.38 ms a token** |
| FFN and staging | 5.19 ms a token | 5.20 ms a token |
| Expert copy | 1.7 ms a token — 8.4% of the token | 1.8 — 18.2% |
| On the card | 18.71 GiB allocated, 2.04 GiB free | **10.21 GiB allocated** |
| The four ranks' last row | byte-identical, `0.0` between rank 0 and every other | byte-identical, `0.0` |
| The first tokens at that depth | `[8374, 4021, 95012]` on all four | the same three |

**2.15×, and the attention column is 3.5× of it.** The two arms are the same prompt, the same weights,
the same deal and the same 128 chunks, and their last rows are byte-identical to each other as well as
to their own ranks — so what the table measures is the split. What it also shows is the ceiling: the
copy's *share* of a token more than doubles, 8.4% to 18.2%, without the copy getting any more expensive
— a token is 2.15× shorter and its bytes are the same 18.6 MiB — and the FFN and staging column does
not move at all. The split is not a licence to keep splitting: at a 2048-token chunk the attention was
15.45 ms of a 20.7 ms token and is now 4.38 of a 9.6 ms one, so the term it fixed is no longer the
term that is left. And it is the goal's own number: the bar this page was written against is a 256k
prefill above a hundred tokens a second, **48.37 does not reach it and 104.04 does**.

The same two runs finish with a two-step decode at that depth, and that number is **not** a served
run's. The bench built the model with `deal="id"` — there was no flag, and `id` is the only deal a
chunk can use — so its steps went through the prefill's own module, and the deal is worth more than
anything else on this page at that depth. Under the deal a *step* is built with, `sorted`, which is
what `pocketllm serve` builds and what the probe above measures, the same depth costs **205.6 ms
against a mean 323.8**; the bench's own decode column read **281.2 ms split and 399.2 replicated**,
which is the same two configurations through the `id` module. Measured directly,
`probe_mimo_v2_decode_host.py` with `POCKETLLM_MIMO_EXPERT_DEAL=id`: **295.7 ms at 262144 against
205.6**, and **270.8 against 177.0 at 32768** — 41 to 53%, so what looked like a hundred milliseconds
of disagreement between two benches was one deal, and not the attention, the arena, or the state a
prefill leaves behind. The column takes `--deal` now and defaults to the served one, and the flag's
own A/B at 8192 tokens of context reads **199.2 against 259.7** — 23%, where the warmed and
interleaved probe reads 41% at an eighth of that depth, because two arms in two processes are not an
A/B and the bench's steps follow a prompt that has just ended. The FFN and staging column does not
move under the split and it is 226 ms of the 281; the `id` deal's own column is where the rest of that
difference sits.

**And the cache goes with it, for free.** A rank that attends over a quarter of the key heads has to
keep a quarter of them, so the same split that buys the step also divides the KV cache: **5.66 GiB of
global layers at 262144 positions become 1.41**, and 0.74 becomes 0.18 at 32768 — a quarter, exactly,
and the only thing in this document that reduces what a 256k context costs on the card. End to end at
256k that is **18.71 GiB allocated against 10.21**, which is the largest single thing the split gives
away for nothing, and it is not a trade: the bytes the attention reads *are* the bytes a rank has to
keep, so the two requirements pull the same way.

**The knob and the control.** `attention_shards(world)` answers four at a world of four and one
anywhere else, because four is the count the weights admit and a world of two or three has no such
partition — those runs keep the whole attention on every rank and deal the experts as before, which is
correct and slower rather than wrong. `POCKETLLM_MIMO_ATTENTION_SHARDS=1` holds it back to one, which is
the A/B's control arm and is what this model did before the stage existed. A group that was handed no
way to join the pieces does not cut them either: `EpGroup.attention_shards` reads the `gather` it was
built with, so a caller that injects its own collectives — or its own arithmetic in place of them, which
is how the single-card tests check the deal — keeps a whole attention rather than a rank asking for the
fourth quarter of a tensor nobody divided.

## A served endpoint

`pocketllm/backends/mimo_backend.py` is the adapter: an OpenAI-compatible server over these four
ranks, one request at a time, one sequence, with the sampler and the stop conditions the HTTP
layer already parses. `pocketllm serve --backend mimo --model <checkpoint>
--tensor-parallel-size 4 --max-model-len 262144 --backend-option prefill_chunk=2048
--backend-option chunk_rows=16` puts `/health`, `/ready`, `/v1/models`, `/v1/chat/completions`,
`/v1/completions`, SSE streaming, `DELETE /v1/requests/{id}` and `/metrics` in front of it.

**The deadlock is the design constraint and not a bug in it.** Every routed layer closes with an
all-reduce at the same point in every rank's program, so the ranks are only ever concurrent by
being *symmetric*: a rank that is not running the request its peers are running is not idle, it is
at a different collective, and NCCL answers a mismatch by hanging. Rank 0 therefore may not begin
a generation it has not told the workers about — one `broadcast_object_list` a request, carrying
the prompt ids, the budget, the sampler and the seed — and a worker's whole loop is that
broadcast, one payload a request. It is also why a cancel has to be agreed rather than acted on:
`_step_sync` is a `broadcast` of one int a step, and a rank that stopped on its own flag would
leave three peers inside a layer's all-reduce. **A stop string rides the same broadcast**, and
that is not a detail: the marker is found on rank 0 and nowhere else, so an implementation that
raised out of the token callback -- which is the obvious one and the single-rank one -- would
unwind rank 0 while its peers were still in a layer. The streamer records the hit instead and the
per-step sync folds it into the flag it was already sending, so every rank leaves at the same
token boundary, one step later, with the same text already cut and sent.

The payload is the whole request and not a hint at it, because the workers have to *reproduce*
it: the same ids, the same budget, the same sampler, the same seed. They are not given the prompt
text — tokenization is rank 0's, and a rank that tokenized it itself would be a second renderer
that could disagree — and they do not need to be told what rank 0 drew, because greedy is
deterministic and the logits are the same sum on every rank.

**What was exercised on the release, four ranks, a 262144-token `--max-model-len`:**

| | Result |
| --- | --- |
| `/health`, `/ready`, `/v1/models` | ready on both routes; the model id the launcher named |
| `POST /v1/chat/completions` | "The capital of France is Paris." — `finish_reason: stop`, 19 prompt and 19 completion tokens, 8.4 s |
| `stream: true` | token deltas, a final chunk carrying `usage`, then `data: [DONE]` |
| `stream: true` with `stop: ["Paris"]` | the stream ends at the marker, `finish_reason: "stop"`, 17 completion tokens, and the text stops at "… is " — the four ranks agreed the stop rather than rank 0 leaving them |
| `POST /v1/completions` | " Paris. It is located in the north-central part of the country…" — `finish_reason: length`, 5 prompt and 24 completion tokens |
| A 5721-token prompt | three chunks of 2048, 43.0 s, a correct answer |
| `DELETE /v1/requests/{id}` mid-stream | `{"cancelled": true}` and `finish_reason: "cancelled"` on the stream, with the server still ready afterwards |
| `/metrics` | `mimo_arena_bytes 53477376`, `mimo_experts_local 64`, `mimo_experts_per_call 16`, `mimo_kv_cache_bytes 6065356800`, `mimo_world 4` |

The arena byte count is the one to read twice: 53,477,376 is 51 MiB, which is the *decode*
arena — two rows a slot — and the second arena a served model keeps is the `id` chunk arena at
408 MiB. A model that serves both has to, because the two deals are not interchangeable: a chunk
needs the experts partitioned and a decode step is 27% faster when the drawings are — 41 to 53%
on a step whose attention is split, measured — and
`forward_chunk` refuses `sorted` at a world over one rather than staging all 256 experts on every
rank. The pair costs 51 MiB of a card that has 2.4 GiB free at 256k, and the dispatch between
them is the row count of the call.

`mimo_kv_cache_bytes` is the other one: 6,065,356,800 is the whole 5.65 GiB a 262144-token cache
takes when every rank holds all of it, and the served path is what the attention split changes
without being asked — `EpGroup.from_env()` answers four at a world of four, so a served run at
that width holds **a quarter of the key heads a rank** and reports a quarter of those bytes,
1.41 GiB. The knob that holds it back is `POCKETLLM_MIMO_ATTENTION_SHARDS=1`, and there is no
reason to set it on this machine.



## Validated performance

**One token, on one card, and a floor.** The number above — 610 ms a token — is the
whole model on the release, and it is a *decode* number: the single-token expert kernel is
what a step uses, and a prompt fed through this path is a prompt fed a token at a time. There
is no batching, no tensor parallelism, and no CUDA kernel behind the attention or the
dense linears. The host reference is not a performance artifact either: it is
float32 on the CPU with no KV cache, so every decode step re-runs the whole prefix —
37 s a token against the card's 0.61 s, which is the ratio a reference is supposed
to have.

What is measured one layer at a time, on the release, bf16, warm cache, best of
three: `tests/bench_mimo_v2_attention.py`. Layer 2 is windowed, layer 5 is global.

| | Windowed (39 layers) | Global (9 layers) |
| --- | ---: | ---: |
| Prefill, 1024-token chunk, 8k context | 38.4 ms (26.6k tok/s) | 131.0 ms (7.8k tok/s) |
| Decode, 4k context | 1.93 ms | 2.71 ms |
| Decode, 64k context | 1.93 ms | 5.09 ms |
| Weights on the card | 180 MiB | 170 MiB |

A windowed layer's decode step does not grow with the context at all, which is the
ring; a global layer's grows with it and is close to the bandwidth a token's keys
cost. Summed over the model with the measured per-layer numbers, one token's attention
is **100 ms at 4k context and 122 ms at 64k** — against 64 ms measured for the whole
48-layer stack at a short context, which is the same sum and a shorter prefix. Both sums
are of the *replicated* path, and both are what the attention split below divides four ways
at the deep end: the same sum at 262144 keys is 202.7 ms whole against 90.8 ms split.

The token's other 580 ms is the copy, and the copy is a per-token quantity because
the routed path is a draw a token. Sharding the experts across four ranks was the one
lever the numbers pointed at, and it was taken: **4.68 GiB a token becomes 1.17 GiB a
rank**, and the token went from 644 to 275 ms. It did not go to the 5.6 tokens a second
that dividing the bytes by four predicted, and the reason is measured rather than argued
— a per-layer collective in this pipeline costs 1.6 ms and not the 0.128 the message
costs alone, because it is a barrier and the four ranks are not equally fast a layer.
That is also the answer to the obvious next move: tensor parallelism for the attention
would divide its 65-88 ms by four and add a collective a layer to do it, and at 1.6 ms a
collective the trade is upside down. The attention's next step is a kernel, not a split —
the same conclusion the dense stack reached on one rank, for a different reason.

**Read that paragraph against the attention-split section below, because the split was
taken and it paid.** The arithmetic was wrong in one term: 1.6 ms is what a *routed*
layer's all-reduce costs, where the message is 16 KiB and the four ranks are waiting on
each other's experts, and an attention join is the same 16 KiB with nothing waiting behind
it — measured on its own the message is **78.7 µs**, and the split is worth 4.04× on a
262144-key chunk and 4.65× on a decode step. If anything should have been revised at the
time it was the phrase *65-88 ms*: at 262144 keys the attention is 202.7 ms over the whole
stack, and a quarter of it is worth far more than a collective.

What *is* left in the 275 ms, per routed layer, is the copy (2.4 ms, at the link's
ceiling), the attention (1.5 ms) and the router (0.5 ms): the thing worth its own stage is
the attention kernel, and the split makes that sharper rather than blunter. A windowed
layer's decode step is 1.551 ms over a 128-slot ring and stays 1.551 after the split,
because that cost is launches and dispatch and not heads — and it is thirty-nine of the
forty-eight layers. Prefill was where the second half of the target lived, and the split is
what met it: 104.04 tokens a second at 262144.

Prefill is the other shape, and it was the second half of the target. Before the stage was run,
the arithmetic in this paragraph said a chunk's traffic is "about 153 GiB for the whole layer
stack no matter the chunk size" — one rank's 256 experts of 47 layers, 14.7 s at the link's
10.4 GiB/s, and the prefill's rate was read off that alone: 35 tokens a second at a 512-token
chunk and 279 at 4096 on one rank, four times that with the bytes dealt over four. The measured
answer is **93 tokens a second at 512, 134 at 1024 and 174 at 2048** on four ranks, and the
widest chunk this card held *at that stage* was 2048 — the tile's row budget is what raised
that, and the widths at and above 2048 are in the width section above.

The byte estimate was right and the extrapolation from it was not, in a way worth recording.
The per-chunk bytes really are a constant (a rank's whole share of every layer is 37.5 GiB
whether the chunk is 256 tokens or 2048), so the copy a token falls as the chunk grows — 149.8
MiB a token at 256 down to 18.7 at 2048 — but the things a token also pays that are *not* the
copy do not fall: the expert kernel and the layer's host work are 3.64 ms a token at 2048 of
which the copy is 1.7, and the attention rises with the width, 1.63 ms a token at 256 to 1.95
at 2048. A rate extrapolated from the copy alone therefore over-predicts the wider chunks the
most, and the truth is a curve that flattens just under 200 tokens a second. The width past
2048 was not a curve at all at that stage: a 4096-token chunk was refused by the attention's
score tile before the rate could be measured, which is the bound the tile budget then removed
— and removed in both directions, since 4096 turns out to be the faster width at 64k and the
slower one at 256k.

## Correctness and precision

Two independent checks, and they cover different failures.

**Per-layer parity against the checkpoint's own remote code.** A 4-layer fixture
of hidden 64 is run through `transformers` with the release's
`modeling_mimo_v2.py`, and the resulting golden holds the fixture's parameters,
its per-layer hidden states, its attention probabilities, its router scores and
every expert's input and output. `tests/test_models_mimo_v2_layer_parity.py` runs
`layers.py` against it: bit-equality of parameters and experts, and agreement of
logits and every layer output to `1e-6`. The fixture is built so that each
semantic under test is active — two attention families, a sink on one of them, V
narrower than QK, a window shorter than the sequence, both FFN kinds, and a top-k
selection the router's correction bias genuinely flips. A final test falsifies the
plausible wrong readings (an additive sink, a shared RoPE table, weighting by the
corrected score, an unnormalised top-k) so a future edit that adopts one of them
fails rather than passing quietly.

**The released tensors.** `tests/test_models_mimo_v2_loader.py` writes a
*miniature* checkpoint with the release's file names, shard convention, six-tensor
expert runs and dtypes, and checks the loader against offsets it parses itself —
including the malformed cases the loader must refuse (a shard holding another
shard's experts, an expert's tensors out of order, an expert interrupted
mid-run). The same properties are then checked on the release when it is on the
host. `tests/test_models_mimo_v2_real_weights.py` builds real layers, runs them,
and pins the three properties that make the bridge usable as a reference: the
experts stay packed, a dequantized-expert cache does not change the arithmetic,
and restricting a layer to the experts its router selected does not change its
output.

**The device path against the host reference.** Six checks, and they answer
different questions. `tests/test_models_mimo_v2_bank.py` reads every expert of the
miniature back out of the segment byte for byte and runs the same property on the
release's layout — including the shard that stores `10, 11, 8, 9`, which is the one
a bank cannot get wrong quietly. `tests/test_models_mimo_v2_device_experts.py`
reproduces the kernel's *own* arithmetic on the host, int8 activations included, to
`4e-7`; that is what separates "the quantisation costs this much" from "we read the
wrong expert", and a wrong expert is 14 to 250 percent off rather than 1e-6.
`tests/test_models_mimo_v2_device_attention.py` holds the attention to `layers.py`'s
own attention function — the reference is the oracle, not a second copy of the same
idea — and then holds the cache to what a cache is: a chunked prefill against a
one-shot one, a decode step against the same token inside the full prefix, a ring that
returns the newest window in time order, and a windowed layer that does not move when
keys outside its window are rewritten while a global layer does.
`tests/test_models_mimo_v2_device_model.py` is the assembly, on a miniature of the
release's own shape whose experts are packed fp4 at scale one — so the codes the
kernel reads and the dense tensors the host reads are the same numbers and a
disagreement is a bug rather than a rounding. It pins the two adds, the unscaled
embedding, the untied head, a stack that says when it is truncated, a cache that
reproduces the prefix it stands for to 1.7e-6, the router's draw by name, and the one
arena serving two routed layers whose layer ids travel with the call. Its release half
is two real layers — layer 0 global and dense, layer 1 windowed and routed — against
the host in float32, fed a token at a time through the cache: 2.8e-3 to 7.9e-3 on a
stream whose peak is 0.55, a logit error of a percent, and the same argmax on every
row. And `scripts/verify_mimo_v2_real_checkpoint.py` runs the whole 48-layer backbone
on the release and greedily decodes, which is the one check that a shape or an offset
error cannot pass.

**The deal, and the shares of a draw.** `tests/test_models_mimo_v2_ep.py` is sixteen
tests and needs no process group, because the property the deal has to have is
arithmetic and not communication: **four ranks' partials, summed in one process, are the
one-rank answer**, and they are summed for both deals over the same draw and the same
hidden state. A deal that dropped a drawing, double-counted one or paired a weight with
the wrong expert cannot pass it. It also pins what makes the two deals different — a
top-8 draw over four ranks is 2, 2, 2, 2 under `sorted` and can be 8, 0, 0, 0 under
`id` — the empty rank's zero, the arena width each deal needs, and that four modules
stage the draw's bytes once between them and not four times over. What it cannot test is
the collective: `make_all_reduce` is a closure around `dist.all_reduce` with nothing in
it to get wrong, and whether four *processes* agree is what the multi-rank run's
byte-for-byte logit comparison answers.

**The split, against the attention it splits.** Two tests, and they are the two ways a split
attention can be wrong. `test_the_four_shares_are_the_whole_attention` runs the four shares over
their own key heads, joins them, and holds the result to the whole layer's — one float32 ULP for a
windowed layer's folded decode step, exact for a chunk — and then asserts that a *permutation* of the
pieces is not the answer, so it cannot pass on symmetry. And
`test_the_join_writes_each_rank_down_its_own_columns` fakes `all_gather_into_tensor` to ask which
columns each rank's piece came out in, because the real one joins along the first axis where the
attention wants the last and a caller that allocates the shape it wants is silently scrambled for
every `rows` over one. `probe_mimo_v2_attention_split.py` then asks the same question of the real
four-rank collective and the released weights, and `probe_mimo_v2_split_tokens.py` asks it of a token
stream: the same greedy tokens, with the split and without, on a real prompt and on a drawn one.

**A chunk against the same rows one at a time.** `tests/test_models_mimo_v2_prefill.py` is
seventeen tests over the prefill, and the one that matters is the layout's: a chunk of six
rows through the grouped kernel against six single-token calls, on the same arena source,
must give the same answer — a `pair_weights` that followed the wrong pair, a slot that held
the wrong expert or a band that summed twice shows up there as a large disagreement rather
than a small one. They agree to **6e-8 of the answer's own peak**, which is not zero: the
grouped kernel tiles K over two stages of shared memory where the single-token one tiles it
in one, and reduces a token's pairs in its own pass, so the float32 accumulation is
reassociated. Everything else in the file is derived from that: the bands, whose one-expert
width is *bit-identical* to one band (a top-k names distinct experts, so a token's pairs in
slot order are already in ascending expert order and the adds are the same adds); the four
shares of a dealt chunk summing to the one-rank answer; the rank that owns none of a chunk's
experts returning a zero rather than an empty tensor, since the collective is unconditional
and cannot be skipped; and a dealt chunk staging a quarter of the experts, once between the
four ranks. The model half covers `mlp`'s dispatch on the row count, `prefill`'s chunking
(one row out, whatever the width, and a decode step after it landing where one pass over the
whole stream lands), the `sorted`-deal refusal at a world over one, and a chunked prompt
taking the same token as a token-at-a-time one. Its release half is two real layers — one
dense, one routed — with the chunk path over a chunk of three: **3.0e-7 of the logits' own
peak of 17.45**, and the same argmax at all six positions.

## Reproduction

```bash
# the checkpoint's own config, read through the schema
python -m src.models.mimo_v2.config /mnt/data3/MiMo-V2.6-Flash-RL

# parity, layout, the host bridge, the bank, the device experts, the attention, the model,
# the deal, the prefill and the served adapter
python -m pytest tests/test_models_mimo_v2_config.py tests/test_models_mimo_v2_quant.py \
    tests/test_models_mimo_v2_qkv_layout.py tests/test_models_mimo_v2_layer_parity.py \
    tests/test_models_mimo_v2_loader.py tests/test_models_mimo_v2_real_weights.py \
    tests/test_models_mimo_v2_bank.py tests/test_models_mimo_v2_device_experts.py \
    tests/test_models_mimo_v2_device_attention.py tests/test_models_mimo_v2_device_model.py \
    tests/test_models_mimo_v2_ep.py tests/test_models_mimo_v2_prefill.py -q
python -m pytest tests/test_mimo_serving.py -q

# what one layer's attention costs, both families, prefill and decode
python tests/bench_mimo_v2_attention.py

# what a whole token costs on one card, and how much of it is the expert copy
python tests/bench_mimo_v2_model.py

# the same token on four cards, the experts dealt out -- and the four-rank agreement check
# `--deal id` is the other deal; the flag sets POCKETLLM_MIMO_EXPERT_DEAL
torchrun --nproc_per_node=4 tests/bench_mimo_v2_ep.py --steps 8
torchrun --nproc_per_node=4 tests/bench_mimo_v2_ep.py --steps 8 --deal id

# a prompt as chunks, four ranks, the chunk width swept and the token-at-a-time floor
torchrun --nproc_per_node=4 tests/bench_mimo_v2_prefill.py \
    --tokens 4096 --chunk 256,512,1024,2048,4096 --floor 16 --band 0

# 256k, four ranks, a 2048-token chunk: 1 h 47 m of prefill after a five-minute build,
# or 42 minutes with the attention split. `POCKETLLM_MIMO_ATTENTION_SHARDS=1` is the control
# arm, and the two together are the A/B the 256k section is made of. The model is built with
# the deal a *step* takes -- `--deal` defaults to the served one -- so the `--decode` rows are
# the served configuration's and the chunk rows are the same either way; `--deal id` is the
# single-module, prefill-only build.
torchrun --nproc_per_node=4 tests/bench_mimo_v2_prefill.py \
    --tokens 262144 --chunk 2048 --band 16 --floor 0 --decode 2

# the same memory state without the prompt that reaches it -- `--capacity` sizes the cache
# apart from the prompt, which is how the tile's own step is measured in minutes
torchrun --nproc_per_node=4 tests/bench_mimo_v2_prefill.py \
    --tokens 32768 --capacity 262144 --chunk 4096 --band 16 --floor 0

# what a four-rank decode step is made of, without a profiler in the way, and with one
torchrun --nproc_per_node=4 tests/probe_mimo_v2_decode_arms.py --steps 8 --prompt 8
torchrun --nproc_per_node=4 tests/probe_mimo_v2_decode_ops.py --steps 3 --prompt 8

# the step's cost at a depth, without the prompt that reaches it: `--depth` fills the cache
torchrun --nproc_per_node=4 tests/probe_mimo_v2_decode_host.py --depth 262144

# where that step's 204.7 ms goes: the copy the kernel waited for, the attention, the kernel
torchrun --nproc_per_node=4 tests/probe_mimo_v2_decode_phases.py --steps 8 --depth 262144

# the attention's copy of its own prefix, span against `cat`, the two arms interleaved
torchrun --nproc_per_node=4 tests/probe_mimo_v2_decode_prefix.py --depth 262144 --steps 4

# whether the previous token's draw predicts this one's: a real prompt, a real greedy stream
torchrun --nproc_per_node=4 tests/probe_mimo_v2_expert_reuse.py --depth 8192 --tokens 32

# the attention split against the whole attention, bit for bit: one card joins its own
# shares with `torch.cat`, four ranks go through the real all-gather
python tests/probe_mimo_v2_attention_split.py
torchrun --nproc_per_node=4 tests/probe_mimo_v2_attention_split.py --rows 1,512

# no token moves: the same greedy stream with the split and without, from two processes
torchrun --nproc_per_node=4 tests/probe_mimo_v2_split_tokens.py --depth 8192 --tokens 16 \
    --random-prompt --out /tmp/tokens_split.txt
POCKETLLM_MIMO_ATTENTION_SHARDS=1 torchrun --nproc_per_node=4 \
    tests/probe_mimo_v2_split_tokens.py --depth 8192 --tokens 16 --random-prompt \
    --out /tmp/tokens_whole.txt
diff /tmp/tokens_split.txt /tmp/tokens_whole.txt

# the served endpoint: chat, completions, streaming, cancel and metrics on four ranks
pocketllm serve --backend mimo --model /mnt/data3/MiMo-V2.6-Flash-RL \
    --tensor-parallel-size 4 --max-model-len 262144 --host 0.0.0.0 --port 8300 \
    --served-model-name mimo-v2.6-flash \
    --backend-option prefill_chunk=2048 --backend-option chunk_rows=16

# the whole backbone on the release, decoded on the CPU
python scripts/verify_mimo_v2_real_checkpoint.py --tokens 8
```

The four-rank run takes about two and a half minutes of wall clock: 84 s to build the
model (four processes mapping the release at once), then 31 s of page registration per
rank, both of which happen concurrently, and then the measured region. A rank that
dies leaves rank 0 blocked in its final `all_gather` — the gathers are participated in
by every rank and printed on one, precisely so that a run fails rather than hangs when
it can — and the process to kill is the rank's `python`, not the `torchrun` wrapper.

The bank is 149.81 GiB of `/dev/shm` and takes twelve minutes to fill the first
time; a run after that attaches in 0.07 s, and four ranks attach to the *same*
segment rather than filling four. `rm -rf
/dev/shm/pocketllm_mimo_experts_*` is how the memory goes back, and the next
`open_expert_bank` refills it. Four ranks that each register the whole mapping take
31 s apiece and do it concurrently, which is 31 s and not 124: the pages are the
same pages and the driver counts them once.

The oracle fixture lives outside this repository (a checkout without it skips the
parity tests); `tests/test_models_mimo_v2_layer_parity.py` documents what the
golden holds and how it was captured.

## Known limitations

- **A chunk's width is capped by the attention's score tile, and the tile's step is a constant.**
  The block loop materialises `[kv_heads, groups, rows, block]` float32 of scores and a global
  layer's late key blocks are visible to the whole chunk; that used to cap a chunk at 2048 tokens,
  and the rows of a slice are now stepped against a fixed 2²⁶-score budget so it no longer does.
  What the constant costs is that it cannot be right at both ends of the context range: it buys a
  64k prompt two steps and a 2048-row chunk at 256k the same two, where a per-rank reading of free
  memory bought a better step on each card and made the four ranks disagree — 28% of a 256k prompt.
  A smaller `block`, a tile that is not float32, or a two-sided tiling would move all of it; the nine
  global layers are the whole of it, and a windowed layer's tile is fixed because its query slice is
  a block plus a window.
- **The width that is fastest is not monotone in the context, and the tile is why.** 4096-token
  chunks are the best at 64k (104.4 against 95.5 at 2048) and did not finish a 256k prompt in four
  hours where 2048 finished in under two. A fixed step does not explain that on its own — a wider
  chunk pays more of the attention, which at 256k is 15.1 ms of every token — and the honest
  statement is that the width was swept at 64k and at a 32k prompt against a 256k cache, and not at
  256k depth itself. Sweep `--chunk` per context; the served configuration is 2048. **Every one of
  those widths was measured with the attention replicated**, and the split below makes the attention
  3.5× cheaper a token, so the 256k width is worth sweeping again before the 2048 default is trusted.
- **A prefill and a decode want different deals, and the deal is per module.** A chunk needs
  the experts partitioned (`id`) and a decode step is faster when the *drawings* are (`sorted`,
  the default): 27% on the replicated attention this page first measured it on, and **41 to 53%
  once the attention is split** — 179.0 against 253.2 ms at eight tokens of context and 177.0
  against 270.8 at 32768. A serving run keeps **both arenas** — 51 MiB for the decode deal and
  408 MiB for the chunk deal at a band of 16 — and dispatches on the row count of the call, which
  is two stores of a rank's share where one would do and 51 MiB of a card that has 2.4 GiB free at
  256k. The alternative is one deal and that much off either a step or a prefill, and it is why
  `bench_mimo_v2_prefill.py` builds the served pair by default.
- **The attention is split and nothing else is.** The experts are dealt over four ranks and the
  attention is divided along the checkpoint's own four-way `qkv_proj` partition, joined by an
  exact all-gather. What is left replicated is the router, the embedding, the head and the dense
  linears — and the last of those is 250 MiB of weights read off the card a token, against a
  collective a layer to divide it, which is why nobody has paid for one. A world that is not four
  keeps the whole attention on every rank, because four is the count the weights admit; a world
  that is four and wants the control arm sets `POCKETLLM_MIMO_ATTENTION_SHARDS=1`.
- **One sequence, one request, no batching.** A second request would have to wait; the
  KV cache, the expert arena and the collectives are all single-sequence. The HTTP layer accepts
  one at a time and the second blocks on a lock.
- **The attention and the dense linears are torch, not kernels.** They are baselines with the
  shapes the kernels have to beat, and the split moved the baseline without moving the conclusion:
  at 262144 on four ranks the attention is **38.8 ms of a 197.2 ms decode step** — a fifth of a
  token, and 44.3 before the prefix copy above went — and the thirty-nine windowed layers are why,
  because a 128-slot ring's decode step costs what a global layer's does: what is left there is
  launches and dispatch and not keys. `probe_mimo_v2_attention_split.py`'s per-layer table is the
  split's *ratios* (4.65× on a global layer's decode step at that depth, 3.68× on a windowed layer's
  chunk) and its absolute milliseconds are three to four times the in-situ column, taken with one
  layer and one card in the loop rather than in the model; the ratios are what the split's claims
  rest on. A kernel is the only thing that reaches the dispatch, and it is the next stage.
- **256k runs, and the decode at that depth is at the target rather than past it.** A 262144-token
  prompt goes through at **104.04 tokens a second**, which is past the hundred this page was written
  against, and a decode step at that depth is **197.2 ms — 5.07 tokens a second — against 323.8
  replicated**, in the step probe that measures the deal a served run's step is built with. Five a
  second was the number this was aiming at and the margin over it is 1.4%, which is small because the
  step is a copy at the link's rate plus everything else in series and both are at their floors: the
  copy is 99.9 ms of the 197.2 — 1198.5 MiB a token at 12.0 GiB/s, which is a PCIe 3.0 x16 link at
  essentially its rate — and hiding it behind the layer's other 97 ms needs a prediction, while
  staging fewer bytes needs fewer experts than the draw's `top_k / world`. What is left to take off
  the step is therefore the dispatch, and the attention is 39 of its 48 layers. A prefill bench's
  `--decode` column read 281.2 for this depth before there was a flag, which is a different deal's
  number; it takes `--deal` now and defaults to the served build.
- **Greedy only, and no sampler.** `argmax`, stopping at the config's own end-of-turn
  tokens, with no temperature, top-p or repetition penalty. The checkpoint's
  `generation_config.json` says `do_sample: false`, so this is its own default — but a
  sampling path is what a serving stack would need, and the logit agreement above is
  the reason a sampler matters: at a tenth-of-a-logit near-tie the card and the host
  draw different tokens, which a distribution-aware sampler absorbs and `argmax` does
  not.
- **No KV cache in the host reference.** Re-running the prefix is deliberate for a
  reference — it is why its numbers can be trusted and why they are 37 s a token —
  but it also means the host cannot be run at a long context to check the device's
  cache at one.
- **Serving is one request, one sequence, no prefix cache and no scheduler.** The adapter is an
  OpenAI-compatible schema over the single-sequence loop in `src/models/mimo_v2/generate.py`, with
  a cancel that is agreed across the ranks and a stop string that is agreed for the same reason. A
  second request waits on a lock rather than being scheduled, a repeated prefix is prefilled again,
  and a cancelled request's KV cache is reset rather than reused. What that buys is a deployment;
  what it does not is concurrency.
- **MTP and DFlash are not executed.** The 3-layer MTP module and the 5-layer
  DFlash drafter are located and described but no speculative path uses them.
- **Vision and audio are out of scope.** The vision tower and the audio encoders
  are part of the checkpoint and are not read.

## Evidence and related notes

- `src/models/mimo_v2/` — the host reference and the pieces of the device path: the bank,
  the routed experts, the attention, the model, the deal over the ranks and the chunk path.
- `tests/test_models_mimo_v2_qkv_layout.py` — the fused projection's row order,
  which the config does not carry and which no shape check can catch.
- `tests/test_models_mimo_v2_layer_parity.py` — the oracle fixture, its contents,
  and what parity means at fixture scale.
- `tests/test_models_mimo_v2_loader.py`, `tests/test_models_mimo_v2_real_weights.py`
  — the checkpoint's layout and the host bridge, on the release.
- `src/models/mimo_v2/bank.py`, `src/models/mimo_v2/device_experts.py`,
  `src/models/mimo_v2/device_attention.py`, `src/models/mimo_v2/device_model.py`,
  `src/models/mimo_v2/ep.py` — the device path, and the measurements in this page.
  `ep.py` carries the two deals and the arithmetic that picks one; it is the only file
  in the tree whose *default* was set by a four-rank measurement.
- `tests/bench_mimo_v2_attention.py`, `tests/bench_mimo_v2_model.py`,
  `tests/bench_mimo_v2_ep.py`, `tests/bench_mimo_v2_prefill.py` — where the attention table,
  the one-rank token table, the four-rank table, the chunk-width table and the 256k numbers come
  from. `bench_mimo_v2_prefill.py`'s `--capacity` sizes the KV cache apart from the prompt, which
  is how the tile's step in a long context is measured in minutes instead of hours.
- `tests/probe_mimo_v2_decode_arms.py` — a decode step with the routed experts replaced by the
  zero an empty rank returns, which is what separates the 38.0 ms a routed path costs *in situ*
  from the 139.6 the rest of a layer does. It is deliberately the only arm: an arm that also
  zeroed the attention was tried and removed, because the residual stream then stops moving,
  every layer routes on the same row, and the subtraction is between two different models.
- `tests/probe_mimo_v2_attention_split.py` — the attention split against the whole attention
  bit for bit, one card and four ranks: a share's projection rows, the joined output, and the
  per-layer whole-against-share timings the split's table is made of. It is also where the one
  inexact row is nailed down to a batched `cuBLAS` tiling and not to the split.
- `tests/probe_mimo_v2_split_tokens.py` — the token stream, real prompt and drawn prompt, with
  the split and without, printing the top-2 logit margin beside the tokens because a stream that
  agrees says very little on its own.
- `tests/probe_mimo_v2_decode_ops.py`, `tests/probe_mimo_v2_decode_phases.py`,
  `tests/probe_mimo_v2_decode_timeline.py` — the profiler's own tables, the per-phase split, and
  the busy union of the device's intervals. The ops probe's first table is the one that found the
  145 `cudaStreamSynchronize` calls; the timeline probe reads `prof.events()` rather than
  exporting a chrome trace, because a 13 MB trace of this step is truncated JSON. The phases probe
  is what separates the expert copy the kernel *waited* for from the copy it hid, and its two arms
  are what the decode step's decomposition above is made of.
- `tests/probe_mimo_v2_decode_prefix.py` — the attention's read of its own span, as a view of the
  cache buffer against the `cat` it replaces, the two arms interleaved and four steps each at
  262144: 197.2 ms a token against 202.3, the attention column 38.8 against 44.3. The `cat` arm is
  the shipped method answering `None`, which is what a wrapped ring answers, so neither arm is a
  patch of the arithmetic.
- `tests/probe_mimo_v2_expert_reuse.py` — whether a token's draw predicts the next one's, per
  layer and per rank, on 8192 tokens of a real document tokenized by the checkpoint's own tokenizer
  and 32 greedy steps of it. It is the measurement that closes the prefetch: a row keeps its expert
  nine to thirteen and a half times in a hundred, and the rank's whole set repeats in 1.9 to 2.4%.
- `tests/probe_mimo_v2_allreduce.py` — the 16 KiB all-reduce on its own, with and without a kernel
  between the messages, at three message sizes, and under `NCCL_P2P_DISABLE=1`. It is where the
  intra-layer **1.6 ms** against the bare **0.126 ms** comes from, and what it says about the
  message itself is that it is latency and not bandwidth: 16 KiB costs **78.7 µs** and 256 KiB —
  sixteen times the bytes — costs **80.1 µs**, while 16 MiB costs 2759.7 µs at 6.08 GB/s, which is
  PCIe's own number and not a ring. Disabling the peers' P2P paths makes the same 16 KiB message
  *faster* (93.3 µs a message in a chain, against 125.8), so the hop was already the host's shared
  memory and the twelve-times gap is the lockstep's and not the transport's.
- `tests/bench_mimo_v2_staging_copy.py` — `_stage` writes one expert as six `copy_` calls because
  the arena holds `w1, w2, w3` separately and the bank holds an expert as one record, and a draw is
  therefore twelve copies of about 2 MiB. This moves the same 25.5 MiB as twelve, six, three, two
  and one copy, and the answer is that the split is free: **2.477 ms against 2.371, 1.04x, and
  0.106 ms of a copy that is 2.4** — all five at 10.05 to 10.50 GiB/s, which is the link's own
  rate. A row-per-piece arena is therefore not worth the layout work, and the copy stall the decode
  profile reports is the link and not the launch count.
- `tests/test_mimo_serving.py` — the adapter without a checkpoint or a process group: the
  payload rank 0 broadcasts, asserted key for key and not as a subset, that only rank 0
  dispatches, and that a worker runs what it was sent.
- `pocketllm/backends/mimo_backend.py` — the served adapter, the one broadcast a request that
  keeps the ranks symmetric, and the cancel and the stop string that have to be collectives;
  `src/models/mimo_v2/generate.py` is the prompt-and-answer loop it drives, kept out of the adapter
  so two adapters cannot disagree about what `max_new_tokens` means.
- `src/models/mimo_v2/device_experts.py:forward_chunk` — the chunk layout, why `bincount` is
  not in it, and what the bands are; `src/models/deepseek_v4_1/device_experts.py:_issue_chunk`
  is the same call in the other heterogeneous path in this tree, with the 3.04x/3.69x
  batched-against-per-row measurement that made it a swap there.
- `src/models/deepseek_v4_1/tp.py` — the same collectives and the same
  injected-closure shape for the other heterogeneous path in this tree.
- The support matrix in [models/README.md](README.md).
