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
one-rank run gives. What the ranks do *not* divide is the attention, which is
replicated on all four and is the next stage's work.

**A prompt is a chunk now, and not a token loop.** `forward_chunk` takes a chunk of
tokens through `moe_multi_token_fp4_forward` — the kernel the V4.1 path already called
and this one did not — whose arena holds a rank's *share* of a layer's experts and
whose pairs are the chunk's drawings grouped by expert, and `prefill` runs a prompt in
chunks of the caller's width. On four ranks a 4096-token prompt goes through at **134
tokens a second at a 1024-token chunk and 174 at 2048**, against **2.93** for the same
prompt fed one token at a time: 46 and 59 times the rate on the same weights, with the
four ranks byte-identical on the prompt's last row. The width is the knob, and what
caps it is the attention's block loop rather than the link — a 4096-token chunk runs
out of card.

What that is not: 256k, serving, or a fast attention. The chunk path is a prompt
through the same torch attention, which is 1.7 to 2 ms a token of the prefill and
replicated on every rank; the cache is sized for 256k and nothing has run there; and
there is no OpenAI-compatible adapter, no batching and no sampler.

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
| Expert parallelism over four ranks | Implemented and verified: 3.63 tok/s, 275 ms a token, four ranks byte-identical and the same tokens as one rank |
| Chunked prefill, grouped multi-token expert kernel | Implemented and verified: 134 tok/s at a 1024-token chunk and 174 at 2048 on four ranks, 46-59x the token-at-a-time loop, four ranks byte-identical |
| 256k context | Not run — the cache is sized for it and nothing has executed at that length |
| OpenAI-compatible serving | Not implemented |
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
counts.

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
staging all of them). On decode, `sorted` — now the default — is **21% faster**:

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
| Router | 0.5 | the gate and the top-k, whose result the host needs |
| Expert copy | 2.4 | 25.5 MiB, at the link's own 10.4 GiB/s |
| Expert kernel | 0.4 | two experts instead of eight |
| The collective | **1.6 in situ** | against **0.128 back to back** |
| The rest | 0.8 | the norms, the adds, the head, the host's own issue |

Read the last two rows together: the same message costs **1.6 ms inside a layer and 0.128 ms
in a loop**, a factor of twelve, and the twelve is the *lockstep* rather than the message. A
collective is a barrier, four ranks' per-layer host work is not equal, and every rank pays
the slowest one's time; the `id` deal doubles that by making the copy itself unequal. It is
why the imbalance above is worth 21%, and it is the measurement that redraws the next
stage's target: the step is now attention plus copy plus a lockstep, and 48 x 5.7 ms is the
275 the `sorted` arm measures.

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
build names `deal="id"` while decode keeps the `sorted` default that is 21% faster there.
The deal is per *module* and not per call, so a serving run that has to do both chooses, and
the choice costs the decode step the 21%.

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

**There is a ceiling and it is not the link.** A 4096-token chunk does not fit: the attention's
block loop materialises `[kv_heads, groups, rows, block]` float32 of scores, and a *global*
layer's late key blocks are visible to the whole chunk, so its tile is 4 × 16 × 4096 × 1024 × 4
bytes — 1 GiB, on a card that is already holding 14.3 GiB of weights and 1.6 GiB of arena.
A windowed layer does not do this: its query slice is a block plus a window, so its tile is
fixed. The nine global layers are what caps the chunk width, and the fix is in that loop
rather than in the expert path — a smaller block, a tile that is not float32, or a two-sided
tiling — which is why the ceiling is recorded here rather than worked around.

**What the four ranks agree on.** The prefill's last row, gathered from all four after the
widest chunk, is byte-identical — `0.0` between rank 0 and every other — and each rank staged
18.7 MiB a token, the same number, because under `id` the shares differ in *which* experts
they are and not in how many. The one-rank comparison in the tests is the check that it stays
that way: a chunk is the same answer as the same rows one at a time to **3.0e-7** of the
logits' own peak on two released layers, and takes the same token at every position.


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
48-layer stack at a short context, which is the same sum and a shorter prefix.

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

What *is* left in the 275 ms, per routed layer, is the copy (2.4 ms, at the link's
ceiling), the attention (1.5 ms, replicated four times over) and the router (0.5 ms): the
two things worth their own stage are the attention kernel and the 10% the lockstep takes
back. Prefill is where the second half of the target lives and it is untouched.

Prefill is the other shape, and it was the second half of the target. Before the stage was run,
the arithmetic in this paragraph said a chunk's traffic is "about 153 GiB for the whole layer
stack no matter the chunk size" — one rank's 256 experts of 47 layers, 14.7 s at the link's
10.4 GiB/s, and the prefill's rate was read off that alone: 35 tokens a second at a 512-token
chunk and 279 at 4096 on one rank, four times that with the bytes dealt over four. The measured
answer is **93 tokens a second at 512, 134 at 1024 and 174 at 2048** on four ranks, and the
widest chunk this card holds is 2048.

The byte estimate was right and the extrapolation from it was not, in a way worth recording.
The per-chunk bytes really are a constant (a rank's whole share of every layer is 37.5 GiB
whether the chunk is 256 tokens or 2048), so the copy a token falls as the chunk grows — 149.8
MiB a token at 256 down to 18.7 at 2048 — but the things a token also pays that are *not* the
copy do not fall: the expert kernel and the layer's host work are 3.64 ms a token at 2048 of
which the copy is 1.7, and the attention rises with the width, 1.63 ms a token at 256 to 1.95
at 2048. A rate extrapolated from the copy alone therefore over-predicts the wider chunks the
most, and the truth is a curve that flattens just under 200 tokens a second — until the
attention's score tile runs out of card, which is what 4096 does.

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
# the deal and the prefill
python -m pytest tests/test_models_mimo_v2_config.py tests/test_models_mimo_v2_quant.py \
    tests/test_models_mimo_v2_qkv_layout.py tests/test_models_mimo_v2_layer_parity.py \
    tests/test_models_mimo_v2_loader.py tests/test_models_mimo_v2_real_weights.py \
    tests/test_models_mimo_v2_bank.py tests/test_models_mimo_v2_device_experts.py \
    tests/test_models_mimo_v2_device_attention.py tests/test_models_mimo_v2_device_model.py \
    tests/test_models_mimo_v2_ep.py tests/test_models_mimo_v2_prefill.py -q

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

- **A chunk's width is capped by the attention's score tile, not by the link.** The block
  loop materialises `[kv_heads, groups, rows, block]` float32 of scores and a global layer's
  late key blocks are visible to the whole chunk, so a 4096-token chunk needs a 1 GiB tile on
  a card already holding 14.3 GiB of weights and 1.6 GiB of arena, and does not fit. 2048 is
  the widest this card holds, the prefill's rate flattens under 200 tokens a second before
  that, and a smaller block, a narrower score tile or a two-sided tiling against the query
  rows is what would move the ceiling. The nine global layers are the whole of it; a windowed
  layer's tile is fixed, because its query slice is a block plus a window.
- **A prefill and a decode want different deals, and the deal is per module.** A chunk needs
  the experts partitioned (`id`) and a decode step is 21% faster when the *drawings* are
  (`sorted`, the default). A serving run that has to do both picks one: `id` costs the decode
  step 73 ms a token, and `sorted` at a world over one cannot prefill at all — the chunk path
  refuses it rather than staging all 256 experts on every rank.
- **No tensor parallelism, no batching, and the attention is replicated.** The experts
  are divided over four ranks and everything else is not: the router, the attention, the
  embedding, the head and the dense layer each run four times over. The checkpoint's fused
  projection is stored in four shards, so a rank that did not de-interleave would read a
  quarter of every head. Measured, splitting the attention is now the wrong trade — a
  collective a layer costs 1.6 ms against the 0.4 ms a quarter of the attention's reads
  would save — so what is missing there is a kernel, not a rank.
- **One sequence, one request, no batching.** A second request would have to wait; the
  KV cache, the expert arena and the collectives are all single-sequence.
- **The attention and the dense linears are torch, not kernels.** They are baselines
  with the shapes the kernels have to beat: 65-88 ms a token for all 48 layers on four
  ranks, and the single-pass decode path loses a factor of twenty-eight to a batch
  dimension of one until the key count passes a window.
- **No 256k run.** The cache is sized for it and the attention's bounds are derived
  from a position rather than from a mask, so nothing in the path is per-context — but
  nothing has executed at that length, and at 262144 a global layer's decode step reads
  256k keys nine times over.
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
- **No serving.** No OpenAI-compatible adapter, no batching, no prefix caching. The
  generation loop in `device_model.py` is greedy and single-request.
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
  the one-rank token table, the four-rank table and the chunk-width table come from.
- `src/models/mimo_v2/device_experts.py:forward_chunk` — the chunk layout, why `bincount` is
  not in it, and what the bands are; `src/models/deepseek_v4_1/device_experts.py:_issue_chunk`
  is the same call in the other heterogeneous path in this tree, with the 3.04x/3.69x
  batched-against-per-row measurement that made it a swap there.
- `src/models/deepseek_v4_1/tp.py` — the same collectives and the same
  injected-closure shape for the other heterogeneous path in this tree.
- The support matrix in [models/README.md](README.md).
