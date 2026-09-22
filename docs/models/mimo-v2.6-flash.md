# MiMo-V2.6-Flash

## Runtime status

**Heterogeneous, one rank, one sequence.** PocketLLM opens the released
checkpoint, derives the expert layout from the shards' own headers, runs the full
48-layer text backbone on the host as a reference, and **runs that same backbone
on one RTX 2080 Ti**: the dense stack and the attention out of the released FP8
weights, the routed experts out of a host-resident bank, one token a step through
a KV cache, to logits. A token costs **610 ms** — 1.64 tokens a second — of which
74% is the expert copy, and the argmax agrees with the float32 host reference at
full depth.

What that is not: a prefill. The routed path has only the single-token kernel, so
a prompt is fed one token at a time and a chunk of tokens has no path at all. Every
number below is a decode number, and the one prefill-shaped number in this page is
a floor rather than a prefill. There is no tensor or expert parallelism, no 256k
run, and no CUDA kernel behind the attention or the dense linears — those are
torch, and they are the baselines the kernels have to beat.

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
| Decode at more than one token, grouped prefill | Attention only: the cache serves a stream of chunks, and the routed path is single-token |
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
| `device_experts.py` | One layer's routed experts computed on a card, staged from the bank as they are drawn |
| `device_attention.py` | One layer's attention on a card, both families, against a per-layer KV cache |
| `device_model.py` | The backbone on a card: embedding, forty-eight layers, final norm, head, one token a step |

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
next stage's expert parallelism is for.

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
single-token loop run over a prompt, so they are a **floor** and not a prefill —
prefill needs the multi-token kernel, which nothing routes into yet. And the
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

## Validated performance

**One token, on one card, and a floor.** The number above — 610 ms a token — is the
whole model on the release, and it is not a prefill: the single-token expert kernel
is the only routed path there is, so a prompt is fed a token at a time. There is no
batching, no tensor parallelism, and no CUDA kernel behind the attention or the
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
the routed path is a draw a token. Sharding the experts across four ranks is the one
lever the numbers point at: **4.68 GiB a token becomes 1.17 GiB a rank**, which at
the 10.4 GiB/s the link sustains is about 113 ms, and 113 plus the 64 the attention
costs is **5.6 tokens a second** with nothing else changed. Sharding the dense stack
as tensor parallelism with it divides the attention's reads by four and puts a token
at about 130 ms, which is 7.7 a second — so the target's 5 is reachable by parallel
work alone, and the attention kernel this page keeps saying is missing is what buys
the margin above it.

Prefill is the other shape, and its arithmetic is not per token. A chunk of a few
thousand tokens draws nearly every expert of every layer — 4096 tokens at top-8 is
32768 draws over 256 experts — so a chunk's traffic is **about 153 GiB for the whole
layer stack no matter the chunk size**, and it is not paid again by the next chunk.
At 10.4 GiB/s that is 14.7 s a chunk, which is 35 tokens a second at a 512-token
chunk and **279 at 4096**. So the prefill target is a matter of chunking and of the
multi-token kernel that has to exist to compute a chunk at all — not of the link,
which the experts are already as close to as storage allows.

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

**The device path against the host reference.** Five checks, and they answer
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

## Reproduction

```bash
# the checkpoint's own config, read through the schema
python -m src.models.mimo_v2.config /mnt/data3/MiMo-V2.6-Flash-RL

# parity, layout, the host bridge, the bank, the device experts, the attention and the model
python -m pytest tests/test_models_mimo_v2_config.py tests/test_models_mimo_v2_quant.py \
    tests/test_models_mimo_v2_qkv_layout.py tests/test_models_mimo_v2_layer_parity.py \
    tests/test_models_mimo_v2_loader.py tests/test_models_mimo_v2_real_weights.py \
    tests/test_models_mimo_v2_bank.py tests/test_models_mimo_v2_device_experts.py \
    tests/test_models_mimo_v2_device_attention.py tests/test_models_mimo_v2_device_model.py -q

# what one layer's attention costs, both families, prefill and decode
python tests/bench_mimo_v2_attention.py

# what a whole token costs on one card, and how much of it is the expert copy
python tests/bench_mimo_v2_model.py

# the whole backbone on the release, decoded on the CPU
python scripts/verify_mimo_v2_real_checkpoint.py --tokens 8
```

The bank is 149.81 GiB of `/dev/shm` and takes twelve minutes to fill the first
time; a run after that attaches in 0.07 s. `rm -rf
/dev/shm/pocketllm_mimo_experts_*` is how the memory goes back, and the next
`open_expert_bank` refills it.

The oracle fixture lives outside this repository (a checkout without it skips the
parity tests); `tests/test_models_mimo_v2_layer_parity.py` documents what the
golden holds and how it was captured.

## Known limitations

- **No prefill.** The routed path has only the single-token kernel, so a prompt is
  fed a token at a time and a chunk of tokens has no path at all. The prefill-shaped
  columns in this page are that same loop, and the 279 tokens a second the link could
  sustain at a 4096-token chunk is arithmetic rather than a measurement. Prefill is
  what the 100 tps target is about and it is the next stage's work.
- **One rank, one sequence, no batching.** No tensor parallelism, no expert
  parallelism, no batching. The checkpoint's fused projection is stored in four shards,
  so a rank that did not de-interleave would read a quarter of every head — and the
  experts are drawn eight a token by whoever is running that token, which is the whole
  of the 4.68 GiB.
- **The attention and the dense linears are torch, not kernels.** They are baselines
  with the shapes the kernels have to beat: 64 ms a token for all 48 layers, and the
  single-pass decode path loses a factor of twenty-eight to a batch dimension of one
  until the key count passes a window.
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

- `src/models/mimo_v2/` — the host reference and the four pieces of the device path:
  the bank, the routed experts, the attention, and the model.
- `tests/test_models_mimo_v2_qkv_layout.py` — the fused projection's row order,
  which the config does not carry and which no shape check can catch.
- `tests/test_models_mimo_v2_layer_parity.py` — the oracle fixture, its contents,
  and what parity means at fixture scale.
- `tests/test_models_mimo_v2_loader.py`, `tests/test_models_mimo_v2_real_weights.py`
  — the checkpoint's layout and the host bridge, on the release.
- `src/models/mimo_v2/bank.py`, `src/models/mimo_v2/device_experts.py`,
  `src/models/mimo_v2/device_attention.py`, `src/models/mimo_v2/device_model.py` —
  the device path, and the measurements in this page.
- `tests/bench_mimo_v2_attention.py`, `tests/bench_mimo_v2_model.py` — where the
  attention table and the token table come from.
- The support matrix in [models/README.md](README.md).
