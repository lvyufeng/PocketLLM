# MiMo-V2.6-Flash

## Runtime status

**Inspect only.** PocketLLM can open the released checkpoint, derive the expert
layout from the shards' own headers, dequantize its three weight formats, and run
the full 48-layer text backbone on the host — but there are **no device kernels
and no server adapter for this checkpoint yet**. Nothing in this page is a
performance claim about the GPU path, because there is no GPU path.

What exists:

| Capability | State |
| --- | --- |
| Checkpoint headers, expert layout, dense-key inventory | Implemented and tested against the release |
| Per-layer CPU parity against the checkpoint's own remote code | Implemented, 21 tests |
| MXFP4 / FP8-block / BF16 dequantizers | Implemented as torch references |
| Full 48-layer backbone on the release, on the host | Implemented and verified: 2.09 nats/token on an English passage against a uniform floor of 11.94, with grammatical greedy continuations |
| KV cache for the host reference | Not implemented — the prefix is re-run every step |
| Device (CUDA) kernels for any part of the model | Not implemented |
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

`src/models/mimo_v2/` is the whole text model, and it is a reference rather than a
runtime:

| Module | What it is |
| --- | --- |
| `config.py` | The schema and the derived geometry: `qkv_out`, per-family KV heads, RoPE dimension and base, window, sink, value scale |
| `layers.py` | The host implementation of one decoder layer and of the backbone |
| `quant.py` | The three storage layouts as torch references (E2M1 codebook, E8M0, MXFP4 unpack, FP8 block dequant) |
| `loader.py` | Header-level access to the release: the expert map, byte ranges, dense tensors, MXFP4 views |
| `weights.py` | The bridge from the shards into `layers.py`, including experts that stay packed until selected |

The routed experts are never expanded. A layer's 256 experts are 3.2 GiB dense
and the model's would be 4.7 TB; `MimoV2Mxfp4Experts` holds the checkpoint's own
uint8 views and dequantizes the eight experts a token actually selects.

## Validated performance

None. There is no device path, and the host reference is not a performance
artifact: it is float32 on the CPU with no KV cache, so every decode step re-runs
the whole prefix and re-reads every selected expert from the checkpoint. It exists
to be the oracle a kernel port is diffed against.

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

`scripts/verify_mimo_v2_real_checkpoint.py` runs the whole 48-layer backbone on
the release and greedily decodes, which is the one check that a shape or an offset
error cannot pass.

## Reproduction

```bash
# the checkpoint's own config, read through the schema
python -m src.models.mimo_v2.config /mnt/data3/MiMo-V2.6-Flash-RL

# parity, layout and the host bridge
python -m pytest tests/test_models_mimo_v2_config.py tests/test_models_mimo_v2_quant.py \
    tests/test_models_mimo_v2_qkv_layout.py tests/test_models_mimo_v2_layer_parity.py \
    tests/test_models_mimo_v2_loader.py tests/test_models_mimo_v2_real_weights.py -q

# the whole backbone on the release, decoded on the CPU
python scripts/verify_mimo_v2_real_checkpoint.py --tokens 8
```

The oracle fixture lives outside this repository (a checkout without it skips the
parity tests); `tests/test_models_mimo_v2_layer_parity.py` documents what the
golden holds and how it was captured.

## Known limitations

- **No device path.** Nothing here is a kernel, and no performance number on this
  page describes anything but the CPU reference.
- **No KV cache in the host reference.** Re-running the prefix is deliberate for a
  reference, and it makes long-context work on the host quadratically expensive.
  256k context is a target of the device path, not something the reference
  demonstrates.
- **No serving.** No OpenAI-compatible adapter, no batching, no prefix caching.
- **MTP and DFlash are not executed.** The 3-layer MTP module and the 5-layer
  DFlash drafter are located and described but no speculative path uses them.
- **Vision and audio are out of scope.** The vision tower and the audio encoders
  are part of the checkpoint and are not read.

## Evidence and related notes

- `src/models/mimo_v2/` — the reference implementation.
- `tests/test_models_mimo_v2_qkv_layout.py` — the fused projection's row order,
  which the config does not carry and which no shape check can catch.
- `tests/test_models_mimo_v2_layer_parity.py` — the oracle fixture, its contents,
  and what parity means at fixture scale.
- `tests/test_models_mimo_v2_loader.py`, `tests/test_models_mimo_v2_real_weights.py`
  — the checkpoint's layout and the host bridge, on the release.
- The support matrix in [models/README.md](README.md).
