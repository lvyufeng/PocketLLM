# Auditing the DeepSeek-V4.1-Flash shards while the checkpoint is arriving

**Date:** 2026-09-16
**Commit:** `b7de882` on `feature/v41-shard-inventory`
**Script:** `scripts/audit_dsv41_headers.py`
**Artifacts:** `/tmp/v41_ckpt.json`, `/tmp/v41_hdr.json` from the two commands in
[Reproducing this record](#reproducing-this-record)
**Checkpoint:** `deepseek-ai/DeepSeek-V4.1-Flash`, 48 shards, 475.24 GiB
**Related:** [DeepSeek-V4.1-Flash](../models/deepseek-v4.1-flash.md)

This is not a throughput measurement, so most of the [benchmarking metadata
rules](../guides/benchmarking.md) do not apply: nothing here generates a token, so there is
no prompt length, no generated-token count, no prefill/decode wall time, no warm or cold
state, and no token parity to report. The items that do apply — checkpoint and variant,
commit, runtime, hardware, and the exact command — are stated. **No V4.1 throughput, latency
or memory figure is measured or implied by anything on this page**, and no weight payload was
read to produce it.

## The question this answers

The audit was written against 48 header prefixes because the checkpoint was not on this
host. It now is, in part: the release is 48 shards and the download is still running, so
there is a second, larger question the prefix tree could never ask — **what do the real
shards say, and can the audit tell a shard that is missing from a shard that is wrong?**

That distinction is the whole design. A check has three outcomes rather than two:

- **passed** — the evidence is local and agrees with the config;
- **failed** — the evidence is local and disagrees, or `model.safetensors.index.json` says
  the checkpoint ships a shard's worth of tensors the header contradicts;
- **undecided** — the evidence sits in a shard that has not been downloaded.

Presence and shape come from different places on purpose. The index names the shard holding
each of the 96,085 tensors, so *is this tensor in the checkpoint* is decidable before any
payload arrives; shapes, dtypes and byte extents come from the headers, so they are only
decidable per shard. Undecided is counted apart from passed and is never reported as a pass,
and `--require-complete` turns every undecided into a failure for callers that need a
binary answer. A failure is never deferred: a bad shape inside a shard that *is* on disk
fails while 28 other shards are still missing.

## Measurement conditions

| Item | Value |
| --- | --- |
| Host | x86_64 CUDA machine |
| GPU | none used; the audit never calls a device API |
| CPU / RAM | 2 × Intel Xeon E5-2696 v4, 2 NUMA nodes, ~1 TiB |
| OS / Python | Ubuntu 22.04.5, kernel 5.15, CPython 3.10.10 (conda `deepseek`) |
| Dependencies | standard library only — no `torch`, `safetensors` or `numpy` |
| Checkpoint | `/mnt/data3/DeepSeek-V4.1-Flash`, 20 of 48 shards on disk at the time of the run |
| Header tree | `/tmp/dsv41`, 48 × `h0000N.bin`, 3,000,001 B each, ~144 MB total |
| Payload read | none, in either run |

The checkpoint directory held `model-00001`–`model-00020` plus `model.safetensors.index.json`
(7,470,294 B), `config.json`, `tokenizer.json` and `inference/`. It is 127 GiB on disk; the
complete set is 475.24 GiB. `/mnt/data3` is a shingled disk, so the download lands on an SSD
staging directory and a serialized mover copies it across — shards 21–23 were in flight
during the run, which is why the counts below are a snapshot rather than a final state.

## Run 1 — the header-prefix tree: 38 of 38

```bash
python scripts/audit_dsv41_headers.py --checkpoint-dir /tmp/dsv41 --header-prefix --json /tmp/v41_hdr.json
```

Exit code 0, `38/38 checks passed`, zero undecided, zero failures. `96,085 tensors` and
`475.24 GiB`, identical to the number this page's model guide has carried since the first
prefix run — the tree is the same 48 headers, so this run is the regression guarantee that
the partial-checkpoint work did not change how a complete header set is read.

38 rather than 39 because the only check the index backs is the index check itself, and
without an index it does not run: `index: {'indexed': False, 'shipped_tensors': 0,
'local_shards': [], 'pending_shards': []}`. With an index present the same header set
reports 39. The engine-group breakdown is `config 12, engram 5, packing 1, scales 3,
inventory 4, experts 3, csa2 5, vision 2, dspark 3`.

## Run 2 — the arriving checkpoint: 31 of 39, 8 undecided, 0 failures

```bash
python scripts/audit_dsv41_headers.py --checkpoint-dir /mnt/data3/DeepSeek-V4.1-Flash --json /tmp/v41_ckpt.json
```

Exit code 0. The headline:

```
DeepSeek-V4.1-Flash header audit: 20 shards in /mnt/data3/DeepSeek-V4.1-Flash
  mode: complete shards
  index: 96,085 tensors over 48 shards, 20 local, 28 not downloaded
  readable: 42,303 of 96,085 tensors (44.0%); presence is checked across the checkpoint, shape only where the shard is here
```

44.0% of the checkpoint's tensors are readable by header alone, and every one of them is
consistent with the config. The 8 undecided checks are exactly the ones whose evidence is
not on disk yet, and each names the shards it is waiting for:

| Undecided check | Waiting on |
| --- | --- |
| `scales: all non-Engram FP8 weights use a 32x32 block` | 27 shards; `observed blocks {(32, 32): 147}` and no other block observed |
| `experts: w1/w2/w3 are FP4 packed into I8 with FP4-block-32 E8M0 scales` | 25 shards |
| `inventory: the known shapes match the config` | `43/48` — the `head.weight` + `norm.weight` shard |
| `engram: the tables are F8_E4M3 with one E8M0 scale per 32 channels` | `47/48`, `48/48` |
| `engram: each Engram layer has its gate and value projection` | `47/48`, `48/48` |
| `scales: the Engram tables use a 1x32 per-row block` | `47/48`, `48/48`; `observed blocks {}` |
| `dspark: the Markov and confidence heads match the config` | `46/48` |
| `dspark: main_proj is n_mtp_layers * dim wide` | `44/48` |

The scale check is the clearest illustration of what undecided means. 147 FP8 weights have
been read and *every one* uses a 32×32 block — but the check asserts a claim over all of
them, and 27 shards' worth have not been read, so it reports the observation and waits
rather than claiming a pass on 44% of the evidence. `observed blocks {}` for the Engram
tables is the same statement from the other side: nothing has been read that could
contradict, and nothing has been read that could confirm.

The 31 that did resolve are not a small set. They include the whole 12-check `config`
group, the index check (`index: every local shard holds exactly the tensors the index
assigns it`), `inventory: every backbone layer has its full tensor set`,
`packing: every tensor's byte extent matches its shape and dtype`, `scales: every
weight/scale pair blocks evenly`, both expert counts (384 per backbone layer, 128 per MTP
layer), all five `csa2` checks, `engram: the tables sit on exactly engram_layer_ids`, both
`vision` checks, and `dspark: every MTP layer carries attn and ffn but only the last
carries the heads`. Those are index questions and geometry questions: they are decided by
the first shard that lands, not by all 48.

### Byte inventory of what is on disk

| Category | Tensors | Bytes |
| --- | ---: | ---: |
| Routed experts | 41,472 | 121.03 GiB |
| Attention | 276 | 2.17 GiB |
| Embedding and head | 1 | 1.23 GiB |
| Vision and aligner | 266 | 0.90 GiB |
| Shared experts | 108 | 0.59 GiB |
| Layer norms and hyper-connections | 180 | 0.13 GiB |
| **Total readable** | **42,303** | **126.06 GiB** |

Largest tensor read: `embed.weight` BF16 `[129280, 5120]`, 1.23 GiB. The routed-expert and
attention rows are proportional to how many backbone layers have landed — 18 layers at
2,304 expert tensors each (18 × 384 experts × 6 tensors = 41,472) — not a fraction of a
fixed total.

### Per-shard contents, as the index predicts them

The index assigns one backbone layer to each of shards 3–42, and the headers agree shard
for shard: shards 3–20 carry 2,334 tensors and 6.88 GiB, except shards 5, 11 and 17, which
carry 2,342 and 6.90 GiB. Those three are backbone layers 2, 8 and 14 — three of the four
`kv_source_layers`. Shard 1 carries the 259 `vision.*` and 4 `aligner.*` tensors; shard 2
carries `embed.weight`, `image_start`, `image_newline` and `image_end`.

The extra 8 tensors on a Full layer, read out of the headers, are
`attn.compressor.{norm.weight, wgate.weight, wkv.weight}` and
`attn.indexer.{k_norm.weight, weights_proj.weight, wk.weight, wq_b.weight, wq_b.scale}`.
This is the CSA2 mode partitioning restated as per-layer tensor counts rather than as a
name list, and it independently reproduces what the model guide records.

Counting the index by layer, rather than by shard, gives an exact per-layer partition over
all 40 backbone layers:

| Layers | Tensors each | Extra beyond layer 0 |
| --- | ---: | --- |
| 0 | 2,334 | — (the baseline set) |
| 1 | 2,340 | +6 — the Engram block (`embed.weight`, `embed.scale`, `q_weight`, `k_weight`, `wkv.weight`, `wkv.scale`) |
| 2, 8 | 2,342 | +8 — the Full set |
| 14 | 2,348 | +14 — the Full set **and** the Engram block |
| 20 | 2,341 | +7 — the Full set minus `compressor.wgate.weight` |
| 24, 28, 32, 36 | 2,337 | +3 — `indexer.{weights_proj.weight, wq_b.weight, wq_b.scale}` |
| all other 32 layers | 2,334 | none |

Every one of the 40 layers is a superset of layer 0 — `missing=[]` for all of them, which
is what `inventory: every backbone layer has its full tensor set` asserts. Two asymmetries
fall out that are worth stating precisely:

- **Layer 20 owns no `compressor.wgate`** because `compress_ratios[20] == 1`: the gate pools
  `compress_ratio` tokens and a ratio of 1 has no group to pool. Its 7 extras are the other
  seven.
- **The Engram tensors live in shards 47 and 48, not in the layer's own shard.** Layer 1's
  six Engram tensors are shipped in shard 47 while layer 1 itself is shard 4, which is why
  shard 4 has 2,334 tensors and the *layer* has 2,340. The index is what makes the two
  counts consistent, and this is the clearest case of why presence had to be separated from
  shape.

The canonicalized name inventory over the full index is **114 distinct patterns**; the 20
local shards show **63** of them, and the 51 that are absent are exactly the layers,
Engram tables, MTP stages and `head`/`norm` that have not been downloaded. No pattern
appears in the local shards that the index does not list.

## What the shards assert, tensor by tensor

Every shape below was read from a header on disk, not from the config or a model card.
Representative rows from backbone layer 2 — a Full CSA2 layer — reproduced with
`--list-tensors 'layers.2.*'`:

| Tensor | dtype | Shape | Bytes |
| --- | --- | --- | ---: |
| `layers.2.attn.wq_a.weight` | F8_E4M3 | `[1280, 5120]` | 6,553,600 |
| `layers.2.attn.wq_a.scale` | F8_E8M0 | `[40, 160]` | 6,400 |
| `layers.2.attn.wq_b.weight` | F8_E4M3 | `[32768, 1280]` | 41,943,040 |
| `layers.2.attn.wq_b.scale` | F8_E8M0 | `[1024, 40]` | 40,960 |
| `layers.2.attn.wkv.weight` | F8_E4M3 | `[512, 5120]` | 2,621,440 |
| `layers.2.attn.wkv.scale` | F8_E8M0 | `[16, 160]` | 2,560 |
| `layers.2.attn.wo_a.weight` | F8_E4M3 | `[8192, 4096]` | 33,554,432 |
| `layers.2.attn.wo_a.scale` | F8_E8M0 | `[256, 128]` | 32,768 |
| `layers.2.attn.wo_b.weight` | F8_E4M3 | `[5120, 8192]` | 41,943,040 |
| `layers.2.attn.wo_b.scale` | F8_E8M0 | `[160, 256]` | 40,960 |
| `layers.2.attn.q_norm.weight` | BF16 | `[1280]` | 2,560 |
| `layers.2.attn.kv_norm.weight` | BF16 | `[512]` | 1,024 |
| `layers.2.attn.attn_sink` | F32 | `[64]` | 256 |
| `layers.2.attn.compressor.wkv.weight` | BF16 | `[512, 5120]` | 5,242,880 |
| `layers.2.attn.compressor.wgate.weight` | BF16 | `[512, 5120]` | 5,242,880 |
| `layers.2.attn.compressor.norm.weight` | BF16 | `[512]` | 1,024 |
| `layers.2.attn.indexer.wk.weight` | BF16 | `[128, 512]` | 131,072 |
| `layers.2.attn.indexer.k_norm.weight` | BF16 | `[128]` | 256 |
| `layers.2.attn.indexer.weights_proj.weight` | BF16 | `[32, 5120]` | 327,680 |
| `layers.2.attn.indexer.wq_b.weight` | F8_E4M3 | `[4096, 1280]` | 5,242,880 |
| `layers.2.attn.indexer.wq_b.scale` | F8_E8M0 | `[128, 40]` | 5,120 |
| `layers.2.ffn.gate.weight` | BF16 | `[384, 5120]` | 3,932,160 |
| `layers.2.ffn.gate.bias` | F32 | `[384]` | 1,536 |
| `layers.2.ffn.gate.bias_vl` | F32 | `[384]` | 1,536 |
| `layers.2.ffn.experts.0.w1.weight` | I8 | `[2304, 2560]` | 5,898,240 |
| `layers.2.ffn.experts.0.w1.scale` | F8_E8M0 | `[2304, 160]` | 368,640 |
| `layers.2.ffn.experts.0.w2.weight` | I8 | `[5120, 1152]` | 5,898,240 |
| `layers.2.ffn.experts.0.w2.scale` | F8_E8M0 | `[5120, 72]` | 368,640 |
| `layers.2.ffn.shared_experts.w1.weight` | F8_E4M3 | `[2304, 5120]` | 11,796,480 |
| `layers.2.ffn.shared_experts.w1.scale` | F8_E8M0 | `[72, 160]` | 11,520 |
| `layers.2.ffn.shared_experts.w2.weight` | F8_E4M3 | `[5120, 2304]` | 11,796,480 |
| `layers.2.hc_attn_fn` | F32 | `[24, 20480]` | 1,966,080 |
| `layers.2.hc_attn_base` | F32 | `[24]` | 96 |
| `layers.2.hc_attn_scale` | F32 | `[3]` | 12 |

Four facts are worth pulling out of the table because they were assumptions before this run
and are now reads:

1. **`ffn.gate.bias_vl` exists on every backbone layer that has landed** — F32 `[384]`, one
   per layer, alongside `ffn.gate.bias`. The model guide previously recorded this tensor as
   appearing nowhere but its own pages; 18 local layers carry it.
2. **The expert tensors are nibble-packed, and the byte extent proves it.**
   `w1.weight` is `I8 [2304, 2560]` = 5,898,240 B, while the logical FP4 matrix is
   `[2304, 5120]` = 11,796,480 elements. The scale `[2304, 160]` gives a block size of 32
   along K (2560 = 5120/2 bytes per row, 160 = 5120/32 blocks). `packing: every tensor's
   byte extent matches its shape and dtype` deciding this is what distinguishes a real
   packed FP4 expert from an unpacked-byte one.
3. **`attn.q_norm.weight` is `[1280]`, the Q LoRA rank**, not the head dimension — so the
   QK norm sits on the compressed query before `wq_b` expands it to `n_heads × head_dim`.
4. **The compressor is BF16 `[512, 5120]` on disk at layer 2**, which is a ratio-2 source:
   the reference declares that matrix FP32 for ratio-2 layers and BF16 for ratio-1, so a
   loader reading the checkpoint has to upcast three of the four copies. This is measured
   here rather than inferred from the reference.

## Reproducing this record

Both runs are standard-library only and read no payload, so either can be repeated against
whatever has been downloaded at the time; the numbers move as shards land, and the report
is designed to be read rather than memorized.

```bash
# the header-prefix tree: 38/38, exit 0
python scripts/audit_dsv41_headers.py \
  --checkpoint-dir /tmp/dsv41 --header-prefix --json /tmp/v41_hdr.json

# the arriving checkpoint: 31/39 with 8 undecided, exit 0
python scripts/audit_dsv41_headers.py \
  --checkpoint-dir /mnt/data3/DeepSeek-V4.1-Flash --json /tmp/v41_ckpt.json

# the same run read strictly: undecided counts as failure, exit 1
python scripts/audit_dsv41_headers.py \
  --checkpoint-dir /mnt/data3/DeepSeek-V4.1-Flash --require-complete

# one layer, tensor by tensor
python scripts/audit_dsv41_headers.py \
  --checkpoint-dir /mnt/data3/DeepSeek-V4.1-Flash --list-tensors 'layers.2.*'
```

`--index PATH` points the presence half at an index other than
`<checkpoint-dir>/model.safetensors.index.json`. The JSON report carries `checkpoint_dir`,
`shards`, `complete`, `tensors`, an `index` block (`indexed`, `shipped_tensors`,
`local_shards`, `pending_shards`) and `checks`, where every check has a `name`, a `status`
of `pass` / `fail` / `undecided`, and a `detail` naming the failing evidence or the shards
being waited on.

The partial behaviour is covered by `tests/test_models_deepseek_v4_1_tensor_audit.py`, which
builds a real partial checkpoint out of the header tree — rewriting an index from the 48
headers and keeping one or two shards — and asserts the boundary rather than the count: a
check whose evidence is missing is undecided and not a pass, a wrong shape inside a
downloaded shard fails while other shards are outstanding, and a header the index contradicts
fails even though the tensor's absence cannot be observed in a header it is not in.

```bash
python -m pytest tests/test_models_deepseek_v4_1_tensor_audit.py -q
```

Expect 16 passed in the `deepseek` environment, 15 of them without a download: the
checkpoint-backed case skips when the release is not on the host.

## Limitations

- **44.0% coverage at this run.** 28 shards, including `head.weight`, `norm.weight` and all
  three MTP stages, had not arrived. Every check that needs them reports undecided; none of
  them is a defect and none of them is a pass. The Engram tables in particular (94.56 GiB
  each, shards 47 and 48) are the last and largest, so the Engram shape checks will be
  undecided for the longest.
- **Metadata only, still.** Nothing on this page reads a tensor value. The audit proves the
  config and the tensor inventory agree with each other; a checkpoint could satisfy every
  check and still be unusable, and a value that is *consistently* wrong in both the config
  and the shapes would pass.
- **A passing check over partial evidence is not asserted.** Where the evidence is
  incomplete the check reports undecided even when every observation so far agrees — the
  `32x32` block check with `147` agreeing observations is undecided, not passing. A caller
  that wants the weaker monotone claim can read the `detail` field, which always carries the
  observation alongside the shard count.
- **The per-layer and per-pattern counts are the index's, not the headers'.** They are
  computed from `model.safetensors.index.json`, which is the release's own statement about
  its shards. For the 20 shards on disk the headers agree with it exactly, which is what
  `index: every local shard holds exactly the tensors the index assigns it` reports; for the
  other 28 that agreement is not yet observable.
- **No execution path.** See [DeepSeek-V4.1-Flash](../models/deepseek-v4.1-flash.md) for
  what a V4.1 runtime would have to add. This page measures a checkpoint, not a runtime.
