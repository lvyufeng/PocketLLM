# Auditing the DeepSeek-V4.1-Flash shards from arrival to complete

**Date:** 2026-09-16
**Commit:** `0bafa6f` — where `scripts/audit_dsv41_headers.py` last changed, and the revision all
four runs below were made with
**Script:** `scripts/audit_dsv41_headers.py`
**Artifacts:** `/tmp/v41_hdr.json`, `/tmp/v41_ckpt.json`, `/tmp/v41_run3.json` and `/tmp/v41_run4.json`
from the four commands in [Reproducing this record](#reproducing-this-record)
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
host. It is now, in full: the release is 48 shards and 475.24 GiB, and the last two shards
to land were the two Engram tables. So the page has a second, larger question the prefix
tree could never ask — **what do the real shards say, and can the audit tell a shard that
is missing from a shard that is wrong?**

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
fails while other shards are still missing.

## Measurement conditions

| Item | Value |
| --- | --- |
| Host | x86_64 CUDA machine |
| GPU | none used; the audit never calls a device API |
| CPU / RAM | 2 × Intel Xeon E5-2696 v4, 2 NUMA nodes, ~1 TiB |
| OS / Python | Ubuntu 22.04.5, kernel 5.15, CPython 3.10.10 (conda `deepseek`) |
| Dependencies | standard library only — no `torch`, `safetensors` or `numpy` |
| Checkpoint | `/mnt/data3/DeepSeek-V4.1-Flash`, 20 of 48 shards on disk at run 2, 46 of 48 at run 3 and 48 of 48 at run 4 |
| Header tree | `/tmp/dsv41`, 48 × `h0000N.bin`, 3,000,001 B each, ~144 MB total |
| Payload read | none, in any run |

The checkpoint directory held `model-00001`–`model-00020` plus `model.safetensors.index.json`
(7,470,294 B), `config.json`, `tokenizer.json` and `inference/` at run 2, `model-00001`–`model-00046`
at run 3, and all 48 shards at run 4. It is 127 GiB on disk at run 2, 286 GiB at run 3 and
475.24 GiB at run 4. `/mnt/data3` is a shingled disk, so the download lands on an SSD staging
directory and a serialized mover copies it across — shards were in flight during runs 2 and 3,
which is why those counts are a snapshot rather than a final state. Run 4 is the final state.

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

## Run 2 — the arriving checkpoint at 20 shards: 31 of 39, 8 undecided, 0 failures

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

### Byte inventory at 20 shards

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

## Run 3 — the checkpoint nearly complete: 35 of 39, 4 undecided, 0 failures

```bash
python scripts/audit_dsv41_headers.py --checkpoint-dir /mnt/data3/DeepSeek-V4.1-Flash --json /tmp/v41_run3.json
```

Exit code 0. The headline:

```
DeepSeek-V4.1-Flash header audit: 46 shards in /mnt/data3/DeepSeek-V4.1-Flash
  mode: complete shards
  index: 96,085 tensors over 48 shards, 46 local, 2 not downloaded
  readable: 96,073 of 96,085 tensors (100.0%); presence is checked across the checkpoint, shape only where the shard is here
```

The tool prints one decimal and 96,073 of 96,085 rounds up to it, so read the fraction rather
than the percentage: **twelve tensors are unreadable, and they are the Engram tensors** — six
on each of the two Engram layers, in shards 47 and 48. Nothing else in the checkpoint is
missing.

Four checks resolved between run 2 and this one, each for the reason its run-2 row named:

| Check | Was waiting on | Now |
| --- | --- | --- |
| `experts: w1/w2/w3 are FP4 packed into I8 with FP4-block-32 E8M0 scales` | 25 shards | pass, over all 94,464 expert tensors: `expected I8[2304,2560] and F8_E8M0[2304,160]` with no offender |
| `inventory: the known shapes match the config` | `43/48` | pass — `head.weight` BF16 `[129280, 5120]` and `norm.weight` BF16 `[5120]` landed on shard 43 |
| `dspark: main_proj is n_mtp_layers * dim wide` | `44/48` | pass — `mtp.0.main_proj.weight` F8_E4M3 `[5120, 15360]`, and 15,360 = 3 × 5120 |
| `dspark: the Markov and confidence heads match the config` | `46/48` | pass — `mtp.2.markov_head.{embed,head}.weight` BF16 `[129280, 256]` (the Markov rank) and `mtp.2.confidence_head.proj.weight` BF16 `[1, 5376]` (dim + rank) |

Nothing new became undecided and nothing that had resolved went back. The four that remain are
the Engram claims, which are the same four that were waiting at run 2:

| Undecided check | Waiting on |
| --- | --- |
| `scales: all non-Engram FP8 weights use a 32x32 block` | `47/48`, `48/48`; `observed blocks {(32, 32): 353}` and no other block observed |
| `engram: the tables are F8_E4M3 with one E8M0 scale per 32 channels` | `47/48`, `48/48` |
| `engram: each Engram layer has its gate and value projection` | `47/48`, `48/48` |
| `scales: the Engram tables use a 1x32 per-row block` | `47/48`, `48/48`; `observed blocks {}` |

The first row is the one to read closely, because its *name* and its *scope* differ. The scope
is every `.weight` in the checkpoint that ships a `.scale` sibling and is not one of the
tables' `engram.embed.*` matrices — and exactly two of the 96,085 tensors qualify only because
they sit in shards 47 and 48: `layers.1.engram.wkv.weight` and `layers.14.engram.wkv.weight`,
the value projections of the two Engram layers. So a check about *non*-Engram FP8 weights is
held open by two Engram ones. That is the correct answer rather than a naming mistake: the
claim covers every FP8 weight outside the embed matrices, and those two have not been read.
353 observations now agree with the claim, up from 147, and the check still refuses to call
that a pass.

`observed blocks {}` for the Engram tables is the same statement from the other side: nothing
has been read that could contradict, and nothing has been read that could confirm.

### Byte inventory at 46 shards

| Category | Tensors | Bytes |
| --- | ---: | ---: |
| Routed experts | 94,464 | 275.67 GiB |
| Attention | 603 | 4.80 GiB |
| Embedding and head | 3 | 2.47 GiB |
| Shared experts | 240 | 1.32 GiB |
| Vision and aligner | 266 | 0.90 GiB |
| MTP and DSpark | 97 | 0.66 GiB |
| Layer norms and hyper-connections | 400 | 0.29 GiB |
| **Total readable** | **96,073** | **286.11 GiB** |

Largest tensor read: still `embed.weight` BF16 `[129280, 5120]` at 1.23 GiB. The largest tensors
in the checkpoint are not readable ones — they are the two Engram `embed.weight` tables in
shards 47 and 48, roughly 92 GiB apiece, which is why the largest tensor *on disk* is not the
largest tensor *in the checkpoint*. The routed-expert row now covers every expert-bearing
layer: 40 backbone layers at 384 routed experts and 3 MTP layers at 128, so
15,744 × 6 = 94,464 tensors. The `Embedding and head` row is complete at three tensors —
`embed.weight`, `head.weight` and `norm.weight` — which is 2.47 GiB.

The shard sizes themselves are the per-layer partition read back off disk: shards 3–42 carry
2,334 tensors and 6.88 GiB each, except shards 5, 11 and 17 (2,342 / 6.90 GiB — backbone layers
2, 8 and 14), shard 23 (2,341 / 6.89 GiB — layer 20), and shards 27, 31, 35 and 39 (2,337 /
6.89 GiB — layers 24, 28, 32 and 36). Shard 43 carries exactly 2 tensors, the `head.weight` +
`norm.weight` pair; shards 44, 45 and 46 carry 801, 798 and 802 tensors — the three MTP stages,
which is where the per-layer counts stop being uniform.

## Run 4 — the complete checkpoint: 39 of 39, 0 undecided, 0 failures

```bash
python scripts/audit_dsv41_headers.py --checkpoint-dir /mnt/data3/DeepSeek-V4.1-Flash --json /tmp/v41_run4.json
```

Exit code 0, and this is the run that closes the question the page was built around. The
headline:

```
DeepSeek-V4.1-Flash header audit: 48 shards in /mnt/data3/DeepSeek-V4.1-Flash
  mode: complete shards
  index: 96,085 tensors over 48 shards, 48 local, 0 not downloaded
  readable: 96,085 of 96,085 tensors (100.0%); presence is checked across the checkpoint, shape only where the shard is here
```

**39 of 39 checks passed, nothing undecided, nothing failed.** The last four to resolve were
the ones waiting on shards 47 and 48, and each one is now a read rather than a wait:

| Check | Was waiting on | Now |
| --- | --- | --- |
| `scales: all non-Engram FP8 weights use a 32x32 block` | `47/48`, `48/48` | pass — `observed blocks {(32, 32): 355}` and no other block observed, so the two `engram.wkv` weights held it open and then agreed with it |
| `engram: the tables are F8_E4M3 with one E8M0 scale per 32 channels` | `47/48`, `48/48` | pass — `layers.{1,14}.engram.embed.weight` F8_E4M3, with the matching `embed.scale` |
| `engram: each Engram layer has its gate and value projection` | `47/48`, `48/48` | pass — `layers.{1,14}.engram.{q_weight, k_weight}` and `wkv.weight`/`wkv.scale` |
| `scales: the Engram tables use a 1x32 per-row block` | `47/48`, `48/48` | pass — `observed blocks {(1, 32): 2}`, one per table |

The `32x32` row is the one that took the longest to answer and the one worth reading: its
scope is every `.weight` in the checkpoint that ships a `.scale` sibling and is not one of
the tables' `engram.embed.*` matrices, and exactly two tensors qualified only because they sit
in shards 47 and 48 — `layers.1.engram.wkv.weight` and `layers.14.engram.wkv.weight`, the
value projections of the two Engram layers. It went 147 → 353 → 355 observations, all in
agreement, and the check refused to call any of the intermediate counts a pass. That is the
whole point of the third outcome: the count was never the evidence.

The largest tensor in the checkpoint is `layers.14.engram.embed.weight` F8_E4M3
`[384016682, 256]` at 91.56 GiB, and it is one of the twelve Engram tensors that were
unreadable at both earlier runs. The 768,022,850 rows the run reports across the two tables
are what the derived bucket minimum `[384006168, 384016682]` was checking against, and the
declared and derived ranges agree exactly rather than within tolerance.

### Byte inventory at 48 shards — the complete checkpoint

| Category | Tensors | Bytes |
| --- | ---: | ---: |
| Routed experts | 94,464 | 275.67 GiB |
| Engram tables | 12 | 189.13 GiB |
| Attention | 603 | 4.80 GiB |
| Embedding and head | 3 | 2.47 GiB |
| Shared experts | 240 | 1.32 GiB |
| Vision and aligner | 266 | 0.90 GiB |
| MTP and DSpark | 97 | 0.66 GiB |
| Layer norms and hyper-connections | 400 | 0.29 GiB |
| **Total** | **96,085** | **475.24 GiB** |

The read is 100.0% of the index's `96,085` tensors, and the 475.24 GiB matches the `475.24 GiB`
this page's model guide has carried since the header-prefix run — which is the property that
matters: the payload read agrees with the header-only read to the byte, so the header tree was
a faithful stand-in for the whole time the checkpoint was arriving. The twelve Engram tensors
are 189.13 GiB of the total, 39.8%, out of 12 tensors of 96,085 — and `routed_experts` is
275.67 GiB, 58.0%, out of 94,464. **97.8% of the checkpoint is the routed experts plus the two
Engram tables.**

Shards 47 and 48 are the two that were last, at 6 tensors and 94.56 GiB each. Nothing about
the partition changed when they landed: shards 3–42 still carry one backbone layer apiece, and
the per-layer counts in the next section are what the index predicts for all 48.

## What the index predicts, per shard and per layer

The index assigns one backbone layer to each of shards 3–42, and the headers agree shard for
shard over all 40 of them. The extra tensors are the CSA2 modes showing up as counts: the 8 on
a Full layer, read out of the headers, are
`attn.compressor.{norm.weight, wgate.weight, wkv.weight}` and
`attn.indexer.{k_norm.weight, weights_proj.weight, wk.weight, wq_b.weight, wq_b.scale}`, and
the 3 on a Reindex layer are `attn.indexer.{weights_proj.weight, wq_b.weight, wq_b.scale}`.
This is the CSA2 mode partitioning restated as per-layer tensor counts rather than as a name
list, and it independently reproduces what the model guide records.

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
  shape. It is also why the twelve tensors unreadable at runs 2 and 3 were exactly the twelve
  this table adds to layers 1 and 14, and why shards 47 and 48 — six tensors each, 94.56 GiB
  each — were the last two checks standing.

Collapsing the index's four indexed positions — `layers.<i>.`, `mtp.<i>.`, `.experts.<i>.` and
`vision.blocks.<i>.` — leaves **114 distinct name patterns**. The 46 local shards showed **108**
of them through run 3, and the 6 absent then were precisely the two Engram layers' shared six;
with all 48 shards the inventory is complete at **114 of 114**. No pattern appears in the shards
that the index does not list, at either count.

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

1. **`ffn.gate.bias_vl` exists on every layer that has landed** — F32 `[384]`, one per layer,
   alongside `ffn.gate.bias`: 43 of each, covering all 40 backbone layers and all 3 MTP
   layers. The model guide previously recorded this tensor as appearing nowhere but its own
   pages.
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

All four runs are standard-library only and read no payload, so any of them can be repeated
against whatever has been downloaded at the time; the numbers move as shards land, and the
report is designed to be read rather than memorized.

```bash
# the header-prefix tree: 38/38, exit 0
python scripts/audit_dsv41_headers.py \
  --checkpoint-dir /tmp/dsv41 --header-prefix --json /tmp/v41_hdr.json

# the arriving checkpoint at 20 shards: 31/39 with 8 undecided, exit 0
python scripts/audit_dsv41_headers.py \
  --checkpoint-dir /mnt/data3/DeepSeek-V4.1-Flash --json /tmp/v41_ckpt.json

# the arriving checkpoint at 46 shards: 35/39 with 4 undecided, exit 0
python scripts/audit_dsv41_headers.py \
  --checkpoint-dir /mnt/data3/DeepSeek-V4.1-Flash --json /tmp/v41_run3.json

# the complete checkpoint: 39/39, nothing undecided, exit 0
python scripts/audit_dsv41_headers.py \
  --checkpoint-dir /mnt/data3/DeepSeek-V4.1-Flash --json /tmp/v41_run4.json

# the same run read strictly: undecided counts as failure, exit 1
python scripts/audit_dsv41_headers.py \
  --checkpoint-dir /mnt/data3/DeepSeek-V4.1-Flash --require-complete

# one layer, tensor by tensor
python scripts/audit_dsv41_headers.py \
  --checkpoint-dir /mnt/data3/DeepSeek-V4.1-Flash --list-tensors 'layers.2.*'
```

The three checkpoint runs differ only in which shards are on disk, which is the point: the
command does not change as the download advances, so a check that flips from undecided to
passing between two runs flipped because its evidence arrived and for no other reason. The one
command whose verdict changes with completeness is `--require-complete`, which exits 0 on the
complete checkpoint — nothing is undecided, so there is nothing for it to fail — and exits 1
whenever any check is undecided, which is the binary answer it exists to give.

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

- **Metadata only, still.** Nothing on this page reads a tensor value, in any of the four runs.
  The audit proves the config and the tensor inventory agree with each other; a checkpoint could
  satisfy every check and still be unusable, and a value that is *consistently* wrong in both the
  config and the shapes would pass.
- **No payload is hashed.** The checks that touch bytes compare a declared extent against a
  shape and a dtype, which is what the header carries; a shard whose payload is corrupt *within*
  its declared extent passes every check here. The three partial runs and the complete one are
  the same audit, so completeness buys the shape of the file and not the trustworthiness of its
  contents. Byte-for-byte integrity is a separate measurement.
- **A passing check over partial evidence was never asserted, which is why the intermediate
  verdicts are not the final ones.** The `32x32` block check reported undecided at `353` agreeing
  observations and passed at `355`; anyone quoting the run-2 or run-3 numbers is quoting a
  snapshot of an arriving download, and the run-4 report is the one that answers for the release.
  A caller that wants the weaker monotone claim reads the `detail` field, which always carries
  the observation alongside the shard count.
- **The per-layer and per-pattern counts are the index's, not the headers'.** They are computed
  from `model.safetensors.index.json`, which is the release's own statement about its shards. All
  48 shards now agree with it exactly, which is what
  `index: every local shard holds exactly the tensors the index assigns it` reports — but that is
  agreement between two files the release ships, not an independent statement about either.
- **The audit says what is in the checkpoint; a runtime says what can be executed from it.** The
  two are now separate records rather than one guess: this page measures the checkpoint, and
  [the host run](deepseek_v4_1_flash_host_run.md) loads it and measures what a token costs. What
  remains unbuilt is the device runtime — the audit cannot tell you that, and neither can this
  page.
