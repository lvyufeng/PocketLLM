# Ascend TP Collective Overlap

Qwen3.8-27B (`hidden_size=5120`, 64 layers: 48 gated-delta linear attention, 16 full attention),
4 x Ascend 910A first generation (`Short_SoC_version=Ascend910`), TP=4, CANN 9.0.0.

The engine overlaps each row-parallel projection's TP all-reduce with the next slice's GEMM. This
document records how many slices that split should use, and what the measurement says about the
collective's cost. Every number is from a run on the machine above; the command is in
[Reproducing](#8-reproducing).

## 1. What the overlap path does

Three sites in the decoder layer end in a row-parallel projection followed by a TP all-reduce:
`mlp.down` (all 64 layers), `lin.out` (the 48 gated-delta layers) and `full.out` (the 16 full
attention layers). `QwenEngine::projection_all_reduce_overlapped` splits the projection's output
rows into `slices` and, for each slice, issues the GEMM on the compute stream and then the
`tp_all_reduce_sum_f16_inplace` for that slice on the HCCL comm stream, so slice *i*'s collective
runs under slice *i+1*'s GEMM. The ceiling is `comm_overlap_slices` (`QWEN_COMM_OVERLAP_SLICES`,
default 4); a row count or world size the path cannot serve falls back to the serial
GEMM-then-collective sequence.

## 2. Why the slice count cannot be a constant

Slicing does not change what has to move across the ring. `slices` slices send the same total bytes
in `slices` calls, so the trade is narrow: it buys overlap of the collective's **byte** term in
exchange for paying its **fixed per-call** term `slices` times instead of once. Splitting one call
into four pays three extra fixed terms, and it is a loss until the byte term is large enough that
overlapping it recovers more than those three fixed terms cost.

Both terms can be priced from the measurements below. At 512 rows the 4-slice pass issues 384 more
collectives than the serial pass and costs **0.612 s more prefill** (1.196 s against 0.584 s), i.e.
1.60 ms per extra collective. The serial pass's 128 collectives cost 0.491 s, i.e. 3.83 ms each
(section 6). So the fixed term is **~1.6 ms** and the byte term **~2.2 ms** at 512 rows — the 4-slice
split pays 4x1.6 ms of fixed cost to overlap 2.2 ms of bytes, which is why it loses by 2x.

The byte term grows 8x from 512 to 4096 rows; the fixed term does not. That is the whole crossover.

## 3. Measured: the constant was 2x wrong at short prompts

TP4, 64 layers, 512-token prompt, 128 new tokens, `QWEN_COMM_OVERLAP_SLICES` forced per run. Three
reps per setting, run interleaved so a slow patch of machine lands on every setting.

| slices | prefill TPS (3 reps) | mean |
|---|---|---|
| 1 (serial) | 882.09 / 882.55 / 862.62 | **875.8** |
| 4 (the shipped default) | 429.12 / 428.61 / 436.35 | **431.4** |
| 8 | 871.51 / 883.83 / 883.42 | **879.6** |

Decode moved only 8.72..8.99 TPS across all nine runs, which bounds the run-to-run drift: decode
takes the serial path at every setting, so a 2.03x prefill effect is not drift.

The slices=8 row is the control that the effect is the splitting and not the row count. At 512 rows
the then-current gate refused it (`per = 512/8 = 64 < 96`), so it fell back to the serial path and
landed on the serial number to within 0.4%. Two settings that resolve to the same code path agree;
the one that does not is 2x away.

At 4096 rows the trade reverses, two reps each:

| slices | prefill TPS (2 reps) | mean |
|---|---|---|
| 1 (serial) | 1206.01 / 1204.42 | **1205.2** |
| 4 | 1267.79 / 1262.53 | **1265.2** |

### Where the crossover is

Rows 1024 and 2048 were unmeasured, and they are what the rule has to get right. 8 new tokens:

| rows | slices=1 | slices=2 | slices=4 |
|---|---|---|---|
| 512 | **875.8** | — | 431.4 |
| 1024 | 1138.77 | **1140.13** | 862.41 |
| 2048 | 1195.12 | **1238.58** | 1185.81 |
| 4096 | 1205.2 | — | **1265.2** |

The optimum is 1 slice up to 1024 rows, 2 at 2048 and 4 at 4096 — i.e. **1024 rows per slice**, with
the 1024-row point a wash between 1 and 2 and every point above losing at 512 rows per slice.

## 4. The rule

`QwenEngine::comm_overlap_slices_for_rows(rows)` doubles the slice count while the resulting
per-slice row count stays at or above `kMinRowsPerOverlapSlice = 1024`, up to the configured
ceiling. Nothing else consumes `comm_overlap_slices` — a runtime knob still scales the ceiling, and
0 or 1 disables slicing.

Verification, no override set, same four shapes:

| prompt | new | slices chosen | prefill TPS | decode TPS |
|---|---|---|---|---|
| 512 | 128 | 1 | 878.8 | 8.73 |
| 1024 | 8 | 1 | 1134.5 | 8.96 |
| 2048 | 8 | 2 | 1238.3 | 9.11 |
| 4096 | 32 | 4 | 1261.6 | 8.91 |

Every point reproduces the optimum the forced sweep found (875.8 / 1140.1 / 1238.6 / 1265.2) to
within the drift the decode column bounds.

### Effect on the documented baseline

| configuration | prompt | new | prefill TPS | decode TPS |
|---|---|---|---|---|
| before (constant 4) | 512 | 128 | 428.1 | 9.21 |
| after (rule) | 512 | 128 | **878.8** | 8.73 |
| before (constant 4) | 4096 | 32 | 1265.3 | 9.55 |
| after (rule) | 4096 | 32 | **1261.6** | 8.91 |

**2.05x prefill at 512 tokens, within spread at 4096.** The 512-token row is worth 2x; the 4096 row
picks 4 slices and is unchanged, which is the intended behaviour of a rule that only ever moves the
short-prompt end.

## 5. Measured: prefill is collective-bound at both prompt lengths

`QWEN_HOST_PROFILE=1`, 512-token prompt, 64 layers, TP4, post-fix build (so the rule selects the
serial path). The profile emits four blocks — kernel warmup, the added `warmup_decode` boundary, the
real prefill, decode — and the real-prefill block's own figures are below. `TOTAL` is the sum of
every scope and so double-counts nested ones; the wall for the same run was
`prefill_seconds=0.584436` / `prefill_tokens_per_s=876.058`.

| phase | seconds | calls | us/call |
|---|---|---|---|
| `STACK.r` (whole 64-layer stack) | 0.54849 | 64 | 8570 |
| `ar.lin.out` | 0.324624 | 48 | 6763 |
| `ar.mlp` | 0.132871 | 64 | 2076 |
| `ar.full.out` | 0.0332871 | 16 | 2080 |
| `ar.hidden_a` | 0.0165539 | 1 | 16554 |
| `tp_all_reduce` (inner measure of the same calls) | 0.507209 | 129 | 3932 |
| `top1_allreduce` | 0.0111021 | 1 | 11102 |
| `layer.attn` | 0.400903 | 64 | 6264 |
| `layer.mlp_down` | 0.133831 | 64 | 2091 |
| `full_attention` | 0.053958 | 16 | 3372 |
| every `pr.*` / `pd.*` projection | <= 0.006359 each | — | — |

Read against the wall this is a sharper statement than the one it replaces:

- `ar.lin.out + ar.mlp + ar.full.out = 0.4908 s`, and `ar.hidden_a` adds the 1 collective the stack
  does not contain, so **0.5072 s of the 0.584 s prefill wall is the 129 TP all-reduce calls** —
  87%.
- `STACK.r` is 0.548 s total, of which 0.491 s is those collectives, so **all of the layer's other
  work — norms, transposes, gated-delta, swiglu, all the projections — is 0.058 s, 0.90 ms per
  layer**.
- The 129 calls are 2 per layer — one `mlp.down` reduce in every layer and one attention-output
  reduce, which is `lin.out` in the 48 gated-delta layers and `full.out` in the 16 full attention
  layers — plus one hidden-state reduce.

The same profile on the pre-fix build confirms the mechanism from the other side: `STACK.r` 1.10998 s
with the three `ar.<site>.ov` scopes summing to 1.0336 s over 512 calls (256 + 192 + 64, i.e. the
layer sites times 4 slices), and the layer's non-collective work at 0.076 s. The collectives
themselves cost 1.0336 s sliced against 0.4908 s serial for the same bytes.

That is also the largest remaining prefill lever, and it is the same one decode needs. There are 2
collectives per layer at ~3.8 ms each at this row count, against a layer whose every other
operation together costs 0.90 ms; removing one of the two per layer would take 0.245 s off the
0.584 s wall. No projection tuning at this prompt length comes close.

### The same attribution at 4096 rows

The short-prompt result could have been an artefact of the serial path: at 4096 rows the rule
selects 4 slices, the byte term is 8x larger, and three of every four collectives are overlapped by
the next slice's GEMM, so the collective share should fall. It does not. Same profile,
4096-token prompt, 2 new tokens, no override (so `comm_overlap_slices_for_rows(4096)` picks 4):

| phase | seconds | calls | us/call |
|---|---|---|---|
| `ar.mlp.ov` | 2.17062 | 256 | 8479 |
| `ar.lin.out.ov` | 0.400025 | 192 | 2083 |
| `ar.full.out.ov` | 0.135582 | 64 | 2119 |
| `ar.hidden_a` | 0.0163618 | 1 | 16362 |
| `STACK.r` (whole 64-layer stack) | 2.9734 | 64 | 46460 |
| `full_attention` | 0.299992 | 16 | 18750 |
| `layer.attn` | 0.773072 | 64 | 12079 |
| `pr.mlp.down` (the GEMMs the collectives overlap) | 0.00835505 | 256 | 33 |
| `pr.lin.out` | 0.00642037 | 192 | 33 |
| every other `pr.*` / `pd.*` projection | <= 0.005759 each | — | — |
| `top1_allreduce` | 0.242522 | 1 | 242522 |
| wall | 3.24111 | — | 1263.77 TPS |

- The `*.ov` scopes sum to **2.706 s over 512 calls**, and `ar.hidden_a` is the 513th, so
  **2.723 s of the 3.241 s wall — 84% — is still TP all-reduce**, at 8x the row count and with the
  4-way split in place. Slicing does what it was supposed to (the projections it overlaps are
  measured at 33 us per slice, 0.0084 s for all 256 of them) and the collective still owns the wall.
- `STACK.r` at 2.973 s against 2.706 s of collectives leaves **0.267 s, 4.2 ms per layer**, for
  everything else in the layer. That is up from 0.90 ms at 512 rows because the GEMM work is real
  at this length — but it is still 8% of the wall.
- `top1_allreduce` is a single call and costs **0.243 s, 7.5% of this prefill**, against 0.011 s for
  the same scope in the decode block of the same run. It is measured, not explained: the scope is
  the only one that grows 20x between the two phases, and this profile cannot yet say why. It is
  the second-largest single item in the prefill wall and is recorded here as open.
- `full_attention` is 0.300 s (9.3%) — at 4096 rows attention has become a real term, unlike at 512,
  which is why it is profiled separately from the collectives rather than folded into them.

The two profiles say the same thing at both ends of the range: a 64-layer TP4 pass issues 129
collectives at 512 rows and 513 at 4096 rows, and either way they are 84-87% of the prefill wall.
More slices move cost between the fixed and byte terms; they do not reduce the number of times the
layer has to stop at a collective.

## 6. What this rules out

- **More slices.** Four is already past the optimum at every row count below 4096 and the ceiling is
  only reachable because of it; the failure mode at short prompts is not subtle. The 4096-row
  profile above is the stronger form of the same point: with the 4-way split applied and half a
  thousand collectives in the pass, they are still 84% of the wall.
- **A cheaper collective.** The fixed term is host-side issue of the collective itself, measured flat
  in payload size in `bench_qwen_ascend_allreduce` (0.404 ms at 10 KB, 0.412 ms at 640 KB, against a
  0.011 ms device event pair), and `hcclOpExpansionMode` 0/1/2/3 leaves it unmoved. Slicing hides the
  byte term; it cannot make the fixed term smaller.
- **Tuning the projections to fix prefill.** Every projection GEMM is 0.0064 s or less at a 512-token
  prompt and 0.0084 s or less at 4096; the collectives over the same layers are 0.49 s and 2.71 s
  respectively.
- **Reading the short-prompt result as a short-prompt problem.** The collective share is 87% at 512
  rows and 84% at 4096, so this is the shape of the pass, not of the prompt.

## 7. Files

- `cpp_engine/engine/qwen_engine.cpp` — `comm_overlap_slices_for_rows`, `kMinRowsPerOverlapSlice`,
  and the call site in `projection_all_reduce_overlapped`.
- `cpp_engine/engine/qwen_layer_components.inl` — the three projection sites that call it.
- `cpp_engine/engine/qwen_layer_components.hpp` and `include/qwen_engine.hpp` — the
  `comm_overlap_slices` config field and its default.
- `cpp_engine/tests/bench_qwen_ascend_allreduce.cpp` — the per-call cost in section 6.

## 8. Reproducing

Four ranks, one process per rank, `HCCL_WHITELIST_DISABLE=1`,
`POCKETLLM_CPP_NCCL_ID_WAIT_ATTEMPTS=12000`, and `scripts/ascend_env.sh` sourced first. Rank *i*
runs:

```bash
python3 -c "print(','.join(str(t) for t in [100,264,4628,286,1879,310,278,4221,13,576]*52)[:4096])" \
  > ids_4096.txt

QWEN_COMM_OVERLAP_SLICES=4 ./cpp_engine/build-ascend/pocketllm_engine \
  --ckpt /mnt/data1/modelscope/Qwen/Qwen3.8-27B \
  --smoke-forward --smoke-layers 0 --resident-bench \
  --token-ids-file ids_4096.txt --max-new-tokens 32 \
  --tp-world 4 --tp-rank 0 --device 0 --nccl-id-path hccl_p4096_s4.id
```

`QWEN_COMM_OVERLAP_SLICES` pins the slice count for the sweep; omit it to measure the rule. The
`prefill_seconds` / `prefill_tokens_per_s` / `decode_seconds` / `decode_tokens_per_s` line on rank 0
is the result. For the section 5 tables add `QWEN_HOST_PROFILE=1`; the profile then prints one
`qwen_phase tag=...` block per measurement boundary and resets between them. The 4096-row profile is
the same command with `--max-new-tokens 2` and a 4096-token id file; two new tokens rather than 32
keeps the decode block short, and the prefill block is unaffected by the generation length.
