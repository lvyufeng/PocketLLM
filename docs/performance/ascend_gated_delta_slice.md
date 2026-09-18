# Ascend gated-delta value-axis slice

Qwen3.8-27B, 4 x Ascend 910A first generation (`Short_SoC_version=Ascend910`), TP=4, CANN 9.0.0.

The gated-delta recurrence is the largest single operator in a prefill layer on this backend: the
in-engine phase profile attributes 2.47 s of an 8.86 s 4966-token prefill to `gated_delta` across its
48 linear-attention layers, more than any other scope. This page records the measurement that located
where that time actually goes, the change that follows from it, and what the change does and does not
buy. The command that produced every number is named alongside it.

**The change is on `master`**, as `kMaxValueSplit` and `value_split_for` in
`cpp_engine/backends/ascend/kernels/qwen_gated_delta_geometry.hpp`, landed by the gated-delta work in
[PR #267](https://github.com/lvyufeng/PocketLLM/pull/267). The numbers below come from the branch
binary that measured it, `bench_qwen_ascend_gated_delta`. That binary was renamed
`bench_qwen_ascend_delta` when the change merged and lost its `--rows` and `--check` flags, so
section 6 states which commands still reproduce against `master` and which do not.

## 1. The op was issue-bound, not element-bound

The kernel walks the recurrence per head, per token, out of UB:

```
S       <- S * decay                      (elementwise, whole matrix)
kv_mem  <- k^T S                          (row vector, value_dim wide)
delta   <- (v - kv_mem) * beta            (row vector)
S       <- S + k delta^T                  (rank-1 update)
out     <- q^T S * q_scale                (row vector)
```

At the TP4 shard shape the shard is 12 value heads over 4 key heads, so the grid is 12 work items on
a part with 30 AI cores: more than half the cores idle. The obvious reading is that the kernel is
under-parallelised, and the obvious fix is to split each head across cores — but only along the value
axis, because `k^T S` and `q^T S` reduce down the key axis and a split there would need a cross-core
reduction.

Before doing that it is worth knowing whether the split pays. It does, but not for the reason the
arithmetic suggests, so the split width is set by measurement rather than by the 30 available cores:

| work per core | measured |
|---|---|
| 128 value columns (whole head) | 10.27 us/head-token |
| 64 value columns | 8.33 us/head-token |

Halving the owned columns removes 19% of the time, not 50%. The per-token cost is dominated by
instruction issue rather than by element count: 384 of the per-token vector instructions are
scalar-driven — 128 `Duplicate` to broadcast k, 128 to broadcast q, 128 `Axpy` for the rank-1 update,
because `dav_c100` has no broadcast intrinsic — and those retire at a fraction of the rate of a
full-width repeat. The fixed part of a slice is therefore ~6.4 us and each further halving only
removes ~1 us.

That is what sets the split at two rather than four:

- two slices: 24 items on 30 cores, **one round** of 8.33 us;
- four slices: 48 items on 30 cores, **two rounds**, so ~2 x 6.9 us — worse.

Four slices would only win if the second round fully overlapped with no straggler and no per-block
fixed cost, which is not what the per-item measurements show.

## 2. The change

A head is split into `value_split` 64-column windows of the value axis, one work item each, with no
shared state and no communication. Each core loads its own `[128, 64]` window of the fp32 state out of
the `[128, 128]` GM pitch, runs the recurrence over its columns, and stores the window back.

`cpp_engine/backends/ascend/kernels/qwen_gated_delta_geometry.hpp` holds the split as one definition
that the device kernel, the host launcher and the benchmark all read. On `master` it is a runtime
decision rather than a constant:

```cpp
constexpr uint32_t kMaxValueSplit = 2;

constexpr uint32_t value_split_for(uint32_t heads, uint32_t cores) {
    return heads * kMaxValueSplit <= cores ? kMaxValueSplit : 1u;
}
```

The launcher queries the device's AI core count, calls `value_split_for` with the head count, and
launches one block per head-slice; the kernel clamps whatever it is handed up to `kMaxValueSplit`,
so an over-large grid is a no-op rather than an out-of-range read. Those are two derivations of the
same number in two translation units, and a drift between them would not fail — it would leave part
of every head's state unwritten and read as a silent accuracy bug. One definition removes the
possibility.

## 3. Op-level result

`cpp_engine/tests/bench_qwen_ascend_gated_delta`, device 0, `--rows 4096 --heads 12 --key-heads 4
--iters 20`, three consecutive runs each. Both arms run through the same launcher and the same bench
binary; only the kernel source differs.

| kernel | us/head-token |
|---|---|
| whole head per core | 10.173 / 10.281 / 10.207 |
| value-axis slice (64 columns) | 8.407 / 8.391 / 8.302 |

**10.22 -> 8.37 us/head-token, -18.1%.**

The same cut is priced a second time in the geometry header that `master` now ships, at **41.3 ->
36.0 ms/layer at 4433 tokens, -13%**. Those are not the same measurement and are not comparable to
each other: this page's A/B is 4096 rows through the branch binary, and the header's is 4433 rows
through `bench_qwen_ascend_delta` on the kernel that replaced this one when the reductions moved from
broadcast-and-fold to accumulation ([PR #267](https://github.com/lvyufeng/PocketLLM/pull/267), which
is also what merged the cut). Both find the same thing, which is the point the header's rule is built
on: a value-axis cut halves every instruction's repeat count and adds none of them.

`--check` in that same branch binary ran the double-precision host reference from
`tests/test_qwen_ascend_group_b` over the same buffers — the flag did not survive the merge, and that
comparison is now the `gated delta recurrence` case inside `test_qwen_ascend_group_b` itself
(`test_gated_delta_recurrence`, rows=2, heads=4). The numbers it produced at the branch shapes were:

| shape | output | state |
|---|---|---|
| rows=4096 | worst_relative=1.068e-04, 0/6291456 mismatches | worst_relative=7.077e-05, 0/196608 mismatches |
| rows=512 | worst_relative=9.655e-05, 0/786432 mismatches | worst_relative=1.211e-05, 0/196608 mismatches |

Zero mismatches over 6.29 M output values. The tolerance is the same one the group-B test used before
the change, and the numbers are of the same order, so the split did not move the error.

## 4. End-to-end result

`scripts/run_qwen_ascend_tp4.sh` with a 4966-token prompt, 9 new tokens and `QWEN_PHASE_PROFILE=1`.
Two runs per arm; the only difference between the arms is the gated-delta kernel source. The TP
all-reduce is left **enabled**, so the 129 collectives of a decode step sit inside the measured
region and the phase profile below is what separates the kernel time from them.

| arm | prefill tokens/s (run 1, run 2) | decode tokens/s (run 1, run 2) |
|---|---|---|
| whole head | 1006.29, 1005.47 | 4.27821, 4.28726 |
| value-axis slice | 1101.57, 1099.65 | 4.23902, 4.26896 |

**Prefill +9.4%; decode flat** (-0.6%, inside run-to-run spread, as expected — decode runs the same
kernel with one token and is not where this cost lives). The decode column is at this commit, before
the overlap-slice and batched-decode merges; the current decode figures are in
[Ascend 910A attention](ascend_attention_optimization.md) section 6.

The phase profile from the same two runs says where the difference is:

| scope (prefill, 48 calls) | whole head | value-axis slice |
|---|---|---|
| `gated_delta` | 2.46620 s | 2.01648 s |
| prefill `TOTAL` | 8.86414 s | 7.99202 s |

The operator itself drops 18.2%, matching the bench, but the prefill only drops 9.8%, because
`gated_delta` is 28% of the profiled prefill rather than all of it. That gap is the whole story of
this change's ceiling: no further tuning of this kernel can act on the other 72% of the prefill.

### The generated tokens are not a usable A/B signal here

The arms were also compared on generated tokens, and that comparison does not survive: the
**baseline** runs disagree with each other. Step-0 greedy token across the four runs, identical
binaries, identical prompt, no other activity on the part:

| run | step-0 token | top logit | checksum |
|---|---|---|---|
| whole head, rep 1 | 125017 | 10.6346 | 10.2468 |
| whole head, rep 2 | 142711 | 10.4235 | 10.2133 |
| value-axis slice, rep 1 | 152058 | 10.939 | 10.0803 |
| value-axis slice, rep 2 | 125017 | 10.6803 | 10.2094 |

Three distinct continuations in four runs. The two runs either side of the kernel swap are no more
alike than two runs of the same kernel, so the divergence is not the slice, and by a margin (top logit
10.42-10.94, checksum 10.08-10.25 on a 4966-token prompt) well above fp16 rounding.

**The conclusion originally drawn from it — that with the TP all-reduce active, prefill logits on this
stack are simply not reproducible run to run — is withdrawn.** The observation is not; the four runs
did disagree. What replaces the reading is smaller than it: the same signature was found later in the
decode path, and was traced there to a pooled workspace slot rather than to the platform. The
partial-RoPE table was uploaded into `WorkspacePool`'s `Intermediate` slot, the one the attention
launchers use for scratch, so a blocking H2D copy could land on the table between an attention kernel
the host had already queued and the rope kernel that read it — one cache sector of the table, once per
process, with the token that comes out of it unconstrained. That is fixed by giving the table its own
slot ([#280](https://github.com/lvyufeng/PocketLLM/pull/280)).

These four runs predate that fix, so the question the paragraph above leaves open is whether the
prefill divergence *is* that race or a second cause of its own. **It has since been re-measured, and
the spread is gone.** Eight runs of one binary over a 4966-token prompt, 9 new tokens, the TP
all-reduce left enabled exactly as above and the same launcher:

| run | step-0 token | top logit | checksum |
|---|---|---|---|
| 1-8, identical | 271 | 14.8354 | 14.8354 |

All eight are the same line to the four decimals the log prints, and all eight carry one identical
9-token sequence. The prompt is not the original one — its text was not retained, which is why the
logit is 14.8354 rather than the 10.42-10.94 the table above spans — so what is compared here is the
run-to-run spread, which is the thing that was broken: 0.52 of top logit across four runs before, none
at all across eight runs after. Reproducibility in this phase is restored, and the attribution above
is supported by it rather than merely left standing on it.

The timings from the same eight runs separate cleanly from the tokens, which is the other half of the
same point: prefill 1450.6-1474.3 tokens/s and decode 9.25-9.58 tokens/s, spread both ways while not
one of the eight moved a token. That is the behaviour this stack is supposed to have, and it is what
the four runs above were missing. Their repeat pairs agreed to under 1% on throughput — the two
whole-head runs came in at 1006.29 and 1005.47 prefill tokens/s — which is exactly what made a
divergence in the tokens look like a result rather than a fault.

What that does not establish is that the mechanism was *the same one*. Both measurements put a
collective in the path and neither can remove it — `QWEN_TP_WORLD=1` OOMs on this checkpoint — so the
re-measurement shows the instability is gone on the tree that carries the fix, not that the fix is
what removed it. The operational consequence below is unchanged and so is rule 8: establishing
run-to-run stability before comparing generated tokens is cheap, and this is the second time on this
backend that skipping it would have read a defect as a property of the stack.

Two consequences for reading this page, unchanged either way. The token sequences are deliberately not
used as evidence above, and anything on this backend that compares generated output across
configurations has to establish run-to-run stability before the comparison means anything. The
`--check` arms in section 3 are unaffected — they compare the operator against a double-precision host
reference in one process, with no collective in the path. The rule is recorded with the other
reporting rules in [Benchmarking and reporting rules](../guides/benchmarking.md).

The section heading stays as it is, for a different reason than the one it was written for: the four
runs above still do not support an A/B, and the comparison that would now be stable cannot be redone
in this form — the branch binary that produced both arms lost its shape flags when the change merged
(section 6), so the two kernels are no longer switchable from `master` on this command line.

## 5. What this does not do

- **It does not make the gated-delta op bandwidth-cheap.** The op moves 1.6 MB of fp32 state per
  (layer, head) either way, split or not. An 18% issue-bound win is what was available; the residual
  is the state traffic itself.
- **It does not close the decode gap.** Decode is measured here at 4.24-4.29 tokens/s, and its profile
  is dominated by dense fp16 weight bytes read per layer (`pr.mlp.gate_up`, `layer.swiglu`,
  `pr.mlp.down`), not by this kernel. See [Ascend 910A attention](ascend_attention_optimization.md)
  section 6 for the arithmetic; weight quantization is the lever there, not kernel work.
- **It does not revisit the four-way split on second-generation silicon.** On a 20- or 24-core part
  the geometry changes completely and the two-round argument above stops applying. It is why
  `value_split_for` is a rule over the head count and the core count rather than a constant of two:
  the shard is 12 heads at TP4 but 24 at TP2, and 24 doubled is 48 items that need two passes over
  the same 30 cores, where the cut loses rather than wins.

## 6. Reproducing

```bash
source scripts/ascend_env.sh
scripts/build_ascend.sh build
cmake --build cpp_engine/build-ascend -j 16 \
    --target bench_qwen_ascend_delta test_qwen_ascend_group_b

# the operator at the shard shape the engine runs; the merged bench calls the row
# count `--tokens` and has no `--rows`
./cpp_engine/build-ascend/tests/bench_qwen_ascend_delta \
    --device 0 --tokens 4096 --heads 12 --key-heads 4 --iters 20

# the parity comparison, now a case inside the group-B test
./cpp_engine/build-ascend/tests/test_qwen_ascend_group_b

# end to end
QWEN_PHASE_PROFILE=1 scripts/run_qwen_ascend_tp4.sh "<4966-token prompt>" 9
```

The kernel-level unsliced baseline is reproduced by setting `kMaxValueSplit` to `1` in
`cpp_engine/backends/ascend/kernels/qwen_gated_delta_geometry.hpp` and rebuilding. Nothing else needs
to change: the launcher's split is clamped to that constant, so the extra grid blocks the launcher
still requests are work items the kernel declines to take.

The 10.22 and 8.37 us/head-token of section 3 and the `--check` mismatches came from the branch
binary, which took the shape as `--rows` and carried its own host-reference check. The merged bench
takes the shape as `--tokens` and reports a rate only, so that A/B is not reproducible from `master`
as a single switch — the sliced arm is, and the parity it checked is covered by the group-B case
above.
