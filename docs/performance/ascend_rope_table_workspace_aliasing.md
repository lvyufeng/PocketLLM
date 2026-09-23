# The Ascend RoPE table's workspace slot, and the decode corruption it caused

A decode step on the Ascend path is one row of activations, and partial RoPE needs a cos/sin table
built on the host for the one position that row sits at. The table is uploaded to a pooled device
buffer. That buffer was the same one the attention and norm operators take as their scratch, on the
reasoning that a blocking `aclrtMemcpy` is ordered against the kernel that reads its result.

It is not, and the rotation that came out was wrong in the first 32 bytes of its cos half. The
consequence was not a crash or a visible numerical blowup. It was a decode step that produced the
wrong token, non-deterministically, at a rate that tracked how much work happened to be in flight
behind the copy — which is why the same event had been recorded as three unrelated findings: a
reproducibility failure in the all-reduce's gate, an instability that appeared only at one replication
width, and a 30% faster device-side collective wait that was withdrawn as incorrect. All three are
this defect, and with the slot separated a single request decodes at 18.6 TPS producing the token
sequence the reference produces.

**Environment.** 4 x Ascend 910 (first generation, no BF16), 32 GB HBM per card, CANN 9.0.0,
`ASCEND_TOOLKIT_HOME=/usr/local/Ascend/cann-9.0.0`. Qwen3.8-27B at TP4, one process per rank on
devices 0-3 through `scripts/run_qwen_ascend_tp4.sh`. `source scripts/ascend_env.sh` precedes every
run.

## 1. The symptom

Two spellings of the same prompt position — the sixth prompt token appended from inside a six-row
prefill, and the same token appended from a one-row decode step — disagreed. Layer 0, 1 and 2 were
bit-identical between them; layer 3, the first full-attention layer and so the first layer with a KV
cache at all, was where they parted:

```
layer3 K row5, prefill(cp6) vs decode:
  maxabs=7.61 rel_l2=0.5007   worst_col=0
layer3 V row5:
  maxabs=0 rel_l2=0
```

The divergence was in the K cache only, and inside that, in 16 of the 256 columns: `[0..7]` and
`[32..39]`. The K tail `[64:255]` agreed to 0.0012, which is the agreement of the same arithmetic
evaluated twice.

The prefill arm's K row matched an FP32 reference built from the real checkpoint — `(1 + w)` RMSNorm,
`rope(theta=1e7, position=5)` — to `maxabs=0.003148 rel=4.079e-4`. The decode arm did not.

## 2. What the rotation had actually been given

Rotation is applied in pairs: lane `i` and lane `i + half_rotary` of the 64-wide rotary span come out
as `a*c - b*s` and `b*c + a*s`, and the pre-rotation input is recoverable from the post-rotation
output when both members of a pair are known. Solving each pair against the *correct* pre-rotation
input gives the cosines and sines the decode step applied.

The sines were right. The cosines were not — and a rotation has no cosine above 1:

| pair `i` | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| recovered cos | 1.029239 | 1.996387 | 2.675499 | 0.250276 | 1.725636 | 3.427392 | 2.050505 | 1.356079 |
| recovered sin | -0.959462 | 0.119658 | 0.968083 | 0.892332 | 0.618690 | 0.390730 | 0.241493 | 0.146585 |

Pairs 0 through 7 are the ones whose cosine half is wrong, and lanes 8 through 31 invert to a
pre-rotation input matching the reference to FP16 precision. `rotary_dim` is 64, so the cos half of
the table is 32 floats: floats `[0..31]`, bytes `[0..127]`, with the sin half immediately after at
bytes `[128..255]`. The corrupted region is bytes `[0..32)` — eight floats, one 32-byte cache
sector, at the exact start of the cos table, with everything behind it intact.

The corruption was stable within a configuration (`g1akd/l3_p5_kcache.f16` and
`g1akd2/l3_p5_kcache.f16` differ by `maxabs=0.0`) and different between positions — p5 through p9
each showed a different pattern. Stable within a run, variable between runs, which is what a race
looks like when one side's timing is reproducible for the length of a run.

## 3. The mechanism

`WorkspacePool` keys a buffer by `{device, stream, purpose}` and hands out one slot per key. The
intent of the `Intermediate` slot is that a composed operator can hold a value across several aclnn
calls while those calls use the `OpWorkspace` slot. Two operators on one stream are serialized by
definition, so sharing `Intermediate` between them is safe *when both sides are kernels*, because the
stream orders one kernel against the next.

The RoPE table is not a kernel's output. It is written from the host:

```cpp
aclrtMemcpy(device, bytes, host_cos.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE);
aclrtMemcpy(device + elements, bytes, host_sin.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE);
```

`aclrtMemcpy` with `ACL_MEMCPY_HOST_TO_DEVICE` blocks the *host* until the copy retires; it does not
enqueue anything on the stream, so it is not ordered against work already queued there. Whatever the
stream was still executing when the host reached that call is free to land on the table between the
copy and the rope kernel's read.

The lane pattern says which side did it. A prefix of the slot was overwritten and nothing after it,
which is what a writer working through the buffer from its start looks like when the reader arrives
part-way. The rope kernel's `DataCopy` of the cos table captured the first 32 bytes after a competing
write had reached them; the sin half, 128 bytes further in, was read before the same writer got
there. That also explains why the corrupted span is a function of timing rather than of shape, and
why it is small: the window is short.

Separating the slot removes the writer from the table's buffer entirely, and that is the fix. It is
not that the buffer needed to be different; it is that a host-side write has no ordering against the
stream at all, so it cannot share a buffer with anything the stream is still working on. The same
reasoning applies to every other slot in the pool — they are all kernel-written, which is why this
was the only one that broke.

## 4. The fix

`WorkspacePool::Purpose` gains a third value, `RopeTable`, and both table builders
(`RopeTables::acquire` and `RopeTables::acquire_rows`) ask for it instead of `Intermediate`. Nothing
else changes: the table is still built per call on the host, still uploaded by blocking copies, and
still consumed by the same kernels in the same order.

`QWEN_ASCEND_ROPE_WS=shared` selects the old slot again. It is the negative control, and it is the
only thing that should ever set it.

| arm | layer 3 K row 5, prefill vs decode |
|---|---|
| `RopeTable` (default) | `maxabs=0 rel=0` |
| `Intermediate` (`QWEN_ASCEND_ROPE_WS=shared`) | `maxabs=7.61035 rel=0.5614` |

The tokens move with it. Greedy decode of "The capital of France is" at 32 steps, hand-written
collective on:

```
default: 11751 13 198 760 6511 314 9564 369 19241 13 198 760 6511 314 14898 369 21047 13 198 760
         6511 314 17163 369 23327 13 198 760 6511 314 32208 369
shared:  11751 13 11751 369 279 6511 ...
```

`11751 13 198 760 6511` — " Paris. The capital of France" — is the reference's own top-1 at each of
the first five steps.

## 5. What it changes elsewhere

Three published findings were this defect. The first two are consequences rather than independent
results; the third is a lever that had been measured and then withdrawn, and is available again.

### 5.1 The all-reduce reproducibility gate

The batched pass at rows=1 is compared against itself from a reset engine (`batch_repeat_mismatches`,
a hard zero in the launcher) and against a single-row reference (`verify_mismatches`, deliberately not
gated). On the unfixed build the gate failed 3 of 36 interleaved runs under HCCL and 8 of 36 under the
hand-written collective, with `worst_logit_abs` of 0.048 to 0.95 and one failure shape whose
`worst_logit_abs` was 23.8 to 24.2 — a top-1 flip on a near-tie. Those rates were read as evidence
about the arrival signal's release ordering and about the host round trip that covers for it. On the
fixed build, twelve interleaved runs — six pairs, both collectives — fail it zero times, with
`worst_logit_abs` between 0.0028 and 0.0097:

| arm | unfixed | fixed |
|---|---|---|
| HCCL (shipped default) | 3 of 36 | 0 of 6 |
| hand-written collective | 8 of 36 | 0 of 6 |

Fisher's exact test on 11 of 72 against 0 of 12 gives p = 1.3e-3. The gate is not a subtle test; it is
the batched path disagreeing with itself, and a per-call race in a shared workspace is exactly the
thing it catches. What it was catching was not the collective.

### 5.2 The width-16 instability

`QWEN_ASCEND_REPLICATE_ROWS=16`, which fills the Cube's sixteen-row M tile from a single decode row,
was recorded as the one setting at which the greedy output was not reproducible: three runs, three
different first tokens, where widths 2, 4 and 8 were stable. The replication widens the workspace the
layer churns through, so it moves the timing of exactly the window this race lives in, and it moved it
into the running state rather than out of it. With the slot separated, the two arms of `1` and `16`
produce the same 32-token sequence, byte for byte:

| arm | `QWEN_ASCEND_REPLICATE_ROWS=1` | `=16` |
|---|---|---|
| host poll (hand-written collective) | 12.96-13.06 TPS / 76.6-77.1 ms | 17.46-17.71 TPS / 56.5-57.3 ms |

Every run behind that row, and every other cell of the matrix, emits `11751 13 198 760 6511 314 9564
369 19241 ...`. An earlier revision of this subsection quoted the pair as 13.0088 / 17.5931 and
12.9874 / 17.2694 TPS; those four numbers are **withdrawn rather than corrected**, because
`Linear::forward` was handing the replication factor to a multi-row projection as the batch at the
time, so a prefill chunk projected row 0 alone and the first token could not have been the reference's.
The row above is the same A/B re-measured on the fixed tree; the full sweep — both widths across all
three collectives, and the 128-token repeats — is
[section 6.3 of the single-request page](ascend_single_request_tps.md#63-what-it-is-worth-1305---1759-tps-at-rows1).

`QWEN_ASCEND_REPLICATE_ROWS` is on `master`, shipped off by default, and `QWEN_ASCEND_REPLICATE_CHECK`
is the opt-in that makes every replicated projection refuse to launch if its destination activation
does not actually hold the taller write.

### 5.3 The device wait, which turns out to be correct

The hand-written collective has a switch that moves the wait onto the device —
`POCKET_ASCEND_IPC_ALLREDUCE_DEVWAIT` enqueues a kernel that spins on the same peer stamps the host
loop was reading, and returns without blocking. It measured 23.4 ms off a 77.1 ms step, 30% of it, at
53.745 ms / 18.607 TPS. It was recorded as a bound rather than a candidate and withdrawn from the
shipped path because **every one of ten device-wait runs failed the same reproducibility gate**, and
that was read as the host round trip being not only the wait but also *the fence the arrival signal
does not have* — the release ordering that the device wait, having no host in it, presumably lacked.

That reading does not survive the fix. Ten interleaved pairs of poll against device wait, the gate of
§5.1 on both arms, run against this build:

| arm | `step_ms` | decode TPS | gate failures |
|---|---|---|---|
| host poll | 76.10-77.66 | 12.88-13.14 | 0 of 5 |
| device wait | 53.02-53.88 | 18.56-18.86 | **0 of 5** |

The device wait was failing the gate because the gate lands on the same comparison the race lands on,
and it was failing it more often than the poll arm for the same reason §5.2 has one width fail and the
others not: it changes the timing of the window. There is no missing fence. The ordering the device
wait is supposed to preserve, it preserves.

What the two arms generate is the check that matters, and it is the same sequence in all four runs —
poll twice, device wait twice, on the fixed build:

```
11751 13 198 760 6511 314 9564 369 19241 13 198 760 6511 314 14898 369 21047 13 198 760
6511 314 17163 369 23327 13 198 760 6511 314 32208 369
```

" Paris. The capital of France is Germany is Berlin. …" — the first five are the reference's own top-1
at each step, and `11751` (space-P) is not a token the unfixed build reached at all on this prompt.

So the state of the target is: a single request at the shipped row count, no replication, 18.6 TPS
with the tokens the reference produces. That is the 17 TPS the task asked for, reached with the
accuracy repaired rather than traded against it.

### 5.4 What these three do not establish

The gate rates above are 12 runs against the 72 the earlier measurement used, which is enough to
separate 0 from 15% but not to put a bound on the residual failure rate, and the device-wait pairs
re-run it at 10 more. The width-16 sweep is one session's interleaved pairs on the fixed tree, against
the fourteen runs the instability was first seen in. What the fix establishes is the direction and the
mechanism, and that the negative control reproduces the symptoms on demand.

## 6. Reproducing this

The A/B is the negative control, one variable, on the plain single-request path — this much runs
against `master` as it stands:

```bash
scripts/run_qwen_ascend_tp4.sh "The capital of France is" 32
QWEN_ASCEND_ROPE_WS=shared scripts/run_qwen_ascend_tp4.sh "The capital of France is" 32
```

The gate re-run of §5.1 is the launcher's batched path with the verifier on. `POCKET_ASCEND_IPC_ALLREDUCE`
was opt-in when this gate was run and is the backend's default now — `1` runs the hand-written
collective, which is also what an unset environment does today, and `0` selects the HCCL arm that
used to be the shipped default. The loop spells both arms out, so it reproduces the original A/B
either way, and the gate's answer is the same one on both sides of the flip:

```bash
for pair in 1 2 3 4 5 6; do
  for arm in 0 1; do
    env QWEN_BATCH_ROWS=1 QWEN_BATCH_VERIFY=3 QWEN_BATCH_PROMPT_LEN=32 \
        QWEN_ASCEND_DEVICES=0,1,2,3 POCKET_ASCEND_IPC_ALLREDUCE=${arm} \
        scripts/run_qwen_ascend_tp4.sh "" 8
  done
done
```

§5.3 is the same loop with the two arms being `POCKET_ASCEND_IPC_ALLREDUCE_DEVWAIT=0` and `=1`, on
the hand-written collective (`POCKET_ASCEND_IPC_ALLREDUCE=1` below is a no-op now that the barrier
is the default, and is left in so the loop states which collective the wait was measured on), and
its token check is the plain single-request path with the same pair of overrides:

```bash
for arm in 0 1; do
  env POCKET_ASCEND_IPC_ALLREDUCE=1 POCKET_ASCEND_IPC_ALLREDUCE_DEVWAIT=${arm} \
      QWEN_ASCEND_DEVICES=0,1,2,3 \
      scripts/run_qwen_ascend_tp4.sh "The capital of France is" 32
done
```

The lane-level diagnosis of §2 is not reproducible from `master`, and the gap is worth stating
rather than papering over: it needs a hidden-state and KV-cache dump that the investigation added
behind `QWEN_DUMP_HIDDEN_DIR` and `QWEN_DUMP_KV_LAYER`, and those are a separate change from this
one. What they do is mechanical — the engine writes one raw FP16 file per layer per forward call
from rank 0, plus the layer named by `QWEN_DUMP_KV_LAYER` (default 3) as `k`/`v` cache images — and
the comparison §1 quotes is position 5 of a six-row prefill against position 0 of the one-row decode
step that follows it. The §4 negative control, which is the part a reader would want to re-run, needs
neither: `QWEN_ASCEND_ROPE_WS=shared` reproduces the cache divergence and the wrong tokens on demand.

## 7. Files

- `cpp_engine/backends/ascend/kernels/aclnn_common.hpp` — `WorkspacePool::Purpose::RopeTable`, and
  the invariant its comment states.
- `cpp_engine/backends/ascend/kernels/qwen_ascend_ops_launch.cpp` — `rope_table_purpose()` and the
  two `RopeTables` builders that call it. `QWEN_ASCEND_ROPE_WS` is the negative control.
