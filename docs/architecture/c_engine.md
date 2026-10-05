# The C engine

`src/` is a C++17 library, `libpocketllm.so`, that runs a checkpoint end to end: it reads the GGUF,
tokenizes, walks the graph, and returns logits. It is not a Python extension — `pyproject.toml`
declares no `ext_modules` and nothing compiles at `pip install` — and the host shell reaches it
through `ctypes` at the eight symbols [`pocketllm.h`](https://github.com/lvyufeng/PocketLLM/tree/main/src/include)
declares.

The pages beside this one describe the **Python** layer: `pocketllm.kernels` is the spec the C ABI
was derived from, `pocketllm.backends` is the registry `pocketllm devices` reads, and
`pocketllm.engine` is the executor the reference backend runs on. This page is the other half — the
native tree that is the runtime a host actually loads.

## What runs today

Qwen3-0.6B, dense, greedy, on `cpu` and on `cuda`. The GGUF reader, the byte-level BPE tokenizer, the
28-block forward pass and the decode loop are all C, and all three are checked against llama.cpp as
the authority:

| Layer | Checked against | Test |
|---|---|---|
| GGUF reader | the Python `GGUFReader`, tensor by tensor | `tests/native/test_gguf_reader.py` |
| Tokenizer | the Python tokenizer and `llama-tokenize` | `tests/native/test_tokenize.py` |
| Forward pass | llama.cpp's low-level API, logits and greedy tokens | `tests/native/test_forward.py` |

llama.cpp is a *test* dependency and not a build one: `libpocketllm.so` links nothing but the CUDA
runtime when it is built. `tests/native/llama_oracle.py` compiles `src/tools/oracle.cpp` against
llama.cpp's headers, and that tool drives the low-level API directly rather than shelling out to
`llama-cli` — which in this build is a chat client, and answers a different question.

Everything else is unbuilt. A checkpoint whose architecture is not `qwen3` opens successfully and
cannot run; `pocketllm run` and `pocketllm serve` still parse their arguments and exit.

## The backend interface

`src/kernel/backend.h` is the design. `Backend` is the set of operations the graph is built from,
over opaque `DeviceBuffer` handles, and `Qwen3Model` is written against it and nothing else — so
`pocketllm_open(session, "cuda")` is the entire difference between running on the host and running on
a card.

```
src/kernel/backend.h          Backend, DeviceBuffer
src/kernel/registry.cpp       which backends this build has
src/kernel/kernels.{h,cpp}    the CPU ops, as free functions
src/kernel/cpu/backend.cpp    them, behind the interface
src/kernel/cuda/backend.cu    the device ops, behind the same interface
```

**One process owns one device.** There is no device index, no rank, no stream per rank and no
collective, and the CUDA backend binds device 0 and stops there. A checkpoint that does not fit is
quantized further; it is never split. Adding a device list back would resurrect a feature this tree
deleted on purpose.

### The CPU ops are the oracle, not the implementation

`kernel/kernels.cpp` holds the CPU math as *free functions over host pointers*, and
`kernel/cpu/backend.cpp` is a thin adapter that calls them. The device kernels are written separately
in `cuda/backend.cu` and share no code with them.

That duplication is deliberate. Those functions are the reference a new kernel is checked against, so
their value is that they were written independently and read simply; a shared implementation would
agree with itself no matter which of the two were wrong. The same reasoning the Python tree applies
to `pocketllm.backends.reference`, applied one level down.

### What the interface carries that a function list could not

- **`copy_to_device` and `copy_device_to_device` are separate.** The CPU cannot tell them apart — both
  are `memcpy` — and that is exactly why the distinction is in the interface. The KV-cache append is a
  device-to-device copy, and expressing it as the host-to-device one works perfectly on the CPU and
  dereferences a device address as host memory on a card. `cuda/backend.cpp`'s `cudaMemcpyDeviceToDevice`
  is the line the distinction exists for.
- **A `DeviceBuffer` is an integer and a byte count**, not a pointer, so the graph may do address
  arithmetic on one to reach a row or a layer without ever dereferencing it. The final projection reads
  the last row of the activation through exactly that, and the KV cache addresses layer slabs through
  it.
- **`attention_scratch` is a query, not a constant.** How much scratch a call needs is a property of
  *how the backend is parallelized*: the CPU now runs the `(query, head)` pairs across its thread pool
  and needs one score row per concurrent task, and a GPU runs them as concurrent blocks and needs one
  row each. The graph cannot know which, so it asks. Both backends size the buffer from the same
  partition the kernel takes — `parallel_tasks` for the CPU, the block count for the card — so the
  allocation and the kernel cannot disagree.
- **`argmax` is an operation.** A caller that only wants the next token transfers four bytes instead
  of the whole logit vector — 600 KB at this vocabulary.

### The C surface is a strict subset of the ABI, on purpose

The ABI declares 17 ops. The engine implements 12 of them as ops — `rms_norm`, `gemm`, `gemm_quant`,
`embedding`, `embedding_quant`, `silu_mul`, `rope`, `attention`, `argmax`, `softmax`,
`logits_temperature`, `topk_sample` — and the rest are either *inlined into the graph* or *not
needed yet*. Neither case is a missing kernel, and the distinction matters when reading the
conformance test, which drives the 12 and deliberately skips the others.

(`embedding_quant` is a second entry point rather than a flag on `embedding` because the two read
different memory — a float table and packed bytes — and it exists because Qwen3 ties its output
projection to the embedding, so `token_embd.weight` is both the graph's first op and a weight the
final matmul contracts against.)

**Inlined, because the graph does it somewhere there is no op.** `add` is `gemm(accumulate=true)`:
the residual is the product's job, so there is no second pass over the activation. `cache_append` is
a `copy_device_to_device` in `qwen3.cpp` rather than a kernel — the positions are always contiguous
and `start_pos`-aligned, so the general scatter the reference op allows (`cache[positions[i]] =
values[i]` for arbitrary indices) has no caller. `cache_truncate` likewise: `reset()` drops the
length without clearing rows, because the next append overwrites what it needs.

**Not needed yet.** `layer_norm` — Qwen3 is all RMSNorm, and the only `layer_norm` string in `src/`
is the metadata key `attention.layer_norm_rms_epsilon`, which is an RMS epsilon. `mul`, `moe_ffn`: no
dense Qwen3 path reaches them.

### The sampling ops are the exception to the backend split

`softmax`, `logits_temperature` and `topk_sample` are implemented on both backends, but unlike every
op above they are **not two independent transcriptions** — the CUDA methods round-trip the logits to
the host and call the same `kernel::` functions the CPU backend calls, so the two agree bit for bit.

The reason is that a sampler is not a data-parallel expression: it ranks a whole distribution,
truncates it, and inverts a cumulative sum at one draw. That is a sequential decision over the entire
vocabulary, and the useful place to make it is the host — a device transcription would have to
re-derive the truncation arithmetic and the tie rules, which is a second place for the token to be
different rather than a check on the first. The transfer is off the hot path: it happens once per
token, on a vector the caller usually has to read anyway to report the argmax.

`test_the_backends_agree_with_each_other` is therefore vacuous for these three — it tests the
transfer, not the arithmetic — and their correctness rests on the reference comparison in
`tests/native/test_op_conformance.py` instead. The one place the C sampler and the numpy reference
may differ is recorded there as a test: the reference accumulates its cumulative sum in float32 and
the kernel in double, and over 151936 additions that gap moves the crossing index in a flat tail. The
kernel is the more accurate of the two, and the test says so by recomputing the cumulative in float64
and showing which side agrees with it.

`softmax`'s conformance bound is likewise worth knowing: it is an *absolute* tolerance and not the
relative one the other float ops use, because a sequential float32 sum over a whole vocabulary
differs from numpy's pairwise one in the last ulps of the row total, and a relative bound there would
be measuring the reduction order rather than the op.

These are gaps in *coverage*, not disagreements about arithmetic, and they close when a model that
needs them arrives. What is worth stating is the other kind of gap — where both sides implement the
op and give **different answers**:

- **`embedding` with an id outside `[0, vocab)`** zeroes the row. The reference indexes the table:
  a negative id wraps to a real row from the end (silently plausible output), and an id past the end
  raises `IndexError` — which on a device is a fault rather than an exception. Zeroing is visible in
  the logits instead.
- **`argmax` of a row containing a NaN** returns the largest *finite* index. `np.argmax` returns the
  first NaN's index, because every comparison against a NaN is false. Different integers, not
  different roundings.

Both are the C behaviour by choice and both are pinned by
`tests/native/test_op_conformance.py`, so a change to either is a failure rather than a drift.

### The packed weights

`src/quant/blocks.h` decodes `q4_k` and `q6_k` one weight at a time, and it is the one header the CPU
and the CUDA kernels *share* — the exception to the independence above, and a deliberate one. What
those two kernels owe each other is correctness of an *algorithm*, and two implementations of an
algorithm can only disagree by one of them being wrong. A bit layout is not an algorithm; it is a
property of the file, and a second transcription of it would be a third place for the layout to be
wrong, agreeing with the first exactly where both misread the same nibble.

`python/pocketllm/quant/k_quants.py` is the other transcription, and it is the oracle the packed
kernels are checked against one op at a time on inputs the test authors. That comparison is
exact — and blind in one direction: if all three read the format the same wrong way, they agree
perfectly. The authority that closes that hole is llama.cpp, which wrote the format and quantized
the file, and it is reached end-to-end rather than per op in
`tests/native/test_quantized_forward.py`.

That file also records where llama.cpp is *not* an oracle, and the answer changed when the CPU kernel
gained its integer path. Both engines now run the same `q4_K` arithmetic — llama.cpp's dot quantizes
the activation to int8 (`block_q8_K`) and ours does too, since `src/quant/q8k.h` — so the GEMM is no
longer the difference. Two others are, and they are named rather than absorbed into a tolerance:

- **The attention reduction.** llama.cpp's `flash_attn_type` default is `AUTO`, which enables flash
  attention on the CPU path, and a flash kernel's running maximum with a rescaled merge is different
  arithmetic from one shift per score row. On this checkpoint the difference is not last-bit:
  llama.cpp's own greedy sequence changes at the second token when flash attention is switched off,
  its logits move by up to 1.16, and the top-2 margin deciding that token is 0.0925 against 0.0196.
  On f16 the same switch moves 0.0004 of the logit spread, which is why the f16 tests never saw it.
  `tests/native/llama_oracle.py` pins the mode by name for this reason, and
  `tests/native/test_quantized_forward.py` records which mode each backend is compared against: the
  CPU's attention lands on the full-softmax token, the card's on the flash one.
- **The activation quantization's error.** It is absolute, so a logit near zero carries a large
  relative error while staying a percent of the range — which is why the quantized comparison is a
  bound on the error relative to the logit spread and not elementwise.

So the two quantized results are compared on their argmax, their logits at that global bound, and
their greedy sequence. Both sequences match their own convention exactly, and both differ from the
f16 sequence from the second token on: quantizing to 4.5 bits per weight is lossy enough to change
the continuation, which is the format working rather than a kernel failing.

### The CPU kernels use the whole machine

The CPU backend is not one core. `kernel/parallel.h` is a persistent `std::thread` pool — created
once, on first use, because the graph makes roughly two hundred kernel calls per token and
spawning threads per call would cost more than the arithmetic — and every kernel that has an
independent-output axis splits it across that pool.

**Not OpenMP.** `-fopenmp` would make `libgomp` a runtime dependency of `libpocketllm.so`, and the
whole reason this library exists is an edge and mobile target where the loader should have to find
as little as possible. A `std::thread` pool needs nothing beyond libstdc++ and pthread.

**The workers spin, and that was the single largest fix in this path.** The pool originally had its
workers wait on a condition variable and the caller notify at the end of each job — the textbook
shape, and wrong for this workload by an order of magnitude. A decode token is about 198 `parallel_for`
calls (28 layers of seven GEMMs plus the embedding and the head) and each is *short*: a 1024×1024
projection is on the order of a hundred microseconds at 22 threads. A futex round trip costs ten to
thirty of those microseconds on this host, so most of a token went into waking threads that had just
gone to sleep. The tell is the scaling: the condvar pool gained nothing at all from the first eight
threads to twenty-two (`tg64`: 22.96 → 22.56 tok/s), because every added thread paid the same wake-up
cost. Replacing the wait with a spin on a generation counter and the notify with a spin on an arrival
count took the same two points to **31.23** and **56.10** tok/s.

The cost is stated rather than hidden: while a job is in flight the pool's cores are at 100% even
when the job is tiny, where the sleeping version let them idle. For a batch server sharing a machine
that is a reason not to do this, and `$POCKETLLM_CPU_THREADS` is the polite setting there. For one
session driving one device from one thread — this engine's whole model — the machine is the session's,
and the idle time was the wake-ups that are no longer happening. There is deliberately no fallback
to sleeping on a long spin: the caller enters the barrier only after running its own share of the
work, so the longest a worker can still be busy is one chunk, and a worker descheduled mid-chunk
would be waited for either way.

The publish of a job and the barrier that ends it are one critical section, and the reason is worth
stating because it is not the obvious one. The barrier alone already guarantees the caller's frame
outlives the job — a worker publishes its arrival with release ordering after its last chunk, so
"arrived" implies "out of the job". What the lock adds is that no *second* job can be published
while one is outstanding: without it, a worker descheduled through a whole job could wake to see the
next generation, join the newer job, and never arrive for the older one — and the older job's caller,
spinning on the arrival count, would wait for an arrival that can no longer come. That is a hang,
not a wrong number, which is why it is a lock rather than an argument.

**The one rule is that a reduction is never split.** Every call site splits an axis whose iterations
touch disjoint memory — the output column of a GEMM, the token of a gather, the element of an
elementwise op — and leaves each output's accumulation over `k` whole inside one task. Splitting `k`
and summing partials would be faster and is forbidden: it changes the association order, so the
result would move with the thread count. `gemm_quant`'s comment says so where someone would be
tempted. Because the partition is a fixed contiguous split of an independent axis, the output is
*bit-identical* to the single-threaded kernel, and `tests/native/test_cpu_parallel.py` holds that
exactly by running each op at one thread and at eight and diffing the bytes.

**The other half of the rule is that the split has to be worth making, and the call site that forgot
is the cautionary example.** `attention` splits `(token, head)` with a grain of `kAttentionGrain`,
and a grain of 32 is larger than the whole job: `partition_size` caps the chunk count at the thread
count *and* at `ceil(total / min_per_task)`, so a decode step (`q_len = 1`, 16 heads) collapses to one
task and runs on one core of twenty-two. The grain means "the smallest span worth waking a thread
for", and reading it off a prefill-sized job — where 16 units is below any sensible wake-up cost — is
what set it to 32. It is now 1: every `(token, head)` unit is a task. The change is worth 1.6× on
decode at 22 threads (`tg32` 48.45 → 77.31) and exactly nothing at one thread, which is what
identifies it as a partition and not an arithmetic change. The `attention` comment states the other
constraint on the same number: the unit is the whole of a task's work and cannot be sliced finer,
because its three passes are chained through `max_score` and `total`.

The thread count is `$POCKETLLM_CPU_THREADS`, falling back to every hardware thread, clamped to
`[1, 256]`. The default is all cores because the requirement is speed with nothing to configure and
the result does not depend on the count; `=1` reproduces a single-threaded number, and on a shared
host it is the polite setting. `build/pocketllm-bench` prints the count it resolved, so a table row
is never ambiguous about how many cores produced it.

### The vectorized packed GEMM

`gemm_quant` is where the FLOPs are — more than 99% of them, at both prefill and decode — so it is
where the work went. The scalar path `dot_q4_k_block_scalar`/`dot_q6_k_block_scalar` decodes one
super-block and returns its contribution, and the GEMM sums those. Both facts are wrong for a
vector kernel: a per-block dot ends in a horizontal reduce, and a 256-weight accumulator is only
sixteen FMAs deep. The AVX2 path folds every block of a row into two lane-wise accumulators and
reduces once per output element instead of once per 256 weights, which is why `dot_row` takes the
whole row and walks the blocks itself.

The *decode* is unchanged and stays exact: the group's `d * scale` and `dmin * minimum` are hoisted
out of the 256-weight walk and computed once per 32 weights, and every weight the vector path reads
is the weight `dequant_q4_k`/`dequant_q6_k` produces. Only the association of the sum moves — the
sum is reassociated and `a * b + c` contracts into an FMA, both of which move the last bits. That was
the accepted trade for `-march=native`, and it is why the safety net is not bit-identity but three
things: the existing tolerance tests, unmodified; a check that the AVX2 decode agrees with the scalar
decode to `1e-6` relative, far below the percent a wrong nibble would move; and the llama.cpp token
match, which is what catches a last-bit change that flips an `argmax`.

`-ffast-math` is deliberately absent. It would add full reassociation and `-ffinite-math-only`, and
the shifted softmax depends on neither being there.

### The integer activation path

The AVX2 path above still decodes a weight to float for every weight it reads. llama.cpp does not:
its `q4_K` and `q6_K` dots quantize the *activation* to int8 once per 256-weight block
(`block_q8_K`) and take the integer route through both operands — unpack the weight to 4 or 6
unsigned bits, multiply against the signed activation byte with `_mm256_maddubs_epi16`, fold the
block's scale in with `_mm256_madd_epi16`, and apply one float multiply per block. `src/quant/q8k.h`
is that block and the quantizer for it, a term-for-term port of `quantize_row_q8_K_ref`, and
`dot_row_q8k` in `kernels.cpp` is the integer dot: `bsums` is what lets a weight's per-group
*minimum* be applied with one `madd` per block instead of a per-weight subtraction.

The cost is a real precision change and it is stated rather than buried: an activation element picks
up up to half a step of its block's scale, and the measured effect is 0.5–0.7% of the output's
magnitude on the conformance shapes, against a `QUANTIZED_RTOL` of 0.05. It is also *the same*
error llama.cpp carries, which is what makes the two engines' quantized logits comparable at a bound
of a few percent of the logit range where they used to be comparable only in ordering.

The exact path is not dead code: a build without AVX2 uses it, and `$POCKETLLM_CPU_EXACT_GEMM`
selects it on one that has AVX2 — which is how the fast path's error is measurable on a real
checkpoint rather than only asserted, and what the tests use to compare the two paths on identical
input.

### The activation quantizer runs on the pool too

Quantizing the activation is the integer path's entry fee, and it used to be paid on one core: the
loop ran on the caller's thread before the parallel walk, so it was the one part of a prefill GEMM
that did not scale. At `m=512, k=1024` that is 2048 independent 256-weight blocks, measured at
**1087 us against a 4244 us call** — a fifth of the GEMM on one core while twenty-one sat at the
barrier. It is now a `parallel_for` over blocks and measures **55.6 us**, 19.5× on the term.

The block is the unit and it stays whole. `quantize_q8_block` is one block's arithmetic — the max
scan, the `-127 / max` scale, the rounding, the `bsums` — and the kernel layer's loop runs it per
block, so the schedule cannot reach inside one. That is what makes the change legitimate rather than
merely faster: each block owns its `d`, `qs` and `bsums` and reads only its own 256 floats, so the
bytes are the serial loop's bytes, and they have to be. The quantized activation is an operand of the
token-for-token llama.cpp match; a schedule that reassociated a block's scale would be a different
model, not a slower one.

The split is: the *block* lives in `src/quant/q8k.h` and the *loop over blocks* lives in
`kernel/kernels.cpp`. `quant/` is the leaf the loader and the reference backend both need, and a leaf
that included `kernel/parallel.h` would drag the thread pool into an install that does not use it.
The serial `quantize_row_q8_k` is still there for callers with no pool to hand — it is the definition
the parallel schedule is checked against.

### The attention score dot, four lanes wide and bit-exact

Decode's cost is not the GEMM once the context is long. At a 512-token context one decode step runs
16 `(token, head)` units × 512 cache rows × 28 layers of 128-wide dots in the attention score pass —
448 MiB of streamed K against a machine whose one core reads a few GB/s. That pass was calling the
scalar `dot`, and a per-pass diagnostic measured it at **2054 us of a 512-row attention call, 1.02
GB/s** — 6× slower than a four-lane `__m128` form over the same bytes (344 us, 6.09 GB/s), on 54% of
the kernel's time.

The scalar `dot` is four independent accumulator chains (`s0..s3`) advancing by four, so lanes 0..3 of
a `__m128` *are* those chains and one vector `mul`/`add` pair replaces four scalar statements. The
horizontal reduce is `(s0 + s1) + (s2 + s3)`, the same order over the same lane values:
`kernels.cpp`'s `dot4`. End to end at the model's own shape — 22 threads pinned to node 0, 512 tokens
of history, the same harness interrupted only by the kernel — the whole attention op falls from
**4321 us/token to 2240** (154.33 → 80.01 us/call over 28 calls), and a decode step's backend time
from 13727 to 11440 us/token.

**The first version of this was `_mm_fmadd_ps` and it was wrong in a way no tolerance caught.** The
shipped scalar loop, compiled as part of a TU with a hundred other functions, does *not* contract its
multiply-add: GCC contracts `s += a[i] * b[i]` only when the multiply has a single use, which the
four separate tails deny it — the shipped `dot` therefore rounds each product before adding it, and an
unconditional FMA does not. The disassembly is not evidence about this and reading it led the first
draft astray. What settled it was a bit-level experiment: 200 000 random 128-wide pairs through an
isolated copy of the shipped source, through the same source in the full TU (observed via
`gemm(m=n=1, k=128)`, the one public path that is exactly one `dot`), and through both candidate
roundings. The shipped function matched "round each product, then add" with **0** mismatches in
200 000 and the fused form with **135 170**.

The consequence of getting it wrong had already been measured: the FMA form moved the model's greedy
completion from 32/32 tokens matching llama.cpp to **1/32**, and the logits error against the f64
oracle at the 2-token prompt was still under `QUANTIZED_RTOL`. A softmax amplifies a last-bit score
difference into a token choice and *hides* it from every bounding test the suite had.

So the vector form is `_mm_mul_ps` into `_mm_add_ps` with `__attribute__((optimize("fp-contract=off")))`,
which reproduces the scalar `dot` bit for bit — verified end to end, not just per dot: float32 logits
identical to the pre-change build at 8 prompt lengths × 2 thread counts, and 64/64 tokens matching
llama.cpp on the recorded prompts. `$POCKETLLM_CPU_SCALAR_DOT` forces `attention` back onto the scalar
`dot`, which is how the *claim* is checked through the public API rather than asserted — one attention
call each way, `memcmp`-identical. That switch exists because the test that would have caught the FMA
draft does not otherwise exist: a float64 oracle at `2 * d * eps` of the output's scale passed the FMA
build with the measured figure at **2% of the bound**.

### The activation scratch is the shape's size, not a fixed one

The integer path needs somewhere to put the quantized activations, and that buffer used to be a
fixed 1024 `Q8KBlock`s on the function's frame — 299 KiB, which is 4–16 blocks of decode and 6144
blocks of a 512-token prefill. A shape past the cap took the exact path instead. The fallback is
*correct*, so nothing failed and nothing was reported: **every prefill longer than 256 tokens was
running the slower kernel, and the only symptom was throughput.** `pp512` on 22 cores measures 98
t/s on the exact path against 180 t/s on the integer one, so this was the largest single term in the
prefill gap.

The buffer is now the shape's size: the stack array when the shape fits (the decode shapes are the
hot path, and 197 GEMM calls per token cannot pay a heap round trip for a 4-block array), a `nothrow`
heap allocation when it does not, and the exact path only if that allocation fails. The test that
would have caught it is in `tests/native/test_cpu_parallel.py`: the default run and a
`$POCKETLLM_CPU_EXACT_GEMM=1` run give *different* answers by construction, so a shape that agrees
with the forced-exact run is a shape where the integer path was skipped.

### What it measures

Measured on the project's x86 host (2× Xeon E5-2696 v4, 88 hardware threads, two NUMA nodes), on
`qwen3-0.6b-q4_k_m.gguf`, `pp32`/`tg32`, median of five. `llama-bench` on the same checkpoint and
thread counts is the other column, at its own default affinity. The three PocketLLM columns are the
same run, round-robin so machine drift lands on all of them: `before` is `kAttentionGrain = 32`,
`after` is the same engine with the grain at 1 and nothing else changed.

| threads | before pp32 | after pp32 | llama.cpp pp32 | before tg32 | after tg32 | llama.cpp tg32 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 29.91 | 29.88 | 69.52 | 16.63 | 16.11 | 18.36 |
| 8 | 143.21 | 153.35 | 396.49 | 50.70 | 63.33 | 74.02 |
| 22 | 293.03 | 293.12 | 719.25 | 48.45 | 77.31 | 81.32 |
| 44 | 397.08 | 422.49 | 982.98 | 48.86 | 72.31 | 82.60 |

Decode at 22 threads goes from 0.60 of llama.cpp to 0.95 on this one line, and the single-thread
column is unmoved at 0.97 — which is what says the change is the partition and not the arithmetic:
at one thread there is no partition to get wrong.

The PocketLLM rows here are taken with the pool spread over both sockets; pinned to one socket the
same build measures 29.34 / 15.36 at one thread, 147.72 / 47.71 at eight and 279.30 / 55.45 at
twenty-two, so the placement moves decode by under 2% — less than the run-to-run spread on this host,
which is why the columns are quoted from the same run.

**The same table measured before the integer activation path**, for the size of that one change:
at one thread 8.62 → 29.60 prefill and 6.07 → 16.62 decode (3.4× and 2.7×) — the single-thread
column was always the diagnostic, and it is the column that closed most of the gap.
Before the spin barrier, decode at 8 threads was 15.48 and at 22 threads 18.22 tok/s, while prefill
barely moved — which is what that diagnosis predicts, since prefill's kernel calls are long enough
that a futex round trip is noise and decode's are not.

**Prefill at 512 tokens, where the scratch cap was the whole story.** The same engine, the fixed cap
against the shape's-size buffer, interleaved in one run, median of three:

| threads | fixed cap pp512 | shape-sized pp512 |
|---:|---:|---:|
| 1 | 7.25 | 17.72 |
| 8 | 39.97 | 78.35 |
| 22 | 96.58 | 172.73 |
| 44 | 112.23 | 207.18 |

That row is between 2.0 and 2.5× and it is a *reversion check*, not an arithmetic change: the
integer path was always correct, it was simply not being reached. The single-thread column is the
tell — 7.25 against 17.72 is the exact-path ratio, unmoved by any thread count.

**The four-lane score dot, interleaved A/B against the pre-change engine**, both pinned to NUMA
node 0, median of the runs shown, `qwen3-0.6b-q4_k_m.gguf`, 22 threads, `llama-bench` on the same
checkpoint and affinity as the other column:

| test | before | after | llama.cpp | after/llama |
|---|---:|---:|---:|---:|
| pp32 | 296.03 | 305.79 | 801.40 | 0.38 |
| tg128 | 89.90 | **95.62** | 94.85 | **1.01** |
| pp512 | 189.20 | **278.47** | 793.06 | 0.35 |
| tg128 after 512-token prefill | 49.02 | **61.17** | 93.25 | 0.66 |
| tg512 (long decode) | 67.59 | **78.53** | 86.35 | 0.91 |

The 1.47× at `tg128`-after-`pp512` is the context effect the section above names: the score dot's
cost grows with the span while the GEMM budget is fixed, so the longer the prompt the larger the
share of decode this fixes. Short-context decode was already within a few percent of llama.cpp and
is now marginally ahead on this measurement — but that row is a near-tie, not a win, and the honest
reading of the table is that decode at short context is at parity and the *gap that remains* is
prefill.

**What the score dot moved, and what it left.** Prefill is 0.35 of llama.cpp's at `pp512` and 0.38 at
`pp32`, and that is the whole of what is left. The score dot is an attention fix, and attention is
decode's term; the prefill gap is unchanged by it, at 0.35 either side of the change (`189.20` →
`278.47` is a 1.47× on our own column and still under llama.cpp's `793.06`).

**Still does not beat llama.cpp, and saying so is the point of the table.** Decode is 88%
of llama.cpp's at one thread (16.11 vs 18.36), 86% at 8 and 95% at 22 — up from 30%, 39% and 61%.
The single-thread ratio is the one that moved the most, because the integer path is a per-core
efficiency fix and llama.cpp has nothing else on this host. Prefill at `pp512` is 172.73 against
llama.cpp's 743.76 at the same 22 threads, and 207.18 against 1254.39 at 44 — still under 0.2, and
the largest remaining term by far.
What is left is not a kernel: at 22 and 44 threads llama.cpp climbs (81.32 → 82.60 on decode) while
ours falls (77.31 → 72.31), and its prefill scales to four to six times ours, on the same cores and
the same checkpoint. That shape — falling above one socket, on a 456 MB weight set read every
token — is where the weights *live*: one allocation, first-touched by the loader on whichever node
it ran on (measured at ~66% node0), so most of a two-socket pool reads across the interconnect.
Placement is the next lever, and it is not in this change.

**Decode also degrades with context length where llama.cpp's holds**: at 22 threads and 1024 tokens
of history ours was 0.56 of llama.cpp's (43.44 vs 77.44), against 0.92 at 64 tokens. Attention's score
pass grows linearly in the span while its weight grows against a fixed GEMM budget, and the per-unit
dot was the scalar one — the four-lane dot above is what that sentence was pointing at, and it moves
the long-context row to 0.91 at 512 tokens of history (`tg512`: 78.53 against 86.35).

**The activation quantizer on the pool, interleaved A/B against the pre-change engine**,
`pp512`, `qwen3-0.6b-q4_k_m.gguf`, median of three, both engines and `llama-bench` round-robin in one
run so machine drift lands on every column. Up to 22 threads both are pinned to NUMA node 0; 44 and
88 run on the default affinity because that is the only way to reach those counts.

| threads | before | after | llama.cpp | after/before | after/llama |
|---:|---:|---:|---:|---:|---:|
| 1 | 24.98 | 24.68 | 65.74 | 0.99 | 0.38 |
| 8 | 124.50 | 131.55 | 385.36 | 1.06 | 0.34 |
| 22 | 273.28 | 315.75 | 793.29 | 1.16 | 0.40 |
| 44 | 401.37 | 536.15 | 1221.34 | 1.34 | 0.44 |
| 88 | 412.00 | 530.50 | 1018.79 | 1.29 | 0.52 |

The one-thread column is flat to 1%, and that is the tell: at one thread the pool hands a lone worker
the whole block range, so the parallel schedule *is* the serial loop and there is nothing for the
change to move. Everything above it is the fifth of a `gemm_quant` call that used to run on one core
while the rest of the pool spun at the barrier, and the gain grows with the thread count because that
is exactly what it was not doing.

Decode is unmoved within noise (`tg64` at 22 threads: 96.4 before, 98.1 after, ±4 run to run): a
decode GEMM is 4–16 blocks, below the point where a split pays for a barrier, and the pool runs it
serially on the caller either way.

**Prefill is still 0.40 of llama.cpp's at its best thread count, and that is the remaining gap.**
The quantizer was a fifth of the *GEMM's* serial work, not a fifth of the whole prefill: `gemm_quant`
is 70% of the op sum and the op sum is 85% of the wall, so a 1.16× on the GEMM carries to ~1.16 on
the total, which is the measured figure. The op-level profile says the same thing more precisely —
`gemm_quant` at `pp512`, 22 threads, 197 calls: **6.98 ms per call before, 5.14 ms after**, on a
prefill that is 1.92 s before and 1.54 s after — and it also says the change did not touch a byte of
attention (518.4 ms → 509.1 ms, drift) or the small ops.

What is left is the row dot itself: our `dot_row_q8k` is a one-output-per-block-walk kernel at
31.9 GFLOP/s per core, against llama.cpp's repacked 8×8 GEMM at 42.2 on the same shape, and prefill's
weight traffic is the part of the problem that only a several-rows-at-a-time kernel can turn into
reuse.

## Building

Out of tree, so the Python package stays free of build artifacts:

```bash
cmake -B build -S src
cmake --build build -j8
```

CUDA is enabled by *availability* rather than by a flag: if a toolkit is found, the backend is
compiled and `pocketllm_open(..., "cuda")` works; if not, the name is refused with a message naming
what the build does provide. A flag would make the presence of a kernel a decision somebody has to
remember on every machine, which is how a release ships without the backend it was tested with.
`POCKETLLM_CUDA=OFF` is the escape hatch, and `POCKETLLM_CUDA_ARCHITECTURES` (default `75`) is the
capability pinned for the project's RTX 2080 Ti development cards.

`CUDAToolkit_ROOT` is deliberately not pinned. On the project's x86 host `CUDA_HOME` names 12.4 while
`nvcc` on `PATH` is 13.0; `/usr/local/cuda` resolves through `update-alternatives` to `cuda-13.0`,
which is what CMake's search finds, so the build agrees with the toolchain without a hard-coded path
that would be wrong on the next machine.

CPU warnings are errors, scoped to `CXX` with a generator expression so nvcc — which spells `-Werror`
differently — is unaffected. `-march=native` is set for the same reason `-O3` is: this is a
build-host choice, and a shipped or mobile build would pin a target architecture instead of taking
the one it is compiled on.

### Measuring it: `pocketllm-bench`

`pocketllm-run` prints generated text and nothing else, so there was no number to review a
performance change against. `build/pocketllm-bench` mirrors `llama-bench`'s shape and its test names —
`pp<N>` for a prefill over `N` tokens with the cache dropped first, `tg<N>` for `N` single-token
decodes against a warm cache — so the two tables can be read side by side. It times with
`steady_clock`, warms up, and reports the median of `--reps`, and it emits each row both as a
Markdown table and as a `bench pp <N> <rate>` line a test can parse. The prompt is synthetic token
ids rather than text, because tokenization is not what is being measured.

## Where the host meets it

`python/pocketllm/native.py` loads the library, checks `pocketllm_abi_version` against the major the
Python side is written against, and wraps each entry point so a negative return becomes an exception
rather than a sentinel. `Engine.open(path, backend)` is one session; `Engine.forward(tokens)` is
`pocketllm_forward`.

The library is found by `$POCKETLLM_CORE_LIB`, then `src/build/libpocketllm.so`, then the repository
root's `build/`. Everything in `tests/native/` skips when it is not there, because a checkout without
a compiler must still be able to run the suite — and a skip is not a pass.