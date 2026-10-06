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
- **The K/V cache width, and llama.cpp's own inconsistency across it.** llama.cpp's f16 cache does
  not answer the same question on its two paths: on this prompt's second generated token — a real
  near-tie, tokens `11` and `13` separated by 0.06–0.5% of the logit spread — a batched prefill says
  `11` where a one-token decode says `13`, so there is no single f16 sequence to match. Its **f32**
  cache says `11` on both. The oracle therefore pins f32
  (`tests/native/llama_oracle.py::KV_TYPE`) even though the CPU engine keeps f16, and that is not a
  mismatch: llama.cpp's own f16 differs from its own f32 by **1.12** in the logits, more than the
  engine's f16 differs from llama.cpp's f32 (**0.89**). Over sixteen greedy tokens the engine's
  decode path reproduces llama.cpp's f32 sequence exactly. The near-tie is present on every cache
  width on both engines; the width only picks a side.

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

The thread count is `$POCKETLLM_CPU_THREADS`, falling back to the **physical core count**, clamped to
`[1, 256]`. `=1` reproduces a single-threaded number, and on a shared host it is the polite setting.
`build/pocketllm-bench` prints the count it resolved, so a table row is never ambiguous about how
many cores produced it.

**The default is cores, not hardware threads, and the difference is 1.3× on decode.** The first
version of this used `hardware_concurrency()`, which on this host is 88 — 44 physical cores × 2
hyperthreads — and it is the wrong number for a pool whose workers *spin*: two hyperthread siblings
that are both spinning are fighting over one core's issue ports and one core's share of the memory
pipeline, and the arithmetic they are supposed to be doing pays for it. Reading the topology from
`sysfs` (`physical_core_count`, one distinct `thread_siblings_list` per core) gives 44, and at that
count both halves of the benchmark are faster than at 88 — prefill because nothing is contending,
decode because decode is memory-bound and the extra threads only add coherence traffic:

| threads | `pp512` | `tg64` |
|---:|---:|---:|
| 22 (one socket's cores) | 521 | 74.2 |
| **44 (both sockets' cores — the default)** | **939** | **72.4** |
| 88 (every hardware thread — the old default) | 862 | 54.5 |

The `taskset` trap below is the same effect seen from the other side: a run pinned to 22 cores that
still builds 88 workers collapses the same way. `physical_core_count` reads the machine's topology
and so does not see an affinity mask either; the advice to set `$POCKETLLM_CPU_THREADS` by hand when
pinning stands.

**Ask for more threads than you have pinned cores and the pool collapses, which is a benchmark trap
rather than a kernel property.** The pool is sized from the machine's topology — the machine, not
the affinity mask — so `taskset -c 0-21` with the default thread count builds 44 workers on 22 cores.
Every worker spins on the generation counter while idle, so 22 threads that cannot run are fighting
the 22 that can, and the run does not degrade gracefully: **`pp512` reads 26 t/s against 500 for the
same binary at `-t 22`**, a 20× loss with no error and no wrong answer. Two rules follow for anyone
measuring this engine:

- **Set `$POCKETLLM_CPU_THREADS` to match the mask** (`taskset -c 0-21` wants `=22`), or do not pin
  at all and let the default take the whole machine.
- **The thread count and the core count are two different numbers, and the interesting comparison is
  at matched core counts.** The table in "What it measures" is laid out that way for this reason.

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

### Rows per weight walk: four, and now eight

With the quantizer parallelized, the GEMM is a walk over weight blocks, and the pairing is still
one activation row to one weight row: `dot_row_q8k` reads each weight block once per output row, so
a 512-token prefill reads the same 590 KB weight panel 512 times. `dot_Rrows_q8k` walks the weight
block once and applies its decoded nibbles to `R` activation rows at a time, keeping `R` independent
integer accumulators and `R` float accumulators live across the same walk.

**Measured at 1.58× to 1.80× on the kernel, and bit-exact.** The `/tmp` probe (`rows4.cpp`: copies
of both kernels, a synthetic panel, one core, both measured back to back in one process) puts
`m=512 n=1024 k=1024` at 1.80× on a quiet machine and 1.58× on a loaded one, with the one-row column
moving by 2× between the two — which is why the ratio is the number quoted here and not a GFLOP/s
figure; the run-to-run spread on this shared host is larger than the effect of most changes. The
end-to-end A/B below is the firmer measurement. The `worst` column of the probe is **0**, which is
the important half: four rows through the batched kernel are the same bytes as four `dot_row_q8k`
calls.

That is a requirement rather than a happy property, and it drove the implementation. Every row's
integer accumulation is over the same values in the same order, its float multiply is per block
(`y.d * wd`, the activation's scale folded in exactly where the one-row kernel folds it), and its
horizontal reduce is the same expression. The first draft of the q6_k branch multiplied by the
weight's `d` alone and left out the activation's `y.d`; the test written with the kernel reported it
as a 537-thousand error on its first run. `tests/native/test_cpu_parallel.py` pins the equivalence
directly: a six-row `gemm_quant` against six one-row calls, rows 0–3 through the batched walk and
rows 4–5 through the one-row kernel, compared byte for byte. The conformance tolerance would have
passed a reassociated sum; this would not.

**The row count is now eight, and getting there took two wrong measurements before the right one.**
The kernel is a template on `R` after the eight-row form, and `$POCKETLLM_CPU_GEMM_RPW=4` selects the
old count, because the two are bit-identical (a new test holds them to the same bytes at `m=16` and
`m=18`, so both the full tiles and the ragged tail are covered) and differ only in speed.

The first probe of the pair, on one core, put R=8 only 5–8% ahead of R=4 — a small margin for
something that halves the panel traffic. The second put both kernels behind the engine's own
`parallel_for` and reported R=8 *behind* R=4 at 22 threads, which looked like a refutation. **It was
not: that probe's traversal ran 6× slower than the engine's own GEMM**, so it was measuring its own
memory behaviour rather than the kernel's. The number that decides it is the engine's, on a quiet
host, at the thread count that matters — `pp512`, `q4_k_m`, all 88 hardware threads, interleaved,
median of six:

| row count | t/s |
|---:|---:|
| 4 (the previous default) | 905 |
| 8 | **978** |

R=8 wins by 8% with the machine full, which is the *opposite* of what register pressure predicts (the
tile wants roughly 29 of the 32 AVX2 registers at R=8 against 17 at R=4) and is why this has to be
measured rather than reasoned about. **Check the host load first**: the same comparison run under a
loaded host read 26 t/s at 44 threads against 500 at 22, all of it contention.

The dispatch lives in `gemm_quant` and is shape-driven: `m / R` groups take the batched path and the
`m % R` tail takes `dot_row_q8k`. At decode `m` is 1, so every call takes the one-row path and the
batched kernel costs a decode nothing — which is why the change is a prefill change and the decode
column of the A/B is flat.

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

### Four query rows per key walk

The four-lane dot made the score pass fast per element; it did not change how many times the key is
read. The score matrix is `q_len × span` over the same `span` key vectors, so a row-at-a-time walk
loads each key row `q_len` times, and at prefill the pairs are a triangular half-matrix. Attention
was **18.3 ms per call and 44% of a 1161 ms `pp512`** at 22 threads once the four-row GEMM landed —
the largest single term in the prefill gap, and the one whose per-element cost had already been
optimized as far as lanes could take it.

`dot_tile` walks `kAttentionRows` query rows of one head against one key vector, so the key is
loaded once for all of them. The partition changed with it: a task is now `(kAttentionRows tokens,
head)` rather than `(token, head)`, and its score scratch is that many rows rather than one. The
constant shipped as 4 when this tiling landed and is **8** today — see "Eight query rows per key
walk" below for the measurement that moved it.

**Bit-exactness here is a lane structure, not a rounding budget.** A natural four-query kernel gives
each row its own `__m256` — eight lanes per row, an eight-lane horizontal reduce — and that is a
different last bit from `dot4`'s four-lane chain, which is exactly the difference the FMA draft
proved fatal (32/32 tokens → 1/32). So the two `__m256` accumulators here hold *two queries each*:
rows 0–1 in the low and high halves of the first, rows 2–3 in the second, loaded at the offsets
`dot4` would load them, combined with the same `_mm_mul_ps`→`_mm_add_ps` and reduced with `dot4`'s
own `((l0 + l1) + (l2 + l3)) + tail`. Every lane is `dot4`'s lane; only which queries share a
register has changed. The probe's per-lane check (`tilecheck.cpp`: four known query rows against one
key, `dot_tile` versus four `dot4` calls) reports 4/4 identical, and the shape sweep — **15 chunk
lengths, 1 through 513** (`1..9, 17, 64, 100, 511, 512, 513`), each at two spans and three thread
counts, base engine against tiled engine, both through the public `attention` op — reports **0
mismatching bytes** across 90 base-against-tiled calls.

The causal tail is not special-cased. A block of four queries shares only the keys up to
`q_offset + t0`; after that row `j` still needs `j` more keys, so the tail runs `dot4` one row at a
time rather than adding a second, ragged tiled kernel — the shipped one-row path, unchanged, and
exactly one tiled code path to be right. A `q_len` that is not a multiple of four, and a decode step
(`q_len = 1`, one block, zero shared keys) both fall through to it, which is why decode is unmoved by
construction rather than by measurement.

**Where the rows-per-walk number comes from.** A dedicated probe (`/tmp/prof/attnrow.cpp`; one
kernel templated on the row count so R = 1, 2, 4 and 8 differ in a bound and not in an
implementation) sweeps both the tiling and the chunk length, all three passes running in every
variant, interleaved best-of-five so drift lands on every column:

| chunk | R=1 | R=2 | R=4 | R=8 |
|---:|---:|---:|---:|---:|
| 1 | 6.5 us | 0.92× | 0.99× | 0.94× |
| 8 | 11.8 us | 1.41× | 1.19× | 1.26× |
| 32 | 79.3 us | 1.19× | 1.39× | 1.29× |
| 128 | 1117 us | 1.21× | 1.34× | 1.32× |
| 512 | 16945 us | 1.20× | **1.42×** | 1.42× |
| 2048 | 330740 us | 1.16× | 1.44× | **1.60×** |

R = 4 is the knee for the shape the benchmarks use, and two effects rather than one produce it. More
rows reuse each key load — the region a block shares is walked once instead of once per row — and more
rows also lengthen the causal *tail*, the `R - 1` keys past the shared region that run `dot4` one row
at a time at R = 1 efficiency. The reuse term grows with `q_len` and the tail term does not, which is
why the ordering flips: at `q_len = 512` the two are even (both R = 4 and R = 8 measure 1.42×), at 128
and below R = 4 leads (1.34× to 1.32×), and at 2048 the reuse wins out and R = 8 leads 1.60× to 1.44×.
R = 2 leaves a sixth of the win on the table at every chunk. Decode is flat across all of them because
a chunk of one is a single block, so every R takes the same serial path.

That the tail is the trade-off is a claim about where the time goes, not a measured decomposition of
it — what the table measures is the two ends, and the crossover between them. Either choice is
defensible at 512; R = 4 is the one that is also the best at the short chunks, which are the ones a
decode-and-short-prompt session actually pays.

The first version of this probe read as a **1.35× win on an output wrong by 3.1**: it advanced one
`s` across the rows of a block and so skipped the keys `t0+1 .. t0+j-1` for every row past the first.
The timings were real and the arithmetic was not, which is why the probe's bit-exactness line is
printed before the timings and the tiled kernel's own test is a byte comparison rather than a
tolerance. The probe ran every chunk at 22 threads pinned to node 0, best of five, which is where the
`q_len = 1` row's 6.5 us comes from: one block per head is 16 units against a pool of 22, so a decode
call barely occupies half the machine whatever the tiling.

**End to end, interleaved A/B against the pre-change engine**, `pp512`, both pinned to NUMA node 0,
`llama-bench` round-robin in the same run, median of three:

| threads | before | after | llama.cpp | after/before | after/llama |
|---:|---:|---:|---:|---:|---:|
| 1 | 35.34 | 38.67 | 65.32 | 1.09 | 0.59 |
| 8 | 173.76 | 198.96 | 382.63 | 1.15 | 0.52 |
| 22 | 417.06 | 480.73 | 792.38 | 1.15 | 0.61 |
| 44 | 705.75 | 798.46 | 1247.77 | 1.13 | 0.64 |
| 88 | 618.86 | 772.21 | 965.90 | 1.25 | 0.80 |

Decode is the control and it holds within this host's spread (`tg64` at 22 threads: 95.98 before,
98.30 after, 97.81 for llama.cpp, ±4 run to run), which is what the construction predicts: `q_len`
is 1, so there is one block per head, no shared keys, and the tiled path is never entered.

**Prefill is 0.61 of llama.cpp's at 22 threads** (480.73 against 792.38), up from 0.52, and 0.59 at
one — the single-thread column moved too, which says this is arithmetic the kernel does less of
rather than a schedule.

The arithmetic behind the 1.15× is worth stating because it was checked before it was measured. The
base profile has `gemm_quant` at 625 ms and attention at 514 ms of a 1161 ms `pp512`, and the probe
says the tiling is 1.42× on the *whole* attention call; 1161 − 514 + 514/1.42 = **1009 ms, or
1.151×** — which is the 1.15× the A/B measured on its own (417.06 → 480.73). The ms columns are a
projection on the profile's run and not a re-profile: the post-change engine was timed end to end,
not re-instrumented. What is left at 22 threads is `gemm_quant` at 625 ms and attention at ~362
projected, so the score pass is no longer the larger half and the four-row GEMM — 1.3× behind
llama.cpp's repacked 8×8 form — is.

### Four query rows per walk over a V row

The score pass got its four-row tiling and the **weighted sum was left alone**, which made it the
second term of the call: at `q_len = 512`, 22 threads, the three passes split score ~6.6 ms, softmax
~0.8 ms, weighted sum ~7.2 ms — the last one streaming the whole `V` slab for every query row and
running `dst[x] += weight * vvec[x]`, two memory operations per one FMA whose multiplicand is a
broadcast register. After the score tiling it was the larger of the two halves of attention.

The tiling is the same idea at the other end of the call: hold `kAttentionRows` accumulators and
walk a V row once for all of them. The split is what the correctness argument turns on, and the
first draft of the probe got it backwards. Row `j` owns keys `[0, span0 + j)`, where
`span0 = q_offset + t0 - first_key + 1` is the part every row of the tile shares, so the walk is two
regions: the **shared prefix** `[0, span0)`, run with the V row outer and the rows inner — the only
shape in which the load is shared at all — and the **triangular remainder** `[span0, span0 + j)`,
run per row because no other row sees those keys. A version that looped `j` outside `i` gave every
row only its own private slice and produced fluent output **0.8 of its own scale off at a 1.58×
"win"** — which is why this is checked by comparison and not by its speed.

**It came out bit-exact**, which it did not have to. Row `r` still adds its own terms in key order,
one `+=` per key, with the same weight; the tiling changes *which row* is being accumulated between
two loads of the same V vector, never the order within a row's own sum. The probe
(`/tmp/prof/attnwsum.cpp`: the shipped `attention` against a copy whose only difference is the last
pass, `Rv = 1` verified byte-identical to the shipped kernel first) reports `relerr 0.00e+00` at
every chunk from 1 to 512, and `test_the_tiled_attention_weighted_sum_is_the_row_at_a_time_one`
holds the end-to-end call to `array_equal` rather than to a tolerance.

The probe's whole-call timings, one core, `q_len = d`-length caches, best of three:

| chunk | shipped | tiled (`Rv = 4`) | whole-call speedup |
|---:|---:|---:|---:|
| 1 | 1.6 us | 1.7 us | 0.93× |
| 8 | 26.6 us | 28.7 us | 0.93× |
| 32 | 336.0 us | 340.8 us | 0.99× |
| 512 | 102342 us | 88575 us | **1.16×** |

The sub-512 rows are below 1.0× because a decode or a short prompt has one block per head — 16 units
against 22 threads — and the tile's extra bookkeeping buys no V reuse there. At the prefill shape
the call is 1.16×, and since attention was ~368 ms of a ~1080 ms `pp512`, that projects to about
0.34 × (1 − 1/1.16) = **4.7% off the total** — which is the 5% the A/B below measures on its own.

**End to end, interleaved A/B, `73cbf8f` against the change**, both pinned to NUMA node 0,
three runs each, `--reps 3`:

| threads | before | after | after/before |
|---:|---:|---:|---:|
| 1 | 38.87 | 40.67 | 1.05 |
| 8 | 200.48 | 212.03 | 1.06 |
| 22 | 479.11 | 501.54 | 1.05 |

`tg64` at 22 threads is the control and it is flat to slightly up (96.7 → 99.6, ±4 run to run),
which is what the construction predicts: a decode step is `q_len = 1`, one block per head, and the
tile is a single row.

Against llama.cpp in the same session the same way (`-t 22` pinned to node 0, `-fa 0` so neither
engine is using flash attention, median of two): **681.0 t/s against our 501.5, or 0.74×**, up from
0.70× when the score tiling landed. With `-fa 1` llama.cpp measures 796.0. Prefill is still the gap
and the remaining term is `gemm_quant`, whose `n = 1024, k = 1024` shape is 305 ms of the ~1070 ms
call on 22 threads — the four-row GEMM behind llama.cpp's eight-row repacked form.

### Eight query rows per key walk

Both tiled passes are bandwidth-bound, and the quantity they are bound by is how many times the K
and V slabs are streamed: `q_len / R × n_heads / kAttentionHeadBatch` slab-lengths, where `R` is
`kAttentionRows`. Everything above is the story of `R = 4`; **`R = 8` halves the traffic again**, and
the measurement below is what moved the constant.

The split of the whole call at `pp512`, 44 threads, one host, by truncating the kernel after each
pass (`$ATTN_PASS` on a scratch copy of the tree — the shipped binary has no such switch):

| | score | softmax | weighted sum | total |
|---|---:|---:|---:|---:|
| `R = 4` | 66 ms | 16 ms | 79 ms | 161 ms |
| `R = 8` | 53 ms | 16 ms | 58 ms | 127 ms |
| `R = 16` | 85 ms | 16 ms | 46 ms | 147 ms |

`R = 8` moves both streamed passes the right way — score 1.25×, weighted sum 1.36× — for **1.27× on
the call**. `R = 16` is where the score pass turns around and gives the win back, and the cause is
registers rather than traffic: `dot_tile_r` holds one live `__m256` per *pair* of query rows, so 16
rows is 8 accumulators plus the key vector and the query loads, which spills. The weighted sum keeps
improving at 16 rows because it trades against a different pressure — a tile row's accumulator is a
`kAttentionSumMaxDim`-wide slice of stack, not a register — which is why the constant is a compromise
and 8 is the value where neither pass has regressed.

**Bit-exactness is not weakened by the row count, and the reason is the pairing.** `dot_tile_r<R>`
gives each *pair* of rows its own accumulator, loads each pair's two four-lane halves exactly where
`dot4` loads them, and combines them with the same `_mm_mul_ps` → `_mm_add_ps`; a pair's arithmetic
does not know how many other pairs exist. So a row's result is the same at `R = 4`, `R = 8` or any
other even `R`, and `test_the_tiled_attention_score_is_the_one_row_dot` — which runs the shipped
kernel against `$POCKETLLM_CPU_SCALAR_DOT` and calls `np.array_equal` — passes unchanged.

**End to end, interleaved, `--reps 5`, median of four rounds**, both engines at their default 44
threads, `-fa 0` on llama.cpp so neither is using flash attention:

| | ours | llama.cpp `-fa 0` | ratio |
|---|---:|---:|---:|
| `pp512` | 966 | 839 | **1.15×** |
| `tg64` | 56.6 | 62.9 | 0.90× |

**Prefill now leads llama.cpp like-for-like.** That is a reversal of the position recorded above —
0.74× at 22 threads in an earlier session — and it is the tile plus the physical-core default plus an
unloaded host, not one of them alone. Against llama.cpp's *shipped* default (`-fa auto`, which is
flash attention on) it is still behind on both: `pp512` 966 against 1238, `tg64` 56.6 against 72.6.

**Decode is the number this change does not move, and that is a construction argument, not a
measurement.** A decode step is `q_len = 1`: one block per head, no shared key region, so the tiled
path is never entered and the single row takes the shipped one-row `dot4`. What decode is short of is
not tile width but *tasks* — 16 heads over 44 threads — and a decode attention call gets **slower**
past about 8 threads (`/tmp/prof/dattn.cpp`, span 512: 99.9 us at 8 threads, 105.0 at 22, 111.4 at
44, 170.5 at 88). llama.cpp shows the same shape on this host (`tg64` 93.4 t/s at 22 threads against
73.0 at 44), which is the signature of two sockets and a resident weight buffer rather than of either
kernel. Widening a decode's tile is therefore the next piece of work, and it needs `q_len > 1` to do
anything at all.

### Two query heads per walk over a key row

The two passes at the two ends of the attention call had each been tiled over *query rows* — four
queries scored against one key row, four outputs accumulated from one V row. What neither addressed is
the axis that decode actually spends its time on: **grouped attention reads each KV head's key row once
per query head, and there are two query heads per KV head.** At 464 rows of context a decode step runs
16 units × ~464 rows × 28 layers of 128-wide dots, and every one of those key rows is loaded *twice*
for the two heads that share it.

The K cache makes that second load expensive rather than free. A KV head's row is `n_head_kv * d` =
1024 floats apart, of which `d` = 128 — 512 bytes of 4096 — belong to the head being scored, so the
walk streams a quarter of its bytes usefully and the row that was just read is four kilobytes of
distance away, not in the same cache line. Scoring two heads of one group together loads the key once
for both.

**Batching heads is exact where tiling rows is not, and the difference is whether the two things share
a producer.** Two heads of one KV group score *different* query vectors against the *same* key vector:
the key load is genuinely shared, and no two lanes share an accumulator, so the batch splits without
touching any sum. Two query *rows* of one head are the opposite case — their spans overlap only
partially, and hoisting the key load or the weight lookup past the row loop changes which products are
summed in what order, which is the last-bit change the softmax turns into a different token. So the
row tiling stays confined to the region a whole tile shares, which is what `dot_tile` is, and the head
batch needs no such region.

The kernel is `dot_pair`: one `__m256` holds query `h0`'s four lanes in the low half and `h1`'s in the
high half, advanced by four at exactly the offsets `dot4` loads, combined with `_mm_mul_ps` into
`_mm_add_ps` and never a fused multiply-add, each half reduced `((l0 + l1) + (l2 + l3)) + tail`. Every
lane is a `dot4` lane, so the two results are bit-identical to two `dot4` calls — the same property the
four-lane dot was built to have, and for the same reason.

A single-core microbenchmark (`/tmp/prof/score2.cpp`, 28 layers × 16 heads, one core, best of three,
span = 1025) isolates the effect:

| body | us | vs shipped |
|---|---:|---:|
| shipped — head outer, strided | 17295 | 1.00× |
| kv head outer, K row loaded once for both heads | **11617** | **1.49×** |
| the same heads, two plain `dot4` calls | 15144 | 1.14× |
| contiguous K, no stride (the ceiling) | 11291 | 1.53× |

Two things in that table are worth keeping. Loading the key once for both heads gets 1.49× of a 1.53×
ceiling, so the strided walk *was* the cost and the paired kernel takes nearly all of it. And **the two
plain `dot4` calls reach only 1.14×**: the point is not that the key row is in L1 for the second read —
it is — but that the second read still issues the loads. The sharing has to be in the register, not in
the cache.

**End to end, where the effect is the decode term it targets.** The instrument is `pocketllm-bench` on
`qwen3-0.6b-q4_k_m.gguf`, both binaries built from the same tree and run interleaved, 44 threads. The
per-op probe over one decode token with 512 rows of context puts the whole attention op at **51.1 ms
against 33.4 ms — 1.53×**, stable to 2% over six interleaved repetitions even with the machine under
load, because the figure is a sum of measured op times rather than a wall clock. Carried to the bench's
own column:

| test | before | after | after/before |
|---|---:|---:|---:|
| `tg64` after a 512-token prefill, 44 threads | 52.0 | **47.2** | **1.10** |
| `pp512`, 44 threads | 501676 us | 493876 us | 1.02 |

The 1.53× on the attention call becomes 1.10× on the step because attention is a little over half of a
decode token at that context — the GEMMs share the token with it. Prefill is flat, which is what the
construction predicts and not a disappointment: at `pp512` each head already scores `kAttentionRows`
query rows per key walk, so the batching's *second* read of the K row is amortized over all of them, and
attention is a much smaller share of a prefill step than of a decode step.

**The measurement this section does not have, and why.** An earlier revision of this page quoted a
`tg64`/`pp512` table taken in one window. It was removed rather than kept: this host is shared, it spent
most of the session at load 30–175 from other users' jobs, and an interleaved A/B only means something
when the two arms see comparable machines. The figures above are the ones that survived six interleaved
repetitions each with the load printed beside them; where a number was not reproducible across those
repetitions it is not on this page.

**The bug the test found before the benchmark did.** The unit count was written
`groups = n_heads / kAttentionHeadBatch`, which is integer division: at Qwen3's 16 heads it is 8 and
nothing is wrong. At an odd head count it truncates, the final unit is never scheduled, and **that
head's output is never written at all** — the caller reads whatever the buffer held. `n_heads` is only
guaranteed to be even for this checkpoint, so the guard is `(n_heads + 1) / 2` and the test
parametrizes head shapes that are not multiples of the batch, including 3 heads over 1 KV head. The
unwritten values were in the `3e-41` denormal range and the test's first form — an even head count —
would have passed.

The same rounding appears in the CPU backend's `attention_scratch`, which sizes the per-task score
region from the partition the kernel takes and now derives both the unit count and the rows-per-unit
from `kAttentionHeadBatch` rather than assuming one head per task.

### The K/V cache is f16, and the widening is where the win is

The cache is the one operand in the graph whose *size* grows with the sequence, and it is the only
thing the attention pass streams. At 464 rows of context a decode step reads 28 layers × 8 KV heads ×
464 rows × 128 values — **4.2 MB per token** in f32 against **2.1 MB** in f16 — and the pass is bound
by that read, not by its arithmetic: the score dot is 82% of the call and the walk is memory-bound
(the probe in the section above measures the same key row being re-read at a four-kilobyte stride).

There is a second reason, and it is the one that settled it: **`-ctk f16 -ctv f16` is llama.cpp's
default.** A f32 cache is not "more correct" than the oracle, it is a different basis — twice the
bytes to read for precision the comparison does not credit. Reading the same number of bytes is what
makes the rest of the comparison a comparison of the walk.

**The f16 cache is read by widening a row to f32 and running the f32 kernels, not by a second family
of f16 dot kernels.** Every score kernel here — `dot`, `dot4`, `dot_pair`, `dot_tile` — is the product
of a measured argument about lane structure, and the score dot is the one place a last-bit change
flips tokens (the FMA experiment above moved a 32/32 token match to 1/32). A parallel `_f16` family
would double that surface with a second kernel under a weaker argument. Widening leaves exactly one
new claim — "this expansion equals each element's width", which is `vcvtph2ps` — and every score
kernel then reads a cache row it has already been verified against.

It is also where the bytes go. A head row is 128 halves = 256 bytes, and the widened row is 512
bytes: L1-resident five times over. `dot_pair` reads a key row twice (once per query head) and
`dot_tile_r` reads it once for `kAttentionRows` rows; both were already reading a 512-byte row in the f32 case, so
this is the access pattern the kernels were written against. A single-core probe that widens a row and
then runs the real score loop measures **393 us against 410 us at span 512** — the conversion costs 4%
and saves half the bytes, and the trade only improves as the cache grows.

**End to end**, `pocketllm-bench` on `qwen3-0.6b-q4_k_m.gguf`, both binaries from the same tree, run
interleaved at 8 threads (`tg64` follows a full `pp1024`, so the decode column is measured against a
1024-row cache):

| test | f32 (before) | f16 (after) | after/before |
|---|---:|---:|---:|
| `tg64` after a 1024-token prefill | 32.1 | **38.7** | **1.19** |
| `pp1024` | 175.4 | **189.2** | **1.08** |
| `tg16` at a 256-row context | 60.2 | 56.1 | 0.93 |

The shape of that table is the design. The win is **1.19× on decode and 1.08× on prefill at a
1024-row context**, because that is where the cache is large enough for halving it to matter. At a
256-row context the cache is a quarter the size and the conversion is not yet paid back — the third
row is a loss, and it is on the page rather than in a footnote because a lever that only pays at long
context is exactly what this one is.

**The bug the change shipped and the profile caught.** The first version put the widened row in a
buffer on the *attention function's* frame. Every pool thread writes that buffer, so eight attention
units raced on one 512-byte line — a data race whose output is not a crash but a plausible, finite,
wrong score row, and whose measured effect was a 2.2× *slowdown* from the false sharing, not the 1.19×
speedup. Moving the buffer inside the per-task lambda (a task is a thread here) fixed both, and the
number above is what it measured once it did. Nothing about the timing said "race"; it looked like a
slow conversion until the buffer was traced.

**The f16 cache is what the engine ships.** `Backend::preferred_kv_dtype` returns `kF16` by default;
the CPU backend takes it and the CUDA one overrides it to `kF32`, because a card at this model size is
nowhere near bandwidth-bound and the trade buys nothing there. The graph asks rather than assuming —
nothing in `qwen3.cpp` names a width — which is what keeps a backend that cannot consume an f16 cache
from silently storing the wrong layout. `opcheck` defaults to f32 (its job is checking an op against
an f32 reference) and takes `kv_dtype 1` to exercise the f16 path; the shipped path is covered end to
end by the token tests instead.

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

**The four-row weight walk, interleaved A/B against the pre-change engine**, same harness and
affinity as the table above, `pp512`, median of three:

| threads | before | after | llama.cpp | after/before | after/llama |
|---:|---:|---:|---:|---:|---:|
| 1 | 24.62 | 34.46 | 65.06 | 1.40 | 0.53 |
| 8 | 130.80 | 173.00 | 381.71 | 1.32 | 0.45 |
| 22 | 318.50 | 415.08 | 791.98 | 1.30 | 0.52 |
| 44 | 531.59 | 718.78 | 1259.03 | 1.35 | 0.57 |
| 88 | 499.97 | 693.08 | 966.43 | 1.39 | 0.72 |

The op-level profile at `pp512` says where it comes from: `gemm_quant` over 197 calls is **79.9 ms
per call to 49.3 ms** at one thread, **13.8 to 8.1** at eight, **5.65 to 3.17** at 22 — a 1.62×,
1.71× and 1.78× on the GEMM, which is the kernel's own 1.58× plus the better cache behaviour the
panel reuse buys. Attention is unmoved (5.20 s → 5.14 s at 22 threads, drift) and so are the small
ops, which is what a change confined to the row dot should look like. Prefill wall time at 22
threads goes 1.65 s → 1.16 s.

The one-thread column is the one that says this is a kernel change and not a scheduling one. The
parallel quantizer's A/B had a flat one-thread column because a lone worker already ran the whole
serial loop; here the single thread is doing *less work* — one weight decode for four rows instead
of four — and it moves 1.40×, which is the kernel's arithmetic and nothing else.

Decode is unmoved within noise at every thread count (`tg64` at 22 threads on the quiet host: 101.0
before, 98.7 after, 96.2 for llama.cpp, against a ±4 run-to-run spread), and that is by construction:
`m` is 1, so the dispatcher routes every decode GEMM to the one-row kernel. The batched path exists
for prefill and costs decode a branch.

**And then the eight-row walk, which is what closed it.** The argument in the paragraph above — that
the four-row walk still spends its time in the weight decode and an eight-row form amortizes it over
eight rows rather than four — turned out to be right, but only the engine could show it: the
single-core probe understated the win and the `parallel_for` probe actively inverted it. `pp512`,
`q4_k_m`, interleaved, median of six, at matched core counts and with `-fa 0` so neither engine is
using flash attention:

| threads | PocketLLM | llama.cpp | ratio |
|---:|---:|---:|---:|
| 22 (node 0) | 526 | 677 | 0.78 |
| 44 | 933 | 815 | **1.14** |
| 88 (both engines' default) | **976** | 823 | **1.19** |

**That table is at 88 threads and is history, not the current default.** Re-measured at each
engine's *current* default — both now pick the physical core count, and llama.cpp's `llama-bench`
prints `44` — on a loaded host, best-of-four interleaved, `pp512`/`tg64`:

| | PocketLLM | llama.cpp | ratio |
|---|---:|---:|---:|
| `pp512`, `-fa 0` on both | 680 | 883 | 0.77 |
| `tg64`, `-fa 0` on both | **64.8** | 50.1 | **1.29** |
| `pp512`, pure defaults | 684 | **1295** | 0.53 |
| `tg64`, pure defaults | 66.3 | **80.7** | 0.82 |

So the current honest position is: **decode ahead on the like-for-like basis, prefill behind on
it, and both behind llama.cpp's shipped default.** The last two rows are the ones a reader running
`llama-bench` with no flags sees, and they are the number that matters to the goal.

**Where the prefill gap is, and where it is not.** The per-op profile of our own `pp512` at 44
threads (817 ms total): `gemm_quant` 521 ms (64%), `attention` 220 ms (27%), `kv_append` 51 ms (6%),
the rest under 3%. Subtract attention and our GEMM half is 597 ms against llama.cpp's whole 580 ms —
so **the GEMM already matches llama.cpp; the entire prefill gap is attention.** llama.cpp's default
is flash attention (`flash_attn_type = AUTO`, which resolves to *on* on the CPU path), and a flash
kernel fuses the score, softmax and weighted-sum passes with an online rescaling instead of writing
the score row out, reading it back for the max, reading it again for the exponentials and a third
time for the weighted sum. Per the `-fa auto` vs `-fa 0` columns above that fusion is worth 1.47× on
prefill and 1.61× on decode to llama.cpp. This engine does not implement it, and that is the
decisive remaining gap.

One thing the earlier win **does not** cover: it is one host, one checkpoint and one prompt length;
`pp32` and longer prompts were not re-measured against this build.

**The measurement that would have said "no" is worth keeping.** Before the engine A/B, a probe put
both row counts behind the engine's own `parallel_for` and reported R=8 *slower* than R=4 at 22
threads — which would have ended the idea. The probe's traversal ran 6× slower than the engine's own
GEMM, so it was measuring its own memory behaviour. A kernel ratio is only transferable if the probe
and the engine are doing the same thing at the same speed; when they are not, the engine is the
instrument. The same lesson is why the thread-count trap above is called out separately: two of the
three "measurements" that framed this stage were artefacts of the harness.

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