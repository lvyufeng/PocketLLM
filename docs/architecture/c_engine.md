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

That file also records where llama.cpp is *not* an oracle. Its `q4_K` dot product does not decode the
weight and multiply it by a float activation: it quantizes the activation to int8 (`block_q8_K`) and
does an integer product. The extra step is an error of its own, and on a one-token prompt it is a
large one — measured, llama.cpp's q4_k answer sits 3.53 (mean absolute) from its own f16 answer,
while ours sits 3.77 away and lands closer to the f16 truth. So the two quantized results are compared
on their argmax and their greedy sequence, not elementwise. Both sequences match exactly, and both
differ from the f16 sequence from the second token on: quantizing to 4.5 bits per weight is lossy
enough to change the continuation, which is the format working rather than a kernel failing.

### The CPU kernels use the whole machine

The CPU backend is not one core. `kernel/parallel.h` is a persistent `std::thread` pool — created
once, on first use, because the graph makes roughly two hundred kernel calls per token and
spawning threads per call would cost more than the arithmetic — and every kernel that has an
independent-output axis splits it across that pool.

**Not OpenMP.** `-fopenmp` would make `libgomp` a runtime dependency of `libpocketllm.so`, and the
whole reason this library exists is an edge and mobile target where the loader should have to find
as little as possible. A `std::thread` pool needs nothing beyond libstdc++ and pthread.

**The one rule is that a reduction is never split.** Every call site splits an axis whose iterations
touch disjoint memory — the output column of a GEMM, the token of a gather, the element of an
elementwise op — and leaves each output's accumulation over `k` whole inside one task. Splitting `k`
and summing partials would be faster and is forbidden: it changes the association order, so the
result would move with the thread count. `gemm_quant`'s comment says so where someone would be
tempted. Because the partition is a fixed contiguous split of an independent axis, the output is
*bit-identical* to the single-threaded kernel, and `tests/native/test_cpu_parallel.py` holds that
exactly by running each op at one thread and at eight and diffing the bytes.

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

### What it measures

Measured on the project's x86 host (2× Xeon E5-2696 v4, 88 hardware threads, two NUMA nodes), on
`qwen3-0.6b-q4_k_m.gguf`, `pp32`/`tg32`, median of five. `llama-bench` on the same checkpoint and
thread counts is the other column.

| threads | PocketLLM pp32 | PocketLLM tg32 | llama.cpp pp32 | llama.cpp tg32 |
|---:|---:|---:|---:|---:|
| 1 | 8.81 | 6.12 | 69.61 | 18.84 |
| 8 | 52.11 | 15.48 | 383.71 | 72.34 |
| 22 | 115.05 | 18.22 | 729.21 | 79.44 |
| 44 | 114.79 | 16.12 | 982.01 | 83.02 |

**This does not yet beat llama.cpp, and saying so is the point of the table.** Decode starts at a
third of llama.cpp's on one thread and stays there — the gap is per-core kernel efficiency, not core
count, because more threads hardly move it. Scaling also stops at the socket boundary (22 threads,
one node): past that the weights are all on the node the loader first touched, and remote reads cost
more than the extra cores return. The two obvious levers, a first-touch-parallel loader and a
quantized-activation (integer) dot the way llama.cpp does it, are follow-up work; neither is in this
change.

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