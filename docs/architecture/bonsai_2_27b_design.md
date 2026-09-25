# Ternary-Bonsai-2-27B: design and measurements

This is task 6 of 6 in [stage 1 of the checkpoint roadmap](pocketllm_new_model_roadmap.md#stage-1--ternary-bonsai-2-27b)
([#387](https://github.com/lvyufeng/PocketLLM/issues/387) under [#381](https://github.com/lvyufeng/PocketLLM/issues/381)),
the last one: wire the checkpoint to `pocketllm serve`, measure it there, and write it down. The five
pages before it are the [reference gate](ternary_bonsai_2_reference_gate.md), the container reader,
the Hadamard, and the [sm_75 dense GEMM](ternary_bonsai_2_dense_gemm.md); the user-facing page is
[the model guide](../models/ternary-bonsai-2-27b.md).

**Verdict: served on one card, and the prefill number the earlier stages recorded was wrong — for a
reason worth the whole page.** A prompt whose token count is a multiple of 64 prefills at
**636–647 tok/s**, which is 99% of what the upstream reference reaches on the same card, not the 35%
that the dense-GEMM stage measured. One token over the boundary costs **11.7 seconds**: a 4,096-token
prompt and a 4,097-token prompt differ by that much, on the same server, same content, one token. The
per-token rate is otherwise flat at 1.55 ms from 2,048 to 22,378 tokens, so the entire "prefill is 2.8×
behind the reference" gap is a ragged-tail effect in the final chunk, and the number this page records
for the stage is the aligned one.

Everything below was measured on this checkout, on **RTX 2080 Ti 0** (22,528 MiB, PCIe, sm_75), the
released `Ternary-Bonsai-2-27B-PTQ1_0.gguf`, one request at a time. Where a number is a single
reading it says so; where it is a difference between two arms, both arms are stated. The prompts are
repository documents, so every one of them can be reproduced from the checkout — `README.md`, the
pages of this documentation tree, and prefixes of them by token count. Token counts are the engine's
own, read from `/metrics`, not estimates.

## Runtime status

| | |
| --- | --- |
| Backend | `pocketllm serve --model <file>.gguf` selects the native C++ engine with no flag |
| Cards | 1, no NVLink needed |
| Context served | 245,760 tokens at an FP16 KV cache; 262,144 with `--kv-cache-dtype fp8` |
| Acceptance | the server answers a chat completion from the released checkpoint on one 2080 Ti |
| Prefill, 4,096-token prompt | 6.44 s — **636.0 tok/s** |
| Decode, at that context | **25.89 tok/s** |
| Memory, weights + runtime | 6,566 MiB, of which 5.53 GiB is the ternary file |
| What it does not do | no batching in practice (measured, below), no speculative decoding, greedy only |

## What serving needed

Two things, and neither was a runtime change.

**Backend selection already worked, at the registry.** The file declares
`general.architecture = qwen35`, and the C++ registry canonicalizes `qwen35` onto the `qwen3_5` name
that the Qwen3.8-27B runtime claims, so the engine that runs the FP8 checkpoint is the engine that
runs this one, unmodified. The gap was on the Python side: the adapter's factory refused a GGUF
checkpoint before it asked the registry. That is the whole of
[#413](https://github.com/lvyufeng/PocketLLM/pull/413) — a `gguf_is_servable` predicate, the file
substituted for a directory when the adapter hands the checkpoint down, and a test that pins the
canonicalization.

**The tokenizer comes out of the file.** A GGUF holds the vocabulary, the merges, the special-token
ids and the chat template, and this one declares `tokenizer.ggml.pre = qwen35`, whose pre-tokenizer is
the same digit-splitting rule the other Qwen-family checkpoints use. `build_gguf_bpe_tokenizer` reads
the header; `tests/test_gguf_tokenizer_pre.py` pins the split on a synthetic vocabulary so a
regression there cannot hide behind a real one.

The released directory holds a **second artifact** beside the file this stage serves
(`Ternary-Bonsai-2-27B-PQ2_0.gguf`, GGML type 142), so a directory naming two checkpoints is refused
with its own message rather than resolved by listing order. `PQ2_0` has no kernel: the loader decodes
its geometry and refuses it by name, which is deliberate — reading a 2.13-bit block as though it were
`PTQ1_0` would produce plausible tokens from the wrong numbers.

## The measurement harness

`scripts/bench_pocketllm_serve_phases.py` splits one served request into prefill and decode from the
**engine's own clock**, not from chunk arrival times: `pocketllm_ttft_seconds_sum/_count` is the time to
the first generated token, `pocketllm_request_duration_seconds_sum` the whole request. By this
repository's [timing convention](../guides/benchmarking.md) the first token is produced by the prompt
forward and therefore belongs to prefill, so `prefill_tps = prompt_tokens / ttft`. The script refuses
to report a row whose metric deltas are not exactly one request's — the trap being a stray health
check or a second client landing inside the window.

`scripts/bench_pocketllm_serve_concurrency.py` is the third of these server benchmarks: several
simultaneous non-streamed requests, aggregate tokens a second against the same figure at concurrency
1. It is non-streamed on purpose — the streaming path holds one lock for a whole generation, because
the engine has a single mutable KV session, so a streamed run would measure the lock.

## The ragged tail: what actually governs prefill

The stage's headline number was 225.9 tok/s at a 4,108-token prompt, against the reference's
642.5 tok/s at 4,096, and it was read as a kernel being 2.8× too slow. It is not a rate. It is a step.

Here is the step, served, on one server configuration, two prompts whose content is identical except
for the last token (`README.md` truncated to 4,079 and to 4,080 tokens; the chat template brings the
rendered prompt to 4,096 and 4,097):

| Rendered prompt | TTFT | Prefill | Decode |
| ---: | ---: | ---: | ---: |
| **4,096 tokens** (64 × 64) | **6.440 s** | **636.0 tok/s** | 25.89 tok/s |
| 4,097 tokens | 18.106 s | 226.3 tok/s | 25.76 tok/s |

One token, 11.7 seconds. And the aligned rate holds at the other boundary:

| Rendered prompt | TTFT | Prefill |
| ---: | ---: | ---: |
| 8,192 tokens (64 × 128) | 12.818 s | 639.1 tok/s |

At the engine, one prefill per document, `prefix_cache` off, a fresh process each so nothing can
resume, the same document cut to a token count — the rate is flat at the top and the step is at the
bottom:

| Tokens | ms a token | tok/s | |
| ---: | ---: | ---: | --- |
| 2,048 (64 × 32) | 1.59 | 628.1 | aligned |
| 4,096 (64 × 64) | 1.55 | 646.7 | aligned |
| 8,192 (64 × 128) | 1.58 | 634.6 | aligned |
| 9,216 (64 × 144) | 1.60 | 625.5 | aligned |
| 4,100 | 4.40 | 227.4 | +4 |
| 4,112 | 2.76 | 362.6 | +16 |
| 4,128 (32 × 129) | 2.02 | 495.3 | +32 |
| 4,159 | 4.40 | 227.4 | +63 |
| 4,161 | 4.46 | 224.4 | +1 into a second tile |

The pattern in the last column is the cost, and it is a function of the **tail**, not of the length or
of the text:

| Tail past the last whole 64-token tile | Penalty | |
| ---: | ---: | --- |
| 1 | 12.2 s | |
| 4 | 11.7 s | |
| 16 | 5.0 s | |
| 32 | 1.9 s | |
| 63 | 11.9 s | |

so the engine is processing the final ragged chunk through something that costs about
**0.4 s for every slot of the tile it cannot fill**, up to ~12 s, and nothing at all when the tile is
full. That fit is from the 4,0xx series above and it is a description, not a law: the longer
documents' totals land within a few seconds of what it predicts rather than on it. The shape of the
cost is a fallback that does the whole model's work again for the ragged chunk rather than tiling it,
which is exactly the "prefill tile geometry" that
[#406](https://github.com/lvyufeng/PocketLLM/issues/406) tracks — but that is a reading of the shape,
not a diagnosis, and one number says it is not a plain re-read: a pass over the 5.53 GiB weight set
costs 11 ms at this card's measured 526 GiB/s, so 11.7 s is a thousand times that. It is within a
factor of the file read off a disk at ~500 MB/s, which is a coincidence this page will not lean on —
the penalty reproduces on the sixth run of a series, long after the file is in the page cache.
**The mechanism is not diagnosed here**; what is measured is the penalty, its trigger, and that it is
not a throughput ceiling.

Three things follow, and they were all checked:

- **It is not the document.** `docs/models/mimo-v2.6-flash.md` whole is 3,422 tokens and measures
  224–228 tok/s; the same file cut to 2,880 tokens (64 × 45) measures 640 tok/s in a fresh process.
  Alternating a prompt and a one-token-variant of it three times each in one process reproduces
  4.4 ms and 1.9 ms a token per variant, to three figures, so it is neither warm-up nor drift.
- **It is not the chunk size.** README at 11,024 tokens, one fresh process per row: chunk 2,048
  → 560.8 tok/s, 4,096 → 515.2, 8,192 → 517.4, 16,384 → 355.5. Smaller chunks are worth a few per
  cent, and no chunk setting crosses the step.
- **It is not the card.** No other process held a GPU during any of these runs
  (`nvidia-smi --query-compute-apps` empty), and the numbers reproduce across sessions.

### What this does to the earlier stage's numbers

The dense-GEMM page measured prefill at a **fixed 512-row shape** — 600 tok/s of dense projections —
and 512 is a multiple of 64, so that number is on the fast path and stands. The stage-6 number that
was wrong is the *whole-model* one: 4,108 tokens is not a multiple of 64, so the 18.19 s it took was
6.4 s of prefill plus a 11.8 s ragged-tail penalty, and reading it as a rate put the model at 35% of
the reference when the aligned figure is 99%.

Served end to end, at `--max-model-len 32768`, one request at a time, greedy, 64 generated tokens:

| Prompt (repository document) | Prompt tokens | TTFT | Prefill | Decode |
| --- | ---: | ---: | ---: | ---: |
| `README.md` cut to an aligned 4,096 | 4,096 | 6.44 s | **636.0 tok/s** | 25.89 tok/s |
| the same cut to 4,097 | 4,097 | 18.11 s | 226.3 tok/s | 25.76 tok/s |
| `README.md` cut to an aligned 8,192 | 8,192 | 12.82 s | 639.1 tok/s | 25.44 tok/s |
| `README.md` whole | 11,040 | 19.2 s | 571–576 tok/s | 25.0 tok/s |
| `docs/architecture/qwen3_8_27b_fp8_design.md` | 12,659 | 33.6 s | 374.6–376.6 tok/s | 24.9 tok/s |
| `docs/performance/serving_throughput_scaling.md` | 22,389 | 56.1 s | 398.9 tok/s | 24.1 tok/s |
| `docs/architecture/ternary_bonsai_2_reference_gate.md` | 6,151 | 27.6 s | 222.6 tok/s | 25.5 tok/s |
| a prompt that repeats the one just served | 5,032 | **0.023 s** | — | 25.6 tok/s |

The multi-chunk rows are the same rule read twice: 11,040 splits into an aligned 8,192 and a 2,848
tail of 32, which the engine answers at 571 tok/s — between the aligned rate and the ragged one,
which is what a 1.9 s tail against 11,040 tokens predicts.

### Against the upstream reference, same card

| | PocketLLM, one card | llama.cpp `prism`, one card |
| --- | ---: | ---: |
| Prefill, aligned | **636.0 tok/s at 4,096** | 642.5 tok/s at 4,096 |
| Prefill, aligned | 639.1 tok/s at 8,192 | 615.2 tok/s at 8,192 |
| Decode | 25.9 tok/s at 4,096 | 30.7 tok/s |

Prefill is level with the reference — 99% at 4,096, 104% at 8,192 — and decode is 84% of it. The
reference's own `llama-bench` prefills in one shot and therefore never pays the tail, which is worth
saying explicitly: the comparison is aligned-to-aligned, and a serving stack that hands the engine
arbitrary prompt lengths does not get to be aligned by accident.

## Memory on one card

| | |
| --- | ---: |
| The ternary file | 5.53 GiB — 402 `ptq1_0` tensors, 5,946,648,928 bytes |
| Weights plus runtime, no request | **6,566 MiB** |
| KV | **64 KiB a token** — 16 full-attention layers × 4 KV heads × 256 × 2 × 2 bytes |
| Idle at `--max-model-len 8192` | 7,078 MiB |
| After the first cold 4K prompt | 8,952 MiB, flat from there |
| The most that fits, FP16 KV | **245,760 tokens** — 15,360 MiB of KV + 6,566 MiB of the rest |
| 262,144 tokens | needs `--kv-cache-dtype fp8`; 15,014 MiB on the card |

The KV half is the same number derived two ways. The architecture gives 64 KiB a token; four fresh
processes at `--max-model-len` 2,048, 8,192, 16,384 and 32,768 measured 6,694, 7,078, 7,590 and
8,614 MiB, which is 64 KiB a token above a **6,566 MiB** intercept, to the megabyte:

| `--max-model-len` | Measured | Intercept + KV |
| ---: | ---: | ---: |
| 2,048 | 6,694 MiB | 6,566 + 128 |
| 8,192 | 7,078 MiB | 6,566 + 512 |
| 16,384 | 7,590 MiB | 6,566 + 1,024 |
| 32,768 | 8,614 MiB | 6,566 + 2,048 |

`--max-model-len 253952` is refused at start with `RuntimeError: device_malloc Qwen runtime tensor`:
the arena does not fit. 245,760 is the largest FP16-KV context, and fp8 halves the KV so the
checkpoint's own 262,144 fits.

**The workspace is a per-thread cache, and it used to leak.** A cold prefill of a long prompt leaves
~430 MiB on the card for a 4K prompt, because the chunk's activation scratch is cached per thread at
its high-water mark rather than reallocated per chunk. That is intended. What was not: the cache was a
trivially destructible `thread_local` holding a raw device pointer, so it was never released when the
thread went away, and `ThreadingHTTPServer` answers each connection on a fresh thread — one workspace
per request, 430 MiB a prompt, linear, until the card filled. Fixed in
[#414](https://github.com/lvyufeng/PocketLLM/pull/414) with a destructor that frees it, and measured
before and after on the same tree, eight cold 4,096-token prompts, one process both times:

| | fresh | after #1 | after #2 | … | after #8 |
| --- | ---: | ---: | ---: | ---: | ---: |
| before | 7,078 | 9,382 | 9,812 | | 12,392 MiB |
| after | 7,078 | 8,952 | 8,952 | | 8,952 MiB |

The direct engine path never showed it because it reuses one thread.

## Concurrency, and why it is off

The batch scheduler is in the engine and the adapter can reach it — `--backend-option
enable_batching=true --backend-option max_batch_size=4`. Two prompts, prefix caching off in both
arms, one timed group of simultaneous non-streamed requests per level, aggregate generated tokens a
second over the whole group against concurrency 1. Both arms were measured on this branch, in one
invocation each, so the only difference between the columns is the flag.

A short prompt — `"The capital of France is"`, 17 prompt tokens, 9 generated, because this model
stops on its own well short of the 64 asked for:

| Concurrency | Serial | Batched |
| ---: | ---: | ---: |
| 1 | 21.49 tok/s (1.00×) | 20.48 tok/s (1.00×) |
| 2 | 21.58 tok/s (1.00×) | 25.47 tok/s (1.24×) |
| 4 | 21.54 tok/s (1.00×) | **36.53 tok/s (1.78×)** |

And a prompt long enough for prefill to matter — the reference-gate page truncated to 2,032 tokens,
which the template renders to 2,044, 64 generated:

| Concurrency | Serial | Batched |
| ---: | ---: | ---: |
| 1 | 5.51 tok/s (1.00×) — 11.61 s | 5.45 tok/s (1.00×) — 11.74 s |
| 2 | 5.44 tok/s (0.99×) — 23.55 s | 5.77 tok/s (1.06×) — 22.17 s |
| 4 | 5.37 tok/s (0.97×) — median 35.75 s, slowest 47.69 s | **6.23 tok/s (1.14×)** — 41.09 s |

Read the serial column as the queue it is: one worker, so the aggregate does not move and the fourth
caller waits as long as the four before it. Batching buys 1.78× on the short prompt and 1.14× on the
long one, and charges for it in latency — at 2,044 tokens the batched group's median is 41.09 s
where the serial queue's median is 35.75 s, so four callers wait longer in total than they would
have queued, and only the queue's last caller (47.69 s) waits longer than that.

**On determinism, this page corrects an earlier reading of its own.** The batching caveat was first
written as "a greedy answer can depend on the batch": four identical requests in one group produced
two distinct texts, and a prompt choosing between `"5:00:00"` and `"0:15:00"` gave one row each.
That measurement came from a build whose greedy runs on that prompt did not stop at `</s>` — all 64
tokens were `"!"` — and on this tree the same probe gives `'The capital of France is **Paris**.'` at
nine tokens, identical across all four rows, with the prefix cache on and off and over repeated
runs. The two are not the same regime, so the caveat is **not** reproduced here and is not claimed
for this build; what remains is the latency trade-off above, and a note that the earlier observation
is unexplained rather than withdrawn.

`--backend-option kv_paged=true` is the other opt-in, and it is measured in the model guide: the
prefix resume disappears (a repeated prompt is forwarded in full every time) and decode loses about
3% (25.0 against 25.8 tok/s). Paging exists for the case where one arena will not do.

**The prefix resume is a resume, not a store.** The engine keeps the state of the request it just
served. A, A, B, B, A gives 0.023 s and 0.014 s TTFT for the two repeats and a full prefill for the A
at the end, so a repeat of the immediately preceding prompt resumes and a repeat of an earlier one
does not, whatever `--enable-prefix-caching` or the arena size says. In the served table above the
resumed request is the one at 0.023 s.

## Correctness and what is not claimed

The generated text is the evidence that the container, the transform and the kernels compose: the
engine test generates `Paris. / The capital of Germany is Berlin. / The capital of Italy is Rome.`
from the released file, all 64 layers, greedy, and the per-kernel evidence — bit-exact integer dots
against the decoded blocks, the loader against the fork's own tiles — is in the earlier stages' pages
and on `tests/`.

Not claimed here: any quality or accuracy figure for the checkpoint (those belong to the authors'
evaluations), speculative decoding (MTP, DSpark and DFlash2 exist in this engine for the FP8
checkpoint and none was run against the ternary artifact), and TP > 1 (the engine supports TP4, which
is how the FP8 sibling is served, but no multi-card run was made for this one).

The alignment penalty is measured but not diagnosed. It is called out here so that the next person to
serve this checkpoint — or any checkpoint on this engine — does not measure a 4,097-token prompt and
conclude the model is slow.

## Evidence

```bash
# the module the server loads; the deepseek env is a 3.11 interpreter, so the
# extension must be built for it and copied into site-packages
PATH=/usr/local/cuda-12.4/bin:$PATH CUDA_HOME=/usr/local/cuda-12.4 \
  cmake --build cpp_engine/build-python -j 16 --target pocketllm_cpp
cp cpp_engine/build-python/python/pocketllm_cpp.cpython-311-x86_64-linux-gnu.so \
   "$(python -c 'import site;print(site.getsitepackages()[0])')"

# the server, from the checkout so `python -m pocketllm` resolves the working tree
python -m pocketllm serve --model /path/to/Ternary-Bonsai-2-27B-PTQ1_0.gguf \
  --served-model-name bonsai --max-model-len 32768 --port 8123

# the phase split, from the engine's own clock; one request at a time
python scripts/bench_pocketllm_serve_phases.py --url http://127.0.0.1:8123 \
  --prompt-file docs/architecture/ternary_bonsai_2_dense_gemm.md --max-tokens 64

# several requests at once, aggregate against concurrency 1
python scripts/bench_pocketllm_serve_concurrency.py --url http://127.0.0.1:8123 \
  --concurrency 1 2 4 --max-tokens 64

# the alignment probe: a repository document cut to an exact token count with
# the checkpoint's own tokenizer. 4,079 + the chat template's 17 = 4,096, which
# is 64 x 64; 4,080 renders to 4,097 and pays the tail. Serve each on a restarted
# server, because the second prompt is a prefix of nothing but the first is of it.
python - <<'PY'
from src.encoding.gguf_tokenizer import build_gguf_bpe_tokenizer
tok, _ = build_gguf_bpe_tokenizer("/path/to/Ternary-Bonsai-2-27B-PTQ1_0.gguf")
ids = tok.encode(open("README.md", encoding="utf-8").read()).ids
for raw, name in ((4079, "aligned.txt"), (4080, "tail.txt")):
    open(f"/tmp/{name}", "w").write(tok.decode(ids[:raw]))
PY

# the C++ tests, which is where the container and the kernels are pinned
cpp_engine/build/tests/test_qwen_ternary_engine     # the capitals, 64 layers
cpp_engine/build/tests/test_qwen_ternary_gemm
cpp_engine/build/tests/test_qwen_hadamard_ops
cpp_engine/build/tests/test_gguf_ternary_reader
python -m pytest tests/test_ptq1_0_layout.py tests/test_gguf_tokenizer_pre.py \
                 tests/test_backend_selection.py -q
```

- [The model guide](../models/ternary-bonsai-2-27b.md) — what a user needs, and the same numbers
  without the probes
- [The reference gate](ternary_bonsai_2_reference_gate.md) — what upstream measures on this card
- [The sm_75 dense GEMM](ternary_bonsai_2_dense_gemm.md) — the two ternary kernels, their correctness
  evidence, and the fixed-512-row measurement this page corrects at the model level
- [#406](https://github.com/lvyufeng/PocketLLM/issues/406) — the prefill tile geometry, which the
  ragged-tail measurement above belongs to
- [Benchmarking and reporting rules](../guides/benchmarking.md) — the convention every number here
  follows
- The support matrix in [models/README.md](../models/README.md)
