# Building and loading the CUDA extensions

`cuda_kernel` and `moe_dispatch_cuda_ext` are compiled in place by `setup.py` and loaded at
runtime by `src/kernels/cuda_loader.py`. Both halves fail quietly on purpose — the loader
returns `None` rather than raising, `--expert-device` falls back to the host expert path with
one line on `progress`, and a missing op surfaces only when a call site reaches it. A full run
of the wrong configuration therefore looks like a run.

This page is the two checks that make a run's build state knowable, and the list of sources that
sit in `src/csrc/` without being compiled.

## What the loader resolves, and in what order

`_find_built_extension` ([`src/kernels/cuda_loader.py:13`][loader]) searches two directories in
order — `build/extensions/`, then the repository root — and inside each one tries the exact name
`cuda_kernel.so` first, then `cuda_kernel<suffix>` for every suffix in that **running
interpreter's** `importlib.machinery.EXTENSION_SUFFIXES`:

```text
deepseek env, Python 3.11.14   ['.cpython-311-x86_64-linux-gnu.so', '.abi3.so', '.so']
base env,     Python 3.10.10   ['.cpython-310-x86_64-linux-gnu.so', '.abi3.so', '.so']
```

The tag is not a preference among the files present. It is a property of the interpreter, so
which build gets loaded is decided by which Python starts the process.

Only when neither directory holds any of those names does the search fall back to globbing
`cuda_kernel*.so` across the same two directories and taking the newest mtime
([`src/kernels/cuda_loader.py:25`][loader]) — and that fallback crosses cpython tags. It is the
one path that can load a build made for a different interpreter.

`build_ext --inplace` writes each extension into the repository root and then copies it into
`build/extensions/` under the same filename ([`setup.py:139-143`][setup]). The copy is what makes
`build/extensions/` the authoritative directory for the interpreter that last built there; the
root keeps whatever older builds were left behind by other interpreters. In this tree the root
holds two builds of `cuda_kernel` — `cpython-310` from 2026-06-15 and `cpython-311` from
2026-09-20 — and `build/extensions/` holds only the 3.11 one. Running under the base 3.10
environment finds nothing in `build/extensions/`, falls through to the root, and loads the
June build without a word.

## The two wrong-build failures, and how to tell them apart

Both are silent and both end with the same line on `progress`:

- **Wrong interpreter tag.** The 3.10 build predates the fp4 MoE ops, so a device-path run under
  the base environment reports `moe_single_token_fp4_forward is not available in the built
  extension` on all 40 layers and keeps the hosts' experts.
- **Stale tree artefact.** A merge changes a source and the tree's `.so` is not rebuilt, so the
  loaded binary is the previous revision's. Nothing about the file is missing — the op is there
  and answers correctly for the old code. This is why a `.so` on disk is not evidence of the
  source state after a merge, and why a rebuild belongs to the diff that motivates it.

The three checks below separate them. The first one is enough on its own to identify *which*
file is live; the third is what says whether that file is the revision you think it is.

```bash
# 1. which file the loader resolves, under the interpreter that will run the model
/home/lvyufeng/miniconda3/envs/deepseek/bin/python -c "
from src.kernels.cuda_loader import load_cuda_kernel
print(load_cuda_kernel())"

# 2. that file against the two copies and the sources it was built from
md5sum build/extensions/cuda_kernel.cpython-311-x86_64-linux-gnu.so \
       cuda_kernel.cpython-311-x86_64-linux-gnu.so
ls -l --time-style=long-iso src/csrc/*.cu src/csrc/*.cpp

# 3. the op the run actually needs -- the only check that sees a stale build
/home/lvyufeng/miniconda3/envs/deepseek/bin/python -c "
from src.kernels.cuda_loader import load_cuda_kernel
m = load_cuda_kernel()
print(len([n for n in dir(m) if not n.startswith('_')]), 'bindings')
print([n for n in dir(m) if 'fp4' in n])"
```

`build_ext` builds **every** extension in `setup.py`, and `pocketllm_cpp` does not survive
`TORCH_CUDA_ARCH_LIST=7.5` — its Qwen NVFP4 kernels are sm_80+. `cuda_kernel` is linked and
copied before that failure, so read the timestamp on the `.so` rather than the exit code:

```bash
PATH=/usr/local/cuda-12.4/bin:$PATH CUDA_HOME=/usr/local/cuda-12.4 TORCH_CUDA_ARCH_LIST=7.5 \
  /home/lvyufeng/miniconda3/envs/deepseek/bin/python setup.py build_ext --inplace
```

## What is compiled is `setup.py`'s list, not the directory

`src/csrc/` is not the compilation unit. The `cuda_kernel` extension lists eight sources
([`setup.py:205-215`][setup]):

```text
src/csrc/cuda_kernel.cpp                    src/csrc/cuda_kernel_impl.cu
src/csrc/minimax_rope_kernel.cu             src/csrc/llama_mmq/gguf_mma_wrapper.cu
src/csrc/qwen4_exp_moe.cu                   src/csrc/qwen4_exp_gated_delta.cu
src/csrc/qwen4_exp_qsa.cu                   src/csrc/qwen4_exp_hyper_connection.cu
```

and `moe_dispatch_cuda_ext` and `deepseek_cpu_moe_ext` each list their own. Everything else in
`src/csrc/` is either a header (all of `llama_mmq/`, `gguf_mma.h`, included by
`cuda_kernel_impl.cu`) or one of the three files below, which are on disk and not compiled:

| File | Why it is not compiled |
| --- | --- |
| `src/csrc/dot_microbench.cpp` | **Deliberate.** It defines `main()` ([`:172`](https://github.com/lvyufeng/PocketLLM/blob/master/src/csrc/dot_microbench.cpp)), which would collide with the extension's module init. `scripts/run_best_scheduler.sh:94` keeps it as a research artefact: its `pmaddubsw` sign-trick rewrite of the int8 dot measured ~2.0x on the inner kernel (30 → 60 GiB/s at dim 2048/4096) and was A/B'd twice at end-to-end delta +0.6% and +0.1%, i.e. inside jitter, because OMP is not on the critical path under the async-overlap configuration. |
| `src/csrc/minimax_gqa_kernel.cu` | **Not compiled, and referenced by nothing.** It defines `gqa_decode_qk_gemv_cuda` and `gqa_decode_attn_v_gemv_cuda`; a repository-wide grep finds no other caller, and no pybind registration. Dead source, harmless. |
| `src/csrc/fused_decode_gqa_attention.cu` | **Not compiled, and referenced by a test.** It defines `fused_decode_gqa_attention_cuda`, but no registration in `cuda_kernel.cpp` binds it and no built extension exports a name like it — so `tests/test_fused_decode_gqa_real.py:53`'s `cuda_mod.fused_decode_gqa_attention(...)` cannot resolve on any tree. |

All three were added by the same commit, `51ab5ab` ("Replace environment variables with typed
QwenKernelOptions API (#108)", 2026-09-13), which added the sources *and* the tests that call them
but did not touch `setup.py` — the compile entries and the binding registration were never added
alongside them.

## A missing op fails, and it does not skip

`pytest.importorskip` and `load_cuda_kernel() -> None` guard the **extension loading**. They do
not guard an individual op. A test that loads the extension successfully and then calls a
binding that does not exist gets an `AttributeError`, which is a failure, not a skip: four tests
in this tree are in that state — `tests/test_minimax_gqa_kernel.py::test_gqa_qk_gemv`,
`::test_gqa_attn_v_gemv`, `::test_gqa_full_attention` (all three call `gqa_decode_qk_gemv` /
`gqa_decode_attn_v_gemv`) and `tests/test_fused_decode_gqa_real.py::test_against_minimax_attention`.
The current 3.11 extension exports 51 bindings and none of them is any of those three names.

So **a failure in those four files is a missing source or a missing registration, not a missing
GPU** — read the error before reading the hardware. Fixing them means either adding the file to
`setup.py`'s source list and binding it, or deleting the test and the source together; leaving a
test that cannot pass spends a future reader's time on the wrong question.

[loader]: https://github.com/lvyufeng/PocketLLM/blob/master/src/kernels/cuda_loader.py
[setup]: https://github.com/lvyufeng/PocketLLM/blob/master/setup.py
