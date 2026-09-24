# DeepSeek-V4.1-Flash: what one request costs, through the launcher

The [device experts page](deepseek_v4_1_flash_device_experts.md) closes its decode section with a
paragraph about what it does not claim, and this page exists to answer it: *"The 512-token prefill is
the probe rather than the launcher — `src/cli/generate_v41.py` does not print a prefill wall — so the
prefill column is the instrumented library at the launcher's own configuration, and the decode column
is the launcher undecorated."* Both columns here are the launcher, over four prompt lengths, one leg
to a process, every figure derived from lines the CLI itself prints.

**One request on four 2080 Ti, through the shipped CLI: prefill 106.0 tokens a second at 32768 prompt
tokens and 103.6 at 262144, decode 4.98 tokens a second at 1024, 4.37 at 32768 and 3.86 at 262144.**
Every figure on this page is the `sorted` expert deal's, which was the default when the legs were run
and is one flag from it now; the default deal's own rates are **1.62×, 1.48× and 1.40× on the prefill
and 5.18, 4.20 and 3.70 tokens a second on the decode**, and the paragraph after the table is where
that matters.
The graphed decode step is **201–202 ms** at 1024 against **354–364 ms** eager — 1.76–1.81× — and the
whole call 32.8 s against 41.3–42.1 s, with the eight 1024-token legs' generated text **identical to
the byte** and the two paths' 64 x 129280 logit matrices **bit-identical** (`max |diff| = 0.000e+00`,
0 rows differing, 64/64 argmax).

| Leg | Prompt | Flags | Prefill | ms a token | Decode | Whole call |
| --- | --- | --- | --- | --- | --- | --- |
| `c1024_off_a2` | 1024 | `--expert-pool-rows 288 --no-decode-graphs` | 55.0 tok/s | 18.2 | **2.82 tok/s** (354 ms) | 41.3 s |
| `c1024_on_a2` | 1024 | `--expert-pool-rows 288 --decode-graphs` | (51.2) | (19.5) | **4.98 tok/s** (201 ms) | 32.8 s |
| `c1024_off_b2` | 1024 | `--expert-pool-rows 288 --no-decode-graphs` | 54.5 tok/s | 18.4 | **2.75 tok/s** (364 ms) | 42.1 s |
| `c1024_on_b2` | 1024 | `--expert-pool-rows 288 --decode-graphs` | (51.5) | (19.4) | **4.95 tok/s** (202 ms) | 32.8 s |
| `c32768c_on` | 32716 | `--expert-pool-rows 148 --prefill-chunk-tokens 4096 --decode-graphs` | **105.98 tok/s** | 9.44 | **4.37 tok/s** (229 ms) | 323.3 s |
| `c262144_on` | 262865 | `--expert-pool-rows 148 --prefill-chunk-tokens 4096 --decode-graphs` | **103.58 tok/s** | 9.66 | **3.86 tok/s** (259 ms) | 2554.5 s |
| `c262144c_on` | 262874 | as above, extended prompt | **103.54 tok/s** | 9.66 | **3.88 tok/s** (258 ms) | 2555.4 s |

`ms a token` is the prefill's, 1000 / the rate. The 1024 rate is in parentheses on the graphed legs
because on that path the same subtraction carries the capture pass — a decode step that is not one of
the 64 tokens — so the figure is a floor rather than a measurement; the eager column beside it is the
clean one, and the two prompts are one unchunked forward rather than the chunked configuration the
long lengths run. **262144 is in the table twice**, on the two prompts below: 103.58 and 103.54
tokens a second, 36 bytes apart in the prompt and 0.04% apart in the rate, and 259 against 258 ms a
decode step.

Every leg is `DEEPSEEK_V41_RESIDENT_EXPERTS=1`, `--threads 22`, `--temperature 0.0`,
`--max-new-tokens 64`, `torchrun --nproc_per_node=4`, and **the `sorted` expert deal** — the deal
every figure on this page was taken on, and one flag from the default since 2026-09-20. That
last one matters for the prefill column: `id`, which is the default now, is priced on the
[device-experts page](deepseek_v4_1_flash_device_experts.md#the-flips-own-two-questions-answered-through-the-launcher)
at **1.621x at 1024 tokens, 1.481x at 32716 and 1.400x at 262874** by the same subtraction this page
uses, and on the
[chunked prefill page](deepseek_v4_1_flash_chunked_prefill.md) at **146.46 tok/s on this same 262144
length** against the `sorted` arm's 103.50 — the numbers here are the deal the run record was taken
on, not the one a bare command line makes. That sitting also prices the flip's other half: decode is
**11 ms a step faster** under `id` at 1024 (204 to 193 ms) and **10 and 12 ms slower** at 32716 and
262874 (228 to 238, 258 to 270), so the decode rates below are one flag from the shipping default's at
every length and the default's own rates are, in order, 5.18, 4.20 and 3.70 tokens a second.

## Run record

| | |
| --- | --- |
| Model | DeepSeek-V4.1-Flash, released checkpoint, fp8 dense + packed-fp4 experts |
| Checkpoint | `/mnt/data3/DeepSeek-V4.1-Flash`, 48 shards, **resident bank attached** (457.8 GiB, one `cudaHostRegister`) |
| Commit | `origin/master` at `f7572f0` (the merge of #304); the worktree's `attention.py`, `decode_pos.py`, `modules.py`, `generate.py` and `graphs.py` byte-identical to it |
| Configuration | TP4, one process a card, `torchrun --nproc_per_node=4`, `--threads 22`, `--temperature 0.0`, `--max-new-tokens 64` |
| Prompts | `/tmp/prompt1024.txt` (1024), `/tmp/prompt32768c.txt` (32716), `/tmp/prompt262144c.txt` (262865); contexts 1088, 32832, 262976 |
| Decode | 64 tokens a leg, `stopped on length` on every leg in the table; first token 271 at 1024 |
| GPUs | 4 x RTX 2080 Ti, 22528 MiB each, `GPU0-GPU1` PHB and `GPU2-GPU3` NV2, all four idle before and after |
| Software | Python 3.11.14, torch 2.9.1+cu128, `deepseek` conda env; master's built `cuda_kernel` / `moe_dispatch` extensions, nothing rebuilt for these legs |
| Driver | `/tmp/run_cap_e2e.sh`, `/tmp/run_cap_e2e2.sh` (the six legs) and `/tmp/run_cap_e2e3.sh` (the two `--dump-logits` legs); logs `/tmp/cap_*.log`, comparison `/tmp/cmp_logits.py` |

## The rate the launcher does not print

The launcher prints two walls and neither of them is the prompt's. `elapsed` covers the whole
`generate()` call, and on every leg here `decode_seconds` starts *after* the prompt's forward — after
`front(torch.tensor([ids]), position, chunk=prefill_chunk)` on the eager path, after
`driver.capture_pass(...)` on the graphed one. So

    prefill ≈ prompt_tokens / (elapsed − decode_seconds)

is a subtraction between two printed numbers rather than a stopwatch inside the library, and the
load is outside both: `started = time.perf_counter()` is taken after `load_backbone` returns, so the
118–127 s of loading is not in `elapsed`.

**That remainder does not survive nowhere else, and three cross-checks say so.** The 1024-token
forward comes out at 18.2 ms a token eager, and the
[chunked prefill page](deepseek_v4_1_flash_chunked_prefill.md)'s width curve — measured with an
instrumented library at chunks of 1024, 2048 and 4096 — gives 18.57 ms a token at chunk 1024. The
32716-token leg is 7.99 chunks of 4096 and 308.7 s of remainder, **38.65 s a chunk**; the
[prefill stack page](deepseek_v4_1_flash_prefill_stack.md) measures the shipping tree's quiet chunk
at 38.07 s and its 262144 leg at 39.39 s. The 262865-token leg is 64.18 chunks and 2537.9 s of
remainder, **39.54 s a chunk**, against that 39.39. Neither the per-chunk figure nor its growth from
32768 to 262144 is a number this page invented.

Two things the subtraction carries that a printed wall would not: on the graphed legs it includes the
capture pass, worth roughly one step of the 40 layers' twice-run bodies, and on all legs it includes
the first token's pick and the `front.reset_state` that precedes the prompt. Both are in the
conservative direction, and both are small against a 308.7 s remainder — the capture is a
millisecond-scale term here, and the first `decode steps:` mark is taken after it.

## What a step is made of

The graphed legs print their own split — host marks around the two captured halves and the eager
expert call between them — and it says the same thing at every length: **the expert call is the
step**.

| Length | Graph A | Eager experts | Graph B | Step |
| --- | --- | --- | --- | --- |
| 1024 | 1.3 | **184.4** | 1.0 | 201 ms |
| 32768 | 1.9 | **211.6** | 1.1 | 229 ms |
| 262144 | 2.1 | **239.6** | 1.2 | 259 ms |

92% of a 1024-token step, 92% of a 32768-token one and 92.5% of a 262144-token one is the routed
experts run eagerly between two graphs. The graphs are 2.3 ms of a step at 1024 and 3.3 ms at
262144; what they remove is the tree's 40 blocks of dispatch, which is why the same step is 201 ms
graphed and 354–364 ms eager at 1024 — 153–163 ms of launch overhead, and 43–46% of the eager step.

`decode graphs:` also reports the replay count as `len(tokens) + 1` — the capture pass's replay and
the step replayed behind it, then one per token — so 65 on every leg here, at **260.0 MiB** of shared
pool over 40 layers.

## The 1024 legs against the older pages, and the 32768/262144 ones

The [live graphed decode page](deepseek_v4_1_flash_decode_graph_live.md) records 408–431 ms a token
on the same 1024-token prompt, and this page records 201–202. They are **not the same
configuration** and must not be subtracted: that page's tree is `e70cc12` (2026-09-18) plus the
`perf/v41-decode-graph` working tree, which predates `149bbe0`'s bank pinning and direct upload
(2026-09-19), so its expert call is still the one that copies each staged row out of the pinned
arena through `_stage`. The device experts page prices exactly that removal at **200/202 ms a decode
token against 341–348 without it**, and its eager arm's 341–348 ms is this page's eager 354–364 ms
as much as its graphed 200/202 is this page's 201/202. What is new here is not the number: it is
that the number comes out of `src/cli/generate_v41.py`'s own two lines, on the merged tree, with the
eight-way text parity below as the acceptance.

| Leg | `--expert-pool-rows` | Evictions | Staged draws | Card peak allocated |
| --- | --- | --- | --- | --- |
| 1024 (graphed) | 288 | 9903 of 288 rows a card | 10191 of 87200 (88.3%) | 15.06 GiB |
| 32716 | 148 | 81051 of 148 rows a card | 81199 of 2622560 (96.9%) | 15.01 GiB |
| 262865 | 148 | 618742 of 148 rows a card | 618890 of 21034480 (97.1%) | 15.89 GiB |

`--expert-pool-rows 288` at 1024 and **148 plus `--prefill-chunk-tokens 4096` at both long
lengths** is not a preference. 288 is −7.2% a chunk at 32768 for +4.34 GiB and dies in the second
chunk at 262144, the flag's own help says so, and the run confirms the second half of it: the 32716
and 262865 legs both load at 148 rows and finish, on the same 22 GiB a card budget, at 15.01 and
15.89 GiB peak allocated.

## Five tokens a second is not a guarantee

The question this sitting was commissioned to answer was whether single-request prefill of 100
tokens a second and decode of 5 tokens a second can be promised. **Prefill: yes at the long lengths,
under the flags above.** 105.98 and 103.58 tokens a second are measured, both above 100, both the
default deal, both on the merged tree.

**Decode: no, not as a flat claim.** 4.98 and 4.95 tokens a second hold at a 1024-token context — but
that is 1088 tokens of a 262976-token ceiling, and the rate degrades with context exactly as the
step's expert call grows:

| Context | Step | Decode |
| --- | --- | --- |
| 1088 | 201–202 ms | 4.95–4.98 tok/s |
| 32832 | 229 ms | 4.37 tok/s |
| 262976 | 258–259 ms | 3.86–3.88 tok/s |

A **guarantee at 1024, 4.37 at 32768 and 3.86 at 262144** is what the measurements support. The
degradation is 28% over a 256× increase in context and it is not the attention's: the eager expert
call moves 184.4 → 211.6 → 239.6 ms across those three lengths while the graphs move 2.3 → 3.0 →
3.3 ms, so 55.2 ms of the 58 ms the step grew is expert staging against a pool that evicts on
essentially every row at the long lengths (96.9% and 97.1% of draws staged).

## The two paths agree, at the tokens and at the logits

**The logits.** Two legs on the merged tree, the same prompt, `--dump-logits` on each, one eager and
one graphed, compared with `/tmp/cmp_logits.py`:

```text
/tmp/capdump_off.pt: decode_graphs=False rows=64 prompt_tokens=1024
  read_bytes delta across the 64 tokens: 0 (per token [0, 0, 0, 0, 0, 0, 0, 0] ...)
/tmp/capdump_on.pt: decode_graphs=True rows=64 prompt_tokens=1024
  read_bytes delta across the 64 tokens: 0 (per token [0, 0, 0, 0, 0, 0, 0, 0] ...)
  tokens vs /tmp/capdump_off.pt: 64/64 identical
    logits: 64 rows x 129280, max |diff| = 0.000e+00 (worst row 0), rows differing = 0
    argmax agreement: 64/64
```

This is the acceptance that matters for #304, because the fix is inside `_TopKStream.push`: when a
stream is capturing, `_recording()` is True and the early-out that compares the held values against
the tile's is skipped. That early-out is **exact rather than a tie-break** — if every held value
exceeds every tile value then the k largest of the union are the k already held, so dropping it
inside a capture changes neither the values nor the positions, tie order aside, and `Indexer` re-sorts
by position at the end. The margin by which the two paths agree is the whole matrix, so the argument
and the measurement say the same thing. `read_bytes` is flat at 0 on both legs, which is the check
that the expert rows came out of the pinned bank rather than off `/mnt/data3`.

These two legs are not in the table above and their walls are not part of the A-B-A-B: 43.9 s and
381 ms a token eager against 33.0 s and 203 ms graphed, taken while the pytest suite was running on
the same host. The eager arm is the one that moved (381 against 354–364 ms); the graphed arm is
203 against 201–202. Parity is not a timing claim, so the contention costs the comparison nothing.

**The text, eight ways.** Every 1024-token leg — four in the first sitting and four in the second,
alternating the flag, on **two trees** — printed the same continuation. Taken as the printed text
minus the prompt, it is **301 bytes** and hashes `md5 d1775d3580129c2953d16060b73c915c` on all eight:

```text
…ote in his notebook that a city is a question its river has already answered. Nobody in the hall
looked up.

Chapter 12. The traveller reached Stockholm on a wet morning in late autumn, having followed the Seine
for the better part of a week. The station was quiet, the platform lamps still burning,
```

Two sittings were needed because the first sitting's two eager legs ran on a tree that did not yet
have the #304 fixes. The fixes are inert on the eager path by argument — `_recording()` is False when
no stream is capturing, so `host_branch` is True and `_TopKStream.push`'s early-out is taken exactly
as before — but *inert by argument* is not *the same tree*, so all four 1024 legs were re-run
interleaved on one tree. The eager arms moved 358/368 → 354/364 ms, inside this host's noise, and the
eight-way hash is what says the two paths and the two trees agree on all 64 argmaxes.

## One leg that measured no decode

`c32768_on`, the first sitting's 32768 leg, reports `1 tokens in 307.2 s, stopped on eos` and no
`decode steps:` or `decode graphs:` line at all. That is not a lost measurement, it is a path:
`_decode_graphs` picks the first token from the prefill's last row **before** it captures anything,
and returns without building a graph when that token is eos — `graphs.py` reports nothing because
`result.driver` stays `None` and `decode_seconds` stays 0. The prompt (`/tmp/prompt32768.txt`) ends
mid-clause, *"…the pl"*, which the model reads as a finished document. Its prefill is still usable —
307.2 s of remainder over 32707 tokens, and it is what the 106.5 tok/s first reading came from — but
it is not in the table above, because a leg with no decode in it cannot be one of a page about
decode.

`/tmp/prompt32768c.txt` and `/tmp/prompt262144c.txt` append `"Chapter 345. The traveller reached"` to
the two long prompts so that the first token cannot be eos, which is a change to the prompt made for a
reason that has nothing to do with any rate on this page: the append is 36 and 36 bytes on 155 KB and
1.2 MB prompts, and both extended legs generate the intended `Chapter 345.` continuation. The second
262144 leg is that prompt *instead of* the original, and it is the check that the append is inert:
103.54 against 103.58 tokens a second, 258 against 259 ms a step.

## Reproduce

```bash
ENV=/home/lvyufeng/miniconda3/envs/deepseek/bin
CKPT=/mnt/data3/DeepSeek-V4.1-Flash

# 1024, A-B-A-B, then the two long lengths. One leg a process.
bash /tmp/run_cap_e2e2.sh

# The rates, from each leg's own two lines.
#   prefill = prompt_tokens / (elapsed - decode_seconds)
grep -a "tokens in\|decode steps\|decode graphs\|prompt .* tokens" /tmp/cap_c262144_on.log

# The two decode paths' logits, one dump each. 64/64 identical, max |diff| = 0.000e+00.
bash /tmp/run_cap_e2e3.sh
"$ENV/python" /tmp/cmp_logits.py /tmp/capdump_off.pt /tmp/capdump_on.pt

# The eight continuations, which must hash to d1775d3580129c2953d16060b73c915c.
```

## What this page does not claim

- **No eager decode at the long lengths.** Both long legs are `--decode-graphs`; the eager column
  exists at 1024 only.
- **No throughput claim.** This is one request, one process a card, and a single stream. Nothing here
  says what four streams or a batch of 32 would do.
- **No warm-page-cache claim.** All six legs ran with the resident bank attached, which is what makes
  the expert rows come out of pinned host memory rather than off `/mnt/data3`.
- **Not the fastest prefill that exists, and not the default's either.** Every leg here is the
  `sorted` deal, which is one flag away from the default rather than the default; the `id` deal is
  1.400× on the same 262144 length and is what a bare command line makes.

## These rates are the launcher's, and the native route is still closed to this checkpoint

Every number on this page comes out of `src/cli/generate_v41.py`, one request at a time. That is a
choice rather than the only possibility: [`pocketllm serve --backend v41`](deepseek_v4_1_flash_served_gate.md)
serves this checkpoint over the same `src/models/deepseek_v4_1` runtime and takes its own readings of
the two columns. What these numbers still have no counterpart in, and what no flag opens, is
`cpp_engine/`: nothing under it can load `/mnt/data3/DeepSeek-V4.1-Flash`, so a *native-engine*
deployment is gated on a runtime that does not exist. Three separate things would each stop it, and
the first is reachable by name:

- **The architecture key has no factory.** The checkpoint's `config.json` declares
  `model_type: deepseek_v41`, and `detect_architecture` returns that string — verified with the built
  native module, `detect_architecture("/mnt/data3/DeepSeek-V4.1-Flash") == "deepseek_v41"` against
  `registered_architectures() == ["deepseek_v4", "qwen3_5"]`. `create_engine` looks its factory up by
  that key, so `cpp_engine --serve --ckpt /mnt/data3/DeepSeek-V4.1-Flash` throws "no engine registered
  for architecture 'deepseek_v41'" rather than starting. The Python server's **native** adapter stops
  one step earlier, on `pocketllm/backends/cpp_backend.py`'s `_ENGINE_KIND_BY_ARCHITECTURE`, which
  carries the same two keys and raises `UnsupportedFeatureError` for anything else — a different
  adapter, `--backend v41`, is the route that does not go through that map at all. The `deepseek_v4`
  engine that is registered is the 43-layer, 4096-hidden, 256-expert, top-8-index-head model that
  `/mnt/data3/DeepSeek-V4-Flash-0731` detects as, which is a different architecture rather than an
  older copy of this one.
- **Its config has no dimensions where the loader looks for them.** V4.1-Flash's `config.json` is the
  multimodal wrapper: the 40 layers, 5120 hidden, 384 experts, 64 heads and 32 index heads are under
  `text_config`, and the top level declares only the token ids and the two sub-configs.
  `ModelConfig::from_hf_config` (`cpp_engine/core/model_config.cpp:146`) reads the root object and
  nothing else, so even a registered factory would come back with `hidden_size` 0 and
  `num_hidden_layers` 0 — a silently empty config rather than an error. Only
  `QwenConfig::from_hf_config` unwraps `text_config`.
- **Engram is not in the engine at all.** The string `engram` does not appear anywhere under
  `cpp_engine/`. The checkpoint carries 12 `engram.*` tensors on its two engram layers, 189.13 GiB of
  the 475.24 GiB total, and the engine has no code path for that lookup
  ([inventory](../architecture/deepseek_v4_1_flash_design.md#tensor-inventory-verified-from-the-shard-headers)).

A fourth gap was the one this page's subject lived in, and that one has closed. `prefill_chunk` used
to be plumbed through `src/models/deepseek_v4_1/generate.py` and `generate_v41.py` and nowhere else,
so the 4096-token chunk these rates are built on — and with it a 262144-token context at a 22 GiB
card — had no counterpart in either server; `--backend v41` now takes it as
`--backend-option prefill_chunk=4096`, and [the served gate page](deepseek_v4_1_flash_served_gate.md)
measures that configuration rather than arguing for it. What the three bullets above describe was
never a slower step but a missing runtime for this geometry, and there is now a Python one, running
this same module. The native one is what remains open, and
[the device experts page](deepseek_v4_1_flash_device_experts.md) records the same boundary from its own
side.
