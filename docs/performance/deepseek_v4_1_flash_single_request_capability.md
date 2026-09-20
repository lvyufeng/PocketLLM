# DeepSeek-V4.1-Flash: what one request costs, through the launcher

The [device experts page](deepseek_v4_1_flash_device_experts.md) closes its decode section with a
paragraph about what it does not claim, and this page exists to answer it: *"The 512-token prefill is
the probe rather than the launcher — `src/cli/generate_v41.py` does not print a prefill wall — so the
prefill column is the instrumented library at the launcher's own configuration, and the decode column
is the launcher undecorated."* Both columns here are the launcher, over four prompt lengths, one leg
to a process, every figure derived from lines the CLI itself prints.

**One request on four 2080 Ti, through the shipped CLI: prefill 106.0 tokens a second at 32768 prompt
tokens and 103.6 at 262144, decode 4.98 tokens a second at 1024, 4.37 at 32768 and 3.86 at 262144.**
The graphed decode step is **201–202 ms** at 1024 against **354–364 ms** eager — 1.76–1.81× — and the
whole call 32.8 s against 41.3–42.1 s, with the eight 1024-token legs' generated text **identical to
the byte**.

| Leg | Prompt | Flags | Prefill | ms a token | Decode | Whole call |
| --- | --- | --- | --- | --- | --- | --- |
| `c1024_off_a2` | 1024 | `--expert-pool-rows 288 --no-decode-graphs` | 55.0 tok/s | 18.2 | **2.82 tok/s** (354 ms) | 41.3 s |
| `c1024_on_a2` | 1024 | `--expert-pool-rows 288 --decode-graphs` | (51.2) | (19.5) | **4.98 tok/s** (201 ms) | 32.8 s |
| `c1024_off_b2` | 1024 | `--expert-pool-rows 288 --no-decode-graphs` | 54.5 tok/s | 18.4 | **2.75 tok/s** (364 ms) | 42.1 s |
| `c1024_on_b2` | 1024 | `--expert-pool-rows 288 --decode-graphs` | (51.5) | (19.4) | **4.95 tok/s** (202 ms) | 32.8 s |
| `c32768c_on` | 32716 | `--expert-pool-rows 148 --prefill-chunk-tokens 4096 --decode-graphs` | **105.98 tok/s** | 9.44 | **4.37 tok/s** (229 ms) | 323.3 s |
| `c262144_on` | 262865 | `--expert-pool-rows 148 --prefill-chunk-tokens 4096 --decode-graphs` | **103.58 tok/s** | 9.66 | **3.86 tok/s** (259 ms) | 2554.5 s |

`ms a token` is the prefill's, 1000 / the rate. The 1024 rate is in parentheses on the graphed legs
because on that path the same subtraction carries the capture pass — a decode step that is not one of
the 64 tokens — so the figure is a floor rather than a measurement; the eager column beside it is the
clean one, and the two prompts are one unchunked forward rather than the chunked configuration the
long lengths run.

Every leg is `DEEPSEEK_V41_RESIDENT_EXPERTS=1`, `--threads 22`, `--temperature 0.0`,
`--max-new-tokens 64`, `torchrun --nproc_per_node=4`, and **the default expert deal (`sorted`)**. That
last one matters for the prefill column: the `id` deal is priced on the
[chunked prefill page](deepseek_v4_1_flash_chunked_prefill.md) at **146.46 tok/s on this same 262144
length** against the `sorted` arm's 103.50 — the numbers here are the shipping default, not the
fastest arm that exists.

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
| Driver | `/tmp/run_cap_e2e.sh` and `/tmp/run_cap_e2e2.sh`; logs `/tmp/cap_*.log` |

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
| 262976 | 259 ms | 3.86 tok/s |

A **guarantee at 1024, 4.37 at 32768 and 3.86 at 262144** is what the measurements support. The
degradation is 28% over a 256× increase in context and it is not the attention's: the eager expert
call moves 184.4 → 211.6 → 239.6 ms across those three lengths while the graphs move 2.3 → 3.0 →
3.3 ms, so 55.2 ms of the 58 ms the step grew is expert staging against a pool that evicts on
essentially every row at the long lengths (96.9% and 97.1% of draws staged).

## The continuation, eight times identical

Every 1024-token leg — four in the first sitting and four in the second, alternating the flag, on
**two trees** — printed the same text. The continuation alone, taken as the printed text minus the
prompt, is **301 bytes** and hashes `md5 d1775d3580129c2953d16060b73c915c` on all eight:

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

Text is a statement about 64 argmaxes and not about the 64 x 129280 rows that produced them, and
#304 touches `_TopKStream.push` — inside a capture the early-out is skipped — so the stronger
acceptance is a logit-level comparison of the two paths **on one tree**, which is what
`--dump-logits` and `/tmp/cmp_logits.py` are for: two legs, one dump each, `max |diff| = 0.000e+00`
over the 64 x 129280 matrix as the bar. **This page carries the text parity; the logit comparison is
recorded in the section below when it lands, and nothing here depends on it.**

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
1.2 MB prompts, and both extended legs generate the intended `Chapter 345.` continuation.

## Reproduce

```bash
ENV=/home/lvyufeng/miniconda3/envs/deepseek/bin
CKPT=/mnt/data3/DeepSeek-V4.1-Flash

# 1024, A-B-A-B, then the two long lengths. One leg a process.
bash /tmp/run_cap_e2e2.sh

# The rates, from each leg's own two lines.
#   prefill = prompt_tokens / (elapsed - decode_seconds)
grep -a "tokens in\|decode steps\|decode graphs\|prompt .* tokens" /tmp/cap_c262144_on.log

# The eight continuations, which must hash to d1775d3580129c2953d16060b73c915c.
```

## What this page does not claim

- **No eager decode at the long lengths.** Both long legs are `--decode-graphs`; the eager column
  exists at 1024 only.
- **No throughput claim.** This is one request, one process a card, and a single stream. Nothing here
  says what four streams or a batch of 32 would do.
- **No warm-page-cache claim.** All six legs ran with the resident bank attached, which is what makes
  the expert rows come out of pinned host memory rather than off `/mnt/data3`.
- **Not the fastest prefill that exists.** The default `sorted` deal is used throughout; the `id`
  deal is 1.41× on the same 262144 length and is still opt-in.
