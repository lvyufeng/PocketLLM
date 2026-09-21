# DeepSeek-V4.1-Flash: the same checkpoint behind the server

[The launcher page](deepseek_v4_1_flash_single_request_capability.md) closes on the reason no serving
number for this checkpoint existed: `deepseek_v41` has no registered engine in `cpp_engine` and no
entry in pocketllm's engine map, its dimensions sit under `text_config` where
`ModelConfig::from_hf_config` does not read, and `engram` appears nowhere under `cpp_engine/` — so
`src/cli/generate_v41.py` under `torchrun` was the only thing that ran it. This page is the other
path. `pocketllm serve --backend v41` puts the OpenAI-compatible server in front of the same
`src/models/deepseek_v4_1` runtime, four ranks, one process a card, and this is what it costs.

**On the reference's own short-context configuration the server clears the deployment gate: 137.5,
140.8 and 138.3 tok/s of prefill over three back-to-back requests at a 1364-token prompt, at 4.53,
4.45 and 4.53 tok/s of decode — with 108.0 and 4.53 on the first request, which is the slow one
because it pays the capture pass.** At the reference's 32768-token leg the two columns go opposite
ways, and neither turns out to be the expert deal: with the deal **held fixed at the reference's own
`sorted`**, the server reads **142.4 and 139.6 tok/s of prefill against the launcher's 105.98** and
**3.91 and 3.85 tok/s of decode against the launcher's 4.37**. The prefill is a third faster, the
decode is a tenth slower, and this page does not get both into one explanation; what it does measure
is that the deal is worth 1.12× on the server's own prefill and nothing on its decode, which is not
the 1.48× the reference measured for the same deal on the launcher.

**The longest leg the runtime accepts is the same story with a wider prefill margin: 150.3, 152.0 and
152.0 tok/s of prefill at 3.53, 3.48 and 3.54 tok/s of decode over three back-to-back requests on a
260244-token prompt**, against the launcher's own 262144 row's 103.54 tok/s at 3.88 — 1.45–1.47× on
prefill, 0.90–0.91× on decode, and 1730.0 s of wall against 2555.4 s. The decode half of the gate is
the half that does not travel: the service reads 4.45–4.53 tok/s at 1364 prompt tokens and 3.48–3.54
at 262144, and the launcher's own 4.98 → 3.88 across the same two lengths.

| Configuration | TP4, one process a card, one process a rank on 4 x RTX 2080 Ti, `DEEPSEEK_V41_RESIDENT_EXPERTS=1`, `OMP_NUM_THREADS=22` |
| Agent | `pocketllm serve --backend v41 --tensor-parallel-size 4`, the CLI's own supervisor, rank 0 binding the listener |
| Short leg | `--max-model-len 2048 --backend-option expert_pool_rows=288 --backend-option prefill_chunk=4096 --backend-option decode_graphs=true --backend-option threads=22` |
| Long leg | as above with `--max-model-len 32768 --backend-option expert_pool_rows=148`, and `expert_deal=sorted` on the arm that is compared with the reference |
| Longest leg | as above with `--max-model-len 262144` — 148 rows and a 4096-token chunk, which is the reference's own long-context pool — on the shipping `id` deal rather than `sorted` |
| Request | `temperature 0.0`, `max_tokens 64`, streamed with `stream_options.include_usage`, filler prompt of 1364, 32524 or 260244 tokens |
| Client | `/tmp/v41_gate.py` — `prompt_tokens / ttft` for prefill and the inter-token cadence of the SSE stream for decode |
| Startup | 457.8 GiB of routed experts pinned in **99.3–102.6 s a rank**, all 48 shards loaded in **130.5–133.7 s**, four `POCKETLLM_RANK_READY`; at 262144 the same pinning takes **92.7–95.6 s a rank (202.4–208.9 ms a GiB)** and the load **124.0–124.9 s** |
| Logs | `/tmp/v41_serve_short_fixed.log`, `/tmp/v41_serve_32k_fixed.log`, `/tmp/v41_serve_32k_sorted.log`, `/tmp/v41_serve_256k.log`; measurement output `/tmp/v41_gate_fixed_32k.txt`, `/tmp/v41_gate_32k_sorted.txt`, `/tmp/v41_gate_256k.txt`, `/tmp/v41_gate_256k_third.txt` |

## Four requests, back to back

One prompt, four requests, nothing restarted between them. Every column is the service's, and every
one of them returned:

| Request | Prompt | ttft | Prefill | Steps | Decode | Wall |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 1364 tok | 12.63 s | 108.0 tok/s | 63 in 13.89 s | 4.53 tok/s | 26.70 s |
| 2 | 1364 tok | 9.92 s | 137.5 tok/s | 63 in 14.00 s | 4.50 tok/s | 24.16 s |
| 3 | 1364 tok | 9.69 s | 140.8 tok/s | 63 in 14.16 s | 4.45 tok/s | 24.02 s |
| 4 | 1364 tok | 9.86 s | 138.3 tok/s | 63 in 13.90 s | 4.53 tok/s | 24.01 s |

The first request costs 2.7 s more of wall than the three behind it and pays it entirely in ttft:
that is `driver.capture_pass(...)` and the replay behind it, which `_decode_graphs` runs after the
prompt's own forward and before the decode clock starts, so it lands in the prefill column here. The
decode column is flat at 13.9–14.2 s across all four, and `reasoning chars 0` on every one of them —
this is `thinking_mode="chat"`, so nothing arrives on the reasoning side of the stream.

That the four requests all returned is the point of the row, not a detail of it: on the first cut of
this adapter only the first one did.

## The defect a launcher cannot have

`graphs=True` installs a recording on every block and leaves it there — it is what
`Generation.driver` documents, and the caller that wants the eager body back is the one that calls
`driver.release()`, which is why the launcher, one request to a process, never needed to. A server
serves the next request, and the next request's *prompt* is what paid. `Block.forward`
(`modules.py:625-628`) hands **every** forward to `block.decode_graph` whatever its width, and
`LayerGraphs.step` writes the input into a sink whose first real pass — the one-token replay the
capture is built from — sized at one row. So a 1364-token prompt entered the previous request's
recording and died there:

    RuntimeError: output with shape [1, 1, 4, 5120] doesn't match the broadcast shape [1, 1364, 4, 5120]

Request 1 fine, every request after it that error, and the client saw the role chunk and then a
stream that ended without a token, because the failure was on the server.

Two fixes, in the two places the responsibility actually sits:

- **The adapter hands the driver back when the request ends.** `V41Backend._run_payload` releases
  `generation.driver` on its success path, so a request's recordings are that request's. This is also
  what makes re-capture correct rather than merely tidy: a capture is taken from one request's prefill
  activation at one request's position, and `capture_pass` rewinds the caches it recorded through.
- **The loop releases its own driver if it is left.** `_decode_graphs` can be abandoned mid-step — the
  adapter's cancellation and stop-string paths both unwind out of `on_token`, and there is no driver
  to hand back when that happens — so its body is wrapped, and a run that unwinds by any exception
  calls `driver.release()` before re-raising. A loop that is left installs nothing.

`tests/test_v41_backend.py` pins the first half (59 tests, over `FakeDriver`) and
`tests/test_models_deepseek_v4_1_generate.py` the second (5 tests, over the real loop with `Pos` and
the model faked): an unwinding callback, a step that raises, a run that finishes and keeps its
graphs for the caller, a run that stops on the prefill's own first token and so never builds one, and
the eager loop that builds none at all.

## 32768, against the reference's own leg

The reference's `c32768c_on` row is **105.98 tok/s prefill and 4.37 tok/s decode** over a 32716-token
prompt, `--expert-pool-rows 148 --prefill-chunk-tokens 4096 --decode-graphs`, whole call 323.3 s. It
is the **`sorted`** expert deal's row, and `sorted` stopped being the default on 2026-09-20 (#306).
The first arm below is therefore the same configuration with one difference that is not a detail —
the deal — and the second arm adds `expert_deal=sorted` and is the one to compare the reference with.

| Arm | Prompt | ttft | Prefill | Decode | Steps | Wall |
| --- | --- | --- | --- | --- | --- | --- |
| default deal (`id`) | 32524 tok | 204.54 s | 159.0 tok/s | 3.82 tok/s | 63 in 16.51 s | 221.34 s |
| default deal (`id`) | 32524 tok | 205.16 s | 158.5 tok/s | 3.95 tok/s | 63 in 15.93 s | 221.36 s |
| `expert_deal=sorted` | 32524 tok | 228.37 s | 142.4 tok/s | 3.91 tok/s | 63 in 16.11 s | 244.75 s |
| `expert_deal=sorted` | 32524 tok | 232.99 s | 139.6 tok/s | 3.85 tok/s | 63 in 16.37 s | 249.64 s |

**The deal is worth 1.12× here and 1.48× on the launcher, so the launcher's deal figure does not
carry over to this path.** 159.0/142.4 = 1.117 on the prefill and 3.82–3.95 against 3.85–3.91 on the
decode, which is a wash — the deal partitions experts over the cards, and what the service pays for
prefill is the *width of the set a rank stages* rather than its share of the rows, so a smaller
effect is the expected direction and the size is the open question. The 148-row pool these legs run
on is also the reference's own long-context setting for a reason unrelated to the deal: 288 rows is
−7.2% a chunk at 32768 for +4.34 GiB and dies in the second chunk at 262144.

**Against the launcher's own `sorted` leg the server is 1.32–1.34× on prefill and 0.88–0.89× on
decode, and the two runs are not separated by anything on this page.** 139.6–142.4 against 105.98,
244.8–249.6 s of wall against 323.3 s for a prompt of the same length — the same pool, the same
chunk, the same deal. The launcher's prefill column is a subtraction between two printed walls and
the server's is a client-side ttft, which makes the server's the *floor* of the two. The two legs
are also two trees, and the direction matters for the attribution: the launcher's rows are
`origin/master` at `f7572f0`, and **the three stacked prefill kernels are ancestors of it** — #296
(`7102c19`), #297 (`aa83816`) and #298 (`b394ddb`) merged at 14:48–14:49 on 2026-09-20, `f7572f0` at
20:06 — which is why the launcher's own 262144 rate of 103.5–104.0 tok/s is the stacked arm's 104.0
and not `98e828f`'s 70.4. What the launcher's rows predate is the `id`/`sorted` deal flip (#306) and
the indexer row split (#318); the deal is held fixed at `sorted` on the compared arm above, so the
1.32–1.34× here is not the kernel work re-measured against an older tree. Decode is the column where the
convention is visible rather than inferred: 253–262 ms a step against the launcher's 229 is ~25 ms
that is not a step, and roughly half of the 4.37 → 3.85–3.91 gap.

**A short prompt on this service is not the short leg above, and the context is why.** The same 1364
tokens on the same code but at `--max-model-len 32768` and 148 rows read **81.9 tok/s of prefill
(16.65 s of ttft, of which ~2.7 s is the first request's capture pass) and 3.87 tok/s of decode**,
against 137.5–140.8 and 4.45–4.53 at 2048 and 288 rows. Both gaps are the context rather than the
prompt: the indexer's candidate path gathers 16384 positions whatever the prompt is, so at a
2048-token context it cannot be the full path at all, and the decode graph replays the attention over
the cache's full width. That is also what the launcher's own table shows between its 1024-context and
32768-context legs, 4.98 against 4.37.

## 262144, the longest leg the runtime accepts

The reference has 262144 in its table twice. The row this section goes against is **`c262144c_on`** —
262874 prompt tokens, **103.54 tok/s of prefill and 3.88 tok/s of decode**, 258 ms a decode step,
whole call 2555.4 s; its sibling `c262144_on` is the same prompt file 36 bytes shorter, 262865 tokens,
103.58 and 3.86 over 2554.5 s, so the two are 0.04% apart on the prefill and 0.5% on the decode and
which one is quoted changes nothing. Both are the `sorted` deal.

| Request | Prompt | ttft | Prefill | Steps | Decode | Wall |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 260244 tok | 1731.00 s | 150.3 tok/s | 63 in 17.87 s | 3.53 tok/s | 1749.16 s |
| 2 | 260244 tok | 1711.83 s | 152.0 tok/s | 63 in 18.09 s | 3.48 tok/s | 1730.23 s |
| 3 | 260244 tok | 1711.93 s | 152.0 tok/s | 63 in 17.78 s | 3.54 tok/s | 1730.01 s |

**Against that row the service is 1.45–1.47× on the prefill, 0.90–0.91× on the decode and 1.48× on
the wall**, 1730.0 s against 2555.4 s — the widest end-to-end margin on this page, and a prefill
margin wider than the 1.32–1.34× at 32768 for the reason the next paragraph is about. Requests 2 and
3 agree to 0.1 s of ttft and 0.4 s of wall, which is what says the first request's 19.2 s of extra
ttft is the capture pass rather than the host: at this length it is 7× the 2.7 s it costs at 1364
prompt tokens, because the recording it replays is over a cache 128 times the width.

**This leg is one flag apart from the row it is compared with, and the flag is in the service's
favour.** It ran the shipping `id` deal and the reference's row is `sorted`, and the pair at 32768
above prices the deal on this path at 1.12× on the prefill, so a like-for-like prefill margin at
262144 is nearer 1.3× than 1.47×. That is *below* the 1.32–1.34× the same comparison reads at 32768,
so on like-for-like terms the service's prefill advantage narrows with length — and the 1.3× is
carried from one length to another rather than measured here. What the leg establishes is what the
gate asks: 150.3–152.0 tok/s of prefill over a prompt the launcher does at 103.54, on three
consecutive requests that all returned.

**The two prompts are not the same text, and that is the largest thing this page has not measured.**
The service was sent 1020 words of filler repeated to 261000, which its chat encoding counts at
260244 tokens; the reference's row is `/tmp/prompt262144c.txt`, a chapter template with a varying
chapter number, tokenized raw, and it is 262874 tokens. A rank stages the *set* of experts the chunks
it sees ask for, and the eviction counters that would say how many distinct rows the filler pulled
are printed by the launcher alone — the served logs end at `POCKETLLM_RANK_READY` — so whether
repeating four sentences costs fewer rows than reading 345 chapters does is not answered here, and it
is a candidate for part of the 1.45–1.47×. The convention itself is reachable and is not what was used:
`V41Backend._tokenize` sends a `/v1/completions` prompt through `tokenizer(prompt)["input_ids"]` with
the comment that this is what the launcher does with `--prompt`, while every leg on this page went
through `/v1/chat/completions` and carries the checkpoint's own chat header instead.

## What the server costs, and what that does to the columns

Three differences make a served column not a launcher column, and none of them is the runtime's. They
are the server's costs; a fourth difference is not a cost at all but what was sent — this page's
filler prompt against the launcher's prose, which applies to every leg here and which the 262144
section works through:

- **Decode is timed at the client.** The SSE cadence here is wall-clock between chunks arriving over
  loopback, through the ASGI server, the JSON serialization and this probe's own parsing; the
  launcher's `decode_seconds` is taken inside `_decode_graphs`. That is 253–262 ms a step at 32768
  against the launcher's graphed 229, 282–287 ms at 262144 against its 258, and 220–225 ms at 1364
  prompt tokens against its 201 — the same ~25 ms a step at all three lengths, which is not a step.
- **Prefill is timed at the client too, so it is a floor rather than a rate.** ttft covers the whole
  request path — tokenizer, the checkpoint's own prompt encoder, the queue, the first forward, the
  capture pass on a graphed run — so `prompt_tokens / ttft` under-reports the forward. It is
  nevertheless the number a caller experiences, which is why it is the column above.
- **Each request re-records its graphs**, since the adapter released them when the last one ended.
  That is 2.7 s of the first request's ttft at 1364 prompt tokens, 19.2 s of it at 262144, and is
  inside every leg of the tables above; it is the price of serving one request at a time with graphs
  on, and it is not in the launcher's numbers at all.

## What it is not

- **Not batched.** `capabilities.supports_batch` is `False` and the scheduler is one mutable KV state
  serialized at the backend boundary: a second request waits for the first. There is no continuous
  batching, no prefix caching and no logprobs.
- **Not speculative.** The three DSpark draft layers are 7.39 GiB the loader leaves in the shards.
- **Not interruptible inside a prompt.** Cancellation is a per-step collective between the ranks, so a
  cancel lands after the prompt's forward rather than during it; this is what the capability's
  `cancellation` detail says.
- **Not a silent failure when a request does not fit.** A prompt longer than the caches were sized for
  is refused *on the stream* — `data: {"error": {...}}` and then `data: [DONE]`, about 0.10 s in, after
  the role chunk — so a client that counts tokens rather than reading the events reports "no tokens"
  and not an error. The refusal names the arithmetic: `this request needs 42686 positions (42684
  prompt tokens and 2 new), and the attention caches were sized at 32768 at startup; raise
  --max-model-len and restart.`

## Reproduce

```bash
# The server. One process a rank is the CLI's own supervisor.
export DEEPSEEK_V41_RESIDENT_EXPERTS=1 OMP_NUM_THREADS=22
/home/lvyufeng/miniconda3/envs/deepseek/bin/python -m pocketllm serve \
  --model /mnt/data3/DeepSeek-V4.1-Flash --backend v41 --tensor-parallel-size 4 \
  --max-model-len 2048 --port 8100 \
  --backend-option expert_pool_rows=288 --backend-option prefill_chunk=4096 \
  --backend-option decode_graphs=true --backend-option threads=22

# The measurement: prompt length in words, or in tokens with the `tok:` prefix.
/home/lvyufeng/miniconda3/envs/deepseek/bin/python /tmp/v41_gate.py 1000 1000 1000 1000

# The longest leg is that command with `--max-model-len 262144` and
# `--backend-option expert_pool_rows=148` — the reference's own long-context pool, 148 rows because
# 288 "dies in the second chunk at 262144" — and then
/home/lvyufeng/miniconda3/envs/deepseek/bin/python /tmp/v41_gate.py tok:261000 tok:261000 tok:261000
```

## What this page does not claim

- **It does not claim the served decode matches the launcher's.** At 32768 it is 3.82–3.95 against
  4.37, at 262144 3.48–3.54 against 3.88 and at 1364 prompt tokens it is 4.45–4.53 against 4.98, and
  this page has not separated the deal, the client-side cadence and the per-request re-capture into a
  single number. The evidence it does have is that all three legs are ~10% down and the short leg is
  the one with no deal in it.
- **It does not claim 4 tokens a second of decode holds at the long lengths.** The service reads
  3.85–3.95 at 32768 and 3.48–3.54 at 262144, against the launcher's 4.37 and 3.88 — so the gate's
  decode half is met on the reference's short-context configuration, 4.45–4.53 against the
  launcher's 4.98, and by neither path at the two long ones.
- **It does not claim the prefill margins are the serving stack's.** The 262144 leg's prompt is a
  repeated filler and the launcher's is prose, a rank stages the set of experts its chunks ask for,
  and the counters that would price that difference are not printed on this path. The deal is a
  second unattributed item at that length, because the leg ran `id` and the row it is compared with
  ran `sorted`.
- **The acceptance is the gate, not a token comparison.** These legs were run with
  `temperature 0.0` and 64 tokens, and their text was not compared against the launcher's on the same
  prompt. What says the served path computes the same thing is the launcher's own bit-identical
  continuation evidence and the eight-way text parity behind it, not anything measured here.
- **Nothing on this page is a concurrency result.** One request at a time, and the second request's
  wall is the first one's plus a wait.
