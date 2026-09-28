# The C++ binary's `--serve` front end is removed

**Affects:** anyone launching `pocketllm_engine --serve`, or the wrappers and harnesses that did it
for them — `scripts/run_cpp_serve_tp4.sh`, `scripts/bench_cpp_openai_{phases,concurrency}.py`,
`scripts/run_serving_sweep.sh` and the `native` golden fixture.

## What changed

The repository had two HTTP front ends: the Python one under `pocketllm serve`, and a second one
compiled into the C++ binary behind `pocketllm_engine --serve`. They were separate request parsers,
separate samplers, separate stop-string scanners and separate token detokenizers, and the two
disagreed about real requests — a `cpp` request through `pocketllm serve` answered with a tool call
as prose while the same checkpoint through the binary answered with `tool_calls`, and the two were
apart the other way round on `response_format`, which the binary applied and the Python host did not
until [#477](https://github.com/lvyufeng/PocketLLM/pull/477) ported it. Issue
[#447](https://github.com/lvyufeng/PocketLLM/issues/447) merges them by keeping the Python front end
and deleting the C++ one, which is what this note records the end of.

**The binary is still built and still useful.** It is the engine's own driver: checkpoint inspection
(`--dump-config`, `--qwen-audit`), the smoke and forward paths, the standalone benchmarks, and the
`--serve`-less test host. What is gone is its HTTP server, its `python_sidecar`, its own `metrics`
collector, its bundled `httplib`, and the request-field and stop-string code that only those used.

## What to do

The Python front end is the only one, so every launch moves to it:

| | Before | Now |
|---|---|---|
| Start the server | `pocketllm_engine --serve --ckpt ...` | `pocketllm serve --model ... --backend cpp` |
| Checkpoint | `--ckpt /path` | `--model /path` |
| Context | `--max-context 8192` | `--max-model-len 8192` |
| Parallelism | `--tp-world 4 --tp-rank R` | `--tensor-parallel-size 4` (the supervisor starts the ranks) |
| Rendezvous | `--nccl-id-path /tmp/x.id` | nothing: the supervisor makes a private per-run path |
| Cards | `--device N` | `--device-ids 0,1,2,3`, one per rank |
| Paged KV | `--kv-paged` | `--backend-option kv_paged=true` |
| KV block | `--kv-block-size 16` | `--backend-option kv_block_size=16` |
| Host / port | `--host` / `--port` | the same two flags |
| Chunked prefill | `--prefill-chunk-tokens N` | the same flag |
| Prefix cache | `--prefix-cache-bytes N` | the same flag |
| KV dtype | `--kv-cache-dtype fp16` | the same flag |
| Batch width | `--max-batch-size N` | the same flag |
| Python host | `--python` / `--sidecar` | gone: the host is the process that started the server |
| Prompt budget | `--prefill-token-budget N` | no flag — see below |
| Request timeout | `--request-timeout-seconds N` | no flag — the Python host has none |
| Reduced depth | `--smoke-layers N` | gone: it truncated the model for debugging a server that no longer exists |

A DeepSeek-V4 TP4 launch, before and after:

```bash
CKPT=/path/to/DeepSeek-V4-Flash PORT=8000 MAX_CONTEXT=8192 bash scripts/run_cpp_serve_tp4.sh
```

```bash
python -m pocketllm serve --model /path/to/DeepSeek-V4-Flash --backend cpp \
    --tensor-parallel-size 4 --max-model-len 8192 --port 8000
```

### Two knobs that have no replacement

`--prefill-token-budget` and `--request-timeout-seconds` were `OpenAIServerConfig` fields of the
deleted server, and neither is a `pocketllm serve` flag.

The budget is the batch scheduler's per-iteration prompt cap
(`BatchScheduler::prefill_token_budget_`); the scheduler keeps its own default of 4096 on the
unified front end, which is the value the wrapper scripts passed anyway, so a launch that was
tuning it to 4096 is unchanged and one that was tuning it to something else can no longer say so.
The cap moves a prompt through the device in pieces; it does not change what is generated.

The timeout bounded how long the C++ server would wait on a request before answering with an error.
The Python host has no equivalent, and a client that needs one sets its own.

## The Ascend consequence

`pocketllm serve --backend cpp` loads `pocketllm_cpp`, the pybind extension, and
`scripts/build_ascend.sh` does not build it: `cpp` is the one runtime that declares `ascend` in
`pocketllm/backends/capabilities.py`, and until that is fixed **the 910B machine has no HTTP serving
path at all**. That is issue [#478](https://github.com/lvyufeng/PocketLLM/issues/478), and it is why
the Ascend serving records under `docs/performance/` cannot be reproduced today even though nothing
about the engine they measured changed.

## What did not change

The engine. Every kernel, every model implementation, the batch scheduler, the C ABI, and the
`pocketllm_engine` binary's non-serving command lines are untouched; so are the recorded number in
`docs/performance/`, which are the engine's and not the front end's. The `cpp` runtime keeps the
request surface it already had through `pocketllm serve` — `stop`, `n`, `logprobs`, `thinking_mode`,
`add_generation_prompt` — plus `response_format`, which the Python host applies with the engine's own
tokenizer and vocabulary.
