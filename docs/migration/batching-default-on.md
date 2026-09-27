# Batching is on by default for the `cpp` backend

**Affects:** anyone running `pocketllm serve --backend cpp`, or constructing `CppBackend` /
`EngineArgs(backend="cpp")` without an explicit width.

## What changed

| | Before | Now |
|---|---|---|
| `pocketllm serve --backend cpp` | The serialized session: one request at a time, no scheduler | The batch scheduler, **8 slots** |
| `--enable-batching` | Did not exist | The CLI spelling, on by default for this backend |
| `--max-batch-size 4` on its own | Accepted, then read by nothing | Asks for the batch path at width 4 |
| `--max-batch-size 8 --no-enable-batching` | The opt-out did not exist | `ConfigurationError` — the two contradict each other |
| `--backend-option enable_batching=true|false` | The only spelling that worked | Still works, and still wins over the flag |
| `--backend-option max_batch_size` | Read only when batching was already on | Still wins over `--max-batch-size` |

The width reaches the engine, which sizes its KV cache for that many slots **at construction** and
cannot grow it afterwards. So a width that stops at the CLI is a width the server refuses at the
first concurrent request, which is what made `--max-batch-size` inert before: it was parsed, put into
`EngineArgs`, and read by nothing on this backend.

## What to do

**If you serve one request at a time and care about its latency**, turn the width down or off:

```bash
pocketllm serve --backend cpp --model ... --no-enable-batching   # the old behaviour, exactly
pocketllm serve --backend cpp --model ... --max-batch-size 2     # a small width, if you want one
```

The batch path costs a lone request more than the serialized session does, in two ways. The
scheduler runs the width's rows whether or not that many requests are present, which with the prompt
cache held fixed is ~17% wall at width 2 and ~18% at width 8. And the scheduler's prefill path does
not consult the prefix cache, so a prompt the serialized session would have resumed for nothing is
re-forwarded in full — which is the larger of the two for a client that repeats its prompt. Both are
measured in
[the concurrency acceptance page](../performance/cpp_openai_concurrency_validation.md#what-the-width-costs-a-lone-request).

**If your checkpoint no longer fits**, the KV arena is the reason. It is
`width × context × bytes a token × full-attention layers`, so a configuration that fitted at width 1
now reserves eight times the KV: on Qwen3.8-27B-FP8 at 64K that is 1,026 MiB a rank becoming about
8 GiB. Either name a smaller width, keep the memory and turn batching off, or lower
`--max-model-len`.

**If you were relying on `--max-batch-size` doing nothing**, that was the defect rather than a
feature.

## What did not change

The `torch`, `v41`, `mimo` and `xing4` backends are unaffected: they have no scheduler for this flag
to select, so it is ignored there and their capabilities still report the width they can actually
honour. `AsyncLLM`'s own concurrency is its executor's `max_workers`, which is still 1 — it adds no
batching of its own.
