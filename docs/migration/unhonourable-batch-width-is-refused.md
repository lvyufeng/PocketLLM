# A `--max-batch-size` that cannot be honoured is now refused

**Affects:** anyone passing `--max-batch-size` above 1, or `--enable-batching`, to
`pocketllm serve --backend cpp` on a build or a checkpoint that cannot provide the batch scheduler.

## What changed

The cpp backend used to answer an unhonourable width with a warning and the serialized session:

```
UserWarning: the batch path was asked for (from --max-batch-size 8) but this native module does
not expose QwenBatchScheduler; falling back to serial execution, which serves one request at a time
```

It now raises `UnsupportedFeatureError` and the process does not start.

| | Before | Now |
|---|---|---|
| `--max-batch-size 8` on a build with no `QwenBatchScheduler` | Warning, then one request at a time | `UnsupportedFeatureError` naming the flag and the build |
| `--enable-batching` on the same build | The same warning | The same refusal |
| `--max-batch-size 8` on a checkpoint whose engine the scheduler will not take | Warning, then one request at a time | `UnsupportedFeatureError`, raised **before** the checkpoint is read |
| No width named, either case | Silent fallback; `capabilities.details["scheduler"]` says which session you got | Unchanged |

A width reaches the engine through the scheduler's own slot allocation, so "the scheduler could not
be built" and "the width cannot be honoured" are one fact rather than two. Accepting the flag and
serving one request at a time was the third state: the number was parsed, reached `EngineArgs`, and
changed nothing.

The default path is untouched. `pocketllm serve --backend cpp` with no width on a scheduler-less
build still runs the serialized session and still reports it in
`capabilities.details["scheduler"]` — refusing there would make the backend unusable on such a build,
and nobody asked for anything.

## What to do

**If you were relying on the width to be ignored**, that was the defect rather than a feature. Name
the width you can have:

```bash
pocketllm serve --backend cpp --model ... --max-batch-size 1     # the serialized session
pocketllm serve --backend cpp --model ... --no-enable-batching   # the same thing, spelled at the flag
```

**If the message says the module does not expose `QwenBatchScheduler`**, the extension was built
without it. `-DPOCKET_BUILD_PYTHON=ON` is not the whole recipe — see
[the native module build](../guides/pocketllm_api.md#native-c-python-module) — and `ldd` on the
installed module is the check that it produced a usable one.

**If the message says the engine is not an `InferenceEngine`**, the checkpoint selects an engine this
host has no batch path for. That is DeepSeek-V4 today: `PersistentEngine` is not bound as an
`InferenceEngine`, so a width cannot be delivered for it even though the native engine's own batched
decode exists and was measured at 1.62x at eight concurrent requests. Serve it at width 1 until that
is fixed.

## What did not change

The `torch`, `v41`, `mimo` and `xing4` backends still refuse a width at parse time, before anything
loads — they declare `supports_batch=False`, so `factory.py` answers them and this backend's
constructor is never reached.
