# A `max_tokens` the context cannot hold is refused

**Affects:** anyone sending `max_tokens` larger than what the prompt leaves to `--backend mimo` or
`--backend xing4`. `v41`, `torch` and `cpp` are unchanged.

## What changed

| | Before | Now |
|---|---|---|
| `max_tokens=10000` against a 4096-token context | Accepted. The runtime sized its cache from the number (`make_cache(len(ids) + budget + 8)`) and generated | **400** `invalid_request_error`: `this request needs N positions (P prompt tokens and B new), and the attention caches were sized at 4096 at startup; raise --max-model-len and restart` |
| a prompt that alone fills the context | Refused | Refused, same message, and the same `ConfigurationError` |
| `max_tokens` that fits | Answered | Answered |

`SamplingParams.token_budget` documents the rule this restores (`pocketllm/api/types.py:339`): an
explicit `max_tokens` is handed back **unchanged**, on the written condition that *the caller's
length check keeps the last word on it*. `v41` had that check (`_validate_length`); `mimo` and
`xing4` did not, and nothing downstream did either — the runtimes size their attention caches from
the budget, so an over-long cap was not an error, it was a silent allocation several hundred
megabytes past the context the operator configured.

Meeting an over-long cap with an allocation rather than a refusal is the failure mode the check
exists to prevent: at 46 KB a token across Xing4's forty layers, a cap large enough is a device OOM
instead of a named refusal, and every size in between is a run quietly using more memory than
`--max-model-len` says it will.

## What to do

**Send a cap that fits.** `max_tokens` may be absent — it then resolves to everything the prompt
leaves — and the answer ends at EOS or at the context limit, whichever comes first. That is what an
OpenAI client that omits the field already gets.

**Or make the context bigger**, which is what the error message names:

```bash
pocketllm serve --backend mimo  --model ... --max-model-len 32768
pocketllm serve --backend xing4 --model ... --max-model-len 32768
```

`--max-model-len` sizes the attention caches **at construction** and cannot be raised afterwards, so
the two numbers have to be decided together: a cap larger than the configured context was never
something the run could serve.

## What did not change

`v41`'s behaviour, the message text, and the HTTP status for an over-long prompt — that refusal was
already a 400. The only runs affected are those that were relying on the allocation.
