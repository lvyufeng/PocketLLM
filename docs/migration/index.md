# Migration notes

Breaking changes to PocketLLM's configuration and default behaviour, one note per change, each
recording what changed, who is affected, and what to set instead.

| Note | The change |
|---|---|
| [Batching is on by default for the `cpp` backend](batching-default-on.md) | `pocketllm serve --backend cpp` now runs the batch scheduler at 8 slots where it used to run the serialized session, so per-request latency and KV memory both change unless a width is named. |
| [The `dsv4` to `pocket` rename](dsv4-to-pocket-rename.md) | Environment variables and module paths were renamed from `DSV4_*` to `POCKETLLM_*`; the old spellings are not read. |
