# Migration notes

Breaking changes to PocketLLM's configuration and default behaviour, one note per change, each
recording what changed, who is affected, and what to set instead.

| Note | The change |
|---|---|
| [`--device` splits into a platform and a card list](device-splits-into-platform-and-cards.md) | `--device` no longer names a card: it is the platform (`auto`/`cuda`/`ascend`/`cpu`), and the cards are `--device-ids 2,3`, one per rank. `--device cuda:2`, `--device 0` and `--backend-option device=…` are refused by name. |
| [Batching is on by default for the `cpp` backend](batching-default-on.md) | `pocketllm serve --backend cpp` now runs the batch scheduler at 8 slots where it used to run the serialized session, so per-request latency and KV memory both change unless a width is named. |
| [The `dsv4` to `pocket` rename](dsv4-to-pocket-rename.md) | Environment variables and module paths were renamed from `DSV4_*` to `POCKETLLM_*`; the old spellings are not read. |
